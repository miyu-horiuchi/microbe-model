"""Deduplicated featurize for species-resolved genomes.

When many BacDive strains share a single species-level representative genome (the
common case after scripts/18), naively running scripts/02 re-downloads + re-runs
pyrodigal on the same FASTA per-strain. This script downloads each unique accession
once, then replicates the resulting feature dict across all bacdive_ids that share it.

Resumable via data/features.jsonl (skips bacdive_ids already in the log).

Usage:
    uv run python scripts/19_featurize_resolved.py --workers 7
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.pipeline import _load_done_ids, _process_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--max-accessions", type=int, default=None,
                        help="Cap how many unique accessions to process (debug).")
    args = parser.parse_args()

    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")

    # Training-ready pool: any phenotype label + a genome accession
    label_cols = list(config.PHENOTYPE_TARGETS.keys())
    has_label = pheno[label_cols].notna().any(axis=1)
    has_genome = pheno["genome_accession"].notna()
    ready = pheno[has_label & has_genome].copy()
    ready["bacdive_id"] = ready["bacdive_id"].astype(int)
    ready["genome_accession"] = ready["genome_accession"].astype(str)

    out_path = config.DATA / "features.jsonl"
    done_ids = _load_done_ids(out_path)
    todo = ready[~ready["bacdive_id"].isin(done_ids)]
    print(f"strains in pool: {len(ready):,}")
    print(f"  already featurized: {len(done_ids):,}")
    print(f"  remaining: {len(todo):,}")

    # Group remaining strains by accession
    by_acc = todo.groupby("genome_accession")["bacdive_id"].apply(list).to_dict()
    accessions = sorted(by_acc.keys())
    if args.max_accessions:
        accessions = accessions[: args.max_accessions]
    print(f"unique accessions to download: {len(accessions):,}")
    print(f"  avg strains per accession: {sum(len(by_acc[a]) for a in accessions) / max(1, len(accessions)):.1f}")
    print(f"workers: {args.workers}\n")

    # Featurize each accession once; the worker tags the result with the *first* bacdive_id
    # of that accession's strain group. We then replicate the feature dict to all sibling
    # bacdive_ids before writing.
    rep_tasks = [(by_acc[acc][0], acc) for acc in accessions]

    n_success = 0
    n_replicated_rows = 0
    start = time.time()
    with open(out_path, "a") as fh, \
         ProcessPoolExecutor(max_workers=args.workers) as pool, \
         tqdm(total=len(rep_tasks), desc="featurize", unit="genome") as bar:
        futures = {pool.submit(_process_one, t): t for t in rep_tasks}
        for fut in as_completed(futures):
            rep_id, acc = futures[fut]
            bar.update(1)
            try:
                feats = fut.result()
            except Exception:
                feats = None
            if not feats:
                continue
            n_success += 1
            for bid in by_acc[acc]:
                row = dict(feats)
                row["bacdive_id"] = bid
                row["genome_accession"] = acc
                fh.write(json.dumps(row) + "\n")
                n_replicated_rows += 1
            fh.flush()
            bar.set_postfix(genomes_ok=n_success, rows=n_replicated_rows)

    print(f"\nfinished in {(time.time() - start) / 60:.1f} min")
    print(f"  unique genomes featurized: {n_success:,}/{len(rep_tasks):,}")
    print(f"  feature rows written: {n_replicated_rows:,}")

    # Materialize parquet
    df = pd.read_json(out_path, lines=True)
    df = df.drop_duplicates(subset=["bacdive_id"], keep="last")
    parquet = config.DATA / "features.parquet"
    df.to_parquet(parquet, index=False)
    print(f"  wrote {len(df):,} rows to {parquet}")


if __name__ == "__main__":
    main()
