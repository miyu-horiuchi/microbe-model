"""Streaming fetch + feature extraction.

Reads training-ready strains (those with both genome accession and ≥1 phenotype label)
from data/bacdive_phenotypes.parquet, then in parallel:
  - downloads each genome FASTA from NCBI Datasets into memory
  - runs pyrodigal + AA-composition feature extraction
  - appends the result to data/features.jsonl

Resumable — re-running picks up where it left off (skips bacdive_ids already in the log).
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.pipeline import stream_fetch_and_featurize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--max", type=int, default=None,
                        help="Cap how many strains to process (default: all training-ready).")
    parser.add_argument("--require-target", default="optimal_temperature_c",
                        help="Filter to strains with this label populated (or 'any' for any label).")
    args = parser.parse_args()

    pheno_path = config.DATA / "bacdive_phenotypes.parquet"
    if not pheno_path.exists():
        raise SystemExit(f"Missing {pheno_path}. Run scripts/01_fetch_bacdive.py first.")
    pheno = pd.read_parquet(pheno_path)

    has_genome = pheno["genome_accession"].notna()
    if args.require_target == "any":
        label_cols = list(config.PHENOTYPE_TARGETS.keys())
        has_label = pheno[label_cols].notna().any(axis=1)
    else:
        has_label = pheno[args.require_target].notna()

    ready = pheno[has_genome & has_label].copy()
    if args.max:
        ready = ready.head(args.max)

    tasks = list(zip(ready["bacdive_id"].astype(int), ready["genome_accession"].astype(str), strict=True))
    out_path = config.DATA / "features.jsonl"
    print(f"Training-ready strains: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {out_path}")
    print()

    start = time.time()
    with tqdm(total=len(tasks), desc="featurize", unit="strain") as bar:
        def progress(n_completed, n_success, n_total):
            bar.n = n_completed
            bar.set_postfix(success=n_success, fail=n_completed - n_success)
            bar.refresh()

        stream_fetch_and_featurize(
            tasks,
            out_path=out_path,
            n_workers=args.workers,
            on_progress=progress,
        )

    elapsed = time.time() - start
    if out_path.exists():
        with open(out_path) as fh:
            rows_in_log = sum(1 for _ in fh)
    else:
        rows_in_log = 0
    print(f"\nFinished in {elapsed/60:.1f} min. {rows_in_log} feature rows in {out_path.name}.")

    # Materialize parquet from JSONL for downstream training
    if rows_in_log:
        df = pd.read_json(out_path, lines=True)
        parquet = config.DATA / "features.parquet"
        df.to_parquet(parquet, index=False)
        print(f"Wrote {len(df)} rows to {parquet}")


if __name__ == "__main__":
    main()
