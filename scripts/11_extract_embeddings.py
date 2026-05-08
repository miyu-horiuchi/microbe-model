"""Extract per-genome ESM-2 embeddings for the full BacDive training corpus.

Designed to run on a CUDA GPU (Lightning AI T4 / A100). Falls back to MPS / CPU
for testing — but at scale you really want GPU.

Reads:
  data/bacdive_phenotypes.parquet (strain list)
  + downloads each genome via the existing pipeline._fetch_fasta_bytes path

Writes:
  data/embeddings.jsonl  (one row per genome, append-only, resumable)
  data/embeddings.parquet (materialized at end)

Resumability: re-running picks up where it left off (same JSONL pattern as
features.jsonl).

Usage:
    # Full corpus on GPU (Lightning AI). ~3-5 hr on T4 with sample_n=50.
    uv run --extra embeddings python scripts/11_extract_embeddings.py \\
        --model facebook/esm2_t30_150M_UR50D --sample-n 50 --batch-size 32

    # Smoke test on Mac MPS with smallest model
    uv run --extra embeddings python scripts/11_extract_embeddings.py \\
        --model facebook/esm2_t6_8M_UR50D --sample-n 20 --max 10
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.features.embeddings import embed_genome, load_esm2
from microbe_model.features.genome import predict_genes
from microbe_model.pipeline import _fetch_fasta_bytes


def _load_done_ids(path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                ids.add(int(row["bacdive_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D",
                        help="HF model id (esm2_t6_8M / t12_35M / t30_150M / t33_650M)")
    parser.add_argument("--sample-n", type=int, default=50,
                        help="Proteins per genome to embed (None = all). 50 is a fast default.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="ESM-2 batch size (32 fits on T4 16GB; raise on A100).")
    parser.add_argument("--max", type=int, default=None,
                        help="Cap how many genomes to process (default: all training-ready).")
    args = parser.parse_args()

    pheno_path = config.DATA / "bacdive_phenotypes.parquet"
    if not pheno_path.exists():
        raise SystemExit(f"Missing {pheno_path}. Run scripts/01_fetch_bacdive.py first.")
    pheno = pd.read_parquet(pheno_path)

    has_genome = pheno["genome_accession"].notna()
    label_cols = list(config.PHENOTYPE_TARGETS.keys())
    has_label = pheno[label_cols].notna().any(axis=1)
    ready = pheno[has_genome & has_label].copy()

    out_path = config.DATA / "embeddings.jsonl"
    done_ids = _load_done_ids(out_path)
    pending = ready[~ready["bacdive_id"].astype(int).isin(done_ids)]
    if args.max:
        pending = pending.head(args.max)
    print(f"Embedding {len(pending):,} genomes (skipping {len(done_ids):,} already done)")

    print(f"Loading {args.model}...")
    tokenizer, model, device = load_esm2(args.model)
    print(f"  device={device}, embed_dim={model.config.hidden_size}, "
          f"sample_n={args.sample_n}, batch_size={args.batch_size}")

    rng = np.random.default_rng(0)
    t0 = time.time()
    n_success = 0
    n_fail = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as log:
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc="embed", unit="genome"):
            bid = int(row["bacdive_id"])
            acc = str(row["genome_accession"])
            try:
                contigs = _fetch_fasta_bytes(acc)
                if not contigs:
                    n_fail += 1
                    continue
                proteins, _, _ = predict_genes(contigs)
                if not proteins:
                    n_fail += 1
                    continue
                vec = embed_genome(
                    proteins, tokenizer, model, device,
                    sample_n=args.sample_n, batch_size=args.batch_size, rng=rng,
                )
            except Exception as exc:  # noqa: BLE001 — single bad genome shouldn't kill the run
                print(f"  skip {acc}: {type(exc).__name__}: {exc}")
                n_fail += 1
                continue
            payload = {
                "bacdive_id": bid,
                "genome_accession": acc,
                "embed_dim": int(len(vec)),
                "embedding": vec.tolist(),
            }
            log.write(json.dumps(payload) + "\n")
            log.flush()
            n_success += 1

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed/60:.1f} min. {n_success} succeeded, {n_fail} failed.")

    # Materialize parquet — flatten the embedding list into per-dim columns
    print("Materializing parquet...")
    rows = []
    with open(out_path) as fh:
        for line in fh:
            row = json.loads(line)
            emb = row["embedding"]
            d = {"bacdive_id": row["bacdive_id"], "genome_accession": row["genome_accession"]}
            d.update({f"emb_{i}": float(v) for i, v in enumerate(emb)})
            rows.append(d)
    df = pd.DataFrame(rows)
    parquet_path = config.DATA / "embeddings.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(df):,} embeddings to {parquet_path}")


if __name__ == "__main__":
    main()
