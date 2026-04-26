"""Scan BacDive and write strain phenotype labels to data/bacdive_phenotypes.parquet.

Uses the v2 public API (no auth). Discovers strain IDs by batch-scanning the
integer ID range — missing IDs are silently dropped server-side, so the scan
is complete in one pass over [start, end].

Usage:
    # Phase 1 smoke test — scan the first ~5K IDs (returns ~3-4K real records)
    uv run python scripts/01_fetch_bacdive.py --end 5000

    # Full BacDive (~150K live records, ~30 min wall time)
    uv run python scripts/01_fetch_bacdive.py --end 200000
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.bacdive import (
    BATCH_SIZE,
    DEFAULT_MAX_ID,
    BacDiveClient,
    cache_record,
    extract_phenotypes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=DEFAULT_MAX_ID)
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip writing per-strain JSON to disk (saves ~150K small files).")
    args = parser.parse_args()

    client = BacDiveClient()
    rows = []
    n_batches = (args.end - args.start) // BATCH_SIZE + 1

    with tqdm(total=n_batches, desc="BacDive batches", unit="batch") as bar:
        for bacdive_id, record in client.iter_records(start=args.start, end=args.end):
            if not args.no_cache:
                cache_record(bacdive_id, record)
            rows.append(extract_phenotypes(record))
            # tqdm advances per batch — track via the integer ID
            if bacdive_id % BATCH_SIZE == 0:
                bar.update(1)
        bar.update(n_batches - bar.n)  # finalize

    df = pd.DataFrame(rows)
    out = config.DATA / "bacdive_phenotypes.parquet"
    df.to_parquet(out, index=False)

    print(f"\nWrote {len(df)} strains to {out}")
    print("Coverage of prediction targets:")
    for col in ("optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"):
        n = df[col].notna().sum()
        print(f"  {col:30s} {n:>6d} / {len(df)} ({100 * n / max(1, len(df)):.1f}%)")
    n_genome = df["genome_accession"].notna().sum()
    print(f"  genome_accession              {n_genome:>6d} / {len(df)} ({100 * n_genome / max(1, len(df)):.1f}%)")
    n_both = df[df["genome_accession"].notna() & df["optimal_temperature_c"].notna()].shape[0]
    print(f"\n  genome + T_opt (training-ready) {n_both:>4d} strains")


if __name__ == "__main__":
    main()
