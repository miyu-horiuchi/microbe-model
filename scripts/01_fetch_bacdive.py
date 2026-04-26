"""Pull strain metadata + phenotype labels from BacDive.

Writes one JSON per strain to data/bacdive/, plus a consolidated parquet table at
data/bacdive_phenotypes.parquet.

Usage:
    uv run python scripts/01_fetch_bacdive.py --limit 1000
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.bacdive import (
    BacDiveClient,
    extract_phenotypes,
    fetch_with_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Max strains to fetch (None=all).")
    args = parser.parse_args()

    client = BacDiveClient()
    rows = []
    for bacdive_id in tqdm(client.iter_strain_ids(limit=args.limit), desc="BacDive", unit="strain"):
        record = fetch_with_cache(client, bacdive_id)
        rows.append(extract_phenotypes(record))

    df = pd.DataFrame(rows)
    out = config.DATA / "bacdive_phenotypes.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")
    print("Coverage of prediction targets:")
    for col in ("optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"):
        print(f"  {col}: {df[col].notna().sum()} / {len(df)}")
    print(f"  genome_accession: {df['genome_accession'].notna().sum()} / {len(df)}")


if __name__ == "__main__":
    main()
