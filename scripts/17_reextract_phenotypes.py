"""Rebuild data/bacdive_phenotypes.parquet from cached data/bacdive/*.json.

Use this after extending extract_phenotypes() to add fields without re-running the
~30-min API scan. Reads every cached JSON, re-applies the extractor, and overwrites
the parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.bacdive import extract_phenotypes


def main() -> None:
    files = sorted(Path(config.BACDIVE_DIR).glob("*.json"))
    print(f"Re-extracting from {len(files):,} cached JSONs in {config.BACDIVE_DIR}")

    rows = []
    for path in tqdm(files, desc="re-extract", unit="strain"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(extract_phenotypes(record))

    df = pd.DataFrame(rows)
    out = config.DATA / "bacdive_phenotypes.parquet"
    df.to_parquet(out, index=False)

    print(f"\nWrote {len(df):,} strains to {out}")
    print("Field coverage:")
    for col in df.columns:
        n = df[col].notna().sum()
        print(f"  {col:30s} {n:>6,} / {len(df):,} ({100 * n / max(1, len(df)):.1f}%)")


if __name__ == "__main__":
    main()
