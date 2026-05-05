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

    # Re-apply species → NCBI genome backfill if the resolved map exists.
    # Otherwise re-extracting silently undoes the work of scripts/18 + 19.
    species_map_path = config.DATA / "species_to_genome.parquet"
    if species_map_path.exists():
        smap = pd.read_parquet(species_map_path)
        hits = smap[smap["status"] == "hit"][["species", "ncbi_accession"]]
        gap = df["genome_accession"].isna() & df["species"].notna()
        df = df.merge(hits, on="species", how="left")
        df.loc[gap & df["ncbi_accession"].notna(), "genome_accession"] = (
            df.loc[gap & df["ncbi_accession"].notna(), "ncbi_accession"]
        )
        df["genome_source"] = "bacdive"
        df.loc[gap & df["ncbi_accession"].notna(), "genome_source"] = "species_resolved"
        df = df.drop(columns=["ncbi_accession"])
        n_filled = (gap & df["genome_source"].eq("species_resolved")).sum()
        print(f"Re-applied species backfill: {n_filled:,} rows")

    out = config.DATA / "bacdive_phenotypes.parquet"
    df.to_parquet(out, index=False)

    print(f"\nWrote {len(df):,} strains to {out}")
    print("Field coverage:")
    for col in df.columns:
        n = df[col].notna().sum()
        print(f"  {col:30s} {n:>6,} / {len(df):,} ({100 * n / max(1, len(df)):.1f}%)")


if __name__ == "__main__":
    main()
