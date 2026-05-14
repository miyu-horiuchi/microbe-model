"""Merge per-genome and per-strain feature tables into a single features.parquet.

Joins:
  data/features.parquet            (per-strain, base genome composition: 355 cols)
+ data/kegg_modules.parquet        (per-genome, KEGG module completeness)
+ data/hmm_features.parquet        (per-genome, unified marker HMM hits)
+ data/isolation_metadata.parquet  (per-strain, BacDive isolation source)

Output: data/features.parquet (overwritten, backup saved to features.parquet.bak).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from microbe_model import config


def main() -> None:
    D: Path = config.DATA

    print("Loading inputs...")
    base = pd.read_parquet(D / "features.parquet")
    kegg = pd.read_parquet(D / "kegg_modules.parquet")
    hmm = pd.read_parquet(D / "hmm_features.parquet")
    iso = pd.read_parquet(D / "isolation_metadata.parquet")

    # Normalize join keys
    base["genome_accession"] = base["genome_accession"].astype(str)
    kegg["genome_accession"] = kegg["genome_accession"].astype(str)
    hmm["genome_accession"] = hmm["genome_accession"].astype(str)
    base["bacdive_id"] = base["bacdive_id"].astype(int)
    iso["bacdive_id"] = iso["bacdive_id"].astype(int)

    print(f"  base:  {len(base):>6,} × {base.shape[1]:>4}")
    print(f"  kegg:  {len(kegg):>6,} × {kegg.shape[1]:>4}")
    print(f"  hmm:   {len(hmm):>6,} × {hmm.shape[1]:>4}")
    print(f"  iso:   {len(iso):>6,} × {iso.shape[1]:>4}")

    # Drop string-typed columns from isolation_metadata; numeric/one-hot encoded
    # versions of the same fields already exist alongside them.
    iso_numeric = iso.select_dtypes(include=["number", "bool"]).copy()
    if "bacdive_id" not in iso_numeric.columns:
        iso_numeric["bacdive_id"] = iso["bacdive_id"].astype(int)
    dropped = sorted(set(iso.columns) - set(iso_numeric.columns))
    if dropped:
        print(f"  dropping {len(dropped)} non-numeric iso cols: {dropped}")

    merged = base.merge(kegg, on="genome_accession", how="left")
    merged = merged.merge(hmm, on="genome_accession", how="left", suffixes=("", "_hmm"))
    merged = merged.merge(iso_numeric, on="bacdive_id", how="left", suffixes=("", "_iso"))

    # Drop columns from secondary tables that collided with base (suffix _iso/_hmm)
    junk = [c for c in merged.columns if c.endswith("_iso_dup") or c.endswith("_hmm_dup")]
    if junk:
        merged = merged.drop(columns=junk)

    print(f"\nmerged: {len(merged):,} rows × {merged.shape[1]} cols")
    print(f"  rows with kegg features: {merged.filter(like='kegg_').notna().any(axis=1).sum():,}")
    print(f"  rows with hmm features:  {merged.filter(like='hmm_').notna().any(axis=1).sum():,}")
    print(f"  rows with iso features:  {merged.filter(like='iso_').notna().any(axis=1).sum():,}")

    out = D / "features.parquet"
    backup = D / "features.parquet.bak"
    if out.exists() and not backup.exists():
        shutil.copy2(out, backup)
        print(f"\nbacked up original to {backup}")
    merged.to_parquet(out, index=False)
    print(f"wrote {out}: {len(merged):,} rows × {merged.shape[1]} cols")


if __name__ == "__main__":
    main()
