"""Build data/strain_media.parquet by walking the BacDive cache.

No network calls — pure local extraction from cached records.
Output: one row per (bacdive_id, medium_id) link.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.mediadive import iter_bacdive_strain_media


def main() -> None:
    rows = []
    n_files = sum(1 for _ in config.BACDIVE_DIR.glob("*.json"))
    print(f"Walking {n_files:,} BacDive cached records...")
    for row in tqdm(iter_bacdive_strain_media(), total=n_files, unit="link"):
        rows.append(row)

    df = pd.DataFrame(rows)
    out = config.DATA / "strain_media.parquet"
    df.to_parquet(out, index=False)

    print(f"\nWrote {len(df):,} strain↔medium links to {out}")
    print(f"  unique strains:  {df['bacdive_id'].nunique():,}")
    print(f"  unique media:    {df['medium_id'].nunique():,}")
    print(f"  growth=yes:      {(df['growth'] == 'yes').sum():,}")
    print(f"  growth=no:       {(df['growth'] == 'no').sum():,}")
    print(f"  growth=weak:     {(df['growth'] == 'weak').sum():,}")
    print("\nTop 10 most-used media:")
    print(df.groupby(['medium_id', 'medium_name']).size().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
