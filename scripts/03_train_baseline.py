"""Train the multi-task XGBoost baseline.

Joins phenotypes + features, derives a stable group column for GroupKFold, trains, saves
the merged training table for the eval renderer, and writes per-target metrics.
"""
from __future__ import annotations

import time

import pandas as pd

from microbe_model import config
from microbe_model.train.baseline import save_results, train_all


def derive_group(row: pd.Series) -> str:
    """Group-K-fold key. Prefer LPSN family (from BacDive); fall back to genus then species."""
    for col in ("family", "genus"):
        val = row.get(col)
        if isinstance(val, str) and val:
            return val
    species = row.get("species")
    if isinstance(species, str) and species:
        return species.split()[0]
    return "__unknown__"


def main() -> None:
    t0 = time.time()
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(derive_group, axis=1)

    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]

    print(f"Training table: {len(df):,} strains × {len(feature_cols)} features")
    print(f"Distinct groups: {df['group'].nunique():,}")
    print(f"Group sizes (top 10): {df['group'].value_counts().head(10).to_dict()}")
    print()

    training_table = config.DATA / "training_table.parquet"
    df.to_parquet(training_table, index=False)
    print(f"Wrote training table to {training_table}")

    results = train_all(df, feature_cols, group_col_override="group")

    out = config.ARTIFACTS / "baseline_results.json"
    save_results(results, out)

    print(f"\nResults summary ({time.time() - t0:.1f}s):\n")
    for target, r in results.items():
        if r.folds:
            metric = r.folds[0].metric_name
            print(f"  {target:25s} {metric:10s} = {r.mean():.4f}  (n_folds={len(r.folds)})")
        else:
            print(f"  {target:25s} skipped (insufficient data)")


if __name__ == "__main__":
    main()
