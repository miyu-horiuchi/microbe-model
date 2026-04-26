"""Train the multi-task XGBoost baseline.

Joins phenotypes + features, derives a `family` column from `species` for group K-fold,
and writes per-target metrics to artifacts/baseline_results.json.
"""
from __future__ import annotations

import pandas as pd

from microbe_model import config
from microbe_model.train.baseline import save_results, train_all


def derive_family(species: str | None) -> str:
    """Crude family proxy: first word of binomial. Replace with GTDB lookup later."""
    if not species:
        return "__unknown__"
    return str(species).split()[0]


def main() -> None:
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df["family"] = df["species"].apply(derive_family)

    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    print(f"Training on {len(df)} strains × {len(feature_cols)} features.")
    print(f"Group counts (top 10): {df['family'].value_counts().head(10).to_dict()}")

    results = train_all(df, feature_cols)

    out = config.ARTIFACTS / "baseline_results.json"
    save_results(results, out)
    print(f"\nWrote results to {out}\n")
    for target, r in results.items():
        if r.folds:
            metric = r.folds[0].metric_name
            print(f"  {target:25s} {metric:10s} = {r.mean():.4f}  (n_folds={len(r.folds)})")
        else:
            print(f"  {target:25s} skipped (insufficient data)")


if __name__ == "__main__":
    main()
