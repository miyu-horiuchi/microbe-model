"""Train v3: hand-crafted features (v1) concatenated with ESM-2 embeddings (v2).

Tests whether embeddings carry complementary signal to the curated features even
when they lose head-to-head. Same train/test splits and XGBoost hyperparameters
as v1 and v2.

Reads:
  data/bacdive_phenotypes.parquet
  data/features.parquet
  data/embeddings.parquet

Writes:
  artifacts/combined_results.json
"""
from __future__ import annotations

import time

import pandas as pd

from microbe_model import config
from microbe_model.train.baseline import save_results, train_all


def derive_group(row: pd.Series) -> str:
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
    embeds = pd.read_parquet(config.DATA / "embeddings.parquet")

    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df = df.merge(embeds, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(derive_group, axis=1)

    v1_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    v2_cols = [c for c in embeds.columns if c.startswith("emb_")]
    feature_cols = v1_cols + v2_cols

    print(f"Training table: {len(df):,} strains × {len(feature_cols)} features "
          f"({len(v1_cols)} hand-crafted + {len(v2_cols)} embedding dims)")
    print(f"Distinct groups: {df['group'].nunique():,}")
    print()

    results = train_all(df, feature_cols, group_col_override="group")

    out = config.ARTIFACTS / "combined_results.json"
    save_results(results, out, feature_cols=feature_cols)
    print(f"\nTrained in {time.time() - t0:.1f}s. Wrote {out}\n")

    print("Results summary:")
    for target, r in results.items():
        if r.folds:
            metric = r.folds[0].metric_name
            print(f"  {target:25s} {metric:10s} = {r.mean():.4f}  (n_folds={len(r.folds)})")
        else:
            print(f"  {target:25s} skipped")


if __name__ == "__main__":
    main()
