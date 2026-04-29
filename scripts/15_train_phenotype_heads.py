"""Pre-train phenotype prediction heads and persist them to disk.

Without this, scripts/recommend.py re-trains 4 targets × (3 quantile heads or 1
classifier) on every call (~30s cold start). Pre-training drops that to ~0s.

For regression targets (T_opt, pH, salt): trains XGBRegressor at quantiles
0.1, 0.5, 0.9 → median + 80% prediction interval.

For classification targets (oxygen): trains XGBClassifier + LabelEncoder.

Reads:
  data/bacdive_phenotypes.parquet
  data/features.parquet

Writes:
  models/phenotype/<target>_q10.ubj           (regression)
  models/phenotype/<target>_q50.ubj           (regression)
  models/phenotype/<target>_q90.ubj           (regression)
  models/phenotype/<target>.ubj               (classification)
  models/phenotype/<target>_classes.json      (classification, label list)
  models/phenotype/feature_cols.json          (shared)
"""
from __future__ import annotations

import json
import time

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from microbe_model import config


def main() -> None:
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]

    out_dir = config.ROOT / "models" / "phenotype"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols))

    print(f"Training set: {len(df):,} strains × {len(feature_cols)} features")
    print(f"Output dir: {out_dir}")
    print()

    for target, task in config.PHENOTYPE_TARGETS.items():
        if target not in df.columns:
            print(f"  {target:30s} skip (column missing)")
            continue
        labeled = df[df[target].notna()]
        if len(labeled) < 100:
            print(f"  {target:30s} skip (only {len(labeled)} labeled)")
            continue
        X = labeled[feature_cols]
        t0 = time.time()
        if task == "regression":
            y = labeled[target].to_numpy(dtype=float)
            for alpha, tag in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
                m = xgb.XGBRegressor(
                    n_estimators=400, max_depth=5, learning_rate=0.05,
                    tree_method="hist", n_jobs=-1,
                    objective="reg:quantileerror", quantile_alpha=alpha,
                )
                m.fit(X, y)
                m.save_model(str(out_dir / f"{target}_{tag}.ubj"))
            print(f"  {target:30s} regression  n={len(labeled):>5}  ({time.time()-t0:.1f}s)")
        else:
            enc = LabelEncoder()
            y_enc = enc.fit_transform(labeled[target].astype(str))
            m = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
            )
            m.fit(X, y_enc)
            m.save_model(str(out_dir / f"{target}.ubj"))
            (out_dir / f"{target}_classes.json").write_text(
                json.dumps(list(enc.classes_))
            )
            print(f"  {target:30s} classifier  n={len(labeled):>5}  classes={list(enc.classes_)}  ({time.time()-t0:.1f}s)")

    print(f"\nWrote phenotype heads to {out_dir}")


if __name__ == "__main__":
    main()
