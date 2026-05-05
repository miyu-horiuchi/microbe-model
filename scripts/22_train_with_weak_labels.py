"""Train baseline using BacDive-first + MediaDive-weak fallback labels for pH/salt.

Compares vs the curated-only baseline (artifacts/baseline_results.json) to see whether
the weak labels are net-helpful for the model. T_opt and oxygen are unaffected (no
weak source). pH and salt get many more training rows but with noisier labels.

Output: artifacts/baseline_results_weak.json
"""
from __future__ import annotations

import time

import pandas as pd

from microbe_model import config
from microbe_model.train.baseline import save_results, train_all

# Reuse the encoders from scripts/03 — copy locally to avoid sys.path gymnastics
import importlib.util
spec = importlib.util.spec_from_file_location("train03", config.ROOT / "scripts" / "03_train_baseline.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)


def main() -> None:
    t0 = time.time()
    catalog = pd.read_parquet(config.DATA / "strain_catalog.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    # Isolation categories live in the original phenotype table — graft them onto the catalog
    iso_in = [c for c in pheno.columns if c.startswith("isolation_cat")]
    pheno_iso = pheno[["bacdive_id"] + iso_in].copy()
    pheno_iso["bacdive_id"] = pheno_iso["bacdive_id"].astype(int)

    catalog["bacdive_id"] = catalog["bacdive_id"].astype(int)
    feats["bacdive_id"] = feats["bacdive_id"].astype(int)
    catalog = catalog.merge(pheno_iso, on="bacdive_id", how="left")
    df = catalog.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(train03.derive_group, axis=1)

    # Coerce numeric targets (catalog stored as object)
    for col in ("optimal_temperature_c", "optimal_ph", "salt_tolerance_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df, iso_cols = train03.encode_isolation_categories(df)
    print(f"Encoded {len(iso_cols)} isolation features")

    # IMPORTANT: do NOT add MediaDive features here. The weak labels for pH and salt
    # are *derived from those same features* (per-strain median across DSMZ media), so
    # including them as inputs leaks the target — the model trivially predicts the
    # matching feature column. The honest test is: do MediaDive-derived weak labels
    # help a genome+isolation model generalize?
    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    feature_cols = feature_cols + iso_cols

    print(f"\nTraining table: {len(df):,} strains × {len(feature_cols)} features")
    for tgt in ("optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"):
        n = df[tgt].notna().sum() if tgt in df.columns else 0
        print(f"  {tgt:25s} labeled={n:>6,}")
    print()

    results = train_all(df, feature_cols, group_col_override="group")
    out = config.ARTIFACTS / "baseline_results_weak.json"
    save_results(results, out, predictions_path=None, feature_cols=feature_cols)

    print(f"\nResults summary ({time.time() - t0:.1f}s):\n")
    for target, r in results.items():
        if r.folds:
            print(f"  {target:25s} {r.folds[0].metric_name:10s} = {r.mean():.4f}")


if __name__ == "__main__":
    main()
