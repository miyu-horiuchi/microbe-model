"""Honest test: do MediaDive-derived weak labels help generalize to BacDive-curated optima?

For pH and salt, two training regimes — same held-out test rows (curated only):
  A. CURATED-ONLY: train on BacDive curated labels.
  B. CURATED + WEAK: train on BacDive curated + MediaDive-derived weak labels.

In both, the test set per fold is the *intersection* of the held-out group with the
curated subset. This isolates whether weak labels help the model do better on the
gold-standard distribution, rather than just helping it predict the medium pH/salt
of the strain itself (which would be circular).

No MediaDive features are used (deployed model parity).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from microbe_model import config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("train03", ROOT / "scripts" / "03_train_baseline.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)


def cv_mae(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    *,
    train_mask: pd.Series,
    test_mask: pd.Series,
    n_splits: int = 5,
) -> tuple[float, int]:
    """5-fold GroupKFold by family. Train on (train_mask & target.notna() & not test fold).
    Evaluate on (test_mask & target.notna() & test fold). Returns mean MAE across folds.
    """
    eligible = df[df[target].notna()].copy()
    eligible[target] = pd.to_numeric(eligible[target], errors="coerce")
    eligible = eligible[eligible[target].notna()]
    groups = eligible["group"].fillna("__unknown__")
    splits = min(n_splits, max(2, groups.nunique()))
    kf = GroupKFold(n_splits=splits)
    maes = []
    n_eval_total = 0
    for tr_idx, te_idx in kf.split(eligible, eligible[target], groups):
        tr = eligible.iloc[tr_idx]
        te = eligible.iloc[te_idx]
        # Apply masks to the row indices we're using
        tr = tr[train_mask.reindex(tr.index, fill_value=False).values]
        te = te[test_mask.reindex(te.index, fill_value=False).values]
        if len(tr) < 100 or len(te) < 50:
            continue
        m = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            tree_method="hist", n_jobs=-1,
        )
        m.fit(tr[feature_cols], tr[target].astype(float))
        preds = m.predict(te[feature_cols])
        maes.append(mean_absolute_error(te[target].astype(float), preds))
        n_eval_total += len(te)
    return (float(np.mean(maes)) if maes else float("nan")), n_eval_total


def main() -> None:
    t0 = time.time()
    catalog = pd.read_parquet(config.DATA / "strain_catalog.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    iso_in = [c for c in pheno.columns if c.startswith("isolation_cat")]
    pheno_iso = pheno[["bacdive_id"] + iso_in].copy()
    pheno_iso["bacdive_id"] = pheno_iso["bacdive_id"].astype(int)

    catalog["bacdive_id"] = catalog["bacdive_id"].astype(int)
    feats["bacdive_id"] = feats["bacdive_id"].astype(int)
    catalog = catalog.merge(pheno_iso, on="bacdive_id", how="left")
    df = catalog.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(train03.derive_group, axis=1)

    # Numeric coercion
    for col in ("optimal_temperature_c", "optimal_ph", "salt_tolerance_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df, iso_cols = train03.encode_isolation_categories(df)
    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    feature_cols = feature_cols + iso_cols

    print(f"\nTraining table: {len(df):,} strains × {len(feature_cols)} features")
    print("Held-out test rows are always BacDive-curated only.\n")

    for target in ("optimal_ph", "salt_tolerance_pct"):
        src_col = f"{target}_source"
        curated = (df[src_col] == "bacdive")
        weak = (df[src_col] == "mediadive_weak")
        print(f"=== {target} ===")
        print(f"  curated rows:        {curated.sum():,}")
        print(f"  weak rows:           {weak.sum():,}")

        # A) CURATED-ONLY training
        mae_a, n_a = cv_mae(df, feature_cols, target,
                            train_mask=curated, test_mask=curated)
        # B) CURATED + WEAK training
        mae_b, n_b = cv_mae(df, feature_cols, target,
                            train_mask=(curated | weak), test_mask=curated)
        delta_pct = 100 * (mae_b - mae_a) / mae_a
        verdict = "HELPS" if mae_b < mae_a - 0.001 else (
            "HURTS" if mae_b > mae_a + 0.001 else "WASH"
        )
        print(f"  A. curated-only  MAE = {mae_a:.4f}  (eval n={n_a:,})")
        print(f"  B. curated+weak  MAE = {mae_b:.4f}  (eval n={n_b:,})")
        print(f"  → Δ = {delta_pct:+.1f}%   [{verdict}]\n")

    print(f"({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
