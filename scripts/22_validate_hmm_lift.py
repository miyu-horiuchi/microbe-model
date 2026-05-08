"""A/B validate the HMM-feature lift on oxygen prediction.

Restricts the training table to the strains with HMM features (currently 100),
then trains oxygen-requirement classifiers twice on identical rows:
  - arm A: composition + isolation + MediaDive features only (the v2 baseline)
  - arm B: same + 30 HMM marker columns

Same GroupKFold-by-family splits, same XGBoost config. Reports macro-F1 per arm
plus per-fold deltas. With only 100 strains the absolute number is noisy, but
the *relative* lift between A and B is a clean test of whether the HMM signal
adds value.

Usage:
    python scripts/22_validate_hmm_lift.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from microbe_model import config
from microbe_model.train.baseline import train_target


def derive_group(row: pd.Series) -> str:
    for col in ("family", "genus"):
        val = row.get(col)
        if isinstance(val, str) and val:
            return val
    species = row.get("species")
    if isinstance(species, str) and species:
        return species.split()[0]
    return "__unknown__"


def encode_isolation_categories(df: pd.DataFrame, *, min_count: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode isolation tags. Lower min_count for the small-N validation slice."""
    import re
    from collections import Counter
    new_cols: list[str] = []
    for level in ("isolation_cat1", "isolation_cat2"):
        if level not in df.columns:
            continue
        tag_counts: Counter[str] = Counter()
        for v in df[level].dropna():
            tag_counts.update(v.split("|"))
        kept = [t for t, n in tag_counts.items() if n >= min_count]
        for tag in sorted(kept):
            slug = tag.lower().replace(">", "gt").replace("<", "lt")
            slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
            col = f"iso_{level.split('_')[1]}_{slug}"
            if col in df.columns:
                continue
            df[col] = df[level].fillna("").apply(lambda v, t=tag: int(t in v.split("|")))
            new_cols.append(col)
    return df, new_cols


def main() -> None:
    t0 = time.time()
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    hmm = pd.read_parquet(config.DATA / "hmm_features_oxygen.parquet")
    print(f"Loaded: pheno={len(pheno):,}, feats={len(feats):,}, hmm={len(hmm):,}")

    pheno["bacdive_id"] = pheno["bacdive_id"].astype(int)
    feats["bacdive_id"] = feats["bacdive_id"].astype(int)
    hmm["bacdive_id"] = hmm["bacdive_id"].astype(int)

    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df = df[df["bacdive_id"].isin(hmm["bacdive_id"])].copy()
    df = df[df["oxygen_requirement"].notna()].copy()
    print(f"After restriction to HMM-featurized strains with oxygen labels: {len(df):,}")

    df["group"] = df.apply(derive_group, axis=1)
    df, iso_cols = encode_isolation_categories(df)

    md_path = config.DATA / "mediadive_features.parquet"
    md_cols: list[str] = []
    if md_path.exists():
        md = pd.read_parquet(md_path)
        md["bacdive_id"] = md["bacdive_id"].astype(int)
        md_cols = [c for c in md.columns if c != "bacdive_id"]
        df = df.merge(md, on="bacdive_id", how="left")

    df = df.merge(hmm.drop(columns=["genome_accession"]), on="bacdive_id", how="left")
    hmm_cols = [c for c in hmm.columns if c not in {"bacdive_id", "genome_accession"}]

    base_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    arm_a_cols = base_cols + iso_cols + md_cols
    arm_b_cols = arm_a_cols + hmm_cols

    print(f"Class distribution: {df['oxygen_requirement'].value_counts().to_dict()}")
    print(f"Distinct groups (families): {df['group'].nunique()}")
    print(f"arm A features: {len(arm_a_cols)}  |  arm B features: {len(arm_b_cols)} (+{len(hmm_cols)} HMM)")
    print()

    print("Arm A — without HMM features")
    res_a = train_target(df, "oxygen_requirement", "classification",
                         feature_cols=arm_a_cols, group_col="group", n_splits=5)
    a_scores = [f.value for f in res_a.folds]
    print(f"  fold F1 macro: {[round(x, 3) for x in a_scores]}  mean={res_a.mean():.3f}")

    print()
    print("Arm B — with HMM features")
    res_b = train_target(df, "oxygen_requirement", "classification",
                         feature_cols=arm_b_cols, group_col="group", n_splits=5)
    b_scores = [f.value for f in res_b.folds]
    print(f"  fold F1 macro: {[round(x, 3) for x in b_scores]}  mean={res_b.mean():.3f}")

    print()
    if a_scores and b_scores and len(a_scores) == len(b_scores):
        deltas = [b - a for a, b in zip(a_scores, b_scores, strict=True)]
        print(f"Per-fold delta (B − A): {[round(x, 3) for x in deltas]}")
        print(f"Mean delta: {np.mean(deltas):+.3f}")
    print(f"Mean F1 lift: {res_a.mean():.3f} → {res_b.mean():.3f}  "
          f"({(res_b.mean() - res_a.mean()):+.3f})")

    if res_b.importances:
        top = sorted(res_b.importances.items(), key=lambda kv: kv[1], reverse=True)[:15]
        print("\nTop 15 features in arm B (XGBoost gain):")
        for name, imp in top:
            tag = "  <-- HMM" if name in hmm_cols else ""
            print(f"  {imp:.4f}  {name}{tag}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
