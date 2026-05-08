"""A/B evaluate the unified HMM features across ALL four phenotype targets.

For each phenotype (T_opt, pH_opt, oxygen, salt), trains XGBoost twice on the
same rows — once without HMM features (arm A), once with (arm B) — and reports
the per-target lift.

This is the dashboard you check after each iteration of the marker library.

Restricts to rows that have HMM coverage so arms A and B see identical data.

Usage:
    python scripts/25_evaluate_all_targets.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from microbe_model import config
from microbe_model.train.baseline import train_target

PHENOTYPE_TARGETS = {
    "optimal_temperature_c": "regression",
    "optimal_ph": "regression",
    "oxygen_requirement": "classification",
    "salt_tolerance_pct": "regression",
}


def derive_group(row: pd.Series) -> str:
    for col in ("family", "genus"):
        val = row.get(col)
        if isinstance(val, str) and val:
            return val
    species = row.get("species")
    if isinstance(species, str) and species:
        return species.split()[0]
    return "__unknown__"


def encode_isolation_categories(df: pd.DataFrame, *, min_count: int = 5) -> tuple[pd.DataFrame, list[str]]:
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
    hmm_path = config.DATA / "hmm_features.parquet"
    if not hmm_path.exists():
        raise SystemExit("data/hmm_features.parquet not found — run scripts/24 first.")
    hmm = pd.read_parquet(hmm_path)
    print(f"Loaded: pheno={len(pheno):,}, feats={len(feats):,}, hmm={len(hmm):,} unique genomes")

    pheno["bacdive_id"] = pheno["bacdive_id"].astype(int)
    feats["bacdive_id"] = feats["bacdive_id"].astype(int)

    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df = df[df["genome_accession"].isin(hmm["genome_accession"])].copy()
    df = df.merge(hmm, on="genome_accession", how="left")
    print(f"Restricted to {len(df):,} strains with HMM coverage")

    df["group"] = df.apply(derive_group, axis=1)
    df, iso_cols = encode_isolation_categories(df)

    md_path = config.DATA / "mediadive_features.parquet"
    md_cols: list[str] = []
    if md_path.exists():
        md = pd.read_parquet(md_path)
        md["bacdive_id"] = md["bacdive_id"].astype(int)
        md_cols = [c for c in md.columns if c != "bacdive_id"]
        df = df.merge(md, on="bacdive_id", how="left")

    base_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    hmm_cols = [c for c in hmm.columns if c != "genome_accession"]
    arm_a_cols = base_cols + iso_cols + md_cols
    arm_b_cols = arm_a_cols + hmm_cols

    print(f"\nFeature counts:  arm A = {len(arm_a_cols)}  |  arm B = {len(arm_b_cols)} (+{len(hmm_cols)} HMM)")
    print(f"Distinct families (groups): {df['group'].nunique():,}")
    print()

    rows: list[dict] = []
    for target, task in PHENOTYPE_TARGETS.items():
        if target not in df.columns:
            continue
        n = df[target].notna().sum()
        if n < 50:
            print(f"--- {target}: skipping ({n} labeled rows)")
            continue
        print(f"--- {target}  ({task}, n={n})")

        res_a = train_target(df, target, task, feature_cols=arm_a_cols,
                             group_col="group", n_splits=5)
        res_b = train_target(df, target, task, feature_cols=arm_b_cols,
                             group_col="group", n_splits=5)
        a, b = res_a.mean(), res_b.mean()

        if task == "regression":
            # MAE — lower is better.
            direction = "↓" if b < a else "↑"
            print(f"  arm A MAE = {a:.3f}  |  arm B MAE = {b:.3f}  |  Δ = {b - a:+.3f} {direction}")
        else:
            # F1 — higher is better.
            direction = "↑" if b > a else "↓"
            print(f"  arm A F1  = {a:.3f}  |  arm B F1  = {b:.3f}  |  Δ = {b - a:+.3f} {direction}")
        print(f"  fold A: {[round(f.value, 3) for f in res_a.folds]}")
        print(f"  fold B: {[round(f.value, 3) for f in res_b.folds]}")
        print()
        rows.append({
            "target": target, "task": task, "n": int(n),
            "arm_a": a, "arm_b": b, "delta": b - a,
        })

    summary = pd.DataFrame(rows)
    out = config.ARTIFACTS / "hmm_lift_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
