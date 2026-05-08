"""Diagnose where the HMM lift came from per phenotype target.

Re-runs train_all() on the HMM-covered subset and dumps, per target:
  - top 25 features by XGBoost gain (with HMM/composition/codon/iso/MD tags)
  - aggregated importance by HMM category (temperature, ph, oxygen, salt,
    vitamin, nitrogen, carbon, special) — useful for spotting which categories
    paid off across multiple targets

Output: artifacts/marker_importance.json + console summary.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from microbe_model import config
from microbe_model.features.markers import all_markers, category_for
from microbe_model.train.baseline import train_all

PHENOTYPE_TARGETS = {
    "optimal_temperature_c": "regression",
    "optimal_ph": "regression",
    "oxygen_requirement": "classification",
    "salt_tolerance_pct": "regression",
}

# Map HMM column -> Pfam category. Built once at module load.
def _hmm_col_categories() -> dict[str, str]:
    out: dict[str, str] = {}
    name_to_pfam: dict[str, str] = {}
    for pfam, (name, _) in all_markers().items():
        name_to_pfam[name] = pfam
    return name_to_pfam


def col_category(col_name: str) -> str:
    """Return one of: hmm:<cat>, composition, codon, tetra, iso, mediadive, baseline."""
    if col_name.startswith("hmm_"):
        # column is hmm_<friendly_name>_<n|score|present>
        rest = col_name[len("hmm_"):]
        for suffix in ("_n", "_score", "_present"):
            if rest.endswith(suffix):
                friendly = rest[: -len(suffix)]
                pfam_for_name = {name: pfam for pfam, (name, _) in all_markers().items()}
                pfam = pfam_for_name.get(friendly)
                if pfam:
                    return f"hmm:{category_for(pfam)}"
                return "hmm:unknown"
    if col_name.startswith("aa_frac_"):
        return "composition"
    if col_name.startswith("codon_"):
        return "codon"
    if col_name.startswith("tetra_"):
        return "tetra"
    if col_name.startswith("iso_"):
        return "iso"
    if col_name.startswith("md_"):
        return "mediadive"
    return "baseline"


def derive_group(row: pd.Series) -> str:
    for col in ("family", "genus"):
        v = row.get(col)
        if isinstance(v, str) and v:
            return v
    s = row.get("species")
    if isinstance(s, str) and s:
        return s.split()[0]
    return "__unknown__"


def encode_iso(df: pd.DataFrame, *, min_count: int = 5) -> tuple[pd.DataFrame, list[str]]:
    import re
    from collections import Counter
    new_cols: list[str] = []
    for level in ("isolation_cat1", "isolation_cat2"):
        if level not in df.columns:
            continue
        c: Counter[str] = Counter()
        for v in df[level].dropna():
            c.update(v.split("|"))
        kept = [t for t, n in c.items() if n >= min_count]
        for tag in sorted(kept):
            slug = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
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
    hmm = pd.read_parquet(config.DATA / "hmm_features.parquet")
    pheno["bacdive_id"] = pheno["bacdive_id"].astype(int)
    feats["bacdive_id"] = feats["bacdive_id"].astype(int)

    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df = df[df["genome_accession"].isin(hmm["genome_accession"])].copy()
    df = df.merge(hmm, on="genome_accession", how="left")

    md_path = config.DATA / "mediadive_features.parquet"
    md_cols: list[str] = []
    if md_path.exists():
        md = pd.read_parquet(md_path)
        md["bacdive_id"] = md["bacdive_id"].astype(int)
        md_cols = [c for c in md.columns if c != "bacdive_id"]
        df = df.merge(md, on="bacdive_id", how="left")

    df["group"] = df.apply(derive_group, axis=1)
    df, iso_cols = encode_iso(df)

    base_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    hmm_cols = [c for c in hmm.columns if c != "genome_accession"]
    feature_cols = base_cols + iso_cols + md_cols + hmm_cols
    print(f"Training on {len(df):,} HMM-covered strains × {len(feature_cols)} features\n")

    results = train_all(df, feature_cols, group_col_override="group")

    report: dict[str, dict] = {}
    for target, r in results.items():
        if not r.importances:
            continue
        ranked = sorted(r.importances.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:25]

        cat_totals: dict[str, float] = {}
        for col, imp in r.importances.items():
            cat = col_category(col)
            cat_totals[cat] = cat_totals.get(cat, 0.0) + imp

        report[target] = {
            "task": r.task,
            "score": r.mean(),
            "n_folds": len(r.folds),
            "top_25_features": [{"name": n, "importance": float(i),
                                 "category": col_category(n)} for n, i in top],
            "category_totals": {k: float(v) for k, v in
                                sorted(cat_totals.items(), key=lambda kv: kv[1], reverse=True)},
        }

        score_label = "MAE" if r.task == "regression" else "F1_macro"
        print(f"--- {target}  ({score_label}={r.mean():.3f}, n_folds={len(r.folds)})")
        print("  top 10 features:")
        for n, i in top[:10]:
            print(f"    {i:.4f}  [{col_category(n):20s}]  {n}")
        print("  category totals:")
        for cat, total in sorted(cat_totals.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            print(f"    {total:.4f}  {cat}")
        print()

    out = config.ARTIFACTS / "marker_importance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {out}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
