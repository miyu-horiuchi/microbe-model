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


def encode_isolation_categories(
    df: pd.DataFrame,
    *,
    min_count: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode isolation_cat1/cat2 (pipe-joined multi-labels).

    Each strain's category cell is "Tag1|Tag2|..." (or NaN). We split, then create one
    iso_<level>_<tag> column per tag that appears in ≥min_count training rows. Strains
    without any isolation info get all-zero rows for these features (XGBoost treats this
    as "no signal" rather than missing).
    """
    new_cols: list[str] = []
    for level in ("isolation_cat1", "isolation_cat2"):
        if level not in df.columns:
            continue
        from collections import Counter
        tag_counts: Counter[str] = Counter()
        for v in df[level].dropna():
            tag_counts.update(v.split("|"))
        kept = [t for t, n in tag_counts.items() if n >= min_count]
        seen_slugs: set[str] = set()
        import re
        for tag in sorted(kept):
            slug = tag.lower().replace(">", "gt").replace("<", "lt")
            slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
            col = f"iso_{level.split('_')[1]}_{slug}"
            if col in seen_slugs:
                continue
            seen_slugs.add(col)
            df[col] = df[level].fillna("").apply(lambda v, t=tag: int(t in v.split("|")))
            new_cols.append(col)
    return df, new_cols


def main() -> None:
    t0 = time.time()
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(derive_group, axis=1)

    df, iso_cols = encode_isolation_categories(df)
    print(f"Encoded {len(iso_cols)} isolation-category features "
          f"({df[iso_cols].sum().sum():.0f} non-zero entries)")

    md_path = config.DATA / "mediadive_features.parquet"
    md_cols: list[str] = []
    if md_path.exists():
        md = pd.read_parquet(md_path)
        md["bacdive_id"] = md["bacdive_id"].astype(int)
        df["bacdive_id"] = df["bacdive_id"].astype(int)
        md_cols = [c for c in md.columns if c != "bacdive_id"]
        df = df.merge(md, on="bacdive_id", how="left")
        n_with_md = df[md_cols[0]].notna().sum() if md_cols else 0
        print(f"Joined MediaDive features ({len(md_cols)} cols) — "
              f"{n_with_md:,}/{len(df):,} training rows have MediaDive data")

    hmm_path = config.DATA / "hmm_features.parquet"
    hmm_cols: list[str] = []
    if hmm_path.exists():
        hmm = pd.read_parquet(hmm_path)
        hmm_cols = [c for c in hmm.columns if c != "genome_accession"]
        df = df.merge(hmm, on="genome_accession", how="left")
        n_with_hmm = df[hmm_cols[0]].notna().sum() if hmm_cols else 0
        print(f"Joined HMM features ({len(hmm_cols)} cols) — "
              f"{n_with_hmm:,}/{len(df):,} training rows have HMM data")

    kegg_path = config.DATA / "kegg_modules.parquet"
    kegg_cols: list[str] = []
    if kegg_path.exists():
        kegg = pd.read_parquet(kegg_path)
        kegg_cols = [c for c in kegg.columns if c != "genome_accession"]
        df = df.merge(kegg, on="genome_accession", how="left")
        n_with_kegg = df[kegg_cols[0]].notna().sum() if kegg_cols else 0
        print(f"Joined KEGG module completeness ({len(kegg_cols)} cols) — "
              f"{n_with_kegg:,}/{len(df):,} training rows have KEGG data")

    iso_meta_path = config.DATA / "isolation_metadata.parquet"
    iso_meta_cols: list[str] = []
    if iso_meta_path.exists():
        iso_meta = pd.read_parquet(iso_meta_path)
        iso_meta["bacdive_id"] = iso_meta["bacdive_id"].astype(int)
        df["bacdive_id"] = df["bacdive_id"].astype(int)
        # Use only the numeric/binary columns; leave free-text out of XGBoost
        keep = ["iso_lat", "iso_lon", "iso_collection_year"]
        keep += [c for c in iso_meta.columns if c.startswith(("iso_continent_", "iso_country_", "iso_host_kingdom_"))]
        iso_meta_cols = [c for c in keep if c in iso_meta.columns]
        df = df.merge(iso_meta[["bacdive_id"] + iso_meta_cols], on="bacdive_id", how="left")
        print(f"Joined isolation metadata ({len(iso_meta_cols)} cols)")

    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    feature_cols = feature_cols + iso_cols + md_cols + hmm_cols + kegg_cols + iso_meta_cols

    print(f"Training table: {len(df):,} strains × {len(feature_cols)} features")
    print(f"Distinct groups: {df['group'].nunique():,}")
    print(f"Group sizes (top 10): {df['group'].value_counts().head(10).to_dict()}")
    print()

    training_table = config.DATA / "training_table.parquet"
    df.to_parquet(training_table, index=False)
    print(f"Wrote training table to {training_table}")

    results = train_all(df, feature_cols, group_col_override="group")

    out = config.ARTIFACTS / "baseline_results.json"
    predictions_out = config.ARTIFACTS / "predictions.parquet"
    save_results(results, out, predictions_path=predictions_out, feature_cols=feature_cols)
    print(f"Wrote per-strain predictions to {predictions_out}")

    print(f"\nResults summary ({time.time() - t0:.1f}s):\n")
    for target, r in results.items():
        if r.folds:
            metric = r.folds[0].metric_name
            print(f"  {target:25s} {metric:10s} = {r.mean():.4f}  (n_folds={len(r.folds)})")
        else:
            print(f"  {target:25s} skipped (insufficient data)")


if __name__ == "__main__":
    main()
