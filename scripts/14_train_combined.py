"""Train v3: hand-crafted features (v1) + ESM-2 embeddings (v2) + isolation tags.

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

import re
import time
from collections import Counter

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


def encode_isolation_categories(
    df: pd.DataFrame,
    *,
    min_count: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode isolation_cat1/cat2 (pipe-joined multi-labels).

    Mirrors the encoder in scripts/03_train_baseline.py so v3 sees the same
    isolation-tag vocabulary as v1.
    """
    new_cols: list[str] = []
    for level in ("isolation_cat1", "isolation_cat2"):
        if level not in df.columns:
            continue
        tag_counts: Counter[str] = Counter()
        for v in df[level].dropna():
            tag_counts.update(v.split("|"))
        kept = [t for t, n in tag_counts.items() if n >= min_count]
        seen_slugs: set[str] = set()
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
    embeds = pd.read_parquet(config.DATA / "embeddings.parquet")

    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    df = df.merge(embeds, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(derive_group, axis=1)

    df, iso_cols = encode_isolation_categories(df)
    print(f"Encoded {len(iso_cols)} isolation-category features "
          f"({df[iso_cols].sum().sum():.0f} non-zero entries)")

    v1_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    v2_cols = [c for c in embeds.columns if c.startswith("emb_")]
    feature_cols = v1_cols + v2_cols + iso_cols

    print(f"Training table: {len(df):,} strains × {len(feature_cols)} features "
          f"({len(v1_cols)} hand-crafted + {len(v2_cols)} embedding dims + {len(iso_cols)} iso tags)")
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
