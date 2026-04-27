"""Train v2 baseline using ESM-2 embeddings as features.

Compares head-to-head with the v0/v1 hand-crafted features. Output is a delta
table — does the embedding really beat the curated features, and on which targets?

Run after scripts/11_extract_embeddings.py has produced data/embeddings.parquet.
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
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    embed_path = config.DATA / "embeddings.parquet"
    if not embed_path.exists():
        raise SystemExit(f"Missing {embed_path}. Run scripts/11_extract_embeddings.py first.")
    embeds = pd.read_parquet(embed_path)

    df = pheno.merge(embeds, on=["bacdive_id", "genome_accession"], how="inner")
    df["group"] = df.apply(derive_group, axis=1)

    feature_cols = [c for c in embeds.columns if c.startswith("emb_")]
    print(f"Training table: {len(df):,} strains × {len(feature_cols)} embedding dims")
    print(f"Distinct groups: {df['group'].nunique():,}")
    print()

    df.to_parquet(config.DATA / "training_table_embeddings.parquet", index=False)
    t0 = time.time()
    results = train_all(df, feature_cols, group_col_override="group")

    out = config.ARTIFACTS / "embedding_results.json"
    save_results(results, out, feature_cols=feature_cols)
    print(f"\nTrained in {time.time() - t0:.1f}s. Wrote {out}\n")

    print("Comparison vs v1 hand-crafted features:")
    for target, r in results.items():
        if r.folds:
            metric = r.folds[0].metric_name
            print(f"  {target:25s} {metric:10s} = {r.mean():.4f}  (n_folds={len(r.folds)})")
        else:
            print(f"  {target:25s} skipped")


if __name__ == "__main__":
    main()
