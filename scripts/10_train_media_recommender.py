"""Train per-medium classifiers and report metrics across all media meeting the count cutoff.

Outputs:
  artifacts/media_recommender_results.json — per-medium PR-AUC + ROC-AUC, fold-by-fold.
  artifacts/media_recommender_report.md   — human-readable summary.
"""
from __future__ import annotations

import time

import pandas as pd

from microbe_model import config
from microbe_model.train.media_recommender import (
    build_training_table,
    save_models,
    save_results,
    train_per_medium,
    train_production_models,
)


def main() -> None:
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    sm = pd.read_parquet(config.DATA / "strain_media.parquet")
    md = pd.read_parquet(config.DATA / "media_metadata.parquet")
    medium_name_by_id = dict(zip(md["medium_id"].astype(str), md["name"], strict=True))

    print(f"Inputs: {len(feats):,} feature rows, {len(sm):,} strain↔medium links")

    X, y_matrix, medium_ids = build_training_table(feats, sm, pheno)
    groups = pheno.set_index("bacdive_id").loc[X.index, "family"].fillna("__unknown__")
    print(f"Training table: {len(X):,} strains × {X.shape[1]} features × {len(medium_ids)} media")
    print(f"Distinct families: {groups.nunique():,}")
    print()

    t0 = time.time()
    results = train_per_medium(X, y_matrix, medium_name_by_id, groups)
    print(f"Trained {len(results)} per-medium classifiers in {time.time() - t0:.1f}s\n")

    out_json = config.ARTIFACTS / "media_recommender_results.json"
    save_results(results, out_json)
    print(f"Wrote {out_json}")

    # Train production models on ALL data for inference + persist
    print("\nFitting production models on full dataset...")
    prod_models = train_production_models(X, y_matrix)
    models_dir = config.ROOT / "models" / "recommender"
    save_models(prod_models, list(X.columns), models_dir)
    print(f"Saved {len(prod_models)} production models to {models_dir}")

    # Headline summary
    rows = [(mid, r.medium_name, r.n_positives, r.n_negatives, r.mean_pr_auc(), r.mean_roc_auc())
            for mid, r in results.items()]
    summary = pd.DataFrame(rows, columns=["medium_id", "name", "n_pos", "n_neg",
                                          "pr_auc", "roc_auc"])
    summary = summary.sort_values("pr_auc", ascending=False)
    print(f"Median PR-AUC: {summary['pr_auc'].median():.3f}")
    print(f"Median ROC-AUC: {summary['roc_auc'].median():.3f}")
    print("\nTop 15 best-modeled media (by PR-AUC):")
    print(summary.head(15).to_string(index=False))
    print("\nWorst 5:")
    print(summary.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
