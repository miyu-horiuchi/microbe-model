"""Append top-K media recommendations to artifacts/uncultured_predictions.parquet.

Without this, the Browse UI can show a row's predicted phenotypes but cannot
answer "what would you put it in?" without a per-row click. Pre-scoring all 5K
uncultured genomes against all 27 trained media classifiers makes the Browse
tab self-contained.

Reads:
  data/uncultured_features.parquet
  data/media_metadata.parquet
  artifacts/uncultured_predictions.parquet
  models/recommender/*.ubj

Writes (overwrites):
  artifacts/uncultured_predictions.parquet
    + top1_medium_id, top1_medium_name, top1_confidence
    + top2_medium_id, top2_medium_name, top2_confidence
    + top3_medium_id, top3_medium_name, top3_confidence
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from microbe_model import config
from microbe_model.train.media_recommender import load_models


def main() -> None:
    t0 = time.time()
    pred_path = config.ARTIFACTS / "uncultured_predictions.parquet"
    feat_path = config.DATA / "uncultured_features.parquet"
    models_dir = config.ROOT / "models" / "recommender"

    if not pred_path.exists():
        raise SystemExit(f"Missing {pred_path}. Run scripts/07_predict_uncultured.py first.")
    if not feat_path.exists():
        raise SystemExit(f"Missing {feat_path}. Run scripts/07_predict_uncultured.py first.")
    if not models_dir.exists():
        raise SystemExit(f"Missing {models_dir}. Run scripts/10_train_media_recommender.py first.")

    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(feat_path)
    models, feature_cols = load_models(models_dir)
    media_meta = pd.read_parquet(config.DATA / "media_metadata.parquet")
    name_by_id = dict(zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True))

    print(f"Scoring {len(feats):,} uncultured genomes against {len(models)} media...")

    # Build the feature matrix once, then batch-predict per medium
    X = feats[feature_cols].to_numpy()
    accessions = feats["genome_accession"].to_numpy()

    medium_ids = list(models.keys())
    proba = np.zeros((len(feats), len(medium_ids)), dtype=np.float32)
    for j, mid in enumerate(medium_ids):
        proba[:, j] = models[mid].predict_proba(X)[:, 1]
        if (j + 1) % 5 == 0:
            print(f"  {j + 1}/{len(medium_ids)} media scored")

    # For each genome, take argsort top-3
    top3_idx = np.argsort(-proba, axis=1)[:, :3]

    rows = []
    for i, acc in enumerate(accessions):
        row = {"genome_accession": acc}
        for k in range(3):
            j = top3_idx[i, k]
            mid = medium_ids[j]
            row[f"top{k + 1}_medium_id"] = mid
            row[f"top{k + 1}_medium_name"] = name_by_id.get(mid, "(unknown)")
            row[f"top{k + 1}_confidence"] = float(proba[i, j])
        rows.append(row)
    media_df = pd.DataFrame(rows)

    # Drop any pre-existing topN_* columns to avoid duplicates on re-runs
    drop_cols = [c for c in preds.columns if c.startswith(("top1_", "top2_", "top3_"))]
    if drop_cols:
        preds = preds.drop(columns=drop_cols)

    merged = preds.merge(media_df, on="genome_accession", how="left")
    merged.to_parquet(pred_path, index=False)
    print(f"\nWrote {len(merged):,} rows to {pred_path}  ({time.time() - t0:.1f}s)")
    print(f"Columns added: top{{1,2,3}}_medium_{{id,name}} + top{{1,2,3}}_confidence")


if __name__ == "__main__":
    main()
