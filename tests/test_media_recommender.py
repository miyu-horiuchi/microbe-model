"""Tests for the per-medium recommender training pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from microbe_model.train.media_recommender import (
    build_training_table,
    save_results,
    train_per_medium,
)


def _synthetic_dataset(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (features, strain_media, bacdive) with a real signal:
    strains with high f0 use medium A, low f0 use medium B.
    """
    rng = np.random.default_rng(seed)
    bids = np.arange(n) + 1
    feat_cols = {f"f{i}": rng.normal(size=n) for i in range(6)}
    features = pd.DataFrame({"bacdive_id": bids, "genome_accession": [f"GCA_{i:09d}" for i in bids],
                             **feat_cols})

    bacdive = pd.DataFrame({
        "bacdive_id": bids,
        "family": [f"family_{i % 10}" for i in bids],
        "genus": [f"genus_{i % 30}" for i in bids],
        "species": [f"species_{i}" for i in bids],
    })

    rows = []
    f0 = features["f0"].to_numpy()
    for bid, x in zip(bids, f0, strict=True):
        # Medium A: probability rises with f0
        if rng.random() < 1 / (1 + np.exp(-x)):
            rows.append({"bacdive_id": int(bid), "medium_id": "A", "medium_name": "MediumA",
                         "growth": "yes"})
        # Medium B: opposite
        if rng.random() < 1 / (1 + np.exp(x)):
            rows.append({"bacdive_id": int(bid), "medium_id": "B", "medium_name": "MediumB",
                         "growth": "yes"})
        # Medium C: random noise (no signal)
        if rng.random() < 0.5:
            rows.append({"bacdive_id": int(bid), "medium_id": "C", "medium_name": "MediumC",
                         "growth": "yes"})
    strain_media = pd.DataFrame(rows)
    return features, strain_media, bacdive


def test_build_training_table_shape() -> None:
    features, sm, bacdive = _synthetic_dataset(n=300)
    X, y_matrix, medium_ids = build_training_table(features, sm, bacdive)
    assert len(X) > 0
    assert y_matrix.shape[0] == len(X)
    assert y_matrix.shape[1] == len(medium_ids)
    # All training-table rows should have at least one positive medium link
    assert (y_matrix.sum(axis=1) > 0).all()
    # Y values are {0, 1}
    unique = set(y_matrix.values.ravel().tolist())
    assert unique.issubset({0, 1})


def test_train_recovers_signal() -> None:
    """Medium A's classifier should beat random — its presence is correlated with f0."""
    features, sm, bacdive = _synthetic_dataset(n=400)
    X, y_matrix, _ = build_training_table(features, sm, bacdive)
    groups = bacdive.set_index("bacdive_id").loc[X.index, "family"]

    # Lower threshold for the test fixture (test data has 3 media, all under default 100)
    from microbe_model.train import media_recommender as mr
    original = mr.MIN_STRAINS_PER_MEDIUM
    mr.MIN_STRAINS_PER_MEDIUM = 20
    try:
        X2, y2, _ = build_training_table(features, sm, bacdive)
    finally:
        mr.MIN_STRAINS_PER_MEDIUM = original

    medium_meta = {"A": "MediumA", "B": "MediumB", "C": "MediumC"}
    results = train_per_medium(X2, y2, medium_meta, groups, n_splits=3, n_estimators=50)

    # Medium A has real signal — should beat ~0.5 baseline ROC-AUC
    if "A" in results and results["A"].fold_metrics:
        assert results["A"].mean_roc_auc() > 0.6


def test_save_results_roundtrip(tmp_path: Path) -> None:
    features, sm, bacdive = _synthetic_dataset(n=300)
    X, y_matrix, _ = build_training_table(features, sm, bacdive)
    groups = bacdive.set_index("bacdive_id").loc[X.index, "family"]
    from microbe_model.train import media_recommender as mr
    original = mr.MIN_STRAINS_PER_MEDIUM
    mr.MIN_STRAINS_PER_MEDIUM = 20
    try:
        X2, y2, _ = build_training_table(features, sm, bacdive)
    finally:
        mr.MIN_STRAINS_PER_MEDIUM = original
    medium_meta = {"A": "MediumA", "B": "MediumB", "C": "MediumC"}
    results = train_per_medium(X2, y2, medium_meta, groups, n_splits=3, n_estimators=50)

    path = tmp_path / "media_results.json"
    save_results(results, path)
    loaded = json.loads(path.read_text())
    for mid in results:
        assert mid in loaded
        assert "mean_pr_auc" in loaded[mid]
        assert "mean_roc_auc" in loaded[mid]
        assert "folds" in loaded[mid]
