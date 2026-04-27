"""Train per-medium binary classifiers to recommend cultivation media for a genome.

Setup:
  - Filter to media used by >= MIN_STRAINS_PER_MEDIUM strains (default 100).
  - For each such medium m, build a binary label: y_i = 1 if strain i has a
    growth=yes link to m, else 0.
  - Train one XGBoost classifier per medium with GroupKFold by family.
  - At inference, output a (n_strains × n_media) probability matrix.

The deliverable: given a new (possibly uncultured) genome, output the top-K media
ranked by predicted probability. This is the "what should I try first?" output
microbiologists actually want.

Limitations:
  - All BacDive `culture medium` entries are growth=yes — we have positive
    examples but no explicit negatives. We construct negatives from strains that
    have *some* media link but not this one. This may bias toward media that are
    just under-recorded.
  - No concentration prediction yet — only recipe selection. v1 will add a
    secondary regression head that adjusts compound concentrations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

MIN_STRAINS_PER_MEDIUM = 100


@dataclass
class MediumModelResult:
    medium_id: str
    medium_name: str
    n_positives: int
    n_negatives: int
    fold_metrics: list[dict] = field(default_factory=list)

    def mean_pr_auc(self) -> float:
        if not self.fold_metrics:
            return float("nan")
        return float(np.mean([m["pr_auc"] for m in self.fold_metrics]))

    def mean_roc_auc(self) -> float:
        if not self.fold_metrics:
            return float("nan")
        return float(np.mean([m["roc_auc"] for m in self.fold_metrics]))


def build_training_table(
    features: pd.DataFrame,
    strain_media: pd.DataFrame,
    bacdive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (X, y_matrix, medium_ids) for media meeting the strain-count threshold.

    X: (n_strains × n_features) feature DataFrame, indexed by bacdive_id
    y_matrix: (n_strains × n_media) {0,1} DataFrame, columns are medium_ids
    """
    # Strains with both genome features and at least one positive medium link
    strain_ids = sorted(set(features["bacdive_id"]).intersection(set(strain_media["bacdive_id"])))
    if not strain_ids:
        raise ValueError("No overlap between feature table and strain_media links.")

    X = features[features["bacdive_id"].isin(strain_ids)].set_index("bacdive_id").sort_index()
    feature_cols = [c for c in X.columns if c not in {"genome_accession"}]
    X = X[feature_cols]

    # Build sparse positive-link table → wide y matrix
    sm = strain_media[strain_media["bacdive_id"].isin(strain_ids)]
    sm = sm[sm["growth"] == "yes"]
    counts = sm.groupby("medium_id").size()
    keep_media = counts[counts >= MIN_STRAINS_PER_MEDIUM].index.tolist()
    sm = sm[sm["medium_id"].isin(keep_media)]

    y_matrix = (
        sm.assign(_one=1)
          .pivot_table(index="bacdive_id", columns="medium_id", values="_one", fill_value=0)
          .reindex(index=X.index, columns=keep_media, fill_value=0)
          .astype(np.uint8)
    )

    return X, y_matrix, keep_media


def train_per_medium(
    X: pd.DataFrame,
    y_matrix: pd.DataFrame,
    medium_metadata: dict[str, str],
    groups: pd.Series,
    *,
    n_splits: int = 5,
    n_estimators: int = 200,
    max_depth: int = 5,
) -> dict[str, MediumModelResult]:
    """Train one classifier per medium with GroupKFold by `groups` (e.g. taxonomic family)."""
    results: dict[str, MediumModelResult] = {}
    splits = min(n_splits, max(2, groups.nunique()))
    kfold = GroupKFold(n_splits=splits)

    for medium_id in y_matrix.columns:
        y = y_matrix[medium_id].to_numpy()
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        result = MediumModelResult(
            medium_id=str(medium_id),
            medium_name=medium_metadata.get(str(medium_id), ""),
            n_positives=n_pos,
            n_negatives=n_neg,
        )

        # Need both classes in train/test
        for fold_idx, (tr_idx, te_idx) in enumerate(kfold.split(X, y, groups)):
            y_tr = y[tr_idx]
            y_te = y[te_idx]
            if y_tr.sum() < 5 or y_te.sum() < 1:
                continue

            scale_pos_weight = (y_tr == 0).sum() / max(1, y_tr.sum())
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                tree_method="hist",
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
            )
            model.fit(X.iloc[tr_idx], y_tr)
            proba = model.predict_proba(X.iloc[te_idx])[:, 1]
            try:
                roc = roc_auc_score(y_te, proba)
                pr = average_precision_score(y_te, proba)
            except ValueError:
                continue
            result.fold_metrics.append({
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "n_test_positives": int(y_te.sum()),
                "roc_auc": float(roc),
                "pr_auc": float(pr),
            })

        results[str(medium_id)] = result
    return results


def save_results(results: dict[str, MediumModelResult], path: Path) -> None:
    payload = {
        mid: {
            "medium_name": r.medium_name,
            "n_positives": r.n_positives,
            "n_negatives": r.n_negatives,
            "mean_pr_auc": r.mean_pr_auc(),
            "mean_roc_auc": r.mean_roc_auc(),
            "folds": r.fold_metrics,
        }
        for mid, r in results.items()
    }
    path.write_text(json.dumps(payload, indent=2))
