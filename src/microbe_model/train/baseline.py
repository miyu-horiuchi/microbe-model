"""Multi-task XGBoost baseline.

One model per phenotype target, evaluated with group K-fold by taxonomic family to prevent
leakage from closely-related strains. This is the v0 "what's the floor on tabular performance"
sanity check before we invest in transformers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from microbe_model import config


@dataclass
class FoldResult:
    target: str
    task: str
    metric_name: str
    value: float
    n_train: int
    n_test: int


@dataclass
class TargetResult:
    target: str
    task: str
    folds: list[FoldResult] = field(default_factory=list)
    importances: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame | None = None  # one row per test-fold sample

    def mean(self) -> float:
        return float(np.mean([f.value for f in self.folds])) if self.folds else float("nan")


def _select_xy(df: pd.DataFrame, target: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    mask = df[target].notna()
    return df.loc[mask, feature_cols], df.loc[mask, target]


def train_target(
    df: pd.DataFrame,
    target: str,
    task: str,
    feature_cols: list[str],
    group_col: str = "family",
    n_splits: int = 5,
) -> TargetResult:
    X, y = _select_xy(df, target, feature_cols)
    groups = df.loc[X.index, group_col].fillna("__unknown__")
    if len(X) < n_splits * 2:
        return TargetResult(target=target, task=task)

    if task == "classification":
        y_str = y.astype(str).to_numpy()
    else:
        y_arr = y.to_numpy(dtype=float)

    n_unique_groups = groups.nunique()
    splits = min(n_splits, max(2, n_unique_groups))
    kfold = GroupKFold(n_splits=splits)

    result = TargetResult(target=target, task=task)
    importance_acc = np.zeros(len(feature_cols), dtype=float)
    fold_count = 0
    pred_rows: list[dict[str, Any]] = []

    split_iter = kfold.split(X, y_str if task == "classification" else y_arr, groups)
    for fold_idx, (tr_idx, te_idx) in enumerate(split_iter):
        if task == "classification":
            # Per-fold encoding: ensures contiguous 0..k-1 labels for xgboost.
            # Test samples whose class never appears in train are dropped from eval.
            fold_encoder = LabelEncoder()
            y_tr = fold_encoder.fit_transform(y_str[tr_idx])
            if len(fold_encoder.classes_) < 2:
                continue
            known = set(fold_encoder.classes_)
            te_mask = np.array([c in known for c in y_str[te_idx]])
            if te_mask.sum() == 0:
                continue
            y_te = fold_encoder.transform(y_str[te_idx][te_mask])

            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mlogloss",
            )
            model.fit(X.iloc[tr_idx], y_tr)
            preds = model.predict(X.iloc[te_idx][te_mask])
            score = f1_score(y_te, preds, average="macro")
            metric = "f1_macro"
            n_test = int(te_mask.sum())
            test_indices = X.iloc[te_idx].index[te_mask]
            pred_labels = fold_encoder.inverse_transform(preds)
            obs_labels = y_str[te_idx][te_mask]
            for idx, p, o in zip(test_indices, pred_labels, obs_labels, strict=True):
                pred_rows.append({
                    "fold": fold_idx, "row_idx": int(idx),
                    "predicted": str(p), "observed": str(o),
                })
        else:
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(X.iloc[tr_idx], y_arr[tr_idx])
            preds = model.predict(X.iloc[te_idx])
            score = mean_absolute_error(y_arr[te_idx], preds)
            metric = "mae"
            n_test = int(len(te_idx))
            test_indices = X.iloc[te_idx].index
            for idx, p, o in zip(test_indices, preds, y_arr[te_idx], strict=True):
                pred_rows.append({
                    "fold": fold_idx, "row_idx": int(idx),
                    "predicted": float(p), "observed": float(o),
                })

        result.folds.append(FoldResult(
            target=target,
            task=task,
            metric_name=metric,
            value=float(score),
            n_train=int(len(tr_idx)),
            n_test=n_test,
        ))
        importance_acc += model.feature_importances_
        fold_count += 1

    if fold_count:
        importance_acc /= fold_count
        result.importances = dict(zip(feature_cols, importance_acc.tolist(), strict=True))
    if pred_rows:
        result.predictions = pd.DataFrame(pred_rows)
    return result


def train_all(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    group_col_override: str | None = None,
) -> dict[str, TargetResult]:
    results: dict[str, TargetResult] = {}
    group_col = group_col_override or "family"
    for target, task in config.PHENOTYPE_TARGETS.items():
        if target not in df.columns:
            continue
        results[target] = train_target(df, target, task, feature_cols, group_col=group_col)
    return results


def save_results(
    results: dict[str, TargetResult],
    path: Path,
    *,
    predictions_path: Path | None = None,
) -> None:
    payload = {
        target: {
            "task": r.task,
            "mean_metric": r.mean(),
            "folds": [f.__dict__ for f in r.folds],
            "top_features": dict(
                sorted(r.importances.items(), key=lambda kv: kv[1], reverse=True)[:20]
            ),
        }
        for target, r in results.items()
    }
    path.write_text(json.dumps(payload, indent=2))

    if predictions_path is not None:
        frames = []
        for target, r in results.items():
            if r.predictions is None or r.predictions.empty:
                continue
            df = r.predictions.copy()
            df["target"] = target
            df["task"] = r.task
            frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(predictions_path, index=False)
