"""Multi-task XGBoost baseline.

One model per phenotype target, evaluated with group K-fold by taxonomic family to prevent
leakage from closely-related strains. This is the v0 "what's the floor on tabular performance"
sanity check before we invest in transformers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y.astype(str))
    else:
        y_enc = y.to_numpy(dtype=float)

    n_unique_groups = groups.nunique()
    splits = min(n_splits, max(2, n_unique_groups))
    kfold = GroupKFold(n_splits=splits)

    result = TargetResult(target=target, task=task)
    importance_acc = np.zeros(len(feature_cols), dtype=float)
    fold_count = 0

    for tr_idx, te_idx in kfold.split(X, y_enc, groups):
        if task == "classification":
            n_classes = len(np.unique(y_enc[tr_idx]))
            if n_classes < 2:
                continue
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mlogloss",
            )
            model.fit(X.iloc[tr_idx], y_enc[tr_idx])
            preds = model.predict(X.iloc[te_idx])
            score = f1_score(y_enc[te_idx], preds, average="macro")
            metric = "f1_macro"
        else:
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(X.iloc[tr_idx], y_enc[tr_idx])
            preds = model.predict(X.iloc[te_idx])
            score = mean_absolute_error(y_enc[te_idx], preds)
            metric = "mae"

        result.folds.append(FoldResult(
            target=target,
            task=task,
            metric_name=metric,
            value=float(score),
            n_train=int(len(tr_idx)),
            n_test=int(len(te_idx)),
        ))
        importance_acc += model.feature_importances_
        fold_count += 1

    if fold_count:
        importance_acc /= fold_count
        result.importances = dict(zip(feature_cols, importance_acc.tolist(), strict=True))
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


def save_results(results: dict[str, TargetResult], path: Path) -> None:
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
