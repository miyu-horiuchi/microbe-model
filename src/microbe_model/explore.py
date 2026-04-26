"""Lightweight data exploration helpers.

Computes Pearson + Spearman correlations between features and regression targets,
and class-mean differences for classification targets. Used by the eval renderer
to surface biologically interpretable signals (e.g. "IVYWREL fraction correlates
with optimal_temperature_c at r=0.71" — confirms the thermophile signature).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def feature_target_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    *,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """For a regression target, return top features by |Spearman correlation|."""
    if target not in df.columns:
        return []
    sub = df[df[target].notna()]
    if len(sub) < 30:
        return []
    rows = []
    y = sub[target].to_numpy(dtype=float)
    for col in feature_cols:
        x = pd.to_numeric(sub[col], errors="coerce").to_numpy()
        mask = ~np.isnan(x)
        if mask.sum() < 30:
            continue
        try:
            rho, p = spearmanr(x[mask], y[mask])
        except Exception:  # noqa: BLE001
            continue
        if np.isnan(rho):
            continue
        rows.append({"feature": col, "spearman_rho": float(rho), "p_value": float(p)})
    rows.sort(key=lambda r: abs(r["spearman_rho"]), reverse=True)
    return rows[:top_n]


def class_mean_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    *,
    top_features: int = 5,
) -> list[dict[str, Any]]:
    """For a classification target, return per-class means for the top-N most varying features.

    Variance is measured across class means — features whose mean differs most across classes
    are most informative.
    """
    if target not in df.columns:
        return []
    sub = df[df[target].notna()]
    if sub[target].nunique() < 2:
        return []

    # Rank features by F-statistic-style metric: variance of class means / pooled variance
    feature_scores = []
    for col in feature_cols:
        vals = pd.to_numeric(sub[col], errors="coerce")
        if vals.notna().sum() < 30:
            continue
        between = sub.assign(_x=vals).groupby(target, dropna=False)["_x"].mean().var()
        within = vals.var()
        if between is None or within is None or within == 0 or pd.isna(between):
            continue
        feature_scores.append((col, float(between / within)))
    feature_scores.sort(key=lambda kv: kv[1], reverse=True)

    output: list[dict[str, Any]] = []
    for col, score in feature_scores[:top_features]:
        means_by_class = (
            sub.assign(_x=pd.to_numeric(sub[col], errors="coerce"))
               .groupby(target, dropna=False)["_x"]
               .mean()
               .to_dict()
        )
        output.append({"feature": col, "score": score, "means": means_by_class})
    return output
