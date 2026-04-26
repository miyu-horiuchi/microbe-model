"""Tests for the feature/target exploration helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from microbe_model.explore import class_mean_features, feature_target_correlations


def _synthetic_with_signal(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f_strong": rng.normal(size=n),
        "f_weak": rng.normal(size=n),
        "f_noise": rng.normal(size=n),
    })
    # Strong signal: target = 30 + 5 * f_strong + noise
    df["optimal_temperature_c"] = 30 + 5 * df["f_strong"] + rng.normal(scale=1, size=n)
    # Make some NaN
    df.loc[rng.random(n) > 0.95, "optimal_temperature_c"] = np.nan
    return df


def test_correlations_rank_strong_feature_first() -> None:
    df = _synthetic_with_signal()
    feats = ["f_strong", "f_weak", "f_noise"]
    corrs = feature_target_correlations(df, feats, "optimal_temperature_c", top_n=3)
    assert len(corrs) == 3
    # Strongest feature should be ranked first
    assert corrs[0]["feature"] == "f_strong"
    assert abs(corrs[0]["spearman_rho"]) > 0.5


def test_correlations_skip_low_n() -> None:
    df = pd.DataFrame({
        "f0": np.arange(10),
        "y": [1.0] * 5 + [np.nan] * 5,
    })
    assert feature_target_correlations(df, ["f0"], "y", top_n=5) == []


def test_correlations_skip_missing_target() -> None:
    df = pd.DataFrame({"f0": [1.0, 2.0]})
    assert feature_target_correlations(df, ["f0"], "missing_target", top_n=5) == []


def test_class_means_orders_most_separating_first() -> None:
    rng = np.random.default_rng(0)
    n = 200
    cls = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({
        "f_separating": np.where(cls == "a", 0, np.where(cls == "b", 5, 10)) + rng.normal(scale=0.5, size=n),
        "f_useless": rng.normal(size=n),
        "target": cls,
    })
    out = class_mean_features(df, ["f_separating", "f_useless"], "target", top_features=2)
    assert len(out) == 2
    assert out[0]["feature"] == "f_separating"
    assert out[1]["feature"] == "f_useless"
    # Class means should differ substantially for f_separating
    means = out[0]["means"]
    assert max(means.values()) - min(means.values()) > 5
