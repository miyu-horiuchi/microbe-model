"""Integration test: train_all + render_report end-to-end on synthetic data.

Exercises the contiguous-class fix in the classification path, the GroupKFold split,
and the markdown rendering — without needing real BacDive or NCBI data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from microbe_model.eval import render_report
from microbe_model.train.baseline import save_results, train_all


def _synthetic_dataset(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=feature_cols)
    df["bacdive_id"] = np.arange(n)
    df["genome_accession"] = [f"GCA_{i:09d}.1" for i in range(n)]
    df["family"] = [f"family_{i % 12}" for i in range(n)]
    df["genus"] = [f"genus_{i % 30}" for i in range(n)]
    df["species"] = [f"species_{i}" for i in range(n)]

    # Regression target with real signal in f0 + noise
    df["optimal_temperature_c"] = 30 + 5 * df["f0"] + rng.normal(scale=2, size=n)
    df["optimal_ph"] = 7.0 + 0.5 * df["f1"] + rng.normal(scale=0.1, size=n)
    df["salt_tolerance_pct"] = np.abs(2 + df["f2"] + rng.normal(scale=0.5, size=n))

    # Classification target — sometimes only some classes appear in a fold
    classes = ["aerobe", "anaerobe", "facultative", "microaerophile", "obligate aerobe"]
    df["oxygen_requirement"] = rng.choice(classes, size=n)

    # Inject some NaNs to mirror real BacDive sparsity
    nan_mask = rng.random(n) > 0.7
    df.loc[nan_mask, "optimal_ph"] = np.nan
    nan_mask = rng.random(n) > 0.5
    df.loc[nan_mask, "salt_tolerance_pct"] = np.nan

    df["group"] = df["family"]
    return df, feature_cols


def test_train_all_handles_classification_with_missing_classes_per_fold(tmp_path: Path) -> None:
    df, feature_cols = _synthetic_dataset(n=200)
    results = train_all(df, feature_cols, group_col_override="group")

    # All four targets should produce at least one fold of results
    for target in ("optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"):
        assert target in results
        assert results[target].folds, f"{target} produced no folds"

    # Regression should beat the always-mean baseline since f0 carries real signal
    temp_result = results["optimal_temperature_c"]
    baseline_mae = float(np.mean(np.abs(
        df["optimal_temperature_c"] - df["optimal_temperature_c"].mean()
    )))
    assert temp_result.mean() < baseline_mae, "model worse than always-mean baseline"


def test_render_report_writes_markdown(tmp_path: Path) -> None:
    df, feature_cols = _synthetic_dataset(n=150)
    results = train_all(df, feature_cols, group_col_override="group")

    results_path = tmp_path / "results.json"
    save_results(results, results_path)

    table_path = tmp_path / "table.parquet"
    df.to_parquet(table_path, index=False)

    out_path = tmp_path / "report.md"
    render_report(results_path, table_path, out_path)
    text = out_path.read_text()

    assert text.startswith("# microbe-model")
    assert "## Per-target results" in text
    assert "optimal_temperature_c" in text
    assert "oxygen_requirement" in text
    assert "## Known limitations" in text
    assert "## Next steps" in text


def test_save_results_roundtrip(tmp_path: Path) -> None:
    df, feature_cols = _synthetic_dataset(n=100)
    results = train_all(df, feature_cols, group_col_override="group")

    path = tmp_path / "results.json"
    save_results(results, path)
    loaded = json.loads(path.read_text())

    for target in results:
        assert target in loaded
        assert "task" in loaded[target]
        assert "mean_metric" in loaded[target]
        assert "folds" in loaded[target]
