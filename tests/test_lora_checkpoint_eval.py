"""Tests for LoRA checkpoint oxygen diagnostics helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "38_eval_lora_checkpoint.py"
    spec = importlib.util.spec_from_file_location("eval_lora_checkpoint", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_oxygen_diagnostics_reports_confusion_and_per_class_metrics() -> None:
    mod = _load_module()
    classes = ["aerobe", "anaerobe", "facultative_anaerobe", "microaerobe"]
    labels = np.array([0, 0, 1, 1, 2, 3])
    probs = np.array([
        [0.90, 0.05, 0.03, 0.02],
        [0.20, 0.70, 0.05, 0.05],
        [0.05, 0.85, 0.05, 0.05],
        [0.10, 0.60, 0.20, 0.10],
        [0.05, 0.10, 0.80, 0.05],
        [0.10, 0.20, 0.60, 0.10],
    ])
    rows = [{"bacdive_id": i, "genome_accession": f"G{i}"} for i in range(len(labels))]

    out = mod.compute_oxygen_diagnostics(probs, labels, rows, classes, top_n_errors=2)

    assert out["n"] == 6
    assert out["accuracy"] == pytest.approx(4 / 6)
    assert out["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
    ]
    assert out["per_class"]["aerobe"]["recall"] == 0.5
    assert out["per_class"]["anaerobe"]["precision"] == pytest.approx(2 / 3)
    assert out["per_class"]["microaerobe"]["f1"] == 0.0
    assert out["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 2 / 3 + 0.0) / 4)
    assert out["macro_f1_all_classes"] == pytest.approx(out["macro_f1"])
    assert out["wrong_predictions"][0]["confidence"] == 0.7
    assert out["wrong_predictions"][0]["true"] == "aerobe"
    assert out["wrong_predictions"][0]["pred"] == "anaerobe"


def test_macro_f1_ignores_zero_support_classes() -> None:
    mod = _load_module()
    classes = ["aerobe", "anaerobe", "facultative_anaerobe", "microaerobe"]
    labels = np.array([0, 1])
    probs = np.array([
        [0.90, 0.10, 0.00, 0.00],
        [0.20, 0.80, 0.00, 0.00],
    ])
    rows = [{"bacdive_id": i, "genome_accession": f"G{i}"} for i in range(len(labels))]

    out = mod.compute_oxygen_diagnostics(probs, labels, rows, classes)

    assert out["macro_f1"] == 1.0
    assert out["macro_f1_all_classes"] == 0.5


def test_render_markdown_includes_key_sections() -> None:
    mod = _load_module()
    diagnostics = {
        "checkpoint": "artifacts/lora/fold0_best.pt",
        "n": 2,
        "accuracy": 0.5,
        "macro_f1": 0.333333333,
        "macro_f1_all_classes": 0.333333333,
        "confusion_matrix": [[1, 0], [1, 0]],
        "classes": ["aerobe", "anaerobe"],
        "per_class": {
            "aerobe": {
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.666666667,
                "support": 1,
                "predicted": 2,
            },
            "anaerobe": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 1,
                "predicted": 0,
            },
        },
        "wrong_predictions": [
            {
                "bacdive_id": 42,
                "genome_accession": "GCA_42",
                "true": "anaerobe",
                "pred": "aerobe",
                "confidence": 0.9,
                "true_probability": 0.1,
                "margin": 0.8,
            }
        ],
    }

    md = mod.render_markdown(diagnostics)

    assert "# LoRA Oxygen Diagnostics" in md
    assert "| True \\ Pred | aerobe | anaerobe |" in md
    assert "| anaerobe | 1 | 0 |" in md
    assert "GCA_42" in md
