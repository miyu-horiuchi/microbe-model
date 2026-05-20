"""Tests for the hybrid prediction script helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "39_predict_hybrid.py"
    spec = importlib.util.spec_from_file_location("predict_hybrid", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_join_features_and_sequences_inner_and_left() -> None:
    mod = _load_module()
    features = pd.DataFrame([
        {"genome_accession": "G1", "feature": 1.0},
        {"genome_accession": "G2", "feature": 2.0},
    ])
    sequences = pd.DataFrame([
        {"genome_accession": "G1", "by_category": {"oxygen": ["AAA"]}},
    ])

    inner = mod.join_features_and_sequences(features, sequences, how="inner")
    assert inner["genome_accession"].tolist() == ["G1"]
    assert inner.loc[0, "by_category"] == {"oxygen": ["AAA"]}

    left = mod.join_features_and_sequences(features, sequences, how="left")
    assert left["genome_accession"].tolist() == ["G1", "G2"]
    assert pd.isna(left.loc[1, "by_category"])


def test_join_features_and_sequences_validates_required_columns() -> None:
    mod = _load_module()
    features = pd.DataFrame([{"genome_accession": "G1"}])
    sequences = pd.DataFrame([{"genome_accession": "G1"}])

    with pytest.raises(ValueError, match="by_category"):
        mod.join_features_and_sequences(features, sequences)


def test_build_hybrid_predictions_orders_core_columns() -> None:
    mod = _load_module()
    joined = pd.DataFrame([
        {"bacdive_id": 1, "genome_accession": "G1", "feature": 1.0},
    ])
    tabular = pd.DataFrame([
        {
            "pred_optimal_temperature_c": 30.0,
            "pred_optimal_temperature_c_low_80": 25.0,
            "pred_optimal_temperature_c_high_80": 35.0,
            "pred_optimal_ph": 7.0,
            "pred_optimal_ph_low_80": 6.5,
            "pred_optimal_ph_high_80": 7.5,
            "pred_salt_tolerance_pct": 1.2,
            "pred_salt_tolerance_pct_low_80": 0.4,
            "pred_salt_tolerance_pct_high_80": 2.0,
        }
    ])
    oxygen = pd.DataFrame([
        {
            "pred_oxygen_requirement": "aerobe",
            "pred_oxygen_requirement_confidence": 0.9,
            "pred_oxygen_requirement_source": "lora",
            "pred_oxygen_requirement_prob_aerobe": 0.9,
            "pred_oxygen_requirement_prob_anaerobe": 0.1,
        }
    ])

    out = mod.build_hybrid_predictions(
        joined,
        tabular_predictions=tabular,
        oxygen_predictions=oxygen,
    )

    assert out.columns[:5].tolist() == [
        "bacdive_id",
        "genome_accession",
        "pred_optimal_temperature_c",
        "pred_optimal_temperature_c_low_80",
        "pred_optimal_temperature_c_high_80",
    ]
    assert out.loc[0, "pred_oxygen_requirement"] == "aerobe"
    assert out.loc[0, "pred_oxygen_requirement_prob_aerobe"] == 0.9


def test_build_hybrid_predictions_falls_back_to_existing_oxygen() -> None:
    mod = _load_module()
    joined = pd.DataFrame([
        {
            "bacdive_id": 1,
            "genome_accession": "G1",
            "pred_oxygen_requirement": "anaerobe",
            "pred_oxygen_requirement_confidence": 0.7,
        },
    ])
    tabular = pd.DataFrame([
        {"pred_optimal_temperature_c": 30.0},
    ])
    oxygen = pd.DataFrame(index=joined.index)

    out = mod.build_hybrid_predictions(
        joined,
        tabular_predictions=tabular,
        oxygen_predictions=oxygen,
    )

    assert out.loc[0, "pred_oxygen_requirement"] == "anaerobe"
    assert out.loc[0, "pred_oxygen_requirement_confidence"] == 0.7
    assert out.loc[0, "pred_oxygen_requirement_source"] == "tabular"


def test_chunk_output_path_uses_range_and_final_suffix(tmp_path: Path) -> None:
    mod = _load_module()

    path = mod.chunk_output_path(
        Path("artifacts/hybrid_predictions.parquet"),
        tmp_path,
        250,
        500,
    )

    assert path == tmp_path / "hybrid_predictions_000250_000500.parquet"
