"""Tests for the Lambda LoRA runner CLI helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "lambda_train_lora.py"
    spec = importlib.util.spec_from_file_location("lambda_train_lora", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_oxy_class_weights() -> None:
    mod = _load_module()

    assert mod._parse_oxy_class_weights("1,1.5,1,1") == (1.0, 1.5, 1.0, 1.0)


def test_parse_oxy_class_weights_requires_one_weight_per_class() -> None:
    mod = _load_module()

    with pytest.raises(Exception, match="must provide 4 values"):
        mod._parse_oxy_class_weights("1,2")
