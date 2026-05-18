"""Tests for LoRA multitask loss helpers."""
from __future__ import annotations

import pytest
import torch

from microbe_model.train.lora_model import masked_multitask_loss

torch.set_num_threads(1)


def _empty_regression_preds() -> dict[str, torch.Tensor]:
    return {
        "temp": torch.zeros(2),
        "ph": torch.zeros(2),
        "salt": torch.zeros(2),
    }


def test_masked_multitask_loss_applies_oxygen_class_weights() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    preds = {**_empty_regression_preds(), "oxy": logits}
    labels = {
        "temp": torch.zeros(2),
        "ph": torch.zeros(2),
        "salt": torch.zeros(2),
        "oxy": torch.tensor([0, 1]),
    }
    label_mask = {
        "temp": torch.zeros(2),
        "ph": torch.zeros(2),
        "salt": torch.zeros(2),
        "oxy": torch.ones(2),
    }
    weights = torch.tensor([1.0, 3.0, 1.0, 1.0])

    loss, per_target = masked_multitask_loss(
        preds,
        labels,
        label_mask,
        target_weights={"temp": 0.0, "ph": 0.0, "salt": 0.0, "oxy": 1.0},
        oxy_class_weights=(1.0, 3.0, 1.0, 1.0),
    )

    expected = torch.nn.functional.cross_entropy(
        logits,
        labels["oxy"],
        weight=weights,
        reduction="none",
    ).mean()
    assert loss.item() == pytest.approx(expected.item())
    assert per_target["oxy"] == pytest.approx(expected.item())
