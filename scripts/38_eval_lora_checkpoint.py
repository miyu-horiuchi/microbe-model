"""Evaluate a LoRA checkpoint's oxygen classification behavior in detail.

Default inputs match the first all-task Lambda run:
    python scripts/38_eval_lora_checkpoint.py

Writes:
    artifacts/lora/fold0_oxygen_diagnostics.json
    artifacts/lora/fold0_oxygen_diagnostics.md

The CLI path needs the LoRA/embedding dependencies (`torch`, `transformers`,
`peft`). The metric helpers are intentionally dependency-light so they can be
unit-tested without loading ESM-2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CLASSES = ["aerobe", "anaerobe", "facultative_anaerobe", "microaerobe"]


def _round_float(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def compute_oxygen_diagnostics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    classes: list[str],
    *,
    top_n_errors: int = 25,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Compute confusion matrix, per-class scores, and confident mistakes."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if probabilities.shape[0] != labels.shape[0] or labels.shape[0] != len(rows):
        raise ValueError("probabilities, labels, and rows must have matching lengths")
    if probabilities.shape[1] != len(classes):
        raise ValueError("probabilities width must match number of classes")

    preds = probabilities.argmax(axis=1)
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true_idx, pred_idx in zip(labels.astype(int), preds.astype(int), strict=True):
        confusion[true_idx, pred_idx] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for idx, name in enumerate(classes):
        tp = int(confusion[idx, idx])
        support = int(confusion[idx, :].sum())
        predicted = int(confusion[:, idx].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": _round_float(precision),
            "recall": _round_float(recall),
            "f1": _round_float(f1),
            "support": support,
            "predicted": predicted,
        }

    wrong_predictions: list[dict[str, Any]] = []
    for i, (true_idx, pred_idx) in enumerate(zip(labels.astype(int), preds.astype(int), strict=True)):
        if true_idx == pred_idx:
            continue
        pred_prob = float(probabilities[i, pred_idx])
        true_prob = float(probabilities[i, true_idx])
        row = rows[i]
        wrong_predictions.append({
            "bacdive_id": row.get("bacdive_id"),
            "genome_accession": row.get("genome_accession"),
            "group": row.get("group"),
            "true": classes[true_idx],
            "pred": classes[pred_idx],
            "confidence": _round_float(pred_prob),
            "true_probability": _round_float(true_prob),
            "margin": _round_float(pred_prob - true_prob),
        })
    wrong_predictions.sort(key=lambda item: (item["confidence"], item["margin"]), reverse=True)

    n = int(labels.shape[0])
    accuracy = float((preds == labels).mean()) if n else 0.0
    out: dict[str, Any] = {
        "checkpoint": checkpoint,
        "n": n,
        "classes": classes,
        "accuracy": _round_float(accuracy),
        "macro_f1": _round_float(float(np.mean(f1_values)) if f1_values else 0.0),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "wrong_predictions": wrong_predictions[:top_n_errors],
    }
    return out


def render_markdown(diagnostics: dict[str, Any]) -> str:
    """Render diagnostics as a compact Markdown report."""
    classes = diagnostics["classes"]
    lines = [
        "# LoRA Oxygen Diagnostics",
        "",
        f"Checkpoint: `{diagnostics.get('checkpoint')}`",
        "",
        f"- Labeled validation rows: `{diagnostics['n']}`",
        f"- Accuracy: `{diagnostics['accuracy']:.4f}`",
        f"- Macro F1: `{diagnostics['macro_f1']:.4f}`",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support | Predicted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls in classes:
        m = diagnostics["per_class"][cls]
        lines.append(
            f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['support']} | {m['predicted']} |"
        )

    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "| True \\ Pred | " + " | ".join(classes) + " |",
        "|---" + "|---:" * len(classes) + "|",
    ])
    for cls, row in zip(classes, diagnostics["confusion_matrix"], strict=True):
        lines.append("| " + cls + " | " + " | ".join(str(int(v)) for v in row) + " |")

    lines.extend([
        "",
        "## High-Confidence Wrong Predictions",
        "",
        "| BacDive ID | Genome | Group | True | Pred | Confidence | True Prob. | Margin |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ])
    wrong = diagnostics.get("wrong_predictions", [])
    if not wrong:
        lines.append("| - | - | - | - | - | - | - | - |")
    for item in wrong:
        lines.append(
            f"| {item.get('bacdive_id')} | {item.get('genome_accession')} | "
            f"{item.get('group') or ''} | {item['true']} | {item['pred']} | "
            f"{item['confidence']:.4f} | {item['true_probability']:.4f} | "
            f"{item['margin']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/lora/fold0_best.pt")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequences", default="data/marker_sequences.jsonl")
    parser.add_argument("--phenotypes", default="data/bacdive_phenotypes.parquet")
    parser.add_argument("--catalog", default="data/strain_catalog.parquet")
    parser.add_argument("--top-n-errors", type=int, default=25)
    parser.add_argument("--output-json", default="artifacts/lora/fold0_oxygen_diagnostics.json")
    parser.add_argument("--output-md", default="artifacts/lora/fold0_oxygen_diagnostics.md")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, else cpu.")
    return parser.parse_args()


def _evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from microbe_model.train.lora_model import LoraModelConfig, OXYGEN_CLASSES, PhenoLoRAModel
    from microbe_model.train.lora_trainer import _build_dataset, _collate, _group_kfold_split

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = LoraModelConfig(**checkpoint["model_cfg"])
    fold = args.fold if args.fold is not None else int(checkpoint.get("train_cfg", {}).get("fold", 0))

    rows = _build_dataset(Path(args.sequences), Path(args.phenotypes), Path(args.catalog))
    _, val_rows = _group_kfold_split(rows, n_splits=5, fold=fold)

    model = PhenoLoRAModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    probs_out: list[np.ndarray] = []
    labels_out: list[int] = []
    rows_out: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(val_rows), args.batch_size):
            chunk = val_rows[start : start + args.batch_size]
            batch = _collate(chunk)
            preds = model(batch["genomes"], device=device)
            probs = torch.softmax(preds["oxy"], dim=-1).detach().cpu().float().numpy()
            labels = batch["labels"]["oxy"].cpu().numpy().astype(int)
            masks = batch["label_mask"]["oxy"].cpu().numpy().astype(bool)
            if masks.any():
                probs_out.append(probs[masks])
                labels_out.extend(labels[masks].tolist())
                rows_out.extend([row for row, keep in zip(chunk, masks, strict=True) if keep])

    if probs_out:
        probabilities = np.concatenate(probs_out, axis=0)
    else:
        probabilities = np.zeros((0, len(OXYGEN_CLASSES)), dtype=float)
    labels = np.array(labels_out, dtype=int)
    return compute_oxygen_diagnostics(
        probabilities,
        labels,
        rows_out,
        list(OXYGEN_CLASSES),
        top_n_errors=args.top_n_errors,
        checkpoint=str(checkpoint_path),
    )


def main() -> None:
    args = parse_args()
    diagnostics = _evaluate_checkpoint(args)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(diagnostics, indent=2) + "\n")
    out_md.write_text(render_markdown(diagnostics))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"oxygen macro_f1={diagnostics['macro_f1']:.4f}  accuracy={diagnostics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
