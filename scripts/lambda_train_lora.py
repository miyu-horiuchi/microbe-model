"""Run LoRA fine-tuning on a prepared Lambda Labs instance or local GPU.

This is the reproducible replacement for the ad hoc one-off script used for the
first Lambda A100 run. It assumes the repo and required data files are present
on the machine where the command is executed.

Example:
    python -u scripts/lambda_train_lora.py --fold 0 --epochs 1
    python -u scripts/lambda_train_lora.py --fold 0 --epochs 1 --target-preset oxygen
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from microbe_model.train.lora_model import LoraModelConfig
from microbe_model.train.lora_trainer import OXY_LABEL_TO_INT, TrainConfig, train_lora


def _parse_oxy_class_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    weights = tuple(float(part) for part in raw.split(","))
    if len(weights) != len(OXY_LABEL_TO_INT):
        classes = ", ".join(OXY_LABEL_TO_INT)
        raise argparse.ArgumentTypeError(
            f"--oxy-class-weights must provide {len(OXY_LABEL_TO_INT)} values for: {classes}"
        )
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--esm-model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-dir", default="artifacts/lora")
    parser.add_argument("--sequences", default="data/marker_sequences.jsonl")
    parser.add_argument("--phenotypes", default="data/bacdive_phenotypes.parquet")
    parser.add_argument("--catalog", default="data/strain_catalog.parquet")
    parser.add_argument(
        "--target-preset",
        choices=("all", "oxygen"),
        default="all",
        help="Use all task losses, or train only the oxygen loss while still reporting all metrics.",
    )
    parser.add_argument(
        "--oxy-class-weights",
        type=_parse_oxy_class_weights,
        default=None,
        help=(
            "Comma-separated oxygen class weights in order "
            "aerobe,anaerobe,facultative_anaerobe,microaerobe. Example: 1,1.5,1,1"
        ),
    )
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    weights = {"temp": 1.0, "ph": 1.0, "salt": 1.0, "oxy": 1.0}
    if args.target_preset == "oxygen":
        weights = {"temp": 0.0, "ph": 0.0, "salt": 0.0, "oxy": 1.0}

    print(
        f"[lambda-lora] fold={args.fold} epochs={args.epochs} "
        f"model={args.esm_model} preset={args.target_preset}",
        flush=True,
    )
    print(f"[lambda-lora] device={device_name}", flush=True)
    print(f"[lambda-lora] target_weights={weights}", flush=True)
    print(f"[lambda-lora] oxy_class_weights={args.oxy_class_weights}", flush=True)

    results = train_lora(
        model_cfg=LoraModelConfig(esm_model_name=args.esm_model, lora_r=args.lora_r),
        train_cfg=TrainConfig(
            fold=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lora_lr=args.lora_lr,
            head_lr=args.head_lr,
            save_dir=args.save_dir,
            bf16=not args.no_bf16,
            temp_weight=weights["temp"],
            ph_weight=weights["ph"],
            salt_weight=weights["salt"],
            oxy_weight=weights["oxy"],
            oxy_class_weights=args.oxy_class_weights,
        ),
        sequences_path=Path(args.sequences),
        phenotypes_path=Path(args.phenotypes),
        catalog_path=Path(args.catalog),
        device=device,
    )
    print(f"[lambda-lora] done best={results.get('best')}", flush=True)


if __name__ == "__main__":
    main()
