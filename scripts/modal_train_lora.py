"""LoRA fine-tune ESM-2 on phenotype prediction (Modal A100/A10G).

Loads data/marker_sequences.jsonl (produced by scripts/36_extract_marker_sequences.py)
+ data/bacdive_phenotypes.parquet + data/strain_catalog.parquet, runs LoRA fine-tuning
for one group-K-fold split, and writes artifacts/lora/fold{N}_results.json back to
the local checkout via Modal's volume mount.

Usage:
    modal run scripts/modal_train_lora.py --fold 0 --epochs 3 --smoke      # tiny smoke
    modal run scripts/modal_train_lora.py --fold 0 --epochs 3              # real run
    modal run scripts/modal_train_lora.py --fold 0 --esm-model facebook/esm2_t30_150M_UR50D
"""
from __future__ import annotations

from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.30",
        "peft>=0.11",
        "scikit-learn>=1.4",
        "pandas>=2.2",
        "pyarrow>=15",
        "numpy>=1.26",
    ])
    # Build-step local-file copies must come before runtime python_source mounts.
    .add_local_file("data/marker_sequences.jsonl", "/data/marker_sequences.jsonl", copy=True)
    .add_local_file("data/bacdive_phenotypes.parquet", "/data/bacdive_phenotypes.parquet", copy=True)
    .add_local_file("data/strain_catalog.parquet", "/data/strain_catalog.parquet", copy=True)
    .add_local_python_source("microbe_model")
)

app = modal.App("microbe-lora-train", image=image)


@app.function(
    gpu="A10G",
    timeout=3600 * 12,
    memory=32_768,
)
def train(
    fold: int = 0,
    epochs: int = 3,
    esm_model: str = "facebook/esm2_t12_35M_UR50D",
    lora_r: int = 8,
    lora_lr: float = 1e-4,
    head_lr: float = 1e-3,
    batch_size: int = 2,
    grad_accum: int = 8,
    smoke_limit: int = 0,
) -> dict:
    """Run one fold of LoRA training inside a Modal container."""
    import json

    import torch

    from microbe_model.train.lora_model import LoraModelConfig
    from microbe_model.train.lora_trainer import TrainConfig, train_lora

    sequences_path = Path("/data/marker_sequences.jsonl")
    if smoke_limit > 0:
        # Truncate the local jsonl to smoke_limit lines (in-container only).
        truncated = Path("/data/marker_sequences_smoke.jsonl")
        with open(sequences_path) as src, open(truncated, "w") as dst:
            for i, line in enumerate(src):
                if i >= smoke_limit:
                    break
                dst.write(line)
        sequences_path = truncated
        print(f"[smoke] truncated to {smoke_limit} sequences", flush=True)

    model_cfg = LoraModelConfig(esm_model_name=esm_model, lora_r=lora_r)
    train_cfg = TrainConfig(
        fold=fold,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_lr=lora_lr,
        head_lr=head_lr,
        save_dir="/artifacts/lora",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = train_lora(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        sequences_path=sequences_path,
        phenotypes_path=Path("/data/bacdive_phenotypes.parquet"),
        catalog_path=Path("/data/strain_catalog.parquet"),
        device=device,
    )

    # Read back the json + checkpoint for return.
    json_path = Path(train_cfg.save_dir) / f"fold{fold}_results.json"
    payload = json.load(open(json_path))
    ckpt_path = Path(train_cfg.save_dir) / f"fold{fold}_best.pt"
    ckpt_bytes = ckpt_path.read_bytes() if ckpt_path.exists() else b""
    return {"results": payload, "ckpt_bytes": ckpt_bytes}


@app.local_entrypoint()
def main(
    fold: int = 0,
    epochs: int = 3,
    esm_model: str = "facebook/esm2_t12_35M_UR50D",
    lora_r: int = 8,
    lora_lr: float = 1e-4,
    head_lr: float = 1e-3,
    batch_size: int = 2,
    grad_accum: int = 8,
    smoke: bool = False,
):
    """Dispatch one fold to Modal, capture results locally."""
    import json

    smoke_limit = 200 if smoke else 0
    print(f"[modal-lora] fold={fold} epochs={epochs} model={esm_model} "
          f"smoke={smoke}", flush=True)

    payload = train.remote(
        fold=fold,
        epochs=epochs,
        esm_model=esm_model,
        lora_r=lora_r,
        lora_lr=lora_lr,
        head_lr=head_lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        smoke_limit=smoke_limit,
    )

    out_dir = Path("artifacts/lora")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"fold{fold}_results{'_smoke' if smoke else ''}.json"
    with open(json_path, "w") as fh:
        json.dump(payload["results"], fh, indent=2)
    print(f"[modal-lora] wrote {json_path}", flush=True)

    if payload["ckpt_bytes"]:
        ckpt_path = out_dir / f"fold{fold}_best{'_smoke' if smoke else ''}.pt"
        ckpt_path.write_bytes(payload["ckpt_bytes"])
        print(f"[modal-lora] wrote {ckpt_path}  ({len(payload['ckpt_bytes'])/1e6:.1f} MB)",
              flush=True)
