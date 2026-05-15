"""Training loop for PhenoLoRAModel — multi-task, masked, group-K-fold compatible."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn, optim

from microbe_model.train.lora_model import (
    CATEGORIES,
    OXYGEN_CLASSES,
    LoraModelConfig,
    PhenoLoRAModel,
    masked_multitask_loss,
)

OXY_LABEL_TO_INT = {c: i for i, c in enumerate(OXYGEN_CLASSES)}


@dataclass
class TrainConfig:
    fold: int = 0
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    lora_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    bf16: bool = True
    max_proteins_per_category: int = 16
    save_dir: str = "artifacts/lora"
    grad_clip: float = 1.0


def _build_dataset(
    sequences_path: Path,
    phenotypes_path: Path,
    catalog_path: Path,  # kept for symmetry; only used if pheno lacks family/genus
) -> list[dict]:
    """Join marker sequences with phenotype labels + family groups → list of records."""
    pheno = pd.read_parquet(phenotypes_path)
    if "family" not in pheno.columns or "genus" not in pheno.columns:
        catalog = pd.read_parquet(catalog_path)
        keep = [c for c in ("family", "genus", "species") if c not in pheno.columns]
        pheno = pheno.merge(
            catalog[["bacdive_id", *keep]].drop_duplicates("bacdive_id"),
            on="bacdive_id",
            how="left",
        )

    rows: list[dict] = []
    with open(sequences_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bacdive_id = int(r["bacdive_id"])
            sub = pheno[pheno["bacdive_id"] == bacdive_id]
            if sub.empty:
                continue
            p_row = sub.iloc[0]

            def _val(col: str):
                v = p_row.get(col)
                if pd.isna(v):
                    return None, 0
                return v, 1

            temp_v, temp_m = _val("optimal_temperature_c")
            ph_v, ph_m = _val("optimal_ph")
            salt_v, salt_m = _val("salt_tolerance_pct")
            oxy_raw, oxy_m = _val("oxygen_requirement")
            if oxy_m and oxy_raw not in OXY_LABEL_TO_INT:
                oxy_m = 0
                oxy_raw = None

            rows.append({
                "bacdive_id": bacdive_id,
                "genome_accession": r["genome_accession"],
                "by_category": r["by_category"],
                "group": (
                    p_row.get("family")
                    or p_row.get("genus")
                    or (p_row.get("species") or "__unk__").split()[0]
                ),
                "labels": {
                    "temp": float(temp_v) if temp_m else 0.0,
                    "ph": float(ph_v) if ph_m else 0.0,
                    "salt": float(salt_v) if salt_m else 0.0,
                    "oxy": OXY_LABEL_TO_INT[oxy_raw] if oxy_m else 0,
                },
                "label_mask": {
                    "temp": temp_m, "ph": ph_m, "salt": salt_m, "oxy": oxy_m,
                },
            })
    return rows


def _group_kfold_split(rows: list[dict], n_splits: int, fold: int):
    from sklearn.model_selection import GroupKFold

    groups = [r["group"] for r in rows]
    indices = np.arange(len(rows))
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(indices, groups=groups))
    train_idx, val_idx = splits[fold]
    train = [rows[i] for i in train_idx]
    val = [rows[i] for i in val_idx]
    return train, val


def _collate(batch: list[dict]) -> dict:
    genomes = [r["by_category"] for r in batch]
    labels = {
        k: torch.tensor([r["labels"][k] for r in batch], dtype=torch.float32)
        for k in ("temp", "ph", "salt")
    }
    labels["oxy"] = torch.tensor([r["labels"]["oxy"] for r in batch], dtype=torch.long)
    label_mask = {
        k: torch.tensor([r["label_mask"][k] for r in batch], dtype=torch.float32)
        for k in ("temp", "ph", "salt", "oxy")
    }
    return {"genomes": genomes, "labels": labels, "label_mask": label_mask}


def _iter_batches(rows: list[dict], batch_size: int, shuffle: bool):
    indices = list(range(len(rows)))
    if shuffle:
        import random
        random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        chunk = [rows[j] for j in indices[i : i + batch_size]]
        yield _collate(chunk)


@torch.no_grad()
def run_validation(model: PhenoLoRAModel, val_rows: list[dict], device: torch.device, batch_size: int) -> dict:
    """Compute validation metrics in inference mode (no grad)."""
    model.eval()
    pred_lists: dict[str, list] = {k: [] for k in ("temp", "ph", "salt", "oxy")}
    label_lists: dict[str, list] = {k: [] for k in ("temp", "ph", "salt", "oxy")}
    mask_lists: dict[str, list] = {k: [] for k in ("temp", "ph", "salt", "oxy")}

    for batch in _iter_batches(val_rows, batch_size, shuffle=False):
        preds = model(batch["genomes"], device=device)
        for k in ("temp", "ph", "salt"):
            pred_lists[k].append(preds[k].cpu().float().numpy())
            label_lists[k].append(batch["labels"][k].cpu().numpy())
            mask_lists[k].append(batch["label_mask"][k].cpu().numpy())
        pred_lists["oxy"].append(preds["oxy"].argmax(dim=-1).cpu().numpy())
        label_lists["oxy"].append(batch["labels"]["oxy"].cpu().numpy())
        mask_lists["oxy"].append(batch["label_mask"]["oxy"].cpu().numpy())

    out: dict = {}
    for k in ("temp", "ph", "salt"):
        preds_arr = np.concatenate(pred_lists[k])
        labels_arr = np.concatenate(label_lists[k])
        masks_arr = np.concatenate(mask_lists[k]).astype(bool)
        if masks_arr.sum() == 0:
            out[k] = {"mae": None, "n": 0}
            continue
        mae = float(np.mean(np.abs(preds_arr[masks_arr] - labels_arr[masks_arr])))
        out[k] = {"mae": mae, "n": int(masks_arr.sum())}

    preds_oxy = np.concatenate(pred_lists["oxy"])
    labels_oxy = np.concatenate(label_lists["oxy"])
    masks_oxy = np.concatenate(mask_lists["oxy"]).astype(bool)
    if masks_oxy.sum() == 0:
        out["oxy"] = {"f1_macro": None, "n": 0}
    else:
        f1 = float(f1_score(labels_oxy[masks_oxy], preds_oxy[masks_oxy], average="macro"))
        out["oxy"] = {"f1_macro": f1, "n": int(masks_oxy.sum())}
    return out


def train_lora(
    *,
    model_cfg: LoraModelConfig,
    train_cfg: TrainConfig,
    sequences_path: Path,
    phenotypes_path: Path,
    catalog_path: Path,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lora] device = {device}", flush=True)

    rows = _build_dataset(sequences_path, phenotypes_path, catalog_path)
    print(f"[lora] loaded {len(rows):,} records with sequences + labels", flush=True)

    train_rows, val_rows = _group_kfold_split(rows, n_splits=5, fold=train_cfg.fold)
    print(f"[lora] fold {train_cfg.fold}: {len(train_rows):,} train / {len(val_rows):,} val",
          flush=True)

    model = PhenoLoRAModel(model_cfg).to(device)
    trainable, total = model.trainable_param_count()
    print(f"[lora] trainable params: {trainable:,} / total: {total:,} "
          f"({100 * trainable / total:.2f}%)", flush=True)

    lora_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("heads."):
            head_params.append(p)
        else:
            lora_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": lora_params, "lr": train_cfg.lora_lr},
            {"params": head_params, "lr": train_cfg.head_lr},
        ],
        weight_decay=train_cfg.weight_decay,
    )

    n_train_batches = math.ceil(len(train_rows) / train_cfg.batch_size)
    total_steps = max(1, n_train_batches * train_cfg.epochs // max(train_cfg.grad_accum, 1))
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_frac))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            step / max(warmup_steps, 1)
            if step < warmup_steps
            else 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))
        ),
    )

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    autocast_dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32

    history: list[dict] = []
    best = {"epoch": -1, "val": None, "score": float("inf")}
    global_step = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        for batch_idx, batch in enumerate(_iter_batches(train_rows, train_cfg.batch_size, shuffle=True)):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")):
                preds = model(batch["genomes"], device=device)
                loss, per_target = masked_multitask_loss(
                    preds,
                    {k: v.to(device) for k, v in batch["labels"].items()},
                    {k: v.to(device) for k, v in batch["label_mask"].items()},
                )

            loss = loss / max(train_cfg.grad_accum, 1)
            loss.backward()
            running_loss += float(loss.detach().cpu()) * max(train_cfg.grad_accum, 1)
            running_n += 1

            if (batch_idx + 1) % train_cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=train_cfg.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    print(f"  ep {epoch+1} step {global_step}: "
                          f"loss={running_loss/max(running_n,1):.4f}  "
                          f"lr_lora={scheduler.get_last_lr()[0]:.2e}",
                          flush=True)

        val_metrics = run_validation(model, val_rows, device, train_cfg.batch_size)
        elapsed = time.time() - t0
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(running_n, 1),
            "val": val_metrics,
            "elapsed_s": elapsed,
        }
        history.append(record)
        print(f"[lora] epoch {epoch+1} done in {elapsed:.0f}s  val={val_metrics}", flush=True)

        score = sum(
            (val_metrics[k]["mae"] or 0.0)
            for k in ("temp", "ph", "salt")
            if val_metrics[k]["mae"] is not None
        ) - (val_metrics["oxy"]["f1_macro"] or 0.0)
        if score < best["score"]:
            best = {"epoch": epoch + 1, "val": val_metrics, "score": score}
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(train_cfg),
                    "state_dict": {k: v for k, v in model.state_dict().items() if "lora" in k.lower() or k.startswith("heads.")},
                },
                save_dir / f"fold{train_cfg.fold}_best.pt",
            )

    results = {
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "history": history,
        "best": best,
    }
    out_json = save_dir / f"fold{train_cfg.fold}_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[lora] wrote {out_json}", flush=True)
    return results
