"""PhenoLoRAModel — LoRA-fine-tuned ESM-2 + per-category mean-pool + multi-task heads.

Architecture:
    For each genome, accept up to K proteins per category × 8 categories.
    For each protein: tokenize → ESM-2(+LoRA) → masked mean-pool over residues → 1 vector.
    For each category: mean-pool over its proteins → 1 category vector.
    Concatenate the 8 category vectors → genome vector.
    Predict the 4 phenotype targets via 4 small heads.

Trainable parameters:
    - LoRA adapters on ESM-2 attention (~1.5M params for t30, r=8 on q+v).
    - Per-protein projection (optional, default skipped — heads operate on raw 8×D vector).
    - 4 prediction heads (~10-50K params combined).

Multi-task loss handles per-target missing labels via a binary mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

CATEGORIES = ["temperature", "ph", "oxygen", "salt", "vitamin", "nitrogen", "carbon", "special"]
OXYGEN_CLASSES = ["aerobe", "anaerobe", "facultative_anaerobe", "microaerobe"]
N_OXYGEN_CLASSES = len(OXYGEN_CLASSES)


@dataclass
class LoraModelConfig:
    esm_model_name: str = "facebook/esm2_t12_35M_UR50D"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: tuple[str, ...] = ("query", "value")
    head_hidden_dim: int = 128
    head_dropout: float = 0.1
    max_seq_len: int = 512               # truncate long proteins to fit memory
    max_proteins_per_cat: int = 6        # cap protein count per category at train time
    gradient_checkpointing: bool = True  # trade compute for memory


class PhenoLoRAModel(nn.Module):
    def __init__(self, cfg: LoraModelConfig):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.esm_model_name)
        base = AutoModel.from_pretrained(cfg.esm_model_name)
        self.embed_dim = base.config.hidden_size

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target),
            bias="none",
        )
        self.esm = get_peft_model(base, lora_cfg)
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()

        # 8 × embed_dim → per-target heads.
        genome_dim = self.embed_dim * len(CATEGORIES)
        hidden = cfg.head_hidden_dim

        def regression_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(genome_dim, hidden),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(hidden, 1),
            )

        self.heads = nn.ModuleDict({
            "temp": regression_head(),
            "ph": regression_head(),
            "salt": regression_head(),
            "oxy": nn.Sequential(
                nn.Linear(genome_dim, hidden),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(hidden, N_OXYGEN_CLASSES),
            ),
        })

    def encode_proteins(self, proteins: list[str], device: torch.device) -> torch.Tensor:
        """Tokenize and ESM-encode a list of proteins → tensor of shape (n_proteins, embed_dim).

        Returns a zero tensor of shape (0, embed_dim) for an empty list.
        """
        if not proteins:
            return torch.zeros((0, self.embed_dim), device=device)
        enc = self.tokenizer(
            proteins,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_seq_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = self.esm(**enc)
        last = outputs.last_hidden_state  # (B, L, D)
        mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def encode_genome(self, by_category: dict[str, list[str]], device: torch.device) -> torch.Tensor:
        """Build one genome vector of shape (8 × embed_dim,) by category-pooling proteins."""
        cat_vectors: list[torch.Tensor] = []
        cap = self.cfg.max_proteins_per_cat
        for cat in CATEGORIES:
            proteins = by_category.get(cat) or []
            if proteins:
                # Cap per-category protein count at train time to control memory.
                # Already sorted shortest-first by the extractor.
                proteins = proteins[:cap]
                per_protein = self.encode_proteins(proteins, device)  # (n, D)
                cat_vec = per_protein.mean(dim=0)
            else:
                cat_vec = torch.zeros(self.embed_dim, device=device)
            cat_vectors.append(cat_vec)
        return torch.cat(cat_vectors, dim=0)

    def forward(self, genomes: list[dict[str, list[str]]], device: torch.device) -> dict[str, torch.Tensor]:
        """Batched forward over a list of genomes (variable protein counts per genome).

        Returns dict of per-target predictions:
            temp, ph, salt: (B,) regression predictions
            oxy:            (B, N_OXYGEN_CLASSES) logits
        """
        genome_vecs = torch.stack(
            [self.encode_genome(g, device) for g in genomes],
            dim=0,
        )  # (B, 8*D)
        return {
            "temp": self.heads["temp"](genome_vecs).squeeze(-1),
            "ph": self.heads["ph"](genome_vecs).squeeze(-1),
            "salt": self.heads["salt"](genome_vecs).squeeze(-1),
            "oxy": self.heads["oxy"](genome_vecs),
        }

    def trainable_param_count(self) -> tuple[int, int]:
        """Return (trainable, total) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def masked_multitask_loss(
    preds: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    label_mask: dict[str, torch.Tensor],
    target_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of per-target losses, weighted by the binary label mask.

    Regression targets: MSE.
    Oxygen: cross-entropy.
    Each per-target loss is averaged over rows with mask=1; if mask is all-zero for a
    target in this batch, that target contributes 0 to the total.
    """
    if target_weights is None:
        target_weights = {"temp": 1.0, "ph": 1.0, "salt": 1.0, "oxy": 1.0}
    per_target_loss: dict[str, float] = {}
    total = preds["temp"].new_zeros(())

    for tgt in ("temp", "ph", "salt"):
        mask = label_mask[tgt].float()
        if mask.sum() == 0:
            per_target_loss[tgt] = 0.0
            continue
        sq = (preds[tgt] - labels[tgt]) ** 2
        loss = (sq * mask).sum() / mask.sum().clamp(min=1.0)
        total = total + target_weights[tgt] * loss
        per_target_loss[tgt] = float(loss.detach().cpu())

    mask_oxy = label_mask["oxy"].float()
    if mask_oxy.sum() > 0:
        logits = preds["oxy"]
        labels_oxy = labels["oxy"].long()
        per_row_loss = nn.functional.cross_entropy(logits, labels_oxy, reduction="none")
        loss = (per_row_loss * mask_oxy).sum() / mask_oxy.sum().clamp(min=1.0)
        total = total + target_weights["oxy"] * loss
        per_target_loss["oxy"] = float(loss.detach().cpu())
    else:
        per_target_loss["oxy"] = 0.0

    return total, per_target_loss
