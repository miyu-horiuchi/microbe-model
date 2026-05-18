# LoRA Fold 0 Results

This page records the completed fold 0 LoRA experiments and the checkpoint to use.

## Recommendation

Use the original all-task fold 0 checkpoint:

- Local checkpoint path: `artifacts/lora/fold0_best.pt`
- Durable release asset: `fold0_best_all_task.pt`
- GitHub release: https://github.com/miyu-horiuchi/microbe-model/releases/tag/lora-fold0-20260518

The all-task checkpoint is the best current fold 0 LoRA result. Oxygen-only training
and the anaerobe-weighted run were useful checks, but neither improved the clean
validation comparison enough to replace the original checkpoint.

## Experiments

All runs used fold 0, ESM-2 t12, LoRA `r=8`, one epoch, batch size 2, gradient
accumulation 8, and Lambda A100 SXM4 GPU training.

| Run | Local result file | Oxygen macro F1 | Oxygen n | Use? |
|---|---|---:|---:|---|
| All-task LoRA | `artifacts/lora/fold0_results.json` | 0.944823 | 2266 | Yes |
| Oxygen-only LoRA | `artifacts/lora/fold0_results_oxygen.json` | 0.916836 | 2214 | No |
| Anaerobe-weighted all-task LoRA | `artifacts/lora_weighted_anaerobe/fold0_results.json` | 0.944776 | 2266 | No |

The anaerobe-weighted run used oxygen class weights:

```text
aerobe=1.0, anaerobe=1.5, facultative_anaerobe=1.0, microaerobe=1.0
```

It slightly improved anaerobe recall in the detailed diagnostic, but its fold 0
training-validation oxygen macro F1 was fractionally lower than the all-task run.

## Checkpoint Assets

The `.pt` files are not committed to git. They are stored as GitHub Release assets:

| Asset | SHA256 |
|---|---|
| `fold0_best_all_task.pt` | `8a73ee20252b1aa710b0480abd307ffbc38b788b1a152a7e63298c525a04be23` |
| `fold0_best_oxygen_only.pt` | `fd10d4a2a7cba5d564fb9ba1f730cace07a0a2173d3622f1f572cfd29306fc95` |
| `fold0_best_weighted_anaerobe.pt` | `c8d34999f570663e020e5644a994f821bf9539a6fcc3e029d5942b8dc7709826` |

## Loading The Best Checkpoint

The checkpoint is a PyTorch dictionary with these keys:

- `epoch`
- `model_cfg`
- `train_cfg`
- `state_dict`

Minimal load pattern:

```python
import torch

from microbe_model.train.lora_model import LoraModelConfig, PhenoLoRAModel

checkpoint = torch.load("artifacts/lora/fold0_best.pt", map_location="cpu")
model_cfg = LoraModelConfig(**checkpoint["model_cfg"])
model = PhenoLoRAModel(model_cfg)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()
```

To regenerate oxygen diagnostics:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python scripts/38_eval_lora_checkpoint.py \
  --checkpoint artifacts/lora/fold0_best.pt \
  --output-json artifacts/lora/fold0_oxygen_diagnostics.json \
  --output-md artifacts/lora/fold0_oxygen_diagnostics.md \
  --batch-size 2
```

## Next GPU Work

Do not spend more GPU on fold 0 variants unless there is a new hypothesis. The next
meaningful validation step is to run the selected all-task LoRA setup across folds
1-4 and report the mean and variance across all five folds. That is a stronger
scientific result, but it is also the next major GPU-cost item.
