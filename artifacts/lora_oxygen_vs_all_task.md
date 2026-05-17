# Oxygen-Only LoRA vs All-Task LoRA

Both runs use fold 0, ESM-2 t12, LoRA r=8, 1 epoch on Lambda A100 SXM4.

| Target | All-task LoRA | Oxygen-only LoRA | Baseline | Readout |
|---|---:|---:|---:|---|
| `optimal_temperature_c` MAE | 3.6660 | 32.2504 | 2.6743 | ignored by oxygen-only loss |
| `optimal_ph` MAE | 0.5598 | 7.0462 | 0.4685 | ignored by oxygen-only loss |
| `salt_tolerance_pct` MAE | 1.8154 | 2.0823 | 1.9171 | not improved by oxygen-only loss |
| `oxygen_requirement` macro F1 | 0.9448 | 0.9168 | 0.4020 | oxygen-only worse than all-task |

## Conclusion

Oxygen-only training did not beat the all-task LoRA run on the oxygen target: `0.9168` macro F1 vs `0.9448` macro F1. The all-task LoRA checkpoint remains the better fold-0 result. The oxygen-only checkpoint is saved locally for inspection but should stay out of git.

## Artifacts

- `artifacts/lora/fold0_results_oxygen.json`
- `artifacts/lora/lambda_fold0_oxygen_1ep_20260517T103524Z.log`
- local-only checkpoint: `artifacts/lora/fold0_best_oxygen.pt`
