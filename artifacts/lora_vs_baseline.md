# LoRA Fold 0 vs Tabular Baseline

Caveat: LoRA is one group fold; baseline is the current five-fold mean.

| Target | LoRA | Baseline | Delta | Verdict |
|---|---:|---:|---:|---|
| `optimal_temperature_c` MAE | 3.6660 | 2.6743 | +0.9917 | worse |
| `optimal_ph` MAE | 0.5598 | 0.4685 | +0.0913 | worse |
| `salt_tolerance_pct` MAE | 1.8154 | 1.9171 | -0.1017 | better |
| `oxygen_requirement` macro F1 | 0.9448 | 0.4020 | +0.5428 | better |

## Recommendation

The first LoRA pass is strongest for oxygen classification. For the next GPU run, use `scripts/lambda_train_lora.py --target-preset oxygen` instead of spending more A100 time optimizing regression losses that underperformed the tabular baseline.

Keep `artifacts/lora/fold0_best.pt` outside git unless it is published to a model store or release asset; the JSON metrics and log are enough for repo history.
