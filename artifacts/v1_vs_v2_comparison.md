# v1 (hand-crafted features) vs v2 (ESM-2 embeddings)

Same train/test splits, same XGBoost hyperparameters. Only difference: input features.

| Target | v1 (n features) | v2 (embedding dim) | Δ |
|---|---|---|---|
| `optimal_temperature_c` | MAE 3.280 | MAE 3.811 | 🔴 +0.532 (+16.2%) |
| `optimal_ph` | MAE 0.520 | MAE 0.544 | 🔴 +0.023 (+4.5%) |
| `oxygen_requirement` | F1 0.279 | F1 0.266 | 🔴 -0.013 (-4.6%) |
| `salt_tolerance_pct` | MAE 2.510 | MAE 2.597 | 🔴 +0.087 (+3.5%) |

## Reading this table

- 🟢 = embeddings beat hand-crafted features
- 🔴 = hand-crafted features beat embeddings
- ⚪ = no difference

Regression: lower MAE is better, so a *negative* delta is good. 
Classification: higher F1 is better, so a *positive* delta is good.

## Interpretation

- **≥ 10% lift on T_opt:** validates the genome-LM direction. Worth investing
  in larger models (ESM-2 t33_650M or Nucleotide Transformer / Evo-1).
- **pH or salt go from broken (≤5%) to working (≥15%):** embeddings recover
  signal that hand-crafted features couldn't capture. Big win for the thesis.
- **No meaningful lift anywhere:** the bottleneck is not feature engineering. 
  Need new data sources (failed cultivation logs, environmental metadata).