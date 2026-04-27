# microbe-model — v0 baseline eval report

_Generated: 2026-04-27T02:16:35+00:00_

## TL;DR

- **`optimal_temperature_c`**: MAE = **3.17** (vs always-predict-mean 5.53, **+43%**)
- **`optimal_ph`**: MAE = **0.54** (vs always-predict-mean 0.55, **+1%**)
- **`oxygen_requirement`**: macro-F1 = **0.283** (vs always-predict-majority 0.072, **+294%**)
- **`salt_tolerance_pct`**: MAE = **2.52** (vs always-predict-mean 2.72, **+7%**)

Trained on **17,065** strains with **33** genome-derived features. Cross-validation: 5-fold GroupKFold by taxonomic family.

## Corpus

- Total strains in feature table: **17,065**
- Labeled-strain counts by target:
  - `optimal_temperature_c`: 17,025
  - `optimal_ph`: 4,654
  - `oxygen_requirement`: 10,434
  - `salt_tolerance_pct`: 4,800

## Target distributions

- `optimal_temperature_c`: n=17,025, mean=31.96, std=8.57, p10=25.00, median=30.00, p90=37.00
- `optimal_ph`: n=4,654, mean=7.19, std=0.83, p10=6.50, median=7.00, p90=8.00
- `salt_tolerance_pct`: n=4,800, mean=3.55, std=4.11, p10=0.00, median=2.50, p90=8.00
- `oxygen_requirement`:
  - `aerobe`: 4,978
  - `anaerobe`: 2,120
  - `facultative anaerobe`: 1,227
  - `obligate aerobe`: 1,029
  - `microaerophile`: 889
  - `obligate anaerobe`: 105
  - `facultative aerobe`: 83
  - `microaerotolerant`: 2
  - `aerotolerant`: 1

## Per-target results (5-fold GroupKFold by family)

Metrics: regression = MAE (lower is better), classification = macro-F1 (higher is better).
Each is shown alongside the dumb-baseline (always-predict-mean / always-predict-majority).

| Target | Task | n labeled | Model metric | Baseline | Improvement |
|---|---|---|---|---|---|
| `optimal_temperature_c` | regression | 17,025 | MAE=3.173 | MAE=5.528 | +42.6% |
| `optimal_ph` | regression | 4,654 | MAE=0.540 | MAE=0.546 | +1.2% |
| `oxygen_requirement` | classification | 10,434 | F1=0.283 | F1=0.072 | +294.3% |
| `salt_tolerance_pct` | regression | 4,800 | MAE=2.523 | MAE=2.720 | +7.3% |

### `optimal_temperature_c` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 2.846 | n=13,620 | n=3,405 |
| 2 | mae = 3.457 | n=13,620 | n=3,405 |
| 3 | mae = 3.456 | n=13,620 | n=3,405 |
| 4 | mae = 2.738 | n=13,620 | n=3,405 |
| 5 | mae = 3.367 | n=13,620 | n=3,405 |

**Top 10 features for `optimal_temperature_c`:**

- `ivywrel_frac` — 0.4960
- `n_predicted_cds` — 0.0539
- `pos_charged_frac` — 0.0393
- `aa_frac_P` — 0.0282
- `aa_frac_C` — 0.0269
- `aa_frac_Y` — 0.0244
- `aa_frac_S` — 0.0224
- `aa_frac_E` — 0.0212
- `mean_isoelectric_point` — 0.0209
- `genome_size_nt` — 0.0195

### `optimal_ph` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 0.487 | n=3,723 | n=931 |
| 2 | mae = 0.571 | n=3,723 | n=931 |
| 3 | mae = 0.554 | n=3,723 | n=931 |
| 4 | mae = 0.554 | n=3,723 | n=931 |
| 5 | mae = 0.532 | n=3,724 | n=930 |

**Top 10 features for `optimal_ph`:**

- `neg_charged_frac` — 0.1119
- `aa_frac_H` — 0.0644
- `ivywrel_frac` — 0.0471
- `aa_frac_Q` — 0.0425
- `aa_frac_E` — 0.0401
- `n_predicted_cds` — 0.0346
- `mean_hydrophobicity` — 0.0335
- `aa_frac_L` — 0.0323
- `aa_frac_C` — 0.0318
- `aa_frac_V` — 0.0311

### `oxygen_requirement` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | f1_macro = 0.274 | n=8,347 | n=2,086 |
| 2 | f1_macro = 0.280 | n=8,347 | n=2,087 |
| 3 | f1_macro = 0.291 | n=8,347 | n=2,087 |
| 4 | f1_macro = 0.267 | n=8,347 | n=2,087 |
| 5 | f1_macro = 0.304 | n=8,348 | n=2,086 |

**Top 10 features for `oxygen_requirement`:**

- `aa_frac_C` — 0.1045
- `genome_size_nt` — 0.0890
- `n_predicted_cds` — 0.0619
- `aa_frac_Q` — 0.0543
- `aa_frac_K` — 0.0373
- `aa_frac_W` — 0.0362
- `aa_frac_M` — 0.0354
- `aa_frac_H` — 0.0352
- `aa_frac_G` — 0.0335
- `ivywrel_frac` — 0.0307

### `salt_tolerance_pct` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 2.183 | n=3,840 | n=960 |
| 2 | mae = 2.590 | n=3,840 | n=960 |
| 3 | mae = 3.110 | n=3,840 | n=960 |
| 4 | mae = 2.440 | n=3,840 | n=960 |
| 5 | mae = 2.291 | n=3,840 | n=960 |

**Top 10 features for `salt_tolerance_pct`:**

- `aa_frac_C` — 0.1585
- `neg_charged_frac` — 0.1448
- `aa_frac_T` — 0.0360
- `mean_isoelectric_point` — 0.0306
- `aa_frac_H` — 0.0301
- `aa_frac_D` — 0.0288
- `aa_frac_L` — 0.0276
- `aa_frac_K` — 0.0265
- `aa_frac_W` — 0.0264
- `ivywrel_frac` — 0.0261

## Feature ↔ target correlations (Spearman, top 10)

Sanity-checks the biology — features known to track each target should appear here at high |ρ|. E.g. `ivywrel_frac` should correlate with `optimal_temperature_c` (Zeldovich 2007 thermophile signature).

### `optimal_temperature_c`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `genome_size_nt` | -0.405 | 0.0e+00 |
| `aa_frac_E` | +0.388 | 0.0e+00 |
| `n_predicted_cds` | -0.386 | 0.0e+00 |
| `ivywrel_frac` | +0.320 | 0.0e+00 |
| `aa_frac_Y` | +0.318 | 0.0e+00 |
| `aa_frac_W` | -0.309 | 0.0e+00 |
| `pos_charged_frac` | +0.299 | 0.0e+00 |
| `aa_frac_A` | -0.295 | 0.0e+00 |
| `neg_charged_frac` | +0.293 | 0.0e+00 |
| `aa_frac_P` | -0.277 | 1.0e-297 |

### `optimal_ph`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `neg_charged_frac` | +0.308 | 4.2e-103 |
| `mean_isoelectric_point` | -0.276 | 3.8e-82 |
| `aa_frac_E` | +0.260 | 6.2e-73 |
| `ivywrel_frac` | +0.166 | 4.0e-30 |
| `aa_frac_D` | +0.111 | 2.6e-14 |
| `mean_hydrophobicity` | -0.109 | 7.3e-14 |
| `aa_frac_C` | -0.095 | 7.6e-11 |
| `pos_charged_frac` | -0.086 | 3.5e-09 |
| `aa_frac_P` | -0.081 | 3.3e-08 |
| `aa_frac_A` | -0.078 | 9.6e-08 |

### `salt_tolerance_pct`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `neg_charged_frac` | +0.227 | 3.2e-57 |
| `mean_isoelectric_point` | -0.204 | 3.5e-46 |
| `ivywrel_frac` | +0.195 | 1.7e-42 |
| `aa_frac_C` | -0.186 | 1.1e-38 |
| `mean_cds_aa_length` | -0.161 | 3.8e-29 |
| `aa_frac_D` | +0.159 | 1.4e-28 |
| `aa_frac_E` | +0.143 | 2.4e-23 |
| `aa_frac_V` | +0.112 | 8.3e-15 |
| `aa_frac_T` | +0.104 | 4.4e-13 |
| `coding_density` | -0.090 | 4.1e-10 |

## Per-family error breakdown (regression targets)

Top 15 most-represented families, MAE per family. Highlights where the model is doing well vs. struggling.

### `optimal_temperature_c`

| Family | n | MAE |
|---|---|---|
| Streptomycetaceae | 798 | 1.451 |
| Bacillaceae | 643 | 4.086 |
| Flavobacteriaceae | 631 | 4.195 |
| Lactobacillaceae | 471 | 3.161 |
| Enterobacteriaceae | 439 | 3.967 |
| Microbacteriaceae | 396 | 2.457 |
| Pseudomonadaceae | 388 | 2.523 |
| Roseobacteraceae | 341 | 2.992 |
| Paenibacillaceae | 319 | 3.474 |
| Pseudonocardiaceae | 306 | 2.184 |
| Moraxellaceae | 269 | 2.723 |
| Sphingomonadaceae | 256 | 1.845 |
| Streptococcaceae | 251 | 3.063 |
| Clostridiaceae | 247 | 4.560 |
| Vibrionaceae | 239 | 3.290 |

### `optimal_ph`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 323 | 0.424 |
| Bacillaceae | 273 | 0.657 |
| Roseobacteraceae | 192 | 0.363 |
| Paenibacillaceae | 126 | 0.484 |
| Microbacteriaceae | 112 | 0.510 |
| Sphingobacteriaceae | 100 | 0.376 |
| Sphingomonadaceae | 96 | 0.365 |
| Streptomycetaceae | 92 | 0.742 |
| Pseudonocardiaceae | 85 | 0.547 |
| Halomonadaceae | 81 | 0.748 |
| Nocardioidaceae | 74 | 0.516 |
| Paracoccaceae | 71 | 0.563 |
| Micrococcaceae | 71 | 0.626 |
| Erythrobacteraceae | 68 | 0.442 |
| Alteromonadaceae | 68 | 0.375 |

### `salt_tolerance_pct`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 285 | 1.491 |
| Streptomycetaceae | 283 | 2.070 |
| Bacillaceae | 246 | 3.241 |
| Microbacteriaceae | 140 | 2.808 |
| Pseudonocardiaceae | 134 | 2.501 |
| Roseobacteraceae | 134 | 1.533 |
| Paenibacillaceae | 125 | 2.387 |
| Pseudomonadaceae | 110 | 3.870 |
| Vibrionaceae | 100 | 2.653 |
| Sphingomonadaceae | 92 | 2.029 |
| Micromonosporaceae | 90 | 1.724 |
| Micrococcaceae | 85 | 3.045 |
| Nocardiaceae | 84 | 2.613 |
| Streptococcaceae | 82 | 1.390 |
| Lactobacillaceae | 78 | 2.506 |

## Known limitations

- **Survivorship bias.** BacDive only contains organisms that have been cultured successfully at least once. The model cannot generalize to truly uncultured strains without explicit out-of-distribution evaluation.
- **Optimum derivation is heuristic.** Most BacDive temperature entries are tagged as `growth` (positive growth at this temperature), not `optimum`. We approximate the optimum as the median of positive-growth temperatures when no explicit optimum is recorded — this can be off by 5°C or more for some strains.
- **Family grouping is naive.** The current `family` column is derived from the genus (first word of binomial name). A proper LPSN/GTDB family assignment would give tighter taxonomic grouping.
- **Feature set is shallow.** No HMM/KEGG annotations, no codon usage indices, no tRNA counts. These are interpretable next steps before moving to genome LMs.
- **Pyrodigal accuracy.** Gene prediction quality drops on highly-fragmented assemblies and atypical genetic codes. Not currently flagged in the feature set.

## Next steps

1. **Add tetranucleotide / codon-usage features.** ~50 extra columns, well-known signal for thermophily.
2. **Replace naive family lookup with LPSN/GTDB join.** Reduces leakage in CV.
3. **Integrate KOMODO media DB** as a richer label source than BacDive alone.
4. **Move to genome embeddings** (Nucleotide Transformer / Evo-1 / DNABERT-2) once the tabular ceiling is established.
5. **Active learning loop**: select novel-family strains where the model is uncertain, prioritize these for wet-lab cultivation testing.
