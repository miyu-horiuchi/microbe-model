# microbe-model — v0 baseline eval report

_Generated: 2026-05-05T10:42:09+00:00_

## TL;DR

- **`optimal_temperature_c`**: MAE = **2.86** (vs always-predict-mean 4.98, **+43%**)
- **`optimal_ph`**: MAE = **0.48** (vs always-predict-mean 0.55, **+12%**)
- **`oxygen_requirement`**: macro-F1 = **0.357** (vs always-predict-majority 0.059, **+507%**)
- **`salt_tolerance_pct`**: MAE = **2.11** (vs always-predict-mean 2.51, **+16%**)

Trained on **46,029** strains with **423** genome-derived features. Cross-validation: 5-fold GroupKFold by taxonomic family.

## Corpus

- Total strains in feature table: **46,029**
- Labeled-strain counts by target:
  - `optimal_temperature_c`: 45,621
  - `optimal_ph`: 5,103
  - `oxygen_requirement`: 21,639
  - `salt_tolerance_pct`: 3,844

## Target distributions

- `optimal_temperature_c`: n=45,621, mean=32.24, std=7.13, p10=27.50, median=30.00, p90=37.00
- `optimal_ph`: n=5,103, mean=7.19, std=0.82, p10=6.50, median=7.00, p90=8.00
- `salt_tolerance_pct`: n=3,844, mean=3.15, std=4.28, p10=0.00, median=2.50, p90=7.00
- `oxygen_requirement`:
  - `aerobe`: 7,803
  - `anaerobe`: 4,193
  - `microaerophile`: 3,804
  - `facultative anaerobe`: 3,389
  - `obligate aerobe`: 2,213
  - `obligate anaerobe`: 136
  - `facultative aerobe`: 87
  - `aerotolerant`: 12
  - `microaerotolerant`: 2

## Per-target results (5-fold GroupKFold by family)

Metrics: regression = MAE (lower is better), classification = macro-F1 (higher is better).
Each is shown alongside the dumb-baseline (always-predict-mean / always-predict-majority).

| Target | Task | n labeled | Model metric | Baseline | Improvement |
|---|---|---|---|---|---|
| `optimal_temperature_c` | regression | 45,621 | MAE=2.857 | MAE=4.981 | +42.6% |
| `optimal_ph` | regression | 5,103 | MAE=0.482 | MAE=0.546 | +11.6% |
| `oxygen_requirement` | classification | 21,639 | F1=0.357 | F1=0.059 | +507.0% |
| `salt_tolerance_pct` | regression | 3,844 | MAE=2.112 | MAE=2.515 | +16.0% |

### `optimal_temperature_c` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 2.953 | n=36,496 | n=9,125 |
| 2 | mae = 2.626 | n=36,497 | n=9,124 |
| 3 | mae = 3.060 | n=36,497 | n=9,124 |
| 4 | mae = 3.265 | n=36,497 | n=9,124 |
| 5 | mae = 2.381 | n=36,497 | n=9,124 |

**Top 10 features for `optimal_temperature_c`:**

- `ivywrel_frac` — 0.1235
- `iso_cat2_thermophilic_gt45_c` — 0.0288
- `iso_cat2_patient` — 0.0251
- `iso_cat2_human` — 0.0234
- `n_predicted_cds` — 0.0216
- `iso_cat1_infection` — 0.0204
- `aa_frac_C` — 0.0143
- `genome_size_nt` — 0.0123
- `tetra_CTAA` — 0.0118
- `aa_frac_D` — 0.0109

### `optimal_ph` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 0.440 | n=4,082 | n=1,021 |
| 2 | mae = 0.568 | n=4,082 | n=1,021 |
| 3 | mae = 0.494 | n=4,082 | n=1,021 |
| 4 | mae = 0.466 | n=4,083 | n=1,020 |
| 5 | mae = 0.444 | n=4,083 | n=1,020 |

**Top 10 features for `optimal_ph`:**

- `md_ph_median` — 0.0518
- `iso_cat2_acidic` — 0.0307
- `iso_cat2_alkaline` — 0.0287
- `neg_charged_frac` — 0.0146
- `aa_frac_H` — 0.0081
- `aa_frac_E` — 0.0077
- `tetra_CTCT` — 0.0071
- `iso_cat2_plant` — 0.0068
- `tetra_AGAC` — 0.0067
- `tetra_CACT` — 0.0065

### `oxygen_requirement` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | f1_macro = 0.353 | n=17,311 | n=4,328 |
| 2 | f1_macro = 0.375 | n=17,311 | n=4,326 |
| 3 | f1_macro = 0.357 | n=17,311 | n=4,328 |
| 4 | f1_macro = 0.274 | n=17,311 | n=4,328 |
| 5 | f1_macro = 0.429 | n=17,312 | n=4,327 |

**Top 10 features for `oxygen_requirement`:**

- `codon_ATA` — 0.0395
- `iso_cat1_host` — 0.0269
- `n_predicted_cds` — 0.0266
- `aa_frac_C` — 0.0195
- `iso_cat1_environmental` — 0.0162
- `codon_CGT` — 0.0144
- `iso_cat1_engineered` — 0.0139
- `iso_cat2_human` — 0.0124
- `genome_size_nt` — 0.0103
- `codon_TAA` — 0.0083

### `salt_tolerance_pct` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 1.926 | n=3,075 | n=769 |
| 2 | mae = 1.893 | n=3,075 | n=769 |
| 3 | mae = 2.746 | n=3,075 | n=769 |
| 4 | mae = 1.870 | n=3,075 | n=769 |
| 5 | mae = 2.128 | n=3,076 | n=768 |

**Top 10 features for `salt_tolerance_pct`:**

- `neg_charged_frac` — 0.0702
- `tetra_ATCC` — 0.0428
- `aa_frac_C` — 0.0298
- `iso_cat2_saline` — 0.0286
- `md_nacl_pct_median` — 0.0256
- `tetra_ACAT` — 0.0255
- `md_nacl_pct_max` — 0.0128
- `aa_frac_T` — 0.0120
- `codon_CCG` — 0.0093
- `tetra_TGAT` — 0.0089

## Feature ↔ target correlations (Spearman, top 10)

Sanity-checks the biology — features known to track each target should appear here at high |ρ|. E.g. `ivywrel_frac` should correlate with `optimal_temperature_c` (Zeldovich 2007 thermophile signature).

### `optimal_temperature_c`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `genome_size_nt` | -0.493 | 0.0e+00 |
| `n_predicted_cds` | -0.482 | 0.0e+00 |
| `aa_frac_P` | -0.391 | 0.0e+00 |
| `aa_frac_Y` | +0.390 | 0.0e+00 |
| `tetra_TCTT` | +0.383 | 0.0e+00 |
| `tetra_TATC` | +0.381 | 0.0e+00 |
| `tetra_GATA` | +0.381 | 0.0e+00 |
| `tetra_AAGA` | +0.381 | 0.0e+00 |
| `tetra_CATA` | +0.380 | 0.0e+00 |
| `tetra_TATG` | +0.379 | 0.0e+00 |

### `optimal_ph`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `md_ph_median` | +0.429 | 4.0e-131 |
| `neg_charged_frac` | +0.304 | 1.6e-109 |
| `mean_isoelectric_point` | -0.278 | 1.8e-91 |
| `aa_frac_E` | +0.256 | 4.5e-77 |
| `md_nacl_pct_max` | +0.218 | 1.9e-33 |
| `md_nacl_pct_median` | +0.212 | 9.9e-32 |
| `iso_cat2_alkaline` | +0.165 | 2.5e-32 |
| `ivywrel_frac` | +0.159 | 2.4e-30 |
| `codon_AAG` | -0.154 | 1.7e-28 |
| `codon_CGA` | +0.153 | 5.8e-28 |

### `salt_tolerance_pct`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `neg_charged_frac` | +0.442 | 1.5e-183 |
| `mean_isoelectric_point` | -0.344 | 1.7e-107 |
| `aa_frac_E` | +0.310 | 3.1e-86 |
| `tetra_GACT` | +0.302 | 4.3e-82 |
| `tetra_AGTC` | +0.302 | 1.0e-81 |
| `md_nacl_pct_max` | +0.298 | 2.9e-52 |
| `md_nacl_pct_median` | +0.290 | 1.6e-49 |
| `tetra_ACTC` | +0.282 | 2.2e-71 |
| `tetra_GAGT` | +0.273 | 1.9e-66 |
| `iso_cat2_saline` | +0.263 | 9.4e-62 |

## Per-family error breakdown (regression targets)

Top 15 most-represented families, MAE per family. Highlights where the model is doing well vs. struggling.

### `optimal_temperature_c`

| Family | n | MAE |
|---|---|---|
| Enterobacteriaceae | 2662 | 3.792 |
| Streptomycetaceae | 2212 | 1.783 |
| Bacillaceae | 1886 | 3.174 |
| Lactobacillaceae | 1732 | 3.709 |
| Pseudomonadaceae | 1621 | 2.488 |
| Myxococcaceae | 1546 | 0.238 |
| Streptococcaceae | 1170 | 2.537 |
| Staphylococcaceae | 1068 | 3.374 |
| Flavobacteriaceae | 981 | 4.116 |
| Corynebacteriaceae | 900 | 2.146 |
| Moraxellaceae | 890 | 3.388 |
| Paenibacillaceae | 760 | 3.081 |
| Microbacteriaceae | 734 | 2.459 |
| Micrococcaceae | 719 | 2.811 |
| Nocardiaceae | 715 | 2.276 |

### `optimal_ph`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 355 | 0.405 |
| Bacillaceae | 298 | 0.606 |
| Roseobacteraceae | 204 | 0.375 |
| Paenibacillaceae | 139 | 0.469 |
| Microbacteriaceae | 120 | 0.446 |
| Sphingobacteriaceae | 114 | 0.336 |
| Sphingomonadaceae | 102 | 0.319 |
| Streptomycetaceae | 98 | 0.513 |
| Pseudonocardiaceae | 93 | 0.479 |
| Halomonadaceae | 82 | 0.584 |
| Micrococcaceae | 82 | 0.613 |
| Nocardioidaceae | 80 | 0.502 |
| Paracoccaceae | 76 | 0.574 |
| Alteromonadaceae | 71 | 0.355 |
| Erythrobacteraceae | 68 | 0.446 |

### `salt_tolerance_pct`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 267 | 1.713 |
| Streptomycetaceae | 264 | 1.987 |
| Bacillaceae | 201 | 3.315 |
| Roseobacteraceae | 127 | 1.395 |
| Pseudonocardiaceae | 123 | 2.280 |
| Paenibacillaceae | 93 | 1.651 |
| Enterococcaceae | 93 | 2.935 |
| Microbacteriaceae | 91 | 2.789 |
| Micromonosporaceae | 90 | 1.609 |
| Sphingomonadaceae | 81 | 1.028 |
| Micrococcaceae | 71 | 2.613 |
| Streptosporangiaceae | 68 | 1.480 |
| Lactobacillaceae | 66 | 2.559 |
| Sphingobacteriaceae | 55 | 1.218 |
| Halomonadaceae | 52 | 2.815 |

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
