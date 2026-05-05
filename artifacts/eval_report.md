# microbe-model — v0 baseline eval report

_Generated: 2026-05-05T06:56:14+00:00_

## TL;DR

- **`optimal_temperature_c`**: MAE = **2.94** (vs always-predict-mean 4.98, **+41%**)
- **`optimal_ph`**: MAE = **0.51** (vs always-predict-mean 0.55, **+7%**)
- **`oxygen_requirement`**: macro-F1 = **0.341** (vs always-predict-majority 0.059, **+479%**)
- **`salt_tolerance_pct`**: MAE = **2.52** (vs always-predict-mean 2.83, **+11%**)

Trained on **46,029** strains with **418** genome-derived features. Cross-validation: 5-fold GroupKFold by taxonomic family.

## Corpus

- Total strains in feature table: **46,029**
- Labeled-strain counts by target:
  - `optimal_temperature_c`: 45,621
  - `optimal_ph`: 5,103
  - `oxygen_requirement`: 21,639
  - `salt_tolerance_pct`: 6,330

## Target distributions

- `optimal_temperature_c`: n=45,621, mean=32.24, std=7.13, p10=27.50, median=30.00, p90=37.00
- `optimal_ph`: n=5,103, mean=7.19, std=0.82, p10=6.50, median=7.00, p90=8.00
- `salt_tolerance_pct`: n=6,330, mean=3.93, std=4.03, p10=0.00, median=3.00, p90=8.00
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
| `optimal_temperature_c` | regression | 45,621 | MAE=2.939 | MAE=4.981 | +41.0% |
| `optimal_ph` | regression | 5,103 | MAE=0.509 | MAE=0.546 | +6.8% |
| `oxygen_requirement` | classification | 21,639 | F1=0.341 | F1=0.059 | +479.5% |
| `salt_tolerance_pct` | regression | 6,330 | MAE=2.517 | MAE=2.827 | +11.0% |

### `optimal_temperature_c` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 3.104 | n=36,496 | n=9,125 |
| 2 | mae = 2.736 | n=36,497 | n=9,124 |
| 3 | mae = 3.146 | n=36,497 | n=9,124 |
| 4 | mae = 3.277 | n=36,497 | n=9,124 |
| 5 | mae = 2.435 | n=36,497 | n=9,124 |

**Top 10 features for `optimal_temperature_c`:**

- `ivywrel_frac` — 0.1267
- `iso_cat2_thermophilic_gt45_c` — 0.0299
- `n_predicted_cds` — 0.0251
- `iso_cat2_human` — 0.0209
- `iso_cat1_infection` — 0.0206
- `iso_cat2_patient` — 0.0178
- `aa_frac_C` — 0.0150
- `genome_size_nt` — 0.0122
- `aa_frac_D` — 0.0113
- `codon_AGG` — 0.0109

### `optimal_ph` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 0.456 | n=4,082 | n=1,021 |
| 2 | mae = 0.626 | n=4,082 | n=1,021 |
| 3 | mae = 0.528 | n=4,082 | n=1,021 |
| 4 | mae = 0.480 | n=4,083 | n=1,020 |
| 5 | mae = 0.454 | n=4,083 | n=1,020 |

**Top 10 features for `optimal_ph`:**

- `iso_cat2_acidic` — 0.0522
- `iso_cat2_alkaline` — 0.0435
- `neg_charged_frac` — 0.0169
- `aa_frac_E` — 0.0086
- `tetra_CTCT` — 0.0084
- `aa_frac_H` — 0.0080
- `mean_isoelectric_point` — 0.0076
- `tetra_CACT` — 0.0074
- `tetra_AGAC` — 0.0071
- `tetra_AGGT` — 0.0059

### `oxygen_requirement` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | f1_macro = 0.315 | n=17,311 | n=4,328 |
| 2 | f1_macro = 0.382 | n=17,311 | n=4,326 |
| 3 | f1_macro = 0.344 | n=17,311 | n=4,328 |
| 4 | f1_macro = 0.259 | n=17,311 | n=4,328 |
| 5 | f1_macro = 0.406 | n=17,312 | n=4,327 |

**Top 10 features for `oxygen_requirement`:**

- `codon_ATA` — 0.0414
- `iso_cat1_host` — 0.0260
- `n_predicted_cds` — 0.0252
- `aa_frac_C` — 0.0191
- `iso_cat1_environmental` — 0.0165
- `codon_CGT` — 0.0148
- `iso_cat1_engineered` — 0.0138
- `genome_size_nt` — 0.0113
- `iso_cat2_human` — 0.0102
- `codon_TAA` — 0.0090

### `salt_tolerance_pct` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 2.218 | n=5,064 | n=1,266 |
| 2 | mae = 2.249 | n=5,064 | n=1,266 |
| 3 | mae = 2.819 | n=5,064 | n=1,266 |
| 4 | mae = 2.350 | n=5,064 | n=1,266 |
| 5 | mae = 2.948 | n=5,064 | n=1,266 |

**Top 10 features for `salt_tolerance_pct`:**

- `aa_frac_C` — 0.0298
- `neg_charged_frac` — 0.0278
- `tetra_ATCC` — 0.0183
- `iso_cat1_environmental` — 0.0142
- `tetra_GACT` — 0.0121
- `iso_cat2_saline` — 0.0114
- `codon_TGC` — 0.0112
- `tetra_CGTT` — 0.0094
- `codon_CGT` — 0.0087
- `iso_cat2_industrial` — 0.0085

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
| `neg_charged_frac` | +0.304 | 1.6e-109 |
| `mean_isoelectric_point` | -0.278 | 1.8e-91 |
| `aa_frac_E` | +0.256 | 4.5e-77 |
| `iso_cat2_alkaline` | +0.165 | 2.5e-32 |
| `ivywrel_frac` | +0.159 | 2.4e-30 |
| `codon_AAG` | -0.154 | 1.7e-28 |
| `codon_CGA` | +0.153 | 5.8e-28 |
| `codon_TGC` | -0.151 | 2.6e-27 |
| `iso_cat2_saline` | +0.137 | 8.9e-23 |
| `tetra_CACT` | +0.135 | 4.3e-22 |

### `salt_tolerance_pct`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `tetra_AGTC` | +0.270 | 4.0e-106 |
| `tetra_GACT` | +0.268 | 1.4e-104 |
| `neg_charged_frac` | +0.221 | 3.9e-71 |
| `ivywrel_frac` | +0.221 | 8.4e-71 |
| `aa_frac_C` | -0.202 | 4.7e-59 |
| `iso_cat1_environmental` | -0.193 | 2.6e-54 |
| `n_contigs` | -0.181 | 1.0e-47 |
| `mean_cds_aa_length` | -0.177 | 8.2e-46 |
| `tetra_ACTC` | +0.176 | 4.5e-45 |
| `tetra_GAGT` | +0.173 | 1.5e-43 |

## Per-family error breakdown (regression targets)

Top 15 most-represented families, MAE per family. Highlights where the model is doing well vs. struggling.

### `optimal_temperature_c`

| Family | n | MAE |
|---|---|---|
| Enterobacteriaceae | 2662 | 4.086 |
| Streptomycetaceae | 2212 | 1.919 |
| Bacillaceae | 1886 | 3.195 |
| Lactobacillaceae | 1732 | 3.537 |
| Pseudomonadaceae | 1621 | 2.576 |
| Myxococcaceae | 1546 | 0.403 |
| Streptococcaceae | 1170 | 2.367 |
| Staphylococcaceae | 1068 | 4.288 |
| Flavobacteriaceae | 981 | 4.202 |
| Corynebacteriaceae | 900 | 2.231 |
| Moraxellaceae | 890 | 3.514 |
| Paenibacillaceae | 760 | 2.967 |
| Microbacteriaceae | 734 | 2.482 |
| Micrococcaceae | 719 | 2.991 |
| Nocardiaceae | 715 | 2.679 |

### `optimal_ph`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 355 | 0.391 |
| Bacillaceae | 298 | 0.678 |
| Roseobacteraceae | 204 | 0.400 |
| Paenibacillaceae | 139 | 0.435 |
| Microbacteriaceae | 120 | 0.438 |
| Sphingobacteriaceae | 114 | 0.353 |
| Sphingomonadaceae | 102 | 0.346 |
| Streptomycetaceae | 98 | 0.599 |
| Pseudonocardiaceae | 93 | 0.495 |
| Halomonadaceae | 82 | 0.603 |
| Micrococcaceae | 82 | 0.619 |
| Nocardioidaceae | 80 | 0.490 |
| Paracoccaceae | 76 | 0.564 |
| Alteromonadaceae | 71 | 0.349 |
| Erythrobacteraceae | 68 | 0.423 |

### `salt_tolerance_pct`

| Family | n | MAE |
|---|---|---|
| Streptococcaceae | 340 | 0.891 |
| Flavobacteriaceae | 312 | 1.834 |
| Bacillaceae | 310 | 3.417 |
| Streptomycetaceae | 309 | 2.116 |
| Pseudomonadaceae | 196 | 4.802 |
| Corynebacteriaceae | 194 | 3.853 |
| Vibrionaceae | 173 | 2.872 |
| Microbacteriaceae | 166 | 2.616 |
| Paenibacillaceae | 150 | 2.096 |
| Roseobacteraceae | 143 | 1.556 |
| Pseudonocardiaceae | 142 | 2.400 |
| Moraxellaceae | 126 | 2.581 |
| Nocardiaceae | 125 | 2.899 |
| Enterococcaceae | 111 | 1.723 |
| Alcaligenaceae | 104 | 4.454 |

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
