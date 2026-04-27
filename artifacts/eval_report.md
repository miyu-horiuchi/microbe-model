# microbe-model — v0 baseline eval report

_Generated: 2026-04-27T11:37:17+00:00_

## TL;DR

- **`optimal_temperature_c`**: MAE = **3.28** (vs always-predict-mean 5.53, **+41%**)
- **`optimal_ph`**: MAE = **0.52** (vs always-predict-mean 0.55, **+5%**)
- **`oxygen_requirement`**: macro-F1 = **0.279** (vs always-predict-majority 0.072, **+289%**)
- **`salt_tolerance_pct`**: MAE = **2.51** (vs always-predict-mean 2.72, **+8%**)

Trained on **17,047** strains with **353** genome-derived features. Cross-validation: 5-fold GroupKFold by taxonomic family.

## Corpus

- Total strains in feature table: **17,047**
- Labeled-strain counts by target:
  - `optimal_temperature_c`: 17,007
  - `optimal_ph`: 4,652
  - `oxygen_requirement`: 10,426
  - `salt_tolerance_pct`: 4,793

## Target distributions

- `optimal_temperature_c`: n=17,007, mean=31.96, std=8.57, p10=25.00, median=30.00, p90=37.00
- `optimal_ph`: n=4,652, mean=7.19, std=0.83, p10=6.50, median=7.00, p90=8.00
- `salt_tolerance_pct`: n=4,793, mean=3.56, std=4.11, p10=0.00, median=2.50, p90=8.00
- `oxygen_requirement`:
  - `aerobe`: 4,973
  - `anaerobe`: 2,120
  - `facultative anaerobe`: 1,226
  - `obligate aerobe`: 1,027
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
| `optimal_temperature_c` | regression | 17,007 | MAE=3.280 | MAE=5.531 | +40.7% |
| `optimal_ph` | regression | 4,652 | MAE=0.520 | MAE=0.547 | +4.8% |
| `oxygen_requirement` | classification | 10,426 | F1=0.279 | F1=0.072 | +288.9% |
| `salt_tolerance_pct` | regression | 4,793 | MAE=2.510 | MAE=2.721 | +7.7% |

### `optimal_temperature_c` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 2.793 | n=13,605 | n=3,402 |
| 2 | mae = 3.529 | n=13,605 | n=3,402 |
| 3 | mae = 3.461 | n=13,606 | n=3,401 |
| 4 | mae = 3.130 | n=13,606 | n=3,401 |
| 5 | mae = 3.486 | n=13,606 | n=3,401 |

**Top 10 features for `optimal_temperature_c`:**

- `ivywrel_frac` — 0.1977
- `n_predicted_cds` — 0.0250
- `pos_charged_frac` — 0.0196
- `aa_frac_E` — 0.0131
- `aa_frac_C` — 0.0119
- `codon_TTG` — 0.0099
- `codon_TGA` — 0.0099
- `codon_AGG` — 0.0098
- `tetra_GCAA` — 0.0084
- `aa_frac_S` — 0.0081

### `optimal_ph` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 0.475 | n=3,721 | n=931 |
| 2 | mae = 0.613 | n=3,721 | n=931 |
| 3 | mae = 0.479 | n=3,722 | n=930 |
| 4 | mae = 0.529 | n=3,722 | n=930 |
| 5 | mae = 0.505 | n=3,722 | n=930 |

**Top 10 features for `optimal_ph`:**

- `neg_charged_frac` — 0.0211
- `tetra_TGCT` — 0.0141
- `aa_frac_H` — 0.0120
- `tetra_CACT` — 0.0097
- `tetra_AGAC` — 0.0088
- `tetra_GAGA` — 0.0081
- `tetra_TCTC` — 0.0077
- `ivywrel_frac` — 0.0077
- `aa_frac_E` — 0.0076
- `tetra_CTCT` — 0.0069

### `oxygen_requirement` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | f1_macro = 0.204 | n=8,340 | n=2,085 |
| 2 | f1_macro = 0.255 | n=8,341 | n=2,085 |
| 3 | f1_macro = 0.286 | n=8,341 | n=2,085 |
| 4 | f1_macro = 0.355 | n=8,341 | n=2,085 |
| 5 | f1_macro = 0.296 | n=8,341 | n=2,085 |

**Top 10 features for `oxygen_requirement`:**

- `codon_ATA` — 0.0381
- `aa_frac_C` — 0.0217
- `genome_size_nt` — 0.0194
- `tetra_CAAA` — 0.0173
- `codon_CAA` — 0.0145
- `aa_frac_Q` — 0.0124
- `tetra_TCAA` — 0.0123
- `n_predicted_cds` — 0.0117
- `aa_frac_K` — 0.0098
- `aa_frac_W` — 0.0088

### `salt_tolerance_pct` — fold-by-fold

| Fold | Metric | Train | Test |
|---|---|---|---|
| 1 | mae = 2.557 | n=3,834 | n=959 |
| 2 | mae = 2.164 | n=3,834 | n=959 |
| 3 | mae = 2.777 | n=3,834 | n=959 |
| 4 | mae = 2.654 | n=3,835 | n=958 |
| 5 | mae = 2.398 | n=3,835 | n=958 |

**Top 10 features for `salt_tolerance_pct`:**

- `aa_frac_C` — 0.0492
- `neg_charged_frac` — 0.0365
- `tetra_CGTT` — 0.0160
- `tetra_GAAA` — 0.0123
- `tetra_GACT` — 0.0110
- `tetra_AACC` — 0.0091
- `codon_CGT` — 0.0073
- `tetra_TGTG` — 0.0071
- `tetra_AGGA` — 0.0066
- `tetra_GGAG` — 0.0065

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
| `codon_TGG` | -0.309 | 0.0e+00 |
| `tetra_TCTT` | +0.300 | 0.0e+00 |
| `pos_charged_frac` | +0.299 | 0.0e+00 |
| `tetra_AAGA` | +0.298 | 0.0e+00 |

### `optimal_ph`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `neg_charged_frac` | +0.308 | 4.5e-103 |
| `mean_isoelectric_point` | -0.276 | 4.4e-82 |
| `aa_frac_E` | +0.260 | 6.3e-73 |
| `ivywrel_frac` | +0.166 | 3.8e-30 |
| `codon_AAG` | -0.163 | 6.2e-29 |
| `codon_TGC` | -0.149 | 1.9e-24 |
| `codon_CGA` | +0.149 | 2.2e-24 |
| `tetra_CACT` | +0.134 | 3.8e-20 |
| `tetra_AGTG` | +0.133 | 6.4e-20 |
| `tetra_ACTC` | +0.119 | 4.0e-16 |

### `salt_tolerance_pct`

| Feature | Spearman ρ | p-value |
|---|---|---|
| `tetra_AGTC` | +0.232 | 9.5e-60 |
| `tetra_GACT` | +0.232 | 1.5e-59 |
| `neg_charged_frac` | +0.227 | 3.2e-57 |
| `mean_isoelectric_point` | -0.204 | 2.9e-46 |
| `ivywrel_frac` | +0.196 | 1.4e-42 |
| `aa_frac_C` | -0.187 | 7.3e-39 |
| `tetra_ACTC` | +0.176 | 1.4e-34 |
| `tetra_GAGT` | +0.173 | 2.5e-33 |
| `tetra_ATGC` | -0.164 | 3.7e-30 |
| `tetra_TCAC` | +0.163 | 5.0e-30 |

## Per-family error breakdown (regression targets)

Top 15 most-represented families, MAE per family. Highlights where the model is doing well vs. struggling.

### `optimal_temperature_c`

| Family | n | MAE |
|---|---|---|
| Streptomycetaceae | 798 | 1.311 |
| Bacillaceae | 643 | 4.423 |
| Flavobacteriaceae | 631 | 4.303 |
| Lactobacillaceae | 471 | 3.389 |
| Enterobacteriaceae | 439 | 3.719 |
| Microbacteriaceae | 396 | 2.467 |
| Pseudomonadaceae | 388 | 2.254 |
| Roseobacteraceae | 341 | 3.054 |
| Paenibacillaceae | 319 | 3.319 |
| Pseudonocardiaceae | 306 | 2.325 |
| Moraxellaceae | 260 | 4.196 |
| Sphingomonadaceae | 256 | 1.890 |
| Streptococcaceae | 251 | 3.510 |
| Clostridiaceae | 247 | 4.372 |
| Vibrionaceae | 237 | 3.256 |

### `optimal_ph`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 323 | 0.412 |
| Bacillaceae | 273 | 0.689 |
| Roseobacteraceae | 192 | 0.389 |
| Paenibacillaceae | 126 | 0.442 |
| Microbacteriaceae | 112 | 0.477 |
| Sphingobacteriaceae | 100 | 0.395 |
| Sphingomonadaceae | 96 | 0.387 |
| Streptomycetaceae | 92 | 0.546 |
| Pseudonocardiaceae | 85 | 0.555 |
| Halomonadaceae | 81 | 0.566 |
| Nocardioidaceae | 74 | 0.495 |
| Paracoccaceae | 71 | 0.577 |
| Micrococcaceae | 71 | 0.598 |
| Erythrobacteraceae | 68 | 0.450 |
| Alteromonadaceae | 68 | 0.365 |

### `salt_tolerance_pct`

| Family | n | MAE |
|---|---|---|
| Flavobacteriaceae | 285 | 1.711 |
| Streptomycetaceae | 283 | 2.141 |
| Bacillaceae | 246 | 3.508 |
| Microbacteriaceae | 140 | 2.795 |
| Pseudonocardiaceae | 134 | 2.345 |
| Roseobacteraceae | 134 | 1.794 |
| Paenibacillaceae | 125 | 2.184 |
| Pseudomonadaceae | 110 | 4.033 |
| Vibrionaceae | 99 | 2.488 |
| Sphingomonadaceae | 92 | 1.809 |
| Micromonosporaceae | 88 | 1.634 |
| Micrococcaceae | 85 | 3.008 |
| Nocardiaceae | 84 | 2.674 |
| Streptococcaceae | 82 | 1.180 |
| Lactobacillaceae | 78 | 1.852 |

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
