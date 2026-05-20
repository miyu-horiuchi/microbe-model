# Media Recommender Dry-Lab Benchmark

This benchmark hides known BacDive/MediaDive strain-medium links and asks
whether the genome-only recommender recovers at least one known medium in
the top-k ranked recommendations.

## Setup

- Split mode: `family`
- Folds: 5
- Evaluation strains: 21050
- Media labels: 40
- Feature columns: 1113
- XGBoost trees per medium per fold: 100

## Ranking Metrics

| Method | MRR | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost medium recommender | 0.588 | 0.450 | 0.660 | 0.775 | 0.722 | 0.194 |
| Global popularity baseline | 0.243 | 0.080 | 0.250 | 0.366 | 0.308 | 0.080 |
| Taxonomic popularity baseline | 0.250 | 0.086 | 0.259 | 0.372 | 0.312 | 0.081 |

## Per-Medium AUC

- Valid media with both classes: 40
- Median ROC-AUC: 0.910
- Median PR-AUC: 0.183

Top media by PR-AUC:

| Medium | Positives | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| 65 GYM STREPTOMYCES MEDIUM | 3434 | 0.901 | 0.976 |
| 9 VY/2 AGAR | 2465 | 0.800 | 0.922 |
| 1076b SP4-Z MEDIUM | 116 | 0.777 | 0.922 |
| 514 BACTO MARINE BROTH (DIFCO 2216) | 1890 | 0.637 | 0.947 |
| 84 ROLLED OATS MINERAL MEDIUM | 1365 | 0.558 | 0.968 |
| 92 TRYPTICASE SOY YEAST EXTRACT MEDIUM | 2899 | 0.447 | 0.869 |
| 11 MRS MEDIUM | 495 | 0.423 | 0.966 |
| 693 COLUMBIA BLOOD MEDIUM | 2503 | 0.419 | 0.878 |
| 830 R2A MEDIUM | 1782 | 0.401 | 0.882 |
| 98 RHIZOBIUM MEDIUM | 151 | 0.343 | 0.969 |

## Interpretation

Hit@k is the easiest practical readout: it measures whether at least one
known growth medium appears in the top-k suggestions. PR-AUC is expected
to be much lower than ROC-AUC because medium labels are sparse and heavily
imbalanced; a high ROC-AUC with modest PR-AUC means the model is useful for
ranking candidates, not for guaranteeing growth.
