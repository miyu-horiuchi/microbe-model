# LoRA Oxygen Diagnostics

Checkpoint: `artifacts/lora/fold0_best.pt`

- Labeled validation rows: `2276`
- Accuracy: `0.9565`
- Macro F1 (supported classes): `0.9460`
- Macro F1 (all configured classes): `0.4730`

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| aerobe | 0.9601 | 0.9797 | 0.9698 | 1623 | 1656 |
| anaerobe | 0.9468 | 0.8989 | 0.9222 | 653 | 620 |
| facultative_anaerobe | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| microaerobe | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

## Confusion Matrix

| True \ Pred | aerobe | anaerobe | facultative_anaerobe | microaerobe |
|---|---:|---:|---:|---:|
| aerobe | 1590 | 33 | 0 | 0 |
| anaerobe | 66 | 587 | 0 | 0 |
| facultative_anaerobe | 0 | 0 | 0 | 0 |
| microaerobe | 0 | 0 | 0 | 0 |

## High-Confidence Wrong Predictions

| BacDive ID | Genome | Group | True | Pred | Confidence | True Prob. | Margin |
|---:|---|---|---|---|---:|---:|---:|
| 499 | GCA_000429505 | Alteromonadaceae | anaerobe | aerobe | 0.9993 | 0.0007 | 0.9986 |
| 481 | GCA_003363485 | Alteromonadaceae | anaerobe | aerobe | 0.9992 | 0.0008 | 0.9983 |
| 498 | GCA_000429485 | Alteromonadaceae | anaerobe | aerobe | 0.9977 | 0.0023 | 0.9954 |
| 168525 | GCA_006386545 | Sphaerotilaceae | anaerobe | aerobe | 0.9975 | 0.0025 | 0.9950 |
| 483 | GCA_000421165 | Alteromonadaceae | anaerobe | aerobe | 0.9959 | 0.0041 | 0.9917 |
| 17841 | GCA_000975055 | Demequinaceae | anaerobe | aerobe | 0.9955 | 0.0045 | 0.9909 |
| 17840 | GCA_000975035 | Demequinaceae | anaerobe | aerobe | 0.9877 | 0.0122 | 0.9755 |
| 164735 | GCF_943590815.1 | Enterobacteriaceae | anaerobe | aerobe | 0.9846 | 0.0154 | 0.9692 |
| 148058 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9819 | 0.0181 | 0.9638 |
| 149706 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9819 | 0.0181 | 0.9638 |
| 156346 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9819 | 0.0181 | 0.9638 |
| 133974 | GCA_001544515 | Burkholderiaceae | anaerobe | aerobe | 0.9799 | 0.0201 | 0.9598 |
| 133484 | GCA_900102145 | Thermaceae | aerobe | anaerobe | 0.9772 | 0.0228 | 0.9544 |
| 5882 | GCA_000215915 | Haloarculaceae | anaerobe | aerobe | 0.9731 | 0.0268 | 0.9463 |
| 133991 | GCA_039544205 | Ferrimonadaceae | anaerobe | aerobe | 0.9621 | 0.0379 | 0.9242 |
| 132346 | GCA_002259755 | Bifidobacteriaceae | aerobe | anaerobe | 0.9601 | 0.0398 | 0.9203 |
| 132623 | GCF_003967195.1 | Granulosicoccaceae | anaerobe | aerobe | 0.9601 | 0.0399 | 0.9202 |
| 24101 | GCA_000242915 | Helicobacteraceae | anaerobe | aerobe | 0.9591 | 0.0409 | 0.9182 |
| 160296 | GCA_010667645 | Bifidobacteriaceae | aerobe | anaerobe | 0.9571 | 0.0428 | 0.9143 |
| 140694 | GCA_003336745 | Thermaceae | aerobe | anaerobe | 0.9524 | 0.0476 | 0.9048 |
| 168303 | GCA_004307015 | Thermaceae | aerobe | anaerobe | 0.9520 | 0.0480 | 0.9039 |
| 134099 | GCA_023349185 | Shewanellaceae | anaerobe | aerobe | 0.9490 | 0.0510 | 0.8980 |
| 159192 | GCA_003721225 | Acidithiobacillaceae | aerobe | anaerobe | 0.9481 | 0.0518 | 0.8963 |
| 154004 | GCF_982443925.1 | Enterobacteriaceae | anaerobe | aerobe | 0.9456 | 0.0543 | 0.8913 |
| 159320 | GCA_009078285 | Bifidobacteriaceae | aerobe | anaerobe | 0.9456 | 0.0543 | 0.8912 |
