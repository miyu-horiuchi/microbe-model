# LoRA Oxygen Diagnostics

Checkpoint: `artifacts/lora_weighted_anaerobe/fold0_best.pt`

- Labeled validation rows: `2266`
- Accuracy: `0.9537`
- Macro F1 (supported classes): `0.9448`
- Macro F1 (all configured classes): `0.4724`

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| aerobe | 0.9582 | 0.9758 | 0.9669 | 1573 | 1602 |
| anaerobe | 0.9428 | 0.9033 | 0.9226 | 693 | 664 |
| facultative_anaerobe | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| microaerobe | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

## Confusion Matrix

| True \ Pred | aerobe | anaerobe | facultative_anaerobe | microaerobe |
|---|---:|---:|---:|---:|
| aerobe | 1535 | 38 | 0 | 0 |
| anaerobe | 67 | 626 | 0 | 0 |
| facultative_anaerobe | 0 | 0 | 0 | 0 |
| microaerobe | 0 | 0 | 0 | 0 |

## High-Confidence Wrong Predictions

| BacDive ID | Genome | Group | True | Pred | Confidence | True Prob. | Margin |
|---:|---|---|---|---|---:|---:|---:|
| 499 | GCA_000429505 | Alteromonadaceae | anaerobe | aerobe | 0.9974 | 0.0026 | 0.9947 |
| 481 | GCA_003363485 | Alteromonadaceae | anaerobe | aerobe | 0.9961 | 0.0039 | 0.9923 |
| 168525 | GCA_006386545 | Sphaerotilaceae | anaerobe | aerobe | 0.9953 | 0.0047 | 0.9905 |
| 6314 | GCA_900475835 | Jonesiaceae | anaerobe | aerobe | 0.9951 | 0.0049 | 0.9902 |
| 498 | GCA_000429485 | Alteromonadaceae | anaerobe | aerobe | 0.9944 | 0.0056 | 0.9889 |
| 17841 | GCA_000975055 | Demequinaceae | anaerobe | aerobe | 0.9938 | 0.0062 | 0.9876 |
| 483 | GCA_000421165 | Alteromonadaceae | anaerobe | aerobe | 0.9904 | 0.0096 | 0.9809 |
| 23140 | GCA_016925555 | Mycoplasmataceae | aerobe | anaerobe | 0.9810 | 0.0187 | 0.9623 |
| 17840 | GCA_000975035 | Demequinaceae | anaerobe | aerobe | 0.9799 | 0.0201 | 0.9598 |
| 148058 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9776 | 0.0223 | 0.9553 |
| 149706 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9776 | 0.0223 | 0.9553 |
| 156346 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9776 | 0.0223 | 0.9553 |
| 140022 | GCF_014068355.1 | Mycoplasmataceae | aerobe | anaerobe | 0.9761 | 0.0236 | 0.9525 |
| 164735 | GCF_943590815.1 | Enterobacteriaceae | anaerobe | aerobe | 0.9693 | 0.0307 | 0.9386 |
| 133974 | GCA_001544515 | Burkholderiaceae | anaerobe | aerobe | 0.9651 | 0.0348 | 0.9303 |
| 133484 | GCA_900102145 | Thermaceae | aerobe | anaerobe | 0.9554 | 0.0443 | 0.9111 |
| 154004 | GCF_982443925.1 | Enterobacteriaceae | anaerobe | aerobe | 0.9536 | 0.0463 | 0.9074 |
| 168303 | GCA_004307015 | Thermaceae | aerobe | anaerobe | 0.9495 | 0.0502 | 0.8993 |
| 133976 | GCA_001544495 | Burkholderiaceae | anaerobe | aerobe | 0.9473 | 0.0526 | 0.8947 |
| 8608 | GCF_900476065.1 | Metamycoplasmataceae | aerobe | anaerobe | 0.9409 | 0.0584 | 0.8825 |
| 132346 | GCA_002259755 | Bifidobacteriaceae | aerobe | anaerobe | 0.9375 | 0.0624 | 0.8751 |
| 133975 | GCA_001544475 | Burkholderiaceae | anaerobe | aerobe | 0.9338 | 0.0661 | 0.8677 |
| 132158 | GCA_000807275 | Orbaceae | anaerobe | aerobe | 0.9336 | 0.0663 | 0.8673 |
| 140694 | GCA_003336745 | Thermaceae | aerobe | anaerobe | 0.9297 | 0.0698 | 0.8599 |
| 133991 | GCA_039544205 | Ferrimonadaceae | anaerobe | aerobe | 0.9252 | 0.0746 | 0.8506 |
