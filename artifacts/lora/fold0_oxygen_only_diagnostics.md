# LoRA Oxygen Diagnostics

Checkpoint: `artifacts/lora/fold0_best_oxygen.pt`

- Labeled validation rows: `2276`
- Accuracy: `0.9446`
- Macro F1 (supported classes): `0.9294`
- Macro F1 (all configured classes): `0.4647`

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| aerobe | 0.9375 | 0.9883 | 0.9622 | 1623 | 1711 |
| anaerobe | 0.9664 | 0.8361 | 0.8966 | 653 | 565 |
| facultative_anaerobe | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| microaerobe | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

## Confusion Matrix

| True \ Pred | aerobe | anaerobe | facultative_anaerobe | microaerobe |
|---|---:|---:|---:|---:|
| aerobe | 1604 | 19 | 0 | 0 |
| anaerobe | 107 | 546 | 0 | 0 |
| facultative_anaerobe | 0 | 0 | 0 | 0 |
| microaerobe | 0 | 0 | 0 | 0 |

## High-Confidence Wrong Predictions

| BacDive ID | Genome | Group | True | Pred | Confidence | True Prob. | Margin |
|---:|---|---|---|---|---:|---:|---:|
| 499 | GCA_000429505 | Alteromonadaceae | anaerobe | aerobe | 0.9991 | 0.0009 | 0.9981 |
| 481 | GCA_003363485 | Alteromonadaceae | anaerobe | aerobe | 0.9988 | 0.0012 | 0.9977 |
| 168525 | GCA_006386545 | Sphaerotilaceae | anaerobe | aerobe | 0.9986 | 0.0014 | 0.9971 |
| 498 | GCA_000429485 | Alteromonadaceae | anaerobe | aerobe | 0.9976 | 0.0024 | 0.9953 |
| 17841 | GCA_000975055 | Demequinaceae | anaerobe | aerobe | 0.9969 | 0.0031 | 0.9939 |
| 483 | GCA_000421165 | Alteromonadaceae | anaerobe | aerobe | 0.9961 | 0.0039 | 0.9922 |
| 133974 | GCA_001544515 | Burkholderiaceae | anaerobe | aerobe | 0.9950 | 0.0049 | 0.9901 |
| 17840 | GCA_000975035 | Demequinaceae | anaerobe | aerobe | 0.9928 | 0.0072 | 0.9856 |
| 164735 | GCF_943590815.1 | Enterobacteriaceae | anaerobe | aerobe | 0.9912 | 0.0087 | 0.9825 |
| 132623 | GCF_003967195.1 | Granulosicoccaceae | anaerobe | aerobe | 0.9903 | 0.0097 | 0.9806 |
| 133976 | GCA_001544495 | Burkholderiaceae | anaerobe | aerobe | 0.9897 | 0.0102 | 0.9795 |
| 148058 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9858 | 0.0141 | 0.9717 |
| 149706 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9858 | 0.0141 | 0.9717 |
| 156346 | GCF_000005845.2 | Enterobacteriaceae | anaerobe | aerobe | 0.9858 | 0.0141 | 0.9717 |
| 133975 | GCA_001544475 | Burkholderiaceae | anaerobe | aerobe | 0.9857 | 0.0142 | 0.9715 |
| 132346 | GCA_002259755 | Bifidobacteriaceae | aerobe | anaerobe | 0.9798 | 0.0201 | 0.9597 |
| 160296 | GCA_010667645 | Bifidobacteriaceae | aerobe | anaerobe | 0.9734 | 0.0264 | 0.9470 |
| 159320 | GCA_009078285 | Bifidobacteriaceae | aerobe | anaerobe | 0.9718 | 0.0281 | 0.9437 |
| 7227 | GCA_000425185 | Pleomorphomonadaceae | aerobe | anaerobe | 0.9599 | 0.0399 | 0.9200 |
| 154004 | GCF_982443925.1 | Enterobacteriaceae | anaerobe | aerobe | 0.9591 | 0.0408 | 0.9183 |
| 24101 | GCA_000242915 | Helicobacteraceae | anaerobe | aerobe | 0.9550 | 0.0445 | 0.9104 |
| 133136 | GCA_030161615 | Demequinaceae | anaerobe | aerobe | 0.9537 | 0.0457 | 0.9080 |
| 159192 | GCA_003721225 | Acidithiobacillaceae | aerobe | anaerobe | 0.9529 | 0.0468 | 0.9060 |
| 8608 | GCF_900476065.1 | Metamycoplasmataceae | aerobe | anaerobe | 0.9278 | 0.0710 | 0.8568 |
| 134099 | GCA_023349185 | Shewanellaceae | anaerobe | aerobe | 0.9243 | 0.0753 | 0.8490 |
