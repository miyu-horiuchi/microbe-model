# Fold 0 Anaerobe-Weighted LoRA Comparison

This run tested whether increasing the oxygen cross-entropy weight for
`anaerobe` improves the fold 0 oxygen result.

Experiment:

- Base setup: all-task LoRA, 1 epoch, fold 0
- Oxygen class weights: `aerobe=1.0`, `anaerobe=1.5`,
  `facultative_anaerobe=1.0`, `microaerobe=1.0`
- Output directory: `artifacts/lora_weighted_anaerobe`

## Training Validation Metric

| Checkpoint | Oxygen macro F1 | Oxygen n | Notes |
|---|---:|---:|---|
| `artifacts/lora/fold0_best.pt` | 0.944823 | 2266 | Original all-task LoRA |
| `artifacts/lora_weighted_anaerobe/fold0_best.pt` | 0.944776 | 2266 | Anaerobe-weighted all-task LoRA |

## Detailed Diagnostic

The weighted checkpoint diagnostic reports:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| aerobe | 0.958177 | 0.975842 | 0.966929 | 1573 |
| anaerobe | 0.942771 | 0.903319 | 0.922623 | 693 |

Confusion matrix:

| True \ Pred | aerobe | anaerobe | facultative_anaerobe | microaerobe |
|---|---:|---:|---:|---:|
| aerobe | 1535 | 38 | 0 | 0 |
| anaerobe | 67 | 626 | 0 | 0 |
| facultative_anaerobe | 0 | 0 | 0 | 0 |
| microaerobe | 0 | 0 | 0 | 0 |

## Conclusion

The `anaerobe=1.5` class-weight experiment does not beat the original all-task
LoRA on fold 0. It slightly improves anaerobe recall in the detailed diagnostic,
but the overall oxygen macro F1 is fractionally lower than the original all-task
checkpoint.

Keep `artifacts/lora/fold0_best.pt` as the best fold 0 checkpoint for now.
