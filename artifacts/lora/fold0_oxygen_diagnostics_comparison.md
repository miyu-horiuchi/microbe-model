# Fold 0 Oxygen Diagnostic Comparison

Both diagnostics were run on the same fold 0 oxygen-labeled validation rows
(`n = 2,276`) using the saved LoRA checkpoints.

| Checkpoint | Accuracy | Macro F1 (supported classes) | Aerobe F1 | Anaerobe F1 | Aerobe -> Anaerobe | Anaerobe -> Aerobe |
|---|---:|---:|---:|---:|---:|---:|
| `fold0_best.pt` | 0.9565 | 0.9460 | 0.9698 | 0.9222 | 33 | 66 |
| `fold0_best_oxygen.pt` | 0.9446 | 0.9294 | 0.9622 | 0.8966 | 19 | 107 |

## Conclusion

The all-task LoRA checkpoint remains the better oxygen model on fold 0. The
oxygen-only checkpoint improves aerobe recall slightly and reduces aerobe
misclassified as anaerobe, but it substantially increases anaerobe misclassified
as aerobe. That drop in anaerobe recall drives the lower macro F1.

The fold 0 validation split has no supported `facultative_anaerobe` or
`microaerobe` rows, so the primary comparison should use macro F1 over supported
classes. Full configured-class macro F1 is also recorded in the JSON files for
visibility, but it is not useful for selecting between these two checkpoints on
this fold.
