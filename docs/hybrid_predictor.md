# Hybrid Predictor

The hybrid predictor combines the strongest current model for each phenotype:

- Temperature: saved tabular XGBoost phenotype head
- pH: saved tabular XGBoost phenotype head
- Salt: saved tabular XGBoost phenotype head
- Oxygen: fold-0 all-task LoRA checkpoint

Script:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev --extra embeddings python scripts/39_predict_hybrid.py \
  --features data/training_table.parquet \
  --marker-sequences data/marker_sequences.jsonl \
  --limit 25 \
  --device mps \
  --output artifacts/hybrid_predictions.parquet
```

Use `--device cuda` on a GPU host, `--device mps` on Apple Silicon, or omit the flag
to let PyTorch choose CUDA when available and CPU otherwise. CPU LoRA inference is
slow because ESM-2 encodes multiple marker proteins per genome.

For larger uncultured-genome batches, use chunked output so progress is durable:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev --extra embeddings python scripts/39_predict_hybrid.py \
  --features artifacts/uncultured_predictions.parquet \
  --marker-sequences data/uncultured_marker_sequences.jsonl \
  --join left \
  --reuse-existing-tabular \
  --device mps \
  --batch-size 2 \
  --chunk-size 250 \
  --chunk-output-dir artifacts/hybrid_chunks \
  --resume-chunks \
  --progress-every 25 \
  --output artifacts/hybrid_predictions.parquet
```

`--resume-chunks` skips existing chunk files and combines all expected chunks into
the final output when the run finishes. `--reuse-existing-tabular` keeps previously
materialized temperature, pH, salt, and media outputs while replacing oxygen with
LoRA where marker sequences are available.

## Inputs

`--features` accepts `.parquet`, `.csv`, `.json`, or `.jsonl` feature rows. Rows must
include `genome_accession` by default. The tabular heads load their feature column
order from `models/phenotype/feature_cols.json`.

`--marker-sequences` must be a JSONL file with the same schema as
`data/marker_sequences.jsonl`:

```json
{
  "bacdive_id": 1,
  "genome_accession": "GCF_004341595.1",
  "by_category": {
    "temperature": ["..."],
    "ph": ["..."],
    "oxygen": ["..."],
    "salt": ["..."],
    "vitamin": ["..."],
    "nitrogen": ["..."],
    "carbon": ["..."],
    "special": ["..."]
  }
}
```

For uncultured genomes, first prepare a marker-sequence JSONL for those same
`genome_accession` values. The existing `data/marker_sequences.jsonl` is keyed to
BacDive training rows, not the uncultured candidate table.

## Output

The output includes:

- `pred_optimal_temperature_c` plus 80% interval columns
- `pred_optimal_ph` plus 80% interval columns
- `pred_salt_tolerance_pct` plus 80% interval columns
- `pred_oxygen_requirement`
- `pred_oxygen_requirement_confidence`
- one probability column per LoRA oxygen class

The default output is `artifacts/hybrid_predictions.parquet`; `.csv`, `.json`, and
`.jsonl` outputs are also supported.
