# External Tool Benchmark Status

This file tracks the apples-to-apples benchmark setup for external tools
on the same held-out BacDive/MediaDive strains used by the dry-lab media
recommender benchmark.

## Held-Out Manifest

- Manifest: `artifacts/external_benchmark_manifest.parquet`
- Rows: 25,728
- Unique genome accessions: 16,154
- Media labels retained: 40
- Fold counts: {"0": 5146, "1": 5146, "2": 5146, "3": 5145, "4": 5145}

Label coverage:

| Target | Labeled rows |
|---|---:|
| Temperature | 25,727 |
| pH | 2,984 |
| Salt | 2,486 |
| Oxygen | 9,283 |
| Medium | 21,050 |

## Local Requirements

- FASTA directory: `data/external_benchmark_fastas`
- FASTAs present: 0 / 16,154 (0.00%)
- FASTA download smoke run: {"attempted": 0, "downloaded": 0, "failed": 0}

| Tool | Local command | Status |
|---|---|---|
| GenomeSPOT | `` | missing |
| CarveMe | `` | missing |
| gapseq | `` | missing |

## Verdict

External baseline execution is not ready on this machine yet: the full held-out FASTA set and one or more external tool binaries/databases are missing.

## Next Commands

Use the manifest to run each external tool against the same rows and folds.
The medium-feasibility tools should be scored by whether at least one known
MediaDive medium is feasible or closest among the tool's predicted feasible
media/metabolite environments.

```bash
PYTHONPATH=src uv run --python 3.11 python scripts/42_prepare_external_benchmarks.py \
  --download-fastas 10
```

For the full benchmark, download the complete FASTA set into the FASTA
directory above, install the external tools plus their databases, then run
tool-specific inference using the `bacdive_id`, `fold`, and
`genome_accession` columns from the manifest.
