# GenomeSPOT Held-Out Benchmark

GenomeSPOT was run on rows selected from the same held-out manifest used
by the microbe-model media benchmark. This run uses a deterministic 5,000
unique-genome subset selected from the family-heldout manifest.

## Setup

- Manifest: `artifacts/external_benchmark_manifest_5k.parquet`
- Limit: 5000
- Required labels: none
- GenomeSPOT source: `data/external_tools/GenomeSPOT-main`
- FASTA directory: `data/external_benchmark_fastas`

## Results

- OK: 5000 / 5000
- Failed/skipped: 0
- Mean runtime per OK genome: 6.37s
- Temperature MAE: 4.393 C
- pH MAE: 0.608
- Salt MAE: 1.981%

## Notes

GenomeSPOT oxygen is a tolerant/not-tolerant label, while microbe-model
uses BacDive oxygen categories. The smoke report keeps raw labels rather
than forcing an evaluation mapping that may hide label-definition mismatch.
