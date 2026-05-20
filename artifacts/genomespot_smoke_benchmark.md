# GenomeSPOT Held-Out Smoke Benchmark

GenomeSPOT was run on rows selected from the same held-out manifest used
by the microbe-model media benchmark. This is a smoke benchmark unless
`limit` equals the full manifest size.

## Setup

- Manifest: `artifacts/external_benchmark_manifest.parquet`
- Limit: 5
- Required labels: temperature, ph, salt, oxygen
- GenomeSPOT source: `data/external_tools/GenomeSPOT-main`
- FASTA directory: `data/external_benchmark_fastas`

## Results

- OK: 5 / 5
- Failed/skipped: 0
- Mean runtime per OK genome: 7.02s
- Temperature MAE: 6.765 C
- pH MAE: 0.839
- Salt MAE: 2.186%

## Notes

GenomeSPOT oxygen is a tolerant/not-tolerant label, while microbe-model
uses BacDive oxygen categories. The smoke report keeps raw labels rather
than forcing an evaluation mapping that may hide label-definition mismatch.
