# Overnight run — summary

_Written 2026-04-27T02:37+00:00_

## Pipeline status

- ✅ BacDive scan: **100,866** strains pulled
  - 19,637 have genome accessions
  - 50,384 have optimal_temperature_c labels
  - **17,054** strains are training-ready (genome + T_opt)
- ✅ Featurize: complete
  - Processed: 17,094 / 17,094
  - Successful: 17,065 (99.8%)
  - Failed: 29 (mostly suppressed/withdrawn NCBI assemblies)
- ✅ Training: see `artifacts/train.log` for stdout
- ✅ Eval report: **`artifacts/eval_report.md`**

## What to read first

1. Open **`artifacts/eval_report.md`** — headline metrics + per-target detail.
2. Check `git log --oneline` to see the commit timeline.

## Files of interest

- ✅ `artifacts/eval_report.md` — headline result + metrics 9.6 KB
- ✅ `artifacts/baseline_results.json` — machine-readable per-fold scores 8.8 KB
- ✅ `data/bacdive_phenotypes.parquet` — phenotype labels (gitignored) 1.7 MB
- ✅ `data/features.parquet` — extracted genome features (gitignored) 5.4 MB
- ✅ `data/training_table.parquet` — merged + group-keyed table used for training (gitignored) 5.8 MB

## Commits since yesterday

- 17518a3 Final overnight commit: trained baseline + eval report + summary
- 72e12e7 Make OVERNIGHT_SUMMARY.md write atomic (avoid race with regen loop)
- 2ea77d1 Add v1 composition features (tetranucleotides + codon usage)
- 316196d Fix predictions parquet type mix + plumb feature_cols through eval
- 7db9544 Add tests for explore module (correlations + class means)
- a22773f Harden post-featurize chain: each phase runs even if previous fails
- eb37476 Add feature↔target correlation analysis to eval report
- a7d692a Update README to reflect current state
- 401687e Eval report enhancements: TL;DR + per-strain predictions + per-family error
- 82997f4 Fix classification fold bug + add end-to-end integration tests
- 8d52535 Add eval report generator + training table persistence + group-col override
- 33535e5 Streaming fetch+featurize pipeline + 6× pyrodigal speedup + GCA version resolution
- e945ca9 Rewrite BacDive client for v2 public API (no auth required)
- 208a477 Scaffold v0: BacDive + NCBI ingestion, genome feature extractor, XGBoost baseline

## Reminders

- The NCBI API key was pasted into chat earlier in this session — rotate it at ncbi.nlm.nih.gov → Account Settings → API Key Management → Revoke + Create New.
- The `caffeinate -dimsu` you started for the overnight run can be stopped now (`Ctrl+C` in that terminal).
- Display sleep / Battery settings: revert to your normal preferences if you changed them last night.
