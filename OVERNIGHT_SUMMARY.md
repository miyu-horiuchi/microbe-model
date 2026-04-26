# Overnight run — summary

_Written 2026-04-26T23:03+00:00_

## Pipeline status

- ✅ BacDive scan: **100,866** strains pulled
  - 19,637 have genome accessions
  - 50,384 have optimal_temperature_c labels
  - **17,054** strains are training-ready (genome + T_opt)
- 🟡 Featurize: in progress (32%)
  - Processed: 5,489 / 17,094
  - Successful: 5,473 (99.7%)
  - Failed: 16 (mostly suppressed/withdrawn NCBI assemblies)
- ⏭ Training: not yet run (waits for featurize completion)
- ⏭ Eval report: not yet generated

## What to read first

1. Wait for `artifacts/eval_report.md` to be generated, then open it.
2. Check `git log --oneline` to see the commit timeline.

## Files of interest

- — `artifacts/eval_report.md` — headline result + metrics 
- — `artifacts/baseline_results.json` — machine-readable per-fold scores 
- ✅ `data/bacdive_phenotypes.parquet` — phenotype labels (gitignored) 1.7 MB
- — `data/features.parquet` — extracted genome features (gitignored) 
- — `data/training_table.parquet` — merged + group-keyed table used for training (gitignored) 

## Commits since yesterday

- 82997f4 Fix classification fold bug + add end-to-end integration tests
- 8d52535 Add eval report generator + training table persistence + group-col override
- 33535e5 Streaming fetch+featurize pipeline + 6× pyrodigal speedup + GCA version resolution
- e945ca9 Rewrite BacDive client for v2 public API (no auth required)
- 208a477 Scaffold v0: BacDive + NCBI ingestion, genome feature extractor, XGBoost baseline

## Reminders

- The NCBI API key was pasted into chat earlier in this session — rotate it at ncbi.nlm.nih.gov → Account Settings → API Key Management → Revoke + Create New.
- The `caffeinate -dimsu` you started for the overnight run can be stopped now (`Ctrl+C` in that terminal).
- Display sleep / Battery settings: revert to your normal preferences if you changed them last night.
