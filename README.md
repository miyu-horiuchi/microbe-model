---
title: microbe-model
emoji: 🦠
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Predict growth conditions and culture media for never-cultured microbes
---

# microbe-model

Predict cultivation conditions (optimal temperature, pH, oxygen requirement, salt tolerance)
for microbial isolates from genome sequence alone. The long-term aim is to lower the cost of
culturing "microbial dark matter" — the >99% of microbial diversity that has not yet been grown
in pure culture.

## Status

v0 — tabular baseline against the full BacDive corpus. No deep model yet; the point of the
v0 is to establish the ceiling on tabular performance before investing in transformers.

## Approach

```
BacDive v2 (phenotype labels) ──┐
                                ├──> joined table (strain, genome_accession, phenotypes)
NCBI Datasets v2 (genomes) ─────┘
                                          │
                                          ▼
                                 streaming featurize
                                 (download → pyrodigal → AA-composition → discard FASTA)
                                          │
                                          ▼
                                 multi-task XGBoost
                                 (5-fold GroupKFold by taxonomic family)
                                          │
                                          ▼
                                 eval report (`artifacts/eval_report.md`)
```

The genome→phenotype features used here have well-established correlations with the target
properties (proteome amino acid composition correlates with optimal growth temperature, GC
content with thermophily, etc.), so even a tabular model has a real signal to learn from.

## Setup

```bash
# Requires Python 3.11 and uv (https://docs.astral.sh/uv/)
uv sync --all-extras

# (optional) NCBI API key raises rate limits; copy .env.example to .env and fill in.
cp .env.example .env
```

## Running the pipeline

```bash
# 1. Scan BacDive (~10 min for full corpus). Writes data/bacdive_phenotypes.parquet.
uv run python scripts/01_fetch_bacdive.py --end 200000

# 2. Streaming fetch + featurize all training-ready strains (~5 hr for ~17K strains
#    with 7 worker processes). Writes data/features.parquet. Resumable.
uv run python scripts/02_fetch_and_featurize.py --require-target any

# 3. Multi-task XGBoost. Writes artifacts/baseline_results.json + predictions.parquet.
uv run python scripts/03_train_baseline.py

# 4. Render the eval report. Writes artifacts/eval_report.md.
uv run python scripts/04_eval.py

# 5. Top-level overnight summary. Writes OVERNIGHT_SUMMARY.md.
uv run python scripts/05_overnight_summary.py
```

For overnight runs, `scripts/run_train_and_eval.sh` chains 03 → 04 → 05.

## Architecture

- **`src/microbe_model/data/bacdive.py`** — v2 REST client (public, no auth). Discovers
  strains by batch-scanning the integer ID range; ~150K live records in 2K calls.
- **`src/microbe_model/pipeline.py`** — streaming fetch + featurize. Each worker process
  downloads a genome FASTA, runs pyrodigal, extracts features, and discards the FASTA —
  no persistent genome storage. Resumable via the JSONL append log.
- **`src/microbe_model/features/genome.py`** — pyrodigal CDS prediction + amino-acid
  composition features. Uses single-genome+train mode (~7× faster than meta on assembled
  genomes) with a meta fallback for very short input.
- **`src/microbe_model/train/baseline.py`** — multi-task XGBoost with per-fold class
  re-encoding for classification (handles non-contiguous class subsets).
- **`src/microbe_model/eval.py`** — markdown report renderer. TL;DR + corpus + per-target
  metrics + per-family error breakdown + limitations + next steps.

## Layout

```
src/microbe_model/
  config.py          # paths, env vars, prediction targets
  data/bacdive.py    # BacDive v2 client
  features/genome.py # pyrodigal + AA-composition feature extraction
  pipeline.py        # streaming async fetch + featurize
  train/baseline.py  # multi-task XGBoost + GroupKFold
  eval.py            # markdown report renderer
scripts/             # numbered pipeline entry points (01–05)
tests/               # 15 unit + integration tests
data/                # (gitignored) parquet tables, JSONL features, BacDive cache
artifacts/           # eval report, training results, logs
```

## What this is *not* yet

- Not a foundation model. No transformer. No genome language model.
- Not a platform. There is no upload UI or active-learning loop.
- Not validated against truly out-of-distribution organisms (BacDive is survivorship-biased
  to organisms that have been cultured at least once).

These are deliberate v0 boundaries — see `OVERNIGHT_SUMMARY.md` after a run for the
headline result and `artifacts/eval_report.md` for the full eval.

## v1 backlog

1. Tetranucleotide and codon-usage features
2. LPSN/GTDB family proper join (for tighter GroupKFold)
3. KOMODO media DB integration as a richer label source
4. Pyrodigal-GV for atypical genetic codes
5. Genome-LM embeddings (Nucleotide Transformer / Evo-1 / DNABERT-2)
6. Active learning loop: highlight novel-family strains where the model is uncertain,
   prioritize for wet-lab cultivation testing.

## Environment variables

`.env` (gitignored) holds:

- `NCBI_API_KEY` — optional, raises NCBI Datasets rate limit from 3 req/s to 10 req/s.

(BacDive's v2 API is public — no token needed.)
