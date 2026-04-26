# microbe-model

Predict cultivation conditions (optimal temperature, pH, oxygen requirement, salt tolerance) for
microbial isolates from genome sequence alone. The long-term aim is to lower the cost of culturing
"microbial dark matter" — the >99% of microbial diversity that has not yet been grown in pure culture.

## Status

v0 — scaffolding the data pipeline + a non-deep-learning baseline. No trained model yet.

## Approach

```
BacDive (phenotype labels) ──┐
                             ├──> joined table (strain, genome_accession, phenotypes)
GTDB / NCBI (genomes) ───────┘
                                       │
                                       ▼
                              feature extraction
                              (genome statistics, codon usage,
                               proteome-level amino acid stats)
                                       │
                                       ▼
                              XGBoost multi-task baseline
                              (group K-fold by family)
                                       │
                                       ▼
                              eval report (MAE, F1, importances)
```

The genome→phenotype features used here have well-established correlations with the target
properties (e.g. proteome amino acid composition correlates with optimal growth temperature),
so even a tabular model has a real signal to learn from. The point of the v0 is to establish
a ceiling before investing in transformer-based approaches.

## Setup

```bash
# Requires Python 3.11 and uv (https://docs.astral.sh/uv/)
uv sync --all-extras
```

## Running the pipeline

```bash
# 1. Pull strain metadata + phenotype labels from BacDive
#    (BacDive v2 API is public as of Feb 2026 — no registration needed)
uv run python scripts/01_fetch_bacdive.py --end 5000          # smoke test, ~5 min
# uv run python scripts/01_fetch_bacdive.py --end 200000      # full BacDive, ~30 min

# 2. Download genomes for strains that have an accession
uv run python scripts/02_fetch_genomes.py

# 3. Extract genome-level features (CDS prediction + amino acid stats)
uv run python scripts/03_extract_features.py

# 4. Train multi-task XGBoost baseline
uv run python scripts/04_train_baseline.py

# 5. Render eval report
uv run python scripts/05_eval.py
```

## Layout

```
src/microbe_model/
  config.py          # paths, constants
  data/
    bacdive.py       # BacDive REST API client
    ncbi.py          # NCBI genome fetcher (Datasets API)
  features/
    genome.py        # gene prediction + tabular feature extraction
  train/
    baseline.py      # multi-task XGBoost + group K-fold eval
scripts/             # runnable entry points (numbered by pipeline order)
tests/               # smoke tests on small fixtures
data/                # (gitignored) cached API responses, genomes, parquet tables
```

## What this is *not* yet

- Not a foundation model. No transformer. No genome language model.
- Not a platform. There is no upload UI or active-learning loop.
- Not validated against held-out organisms. The eval scaffolding exists; the data does not.

These are deliberate v0 boundaries. See the project notes for the longer-term plan.

## Environment variables

Copy `.env.example` to `.env` and fill in:

- `NCBI_API_KEY` — optional, raises NCBI rate limit from 3 req/s to 10 req/s.

(BacDive's v2 API was opened to the public in February 2026 — no registration or token needed.)
