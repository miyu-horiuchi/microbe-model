---
title: microbe-model
emoji: 🦠
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict growth media for never-cultured microbes
---

# microbe-model

Predict cultivation conditions (optimal temperature, pH, oxygen requirement, salt tolerance)
for microbial isolates from genome sequence alone. The long-term aim is to lower the cost of
culturing "microbial dark matter" — the >99% of microbial diversity that has not yet been grown
in pure culture.

## Status

v3 — multi-source feature fusion. The corpus is 22,301 unique genomes from BacDive,
each described by **five parallel feature paths** that XGBoost then combines:

1. **Composition / codon / tetranucleotide statistics** (~355 cols, v0)
2. **MediaDive recipe metadata** (medium pH, NaCl content of media this strain grows on)
3. **Curated Pfam HMM markers** (48 verified, 8 categories: T_opt, pH, oxygen, salt,
   vitamins, nitrogen, carbon, special)
4. **KEGG module completeness** — fractional 0–1 score for ~570 metabolic pathways via
   KOfam + KEGG module rules (in progress)
5. **Isolation metadata** — country / continent / lat-lon / collection year / inferred
   host kingdom from raw BacDive JSONs
6. **ESM-2 protein-language-model embeddings** (t6 320-dim done; t30 640-dim partial,
   running on Modal A10G)

Latest A/B lift over the composition-only baseline:
- **Salt MAE: 2.11 → 1.99 (-5.6%)**
- **Oxygen F1: 0.357 → 0.379 (+5.9%)**
- T_opt and pH neutral so far (the broad Pfam signals saturate; KEGG modules expected to
  break the ceiling)

## Approach

```
BacDive v2 (phenotype labels) ──┐
                                ├──> joined table (strain, genome_accession, phenotypes)
NCBI Datasets v2 (genomes) ─────┘
                                          │
                                          ▼
                                 streaming featurize
                                 (download → pyrodigal → discard FASTA)
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼              ▼              ▼              ▼              ▼
   composition /   Pfam HMM scan  KEGG/KOfam scan   ESM-2 mean-pool   isolation
   codon / tetra   (48 markers)   (570 modules)     (t30 on Modal)    metadata
            └─────────────────────────────┼─────────────────────────────┘
                                          ▼
                                 multi-task XGBoost
                                 (5-fold GroupKFold by family)
                                          │
                                          ▼
                                 phenotype heads + medium recommender
```

The five feature paths are **independent**: each describes the same genome a different
way, and XGBoost decides which to weight per phenotype. The marker-importance diagnostic
shows oxygen leans hard on Pfam HMMs (COX1, hydrogenases), T_opt leans on composition
(IVYWREL fraction), salt uses both, and KEGG modules are expected to dominate the
medium-recommendation side because every "missing biosynthetic pathway" maps directly
to "this compound goes in the recipe."

## Setup

```bash
# Requires Python 3.11 and uv (https://docs.astral.sh/uv/)
uv sync --all-extras

# (optional) NCBI API key raises rate limits; copy .env.example to .env and fill in.
cp .env.example .env
```

## Running the pipeline

```bash
# === core pipeline (composition + tabular features) ===
uv run python scripts/01_fetch_bacdive.py --end 200000        # ~10 min
uv run python scripts/02_fetch_and_featurize.py --require-target any  # ~5 hr, resumable
uv run python scripts/03_train_baseline.py                    # multi-task XGBoost
uv run python scripts/04_eval.py                              # eval report
uv run python scripts/05_overnight_summary.py                 # OVERNIGHT_SUMMARY.md

# === Pfam HMM markers (curated, ~5 hr scan once) ===
uv run python scripts/23_verify_markers.py                    # validate Pfam IDs
uv run python scripts/24_unified_hmm_scan.py --workers 8      # scan 22K genomes
uv run python scripts/25_evaluate_all_targets.py              # A/B lift report
uv run python scripts/26_marker_importance.py                 # which markers paid off

# === KEGG module completeness (~570 modules) ===
uv run python scripts/27_fetch_kegg_modules.py                # ~1 min, REST API
uv run python scripts/28_kofam_scan.py --fetch-only           # download KOfam, ~10 min
uv run python scripts/28_kofam_scan.py --workers 8            # full scan, ~14-18 hr
uv run python scripts/29_compute_kegg_completeness.py         # ~1 min, materialize parquet

# === Environment-of-origin enrichment (no compute) ===
uv run python scripts/30_parse_isolation_metadata.py          # ~30 sec; lat/lon/host/etc

# === ESM-2 protein embeddings ===
# Local on MPS (slow):
uv run --extra embeddings python scripts/11_extract_embeddings.py \
    --model facebook/esm2_t30_150M_UR50D --sample-n 50

# Or on Modal A10G GPUs (much faster, requires `modal setup` + ncbi-key secret):
modal run scripts/modal_embed.py
uv run python scripts/_materialize_embeddings.py             # JSONL → parquet
```

For overnight runs, `scripts/run_train_and_eval.sh` chains the core pipeline. The HMM,
KEGG, and embedding paths are independent — once their per-genome parquets exist
(`data/hmm_features.parquet`, `data/kegg_modules.parquet`, `data/embeddings.parquet`),
`03_train_baseline.py` and `10_train_media_recommender.py` auto-merge them.

## Architecture

### Core
- **`src/microbe_model/data/bacdive.py`** — v2 REST client (public, no auth). Discovers
  strains by batch-scanning the integer ID range; ~150K live records in 2K calls.
- **`src/microbe_model/pipeline.py`** — streaming fetch + featurize. Each worker process
  downloads a genome FASTA, runs pyrodigal, extracts features, and discards the FASTA —
  no persistent genome storage. Resumable via the JSONL append log.
- **`src/microbe_model/features/genome.py`** — pyrodigal CDS prediction + amino-acid
  composition / codon / tetranucleotide features.
- **`src/microbe_model/train/baseline.py`** — multi-task XGBoost with per-fold class
  re-encoding for classification.
- **`src/microbe_model/eval.py`** — markdown report renderer.

### Feature paths
- **`src/microbe_model/features/markers.py`** — 48 verified Pfam markers across 8 categories
  (T_opt, pH, oxygen, salt, vitamins, nitrogen, carbon, special). All IDs validated via
  `scripts/23_verify_markers.py` against InterPro DESC fields.
- **`src/microbe_model/features/kegg_modules.py`** — KEGG module rule parser (boolean
  AND / OR / parens grammar) + AST evaluator for fractional & strict completeness scoring.
- **`src/microbe_model/features/embeddings.py`** — frozen ESM-2 forward pass + mean-pool
  per protein → per-proteome 320/640-dim vector (model-size dependent).

### Scanners (numbered scripts)
- **`24_unified_hmm_scan.py`** — pyhmmer scan over the 48-marker Pfam library, dedup'd
  by genome accession, streams to `data/hmm_features.parquet`.
- **`28_kofam_scan.py`** — same architecture but against KOfam (~3K KEGG-relevant HMMs);
  output is per-genome KO sets.
- **`29_compute_kegg_completeness.py`** — applies the KEGG module rules to KO hits,
  yields ~570 fractional-completeness columns per genome.
- **`30_parse_isolation_metadata.py`** — parses raw BacDive JSONs for lat/lon/country/
  host species; outputs `data/isolation_metadata.parquet` with one-hot encodings.
- **`modal_embed.py`** — Modal app for ESM-2 t30 (or t33) extraction on A10G GPUs.

## Layout

```
src/microbe_model/
  config.py            # paths, env vars, prediction targets
  data/bacdive.py      # BacDive v2 client
  features/
    genome.py          # pyrodigal + composition / codon / tetra
    composition.py     # tetranucleotide + codon-usage helpers
    markers.py         # 48 verified Pfam markers (8 categories)
    kegg_modules.py    # KEGG module rule parser + AST evaluator
    embeddings.py      # ESM-2 mean-pool helpers
  pipeline.py          # streaming async fetch + featurize
  train/
    baseline.py        # multi-task XGBoost + GroupKFold
    media_recommender.py  # per-medium binary classifiers
  eval.py              # markdown report renderer
scripts/               # numbered pipeline entry points (01–30 + modal_*.py)
tests/                 # unit + integration tests
data/                  # (gitignored) parquet tables, JSONL features, BacDive cache
artifacts/             # eval report, training results, logs
models/                # trained phenotype heads + per-medium recommender models (LFS)
```

## What this is *not* yet

- Not a foundation model. No transformer. No genome language model.
- Not a platform. There is no upload UI or active-learning loop.
- Not validated against truly out-of-distribution organisms (BacDive is survivorship-biased
  to organisms that have been cultured at least once).

These are deliberate v0 boundaries — see `OVERNIGHT_SUMMARY.md` after a run for the
headline result and `artifacts/eval_report.md` for the full eval.

## v1 backlog (partially shipped — see Status)

✅ Done:
- Tetranucleotide and codon-usage features (v0.1)
- MediaDive recipe metadata as a richer label source (v0.2)
- ESM-2 t6 mean-pooled embeddings (v2)
- 48 verified Pfam markers across 8 categories (v3)
- KEGG module completeness pipeline (v3, scan in progress)
- Isolation metadata enrichment from raw BacDive JSON (v3)
- Modal-based GPU embedding extraction (t30 in flight)

🔬 Open:
- LPSN/GTDB family proper join (for tighter GroupKFold)
- Pyrodigal-GV for atypical genetic codes
- Per-marker pooling for ESM-2 (embed only HMMER-hit proteins, average within family —
  unlocks more of ESM-2's signal than mean-of-means does)
- Co-occurrence / cross-feeding context from public metagenomes (MGnify, EMP, HMP)
- Active learning loop: highlight novel-family strains where the model is uncertain,
  prioritize for wet-lab cultivation testing.

## Environment variables

`.env` (gitignored) holds:

- `NCBI_API_KEY` — optional, raises NCBI Datasets rate limit from 3 req/s to 10 req/s.

(BacDive's v2 API is public — no token needed.)
