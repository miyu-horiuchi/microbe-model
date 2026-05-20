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

v5 — hybrid predictor on top of the v4 multi-source feature stack.
46,029 BacDive strains over 22,300 unique genomes, each described by **six parallel
feature paths** that XGBoost then combines (6,312 features total per genome):

1. **Composition / codon / tetranucleotide statistics** (~355 cols, v0)
2. **MediaDive recipe metadata** (medium pH, NaCl content of media this strain grows on)
3. **Curated Pfam HMM markers** (144 cols, 8 categories: T_opt, pH, oxygen, salt,
   vitamins, nitrogen, carbon, special)
4. **KEGG module completeness** — fractional 0–1 score for 570 metabolic pathways via
   completed KOfam scan + KEGG module rules
5. **Isolation metadata** — country / continent / lat-lon / collection year / inferred
   host kingdom from raw BacDive JSONs (46 cols + 65 one-hot category encodings)
6. **Phenotype-targeted ESM-2 embeddings (PTPE)** — for each genome, embed only the
   proteins matching curated phenotype-relevant HMMs (cytochromes for oxygen, heat-shock
   for temperature, Na⁺/H⁺ antiporters for pH/salt, etc.) with frozen ESM-2 t30, then
   mean-pool per category. 8 markers × 641 dims = 5,128 cols.

Current tabular 5-fold GroupKFold CV (full feature stack):

| Target | Metric | v3 (pre-PTPE) | v4 (+PTPE) | Δ |
|---|---|---|---|---|
| optimal_temperature_c | MAE °C | 2.74 | **2.67** | −2.4% |
| optimal_ph | MAE | 0.47 | **0.47** | −1.0% |
| oxygen_requirement | F1-macro | 0.41 | **0.40** | −2.4% (slight regression) |
| salt_tolerance_pct | MAE % | 1.94 | **1.92** | −1.1% |

PTPE adds modest, mixed lift on the regressors and slightly hurts oxygen F1. Frozen
mean-pool may not be unlocking the PLM signal. A first fold-0 LoRA fine-tune of
ESM-2 t12 is now complete and is strongest for oxygen classification: the best
all-task LoRA checkpoint reaches `0.9448` oxygen macro F1 on fold 0, versus `0.4020`
for the current tabular five-fold mean. Oxygen-only and anaerobe-weighted variants
did not beat the original all-task checkpoint. See [docs/lora_results.md](docs/lora_results.md) for the
checkpoint release, metrics, and load instructions.
For practical prediction, use the hybrid predictor in [docs/hybrid_predictor.md](docs/hybrid_predictor.md):
tabular XGBoost heads for temperature/pH/salt plus LoRA for oxygen. The deployed UI
surfaces whether each oxygen value came from `LoRA` or the `tabular` fallback.

Media recommendation now has a dry-lab held-out benchmark in
[artifacts/media_recommender_drylab_benchmark.md](artifacts/media_recommender_drylab_benchmark.md).
On 5-fold family-heldout MediaDive links, the XGBoost recommender recovers at least
one known medium in the top 5 for `77.5%` of evaluable strains, compared with `36.6%`
for global medium popularity and `37.2%` for the taxonomic-popularity baseline.
Median per-medium ROC-AUC is `0.910`; median PR-AUC is `0.183`, reflecting sparse,
imbalanced medium labels.

External-tool benchmarking is prepared in
[artifacts/external_benchmark_status.md](artifacts/external_benchmark_status.md).
The manifest pins the same family-heldout strains for GenomeSPOT condition-trait
comparison and CarveMe/gapseq-style medium-feasibility comparison, but the local
machine still needs the full held-out genome FASTA set and the external tool
binaries/databases before those baselines can be run.

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
                      phenotype heads + medium recommender + LoRA oxygen head
```

The six feature paths are **independent**: each describes the same genome a different
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

# === Phenotype-targeted ESM-2 embeddings (PTPE) ===
# Embed only proteins matching curated phenotype HMMs, pool per category.
uv run modal run scripts/modal_per_marker_embed.py --model facebook/esm2_t30_150M_UR50D
uv run python scripts/_materialize_per_marker_embeddings.py  # JSONL → parquet

# === Hybrid predictor ===
# Tabular XGBoost for temperature/pH/salt, fold-0 LoRA for oxygen.
PYTHONPATH=src uv run --python 3.11 --extra dev --extra embeddings python scripts/39_predict_hybrid.py \
    --features data/training_table.parquet \
    --marker-sequences data/marker_sequences.jsonl \
    --device mps \
    --output artifacts/hybrid_predictions.parquet

# Chunked uncultured-catalog run; keeps tabular values and replaces oxygen with LoRA.
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

# === External benchmark manifest ===
# Pins the same held-out strains/folds for GenomeSPOT, CarveMe, and gapseq runs.
PYTHONPATH=src uv run --python 3.11 python scripts/42_prepare_external_benchmarks.py

# Optional smoke download of 10 missing genome FASTAs for external-tool setup checks.
PYTHONPATH=src uv run --python 3.11 python scripts/42_prepare_external_benchmarks.py \
    --download-fastas 10
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

### UI and API
- **`api/main.py`** — FastAPI backend for the Hugging Face Space. It serves the React
  build, recommender models, catalog API, NCBI lookup, and on-demand genome prediction.
- **`web/`** — React/Vite frontend used by the Docker Space at
  <https://huggingface.co/spaces/miyuiu/microbe-model>.
- **Hybrid catalog behavior** — `/api/catalog` always loads
  `artifacts/uncultured_predictions.parquet`; if `artifacts/hybrid_predictions.parquet`
  exists, the API overlays matching `pred_*` columns by `genome_accession`.
  Oxygen rows include `O2_source` so the UI can show `LoRA` vs `tabular`.
- **Live `/api/predict` behavior** — on-demand predictions currently use the deployed
  tabular phenotype heads and return per-phenotype `source` metadata. LoRA-backed
  oxygen is used for precomputed hybrid catalog rows when the hybrid artifact is present.

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
scripts/               # numbered pipeline entry points (01–42 + modal_*.py)
api/                   # FastAPI backend for the Docker/Hugging Face Space
web/                   # React/Vite frontend for the deployed UI
tests/                 # unit + integration tests
data/                  # (gitignored) parquet tables, JSONL features, BacDive cache
artifacts/             # eval report, training results, logs
models/                # trained phenotype heads + per-medium recommender models (LFS)
```

## What this is *not* yet

- Not an end-to-end foundation model. LoRA only fine-tunes an ESM-2 marker-protein
  encoder for phenotype prediction; the system is still mostly tabular XGBoost plus
  a targeted oxygen LoRA head.
- Not a full active-learning platform. The UI can score an accession/name/FASTA, but it
  does not yet store experiments, close the lab feedback loop, or retrain from new assays.
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
- KEGG module completeness pipeline — full KOfam scan complete (v4)
- Isolation metadata enrichment from raw BacDive JSON (v3)
- Modal-based GPU embedding extraction (t30 complete, 22,300 genomes)
- **Phenotype-targeted ESM-2 embeddings (PTPE)** — HMM-gated mean-pool per category (v4)
- **Fold-0 LoRA fine-tune of ESM-2 t12** — best result is the all-task checkpoint,
  stored in the `lora-fold0-20260518` GitHub Release

🔬 Open:
- **Run LoRA across folds 1-4** if publication-grade validation is needed; fold 0
  is promising for oxygen, but it is still only one group fold
- **Run external baselines** on the prepared held-out manifest once FASTAs and
  third-party databases are local: GenomeSPOT for condition traits, and
  CarveMe/gapseq-style metabolic reconstructions for medium feasibility.
- **Attention-pooled** per-category genome encoder instead of mean-pool (most novel
  methodological direction)
- LPSN/GTDB family proper join (for tighter GroupKFold)
- Pyrodigal-GV for atypical genetic codes
- Co-occurrence / cross-feeding context from public metagenomes (MGnify, EMP, HMP)
- Active learning loop: highlight novel-family strains where the model is uncertain,
  prioritize for wet-lab cultivation testing.

## Environment variables

`.env` (gitignored) holds:

- `NCBI_API_KEY` — optional, raises NCBI Datasets rate limit from 3 req/s to 10 req/s.

(BacDive's v2 API is public — no token needed.)
