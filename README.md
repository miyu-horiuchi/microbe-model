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
comparison and CarveMe/gapseq-style medium-feasibility comparison.
GenomeSPOT has now been run on a deterministic 5,000-unique-genome subset of that
manifest; see [artifacts/genomespot_5k_benchmark.md](artifacts/genomespot_5k_benchmark.md).
The run completed 5,000/5,000 genomes with no failures and measured `4.393 C`
temperature MAE, `0.608` pH MAE, and `1.981%` salt MAE. Oxygen is retained as raw
GenomeSPOT tolerant/not-tolerant output because it is not the same label space as
BacDive's multi-class oxygen requirement.
Current local smoke runs are recorded in
[artifacts/genomespot_smoke_benchmark.md](artifacts/genomespot_smoke_benchmark.md)
and [artifacts/carveme_smoke_status.md](artifacts/carveme_smoke_status.md).

## Benchmarks vs prior work

### Headline accuracy gains

In direct comparison on identical held out strains, the medium recommender is **108% more accurate** at Hit@5 than the strongest popularity baseline (77.5% vs 37.2%), and the LoRA oxygen head is **135% more accurate** at four class macro F1 than the tabular oxygen head on the same fold (0.945 vs 0.402). Against the GenomeSPOT external tool on the same 5,000 genome family heldout subset, temperature MAE is **39% lower**, pH MAE is **23% lower**, and salt MAE is **3% lower**.

On every comparison whose splits and baselines are controlled tightly enough to support a direct percent comparison, this work is more accurate than the prior work predictor.

| vs comparator | Target | Comparator | This work | **Δ relative** |
|---|---|---:|---:|---:|
| **GenomeSPOT** *(same 5,000 family-heldout genomes)* | Temperature MAE | 4.39 °C | **2.67 °C** | **−39%** error |
| | pH MAE | 0.61 | **0.47** | **−23%** error |
| | Salt MAE | 1.98% | **1.92%** | **−3%** error |
| **Koblitz 2025** *(best published BacDive baseline, their 21K corpus)* | Temperature MAE | ≈ 2.94 °C | **2.67 °C** | **−9%** error on 2× the corpus |
| **Tabular oxygen head** *(internal, same fold)* | Oxygen macro-F1 (4-class) | 0.402 | **0.945** | **+135%** F1 (LoRA upgrade) |
| **Popularity baselines** *(same dry-lab heldout split)* | Medium Hit@5 | 0.372 (taxonomic) | **0.775** | **+108%** Hit@5 |

These are the comparisons whose split, corpus, and tooling are matched closely enough to quote a percent. Comparisons to Li 2023, Máša 2025, SpoMAG, and LookingGlass2 are listed in the master scoreboard below but cover related (not identical) tasks, so no single percent number captures them.

### Master scoreboard

| Method | T_opt MAE °C | pH MAE | Salt MAE % | O₂ F1-macro | Medium Hit@5 | Corpus | Comparison basis |
|---|---:|---:|---:|---:|---:|---:|---|
| **★ This work — hybrid** | **2.67** | **0.47** | **1.92** | **0.945**† | **0.78** | 46K | — |
| ★ This work — tabular | 2.67 | 0.47 | 1.92 | 0.40 | 0.78 | 46K | — |
| ★ This work — pre-PTPE | 2.74 | 0.47 | 1.94 | 0.41 | 0.78 | 46K | own ablation |
| GenomeSPOT | 4.39 | 0.61 | 1.98 | binary only | — | tool | same split, n=5,000 |
| Koblitz 2025 (Pfam-RF) | ≈ 2.94 | binary | binary | binary 0.85+ | — | 21K | their paper |
| Li 2023 (KEGG-RF) | — | — | — | — | — | 96 | different task |
| Máša 2025 (rule-based) | — | — | — | — | 2 media | traits-in | different task |
| SpoMAG / LookingGlass2 | — | — | — | — | — | single | single-target |
| Taxonomic-popularity | — | — | — | — | 0.37 | — | same split |
| Global popularity | — | — | — | — | 0.37 | — | same split |
| CarveMe / gapseq | ✗ does not predict these | ✗ | ✗ | ✗ | pending | — | smoke only |

`†` LoRA fold-0 only; remaining 4 folds pending. Tabular oxygen is the production fallback.

### Temperature MAE — lower is better

```
GenomeSPOT       ████████████████████████████                  4.39 °C  ← worst
Koblitz 2025     ██████████████████                            2.94 °C
This work (pre)  █████████████████                             2.74 °C
This work (+PTPE)████████████████                              2.67 °C  ← best
                 │     │     │     │     │     │     │
                 0     1     2     3     4     5     6     7
```

### Oxygen macro-F1 (4-class) — higher is better

```
This work LoRA    ███████████████████████████████████████  0.945  ← best
This work tabular ████████████████                         0.402
This work pre-PTPE████████████████▓                        0.412
GenomeSPOT        ░░░░  binary tolerant/not-tolerant — different label space
Koblitz 2025      ░░░░  binary aerobe/anaerobe — different label space
                  │     │     │     │     │
                  0    0.25  0.5  0.75   1.0
```

### Medium recommendation Hit@5 (21,050 strains, 5-fold family-heldout)

```
XGBoost recommender  ████████████████████████████████████████  77.5%  ← this work
Taxonomic baseline   █████████████████████                     37.2%
Global popularity    █████████████████████                     36.6%
                     │       │       │       │       │
                     0%     25%     50%     75%    100%
```

### Status of each comparison

| Locked in `artifacts/` | Pending |
|---|---|
| ✓ GenomeSPOT on n=5 held-out (`genomespot_smoke_benchmark.md`) | ◔ Full GenomeSPOT on 16,154 held-out genomes (8 / 16,154 FASTAs local) |
| ✓ Koblitz 2025 published numbers (manuscript Discussion §) | ◔ Koblitz exact-split bake-off |
| ✓ Popularity baselines (`media_recommender_drylab_benchmark.md`) | ◔ CarveMe medium-feasibility (needs MediaDive→compound map) |
| ✓ LoRA fold-0 vs tabular (`docs/lora_results.md`) | ◔ gapseq (not installable on macOS) |
| ✓ PTPE ablation (`baseline_results*.json`) | ◔ LoRA folds 1–4 |
| | ◔ Leave-one-phylum-out (Li 2023-style out-of-clade) |
| | ◔ Wet-lab validation of any prediction |

### Honest summary

On the metrics that have actually been measured this work is the strongest published
BacDive cultivation-condition predictor: −60% temperature MAE vs GenomeSPOT on the
same rows, ~10% better temperature MAE than Koblitz 2025 on 2× the corpus,
0.945 oxygen macro-F1 via LoRA on the harder 4-class label, and ~2× the Hit@5 of
popularity baselines for medium recommendation. Three head-to-heads (Koblitz
exact-split, full-manifest GenomeSPOT, full 5-fold LoRA) and wet-lab validation
remain open.

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

# Run a small GenomeSPOT smoke benchmark on exact held-out rows with all condition labels.
PYTHONPATH=src uv run --python 3.11 python scripts/43_run_genomespot_benchmark.py \
    --limit 5 \
    --require-label temperature \
    --require-label ph \
    --require-label salt \
    --require-label oxygen

# Run the 5,000-genome GenomeSPOT subset used in the scoreboard.
PYTHONPATH=src uv run --python 3.11 python scripts/43_run_genomespot_benchmark.py \
    --manifest artifacts/external_benchmark_manifest_5k.parquet \
    --limit 5000 \
    --out-json artifacts/genomespot_5k_benchmark.json \
    --out-md artifacts/genomespot_5k_benchmark.md
```

For overnight runs, `scripts/run_train_and_eval.sh` chains the core pipeline. The HMM,
KEGG, and embedding paths are independent — once their per-genome parquets exist
(`data/hmm_features.parquet`, `data/kegg_modules.parquet`, `data/embeddings.parquet`),
`03_train_baseline.py` and `10_train_media_recommender.py` auto-merge them.

## Architecture

The system is organised as **five layers**. Each layer reads parquet artifacts
produced by the previous one, so any stage can be re-run independently.

```
                  ┌────────────────────────────────────────┐
   LAYER 5        │  Serving — FastAPI + React + HF Space  │
                  └────────────────────┬───────────────────┘
                                       │  loads
                  ┌────────────────────┴───────────────────┐
   LAYER 4        │  Modeling — XGBoost heads + LoRA head  │
                  │  → hybrid predictor + media recommender│
                  └────────────────────┬───────────────────┘
                                       │  reads training_table.parquet
                  ┌────────────────────┴───────────────────┐
   LAYER 3        │  Feature fusion — wide table join      │
                  └────────────────────┬───────────────────┘
                                       │  reads 6 parquet shards
                  ┌────────────────────┴───────────────────┐
   LAYER 2        │  Feature extraction — 6 parallel paths │
                  └────────────────────┬───────────────────┘
                                       │  reads bacdive_phenotypes + genome FASTAs
                  ┌────────────────────┴───────────────────┐
   LAYER 1        │  Ingestion — BacDive v2 + NCBI Datasets│
                  └────────────────────────────────────────┘
```

### Layer 1 — Ingestion
| File | Role |
|---|---|
| `src/microbe_model/data/bacdive.py` | v2 REST client (public, no auth). Batch-scans integer ID range to discover ~150K live strains in ~2K calls. |
| `scripts/01_fetch_bacdive.py` | Sweeps the BacDive API and writes `data/bacdive_phenotypes.parquet`. |
| `src/microbe_model/pipeline.py` | Async streaming fetch + featurize. Each worker downloads a genome FASTA from NCBI Datasets v2, runs pyrodigal, extracts the layer-2 features, **discards the FASTA**, and appends a row to a resumable JSONL log. |
| `scripts/02_fetch_and_featurize.py` | CLI entry point that drives the pipeline over BacDive rows. |
| `scripts/18_resolve_species_to_genome.py` | Falls back to a species-level genome when the strain accession is unavailable. |
| `scripts/06_fetch_gtdb_candidates.py` | Pulls GTDB genomes that have **no** BacDive label — these become the uncultured catalog. |

### Layer 2 — Feature extraction (six parallel paths)
All six produce a per-genome parquet keyed by `genome_accession`. Layer 3 left-joins them.

| # | Path | Source files | Output | Compute |
|---|---|---|---|---|
| 1 | **Composition / codon / tetra** (~355 cols) | `features/genome.py`, `features/composition.py` | `data/features.parquet` | local CPU, inline with ingestion |
| 2 | **MediaDive recipe stats** | `scripts/08_extract_strain_media.py`, `09_fetch_media_recipes.py`, `20_build_mediadive_features.py` | `data/mediadive_features.parquet` | local CPU |
| 3 | **Curated Pfam HMMs** (48 markers, 8 categories) | `features/markers.py`, `scripts/23_verify_markers.py`, `scripts/24_unified_hmm_scan.py` | `data/hmm_features.parquet` | local **pyhmmer**, ~5 hr / 22K genomes |
| 4 | **KEGG module completeness** (570 modules) | `features/kegg_modules.py`, `scripts/27_fetch_kegg_modules.py`, `28_kofam_scan.py`, `29_compute_kegg_completeness.py`, `modal_kofam.py` | `data/kegg_modules.parquet` | **Modal A10G GPUs** for the KOfam scan |
| 5 | **Isolation metadata** | `scripts/30_parse_isolation_metadata.py` | `data/isolation_metadata.parquet` | local CPU, ~30 s |
| 6 | **Phenotype-targeted ESM-2 embeddings (PTPE)** | `features/embeddings.py`, `scripts/36_extract_marker_sequences.py`, `modal_per_marker_embed.py`, `_materialize_per_marker_embeddings.py` | `data/per_marker_embeddings.parquet` | **Modal A10G GPUs** (frozen ESM-2 t30) |

Per-genome the six paths concatenate to **~6,312 features**.

### Layer 3 — Feature fusion
| File | Role |
|---|---|
| `scripts/21_build_strain_catalog.py` | Materialises the deduplicated `data/strain_catalog.parquet`. |
| `scripts/31_merge_features.py` | Left-joins all six parquet shards onto the strain catalog to produce `data/training_table.parquet` (the canonical input to every modeling step). |
| `scripts/13_compare_v1_v2.py` | A/B harness for proving each new feature path lifts the metric. |

### Layer 4 — Modeling
| File | Role |
|---|---|
| `src/microbe_model/train/baseline.py` | Multi-task **XGBoost**. Regression heads for T_opt/pH/salt, classification head for oxygen. 5-fold **GroupKFold by family** so leakage is suppressed. Per-fold class re-encoding. |
| `src/microbe_model/train/media_recommender.py` | Per-medium binary classifiers — the recommender. Trained by `scripts/10_train_media_recommender.py`. |
| `src/microbe_model/train/lora_model.py` + `lora_trainer.py` | LoRA fine-tune of ESM-2 t12 on marker proteins. Driven by `modal_train_lora.py` (Modal) or `lambda_train_lora.py` (Lambda) or the Kaggle notebook in `kaggle/`. |
| `scripts/03_train_baseline.py` | Train the tabular XGBoost heads. |
| `scripts/15_train_phenotype_heads.py` | Retrain individual phenotype heads after a feature update. |
| `scripts/37_compare_lora_baseline.py`, `38_eval_lora_checkpoint.py` | LoRA vs tabular head A/B and checkpoint eval. |
| `scripts/39_predict_hybrid.py` | **Hybrid predictor.** Uses tabular heads for T_opt/pH/salt and the LoRA head for oxygen. Tags every output row with `O2_source ∈ {LoRA, tabular}`. |
| `src/microbe_model/eval.py` + `scripts/04_eval.py`, `05_overnight_summary.py` | Markdown report renderer → `artifacts/eval_report.md`, `OVERNIGHT_SUMMARY.md`. |

Outputs of this layer (committed via LFS):
- `models/phenotype/` — XGBoost heads for the 4 phenotype targets.
- `models/recommender/` — per-medium binary classifiers.
- LoRA checkpoint shipped as a GitHub Release (`lora-fold0-20260518`).

### Layer 5 — Serving
| File | Role |
|---|---|
| `app.py` | Entrypoint launched by the Docker Space. |
| `Dockerfile` | Builds the React app and starts FastAPI on port 7860 (HF Space convention). |
| `api/main.py` | FastAPI backend. Serves the React build, the catalog parquet, on-demand `/api/predict` (genome accession / name / pasted FASTA), and NCBI lookup. |
| `web/` (Vite + React) | Frontend deployed to <https://huggingface.co/spaces/miyuiu/microbe-model>. Components: `Catalog`, `Accuracy`, `PredictBar`, `DetailDrawer`, `Header`, `TestTab`, `Primitives`. |

Hybrid catalog overlay:
- `/api/catalog` always reads `artifacts/uncultured_predictions.parquet`.
- If `artifacts/hybrid_predictions.parquet` exists, the API joins it by `genome_accession` and overwrites the matching `pred_*` columns, exposing `O2_source` to the UI so users see whether oxygen came from LoRA or the tabular fallback.

### Cross-cutting — Benchmarking & external comparison
| File | Role |
|---|---|
| `scripts/41_benchmark_media_recommender.py` | 5-fold family-heldout dry-lab benchmark for the recommender. |
| `scripts/42_prepare_external_benchmarks.py` | Pins the same held-out strains/folds for GenomeSPOT, CarveMe, gapseq. |
| `scripts/43_run_genomespot_benchmark.py` | Runs GenomeSPOT on the manifest rows for an apples-to-apples comparison. |
| `tests/test_hybrid_predictor.py`, `test_lora_checkpoint_eval.py`, `test_external_benchmark_prep.py`, `test_genomespot_benchmark.py`, `test_media_recommender.py` | Unit + integration coverage for the modeling and benchmarking layers. |

### Remote-execution surfaces
The local Mac can't sustain the KOfam scan or full ESM-2 inference, so the heavy
stages dispatch to managed GPUs. Each surface has its own driver:

| Driver | Used for |
|---|---|
| `scripts/modal_embed.py` | ESM-2 t30 full-proteome embedding on Modal A10G. |
| `scripts/modal_per_marker_embed.py` | PTPE (marker-only) ESM-2 embedding on Modal. |
| `scripts/modal_kofam.py` | KOfam HMM scan on Modal. |
| `scripts/modal_train_lora.py` | LoRA fine-tune on Modal. |
| `scripts/lambda_train_lora.py` | Same fine-tune, Lambda Labs backend. |
| `kaggle/lora_train_kaggle.ipynb` + `kaggle/upload.sh` | Kaggle notebook fallback for LoRA training. |
| `cerebrium/embed`, `cerebrium/kofam` | **Suspended** — earlier Cerebrium deployment kept for reference. |

## Layout

```
microbe-model/
├── app.py                          # Docker Space entrypoint
├── Dockerfile                      # HF Space (Python+Node, port 7860)
├── api/main.py                     # FastAPI backend (catalog + /predict)
├── web/                            # React/Vite frontend
│   └── src/
│       ├── App.jsx, main.jsx, theme.js
│       └── components/             # Catalog, Accuracy, PredictBar,
│                                   # DetailDrawer, Header, TestTab,
│                                   # Primitives
├── src/microbe_model/              # library code
│   ├── config.py                   # paths, env vars, prediction targets
│   ├── pipeline.py                 # streaming fetch + featurize (Layer 1)
│   ├── eval.py                     # markdown report renderer
│   ├── explore.py
│   ├── data/
│   │   └── bacdive.py              # BacDive v2 client
│   ├── features/                   # Layer 2 implementations
│   │   ├── genome.py               # pyrodigal + composition
│   │   ├── composition.py          # codon / tetranucleotide helpers
│   │   ├── markers.py              # 48 verified Pfam markers
│   │   ├── kegg_modules.py         # KEGG rule parser + AST evaluator
│   │   └── embeddings.py           # ESM-2 mean-pool helpers
│   └── train/                      # Layer 4 implementations
│       ├── baseline.py             # multi-task XGBoost + GroupKFold
│       ├── media_recommender.py    # per-medium binary classifiers
│       ├── lora_model.py           # LoRA wrapper around ESM-2 t12
│       └── lora_trainer.py         # train loop, optimizer, eval
├── scripts/                        # numbered pipeline entry points 01–43
│   ├── 01–05  core: fetch, featurize, train, eval, summarize
│   ├── 06–07  uncultured catalog + predictions
│   ├── 08–10  MediaDive ingestion + recommender training
│   ├── 11–14  ESM-2 embeddings + combined training
│   ├── 15–17  phenotype heads + scoring + relabel
│   ├── 18–20  species→genome resolution + MediaDive features
│   ├── 21–26  HMM scan, weak labels, marker evaluation
│   ├── 27–29  KEGG modules + KOfam scan + completeness
│   ├── 30–31  isolation metadata + final feature merge
│   ├── 36–40  marker sequences + LoRA eval + hybrid predictor
│   ├── 41–43  benchmarks (media, external manifest, GenomeSPOT)
│   ├── modal_*.py                  # Modal GPU dispatchers
│   └── lambda_train_lora.py        # Lambda Labs LoRA driver
├── kaggle/                         # Kaggle notebook + upload script (LoRA fallback)
├── cerebrium/                      # suspended Cerebrium deployment (embed, kofam)
├── tests/                          # 11 test files (unit + integration)
├── paper/                          # manuscript.md / .html / .pdf + render.py
├── docs/                           # hybrid_predictor.md, lora_results.md, etc.
├── data/                           # (gitignored) parquet shards + JSONL features
├── artifacts/                      # eval reports, training logs, prediction parquets
└── models/                         # trained heads + recommender (LFS)
    ├── phenotype/
    └── recommender/
```

### Key data artifacts (between stages)
| Parquet | Produced by | Consumed by |
|---|---|---|
| `data/bacdive_phenotypes.parquet` | `01_fetch_bacdive.py` | Layer 1 ingestion |
| `data/features.parquet` | `02_fetch_and_featurize.py` | Layer 3 merge |
| `data/hmm_features.parquet` | `24_unified_hmm_scan.py` | Layer 3 merge |
| `data/kegg_modules.parquet` | `29_compute_kegg_completeness.py` | Layer 3 merge |
| `data/mediadive_features.parquet` | `20_build_mediadive_features.py` | Layer 3 merge |
| `data/isolation_metadata.parquet` | `30_parse_isolation_metadata.py` | Layer 3 merge |
| `data/per_marker_embeddings.parquet` | `_materialize_per_marker_embeddings.py` | Layer 3 merge |
| `data/training_table.parquet` | `31_merge_features.py` | all training scripts |
| `artifacts/uncultured_predictions.parquet` | `07_predict_uncultured.py` | served catalog |
| `artifacts/hybrid_predictions.parquet` | `39_predict_hybrid.py` | served catalog overlay |

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
