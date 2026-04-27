# Running ESM-2 embedding extraction on Lightning AI

Embedding 17K BacDive genomes is too slow on the M2 Mac (~80 hours for sampled
extraction with the smallest model). Run it once on a Lightning AI Studio (free
T4 tier covers it) and pull the resulting parquet back to the Mac for training.

## Setup (one-time, ~5 min)

1. Sign in at https://lightning.ai/, create a new Studio.
2. Pick **T4 (16 GB)** for the GPU. Free tier gives ~80 hr/month.
3. In the Studio terminal:
   ```bash
   git clone https://github.com/miyu-horiuchi/microbe-model.git
   cd microbe-model
   pip install uv
   uv sync --all-extras
   ```
4. Add `.env` with the NCBI API key (same one you have locally):
   ```bash
   echo 'NCBI_API_KEY=...' > .env
   ```

## Pull the source data (~10 min)

The phenotype labels need to come from somewhere. Two options:

**Option A — re-run the BacDive scan there:**
```bash
uv run python scripts/01_fetch_bacdive.py --end 200000
```
~10 min, no network input needed beyond the API.

**Option B — copy `data/bacdive_phenotypes.parquet` from the Mac:**
```bash
# On the Mac, get the path
ls -la data/bacdive_phenotypes.parquet
# Upload via Lightning's drag-and-drop UI, or scp / rsync
```
Same result, faster.

## Run the extraction (~3-5 hr on T4)

```bash
uv run python scripts/11_extract_embeddings.py \
    --model facebook/esm2_t30_150M_UR50D \
    --sample-n 50 \
    --batch-size 32
```

Resumable — if the studio times out, just re-run and it picks up from where it
left off (writes per-genome rows to `data/embeddings.jsonl` as it goes).

For a faster smoke test:
```bash
uv run python scripts/11_extract_embeddings.py \
    --model facebook/esm2_t6_8M_UR50D --sample-n 20 --max 200
```

## Pull results back to the Mac

```bash
# Lightning Studio download:
# Right-click data/embeddings.parquet -> Download
```

Or push to a GitHub release / HuggingFace dataset for portability.

## Train + compare locally on the Mac (no GPU needed)

```bash
uv run python scripts/12_train_with_embeddings.py
```

This trains XGBoost on the embedding columns and writes
`artifacts/embedding_results.json`. Compare to v1 baseline:

| Target | v1 (353 hand-crafted features) | v2 (640-dim ESM-2 embeddings) |
|--------|-------------------------------|------------------------------|
| T_opt | MAE ?? | MAE ?? |
| pH | MAE ?? | MAE ?? |
| Oxygen | F1 ?? | F1 ?? |
| Salt | MAE ?? | MAE ?? |

If embeddings give a ≥10% lift on T_opt or fix pH/salt, this validates the
"genome LM" direction and we proceed to a heavier model (ESM-2 t33_650M, or
Nucleotide Transformer for DNA-native).

## Cost estimate

- T4 (free): ~3-5 hr per full pass with `t30_150M`, sample_n=50, batch_size=32
- A100 (paid, ~$1.30/hr): ~30 min same workload
- Storage: embeddings.parquet ~50 MB at 640-dim × 17K genomes
