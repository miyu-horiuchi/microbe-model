# Kaggle migration — LoRA fine-tune of ESM-2 on BacDive phenotypes

This directory packages everything you need to run the LoRA fine-tune on Kaggle's
free P100 GPU (30 hr/week quota). Costs $0; one epoch fits in a single 12-h session.

## One-time setup

1. **Make a Kaggle account** and verify it via phone (required to enable internet +
   GPU). https://www.kaggle.com/account
2. **Install the Kaggle CLI** locally so you can upload datasets without dragging
   1.3 GB through the browser:

   ```bash
   uv pip install kaggle
   ```

3. **Download your Kaggle API token** from https://www.kaggle.com/settings → "Create
   new API token". Save it to `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`.

## Upload the three datasets

Run the helper, which packages and uploads all three at once (and re-runs as
"new version" pushes on subsequent invocations so the Kaggle URLs stay stable):

```bash
export KAGGLE_USERNAME=<your-kaggle-handle>
bash kaggle/upload.sh
```

This creates (or updates) three datasets under your account:

| Slug | Size | License | Contents |
|---|---|---|---|
| `bacdive-marker-sequences` | 1.3 GB | CC0-1.0 | `marker_sequences.jsonl` |
| `bacdive-tables` | ~50 MB | CC-BY-4.0 | `bacdive_phenotypes.parquet`, `strain_catalog.parquet` |
| `microbe-model-code` | ~120 KB | MIT | The `microbe_model/` Python package |

If you ever need to wipe + re-stage the local copies, run with `FORCE_RECREATE=1 bash kaggle/upload.sh`.

## Running on Kaggle

1. Open https://www.kaggle.com → "Create" → "New Notebook".
2. Upload `kaggle/lora_train_kaggle.ipynb` (or paste the contents).
3. **Settings (right rail):**
   - Accelerator → **GPU P100**
   - Persistence → **Files only** (so checkpoints survive between sessions)
   - Internet → **on** (needed to fetch ESM-2 weights from HuggingFace)
4. **Add inputs (right rail):**
   - `<YOUR-KAGGLE-USERNAME>/bacdive-marker-sequences`
   - `<YOUR-KAGGLE-USERNAME>/bacdive-tables`
   - `<YOUR-KAGGLE-USERNAME>/microbe-model-code`
5. Adjust the input paths in cell 2 of the notebook to match the names of the
   datasets you uploaded (Kaggle slugifies them so the folder name under
   `/kaggle/input/` will be e.g. `bacdive-marker-sequences/`).
6. **Run all cells.** Training will print loss every 50 steps; expect ~10 h for
   one epoch of fold 0 on P100. The trainer saves `fold0_best.pt` and
   `fold0_results.json` to `/kaggle/working/`, both downloadable from the Output
   tab when the session ends.

## Resuming across sessions (only if you want >1 epoch)

If you need 3 epochs total, your simplest path is three separate Kaggle sessions,
each running 1 epoch starting from the previous session's checkpoint:

1. After session 1 finishes, **download** `/kaggle/working/fold0_best.pt`.
2. **Create a new Kaggle Dataset** called `lora-fold0-ckpt-epoch1` containing it.
3. In session 2, add this dataset as input and load the checkpoint state-dict
   into the model before training begins. Add a cell like:

   ```python
   import torch
   ckpt = torch.load(\"/kaggle/input/lora-fold0-ckpt-epoch1/fold0_best.pt\")
   model.load_state_dict(ckpt[\"state_dict\"], strict=False)
   ```
4. Repeat for session 3.

For a "publishable, modest-cost LoRA result", running 1 epoch in 1 session is
usually enough — LoRA reaches most of its gain in the first pass through the data.

## What to do with the result

When the Kaggle run finishes, download `fold0_results.json` from
`/kaggle/working/` and drop it at `artifacts/lora/fold0_results.json` locally.
A follow-up script will compare the LoRA per-target metrics to the
frozen-PTPE XGBoost baseline in `artifacts/baseline_results.json`.
