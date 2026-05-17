#!/usr/bin/env bash
# Upload the three Kaggle datasets needed to run the LoRA fine-tune notebook.
#
# Required env:
#   KAGGLE_USERNAME    your Kaggle handle (must match ~/.kaggle/kaggle.json)
#
# Optional env:
#   KAGGLE_VERSION_NOTES  short note saved on each dataset version (default: "auto upload")
#   FORCE_RECREATE        if "1", delete the local staging dirs even if present
#
# This script is idempotent. The first time, it creates each dataset; on
# subsequent runs it pushes a new version of each (so the Kaggle URLs stay stable
# and the notebook keeps working without re-attaching).

set -euo pipefail

# ---------- preflight ----------
if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "error: set KAGGLE_USERNAME (your Kaggle handle) and re-run." >&2
  exit 2
fi

if ! uv run kaggle --version >/dev/null 2>&1; then
  echo "error: kaggle CLI not callable via 'uv run kaggle'. Install with: uv pip install kaggle" >&2
  exit 2
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" && ! -f "${HOME}/.kaggle/access_token" ]]; then
  echo "error: no Kaggle credentials found." >&2
  echo "       Place either ~/.kaggle/kaggle.json or ~/.kaggle/access_token (chmod 600)." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_NOTES="${KAGGLE_VERSION_NOTES:-auto upload}"

upload_dataset() {
  local slug="$1"           # short id, e.g. bacdive-marker-sequences
  local title="$2"          # human-readable title (≤50 chars)
  local license="$3"        # license string, e.g. CC0-1.0
  local stage_dir="$4"      # local directory containing the files
  local has_subdirs="$5"    # "1" if subdirs need zipping; "0" for flat files only
  local kaggle_id="${KAGGLE_USERNAME}/${slug}"
  local dir_mode_flag=""
  if [[ "${has_subdirs}" == "1" ]]; then
    dir_mode_flag="--dir-mode zip"
  fi

  printf '\n=== %s ===\n' "${kaggle_id}"
  cat >"${stage_dir}/dataset-metadata.json" <<JSON
{
  "title": "${title}",
  "id": "${kaggle_id}",
  "licenses": [{"name": "${license}"}]
}
JSON

  # First-time vs. update: try `datasets status`; if it 404s, create; else version.
  # For flat payloads we skip --dir-mode zip so a 1.3 GB local archive isn't built
  # before upload (would blow up a tight disk). For payloads with subdirs we must
  # use --dir-mode zip because the CLI skips folders otherwise.
  if uv run kaggle datasets status "${kaggle_id}" >/dev/null 2>&1; then
    echo "[update] pushing new version of ${kaggle_id}"
    uv run kaggle datasets version -p "${stage_dir}" -m "${VERSION_NOTES}" ${dir_mode_flag}
  else
    echo "[create] uploading new dataset ${kaggle_id}"
    uv run kaggle datasets create -p "${stage_dir}" ${dir_mode_flag}
  fi
}

# ---------- A. marker-sequences (1.3 GB JSONL) ----------
# Use symlinks to avoid duplicating the 1.3 GB file on disk.
A_DIR="/tmp/kaggle-marker-sequences"
rm -rf "${A_DIR}"
mkdir -p "${A_DIR}"
ln -sf "${REPO_ROOT}/data/marker_sequences.jsonl" "${A_DIR}/marker_sequences.jsonl"
upload_dataset bacdive-marker-sequences \
  "BacDive HMM-gated marker sequences" \
  "CC0-1.0" \
  "${A_DIR}" \
  "0"

# ---------- B. bacdive-tables (~50 MB) ----------
B_DIR="/tmp/kaggle-bacdive-tables"
rm -rf "${B_DIR}"
mkdir -p "${B_DIR}"
ln -sf "${REPO_ROOT}/data/bacdive_phenotypes.parquet" "${B_DIR}/bacdive_phenotypes.parquet"
ln -sf "${REPO_ROOT}/data/strain_catalog.parquet" "${B_DIR}/strain_catalog.parquet"
upload_dataset bacdive-tables \
  "BacDive phenotype labels + strain catalog" \
  "CC-BY-4.0" \
  "${B_DIR}" \
  "0"

# ---------- C. microbe-model-code (~120 KB) ----------
C_DIR="${REPO_ROOT}/kaggle/microbe_model_code"
if [[ ! -d "${C_DIR}/microbe_model" ]]; then
  echo "error: expected ${C_DIR}/microbe_model/ to exist (re-run setup)." >&2
  exit 2
fi
upload_dataset microbe-model-code \
  "microbe-model Python package (LoRA trainer)" \
  "MIT" \
  "${C_DIR}" \
  "1"

# ---------- summary ----------
cat <<EOF

All three datasets uploaded under ${KAGGLE_USERNAME}/:
  - ${KAGGLE_USERNAME}/bacdive-marker-sequences
  - ${KAGGLE_USERNAME}/bacdive-tables
  - ${KAGGLE_USERNAME}/microbe-model-code

Next: open https://www.kaggle.com/code/new, upload kaggle/lora_train_kaggle.ipynb,
attach the three datasets above as inputs, set Accelerator=P100 and Internet=on,
and Run All.
EOF
