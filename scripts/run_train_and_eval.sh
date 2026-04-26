#!/usr/bin/env bash
# Run train + eval + write the eval report. Used as the post-featurize step.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[$(date -u +%FT%TZ)] Starting train..."
uv run python scripts/03_train_baseline.py 2>&1 | tee artifacts/train.log

echo "[$(date -u +%FT%TZ)] Starting eval report..."
uv run python scripts/04_eval.py 2>&1 | tee artifacts/eval.log

echo "[$(date -u +%FT%TZ)] Writing overnight summary..."
uv run python scripts/05_overnight_summary.py 2>&1 | tee -a artifacts/eval.log

echo "[$(date -u +%FT%TZ)] Done."
