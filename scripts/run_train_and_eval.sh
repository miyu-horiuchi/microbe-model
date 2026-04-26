#!/usr/bin/env bash
# Run train + eval + summary. Used as the post-featurize step.
#
# Each phase runs unconditionally; a failure does not abort subsequent phases.
# Per-phase logs go to artifacts/{train,eval,summary}.log so the user can read
# each one individually, and a roll-up timeline goes to artifacts/run.log.
set -uo pipefail
cd "$(dirname "$0")/.."

ts() { date -u +%FT%TZ; }
mkdir -p artifacts
roll_log="artifacts/run.log"
: >"$roll_log"

run_phase() {
    local name="$1"; shift
    local phase_log="artifacts/${name}.log"
    echo "[$(ts)] >>> ${name}" | tee -a "$roll_log"
    if "$@" >"$phase_log" 2>&1; then
        echo "[$(ts)] <<< ${name} OK" | tee -a "$roll_log"
    else
        local rc=$?
        echo "[$(ts)] <<< ${name} FAILED (exit $rc) — see $phase_log" | tee -a "$roll_log"
    fi
}

run_phase "train"   uv run python scripts/03_train_baseline.py
run_phase "eval"    uv run python scripts/04_eval.py
run_phase "summary" uv run python scripts/05_overnight_summary.py

echo "[$(ts)] all phases attempted" | tee -a "$roll_log"
