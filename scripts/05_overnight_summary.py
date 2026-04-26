"""Render a one-page OVERNIGHT_SUMMARY.md the user reads first.

Pulls headline numbers from the eval report + featurize log + git history,
and renders a single short page that orients the reader cold.
"""
from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microbe_model import config

ROOT = config.ROOT


def _read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as fh:
        return sum(1 for _ in fh)


def _read_featurize_log_summary(path: Path) -> dict[str, str | int]:
    """Parse the last tqdm line for success/fail counts."""
    if not path.exists():
        return {}
    with open(path) as fh:
        text = fh.read().replace("\r", "\n")
    last_match = None
    pattern = re.compile(r"(\d+)/(\d+).*?fail=(\d+).*?success=(\d+)")
    for m in pattern.finditer(text):
        last_match = m
    if last_match is None:
        # Fall back to alternate ordering (success first then fail)
        alt = re.compile(r"(\d+)/(\d+).*?success=(\d+).*?fail=(\d+)")
        for m in alt.finditer(text):
            last_match = m
        if last_match is None:
            return {}
        completed, total, success, fail = (int(last_match.group(i)) for i in range(1, 5))
    else:
        completed, total, fail, success = (int(last_match.group(i)) for i in range(1, 5))
    return {
        "completed": completed,
        "total": total,
        "success": success,
        "fail": fail,
    }


def _git_log_since_yesterday() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "--oneline", "--since=yesterday"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> None:
    pheno_path = config.DATA / "bacdive_phenotypes.parquet"
    features_jsonl = config.DATA / "features.jsonl"
    featurize_log = config.ARTIFACTS / "featurize.log"
    eval_report = config.ARTIFACTS / "eval_report.md"
    blockers = config.ARTIFACTS / "blockers.md"
    train_log = config.ARTIFACTS / "train.log"

    pheno = pd.read_parquet(pheno_path) if pheno_path.exists() else None
    feats_n = _read_jsonl_count(features_jsonl)
    summary = _read_featurize_log_summary(featurize_log)

    lines: list[str] = []
    lines.append("# Overnight run — summary")
    lines.append("")
    lines.append(f"_Written {datetime.now(UTC).isoformat(timespec='minutes')}_")
    lines.append("")

    # Pipeline status
    lines.append("## Pipeline status")
    lines.append("")
    if pheno is not None:
        lines.append(f"- ✅ BacDive scan: **{len(pheno):,}** strains pulled")
        n_genome = pheno["genome_accession"].notna().sum()
        n_temp = pheno["optimal_temperature_c"].notna().sum()
        n_train_ready = (
            pheno["genome_accession"].notna() & pheno["optimal_temperature_c"].notna()
        ).sum()
        lines.append(f"  - {n_genome:,} have genome accessions")
        lines.append(f"  - {n_temp:,} have optimal_temperature_c labels")
        lines.append(f"  - **{n_train_ready:,}** strains are training-ready (genome + T_opt)")
    else:
        lines.append("- ❌ BacDive scan did not complete (data/bacdive_phenotypes.parquet missing)")

    if summary:
        pct = 100 * summary["completed"] / max(1, summary["total"])
        success_rate = 100 * summary["success"] / max(1, summary["completed"])
        if summary["completed"] >= summary["total"]:
            status = "✅"
            descriptor = "complete"
        else:
            status = "🟡"
            descriptor = f"in progress ({pct:.0f}%)"
        lines.append(f"- {status} Featurize: {descriptor}")
        lines.append(f"  - Processed: {summary['completed']:,} / {summary['total']:,}")
        lines.append(f"  - Successful: {summary['success']:,} ({success_rate:.1f}%)")
        lines.append(f"  - Failed: {summary['fail']:,} (mostly suppressed/withdrawn NCBI assemblies)")
    elif feats_n:
        lines.append(f"- 🟡 Featurize: {feats_n:,} feature rows (log not parseable)")
    else:
        lines.append("- ❌ Featurize did not run")

    if train_log.exists():
        lines.append("- ✅ Training: see `artifacts/train.log` for stdout")
    else:
        lines.append("- ⏭ Training: not yet run (waits for featurize completion)")

    if eval_report.exists():
        lines.append(f"- ✅ Eval report: **`{eval_report.relative_to(ROOT)}`**")
    else:
        lines.append("- ⏭ Eval report: not yet generated")

    lines.append("")

    # What to read first
    lines.append("## What to read first")
    lines.append("")
    next_steps: list[str] = []
    if eval_report.exists():
        next_steps.append(f"Open **`{eval_report.relative_to(ROOT)}`** — headline metrics + per-target detail.")
    else:
        next_steps.append("Wait for `artifacts/eval_report.md` to be generated, then open it.")
    if blockers.exists():
        next_steps.append(f"Open **`{blockers.relative_to(ROOT)}`** — anything I got stuck on.")
    next_steps.append("Check `git log --oneline` to see the commit timeline.")
    for i, step in enumerate(next_steps, start=1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # Files written this run
    lines.append("## Files of interest")
    lines.append("")
    files_of_interest = [
        ("artifacts/eval_report.md", "headline result + metrics"),
        ("artifacts/baseline_results.json", "machine-readable per-fold scores"),
        ("data/bacdive_phenotypes.parquet", "phenotype labels (gitignored)"),
        ("data/features.parquet", "extracted genome features (gitignored)"),
        ("data/training_table.parquet", "merged + group-keyed table used for training (gitignored)"),
    ]
    for rel, desc in files_of_interest:
        path = ROOT / rel
        marker = "✅" if path.exists() else "—"
        size_mb = f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else ""
        lines.append(f"- {marker} `{rel}` — {desc} {size_mb}")
    lines.append("")

    # Commits overnight
    commits = _git_log_since_yesterday()
    if commits:
        lines.append("## Commits since yesterday")
        lines.append("")
        for c in commits:
            lines.append(f"- {c}")
        lines.append("")

    # Reminders
    lines.append("## Reminders")
    lines.append("")
    lines.append("- The NCBI API key was pasted into chat earlier in this session — "
                 "rotate it at ncbi.nlm.nih.gov → Account Settings → API Key Management → "
                 "Revoke + Create New.")
    lines.append("- The `caffeinate -dimsu` you started for the overnight run can be stopped "
                 "now (`Ctrl+C` in that terminal).")
    lines.append("- Display sleep / Battery settings: revert to your normal preferences if "
                 "you changed them last night.")
    lines.append("")

    out_path = ROOT / "OVERNIGHT_SUMMARY.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
