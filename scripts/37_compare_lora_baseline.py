"""Compare the fold-0 LoRA result against the current tabular baseline.

The LoRA run is a single fold, while artifacts/baseline_results.json is a
five-fold mean. This script still makes the direction of change explicit and
writes a small Markdown report for the experiment log.

Usage:
    python scripts/37_compare_lora_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

from microbe_model import config


TARGETS = {
    "optimal_temperature_c": ("temp", "mae", "MAE", False),
    "optimal_ph": ("ph", "mae", "MAE", False),
    "salt_tolerance_pct": ("salt", "mae", "MAE", False),
    "oxygen_requirement": ("oxy", "f1_macro", "macro F1", True),
}


def verdict(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-9:
        return "tie"
    is_better = delta > 0 if higher_is_better else delta < 0
    return "better" if is_better else "worse"


def main() -> None:
    lora_path = config.ARTIFACTS / "lora" / "fold0_results.json"
    baseline_path = config.ARTIFACTS / "baseline_results.json"
    if not lora_path.exists():
        raise SystemExit(f"Missing {lora_path}")
    if not baseline_path.exists():
        raise SystemExit(f"Missing {baseline_path}")

    lora = json.loads(lora_path.read_text())["best"]["val"]
    baseline = json.loads(baseline_path.read_text())

    lines = [
        "# LoRA Fold 0 vs Tabular Baseline",
        "",
        "Caveat: LoRA is one group fold; baseline is the current five-fold mean.",
        "",
        "| Target | LoRA | Baseline | Delta | Verdict |",
        "|---|---:|---:|---:|---|",
    ]

    for target, (lora_key, metric_key, label, higher_is_better) in TARGETS.items():
        lora_value = float(lora[lora_key][metric_key])
        baseline_value = float(baseline[target]["mean_metric"])
        delta = lora_value - baseline_value
        lines.append(
            f"| `{target}` {label} | {lora_value:.4f} | {baseline_value:.4f} | "
            f"{delta:+.4f} | {verdict(delta, higher_is_better)} |"
        )

    lines.extend([
        "",
        "## Recommendation",
        "",
        "The first LoRA pass is strongest for oxygen classification. For the next GPU run, "
        "use `scripts/lambda_train_lora.py --target-preset oxygen` instead of spending "
        "more A100 time optimizing regression losses that underperformed the tabular baseline.",
        "",
        "Keep `artifacts/lora/fold0_best.pt` outside git unless it is published to a model "
        "store or release asset; the JSON metrics and log are enough for repo history.",
    ])

    out = config.ARTIFACTS / "lora_vs_baseline.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
