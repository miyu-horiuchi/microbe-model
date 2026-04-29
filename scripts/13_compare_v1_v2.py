"""Side-by-side comparison: v1 hand-crafted features vs v2 ESM-2 embeddings.

Reads:
  artifacts/baseline_results.json       (v1)
  artifacts/embedding_results.json      (v2)

Writes:
  artifacts/v1_vs_v2_comparison.md
"""
from __future__ import annotations

import json

from microbe_model import config


def main() -> None:
    v1_path = config.ARTIFACTS / "baseline_results.json"
    v2_path = config.ARTIFACTS / "embedding_results.json"
    if not v1_path.exists() or not v2_path.exists():
        raise SystemExit(f"Need both {v1_path} and {v2_path}")

    v1 = json.loads(v1_path.read_text())
    v2 = json.loads(v2_path.read_text())
    v1.pop("__meta__", None)
    v2.pop("__meta__", None)

    lines = [
        "# v1 (hand-crafted features) vs v2 (ESM-2 embeddings)",
        "",
        "Same train/test splits, same XGBoost hyperparameters. Only difference: input features.",
        "",
        "| Target | v1 (n features) | v2 (embedding dim) | Δ |",
        "|---|---|---|---|",
    ]
    for target in ("optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"):
        v1_metric = v1.get(target, {}).get("mean_metric")
        v2_metric = v2.get(target, {}).get("mean_metric")
        task = v1.get(target, {}).get("task") or v2.get(target, {}).get("task", "?")
        if v1_metric is None or v2_metric is None:
            lines.append(f"| `{target}` | — | — | — |")
            continue
        if task == "regression":
            delta = v2_metric - v1_metric
            arrow = "🟢" if delta < 0 else ("🔴" if delta > 0 else "⚪")
            lines.append(f"| `{target}` | MAE {v1_metric:.3f} | MAE {v2_metric:.3f} | "
                         f"{arrow} {delta:+.3f} ({delta / v1_metric * 100:+.1f}%) |")
        else:
            delta = v2_metric - v1_metric
            arrow = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
            lines.append(f"| `{target}` | F1 {v1_metric:.3f} | F1 {v2_metric:.3f} | "
                         f"{arrow} {delta:+.3f} ({delta / max(0.001, v1_metric) * 100:+.1f}%) |")

    lines.extend([
        "",
        "## Reading this table",
        "",
        "- 🟢 = embeddings beat hand-crafted features",
        "- 🔴 = hand-crafted features beat embeddings",
        "- ⚪ = no difference",
        "",
        "Regression: lower MAE is better, so a *negative* delta is good. ",
        "Classification: higher F1 is better, so a *positive* delta is good.",
        "",
        "## Interpretation",
        "",
        "- **≥ 10% lift on T_opt:** validates the genome-LM direction. Worth investing",
        "  in larger models (ESM-2 t33_650M or Nucleotide Transformer / Evo-1).",
        "- **pH or salt go from broken (≤5%) to working (≥15%):** embeddings recover",
        "  signal that hand-crafted features couldn't capture. Big win for the thesis.",
        "- **No meaningful lift anywhere:** the bottleneck is not feature engineering. ",
        "  Need new data sources (failed cultivation logs, environmental metadata).",
    ])

    out_path = config.ARTIFACTS / "v1_vs_v2_comparison.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
