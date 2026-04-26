"""Evaluation report generation.

Renders a markdown report from a trained-results JSON (the output of train/baseline.py)
joined with the source dataset. Designed to be readable cold — every number includes
a comparison baseline so the reader can interpret it without context.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microbe_model import config


def _baseline_mae(y: np.ndarray) -> float:
    """MAE of the always-predict-mean baseline (sanity floor)."""
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs(y - np.mean(y))))


def _baseline_f1(y: np.ndarray) -> float:
    """Macro-F1 of the always-predict-majority baseline."""
    from sklearn.metrics import f1_score
    if len(y) == 0:
        return float("nan")
    values, counts = np.unique(y, return_counts=True)
    majority = values[np.argmax(counts)]
    pred = np.full_like(y, majority)
    return float(f1_score(y, pred, average="macro"))


def render_report(
    results_path: Path,
    dataset_path: Path,
    out_path: Path,
    *,
    n_strains: int | None = None,
    runtime_seconds: float | None = None,
    predictions_path: Path | None = None,
) -> None:
    results: dict[str, Any] = json.loads(results_path.read_text())
    df = pd.read_parquet(dataset_path)
    predictions = (
        pd.read_parquet(predictions_path)
        if predictions_path is not None and predictions_path.exists()
        else None
    )

    lines: list[str] = []
    lines.append("# microbe-model — v0 baseline eval report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(UTC).isoformat(timespec='seconds')}_")
    lines.append("")

    # Section: TL;DR — the headline number
    lines.append("## TL;DR")
    lines.append("")
    headline_lines = []
    for target, r in results.items():
        if not r["folds"]:
            continue
        y = df[target].dropna().to_numpy()
        if r["task"] == "regression":
            baseline = _baseline_mae(y.astype(float))
            improvement = (baseline - r["mean_metric"]) / max(0.001, baseline) * 100
            headline_lines.append(
                f"- **`{target}`**: MAE = **{r['mean_metric']:.2f}** "
                f"(vs always-predict-mean {baseline:.2f}, **{improvement:+.0f}%**)"
            )
        else:
            baseline = _baseline_f1(y)
            improvement = (r["mean_metric"] - baseline) / max(0.001, baseline) * 100
            headline_lines.append(
                f"- **`{target}`**: macro-F1 = **{r['mean_metric']:.3f}** "
                f"(vs always-predict-majority {baseline:.3f}, **{improvement:+.0f}%**)"
            )
    lines.extend(headline_lines if headline_lines else [
        "- _No targets trained successfully — see logs._"
    ])
    lines.append("")
    lines.append(f"Trained on **{len(df):,}** strains with **{len([c for c in df.columns if c.startswith(('aa_frac_', 'genome_size', 'gc_', 'n_predicted', 'coding_', 'mean_', 'aromatic_', 'pos_', 'neg_', 'ivywrel_', 'median_'))])}** genome-derived features. "
                 f"Cross-validation: 5-fold GroupKFold by taxonomic family.")
    lines.append("")

    # Section: corpus
    lines.append("## Corpus")
    lines.append("")
    lines.append(f"- Total strains in feature table: **{len(df):,}**")
    if n_strains is not None:
        lines.append(f"- Total strains attempted (had genome accession + label): {n_strains:,}")
        lines.append(f"- Feature-extraction success rate: {100 * len(df) / max(1, n_strains):.1f}%")
    if runtime_seconds is not None:
        lines.append(f"- Featurize wall time: {runtime_seconds / 60:.1f} min")
    # Per-target label counts
    lines.append("- Labeled-strain counts by target:")
    for target in ("optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"):
        if target in df.columns:
            n = df[target].notna().sum()
            lines.append(f"  - `{target}`: {n:,}")
    lines.append("")

    # Section: data exploration — distributions of the regression targets
    lines.append("## Target distributions")
    lines.append("")
    for target in ("optimal_temperature_c", "optimal_ph", "salt_tolerance_pct"):
        if target not in df.columns:
            continue
        y = df[target].dropna()
        if len(y) == 0:
            continue
        lines.append(
            f"- `{target}`: n={len(y):,}, mean={y.mean():.2f}, "
            f"std={y.std():.2f}, p10={y.quantile(0.1):.2f}, "
            f"median={y.median():.2f}, p90={y.quantile(0.9):.2f}"
        )
    if "oxygen_requirement" in df.columns:
        lines.append("- `oxygen_requirement`:")
        for cls, n in df["oxygen_requirement"].value_counts().head(10).items():
            lines.append(f"  - `{cls}`: {n:,}")
    lines.append("")

    # Section: per-target results
    lines.append("## Per-target results (5-fold GroupKFold by family)")
    lines.append("")
    lines.append("Metrics: regression = MAE (lower is better), classification = macro-F1 (higher is better).")
    lines.append("Each is shown alongside the dumb-baseline (always-predict-mean / always-predict-majority).")
    lines.append("")
    lines.append("| Target | Task | n labeled | Model metric | Baseline | Improvement |")
    lines.append("|---|---|---|---|---|---|")
    for target, r in results.items():
        if not r["folds"]:
            lines.append(f"| {target} | {r['task']} | — | _skipped (insufficient data)_ | — | — |")
            continue
        y = df[target].dropna().to_numpy()
        n_labeled = len(y)
        if r["task"] == "regression":
            baseline = _baseline_mae(y.astype(float))
            mean = r["mean_metric"]
            improvement = f"{(baseline - mean) / baseline * 100:+.1f}%"
            lines.append(f"| `{target}` | regression | {n_labeled:,} | "
                         f"MAE={mean:.3f} | MAE={baseline:.3f} | {improvement} |")
        else:
            baseline = _baseline_f1(y)
            mean = r["mean_metric"]
            improvement = f"{(mean - baseline) / max(0.01, baseline) * 100:+.1f}%"
            lines.append(f"| `{target}` | classification | {n_labeled:,} | "
                         f"F1={mean:.3f} | F1={baseline:.3f} | {improvement} |")
    lines.append("")

    # Section: per-fold detail
    for target, r in results.items():
        if not r["folds"]:
            continue
        lines.append(f"### `{target}` — fold-by-fold")
        lines.append("")
        lines.append("| Fold | Metric | Train | Test |")
        lines.append("|---|---|---|---|")
        for i, f in enumerate(r["folds"]):
            lines.append(f"| {i+1} | {f['metric_name']} = {f['value']:.3f} | "
                         f"n={f['n_train']:,} | n={f['n_test']:,} |")
        lines.append("")

        top = r.get("top_features", {})
        if top:
            lines.append(f"**Top 10 features for `{target}`:**")
            lines.append("")
            for name, importance in list(top.items())[:10]:
                lines.append(f"- `{name}` — {importance:.4f}")
            lines.append("")

    # Section: feature-target correlations (data-exploration sanity check)
    feature_cols = [
        c for c in df.columns
        if c.startswith(("aa_frac_", "genome_size", "gc_", "n_predicted", "coding_",
                          "mean_", "aromatic_", "pos_", "neg_", "ivywrel_", "median_"))
    ]
    if feature_cols:
        from microbe_model.explore import feature_target_correlations
        lines.append("## Feature ↔ target correlations (Spearman, top 10)")
        lines.append("")
        lines.append("Sanity-checks the biology — features known to track each target should "
                     "appear here at high |ρ|. E.g. `ivywrel_frac` should correlate with "
                     "`optimal_temperature_c` (Zeldovich 2007 thermophile signature).")
        lines.append("")
        for target in ("optimal_temperature_c", "optimal_ph", "salt_tolerance_pct"):
            corrs = feature_target_correlations(df, feature_cols, target, top_n=10)
            if not corrs:
                continue
            lines.append(f"### `{target}`")
            lines.append("")
            lines.append("| Feature | Spearman ρ | p-value |")
            lines.append("|---|---|---|")
            for row in corrs:
                lines.append(f"| `{row['feature']}` | {row['spearman_rho']:+.3f} | "
                             f"{row['p_value']:.1e} |")
            lines.append("")

    # Section: per-phylum error breakdown (regression targets only)
    if predictions is not None and not predictions.empty and "row_idx" in predictions.columns:
        joined = predictions.merge(
            df[["genus", "family"]].rename_axis("row_idx").reset_index(),
            on="row_idx",
            how="left",
        )
        regression_preds = joined[joined["task"] == "regression"]
        if not regression_preds.empty:
            lines.append("## Per-family error breakdown (regression targets)")
            lines.append("")
            lines.append("Top 15 most-represented families, MAE per family. Highlights where the "
                         "model is doing well vs. struggling.")
            lines.append("")
            for target in regression_preds["target"].unique():
                sub = regression_preds[regression_preds["target"] == target].copy()
                sub["abs_error"] = (
                    pd.to_numeric(sub["predicted"]) - pd.to_numeric(sub["observed"])
                ).abs()
                grp = (sub.groupby("family", dropna=False)
                       .agg(n=("abs_error", "size"), mae=("abs_error", "mean"))
                       .sort_values("n", ascending=False)
                       .head(15))
                if grp.empty:
                    continue
                lines.append(f"### `{target}`")
                lines.append("")
                lines.append("| Family | n | MAE |")
                lines.append("|---|---|---|")
                for fam, row in grp.iterrows():
                    fam_label = fam if pd.notna(fam) else "_(no family)_"
                    lines.append(f"| {fam_label} | {int(row['n'])} | {row['mae']:.3f} |")
                lines.append("")

    # Section: limitations
    lines.append("## Known limitations")
    lines.append("")
    lines.append("- **Survivorship bias.** BacDive only contains organisms that have been cultured "
                 "successfully at least once. The model cannot generalize to truly uncultured strains "
                 "without explicit out-of-distribution evaluation.")
    lines.append("- **Optimum derivation is heuristic.** Most BacDive temperature entries are tagged "
                 "as `growth` (positive growth at this temperature), not `optimum`. We approximate "
                 "the optimum as the median of positive-growth temperatures when no explicit "
                 "optimum is recorded — this can be off by 5°C or more for some strains.")
    lines.append("- **Family grouping is naive.** The current `family` column is derived from the "
                 "genus (first word of binomial name). A proper LPSN/GTDB family assignment would "
                 "give tighter taxonomic grouping.")
    lines.append("- **Feature set is shallow.** No HMM/KEGG annotations, no codon usage indices, no "
                 "tRNA counts. These are interpretable next steps before moving to genome LMs.")
    lines.append("- **Pyrodigal accuracy.** Gene prediction quality drops on highly-fragmented "
                 "assemblies and atypical genetic codes. Not currently flagged in the feature set.")
    lines.append("")

    # Section: next steps
    lines.append("## Next steps")
    lines.append("")
    lines.append("1. **Add tetranucleotide / codon-usage features.** ~50 extra columns, "
                 "well-known signal for thermophily.")
    lines.append("2. **Replace naive family lookup with LPSN/GTDB join.** Reduces leakage in CV.")
    lines.append("3. **Integrate KOMODO media DB** as a richer label source than BacDive alone.")
    lines.append("4. **Move to genome embeddings** (Nucleotide Transformer / Evo-1 / DNABERT-2) "
                 "once the tabular ceiling is established.")
    lines.append("5. **Active learning loop**: select novel-family strains where the model is "
                 "uncertain, prioritize these for wet-lab cultivation testing.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    render_report(
        results_path=config.ARTIFACTS / "baseline_results.json",
        dataset_path=config.DATA / "training_table.parquet",
        out_path=config.ARTIFACTS / "eval_report.md",
        predictions_path=config.ARTIFACTS / "predictions.parquet",
    )
