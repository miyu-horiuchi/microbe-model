"""Render paper/figures/*.png — three horizontal-bar comparison figures
that visualise the prior-work scoreboard.

Re-run with:

    uv run --python 3.11 --with matplotlib python paper/figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["EB Garamond", "Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222",
        "xtick.color": "#222",
        "ytick.color": "#222",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

THIS_WORK = "#111111"
OTHER = "#888888"


def hbar(
    ax,
    labels: list[str],
    values: list[float],
    is_this_work: list[bool],
    *,
    value_fmt: str,
    xlabel: str,
    annotate_best: bool = True,
    best_is_min: bool = False,
) -> None:
    y = list(range(len(labels)))[::-1]
    colors = [THIS_WORK if w else OTHER for w in is_this_work]
    ax.barh(y, values, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    span = max(values) - min(values) if values else 1
    pad = span * 0.02
    for yi, v in zip(y, values):
        ax.text(v + pad, yi, value_fmt.format(v), va="center", fontsize=8)


def fig_temperature() -> None:
    labels = ["GenomeSPOT", "Koblitz 2025", "This work (pre-PTPE)", "This work (+PTPE)"]
    values = [4.39, 2.94, 2.74, 2.67]
    is_this_work = [False, False, True, True]
    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    hbar(
        ax,
        labels,
        values,
        is_this_work,
        value_fmt="{:.2f} °C",
        xlabel="Optimum temperature MAE (°C) — lower is better",
        best_is_min=True,
    )
    ax.set_xlim(0, max(values) * 1.25)
    fig.savefig(OUT_DIR / "fig_temperature_mae.png")
    plt.close(fig)


def fig_oxygen() -> None:
    rows = [
        ("Koblitz 2025", 0.04, None, "#cccccc", "binary aerobe / anaerobe — different label space"),
        ("GenomeSPOT", 0.04, None, "#cccccc", "binary tolerant / not-tolerant — different label space"),
        ("This work — pre-PTPE", 0.412, "{:.3f}", THIS_WORK, None),
        ("This work — tabular", 0.402, "{:.3f}", THIS_WORK, None),
        ("This work — LoRA", 0.945, "{:.3f}", THIS_WORK, None),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    y = list(range(len(rows)))[::-1]
    bars = ax.barh(y, values, color=colors, height=0.55)
    for bar, row in zip(bars, rows):
        if row[4] is not None:
            bar.set_hatch("////")
            bar.set_edgecolor("#888")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Oxygen requirement macro-F1 (4-class) — higher is better")
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.05)
    for yi, row in zip(y, rows):
        value, value_fmt, note = row[1], row[2], row[4]
        if note is None:
            ax.text(value + 0.02, yi, value_fmt.format(value), va="center", fontsize=8)
        else:
            ax.text(value + 0.02, yi, note, va="center", fontsize=8, color="#555", style="italic")
    fig.savefig(OUT_DIR / "fig_oxygen_f1.png")
    plt.close(fig)


def fig_medium() -> None:
    labels = [
        "Global popularity",
        "Taxonomic popularity",
        "XGBoost recommender (this work)",
    ]
    values = [0.366, 0.372, 0.775]
    is_this_work = [False, False, True]
    fig, ax = plt.subplots(figsize=(5.2, 1.55))
    hbar(
        ax,
        labels,
        values,
        is_this_work,
        value_fmt="{:.1%}",
        xlabel="Medium recommender Hit@5, 5-fold family-heldout — higher is better",
        best_is_min=False,
    )
    ax.set_xlim(0, 1.0)
    fig.savefig(OUT_DIR / "fig_medium_hit5.png")
    plt.close(fig)


def main() -> None:
    fig_temperature()
    fig_oxygen()
    fig_medium()
    print(f"Wrote 3 figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
