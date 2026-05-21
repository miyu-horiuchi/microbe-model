"""Generate architecture diagrams for paper/architecture.pdf.

Re-run with:
    uv run --python 3.11 --with matplotlib python paper/architecture_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["EB Garamond", "Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

LAYER_FILL = "#f3f3f3"
LAYER_EDGE = "#222"
ACCENT = "#111"
GHOST = "#bbbbbb"


def box(ax, x, y, w, h, text, *, fill=LAYER_FILL, fontsize=9, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.9,
        edgecolor=LAYER_EDGE,
        facecolor=fill,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )


def arrow(ax, x0, y0, x1, y1, *, color="#222", style="->", lw=0.9):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(a)


def fig_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Layer 1 — Ingestion
    box(ax, 0.5, 10.5, 9, 1.0,
        "Layer 1 — Ingestion\nBacDive v2 REST   joined with   NCBI Datasets v2   >>   streaming pyrodigal CDS   (FASTA discarded)",
        fontsize=9)

    # Layer 2 — Six parallel feature paths
    box(ax, 0.5, 6.6, 9, 3.4,
        "", fill=LAYER_FILL)
    ax.text(5, 9.7, "Layer 2 — Six parallel feature paths (per genome)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    paths = [
        ("composition\ncodon, tetra\n~355 cols", "local CPU"),
        ("MediaDive\nrecipe stats\n5 cols", "local CPU"),
        ("Pfam HMMs\n48 markers\n144 cols", "pyhmmer\nlocal"),
        ("KEGG modules\nKOfam scan\n570 cols", "Modal GPU"),
        ("Isolation\nmeta (lat/lon)\n111 cols", "local CPU"),
        ("PTPE: ESM-2\nHMM-gated\n5,128 cols", "Modal GPU"),
    ]
    x = 0.7
    w = 1.46
    gap = 0.04
    for label, where in paths:
        box(ax, x, 7.5, w, 1.6, label, fontsize=7.8)
        ax.text(x + w / 2, 7.18, where, ha="center", va="center", fontsize=7, color="#444", style="italic")
        x += w + gap

    # Layer 3 — Fusion
    box(ax, 0.5, 5.4, 9, 0.9,
        "Layer 3 — Fusion: left-join on genome_accession   >>   training_table.parquet   (6,313 cols x 46,029 rows)",
        fontsize=9)

    # Layer 4 — Modeling (hybrid predictor)
    box(ax, 0.5, 2.5, 9, 2.5, "", fill=LAYER_FILL)
    ax.text(5, 4.7, "Layer 4 — Hybrid predictor (5-fold GroupKFold by family)",
            ha="center", va="center", fontsize=9, fontweight="bold")
    box(ax, 0.8, 3.0, 2.7, 1.4,
        "Tabular XGBoost\nT_opt, pH, salt heads\nregression",
        fontsize=8)
    box(ax, 3.65, 3.0, 2.7, 1.4,
        "LoRA on ESM-2 t12\nHMM-gated marker proteins\n4-class oxygen head",
        fontsize=8)
    box(ax, 6.5, 3.0, 2.7, 1.4,
        "Medium recommender\n40 per-medium\nXGBoost classifiers",
        fontsize=8)

    # Layer 5 — Serving
    box(ax, 0.5, 1.1, 9, 0.9,
        "Layer 5 — Serving: FastAPI   >>   React/Vite   >>   HuggingFace Docker Space",
        fontsize=9)

    # Arrows between layers
    for y0, y1 in [(10.5, 10.0), (7.0, 6.3), (5.4, 5.0), (2.5, 2.0)]:
        arrow(ax, 5, y0, 5, y1)

    fig.savefig(OUT_DIR / "arch_system_overview.png")
    plt.close(fig)


def fig_hybrid_decision() -> None:
    """Per-target model-selection matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    targets = ["Temperature", "pH", "Salt", "Oxygen", "Medium"]
    tabular = [0.94, 0.93, 0.65, 0.18, 0.86]
    lora = [0.62, 0.55, 0.62, 1.00, 0.0]

    # heatmap-style cells with text
    cell_w = 1.6
    cell_h = 1.0
    x0 = 1.0
    for i, t in enumerate(targets):
        x = x0 + i * cell_w
        ax.text(x + cell_w / 2, 3.4, t, ha="center", va="center", fontsize=8, fontweight="bold")
        # Tabular row
        choice_t = tabular[i] >= lora[i]
        box(ax, x, 2.0, cell_w - 0.1, cell_h, "tabular", fontsize=8,
            fill=("#cfcfcf" if choice_t else "#f5f5f5"))
        # LoRA row
        choice_l = not choice_t
        if i < 4:
            box(ax, x, 0.85, cell_w - 0.1, cell_h, "LoRA", fontsize=8,
                fill=("#cfcfcf" if choice_l else "#f5f5f5"))
        else:
            box(ax, x, 0.85, cell_w - 0.1, cell_h, "(n/a)", fontsize=8, fill="#f5f5f5")
    ax.text(0.85, 2.5, "Tabular", ha="right", va="center", fontsize=8, fontweight="bold")
    ax.text(0.85, 1.35, "LoRA",    ha="right", va="center", fontsize=8, fontweight="bold")
    ax.text(5.0, 0.2, "shaded cell = production choice for this target", ha="center", va="center",
            fontsize=7.5, color="#555", style="italic")

    fig.savefig(OUT_DIR / "arch_hybrid_decision.png")
    plt.close(fig)


def fig_feature_target_matrix() -> None:
    """Which feature paths matter for which targets."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    feature_paths = [
        "composition / codon",
        "MediaDive recipes",
        "Pfam HMMs (48)",
        "KEGG modules (570)",
        "isolation metadata",
        "PTPE embeddings",
        "LoRA on markers",
    ]
    targets = ["T_opt", "pH", "O2", "salt", "medium"]
    # Rows = feature path, cols = target. 0=n/a, 1=weak, 2=contributes, 3=dominant
    importance = np.array([
        [3, 1, 1, 2, 1],   # composition
        [1, 3, 1, 3, 2],   # MediaDive
        [2, 2, 3, 2, 1],   # Pfam HMMs
        [1, 2, 2, 1, 3],   # KEGG modules
        [2, 2, 1, 2, 1],   # isolation
        [2, 2, 2, 2, 1],   # PTPE
        [0, 0, 3, 0, 0],   # LoRA
    ])
    cmap = ["#ffffff", "#dddddd", "#999999", "#222222"]
    for i, _ in enumerate(feature_paths):
        for j, _ in enumerate(targets):
            v = importance[i, j]
            ax.add_patch(plt.Rectangle((j, len(feature_paths) - 1 - i), 1, 1,
                                       facecolor=cmap[v], edgecolor="#888", linewidth=0.5))
    ax.set_xlim(0, len(targets))
    ax.set_ylim(0, len(feature_paths))
    ax.set_xticks([j + 0.5 for j in range(len(targets))])
    ax.set_xticklabels(targets)
    ax.set_yticks([len(feature_paths) - 1 - i + 0.5 for i in range(len(feature_paths))])
    ax.set_yticklabels(feature_paths)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Legend
    handles = [
        mpatches.Patch(facecolor=cmap[0], edgecolor="#888", label="n/a"),
        mpatches.Patch(facecolor=cmap[1], edgecolor="#888", label="weak"),
        mpatches.Patch(facecolor=cmap[2], edgecolor="#888", label="contributes"),
        mpatches.Patch(facecolor=cmap[3], edgecolor="#888", label="dominant"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.18),
              frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / "arch_feature_target_matrix.png")
    plt.close(fig)


def main() -> None:
    fig_system_overview()
    fig_hybrid_decision()
    fig_feature_target_matrix()
    print(f"Wrote 3 architecture figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
