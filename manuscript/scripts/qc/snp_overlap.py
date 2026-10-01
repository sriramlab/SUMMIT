"""Plot snp overlap for the SUMMIT manuscript."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from types import SimpleNamespace as CohortResult


def build_overlap_matrix(results):
    return np.asarray(data["overlap_matrix"]), data["overlap_text"]


COHORT_ORDER = ["EUR_300k", "EUR", "SAS", "AFR"]


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "maf_overlap",
    ["#F7FBFC", "#D7EBF0", "#88B8C9", "#2C5B77", "#17384B"],
)


MAF_BINS: Tuple[Tuple[str, float, float, bool, str], ...] = (
    ("maf_0p01_0p05", 0.01, 0.05, False, "[0.01,0.05)"),
    ("maf_0p05_0p10", 0.05, 0.10, False, "[0.05,0.10)"),
    ("maf_0p10_0p20", 0.10, 0.20, False, "[0.10,0.20)"),
    ("maf_0p20_0p30", 0.20, 0.30, False, "[0.20,0.30)"),
    ("maf_0p30_0p50", 0.30, 0.50, True, "[0.30,0.50]"),
)


DENSITY_EDGES = np.linspace(0.01, 0.50, 81)


def format_count_short(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def format_axis_count(value: float, _pos: int) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def make_figure(results: Dict[str, CohortResult], maf_panel_stat: str) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), constrained_layout=True)
    ax_totals = axes[0, 0]
    ax_heat = axes[0, 1]
    ax_density = axes[1, 0]
    ax_bins = axes[1, 1]

    totals = [results[cohort].total_M for cohort in COHORT_ORDER]
    y = np.arange(len(COHORT_ORDER))
    bars = ax_totals.barh(
        y,
        totals,
        color=[results[cohort].color for cohort in COHORT_ORDER],
        height=0.72,
    )
    ax_totals.set_yticks(y, [results[cohort].label for cohort in COHORT_ORDER])
    ax_totals.invert_yaxis()
    ax_totals.set_xlabel("SNPs")
    ax_totals.set_title("A. SNP set size", loc="left")
    ax_totals.grid(axis="x", color="#D9E1E6", linewidth=0.8)
    ax_totals.xaxis.set_major_formatter(FuncFormatter(format_axis_count))
    xmax = max(totals) * 1.12
    ax_totals.set_xlim(0, xmax)
    for bar, value in zip(bars, totals):
        ax_totals.text(
            bar.get_width() + max(totals) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            format_count_short(value),
            va="center",
            ha="left",
            fontsize=10,
            color="#243B53",
        )

    overlap_matrix, overlap_text = build_overlap_matrix(results)
    image = ax_heat.imshow(overlap_matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
    labels = [results[cohort].label for cohort in COHORT_ORDER]
    ax_heat.set_xticks(np.arange(len(COHORT_ORDER)), labels)
    ax_heat.set_yticks(np.arange(len(COHORT_ORDER)), labels)
    plt.setp(ax_heat.get_xticklabels(), rotation=0, ha="center")
    ax_heat.set_title("B. Pairwise SNP overlap", loc="left")
    for i in range(len(COHORT_ORDER)):
        for j in range(len(COHORT_ORDER)):
            value = overlap_matrix[i, j]
            text_color = "white" if value >= 0.58 else "#102A43"
            ax_heat.text(
                j,
                i,
                overlap_text[i][j],
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                linespacing=1.1,
            )
    divider = make_axes_locatable(ax_heat)
    cax = divider.append_axes("right", size="4.5%", pad=0.12)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_ylabel(
        "Intersection / smaller cohort", rotation=90, va="center", labelpad=14
    )

    for cohort in COHORT_ORDER:
        edges = DENSITY_EDGES
        density = np.asarray(results[cohort].hist_counts)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_density.plot(
            centers,
            density,
            color=results[cohort].color,
            linewidth=2.4,
            label=results[cohort].label,
        )
    ax_density.set_xlim(0.01, 0.50)
    ax_density.set_xlabel("Minor allele frequency")
    ax_density.set_ylabel("Density" if maf_panel_stat == "density" else "SNPs")
    ax_density.set_title("C. MAF distributions", loc="left")
    ax_density.grid(axis="y", color="#D9E1E6", linewidth=0.8)
    ax_density.legend(loc="upper right", frameon=False, ncol=2)

    bin_labels = [label for _name, _lo, _hi, _inc, label in MAF_BINS]
    x = np.arange(len(bin_labels))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(COHORT_ORDER))
    for offset, cohort in zip(offsets, COHORT_ORDER):
        counts = [
            results[cohort].maf_bin_counts[name]
            for name, _lo, _hi, _inc, _label in MAF_BINS
        ]
        ax_bins.bar(
            x + offset,
            counts,
            width=width,
            color=results[cohort].color,
            edgecolor="white",
            linewidth=0.6,
        )
    ax_bins.set_xticks(x, bin_labels, rotation=0, ha="center")
    ax_bins.set_ylabel("SNPs")
    ax_bins.set_title("D. Counts by MAF bin", loc="left")
    ax_bins.grid(axis="y", color="#D9E1E6", linewidth=0.8)
    ax_bins.yaxis.set_major_formatter(FuncFormatter(format_axis_count))

    return fig


if __name__ == "__main__":
    import json
    from pathlib import Path

    data = json.loads(Path("data/qc/snp_overlap.json").read_text())
    results = {pop: CohortResult(**row) for pop, row in data["cohorts"].items()}
    fig = make_figure(results, "count")
    fig.savefig("figs/supplementary/fig_s01_snp_sets.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
