"""Plot numvec histogram for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METADATA_COLS = {"CHR", "SNP", "BP", "CM", "MAF"}


PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


COHORT_LABELS = {
    "AFR": "AFR",
    "SAS": "SAS",
    "EUR": "EUR",
    "EUR_300k": "EUR (300k)",
}


DEFAULT_COV_LABELS = {
    "AFR": "10pc",
    "EUR": "10pc",
    "SAS": "25pc",
    "EUR_300k": "10pc",
}


NUMVEC_COLORS = {
    10: "#9e3d22",
    100: "#3f6f9f",
    1000: "#2f7f5f",
    10000: "#2f2f2f",
}


def layout_shape(layout: str, n_panels: int) -> Tuple[int, int, Tuple[float, float]]:
    if layout == "row":
        return 1, n_panels, (7.4, 2.15)
    if layout == "grid":
        return 2, int(math.ceil(n_panels / 2)), (6.9, 5.1)
    raise ValueError(f"Unknown layout: {layout}")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def setup_axes(layout: str, n_panels: int):
    nrows, ncols, figsize = layout_shape(layout, n_panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat


def plot_histogram_figure(
    data_dir: Path,
    pops: Sequence[str],
    numvecs: Sequence[int],
    seed: int,
    layout: str,
    hist_clip: Tuple[float, float],
    bins: int,
    outdir: Path,
    dpi: int,
) -> None:
    fig, axes = setup_axes(layout, len(pops))
    handles = []
    labels = []
    for idx, (ax, pop) in enumerate(zip(axes, pops)):
        hist = pd.read_csv(data_dir)
        for numvec in numvecs:
            sub = hist[hist["pop"].eq(pop) & hist["numvec"].eq(numvec)]
            centers = sub["center"].to_numpy()
            counts = sub["density"].to_numpy()
            color = NUMVEC_COLORS.get(numvec, "#333333")
            (line,) = ax.plot(
                centers, counts, color=color, linewidth=1.15, label=f"{numvec:,}"
            )
            ax.fill_between(centers, counts, color=color, alpha=0.08, linewidth=0)
            if idx == 0:
                handles.append(line)
                labels.append(f"{numvec:,}")

        ax.grid(
            True, which="major", axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.7
        )
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="upper"))
        ax.set_title(COHORT_LABELS.get(pop, pop), pad=3)
        ax.text(
            -0.12,
            1.12,
            PANEL_LABELS[idx],
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
        ax.set_xlabel("LD score")
        if idx % (2 if layout == "grid" else len(pops)) == 0:
            ax.set_ylabel("Density")

    fig.legend(
        handles,
        labels,
        title="Random vectors",
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.subplots_adjust(
        top=0.78 if layout == "row" else 0.88,
        wspace=0.32 if layout == "row" else 0.28,
        hspace=0.55 if layout == "grid" else 0.1,
    )
    for ext in ["pdf", "png"]:
        fig.savefig(outdir / f"fig_s21_ld_projection_distributions.{ext}", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    plot_histogram_figure(
        Path("data/ld/numvec_histogram.csv"),
        ["EUR_300k", "EUR", "SAS", "AFR"],
        [10, 100, 1000, 10000],
        0,
        "row",
        (0.5, 99.5),
        160,
        Path("figs/supplementary"),
        600,
    )
