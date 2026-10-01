"""Plot reference swap for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import diagnostic
import enrichment_layout as merged


def set_style() -> None:
    merged.set_style()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.7,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 360,
        }
    )


def load_reference_swap() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the same filters as the supplementary diagnostic figure."""
    df = pd.read_csv(diagnostic.INPUT, sep="\t")
    df = df[
        ~df["trait"].map(diagnostic.source_from_trait).isin(diagnostic.DROP_SOURCES)
    ].copy()
    df["is_maf"] = df["annotation"].map(diagnostic.is_maf_annotation)

    positive = df[(df["eur_enrichment"] > 0) & (df["sas_enrichment"] > 0)].copy()
    positive["log2_eur_enrichment"] = np.log2(positive["eur_enrichment"])
    positive["log2_sas_enrichment"] = np.log2(positive["sas_enrichment"])
    positive["log2_enrichment_ratio"] = np.log2(
        positive["enrichment_ratio_sas_over_eur"]
    )

    non_maf = positive[~positive["is_maf"]].copy()
    maf = positive[positive["is_maf"]].copy()
    maf["maf_bin"] = maf["annotation"].str.extract(r"MAFbin(\d+)")[0].astype(int)
    return positive, non_maf, maf


def add_panel_label(
    fig: plt.Figure, ax: plt.Axes, label: str, dx: float, dy: float = 0.010
) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 + dx,
        box.y1 + dy,
        label,
        ha="left",
        va="bottom",
        fontsize=17.0,
        fontweight="bold",
        color="#111827",
    )


def draw_supplementary_scatter(
    ax: plt.Axes,
    positive: pd.DataFrame,
    non_maf: pd.DataFrame,
    maf: pd.DataFrame,
) -> None:
    """Reproduce panel A of the supplementary diagnostic figure."""
    ax.scatter(
        non_maf["log2_eur_enrichment"],
        non_maf["log2_sas_enrichment"],
        s=13,
        color=diagnostic.NON_MAF_COLOR,
        alpha=0.45,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        maf["log2_eur_enrichment"],
        maf["log2_sas_enrichment"],
        s=22,
        marker="^",
        color=diagnostic.MAF_COLOR,
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )
    diagnostic.add_density_contours(
        ax,
        non_maf["log2_eur_enrichment"],
        non_maf["log2_sas_enrichment"],
        color=diagnostic.CONTOUR_COLOR,
    )

    xmin = -4.0
    xmax = (
        max(
            positive["log2_eur_enrichment"].max(), positive["log2_sas_enrichment"].max()
        )
        + 0.25
    )
    ax.plot([xmin, xmax], [xmin, xmax], ls="--", lw=1.2, color="#333333", zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel("log2 enrichment using EUR LD reference")
    ax.set_ylabel("log2 enrichment using SAS LD reference")
    ax.grid(True, color="#e1e5ea", linewidth=0.8)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=diagnostic.NON_MAF_COLOR,
            markeredgecolor="none",
            markersize=5.5,
            alpha=0.65,
            label=f"Non-MAF bins (n={len(non_maf)})",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=diagnostic.MAF_COLOR,
            markeredgecolor="none",
            markersize=6.2,
            alpha=0.85,
            label=f"MAF bins (n={len(maf)})",
        ),
        Line2D(
            [0],
            [0],
            color=diagnostic.CONTOUR_COLOR,
            lw=1.5,
            label="Non-MAF density contour",
        ),
        Line2D([0], [0], color="#333333", lw=1.2, ls="--", label="Identity line"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        framealpha=0.92,
        handlelength=1.8,
    )


def format_diagnostic_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.1)
