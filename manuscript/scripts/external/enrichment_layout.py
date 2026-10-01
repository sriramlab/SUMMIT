"""Plot enrichment layout for the SUMMIT manuscript."""
from __future__ import annotations

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
import enrichment as enrich

TAG = "without_mvp_hare"


def set_style() -> None:
    enrich.covplot.set_style()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.4,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.9,
            "xtick.labelsize": 10.3,
            "ytick.labelsize": 10.3,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.75,
            "figure.dpi": 170,
            "savefig.dpi": 340,
        }
    )


def load_enrichment():
    enrich_df = enrich.load_enrichment(TAG)
    enrich_jack = enrich.load_enrichment_jackknife(TAG)
    enrich_df = enrich_df[
        enrich_df["annotation"].isin(enrich.SELECTED_PLUS_QTL)
        & enrich_df["pop"].isin(enrich.covplot.POPS)
    ].copy()
    enrich_jack = enrich_jack[
        enrich_jack["annotation"].isin(enrich.SELECTED_PLUS_QTL)
        & enrich_jack["pop"].isin(enrich.covplot.POPS)
    ].copy()
    return enrich_df, enrich_jack


def add_enrichment_legend(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=enrich.covplot.POP_COLOR[pop],
            label=enrich.covplot.POP_LABEL[pop],
            markersize=5.3,
            lw=1.15,
        )
        for pop in enrich.covplot.POPS
    ]
    handles.append(
        Line2D([0], [0], marker="D", color="0.22", label="median", markersize=5.4, lw=0)
    )
    boxes = [ax.get_position() for ax in axes]
    external_box = boxes[1]
    legend_x = external_box.x1 - 0.010
    legend_y = external_box.y0 + 0.86 * (external_box.y1 - external_box.y0)
    legend = fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(legend_x, legend_y),
        frameon=True,
        fancybox=False,
        ncol=1,
        handlelength=1.05,
        handletextpad=0.42,
        borderpad=0.30,
        labelspacing=0.28,
        fontsize=9.8,
        borderaxespad=0.0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#9ca3af")
    legend.get_frame().set_linewidth(0.75)
    legend.get_frame().set_alpha(0.96)


def draw_enrichment_block(
    fig: plt.Figure, spec, enrich_df, enrich_jack
) -> list[plt.Axes]:
    grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.095)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    y_base = np.arange(len(enrich.SELECTED_PLUS_QTL))[::-1]
    y_lookup = {ann: y for ann, y in zip(enrich.SELECTED_PLUS_QTL, y_base)}
    offsets = {"EUR_300k": 0.21, "SAS": 0.0, "AFR": -0.21}
    rng = np.random.default_rng(20260528)

    for ax, (kind, title) in zip(
        axes, [("local", "UKB"), ("external", "External GWAS")]
    ):
        dat = enrich_df[
            enrich_df["dataset_kind"].eq(kind)
            & enrich_df["annotation"].isin(enrich.SELECTED_PLUS_QTL)
        ]
        summ = enrich_jack[
            enrich_jack["dataset_kind"].eq(kind)
            & enrich_jack["annotation"].isin(enrich.SELECTED_PLUS_QTL)
        ]
        ax.axvline(1.0, color="0.25", lw=0.85, ls="--", zorder=1)
        ax.grid(axis="x", color="0.90", lw=0.55)
        ax.set_axisbelow(True)

        for pop in enrich.covplot.POPS:
            color = enrich.covplot.POP_COLOR[pop]
            pop_dat = dat[dat["pop"].eq(pop) & np.isfinite(dat["signed_enrichment"])]
            pop_summ = summ[summ["pop"].eq(pop)]
            for ann in enrich.SELECTED_PLUS_QTL:
                sub = pop_dat[pop_dat["annotation"].eq(ann)]
                if sub.empty:
                    continue
                y = y_lookup[ann] + offsets[pop]
                vals = sub["signed_enrichment"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                nonneg = vals[vals >= 0.0]
                if nonneg.size:
                    jitter = rng.normal(0.0, 0.030, size=nonneg.size)
                    in_range = nonneg <= enrich.XMAX_ENRICH
                    ax.scatter(
                        nonneg[in_range],
                        y + jitter[in_range],
                        s=7.0,
                        color=color,
                        alpha=0.20,
                        linewidths=0,
                        zorder=3,
                    )
                    if np.any(~in_range):
                        ax.scatter(
                            np.full(np.sum(~in_range), enrich.XMAX_ENRICH),
                            y + jitter[~in_range],
                            s=9.0,
                            color=color,
                            alpha=0.22,
                            marker=">",
                            linewidths=0,
                            zorder=3,
                        )
                row = pop_summ[pop_summ["annotation"].eq(ann)]
                if row.empty:
                    continue
                r = row.iloc[0]
                estimate = float(r["estimate"])
                se = float(r["se"])
                enrich.draw_enrichment_jackknife_ci(
                    ax, estimate - se, estimate + se, y, color
                )
                enrich.draw_enrichment_median(ax, estimate, y, color)

        ax.set_xlim(0.0, enrich.XMAX_ENRICH)
        ax.set_xticks([0, 1, 4, 8, 12])
        ax.set_ylim(-0.65, len(enrich.SELECTED_PLUS_QTL) - 0.35)
        ax.set_title(title, pad=5.0, fontsize=14.0)
        ax.set_xlabel("Trait-level enrichment", fontsize=12.2, labelpad=2.5)
        ax.tick_params(labelsize=10.3, pad=1.2, length=2.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(
        [enrich.covplot.ANNOTATION_LABEL[a] for a in enrich.SELECTED_PLUS_QTL],
        fontsize=10.3,
    )
    axes[0].tick_params(axis="y", labelsize=10.3, pad=2.0)
    axes[1].tick_params(labelleft=False)
    return axes
