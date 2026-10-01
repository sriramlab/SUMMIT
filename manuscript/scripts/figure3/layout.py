"""Plot layout for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import common
import normalization as panel_d
import method_style as v2
import architecture as v3

METRIC_LABEL = {
    "rg": r"Genome-wide $r_g$ error",
    "gencov": r"Genome-wide $\gamma_g$ error",
}


PANEL_A_HIGH = 0.15


def draw_panel_a(ax: plt.Axes, calibration: pd.DataFrame) -> None:
    """Null-r_g FPR on a useful central scale with labelled off-scale values."""
    centers = np.arange(len(v3.POPS), dtype=float)
    offsets = dict(zip(common.sim.METHODS, np.linspace(-0.27, 0.27, 3)))
    ax.axhline(
        0.05,
        color="#6E6E6E",
        linewidth=common._lw(1.2),
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    for boundary in centers[:-1] + 0.5:
        ax.axvline(boundary, color="#EBEBEB", linewidth=common._lw(1.0), zorder=0)

    cap = PANEL_A_HIGH - 0.006
    for method in common.sim.METHODS:
        color = common.sim.METHOD_COLOR[method]
        marker = common.sim.METHOD_MARKER[method]
        xs, plotted = [], []
        for pop_idx, pop in enumerate(v3.POPS):
            row = calibration[
                calibration["pop"].astype(str).eq(pop)
                & calibration["method"].astype(str).eq(method)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing calibration row for {pop}/{method}")
            result = row.iloc[0]
            value = float(result["fpr"])
            low = float(result["fpr_lo"])
            high = float(result["fpr_hi"])
            x = centers[pop_idx] + offsets[method]
            xs.append(x)
            if value > PANEL_A_HIGH:
                y = cap
                ax.scatter(
                    x,
                    y,
                    marker="^",
                    s=common._scatter_size(72),
                    color=color,
                    edgecolor="white",
                    linewidth=common._lw(0.9),
                    zorder=5,
                )
                ax.text(
                    x,
                    y - 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="top",
                    fontsize=common._fs(9.2),
                    color=color,
                    fontweight="bold",
                    zorder=6,
                )
            else:
                y = value
                shown_low = max(0.0, low)
                shown_high = min(cap, high)
                ax.errorbar(
                    [x],
                    [y],
                    yerr=np.asarray([[y - shown_low], [shown_high - y]]),
                    fmt=marker,
                    markersize=common._ms(8.0),
                    linewidth=common._lw(2.0),
                    capsize=common._ms(3.8),
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=common._lw(1.0),
                    zorder=3,
                )
            plotted.append(y)
        ax.plot(
            xs, plotted, color=color, linewidth=common._lw(1.5), alpha=0.35, zorder=2
        )

    ax.set_xlim(-0.5, len(centers) - 0.5)
    ax.set_ylim(0.0, PANEL_A_HIGH)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_xticks(centers)
    ax.set_xticklabels(v3.POPS)
    ax.set_ylabel("False-positive rate")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    common.style_axis(ax)


def draw_panel_b(ax: plt.Axes, summary: pd.DataFrame, metric: str) -> None:
    methods = list(common.sim.METHODS)
    x_lookup = {1: 0.0, 8: 1.0, 24: 2.0}
    offsets = dict(zip(methods, np.linspace(-0.18, 0.18, len(methods))))
    axis_low = float(summary["axis_low"].iloc[0])
    axis_high = float(summary["axis_high"].iloc[0])
    ax.axhline(
        0.0,
        color="#6E6E6E",
        linewidth=common._lw(1.2),
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    for method in methods:
        color = common.sim.METHOD_COLOR[method]
        marker = common.sim.METHOD_MARKER[method]
        mean_x, mean_y = [], []
        for row in (
            summary[summary["method"].eq(method)]
            .sort_values("nbins")
            .itertuples(index=False)
        ):
            x = x_lookup[int(row.nbins)] + offsets[method]
            artists = ax.bxp(
                [
                    {
                        "med": float(row.median),
                        "q1": float(row.q25),
                        "q3": float(row.q75),
                        "whislo": float(row.q025),
                        "whishi": float(row.q975),
                        "fliers": [],
                    }
                ],
                positions=[x],
                widths=0.13,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": color,
                    "edgecolor": color,
                    "alpha": 0.22,
                    "linewidth": common._lw(1.4),
                },
                medianprops={"color": color, "linewidth": common._lw(1.6)},
                whiskerprops={"color": color, "linewidth": common._lw(1.5)},
                capprops={"color": color, "linewidth": common._lw(1.5)},
            )
            for element in ("boxes", "medians", "whiskers", "caps"):
                for artist in artists[element]:
                    artist.set_zorder(2)
            ax.scatter(
                x,
                float(row.mean),
                s=common._scatter_size(58),
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=common._lw(0.9),
                zorder=4,
            )
            mean_x.append(x)
            mean_y.append(float(row.mean))
        ax.plot(
            mean_x, mean_y, color=color, linewidth=common._lw(1.5), alpha=0.38, zorder=3
        )

    ax.set_title("EUR (300k)", pad=5)
    ax.set_xlim(-0.49, 2.49)
    ax.set_ylim(axis_low, axis_high)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1", "8", "24"])
    ax.set_xlabel("Fitted bins")
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    common.style_axis(ax)


def draw_panel_c(ax: plt.Axes, summary: pd.DataFrame, metric: str) -> None:
    centers = np.arange(len(v3.POPS), dtype=float)
    method_offsets = dict(zip(v3.SIGNED_METHODS, np.linspace(-0.27, 0.27, 3)))
    profile_offsets = {"common_positive": -0.078, "low_positive": 0.078}
    ax.axhline(
        0.0,
        color="#6E6E6E",
        linewidth=common._lw(1.2),
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    v3.shade_population_bands(ax, centers, 0.5)

    for pop_idx, pop in enumerate(v3.POPS):
        for method in v3.SIGNED_METHODS:
            color = v2.signed_method_color(method)
            marker = v2.signed_method_marker(method)
            xs, ys = [], []
            for profile in v3.SIGNED_PROFILES:
                row = summary[
                    summary["pop"].eq(pop)
                    & summary["method"].eq(method)
                    & summary["profile"].eq(profile)
                ]
                if len(row) != 1:
                    raise RuntimeError(f"Missing Panel-C row {pop}/{method}/{profile}")
                result = row.iloc[0]
                x = centers[pop_idx] + method_offsets[method] + profile_offsets[profile]
                mean = float(result["mean"])
                low = float(result["mean_ci95_low"])
                high = float(result["mean_ci95_high"])
                filled = profile == "low_positive"
                ax.errorbar(
                    [x],
                    [mean],
                    yerr=np.asarray([[mean - low], [high - mean]]),
                    fmt=marker,
                    markersize=common._ms(7.2),
                    linewidth=common._lw(1.8),
                    capsize=common._ms(3.2),
                    color=color,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color,
                    markeredgewidth=common._lw(1.4),
                    zorder=3,
                )
                xs.append(x)
                ys.append(mean)
            ax.plot(
                xs, ys, color=color, linewidth=common._lw(1.4), alpha=0.45, zorder=2
            )

    if metric == "rg":
        ax.set_ylim(-1.75, 1.75)
        ax.set_yticks([-1.5, -0.75, 0.0, 0.75, 1.5])
    else:
        ax.set_ylim(-0.23, 0.14)
        ax.set_yticks([-0.2, -0.1, 0.0, 0.1])
    ax.set_xlim(-0.5, len(v3.POPS) - 0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels(v3.POPS)
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    common.style_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=common._ms(6.4),
            markerfacecolor="white",
            markeredgecolor="#4D4D4D",
            markeredgewidth=common._lw(1.4),
            label="High-MAF positive",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=common._ms(6.4),
            markerfacecolor="#4D4D4D",
            markeredgecolor="#4D4D4D",
            markeredgewidth=common._lw(1.4),
            label="Low-MAF positive",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.1,
        fontsize=common._fs(10.5),
    )


def build_figure(
    metric: str,
    calibration: pd.DataFrame,
    panel_b_summary: pd.DataFrame,
    panel_c_summary: pd.DataFrame,
    panel_d_records: pd.DataFrame,
) -> plt.Figure:
    common.set_style(profile="readable")
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(18.6, 9.5))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.121,
        top=0.885,
        wspace=0.42,
        hspace=0.60,
        height_ratios=[0.95, 1.05],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    draw_panel_a(ax_a, calibration)
    draw_panel_b(ax_b, panel_b_summary, metric)
    draw_panel_c(ax_c, panel_c_summary, metric)

    panel_d_grid = gs[1, :].subgridspec(1, 3, wspace=0.34)
    axes_d = [fig.add_subplot(panel_d_grid[0, idx]) for idx in range(3)]
    panel_d.draw_direct(axes_d, panel_d_records)

    fig.legend(
        handles=common.method_legend_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.35,
        handletextpad=0.5,
    )
    for ax, label in zip([ax_a, ax_b, ax_c, axes_d[0]], "ABCD"):
        common.add_panel_label(fig, ax, label, dx=-0.056, dy=0.010)
    return fig
