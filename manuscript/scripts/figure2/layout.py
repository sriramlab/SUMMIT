"""Plot layout for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


POPS = ("EUR", "SAS", "AFR")


METHODS = ("summit", "covldsc", "sumher_ldak")


def draw_top_panels(
    fig,
    spec_a,
    spec_b,
    coding_discrimination,
    coding_calibration,
    *,
    common,
    panel_source,
):
    holder = type(
        "CodingPanelHolder",
        (),
        {"panel_b": coding_discrimination},
    )()
    axes_a = panel_source.draw_enrichment_recovery(fig, spec_a, holder)
    ax_b = fig.add_subplot(spec_b)
    panel_source.draw_calibration_panel(
        ax_b,
        coding_calibration,
        ylim=(0.0, 0.30),
        ylabel="FPR",
        show_counts=False,
    )
    return axes_a, ax_b


def draw_curve_group(
    fig,
    spec,
    curves: pd.DataFrame,
    metric: str,
    *,
    common,
    show_intervals: bool,
):
    inner = spec.subgridspec(1, 3, wspace=0.22)
    axes = []
    if metric == "fpr":
        fpr_rows = curves[curves["metric"].eq("fpr")]
        fpr_max = float(
            np.nanmax(fpr_rows[["estimate", "ci_hi"]].to_numpy(dtype=float))
        )
        fpr_upper = max(
            0.24,
            np.ceil((fpr_max + 0.01) / 0.05) * 0.05,
        )
        fpr_tick_step = 0.10 if fpr_upper > 0.30 else 0.05
        fpr_ticks = np.arange(
            0.0,
            fpr_upper + 0.5 * fpr_tick_step,
            fpr_tick_step,
        )
    for pop_index, pop in enumerate(POPS):
        ax = fig.add_subplot(inner[0, pop_index])
        axes.append(ax)
        if metric == "fpr":
            ax.plot(
                [0.0, 0.20],
                [0.0, 0.20],
                color="#6E6E6E",
                linewidth=common._lw(1.2),
                linestyle=(0, (4, 2)),
                zorder=1,
            )
        for method in METHODS:
            cell = curves[
                curves["pop"].eq(pop)
                & curves["method"].eq(method)
                & curves["metric"].eq(metric)
            ].sort_values("alpha")
            x = cell["alpha"].to_numpy(dtype=float)
            y = cell["estimate"].to_numpy(dtype=float)
            if show_intervals:
                ax.fill_between(
                    x,
                    cell["ci_lo"].to_numpy(dtype=float),
                    cell["ci_hi"].to_numpy(dtype=float),
                    color=common.sim.METHOD_COLOR[method],
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                x,
                y,
                color=common.sim.METHOD_COLOR[method],
                linewidth=common._lw(1.8),
                marker=common.sim.METHOD_MARKER[method],
                markersize=common._ms(3.8),
                markevery=[0, 5, 10, 15, len(x) - 1],
                markerfacecolor=common.sim.METHOD_COLOR[method],
                markeredgecolor="white",
                markeredgewidth=common._lw(0.65),
                zorder=3,
            )
        ax.axvline(
            0.05,
            color="#A0A0A0",
            linewidth=common._lw(0.9),
            linestyle=(0, (2, 2)),
            zorder=0,
        )
        ax.set_xlim(0.0, 0.20)
        ax.set_xticks(
            [0.0, 0.05, 0.10, 0.20],
            ["0", "0.05", "0.1", "0.2"],
        )
        ax.set_xlabel(r"Nominal $\alpha$")
        ax.set_title(pop, pad=6)
        if metric == "power":
            ax.set_ylim(0.0, 1.04)
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ylabel = "Power"
        else:
            ax.set_ylim(0.0, fpr_upper)
            ax.set_yticks(fpr_ticks)
            ylabel = "FPR"
        if pop_index == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        common.style_axis(ax)
    return axes


def add_legend_and_labels(
    fig,
    axes_a,
    ax_b,
    axes_c,
    axes_d,
    *,
    common,
) -> None:
    fig.legend(
        handles=common.method_legend_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.985),
        columnspacing=1.35,
        handletextpad=0.5,
    )
    common.add_panel_label(fig, axes_a, "A", dx=-0.050, dy=0.010)
    common.add_panel_label(fig, ax_b, "B", dx=-0.060, dy=0.010)
    common.add_panel_label(fig, axes_c, "C", dx=-0.050, dy=0.010)
    common.add_panel_label(fig, axes_d, "D", dx=-0.060, dy=0.010)


def save_figure(fig, output_prefix: Path) -> None:
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_power_and_calibration_curves(
    coding_discrimination,
    coding_calibration,
    curves,
    output_prefix,
    *,
    common,
    panel_source,
) -> None:
    fig = plt.figure(figsize=(15.2, 8.9))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.06],
        hspace=0.58,
        top=0.87,
        bottom=0.10,
        left=0.068,
        right=0.988,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.34)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
    axes_a, ax_b = draw_top_panels(
        fig,
        top[0, 0],
        top[0, 1],
        coding_discrimination,
        coding_calibration,
        common=common,
        panel_source=panel_source,
    )
    axes_c = draw_curve_group(
        fig,
        bottom[0, 0],
        curves,
        "power",
        common=common,
        show_intervals=True,
    )
    axes_d = draw_curve_group(
        fig,
        bottom[0, 1],
        curves,
        "fpr",
        common=common,
        show_intervals=True,
    )
    add_legend_and_labels(fig, axes_a, ax_b, axes_c, axes_d, common=common)
    save_figure(fig, output_prefix)
