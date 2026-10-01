"""Plot mafld for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
import numpy as np
import pandas as pd
import common
import layout as fig1

POPS = ["EUR", "SAS", "AFR"]


ARCHITECTURES = ["GCTA", "LDAK"]


ARCHITECTURE_TITLES = {
    "GCTA": "Equal-variance effects",
    "LDAK": "LDAK-weighted causal sampling",
}


METHODS = ["SUMMIT-GW", "cov-LDSC-W", "SumHer-LDAK"]


METHOD_TO_KEY = {
    "SUMMIT-GW": "covsumrhe",
    "cov-LDSC-W": "covldsc",
    "SumHer-LDAK": "sumher_ldak_20000",
}


METHOD_LABEL = {
    "SUMMIT-GW": "SUMMIT",
    "cov-LDSC-W": "cov-LDSC",
    "SumHer-LDAK": "SumHer-LDAK",
}


def draw_panel_c_median(
    summary: pd.DataFrame,
):
    """Create the Panel C drawing closure used by the established layout."""

    def draw(fig: plt.Figure, spec) -> list[plt.Axes]:
        inner = spec.subgridspec(1, 2, wspace=0.26)
        axes = [fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])]
        centers = np.arange(len(POPS), dtype=float)
        offsets = np.linspace(-0.22, 0.22, len(METHODS))
        finite = summary["q75_rel_mse"].to_numpy(dtype=float)
        ymax = max(1000.0, float(np.nanmax(finite)) * 1.4)
        ylim = (0.65, ymax)

        for ax, architecture in zip(axes, ARCHITECTURES):
            cell = summary[summary["architecture"].eq(architecture)]
            for method_index, method in enumerate(METHODS):
                part = cell[cell["method"].eq(method)].set_index("pop").loc[POPS]
                medians: list[float] = []
                for pop_index, pop in enumerate(POPS):
                    row = part.loc[pop]
                    xpos = centers[pop_index] + offsets[method_index]
                    median = float(row["median_rel_mse"])
                    q25 = float(row["q25_rel_mse"])
                    q75 = float(row["q75_rel_mse"])
                    medians.append(median)
                    if method != "SUMMIT-GW" and q25 != q75:
                        color = fig1.point_color(METHOD_TO_KEY[method])
                        ax.vlines(
                            xpos,
                            q25,
                            q75,
                            color=color,
                            linewidth=fig1.ERROR_BAR_LW,
                            zorder=3,
                        )
                        ax.hlines(
                            [q25, q75],
                            xpos - fig1.ERROR_CAP_HALF_WIDTH,
                            xpos + fig1.ERROR_CAP_HALF_WIDTH,
                            color=color,
                            linewidth=fig1.ERROR_BAR_LW,
                            zorder=3,
                        )
                    fig1.scatter_main_point(
                        ax,
                        xpos,
                        median,
                        METHOD_TO_KEY[method],
                        zorder=4,
                    )
                ax.plot(
                    centers + offsets[method_index],
                    medians,
                    color=fig1.point_color(METHOD_TO_KEY[method]),
                    linewidth=1.5,
                    alpha=0.35,
                    zorder=2,
                )

            fig1.style_log_ratio_axis(ax, ylim)
            ax.set_title(ARCHITECTURE_TITLES[architecture], pad=6)
            ax.set_xticks(centers)
            ax.set_xticklabels(POPS)
            ax.set_xlim(-0.5, len(POPS) - 0.5)

        axes[0].set_ylabel("Relative MSE\nvs SUMMIT")
        axes[1].tick_params(axis="y", which="both", left=True, labelleft=False)
        axes[1].spines["left"].set_visible(True)
        return axes

    return draw


def draw_panel_d_histogram(source: pd.DataFrame):
    """Create population-faceted setting-level histograms."""
    positive = source.loc[
        source["absolute_relative_bias"] > 0, "absolute_relative_bias"
    ]
    lower = 10 ** np.floor(np.log10(float(positive.min())))
    upper = 10 ** np.ceil(np.log10(float(positive.max())))
    bins = np.geomspace(lower, upper, 13)
    tick_candidates = [
        (0.0001, r"$10^{-4}$"),
        (0.01, r"$10^{-2}$"),
        (1.0, r"$1$"),
        (100.0, r"$10^{2}$"),
    ]
    ticks = [tick for tick, _ in tick_candidates if lower <= tick <= upper]
    tick_labels = [label for tick, label in tick_candidates if lower <= tick <= upper]

    def draw(fig: plt.Figure, spec) -> list[plt.Axes]:
        inner = spec.subgridspec(1, 3, wspace=0.22)
        axes = [fig.add_subplot(inner[0, index]) for index in range(len(POPS))]

        for ax, pop in zip(axes, POPS):
            cell = source[source["pop"].eq(pop)]
            for method in METHODS:
                values = cell.loc[
                    cell["method"].eq(method), "absolute_relative_bias"
                ].to_numpy(dtype=float)
                if len(values) != 8:
                    raise RuntimeError(
                        f"Expected eight Panel D values for {pop}/{method}"
                    )
                color = fig1.point_color(METHOD_TO_KEY[method])
                ax.hist(
                    values,
                    bins=bins,
                    histtype="stepfilled",
                    color=color,
                    alpha=0.08,
                )
                ax.hist(
                    values,
                    bins=bins,
                    histtype="step",
                    color=color,
                    linewidth=2.4,
                    label=METHOD_LABEL[method],
                )

            ax.set_xscale("log")
            ax.set_xlim(lower, upper)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.set_xticklabels(tick_labels)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(0, 8.6)
            ax.set_yticks([0, 2, 4, 6, 8])
            ax.set_title(pop, pad=6)
            ax.grid(axis="y")
            ax.grid(axis="x", visible=False)
            common.style_axis(ax)

        axes[0].set_ylabel("Number of settings")
        axes[1].set_xlabel("Absolute relative bias")
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)
        return axes

    return draw
