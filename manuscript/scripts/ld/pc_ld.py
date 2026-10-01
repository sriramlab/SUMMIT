"""Plot pc ld for the SUMMIT manuscript."""
from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
from pathlib import Path
import matplotlib.pyplot as plt


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]


POPS = ["EUR_300k", "EUR", "SAS", "AFR"]


EUR_POPS = ["EUR_300k", "EUR"]


LOG_POPS = {"SAS", "AFR"}


BROKEN_POPS = LOG_POPS


ROW_FIGSIZE = (6.78, 2.23)


POP_COLOR = {
    "EUR": TAB20[0],
    "EUR_300k": TAB20[1],
    "AFR": TAB20[6],
    "SAS": TAB20[4],
}


POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "AFR": "AFR",
    "SAS": "SAS",
}


def resolve_png_path(fig_path: Path) -> Path:
    return fig_path.with_suffix(".png")


def set_row_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
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
        }
    )


def style_axis(
    ax: plt.Axes,
    x_values: list[int],
    *,
    grid_linewidth: float = 0.8,
    tick_labelsize: float | None = 11,
) -> None:
    ax.set_xlim(min(x_values) - 2, max(x_values) + 2)
    ax.set_xticks(x_values)
    ax.grid(axis="y", color="#d9d9d9", linewidth=grid_linewidth)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", labelsize=tick_labelsize)


def padded_limits(values: pd.Series, yscale: str) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot set y-limits from an empty series")

    y_min = float(clean.min())
    y_max = float(clean.max())

    if yscale == "log":
        if (clean <= 0).any():
            raise ValueError(
                "--yscale log requires all plotted mean LD scores to be > 0"
            )
        if y_min == y_max:
            return y_min / 1.12, y_max * 1.12
        return y_min / 1.08, y_max * 1.08

    if y_min == y_max:
        pad = max(abs(y_max) * 0.08, 0.1)
    else:
        pad = (y_max - y_min) * 0.08
    lower = y_min - pad
    if y_min >= 0:
        lower = max(0.0, lower)
    return lower, y_max + pad


def apply_yscale(ax: plt.Axes, yscale: str) -> None:
    if yscale == "log":
        ax.set_yscale("log")


def yscale_for_pop(pop: str, yscale: str) -> str:
    if yscale == "mixed":
        return "log" if pop in LOG_POPS else "linear"
    return yscale


def format_log_axis(ax: plt.Axes) -> None:
    y_min, y_max = ax.get_ylim()
    if y_min <= 0:
        return

    if y_max / y_min < 20:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    ax.yaxis.set_minor_formatter(NullFormatter())


def set_shared_y_limits(
    axes: list[plt.Axes],
    values: pd.Series,
    yscale: str,
) -> None:
    y_min, y_max = padded_limits(values, yscale)
    for ax in axes:
        ax.set_ylim(y_min, y_max)


def add_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    kwargs = dict(color="#606060", clip_on=False, linewidth=1.1)
    diag = 0.012
    ax_top.plot((-diag, diag), (-diag, diag), transform=ax_top.transAxes, **kwargs)
    ax_top.plot(
        (1 - diag, 1 + diag), (-diag, diag), transform=ax_top.transAxes, **kwargs
    )
    ax_bottom.plot(
        (-diag, diag), (1 - diag, 1 + diag), transform=ax_bottom.transAxes, **kwargs
    )
    ax_bottom.plot(
        (1 - diag, 1 + diag),
        (1 - diag, 1 + diag),
        transform=ax_bottom.transAxes,
        **kwargs,
    )


def plot_pop_line(
    ax: plt.Axes,
    pop_df: pd.DataFrame,
    pop: str,
    *,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.4,
    markersize: float = 6.0,
) -> None:
    ax.plot(
        pop_df["num_pcs"],
        pop_df["mean_ldscore"],
        marker="o",
        color=POP_COLOR[pop],
        linewidth=linewidth,
        markersize=markersize,
        label=label,
        linestyle=linestyle,
        zorder=3,
    )


def plot_four_panels(df: pd.DataFrame, out_path: Path, yscale: str) -> None:
    if yscale == "broken":
        plot_four_panels_broken(df, out_path)
        return

    set_row_style()
    fig, axes = plt.subplots(
        1,
        len(POPS),
        figsize=ROW_FIGSIZE,
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    x_values = sorted(df["num_pcs"].unique())
    axes_by_pop = {}

    for ax, pop in zip(axes.ravel(), POPS):
        pop_df = df[df["pop"] == pop].sort_values("num_pcs")
        if pop_df.empty:
            ax.set_visible(False)
            continue

        plot_pop_line(ax, pop_df, pop, linewidth=1.15, markersize=3.0)
        ax.set_title(POP_LABEL[pop], pad=3, color=POP_COLOR[pop])
        style_axis(ax, x_values, grid_linewidth=0.45, tick_labelsize=None)
        panel_yscale = yscale_for_pop(pop, yscale)
        apply_yscale(ax, panel_yscale)
        ax.set_ylim(*padded_limits(pop_df["mean_ldscore"], panel_yscale))
        axes_by_pop[pop] = ax

    eur_axes = [axes_by_pop[pop] for pop in EUR_POPS if pop in axes_by_pop]
    eur_values = df[df["pop"].isin(EUR_POPS)]["mean_ldscore"]
    if eur_axes and not eur_values.empty:
        set_shared_y_limits(eur_axes, eur_values, yscale_for_pop("EUR", yscale))
    for pop, ax in axes_by_pop.items():
        if yscale_for_pop(pop, yscale) == "log":
            format_log_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=POP_COLOR[pop],
            marker="o",
            linewidth=1.15,
            markersize=3.2,
            label=POP_LABEL[pop],
        )
        for pop in POPS
        if not df[df["pop"] == pop].empty
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.8,
        columnspacing=1.15,
    )

    fig.supxlabel("Number of genotype PCs", y=0.045)
    fig.supylabel("Mean total LD score", x=0.055)
    fig.tight_layout(rect=[0.04, 0.07, 1, 0.89], w_pad=1.0)
    save_figure(fig, out_path)


def plot_four_panels_broken(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.11,
        right=0.98,
        bottom=0.12,
        top=0.86,
        hspace=0.52,
        wspace=0.28,
    )
    x_values = sorted(df["num_pcs"].unique())
    axes_by_pop: dict[str, plt.Axes] = {}

    for idx, pop in enumerate(POPS):
        row = idx // 2
        col = idx % 2
        pop_df = df[df["pop"] == pop].sort_values("num_pcs")
        if pop_df.empty:
            continue

        if pop in BROKEN_POPS:
            sub = outer[row, col].subgridspec(
                2,
                1,
                height_ratios=[0.9, 2.6],
                hspace=0.06,
            )
            ax_top = fig.add_subplot(sub[0])
            ax_bottom = fig.add_subplot(sub[1], sharex=ax_top)
            high_df = pop_df[pop_df["num_pcs"] == 0]
            low_df = pop_df[pop_df["num_pcs"] != 0]

            if not high_df.empty:
                plot_pop_line(ax_top, high_df, pop, linestyle="None")
                ax_top.set_ylim(*padded_limits(high_df["mean_ldscore"], "linear"))
            if not low_df.empty:
                plot_pop_line(ax_bottom, low_df, pop)
                ax_bottom.set_ylim(*padded_limits(low_df["mean_ldscore"], "linear"))

            style_axis(ax_top, x_values)
            style_axis(ax_bottom, x_values)
            ax_top.spines["bottom"].set_visible(False)
            ax_bottom.spines["top"].set_visible(False)
            ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
            ax_bottom.tick_params(axis="x", top=False)
            ax_top.set_title(POP_LABEL[pop], fontsize=14, pad=8, color=POP_COLOR[pop])
            add_break_marks(ax_top, ax_bottom)
            axes_by_pop[pop] = ax_bottom
            continue

        ax = fig.add_subplot(outer[row, col])
        plot_pop_line(ax, pop_df, pop)
        ax.set_title(POP_LABEL[pop], fontsize=14, pad=8, color=POP_COLOR[pop])
        style_axis(ax, x_values)
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)
        axes_by_pop[pop] = ax

    eur_axes = [axes_by_pop[pop] for pop in EUR_POPS if pop in axes_by_pop]
    eur_values = df[df["pop"].isin(EUR_POPS)]["mean_ldscore"]
    if eur_axes and not eur_values.empty:
        set_shared_y_limits(eur_axes, eur_values, "linear")

    handles = [
        Line2D(
            [0],
            [0],
            color=POP_COLOR[pop],
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=POP_LABEL[pop],
        )
        for pop in POPS
        if not df[df["pop"] == pop].empty
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.2,
        columnspacing=1.4,
        fontsize=11,
    )

    fig.supxlabel("Number of genotype PCs", fontsize=13, y=0.07)
    fig.supylabel("Mean total LD score", fontsize=13, x=0.065)
    save_figure(fig, out_path)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = resolve_png_path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    print(f"Wrote figure to {out_path}")
    print(f"Wrote figure to {png_path}")


if __name__ == "__main__":
    plot_four_panels(
        pd.read_csv("data/ld/pc_ld_summary.csv"),
        Path("figs/supplementary/fig_s03_pc_sensitivity_ld.pdf"),
        "mixed",
    )
