"""Plot runtime scaling for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


X_COL: Dict[str, Tuple[str, str]] = {
    "N": ("n", "Samples (N)"),
    "M": ("m", "Variants (M)"),
    "K": ("k", "Annotation bins (K)"),
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def pretty_number(x: float, _pos=None) -> str:
    if not np.isfinite(x) or x <= 0:
        return ""
    if x >= 1000:
        return f"{int(round(x)):,}"
    if abs(x - round(x)) < 1e-8:
        return f"{int(round(x))}"
    return f"{x:g}"


def format_axis(ax, *, log_x: bool, log_y: bool, x_values=None) -> None:
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=9)
        )
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pretty_number))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7, integer=True))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pretty_number))

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=9)
        )
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pretty_number))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pretty_number))


def prep_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "n",
        "m",
        "k",
        "numvec",
        "n_runs",
        "elapsed_mean",
        "elapsed_median",
        "elapsed_sd",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_runtime(
    runtime: pd.DataFrame, outdir: Path, dpi: int, share_y: bool = False
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), squeeze=False, sharey=share_y)
    for idx, (ax, exp) in enumerate(zip(axes[0], ["N", "M", "K"])):
        work = runtime.loc[runtime["experiment"].eq(exp)].copy()
        if work.empty:
            ax.set_visible(False)
            continue
        x_col, x_label = X_COL[exp]
        work = work.sort_values(x_col)
        x = work[x_col].to_numpy(dtype=float)
        y = work["elapsed_median"].to_numpy(dtype=float)
        yerr = work["elapsed_sd"].fillna(0).to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=5.0,
            linewidth=1.9,
            capsize=3.2,
            color="#2f6f9f",
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        log_axes = exp in {"N", "M"}
        format_axis(ax, log_x=log_axes, log_y=log_axes, x_values=x)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Elapsed seconds" if idx == 0 else "")
        ax.set_title(f"Runtime scaling with {exp}")
        ax.grid(True, which="major", alpha=0.32, linewidth=0.65)
        ax.grid(True, which="minor", alpha=0.14, linewidth=0.45)
    fig.tight_layout()

    fig.savefig(outdir / "fig_s23_runtime_scaling.png", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    plot_runtime(
        prep_summary(pd.read_csv("data/ld/runtime_scaling.csv")),
        Path("figs/supplementary"),
        300,
    )
