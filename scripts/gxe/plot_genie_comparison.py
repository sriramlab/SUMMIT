#!/usr/bin/env python3
"""Plot aligned SUMMIT and GENIE component estimates with two-axis SE bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENTS = ("G", "GxE", "NxE")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path, help="Aligned comparison TSV.")
    parser.add_argument("output", type=Path, help="Output path without an extension.")
    parser.add_argument(
        "--title",
        default="Age interaction: SUMMIT versus GENIE",
        help="Figure title.",
    )
    return parser.parse_args()


def _finite_component(table: pd.DataFrame, component: str) -> pd.DataFrame:
    subset = table.loc[table["component"].astype(str) == component].copy()
    numeric = ["summit_estimate", "summit_se", "genie_estimate", "genie_se"]
    for column in numeric:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    keep = np.isfinite(subset[numeric].to_numpy(dtype=np.float64)).all(axis=1)
    subset = subset.loc[keep]
    if subset.empty:
        raise ValueError(f"No finite {component} rows were found.")
    if (subset[["summit_se", "genie_se"]] < 0.0).any(axis=None):
        raise ValueError(f"Negative standard error in {component} rows.")
    return subset


def make_figure(table: pd.DataFrame, title: str) -> plt.Figure:
    required = {
        "trait", "component", "summit_estimate", "summit_se",
        "genie_estimate", "genie_se",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Comparison table is missing columns: {missing}.")

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    color = "#1f6f8b"
    for axis, component in zip(axes, COMPONENTS):
        subset = _finite_component(table, component)
        x = subset["genie_estimate"].to_numpy(dtype=np.float64)
        y = subset["summit_estimate"].to_numpy(dtype=np.float64)
        xerr = subset["genie_se"].to_numpy(dtype=np.float64)
        yerr = subset["summit_se"].to_numpy(dtype=np.float64)
        lower = float(min(np.min(x - xerr), np.min(y - yerr)))
        upper = float(max(np.max(x + xerr), np.max(y + yerr)))
        padding = 0.06 * max(upper - lower, np.finfo(np.float64).eps)
        limits = (lower - padding, upper + padding)

        axis.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            fmt="o", markersize=4.2, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.35,
            ecolor=color, elinewidth=0.65, alpha=0.58, capsize=0,
        )
        axis.plot(limits, limits, color="#333333", linewidth=1.0, linestyle="--")
        axis.axhline(0.0, color="#b7b7b7", linewidth=0.6)
        axis.axvline(0.0, color="#b7b7b7", linewidth=0.6)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        correlation = float(np.corrcoef(x, y)[0, 1])
        axis.set_title(f"{component}  (r = {correlation:.3f})")
        axis.set_xlabel("GENIE estimate")
        axis.grid(color="#e4e4e4", linewidth=0.55, alpha=0.75)
    axes[0].set_ylabel("SUMMIT estimate")
    figure.suptitle(title, fontsize=13)
    return figure


def main() -> None:
    args = _arguments()
    table = pd.read_csv(args.table, sep="\t")
    figure = make_figure(table, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        target = Path(f"{args.output}.{extension}")
        figure.savefig(target, dpi=300 if extension == "png" else None)
    plt.close(figure)


if __name__ == "__main__":
    main()
