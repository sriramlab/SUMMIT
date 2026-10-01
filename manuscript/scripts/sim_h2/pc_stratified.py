#!/usr/bin/env python3
"""Main-text style relMSE/error figure for the PC-stratified simulation."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


TAB20 = list(plt.get_cmap("tab20").colors)

METHOD_ORDER = ["covsumrhe", "sumrhe", "covldsc", "ldsc", "sumher_gcta", "sumher_ldak"]

PRETTY_LABEL = {
    "covsumrhe": "SUMMIT",
    "sumrhe": "SUMMIT\n(no PC)",
    "covldsc": "cov-LDSC",
    "ldsc": "LDSC",
    "sumher_gcta": "SumHer\n(GCTA)",
    "sumher_ldak": "SumHer\n(LDAK)",
}

METHOD_COLOR = {
    "covsumrhe": TAB20[4],
    "sumrhe": TAB20[5],
    "covldsc": TAB20[2],
    "ldsc": TAB20[10],
    "sumher_gcta": TAB20[6],
    "sumher_ldak": TAB20[8],
}

TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
XTICK_LABEL_FONTSIZE = 12
YTICK_LABEL_FONTSIZE = 12


def stacked_figsize(n_cols: int, n_methods: int) -> tuple[float, float]:
    """Compact supplemental figure size; keeps text readable after LaTeX scaling."""
    col_width = 5.3 if n_methods >= 6 else 3.8
    return (max(10.5, col_width * n_cols), 7.4)


def xtick_labelsize(n_methods: int) -> float:
    if n_methods >= 7:
        return 8.5
    if n_methods >= 6:
        return 9.5
    return XTICK_LABEL_FONTSIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/sim_h2/pc_stratified.tsv",
    )
    parser.add_argument(
        "--out-prefix",
        default="figs/supplementary/fig_s10_pc_stratified_simulations",
    )
    parser.add_argument("--true-h2", type=float, default=0.25)
    parser.add_argument("--base", default="covsumrhe")
    parser.add_argument("--pops", default="EUR,SAS,AFR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yscale", choices=["auto", "linear", "log"], default="auto")
    return parser.parse_args()


def color_from_key(method: str):
    return METHOD_COLOR.get(method, TAB20[sum(method.encode("utf-8")) % len(TAB20)])


def label_from_key(method: str) -> str:
    return PRETTY_LABEL.get(method, method)


def auto_yscale(vals: np.ndarray) -> str:
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return "linear"
    if float(np.nanmax(vals)) > 3.0 or float(np.nanmin(vals)) < 0.33:
        return "log"
    return "linear"


def set_log_ylim_with_pad(ax, vals: np.ndarray):
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        ax.set_ylim(0.5, 2.0)
        return
    lo = min(float(np.nanmin(vals)), 1.0 / 1.5)
    hi = max(float(np.nanmax(vals)), 1.5)
    ax.set_ylim(max(lo / 1.3, 1e-8), hi * 1.3)


def set_linear_ylim_with_pad(ax, vals: np.ndarray):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ax.set_ylim(0.5, 2.0)
        return
    lo = min(float(np.nanmin(vals)), 1.0)
    hi = max(float(np.nanmax(vals)), 1.0)
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.15
    ax.set_ylim(lo - pad, hi + pad)


def robust_error_ylim(err: np.ndarray) -> tuple[float, float]:
    err = err[np.isfinite(err)]
    if err.size == 0:
        return (-1.0, 1.0)

    q1 = float(np.nanpercentile(err, 25))
    q3 = float(np.nanpercentile(err, 75))
    iqr = q3 - q1
    if iqr <= 0 or not np.isfinite(iqr):
        lo = float(np.nanmin(err))
        hi = float(np.nanmax(err))
    else:
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
        lo = (
            float(np.nanmin(err[err >= fence_lo]))
            if np.any(err >= fence_lo)
            else float(np.nanmin(err))
        )
        hi = (
            float(np.nanmax(err[err <= fence_hi]))
            if np.any(err <= fence_hi)
            else float(np.nanmax(err))
        )

    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    span = hi - lo
    pad = 0.15 * span if span > 0 else 0.25
    return (lo - pad, hi + pad)


def robust_error_ylim_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "method",
    group_order: list[str] | None = None,
) -> tuple[float, float]:
    lows, highs = [], []
    groups = (
        group_order
        if group_order is not None
        else sorted(df[group_col].dropna().unique().tolist())
    )

    for group in groups:
        vals = pd.to_numeric(
            df.loc[df[group_col] == group, value_col], errors="coerce"
        ).to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        q1 = float(np.nanpercentile(vals, 25))
        q3 = float(np.nanpercentile(vals, 75))
        iqr = q3 - q1
        if iqr <= 0 or not np.isfinite(iqr):
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
        else:
            fence_lo = q1 - 1.5 * iqr
            fence_hi = q3 + 1.5 * iqr
            lo = (
                float(np.nanmin(vals[vals >= fence_lo]))
                if np.any(vals >= fence_lo)
                else float(np.nanmin(vals))
            )
            hi = (
                float(np.nanmax(vals[vals <= fence_hi]))
                if np.any(vals <= fence_hi)
                else float(np.nanmax(vals))
            )

        lows.extend([lo, q1, 0.0])
        highs.extend([hi, q3, 0.0])

    if not lows or not highs:
        return (-1.0, 1.0)

    lo = float(np.nanmin(lows))
    hi = float(np.nanmax(highs))
    span = hi - lo
    pad = 0.22 * span if span > 0 else 0.25
    return (lo - pad, hi + pad)


def compute_relmse(
    df: pd.DataFrame, methods: list[str], base: str, true_h2: float
) -> pd.DataFrame:
    rows = []
    for pop, pop_df in df.groupby("pop", sort=False):
        mse_by_method = {}
        for method in methods:
            est = pd.to_numeric(
                pop_df.loc[pop_df["method"] == method, "estimate"], errors="coerce"
            ).to_numpy(float)
            est = est[np.isfinite(est)]
            if est.size == 0:
                mse_by_method[method] = np.nan
            else:
                mse_by_method[method] = float(np.mean((est - true_h2) ** 2))

        base_mse = mse_by_method.get(base, np.nan)
        for method in methods:
            mse = mse_by_method.get(method, np.nan)
            rel_mse = np.nan
            if np.isfinite(mse) and np.isfinite(base_mse) and base_mse > 0:
                rel_mse = mse / base_mse
            rows.append(dict(pop=pop, method=method, mse=mse, rel_mse=rel_mse))

    return pd.DataFrame(rows)


def plot_relmse_panel(ax, rel: pd.DataFrame, pop: str, methods: list[str], yscale: str):
    sub = rel[rel["pop"] == pop].copy()
    rel_vals = pd.to_numeric(sub["rel_mse"], errors="coerce").to_numpy(float)
    x = np.arange(len(methods), dtype=float)

    for i, method in enumerate(methods):
        val = sub.loc[sub["method"] == method, "rel_mse"]
        if val.empty or not np.isfinite(float(val.iloc[0])):
            continue
        ax.scatter(
            [x[i]],
            [float(val.iloc[0])],
            s=90,
            marker="D",
            color=color_from_key(method),
            zorder=4,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label_from_key(m) for m in methods], rotation=0, ha="center")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=xtick_labelsize(len(methods)))
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)

    for tick in ax.get_xticklabels():
        if tick.get_text() == label_from_key("covsumrhe"):
            tick.set_fontweight("bold")

    if yscale == "auto":
        panel_scale = auto_yscale(rel_vals)
    else:
        panel_scale = yscale

    ax.set_yscale(panel_scale)
    if panel_scale == "log":
        set_log_ylim_with_pad(ax, rel_vals)
    else:
        set_linear_ylim_with_pad(ax, rel_vals)


def plot_error_panel(
    ax, df: pd.DataFrame, pop: str, methods: list[str], true_h2: float
):
    sub = df[df["pop"] == pop].copy()
    sub = sub[sub["method"].isin(methods)].copy()
    sub["err"] = pd.to_numeric(sub["estimate"], errors="coerce") - true_h2
    sub["pretty"] = sub["method"].map(label_from_key)

    order_pretty = [label_from_key(m) for m in methods]
    palette = {label_from_key(m): color_from_key(m) for m in methods}
    sns.boxplot(
        data=sub,
        x="pretty",
        y="err",
        order=order_pretty,
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1.2,
    )

    x = np.arange(len(methods), dtype=float)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.8)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(order_pretty, rotation=0, ha="center")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=xtick_labelsize(len(methods)))
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)
    for tick in ax.get_xticklabels():
        if tick.get_text() == label_from_key("covsumrhe"):
            tick.set_fontweight("bold")

    ax.set_ylim(*robust_error_ylim_by_group(sub, "err", group_order=methods))


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, sep="\t")
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["method"] = df["method"].astype(str)
    df["pop"] = df["pop"].astype(str)

    pops = [p.strip() for p in args.pops.split(",") if p.strip()]
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    if args.base not in methods:
        raise SystemExit(
            f"ERROR: baseline method '{args.base}' is absent from {source}"
        )
    methods = [args.base] + [m for m in methods if m != args.base]

    missing = sorted(set(pops) - set(df["pop"]))
    if missing:
        raise SystemExit(f"ERROR: source data missing populations: {missing}")

    rel = compute_relmse(df, methods, args.base, args.true_h2)
    rel.to_csv(f"{out_prefix}_relmse.tsv", sep="\t", index=False)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        2, len(pops), figsize=stacked_figsize(len(pops), len(methods)), sharey=False
    )
    if len(pops) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, pop in enumerate(pops):
        ax = axes[0, col]
        plot_relmse_panel(ax, rel, pop, methods, args.yscale)
        ax.set_title(pop, fontsize=TITLE_FONTSIZE)
        if col == 0:
            ax.set_ylabel("Relative MSE", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")

    for col, pop in enumerate(pops):
        ax = axes[1, col]
        plot_error_panel(ax, df, pop, methods, args.true_h2)
        if col == 0:
            ax.set_ylabel("Error (estimate - true $h^2$)", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        ax.set_title("")

    fig.tight_layout()
    pdf = f"{out_prefix}.pdf"
    png = f"{out_prefix}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"[write] {out_prefix}_relmse.tsv")
    print(f"[write] {pdf}")
    print(f"[write] {png}")


if __name__ == "__main__":
    main()
