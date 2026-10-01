#!/usr/bin/env python3
"""
Compare SUMMIT total h2 estimates from 8-bin and 24-bin MAF-LD annotations.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]

POPS = ["EUR_300k", "EUR", "SAS", "AFR"]
POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "SAS": "SAS",
    "AFR": "AFR",
}
POP_COLOR = {
    "EUR_300k": TAB20[1],
    "EUR": TAB20[0],
    "SAS": TAB20[4],
    "AFR": TAB20[6],
}
POP_MARKER = {
    "EUR_300k": "o",
    "EUR": "s",
    "SAS": "^",
    "AFR": "D",
}
SINGLE_PANEL_ALPHA = 0.66
SEPARATE_PANEL_ALPHA = 0.78
ERRORBAR_ALPHA = 0.24
TICK_STEP = 0.2
TICK_LABEL_PAD = 1.75
SINGLE_PANEL_TICK_LABELSIZE = 11
SEPARATE_PANEL_TICK_LABELSIZE = 9.5


def load_phen_list(path: str) -> list[str]:
    phens = []
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                phens.append(line.split()[0])
    return phens


def build_phen_name_map(df: pd.DataFrame) -> dict[str, str]:
    if "phen_name" not in df.columns:
        return {}

    phen2name = {}
    for phen, group in df[["phen", "phen_name"]].dropna().groupby("phen", sort=False):
        names = [str(x).strip() for x in group["phen_name"]]
        names = [x for x in names if x and x.lower() != "nan"]
        phen2name[str(phen)] = names[0] if names else str(phen)
    return phen2name


def load_paired_estimates(
    outs_base: str,
    csv_name: str,
    phen_list: list[str],
    method: str,
) -> pd.DataFrame:
    rows = []
    phen_keep = set(phen_list)

    for pop in POPS:
        csv_path = os.path.join(outs_base, pop, csv_name)
        if not os.path.exists(csv_path):
            print(f"[warn] missing CSV for {pop}: {csv_path}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)
        required = {"phen", "phen_name", "method", "num_bins", "h2", "h2_se", "window"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{csv_path} missing required columns: {missing}")

        df["phen"] = df["phen"].astype(str)
        df["method"] = df["method"].astype(str)
        df["num_bins"] = pd.to_numeric(df["num_bins"], errors="coerce")
        df["h2"] = pd.to_numeric(df["h2"], errors="coerce")
        df["h2_se"] = pd.to_numeric(df["h2_se"], errors="coerce")

        df = df[
            df["phen"].isin(phen_keep)
            & (df["method"] == method)
            & df["num_bins"].isin([8, 24])
            & df["h2"].notna()
        ].copy()
        if df.empty:
            continue

        df["window_sort"] = pd.to_numeric(df["window"], errors="coerce").fillna(-1)
        df = (
            df.sort_values("window_sort")
            .groupby(["phen", "method", "num_bins"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        phen2name = build_phen_name_map(df)

        pivot = df.pivot_table(
            index="phen", columns="num_bins", values=["h2", "h2_se"], aggfunc="first"
        )
        pivot.columns = [f"{metric}_{int(bins)}bin" for metric, bins in pivot.columns]
        pivot = pivot.reset_index()
        pivot["pop"] = pop
        pivot["phen_name"] = pivot["phen"].map(phen2name).fillna(pivot["phen"])
        rows.append(pivot)

    if not rows:
        raise RuntimeError("No paired SUMMIT rows found.")

    paired = pd.concat(rows, ignore_index=True)
    paired = paired.dropna(subset=["h2_8bin", "h2_24bin"]).copy()
    paired["delta_24_minus_8"] = paired["h2_24bin"] - paired["h2_8bin"]
    paired["abs_delta"] = paired["delta_24_minus_8"].abs()
    return paired


def compute_axis_limits(paired: pd.DataFrame) -> tuple[float, float]:
    x = paired["h2_8bin"].to_numpy(dtype=float)
    y = paired["h2_24bin"].to_numpy(dtype=float)
    xy_min = min(np.nanmin(x), np.nanmin(y))
    xy_max = max(np.nanmax(x), np.nanmax(y))
    pad = 0.04 * (xy_max - xy_min)
    axis_low = np.floor((xy_min - pad) / 0.05) * 0.05
    axis_high = np.ceil((xy_max + pad) / 0.05) * 0.05
    return axis_low, axis_high


def summarize_pairs(paired: pd.DataFrame) -> tuple[int, float, float]:
    x = paired["h2_8bin"].to_numpy(dtype=float)
    y = paired["h2_24bin"].to_numpy(dtype=float)
    pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(paired) > 1 else np.nan
    median_abs_delta = float(np.nanmedian(paired["abs_delta"]))
    return len(paired), pearson_r, median_abs_delta


def print_stats(paired: pd.DataFrame, separate_plots: bool) -> None:
    n_pairs, pearson_r, median_abs_delta = summarize_pairs(paired)
    print(
        f"[stats] all cohorts: n={n_pairs}, "
        f"Pearson r={pearson_r:.3f}, "
        f"median |delta h2|={median_abs_delta:.3f}"
    )

    if not separate_plots:
        return

    for pop in POPS:
        sub = paired[paired["pop"] == pop]
        if sub.empty:
            continue
        n_pairs, pearson_r, median_abs_delta = summarize_pairs(sub)
        print(
            f"[stats] {POP_LABEL[pop]}: n={n_pairs}, "
            f"Pearson r={pearson_r:.3f}, "
            f"median |delta h2|={median_abs_delta:.3f}"
        )


def add_identity_line(ax, axis_low: float, axis_high: float) -> None:
    ax.plot(
        [axis_low, axis_high],
        [axis_low, axis_high],
        color="0.25",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )


def common_ticks(
    axis_low: float, axis_high: float, step: float = TICK_STEP
) -> np.ndarray:
    tick_low = np.ceil(axis_low / step) * step
    tick_high = np.floor(axis_high / step) * step
    return np.arange(tick_low, tick_high + step / 2, step)


def format_axes(ax, axis_low: float, axis_high: float, tick_labelsize: float) -> None:
    ticks = common_ticks(axis_low, axis_high)
    ax.set_xlim(axis_low, axis_high)
    ax.set_ylim(axis_low, axis_high)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(
        axis="both", which="major", labelsize=tick_labelsize, pad=TICK_LABEL_PAD
    )
    ax.set_aspect("equal", adjustable="box")


def add_stats_box(ax, paired: pd.DataFrame, label: str, fontsize: float = 11) -> None:
    n_pairs, pearson_r, median_abs_delta = summarize_pairs(paired)
    ax.text(
        0.04,
        0.96,
        f"$n$ = {n_pairs} {label}\nPearson $r$ = {pearson_r:.3f}\nMedian $|\\Delta h^2|$ = {median_abs_delta:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.78",
            "alpha": 0.92,
        },
    )


def add_errorbars(ax, sub: pd.DataFrame, pop: str) -> None:
    ax.errorbar(
        sub["h2_8bin"],
        sub["h2_24bin"],
        xerr=sub["h2_se_8bin"],
        yerr=sub["h2_se_24bin"],
        fmt="none",
        ecolor=POP_COLOR[pop],
        elinewidth=0.65,
        capsize=0,
        alpha=ERRORBAR_ALPHA,
        zorder=2,
    )


def plot_single_panel(
    paired: pd.DataFrame,
    outfile: str,
    axis_low: float,
    axis_high: float,
    remove_stats: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    add_identity_line(ax, axis_low, axis_high)

    for pop in POPS:
        sub = paired[paired["pop"] == pop]
        if sub.empty:
            continue
        add_errorbars(ax, sub, pop)
        ax.scatter(
            sub["h2_8bin"],
            sub["h2_24bin"],
            s=50,
            marker=POP_MARKER[pop],
            facecolor=POP_COLOR[pop],
            edgecolor="black",
            linewidth=0.45,
            alpha=SINGLE_PANEL_ALPHA,
            label=POP_LABEL[pop],
            zorder=3,
        )

    format_axes(ax, axis_low, axis_high, SINGLE_PANEL_TICK_LABELSIZE)
    ax.set_xlabel("SUMMIT total $h^2$ (8-bin MAF-LD)")
    ax.set_ylabel("SUMMIT total $h^2$ (24-bin MAF-LD)")

    if not remove_stats:
        add_stats_box(ax, paired, "trait-cohort pairs", fontsize=11)

    legend = ax.legend(
        title="Cohort",
        loc="lower right",
        frameon=True,
        fontsize=10.5,
        title_fontsize=11,
        borderpad=0.55,
        handletextpad=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("0.78")
    legend.get_frame().set_linewidth(0.8)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_separate_panels(
    paired: pd.DataFrame,
    outfile: str,
    axis_low: float,
    axis_high: float,
    remove_stats: bool,
) -> None:
    fig, axes = plt.subplots(
        1, len(POPS), figsize=(13.0, 3.9), sharex=True, sharey=True
    )

    for ax, pop in zip(axes, POPS):
        sub = paired[paired["pop"] == pop]
        add_identity_line(ax, axis_low, axis_high)
        if not sub.empty:
            add_errorbars(ax, sub, pop)
        ax.scatter(
            sub["h2_8bin"],
            sub["h2_24bin"],
            s=46,
            marker=POP_MARKER[pop],
            facecolor=POP_COLOR[pop],
            edgecolor="black",
            linewidth=0.45,
            alpha=SEPARATE_PANEL_ALPHA,
            zorder=3,
        )
        ax.set_title(POP_LABEL[pop], fontsize=13, pad=7)
        format_axes(ax, axis_low, axis_high, SEPARATE_PANEL_TICK_LABELSIZE)
        if not remove_stats and not sub.empty:
            add_stats_box(ax, sub, "traits", fontsize=8.6)
        sns.despine(ax=ax)

    fig.tight_layout(w_pad=0.4)
    fig.supxlabel("Total $h^2$ (8-bin MAF-LD)", fontsize=13, y=-0.035)
    fig.supylabel("Total $h^2$ (24-bin MAF-LD)", fontsize=13, x=-0.015)

    fig.savefig(outfile, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outs-base",
        default="data/real_h2",
        help="Base directory with per-pop real-phenotype CSVs.",
    )
    parser.add_argument(
        "--csv-name",
        default="estimates.csv",
        help="Per-pop CSV filename.",
    )
    parser.add_argument(
        "--phen-list",
        default="data/traits/traits.txt",
        help="Phenotype list used for the plotted traits.",
    )
    parser.add_argument(
        "--method",
        default="sumrhe",
        help="Method key for SUMMIT in the parsed CSV.",
    )
    parser.add_argument(
        "--outfile",
        default="figs/real_h2/total_h2_8v24_comparison.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--source-data",
        default=None,
        help="Optional paired source-data TSV path. Defaults to outfile stem + _source_data.tsv.",
    )
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        help="Make a single-row, four-panel figure with one subplot per population.",
    )
    parser.add_argument(
        "--remove-stats",
        action="store_true",
        help="Remove the statistics box from the plot.",
    )
    args = parser.parse_args()

    phen_list = load_phen_list(args.phen_list)
    paired = load_paired_estimates(
        args.outs_base, args.csv_name, phen_list, args.method
    )

    outfile = args.outfile
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    source_data = args.source_data
    if source_data is None:
        root, _ext = os.path.splitext(outfile)
        source_data = f"{root}_source_data.tsv"
    source_dir = os.path.dirname(source_data)
    if source_dir:
        os.makedirs(source_dir, exist_ok=True)

    paired.sort_values(["pop", "phen"]).to_csv(source_data, sep="\t", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    axis_low, axis_high = compute_axis_limits(paired)
    if args.separate_plots:
        plot_separate_panels(paired, outfile, axis_low, axis_high, args.remove_stats)
    else:
        plot_single_panel(paired, outfile, axis_low, axis_high, args.remove_stats)

    print_stats(paired, separate_plots=args.separate_plots)
    print(f"[info] wrote {outfile}")
    print(f"[info] wrote {source_data}")
    print(f"[info] paired trait-cohort rows: {len(paired)}")


if __name__ == "__main__":
    main()
