#!/usr/bin/env python3
"""
plot_total_h2_maintext.py

Main-text style total h2 plot:
- Each row = method
- X-axis = trait (phenotype)
- Within each method row, bars are grouped by population (fixed pop colors)
- Default: plot only 24-bin results (less crowded)
- If --bins includes both 8 and 24, generate two separate PDF files (one per bins)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import colormaps
from matplotlib.colors import to_hex


# ------------------ Colors ------------------
TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]

POPS = ["EUR_300k", "EUR", "SAS", "AFR"]

# Fixed colors for populations (EUR and EUR_300k visually similar but distinct)
POP_COLOR = {
    "EUR": TAB20[0],  # blue
    "EUR_300k": TAB20[1],  # lighter blue (paired in tab20)
    "AFR": TAB20[4],  # green-ish
    "SAS": TAB20[6],  # red-ish / orange-ish
}

POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "AFR": "AFR",
    "SAS": "SAS",
}

BASE_AXES_LEFT = 0.03396
BASE_AXES_RIGHT = 0.83400
BASE_AXES_BOTTOM = 0.06926
BASE_AXES_TOP = 0.96895
BASE_AXES_HSPACE = 0.23300
BASE_LEGEND_X = 0.84000
LEGEND_WIDTH_IN = 1.99
RIGHT_PAD_IN = 1.00
METHOD_TITLE_FONTSIZE = 24
Y_LABEL_FONTSIZE = 22

METHOD_LABEL = {
    "sumrhe": "SUMMIT",
    "sumher": "SumHer-GCTA",
    "sumher_ldak": "SumHer-LDAK",
    "covldsc": "cov-LDSC",
    "ldsc": "LDSC",
}


def load_phen_list(path: str):
    phens = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            phens.append(parts[0])
    return phens


def _build_phen_clean_map(df: pd.DataFrame) -> dict:
    if df.empty or "phen" not in df.columns:
        return {}
    if "phen_name" not in df.columns:
        return {p: p for p in df["phen"].astype(str).unique()}

    tmp = df[["phen", "phen_name"]].copy()
    tmp["phen"] = tmp["phen"].astype(str)
    tmp["phen_name"] = tmp["phen_name"].astype(str)

    phen2clean = {}
    for phen, g in tmp.groupby("phen", sort=False):
        vals = [v.strip() for v in g["phen_name"].tolist() if isinstance(v, str)]
        vals = [v for v in vals if v and v.lower() != "nan"]
        phen2clean[phen] = vals[0] if vals else phen
    return phen2clean


def _resolve_outfile(outfile: str, bins_val: int, multi_bins: bool) -> str:
    """
    If user provides {bins} placeholder, use it.
    Else if multi-bins, append _{bins}bins before extension.
    Else use as-is.
    """
    if "{bins}" in outfile:
        return outfile.format(bins=bins_val)

    if multi_bins:
        root, ext = os.path.splitext(outfile)
        if not ext:
            ext = ".pdf"
        return f"{root}_{bins_val}bins{ext}"

    # single bins => keep the given path
    return outfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outs-base",
        default="data/real_h2",
        help="Base directory with per-pop MAF-LD CSVs (default: data/real_h2)",
    )
    parser.add_argument(
        "--csv-name",
        default="estimates.csv",
        help="Per-pop CSV filename (default: estimates.csv)",
    )
    parser.add_argument(
        "--phen-list",
        default="phen_list/traits.txt",
        help="Path to phen_list.txt (default: phen_list/traits.txt)",
    )
    parser.add_argument(
        "--bins",
        default="24",
        help="Comma-separated bins to include (default: 24). If multiple (e.g. 8,24), makes separate PDFs.",
    )
    parser.add_argument(
        "--phen-order-bins",
        type=int,
        default=None,
        help="If set, order phenotypes using this num_bins value for every output figure.",
    )
    parser.add_argument(
        "--outfile",
        default="mainfigs/total_h2_maintext_{bins}bins.pdf",
        help="Output figure path. If multiple --bins, either include {bins} or we append _{bins}bins (default: figs/total_h2_maintext_{bins}bins.pdf)",
    )
    args = parser.parse_args()

    outs_base = args.outs_base
    csv_name = args.csv_name
    phen_list_path = args.phen_list
    outfile = args.outfile
    phen_order_bins = args.phen_order_bins

    try:
        bins_keep = [int(x) for x in args.bins.split(",") if x.strip()]
    except Exception:
        raise RuntimeError(
            f"Could not parse --bins '{args.bins}' (expected like 8 or 24 or 8,24)"
        )
    if not bins_keep:
        raise RuntimeError("No bins provided via --bins")

    bins_keep = sorted(set(bins_keep), key=lambda b: (b != 24, b))  # prefer 24 first
    multi_bins = len(bins_keep) > 1
    bins_needed = set(bins_keep)
    if phen_order_bins is not None:
        bins_needed.add(phen_order_bins)

    # phenotype ordering base
    phen_list = load_phen_list(phen_list_path)
    phen_list = [
        p for p in phen_list if p != "fev1_fvc" and p != "forced_expiratory_volume_1s"
    ]

    # read CSVs
    dfs = []
    for pop in POPS:
        csv_path = os.path.join(outs_base, pop, csv_name)
        if not os.path.exists(csv_path):
            print(f"[warn] missing CSV for {pop}: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        df["pop"] = pop
        if "phen_name" not in df.columns:
            df["phen_name"] = df["phen"]
        dfs.append(df)

    if not dfs:
        print("[error] no CSVs found for any population", file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)

    # normalize types
    df_all["phen"] = df_all["phen"].astype(str)
    df_all["phen_name"] = df_all["phen_name"].astype(str)
    df_all["method"] = df_all["method"].astype(str)
    df_all["pop"] = df_all["pop"].astype(str)

    # required columns
    for c in ["pop", "phen", "method", "num_bins", "h2", "h2_se", "window"]:
        if c not in df_all.columns:
            raise RuntimeError(
                f"CSV missing required column '{c}'. Found: {list(df_all.columns)}"
            )

    # filter phen + bins (bins further filtered per-figure)
    df_all = df_all[df_all["phen"].isin(phen_list)].copy()
    df_all = df_all[
        (df_all["phen"] != "fev1_fvc")
        & (df_all["phen"] != "forced_expiratory_volume_1s")
    ].copy()
    df_all["num_bins"] = pd.to_numeric(df_all["num_bins"], errors="coerce")
    df_all = df_all[df_all["num_bins"].isin(bins_needed)].copy()

    # phen -> clean map
    phen2clean = _build_phen_clean_map(df_all)

    # pick largest window per (pop, phen, method, num_bins)
    df_all["window_sort"] = pd.to_numeric(df_all["window"], errors="coerce").fillna(-1)
    df_all_sorted = df_all.sort_values("window_sort")
    df_sel = (
        df_all_sorted.groupby(["pop", "phen", "method", "num_bins"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if (
        phen_order_bins is not None
        and df_sel[df_sel["num_bins"] == phen_order_bins].empty
    ):
        raise RuntimeError(f"No rows found for --phen-order-bins={phen_order_bins}")

    # y-axis (UNCHANGED)
    df_sel["h2_num_all"] = pd.to_numeric(df_sel["h2"], errors="coerce")
    y_high = 1.5
    y_low = -0.30
    y_ticks = np.arange(-0.25, y_high + 1e-8, 0.25)

    # plotting style (keep consistent)
    sns.set_theme(style="white", context="talk")

    # ensure output directory exists
    # (done per-figure below once we resolve each outfile)

    # method ordering preference
    base_preferred = ["sumrhe", "covldsc", "ldsc", "sumher", "sumher_ldak"]

    for bins_val in bins_keep:
        df_bins = df_sel[df_sel["num_bins"] == bins_val].copy()
        if df_bins.empty:
            print(f"[warn] no rows for num_bins={bins_val}; skipping", file=sys.stderr)
            continue

        # populations present (fixed order)
        pops_present = [p for p in POPS if p in set(df_bins["pop"].unique())]
        if not pops_present:
            print(
                f"[warn] no populations present for num_bins={bins_val}; skipping",
                file=sys.stderr,
            )
            continue

        # methods present (preferred order + any extras)
        methods_present = [
            m for m in base_preferred if m in set(df_bins["method"].unique())
        ] + [m for m in sorted(df_bins["method"].unique()) if m not in base_preferred]
        if not methods_present:
            print(
                f"[warn] no methods present for num_bins={bins_val}; skipping",
                file=sys.stderr,
            )
            continue

        # phenotype order (DESC by EUR_300k SUMMIT; optionally from a fixed bins value)
        order_bins_val = phen_order_bins if phen_order_bins is not None else bins_val
        df_order = df_sel[df_sel["num_bins"] == order_bins_val].copy()
        if df_order.empty:
            raise RuntimeError(
                f"No rows found for phenotype ordering with num_bins={order_bins_val}"
            )

        phen_present = set(df_bins["phen"].unique())
        ref_pop = "EUR_300k"
        df_ref = df_order[df_order["pop"] == ref_pop].copy()

        phen_order = []
        if not df_ref.empty:
            df_ref_sumrhe = df_ref[df_ref["method"] == "sumrhe"]
            if not df_ref_sumrhe.empty:
                phen_scores = df_ref_sumrhe.set_index("phen")["h2"]
            else:
                phen_scores = df_ref.groupby("phen")["h2"].mean()

            phen_scores = pd.to_numeric(phen_scores, errors="coerce").sort_values(
                ascending=False
            )
            phen_order = [p for p in phen_scores.index if p in phen_present]
        else:
            phen_scores = pd.to_numeric(
                df_order.groupby("phen")["h2"].mean(), errors="coerce"
            ).sort_values(ascending=False)
            phen_order = [p for p in phen_scores.index if p in phen_present]

        # append remaining phens (phen_list order)
        phen_order_set = set(phen_order)
        phen_order += [
            p for p in phen_list if (p in phen_present and p not in phen_order_set)
        ]

        phen_order_clean = [phen2clean.get(p, p) for p in phen_order]

        # figure geometry (analogous)
        n_methods = len(methods_present)
        fig_height = 3.5 * n_methods
        base_fig_width = max(16.0, 1.1 * max(1, len(phen_order)))
        axes_left_in = BASE_AXES_LEFT * base_fig_width
        axes_right_in = BASE_AXES_RIGHT * base_fig_width
        legend_x_in = BASE_LEGEND_X * base_fig_width
        fig_width = max(16.0, legend_x_in + LEGEND_WIDTH_IN + RIGHT_PAD_IN)
        axes_left = axes_left_in / fig_width
        axes_right = axes_right_in / fig_width
        legend_x = legend_x_in / fig_width

        fig, axes = plt.subplots(
            n_methods, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
        if n_methods == 1:
            axes = [axes]

        # grouped bars geometry (now: pop bars per trait, within each method row)
        x = np.arange(len(phen_order))
        group_width = 0.60
        P = max(1, len(pops_present))
        bar_w = group_width / P

        for ax_idx, (ax, method) in enumerate(zip(axes, methods_present)):
            df_m = df_bins[df_bins["method"] == method].copy()
            if df_m.empty:
                ax.text(0.5, 0.5, f"No data for {method}", ha="center", va="center")
                ax.set_axis_off()
                continue

            # Convert estimates to numeric values.
            df_m["phen"] = df_m["phen"].astype(str)
            df_m["pop"] = df_m["pop"].astype(str)
            df_m["h2_num"] = pd.to_numeric(df_m["h2"], errors="coerce")
            df_m["h2_se_num"] = pd.to_numeric(df_m["h2_se"], errors="coerce")

            # duplicates safety (should be none after df_sel)
            dup = df_m.duplicated(["phen", "pop"], keep=False)
            if dup.any():
                ndup = int(dup.sum())
                print(
                    f"[warn] method={method}, bins={bins_val}: found {ndup} duplicated (phen,pop) rows; using last",
                    file=sys.stderr,
                )
                df_m = (
                    df_m.sort_values("window_sort")
                    .groupby(["phen", "pop"], as_index=False)
                    .tail(1)
                )

            h2_mat = df_m.pivot(index="phen", columns="pop", values="h2_num").reindex(
                index=phen_order, columns=pops_present
            )
            se_mat = df_m.pivot(
                index="phen", columns="pop", values="h2_se_num"
            ).reindex(index=phen_order, columns=pops_present)

            for j, pop in enumerate(pops_present):
                offset = (j - (P - 1) / 2.0) * bar_w
                color = POP_COLOR.get(
                    pop, plt.rcParams["axes.prop_cycle"].by_key()["color"][j % 10]
                )

                for i, phen in enumerate(phen_order):
                    h = (
                        h2_mat.loc[phen, pop]
                        if (phen in h2_mat.index and pop in h2_mat.columns)
                        else np.nan
                    )
                    s = (
                        se_mat.loc[phen, pop]
                        if (phen in se_mat.index and pop in se_mat.columns)
                        else np.nan
                    )
                    if not np.isfinite(h):
                        continue

                    bx = x[i] + offset
                    rects = ax.bar(
                        bx,
                        float(h),
                        width=bar_w * 0.85,
                        color=color,
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=4,
                        align="center",
                    )
                    rect = rects[0]

                    if np.isfinite(s) and float(s) > 0.0:
                        x_center = rect.get_x() + rect.get_width() / 2.0
                        ax.errorbar(
                            x_center,
                            float(h),
                            yerr=float(s),
                            fmt="none",
                            ecolor="black",
                            elinewidth=1.0,
                            capsize=3,
                            zorder=6,
                        )

            ax.set_ylabel("Total h²", fontsize=Y_LABEL_FONTSIZE)
            ax.set_title(
                METHOD_LABEL.get(method, method), fontsize=METHOD_TITLE_FONTSIZE
            )

            ax.set_ylim(y_low, y_high)
            ax.set_yticks(y_ticks)

            ax.set_xticks(x)
            ax.set_xticklabels(phen_order_clean)
            for label in ax.get_xticklabels():
                label.set_rotation(18)

            if ax_idx < n_methods - 1:
                ax.set_xlabel("")

            ax.axhspan(y_low, 0.0, color="0.96", zorder=0)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, zorder=2)

            ax.set_axisbelow(True)
            ax.grid(False, axis="x")
            ax.grid(True, axis="y", which="major", linestyle=":", alpha=0.45)

            from matplotlib.ticker import AutoMinorLocator

            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.20)

        axes[-1].set_xlabel("Phenotype")

        # legend: populations
        handles = []
        labels = []
        for pop in pops_present:
            handles.append(
                Patch(facecolor=POP_COLOR.get(pop, "0.5"), edgecolor="black")
            )
            labels.append(POP_LABEL.get(pop, pop))

        fig.legend(
            handles,
            labels,
            title="Population",
            loc="center left",
            bbox_to_anchor=(legend_x, 0.5),
            borderaxespad=0.0,
            ncol=1,
            columnspacing=1.0,
            handletextpad=0.2,
        )

        out_path = _resolve_outfile(outfile, bins_val, multi_bins=multi_bins)
        outdir = os.path.dirname(out_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        fig.subplots_adjust(
            left=axes_left,
            right=axes_right,
            bottom=BASE_AXES_BOTTOM,
            top=BASE_AXES_TOP,
            hspace=BASE_AXES_HSPACE,
        )
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[ok] saved figure to {out_path}")


if __name__ == "__main__":
    main()
