#!/usr/bin/env python3
# relmse_mafld_maintext.py
#
# Main-text figures for MAF-LD partitioning (local_h2 sims), focusing on TOTAL h2:
#   (A) relMSE summary (2 x 3): rows = true architecture (GCTA/LDAK), cols = pops (AFR/EUR/SAS)
#       - relative MSE vs covsumrhe (baseline labeled "SUMMIT") by default
#       - per-setting dots (jittered) + IQR whisker + median diamond
#       - baseline shown as a single diamond at y=1
#   (B) pooled error boxplots (3 x 2): rows = pops (AFR/EUR/SAS), cols = true architecture (GCTA/LDAK)
#       - error = (estimated total h2 - true total h2) pooled across all settings/replicates
#       - baseline included and shown first; "SUMMIT" tick bolded
#       - whisker-safe y-lims so nothing gets clipped
#
# Expected CSV per pop:
#   {outs_root}/{POP}/estimates.csv
#
# Example:
#   python relmse_mafld_maintext.py
#
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Config: methods/windows and plotting labels
# -------------------------
ALLOWED_METHODS = {
    "rhe",
    "sumrhe",
    "covsumrhe",
    "ldsc",
    "covldsc",
    "sumher",
    "sumher_ldak",
}

# Window size for LDSC and SumHer.
KEEP_WINDOWS = {20000}  # kb, applies to ldsc/sumher/sumher_ldak only

TAB20 = list(plt.get_cmap("tab20").colors)

# Pretty labels (paper-friendly)
PRETTY_LABEL = {
    "covsumrhe": "SUMMIT",
    "rhe": "RHE-mc",
    "covldsc": "cov-LDSC",
    "ldsc_50000": "LDSC",
    "sumher_50000": "SumHer\n(GCTA)",
    "sumher_ldak_50000": "SumHer\n(LDAK)",
    "ldsc_20000": "LDSC",
    "sumher_20000": "SumHer\n(GCTA)",
    "sumher_ldak_20000": "SumHer\n(LDAK)",
    # (optional if ever present/kept)
    "ldsc_2000": "LDSC",
    "sumher_2000": "SumHer\n(GCTA)",
    "sumher_ldak_2000": "SumHer\n(LDAK)",
}

METHOD_COLOR = {
    "covsumrhe": TAB20[4],
    "rhe": TAB20[0],
    "covldsc": TAB20[2],
    "ldsc_50000": TAB20[10],
    "sumher_50000": TAB20[6],
    "sumher_ldak_50000": TAB20[8],
    "ldsc_20000": TAB20[10],
    "sumher_20000": TAB20[6],
    "sumher_ldak_20000": TAB20[8],
}

# Method order, with the baseline first.
DEFAULT_METHODS = [
    "covsumrhe",
    "rhe",
    "covldsc",
    "ldsc_20000",
    "sumher_20000",
    "sumher_ldak_20000",
]

TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
XTICK_LABEL_FONTSIZE = 12
YTICK_LABEL_FONTSIZE = 12


def stacked_figsize(n_cols: int, n_methods: int) -> tuple[float, float]:
    """Compact supplemental figure size; keeps text readable after LaTeX scaling."""
    col_width = 5.3 if n_methods >= 6 else 3.8
    return (max(10.5, col_width * n_cols), 7.3)


def xtick_labelsize(n_methods: int) -> float:
    if n_methods >= 7:
        return 8.5
    if n_methods >= 6:
        return 9.5
    return XTICK_LABEL_FONTSIZE


# -------------------------
# Helpers
# -------------------------
def parse_arch_and_base_setting(s):
    s = str(s)
    arch = "ldak" if s.endswith("_ldak") else "gcta"
    base = s.replace("_ldak", "")
    return arch, base


def nice_label_from_key(k: str) -> str:
    return PRETTY_LABEL.get(k, k)


def color_from_key(k: str):
    if k in METHOD_COLOR:
        return METHOD_COLOR[k]
    return TAB20[sum(k.encode("utf-8")) % len(TAB20)]


def method_window_key(method: str, window: float | int | None) -> str:
    method = str(method)
    if method in ("ldsc", "sumher", "sumher_ldak"):
        if window is None or not np.isfinite(window):
            return f"{method}_NA"
        return f"{method}_{int(window)}"
    return method


def restrict_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Keep allowed methods; for windowed methods keep only KEEP_WINDOWS."""
    if df.empty:
        return df
    df = df[df["method"].isin(ALLOWED_METHODS)].copy()
    is_windowed = df["method"].isin({"ldsc", "sumher", "sumher_ldak"})
    keep = pd.concat(
        [
            df[~is_windowed],
            df[is_windowed & df["window"].isin(KEEP_WINDOWS)],
        ],
        ignore_index=True,
    )
    return keep


def parse_methods_arg(s: str) -> List[str]:
    out = []
    for tok in str(s).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def median_pos(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    return float(np.median(x))


def auto_yscale(rel_vals: np.ndarray) -> str:
    """Heuristic: use log if ratios have wide range."""
    v = rel_vals[np.isfinite(rel_vals) & (rel_vals > 0)]
    if v.size == 0:
        return "linear"
    vmax = float(np.nanmax(v))
    vmin = float(np.nanmin(v))
    if vmax > 3.0 or vmin < 0.33:
        return "log"
    return "linear"


def set_log_ylim_with_pad(ax, vals: np.ndarray):
    v = vals[np.isfinite(vals) & (vals > 0)]
    if v.size == 0:
        ax.set_ylim(0.5, 2.0)
        return
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    lo = min(lo, 1.0 / 1.5)
    hi = max(hi, 1.0 * 1.5)
    ax.set_ylim(max(lo / 1.3, 1e-6), hi * 1.3)


def set_linear_ylim_with_pad(ax, vals: np.ndarray):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        ax.set_ylim(0.5, 2.0)
        return
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    lo = min(lo, 1.0)
    hi = max(hi, 1.0)
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.15
    ax.set_ylim(lo - pad, hi + pad)


def robust_error_ylim(err: np.ndarray) -> Tuple[float, float]:
    """
    Compute y-limits that won't clip seaborn boxplot whiskers (showfliers=False).
    Whiskers go to the most extreme points within [Q1-1.5*IQR, Q3+1.5*IQR].
    """
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return (-1.0, 1.0)

    q1 = float(np.nanpercentile(e, 25))
    q3 = float(np.nanpercentile(e, 75))
    iqr = q3 - q1

    if iqr <= 0 or not np.isfinite(iqr):
        lo = float(np.nanmin(e))
        hi = float(np.nanmax(e))
    else:
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
        lo = (
            float(np.nanmin(e[e >= fence_lo]))
            if np.any(e >= fence_lo)
            else float(np.nanmin(e))
        )
        hi = (
            float(np.nanmax(e[e <= fence_hi]))
            if np.any(e <= fence_hi)
            else float(np.nanmax(e))
        )

    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    span = hi - lo
    pad = 0.15 * span if span > 0 else 0.25
    return (lo - pad, hi + pad)


def robust_error_ylim_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "method_window",
    group_order: List[str] | None = None,
) -> Tuple[float, float]:
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


def compute_relmse_by_setting(
    df: pd.DataFrame,
    base_key: str,
    setting_cols: List[str],
) -> pd.DataFrame:
    """
    Compute relMSE per (setting_cols + method_window).

    relMSE(method, setting) = MSE(method, setting) / MSE(base, setting),
    where MSE is computed across replicates pooled within the setting.
    """
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["h2", "true_h2", "method_window"])
    if d.empty:
        return pd.DataFrame()

    d["sqerr"] = (d["h2"].astype(float) - d["true_h2"].astype(float)) ** 2

    mse_tbl = (
        d.groupby(setting_cols + ["method_window"], as_index=False)["sqerr"]
        .mean()
        .rename(columns={"sqerr": "mse"})
    )

    base_tbl = mse_tbl[mse_tbl["method_window"] == base_key].copy()
    base_tbl = base_tbl.drop(columns=["method_window"]).rename(
        columns={"mse": "mse_base"}
    )

    out = mse_tbl.merge(base_tbl, on=setting_cols, how="inner")
    out["rel_mse"] = out["mse"] / out["mse_base"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["rel_mse", "mse_base"])
    out = out[out["mse_base"] > 0].copy()
    return out


def load_one_pop(pop: str, outs_root: str, csv_name: str) -> pd.DataFrame:
    p = os.path.join(outs_root, pop, csv_name)
    if not os.path.exists(p):
        print(f"[warn] missing CSV for {pop}: {p}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame()
    df["pop"] = pop
    return df


def preprocess_df(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all

    # Parse arch & base_setting from 'setting'
    if "setting" in df_all.columns:
        archs, bases = zip(*df_all["setting"].map(parse_arch_and_base_setting))
        df_all["arch"] = archs
        df_all["base_setting"] = bases
    else:
        df_all["arch"] = "gcta"
        df_all["base_setting"] = ""

    # Numeric coerce
    for col in ["window", "p_causal", "true_h2", "h2", "num_bins"]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all["window"] = df_all.get("window", -1).fillna(-1).astype(int)

    # Restrict methods/windows
    df = restrict_methods(df_all)
    if df.empty:
        return df

    # Create method_window keys
    df["method"] = df["method"].astype(str)
    df["method_window"] = [
        method_window_key(m, w)
        for m, w in zip(df["method"].tolist(), df["window"].tolist())
    ]

    # Hygiene
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def bold_summit_ticks(ax):
    for t in ax.get_xticklabels():
        if t.get_text() == "SUMMIT":
            t.set_fontweight("bold")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pops",
        nargs="*",
        default=["EUR", "SAS", "AFR"],
        help="Populations (default: EUR SAS AFR)",
    )
    ap.add_argument(
        "--outs-root",
        default="data/sim_h2/mafld",
        help="Root containing per-pop outputs",
    )
    ap.add_argument(
        "--csv-name", default="estimates.csv", help="CSV under each pop dir"
    )
    ap.add_argument(
        "--figdir", default="figs/supplementary", help="Output figure directory"
    )
    ap.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method keys (baseline first)",
    )
    ap.add_argument(
        "--base", default="covsumrhe", help="Baseline method key (default: covsumrhe)"
    )
    ap.add_argument(
        "--yscale",
        choices=["auto", "linear", "log"],
        default="auto",
        help="Y-scale for relMSE panels",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for jitter")
    args = ap.parse_args()

    pops = [str(p) for p in args.pops]
    methods_req = parse_methods_arg(args.methods)
    base_key = str(args.base).strip()

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load and preprocess
    dfs = []
    for pop in pops:
        d = load_one_pop(pop, args.outs_root, args.csv_name)
        if not d.empty:
            dfs.append(d)
    if not dfs:
        raise SystemExit(
            "[error] No rows found across requested pops. Check --outs-root/--csv-name."
        )

    df_all = preprocess_df(pd.concat(dfs, ignore_index=True))
    if df_all.empty:
        raise SystemExit("[error] No rows after method/window filtering.")

    # Drop sumrhe for main text by default (consistent w/ your genome-wide main text)
    df_all = df_all[df_all["method_window"] != "sumrhe"].copy()

    available = set(df_all["method_window"].unique())
    if base_key not in available:
        raise SystemExit(
            f"[error] baseline '{base_key}' not present after filtering.\n"
            f"Present keys (sample): {sorted(list(available))[:50]}"
        )

    # Resolve method list: baseline first, then requested methods in order if present
    methods_all = [base_key] + [
        m for m in methods_req if m != base_key and m in available
    ]
    if len(methods_all) == 1:
        raise SystemExit(
            "[error] After filtering, only baseline remains. Check --methods or data availability."
        )

    # Define "setting" keys (exclude replicate-specific columns)
    # Match base_setting and true_h2 when comparing relative MSE.
    setting_cols = ["pop", "arch", "base_setting", "p_causal", "true_h2"]

    # relMSE per setting
    rel_tbl = compute_relmse_by_setting(
        df_all[df_all["method_window"].isin(methods_all)], base_key, setting_cols
    )
    if rel_tbl.empty:
        raise SystemExit(
            "[error] relMSE table is empty (likely missing baseline rows for some settings)."
        )

    sns.set_style("whitegrid")

    # ============================================================
    # Figure A: relMSE summary (2 x 3): rows arch, cols pops
    # ============================================================
    arch_order = ["gcta", "ldak"]
    n_rows, n_cols = 2, len(pops)

    figA, axesA = plt.subplots(
        n_rows, n_cols, figsize=stacked_figsize(n_cols, len(methods_all)), sharey=False
    )
    if n_cols == 1:
        axesA = np.array([[axesA[0]], [axesA[1]]])

    # Column titles: pops
    for j, pop in enumerate(pops):
        axesA[0, j].set_title(pop, fontsize=TITLE_FONTSIZE)

    for i, arch in enumerate(arch_order):
        for j, pop in enumerate(pops):
            ax = axesA[i, j]

            sub = rel_tbl[(rel_tbl["pop"] == pop) & (rel_tbl["arch"] == arch)].copy()
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # y-scale decided from non-baseline methods
            sub_nonbase = sub[sub["method_window"] != base_key]
            all_rel_vals = sub_nonbase["rel_mse"].to_numpy(float)

            scale = args.yscale if args.yscale != "auto" else auto_yscale(all_rel_vals)
            # if pop.startswith("EUR"):
            #     scale = "linear"

            x = np.arange(len(methods_all), dtype=float)

            for k, m in enumerate(methods_all):
                if m == base_key:
                    # baseline: show a single diamond at y=1
                    ax.scatter(
                        [x[k]],
                        [1.0],
                        s=90,
                        marker="D",
                        color=color_from_key(m),
                        zorder=4,
                    )
                    continue

                vals = sub[sub["method_window"] == m]["rel_mse"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    jitter = rng.uniform(-0.18, 0.18, size=vals.size)
                    ax.scatter(
                        np.full(vals.size, x[k]) + jitter,
                        vals,
                        s=22,
                        alpha=0.28,
                        color=color_from_key(m),
                        edgecolors="none",
                        zorder=2,
                    )

                    # IQR whisker + median
                    q25 = float(np.nanpercentile(vals, 25))
                    q75 = float(np.nanpercentile(vals, 75))
                    center = median_pos(vals)

                    if np.isfinite(q25) and np.isfinite(q75):
                        ax.plot(
                            [x[k], x[k]],
                            [q25, q75],
                            color=color_from_key(m),
                            linewidth=2.2,
                            zorder=3,
                        )
                        ax.plot(
                            [x[k] - 0.08, x[k] + 0.08],
                            [q25, q25],
                            color=color_from_key(m),
                            linewidth=2.2,
                            zorder=3,
                        )
                        ax.plot(
                            [x[k] - 0.08, x[k] + 0.08],
                            [q75, q75],
                            color=color_from_key(m),
                            linewidth=2.2,
                            zorder=3,
                        )

                    if np.isfinite(center) and center > 0:
                        ax.scatter(
                            [x[k]],
                            [center],
                            s=90,
                            marker="D",
                            color=color_from_key(m),
                            zorder=4,
                        )

            ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, zorder=1)

            # y-axis labels only on first column
            if j == 0:
                ax.set_ylabel(
                    "Relative MSE\n(GCTA arch)"
                    if arch == "gcta"
                    else "Relative MSE\n(LDAK arch)",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            else:
                ax.set_ylabel("")
                # ax.tick_params(axis="y", labelleft=False)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [nice_label_from_key(m) for m in methods_all], rotation=0, ha="center"
            )
            ax.tick_params(axis="x", labelsize=xtick_labelsize(len(methods_all)))
            ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)
            bold_summit_ticks(ax)

            if scale == "log":
                ax.set_yscale("log")
                set_log_ylim_with_pad(ax, all_rel_vals)
            else:
                ax.set_yscale("linear")
                set_linear_ylim_with_pad(ax, all_rel_vals)

    figA.tight_layout()
    outA = figdir / "fig_s12_mafld_h2_mse.pdf"
    figA.savefig(outA, dpi=300, bbox_inches="tight")
    print(f"[ok] saved {outA}")

    # ============================================================
    # Figure B: pooled error boxplots (2 x 3): rows arch, cols pops
    # ============================================================
    arch_order = ["gcta", "ldak"]
    n_rowsB, n_colsB = 2, len(pops)
    figB, axesB = plt.subplots(
        n_rowsB,
        n_colsB,
        figsize=stacked_figsize(n_colsB, len(methods_all)),
        sharey=False,
    )

    if n_colsB == 1:
        axesB = np.array([[axesB[0]], [axesB[1]]])

    # Column titles (pops)
    for j, pop in enumerate(pops):
        axesB[0, j].set_title(pop, fontsize=TITLE_FONTSIZE)

    for i, arch in enumerate(arch_order):
        for j, pop in enumerate(pops):
            ax = axesB[i, j]

            sub = df_all[
                (df_all["pop"] == pop)
                & (df_all["arch"] == arch)
                & (df_all["method_window"].isin(methods_all))
            ].copy()

            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            sub["err"] = sub["h2"].astype(float) - sub["true_h2"].astype(float)
            sub["pretty"] = sub["method_window"].map(nice_label_from_key)

            order_pretty = [nice_label_from_key(m) for m in methods_all]
            palette = {nice_label_from_key(m): color_from_key(m) for m in methods_all}

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

            ax.axhline(0.0, color="red", linestyle="--", linewidth=1.8)

            # Row labels on first column only
            if j == 0:
                ax.set_ylabel(
                    "Error (estimate − true $h^2$)\n(GCTA arch)"
                    if arch == "gcta"
                    else "Error (estimate − true $h^2$)\n(LDAK arch)",
                    fontsize=14,
                )
            else:
                ax.set_ylabel("")
                # ax.tick_params(axis="y", labelleft=False)

            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
            ax.tick_params(axis="x", labelsize=xtick_labelsize(len(methods_all)))
            ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)
            bold_summit_ticks(ax)

            ax.set_ylim(
                *robust_error_ylim_by_group(sub, "err", group_order=methods_all)
            )

    figB.tight_layout()
    outB = figdir / "fig_s13_mafld_h2_error.pdf"
    figB.savefig(outB, dpi=300, bbox_inches="tight")
    print(f"[ok] saved {outB}")


if __name__ == "__main__":
    main()
