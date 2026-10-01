#!/usr/bin/env python3
# relmse_main.py
#
# Main-text friendly stacked figure:
#   2 x 3 subplots (rows: relMSE / pooled error; cols: populations)
#
# Key behaviors:
#  - Default --base is covsumrhe (SUMMIT).
#  - Top row (relMSE): reserve an EMPTY first x-slot for SUMMIT to align with bottom row.
#       * We include SUMMIT in the x-axis order but do NOT plot anything at that slot.
#       * We also leave its tick label blank (empty string), so it’s just an empty tick.
#  - Bottom row (pooled error): INCLUDE baseline (SUMMIT) and show it first.
#  - EUR_* relMSE panel forced to linear y-scale.
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Labels + colors
# -------------------------
TAB20 = list(plt.get_cmap("tab20").colors)

PRETTY_LABEL = {
    "rhe": "RHE-mc",
    "covsumrhe": "SUMMIT",
    "covldsc": "cov-LDSC",
    "ldsc_20000": "LDSC",
    "ldsc_50000": "LDSC",
    "sumher_20000": "SumHer\n(GCTA)",
    "sumher_ldak_20000": "SumHer\n(LDAK)",
    "hdl": "HDL",
}

METHOD_COLOR = {
    "rhe": TAB20[0],
    "covsumrhe": TAB20[4],
    "covldsc": TAB20[2],
    "ldsc_2000": TAB20[11],
    "ldsc_20000": TAB20[10],
    "ldsc_50000": TAB20[10],
    "sumher_2000": TAB20[7],
    "sumher_20000": TAB20[6],
    "sumher_ldak_2000": TAB20[9],
    "sumher_ldak_20000": TAB20[8],
    "hdl": TAB20[19],
}

DEFAULT_METHODS = [
    "rhe",
    "covldsc",
    "ldsc_20000",
    "covsumrhe",
    "sumher_20000",
    "sumher_ldak_20000",
    "hdl",
]
WINDOW_METHODS = [
    "covsumrhe",
    "ldsc_2000",
    "ldsc_20000",
    "sumher_2000",
    "sumher_20000",
    "sumher_ldak_2000",
    "sumher_ldak_20000",
]
WINDOW_PRETTY_LABEL = {
    "covsumrhe": "SUMMIT",
    "ldsc_2000": "LDSC\n(2Mb)",
    "ldsc_20000": "LDSC\n(20Mb)",
    "sumher_2000": "SumHer\nGCTA\n(2Mb)",
    "sumher_20000": "SumHer\nGCTA\n(20Mb)",
    "sumher_ldak_2000": "SumHer\nLDAK\n(2Mb)",
    "sumher_ldak_20000": "SumHer\nLDAK\n(20Mb)",
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


# -------------------------
# Helpers
# -------------------------
def nice_label_from_key(k: str) -> str:
    return PRETTY_LABEL.get(k, k)


def plot_label_from_key(k: str, label_map: Optional[dict[str, str]] = None) -> str:
    if label_map is not None and k in label_map:
        return label_map[k]
    return nice_label_from_key(k)


def color_from_key(k: str):
    if k in METHOD_COLOR:
        return METHOD_COLOR[k]
    return TAB20[sum(k.encode("utf-8")) % len(TAB20)]


def method_window_key(method: str, window: float | int | None) -> str:
    """Create canonical method_window key from (method, window)."""
    if method in ("ldsc", "sumher", "sumher_ldak"):
        if window is None or not np.isfinite(window):
            return f"{method}_NA"
        return f"{method}_{int(window)}"
    return method


def resolve_base_key(base: str, base_window: int, available: set[str]) -> str:
    """Resolve --base into a method_window key."""
    if base in ("ldsc", "sumher", "sumher_ldak"):
        key = f"{base}_{int(base_window)}"
    else:
        key = base

    if key not in available:
        cands = sorted([k for k in available if k == base or k.startswith(base + "_")])
        if not cands:
            cands = sorted(list(available))[:40]
        raise SystemExit(
            f"ERROR: --base '{base}' resolved to '{key}', but that key isn't present.\n"
            f"Some available keys: {cands}\n"
            f"Tip: windowed base example: --base ldsc --base_window 20000 (or --base ldsc_20000)"
        )
    return key


def parse_methods_arg(s: str) -> List[str]:
    out = []
    for tok in s.split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def gmean_pos(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(x))))


def compute_relmse_by_setting(df: pd.DataFrame, base_key: str) -> pd.DataFrame:
    """
    Returns relMSE per (method_window, h2, pcausal):
      mse = mean((estimate - true_h2)^2) over replicates
      rel_mse = mse / mse_base
    """
    g = df.groupby(["method_window", "h2", "pcausal"], as_index=False)["estimate"]
    mse = g.apply(
        lambda s: np.mean((s.to_numpy(float) - float(s.name[1])) ** 2)
        if len(s)
        else np.nan
    )
    mse = (
        mse.rename(columns={None: "mse"})
        if None in mse.columns
        else mse.rename(columns={"estimate": "mse"})
    )

    base = mse[mse["method_window"] == base_key].copy()
    base = base.rename(columns={"mse": "mse_base"}).drop(columns=["method_window"])
    out = mse.merge(base, on=["h2", "pcausal"], how="inner")
    out["rel_mse"] = out["mse"] / out["mse_base"]
    return out[["method_window", "h2", "pcausal", "rel_mse"]]


def summarize_relmse(rel: pd.DataFrame, method_order: List[str]) -> pd.DataFrame:
    """Summarize relMSE across settings for each method."""
    rows = []
    for m in method_order:
        sub = rel[rel["method_window"] == m]["rel_mse"].to_numpy(float)
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            rows.append(
                dict(
                    method_window=m,
                    n=0,
                    gmean=np.nan,
                    median=np.nan,
                    q25=np.nan,
                    q75=np.nan,
                    min=np.nan,
                    max=np.nan,
                )
            )
            continue
        rows.append(
            dict(
                method_window=m,
                n=int(sub.size),
                gmean=gmean_pos(sub),
                median=float(np.nanmedian(sub)),
                q25=float(np.nanpercentile(sub, 25)),
                q75=float(np.nanpercentile(sub, 75)),
                min=float(np.nanmin(sub)),
                max=float(np.nanmax(sub)),
            )
        )
    return pd.DataFrame(rows)


def auto_yscale(rel_vals: np.ndarray) -> str:
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
    y-lims that won't clip seaborn boxplot whiskers (showfliers=False).
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
    group_order: Optional[list[str]] = None,
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


def plot_methods_for_relmse_and_error(present_methods: list[str], base_key: str):
    """
    relMSE: exclude baseline
    error: include baseline and place it FIRST
    """
    present = list(present_methods)
    if base_key not in present:
        present.append(base_key)

    plot_rel = [m for m in present if m != base_key]
    plot_err = [base_key] + [m for m in present if m != base_key]
    return plot_rel, plot_err


def _isclose_series(s: pd.Series, val: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return pd.Series(
        np.isfinite(x) & np.isclose(x, float(val), rtol=0, atol=1e-12), index=s.index
    )


# -------------------------
# NEW: SE column detection + metrics
# -------------------------
def detect_reported_se_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a reported SE column for h2 estimates.
    Priority: common names, then any numeric column that looks like an SE.
    """
    preferred = [
        "h2_se",
        "se",
        "se_h2",
        "h2se",
        "estimate_se",
        "se_estimate",
        "jackknife_se",
        "jk_se",
        "se_jk",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    # fall back: look for something with "se" in the name that is numeric-ish
    cands = []
    for c in df.columns:
        cl = str(c).lower()
        if "se" in cl or "std_err" in cl or "stderr" in cl:
            cands.append(c)

    for c in cands:
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s.to_numpy(dtype=float)).any():
            return c

    return None


def compute_mse_bias_true_se_reported_se_by_setting(
    df: pd.DataFrame, se_col: Optional[str]
) -> pd.DataFrame:
    """
    Per (method_window, h2, pcausal) compute over replicates:
      mse      = mean((est - true)^2)
      bias     = mean(est - true)
      true_se  = sd(est)  [empirical across replicates]
      rep_se   = mean(reported_se) if available
    """
    rows = []
    use_rep_se = se_col is not None and se_col in df.columns
    for (m, h2, pc), sub in df.groupby(
        ["method_window", "h2", "pcausal"], dropna=False
    ):
        est = pd.to_numeric(sub["estimate"], errors="coerce").to_numpy(float)
        true = float(h2) if pd.notnull(h2) else np.nan
        est = est[np.isfinite(est)]
        if est.size == 0 or not np.isfinite(true):
            rows.append(
                dict(
                    method_window=m,
                    h2=h2,
                    pcausal=pc,
                    nrep=int(est.size),
                    mse=np.nan,
                    bias=np.nan,
                    true_se=np.nan,
                    rep_se=np.nan,
                )
            )
            continue

        err = est - true
        mse = float(np.mean(err * err))
        bias = float(np.mean(err))
        true_se = float(np.std(est, ddof=1)) if est.size >= 2 else np.nan

        rep_se = np.nan
        if use_rep_se:
            rep = pd.to_numeric(sub[se_col], errors="coerce").to_numpy(float)
            rep = rep[np.isfinite(rep)]
            rep_se = float(np.mean(rep)) if rep.size else np.nan

        rows.append(
            dict(
                method_window=m,
                h2=true,
                pcausal=pc,
                nrep=int(est.size),
                mse=mse,
                bias=bias,
                true_se=true_se,
                rep_se=rep_se,
            )
        )

    return pd.DataFrame(rows)


def summarize_metrics_across_settings(
    per_setting: pd.DataFrame, method_order: List[str]
) -> pd.DataFrame:
    """
    Aggregate per-setting metrics into one row per method.
    We average across settings (each (h2,pcausal) gets equal weight).
    """
    out_rows = []
    for m in method_order:
        sub = per_setting[per_setting["method_window"] == m].copy()
        if sub.shape[0] == 0:
            out_rows.append(
                dict(
                    method_window=m,
                    n_settings=0,
                    n_reps_total=0,
                    mse=np.nan,
                    bias=np.nan,
                    true_se=np.nan,
                    mean_reported_se=np.nan,
                )
            )
            continue

        out_rows.append(
            dict(
                method_window=m,
                n_settings=int(sub.shape[0]),
                n_reps_total=int(
                    pd.to_numeric(sub["nrep"], errors="coerce").fillna(0).sum()
                ),
                mse=float(
                    np.nanmean(
                        pd.to_numeric(sub["mse"], errors="coerce").to_numpy(float)
                    )
                ),
                bias=float(
                    np.nanmean(
                        pd.to_numeric(sub["bias"], errors="coerce").to_numpy(float)
                    )
                ),
                true_se=float(
                    np.nanmean(
                        pd.to_numeric(sub["true_se"], errors="coerce").to_numpy(float)
                    )
                ),
                mean_reported_se=float(
                    np.nanmean(
                        pd.to_numeric(sub["rep_se"], errors="coerce").to_numpy(float)
                    )
                ),
            )
        )

    return pd.DataFrame(out_rows)


# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pops",
    default="EUR,SAS,AFR",
    help="Comma-separated population codes (default: EUR,SAS,AFR)",
)
parser.add_argument(
    "--indir",
    default="data/sim_h2/total",
    help="Input root directory containing {POP}/estimates.csv",
)
parser.add_argument(
    "--figdir",
    default="figs/supplementary",
    help="Output figure directory (default: figs)",
)
parser.add_argument(
    "--methods",
    default=",".join(DEFAULT_METHODS),
    help="Comma-separated method keys to show",
)
parser.add_argument(
    "--base",
    default="covsumrhe",  # <-- default changed as requested
    help="Baseline method for relMSE. Can be explicit key (e.g. covsumrhe, ldsc_20000) "
    "or family (ldsc,sumher,sumher_ldak) with --base_window.",
)
parser.add_argument(
    "--base_window",
    type=int,
    default=20000,
    help="If --base is a family, use this window kb.",
)
parser.add_argument(
    "--yscale",
    choices=["auto", "linear", "log"],
    default="auto",
    help="Y-scale for relMSE panels.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
parser.add_argument(
    "--plot-window",
    action="store_true",
    help="Also write supplementary stacked figures comparing SUMMIT to LDSC, SumHer-GCTA, "
    "and SumHer-LDAK at 2000kb and 20000kb windows.",
)
# NEW
parser.add_argument(
    "--stratify-h2",
    action="store_true",
    help="If set, columns are true h2 values (within each population). "
    "If multiple pops are specified, write one PDF per population.",
)
parser.add_argument(
    "--verbose",
    nargs="?",  # optional value
    const="h2",  # if user passes just --verbose
    default=None,  # if not provided
    choices=["h2", "extra"],
    help="Verbose diagnostics. "
    "--verbose => stratify by h2. "
    "--verbose extra => stratify by (h2, pcausal) for every setting.",
)

args = parser.parse_args()

pops = [p.strip() for p in args.pops.split(",") if p.strip()]
method_order = parse_methods_arg(args.methods)
figdir = Path(args.figdir)
figdir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed)

sns.set_style("whitegrid")


# -------------------------
# Load per-pop data + compute summaries
# -------------------------
per_pop = {}
metrics_all_rows = []  # aggregated across settings (existing CSV)

metrics_byh2_rows = []
relmse_byh2_rows = []
# NEW: verbose per-(h2,pcausal) CSVs (only for --verbose extra)
metrics_bysetting_rows = []
relmse_bysetting_rows = []

for pop in pops:
    f = Path(args.indir) / pop / "estimates.csv"
    if not f.exists():
        raise SystemExit(f"ERROR: missing file: {f}")

    df = pd.read_csv(f)

    df["h2"] = pd.to_numeric(df.get("h2"), errors="coerce")
    df["pcausal"] = pd.to_numeric(df.get("pcausal"), errors="coerce")
    df["estimate"] = pd.to_numeric(df.get("estimate"), errors="coerce")
    df["window"] = pd.to_numeric(df.get("window"), errors="coerce")

    df["method"] = df["method"].astype(str)
    df["method_window"] = [
        method_window_key(m, w)
        for m, w in zip(df["method"].tolist(), df["window"].tolist())
    ]

    available = set(df["method_window"].unique())
    present_methods = [m for m in method_order if m in available]
    present_methods = [
        m for m in present_methods if m != "sumrhe"
    ]  # keep main-text clean

    if len(present_methods) == 0:
        raise SystemExit(
            f"ERROR: None of requested methods are present for pop={pop}.\n"
            f"Requested: {method_order}\n"
            f"Available sample: {sorted(list(available))[:40]}"
        )

    base_key = resolve_base_key(args.base, args.base_window, available)

    # keep only present + base
    keep_keys = set(present_methods) | {base_key}
    df_use = df[df["method_window"].isin(keep_keys)].copy()

    # detect & coerce reported SE column if present
    se_col = detect_reported_se_col(df_use)
    if se_col is not None:
        df_use[se_col] = pd.to_numeric(df_use[se_col], errors="coerce")

    rel = compute_relmse_by_setting(df_use, base_key=base_key)

    # Use the same method order in both rows, with the baseline first.
    methods_all = [base_key] + [m for m in present_methods if m != base_key]
    methods_no_base = [
        m for m in methods_all if m != base_key
    ]  # for relMSE plotting only

    # Summarize across settings (exclude baseline; it's identically 1)
    rel_sum = summarize_relmse(
        rel[rel["method_window"].isin(methods_no_base)], methods_no_base
    )

    # Compute MSE/bias/true SE/reported SE per method (averaged across settings)
    per_setting_metrics = compute_mse_bias_true_se_reported_se_by_setting(
        df_use, se_col=se_col
    )
    metrics_sum = summarize_metrics_across_settings(per_setting_metrics, methods_all)
    metrics_sum["pop"] = pop
    metrics_sum["pretty"] = metrics_sum["method_window"].map(nice_label_from_key)
    metrics_sum["reported_se_col"] = se_col if se_col is not None else ""
    metrics_sum["base_key"] = base_key
    metrics_all_rows.append(metrics_sum)

    # list of true h2 values for stratified plotting / verbose diagnostics
    h2_vals = pd.to_numeric(df_use["h2"], errors="coerce").dropna().to_numpy(float)
    h2_vals = sorted(np.unique(h2_vals[np.isfinite(h2_vals)]).tolist())

    if len(h2_vals) == 0:
        raise SystemExit(f"ERROR: pop={pop} has no finite h2 values in estimates.csv")

    per_pop[pop] = dict(
        df=df_use,
        base_key=base_key,
        present_methods=present_methods,
        methods_all=methods_all,
        methods_no_base=methods_no_base,
        rel=rel,
        rel_sum=rel_sum,
        metrics_sum=metrics_sum,
        reported_se_col=se_col,
        h2_vals=h2_vals,
    )

    # Print quick summary (relMSE)
    tbl = rel_sum.copy()
    tbl["pretty"] = tbl["method_window"].map(nice_label_from_key)
    tbl = tbl[["pretty", "n", "gmean", "median", "min", "max"]]
    print(
        f"\n=== {pop}: relMSE aggregated across settings (vs {nice_label_from_key(base_key)}) ==="
    )
    print(tbl.to_string(index=False))
    if base_key == "covsumrhe" and "rhe" in rel["method_window"].unique():
        rhe_rel = pd.to_numeric(
            rel[rel["method_window"] == "rhe"]["rel_mse"], errors="coerce"
        ).to_numpy(float)
        summit_vs_rhe = 1.0 / rhe_rel[np.isfinite(rhe_rel) & (rhe_rel > 0)]
        if summit_vs_rhe.size:
            print(
                f"SUMMIT relMSE vs RHE-mc (inverse of RHE-mc/SUMMIT): "
                f"median={np.nanmedian(summit_vs_rhe):.3g}, "
                f"min={np.nanmin(summit_vs_rhe):.3g}, "
                f"max={np.nanmax(summit_vs_rhe):.3g}"
            )

    # Print MSE/bias/SE summary
    mt = metrics_sum.copy()
    mt = mt[
        [
            "pretty",
            "n_settings",
            "n_reps_total",
            "mse",
            "bias",
            "true_se",
            "mean_reported_se",
        ]
    ]
    print(f"\n=== {pop}: MSE / bias / SE (averaged across settings) ===")
    if se_col is None:
        print("NOTE: No reported-SE column detected; 'mean_reported_se' will be NaN.")
    print(mt.to_string(index=False))

    # -------------------------
    # Verbose diagnostics stratified by h2
    # -------------------------
    if args.verbose == "h2":
        for h2v in h2_vals:
            # relMSE stratified by h2 (aggregated across pcausal within that h2)
            rel_h2 = rel[_isclose_series(rel["h2"], float(h2v))].copy()
            rel_sum_h2 = summarize_relmse(
                rel_h2[rel_h2["method_window"].isin(methods_no_base)],
                methods_no_base,
            )
            rel_sum_h2["pop"] = pop
            rel_sum_h2["h2"] = float(h2v)
            rel_sum_h2["base_key"] = base_key
            rel_sum_h2["pretty"] = rel_sum_h2["method_window"].map(nice_label_from_key)
            relmse_byh2_rows.append(rel_sum_h2)

            # MSE/bias/SE stratified by h2 (aggregated across pcausal within that h2)
            per_setting_h2 = per_setting_metrics[
                _isclose_series(per_setting_metrics["h2"], float(h2v))
            ].copy()
            metrics_sum_h2 = summarize_metrics_across_settings(
                per_setting_h2, methods_all
            )
            metrics_sum_h2["pop"] = pop
            metrics_sum_h2["h2"] = float(h2v)
            metrics_sum_h2["pretty"] = metrics_sum_h2["method_window"].map(
                nice_label_from_key
            )
            metrics_sum_h2["reported_se_col"] = se_col if se_col is not None else ""
            metrics_sum_h2["base_key"] = base_key
            metrics_byh2_rows.append(metrics_sum_h2)

            print(
                f"\n--- {pop}: diagnostics stratified by true h2 = {float(h2v):g} ---"
            )
            t1 = rel_sum_h2[["pretty", "n", "gmean", "median", "min", "max"]].copy()
            print(f"[relMSE vs {nice_label_from_key(base_key)}]")
            print(t1.to_string(index=False))

            t2 = metrics_sum_h2[
                [
                    "pretty",
                    "n_settings",
                    "n_reps_total",
                    "mse",
                    "bias",
                    "true_se",
                    "mean_reported_se",
                ]
            ].copy()
            print("[MSE / bias / SE]")
            if se_col is None:
                print(
                    "NOTE: No reported-SE column detected; 'mean_reported_se' will be NaN."
                )
            print(t2.to_string(index=False))

    # -------------------------
    # Verbose EXTRA: diagnostics stratified by (h2, pcausal)
    # -------------------------
    elif args.verbose == "extra":
        # enumerate unique settings for this pop
        settings = (
            df_use[["h2", "pcausal"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["h2", "pcausal"], kind="mergesort")
            .to_numpy(float)
        )

        for h2v, pcv in settings:
            h2v = float(h2v)
            pcv = float(pcv)

            # relMSE for this exact setting: one value per method_window
            rel_set = rel[
                _isclose_series(rel["h2"], h2v) & _isclose_series(rel["pcausal"], pcv)
            ].copy()
            rel_set["pop"] = pop
            rel_set["base_key"] = base_key
            rel_set["pretty"] = rel_set["method_window"].map(nice_label_from_key)
            rel_set = rel_set[rel_set["method_window"].isin(methods_no_base)].copy()
            relmse_bysetting_rows.append(
                rel_set[
                    [
                        "pop",
                        "h2",
                        "pcausal",
                        "method_window",
                        "pretty",
                        "rel_mse",
                        "base_key",
                    ]
                ]
            )

            # MSE/bias/SE for this exact setting: already computed per (method,h2,pcausal)
            met_set = per_setting_metrics[
                _isclose_series(per_setting_metrics["h2"], h2v)
                & _isclose_series(per_setting_metrics["pcausal"], pcv)
            ].copy()
            met_set["pop"] = pop
            met_set["pretty"] = met_set["method_window"].map(nice_label_from_key)
            met_set["reported_se_col"] = se_col if se_col is not None else ""
            met_set["base_key"] = base_key
            metrics_bysetting_rows.append(
                met_set[
                    [
                        "pop",
                        "h2",
                        "pcausal",
                        "method_window",
                        "pretty",
                        "nrep",
                        "mse",
                        "bias",
                        "true_se",
                        "rep_se",
                        "reported_se_col",
                        "base_key",
                    ]
                ]
            )

            # Print (ordered)
            print(
                f"\n--- {pop}: diagnostics for setting h2={h2v:g}, pcausal={pcv:g} ---"
            )

            # relMSE printout (ordered by your method order)
            rel_print = rel_set[["pretty", "rel_mse"]].copy()
            order = [nice_label_from_key(m) for m in methods_no_base]
            rel_print["__ord"] = rel_print["pretty"].map(
                {k: i for i, k in enumerate(order)}
            )
            rel_print = rel_print.sort_values("__ord").drop(columns="__ord")
            print(f"[relMSE vs {nice_label_from_key(base_key)}]")
            print(rel_print.to_string(index=False))

            # metrics printout (ordered baseline first then others)
            met_print = met_set[
                ["pretty", "nrep", "mse", "bias", "true_se", "rep_se"]
            ].copy()
            order2 = [nice_label_from_key(m) for m in methods_all]
            met_print["__ord"] = met_print["pretty"].map(
                {k: i for i, k in enumerate(order2)}
            )
            met_print = met_print.sort_values("__ord").drop(columns="__ord")
            print("[MSE / bias / SE]")
            if se_col is None:
                print("NOTE: No reported-SE column detected; 'rep_se' will be NaN.")
            print(met_print.to_string(index=False))


# Save combined CSV (aggregated across settings)
metrics_all = (
    pd.concat(metrics_all_rows, ignore_index=True)
    if metrics_all_rows
    else pd.DataFrame()
)
metrics_out = figdir / "fig_s06_h2_metrics.csv"
metrics_all.to_csv(metrics_out, index=False)
print(f"\nSaved metrics CSV: {metrics_out}")

# NEW: Save per-h2 diagnostic CSVs (only with --verbose)
if args.verbose == "h2":
    if metrics_byh2_rows:
        metrics_byh2 = pd.concat(metrics_byh2_rows, ignore_index=True)
        out2 = figdir / "fig_s06_h2_metrics_by_h2.csv"
        metrics_byh2.to_csv(out2, index=False)
        print(f"Saved per-h2 metrics CSV: {out2}")

    if relmse_byh2_rows:
        relmse_byh2 = pd.concat(relmse_byh2_rows, ignore_index=True)
        out3 = figdir / f"method_relmse_byh2_base_{args.base.replace('/', '_')}.csv"
        relmse_byh2.to_csv(out3, index=False)
        print(f"Saved per-h2 relMSE CSV: {out3}")
elif args.verbose == "extra":
    if metrics_bysetting_rows:
        metrics_bysetting = pd.concat(metrics_bysetting_rows, ignore_index=True)
        out4 = figdir / "fig_s06_h2_metrics_by_setting.csv"
        metrics_bysetting.to_csv(out4, index=False)
        print(f"Saved per-setting metrics CSV: {out4}")

    if relmse_bysetting_rows:
        relmse_bysetting = pd.concat(relmse_bysetting_rows, ignore_index=True)
        out5 = (
            figdir / f"method_relmse_bysetting_base_{args.base.replace('/', '_')}.csv"
        )
        relmse_bysetting.to_csv(out5, index=False)
        print(f"Saved per-setting relMSE CSV: {out5}")


# -------------------------
# Figures
# -------------------------
def _plot_panel_relmse(
    ax,
    pop: str,
    rel_sub: pd.DataFrame,
    rel_sum_sub: pd.DataFrame,
    methods_all: list[str],
    methods_no_base: list[str],
    base_key: str,
    yscale: str,
    yvals_for_lim: np.ndarray,
    rng: np.random.Generator,
    label_map: Optional[dict[str, str]] = None,
):
    n = len(methods_all)
    x = np.arange(n, dtype=float)

    # Plot non-baseline methods at their assigned x positions.
    for i, m in enumerate(methods_all):
        if m == base_key:
            ax.scatter(
                [x[i]], [1.0], s=90, marker="D", color=color_from_key(m), zorder=4
            )
            continue

        sub = rel_sub[rel_sub["method_window"] == m]["rel_mse"].to_numpy(float)
        sub = sub[np.isfinite(sub)]
        if sub.size:
            jitter = rng.uniform(-0.18, 0.18, size=sub.size)
            ax.scatter(
                np.full(sub.size, x[i]) + jitter,
                sub,
                s=22,
                alpha=0.38,
                color=color_from_key(m),
                edgecolors="none",
                zorder=2,
            )

        row = rel_sum_sub[rel_sum_sub["method_window"] == m]
        if row.shape[0] == 1:
            q25 = float(row["q25"].iloc[0])
            q75 = float(row["q75"].iloc[0])
            gm = float(row["gmean"].iloc[0])

            if np.isfinite(q25) and np.isfinite(q75):
                ax.plot(
                    [x[i], x[i]],
                    [q25, q75],
                    color=color_from_key(m),
                    linewidth=2.2,
                    zorder=3,
                )
                ax.plot(
                    [x[i] - 0.08, x[i] + 0.08],
                    [q25, q25],
                    color=color_from_key(m),
                    linewidth=2.2,
                    zorder=3,
                )
                ax.plot(
                    [x[i] - 0.08, x[i] + 0.08],
                    [q75, q75],
                    color=color_from_key(m),
                    linewidth=2.2,
                    zorder=3,
                )

            if np.isfinite(gm) and (gm > 0):
                ax.scatter(
                    [x[i]], [gm], s=90, marker="D", color=color_from_key(m), zorder=4
                )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, zorder=1)

    ax.set_xticks(x)
    ticklabs = [plot_label_from_key(m, label_map) for m in methods_all]
    ax.set_xticklabels(ticklabs, rotation=0, ha="center")
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=xtick_labelsize(n))
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)

    for t in ax.get_xticklabels():
        if t.get_text() == plot_label_from_key(base_key, label_map):
            t.set_fontweight("bold")

    if yscale == "log":
        ax.set_yscale("log")
        set_log_ylim_with_pad(ax, yvals_for_lim)
    else:
        ax.set_yscale("linear")
        set_linear_ylim_with_pad(ax, yvals_for_lim)


def _plot_panel_error(
    ax,
    df_sub: pd.DataFrame,
    methods_all: list[str],
    ylims: tuple[float, float],
    base_key: Optional[str] = None,
    label_map: Optional[dict[str, str]] = None,
):
    n = len(methods_all)
    x = np.arange(n, dtype=float)

    err = df_sub[df_sub["method_window"].isin(methods_all)].copy()
    err["err"] = err["estimate"] - err["h2"]
    err["pretty"] = err["method_window"].map(
        lambda m: plot_label_from_key(m, label_map)
    )

    order_pretty = [plot_label_from_key(m, label_map) for m in methods_all]
    palette = {
        plot_label_from_key(m, label_map): color_from_key(m) for m in methods_all
    }

    sns.boxplot(
        data=err,
        x="pretty",
        y="err",
        order=order_pretty,
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1.2,
    )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.8)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(order_pretty, rotation=0, ha="center")
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=xtick_labelsize(n))
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)

    base_label = (
        plot_label_from_key(base_key, label_map) if base_key is not None else "SUMMIT"
    )
    for t in ax.get_xticklabels():
        if t.get_text() == base_label:
            t.set_fontweight("bold")

    ax.set_ylim(*ylims)


if not args.stratify_h2:
    # -------------------------
    # Combined Figure: 2 x N_pops
    # -------------------------
    n_pops = len(pops)
    max_methods = max(len(per_pop[p]["methods_all"]) for p in pops)
    fig, axes = plt.subplots(
        2, n_pops, figsize=stacked_figsize(n_pops, max_methods), sharey=False
    )
    if n_pops == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # ---- Row 1: relMSE ----
    for col, pop in enumerate(pops):
        ax = axes[0, col]
        base_key = per_pop[pop]["base_key"]
        methods_all = per_pop[pop]["methods_all"]
        methods_no_base = per_pop[pop]["methods_no_base"]
        rel = per_pop[pop]["rel"]
        rel_sum = per_pop[pop]["rel_sum"]

        all_rel_vals = rel[rel["method_window"].isin(methods_no_base)][
            "rel_mse"
        ].to_numpy(float)
        scale = args.yscale if args.yscale != "auto" else auto_yscale(all_rel_vals)
        if pop.startswith("EUR"):
            scale = "linear"

        _plot_panel_relmse(
            ax=ax,
            pop=pop,
            rel_sub=rel,
            rel_sum_sub=rel_sum,
            methods_all=methods_all,
            methods_no_base=methods_no_base,
            base_key=base_key,
            yscale=scale,
            yvals_for_lim=all_rel_vals,
            rng=rng,
        )

        ax.set_title(pop, fontsize=TITLE_FONTSIZE)
        if col == 0:
            ax.set_ylabel("Relative MSE", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")

    # ---- Row 2: pooled error ----
    for col, pop in enumerate(pops):
        ax = axes[1, col]
        df_use = per_pop[pop]["df"].copy()
        methods_all = per_pop[pop]["methods_all"]
        df_ylim = df_use.copy()
        df_ylim["err"] = df_ylim["estimate"] - df_ylim["h2"]
        ylims = robust_error_ylim_by_group(df_ylim, "err", group_order=methods_all)

        _plot_panel_error(ax=ax, df_sub=df_use, methods_all=methods_all, ylims=ylims)

        if col == 0:
            ax.set_ylabel("Error (estimate − true $h^2$)", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        ax.set_title("")

    fig.tight_layout()
    out = figdir / "fig_s06_h2_simulations.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")

else:
    # -------------------------
    # Stratified by h2:
    #   - if 1 pop: one 2 x (#h2) figure
    #   - if multiple pops: one PDF per pop
    # -------------------------
    for pop in pops:
        df_use = per_pop[pop]["df"].copy()
        base_key = per_pop[pop]["base_key"]
        methods_all = per_pop[pop]["methods_all"]
        methods_no_base = per_pop[pop]["methods_no_base"]
        rel = per_pop[pop]["rel"]
        h2_vals = per_pop[pop]["h2_vals"]

        n_h2 = len(h2_vals)
        fig, axes = plt.subplots(
            2, n_h2, figsize=stacked_figsize(n_h2, len(methods_all)), sharey=False
        )
        if n_h2 == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        # Use a single y-scale + y-lims across columns within a pop
        all_rel_vals_pop = rel[rel["method_window"].isin(methods_no_base)][
            "rel_mse"
        ].to_numpy(float)
        scale = args.yscale if args.yscale != "auto" else auto_yscale(all_rel_vals_pop)
        if pop.startswith("EUR"):
            scale = "linear"

        # Use a single error y-lim across columns within a pop
        df_ylim = df_use.copy()
        df_ylim["err"] = df_ylim["estimate"] - df_ylim["h2"]
        err_ylims = robust_error_ylim_by_group(df_ylim, "err", group_order=methods_all)

        # Row 1 + Row 2 per h2 column
        for col, h2v in enumerate(h2_vals):
            # --- relMSE column (only this h2) ---
            ax_top = axes[0, col]
            rel_h2 = rel[_isclose_series(rel["h2"], float(h2v))].copy()
            rel_sum_h2 = summarize_relmse(
                rel_h2[rel_h2["method_window"].isin(methods_no_base)], methods_no_base
            )

            _plot_panel_relmse(
                ax=ax_top,
                pop=pop,
                rel_sub=rel_h2,
                rel_sum_sub=rel_sum_h2,
                methods_all=methods_all,
                methods_no_base=methods_no_base,
                base_key=base_key,
                yscale=scale,
                yvals_for_lim=all_rel_vals_pop,  # shared limits across columns
                rng=rng,
            )
            ax_top.set_title(f"$h^2$ = {float(h2v):g}", fontsize=TITLE_FONTSIZE)
            if col == 0:
                ax_top.set_ylabel("Relative MSE", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax_top.set_ylabel("")

            # --- error column (only this h2) ---
            ax_bot = axes[1, col]
            df_h2 = df_use[_isclose_series(df_use["h2"], float(h2v))].copy()

            _plot_panel_error(
                ax=ax_bot, df_sub=df_h2, methods_all=methods_all, ylims=err_ylims
            )
            if col == 0:
                ax_bot.set_ylabel(
                    "Error (estimate − true $h^2$)", fontsize=AXIS_LABEL_FONTSIZE
                )
            else:
                ax_bot.set_ylabel("")
            ax_bot.set_title("")

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out = (
            figdir
            / f"{pop}_stacked_relmse_error_byh2_base_{args.base.replace('/', '_')}.pdf"
        )
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {out}")


def _prepare_plot_data_for_methods(
    pop: str,
    indir: Path,
    requested_methods: list[str],
    base: str,
    base_window: int,
    plot_name: str,
    label_map: Optional[dict[str, str]] = None,
) -> dict:
    f = indir / pop / "estimates.csv"
    if not f.exists():
        raise SystemExit(f"ERROR: missing file: {f}")

    df = pd.read_csv(f)
    df["h2"] = pd.to_numeric(df.get("h2"), errors="coerce")
    df["pcausal"] = pd.to_numeric(df.get("pcausal"), errors="coerce")
    df["estimate"] = pd.to_numeric(df.get("estimate"), errors="coerce")
    df["window"] = pd.to_numeric(df.get("window"), errors="coerce")

    df["method"] = df["method"].astype(str)
    df["method_window"] = [
        method_window_key(m, w)
        for m, w in zip(df["method"].tolist(), df["window"].tolist())
    ]

    available = set(df["method_window"].unique())
    present_methods = [m for m in requested_methods if m in available]
    missing_methods = [m for m in requested_methods if m not in available]
    if missing_methods:
        print(
            f"WARNING: pop={pop} missing {plot_name} methods: {', '.join(missing_methods)}"
        )

    base_key = resolve_base_key(base, base_window, available)
    if base_key not in present_methods:
        present_methods = [base_key] + present_methods

    non_base_methods = [m for m in present_methods if m != base_key]
    if len(non_base_methods) == 0:
        raise SystemExit(
            f"ERROR: No non-baseline {plot_name} methods are present for pop={pop}.\n"
            f"Requested: {requested_methods}\n"
            f"Available sample: {sorted(list(available))[:40]}"
        )

    methods_all = [base_key] + non_base_methods
    methods_no_base = [m for m in methods_all if m != base_key]
    df_use = df[df["method_window"].isin(methods_all)].copy()

    rel = compute_relmse_by_setting(df_use, base_key=base_key)
    rel_sum = summarize_relmse(
        rel[rel["method_window"].isin(methods_no_base)], methods_no_base
    )

    h2_vals = pd.to_numeric(df_use["h2"], errors="coerce").dropna().to_numpy(float)
    h2_vals = sorted(np.unique(h2_vals[np.isfinite(h2_vals)]).tolist())
    if len(h2_vals) == 0:
        raise SystemExit(f"ERROR: pop={pop} has no finite h2 values for {plot_name}")

    tbl = rel_sum.copy()
    tbl["pretty"] = tbl["method_window"].map(
        lambda m: plot_label_from_key(m, label_map)
    )
    tbl = tbl[["pretty", "n", "gmean", "median", "min", "max"]]
    print(
        f"\n=== {pop}: {plot_name} relMSE aggregated across settings (vs {plot_label_from_key(base_key, label_map)}) ==="
    )
    print(tbl.to_string(index=False))

    return dict(
        df=df_use,
        base_key=base_key,
        present_methods=present_methods,
        methods_all=methods_all,
        methods_no_base=methods_no_base,
        rel=rel,
        rel_sum=rel_sum,
        h2_vals=h2_vals,
    )


if args.plot_window:
    window_base = "covsumrhe"
    window_base_tag = window_base.replace("/", "_")
    window_per_pop = {
        pop: _prepare_plot_data_for_methods(
            pop=pop,
            indir=Path(args.indir),
            requested_methods=WINDOW_METHODS,
            base=window_base,
            base_window=args.base_window,
            plot_name="--plot-window",
            label_map=WINDOW_PRETTY_LABEL,
        )
        for pop in pops
    }

    if not args.stratify_h2:
        n_pops = len(pops)
        max_methods = max(len(window_per_pop[p]["methods_all"]) for p in pops)
        fig, axes = plt.subplots(
            2, n_pops, figsize=stacked_figsize(n_pops, max_methods), sharey=False
        )
        if n_pops == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for col, pop in enumerate(pops):
            ax = axes[0, col]
            base_key = window_per_pop[pop]["base_key"]
            methods_all = window_per_pop[pop]["methods_all"]
            methods_no_base = window_per_pop[pop]["methods_no_base"]
            rel = window_per_pop[pop]["rel"]
            rel_sum = window_per_pop[pop]["rel_sum"]

            all_rel_vals = rel[rel["method_window"].isin(methods_no_base)][
                "rel_mse"
            ].to_numpy(float)
            scale = args.yscale if args.yscale != "auto" else auto_yscale(all_rel_vals)
            if pop.startswith("EUR"):
                scale = "linear"

            _plot_panel_relmse(
                ax=ax,
                pop=pop,
                rel_sub=rel,
                rel_sum_sub=rel_sum,
                methods_all=methods_all,
                methods_no_base=methods_no_base,
                base_key=base_key,
                yscale=scale,
                yvals_for_lim=all_rel_vals,
                rng=rng,
                label_map=WINDOW_PRETTY_LABEL,
            )

            ax.set_title(pop, fontsize=TITLE_FONTSIZE)
            if col == 0:
                ax.set_ylabel("Relative MSE", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")

        for col, pop in enumerate(pops):
            ax = axes[1, col]
            df_use = window_per_pop[pop]["df"].copy()
            base_key = window_per_pop[pop]["base_key"]
            methods_all = window_per_pop[pop]["methods_all"]
            df_ylim = df_use.copy()
            df_ylim["err"] = df_ylim["estimate"] - df_ylim["h2"]
            ylims = robust_error_ylim_by_group(df_ylim, "err", group_order=methods_all)

            _plot_panel_error(
                ax=ax,
                df_sub=df_use,
                methods_all=methods_all,
                ylims=ylims,
                base_key=base_key,
                label_map=WINDOW_PRETTY_LABEL,
            )

            if col == 0:
                ax.set_ylabel(
                    "Error (estimate − true $h^2$)", fontsize=AXIS_LABEL_FONTSIZE
                )
            else:
                ax.set_ylabel("")
            ax.set_title("")

        fig.tight_layout()
        out = figdir / "fig_s05_h2_window_sensitivity.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {out}")

    else:
        for pop in pops:
            df_use = window_per_pop[pop]["df"].copy()
            base_key = window_per_pop[pop]["base_key"]
            methods_all = window_per_pop[pop]["methods_all"]
            methods_no_base = window_per_pop[pop]["methods_no_base"]
            rel = window_per_pop[pop]["rel"]
            h2_vals = window_per_pop[pop]["h2_vals"]

            n_h2 = len(h2_vals)
            fig, axes = plt.subplots(
                2, n_h2, figsize=stacked_figsize(n_h2, len(methods_all)), sharey=False
            )
            if n_h2 == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            all_rel_vals_pop = rel[rel["method_window"].isin(methods_no_base)][
                "rel_mse"
            ].to_numpy(float)
            scale = (
                args.yscale if args.yscale != "auto" else auto_yscale(all_rel_vals_pop)
            )
            if pop.startswith("EUR"):
                scale = "linear"

            df_ylim = df_use.copy()
            df_ylim["err"] = df_ylim["estimate"] - df_ylim["h2"]
            err_ylims = robust_error_ylim_by_group(
                df_ylim, "err", group_order=methods_all
            )

            for col, h2v in enumerate(h2_vals):
                ax_top = axes[0, col]
                rel_h2 = rel[_isclose_series(rel["h2"], float(h2v))].copy()
                rel_sum_h2 = summarize_relmse(
                    rel_h2[rel_h2["method_window"].isin(methods_no_base)],
                    methods_no_base,
                )

                _plot_panel_relmse(
                    ax=ax_top,
                    pop=pop,
                    rel_sub=rel_h2,
                    rel_sum_sub=rel_sum_h2,
                    methods_all=methods_all,
                    methods_no_base=methods_no_base,
                    base_key=base_key,
                    yscale=scale,
                    yvals_for_lim=all_rel_vals_pop,
                    rng=rng,
                    label_map=WINDOW_PRETTY_LABEL,
                )
                ax_top.set_title(f"$h^2$ = {float(h2v):g}", fontsize=TITLE_FONTSIZE)
                if col == 0:
                    ax_top.set_ylabel("Relative MSE", fontsize=AXIS_LABEL_FONTSIZE)
                else:
                    ax_top.set_ylabel("")

                ax_bot = axes[1, col]
                df_h2 = df_use[_isclose_series(df_use["h2"], float(h2v))].copy()

                _plot_panel_error(
                    ax=ax_bot,
                    df_sub=df_h2,
                    methods_all=methods_all,
                    ylims=err_ylims,
                    base_key=base_key,
                    label_map=WINDOW_PRETTY_LABEL,
                )
                if col == 0:
                    ax_bot.set_ylabel(
                        "Error (estimate − true $h^2$)", fontsize=AXIS_LABEL_FONTSIZE
                    )
                else:
                    ax_bot.set_ylabel("")
                ax_bot.set_title("")

            fig.tight_layout(rect=[0, 0, 1, 0.97])

            out = (
                figdir
                / f"{pop}_stacked_relmse_error_window_byh2_base_{window_base_tag}.pdf"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"\nSaved: {out}")
