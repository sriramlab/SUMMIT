#!/usr/bin/env python3
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
    "ldsc_50000": "LDSC",
    "sumher_50000": "SumHer\n(GCTA)",
    "sumher_ldak_50000": "SumHer\n(LDAK)",
}

METHOD_COLOR = {
    "rhe": TAB20[0],
    "covsumrhe": TAB20[4],
    "covldsc": TAB20[2],
    "ldsc_50000": TAB20[10],
    "sumher_50000": TAB20[6],
    "sumher_ldak_50000": TAB20[8],
}

DEFAULT_METHODS = [
    "rhe",
    "covldsc",
    "ldsc_50000",
    "covsumrhe",
    "sumher_50000",
    "sumher_ldak_50000",
]

TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
XTICK_LABEL_FONTSIZE = 12
YTICK_LABEL_FONTSIZE = 12


def stacked_figsize(n_cols: int, n_methods: int) -> tuple[float, float]:
    """Compact supplemental figure size; keeps text readable after LaTeX scaling."""
    col_width = 4.8 if n_methods >= 6 else 3.8
    return (max(10.5, col_width * n_cols), 7.4)


# -------------------------
# Helpers
# -------------------------
def nice_label_from_key(k: str) -> str:
    return PRETTY_LABEL.get(k, k)


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
    # allow passing already-resolved keys like ldsc_50000
    if base in available:
        return base

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
            f"Tip: windowed base example: --base ldsc --base_window 50000 (or --base ldsc_50000)"
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


def auto_yscale(rel_vals: np.ndarray) -> str:
    v = rel_vals[np.isfinite(rel_vals) & (rel_vals > 0)]
    if v.size == 0:
        return "linear"
    vmax = float(np.nanmax(v))
    vmin = float(np.nanmin(v))
    if vmax > 100.0 or vmin < 0.33:
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
    """y-lims that won't clip seaborn boxplot whiskers (showfliers=False)."""
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


# -------------------------
# Binary-specific: scale selection
# -------------------------
def select_scale_columns(df: pd.DataFrame, scale: str) -> tuple[str, str, str]:
    """
    Returns (est_col, se_col, true_col_for_mse) in df.
    - scale = "liab": use h2_liab, se_liab, true = h2_true
    - scale = "obs" : use h2_obs, se_obs, true = h2_true / scale_factor  (row-wise)
    """
    scale = scale.lower().strip()
    if scale not in ("liab", "obs"):
        raise ValueError("--scale must be 'liab' or 'obs'")

    if scale == "liab":
        need = ["h2_liab", "se_liab", "h2_true"]
        for c in need:
            if c not in df.columns:
                raise SystemExit(f"ERROR: missing column '{c}' in binary parsed CSV.")
        return "h2_liab", "se_liab", "h2_true"

    # obs scale
    need = ["h2_obs", "se_obs", "h2_true", "scale_factor"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"ERROR: missing column '{c}' in binary parsed CSV.")
    # create true_obs if not already present
    if "h2_true_obs" not in df.columns:
        sf = pd.to_numeric(df["scale_factor"], errors="coerce").to_numpy(float)
        ht = pd.to_numeric(df["h2_true"], errors="coerce").to_numpy(float)
        true_obs = np.full_like(ht, np.nan, dtype=float)
        ok = np.isfinite(ht) & np.isfinite(sf) & (sf != 0)
        true_obs[ok] = ht[ok] / sf[ok]
        df["h2_true_obs"] = true_obs
    return "h2_obs", "se_obs", "h2_true_obs"


# -------------------------
# Metrics
# -------------------------
def compute_relmse_by_setting(
    df: pd.DataFrame,
    base_key: str,
    est_col: str,
    true_col: str,
    setting_col: str,
) -> pd.DataFrame:
    """
    Returns relMSE per (method_window, true_h2, setting):
      mse = mean((estimate - true_h2)^2) over replicates
      rel_mse = mse / mse_base
    """
    rows = []
    for (mw, h2t, s), sub in df.groupby(
        ["method_window", true_col, setting_col], dropna=False
    ):
        est = pd.to_numeric(sub[est_col], errors="coerce").to_numpy(float)
        est = est[np.isfinite(est)]
        true = float(h2t) if pd.notnull(h2t) else np.nan
        mse = (
            float(np.mean((est - true) ** 2))
            if (est.size and np.isfinite(true))
            else np.nan
        )
        rows.append({"method_window": mw, "h2_true": true, setting_col: s, "mse": mse})

    mse_df = pd.DataFrame(rows)
    base = mse_df[mse_df["method_window"] == base_key].copy()
    base = base.rename(columns={"mse": "mse_base"}).drop(columns=["method_window"])

    out = mse_df.merge(base, on=["h2_true", setting_col], how="inner")
    out["rel_mse"] = out["mse"] / out["mse_base"]
    return out[["method_window", "h2_true", setting_col, "rel_mse"]]


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


def compute_mse_bias_true_se_reported_se_by_setting(
    df: pd.DataFrame,
    est_col: str,
    true_col: str,
    setting_col: str,
    se_col: Optional[str],
) -> pd.DataFrame:
    """
    Per (method_window, true_h2, setting) compute over replicates:
      mse      = mean((est - true)^2)
      bias     = mean(est - true)
      true_se  = sd(est)  [empirical across replicates]
      rep_se   = mean(reported_se) if available
    """
    rows = []
    use_rep_se = se_col is not None and se_col in df.columns

    for (m, h2t, s), sub in df.groupby(
        ["method_window", true_col, setting_col], dropna=False
    ):
        est = pd.to_numeric(sub[est_col], errors="coerce").to_numpy(float)
        est = est[np.isfinite(est)]
        true = float(h2t) if pd.notnull(h2t) else np.nan

        if est.size == 0 or not np.isfinite(true):
            rows.append(
                dict(
                    method_window=m,
                    h2_true=true,
                    setting=s,
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
                h2_true=true,
                setting=s,
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
    """Average per-setting metrics into one row per method (each setting equally weighted)."""
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
    "--pops", default="EUR,SAS,AFR", help="Comma-separated population codes"
)
parser.add_argument(
    "--indir",
    default="data/sim_h2/binary",
    help="Input root directory containing {POP}/estimates.csv",
)
parser.add_argument(
    "--figdir", default="figs/supplementary", help="Output figure directory"
)
parser.add_argument(
    "--methods",
    default=",".join(DEFAULT_METHODS),
    help="Comma-separated method keys to show",
)
parser.add_argument(
    "--base",
    default="covsumrhe",
    help="Baseline method for relMSE. Can be explicit key (e.g. covsumrhe, ldsc_50000) "
    "or family (ldsc,sumher,sumher_ldak) with --base_window.",
)
parser.add_argument(
    "--base_window",
    type=int,
    default=50000,
    help="If --base is a family, use this window kb.",
)
parser.add_argument(
    "--yscale",
    choices=["auto", "linear", "log"],
    default="auto",
    help="Y-scale for relMSE panels.",
)
parser.add_argument(
    "--scale",
    choices=["liab", "obs"],
    default="liab",
    help="Which scale to evaluate/plot: liability-scale (liab) or observed-scale (obs).",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
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
metrics_all_rows = []
setting_col = "tag"  # binary parsed CSV uses "tag" as the scenario label (analogous to pcausal label)

for pop in pops:
    f = Path(args.indir) / pop / "estimates.csv"
    if not f.exists():
        raise SystemExit(f"ERROR: missing file: {f}")

    df = pd.read_csv(f)

    # minimal required columns for binary parsed CSV
    req = ["method", "run", "window", "h2_true", setting_col]
    for c in req:
        if c not in df.columns:
            raise SystemExit(f"ERROR: missing column '{c}' in {f}")

    df["window"] = pd.to_numeric(df.get("window"), errors="coerce")
    df["h2_true"] = pd.to_numeric(df.get("h2_true"), errors="coerce")
    df[setting_col] = df[setting_col].astype(str)

    df["method"] = df["method"].astype(str)
    df["method_window"] = [
        method_window_key(m, w)
        for m, w in zip(df["method"].tolist(), df["window"].tolist())
    ]

    # choose estimate/se/true columns for the requested scale
    est_col, se_col, true_col = select_scale_columns(df, scale=args.scale)

    # coerce columns to numeric
    df[est_col] = pd.to_numeric(df[est_col], errors="coerce")
    df[se_col] = pd.to_numeric(df[se_col], errors="coerce")
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce")

    available = set(df["method_window"].unique())
    present_methods = [m for m in method_order if m in available]
    present_methods = [
        m for m in present_methods if m != "sumrhe"
    ]  # keep main-text clean (same as continuous)

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

    rel = compute_relmse_by_setting(
        df_use,
        base_key=base_key,
        est_col=est_col,
        true_col=true_col,
        setting_col=setting_col,
    )

    # Use the same method order in both rows, with the baseline first.
    methods_all = [base_key] + [m for m in present_methods if m != base_key]
    methods_no_base = [m for m in methods_all if m != base_key]  # relMSE plotting only

    rel_sum = summarize_relmse(
        rel[rel["method_window"].isin(methods_no_base)], methods_no_base
    )

    # per-setting metrics + aggregate (include baseline here)
    per_setting_metrics = compute_mse_bias_true_se_reported_se_by_setting(
        df_use,
        est_col=est_col,
        true_col=true_col,
        setting_col=setting_col,
        se_col=se_col,
    )
    metrics_sum = summarize_metrics_across_settings(per_setting_metrics, methods_all)
    metrics_sum["pop"] = pop
    metrics_sum["pretty"] = metrics_sum["method_window"].map(nice_label_from_key)
    metrics_sum["se_col_used"] = se_col
    metrics_sum["est_col_used"] = est_col
    metrics_sum["true_col_used"] = true_col
    metrics_sum["base_key"] = base_key
    metrics_sum["scale"] = args.scale
    metrics_all_rows.append(metrics_sum)

    per_pop[pop] = dict(
        df=df_use,
        base_key=base_key,
        present_methods=present_methods,
        methods_all=methods_all,
        methods_no_base=methods_no_base,
        rel=rel,
        rel_sum=rel_sum,
        metrics_sum=metrics_sum,
        est_col=est_col,
        se_col=se_col,
        true_col=true_col,
    )

    # print summaries
    tbl = rel_sum.copy()
    tbl["pretty"] = tbl["method_window"].map(nice_label_from_key)
    tbl = tbl[["pretty", "n", "gmean", "median", "min", "max"]]
    print(
        f"\n=== {pop}: relMSE aggregated across settings (vs {nice_label_from_key(base_key)}) [{args.scale}] ==="
    )
    print(tbl.to_string(index=False))

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
            "est_col_used",
        ]
    ]
    print(f"\n=== {pop}: MSE / bias / SE (averaged across settings) [{args.scale}] ===")
    print(mt.to_string(index=False))

# save combined metrics CSV
metrics_all = (
    pd.concat(metrics_all_rows, ignore_index=True)
    if metrics_all_rows
    else pd.DataFrame()
)
metrics_out = figdir / f"fig_s08_binary_h2_metrics_{args.scale}.csv"
metrics_all.to_csv(metrics_out, index=False)
print(f"\nSaved metrics CSV: {metrics_out}")


# -------------------------
# Combined Figure: 2 x N_pops (relMSE + pooled error)
# -------------------------
n_pops = len(pops)
max_methods = max(len(per_pop[p]["methods_all"]) for p in pops)
fig, axes = plt.subplots(
    2, n_pops, figsize=stacked_figsize(n_pops, max_methods), sharey=False
)
if n_pops == 1:
    axes = np.array([[axes[0]], [axes[1]]])

# ---- Row 1: relMSE (baseline shown as a diamond at 1) ----
for col, pop in enumerate(pops):
    ax = axes[0, col]
    base_key = per_pop[pop]["base_key"]
    methods_all = per_pop[pop]["methods_all"]
    methods_no_base = per_pop[pop]["methods_no_base"]
    rel = per_pop[pop]["rel"]
    rel_sum = per_pop[pop]["rel_sum"]

    n = len(methods_all)
    x = np.arange(n, dtype=float)

    all_rel_vals = rel[rel["method_window"].isin(methods_no_base)]["rel_mse"].to_numpy(
        float
    )
    scale = args.yscale if args.yscale != "auto" else auto_yscale(all_rel_vals)
    if pop.startswith("EUR"):
        scale = "linear"

    for i, m in enumerate(methods_all):
        if m == base_key:
            ax.scatter(
                [x[i]], [1.0], s=90, marker="D", color=color_from_key(m), zorder=4
            )
            continue

        sub = rel[rel["method_window"] == m]["rel_mse"].to_numpy(float)
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

        row = rel_sum[rel_sum["method_window"] == m]
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
    ax.set_title(pop, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(
        "Relative MSE" if col == 0 else "",
        fontsize=AXIS_LABEL_FONTSIZE if col == 0 else 12,
    )

    ax.set_xticks(x)
    ticklabs = [nice_label_from_key(m) for m in methods_all]
    ax.set_xticklabels(ticklabs, rotation=0, ha="center")

    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=XTICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)
    for t in ax.get_xticklabels():
        if t.get_text() == nice_label_from_key(base_key):
            t.set_fontweight("bold")

    if scale == "log":
        ax.set_yscale("log")
        set_log_ylim_with_pad(ax, all_rel_vals)
    else:
        ax.set_yscale("linear")
        set_linear_ylim_with_pad(ax, all_rel_vals)

# ---- Row 2: pooled error (estimate − true) ----
for col, pop in enumerate(pops):
    ax = axes[1, col]
    df_use = per_pop[pop]["df"].copy()
    methods_all = per_pop[pop]["methods_all"]
    est_col = per_pop[pop]["est_col"]
    true_col = per_pop[pop]["true_col"]

    n = len(methods_all)
    x = np.arange(n, dtype=float)

    err = df_use[df_use["method_window"].isin(methods_all)].copy()
    err["err"] = pd.to_numeric(err[est_col], errors="coerce") - pd.to_numeric(
        err[true_col], errors="coerce"
    )
    err["pretty"] = err["method_window"].map(nice_label_from_key)

    order_pretty = [nice_label_from_key(m) for m in methods_all]
    palette = {nice_label_from_key(m): color_from_key(m) for m in methods_all}

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
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        "Error (estimate − true $h^2$)" if col == 0 else "",
        fontsize=AXIS_LABEL_FONTSIZE if col == 0 else 12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(order_pretty, rotation=0, ha="center")
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=XTICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)

    for t in ax.get_xticklabels():
        if t.get_text() == "SUMMIT":
            t.set_fontweight("bold")

    ax.set_ylim(*robust_error_ylim(err["err"].to_numpy(float)))

# Layout + save
fig.tight_layout()
out = figdir / "fig_s08_binary_h2_simulations.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out}")
