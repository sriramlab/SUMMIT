#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib import colormaps
from matplotlib.colors import to_hex

TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]

METHOD_COLOR = {
    "rhe": TAB20[0],
    "covsumrhe": TAB20[4],
    "covldsc": TAB20[2],
    "ldsc_50Mb": TAB20[10],
    "sumher_50Mb": TAB20[6],
    "sumher_ldak_50Mb": TAB20[8],
    "ldsc_20Mb": TAB20[10],
    "sumher_20Mb": TAB20[6],
    "sumher_ldak_20Mb": TAB20[8],
    # (optional) if you ever plot these too:
    # "sumrhe": TAB20[1],
    # "ldsc_2Mb": TAB20[11],
    # "sumher_2Mb": TAB20[7],
    # "sumher_ldak_2Mb": TAB20[9],
}

TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12


# -------------------- Methods & labels --------------------
METHODS_ALL = {"rhe", "sumrhe", "covsumrhe", "ldsc", "covldsc", "sumher", "sumher_ldak"}
WINDOWED_METHODS = {"ldsc", "sumher", "sumher_ldak"}

WINDOWS_KEEP_DEFAULT = {2000, 20000, 50000}  # 2Mb, 20Mb, 50Mb (kb)
WINDOWS_KEEP_BEST = {20000}  # 20Mb only

# Place cov-LDSC after LDSC_50Mb.
LABEL_ORDER = [
    "rhe",
    "sumrhe",
    "covsumrhe",
    "sumher_2Mb",
    "sumher_ldak_2Mb",
    "ldsc_2Mb",
    "sumher_20Mb",
    "sumher_ldak_20Mb",
    "ldsc_20Mb",
    "sumher_50Mb",
    "sumher_ldak_50Mb",
    "ldsc_50Mb",
    "covldsc",
]

# Main-text minimalist method set & order
MAIN_TEXT_KEEP = {
    "rhe",  # RHE-mc
    "covsumrhe",  # SUMMIT-cov (but displayed as SUMMIT in main text)
    "covldsc",  # cov-LDSC
    "ldsc_20Mb",  # LDSC 20Mb
    "sumher_20Mb",  # SumHer-GCTA 20Mb
    "sumher_ldak_20Mb",  # SumHer-LDAK 20Mb
}
MAIN_TEXT_ORDER = [
    "covsumrhe",  # show as SUMMIT
    "covldsc",
    "rhe",
    "ldsc_20Mb",
    "sumher_20Mb",
    "sumher_ldak_20Mb",
]

PRETTY_LABEL = {
    "rhe": "RHE-mc",
    "sumrhe": "SUMMIT",
    "covsumrhe": "SUMMIT-cov",
    "covldsc": "cov-LDSC",
    "sumher_2Mb": "SumHer-GCTA (2Mb)",
    "sumher_ldak_2Mb": "SumHer-LDAK (2Mb)",
    "ldsc_2Mb": "LDSC (2Mb)",
    "sumher_20Mb": "SumHer-GCTA (20Mb)",
    "sumher_ldak_20Mb": "SumHer-LDAK (20Mb)",
    "ldsc_20Mb": "LDSC (20Mb)",
    "sumher_50Mb": "SumHer-GCTA (50Mb)",
    "sumher_ldak_50Mb": "SumHer-LDAK (50Mb)",
    "ldsc_50Mb": "LDSC (50Mb)",
}


def pretty_methods(ms):
    return [PRETTY_LABEL.get(m, m) for m in ms]


STYLE = {
    # Base style; emphasis handled in _draw()
    "sumrhe": ("-", "o"),
    "covsumrhe": ("-", "P"),
    "covldsc": ("-", "X"),
    "rhe": ("--", "s"),
    "sumher_2Mb": ("-", "^"),
    "sumher_ldak_2Mb": ("-", "v"),
    "ldsc_2Mb": ("-", "D"),
    "sumher_20Mb": ("-", "^"),
    "sumher_ldak_20Mb": ("-", "v"),
    "ldsc_20Mb": ("-", "D"),
    "sumher_50Mb": (":", "^"),
    "sumher_ldak_50Mb": (":", "v"),
    "ldsc_50Mb": (":", "D"),
}


def get_pretty_label_map(main_text: bool) -> Dict[str, str]:
    """
    Plot-time label mapping.
    For --main-text: show covsumrhe as "SUMMIT" (and you won't plot sumrhe anyway).
    """
    m = dict(PRETTY_LABEL)
    if main_text:
        m["covsumrhe"] = "SUMMIT"
    return m


# -------------------- Helpers --------------------
def restrict_methods_and_windows(
    df: pd.DataFrame, best_results: bool = False
) -> pd.DataFrame:
    """
    Keep only methods we care about and, for windowed methods, only selected windows.

    If best_results=True: keep only 20Mb for windowed methods.
    Else: keep 2Mb, 20Mb, and 50Mb.
    """
    df = df[df["method"].isin(METHODS_ALL)].copy()
    if "window" not in df.columns:
        df["window"] = -1

    windows_keep = WINDOWS_KEEP_BEST if best_results else WINDOWS_KEEP_DEFAULT
    has_window = df["method"].isin(WINDOWED_METHODS)

    keep = pd.concat(
        [
            df[~has_window],
            df[has_window & df["window"].isin(windows_keep)],
        ],
        ignore_index=True,
    )
    return keep


def method_label(row):
    """Map (method, window) -> plotting label."""
    m = str(row["method"])
    w = int(row.get("window", -1)) if not pd.isna(row.get("window", -1)) else -1

    # windowed methods
    if m == "ldsc":
        if w == 2000:
            return "ldsc_2Mb"
        if w == 20000:
            return "ldsc_20Mb"
        if w == 50000:
            return "ldsc_50Mb"
        return None
    if m == "sumher":
        if w == 2000:
            return "sumher_2Mb"
        if w == 20000:
            return "sumher_20Mb"
        if w == 50000:
            return "sumher_50Mb"
        return None
    if m == "sumher_ldak":
        if w == 2000:
            return "sumher_ldak_2Mb"
        if w == 20000:
            return "sumher_ldak_20Mb"
        if w == 50000:
            return "sumher_ldak_50Mb"
        return None

    # non-windowed methods
    if m in ("rhe", "sumrhe", "covsumrhe", "covldsc"):
        return m

    return None


def attach_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method_label"] = df.apply(method_label, axis=1)
    return df[~df["method_label"].isna()].copy()


def to_long(
    df: pd.DataFrame, prefix: str, out_col: str, se_col: str, allow_missing_se=False
) -> pd.DataFrame:
    """
    Wide -> long for enrichment columns.

    Expects columns:
      enr_0, enr_1, enr_se_0, enr_se_1   (prefix='enr')
    """
    if df.empty:
        return df.copy()

    B = int(df["num_bins"].mode().iat[0])

    base_keep = [
        "pop",
        "true_h2",
        "arch",
        "scenario",
        "method",
        "method_label",
        "run",
        "num_bins",
        "p_causal",
        "window",
    ]
    keep = [c for c in base_keep if c in df.columns]

    val_cols = [f"{prefix}_{j}" for j in range(B)]
    se_cols = [f"{prefix}_se_{j}" for j in range(B)]

    for c in val_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing value column: {c}")
    if allow_missing_se:
        for c in se_cols:
            if c not in df.columns:
                df[c] = np.nan
    else:
        missing = [c for c in se_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing SE columns: {missing[:5]} ...")

    sub = df[keep + val_cols + se_cols].copy()

    v = sub.melt(
        id_vars=keep, value_vars=val_cols, var_name="bin_v", value_name=out_col
    )
    v["bin"] = v["bin_v"].str.split("_").str[-1].astype(int)
    v.drop(columns=["bin_v"], inplace=True)

    s = sub.melt(id_vars=keep, value_vars=se_cols, var_name="bin_s", value_name=se_col)
    s["bin"] = s["bin_s"].str.split("_").str[-1].astype(int)
    s.drop(columns=["bin_s"], inplace=True)

    return v.merge(s, on=keep + ["bin"], how="inner")


# -------------------- ROC / PR utilities --------------------
def _safe_array(x):
    a = np.asarray(x, float)
    return a[np.isfinite(a)]


def roc_curve_from_scores(scores: np.ndarray, labels: np.ndarray):
    """Simple ROC from continuous scores and binary labels."""
    s = _safe_array(scores)
    y = _safe_array(labels).astype(int)
    m = min(len(s), len(y))
    if m == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    s, y = s[:m], y[:m]
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / P
    fpr = fp / N
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr


def pr_curve_from_scores(scores: np.ndarray, labels: np.ndarray):
    """Precision–Recall curve (positive = 1)."""
    s = _safe_array(scores)
    y = _safe_array(labels).astype(int)
    m = min(len(s), len(y))
    if m == 0:
        return None, None
    s, y = s[:m], y[:m]
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    P = yy.sum()
    if P == 0:
        return None, None
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P
    R = np.concatenate([[0.0], recall, [1.0]])
    Pn = np.concatenate([[1.0], precision, [precision[-1]]])
    orderR = np.argsort(R)
    return R[orderR], Pn[orderR]


def _interp_on_grid(x, y, grid):
    if x is None or y is None or len(x) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    if not np.all(np.diff(x) >= 0):
        ordx = np.argsort(x)
        x = x[ordx]
        y = y[ordx]
    yi = np.interp(grid, x, y, left=y[0], right=y[-1])
    yi[~np.isfinite(yi)] = np.nan
    return yi


# -------------------- Resampling pooled curves (balanced) --------------------
StratumKey = Tuple[float, float]  # (true_h2, p_causal)


def _quantile_nan(a: np.ndarray, q: float, axis: int = 0):
    """np.nanquantile wrapper that tolerates all-nan slices."""
    with np.errstate(all="ignore"):
        return np.nanquantile(a, q, axis=axis)


def _sample_balanced_groups(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    group_cols: List[str],
    replace_if_needed: bool,
) -> pd.DataFrame:
    """
    Sample ~evenly across groups defined by group_cols.
    Falls back gracefully if groups are missing/imbalanced.
    """
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()

    # Sample directly when no grouping is requested.
    if not group_cols:
        rep = replace_if_needed and (len(df) < n)
        idx = rng.choice(df.index.to_numpy(), size=n, replace=rep)
        return df.loc[idx].copy()

    if any(c not in df.columns for c in group_cols):
        rep = replace_if_needed and (len(df) < n)
        idx = rng.choice(df.index.to_numpy(), size=n, replace=rep)
        return df.loc[idx].copy()

    groups = list(df.groupby(group_cols, dropna=False))
    G = len(groups)
    if G <= 1:
        rep = replace_if_needed and (len(df) < n)
        idx = rng.choice(df.index.to_numpy(), size=n, replace=rep)
        return df.loc[idx].copy()

    base = n // G
    rem = n % G

    chosen_idx = []
    order = rng.permutation(G)
    for t, gi in enumerate(order):
        _, g = groups[gi]
        k = base + (1 if t < rem else 0)
        if k <= 0:
            continue
        rep = replace_if_needed and (len(g) < k)
        chosen_idx.extend(rng.choice(g.index.to_numpy(), size=k, replace=rep).tolist())

    if len(chosen_idx) < n:
        remaining = df.index.difference(pd.Index(chosen_idx))
        need = n - len(chosen_idx)
        if len(remaining) > 0:
            take = min(need, len(remaining))
            chosen_idx.extend(
                rng.choice(remaining.to_numpy(), size=take, replace=False).tolist()
            )
        if len(chosen_idx) < n and replace_if_needed:
            need = n - len(chosen_idx)
            chosen_idx.extend(
                rng.choice(df.index.to_numpy(), size=need, replace=True).tolist()
            )

    return df.loc[pd.Index(chosen_idx)].copy()


def build_balanced_curves(
    enr_long: pd.DataFrame,
    fpr_grid: np.ndarray,
    rec_grid: np.ndarray,
    *,
    n_resamples: int = 200,
    seed: int = 0,
    balance_cols: Optional[List[str]] = None,
    replace_if_needed: bool = False,
    verbose: bool = False,
):
    """
    enr_long must contain (at minimum):
      pop, method_label, z, y, true_h2, p_causal
    Optional:
      arch, scenario  (used if in balance_cols)

    We resample positives to match negatives *within each (true_h2, p_causal) stratum*,
    concatenate across strata, and compute ROC/PR for each resample.
    Then return mean + quantile bands across resamples.
    """
    if balance_cols is None:
        balance_cols = []

    mean_roc: Dict[Tuple[str, str], dict] = {}
    mean_pr: Dict[Tuple[str, str], dict] = {}

    keys = sorted(enr_long.groupby(["pop", "method_label"], dropna=False).groups.keys())

    rng_master = np.random.default_rng(seed)

    for pop, method_label in keys:
        g = enr_long[
            (enr_long["pop"] == pop) & (enr_long["method_label"] == method_label)
        ].copy()
        if g.empty:
            continue

        strata = []
        for (h2, pc), gg in g.groupby(["true_h2", "p_causal"], dropna=False):
            gg = gg[np.isfinite(gg["z"]) & np.isfinite(gg["y"])].copy()
            if gg.empty:
                continue

            pos = gg[gg["y"].astype(int) == 1].copy()
            neg = gg[gg["y"].astype(int) == 0].copy()
            if len(pos) == 0 or len(neg) == 0:
                continue

            strata.append(((float(h2), float(pc)), pos, neg))

        if not strata:
            if verbose:
                print(f"[warn] no usable strata for {pop}/{method_label}")
            continue

        roc_stack, pr_stack = [], []
        auroc_list, aupr_list, n_list = [], [], []

        sub_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)

        for _ in range(int(n_resamples)):
            scores_all, labels_all = [], []

            for (_, _), pos, neg in strata:
                n_neg = len(neg)
                n_pos = len(pos)
                n_take = min(n_neg, n_pos)
                if n_take <= 0:
                    continue

                if n_neg == n_take:
                    neg_s = neg
                else:
                    idx = rng.choice(neg.index.to_numpy(), size=n_take, replace=False)
                    neg_s = neg.loc[idx]

                pos_s = _sample_balanced_groups(
                    pos,
                    n_take,
                    rng,
                    group_cols=balance_cols,
                    replace_if_needed=replace_if_needed,
                )

                scores_all.append(pos_s["z"].to_numpy(float))
                labels_all.append(np.ones(len(pos_s), dtype=int))
                scores_all.append(neg_s["z"].to_numpy(float))
                labels_all.append(np.zeros(len(neg_s), dtype=int))

            if not scores_all:
                continue

            scores = np.concatenate(scores_all)
            labels = np.concatenate(labels_all)
            mask = np.isfinite(scores) & np.isfinite(labels)
            scores = scores[mask]
            labels = labels[mask].astype(int)

            if scores.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
                continue

            fpr, tpr = roc_curve_from_scores(scores, labels)
            R, P = pr_curve_from_scores(scores, labels)

            auroc = float(np.trapz(tpr, fpr)) if fpr is not None else np.nan
            aupr = (
                float(np.trapz(P, R)) if (R is not None and P is not None) else np.nan
            )

            roc_y = _interp_on_grid(fpr, tpr, fpr_grid)
            pr_y = (
                _interp_on_grid(R, P, rec_grid)
                if (R is not None and P is not None)
                else np.full_like(rec_grid, np.nan)
            )

            roc_stack.append(roc_y)
            pr_stack.append(pr_y)
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            n_list.append(int(scores.size))

        if len(roc_stack) == 0:
            if verbose:
                print(f"[warn] no resamples produced curves for {pop}/{method_label}")
            continue

        roc_arr = np.vstack(roc_stack)
        pr_arr = np.vstack(pr_stack)

        roc_mean = np.nanmean(roc_arr, axis=0)
        pr_mean = np.nanmean(pr_arr, axis=0)

        roc_lo = _quantile_nan(roc_arr, 0.05, axis=0)
        roc_hi = _quantile_nan(roc_arr, 0.95, axis=0)
        pr_lo = _quantile_nan(pr_arr, 0.05, axis=0)
        pr_hi = _quantile_nan(pr_arr, 0.95, axis=0)

        mean_roc[(pop, method_label)] = {
            "x": fpr_grid,
            "mean": roc_mean,
            "lo": roc_lo,
            "hi": roc_hi,
            "n": int(np.nanmedian(np.asarray(n_list, float))),
            "auroc_mean": float(np.nanmean(auroc_list)),
            "auroc_sd": float(np.nanstd(auroc_list)),
            "n_resamples": int(len(auroc_list)),
        }
        mean_pr[(pop, method_label)] = {
            "x": rec_grid,
            "mean": pr_mean,
            "lo": pr_lo,
            "hi": pr_hi,
            "n": int(np.nanmedian(np.asarray(n_list, float))),
            "aupr_mean": float(np.nanmean(aupr_list)),
            "aupr_sd": float(np.nanstd(aupr_list)),
            "n_resamples": int(len(aupr_list)),
        }

    return mean_roc, mean_pr


# -------------------- Dodging utilities (unchanged) --------------------
def _compute_dodge_base(xs, dodge_frac=0.5):
    xs = np.array(sorted(xs), dtype=float)
    if xs.size >= 2:
        diffs = np.diff(xs)
        pos = diffs[diffs > 0]
        min_gap = float(np.min(pos)) if pos.size else 0.0
    else:
        min_gap = max(1e-3, float(xs[0]) * 0.1) if xs.size else 1e-3
    base_delta = (min_gap if min_gap > 0 else 1e-3) * float(dodge_frac)
    return base_delta


def _collision_offsets_on_grid(
    y_by_method,
    methods_ord,
    xs,
    base_delta,
    y_thresh=0.02,
    anchor_method=None,
    anchor_position="center",
    smooth=True,
):
    offsets = {}
    xs = np.asarray(xs, float)
    L = len(xs)
    for i in range(L):
        pairs = []
        for m in methods_ord:
            yi = y_by_method.get(m, None)
            if yi is None:
                continue
            val = yi[i]
            if np.isfinite(val):
                pairs.append((m, float(val)))
        if len(pairs) <= 1:
            for m, _ in pairs:
                offsets[(m, i)] = 0.0
            continue

        pairs.sort(key=lambda t: t[1])
        clusters = []
        cur = [pairs[0]]
        for j in range(1, len(pairs)):
            if abs(pairs[j][1] - pairs[j - 1][1]) <= y_thresh:
                cur.append(pairs[j])
            else:
                clusters.append(cur)
                cur = [pairs[j]]
        clusters.append(cur)

        for cluster in clusters:
            k = len(cluster)
            if k == 1:
                offsets[(cluster[0][0], i)] = 0.0
                continue
            base = np.linspace(-base_delta, base_delta, k)
            if anchor_method and any(m == anchor_method for m, _ in cluster):
                anchor_idx = [
                    idx for idx, (m, _) in enumerate(cluster) if m == anchor_method
                ][0]
                if anchor_position == "right":
                    shift = (k - 1) - anchor_idx
                elif anchor_position == "left":
                    shift = 0 - anchor_idx
                else:
                    zero_idx = int(np.argmin(np.abs(base)))
                    shift = zero_idx - anchor_idx
                base = np.roll(base, shift)
            for idx, (m, _) in enumerate(cluster):
                offsets[(m, i)] = float(base[idx])

    if smooth and L >= 3:
        for m in methods_ord:
            vals = np.array([offsets.get((m, i), 0.0) for i in range(L)], dtype=float)
            sm = vals.copy()
            for i in range(L):
                lo = max(0, i - 1)
                hi = min(L - 1, i + 1)
                sm[i] = np.mean(vals[lo : hi + 1])
            for i in range(L):
                key = (m, i)
                if key in offsets:
                    offsets[key] = float(sm[i])

    return offsets


# -------------------- Plotting --------------------
def make_2x3_curves(
    mean_roc: dict,
    mean_pr: dict,
    pops_order,
    outfile: str,
    *,
    highlight_method="sumrhe",
    ribbon_alpha=0.10,
    dodge=True,
    dodge_frac=0.45,
    y_thresh_roc=0.02,
    y_thresh_pr=0.02,
    roc_anchor="center",
    pr_anchor="right",
    suptitle: Optional[str] = None,
    compact_subplot_titles: bool = False,
    label_order_override: Optional[List[str]] = None,
    pretty_label_map: Optional[Dict[str, str]] = None,
):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    sns.set_style("whitegrid")

    if pretty_label_map is None:
        pretty_label_map = PRETTY_LABEL

    present = sorted({m for (_, m) in list(mean_roc.keys()) + list(mean_pr.keys())})

    base_ref = label_order_override if label_order_override is not None else LABEL_ORDER
    base_order = [m for m in base_ref if m in present]
    rest = [m for m in present if m not in set(base_order)]
    order = base_order + rest

    # Fixed, pre-specified colors (fallback to tab20 if missing)
    palette = sns.color_palette(n_colors=20)
    fallback = {m: palette[i % len(palette)] for i, m in enumerate(order)}

    color_map = {}
    for i, m in enumerate(order):
        if m in METHOD_COLOR:
            color_map[m] = METHOD_COLOR[m]
        else:
            color_map[m] = fallback[m]

    # Always highlight SUMMIT + SUMMIT-cov; also allow an extra highlight via CLI.
    highlight = {"sumrhe", "covsumrhe"}
    if highlight_method:
        highlight.add(str(highlight_method))

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), sharex="col", sharey=False)

    def pop_title(p):
        return "EUR" if p == "EUR_hdl" else p

    def _draw(ax, x, mean, lo, hi, method, x_offsets=None):
        ls, mk = STYLE.get(method, ("-", None))
        col = color_map[method]
        is_hi = method in highlight

        if is_hi:
            lw = 2.7
            alpha = 0.92
            ms = 6.2
            markevery = 20
            path_fx = [
                pe.Stroke(linewidth=lw + 1.1, foreground="white", alpha=0.65),
                pe.Normal(),
            ]
            mfc = col
            mec = col
            mew = 0.0
            zord = 5
        else:
            lw = 1.75
            alpha = 0.42
            ms = 4.2
            markevery = 28
            path_fx = None
            mfc = "none"
            mec = col
            mew = 1.05
            zord = 3

        x_shift = np.clip(x + (x_offsets if x_offsets is not None else 0.0), 0.0, 1.0)

        if ribbon_alpha > 0.0 and lo is not None and hi is not None:
            ax.fill_between(
                x_shift,
                lo,
                hi,
                color=col,
                alpha=ribbon_alpha,
                linewidth=0,
                zorder=zord - 1,
            )

        ax.plot(
            x_shift,
            mean,
            linestyle=ls,
            marker=mk,
            markevery=markevery,
            markersize=ms,
            linewidth=lw,
            color=col,
            zorder=zord,
            path_effects=path_fx,
            label=pretty_label_map.get(method, method),
            alpha=alpha,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markeredgewidth=mew,
        )

    # --- Top row: ROC
    for col, pop in enumerate(pops_order[:3]):
        ax = axes[0, col]
        methods_here = [m for m in order if (pop, m) in mean_roc]
        if (not pop) or (not methods_here):
            ax.set_visible(False)
            continue

        xs = mean_roc[(pop, methods_here[0])]["x"]
        x_base_delta = _compute_dodge_base(xs, dodge_frac=dodge_frac) if dodge else 0.0
        y_map = {m: mean_roc[(pop, m)]["mean"] for m in methods_here}

        anchor = (
            "sumrhe"
            if "sumrhe" in methods_here
            else ("covsumrhe" if "covsumrhe" in methods_here else None)
        )

        offsets = (
            _collision_offsets_on_grid(
                y_map,
                methods_here,
                xs,
                x_base_delta,
                y_thresh=y_thresh_roc,
                anchor_method=anchor,
                anchor_position=roc_anchor,
                smooth=True,
            )
            if dodge
            else {}
        )

        for m in order:
            if (pop, m) not in mean_roc:
                continue
            d = mean_roc[(pop, m)]
            xoff = (
                np.array([offsets.get((m, i), 0.0) for i in range(len(d["x"]))])
                if dodge
                else None
            )
            _draw(
                ax,
                d["x"],
                d["mean"],
                d.get("lo", None),
                d.get("hi", None),
                m,
                x_offsets=xoff,
            )

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if compact_subplot_titles:
            ax.set_title(f"{pop_title(pop)} — ROC", fontsize=TITLE_FONTSIZE)
        else:
            ax.set_title(
                f"{pop_title(pop)} — ROC (balanced resampling)", fontsize=TITLE_FONTSIZE
            )
        ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(
            "True Positive Rate" if col == 0 else "", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    # --- Bottom row: PR
    for col, pop in enumerate(pops_order[:3]):
        ax = axes[1, col]
        methods_here = [m for m in order if (pop, m) in mean_pr]
        if (not pop) or (not methods_here):
            ax.set_visible(False)
            continue

        xs = mean_pr[(pop, methods_here[0])]["x"]
        x_base_delta = _compute_dodge_base(xs, dodge_frac=dodge_frac) if dodge else 0.0
        y_map = {m: mean_pr[(pop, m)]["mean"] for m in methods_here}

        anchor = (
            "sumrhe"
            if "sumrhe" in methods_here
            else ("covsumrhe" if "covsumrhe" in methods_here else None)
        )

        offsets = (
            _collision_offsets_on_grid(
                y_map,
                methods_here,
                xs,
                x_base_delta,
                y_thresh=y_thresh_pr,
                anchor_method=anchor,
                anchor_position=pr_anchor,
                smooth=True,
            )
            if dodge
            else {}
        )

        for m in order:
            if (pop, m) not in mean_pr:
                continue
            d = mean_pr[(pop, m)]
            xoff = (
                np.array([offsets.get((m, i), 0.0) for i in range(len(d["x"]))])
                if dodge
                else None
            )
            _draw(
                ax,
                d["x"],
                d["mean"],
                d.get("lo", None),
                d.get("hi", None),
                m,
                x_offsets=xoff,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if compact_subplot_titles:
            ax.set_title(f"{pop_title(pop)} — PR", fontsize=TITLE_FONTSIZE)
        else:
            ax.set_title(
                f"{pop_title(pop)} — PR (balanced resampling)", fontsize=TITLE_FONTSIZE
            )
        ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Precision" if col == 0 else "", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    # ---- Global legend ----
    handles = []
    for m in order:
        ls, mk = STYLE.get(m, ("-", None))
        colr = color_map[m]
        is_hi = m in {"sumrhe", "covsumrhe"} or (
            highlight_method and m == str(highlight_method)
        )

        if is_hi:
            lw = 2.9
            ms = 6.0
            alpha = 0.95
            mfc = colr
            mec = colr
            mew = 0.0
        else:
            lw = 1.7
            ms = 4.2
            alpha = 0.65
            mfc = "none"
            mec = colr
            mew = 1.0

        h = Line2D(
            [0],
            [0],
            color=colr,
            linestyle=ls,
            marker=mk,
            linewidth=lw,
            markersize=ms,
            alpha=alpha,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markeredgewidth=mew,
            label=pretty_label_map.get(m, m),
        )
        handles.append(h)

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center left",
        bbox_to_anchor=(0.89, 0.5),
        frameon=True,
        title=None,
        fontsize=LEGEND_FONTSIZE,
    )

    # Suptitle handling:
    #   suptitle is None -> use default
    #   suptitle == ""   -> no suptitle
    #   else             -> use provided
    if suptitle is None:
        suptitle = "ROC & PR curves by population (coding: non-null vs null; balanced resampling)"

    if suptitle != "":
        fig.suptitle(suptitle, fontsize=TITLE_FONTSIZE, y=0.98)
        rect = (0, 0, 0.88, 0.96)
    else:
        rect = (0, 0, 0.88, 1.0)

    fig.tight_layout(rect=rect)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved {outfile}")


# -------------------- IO / preprocessing --------------------
def _read_one_csv(pop: str, base_dir: str) -> Optional[pd.DataFrame]:
    csv = os.path.join(base_dir, pop, "estimates.csv")
    if not os.path.exists(csv):
        print(f"[warn] missing {csv}; skipping {pop}")
        return None
    df = pd.read_csv(csv)
    if df.empty:
        print(f"[warn] empty {csv}; skipping {pop}")
        return None
    df["pop"] = pop
    return df


def _basic_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["window", "num_bins", "p_causal", "true_h2", "run"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "arch" in df.columns:
        df["arch"] = df["arch"].astype(str).str.lower()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str)

    if "window" in df.columns:
        df["window"] = df["window"].fillna(-1).astype(int)
    else:
        df["window"] = -1

    if "num_bins" in df.columns and df["num_bins"].notna().any():
        df["num_bins"] = df["num_bins"].fillna(df["num_bins"].mode().iat[0]).astype(int)

    return df


def _prep_long_for_scores(
    df: pd.DataFrame,
    *,
    sim_type: str,
    best_results: bool,
    true_h2_vals: Optional[List[float]],
    p_causals: Optional[List[float]],
    archs: Optional[List[str]],
    scenarios: Optional[List[str]],
    coding_bin: int,
) -> pd.DataFrame:
    """
    Returns long df with columns:
      pop, method_label, z, y, true_h2, p_causal, [arch, scenario]
    where:
      y = 1 for nonnull (positive), 0 for null (negative),
      and only coding_bin rows are kept.
    """
    df = _basic_types(df)

    # method/window filtering & labeling
    df = restrict_methods_and_windows(df, best_results=best_results)
    df = attach_method_labels(df)
    if df.empty:
        return df

    B = int(df["num_bins"].mode().iat[0])
    if B != 2:
        print(
            f"[warn] pop {df['pop'].iat[0] if 'pop' in df.columns else '??'}: expected num_bins=2, got {B}; skipping."
        )
        return df.iloc[0:0].copy()

    # Filters shared
    if true_h2_vals is not None and "true_h2" in df.columns:
        df = df[df["true_h2"].isin(true_h2_vals)].copy()
    if p_causals is not None and "p_causal" in df.columns:
        df = df[df["p_causal"].isin(p_causals)].copy()

    # Filters that only exist for non-null (arch/scenario)
    if archs is not None and "arch" in df.columns:
        archs_norm = [a.lower() for a in archs]
        df = df[df["arch"].isin(archs_norm)].copy()
    if scenarios is not None and "scenario" in df.columns:
        df = df[df["scenario"].isin(scenarios)].copy()

    if df.empty:
        return df

    # long-format enrichment
    if not any(c.startswith("enr_") for c in df.columns):
        print(f"[warn] enrichment columns missing for pop {df['pop'].iat[0]}; skipping")
        return df.iloc[0:0].copy()

    enr_long = to_long(df, "enr", "enrichment", "enrichment_se", allow_missing_se=True)
    if enr_long.empty:
        return enr_long

    # Keep coding bin only
    enr_long = enr_long[enr_long["bin"].astype(int) == int(coding_bin)].copy()
    if enr_long.empty:
        return enr_long

    # z = (enrichment_hat - 1) / SE
    est = enr_long["enrichment"].to_numpy(float)
    se = enr_long["enrichment_se"].to_numpy(float)
    z = np.full_like(est, np.nan, dtype=float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    z[ok] = (est[ok] - 1.0) / se[ok]
    enr_long["z"] = z

    enr_long = enr_long[enr_long["method_label"].notna()].copy()
    enr_long = enr_long[np.isfinite(enr_long["z"])].copy()
    if enr_long.empty:
        return enr_long

    # Label by sim type
    if sim_type == "nonnull":
        enr_long["y"] = 1
    elif sim_type == "null":
        enr_long["y"] = 0
    else:
        raise ValueError("sim_type must be 'nonnull' or 'null'")

    enr_long["sim_type"] = sim_type

    # Ensure arch/scenario exist so balancing code can rely on columns if desired
    if "arch" not in enr_long.columns:
        enr_long["arch"] = "na"
    if "scenario" not in enr_long.columns:
        enr_long["scenario"] = "na"

    return enr_long


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Average ROC/PR curves (2-bin contig_h2 sims) using PI’s suggested comparison:\n"
            "coding-bin scores from non-null sims (positive) vs coding-bin scores from null sims (negative),\n"
            "with stratified balanced resampling within (true_h2, p_causal)."
        )
    )
    ap.add_argument("--pops", nargs="*", default=["EUR", "SAS", "AFR"])

    ap.add_argument(
        "--outs_base_nonnul",
        default="data/sim_h2/coding",
        help="Base dir for NON-NULL sims; expects pop/estimates.csv",
    )
    ap.add_argument(
        "--outs_base_null",
        default="data/sim_h2/coding_null",
        help="Base dir for NULL sims; expects pop/estimates.csv",
    )

    # Main-text simplification
    ap.add_argument(
        "--main-text",
        action="store_true",
        help=(
            "Main-text mode: only plot {LDSC,SumHer,SumHer-LDAK}@50Mb + RHE-mc + cov-LDSC + SUMMIT-cov, "
            "and display SUMMIT-cov as 'SUMMIT'. Also uses compact subplot titles."
        ),
    )

    # Filters
    ap.add_argument("--true_h2_vals", type=float, nargs="*", default=None)
    ap.add_argument(
        "--p_causals",
        type=float,
        nargs="*",
        default=[1.0, 0.01],
        help="p_causal values to include (applied to BOTH non-null and null). Default matches your shared set.",
    )
    ap.add_argument(
        "--archs",
        nargs="*",
        default=None,
        help="Optional: restrict NON-NULL to these arch values (e.g., GCTA LDAK).",
    )
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help='Optional: restrict NON-NULL to these scenarios (e.g., "weak" "strong").',
    )

    ap.add_argument(
        "--coding_bin",
        type=int,
        default=0,
        help="Which bin index is the coding bin (default 0). We only use this bin from both non-null and null.",
    )

    ap.add_argument(
        "--best-results",
        action="store_true",
        help="If set: for LDSC/SumHer methods only plot 50Mb (window=50000kb).",
    )

    # Resampling knobs
    ap.add_argument(
        "--n_resamples", type=int, default=200, help="Number of stratified resamples."
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--balance_arch_scenario",
        action="store_true",
        help="If set: when sampling positives (non-null), sample ~evenly across (arch, scenario) within each stratum.",
    )
    ap.add_argument(
        "--replace_if_needed",
        action="store_true",
        help="If set: allow sampling with replacement when a group is too small (normally unnecessary).",
    )
    ap.add_argument("--verbose", action="store_true")

    # Output / plotting
    ap.add_argument(
        "--outfile",
        default="figs/supplementary/fig_s14_coding_power.pdf",
        help="Output PDF for curves.",
    )
    ap.add_argument("--grid_points", type=int, default=201)
    ap.add_argument(
        "--highlight_method",
        default="sumrhe",
        help="Extra method label to highlight (SUMMIT + SUMMIT-cov are always highlighted).",
    )
    ap.add_argument(
        "--ribbon_alpha",
        type=float,
        default=0.10,
        help="Alpha for 5–95% band ribbons (0 to disable).",
    )

    # Dodging knobs
    ap.add_argument("--no_dodge", action="store_true")
    ap.add_argument("--dodge_frac", type=float, default=0.45)
    ap.add_argument("--y_thresh_roc", type=float, default=0.02)
    ap.add_argument("--y_thresh_pr", type=float, default=0.02)
    ap.add_argument(
        "--roc_anchor", choices=["center", "right", "left"], default="center"
    )
    ap.add_argument("--pr_anchor", choices=["center", "right", "left"], default="right")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    # If main-text: force best-results (50Mb-only for windowed methods)
    if args.main_text:
        args.best_results = True
        # if user didn't override highlight_method, default to covsumrhe (the plotted "SUMMIT")
        if args.highlight_method == "sumrhe":
            args.highlight_method = "covsumrhe"

    all_rows = []

    for pop in args.pops:
        # --- NON-NULL ---
        df_nn = _read_one_csv(pop, args.outs_base_nonnul)
        if df_nn is not None:
            nn_long = _prep_long_for_scores(
                df_nn,
                sim_type="nonnull",
                best_results=args.best_results,
                true_h2_vals=args.true_h2_vals,
                p_causals=args.p_causals,
                archs=args.archs,
                scenarios=args.scenarios,
                coding_bin=args.coding_bin,
            )
            if not nn_long.empty:
                all_rows.append(nn_long)
            else:
                print(f"[warn] non-null: no usable rows for {pop}")

        # --- NULL ---
        df_null = _read_one_csv(pop, args.outs_base_null)
        if df_null is not None:
            null_long = _prep_long_for_scores(
                df_null,
                sim_type="null",
                best_results=args.best_results,
                true_h2_vals=args.true_h2_vals,
                p_causals=args.p_causals,
                archs=None,  # null won't have these; ignore
                scenarios=None,  # null won't have these; ignore
                coding_bin=args.coding_bin,
            )
            if not null_long.empty:
                all_rows.append(null_long)
            else:
                print(f"[warn] null: no usable rows for {pop}")

    if not all_rows:
        raise SystemExit("No data collected from non-null + null; nothing to plot.")

    enr_all = pd.concat(all_rows, ignore_index=True)

    # Main-text: keep only minimal set of method_labels
    if args.main_text:
        enr_all = enr_all[enr_all["method_label"].isin(MAIN_TEXT_KEEP)].copy()

    # sanity: need both classes per pop/method
    chk = enr_all.groupby(["pop", "method_label"])["y"].nunique()
    bad = chk[chk < 2]
    if len(bad) > 0:
        for (pop, m), _ in bad.items():
            print(
                f"[warn] pop={pop} method={m}: missing one class (null or non-null); will be skipped."
            )

    grid = np.linspace(0.0, 1.0, int(args.grid_points))

    balance_cols = ["arch", "scenario"] if args.balance_arch_scenario else []
    mean_roc, mean_pr = build_balanced_curves(
        enr_all,
        fpr_grid=grid,
        rec_grid=grid,
        n_resamples=args.n_resamples,
        seed=args.seed,
        balance_cols=balance_cols,
        replace_if_needed=args.replace_if_needed,
        verbose=args.verbose,
    )

    # ---- Print AUROC / AUPR summary table ----
    pretty_map = get_pretty_label_map(args.main_text)
    summary_rows = []
    for (pop, method), rec in mean_roc.items():
        pr = mean_pr.get((pop, method), {})
        summary_rows.append(
            {
                "pop": pop,
                "method": method,
                "pretty_method": pretty_map.get(method, method),
                "n_median": rec.get("n", np.nan),
                "n_resamples": rec.get("n_resamples", np.nan),
                "AUROC_mean": rec.get("auroc_mean", np.nan),
                "AUROC_sd": rec.get("auroc_sd", np.nan),
                "AUPR_mean": pr.get("aupr_mean", np.nan),
                "AUPR_sd": pr.get("aupr_sd", np.nan),
            }
        )
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["pop", "method"])
        print(
            "\n=== Balanced (coding non-null vs coding null): AUROC / AUPR over resamples ==="
        )
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    methods_present = set(enr_all["method_label"].unique().tolist())
    pops_order = [
        p
        for p in args.pops
        if any((p, m) in mean_roc for m in methods_present)
        or any((p, m) in mean_pr for m in methods_present)
    ]
    if len(pops_order) == 0:
        raise SystemExit("No populations left after filtering; cannot draw figure.")
    if len(pops_order) < 3:
        pops_order = pops_order + [""] * (3 - len(pops_order))

    # For main-text: no suptitle (cleaner)
    suptitle = None
    if args.main_text:
        suptitle = ""
    else:
        suptitle = (
            f"ROC & PR (coding bin={args.coding_bin}): non-null vs null — "
            f"stratified balanced resampling (R={args.n_resamples})"
        )

    make_2x3_curves(
        mean_roc,
        mean_pr,
        pops_order,
        args.outfile,
        highlight_method=args.highlight_method,
        ribbon_alpha=float(args.ribbon_alpha),
        dodge=(not args.no_dodge),
        dodge_frac=args.dodge_frac,
        y_thresh_roc=args.y_thresh_roc,
        y_thresh_pr=args.y_thresh_pr,
        roc_anchor=args.roc_anchor,
        pr_anchor=args.pr_anchor,
        suptitle=suptitle,
        compact_subplot_titles=args.main_text,
        label_order_override=(MAIN_TEXT_ORDER if args.main_text else None),
        pretty_label_map=pretty_map,
    )


if __name__ == "__main__":
    main()
