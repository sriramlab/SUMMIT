#!/usr/bin/env python3
"""
plot_pop_diff.py

Pairwise population-difference tests for total h2 (per trait), using reported SEs.

Wald z-statistic:
    z = (h2_pop1 - h2_pop2) / sqrt(se_pop1^2 + se_pop2^2)

P-values: two-sided normal approximation.
Multiple testing columns:
  - nominal: p < alpha
  - BH: q < fdr   (for reporting; no threshold line plotted)
  - Bonferroni: applied PER METHOD (and per bin): within each (method, num_bins)

Outputs:
  1) --out-csv: all tests
  2) --out-sig-csv: only significant rows (any threshold)
  3) --out-fig: Wald z-statistic scatter+box plots
       - One subplot per bin (8 or 24). If both, two rows.
       - X-axis: population pairs
       - Within each pop-pair, methods grouped side-by-side (grouped boxplots).
       - Boxplots colored by method (hard-coded palette).
       - Points optionally colored by trait class (--color-by-group), using a pastel palette
         designed to be orthogonal to method colors.

Plot threshold lines:
  - nominal (global; gray dashed)
  - Bonferroni (PER METHOD; dashed lines in the method's color)

Example:
./plot_pop_diff.py \
  --outs-base data/real_h2 \
  --csv-name estimates.csv \
  --phen-list data/traits/trait_categories.tsv \
  --bins 8,24 \
  --color-by-group \
  --print-hits sumrhe,covldsc \
  --out-csv figs/pop_diff_tests.csv \
  --out-sig-csv figs/pop_diff_significant.csv \
  --out-fig figs/pop_diff_tstats.pdf
"""

import os
import sys
import math
import argparse
import colorsys
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------
# Hard-coded method colors (boxplots)
# ---------------------------
TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
METHOD_COLOR = {
    "sumrhe": TAB20[4],
    "covsumrhe": TAB20[4],  # treat as SUMMIT-family if present
    "sumher": TAB20[6],
    "sumher_ldak": TAB20[8],
    "ldsc": TAB20[10],
    "covldsc": TAB20[2],
    "rhe": TAB20[0],
}

BASE_LABEL = {
    "sumrhe": "SUMMIT",
    "covsumrhe": "SUMMIT",
    "sumher": "SumHer-GCTA",
    "sumher_ldak": "SumHer-LDAK",
    "covldsc": "cov-LDSC",
    "ldsc": "LDSC",
    "rhe": "RHE-mc",
}

DEFAULT_POPS = ["AFR", "EUR", "EUR_300k", "SAS"]


# ---------------------------
# Normal helpers (scipy optional)
# ---------------------------
try:
    from scipy.stats import norm as _scipy_norm  # type: ignore

    def norm_cdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))

    def norm_isf(q: float) -> float:
        return float(_scipy_norm.isf(q))

except Exception:

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Acklam inverse normal CDF approximation (ppf)
    def _norm_ppf(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            raise ValueError("p must be in (0,1)")

        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]

        plow = 0.02425
        phigh = 1.0 - plow

        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            return num / den

        if p > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            return -(num / den)

        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return num / den

    def norm_isf(q: float) -> float:
        if q <= 0.0:
            return float("inf")
        if q >= 1.0:
            return float("-inf")
        return _norm_ppf(1.0 - q)


def two_sided_p_from_z(z: float) -> float:
    az = abs(float(z))
    return max(0.0, min(1.0, 2.0 * (1.0 - norm_cdf(az))))


# ---------------------------
# Multiple testing
# ---------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q_rev = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q_rev)
    out[order] = np.clip(q_rev, 0.0, 1.0)
    return out


# ---------------------------
# Phen list + class
# ---------------------------
def load_phen_list_with_class(
    path: str,
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    phen_order: List[str] = []
    phen2class: Dict[str, str] = {}
    phen2abbr: Dict[str, str] = {}

    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise RuntimeError(f"Phen list line has <4 columns: '{line}'")
            phen, _code, abbr, cls = parts[0], parts[1], parts[2], parts[3]
            phen_order.append(phen)
            phen2class[phen] = cls
            phen2abbr[phen] = abbr

    return phen_order, phen2class, phen2abbr


def _domain_of_class(cls: str) -> str:
    return cls.split("_", 1)[0] if "_" in cls else cls


def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r = int(max(0, min(1, rgb[0])) * 255)
    g = int(max(0, min(1, rgb[1])) * 255)
    b = int(max(0, min(1, rgb[2])) * 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def build_class_palette_pastel(classes: List[str]) -> Dict[str, str]:
    """
    Pastel, domain-grouped palette to avoid clashing with METHOD_COLOR (tab20, saturated).

    Strategy:
      - assign each DOMAIN a base pastel hue from Pastel1/Pastel2
      - for classes within a domain, vary LIGHTNESS slightly (same hue family)
    """
    classes = sorted(set([str(c) for c in classes if str(c) != "nan"]))
    domain2classes: Dict[str, List[str]] = {}
    for cls in classes:
        domain2classes.setdefault(_domain_of_class(cls), []).append(cls)

    # base pastel colors per domain
    base_cmap = colormaps["Pastel1"] if "Pastel1" in colormaps else colormaps["Pastel2"]
    base_colors = [to_hex(c) for c in base_cmap.colors]
    domains = sorted(domain2classes.keys())
    if not domains:
        return {}

    palette: Dict[str, str] = {}
    for di, domain in enumerate(domains):
        base_hex = base_colors[di % len(base_colors)]
        r, g, b = _hex_to_rgb01(base_hex)
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        cls_list = sorted(domain2classes[domain])
        k = len(cls_list)

        # keep pastel saturation; vary lightness for sibling classes
        # (more classes -> wider lightness spread)
        lightness = np.linspace(0.55, 0.85, max(1, k))
        sat = min(0.55, max(0.25, s if s > 0 else 0.45))

        for c, li in zip(cls_list, lightness):
            rr, gg, bb = colorsys.hls_to_rgb(h, float(li), float(sat))
            palette[c] = _rgb01_to_hex((rr, gg, bb))

    return palette


# ---------------------------
# Data loading (consistent with your plot_total_h2.py)
# ---------------------------
def load_all_pops_csv(outs_base: str, csv_name: str, pops: List[str]) -> pd.DataFrame:
    dfs = []
    for pop in pops:
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
        raise RuntimeError("[error] no CSVs found for any population")

    return pd.concat(dfs, ignore_index=True)


def select_largest_window(df_all: pd.DataFrame) -> pd.DataFrame:
    need = ["pop", "phen", "method", "num_bins", "h2", "h2_se", "window"]
    for c in need:
        if c not in df_all.columns:
            raise RuntimeError(
                f"CSV missing required column '{c}'. Found: {list(df_all.columns)}"
            )

    df = df_all.copy()
    df["phen"] = df["phen"].astype(str)
    df["method"] = df["method"].astype(str)
    df["pop"] = df["pop"].astype(str)
    df["num_bins"] = pd.to_numeric(df["num_bins"], errors="coerce")
    df["window_sort"] = pd.to_numeric(df["window"], errors="coerce").fillna(-1)

    df_sorted = df.sort_values("window_sort")
    df_sel = (
        df_sorted.groupby(["pop", "phen", "method", "num_bins"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return df_sel


# ---------------------------
# Tests
# ---------------------------
def build_pairwise_tests(
    df_sel: pd.DataFrame,
    phen_order: List[str],
    phen2class: Dict[str, str],
    phen2abbr: Dict[str, str],
    pops_present: List[str],
    bins_keep: List[int],
    methods_keep: Optional[List[str]],
) -> pd.DataFrame:
    df = df_sel.copy()

    df["h2_num"] = pd.to_numeric(df["h2"], errors="coerce")
    df["h2_se_num"] = pd.to_numeric(df["h2_se"], errors="coerce")
    df = df[np.isfinite(df["h2_num"]) & np.isfinite(df["h2_se_num"])].copy()
    df = df[df["h2_se_num"] > 0].copy()

    df = df[df["phen"].isin(phen_order)].copy()
    df = df[df["num_bins"].isin(bins_keep)].copy()
    df = df[df["pop"].isin(pops_present)].copy()

    if methods_keep is not None:
        df = df[df["method"].isin(methods_keep)].copy()

    key_cols = ["pop", "phen", "method", "num_bins"]
    df_key = df[key_cols + ["h2_num", "h2_se_num"]].copy()

    dup = df_key.duplicated(key_cols, keep=False)
    if dup.any():
        ndup = int(dup.sum())
        print(
            f"[warn] duplicated (pop,phen,method,num_bins) rows: {ndup}; using last",
            file=sys.stderr,
        )
        df_key = df_key.drop_duplicates(key_cols, keep="last")

    lookup = df_key.set_index(key_cols)

    pop_pairs = list(combinations(pops_present, 2))
    rows = []
    for pop1, pop2 in pop_pairs:
        pop_pair = f"{pop1} vs {pop2}"
        for phen in phen_order:
            cls = phen2class.get(phen, "unknown")
            abbr = phen2abbr.get(phen, phen)
            for method in sorted(df["method"].unique()):
                for nb in sorted(set(bins_keep)):
                    k1 = (pop1, phen, method, float(nb))
                    k2 = (pop2, phen, method, float(nb))
                    if k1 not in lookup.index or k2 not in lookup.index:
                        continue

                    h1, s1 = (
                        lookup.loc[k1, ["h2_num", "h2_se_num"]].astype(float).tolist()
                    )
                    h2, s2 = (
                        lookup.loc[k2, ["h2_num", "h2_se_num"]].astype(float).tolist()
                    )
                    se_diff = math.sqrt(float(s1) ** 2 + float(s2) ** 2)
                    if not np.isfinite(se_diff) or se_diff <= 0:
                        continue

                    diff = float(h1) - float(h2)
                    tstat = diff / se_diff
                    pval = two_sided_p_from_z(tstat)

                    rows.append(
                        {
                            "method": method,
                            "num_bins": int(nb),
                            "pop1": pop1,
                            "pop2": pop2,
                            "pop_pair": pop_pair,
                            "phen": phen,
                            "phen_abbr": abbr,
                            "class": cls,
                            "h2_pop1": float(h1),
                            "se_pop1": float(s1),
                            "h2_pop2": float(h2),
                            "se_pop2": float(s2),
                            "diff": float(diff),
                            "se_diff": float(se_diff),
                            "t_stat": float(tstat),
                            "p_value": float(pval),
                        }
                    )

    return pd.DataFrame(rows)


def apply_multipletesting(
    df_tests: pd.DataFrame,
    alpha: float,
    fdr: float,
    bh_scope: str,
) -> pd.DataFrame:
    """
    BH q-values (scope configurable).
    Bonferroni is ALWAYS per (method, num_bins), per your request.

    bh_scope:
      - "bin": BH within each num_bins across ALL methods+pop_pairs+traits
      - "method_bin": BH within each (method, num_bins)
      - "method_bin_pop_pair": BH within each (method, num_bins, pop_pair)
    """
    if bh_scope not in {"bin", "method_bin", "method_bin_pop_pair"}:
        raise RuntimeError(
            "bh_scope must be one of: bin, method_bin, method_bin_pop_pair"
        )

    if bh_scope == "bin":
        bh_group_cols = ["num_bins"]
    elif bh_scope == "method_bin":
        bh_group_cols = ["method", "num_bins"]
    else:
        bh_group_cols = ["method", "num_bins", "pop_pair"]

    df = df_tests.copy()
    df["reject_nominal"] = df["p_value"] < alpha

    # BH
    df["q_bh"] = np.nan
    df["reject_bh"] = False
    for _, idxs in df.groupby(bh_group_cols, sort=False).groups.items():
        idx = np.asarray(list(idxs), dtype=int)
        p = df.loc[idx, "p_value"].astype(float).to_numpy()
        if p.size == 0:
            continue
        q = bh_qvalues(p)
        df.loc[idx, "q_bh"] = q
        df.loc[idx, "reject_bh"] = q < fdr

    # Bonferroni PER METHOD (and bin)
    df["bonf_m"] = np.nan
    df["p_bonf"] = np.nan
    df["reject_bonf"] = False
    for (mth, nb), idxs in df.groupby(
        ["method", "num_bins"], sort=False
    ).groups.items():
        idx = np.asarray(list(idxs), dtype=int)
        p = df.loc[idx, "p_value"].astype(float).to_numpy()
        m = int(p.size)
        if m <= 0:
            continue
        df.loc[idx, "bonf_m"] = m
        df.loc[idx, "p_bonf"] = np.minimum(1.0, p * m)
        df.loc[idx, "reject_bonf"] = p < (alpha / m)

    return df


# ---------------------------
# Plotting helpers
# ---------------------------
def _method_color_map(methods: List[str]) -> Dict[str, str]:
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", TAB20)
    out = {}
    j = 0
    for m in methods:
        if m in METHOD_COLOR:
            out[m] = METHOD_COLOR[m]
        else:
            out[m] = default_cycle[j % len(default_cycle)]
            j += 1
    return out


def _pretty_pair_label(pp: str) -> str:
    a, b = [x.strip() for x in pp.split("vs")]
    a2 = "EUR(300k)" if a == "EUR_300k" else a
    b2 = "EUR(300k)" if b == "EUR_300k" else b
    return f"{a2}–{b2}"


def _t_bonf_for_method_bin(alpha: float, m: int) -> float:
    m = max(1, int(m))
    return float(norm_isf((alpha / m) / 2.0))


def make_plots_grouped_methods(
    df_tests: pd.DataFrame,
    out_fig: str,
    pops_present: List[str],
    bins_plot: List[int],
    methods_present: List[str],
    color_by_group: bool,
    class_palette: Dict[str, str],
    alpha: float,
):
    sns.set_theme(style="white", context="talk")

    pop_pairs = [f"{a} vs {b}" for a, b in combinations(pops_present, 2)]
    if not pop_pairs:
        raise RuntimeError("Need >= 2 pops to plot pairs.")

    nrows = len(bins_plot)
    fig_w = max(16.0, 2.25 * len(pop_pairs) + 0.9 * max(1, len(methods_present)))
    fig_h = max(5.0, 4.2 * nrows)

    fig, axes = plt.subplots(nrows, 1, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if nrows == 1:
        axes = [axes]

    method_colors = _method_color_map(methods_present)

    # legends: method always; class only if enabled
    method_handles = [
        Patch(
            facecolor=method_colors[m],
            edgecolor="black",
            alpha=0.35,
            label=BASE_LABEL.get(m, m),
        )
        for m in methods_present
    ]

    class_handles: List[Line2D] = []
    if color_by_group:
        classes = sorted(df_tests["class"].dropna().astype(str).unique().tolist())
        for c in classes:
            if c not in class_palette:
                continue
            class_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=class_palette[c],
                    label=c,
                    markersize=8,
                )
            )

    rng = np.random.default_rng(0)

    # nominal threshold is global
    t_nom = float(norm_isf(alpha / 2.0))

    # Per-bin plotting
    for ax, nb in zip(axes, bins_plot):
        sub = df_tests[df_tests["num_bins"] == int(nb)].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {nb} bins", ha="center", va="center")
            ax.set_axis_off()
            continue

        x_centers = np.arange(len(pop_pairs), dtype=float)
        methods_bin = [
            m for m in methods_present if m in set(sub["method"].astype(str).unique())
        ]
        M = max(1, len(methods_bin))

        group_width = 0.80
        box_w = group_width / M
        offsets = [(j - (M - 1) / 2.0) * box_w for j in range(M)]

        # Compute per-method Bonf thresholds (per method, per bin)
        method2m = {}
        method2tbonf = {}
        for mth in methods_bin:
            m = int((sub["method"] == mth).sum())
            method2m[mth] = m
            method2tbonf[mth] = _t_bonf_for_method_bin(alpha, m)

        # Draw grouped boxplots + scatters
        for j, mth in enumerate(methods_bin):
            pos = x_centers + offsets[j]
            data = []
            for pp in pop_pairs:
                vals = (
                    sub[(sub["pop_pair"] == pp) & (sub["method"] == mth)]["t_stat"]
                    .astype(float)
                    .to_numpy()
                )
                vals = vals[np.isfinite(vals)]
                data.append(vals)

            bp = ax.boxplot(
                data,
                positions=pos,
                widths=box_w * 0.88,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                manage_ticks=False,
                zorder=2,
            )

            face = method_colors.get(mth, "0.7")
            for patch in bp["boxes"]:
                patch.set_facecolor(face)
                patch.set_alpha(0.28)
                patch.set_edgecolor("black")
                patch.set_linewidth(1.0)
            for key in ["whiskers", "caps", "medians"]:
                for item in bp[key]:
                    item.set_color("black")
                    item.set_linewidth(1.0)

            # scatter overlay
            for i, pp in enumerate(pop_pairs):
                g = sub[(sub["pop_pair"] == pp) & (sub["method"] == mth)].copy()
                if g.empty:
                    continue
                x0 = pos[i]
                jitter = rng.normal(0.0, box_w * 0.10, size=len(g))
                xs = x0 + jitter
                ys = g["t_stat"].astype(float).to_numpy()

                if color_by_group:
                    cols = [
                        class_palette.get(str(c), "#777777")
                        for c in g["class"].astype(str).tolist()
                    ]
                else:
                    cols = [method_colors.get(mth, "0.15")] * len(g)

                ax.scatter(
                    xs,
                    ys,
                    s=28,
                    c=cols,
                    alpha=0.88,
                    edgecolors="black",
                    linewidths=0.25,
                    zorder=3,
                )

        # Threshold lines: nominal (gray) + per-method bonf (method-colored)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, zorder=1)
        ax.axhline(
            +t_nom, color="0.35", linestyle="--", linewidth=1.3, alpha=0.9, zorder=1
        )
        ax.axhline(
            -t_nom, color="0.35", linestyle="--", linewidth=1.3, alpha=0.9, zorder=1
        )

        for mth in methods_bin:
            t_b = method2tbonf[mth]
            col = method_colors.get(mth, "0.2")
            ax.axhline(
                +t_b, color=col, linestyle="--", linewidth=1.2, alpha=0.55, zorder=1
            )
            ax.axhline(
                -t_b, color=col, linestyle="--", linewidth=1.2, alpha=0.55, zorder=1
            )

        # ---- Reasonable y-limits (not too wide) but must include Bonf lines ----
        tvals = sub["t_stat"].astype(float).to_numpy()
        tvals = tvals[np.isfinite(tvals)]
        # Use the 95th percentile of |t| for display limits.
        base = float(np.nanpercentile(np.abs(tvals), 95)) if tvals.size else 4.0
        base = max(3.5, base * 1.5)
        # ensure we include the maximum Bonf line with padding
        t_bonf_max = max([abs(v) for v in method2tbonf.values()] + [abs(t_nom)])
        lim = max(base, t_bonf_max * 1.5)
        # still cap to keep plot readable, but never below required threshold
        lim = min(max(lim, t_bonf_max * 1.55), 30.0)

        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Wald z-statistic")  # (Δh² / sqrt(se₁² + se₂²))

        # Title includes Bonf m per method (compact)
        bonf_parts = []
        for mth in methods_bin:
            bonf_parts.append(f"{BASE_LABEL.get(mth, mth)} m={method2m[mth]}")
        bonf_txt = "; ".join(bonf_parts)
        if len(axes) < 2:
            ax.set_title(f"Population-pair h² significance")
        else:
            ax.set_title(f"Population-pair h² significance: {nb} bins")

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.grid(False, axis="x")

        ax.set_xticks(x_centers)
        # ax.set_xticklabels([_pretty_pair_label(pp) for pp in pop_pairs], rotation=15, ha="right")
        ax.set_xticklabels([_pretty_pair_label(pp) for pp in pop_pairs])

    axes[-1].set_xlabel("Population pair")

    outdir = os.path.dirname(out_fig)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Legends: NO dashed-line legend (per your request)
    if color_by_group and class_handles:
        fig.legend(
            handles=method_handles,
            title="Method (box color)",
            loc="center left",
            bbox_to_anchor=(0.86, 0.70),
            borderaxespad=0.0,
            ncol=1,
            handletextpad=0.2,
        )
        fig.legend(
            handles=class_handles,
            title="Trait class (point color)",
            loc="center left",
            bbox_to_anchor=(0.86, 0.28),
            borderaxespad=0.0,
            ncol=1,
            handletextpad=0.2,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
    else:
        fig.legend(
            handles=method_handles,
            title="Method",
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
            borderaxespad=0.0,
            ncol=1,
            handletextpad=0.2,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[ok] saved figure to {out_fig}")


# ---------------------------
# Significant-settings CSV + terminal printout
# ---------------------------
def write_sig_csv(df_tests: pd.DataFrame, out_sig_csv: str) -> pd.DataFrame:
    sig_mask = df_tests[["reject_nominal", "reject_bh", "reject_bonf"]].any(axis=1)
    df_sig = df_tests.loc[sig_mask].copy()

    outdir = os.path.dirname(out_sig_csv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df_sig.to_csv(out_sig_csv, index=False)
    print(f"[ok] saved significant-only CSV to {out_sig_csv}  (rows={len(df_sig)})")
    return df_sig


def print_hits(df_sig: pd.DataFrame, methods_to_print: List[str]):
    if not methods_to_print:
        return

    missing = [
        m
        for m in methods_to_print
        if m not in set(df_sig["method"].astype(str).unique())
    ]
    if missing:
        print(
            f"\n[print-hits] requested methods not found among significant rows: {', '.join(missing)}"
        )

    df_m = df_sig[df_sig["method"].isin(methods_to_print)].copy()
    if df_m.empty:
        print("\n[print-hits] No significant hits for requested methods.")
        return

    def _list_traits(g: pd.DataFrame, col: str) -> List[str]:
        gg = g[g[col]].copy()
        if gg.empty:
            return []
        gg["abs_t"] = gg["t_stat"].abs()
        gg = gg.sort_values("abs_t", ascending=False)
        return gg["phen_abbr"].astype(str).tolist()

    print(
        "\n[print-hits] Significant population-pair differences (trait abbreviations):"
    )
    for (mth, nb, pp), g in df_m.groupby(
        ["method", "num_bins", "pop_pair"], sort=False
    ):
        nom = _list_traits(g, "reject_nominal")
        bh = _list_traits(g, "reject_bh")
        bon = _list_traits(g, "reject_bonf")

        if not (nom or bh or bon):
            continue

        label = BASE_LABEL.get(mth, mth)
        print(f"\n  method={label}  bins={int(nb)}  pair={pp}")
        if nom:
            print(f"    nominal:   {', '.join(nom)}")
        if bh:
            print(f"    BH:        {', '.join(bh)}")
        if bon:
            print(f"    bonferr.:  {', '.join(bon)}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outs-base", default="data/real_h2")
    parser.add_argument("--csv-name", default="estimates.csv")
    parser.add_argument("--phen-list", default="data/traits/trait_categories.tsv")
    parser.add_argument(
        "--bins", default="8,24", help="Comma-separated bins (e.g. 8,24 or 24)"
    )
    parser.add_argument("--pops", default=",".join(DEFAULT_POPS))
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated methods to keep (default: keep all present)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--fdr", type=float, default=0.05, help="BH FDR threshold (reporting only)"
    )
    parser.add_argument(
        "--bh-scope",
        default="method_bin",
        choices=["bin", "method_bin", "method_bin_pop_pair"],
        help="BH family definition (Bonf is always per method+bin)",
    )
    parser.add_argument("--color-by-group", action="store_true")
    parser.add_argument(
        "--print-hits",
        default="",
        help="Comma-separated method(s) to print significant hits on terminal (e.g. sumrhe,covldsc)",
    )
    parser.add_argument("--out-csv", default="figs/pop_diff_tests.csv")
    parser.add_argument("--out-sig-csv", default="figs/pop_diff_significant.csv")
    parser.add_argument("--out-fig", default="figs/pop_diff_tstats.pdf")
    args = parser.parse_args()

    try:
        bins_keep = [int(x) for x in args.bins.split(",") if x.strip()]
    except Exception:
        raise RuntimeError(f"Could not parse --bins '{args.bins}' (expected like 8,24)")

    pops = [p.strip() for p in args.pops.split(",") if p.strip()]

    methods_keep = None
    if args.methods.strip():
        methods_keep = [m.strip() for m in args.methods.split(",") if m.strip()]

    methods_to_print: List[str] = []
    if args.print_hits.strip():
        methods_to_print = [m.strip() for m in args.print_hits.split(",") if m.strip()]

    phen_order, phen2class, phen2abbr = load_phen_list_with_class(args.phen_list)

    df_all = load_all_pops_csv(args.outs_base, args.csv_name, pops)
    df_sel = select_largest_window(df_all)

    pops_present = [p for p in pops if p in set(df_sel["pop"].astype(str).unique())]
    if len(pops_present) < 2:
        raise RuntimeError("Need at least 2 populations with data to compare.")

    df_tests = build_pairwise_tests(
        df_sel=df_sel,
        phen_order=phen_order,
        phen2class=phen2class,
        phen2abbr=phen2abbr,
        pops_present=pops_present,
        bins_keep=bins_keep,
        methods_keep=methods_keep,
    )
    if df_tests.empty:
        raise RuntimeError(
            "No tests were constructed (check filtering / missing data)."
        )

    df_tests = apply_multipletesting(
        df_tests=df_tests,
        alpha=float(args.alpha),
        fdr=float(args.fdr),
        bh_scope=str(args.bh_scope),
    )

    outdir = os.path.dirname(args.out_csv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df_tests.to_csv(args.out_csv, index=False)
    print(f"[ok] saved CSV to {args.out_csv}  (rows={len(df_tests)})")

    df_sig = write_sig_csv(df_tests, args.out_sig_csv)
    print_hits(df_sig, methods_to_print)

    # plot sets
    methods_present = sorted(df_tests["method"].astype(str).unique().tolist())
    if methods_keep is not None:
        methods_present = [m for m in methods_keep if m in set(methods_present)]

    bins_present = sorted(
        df_tests["num_bins"].astype(int).unique().tolist(), key=lambda b: (b != 8, b)
    )
    bins_plot = [
        b
        for b in sorted(set(bins_keep), key=lambda x: (x != 8, x))
        if b in set(bins_present)
    ]

    # pastel trait-class palette (orthogonal to method colors)
    class_palette = build_class_palette_pastel(
        df_tests["class"].astype(str).unique().tolist()
    )

    make_plots_grouped_methods(
        df_tests=df_tests,
        out_fig=args.out_fig,
        pops_present=pops_present,
        bins_plot=bins_plot,
        methods_present=methods_present,
        color_by_group=bool(args.color_by_group),
        class_palette=class_palette,
        alpha=float(args.alpha),
    )

    print("\n[summary] significant counts by method & bins:")
    for (m, nb), g in df_tests.groupby(["method", "num_bins"], sort=False):
        # bonf is now method+bin by construction
        print(
            f"  {m:12s} bins={int(nb):2d}  tests={len(g):5d}  "
            f"nominal={int(g['reject_nominal'].sum()):4d}  "
            f"BH={int(g['reject_bh'].sum()):4d}  "
            f"bonf={int(g['reject_bonf'].sum()):4d}  "
            f"(bonf_m={int(np.nanmax(g['bonf_m'])) if np.isfinite(np.nanmax(g['bonf_m'])) else 'NA'})"
        )


if __name__ == "__main__":
    main()
