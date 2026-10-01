"""Plot calibration style for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.patheffects as pe
from matplotlib import colormaps
from matplotlib.colors import to_hex
import math
import numpy as np
import pandas as pd
import seaborn as sns


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]


PRETTY_LABEL = {
    "rhe": "RHE-mc",
    "sumrhe": "SUMMIT",
    "covsumrhe": "SUMMIT-cov",
    "ldsc_2000": "LDSC (2Mb)",
    # "ldsc_20000": "LDSC (20Mb)",
    "ldsc_20000": "LDSC",
    "sumher_2000": "SumHer-GCTA (2Mb)",
    # "sumher_20000": "SumHer-GCTA (20Mb)",
    "sumher_20000": "SumHer\n(GCTA)",
    "sumher_ldak_2000": "SumHer-LDAK (2Mb)",
    # "sumher_ldak_20000": "SumHer-LDAK (20Mb)",
    "sumher_ldak_20000": "SumHer\n(LDAK)",
    "covldsc": "cov-LDSC",
    "covldsc_const": "cov-LDSC (constr.)",
}


METHOD_COLOR = {
    "rhe": TAB20[0],
    "sumrhe": TAB20[5],
    "covsumrhe": TAB20[4],
    "sumher_2000": TAB20[7],
    "sumher_20000": TAB20[6],
    "sumher_ldak_2000": TAB20[9],
    "sumher_ldak_20000": TAB20[8],
    "ldsc_2000": TAB20[11],
    "ldsc_20000": TAB20[10],
    "covldsc": TAB20[2],
    "covldsc_const": TAB20[3],
}


TITLE_FONTSIZE = 16


AXIS_LABEL_FONTSIZE = 15


TICK_LABEL_FONTSIZE = 12


LEGEND_FONTSIZE = 12


def _get_method_color(method_label: str):
    if method_label in METHOD_COLOR:
        return METHOD_COLOR[method_label]
    base = str(method_label).split("_", 1)[0]  # e.g., ldsc_2000 -> ldsc
    return METHOD_COLOR.get(base, None)


def _method_family(m: str) -> str:
    if m == "rhe":
        return "rhe"
    if m in ("sumrhe", "covsumrhe"):
        return "sumrhe"
    if m == "covldsc" or m.startswith("ldsc_"):
        return "ldsc"
    if m.startswith("sumher_") or m.startswith("sumher_ldak_"):
        return "sumher"
    return "other"


MARKER_BY_FAMILY = {
    "rhe": "s",  # square
    "sumrhe": "o",  # circle
    "ldsc": "^",  # triangle up
    "sumher": "D",  # diamond
    "other": "o",
}


def _norm_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return math.nan
    # Rational approximation (Acklam)
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
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / ((((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    )


def zcrit_two_sided(alpha: float) -> float:
    # two-sided: P(|Z| > z_crit) = alpha
    return _norm_ppf(1.0 - float(alpha) / 2.0)


def z_for_ci(ci_level: float) -> float:
    # symmetric normal critical for CI level (e.g., 0.95 -> 1.96)
    alpha = 1.0 - float(ci_level)
    return zcrit_two_sided(alpha)


def _compute_dodge_offsets(methods_ord, xs, dodge_frac=0.5):
    xs = np.array(sorted(xs), dtype=float)
    if xs.size >= 2:
        diffs = np.diff(xs)
        pos = diffs[diffs > 0]
        min_gap = float(np.min(pos)) if pos.size else 0.0
    else:
        min_gap = max(1e-3, float(xs[0]) * 0.1) if xs.size else 1e-3
    base_delta = (min_gap if min_gap > 0 else 1e-3) * float(dodge_frac)
    k = len(methods_ord)
    offsets = np.linspace(-base_delta, base_delta, k) if k > 1 else np.array([0.0])
    return {m: offsets[i] for i, m in enumerate(methods_ord)}, base_delta


def _collision_offsets(
    df, methods_ord, xs, base_delta, y_thresh=0.02, anchor_methods=None, smooth=True
):
    """
    For each alpha, cluster methods whose empirical rates are within y_thresh and assign
    symmetric horizontal offsets in [-base_delta, base_delta]. Anchor methods (if present)
    are kept near zero within their cluster.
    """
    anchor_methods = set(anchor_methods or [])

    ymap = {m: {} for m in methods_ord}
    for m in methods_ord:
        gm = df[df["method_label"] == m][["alpha", "empirical"]]
        for _, r in gm.iterrows():
            ymap[m][float(r["alpha"])] = float(r["empirical"])

    offsets = {}
    xs = [float(a) for a in xs]

    for a in xs:
        pairs = [(m, ymap[m][a]) for m in methods_ord if a in ymap[m]]
        if len(pairs) <= 1:
            for m, _ in pairs:
                offsets[(m, a)] = 0.0
            continue

        pairs.sort(key=lambda t: t[1])
        clusters = []
        cur = [pairs[0]]
        for i in range(1, len(pairs)):
            if abs(pairs[i][1] - pairs[i - 1][1]) <= y_thresh:
                cur.append(pairs[i])
            else:
                clusters.append(cur)
                cur = [pairs[i]]
        clusters.append(cur)

        for cluster in clusters:
            k = len(cluster)
            if k == 1:
                offsets[(cluster[0][0], a)] = 0.0
                continue

            base = np.linspace(-base_delta, base_delta, k)
            anchor_in_cluster = [
                i for i, (m, _) in enumerate(cluster) if m in anchor_methods
            ]
            if anchor_in_cluster:
                anchor_idx = anchor_in_cluster[0]
                zero_idx = int(np.argmin(np.abs(base)))
                base = np.roll(base, zero_idx - anchor_idx)

            for i, (m, _) in enumerate(cluster):
                offsets[(m, a)] = float(base[i])

    if smooth and len(xs) >= 3:
        xs_sorted = sorted(xs)
        for m in methods_ord:
            vals = np.array([offsets.get((m, a), 0.0) for a in xs_sorted], dtype=float)
            sm = vals.copy()
            for i in range(len(xs_sorted)):
                lo = max(0, i - 1)
                hi = min(len(xs_sorted) - 1, i + 1)
                sm[i] = np.mean(vals[lo : hi + 1])
            for i, a in enumerate(xs_sorted):
                key = (m, a)
                if key in offsets:
                    offsets[key] = float(sm[i])

    return offsets


def _plot_lines(
    ax,
    df,
    methods_ord,
    title,
    xlabel,
    ylabel,
    dodge=True,
    dodge_frac=0.5,
    y_thresh=0.02,
    lw=1.35,
    ms=5.4,
    highlight_methods=("sumrhe", "covsumrhe"),
    pretty_label_map=None,
    show_ribbon: bool = False,
    ribbon_kind: str = "ci",  # "ci" or "se"
    ribbon_alpha: float = 0.18,
    ci_level: float = 0.95,
    alpha_grid: list[float] | None = None,
):
    """
    Unify with plot_sims_maintext:
      - use the same alpha VALUES (points),
      - but keep an even-spaced x-axis grid (do NOT set xticks=alphas).
    """
    import matplotlib.ticker as mticker

    if pretty_label_map is None:
        pretty_label_map = PRETTY_LABEL

    if df is None or df.empty:
        ax.set_visible(False)
        return

    # ---- enforce same alpha VALUES as plot_sims_maintext ----
    if alpha_grid is None:
        alpha_grid = [1e-3, 1e-2, 0.05, 0.10, 0.20]
    grid = np.round(np.array([float(a) for a in alpha_grid], dtype=float), 12)

    df = df.copy()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df = df.dropna(subset=["alpha"]).copy()
    df["alpha"] = df["alpha"].astype(float).round(12)

    # filter to grid robustly
    a_vals = df["alpha"].to_numpy(dtype=float)
    keep = np.zeros(df.shape[0], dtype=bool)
    for a in grid:
        keep |= np.isclose(a_vals, a, atol=1e-12, rtol=0.0)
    df = df.loc[keep].copy()

    if df.empty:
        ax.set_visible(False)
        return

    xs = sorted(df["alpha"].unique().tolist())
    if not xs:
        ax.set_visible(False)
        return

    # highlight set
    hi = set(highlight_methods or [])
    if "sumrhe" in hi:
        hi.add("covsumrhe")

    # colors stable
    palette = sns.color_palette(n_colors=max(3, len(methods_ord)))
    color_map = {}
    pal_i = 0
    for m in methods_ord:
        c = _get_method_color(m)
        if c is None:
            c = palette[pal_i % len(palette)]
            pal_i += 1
        color_map[m] = c

    # ideal line
    ax.plot(
        xs,
        xs,
        linestyle="--",
        linewidth=2.2,
        color="0.35",
        alpha=0.7,
        label="Ideal (y = x)",
        zorder=1,
    )

    _, base_delta = _compute_dodge_offsets(methods_ord, xs, dodge_frac=dodge_frac)
    offsets = (
        _collision_offsets(
            df,
            methods_ord,
            xs,
            base_delta,
            y_thresh=y_thresh,
            anchor_methods=hi,
            smooth=True,
        )
        if dodge
        else {}
    )

    zci = z_for_ci(ci_level)

    for m in methods_ord:
        gm = df[df["method_label"] == m].sort_values("alpha")
        if gm.empty:
            continue

        a = gm["alpha"].to_numpy(dtype=float)
        y = gm["empirical"].to_numpy(dtype=float)

        off = np.array([offsets.get((m, float(ai)), 0.0) for ai in a], dtype=float)
        x = np.clip(a + off, 0.0, 1.0)

        fam = _method_family(m)
        mk = MARKER_BY_FAMILY.get(fam, "o")
        col = color_map[m]
        is_hi = m in hi

        if is_hi:
            this_lw = lw * 2.1
            this_ms = ms * 1.25
            this_alpha = 0.95
            this_z = 6
            path_fx = [
                pe.Stroke(linewidth=this_lw + 1.25, foreground="white", alpha=0.75),
                pe.Normal(),
            ]
            mfc, mec, mew = col, col, 0.0
        else:
            this_lw = lw
            this_ms = ms * 0.95
            this_alpha = 0.65
            this_z = 4
            path_fx = None
            mfc, mec, mew = "none", col, 1.1

        # ribbon
        if show_ribbon:
            lo = hi_arr = None
            if (
                ribbon_kind == "ci"
                and ("emp_lo" in gm.columns)
                and ("emp_hi" in gm.columns)
            ):
                lo = gm["emp_lo"].to_numpy(dtype=float)
                hi_arr = gm["emp_hi"].to_numpy(dtype=float)
            elif ribbon_kind == "se" and ("emp_se" in gm.columns):
                se = gm["emp_se"].to_numpy(dtype=float)
                lo = np.clip(y - zci * se, 0.0, 1.0)
                hi_arr = np.clip(y + zci * se, 0.0, 1.0)
            else:
                if "n_noncausal" in gm.columns:
                    n = gm["n_noncausal"].to_numpy(dtype=float)
                    se = np.sqrt(
                        np.clip(y * (1.0 - y), 0.0, 1.0) / np.clip(n, 1.0, np.inf)
                    )
                    lo = np.clip(y - zci * se, 0.0, 1.0)
                    hi_arr = np.clip(y + zci * se, 0.0, 1.0)

            if lo is not None and hi_arr is not None:
                ax.fill_between(
                    x,
                    lo,
                    hi_arr,
                    color=col,
                    alpha=float(ribbon_alpha) if is_hi else float(ribbon_alpha) * 0.65,
                    linewidth=0,
                    zorder=this_z - 1,
                )

        ax.plot(
            x,
            y,
            marker=mk,
            linewidth=this_lw,
            markersize=this_ms,
            alpha=this_alpha,
            zorder=this_z,
            path_effects=path_fx,
            color=col,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markeredgewidth=mew,
            label=pretty_label_map.get(m, m),
        )

    # Axis ranges with a small gap at the upper limit.
    xmin = max(0.0, float(min(xs)) - 1.5 * base_delta)
    xmax = min(1.0, float(max(xs)) + 1.5 * base_delta)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.02, 1.02)

    # ---- even-spaced grid + ticks (NOT per-alpha) ----
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)


def _add_shared_legend(fig, axes, title="Method", right_margin=0.88):
    all_axes = np.atleast_1d(axes).ravel()
    handles, labels = [], []
    for ax in all_axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles.extend(h)
            labels.extend(l)
    seen = set()
    H, L = [], []
    for h, l in zip(handles, labels):
        if l in seen or l == "_nolegend_":
            continue
        seen.add(l)
        H.append(h)
        L.append(l)
    if H:
        fig.legend(
            H,
            L,
            loc="center left",
            bbox_to_anchor=(0.89, 0.5),
            bbox_transform=fig.transFigure,
            frameon=True,
            title=title,
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=(0, 0, right_margin, 1))
    else:
        fig.tight_layout()
