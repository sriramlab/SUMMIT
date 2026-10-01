#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import erfc, sqrt
from pathlib import Path
from typing import Iterable
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path("data/sim_rg/total")
DEFAULT_OUT = Path("figs/supplementary/fig_s19_sumcore_overlap_modes.pdf")
DEFAULT_POPS = ["EUR_300k", "EUR", "SAS", "AFR"]
DEFAULT_METHOD_X = "summit_cov"
DEFAULT_METHOD_Y = "summit_unc_cov"

TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
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
POINT_COLOR = TAB20[0]
OWN_SCALE_POPS = {"EUR_300k"}
TEXT_DARK = "#111827"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Supplementary SUM-CORE supplied- vs summary-estimated-overlap benchmark figure."
    )
    ap.add_argument(
        "--base",
        default=str(DEFAULT_BASE),
        help="Base directory containing POP/estimates.csv.",
    )
    ap.add_argument(
        "--pops",
        nargs="+",
        default=DEFAULT_POPS,
        help="Cohorts to plot in display order.",
    )
    ap.add_argument(
        "--infile", default="estimates.csv", help="Per-pop simulation summary filename."
    )
    ap.add_argument("--method-x", default=DEFAULT_METHOD_X, help="Method on x-axis.")
    ap.add_argument("--method-y", default=DEFAULT_METHOD_Y, help="Method on y-axis.")
    ap.add_argument("--h2", type=float, default=0.25, help="Simulation h2 to include.")
    ap.add_argument(
        "--pol",
        type=float,
        default=1.0,
        help="Simulation polygenicity parameter to include.",
    )
    ap.add_argument("--rho-g", type=float, default=0.0, help="True rg to include.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output PDF path.")
    ap.add_argument(
        "--also-png",
        action="store_true",
        default=True,
        help="Also write a PNG preview.",
    )
    ap.add_argument(
        "--no-png",
        dest="also_png",
        action="store_false",
        help="Do not write a PNG preview.",
    )
    ap.add_argument(
        "--rg-error-bars",
        action="store_true",
        help="Draw x/y rg standard-error bars in the rg panels.",
    )
    ap.add_argument(
        "--overlap-error-bars",
        action="store_true",
        help="Draw x/y standard-error bars in the overlap-covariance panels.",
    )
    ap.add_argument("--point-size", type=float, default=18.0)
    ap.add_argument("--alpha", type=float, default=0.68)
    ap.add_argument("--error-alpha", type=float, default=0.18)
    ap.add_argument("--error-lw", type=float, default=0.55)
    ap.add_argument("--dpi", type=int, default=450)
    ap.add_argument("--width", type=float, default=14.2)
    ap.add_argument("--height", type=float, default=9.0)
    return ap.parse_args()


def set_style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_inputs(base: Path, pops: Iterable[str], infile: str) -> pd.DataFrame:
    frames = []
    for pop in pops:
        path = base / pop / infile
        if not path.exists():
            raise FileNotFoundError(f"Missing input for {pop}: {path}")
        df = pd.read_csv(path)
        df["pop"] = pop
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for col in [
        "h2",
        "pol",
        "rho_g",
        "i",
        "rg",
        "rg_se",
        "intercept_c",
        "intercept_se",
        "h2_1",
        "h2_2",
        "gamma_g",
        "gamma_se",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def filter_scenario(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    need = {
        "pop",
        "method",
        "h2",
        "pol",
        "rho_g",
        "i",
        "rg",
        "rg_se",
        "intercept_c",
        "intercept_se",
    }
    missing = sorted(need.difference(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    keep = (
        np.isclose(df["h2"].to_numpy(float), args.h2, atol=1e-12, rtol=0.0)
        & np.isclose(df["pol"].to_numpy(float), args.pol, atol=1e-12, rtol=0.0)
        & np.isclose(df["rho_g"].to_numpy(float), args.rho_g, atol=1e-12, rtol=0.0)
        & df["method"].astype(str).isin([args.method_x, args.method_y]).to_numpy()
    )
    out = df.loc[keep].copy()
    if out.empty:
        raise RuntimeError(
            f"No rows after filtering h2={args.h2}, pol={args.pol}, rho_g={args.rho_g}, "
            f"methods={args.method_x},{args.method_y}."
        )
    return out


def align_methods(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keys = ["pop", "h2", "pol", "rho_g", "i"]
    value_cols = [
        "rg",
        "rg_se",
        "intercept_c",
        "intercept_se",
        "h2_1",
        "h2_2",
        "gamma_g",
        "gamma_se",
    ]
    x = df[df["method"].astype(str) == args.method_x][keys + value_cols].copy()
    y = df[df["method"].astype(str) == args.method_y][keys + value_cols].copy()
    if x.duplicated(keys).any() or y.duplicated(keys).any():
        raise ValueError(
            "Duplicate method rows for the same pop/scenario/replicate key."
        )
    out = x.merge(y, on=keys, how="outer", suffixes=("_x", "_y"), indicator=True)
    return out.sort_values(["pop", "i"]).reset_index(drop=True)


def normal_two_sided_p(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan, dtype=float)
    finite = np.isfinite(z)
    if np.any(finite):
        out[finite] = np.fromiter(
            (erfc(abs(float(v)) / sqrt(2.0)) for v in z[finite]), dtype=float
        )
    return out


def finite_pair(tab: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    keep = np.isfinite(tab[x].to_numpy(float)) & np.isfinite(tab[y].to_numpy(float))
    return tab.loc[keep].copy()


def method_valid(tab: pd.DataFrame, suffix: str) -> np.ndarray:
    return (
        np.isfinite(tab[f"rg_{suffix}"].to_numpy(float))
        & np.isfinite(tab[f"rg_se_{suffix}"].to_numpy(float))
        & (tab[f"rg_se_{suffix}"].to_numpy(float) > 0)
    )


def summarize_pop(tab: pd.DataFrame, pop: str, alpha: float = 0.05) -> dict:
    total = int(tab.shape[0])
    valid_x = method_valid(tab, "x")
    valid_y = method_valid(tab, "y")
    paired_rg = finite_pair(tab, "rg_x", "rg_y")
    paired_se = finite_pair(tab, "rg_se_x", "rg_se_y")
    paired_c = finite_pair(tab, "intercept_c_x", "intercept_c_y")

    z_x = tab.loc[valid_x, "rg_x"].to_numpy(float) / tab.loc[
        valid_x, "rg_se_x"
    ].to_numpy(float)
    z_y = tab.loc[valid_y, "rg_y"].to_numpy(float) / tab.loc[
        valid_y, "rg_se_y"
    ].to_numpy(float)
    p_x = normal_two_sided_p(z_x)
    p_y = normal_two_sided_p(z_y)

    def corr(df: pd.DataFrame, xcol: str, ycol: str) -> float:
        if df.shape[0] < 2:
            return np.nan
        x = df[xcol].to_numpy(float)
        y = df[ycol].to_numpy(float)
        if np.std(x) <= 0 or np.std(y) <= 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    d_rg = paired_rg["rg_y"].to_numpy(float) - paired_rg["rg_x"].to_numpy(float)
    d_se = paired_se["rg_se_y"].to_numpy(float) - paired_se["rg_se_x"].to_numpy(float)
    d_c = paired_c["intercept_c_y"].to_numpy(float) - paired_c[
        "intercept_c_x"
    ].to_numpy(float)

    return {
        "pop": pop,
        "pop_label": POP_LABEL.get(pop, pop),
        "n_total": total,
        "n_valid_x": int(valid_x.sum()),
        "n_valid_y": int(valid_y.sum()),
        "n_paired_rg": int(paired_rg.shape[0]),
        "n_paired_se": int(paired_se.shape[0]),
        "n_paired_intercept": int(paired_c.shape[0]),
        "valid_rate_x": float(valid_x.mean()) if total else np.nan,
        "valid_rate_y": float(valid_y.mean()) if total else np.nan,
        "paired_valid_rate_rg": float(paired_rg.shape[0] / total) if total else np.nan,
        "fpr_0p05_x": float(np.nanmean(p_x <= alpha)) if p_x.size else np.nan,
        "fpr_0p05_y": float(np.nanmean(p_y <= alpha)) if p_y.size else np.nan,
        "mean_rg_x": float(np.nanmean(tab["rg_x"])),
        "mean_rg_y": float(np.nanmean(tab["rg_y"])),
        "sd_rg_x": float(np.nanstd(tab["rg_x"], ddof=1)),
        "sd_rg_y": float(np.nanstd(tab["rg_y"], ddof=1)),
        "corr_rg": corr(paired_rg, "rg_x", "rg_y"),
        "rmse_delta_rg": float(np.sqrt(np.mean(d_rg * d_rg))) if d_rg.size else np.nan,
        "median_rg_se_x": float(np.nanmedian(tab["rg_se_x"])),
        "median_rg_se_y": float(np.nanmedian(tab["rg_se_y"])),
        "corr_rg_se": corr(paired_se, "rg_se_x", "rg_se_y"),
        "rmse_delta_rg_se": float(np.sqrt(np.mean(d_se * d_se)))
        if d_se.size
        else np.nan,
        "mean_intercept_x": float(np.nanmean(tab["intercept_c_x"])),
        "mean_intercept_y": float(np.nanmean(tab["intercept_c_y"])),
        "median_intercept_se_x": float(np.nanmedian(tab["intercept_se_x"])),
        "median_intercept_se_y": float(np.nanmedian(tab["intercept_se_y"])),
        "corr_intercept": corr(paired_c, "intercept_c_x", "intercept_c_y"),
        "rmse_delta_intercept": float(np.sqrt(np.mean(d_c * d_c)))
        if d_c.size
        else np.nan,
    }


def panel_limits(
    tab: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    zero_floor: bool = False,
    symmetric_zero: bool = False,
    xerr: str | None = None,
    yerr: str | None = None,
) -> tuple[float, float]:
    vals = []
    for col, err in [(xcol, xerr), (ycol, yerr)]:
        v = tab[col].to_numpy(float)
        if err is not None and err in tab.columns:
            e = tab[err].to_numpy(float)
            vals.extend([v - e, v + e])
        else:
            vals.append(v)
    arr = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0) if zero_floor else (-1.0, 1.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if zero_floor:
        lo = 0.0
    if symmetric_zero:
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound
    if hi <= lo:
        pad = max(0.01, abs(hi) * 0.08)
    else:
        pad = 0.08 * (hi - lo)
    lo2 = lo if zero_floor else lo - pad
    hi2 = hi + pad
    if symmetric_zero:
        bound = max(abs(lo2), abs(hi2))
        lo2, hi2 = -bound, bound
    return lo2, hi2


def add_identity_and_zero(
    ax: plt.Axes, limits: tuple[float, float], *, show_zero: bool
) -> None:
    lo, hi = limits
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.0,
        color="k",
        alpha=0.65,
        zorder=1,
    )
    if show_zero and lo < 0 < hi:
        ax.axhline(0.0, color="0.55", lw=0.65, alpha=0.55, zorder=0)
        ax.axvline(0.0, color="0.55", lw=0.65, alpha=0.55, zorder=0)


def add_errorbars(
    ax: plt.Axes,
    tab: pd.DataFrame,
    xcol: str,
    ycol: str,
    xerr: str,
    yerr: str,
    color: str,
    args: argparse.Namespace,
) -> None:
    keep = (
        np.isfinite(tab[xcol].to_numpy(float))
        & np.isfinite(tab[ycol].to_numpy(float))
        & np.isfinite(tab[xerr].to_numpy(float))
        & np.isfinite(tab[yerr].to_numpy(float))
    )
    if not np.any(keep):
        return
    ax.errorbar(
        tab.loc[keep, xcol],
        tab.loc[keep, ycol],
        xerr=tab.loc[keep, xerr],
        yerr=tab.loc[keep, yerr],
        fmt="none",
        ecolor=color,
        elinewidth=args.error_lw,
        alpha=args.error_alpha,
        capsize=0,
        zorder=2,
    )


def draw_scatter_panel(
    ax: plt.Axes,
    tab: pd.DataFrame,
    *,
    pop: str,
    xcol: str,
    ycol: str,
    xerr: str | None,
    yerr: str | None,
    show_errorbars: bool,
    zero_floor: bool,
    symmetric_zero: bool,
    show_zero: bool,
    annotation: str,
    limits: tuple[float, float] | None,
    args: argparse.Namespace,
) -> None:
    tab = finite_pair(tab, xcol, ycol)
    color = POP_COLOR.get(pop, POINT_COLOR)

    if limits is None:
        limits = panel_limits(
            tab,
            xcol,
            ycol,
            zero_floor=zero_floor,
            symmetric_zero=symmetric_zero,
            xerr=xerr if show_errorbars else None,
            yerr=yerr if show_errorbars else None,
        )
    add_identity_and_zero(ax, limits, show_zero=show_zero)

    if show_errorbars and xerr is not None and yerr is not None:
        add_errorbars(ax, tab, xcol, ycol, xerr, yerr, color, args)

    sns.scatterplot(
        data=tab,
        x=xcol,
        y=ycol,
        ax=ax,
        s=args.point_size,
        color=color,
        alpha=args.alpha,
        linewidth=0.0,
        zorder=3,
        rasterized=True,
    )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT_DARK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.7,
        },
        zorder=5,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def format_r(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.3f}"


def row_limits_by_pop(
    aligned: pd.DataFrame,
    row_specs: list[dict],
    pops: list[str],
    args: argparse.Namespace,
) -> dict[tuple[int, str], tuple[float, float]]:
    small_pops = [pop for pop in pops if pop not in OWN_SCALE_POPS]
    limits: dict[tuple[int, str], tuple[float, float]] = {}
    pop_col = aligned["pop"].astype(str)

    for row, spec in enumerate(row_specs):
        shared_small = None
        if small_pops:
            shared_small = panel_limits(
                aligned.loc[pop_col.isin(small_pops)].copy(),
                spec["xcol"],
                spec["ycol"],
                zero_floor=bool(spec["zero_floor"]),
                symmetric_zero=bool(spec["symmetric_zero"]),
                xerr=spec["xerr"] if bool(spec["show_errorbars"]) else None,
                yerr=spec["yerr"] if bool(spec["show_errorbars"]) else None,
            )

        for pop in pops:
            if pop in OWN_SCALE_POPS or shared_small is None:
                tab = aligned.loc[pop_col == pop].copy()
                limits[(row, pop)] = panel_limits(
                    tab,
                    spec["xcol"],
                    spec["ycol"],
                    zero_floor=bool(spec["zero_floor"]),
                    symmetric_zero=bool(spec["symmetric_zero"]),
                    xerr=spec["xerr"] if bool(spec["show_errorbars"]) else None,
                    yerr=spec["yerr"] if bool(spec["show_errorbars"]) else None,
                )
            else:
                limits[(row, pop)] = shared_small

    return limits


def set_shared_ticks(ax: plt.Axes, limits: tuple[float, float]) -> None:
    lo, hi = limits
    locator = mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(lo, hi)
    ticks = ticks[(ticks >= lo - 1e-12) & (ticks <= hi + 1e-12)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=False))
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=False))


def draw_figure(
    aligned: pd.DataFrame, metrics: pd.DataFrame, args: argparse.Namespace
) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        len(args.pops),
        figsize=(args.width, args.height),
        sharex=False,
        sharey=False,
        constrained_layout=False,
    )
    if len(args.pops) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    row_specs = [
        {
            "xcol": "rg_x",
            "ycol": "rg_y",
            "xerr": "rg_se_x",
            "yerr": "rg_se_y",
            "show_errorbars": args.rg_error_bars,
            "zero_floor": False,
            "symmetric_zero": True,
            "show_zero": True,
            "left_label": r"Summary-estimated-overlap $\hat r_g$",
            "bottom_label": r"Supplied-overlap $\hat r_g$",
        },
        {
            "xcol": "rg_se_x",
            "ycol": "rg_se_y",
            "xerr": None,
            "yerr": None,
            "show_errorbars": False,
            "zero_floor": True,
            "symmetric_zero": False,
            "show_zero": False,
            "left_label": r"Summary-estimated-overlap SE$(\hat r_g)$",
            "bottom_label": r"Supplied-overlap SE$(\hat r_g)$",
        },
        {
            "xcol": "intercept_c_x",
            "ycol": "intercept_c_y",
            "xerr": "intercept_se_x",
            "yerr": "intercept_se_y",
            "show_errorbars": args.overlap_error_bars,
            "zero_floor": False,
            "symmetric_zero": True,
            "show_zero": True,
            "left_label": r"Summary-estimated $\hat c_{\rm ov}$",
            "bottom_label": r"Supplied $\hat c_{\rm ov}$",
        },
    ]

    row_limits = row_limits_by_pop(aligned, row_specs, args.pops, args)
    letters = [chr(ord("A") + i) for i in range(len(args.pops))]
    for col, pop in enumerate(args.pops):
        sub = aligned[aligned["pop"].astype(str) == pop].copy()
        m = metrics.loc[metrics["pop"] == pop].iloc[0]
        axes[0, col].text(
            0.0,
            1.03,
            f"{letters[col]}.",
            transform=axes[0, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=TEXT_DARK,
            clip_on=False,
        )
        axes[0, col].text(
            0.105,
            1.03,
            POP_LABEL.get(pop, pop),
            transform=axes[0, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=POP_COLOR.get(pop, TEXT_DARK),
            clip_on=False,
        )

        annotations = [
            f"r={format_r(float(m.corr_rg))}",
            f"r={format_r(float(m.corr_rg_se))}",
            f"r={format_r(float(m.corr_intercept))}",
        ]

        for row, spec in enumerate(row_specs):
            ax = axes[row, col]
            draw_scatter_panel(
                ax,
                sub,
                pop=pop,
                xcol=spec["xcol"],
                ycol=spec["ycol"],
                xerr=spec["xerr"],
                yerr=spec["yerr"],
                show_errorbars=bool(spec["show_errorbars"]),
                zero_floor=bool(spec["zero_floor"]),
                symmetric_zero=bool(spec["symmetric_zero"]),
                show_zero=bool(spec["show_zero"]),
                annotation=annotations[row],
                limits=row_limits[(row, pop)],
                args=args,
            )
            ax.set_xlim(*row_limits[(row, pop)])
            ax.set_ylim(*row_limits[(row, pop)])
            set_shared_ticks(ax, row_limits[(row, pop)])
            if col == 0:
                ax.set_ylabel(spec["left_label"])
            else:
                ax.set_ylabel("")
            ax.set_xlabel(spec["bottom_label"])

    fig.subplots_adjust(
        left=0.07, right=0.995, bottom=0.075, top=0.955, wspace=0.17, hspace=0.34
    )
    return fig


def write_outputs(
    aligned: pd.DataFrame,
    metrics: pd.DataFrame,
    fig: plt.Figure,
    out: Path,
    also_png: bool,
    dpi: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if also_png:
        fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    aligned.to_csv(out.with_name(out.stem + "__source_data.csv"), index=False)
    metrics.to_csv(out.with_name(out.stem + "__metrics.csv"), index=False)


def main() -> None:
    args = parse_args()
    set_style()
    base = Path(args.base)
    df = read_inputs(base, args.pops, args.infile)
    df = filter_scenario(df, args)
    aligned = align_methods(df, args)

    metric_rows = []
    for pop in args.pops:
        metric_rows.append(
            summarize_pop(aligned[aligned["pop"].astype(str) == pop].copy(), pop)
        )
    metrics = pd.DataFrame(metric_rows)

    fig = draw_figure(aligned, metrics, args)
    out = Path(args.out)
    write_outputs(aligned, metrics, fig, out, args.also_png, args.dpi)
    plt.close(fig)

    print(f"[write] {out}")
    if args.also_png:
        print(f"[write] {out.with_suffix('.png')}")
    print(f"[write] {out.with_name(out.stem + '__metrics.csv')}")
    for _, row in metrics.iterrows():
        print(
            "[{pop}] paired={n_paired_rg}/{n_total} "
            "FPR={fpr_0p05_x:.2f}/{fpr_0p05_y:.2f} "
            "corr_rg={corr_rg:.3f} median_se={median_rg_se_x:.4g}/{median_rg_se_y:.4g}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
