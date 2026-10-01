#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ALL_BIN_SETTINGS = (1, 8, 24)
ARCH_PREFERRED_ORDER = ("GCTA", "LDAK")
ARCH_TOKEN_RE = re.compile(r"(?:^|_)LDAK(?:_|$)", re.IGNORECASE)


def pretty_pop_label(pop: object) -> str:
    return {"EUR_300k": "EUR (300k)"}.get(str(pop), str(pop))


def infer_arch_token(raw: object) -> str:
    name = Path(str(raw)).name
    if ARCH_TOKEN_RE.search(name):
        return "LDAK"
    return "GCTA"


def normalize_arch(raw: object) -> str:
    if raw is None:
        return "GCTA"
    try:
        if pd.isna(raw):
            return "GCTA"
    except Exception:
        pass

    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return "GCTA"
    if s.upper() in {"GCTA", "LDAK"}:
        return s.upper()
    return infer_arch_token(s)


def ensure_arch_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "arch" not in d.columns:
        d["arch"] = "GCTA"
    d["arch"] = d["arch"].apply(normalize_arch)
    return d


# -------------------------
# Style
# -------------------------
def set_maintext_style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.12)
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
        }
    )


def save_figure(fig: plt.Figure, outpath: Path, also_png: bool = False) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    if also_png:
        fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")


# -------------------------
# Basic I/O
# -------------------------
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] missing: {path}")
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}")
        return None


def load_concat(base: Path, pops: Sequence[str], infile: str) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for pop in pops:
        p = base / pop / infile
        d = _safe_read_csv(p)
        if d is None or d.shape[0] == 0:
            continue
        d = d.copy()
        d["pop"] = str(pop)
        dfs.append(d)
    if not dfs:
        raise RuntimeError("No input data loaded. Check --base/--pops/--infile.")
    return pd.concat(dfs, ignore_index=True)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------------------------
# Method mapping
# -------------------------
def _method_window_match(raw: str, window_kb: int) -> bool:
    s = str(raw)
    if "kb" not in s:
        return True
    return f"_{int(window_kb)}kb" in s


def map_raw_method_to_plot(
    raw: str,
    sumher_window_kb: int = 50000,
    include_sumher_all_windows: bool = False,
) -> str | None:
    s = str(raw)

    if s in {"summit", "summit_cov"}:
        return "summit"

    if s == "ldsc":
        return "ldsc"
    if s in {"ldsc_cov", "covldsc"}:
        return "covldsc"

    if s.startswith("sumher_ldak"):
        if include_sumher_all_windows or _method_window_match(s, sumher_window_kb):
            return "sumher_ldak"
        return None

    if s.startswith("sumher"):
        if include_sumher_all_windows or _method_window_match(s, sumher_window_kb):
            return "sumher"
        return None

    if s.startswith("hdl"):
        return "hdl"

    return None


# -------------------------
# Truth loading
# -------------------------
SCEN_RE = re.compile(
    r"^(?:sims_)?(?P<h2_cfg>flatH2|gradH2)_(?P<rg_cfg>Constrg|Gradrg|Nullrg)(?:$|_.*)",
    re.IGNORECASE,
)


class TrueScenario(dict):
    pass


def _parse_scalar_or_csv(x: object, n: int) -> np.ndarray:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.full(n, np.nan, dtype=float)

    s = str(x).strip().strip('"').strip("'")
    if s == "":
        return np.full(n, np.nan, dtype=float)

    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = np.asarray([float(p) for p in parts], dtype=float)

    if vals.size == 1:
        return np.repeat(vals[0], n).astype(float)
    if vals.size == n:
        return vals.astype(float)

    raise ValueError(f"Expected 1 or {n} values, got {vals.size} from {x!r}")


def safe_total_rg(gamma_total: float, h1_total: float, h2_total: float) -> float:
    if (
        not np.isfinite(gamma_total)
        or not np.isfinite(h1_total)
        or not np.isfinite(h2_total)
    ):
        return np.nan
    if h1_total <= 0.0 or h2_total <= 0.0:
        return np.nan
    return float(gamma_total / np.sqrt(h1_total * h2_total))


def _sqrt_positive_product(x: object, y: object) -> np.ndarray:
    """
    Elementwise sqrt(x * y), but only when both inputs are finite and > 0.
    Otherwise returns NaN.

    This matches the denominator used in rg = gamma / sqrt(h2_1 * h2_2),
    where the denominator is only meaningful when both h2 terms are positive.
    """
    xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    out = np.full(xa.shape, np.nan, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya) & (xa > 0.0) & (ya > 0.0)
    out[mask] = np.sqrt(xa[mask] * ya[mask])
    return out


def load_true_params(
    path: Path | None,
) -> dict[tuple[str, str, str, str], TrueScenario]:
    """
    Reads all_params.mafld.8.csv and returns a mapping keyed by
    (pop, h2_cfg, rg_cfg, arch).

    The architecture is taken from an explicit `arch` column when present;
    otherwise it is inferred from the out_prefix (LDAK token -> LDAK, else GCTA).

    Stored fields:
      h2_1:   per-bin h2 for trait 1 (len 8)
      h2_2:   per-bin h2 for trait 2 (len 8)
      denom:  per-bin rg denominator sqrt(h2_1 * h2_2) (len 8)
      rg:     per-bin rg (len 8)
      gamma:  per-bin genetic covariance (len 8)

      total_h2_1, total_h2_2, total_denom, total_gamma, total_rg_from_bins
    """
    if path is None:
        return {}
    if not path.exists():
        print(f"[warn] true-params file not found: {path}")
        return {}

    df = pd.read_csv(path)
    need = {"pop", "out_prefix", "sig1", "sig2", "rho_g"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(
            f"[warn] true-params file missing columns {missing}; skipping truth lines requiring that file."
        )
        return {}

    out: dict[tuple[str, str, str, str], TrueScenario] = {}
    for _, row in df.iterrows():
        out_prefix = str(row["out_prefix"]).strip()
        m = SCEN_RE.match(out_prefix)
        if not m:
            continue

        h2_cfg = m.group("h2_cfg")
        rg_cfg = m.group("rg_cfg")
        pop = str(row["pop"]).strip()
        arch = (
            normalize_arch(row["arch"])
            if "arch" in df.columns
            else infer_arch_token(out_prefix)
        )

        h2_1 = np.asarray(_parse_scalar_or_csv(row["sig1"], 8), dtype=float)
        h2_2 = np.asarray(_parse_scalar_or_csv(row["sig2"], 8), dtype=float)
        rg = np.asarray(_parse_scalar_or_csv(row["rho_g"], 8), dtype=float)

        denom = np.asarray(_sqrt_positive_product(h2_1, h2_2), dtype=float)
        gamma = rg * denom

        total_h2_1 = float(np.nansum(h2_1))
        total_h2_2 = float(np.nansum(h2_2))
        total_denom = float(_sqrt_positive_product(total_h2_1, total_h2_2))
        total_gamma = float(np.nansum(gamma))
        total_rg_from_bins = safe_total_rg(total_gamma, total_h2_1, total_h2_2)

        out[(pop, h2_cfg, rg_cfg, arch)] = dict(
            h2_1=h2_1,
            h2_2=h2_2,
            denom=denom,
            rg=rg,
            gamma=gamma,
            total_h2_1=total_h2_1,
            total_h2_2=total_h2_2,
            total_denom=total_denom,
            total_gamma=total_gamma,
            total_rg_from_bins=total_rg_from_bins,
        )

    return out


def get_truth_record(
    true_map: dict[tuple[str, str, str, str], TrueScenario],
    pop: str,
    h2_cfg: str,
    rg_cfg: str,
    arch: str,
) -> TrueScenario | None:
    arch_norm = normalize_arch(arch)

    for cand_arch in [arch_norm, "GCTA", "LDAK"]:
        key = (pop, h2_cfg, rg_cfg, cand_arch)
        if key in true_map:
            return true_map[key]

    for (p, h, r, _a), rec in true_map.items():
        if p == pop and h == h2_cfg and r == rg_cfg:
            return rec

    return None


def true_total_rg_for_cfg(
    rg_cfg: str,
    true_rg_constr: float,
    true_rg_grad: float,
    true_rg_null: float,
) -> float:
    s = str(rg_cfg).strip().lower()
    if s == "constrg":
        return float(true_rg_constr)
    if s == "gradrg":
        return float(true_rg_grad)
    if s == "nullrg":
        return float(true_rg_null)
    return float("nan")


def truth_total_for_estimand(
    estimand: str,
    truth_rec: dict | None,
    rg_cfg: str,
    true_rg_constr: float,
    true_rg_grad: float,
    true_rg_null: float,
) -> float:
    estimand = str(estimand)

    if estimand == "rg":
        if truth_rec is not None and "total_rg_from_bins" in truth_rec:
            return float(truth_rec["total_rg_from_bins"])
        return true_total_rg_for_cfg(
            rg_cfg=rg_cfg,
            true_rg_constr=true_rg_constr,
            true_rg_grad=true_rg_grad,
            true_rg_null=true_rg_null,
        )

    if truth_rec is None:
        return float("nan")

    if estimand == "h2_1":
        return float(truth_rec["total_h2_1"])
    if estimand == "h2_2":
        return float(truth_rec["total_h2_2"])
    if estimand == "denom":
        return float(truth_rec["total_denom"])
    if estimand == "gamma":
        return float(truth_rec["total_gamma"])

    raise ValueError(f"Unknown estimand: {estimand}")


# -------------------------
# Estimand metadata
# -------------------------
def estimand_total_col(estimand: str) -> str:
    return {
        "h2_1": "h2_1",
        "h2_2": "h2_2",
        "denom": "denom",
        "gamma": "gamma_g",
        "rg": "rg",
    }[str(estimand)]


def estimand_bin_prefix(estimand: str) -> str:
    return {
        "h2_1": "h2_1_bin_",
        "h2_2": "h2_2_bin_",
        "denom": "denom_bin_",
        "gamma": "gamma_bin_",
        "rg": "rg_bin_",
    }[str(estimand)]


def estimand_math_label(estimand: str) -> str:
    return {
        "h2_1": r"$h^2_1$",
        "h2_2": r"$h^2_2$",
        "denom": r"$\sqrt{h^2_1 h^2_2}$",
        "gamma": r"$\gamma_g$",
        "rg": r"$r_g$",
    }[str(estimand)]


def estimand_title_label(estimand: str) -> str:
    return {
        "h2_1": r"trait-1 $h^2$",
        "h2_2": r"trait-2 $h^2$",
        "denom": r"$r_g$ denominator $\sqrt{h^2_1 h^2_2}$",
        "gamma": r"genetic covariance $\gamma_g$",
        "rg": r"genetic correlation $r_g$",
    }[str(estimand)]


# -------------------------
# Prep / reshape
# -------------------------
def _truthy_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def prepare_df(
    df: pd.DataFrame,
    bin_setting: int,
    methods_keep: Sequence[str],
    estimand: str,
) -> pd.DataFrame:
    d = df.copy()
    d = ensure_arch_column(d)
    d["method"] = d["method"].astype(str)
    d["method_plot"] = d["method"].apply(map_raw_method_to_plot)
    d = d.dropna(subset=["method_plot"]).copy()
    d = d[d["method_plot"].isin(methods_keep)].copy()

    estimand = str(estimand)

    if estimand == "denom":
        # Need the h2 columns first so we can derive denom and denom_bin_k.
        numeric_cols = ["nbins", "i", "h2_1", "h2_2"]
        numeric_cols += [f"h2_1_bin_{k}" for k in range(24)]
        numeric_cols += [f"h2_2_bin_{k}" for k in range(24)]
        d = coerce_numeric(d, numeric_cols)

        need = {"method", "h2_cfg", "rg_cfg", "nbins", "i", "pop", "h2_1", "h2_2"}
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns for estimand='denom': {missing}"
            )

        d["denom"] = _sqrt_positive_product(
            d["h2_1"].to_numpy(float), d["h2_2"].to_numpy(float)
        )

        for k in range(24):
            c1 = f"h2_1_bin_{k}"
            c2 = f"h2_2_bin_{k}"
            if c1 in d.columns and c2 in d.columns:
                d[f"denom_bin_{k}"] = _sqrt_positive_product(
                    d[c1].to_numpy(float),
                    d[c2].to_numpy(float),
                )
    else:
        total_col = estimand_total_col(estimand)
        bin_prefix = estimand_bin_prefix(estimand)

        need = {"method", "h2_cfg", "rg_cfg", "nbins", "i", "pop", total_col}
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        numeric_cols = ["nbins", "i", total_col] + [
            f"{bin_prefix}{k}" for k in range(24)
        ]
        d = coerce_numeric(d, numeric_cols)

    total_col = estimand_total_col(estimand)
    bin_prefix = estimand_bin_prefix(estimand)

    d = d[d["nbins"] == int(bin_setting)].copy()

    if "parse_ok" in d.columns:
        d = d[_truthy_mask(d["parse_ok"])].copy()

    needed_bin_cols = [f"{bin_prefix}{k}" for k in range(int(bin_setting))]
    missing_bins = [c for c in needed_bin_cols if c not in d.columns]
    if missing_bins:
        raise ValueError(
            f"Input CSV is missing per-bin columns for estimand={estimand!r} and --bin {bin_setting}: {missing_bins}"
        )

    d["h2_cfg"] = d["h2_cfg"].astype(str)
    d["rg_cfg"] = d["rg_cfg"].astype(str)
    d["arch"] = d["arch"].astype(str)
    d["pop"] = d["pop"].astype(str)
    d = d.dropna(subset=["pop", "method_plot", "h2_cfg", "rg_cfg", "arch", "i"]).copy()
    return d


def prepare_df_all_bins(
    df: pd.DataFrame,
    methods_keep: Sequence[str],
    estimand: str,
    bin_settings: Sequence[int] = ALL_BIN_SETTINGS,
) -> pd.DataFrame:
    d = df.copy()
    d = ensure_arch_column(d)
    d["method"] = d["method"].astype(str)
    d["method_plot"] = d["method"].apply(map_raw_method_to_plot)
    d = d.dropna(subset=["method_plot"]).copy()
    d = d[d["method_plot"].isin(methods_keep)].copy()

    estimand = str(estimand)

    if estimand == "denom":
        d = coerce_numeric(d, ["nbins", "i", "h2_1", "h2_2"])

        need = {"method", "h2_cfg", "rg_cfg", "nbins", "i", "pop", "h2_1", "h2_2"}
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns for estimand='denom': {missing}"
            )

        d["denom"] = _sqrt_positive_product(
            d["h2_1"].to_numpy(float), d["h2_2"].to_numpy(float)
        )
    else:
        total_col = estimand_total_col(estimand)
        need = {"method", "h2_cfg", "rg_cfg", "nbins", "i", "pop", total_col}
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        d = coerce_numeric(d, ["nbins", "i", total_col])

    keep_bins = {int(b) for b in bin_settings}
    d = d[d["nbins"].isin(sorted(keep_bins))].copy()
    d["nbins"] = d["nbins"].astype(int)

    if "parse_ok" in d.columns:
        d = d[_truthy_mask(d["parse_ok"])].copy()

    total_col = estimand_total_col(estimand)

    d["h2_cfg"] = d["h2_cfg"].astype(str)
    d["rg_cfg"] = d["rg_cfg"].astype(str)
    d["arch"] = d["arch"].astype(str)
    d["pop"] = d["pop"].astype(str)
    d = d.dropna(
        subset=["pop", "method_plot", "h2_cfg", "rg_cfg", "arch", "i", total_col]
    ).copy()
    return d


def build_raw_tick_df(
    df: pd.DataFrame,
    bin_setting: int,
    include_total: bool,
    estimand: str,
) -> pd.DataFrame:
    total_col = estimand_total_col(estimand)
    bin_prefix = estimand_bin_prefix(estimand)

    id_cols = ["pop", "arch", "method_plot", "h2_cfg", "rg_cfg", "i", "nbins"]
    value_cols = [f"{bin_prefix}{k}" for k in range(int(bin_setting))]

    long_bins = (
        df[id_cols + value_cols]
        .melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name="tick_col",
            value_name="est_raw",
        )
        .copy()
    )
    long_bins["tick_kind"] = "bin"
    long_bins["tick_idx"] = (
        long_bins["tick_col"].astype(str).str.extract(r"_(\d+)$")[0].astype(int)
    )
    long_bins["tick_order"] = long_bins["tick_idx"] + 1
    long_bins["tick_name"] = long_bins["tick_idx"].map(lambda k: f"bin_{int(k) + 1}")
    long_bins = long_bins.drop(columns=["tick_col"])

    frames = [long_bins]
    if include_total:
        total_df = (
            df[id_cols + [total_col]].copy().rename(columns={total_col: "est_raw"})
        )
        total_df["tick_kind"] = "total"
        total_df["tick_idx"] = int(bin_setting)
        total_df["tick_order"] = int(bin_setting) + 1
        total_df["tick_name"] = "total"
        frames.append(total_df)

    out = pd.concat(frames, ignore_index=True)
    out["valid_raw"] = np.isfinite(out["est_raw"].to_numpy(float))
    return out


def _nbins_tick_order_map(bin_settings: Sequence[int]) -> dict[int, int]:
    ordered = [int(b) for b in bin_settings]
    return {b: i + 1 for i, b in enumerate(ordered)}


def _nbins_tick_label(nbins: int) -> str:
    n = int(nbins)
    return f"{n} bin" if n == 1 else f"{n} bins"


def build_raw_total_across_bins_df(
    df: pd.DataFrame,
    estimand: str,
    bin_settings: Sequence[int] = ALL_BIN_SETTINGS,
) -> pd.DataFrame:
    total_col = estimand_total_col(estimand)
    keep_bins = [int(b) for b in bin_settings]
    order_map = _nbins_tick_order_map(keep_bins)

    d = df[df["nbins"].isin(keep_bins)].copy()
    id_cols = ["pop", "arch", "method_plot", "h2_cfg", "rg_cfg", "i", "nbins"]
    out = d[id_cols + [total_col]].copy().rename(columns={total_col: "est_raw"})
    out["tick_kind"] = "fit_nbins"
    out["tick_idx"] = out["nbins"].astype(int)
    out["tick_order"] = out["tick_idx"].map(order_map).astype(int)
    out["tick_name"] = out["tick_idx"].map(lambda x: f"{int(x)}bin")
    out["tick_label"] = out["tick_idx"].map(_nbins_tick_label)
    out["valid_raw"] = np.isfinite(out["est_raw"].to_numpy(float))
    return out


def add_overlap_flag(
    raw_df: pd.DataFrame, methods_required: Sequence[str]
) -> pd.DataFrame:
    key_cols = ["pop", "arch", "h2_cfg", "rg_cfg", "i", "tick_name"]
    want_n = int(len(list(methods_required)))

    valid_methods = (
        raw_df.loc[raw_df["valid_raw"].fillna(False), key_cols + ["method_plot"]]
        .drop_duplicates()
        .groupby(key_cols, dropna=False)["method_plot"]
        .nunique()
        .rename("n_methods_valid")
        .reset_index()
    )

    out = raw_df.merge(valid_methods, on=key_cols, how="left")
    out["n_methods_valid"] = out["n_methods_valid"].fillna(0).astype(int)
    out["overlap_ok"] = out["n_methods_valid"] >= want_n
    return out


# -------------------------
# Truth tables / transforms
# -------------------------
def build_truth_df(
    panels: pd.DataFrame,
    estimand: str,
    bin_setting: int,
    include_total: bool,
    plot_mode: str,
    true_map: dict[tuple[str, str, str, str], TrueScenario],
    true_rg_constr: float,
    true_rg_grad: float,
    true_rg_null: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_bins = int(bin_setting)
    n_points = n_bins + (1 if include_total else 0)

    for panel in panels.itertuples(index=False):
        pop = str(panel.pop)
        arch = normalize_arch(panel.arch)
        h2_cfg = str(panel.h2_cfg)
        rg_cfg = str(panel.rg_cfg)
        rec = get_truth_record(
            true_map, pop=pop, h2_cfg=h2_cfg, rg_cfg=rg_cfg, arch=arch
        )

        raw_truth = np.full(n_points, np.nan, dtype=float)
        total_truth = truth_total_for_estimand(
            estimand=estimand,
            rg_cfg=rg_cfg,
            truth_rec=rec,
            true_rg_constr=true_rg_constr,
            true_rg_grad=true_rg_grad,
            true_rg_null=true_rg_null,
        )

        if n_bins == 1:
            raw_truth[0] = total_truth
        elif n_bins == 8 and rec is not None:
            raw_truth[:8] = np.asarray(rec[estimand], dtype=float)

        if include_total:
            raw_truth[-1] = total_truth

        truth_plot = raw_truth.copy()
        if plot_mode in {"error", "mse"}:
            truth_plot = np.where(np.isfinite(raw_truth), 0.0, np.nan)

        for j in range(n_bins):
            rows.append(
                {
                    "pop": pop,
                    "arch": arch,
                    "h2_cfg": h2_cfg,
                    "rg_cfg": rg_cfg,
                    "tick_kind": "bin",
                    "tick_idx": j,
                    "tick_order": j + 1,
                    "tick_name": f"bin_{j + 1}",
                    "truth_estimate": raw_truth[j],
                    "truth_plot": truth_plot[j],
                }
            )

        if include_total:
            rows.append(
                {
                    "pop": pop,
                    "arch": arch,
                    "h2_cfg": h2_cfg,
                    "rg_cfg": rg_cfg,
                    "tick_kind": "total",
                    "tick_idx": n_bins,
                    "tick_order": n_bins + 1,
                    "tick_name": "total",
                    "truth_estimate": raw_truth[-1],
                    "truth_plot": truth_plot[-1],
                }
            )

    return pd.DataFrame(rows)


def build_truth_total_across_bins_df(
    panels: pd.DataFrame,
    estimand: str,
    plot_mode: str,
    true_map: dict[tuple[str, str, str, str], TrueScenario],
    true_rg_constr: float,
    true_rg_grad: float,
    true_rg_null: float,
    bin_settings: Sequence[int] = ALL_BIN_SETTINGS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keep_bins = [int(b) for b in bin_settings]
    order_map = _nbins_tick_order_map(keep_bins)

    for panel in panels.itertuples(index=False):
        pop = str(panel.pop)
        arch = normalize_arch(panel.arch)
        h2_cfg = str(panel.h2_cfg)
        rg_cfg = str(panel.rg_cfg)
        rec = get_truth_record(
            true_map, pop=pop, h2_cfg=h2_cfg, rg_cfg=rg_cfg, arch=arch
        )

        total_truth = truth_total_for_estimand(
            estimand=estimand,
            rg_cfg=rg_cfg,
            truth_rec=rec,
            true_rg_constr=true_rg_constr,
            true_rg_grad=true_rg_grad,
            true_rg_null=true_rg_null,
        )
        truth_plot = (
            0.0
            if (plot_mode in {"error", "mse"} and np.isfinite(total_truth))
            else total_truth
        )

        for nbins in keep_bins:
            rows.append(
                {
                    "pop": pop,
                    "arch": arch,
                    "h2_cfg": h2_cfg,
                    "rg_cfg": rg_cfg,
                    "tick_kind": "fit_nbins",
                    "tick_idx": int(nbins),
                    "tick_order": int(order_map[int(nbins)]),
                    "tick_name": f"{int(nbins)}bin",
                    "truth_estimate": total_truth,
                    "truth_plot": truth_plot,
                }
            )

    return pd.DataFrame(rows)


def apply_plot_transform(plot_df: pd.DataFrame, plot_mode: str) -> pd.DataFrame:
    d = plot_df.copy()
    est = d["est_raw"].to_numpy(float)
    tru = d["truth_estimate"].to_numpy(float)

    if plot_mode == "estimate":
        d["plot_value"] = est
    elif plot_mode == "error":
        out = np.full(est.shape, np.nan, dtype=float)
        mask = np.isfinite(est) & np.isfinite(tru)
        out[mask] = est[mask] - tru[mask]
        d["plot_value"] = out
    elif plot_mode == "mse":
        out = np.full(est.shape, np.nan, dtype=float)
        mask = np.isfinite(est) & np.isfinite(tru)
        out[mask] = np.square(est[mask] - tru[mask])
        d["plot_value"] = out
    else:
        raise ValueError(f"Unknown plot mode: {plot_mode}")

    return d


def validate_truth_requirements(
    plot_df: pd.DataFrame,
    plot_mode: str,
    bin_setting: int,
    total_only: bool = False,
) -> None:
    if plot_mode not in {"error", "mse"}:
        return

    bin_setting = int(bin_setting)
    if not total_only and plot_mode == "error" and bin_setting not in {1, 8}:
        raise ValueError("--plot-error is only supported for --bin 1 or --bin 8.")
    if not total_only and plot_mode == "mse" and bin_setting not in {1, 8, 24}:
        raise ValueError(
            "--plot-mse is only supported for --bin 1, --bin 8, or total-only for --bin 24."
        )

    relevant = np.ones(plot_df.shape[0], dtype=bool)
    if not total_only and plot_mode == "mse" and bin_setting == 24:
        relevant = plot_df["tick_kind"].astype(str).eq("total").to_numpy(bool)

    bad = plot_df[
        relevant
        & np.isfinite(plot_df["est_raw"].to_numpy(float))
        & ~np.isfinite(plot_df["truth_estimate"].to_numpy(float))
    ].copy()
    if bad.shape[0] == 0:
        return

    show = (
        bad[["pop", "arch", "h2_cfg", "rg_cfg", "tick_name"]]
        .drop_duplicates()
        .sort_values(["pop", "arch", "h2_cfg", "rg_cfg", "tick_name"])
        .head(10)
    )
    raise RuntimeError(
        "Missing truth values needed for the requested truth-referenced plot. "
        f"Examples:\n{show.to_string(index=False)}"
    )


# -------------------------
# Count table
# -------------------------
def compute_n_valid_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = [
        "pop",
        "method_plot",
        "arch",
        "h2_cfg",
        "rg_cfg",
        "tick_kind",
        "tick_idx",
        "tick_order",
        "tick_name",
    ]
    out = (
        raw_df.groupby(grp_cols, dropna=False)["valid_raw"]
        .sum()
        .rename("n_valid")
        .reset_index()
    )
    out["n_valid"] = out["n_valid"].astype(float)
    return out


# -------------------------
# Labels / formatting
# -------------------------
def parse_csv_arg(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def make_bin_ticklabels(bin_setting: int, include_total: bool) -> list[str]:
    """
    parse_part.py flattens away the original bin names, so for 8 bins we only
    reconstruct the MAF-LD structure described by the user. 24-bin labels are
    numeric indices.
    """
    if bin_setting == 1:
        labels = ["1\nAll"]
    elif bin_setting == 8:
        labels = []
        for i in range(8):
            ldq = (i % 4) + 1
            labels.append(f"{i + 1}\nQ{ldq}")
    else:
        labels = [str(i) for i in range(1, bin_setting + 1)]

    if include_total:
        labels.append("Total")
    return labels


def make_all_bin_ticklabels(
    bin_settings: Sequence[int] = ALL_BIN_SETTINGS,
) -> list[str]:
    return [_nbins_tick_label(b) for b in bin_settings]


def maybe_annotate_8bin_groups(ax: plt.Axes) -> None:
    ymin, ymax = ax.get_ylim()
    y = ymax - 0.03 * (ymax - ymin)
    ax.axvline(4.5, color="0.55", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(2.5, y, "MAF < 0.1", ha="center", va="top", fontsize=7.8, color="0.35")
    ax.text(6.5, y, "MAF ≥ 0.1", ha="center", va="top", fontsize=7.8, color="0.35")


def pretty_h2_label(h2_cfg: str) -> str:
    return {"flatH2": "flatH2", "gradH2": "gradH2"}.get(h2_cfg, h2_cfg)


def pretty_arch_label(arch: str) -> str:
    return normalize_arch(arch)


def arch_order_from_df(df: pd.DataFrame) -> list[str]:
    if "arch" not in df.columns or df.shape[0] == 0:
        return ["GCTA"]

    present: list[str] = []
    for raw in pd.Series(df["arch"]).dropna().tolist():
        arch = normalize_arch(raw)
        if arch not in present:
            present.append(arch)

    ordered = [a for a in ARCH_PREFERRED_ORDER if a in present]
    extras = sorted([a for a in present if a not in ordered])
    out = ordered + extras
    return out if out else ["GCTA"]


def pretty_rg_label(rg_cfg: str) -> str:
    return {"Nullrg": "Nullrg", "Constrg": "Constrg", "Gradrg": "Gradrg"}.get(
        rg_cfg, rg_cfg
    )


def _cfg_eq(series: pd.Series, target: str) -> pd.Series:
    return series.astype(str).str.lower().eq(str(target).lower())


def parse_ylim_arg(s: str | None) -> tuple[float, float] | None:
    if s is None:
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--ylim must look like: min,max")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("--ylim must contain two finite numbers with min < max.")
    return (lo, hi)


def mode_ylabel(plot_mode: str, estimand: str) -> str:
    mathlab = estimand_math_label(estimand)
    if plot_mode == "estimate":
        return f"Estimated {mathlab}"
    if plot_mode == "error":
        return rf"$\hat{{{mathlab[1:-1]}}}$ - true {mathlab}"
    if plot_mode == "mse":
        return f"Squared error in {mathlab}"
    return "Number of valid estimates"


def mode_title(plot_mode: str) -> str:
    if plot_mode == "estimate":
        return "estimates"
    if plot_mode == "error":
        return "errors"
    if plot_mode == "mse":
        return "squared errors"
    return "valid-estimate counts"


def center_stat(vals: np.ndarray, center: str) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    if center == "median":
        return float(np.nanmedian(vals))
    return float(np.nanmean(vals))


# -------------------------
# Y scaling
# -------------------------
def compute_ylim_from_values(
    values: np.ndarray,
    refs: np.ndarray,
    user_ylim: tuple[float, float] | None,
    nonnegative: bool = False,
) -> tuple[float, float]:
    if user_ylim is not None:
        return user_ylim

    vals = np.asarray(values, dtype=float)
    refs = np.asarray(refs, dtype=float)

    vals = vals[np.isfinite(vals)]
    refs = refs[np.isfinite(refs)]

    if nonnegative:
        vals = vals[vals >= 0.0]
        refs = refs[refs >= 0.0]
        all_vals = np.concatenate([vals, refs, np.array([0.0], dtype=float)])
        if all_vals.size == 0:
            return (0.0, 1.0)

        hi = float(np.nanmax(all_vals))
        if not np.isfinite(hi) or hi <= 0.0:
            return (0.0, 1.0)

        pad = 0.08 * hi
        if pad <= 0.0:
            pad = 0.1
        return (0.0, hi + pad)

    all_vals = np.concatenate([vals, refs, np.array([0.0], dtype=float)])
    if all_vals.size == 0:
        return (-1.0, 1.0)

    lo = float(np.nanmin(all_vals))
    hi = float(np.nanmax(all_vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)

    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    if math.isclose(lo, hi):
        pad = 0.1 if hi == 0.0 else 0.1 * abs(hi)
        return (lo - pad, hi + pad)

    pad = 0.08 * (hi - lo)
    return (lo - pad, hi + pad)


def compute_ylim_map(
    h2_cfg: str,
    arch: str,
    pops: Sequence[str],
    y_scale: str,
    plot_mode: str,
    plot_df: pd.DataFrame | None,
    counts_df: pd.DataFrame | None,
    truth_df: pd.DataFrame | None,
    user_ylim: tuple[float, float] | None,
    center: str,
) -> dict[str, tuple[float, float]]:
    nonnegative = plot_mode in {"mse", "nvalid"}

    def _visible_box_values(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.empty(0, dtype=float)
        if vals.size < 4:
            return vals

        q1, q3 = np.nanquantile(vals, [0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0.0:
            return vals

        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        keep = (vals >= lo_fence) & (vals <= hi_fence)
        vis = vals[keep]
        return vis if vis.size else vals

    def _drawn_values_for(pop: str | None) -> tuple[np.ndarray, np.ndarray]:
        if plot_mode == "nvalid":
            assert counts_df is not None
            sub = counts_df[
                _cfg_eq(counts_df["h2_cfg"], h2_cfg) & _cfg_eq(counts_df["arch"], arch)
            ].copy()
            if pop is not None:
                sub = sub[sub["pop"].astype(str).eq(str(pop))].copy()

            drawn: list[float] = []
            if sub.shape[0]:
                group_cols = ["pop", "rg_cfg", "method_plot", "tick_order"]
                for _, g in sub.groupby(group_cols, dropna=False):
                    y = g["n_valid"].to_numpy(float)
                    y = y[np.isfinite(y)]
                    if y.size:
                        drawn.append(float(y[0]))
            return np.asarray(drawn, dtype=float), np.empty(0, dtype=float)

        assert plot_df is not None and truth_df is not None
        sub = plot_df[
            _cfg_eq(plot_df["h2_cfg"], h2_cfg) & _cfg_eq(plot_df["arch"], arch)
        ].copy()
        tr = truth_df[
            _cfg_eq(truth_df["h2_cfg"], h2_cfg) & _cfg_eq(truth_df["arch"], arch)
        ].copy()

        if pop is not None:
            sub = sub[sub["pop"].astype(str).eq(str(pop))].copy()
            tr = tr[tr["pop"].astype(str).eq(str(pop))].copy()

        drawn: list[float] = []
        if sub.shape[0]:
            group_cols = ["pop", "rg_cfg", "method_plot", "tick_order"]
            for _, g in sub.groupby(group_cols, dropna=False):
                y = g["plot_value"].to_numpy(float)
                y = y[np.isfinite(y)]
                if y.size == 0:
                    continue

                vis = _visible_box_values(y)
                if vis.size:
                    drawn.append(float(np.nanmin(vis)))
                    drawn.append(float(np.nanmax(vis)))

                ctr = center_stat(y, center)
                if np.isfinite(ctr):
                    drawn.append(float(ctr))

        refs = (
            tr["truth_plot"].to_numpy(float)
            if tr.shape[0]
            else np.empty(0, dtype=float)
        )
        return np.asarray(drawn, dtype=float), refs

    if y_scale not in {"shared", "independent"}:
        raise ValueError(f"Unknown y-scale: {y_scale}")

    if y_scale == "shared":
        values, refs = _drawn_values_for(None)
        shared_ylim = compute_ylim_from_values(
            values=values,
            refs=refs,
            user_ylim=user_ylim,
            nonnegative=nonnegative,
        )
        return {str(pop): shared_ylim for pop in pops}

    out: dict[str, tuple[float, float]] = {}
    for pop in pops:
        values, refs = _drawn_values_for(pop)
        out[str(pop)] = compute_ylim_from_values(
            values=values,
            refs=refs,
            user_ylim=user_ylim,
            nonnegative=nonnegative,
        )
    return out


# -------------------------
# Plot helpers
# -------------------------
def panel_has_any_data(
    mode: str,
    sub: pd.DataFrame,
    truth_vals: np.ndarray | None,
) -> bool:
    if mode == "nvalid":
        return sub.shape[0] > 0
    if sub.shape[0] > 0 and np.isfinite(sub["plot_value"].to_numpy(float)).any():
        return True
    if (
        truth_vals is not None
        and np.isfinite(np.asarray(truth_vals, dtype=float)).any()
    ):
        return True
    return False


def draw_method_box_and_line(
    ax: plt.Axes,
    sub: pd.DataFrame,
    method: str,
    x_base: np.ndarray,
    color: str,
    marker: str,
    offset: float,
    width: float,
    center: str,
) -> None:
    sm = sub[sub["method_plot"] == method].copy()
    if sm.shape[0] == 0:
        return

    box_data: list[np.ndarray] = []
    box_pos: list[float] = []
    center_y: list[float] = []

    for x in x_base:
        vals = sm.loc[sm["tick_order"] == int(x), "plot_value"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            box_data.append(vals)
            box_pos.append(float(x) + offset)
            center_y.append(center_stat(vals, center))
        else:
            center_y.append(np.nan)

    if box_data:
        ax.boxplot(
            box_data,
            positions=box_pos,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(
                facecolor=to_rgba(color, 0.24), edgecolor=color, linewidth=1.0
            ),
            whiskerprops=dict(color=color, linewidth=0.95),
            capprops=dict(color=color, linewidth=0.95),
            medianprops=dict(color=color, linewidth=1.25),
        )

    ax.plot(
        x_base + offset,
        np.asarray(center_y, dtype=float),
        color=color,
        linewidth=1.7,
        marker=marker,
        markersize=4.0,
        alpha=0.95,
        zorder=3,
    )


def draw_method_count_line(
    ax: plt.Axes,
    sub: pd.DataFrame,
    method: str,
    x_base: np.ndarray,
    color: str,
    marker: str,
    offset: float,
) -> None:
    sm = sub[sub["method_plot"] == method].copy()

    yvals: list[float] = []
    for x in x_base:
        vals = sm.loc[sm["tick_order"] == int(x), "n_valid"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        yvals.append(float(vals[0]) if vals.size else 0.0)

    ax.plot(
        x_base + offset,
        np.asarray(yvals, dtype=float),
        color=color,
        linewidth=1.9,
        marker=marker,
        markersize=5.0,
        alpha=0.95,
        zorder=3,
    )


def truth_style_for_panel(
    mode: str, bin_setting: int, truth_curve: np.ndarray | None
) -> tuple[str, str] | None:
    if mode == "nvalid" or truth_curve is None:
        return None
    t = np.asarray(truth_curve, dtype=float)
    if not np.isfinite(t).any():
        return None

    if mode == "estimate":
        per_bin_truth = False
        if int(bin_setting) == 8:
            n_take = min(8, t.size)
            per_bin_truth = np.isfinite(t[:n_take]).any()
        if per_bin_truth:
            return ("black", "True (8-bin)")
        return (to_hex(colormaps["tab20"].colors[0]), "True")

    return (to_hex(colormaps["tab20"].colors[0]), "Truth/reference = 0")


def flat_truth_style_for_total_across_bins(mode: str) -> tuple[str, str] | None:
    if mode == "nvalid":
        return None
    if mode == "estimate":
        return ("black", "True total")
    return (to_hex(colormaps["tab20"].colors[0]), "Truth/reference = 0")


def subtitle_for_mode(
    mode: str,
    center: str,
    estimand: str,
    bin_setting: int,
    include_total: bool,
    overlap_only: bool,
) -> str:
    mathlab = estimand_math_label(estimand)
    total_only_24bin_mse = mode == "mse" and int(bin_setting) == 24

    if mode == "estimate":
        parts = [f"lines = {center}", f"boxes = replicate {mathlab} distribution"]
        if include_total:
            parts.append(f"last x-tick = total {mathlab}")
        if int(bin_setting) == 8:
            parts.append(
                "dashed black = 8-bin truth from all_params.mafld.8.csv when available"
            )
        elif int(bin_setting) == 1:
            parts.append("dashed blue = scalar truth for the fitted 1-bin parameter")
        else:
            parts.append(
                "per-bin truth omitted for 24-bin fits; total truth shown at the last tick when available"
            )
    elif mode == "error":
        parts = [
            f"lines = {center} error",
            f"boxes = replicate {mathlab} error distribution",
            "dashed blue = truth/reference at 0",
        ]
        if include_total:
            parts.append(f"last x-tick = total {mathlab} error")
    elif mode == "mse":
        if total_only_24bin_mse:
            parts = [
                f"24-bin fits: only total {mathlab} squared error is plotted",
                f"lines = {center} total squared error",
                f"boxes = replicate total {mathlab} squared-error distribution",
                "dashed blue = truth/reference at 0",
            ]
        else:
            parts = [
                f"lines = {center} squared error",
                f"boxes = replicate {mathlab} squared-error distribution",
                "dashed blue = truth/reference at 0",
            ]
            if include_total:
                parts.append(f"last x-tick = total {mathlab} squared error")
    elif mode == "nvalid":
        parts = ["lines = number of finite estimates"]
        if include_total:
            parts.append("last x-tick = total valid-count")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if overlap_only:
        parts.append("overlap-only = both methods finite at the same replicate/tick")
    return "; ".join(parts)


def subtitle_for_total_across_bins_mode(
    mode: str,
    center: str,
    estimand: str,
    overlap_only: bool,
) -> str:
    mathlab = estimand_math_label(estimand)

    if mode == "estimate":
        parts = [
            f"lines = {center} total {mathlab}",
            f"boxes = replicate total {mathlab} distribution",
            "x-axis = fitted bin setting (1, 8, 24)",
            "dashed line = true total value",
        ]
    elif mode == "error":
        parts = [
            f"lines = {center} total error",
            f"boxes = replicate total {mathlab} error distribution",
            "x-axis = fitted bin setting (1, 8, 24)",
            "dashed line = truth/reference at 0",
        ]
    elif mode == "mse":
        parts = [
            f"lines = {center} total squared error",
            f"boxes = replicate total {mathlab} squared-error distribution",
            "x-axis = fitted bin setting (1, 8, 24)",
            "dashed line = truth/reference at 0",
        ]
    elif mode == "nvalid":
        parts = [
            "lines = number of finite total estimates",
            "x-axis = fitted bin setting (1, 8, 24)",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if overlap_only:
        parts.append(
            "overlap-only = both methods finite at the same replicate/bin-setting"
        )
    return "; ".join(parts)


def make_outstem(
    outprefix: str,
    estimand: str,
    mode: str,
    hide_full: bool,
    y_scale: str,
    overlap_only: bool,
    plot_all_bins: bool = False,
) -> str:
    stem = str(outprefix)
    stem += f"_{estimand}"
    if mode == "error":
        stem += "_error"
    elif mode == "mse":
        stem += "_mse"
    elif mode == "nvalid":
        stem += "_nvalid"

    if hide_full:
        stem += "_nofull"
    if y_scale == "independent":
        stem += "_indepy"
    if overlap_only:
        stem += "_overlap"
    if plot_all_bins:
        stem += "_allbins"
    return stem


# -------------------------
# Main plotting
# -------------------------
def plot_partitioned_grid(
    raw_df: pd.DataFrame,
    plot_df: pd.DataFrame | None,
    counts_df: pd.DataFrame | None,
    truth_df: pd.DataFrame | None,
    estimand: str,
    h2_cfg: str,
    arch: str,
    pops: Sequence[str],
    rg_order: Sequence[str],
    bin_setting: int,
    include_total: bool,
    outpath: Path,
    mode: str,
    center: str,
    palette: dict[str, str],
    method_label_map: dict[str, str],
    user_ylim: tuple[float, float] | None,
    y_scale: str,
    overlap_only: bool,
) -> None:
    total_only_24bin_mse = mode == "mse" and int(bin_setting) == 24

    nrows = len(rg_order)
    ncols = len(pops)

    if total_only_24bin_mse:
        effective_ticks = 1
        panel_w = 2.8
        xticklabels = ["Total"]
    else:
        effective_ticks = int(bin_setting) + (1 if include_total else 0)
        if effective_ticks <= 2:
            panel_w = 2.8
        elif effective_ticks <= 9:
            panel_w = 3.9
        else:
            panel_w = 5.4
        xticklabels = make_bin_ticklabels(
            int(bin_setting), include_total=bool(include_total)
        )

    fig_w = panel_w * ncols + 1.8
    fig_h = 2.45 * nrows + 1.45

    if y_scale == "shared":
        sharey = True
    elif y_scale == "independent":
        sharey = "col"
    else:
        raise ValueError(f"Unknown y-scale: {y_scale}")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=sharey,
        constrained_layout=False,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    x_base = np.arange(1, effective_ticks + 1, dtype=float)

    ylim_map = compute_ylim_map(
        h2_cfg=h2_cfg,
        arch=arch,
        pops=pops,
        y_scale=y_scale,
        plot_mode=mode,
        plot_df=plot_df,
        counts_df=counts_df,
        truth_df=truth_df,
        user_ylim=user_ylim,
        center=center,
    )

    methods_order = ["summit", "covldsc"]
    offsets = {"summit": -0.18, "covldsc": 0.18}
    markers = {"summit": "o", "covldsc": "s"}
    width = 0.28

    any_truth_drawn = False
    truth_legend_label: str | None = None

    for r, rg_cfg in enumerate(rg_order):
        for c, pop in enumerate(pops):
            ax = axes[r, c]

            raw_sub = raw_df[
                _cfg_eq(raw_df["h2_cfg"], h2_cfg)
                & _cfg_eq(raw_df["arch"], arch)
                & _cfg_eq(raw_df["rg_cfg"], rg_cfg)
                & raw_df["pop"].astype(str).eq(str(pop))
            ].copy()

            if mode == "nvalid":
                assert counts_df is not None
                panel_sub = counts_df[
                    _cfg_eq(counts_df["h2_cfg"], h2_cfg)
                    & _cfg_eq(counts_df["arch"], arch)
                    & _cfg_eq(counts_df["rg_cfg"], rg_cfg)
                    & counts_df["pop"].astype(str).eq(str(pop))
                ].copy()
                truth_curve = None
            else:
                assert plot_df is not None and truth_df is not None
                panel_sub = plot_df[
                    _cfg_eq(plot_df["h2_cfg"], h2_cfg)
                    & _cfg_eq(plot_df["arch"], arch)
                    & _cfg_eq(plot_df["rg_cfg"], rg_cfg)
                    & plot_df["pop"].astype(str).eq(str(pop))
                ].copy()

                tr_sub = truth_df[
                    _cfg_eq(truth_df["h2_cfg"], h2_cfg)
                    & _cfg_eq(truth_df["arch"], arch)
                    & _cfg_eq(truth_df["rg_cfg"], rg_cfg)
                    & truth_df["pop"].astype(str).eq(str(pop))
                ].copy()

                if total_only_24bin_mse:
                    panel_sub = panel_sub[
                        panel_sub["tick_kind"].astype(str).eq("total")
                    ].copy()
                    tr_sub = tr_sub[tr_sub["tick_kind"].astype(str).eq("total")].copy()

                    if panel_sub.shape[0]:
                        panel_sub["tick_idx"] = 0
                        panel_sub["tick_order"] = 1
                        panel_sub["tick_name"] = "total"
                    if tr_sub.shape[0]:
                        tr_sub["tick_idx"] = 0
                        tr_sub["tick_order"] = 1
                        tr_sub["tick_name"] = "total"

                truth_curve = tr_sub.sort_values("tick_order")["truth_plot"].to_numpy(
                    float
                )
                if truth_curve.size == 0:
                    truth_curve = None

            if not panel_has_any_data(mode=mode, sub=panel_sub, truth_vals=truth_curve):
                if r == 0:
                    ax.set_title(pretty_pop_label(pop))
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            if mode == "nvalid":
                for method in methods_order:
                    draw_method_count_line(
                        ax=ax,
                        sub=panel_sub,
                        method=method,
                        x_base=x_base,
                        color=palette[method],
                        marker=markers[method],
                        offset=offsets[method],
                    )
            else:
                for method in methods_order:
                    draw_method_box_and_line(
                        ax=ax,
                        sub=panel_sub,
                        method=method,
                        x_base=x_base,
                        color=palette[method],
                        marker=markers[method],
                        offset=offsets[method],
                        width=width,
                        center=center,
                    )

                truth_style = truth_style_for_panel(
                    mode=mode,
                    bin_setting=int(bin_setting),
                    truth_curve=truth_curve,
                )
                if truth_style is not None and truth_curve is not None:
                    t = np.asarray(truth_curve, dtype=float)
                    if np.isfinite(t).any():
                        ax.plot(
                            x_base,
                            t,
                            color=truth_style[0],
                            linestyle="--",
                            linewidth=1.6,
                            marker="D",
                            markersize=3.5,
                            alpha=0.90,
                            zorder=4,
                        )
                        any_truth_drawn = True
                        truth_legend_label = truth_style[1]

            ax.axhline(
                0.0, color="0.45", linestyle="--", linewidth=0.85, alpha=0.8, zorder=1
            )

            if include_total and not total_only_24bin_mse:
                ax.axvline(
                    float(bin_setting) + 0.5,
                    color="0.55",
                    linestyle=":",
                    linewidth=0.9,
                    zorder=1,
                )

            ax.set_xlim(0.45, effective_ticks + 0.55)
            ylo, yhi = ylim_map[str(pop)]
            ax.set_ylim(ylo, yhi)

            ax.set_xticks(x_base)
            ax.set_xticklabels(xticklabels)
            if total_only_24bin_mse:
                ax.tick_params(axis="x", labelrotation=0)
            elif int(bin_setting) >= 24:
                ax.tick_params(axis="x", labelrotation=90)
            else:
                ax.tick_params(axis="x", labelrotation=0)

            if int(bin_setting) == 8 and not total_only_24bin_mse:
                maybe_annotate_8bin_groups(ax)

            if mode == "nvalid":
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if r == 0:
                ax.set_title(pretty_pop_label(pop))

            if c == 0:
                ax.set_ylabel(
                    f"{pretty_rg_label(rg_cfg)}\n{mode_ylabel(mode, estimand)}"
                )
            else:
                ax.set_ylabel("")
                if y_scale == "shared":
                    ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.tick_params(axis="y", labelleft=True)

            if r == nrows - 1:
                ax.set_xlabel("Target" if total_only_24bin_mse else "Bin")
                ax.tick_params(axis="x", labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    if total_only_24bin_mse:
        fig_title = (
            f"{pretty_h2_label(h2_cfg)} · {pretty_arch_label(arch)}: total {estimand_title_label(estimand)} "
            f"{mode_title(mode)} from 24-bin fits"
        )
    else:
        fig_title = (
            f"{pretty_h2_label(h2_cfg)} · {pretty_arch_label(arch)}: per-bin {estimand_title_label(estimand)} "
            f"{mode_title(mode)} ({int(bin_setting)} fitted bins)"
        )

    # fig.suptitle(
    #     fig_title,
    #     y=0.995,
    #     fontsize=13,
    # )
    # fig.text(
    #     0.5,
    #     0.965,
    #     subtitle_for_mode(
    #         mode=mode,
    #         center=center,
    #         estimand=estimand,
    #         bin_setting=int(bin_setting),
    #         include_total=bool(include_total),
    #         overlap_only=bool(overlap_only),
    #     ),
    #     ha="center",
    #     va="top",
    #     fontsize=9,
    # )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=palette["summit"],
            marker="o",
            linewidth=1.7,
            markersize=5,
            label=method_label_map["summit"],
        ),
        Line2D(
            [0],
            [0],
            color=palette["covldsc"],
            marker="s",
            linewidth=1.7,
            markersize=5,
            label=method_label_map["covldsc"],
        ),
    ]
    if any_truth_drawn and truth_legend_label is not None:
        truth_style = truth_style_for_panel(
            mode=mode,
            bin_setting=int(bin_setting),
            truth_curve=np.array([0.0]),
        )
        assert truth_style is not None
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=truth_style[0],
                linestyle="--",
                marker="D",
                linewidth=1.6,
                markersize=4.5,
                label=truth_legend_label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        bbox_transform=fig.transFigure,
        frameon=True,
        title="Curve",
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.91, 0.95))
    save_figure(fig, outpath)
    plt.close(fig)
    print(f"[write] {outpath}")


def plot_total_across_bins_grid(
    raw_df: pd.DataFrame,
    plot_df: pd.DataFrame | None,
    counts_df: pd.DataFrame | None,
    truth_df: pd.DataFrame | None,
    estimand: str,
    h2_cfg: str,
    arch: str,
    pops: Sequence[str],
    rg_order: Sequence[str],
    outpath: Path,
    mode: str,
    center: str,
    palette: dict[str, str],
    method_label_map: dict[str, str],
    user_ylim: tuple[float, float] | None,
    y_scale: str,
    overlap_only: bool,
    bin_settings: Sequence[int] = ALL_BIN_SETTINGS,
) -> None:
    nrows = len(rg_order)
    ncols = len(pops)
    effective_ticks = len(list(bin_settings))
    xticklabels = make_all_bin_ticklabels(bin_settings)

    panel_w = 3.0
    fig_w = panel_w * ncols + 1.8
    fig_h = 2.45 * nrows + 1.45

    if y_scale == "shared":
        sharey = True
    elif y_scale == "independent":
        sharey = "col"
    else:
        raise ValueError(f"Unknown y-scale: {y_scale}")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=sharey,
        constrained_layout=False,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    x_base = np.arange(1, effective_ticks + 1, dtype=float)

    ylim_map = compute_ylim_map(
        h2_cfg=h2_cfg,
        arch=arch,
        pops=pops,
        y_scale=y_scale,
        plot_mode=mode,
        plot_df=plot_df,
        counts_df=counts_df,
        truth_df=truth_df,
        user_ylim=user_ylim,
        center=center,
    )

    methods_order = ["summit", "covldsc"]
    offsets = {"summit": -0.18, "covldsc": 0.18}
    markers = {"summit": "o", "covldsc": "s"}
    width = 0.28

    any_truth_drawn = False
    truth_legend_label: str | None = None

    for r, rg_cfg in enumerate(rg_order):
        for c, pop in enumerate(pops):
            ax = axes[r, c]

            if mode == "nvalid":
                assert counts_df is not None
                panel_sub = counts_df[
                    _cfg_eq(counts_df["h2_cfg"], h2_cfg)
                    & _cfg_eq(counts_df["arch"], arch)
                    & _cfg_eq(counts_df["rg_cfg"], rg_cfg)
                    & counts_df["pop"].astype(str).eq(str(pop))
                ].copy()
                truth_curve = None
            else:
                assert plot_df is not None and truth_df is not None
                panel_sub = plot_df[
                    _cfg_eq(plot_df["h2_cfg"], h2_cfg)
                    & _cfg_eq(plot_df["arch"], arch)
                    & _cfg_eq(plot_df["rg_cfg"], rg_cfg)
                    & plot_df["pop"].astype(str).eq(str(pop))
                ].copy()
                tr_sub = truth_df[
                    _cfg_eq(truth_df["h2_cfg"], h2_cfg)
                    & _cfg_eq(truth_df["arch"], arch)
                    & _cfg_eq(truth_df["rg_cfg"], rg_cfg)
                    & truth_df["pop"].astype(str).eq(str(pop))
                ].copy()
                truth_curve = tr_sub.sort_values("tick_order")["truth_plot"].to_numpy(
                    float
                )
                if truth_curve.size == 0:
                    truth_curve = None

            if not panel_has_any_data(mode=mode, sub=panel_sub, truth_vals=truth_curve):
                if r == 0:
                    ax.set_title(pretty_pop_label(pop))
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            if mode == "nvalid":
                for method in methods_order:
                    draw_method_count_line(
                        ax=ax,
                        sub=panel_sub,
                        method=method,
                        x_base=x_base,
                        color=palette[method],
                        marker=markers[method],
                        offset=offsets[method],
                    )
            else:
                for method in methods_order:
                    draw_method_box_and_line(
                        ax=ax,
                        sub=panel_sub,
                        method=method,
                        x_base=x_base,
                        color=palette[method],
                        marker=markers[method],
                        offset=offsets[method],
                        width=width,
                        center=center,
                    )

                truth_style = flat_truth_style_for_total_across_bins(mode)
                if truth_style is not None and truth_curve is not None:
                    truth_vals = np.asarray(truth_curve, dtype=float)
                    truth_vals = truth_vals[np.isfinite(truth_vals)]
                    if truth_vals.size:
                        truth_level = float(truth_vals[0])
                        ax.axhline(
                            truth_level,
                            color=truth_style[0],
                            linestyle="--",
                            linewidth=1.6,
                            alpha=0.90,
                            zorder=2,
                        )
                        any_truth_drawn = True
                        truth_legend_label = truth_style[1]

            if mode == "estimate":
                ax.axhline(
                    0.0,
                    color="0.45",
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.8,
                    zorder=1,
                )

            ax.set_xlim(0.45, effective_ticks + 0.55)
            ylo, yhi = ylim_map[str(pop)]
            ax.set_ylim(ylo, yhi)

            ax.set_xticks(x_base)
            ax.set_xticklabels(xticklabels)
            ax.tick_params(axis="x", labelrotation=0)

            if mode == "nvalid":
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if r == 0:
                ax.set_title(pretty_pop_label(pop))

            if c == 0:
                ax.set_ylabel(
                    f"{pretty_rg_label(rg_cfg)}\n{mode_ylabel(mode, estimand)}"
                )
            else:
                ax.set_ylabel("")
                if y_scale == "shared":
                    ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.tick_params(axis="y", labelleft=True)

            if r == nrows - 1:
                ax.set_xlabel("Fitted bin setting")
                ax.tick_params(axis="x", labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    # fig_title = (
    #     f"{pretty_h2_label(h2_cfg)} · {pretty_arch_label(arch)}: total {estimand_title_label(estimand)} "
    #     f"{mode_title(mode)} across fitted bin settings"
    # )
    # fig.suptitle(fig_title, y=0.995, fontsize=13)
    # fig.text(
    #     0.5,
    #     0.965,
    #     subtitle_for_total_across_bins_mode(
    #         mode=mode,
    #         center=center,
    #         estimand=estimand,
    #         overlap_only=bool(overlap_only),
    #     ),
    #     ha="center",
    #     va="top",
    #     fontsize=9,
    # )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=palette["summit"],
            marker="o",
            linewidth=1.7,
            markersize=5,
            label=method_label_map["summit"],
        ),
        Line2D(
            [0],
            [0],
            color=palette["covldsc"],
            marker="s",
            linewidth=1.7,
            markersize=5,
            label=method_label_map["covldsc"],
        ),
    ]
    if any_truth_drawn and truth_legend_label is not None:
        truth_style = flat_truth_style_for_total_across_bins(mode)
        assert truth_style is not None
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=truth_style[0],
                linestyle="--",
                linewidth=1.6,
                label=truth_legend_label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        bbox_transform=fig.transFigure,
        frameon=True,
        title="Curve",
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.91, 0.95))
    save_figure(fig, outpath)
    plt.close(fig)
    print(f"[write] {outpath}")


def print_verbose_total_rg_diagnostics(df: pd.DataFrame, truth_map: dict) -> None:
    need = {
        "pop",
        "arch",
        "h2_cfg",
        "rg_cfg",
        "method_plot",
        "h2_1",
        "h2_2",
        "gamma_g",
        "rg",
    }
    missing = need.difference(df.columns)
    if missing:
        print(
            f"[verbose] skipped total-rg diagnostics; missing columns: {sorted(missing)}"
        )
        return

    d = df.copy()
    d["rg_from_components"] = np.where(
        np.isfinite(d["gamma_g"])
        & np.isfinite(d["h2_1"])
        & np.isfinite(d["h2_2"])
        & (d["h2_1"] > 0)
        & (d["h2_2"] > 0),
        d["gamma_g"] / np.sqrt(d["h2_1"] * d["h2_2"]),
        np.nan,
    )

    rows = []
    for (pop, arch, h2_cfg, rg_cfg, method), sub in d.groupby(
        ["pop", "arch", "h2_cfg", "rg_cfg", "method_plot"], dropna=False
    ):
        rec = get_truth_record(
            truth_map,
            pop=str(pop),
            h2_cfg=str(h2_cfg),
            rg_cfg=str(rg_cfg),
            arch=str(arch),
        )
        truth_total = np.nan
        if rec is not None and "total_rg_from_bins" in rec:
            truth_total = float(rec["total_rg_from_bins"])

        rg = pd.to_numeric(sub["rg"], errors="coerce").to_numpy(float)
        rgc = pd.to_numeric(sub["rg_from_components"], errors="coerce").to_numpy(float)
        h1 = pd.to_numeric(sub["h2_1"], errors="coerce").to_numpy(float)
        h2 = pd.to_numeric(sub["h2_2"], errors="coerce").to_numpy(float)
        g = pd.to_numeric(sub["gamma_g"], errors="coerce").to_numpy(float)

        rows.append(
            {
                "pop": pop,
                "arch": arch,
                "h2_cfg": h2_cfg,
                "rg_cfg": rg_cfg,
                "method": method,
                "n": int(np.sum(np.isfinite(rg))),
                "mean_rg_reported": float(np.nanmean(rg)),
                "median_rg_reported": float(np.nanmedian(rg)),
                "mean_rg_from_components": float(np.nanmean(rgc)),
                "median_rg_from_components": float(np.nanmedian(rgc)),
                "mean_h2_1": float(np.nanmean(h1)),
                "mean_h2_2": float(np.nanmean(h2)),
                "mean_gamma": float(np.nanmean(g)),
                "true_total_rg": truth_total,
                "bias": float(np.nanmean(rg) - truth_total)
                if np.isfinite(truth_total)
                else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["pop", "arch", "h2_cfg", "rg_cfg", "method"])
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print("\n[verbose] total-rg diagnostics")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", type=str, default="data/sim_rg/partitioned")
    ap.add_argument("--infile", type=str, default="estimates.csv")
    ap.add_argument("--pops", type=str, default="EUR_300k,EUR,SAS,AFR")
    ap.add_argument("--rg-order", type=str, default="Constrg,Gradrg,Nullrg")

    ap.add_argument(
        "--bin", dest="bin_setting", type=int, default=8, choices=[1, 8, 24]
    )
    ap.add_argument(
        "--estimand",
        type=str,
        default="rg",
        choices=["h2_1", "h2_2", "denom", "gamma", "rg"],
        help="Which estimand to compare across methods.",
    )

    ap.add_argument("--true-params", type=str, default="all_params.mafld.8.csv")

    ap.add_argument("--outdir", type=str, default="./sims_part_mafld_maintext")
    ap.add_argument("--outprefix", type=str, default="gencor_sims_part_mafld")

    ap.add_argument(
        "--center",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Summary statistic used for the method line across replicates.",
    )
    ap.add_argument(
        "--ylim",
        type=str,
        default=None,
        help="Optional fixed y-limits as min,max (example: --ylim -0.2,0.6).",
    )
    ap.add_argument(
        "--y-scale",
        type=str,
        default="independent",
        choices=["shared", "independent"],
        help="shared = one y-range per H2 figure; independent = one y-range per population column, shared across its rg rows.",
    )
    ap.add_argument(
        "--hide-full", action="store_true", help="Hide the last Total x-tick."
    )
    ap.add_argument(
        "--overlap-only",
        action="store_true",
        help="Keep only replicate/tick values where both methods are finite.",
    )
    ap.add_argument(
        "--plot-all-bins",
        action="store_true",
        help="Plot only the total estimand, comparing the 1-bin, 8-bin, and 24-bin fits in the same subplots.",
    )

    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--plot-error", action="store_true", help="Plot estimate minus truth."
    )
    mode_group.add_argument(
        "--plot-mse", action="store_true", help="Plot squared error relative to truth."
    )
    mode_group.add_argument(
        "--plot-n-valid",
        action="store_true",
        help="Plot only the number of finite estimates per method and x-tick.",
    )

    ap.add_argument("--true-rg-constr", type=float, default=0.3)
    ap.add_argument("--true-rg-grad", type=float, default=0.3)
    ap.add_argument("--true-rg-null", type=float, default=0.0)

    args = ap.parse_args()

    set_maintext_style()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pops = parse_csv_arg(args.pops)
    rg_order = parse_csv_arg(args.rg_order)
    if not pops:
        raise ValueError("Empty --pops.")
    if not rg_order:
        raise ValueError("Empty --rg-order.")

    plot_mode = "estimate"
    if bool(args.plot_error):
        plot_mode = "error"
    elif bool(args.plot_mse):
        plot_mode = "mse"
    elif bool(args.plot_n_valid):
        plot_mode = "nvalid"

    plot_all_bins = bool(args.plot_all_bins)
    bin_setting = int(args.bin_setting)

    if not plot_all_bins:
        if plot_mode == "error" and bin_setting not in {1, 8}:
            raise ValueError("--plot-error is only supported for --bin 1 or --bin 8.")
        if plot_mode == "mse" and bin_setting not in {1, 8, 24}:
            raise ValueError(
                "--plot-mse is only supported for --bin 1, --bin 8, or total-only for --bin 24."
            )
    else:
        if args.hide_full:
            print(
                "[note] --plot-all-bins shows only totals across {1, 8, 24}; ignoring --hide-full."
            )
        if bin_setting != 8:
            print(
                "[note] --plot-all-bins ignores --bin and compares total estimates from the 1-bin, 8-bin, and 24-bin fits."
            )

    hide_full_effective = bool(args.hide_full) and not plot_all_bins
    if plot_mode == "mse" and bin_setting == 24 and hide_full_effective:
        print(
            "[note] --plot-mse with --bin 24 plots only the total estimate; ignoring --hide-full."
        )
        hide_full_effective = False

    TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
    palette = {
        "summit": TAB20[4],
        "covldsc": TAB20[2],
    }
    method_label_map = {
        "summit": "SUMMIT",
        "covldsc": "cov-LDSC",
    }

    df0 = load_concat(base=base, pops=pops, infile=args.infile)
    true_map = load_true_params(Path(args.true_params)) if args.true_params else {}
    user_ylim = parse_ylim_arg(args.ylim)

    if plot_all_bins:
        df = prepare_df_all_bins(
            df=df0,
            methods_keep=["summit", "covldsc"],
            # methods_keep=["summit"],
            estimand=str(args.estimand),
            bin_settings=ALL_BIN_SETTINGS,
        )
        if df.shape[0] == 0:
            raise RuntimeError(
                f"No rows after filtering to summit/covldsc and nbins in {ALL_BIN_SETTINGS} for estimand={args.estimand}."
            )

        raw_df = build_raw_total_across_bins_df(
            df=df,
            estimand=str(args.estimand),
            bin_settings=ALL_BIN_SETTINGS,
        )
        raw_df = add_overlap_flag(raw_df, methods_required=["summit", "covldsc"])
        if bool(args.overlap_only):
            raw_df = raw_df[raw_df["overlap_ok"]].copy()

        panels = raw_df[["pop", "arch", "h2_cfg", "rg_cfg"]].drop_duplicates().copy()
        truth_df = build_truth_total_across_bins_df(
            panels=panels,
            estimand=str(args.estimand),
            plot_mode=plot_mode,
            true_map=true_map,
            true_rg_constr=float(args.true_rg_constr),
            true_rg_grad=float(args.true_rg_grad),
            true_rg_null=float(args.true_rg_null),
            bin_settings=ALL_BIN_SETTINGS,
        )

        plot_df: pd.DataFrame | None = None
        counts_df: pd.DataFrame | None = None
        if plot_mode == "nvalid":
            counts_df = compute_n_valid_df(raw_df)
        else:
            plot_df = raw_df.merge(
                truth_df,
                on=[
                    "pop",
                    "arch",
                    "h2_cfg",
                    "rg_cfg",
                    "tick_kind",
                    "tick_idx",
                    "tick_order",
                    "tick_name",
                ],
                how="left",
            )
            validate_truth_requirements(
                plot_df=plot_df,
                plot_mode=plot_mode,
                bin_setting=bin_setting,
                total_only=True,
            )
            plot_df = apply_plot_transform(plot_df=plot_df, plot_mode=plot_mode)

        outstem = make_outstem(
            outprefix=str(args.outprefix),
            estimand=str(args.estimand),
            mode=plot_mode,
            hide_full=False,
            y_scale=str(args.y_scale),
            overlap_only=bool(args.overlap_only),
            plot_all_bins=True,
        )

        arch_order = arch_order_from_df(raw_df)
        for h2_cfg in ["flatH2", "gradH2"]:
            for arch in arch_order:
                combo_mask = _cfg_eq(raw_df["h2_cfg"], h2_cfg) & _cfg_eq(
                    raw_df["arch"], arch
                )
                if not combo_mask.any():
                    continue
                outpath = outdir / f"{outstem}_{h2_cfg}_{arch}.pdf"
                plot_total_across_bins_grid(
                    raw_df=raw_df,
                    plot_df=plot_df,
                    counts_df=counts_df,
                    truth_df=truth_df,
                    estimand=str(args.estimand),
                    h2_cfg=h2_cfg,
                    arch=arch,
                    pops=pops,
                    rg_order=rg_order,
                    outpath=outpath,
                    mode=plot_mode,
                    center=str(args.center),
                    palette=palette,
                    method_label_map=method_label_map,
                    user_ylim=user_ylim,
                    y_scale=str(args.y_scale),
                    overlap_only=bool(args.overlap_only),
                    bin_settings=ALL_BIN_SETTINGS,
                )
    else:
        df = prepare_df(
            df=df0,
            bin_setting=bin_setting,
            methods_keep=["summit", "covldsc"],
            estimand=str(args.estimand),
        )
        if df.shape[0] == 0:
            raise RuntimeError(
                f"No rows after filtering to summit/covldsc and nbins=={bin_setting} for estimand={args.estimand}."
            )

        include_total = not hide_full_effective
        raw_df = build_raw_tick_df(
            df=df,
            bin_setting=bin_setting,
            include_total=include_total,
            estimand=str(args.estimand),
        )
        raw_df = add_overlap_flag(raw_df, methods_required=["summit", "covldsc"])
        if bool(args.overlap_only):
            raw_df = raw_df[raw_df["overlap_ok"]].copy()

        panels = raw_df[["pop", "arch", "h2_cfg", "rg_cfg"]].drop_duplicates().copy()
        truth_df = build_truth_df(
            panels=panels,
            estimand=str(args.estimand),
            bin_setting=bin_setting,
            include_total=include_total,
            plot_mode=plot_mode,
            true_map=true_map,
            true_rg_constr=float(args.true_rg_constr),
            true_rg_grad=float(args.true_rg_grad),
            true_rg_null=float(args.true_rg_null),
        )

        plot_df = None
        counts_df = None

        if plot_mode == "nvalid":
            counts_df = compute_n_valid_df(raw_df)
        else:
            plot_df = raw_df.merge(
                truth_df,
                on=[
                    "pop",
                    "arch",
                    "h2_cfg",
                    "rg_cfg",
                    "tick_kind",
                    "tick_idx",
                    "tick_order",
                    "tick_name",
                ],
                how="left",
            )
            validate_truth_requirements(
                plot_df=plot_df, plot_mode=plot_mode, bin_setting=bin_setting
            )
            plot_df = apply_plot_transform(plot_df=plot_df, plot_mode=plot_mode)

        outstem = make_outstem(
            outprefix=str(args.outprefix),
            estimand=str(args.estimand),
            mode=plot_mode,
            hide_full=hide_full_effective,
            y_scale=str(args.y_scale),
            overlap_only=bool(args.overlap_only),
            plot_all_bins=False,
        )

        arch_order = arch_order_from_df(raw_df)
        for h2_cfg in ["flatH2", "gradH2"]:
            for arch in arch_order:
                combo_mask = _cfg_eq(raw_df["h2_cfg"], h2_cfg) & _cfg_eq(
                    raw_df["arch"], arch
                )
                if not combo_mask.any():
                    continue
                outpath = outdir / f"{outstem}_{bin_setting}bins_{h2_cfg}_{arch}.pdf"
                plot_partitioned_grid(
                    raw_df=raw_df,
                    plot_df=plot_df,
                    counts_df=counts_df,
                    truth_df=truth_df,
                    estimand=str(args.estimand),
                    h2_cfg=h2_cfg,
                    arch=arch,
                    pops=pops,
                    rg_order=rg_order,
                    bin_setting=bin_setting,
                    include_total=include_total,
                    outpath=outpath,
                    mode=plot_mode,
                    center=str(args.center),
                    palette=palette,
                    method_label_map=method_label_map,
                    user_ylim=user_ylim,
                    y_scale=str(args.y_scale),
                    overlap_only=bool(args.overlap_only),
                )

    print("[done]")


if __name__ == "__main__":
    main()
