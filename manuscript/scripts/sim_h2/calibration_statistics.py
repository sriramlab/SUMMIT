"""Plot calibration statistics for the SUMMIT manuscript."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


METHODS_ALL = {
    "rhe",
    "sumrhe",
    "covsumrhe",
    "ldsc",
    "covldsc",
    "covldsc_const",
    "sumher",
    "sumher_ldak",
}


WINDOWED_METHODS = {"ldsc", "sumher", "sumher_ldak"}


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


def zcrit_one_sided(alpha: float) -> float:
    # one-sided, right tail: P(Z > z_crit) = alpha
    return _norm_ppf(1.0 - float(alpha))


def zcrit_two_sided(alpha: float) -> float:
    # two-sided: P(|Z| > z_crit) = alpha
    return _norm_ppf(1.0 - float(alpha) / 2.0)


def z_for_ci(ci_level: float) -> float:
    # symmetric normal critical for CI level (e.g., 0.95 -> 1.96)
    alpha = 1.0 - float(ci_level)
    return zcrit_two_sided(alpha)


def restrict_methods_and_windows(
    df: pd.DataFrame, windows_keep: set[int]
) -> pd.DataFrame:
    """
    Keep only methods we care about and, for windowed methods, only windows_keep.
    Non-windowed methods (rhe/sumrhe/covsumrhe/covldsc) always kept.
    """
    df = df[df["method"].isin(METHODS_ALL)].copy()
    if "window" not in df.columns:
        df["window"] = -1
    df["window"] = pd.to_numeric(df["window"], errors="coerce").fillna(-1).astype(int)
    has_w = df["method"].isin(WINDOWED_METHODS)
    return pd.concat(
        [df[~has_w], df[has_w & df["window"].isin(windows_keep)]], ignore_index=True
    )


def attach_method_labels(df: pd.DataFrame, windows_keep: set[int]) -> pd.DataFrame:
    """Map (method, window) -> labels like 'ldsc_2000', 'sumher_50000', etc."""
    df = df.copy()

    def _ml(row):
        m = str(row["method"])
        w = int(row.get("window", -1))
        if m in WINDOWED_METHODS:
            return f"{m}_{w}" if w in windows_keep else None
        return m

    df["method_label"] = df.apply(_ml, axis=1)
    return df[~df["method_label"].isna()].copy()


def to_long(
    df: pd.DataFrame, prefix: str, out_col: str, se_col: str, allow_missing_se=False
) -> pd.DataFrame:
    """
    Wide -> long for 2-bin contig_h2 *bin-level* columns.

    For prefix='enr' expects:
      enr_0, enr_1, enr_se_0, enr_se_1
    """
    if df.empty:
        return df.copy()

    B = int(df["num_bins"].mode().iat[0])

    base_keep = [
        "pop",
        "true_h2",
        "method",
        "method_label",
        "run",
        "num_bins",
        "p_causal",
        "window",
        "arch",
        "scenario",
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


_CLUSTER_CANDIDATES = ["run", "rep", "replicate", "seed", "sim_id", "trial", "iter"]


def _as_key_str(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        out = x.round(12).astype("string")
        return out.fillna("__NA__")
    return series.astype("string").fillna("__NA__")


def _make_cluster_id(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        # fallback: per-row unique cluster
        return pd.Series(np.arange(df.shape[0]), index=df.index, dtype="int64").astype(
            "string"
        )
    tmp = df.copy()
    for c in cols:
        tmp[c] = _as_key_str(tmp[c]) if c in tmp.columns else "__NA__"
    return tmp[cols].astype("string").agg("|".join, axis=1)


def _infer_cluster_cols(df: pd.DataFrame, user_cols: list[str] | None) -> list[str]:
    """
    Prefer a cluster id that corresponds to an *independent simulation replicate*.
    run alone is often NOT unique across settings, so we include setting columns when present.
    """
    if user_cols:
        cols = [c for c in user_cols if c in df.columns]
        return cols

    cols = []
    for c in ["arch", "scenario", "true_h2", "p_causal"]:
        if c in df.columns:
            cols.append(c)

    # pick first available replicate-like id
    rep = None
    for c in _CLUSTER_CANDIDATES:
        if c in df.columns:
            rep = c
            break
    if rep is not None:
        cols.append(rep)

    # for enrichment long-form: include bin if present (doesn't hurt, usually constant for null bin)
    if "bin" in df.columns:
        cols.append("bin")

    return cols


def _extract_null_units(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Return replicate-level (or row-level) null units with z-values:
      columns: pop, method_label, z, plus cluster columns if present.
    """
    if long_df.empty:
        return pd.DataFrame()

    if metric == "enr":
        if (
            "enrichment" not in long_df.columns
            or "enrichment_se" not in long_df.columns
        ):
            return pd.DataFrame()
        df = long_df[long_df["is_causal"] == 0].copy()
        # z = (enr - 1) / se
        df["z"] = (
            pd.to_numeric(df["enrichment"], errors="coerce") - 1.0
        ) / pd.to_numeric(df["enrichment_se"], errors="coerce")
    elif metric == "h2":
        if (
            "h2" not in long_df.columns
            or "h2_se" not in long_df.columns
            or "true_h2" not in long_df.columns
        ):
            return pd.DataFrame()
        df = long_df.copy()
        df["z"] = (
            pd.to_numeric(df["h2"], errors="coerce")
            - pd.to_numeric(df["true_h2"], errors="coerce")
        ) / pd.to_numeric(df["h2_se"], errors="coerce")
    else:
        return pd.DataFrame()

    # validity
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df = df[np.isfinite(df["z"].to_numpy(dtype=float))].copy()

    need = ["pop", "method_label", "z"]
    extra = [
        c
        for c in [
            "arch",
            "scenario",
            "true_h2",
            "p_causal",
            "run",
            "rep",
            "seed",
            "sim_id",
            "trial",
            "iter",
            "bin",
        ]
        if c in df.columns
    ]
    keep = list(dict.fromkeys(need + extra))
    return df[keep].copy()


def build_calibration_overall_bootstrap(
    long_df: pd.DataFrame,
    alpha_vals: list[float],
    metric: str,
    both_sides: bool,
    B: int,
    seed: int,
    ci_level: float,
    cluster_cols: list[str] | None,
    complete_case: bool = False,
    methods_ord_for_cc: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bootstrap calibration curves per (pop, method_label, alpha).
    - cluster bootstrap if cluster_cols not empty; else row bootstrap.
    - complete_case (optional): keep only clusters where all methods in methods_ord_for_cc have >=1 valid unit.
    """
    units = _extract_null_units(long_df, metric=metric)
    if units.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(int(seed))
    zci = z_for_ci(ci_level)

    # cluster id
    cols = _infer_cluster_cols(units, cluster_cols)
    units = units.copy()
    units["_cluster"] = _make_cluster_id(units, cols)

    # optional complete-case across methods
    if complete_case and methods_ord_for_cc:
        keep_methods = [
            m for m in methods_ord_for_cc if m in units["method_label"].unique()
        ]
        if keep_methods:
            # for each pop, cluster: require all keep_methods appear
            grp = (
                units[units["method_label"].isin(keep_methods)]
                .groupby(["pop", "_cluster"])["method_label"]
                .nunique()
                .reset_index(name="k")
            )
            need = len(keep_methods)
            good = grp[grp["k"] == need][["pop", "_cluster"]]
            if not good.empty:
                units = units.merge(
                    good.assign(_keep=1), on=["pop", "_cluster"], how="inner"
                ).drop(columns=["_keep"])
            else:
                return pd.DataFrame()

    out_rows = []
    alphas = [float(a) for a in alpha_vals]

    for (pop, meth), sub in units.groupby(["pop", "method_label"], dropna=False):
        z = sub["z"].to_numpy(dtype=float)
        if z.size == 0:
            continue

        # point estimate curve
        if both_sides:
            f_hat = np.array(
                [(np.abs(z) >= zcrit_two_sided(a)).mean() for a in alphas], dtype=float
            )
            n_sig_hat = np.array(
                [(np.abs(z) >= zcrit_two_sided(a)).sum() for a in alphas], dtype=int
            )
        else:
            f_hat = np.array(
                [(z >= zcrit_one_sided(a)).mean() for a in alphas], dtype=float
            )
            n_sig_hat = np.array(
                [(z >= zcrit_one_sided(a)).sum() for a in alphas], dtype=int
            )
        n_total = int(z.size)

        # bootstrap distribution
        clusters = pd.unique(sub["_cluster"])
        use_cluster = clusters.size >= 2
        if use_cluster:
            # cluster -> indices
            idx_by = {}
            for i, cid in enumerate(clusters):
                idx_by[cid] = sub.index[sub["_cluster"] == cid].to_numpy()
            C = int(clusters.size)
            boot = np.empty((B, len(alphas)), dtype=float)
            for b in range(B):
                picks = rng.integers(0, C, size=C, endpoint=False)
                idx = np.concatenate([idx_by[clusters[i]] for i in picks], axis=0)
                zb = units.loc[idx, "z"].to_numpy(dtype=float)
                if both_sides:
                    boot[b, :] = np.array(
                        [(np.abs(zb) >= zcrit_two_sided(a)).mean() for a in alphas],
                        dtype=float,
                    )
                else:
                    boot[b, :] = np.array(
                        [(zb >= zcrit_one_sided(a)).mean() for a in alphas], dtype=float
                    )
            n_clusters = C
        else:
            # row bootstrap
            n = int(z.size)
            boot = np.empty((B, len(alphas)), dtype=float)
            for b in range(B):
                idx = rng.integers(0, n, size=n, endpoint=False)
                zb = z[idx]
                if both_sides:
                    boot[b, :] = np.array(
                        [(np.abs(zb) >= zcrit_two_sided(a)).mean() for a in alphas],
                        dtype=float,
                    )
                else:
                    boot[b, :] = np.array(
                        [(zb >= zcrit_one_sided(a)).mean() for a in alphas], dtype=float
                    )
            n_clusters = np.nan

        # summarize
        boot_se = boot.std(axis=0, ddof=1)
        lo = np.quantile(boot, (1.0 - ci_level) / 2.0, axis=0)
        hi = np.quantile(boot, 1.0 - (1.0 - ci_level) / 2.0, axis=0)

        for k, a in enumerate(alphas):
            out_rows.append(
                {
                    "pop": pop,
                    "method_label": meth,
                    "alpha": float(a),
                    "n_sig": int(n_sig_hat[k]),
                    "n_noncausal": int(n_total),
                    "empirical": float(f_hat[k]),
                    "emp_se": float(boot_se[k]),
                    "emp_lo": float(lo[k]),
                    "emp_hi": float(hi[k]),
                    "n_clusters": n_clusters,
                }
            )

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)
