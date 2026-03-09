import numpy as np
import pandas as pd
import os
import re
import time
import datetime
import glob
from pathlib import Path

# ----------------------- Time helpers ----------------------- #
def _get_time():
    current_time = time.time()
    return current_time


def _get_timestr(current_time):
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    timestr = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)))+" "+str(timezone)
    return timestr

# ----------------------- I/O & parsing helpers ----------------------- #

def _parse_column_name(df_hdr, names, default_pos):
    cols = list(df_hdr.columns)
    cols_lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    if default_pos >= len(cols):
        raise ValueError(f"Could not infer column {names}; header too short.")
    return cols[default_pos]


def _parse_sumdir(h2_path):
    if h2_path is None:
        raise ValueError("h2_path must be provided.")
    p = Path(h2_path)
    if p.is_dir():
        out = sorted(
            [str(x) for x in p.iterdir() if x.is_file() and not x.name.startswith(".")]
        )
        if not out:
            raise ValueError(f"No files found in h2_path directory: {h2_path}")
        return out
    if p.is_file():
        return [str(p)]
    raise ValueError(f"Could not resolve h2_path: {h2_path}")


def _parse_rg_pair(rg):
    if rg is None:
        raise ValueError("--rg must be provided.")
    parts = [x.strip() for x in str(rg).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--rg must be exactly two comma-separated sumstats paths.")
    for p in parts:
        if not Path(p).is_file():
            raise ValueError(f"Could not find sumstats file: {p}")
    return parts


def _phen_name_from_path(path: str) -> str:
    name = os.path.basename(path)
    for suf in (".sumstats.gz", ".sumstats", ".txt.gz", ".txt", ".tsv.gz", ".tsv", ".gz"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


def _parse_verbose(verbose) -> int:
    if isinstance(verbose, str):
        s = verbose.strip().lower()
        if s in ("0", "false", "none", "off"):
            return 0
        if s in ("1", "true", "yes", "on"):
            return 1
        if s == "max":
            return 2
        return 1
    return 1 if bool(verbose) else 0


def _resolve_chisq_threshold(nmax: float, raw=None):
    if raw is None:
        return None, "none"
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "auto":
            return float(max(80.0, 0.001 * float(nmax))), "auto"
        if s in ("none", "null"):
            return None, "none"
        try:
            return float(s), "manual"
        except Exception as e:
            raise ValueError(f"Invalid chisq_threshold string value: {raw!r}") from e
    try:
        return float(raw), "manual"
    except Exception as e:
        raise ValueError(f"Invalid chisq_threshold value: {raw!r}") from e


# -----------------------
# Batched vectorized trace estimators
# -----------------------

def _calc_trace_from_ld_batch(ldsum, n, m1, m2, delta=None):
    ldsum = np.asarray(ldsum, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)

    denom = m1 * m2

    # (..., K, K) with broadcasted compute
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ldsum * (n**2) / denom + n

    # where denom <= 0 (or NaN), fall back to n
    valid = (denom > 0) & np.isfinite(denom)
    out = np.where(valid, out, n)

    # ---- optional δ-based correction: out_{kℓ} ← out_{kℓ} - N * δ_{kℓ} ----
    if delta is not None:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
            raise ValueError(f"delta must be KxK; got shape {delta.shape}")

        # Get scalar N (first element of n broadcast)
        n_scalar = float(n.ravel()[0])

        # Correction term: N * δ_{kℓ}, broadcast across jackknife rows, but
        # only for valid (denom > 0) entries.
        corr = n_scalar * delta  # (K, K)

        # Broadcast corr and apply only where denom is valid
        out = np.where(valid, out - corr, out)

    return out


def _calc_rg_trace_from_ld_batch(ldsum, n1, n2, m1, m2):
    ldsum = np.asarray(ldsum, dtype=np.float64)
    n1 = float(n1)
    n2 = float(n2)
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)

    denom = m1 * m2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ldsum * (n1 * n2) / denom

    valid = (denom > 0) & np.isfinite(denom)
    out = np.where(valid, out, 0.0)
    return out


def estimate_offdiag_variances_from_jackknife(trace_KK):
    trace_KK = np.asarray(trace_KK, dtype=np.float64)
    if trace_KK.ndim != 3:
        raise ValueError("trace_KK must have shape (B+1, K, K)")

    B_plus, K, K2 = trace_KK.shape
    if K != K2:
        raise ValueError("trace_KK last two dimensions must be equal (KxK)")

    B = B_plus - 1

    # If we don't have enough jackknife blocks, just fall back to 0.5 weights.
    if B <= 1:
        var1 = np.zeros((K, K), dtype=np.float64)
        var2 = np.zeros((K, K), dtype=np.float64)
        cov12 = np.zeros((K, K), dtype=np.float64)
        w_opt = np.full((K, K), 0.5, dtype=np.float64)
        return var1, var2, cov12, w_opt

    jack = trace_KK[:B]  # (B, K, K)

    var1 = np.zeros((K, K), dtype=np.float64)
    var2 = np.zeros((K, K), dtype=np.float64)
    cov12 = np.zeros((K, K), dtype=np.float64)
    w_opt = np.full((K, K), 0.5, dtype=np.float64)

    denom = float(B - 1)

    for k in range(K):
        for l in range(K):
            if k == l:
                continue

            x = jack[:, k, l]
            y = jack[:, l, k]
            mx = x.mean()
            my = y.mean()
            dx = x - mx
            dy = y - my

            v1 = np.dot(dx, dx) / denom
            v2 = np.dot(dy, dy) / denom
            c12 = np.dot(dx, dy) / denom

            var1[k, l] = v1
            var2[k, l] = v2
            cov12[k, l] = c12

            # Optimal linear weight:
            # w* = (sigma2^2 - cov12) / (sigma1^2 + sigma2^2 - 2*cov12)
            denom_w = v1 + v2 - 2.0 * c12
            if denom_w <= 0.0 or not np.isfinite(denom_w):
                w = 0.5
            else:
                w = (v2 - c12) / denom_w
                if not np.isfinite(w):
                    w = 0.5
                else:
                    # Clamp to [0, 1] to avoid crazy weights from noise
                    if w < 0.0:
                        w = 0.0
                    elif w > 1.0:
                        w = 1.0

            w_opt[k, l] = w
            w_opt[l, k] = 1.0 - w

    # Diagonals: trivial, no off-diagonal ambiguity
    for k in range(K):
        var1[k, k] = 0.0
        var2[k, k] = 0.0
        cov12[k, k] = 0.0
        w_opt[k, k] = 0.5

    return var1, var2, cov12, w_opt


def symmetrize_trace_with_jackknife(
    trace_KK,
    logger=None,
    verbose=False,
    jk_block_sizes=None,  # only meaningful for delete-1 partition blocks
    jk_n_units=None,      # required for delete-d
    jk_delete_d: int = 1, # d
    nan_policy: str = "omit",
):
    trace_KK = np.asarray(trace_KK, dtype=np.float64)
    if trace_KK.ndim != 3:
        raise ValueError("trace_KK must have shape (B+1, K, K)")

    B_plus, K, K2 = trace_KK.shape
    if K != K2:
        raise ValueError("trace_KK last two dimensions must be equal (KxK)")

    B = B_plus - 1

    if B <= 0:
        # no replicates: just average full row
        sym = trace_KK.copy()
        full = sym[-1]
        for k in range(K):
            for l in range(k + 1, K):
                v = 0.5 * (full[k, l] + full[l, k])
                full[k, l] = full[l, k] = v
        sym[-1] = full
        return sym

    sym = trace_KK.copy()

    # diagnostics (full row)
    if logger is not None:
        full_before = trace_KK[B]
        tri = np.triu_indices(K, k=1)
        max_asym_before = float(np.abs(full_before - full_before.T)[tri].max(initial=0.0))

    # ----------------------------
    # w_opt estimators
    # ----------------------------
    def _w_opt_delete_d(trace_KK, n_units: int, delete_d: int):
        jack = trace_KK[:B]  # (B,K,K)
        w_opt = np.full((K, K), 0.5, dtype=np.float64)
        for k in range(K):
            w_opt[k, k] = 0.5

        U = int(n_units)
        d = int(delete_d)
        if not (1 <= d < U):
            return w_opt

        # factor cancels in w*, but keep for clarity
        factor = float(U - d) / float(d)

        for k in range(K):
            for l in range(k + 1, K):
                x = jack[:, k, l]
                y = jack[:, l, k]

                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() <= 1:
                    continue

                xf = x[mask]
                yf = y[mask]

                if nan_policy == "omit":
                    mx = float(np.mean(xf))
                    my = float(np.mean(yf))
                else:
                    # propagate
                    if not (np.isfinite(xf).all() and np.isfinite(yf).all()):
                        continue
                    mx = float(xf.mean())
                    my = float(yf.mean())

                dx = xf - mx
                dy = yf - my

                v1 = factor * float(np.mean(dx * dx))
                v2 = factor * float(np.mean(dy * dy))
                c12 = factor * float(np.mean(dx * dy))

                denom_w = v1 + v2 - 2.0 * c12
                if denom_w <= 0.0 or not np.isfinite(denom_w):
                    wstar = 0.5
                else:
                    wstar = (v2 - c12) / denom_w
                    if not np.isfinite(wstar):
                        wstar = 0.5
                    else:
                        wstar = 0.0 if wstar < 0.0 else (1.0 if wstar > 1.0 else wstar)

                w_opt[k, l] = wstar
                w_opt[l, k] = 1.0 - wstar

        return w_opt

    def _w_opt_delete1_weighted(trace_KK, jk_block_sizes):
        """
        Your existing delete-1 delete-m pseudovalue approach.

        IMPORTANT:
        This assumes blocks form a PARTITION and sum(m_b)=M_total.
        """
        m = np.asarray(jk_block_sizes, dtype=np.float64).ravel()
        if m.size != B:
            raise ValueError(f"jk_block_sizes must have length B={B}, got {m.size}")

        good = np.isfinite(m) & (m > 0)
        if good.sum() <= 1:
            return np.full((K, K), 0.5, dtype=np.float64)

        m = m[good]
        M = float(m.sum())
        if not (np.isfinite(M) and M > 0):
            return np.full((K, K), 0.5, dtype=np.float64)

        w = m / M
        jack = trace_KK[:B][good, :, :]
        full = trace_KK[B]

        w_opt = np.full((K, K), 0.5, dtype=np.float64)
        for k in range(K):
            w_opt[k, k] = 0.5

        for k in range(K):
            for l in range(k + 1, K):
                x_full = float(full[k, l])
                y_full = float(full[l, k])

                x = jack[:, k, l]
                y = jack[:, l, k]

                PVx = (M * x_full - (M - m) * x) / m
                PVy = (M * y_full - (M - m) * y) / m

                finite = np.isfinite(PVx) & np.isfinite(PVy)
                if finite.sum() <= 1:
                    continue

                wf = w[finite]
                wf_sum = float(wf.sum())
                if not (np.isfinite(wf_sum) and wf_sum > 0):
                    continue
                wf = wf / wf_sum

                PVx_f = PVx[finite]
                PVy_f = PVy[finite]

                mx = float(np.sum(wf * PVx_f))
                my = float(np.sum(wf * PVy_f))
                dx = PVx_f - mx
                dy = PVy_f - my

                w2 = float(np.sum(wf * wf))
                denom = 1.0 - w2
                if not (np.isfinite(denom) and denom > 0):
                    continue

                num1 = float(np.sum((wf * wf) * (dx * dx)))
                num2 = float(np.sum((wf * wf) * (dy * dy)))
                numc = float(np.sum((wf * wf) * (dx * dy)))

                v1 = num1 / denom
                v2 = num2 / denom
                c12 = numc / denom

                denom_w = v1 + v2 - 2.0 * c12
                if denom_w <= 0.0 or not np.isfinite(denom_w):
                    wstar = 0.5
                else:
                    wstar = (v2 - c12) / denom_w
                    if not np.isfinite(wstar):
                        wstar = 0.5
                    else:
                        wstar = 0.0 if wstar < 0.0 else (1.0 if wstar > 1.0 else wstar)

                w_opt[k, l] = wstar
                w_opt[l, k] = 1.0 - wstar

        return w_opt

    # Choose w_opt
    if int(jk_delete_d) > 1:
        if jk_n_units is None:
            raise ValueError("delete-d symmetrization requires jk_n_units (e.g. 22 chromosomes).")
        w_opt = _w_opt_delete_d(trace_KK, int(jk_n_units), int(jk_delete_d))
        if logger is not None and verbose:
            logger._log(
                f"[Trace] symmetrize: delete-d mode used "
                f"(U={int(jk_n_units)}, d={int(jk_delete_d)}, R={B})."
            )
    elif jk_block_sizes is not None:
        w_opt = _w_opt_delete1_weighted(trace_KK, jk_block_sizes)
        if logger is not None and verbose:
            logger._log(f"[Trace] symmetrize: delete-1 weighted blocks used (B={B}).")
    else:
        # legacy equal-weight estimate
        _, _, _, w_opt = estimate_offdiag_variances_from_jackknife(trace_KK)

    # Apply weights to symmetrize all replicate rows and full row
    for k in range(K):
        for l in range(k + 1, K):
            w = float(w_opt[k, l])

            x = trace_KK[:B, k, l]
            y = trace_KK[:B, l, k]
            v_jk = w * x + (1.0 - w) * y
            sym[:B, k, l] = v_jk
            sym[:B, l, k] = v_jk

            x_full = trace_KK[B, k, l]
            y_full = trace_KK[B, l, k]
            v_full = w * x_full + (1.0 - w) * y_full
            sym[B, k, l] = v_full
            sym[B, l, k] = v_full

    if logger is not None and verbose:
        full_after = sym[B]
        tri = np.triu_indices(K, k=1)
        max_asym_after = float(np.abs(full_after - full_after.T)[tri].max(initial=0.0))
        logger._log(
            f"[Trace] Jackknife-based symmetrization: max off-diagonal asym "
            f"(before, after) = ({max_asym_before:.4e}, {max_asym_after:.4e})"
        )

    return sym


# -----------------------
# Jackknife helpers
# -----------------------

def _calc_jackknife_se_from_delete_sets(
    alist,
    D,
    unit_sizes,
    axis=0,
    center="mean",
    nan_policy="omit",
):
    """
    Overlap-aware jackknife SE for chr-mode delete-* replicates.

    Design
    ------
    1) Exact delete-1 LOCO full set (R == U and D is a permutation of I):
       use the exact unequal-unit delete-1 pseudovalue jackknife.

    2) General delete-d / sampled delete-1:
       reconstruct unit-level pseudovalues from the overlapping delete-set equations
           M * theta_full - (M - m_S) * theta_{-S} ~= sum_{u in S} g_u
       where g_u = m_u * PV_u,
       then apply the unequal-unit delete-1 variance formula to the recovered PV_u.

       This is the overlap-aware generalization you were aiming for with your
       second implementation, but fixed so that it also works for LOCO delete-1,
       handles NaNs per-parameter, and falls back safely if the delete-set design
       is not identifiable.

    Notes
    -----
    - For equal-size units and full combinatorial delete-d, this is asymptotically
      equivalent to the usual delete-d jackknife scaling.
    - For unequal chromosome sizes, this is the safer formulation.
    """
    a = np.asarray(alist)
    est_full = np.take(a, indices=-1, axis=axis)

    # First R slices = replicates, last slice = full
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(0, -1)
    reps = a[tuple(slicer)]
    reps = np.moveaxis(reps, axis, 0)  # (R, ...)
    R = reps.shape[0]

    if R <= 0:
        return est_full, np.full_like(np.asarray(est_full, dtype=np.float64), np.nan, dtype=np.float64)

    D = np.asarray(D, dtype=np.float64, order="C")
    if D.ndim != 2:
        raise ValueError("D must be 2D with shape (R, U).")
    if D.shape[0] != R:
        raise ValueError(f"D has R={D.shape[0]} rows but alist has R={R} replicates.")

    U = int(D.shape[1])

    unit_sizes = np.asarray(unit_sizes, dtype=np.float64).ravel()
    if unit_sizes.size != U:
        raise ValueError(f"unit_sizes must have length U={U}, got {unit_sizes.size}.")

    M = float(np.sum(unit_sizes))
    if not (np.isfinite(M) and M > 0.0):
        raise ValueError(f"Total unit size M must be positive finite; got {M}.")

    reps_flat = np.asarray(reps, dtype=np.float64).reshape(R, -1)  # (R, P)
    full_flat = np.asarray(est_full, dtype=np.float64).reshape(-1) # (P,)
    P = full_flat.size

    m_del = D @ unit_sizes  # (R,)

    good_rep_base = np.isfinite(m_del) & (m_del > 0.0) & (m_del < M)
    good_u_base = np.isfinite(unit_sizes) & (unit_sizes > 0.0) & (unit_sizes < M)

    use_u = np.flatnonzero(good_u_base)
    if use_u.size == 0:
        return est_full, np.full_like(np.asarray(est_full, dtype=np.float64), np.nan, dtype=np.float64)

    m_u = unit_sizes[use_u]
    w_u = m_u / M

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _center_on_values(vals, weights, full_scalar):
        vals = np.asarray(vals, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        if center == "full":
            return float(full_scalar)

        finite = np.isfinite(vals) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(finite):
            return np.nan

        vf = vals[finite]
        wf = weights[finite]
        sw = float(wf.sum())
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan
        wf = wf / sw

        if center == "mean":
            return float(np.sum(wf * vf))
        elif center == "median":
            return float(np.median(vf))
        else:
            raise ValueError("center must be one of {'full','mean','median'}")

    def _se_from_unit_pseudovalues(pv_all, full_scalar):
        """
        pv_all : shape (U,)
        """
        pv_all = np.asarray(pv_all, dtype=np.float64)

        if nan_policy == "propagate":
            valid = good_u_base
            if not (np.isfinite(full_scalar) and np.all(np.isfinite(pv_all[valid]))):
                return np.nan

            vals = pv_all[valid]
            ww = (unit_sizes[valid] / M).astype(np.float64)
            ctr = _center_on_values(vals, ww, full_scalar)
            if not np.isfinite(ctr):
                return np.nan

            diffs = vals - ctr
            denom = 1.0 - ww
            with np.errstate(divide="ignore", invalid="ignore"):
                term = (ww * ww / denom) * (diffs * diffs)
            var = float(np.sum(term))
            return float(np.sqrt(var)) if np.isfinite(var) else np.nan

        # omit
        valid = good_u_base & np.isfinite(pv_all)
        if not np.any(valid):
            return np.nan

        vals = pv_all[valid]
        ww = (unit_sizes[valid] / M).astype(np.float64)
        sw = float(ww.sum())
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan
        ww = ww / sw

        ctr = _center_on_values(vals, ww, full_scalar)
        if not np.isfinite(ctr):
            return np.nan

        diffs = vals - ctr
        denom = 1.0 - ww
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (ww * ww / denom) * (diffs * diffs)
        term = np.where(np.isfinite(term), term, 0.0)

        w2 = float(np.sum(ww * ww))
        n_eff = (1.0 / w2) if (w2 > 0.0 and np.isfinite(w2)) else 0.0
        var = float(np.sum(term))

        if not (np.isfinite(var) and n_eff > 1.0):
            return np.nan
        return float(np.sqrt(var))

    def _se_direct_delete_d(rep_vals, full_scalar):
        """
        Fallback only if the delete-set design is underidentified for pseudovalue
        reconstruction. This is not the preferred path for your chromosome setting.
        """
        rep_vals = np.asarray(rep_vals, dtype=np.float64)
        valid = good_rep_base & np.isfinite(rep_vals)

        if nan_policy == "propagate":
            if not (np.isfinite(full_scalar) and np.all(valid)):
                return np.nan
            x = rep_vals
            sc = (M - m_del) / m_del
        else:
            if not np.any(valid):
                return np.nan
            x = rep_vals[valid]
            sc = ((M - m_del[valid]) / m_del[valid]).astype(np.float64)

        if center == "full":
            ctr = float(full_scalar)
        elif center == "mean":
            ctr = float(np.mean(x))
        elif center == "median":
            ctr = float(np.median(x))
        else:
            raise ValueError("center must be one of {'full','mean','median'}")

        diffs = x - ctr
        var = float(np.mean(sc * diffs * diffs))
        return float(np.sqrt(var)) if np.isfinite(var) else np.nan

    # ------------------------------------------------------------------
    # Detect exact delete-1 LOCO full set: D is a permutation of I
    # ------------------------------------------------------------------
    mask = D > 0.5
    exact_loco = (
        R == U
        and np.all(mask.sum(axis=1) == 1)
        and np.all(mask.sum(axis=0) == 1)
    )

    se_flat = np.full(P, np.nan, dtype=np.float64)

    # ------------------------------------------------------------------
    # Case A: exact LOCO delete-1 (exact unequal-unit pseudovalues)
    # ------------------------------------------------------------------
    if exact_loco:
        unit_of_rep = mask.argmax(axis=1)  # replicate r deletes unit unit_of_rep[r]
        rep_for_unit = np.empty(U, dtype=np.int64)  # inverse map: unit u -> replicate index
        rep_for_unit[unit_of_rep] = np.arange(R, dtype=np.int64)

        for p in range(P):
            full_p = float(full_flat[p])
            rep_p = reps_flat[:, p]

            if nan_policy == "propagate" and (not np.isfinite(full_p) or not np.all(np.isfinite(rep_p))):
                se_flat[p] = np.nan
                continue
            if not np.isfinite(full_p):
                se_flat[p] = np.nan
                continue

            pv_all = np.full(U, np.nan, dtype=np.float64)
            for u in use_u:
                r = int(rep_for_unit[u])
                th_r = rep_p[r]
                if np.isfinite(th_r):
                    mu = float(unit_sizes[u])
                    pv_all[u] = (M * full_p - (M - mu) * th_r) / mu

            se_flat[p] = _se_from_unit_pseudovalues(pv_all, full_p)

        return est_full, se_flat.reshape(est_full.shape)

    # ------------------------------------------------------------------
    # Case B: general delete-d / sampled delete-1
    # overlap-aware reconstruction of unit pseudovalues
    # ------------------------------------------------------------------
    D_use = D[:, use_u]  # (R, U_use)
    U_use = D_use.shape[1]

    for p in range(P):
        full_p = float(full_flat[p])
        rep_p = reps_flat[:, p]

        if not np.isfinite(full_p):
            se_flat[p] = np.nan
            continue

        valid_rep = good_rep_base & np.isfinite(rep_p)
        if nan_policy == "propagate" and not np.all(valid_rep):
            se_flat[p] = np.nan
            continue
        if nan_policy == "omit" and not np.any(valid_rep):
            se_flat[p] = np.nan
            continue

        Dv = D_use[valid_rep, :]  # (Rv, U_use)
        yv = M * full_p - (M - m_del[valid_rep]) * rep_p[valid_rep]  # (Rv,)

        # If the delete-set design is not identifiable, fall back safely.
        # This can happen if too few random subsets were sampled.
        rank = np.linalg.matrix_rank(Dv) if Dv.size else 0
        if (Dv.shape[0] < U_use) or (rank < U_use):
            se_flat[p] = _se_direct_delete_d(rep_p, full_p)
            continue

        # Solve Dv g ~= yv for g_u = m_u * PV_u
        # Prefer normal equations solve; add tiny ridge only if needed.
        AtA = Dv.T @ Dv
        Aty = Dv.T @ yv

        try:
            g = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            tr = float(np.trace(AtA))
            lam = 1e-10 * (tr / U_use if (np.isfinite(tr) and tr > 0.0) else 1.0)
            try:
                g = np.linalg.solve(AtA + lam * np.eye(U_use, dtype=np.float64), Aty)
            except np.linalg.LinAlgError:
                g = np.linalg.lstsq(Dv, yv, rcond=None)[0]

        pv_use = g / m_u
        pv_all = np.full(U, np.nan, dtype=np.float64)
        pv_all[use_u] = pv_use

        se_flat[p] = _se_from_unit_pseudovalues(pv_all, full_p)

    return est_full, se_flat.reshape(est_full.shape)

