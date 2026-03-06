import numpy as np
import pandas as pd
import os
import re
import time
import datetime
import glob

# ----------------------- I/O & parsing helpers ----------------------- #
def _read_multiple_lines(file_path, num_lines, sep=','):
    '''
    Processes a input file (num_lines) lines at a time
    '''
    values = pd.read_csv(file_path, chunksize=num_lines)
    for val in values:
        yield val.to_numpy()

def _read_with_optional_header(file_path):
    with open(file_path, 'r') as fd:
        line = fd.readline().strip()
        try:
            vals = [float(x) for x in line.split()]
            is_header = False
        except ValueError:
            is_header = True
    if is_header:
        header = line.split()
        data = np.loadtxt(file_path, skiprows=1)
        return header, data
    else:
        data = np.loadtxt(file_path)
        return None, data

def _find_matching_files(regex, prefix):
    '''
    regex file matching. returns a list of matches (with specified path prefix)
    '''
    pattern = re.compile(regex)
    files = os.listdir(prefix)
    return [prefix+file for file in files if pattern.match(file)]

def _get_time():
    current_time = time.time()
    return current_time

def _get_timestr(current_time):
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    timestr = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)))+" "+str(timezone)
    return timestr

def _parse_sumdir(path):
    '''
    check whether the path for sumstats is a directory or a file (or even regex).
    TODO: allow regex matching for file names
    '''
    if not os.path.exists(path):
        raise ValueError(f"--h2 path '{path}' does not exist")
    # if dir, glob for anything with “.sumstat” in the name
    if os.path.isdir(path):
        pattern = os.path.join(path.rstrip("/"), "*.sumstat*")
        sum_files = sorted(glob.glob(pattern))
        if not sum_files:
            raise ValueError(f"--h2 path '{path}' contains no '*.sumstat*' files")
        return sum_files

    # if file, only accept if it has “.sumstat” in the basename
    if os.path.isfile(path):
        name = os.path.basename(path)
        if ".sumstat" in name:
            return [path]
        else:
            raise ValueError(f"--h2 file '{path}' is not a '*.sumstat*' file")
    raise ValueError(f"--h2 path '{path}' is invalid")

def _parse_rgdir(rg):
    """
    Parse an --rg argument string into exactly two sumstat file paths.
    Accepts any filename containing '.sumstat' (e.g. .sumstat, .sumstat.gz, etc.)
    """
    if rg is None:
        raise ValueError("--rg must be provided for genetic correlation.")

    paths = rg.split(",")
    if len(paths) != 2:
        raise ValueError("--rg must be exactly two comma-separated '*.sumstat*' files.")

    validated = []
    for p in paths:
        if not os.path.isfile(p):
            raise ValueError(f"--rg path '{p}' does not exist or is not a file.")
        if ".sumstat" not in os.path.basename(p):
            raise ValueError(f"--rg file '{p}' is not a valid '*.sumstat*' file.")
        validated.append(p)

    return validated

def _parse_column_name(df, letters, min_index=3):
    '''
    select only the column names that include certain letters & after certain column index
    '''
    matching_columns = [col for col in df.columns[min_index:] if any(letter in col for letter in letters)]

    if len(matching_columns) == 0:
        raise ValueError(f"No column containing any of the letters {letters} found starting from column {min_index}.")
    elif len(matching_columns) > 1:
        raise ValueError(f"Multiple columns containing the letters {letters} found: {matching_columns}. Expected only one.")
    else:
        return matching_columns[0]

def _normalize_enrich_mode(mode: str) -> str:
    if mode is None:
        return "auto"
    m = str(mode).strip().lower().replace("_", "-").replace(" ", "-")
    if m in ("auto",):
        return "auto"
    if m in ("overlap", "overlapping"):
        return "overlap"
    if m in ("non-overlap", "nonoverlap", "nonoverlapping", "component", "components"):
        return "non-overlap"
    if m in ("both", "all"):
        return "both"
    raise ValueError(f"Invalid enrich_mode={mode!r}. Choose from: auto, overlap, non-overlap, both.")

def _has_overlapping_annotations(A: np.ndarray) -> bool:
    """
    Returns True if any SNP has >1 nonzero annotation entry.
    Works for binary or continuous weights. Exact-zero based.
    """
    # np.count_nonzero is C-optimized and avoids a big Python loop.
    return bool(np.any(np.count_nonzero(A, axis=1) > 1))


def _suggested_chisq_max(nmax: float) -> float:
    if not np.isfinite(nmax) or nmax <= 0:
        return 80.0
    return float(max(80.0, 0.001 * nmax))


def _resolve_chisq_threshold(nmax: float, raw=None):
    if raw is None:
        return None, "none"

    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "auto":
            return float(_suggested_chisq_max(nmax)), "auto"
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


def _calc_jackknife_se(
    alist,
    axis=0,
    center="mean",
    nan_policy="omit",
    weights=None,
    use_pseudovalues=False,
):
    """
    Jackknife SE along axis for arrays shaped (B+1, ...), where the last slice
    is the full-sample estimate and the first B are jackknife replicates.

    - If use_pseudovalues=False OR weights is None:
      Standard equal-weight jackknife:
        Var = (B-1)/B * sum_b (theta_b - center)^2
      with optional NaN omission.

    - If use_pseudovalues=True AND weights provided:
      Delete-m (unequal block sizes) via pseudovalues:
        PV_b = (M*theta_full - (M-m_b)*theta_{-b}) / m_b
      and variance computed using the unequal-weight delete-1 formula:
        Var = sum_b [ w_b^2/(1-w_b) * (PV_b - PV_center)^2 ],
      where w_b = m_b / M.

    Returns
    -------
    est_full, se
    """
    a = np.asarray(alist)
    est_full = np.take(a, indices=-1, axis=axis)

    # LOO replicates = all but last along axis
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(0, -1)
    reps = a[tuple(slicer)]
    reps = np.moveaxis(reps, axis, 0)  # (B, ...)
    B = reps.shape[0]

    if B <= 0:
        return est_full, np.full_like(est_full, np.nan, dtype=np.float64)

    # ----------------------------
    # Branch 1: legacy equal-weight jackknife
    # ----------------------------
    if (not use_pseudovalues) or (weights is None):
        if center == "full":
            center_arr = est_full
        elif center == "mean":
            center_arr = np.nanmean(reps, axis=0) if nan_policy == "omit" else np.mean(reps, axis=0)
        elif center == "median":
            center_arr = np.nanmedian(reps, axis=0) if nan_policy == "omit" else np.median(reps, axis=0)
        else:
            raise ValueError("center must be one of {'full','mean','median'}")

        diffs = np.asarray(reps, dtype=np.float64) - np.asarray(center_arr, dtype=np.float64)

        if nan_policy == "omit":
            finite = np.isfinite(diffs)
            m_eff = finite.sum(axis=0).astype(np.float64)  # per-coordinate effective replicate count
            diffs = np.where(finite, diffs, 0.0)
            ss = np.sum(diffs * diffs, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                var = (np.maximum(m_eff - 1.0, 0.0) / np.maximum(m_eff, 1.0)) * ss
            se = np.sqrt(var)
            se = np.where(m_eff >= 1.0, se, np.nan)
        else:
            # propagate
            ss = np.sum(diffs * diffs, axis=0)
            var = ((B - 1.0) / B) * ss
            se = np.sqrt(var)

        return est_full, np.asarray(se, dtype=np.float64).reshape(est_full.shape)

    # ----------------------------
    # Branch 2: weighted delete-m via pseudovalues (delete-1 partition scheme)
    # ----------------------------
    m = np.asarray(weights, dtype=np.float64).ravel()
    if m.size != B:
        raise ValueError(f"weights must have length B={B}, got {m.size}")

    good = np.isfinite(m) & (m > 0.0)
    if not np.any(good):
        return est_full, np.full_like(est_full, np.nan, dtype=np.float64)

    reps = np.asarray(reps, dtype=np.float64)[good, ...]
    m = m[good]
    B2 = reps.shape[0]
    M = float(np.sum(m))

    if B2 <= 0 or (not np.isfinite(M)) or M <= 0.0:
        return est_full, np.full_like(est_full, np.nan, dtype=np.float64)

    w = (m / M).astype(np.float64)  # (B2,)

    # avoid pathological w==1
    goodw = np.isfinite(w) & (w > 0.0) & (w < 1.0)
    if not np.any(goodw):
        return est_full, np.full_like(est_full, np.nan, dtype=np.float64)

    reps = reps[goodw, ...]
    m = m[goodw]
    w = w[goodw]
    B2 = reps.shape[0]

    # Broadcast shapes
    reshape = (B2,) + (1,) * (reps.ndim - 1)
    m_b = m.reshape(reshape)
    w_b = w.reshape(reshape)

    # Pseudovalues
    est_full_f = np.asarray(est_full, dtype=np.float64)
    PV = (M * est_full_f - (M - m_b) * reps) / m_b  # (B2, ...)

    # Center on PV scale
    if center == "full":
        center_arr = est_full_f
    elif center == "median":
        center_arr = np.nanmedian(PV, axis=0) if nan_policy == "omit" else np.median(PV, axis=0)
    elif center == "mean":
        if nan_policy == "propagate":
            if not np.isfinite(PV).all():
                return est_full, np.full_like(est_full, np.nan, dtype=np.float64)
            center_arr = np.sum(w_b * PV, axis=0)  # weights sum to 1
        else:
            finite = np.isfinite(PV)
            w_eff = w_b * finite
            sw = np.sum(w_eff, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_norm = np.where(sw > 0, w_eff / sw, 0.0)
            center_arr = np.sum(w_norm * PV, axis=0)
    else:
        raise ValueError("center must be one of {'full','mean','median'}")

    diffs = PV - center_arr

    # Var = Σ w^2/(1-w) * diffs^2 (delete-1 unequal-size)
    if nan_policy == "propagate":
        if not np.isfinite(diffs).all():
            return est_full, np.full_like(est_full, np.nan, dtype=np.float64)
        denom = 1.0 - w_b
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (w_b * w_b / denom) * (diffs * diffs)
        var = np.sum(term, axis=0)
        var = np.where(np.isfinite(var), var, np.nan)
        se = np.sqrt(var)
        return est_full, np.asarray(se, dtype=np.float64).reshape(est_full.shape)

    # omit NaNs per-coordinate
    finite = np.isfinite(diffs)
    w_eff = w_b * finite
    sw = np.sum(w_eff, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_norm = np.where(sw > 0, w_eff / sw, 0.0)

    denom = 1.0 - w_norm

    # effective dof guard
    w2 = np.sum(w_norm * w_norm, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(w2 > 0, 1.0 / w2, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        term = (w_norm * w_norm / denom) * (diffs * diffs)
    term = np.where(np.isfinite(term), term, 0.0)

    var = np.sum(term, axis=0)
    var = np.where((sw > 0) & (n_eff > 1.0) & np.isfinite(var), var, np.nan)
    se = np.sqrt(var)

    return est_full, np.asarray(se, dtype=np.float64).reshape(est_full.shape)


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


# -----------------------
# rg helpers
# -----------------------


def _solve_linear_equation(X, y, method="auto"):
    """
    Solve A x = b.

    Supports batched solves:
      X shape: (..., p, p)
      y shape: (..., p) or (..., p, k)

    For SPD matrices, Cholesky is fastest and most stable.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if method == "lstsq":
        # Keep original behavior if explicitly requested
        return np.linalg.lstsq(X, y, rcond=None)[0]

    # Try Cholesky (fast path)
    try:
        L = np.linalg.cholesky(X)  # (..., p, p)

        # forward solve L z = y
        z = np.linalg.solve(L, y[..., None]).squeeze(-1)  # (..., p) or (..., p, k)

        # backward solve L.T x = z
        x = np.linalg.solve(np.swapaxes(L, -1, -2), z[..., None]).squeeze(-1)
        return x

    except np.linalg.LinAlgError:
        # Fall back to generic solver (still batched)
        return np.linalg.solve(X, y)


def bivariate_regression_partitioned_jn(
    l2_bins,
    y,
    nblks,
    n1,
    n2,
    nsnps_blk,
    blk_idx=None,
    weight_floor=None,
    weight_cap_quantile=None,
    chisq1=None,
    chisq2=None,
    chisq_threshold=None,
    chisq_mode="either",  # "either" (default), "both", "max"
):
    """
    Leave-one-block-out WLS regression of y = z1*z2 on partitioned LD scores + intercept.

    If chisq_threshold is not None, we *temporarily* exclude SNPs with large chi^2
    (typically > 30) from THIS regression only by setting their weights to 0.
    This keeps the full SNP indexing and jackknife block structure unchanged.

    Recommended setting (LDSC-style for rg intercept step):
        chisq_mode="either", chisq_threshold=30, chisq1=z1^2, chisq2=z2^2
    """
    L = np.asarray(l2_bins, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()
    nsnps_blk = np.asarray(nsnps_blk, dtype=np.float64)

    if L.ndim != 2:
        raise ValueError("l2_bins must be 2-D (M, K)")

    M, K = L.shape
    if y.size != M:
        raise ValueError(f"y must have shape (M,), got {y.shape}, expected M={M}")

    nblks = int(nblks)
    if nblks <= 0:
        raise ValueError("nblks must be positive")

    if nsnps_blk.shape != (nblks + 1, K):
        raise ValueError(f"nsnps_blk must be (nblks+1, K)=({nblks+1},{K}), got {nsnps_blk.shape}")

    n1 = float(n1)
    n2 = float(n2)
    if not (np.isfinite(n1) and np.isfinite(n2) and n1 > 0 and n2 > 0):
        raise ValueError("n1 and n2 must be positive finite")

    sN = np.sqrt(n1 * n2)

    # ----------------------------
    # block index
    # ----------------------------
    if blk_idx is None:
        blk_size = M // nblks
        if blk_size == 0:
            raise ValueError("Too many jackknife blocks (nblks > M).")

        blk_idx = np.repeat(np.arange(nblks, dtype=np.int64), blk_size)
        if blk_idx.size < M:
            blk_idx = np.concatenate(
                [blk_idx, np.full(M - blk_idx.size, nblks - 1, dtype=np.int64)]
            )
    else:
        blk_idx = np.asarray(blk_idx, dtype=np.int64).ravel()
        if blk_idx.size != M:
            raise ValueError(f"blk_idx must have length M={M}, got {blk_idx.size}")
        if blk_idx.min() < 0 or blk_idx.max() >= nblks:
            raise ValueError(f"blk_idx values must be in [0, nblks-1]=[0,{nblks-1}]")

    # Derive contiguous block bounds once (assumes piecewise-constant blk_idx)
    change = np.flatnonzero(blk_idx[1:] != blk_idx[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [M]))
    run_blk = blk_idx[starts]

    blk_starts = np.zeros(nblks, dtype=np.int64)
    blk_ends = np.zeros(nblks, dtype=np.int64)
    for r, b in enumerate(run_blk):
        blk_starts[int(b)] = int(starts[r])
        blk_ends[int(b)] = int(ends[r])

    # ----------------------------
    # chisq-based keep mask for intercept regression
    # ----------------------------
    keep = np.ones(M, dtype=bool)
    if chisq_threshold is not None:
        thr = float(chisq_threshold)
        if not (np.isfinite(thr) and thr > 0):
            raise ValueError("chisq_threshold must be positive finite")
        if chisq1 is None or chisq2 is None:
            raise ValueError("chisq1 and chisq2 must be provided when chisq_threshold is set")

        c1 = np.asarray(chisq1, dtype=np.float64).ravel()
        c2 = np.asarray(chisq2, dtype=np.float64).ravel()
        if c1.size != M or c2.size != M:
            raise ValueError(f"chisq1/chisq2 must have length M={M}")

        # require finite chisq for kept SNPs
        finite = np.isfinite(c1) & np.isfinite(c2)
        keep &= finite

        mode = str(chisq_mode).lower()
        if mode == "either":
            keep &= (c1 <= thr) & (c2 <= thr)  # drop if either > thr
        elif mode == "both":
            keep &= ~((c1 > thr) & (c2 > thr))  # drop only if both > thr
        elif mode == "max":
            keep &= (np.maximum(c1, c2) <= thr)
        else:
            raise ValueError("chisq_mode must be one of {'either','both','max'}")

    # ----------------------------
    # base weights w = 1 / sum_k l2_{j,k}, with optional floor/cap
    # and then set w=0 for excluded SNPs.
    # ----------------------------
    ltot = L.sum(axis=1)

    if weight_floor is None:
        bad = keep & ((~np.isfinite(ltot)) | (ltot <= 0.0))
        if np.any(bad):
            nb = int(bad.sum())
            mn = float(np.nanmin(ltot))
            idx_bad = np.flatnonzero(bad)[:10]
            raise ValueError(
                "Total LD (ltot) must be finite and > 0 for weights w=1/ltot on KEPT SNPs.\n"
                f"Found {nb}/{M} kept SNPs with ltot <= 0 or non-finite (min={mn}).\n"
                "Pass weight_floor to clamp.\n"
                f"First bad kept indices: {idx_bad.tolist()}"
            )

        # for dropped SNPs, ltot value doesn't matter (we'll set w=0); keep it safe anyway
        ltot_safe = np.where(keep, ltot, 1.0)
    else:
        eps = float(weight_floor)
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError("weight_floor must be positive finite")

        ltot_safe = np.where(np.isfinite(ltot), ltot, eps)
        ltot_safe = np.maximum(ltot_safe, eps)
        ltot_safe = np.where(keep, ltot_safe, 1.0)

    w = 1.0 / ltot_safe
    w[~keep] = 0.0

    # Optional cap (apply only to positive weights, otherwise zeros distort quantile)
    if weight_cap_quantile is not None:
        q = float(weight_cap_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("weight_cap_quantile must be in (0,1)")

        wpos = w[w > 0]
        if wpos.size > 0:
            cap = float(np.quantile(wpos, q))
            if np.isfinite(cap) and cap > 0:
                w = np.minimum(w, cap)

    # Sanity: need enough kept weight mass to fit (K+1) params
    if w.sum() <= 0:
        raise ValueError(
            "After chisq filtering + weighting, no SNPs remain for intercept regression (sum(w)=0)."
        )

    # not a strict requirement, but helps catch pathological filtering
    if int((w > 0).sum()) < (K + 5):
        raise ValueError(
            f"Too few SNPs after chisq filtering for stable regression: kept={(w > 0).sum()} < K+5={K+5}. "
            "Relax chisq_threshold or check inputs."
        )

    wy = w * y

    # ----------------------------
    # total normal equations (p = K+1)
    # ----------------------------
    S00_tot = float(w.sum())
    S0_tot = L.T @ w
    LW = L * w[:, None]
    SLL_tot = LW.T @ L
    S0y_tot = float(wy.sum())
    SLy_tot = L.T @ wy

    p = K + 1
    SXX_tot = np.empty((p, p), dtype=np.float64)
    SXY_tot = np.empty((p,), dtype=np.float64)

    SXX_tot[0, 0] = S00_tot
    SXX_tot[0, 1:] = S0_tot
    SXX_tot[1:, 0] = S0_tot
    SXX_tot[1:, 1:] = SLL_tot

    SXY_tot[0] = S0y_tot
    SXY_tot[1:] = SLy_tot

    # ----------------------------
    # per-block contributions
    # ----------------------------
    SXX_blk = np.zeros((nblks, p, p), dtype=np.float64)
    SXY_blk = np.zeros((nblks, p), dtype=np.float64)

    for b in range(nblks):
        s = int(blk_starts[b])
        e = int(blk_ends[b])
        if e <= s:
            continue

        Lb = L[s:e, :]
        wb = w[s:e]
        if wb.sum() <= 0:
            continue

        wyb = wy[s:e]

        S00 = float(wb.sum())
        S0 = Lb.T @ wb
        SLy = Lb.T @ wyb
        S0y = float(wyb.sum())
        SLL = (Lb * wb[:, None]).T @ Lb

        Sb = SXX_blk[b]
        Sb[0, 0] = S00
        Sb[0, 1:] = S0
        Sb[1:, 0] = S0
        Sb[1:, 1:] = SLL

        tb = SXY_blk[b]
        tb[0] = S0y
        tb[1:] = SLy

    # LOO systems
    SXX = SXX_tot[None, :, :] - SXX_blk  # (B, p, p)
    SXY = SXY_tot[None, :] - SXY_blk     # (B, p)

    # Some LOO replicates could end up with (almost) no weight if the dropped SNPs cluster.
    # Solve only the valid ones; others -> NaN.
    beta_j = np.full((nblks, p), np.nan, dtype=np.float64)
    valid = SXX[:, 0, 0] > 0  # intercept weight mass in LOO replicate

    if np.any(valid):
        Sv = SXX[valid]
        tv = SXY[valid]
        try:
            beta_j[valid] = np.linalg.solve(Sv, tv[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # per-replicate fallback
            for ii, b in enumerate(np.flatnonzero(valid)):
                try:
                    beta_j[b] = np.linalg.solve(SXX[b], SXY[b])
                except np.linalg.LinAlgError:
                    beta_j[b] = np.linalg.lstsq(SXX[b], SXY[b], rcond=None)[0]

    # Full solve
    try:
        beta_full = np.linalg.solve(SXX_tot, SXY_tot)
    except np.linalg.LinAlgError:
        beta_full = np.linalg.lstsq(SXX_tot, SXY_tot, rcond=None)[0]

    beta_all = np.vstack([beta_j, beta_full[None, :]])  # (B+1, p)

    # Scale slopes -> gamma
    scale = nsnps_blk / sN  # (B+1, K)
    gamma_all = scale * beta_all[:, 1:]  # (B+1, K)
    c_all = beta_all[:, 0]               # (B+1,)

    return gamma_all, c_all


def compute_t1_all_jn(annot, y, blk_idx, nblks):
    """
    Same math as before, but robust if a block appears in multiple segments:

      T1_full   = A^T y
      T1_blk[b] = sum over all segments belonging to block b of (A_seg^T y_seg)
      T1_LOO    = full - blk
    """
    A = np.asarray(annot, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()
    blk_idx = np.asarray(blk_idx, dtype=np.int64).ravel()

    if A.ndim != 2:
        raise ValueError("annot must be 2D (M,K)")

    M, K = A.shape
    if y.size != M:
        raise ValueError("y length mismatch with annot")
    if blk_idx.size != M:
        raise ValueError("blk_idx length mismatch with annot")

    nblks = int(nblks)
    if blk_idx.min(initial=0) < 0 or blk_idx.max(initial=0) >= nblks:
        raise ValueError(f"blk_idx values must be in [0, nblks-1]=[0,{nblks-1}]")

    T1_full = A.T @ y  # (K,)

    # segments
    change = np.flatnonzero(blk_idx[1:] != blk_idx[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [M]))
    seg_blk = blk_idx[starts]

    T1_blk = np.zeros((nblks, K), dtype=np.float64)
    for r in range(starts.size):
        b = int(seg_blk[r])
        s = int(starts[r])
        e = int(ends[r])
        if e <= s:
            continue
        T1_blk[b] += A[s:e, :].T @ y[s:e]

    T1_all = np.empty((nblks + 1, K), dtype=np.float64)
    T1_all[:nblks] = T1_full[None, :] - T1_blk
    T1_all[nblks] = T1_full
    return T1_all


def solve_score_gamma_from_intercept_jn(
    ld_sum_all,
    t1_all,
    nsnps_blk,
    c_all,
    n1,
    n2,
    ridge_rel=0,
):
    ld_sum_all = np.asarray(ld_sum_all, dtype=np.float64)
    t1_all = np.asarray(t1_all, dtype=np.float64)
    nsnps_blk = np.asarray(nsnps_blk, dtype=np.float64)
    c_all = np.asarray(c_all, dtype=np.float64).ravel()

    Bp1, K, K2 = ld_sum_all.shape
    if K2 != K:
        raise ValueError("ld_sum_all must be (B+1, K, K)")
    if t1_all.shape != (Bp1, K):
        raise ValueError("t1_all must be (B+1, K)")
    if nsnps_blk.shape != (Bp1, K):
        raise ValueError("nsnps_blk must be (B+1, K)")
    if c_all.size != Bp1:
        raise ValueError("c_all must be (B+1,)")

    sN = np.sqrt(float(n1) * float(n2))
    I = np.eye(K, dtype=np.float64)

    # rhs_all: (B+1, K)
    rhs_all = (t1_all - nsnps_blk * c_all[:, None]) / sN

    # If any empty/invalid bins exist, do the safe slow loop (preserves semantics).
    badM = ~(np.isfinite(nsnps_blk) & (nsnps_blk > 0))
    if np.any(badM):
        gamma_all = np.full((Bp1, K), np.nan, dtype=np.float64)
        for b in range(Bp1):
            Mvec = nsnps_blk[b]
            good = ~badM[b]
            if not np.any(good):
                continue

            A = ld_sum_all[b][np.ix_(good, good)]
            rhs = rhs_all[b][good]

            tr = float(np.trace(A))
            lam = ridge_rel * (tr / A.shape[0] if np.isfinite(tr) and tr != 0.0 else 1.0)
            A_reg = A + lam * np.eye(A.shape[0], dtype=np.float64)

            try:
                g = np.linalg.solve(A_reg, rhs)
            except np.linalg.LinAlgError:
                g = np.linalg.lstsq(A_reg, rhs, rcond=None)[0]

            out = np.full(K, np.nan, dtype=np.float64)
            out[good] = Mvec[good] * g
            gamma_all[b] = out

        return gamma_all

    # ---- fast stacked solve (no empty bins) ----
    tr = np.trace(ld_sum_all, axis1=1, axis2=2)  # (B+1,)
    tr_eff = np.where(np.isfinite(tr) & (tr != 0.0), tr / K, 1.0)
    lam = ridge_rel * tr_eff  # (B+1,)

    A_reg = ld_sum_all + lam[:, None, None] * I[None, :, :]  # (B+1,K,K)
    g_all = np.linalg.solve(A_reg, rhs_all[..., None])[..., 0]  # (B+1,K)
    gamma_all = nsnps_blk * g_all  # (B+1,K)

    return gamma_all
