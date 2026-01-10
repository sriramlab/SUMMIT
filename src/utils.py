import numpy as np
import pandas as pd
import os
import re
import time
import datetime
import glob

def _replace_None(li: list):
    """
    replace the None elements in the list with zero
    """
    for i, val in enumerate(li):
        if val is None:
            li[i] = .0
    return li

def _partition_bin_non_overlapping(jn_values: np.ndarray, jn_annot: np.ndarray, nbins: int):
    """
    Partition the first array (a 1D np array) by the annotation (a 1D array). return a nested list.
    This function assumes that a SNP can belong to only one bin.
    """
    partitions = {i: [] for i in range(nbins)}
    for z, idx in zip(jn_values, jn_annot):
        partitions[idx.argmax()].append(z)
    snp_cnts = [len(partitions[i]) - sum(1 for s in partitions[i] if s is None) for i in range(nbins)]
    return [_replace_None(partitions[i]) for i in range(nbins)], snp_cnts


def _partition_bin_overlapping(jn_values: np.ndarray, jn_annot: np.ndarray, nbins: int):
    """
    Partition the 1D array `jn_values` into `nbins` lists using the 2D
    indicator/weight matrix `jn_annot` (shape: [num_snps, nbins]).
    A SNP belongs to bin b if jn_annot[i, b] != 0.
    """
    import numpy as np

    jn_values = np.asarray(jn_values)
    jn_annot  = np.asarray(jn_annot)

    # Handle single-bin edge case: allow 1D annot
    if jn_annot.ndim == 1:
        jn_annot = jn_annot.reshape(-1, 1)

    if jn_annot.shape[0] != jn_values.shape[0]:
        raise ValueError("jn_values and jn_annot must have the same number of rows (SNPs).")

    partitions = {i: [] for i in range(nbins)}

    # Vectorized selection per bin; treat any non-zero as membership
    for b in range(nbins):
        col = jn_annot[:, b]
        mask = (col != 0)  # works for bool or numeric (binary/continuous)
        partitions[b] = jn_values[mask].tolist()

    # Count SNPs per bin (you never append None, so no need to subtract)
    snp_cnts = [len(partitions[i]) for i in range(nbins)]

    # Preserve your existing return shape and None-handling helper
    return [_replace_None(partitions[i]) for i in range(nbins)], snp_cnts


def _calc_lsum(tr, n, m1, m2):
    '''
    Calculate the sum of the LD scores from the trace estimates
    '''
    return (tr - n)*(m1*m2)/pow(n,2)

def _calc_trace_from_ld(ldsum, n, m1, m2):
    '''
    Calculate the trace from the sum of the LD scores
    '''
    return ldsum*pow(n, 2)/(m1*m2) + n

def _calc_rg_trace_from_ld(ldsum, n1, n2, m1, m2):
    '''
    Calculate the trace from the sum of the LD scores for unconstrained rg calculation
    '''
    return ldsum*n1*n2/(m1*m2)

def _calc_rg_const_trace_from_ld(ldsum, n1, n2, m1, m2):
    '''
    Calculate the trace from the sum of the LD scores for constrained rg calculation
    '''
    return ldsum*n1*n2/(m1*m2)


# ----------------------- Batched vectorized versions ----------------------- #

def _calc_trace_from_ld_batch(ldsum, n, m1, m2, delta=None):
    """
    Batched version of _calc_trace_from_ld with broadcasting.
    Inputs
      ldsum : (..., K, K)
      n     : scalar or broadcastable to (..., 1, 1)  # sample size N
      m1    : (..., K, 1)  LOO bin counts for 'row' bin
      m2    : (..., 1, K)  LOO bin counts for 'col' bin
      delta : optional (K, K) block-level δ matrix.
              If provided, applies a finite-sample correction:
                  trace_corr = trace_gaussian - N * δ,
              broadcasted over jackknife rows.
    Returns
      trace : (..., K, K)
    Fills positions with zero denominator with n (your original fill).
    """
    ldsum = np.asarray(ldsum, dtype=np.float64)
    n     = np.asarray(n,     dtype=np.float64)
    m1    = np.asarray(m1,    dtype=np.float64)
    m2    = np.asarray(m2,    dtype=np.float64)

    denom = m1 * m2                         # (..., K, K)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = ldsum * (n ** 2) / denom + n  # broadcasted compute

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
        corr = n_scalar * delta            # (K, K)
        # Broadcast corr and apply only where denom is valid
        out = np.where(valid, out - corr, out)

    return out


def _calc_rg_trace_from_ld_batch(ldsum, n1, n2, m1, m2):
    """
    Batched version of _calc_rg_trace_from_ld with broadcasting.
    Inputs
      ldsum : (..., K, K)
      n1,n2 : scalars
      m1    : (..., K, 1)
      m2    : (..., 1, K)
    Returns
      rg_trace : (..., K, K)
    Fills positions with zero denominator with 0.0.
    """
    ldsum = np.asarray(ldsum, dtype=np.float64)
    n1 = float(n1)
    n2 = float(n2)
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)

    denom = m1 * m2
    with np.errstate(divide='ignore', invalid='ignore'):
        out = ldsum * (n1 * n2) / denom
    valid = (denom > 0) & np.isfinite(denom)
    out = np.where(valid, out, 0.0)
    return out


def estimate_offdiag_variances_from_jackknife(trace_KK):
    """
    Estimate per-pair (k,l) variances and covariance of the two off-diagonal
    trace estimators using jackknife blocks.

    Parameters
    ----------
    trace_KK : ndarray, shape (B+1, K, K)
        Jackknife + full-sample trace estimates for the KxK bin-binned matrix.
        Convention: rows 0..B-1 = LOO jackknife, row B = full-sample.

    Returns
    -------
    var1 : ndarray, shape (K, K)
        Estimated Var[T_{kl}^{(1)}] from entries (k,l) across jackknife blocks.
    var2 : ndarray, shape (K, K)
        Estimated Var[T_{kl}^{(2)}] from entries (l,k) across jackknife blocks.
    cov12 : ndarray, shape (K, K)
        Estimated Cov[T_{kl}^{(1)}, T_{kl}^{(2)}].
    w_opt : ndarray, shape (K, K)
        Optimal weights for combining the two off-diagonal estimates:
        T_sym_{kl} = w_opt[k,l] * T_{kl} + (1 - w_opt[k,l]) * T_{lk}.
        Diagonal entries are set to 0.5 by convention.
    """
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


def symmetrize_trace_with_jackknife(trace_KK, logger=None, verbose=False):
    """
    Symmetrize a (B+1, K, K) trace tensor using jackknife-based optimal weights.

    Parameters
    ----------
    trace_KK : ndarray, shape (B+1, K, K)
        Jackknife + full-sample trace matrices.
    logger : object with a `_log(str)` method, optional
        If provided, logs before/after asymmetry diagnostics.

    Returns
    -------
    sym : ndarray, shape (B+1, K, K)
        Symmetrized trace tensor (off-diagonals averaged with optimal weights).
    """
    trace_KK = np.asarray(trace_KK, dtype=np.float64)
    B_plus, K, K2 = trace_KK.shape
    if K != K2:
        raise ValueError("trace_KK last two dimensions must be equal (KxK)")

    B = B_plus - 1
    if B < 0:
        raise ValueError("trace_KK must have at least one row (full sample)")

    # Estimate variances & optimal weights
    _, _, _, w_opt = estimate_offdiag_variances_from_jackknife(trace_KK)

    sym = trace_KK.copy()

    if B == 0:
        # No jackknife; at least symmetrize the full-sample row via simple average
        full = sym[0]
        for k in range(K):
            for l in range(k + 1, K):
                v = 0.5 * (full[k, l] + full[l, k])
                full[k, l] = full[l, k] = v
        sym[0] = full
        return sym

    # Before-symmetry diagnostics (full row only)
    if logger is not None:
        full_before = trace_KK[B]
        asym_before = np.abs(full_before - full_before.T)
        tri = np.triu_indices(K, k=1)
        max_asym_before = float(asym_before[tri].max(initial=0.0))

    # Apply optimal weights to all jackknife rows and full row
    for k in range(K):
        for l in range(k + 1, K):
            w = w_opt[k, l]

            # Jackknife rows 0..B-1
            x = trace_KK[:B, k, l]
            y = trace_KK[:B, l, k]
            v_jk = w * x + (1.0 - w) * y
            sym[:B, k, l] = v_jk
            sym[:B, l, k] = v_jk

            # Full-sample row at index B
            x_full = trace_KK[B, k, l]
            y_full = trace_KK[B, l, k]
            v_full = w * x_full + (1.0 - w) * y_full
            sym[B, k, l] = v_full
            sym[B, l, k] = v_full

    # After-symmetry diagnostics
    if logger is not None:
        full_after = sym[B]
        asym_after = np.abs(full_after - full_after.T)
        tri = np.triu_indices(K, k=1)
        max_asym_after = float(asym_after[tri].max(initial=0.0))
        if verbose:
            logger._log(
                f"[Trace] Jackknife-based symmetrization: max off-diagonal asym "
                f"(before, after) = ({max_asym_before:.4e}, {max_asym_after:.4e})"
            )

    return sym



# ----------------------- Jackknife helpers ----------------------- #

def _calc_jn_subsample(alist):
    """
    From a list/array return an array of leave-one-out (jackknife) subsamples.
    The last element is the sum of all elements.
    """
    total = sum(alist)
    jn_sub = [total - val for val in alist]
    jn_sub.append(total)
    return np.array(jn_sub)


def _calc_jackknife_se(alist, axis=0, center='mean', nan_policy='propagate'):
    """
    Jackknife SE along `axis` for arrays shaped (B+1, ...), where the last slice
    is the full-sample estimate and the first B are LOO replicates.

    Matches the legacy (slow) implementation by default:
      center='full' and nan_policy='propagate'  ->  centers at full and propagates NaNs.

    Options:
      center: 'full' (legacy) or 'mean' (standard jackknife center at LOO mean)
      nan_policy: 'propagate' (legacy), or 'omit' (ignore NaNs per-coordinate)

    Returns: (est_full, se_jk) with est_full = last slice on `axis`.
    """
    a = np.asarray(alist)
    # full-sample estimate (last slice on axis)
    est_full = np.take(a, indices=-1, axis=axis)

    # LOO replicates = all but last
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(0, -1)
    reps = a[tuple(slicer)]                   # shape: (n, ...)

    # move jk axis to front
    reps = np.moveaxis(reps, axis, 0)         # (n, ...)

    # center choice
    if center == 'full':
        # broadcast est_full across replicate axis
        center_arr = est_full
    elif center == 'mean':
        center_arr = np.nanmean(reps, axis=0) if nan_policy == 'omit' else reps.mean(axis=0)
    else:
        raise ValueError("center must be 'full' or 'mean'")

    diffs = reps - center_arr  # (n, ...)

    if nan_policy == 'omit':
        finite = np.isfinite(diffs)
        m = finite.sum(axis=0)                         # effective replicates per coordinate
        diffs = np.where(finite, diffs, 0.0)
        ss = (diffs * diffs).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            var_jk = (np.maximum(m - 1, 0) / np.maximum(m, 1)) * ss
            se_jk = np.sqrt(var_jk)
            se_jk = np.where(m < 1, np.nan, se_jk)
    else:  # 'propagate' (legacy behavior)
        ss = (diffs * diffs).sum(axis=0)
        n = diffs.shape[0]
        se_jk = np.sqrt((n - 1) / n * ss)

    return est_full, se_jk


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

def _map_idx(snpid, npartition):
    '''
    create a mapping of SNP id -> idx
    '''
    mapping = {}
    partition = _partition(snpid, npartition)
    for idx, part in enumerate(partition):
        for snp in part:
            mapping[snp] = idx
    return mapping

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


    
def _parse_column(df, letters, min_index=3):
    '''
    select the column of a dataframe including certain letter & after certain column index
    '''
    matching_columns = [col for col in df.columns[min_index:] if any(letter in col for letter in letters)]
    
    if len(matching_columns) == 0:
        raise ValueError(f"No column containing any of the letters {letters} found starting from column {min_index}.")
    elif len(matching_columns) > 1:
        raise ValueError(f"Multiple columns containing the letters {letters} found: {matching_columns}. Expected only one.")
    else:
        return df[matching_columns[0]].values

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

def _solve_linear_equation(X, y, method='auto'):
    """
    Solve A x = b.
    Supports batched solves:
      X shape: (..., p, p)
      y shape: (..., p) or (..., p, k)
    For SPD matrices, Cholesky is fastest and most stable.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if method == 'lstsq':
        # Keep your original behavior if explicitly requested
        return np.linalg.lstsq(X, y, rcond=None)[0]

    # Try Cholesky (fast path)
    try:
        L = np.linalg.cholesky(X)               # (..., p, p)
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
    smooth_window=100,
    segment_ids=None,
    blk_idx=None,
    # NEW:
    positions_bp=None,
    smooth_bp_window=None,
    weight_floor=None,   # e.g. 1e-8 to avoid hard failure
):
    """
    Leave-one-(macro)-block-out per-SNP WLS for partitioned SUMCORE,
    with smoothed weights.

    Model:
        E[y_j] = c + sum_k beta_k * l2_{j,k},   where y_j = z1_j * z2_j

    Weights (variance proxy):
        ltot_j = sum_k l2_{j,k}
        ltot_smooth_j = smoothing(ltot_j) either:
          - SNP-count moving average within segment (default), or
          - physical bp-window within segment (if smooth_bp_window provided)
        w_j = 1 / ltot_smooth_j

    Jackknife blocks:
        If blk_idx is provided, it MUST match Trace exactly (recommended).
        Otherwise falls back to legacy M//nblks block construction.

    New smoothing options:
      - segment_ids: e.g., chromosome per SNP. Smoothing will NOT cross segment boundaries.
      - positions_bp + smooth_bp_window: physical window smoothing.
          smooth_bp_window is TOTAL window length in base-pairs (centered window, +/- smooth_bp_window/2).
    """
    import numpy as np

    l2_bins = np.asarray(l2_bins, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    nsnps_blk = np.asarray(nsnps_blk, dtype=np.float64)

    if l2_bins.ndim != 2:
        raise ValueError("l2_bins must be 2-D (M, K)")
    M, K = l2_bins.shape
    if y.size != M:
        raise ValueError(f"y must have shape (M,), got {y.shape}, expected M={M}")

    nblks = int(nblks)
    if nblks <= 0:
        raise ValueError("nblks must be positive")
    if nsnps_blk.shape != (nblks + 1, K):
        raise ValueError(f"nsnps_blk must be (nblks+1, K)=({nblks+1},{K}), got {nsnps_blk.shape}")

    n1 = float(n1); n2 = float(n2)
    if not (np.isfinite(n1) and np.isfinite(n2) and n1 > 0 and n2 > 0):
        raise ValueError("n1 and n2 must be positive finite")

    if not np.all(np.isfinite(y)):
        bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"Found {bad} non-finite entries in y; drop these SNPs upstream.")
    if not np.all(np.isfinite(l2_bins)):
        bad = int((~np.isfinite(l2_bins)).sum())
        raise ValueError(f"Found {bad} non-finite entries in l2_bins; drop/fix these SNPs upstream.")

    smooth_window = int(smooth_window) if smooth_window is not None else None
    if smooth_window is not None and smooth_window <= 0:
        raise ValueError("smooth_window must be positive or None")

    if smooth_bp_window is not None:
        smooth_bp_window = int(smooth_bp_window)
        if smooth_bp_window <= 0:
            raise ValueError("smooth_bp_window must be positive")

    # ----------------------------
    # Helpers: smoothing
    # ----------------------------
    def _moving_average_1d_count(x, win):
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if win is None or win <= 1 or n == 0:
            return x.copy()
        half = win // 2
        cs = np.empty(n + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(x, out=cs[1:])

        idx = np.arange(n, dtype=np.int64)
        start = np.maximum(idx - half, 0)
        end = np.minimum(idx + half + 1, n)
        sums = cs[end] - cs[start]
        lens = (end - start).astype(np.float64)
        return sums / lens

    def _moving_average_1d_bp(x, bp, win_bp):
        """
        Centered window smoothing by bp distance: includes SNPs within +/- win_bp/2.
        Requires bp to be nondecreasing in this slice.
        O(n) two-pointer with prefix sums.
        """
        x = np.asarray(x, dtype=np.float64)
        bp = np.asarray(bp, dtype=np.int64)
        n = x.size
        if n == 0:
            return x.copy()
        if n == 1:
            return x.copy()
        if np.any(bp[1:] < bp[:-1]):
            raise ValueError("positions_bp must be nondecreasing within each segment/run for bp-window smoothing.")

        half = win_bp // 2
        cs = np.empty(n + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(x, out=cs[1:])

        out = np.empty(n, dtype=np.float64)
        left = 0
        right = 0
        for i in range(n):
            # move left up until bp[i] - bp[left] <= half
            while left < n and (bp[i] - bp[left] > half):
                left += 1
            # move right up until bp[right] - bp[i] > half (right is exclusive)
            if right < i:
                right = i
            while right < n and (bp[right] - bp[i] <= half):
                right += 1
            s = cs[right] - cs[left]
            m = right - left
            out[i] = s / float(m) if m > 0 else x[i]
        return out

    def _iter_runs(seg):
        """Yield contiguous [start,end) runs where seg is constant."""
        n = seg.size
        if n == 0:
            return
        s = 0
        for i in range(1, n):
            if seg[i] != seg[i - 1]:
                yield s, i
                s = i
        yield s, n

    # ----------------------------
    # Compute smoothed total-LD for weights
    # ----------------------------
    ltot = l2_bins.sum(axis=1)

    # Choose smoothing mode
    use_bp = (smooth_bp_window is not None)
    if use_bp:
        if segment_ids is None or positions_bp is None:
            raise ValueError("bp-window smoothing requires segment_ids (e.g., chr) and positions_bp.")
        seg = np.asarray(segment_ids)
        bp = np.asarray(positions_bp, dtype=np.int64)
        if seg.shape[0] != M or bp.shape[0] != M:
            raise ValueError("segment_ids and positions_bp must both have length M.")
        ltot_smooth = np.empty(M, dtype=np.float64)
        for s, e in _iter_runs(seg):
            ltot_smooth[s:e] = _moving_average_1d_bp(ltot[s:e], bp[s:e], smooth_bp_window)
    else:
        # SNP-count smoothing (default)
        if segment_ids is None:
            ltot_smooth = _moving_average_1d_count(ltot, smooth_window)
        else:
            seg = np.asarray(segment_ids)
            if seg.shape[0] != M:
                raise ValueError(f"segment_ids must have length M={M}, got {seg.shape[0]}")
            ltot_smooth = np.empty(M, dtype=np.float64)
            for s, e in _iter_runs(seg):
                ltot_smooth[s:e] = _moving_average_1d_count(ltot[s:e], smooth_window)

    # Enforce positivity for weights
    bad = (~np.isfinite(ltot_smooth)) | (ltot_smooth <= 0.0)
    if np.any(bad):
        if weight_floor is None:
            nb = int(bad.sum())
            mn = float(np.nanmin(ltot_smooth))
            idx_bad = np.flatnonzero(bad)[:10]
            raise ValueError(
                f"Smoothed total LD (ltot_smooth) must be finite and > 0 for weights w=1/ltot_smooth.\n"
                f"Found {nb}/{M} SNPs with ltot_smooth <= 0 or non-finite (min={mn}).\n"
                f"Try: larger smooth_window, and/or smooth_bp_window with segment_ids+positions_bp.\n"
                f"First bad indices: {idx_bad.tolist()}"
            )
        else:
            eps = float(weight_floor)
            ltot_smooth = np.where(np.isfinite(ltot_smooth), ltot_smooth, eps)
            ltot_smooth = np.maximum(ltot_smooth, eps)

    w = 1.0 / ltot_smooth

    # ----------------------------
    # Per-SNP design matrix
    # ----------------------------
    X = np.empty((M, K + 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = l2_bins

    # ----------------------------
    # Macro-block jackknife (MUST match Trace)
    # ----------------------------
    if blk_idx is None:
        # legacy fallback
        blk_size = M // nblks
        if blk_size == 0:
            raise ValueError("Too many jackknife blocks (nblks > M).")
        blk_idx = np.repeat(np.arange(nblks, dtype=np.int64), blk_size)
        if blk_idx.size < M:
            blk_idx = np.concatenate([blk_idx, np.full(M - blk_idx.size, nblks - 1, dtype=np.int64)])
    else:
        blk_idx = np.asarray(blk_idx, dtype=np.int64).ravel()
        if blk_idx.size != M:
            raise ValueError(f"blk_idx must have length M={M}, got {blk_idx.size}")
        if blk_idx.min() < 0 or blk_idx.max() >= nblks:
            raise ValueError(f"blk_idx values must be in [0, nblks-1]=[0,{nblks-1}]")

    # ----------------------------
    # Normal equations totals
    # ----------------------------
    WX = w[:, None] * X
    SXX_tot = WX.T @ X
    SXY_tot = WX.T @ y

    # Per-block contributions via bincount
    SXX_blk = np.zeros((nblks, K + 1, K + 1), dtype=np.float64)
    SXY_blk = np.zeros((nblks, K + 1), dtype=np.float64)
    tmp = np.empty(M, dtype=np.float64)

    for c in range(K + 1):
        Xc = X[:, c]
        for d in range(c, K + 1):
            tmp[:] = w * Xc * X[:, d]
            S = np.bincount(blk_idx, weights=tmp, minlength=nblks).astype(np.float64, copy=False)
            SXX_blk[:, c, d] = S
            if d != c:
                SXX_blk[:, d, c] = S

        tmp[:] = w * Xc * y
        SXY_blk[:, c] = np.bincount(blk_idx, weights=tmp, minlength=nblks).astype(np.float64, copy=False)

    # LOO totals
    SXX = SXX_tot[None, :, :] - SXX_blk
    SXY = SXY_tot[None, :] - SXY_blk

    # Solve
    try:
        beta_j = np.linalg.solve(SXX, SXY[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta_j = np.vstack([np.linalg.lstsq(SXX[b], SXY[b], rcond=None)[0] for b in range(nblks)])

    try:
        beta_full = np.linalg.solve(SXX_tot, SXY_tot)
    except np.linalg.LinAlgError:
        beta_full = np.linalg.lstsq(SXX_tot, SXY_tot, rcond=None)[0]

    beta_all = np.vstack([beta_j, beta_full[None, :]])  # (B+1, K+1)

    # Scale slopes -> gamma
    scale = nsnps_blk / np.sqrt(n1 * n2)                # (B+1, K)
    gamma_all = scale * beta_all[:, 1:]                 # (B+1, K)
    c_all = beta_all[:, 0]                              # (B+1,)
    return gamma_all, c_all

def compute_t1_all_jn(annot, y, blk_idx, nblks):
    """
    Compute T1_all (B+1, K) where:
      T1_all[b,k] = sum_{j not in block b} A[j,k] * y[j]   for b=0..B-1
      T1_all[B,k] = sum_{j} A[j,k] * y[j]                 (full)

    annot: (M,K), y: (M,), blk_idx: (M,)
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
    if blk_idx.min() < 0 or blk_idx.max() >= nblks:
        raise ValueError("blk_idx out of range")

    # full T1
    T1_full = A.T @ y  # (K,)

    # per-block contribution: sum_{j in block b} A[j,k]*y[j]
    T1_blk = np.zeros((nblks, K), dtype=np.float64)
    tmp = np.empty(M, dtype=np.float64)
    for k in range(K):
        tmp[:] = A[:, k] * y
        T1_blk[:, k] = np.bincount(blk_idx, weights=tmp, minlength=nblks).astype(np.float64, copy=False)

    # LOO = full - blk, plus full row at end
    T1_all = np.empty((nblks + 1, K), dtype=np.float64)
    T1_all[:nblks] = T1_full[None, :] - T1_blk
    T1_all[nblks] = T1_full
    return T1_all


def solve_score_gamma_from_intercept_jn(
    ld_sum_all,    # (B+1, K, K) = A^T L (LOO + full)
    t1_all,        # (B+1, K)    = A^T y (LOO + full)
    nsnps_blk,     # (B+1, K)    = M_k (LOO + full)
    c_all,         # (B+1,)      = intercept estimates
    n1, n2,
    ridge_rel=1e-12,
):
    """
    Two-step SUMCORE/SCORE plug-in solve for partitioned gamma using intercept c_all.

    For each replicate b:
      rhs = (t1_all[b] - nsnps_blk[b]*c_all[b]) / sqrt(n1*n2)
      Solve: ld_sum_all[b] @ g = rhs,  where g = gamma / M (elementwise over columns)
      Then:  gamma = M * g  (elementwise)

    Returns:
      gamma_all: (B+1, K)
    """
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

    n1 = float(n1); n2 = float(n2)
    sN = np.sqrt(n1 * n2)

    gamma_all = np.full((Bp1, K), np.nan, dtype=np.float64)

    I = np.eye(K, dtype=np.float64)

    for b in range(Bp1):
        Mvec = nsnps_blk[b].copy()  # (K,)
        # rows with M=0 are meaningless; keep them NaN
        badM = ~(np.isfinite(Mvec) & (Mvec > 0))
        if np.all(badM):
            continue

        rhs = (t1_all[b] - Mvec * c_all[b]) / sN
        rhs[badM] = np.nan

        A = ld_sum_all[b].copy()

        # ridge for stability (scaled to matrix magnitude)
        tr = float(np.trace(A))
        lam = ridge_rel * (tr / K if np.isfinite(tr) and tr != 0.0 else 1.0)
        A_reg = A + lam * I

        # solve for g = gamma / M (column-wise scaling happens later)
        try:
            g = np.linalg.solve(A_reg, rhs)
        except np.linalg.LinAlgError:
            g = np.linalg.lstsq(A_reg, rhs, rcond=None)[0]

        gamma = Mvec * g
        gamma[badM] = np.nan
        gamma_all[b] = gamma

    return gamma_all