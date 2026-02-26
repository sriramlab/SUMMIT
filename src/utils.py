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
    Partition the 1D array `jn_values` into `nbins` lists using an overlapping /
    continuous annotation matrix `jn_annot` (shape: [num_snps, nbins]).

    Continuous/overlap-correct semantics:
      - A SNP contributes to bin b with weight w = jn_annot[i,b].
      - Returned partition list contains weighted values: w * jn_values[i]
        (and includes only SNPs with w != 0).
      - Returned "snp_cnts" is actually BIN MASS: sum_i w_i (not nnz count).

    This matches the general pattern used in SUMRHE continuous refactors:
      sums use A-weights (A^T y) and "counts" are sum(A).

    Returns:
      partitions: list of length nbins; each is list of weighted values (None -> 0)
      snp_cnts:  list of length nbins; each is sum of weights in that bin
    """
    jn_values = np.asarray(jn_values)
    jn_annot  = np.asarray(jn_annot)

    if jn_values.ndim != 1:
        jn_values = jn_values.ravel()

    # Handle single-bin edge case: allow 1D annot
    if jn_annot.ndim == 1:
        jn_annot = jn_annot.reshape(-1, 1)

    if jn_annot.ndim != 2:
        raise ValueError("jn_annot must be 2D (M, nbins).")
    M, K = jn_annot.shape
    if K != nbins:
        raise ValueError(f"jn_annot has K={K} columns but nbins={nbins}.")
    if jn_values.shape[0] != M:
        raise ValueError("jn_values and jn_annot must have the same number of rows (SNPs).")

    if not np.all(np.isfinite(jn_annot)):
        raise ValueError("jn_annot contains non-finite values.")

    partitions = {b: [] for b in range(nbins)}
    snp_mass = np.zeros(nbins, dtype=np.float64)

    # Weighted partition per bin
    for b in range(nbins):
        w = jn_annot[:, b].astype(np.float64, copy=False)
        mask = (w != 0.0)
        if np.any(mask):
            partitions[b] = (jn_values[mask] * w[mask]).tolist()
            snp_mass[b] = float(w[mask].sum())
        else:
            partitions[b] = []
            snp_mass[b] = 0.0

    return [_replace_None(partitions[i]) for i in range(nbins)], snp_mass.tolist()



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


def _calc_jackknife_se(alist, axis=0, center="mean", nan_policy="propagate"):
    """
    Jackknife SE along `axis` for arrays shaped (B+1, ...), where the last slice
    is the full-sample estimate and the first B are LOO replicates.

    Options:
      center:
        - 'full'   : center at the full-sample estimate (legacy behavior)
        - 'mean'   : center at the mean of LOO replicates (standard jackknife)
        - 'median' : center at the median of LOO replicates (robust-ish, often more conservative)
      nan_policy:
        - 'propagate' : propagate NaNs (legacy)
        - 'omit'      : ignore NaNs per-coordinate (uses nanmean/nanmedian and effective m)

    Returns: (est_full, se_jk)
      est_full = last slice on `axis`.
    """
    a = np.asarray(alist)

    # full-sample estimate (last slice on axis)
    est_full = np.take(a, indices=-1, axis=axis)

    # LOO replicates = all but last
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(0, -1)
    reps = a[tuple(slicer)]  # shape: (n, ...)

    # move jk axis to front -> (n, ...)
    reps = np.moveaxis(reps, axis, 0)
    n = reps.shape[0]

    # center choice
    if center == "full":
        center_arr = est_full
    elif center == "mean":
        center_arr = np.nanmean(reps, axis=0) if nan_policy == "omit" else reps.mean(axis=0)
    elif center == "median":
        center_arr = np.nanmedian(reps, axis=0) if nan_policy == "omit" else np.median(reps, axis=0)
    else:
        raise ValueError("center must be 'full', 'mean', or 'median'")

    diffs = reps - center_arr  # (n, ...)

    if nan_policy == "omit":
        finite = np.isfinite(diffs)
        m = finite.sum(axis=0)  # effective replicates per coordinate
        diffs = np.where(finite, diffs, 0.0)
        ss = (diffs * diffs).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_jk = (np.maximum(m - 1, 0) / np.maximum(m, 1)) * ss
            se_jk = np.sqrt(var_jk)
            se_jk = np.where(m < 1, np.nan, se_jk)
    else:  # 'propagate'
        ss = (diffs * diffs).sum(axis=0)
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

    n1 = float(n1); n2 = float(n2)
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
            blk_idx = np.concatenate([blk_idx, np.full(M - blk_idx.size, nblks - 1, dtype=np.int64)])
    else:
        blk_idx = np.asarray(blk_idx, dtype=np.int64).ravel()
        if blk_idx.size != M:
            raise ValueError(f"blk_idx must have length M={M}, got {blk_idx.size}")
        if blk_idx.min() < 0 or blk_idx.max() >= nblks:
            raise ValueError(f"blk_idx values must be in [0, nblks-1]=[0,{nblks-1}]")

    # Derive contiguous block bounds once (assumes piecewise-constant blk_idx)
    change = np.flatnonzero(blk_idx[1:] != blk_idx[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change, [M]))
    run_blk = blk_idx[starts]

    blk_starts = np.zeros(nblks, dtype=np.int64)
    blk_ends   = np.zeros(nblks, dtype=np.int64)
    for r, b in enumerate(run_blk):
        blk_starts[int(b)] = int(starts[r])
        blk_ends[int(b)]   = int(ends[r])

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
                f"Total LD (ltot) must be finite and > 0 for weights w=1/ltot on KEPT SNPs.\n"
                f"Found {nb}/{M} kept SNPs with ltot <= 0 or non-finite (min={mn}).\n"
                f"Pass weight_floor to clamp.\n"
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
        raise ValueError("After chisq filtering + weighting, no SNPs remain for intercept regression (sum(w)=0).")
    # not a strict requirement, but helps catch pathological filtering
    if int((w > 0).sum()) < (K + 5):
        raise ValueError(
            f"Too few SNPs after chisq filtering for stable regression: kept={(w>0).sum()} < K+5={K+5}. "
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
        s = int(blk_starts[b]); e = int(blk_ends[b])
        if e <= s:
            continue
        Lb = L[s:e, :]
        wb = w[s:e]
        if wb.sum() <= 0:
            continue
        wyb = wy[s:e]

        S00 = float(wb.sum())
        S0  = Lb.T @ wb
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
    SXX = SXX_tot[None, :, :] - SXX_blk          # (B, p, p)
    SXY = SXY_tot[None, :]    - SXY_blk          # (B, p)

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

    beta_all = np.vstack([beta_j, beta_full[None, :]])         # (B+1, p)

    # Scale slopes -> gamma
    scale = nsnps_blk / sN                                      # (B+1, K)
    gamma_all = scale * beta_all[:, 1:]                         # (B+1, K)
    c_all = beta_all[:, 0]                                      # (B+1,)
    return gamma_all, c_all

def compute_t1_all_jn(annot, y, blk_idx, nblks):
    """
    Same math, faster:
      T1_full = A^T y
      T1_blk[b] = (A_b)^T y_b   for contiguous blocks
      T1_LOO = full - blk
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

    # Full: A^T y (BLAS)
    T1_full = A.T @ y  # (K,)

    # Derive contiguous block bounds once
    change = np.flatnonzero(blk_idx[1:] != blk_idx[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change, [M]))
    run_blk = blk_idx[starts]

    blk_starts = np.zeros(nblks, dtype=np.int64)
    blk_ends   = np.zeros(nblks, dtype=np.int64)
    for b in range(nblks):
        blk_starts[b] = 0
        blk_ends[b] = 0
    for r, b in enumerate(run_blk):
        blk_starts[b] = starts[r]
        blk_ends[b] = ends[r]

    T1_blk = np.zeros((nblks, K), dtype=np.float64)
    for b in range(nblks):
        s = int(blk_starts[b]); e = int(blk_ends[b])
        if e <= s:
            continue
        T1_blk[b] = A[s:e, :].T @ y[s:e]   # gemv

    T1_all = np.empty((nblks + 1, K), dtype=np.float64)
    T1_all[:nblks] = T1_full[None, :] - T1_blk
    T1_all[nblks] = T1_full
    return T1_all


def solve_score_gamma_from_intercept_jn(
    ld_sum_all,
    t1_all,
    nsnps_blk,
    c_all,
    n1, n2,
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
    tr = np.trace(ld_sum_all, axis1=1, axis2=2)                  # (B+1,)
    tr_eff = np.where(np.isfinite(tr) & (tr != 0.0), tr / K, 1.0)
    lam = ridge_rel * tr_eff                                     # (B+1,)
    A_reg = ld_sum_all + lam[:, None, None] * I[None, :, :]      # (B+1,K,K)

    g_all = np.linalg.solve(A_reg, rhs_all[..., None])[..., 0]   # (B+1,K)
    gamma_all = nsnps_blk * g_all                                 # (B+1,K)
    return gamma_all