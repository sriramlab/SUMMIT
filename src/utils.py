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
    Partition the first array (a 1D np array) by the annotation (a 1D array). return a nested list.
    This function assumes that a SNP can belong to multiple bins.
    """
    partitions = {i: [] for i in range(nbins)}
    for bin in range(nbins):
        partitions[bin] = [jn_values[snp] for snp in range(len(jn_values)) if jn_annot[row, bin]]
    snp_cnts = [len(partitions[i]) - sum(1 for s in partitions[i] if s is None) for i in range(nbins)]
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

def _calc_jn_subsample(alist):
    '''
    From a list/array return an array of leave-one-out (jackknife) subsamples
    the last element is the sum of all elements
    '''
    total = sum(alist)
    jn_sub = [total - val for val in alist]
    jn_sub.append(total)
    return np.array(jn_sub)

def _calc_jackknife_se(alist):
    '''
    alist should have shape (nblks+1,) where the last value is the total estimate
    '''
    n_total = alist.shape[0]
    nblks   = n_total - 1
    leave_out = alist[:nblks, ...]
    est_full  = alist[-1, ...]
    sum_sq = np.sum((leave_out - est_full)**2, axis=0)
    se_jk = np.sqrt((nblks - 1) / nblks * sum_sq)

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

def _solve_linear_equation(X, y, method='lstsq'):
    '''
    Solve system of linear equations (either least square or QR)
    '''
    if (method == 'lstsq'):
        return np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        Q, R = scipy.linalg.qr(X)
        return scipy.linalg.solve_triangular(R, np.dot(Q.T, y))

import numpy as np

def bivariate_regression_partitioned_jn(l2_bins, y, w, nblks,
                                        n1, n2, nsnps_blk):
    """
    Leave-one-block-out jack-knife WLS for partitioned SCORE / SUMCORE
    **using pre-computed SNP counts per replicate**.

    Parameters
    ----------
    l2_bins : (M, K) ndarray[float]
        LD scores for K annotation bins.  Non-member SNPs must carry 0.0.
    y       : (M,) ndarray[float]
        SNP-wise product z1 * z2.
    w       : (M,) ndarray[float]
        Positive, finite WLS weights (≈ 1 / l2_total).
    nblks   : int
        Number of jack-knife blocks.
    n1, n2  : float
        GWAS sample sizes.
    nsnps_blk : (nblks+1, K) ndarray[int]
        SNP counts **after** leaving out block j (rows 0…nblks-1)
        and for the full data set (last row).  That is,
            nsnps_blk[j, k] = #SNPs of bin k present in replicate j.

    Returns
    -------
    gamma_all : (nblks+1, K) ndarray[float]
        γ̂ for every replicate and annotation bin.
    c_all     : (nblks+1,) ndarray[float]
        Intercept estimates (last element = full data).
    """
    # ---------- validation --------------------------------------------------
    l2_bins   = np.asarray(l2_bins, dtype=float)
    y         = np.asarray(y,       dtype=float).ravel()
    w         = np.asarray(w,       dtype=float).ravel()
    nsnps_blk = np.asarray(nsnps_blk, dtype=int)

    if l2_bins.ndim != 2:
        raise ValueError("l2_bins must be 2-D (M, K)")
    M, K = l2_bins.shape
    if y.size != M or w.size != M:
        raise ValueError("Shapes of l2_bins, y, w are inconsistent")
    if nsnps_blk.shape != (nblks + 1, K):
        raise ValueError("nsnps_blk must be (nblks+1, K)")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Weights must be positive and finite")

    # ---------- design matrix  X = [1 | l2_bin1 … l2_binK] ------------------
    X = np.empty((M, K + 1), dtype=float)
    X[:, 0]  = 1.0
    X[:, 1:] = l2_bins

    # ---------- jack-knife block index -------------------------------------
    blk_idx = np.repeat(np.arange(nblks), M // nblks)
    blk_idx = np.append(blk_idx,
                        np.full(M - blk_idx.size, nblks - 1))

    # ---------- totals over all SNPs ---------------------------------------
    WX       = w[:, None] * X                # (M, K+1)
    SXX_tot  = WX.T @ X                      # (K+1, K+1)
    SXY_tot  = WX.T @ y                      # (K+1,)

    # ---------- per-block contributions via bincount -----------------------
    SXX_blk = np.zeros((nblks, K + 1, K + 1), dtype=float)
    SXY_blk = np.zeros((nblks, K + 1),        dtype=float)
    tmp = np.empty(M, dtype=float)

    for c in range(K + 1):
        for d in range(c, K + 1):            # exploit symmetry
            tmp[:] = w * X[:, c] * X[:, d]
            S = np.bincount(blk_idx, tmp, minlength=nblks)
            SXX_blk[:, c, d] = S
            if d != c:
                SXX_blk[:, d, c] = S
        tmp[:] = w * X[:, c] * y
        SXY_blk[:, c] = np.bincount(blk_idx, tmp, minlength=nblks)

    # ---------- leave-one-block-out totals ---------------------------------
    SXX = SXX_tot[None, :, :] - SXX_blk       # (nblks, K+1, K+1)
    SXY = SXY_tot[None, :]   - SXY_blk        # (nblks, K+1)

    # ---------- solve  (Xᵀ W X) β = Xᵀ W y  -------------------------------
    beta_j = np.linalg.solve(SXX, SXY[..., None])[..., 0]  # (nblks, K+1)
    beta_full = np.linalg.solve(SXX_tot, SXY_tot)
    beta_all  = np.vstack([beta_j, beta_full[None, :]])    # (nblks+1, K+1)

    # ---------- scale to γ̂ -------------------------------------------------
    scale      = nsnps_blk / np.sqrt(n1 * n2)              # (nblks+1, K)
    gamma_all  = scale * beta_all[:, 1:]                   # drop intercept
    c_all      = beta_all[:, 0]

    return gamma_all, c_all
