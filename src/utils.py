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

def _bivariate_regression_jn(l2, y, w, nblks, n1, n2, nsnps):
    """
    Jackknife WLS for y_j = beta * l2_j + c + e_j with arbitrary weights w_j.
    Returns gamma estimates such that the *last* element is the point estimate
    (using all SNPs) and the first nblks elements are leave-one-block-out jackknife.

    Parameters
    ----------
    l2    : array_like, shape (M,)
        LD scores.
    y     : array_like, shape (M,)
        Response (e.g., z1*z2).
    w     : array_like, shape (M,)
        Weights for WLS.
    nblks : int
        Number of jackknife blocks.
    n1, n2: float
        Sample sizes for scaling gamma.
    nsnps : array_like, shape (nblks+1,)
        Number of SNPs included in each subsample.  Last entry corresponds to full data.

    Returns
    -------
    gamma_all : ndarray, shape (nblks+1,)
        Gamma estimates: [gamma_j1,...,gamma_jn, gamma_full].
    c_all     : ndarray, shape (nblks+1,)
        Intercept estimates: [c_j1,...,c_jn, c_full].
    """
    l2 = np.squeeze(np.asarray(l2))
    y = np.squeeze(np.asarray(y))
    w = np.squeeze(np.asarray(w))
    nsnps = np.squeeze(np.asarray(nsnps))
    if l2.ndim != 1 or y.ndim != 1 or w.ndim != 1:
        raise ValueError("l2, y, and w must be 1D arrays")
    M = l2.shape[0]
    if y.shape[0] != M or w.shape[0] != M:
        raise ValueError("l2, y, and w must have the same length")
    if nsnps.ndim != 1 or nsnps.shape[0] != nblks + 1:
        raise ValueError("nsnps must have length nblks+1")

    # full-data normal-equation totals
    A00_tot = np.dot(w, l2 * l2)
    A01_tot = np.dot(w, l2)
    A11_tot = np.sum(w)
    b0_tot = np.dot(w * l2, y)
    b1_tot = np.dot(w, y)

    # assign SNPs to blocks
    blk_size = M // nblks
    blk_idx = np.empty(M, dtype=int)
    for i in range(nblks - 1):
        start = i * blk_size
        blk_idx[start:start + blk_size] = i
    blk_idx[(nblks - 1) * blk_size:] = nblks - 1

    # per-block contributions
    A00_blk = np.bincount(blk_idx, weights=w * l2 * l2, minlength=nblks)
    A01_blk = np.bincount(blk_idx, weights=w * l2, minlength=nblks)
    A11_blk = np.bincount(blk_idx, weights=w, minlength=nblks)
    b0_blk = np.bincount(blk_idx, weights=w * l2 * y, minlength=nblks)
    b1_blk = np.bincount(blk_idx, weights=w * y, minlength=nblks)

    # leave-one-block-out totals
    A00 = A00_tot - A00_blk
    A01 = A01_tot - A01_blk
    A11 = A11_tot - A11_blk
    b0 = b0_tot - b0_blk
    b1 = b1_tot - b1_blk

    # jackknife estimates
    denom = A00 * A11 - A01**2
    beta_j = (b0 * A11 - A01 * b1) / denom
    c_j = (A00 * b1 - b0 * A01) / denom

    # full-data estimate
    denom0 = A00_tot * A11_tot - A01_tot**2
    beta0 = (b0_tot * A11_tot - A01_tot * b1_tot) / denom0
    c0 = (A00_tot * b1_tot - b0_tot * A01_tot) / denom0

    # assemble: jackknife estimates first, then full-data
    beta_all = np.concatenate((beta_j, [beta0]))
    c_all = np.concatenate((c_j, [c0]))

    # recover gamma: scale by nsnps (aligned so last nsnps is full data)
    gamma_all = (nsnps / np.sqrt(n1 * n2)) * beta_all
    return gamma_all, c_all
