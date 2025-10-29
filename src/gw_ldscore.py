"""
Stochastically estimate (partitioned) genome-wide LD scores. Some part of the code is modified from Eric Liu's script
"""
import utils
import numpy as np
import pandas as pd
from bed_reader import open_bed
import multiprocessing as mp
from tqdm import tqdm
from scipy.linalg import blas as fblas
import sys
import gc
from multiprocessing import shared_memory
import atexit
import os
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
_THREAD_LOCAL = threading.local()

os.environ.setdefault("MALLOC_ARENA_MAX", "2")           # limit per-process arenas
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")  # bytes; encourage trims
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")  # bytes; mmap large blocks

def limit_blas_threads(n: int = 4):
    """
    Cap BLAS/OpenMP thread teams so mp.Pool workers don't oversubscribe the CPU.
    Call this BEFORE spawning any Pools.
    """
    import os
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["BLIS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        threadpool_limits(limits=n)
    except Exception:
        pass


def _bytes_human(n):
    if n is None: return "n/a"
    if n < 1024: return f"{n} B"
    for unit in ["KB","MB","GB","TB","PB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:,.2f} {unit}"
    return f"{n:,.2f} EB"

def _rss_snapshot(label, logger=None, include_children=True):
    import os, psutil
    rss = pss = None
    try:
        pid = os.getpid()
        with open(f"/proc/{pid}/smaps_rollup", "r") as f:
            for line in f:
                if line.startswith("Pss:"):
                    # kB → bytes
                    pss = int(line.split()[1]) * 1024
                elif line.startswith("Rss:"):
                    rss = int(line.split()[1]) * 1024
    except Exception:
        try:
            p = psutil.Process()
            rss = p.memory_info().rss
        except Exception:
            pass

    msg = f"[mem] {label}: RSS={_bytes_human(rss)}; PSS≈{_bytes_human(pss)}"
    if logger:
        try: logger._log(msg)
        except Exception: print(msg, file=sys.stderr, flush=True)
    else:
        print(msg, file=sys.stderr, flush=True)


def _resvar_worker_thread(span,
                          bed_prefix: str,
                          row_sel,
                          dtype: np.dtype,
                          ddof: int,
                          C: np.ndarray | None,
                          R: np.ndarray | None,
                          N_eff: float,
                          eps: float):
    """
    Worker for Var(M x_m) on SNP columns [s:e], using threads
    """
    s, e = span

    # Thread-local bed handle (avoid cross-thread sharing of a single handle)
    G = getattr(_THREAD_LOCAL, "G", None)
    Gid = getattr(_THREAD_LOCAL, "bed_prefix", None)
    if (G is None) or (Gid != bed_prefix):
        G = open_bed(bed_prefix + ".bed")
        _THREAD_LOCAL.G = G
        _THREAD_LOCAL.bed_prefix = bed_prefix

    # Row selection
    rows = row_sel if row_sel is not None else slice(None)

    # 1) Read & standardize in raw space
    geno = G.read(index=np.s_[rows, s:e], dtype=dtype)          # (N × L)
    means = np.nanmean(geno, axis=0, dtype=dtype)
    stds  = np.nanstd( geno, axis=0, dtype=dtype, ddof=ddof)
    stds[stds == 0] = 1.0
    np.subtract(geno, means, out=geno)
    np.divide(  geno, stds,  out=geno)
    np.nan_to_num(geno, copy=False)
    geno = np.asfortranarray(geno, dtype=dtype)

    # 2) Project: Y = (I - C R) * geno  (C: N×p with orthonormal cols; R = C^T in same dtype)
    if C is not None and R is not None:
        tmp = R @ geno              # (p × L)
        Y   = geno - (C @ tmp)      # (N × L)
        del tmp
    else:
        Y = geno

    # 3) Variance with denominator (N_eff - 1), all in dtype
    resvar = np.sum(Y * Y, axis=0, dtype=dtype) / float(N_eff - 1)
    inv_sqrt_resvar = (1.0 / np.sqrt(np.maximum(resvar, eps))).astype(dtype, copy=False)
    return (s, e, inv_sqrt_resvar)

# -------------------- shared-memory worker globals --------------------
_g_Xz2d = None           # (N × V*B), holds either Xz or M·Xz depending on phase
_g_meansq = None         # (M × B)
_g_xz_locks = None
_g_shm_xz = None
_g_shm_ms = None

def _worker_cleanup():
    global _g_Xz2d, _g_meansq, _g_shm_xz, _g_shm_ms
    _g_Xz2d = None
    _g_meansq = None
    try:
        if _g_shm_xz is not None:
            _g_shm_xz.close()
    except Exception:
        pass
    try:
        if _g_shm_ms is not None:
            _g_shm_ms.close()
    except Exception:
        pass
    _g_shm_xz = None
    _g_shm_ms = None

def _init_shared(xz_name, xz_shape2d, meansq_name, meansq_shape, dtype_str, xz_locks):
    import numpy as _np
    from multiprocessing import shared_memory as _sm
    global _g_Xz2d, _g_meansq, _g_xz_locks, _g_shm_xz, _g_shm_ms

    dt = _np.dtype(dtype_str)

    if xz_name is not None:
        _g_shm_xz = _sm.SharedMemory(name=xz_name)
        nrows, ncols = xz_shape2d
        _g_Xz2d = _np.frombuffer(_g_shm_xz.buf, dtype=dt, count=nrows*ncols)\
                  .reshape((nrows, ncols), order='F')
    else:
        _g_shm_xz = None
        _g_Xz2d = None

    if meansq_name is not None:
        _g_shm_ms = _sm.SharedMemory(name=meansq_name)
        M, B = meansq_shape
        _g_meansq = _np.ndarray((M, B), dtype=dt, buffer=_g_shm_ms.buf)
    else:
        _g_shm_ms = None
        _g_meansq = None

    _g_xz_locks = xz_locks
    atexit.register(_worker_cleanup)

# -------------------- covariate reader → orthonormal Q --------------------
def read_cov(
    cov_filename: str,
    fam_filename: str,
    std: bool = True,
    cov_impute_method: str = "ignore",   # drop rows with any NA
    one_hot_conversion: bool = False,
    categorical_threshold: int = 100,
    logger=None,
    verbose=False,
    sample_idx=None,
    ddof = 1
):
    fam = pd.read_csv(fam_filename, sep=r'\s+', header=None, usecols=[0,1], names=['FID','IID'])
    cov = pd.read_csv(cov_filename, sep=r'\s+')

    merged = fam.merge(cov, on=['FID','IID'], how='left', indicator=True)
    n_missing_in_cov = (merged['_merge'] != 'both').sum()
    if n_missing_in_cov:
        raise ValueError(f"{n_missing_in_cov} .fam samples not found in covariate file (FID/IID mismatch).")
    merged.drop(columns=['_merge'], inplace=True)
    
    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx, dtype=int)
        merged = merged.iloc[sample_idx].reset_index(drop=True)

    df = merged.drop(columns=['FID','IID']).copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if one_hot_conversion:
        small = [c for c in df.columns if df[c].nunique() <= categorical_threshold]
        if small:
            df = pd.get_dummies(df, columns=small, drop_first=True)

    if cov_impute_method == "ignore":
        keep_mask = ~df.isna().any(axis=1)
        dropped = (~keep_mask).sum()
        if logger: logger._log(f"Dropping {dropped} samples due to missing covariates.")
        df = df.loc[keep_mask].reset_index(drop=True)
    else:
        df = df.apply(lambda s: s.fillna(s.mean()), axis=0)
        keep_mask = np.ones(len(df), dtype=bool)

    zvc = df.std(ddof=0) == 0
    if zvc.any():
        drop_cols = zvc.index[zvc].tolist()
        if logger: logger._log(f"Dropping {len(drop_cols)} constant covariates: {drop_cols[:10]}{'...' if len(drop_cols)>10 else ''}")
        df.drop(columns=drop_cols, inplace=True)

    if std and not df.empty:
        df = (df - df.mean()) / df.std(ddof=ddof)
        bad_cols = [c for c in df.columns if df[c].isna().all()]
        if bad_cols:
            if logger: logger._log(f"Dropping malformed covariate columns after standardization: {bad_cols}")
            df.drop(columns=bad_cols, inplace=True)

    if df.empty:
        raise ValueError("After cleaning, no usable covariates remain.")

    C64 = df.to_numpy(dtype=np.float64)
    Q, _ = np.linalg.qr(C64, mode='reduced')     # Q: (N_kept × p_eff)
    C = np.asfortranarray(Q)
    R = np.asfortranarray(Q.T)

    km = np.flatnonzero(keep_mask.values) if isinstance(keep_mask, pd.Series) else np.flatnonzero(keep_mask)
    if sample_idx is not None:
        keep_idx_global = np.asarray(sample_idx, dtype=int)[km]
    else:
        keep_idx_global = km
        
    if logger:
        logger._log(f"Read {cov_filename}: kept {C.shape[0]} samples, {C.shape[1]} effective covariates. "
                    f"C shape={C.shape}, R shape=({R.shape[0]},{R.shape[1]}).")

    return C, R, keep_idx_global

# -------------------- main class --------------------
class GenomewideLDScore:
    def __init__(self,
                bed_path,
                annot_path,
                out_path,
                log,
                rand_dist,
                covar_path=None,
                num_vecs=10,
                num_workers=4,
                step_size=1000,
                seed=None,
                verbose=False,
                dtype='float32',
                num_threads: int = 1,
                eps_var: float = 1e-10,
                rand_samp=None, # float in (0,1] or int in [100, N]
                ddof = 1):
        # Cap BLAS threads before any Pools spawn
        self.num_threads = int(num_threads)
        try:
            limit_blas_threads(self.num_threads)
        except NameError:
            pass  # if helper isn't defined here

        self.eps_var = float(eps_var)

        self.G = open_bed(bed_path + ".bed")
        self.nsamp, self.nsnps = self.G.shape
        self.nvecs = num_vecs
        self.nworkers = num_workers
        self.step_size = step_size
        self.log = log
        self.verbose = verbose
        self.dtype = np.float32 if dtype in (np.float32, 'float32', 'f4') else np.float64
        self.rand_dist = rand_dist
        self.root_seed = seed
        rng = np.random.default_rng(self.root_seed)
        self.ddof = ddof
        
        self.start_time = utils._get_time()
        self.log._log("Genome-wide LD score calculation started at: "+utils._get_timestr(self.start_time))
        
        # -------- resolve random subsample of individuals --------
        base_idx = np.arange(self.nsamp, dtype=int)
        sel_idx = None
        if rand_samp is not None:
            if isinstance(rand_samp, (float, np.floating)):
                if not (0.0 < rand_samp <= 1.0):
                    raise ValueError("--rand-samp float must be in (0,1].")
                k = int(np.floor(rand_samp * self.nsamp))
                k = max(1, min(k, self.nsamp))
            else:
                k = int(rand_samp)
                if not (100 <= k <= self.nsamp):
                    raise ValueError("--rand-samp int must be in [100, N].")
            sel_idx = np.sort(rng.choice(base_idx, size=k, replace=False))
            self.log._log(f"Randomly subsampling individuals: {k}/{self.nsamp} ({k/self.nsamp:.1%})")

        # read .bim and annotation
        self._read_bim(bed_path + ".bim")
        if annot_path is not None:
            self._read_annot(annot_path)
        else:
            self._read_annot(None)

        # covariates → orthonormal Q (C) and Q^T (cov_R); drop NA rows
        if covar_path is not None:
            fam_file = bed_path + ".fam"
            C, R, keep_idx_global = read_cov(
                cov_filename=covar_path,
                fam_filename=fam_file,
                std=True,
                cov_impute_method="ignore",
                one_hot_conversion=False,
                categorical_threshold=100,
                logger=self.log,
                verbose=self.verbose,
                sample_idx=sel_idx if sel_idx is not None else None,
                ddof = self.ddof
            )
            # Final selected rows are those covariate-kept (already global indices)
            self.row_sel = np.asarray(keep_idx_global, dtype=int)
            self.C = np.asarray(C, dtype=self.dtype, order='F')
            self.cov_R = np.asarray(R, dtype=self.dtype, order='F')
            self.nsamp = self.C.shape[0]
            self.log._log(f"Final sample count after covariate filtering/subsample: {self.nsamp}")
        else:
            # No covariates: just use the random subsample or all rows
            self.row_sel = sel_idx if sel_idx is not None else None
            self.C = None
            self.cov_R = None
            if self.row_sel is not None:
                self.nsamp = len(self.row_sel)
                self.log._log(f"No covariates. Using random subsample: {self.nsamp} individuals.")
            else:
                self.log._log("No covariates and no subsampling: using all individuals.")

        self.p_eff = self.C.shape[1] if self.C is not None else 0
        self.N_eff = self.nsamp - self.p_eff
        if self.N_eff <= 1:
            raise ValueError(f"N_eff={self.N_eff} is too small after projection.")
        
        self.outpath = out_path

        # storage used by correlation mode
        self.inv_sqrt_resvar_all = None # (M,) 1/sqrt(Var(M x_m)+eps)

    def _auto_vchunk(self, phase: str) -> int:
        """
        Conservative V-chunk sizing.
        Reserve headroom for the parent shared arrays (Xz2d + meansq),
        then give each worker ~20% of the remaining available memory.
        Floor at 64 and round up to a multiple of 64 for BLAS.
        """
        b = np.dtype(self.dtype).itemsize
        N   = int(self.nsamp)
        M   = int(self.nsnps)
        B   = int(self.nbins)
        V   = int(self.nvecs)
        S   = int(min(self.step_size, self.nsnps))
        W   = max(1, int(self.nworkers))

        # Parent SHM footprint we already committed (or will commit): N*(V*B) + M*B
        parent_bytes = (N * (V * B) + M * B) * b

        try:
            import psutil
            avail = int(psutil.virtual_memory().available)
        except Exception:
            avail = 8 * (1024 ** 3)  # 8 GB fallback

        # Guard: do not assume we can use all available memory.
        # Reserve at least the parent footprint and only use a small slice of the rest.
        remain = max(0, avail - parent_bytes)
        budget_per_worker = max(64 * (1024 ** 2), int(0.20 * remain / W))  # 20% of the remainder

        geno_blk = N * S * b
        extra = budget_per_worker - geno_blk
        if extra <= 0:
            return min(V, 64)  # bare minimum

        bytes_per_v = S * b  # one (L×1) column
        v_chunk = int(extra // bytes_per_v)
        v_chunk = max(64, min(V, v_chunk))
        # round up to multiple of 64 for nicer GEMM kernels
        v_chunk = min(V, ((v_chunk + 63) // 64) * 64)
        return v_chunk
    
    def _precompute_residual_variances(self):
        """
        Precompute inv sqrt residual variances per SNP, consistent with XtXz.
        Handles: standardization (ddof), nan->0 imputation, and optional projection.
        """
        row_sel = self.row_sel if self.row_sel is not None else slice(None)
        inv = np.empty(self.nsnps, dtype=self.dtype)

        chunks = [(s, min(self.nsnps, s + self.step_size))
                for s in range(0, self.nsnps, self.step_size)]
        if not chunks:
            self.log._log("[warn] No SNP chunks formed; returning zeros.")
            return np.zeros(self.nsnps, dtype=self.dtype)

        for s, e in chunks:
            G = self.G.read(index=np.s_[row_sel, s:e], dtype=self.dtype)  # (N × L)
            means = np.nanmean(G, axis=0, dtype=self.dtype)
            stds  = np.nanstd( G, axis=0, dtype=self.dtype, ddof=self.ddof)
            stds[stds == 0] = 1.0
            G = (G - means) / stds
            np.nan_to_num(G, copy=False)  # mean-imputation at 0 after centering

            if self.C is not None:
                tmp = self.cov_R @ G
                G   = G - (self.C @ tmp)
                del tmp
                N_denom = float(self.N_eff)
            else:
                N_denom = float(self.nsamp)

            var = np.sum(G * G, axis=0, dtype=self.dtype) / max(N_denom - 1.0, 1.0)
            inv[s:e] = (1.0 / np.sqrt(np.maximum(var, self.eps_var))).astype(self.dtype, copy=False)

        return inv

    # ------------------ Phase 1 worker: build X_k z ------------------
    def _compute_Xz_blk(self, task):
        """
        Phase 1 worker (block j):
        - reads/standardizes geno block (N × L)
        - builds Xz for all bins in this block, in v-chunks
        - uses chunk-invariant RNG so results are independent of vchunk
        - accumulates into global _g_Xz2d via local scratch to avoid double-add
        """
        j, blk_start, blk_end = task
        L = blk_end - blk_start
        row_sel = self.row_sel if self.row_sel is not None else slice(None)

        # ---- read & standardize genotype block ----
        geno = self.G.read(index=np.s_[row_sel, blk_start:blk_end], dtype=self.dtype)
        geno = np.array(geno, dtype=self.dtype, order='F', copy=True)

        means = np.nanmean(geno, axis=0, dtype=self.dtype)
        stds  = np.nanstd(geno, axis=0, dtype=self.dtype, ddof=self.ddof)
        stds[stds == 0] = 1.0

        np.subtract(geno, means, out=geno)      # in place
        np.divide(  geno, stds,  out=geno)      # in place
        np.nan_to_num(geno, copy=False)

        inv_sqrt = self.inv_sqrt_resvar_all[blk_start:blk_end]  # (L,)
        ann_blk  = self.annot[blk_start:blk_end]                # (L × nbins)

        # ---- indices & per-bin scale precompute ----
        idxs, scales = [], []
        Kmax = 0
        for k in range(self.nbins):
            bi = np.nonzero(ann_blk[:, k])[0]
            idxs.append(bi)
            if bi.size:
                Kmax = max(Kmax, int(bi.size))
                sk = np.empty(bi.size, dtype=self.dtype)
                np.take(inv_sqrt, bi, out=sk)                 # sk = inv_sqrt[bi]
                tmp = np.empty_like(sk)
                np.take(ann_blk[:, k], bi, out=tmp)           # tmp = a_{i,k}
                np.sqrt(tmp, out=tmp)                         # sqrt(a_{i,k})
                sk *= tmp                                     # inv_sqrt * sqrt(a)
                scales.append(sk)
            else:
                scales.append(None)
        if self.nbins == 0:
            return 1

        # ---- fixed workspaces ----
        N = geno.shape[0]
        vchunk = int(getattr(self, "v_chunk_xz", 0) or self.nvecs)
        vchunk = max(1, min(vchunk, self.nvecs))

        A_buf = np.empty((N, Kmax), dtype=self.dtype, order='F')
        B_buf = np.empty((Kmax, vchunk), dtype=self.dtype, order='F')

        gemm = fblas.sgemm if self.dtype is np.float32 else fblas.dgemm

        # ---- RNG + accumulation per v-chunk ----
        # NOTE: each Z column is generated from a seed keyed by its GLOBAL index (vglob),
        #       so results are invariant to how we split v into chunks.
        for v0 in range(0, self.nvecs, vchunk):
            v1 = min(self.nvecs, v0 + vchunk)
            Vt = v1 - v0

            # Build Z (L × Vt) in Fortran order, column-by-column with global-index seeds
            Z = np.empty((L, Vt), dtype=self.dtype, order='F')
            if self.root_seed is None:
                # Unseeded: still make it chunk-invariant by drawing per-column
                for t, vglob in enumerate(range(v0, v1)):
                    rng = np.random.default_rng(None)
                    Z[:, t] = rng.standard_normal(L).astype(self.dtype, copy=False)
            else:
                rseed = int(self.root_seed)
                jj    = int(j)
                for t, vglob in enumerate(range(v0, v1)):
                    rng = np.random.default_rng([rseed, jj, int(vglob)])
                    if self.rand_dist == "rademacher":
                        Zi = rng.integers(0, 2, size=L, dtype=np.int8)
                        Zi *= 2; Zi -= 1
                        Z[:, t] = Zi.astype(self.dtype, copy=False)
                    else:
                        Z[:, t] = rng.standard_normal(L).astype(self.dtype, copy=False)

            # Optional spherical normalization (column-wise)
            if self.rand_dist == "spherical":
                norms = np.linalg.norm(Z, axis=0)
                norms[norms == 0] = 1.0
                Z /= norms
                Z *= np.sqrt(L).astype(self.dtype)

            # --- per-bin accumulation using fixed workspaces (local scratch → add once) ---
            for k in range(self.nbins):
                bi = idxs[k]
                K  = int(bi.size)
                if K == 0:
                    continue

                # A_view: (N × K), gather and scale in-place
                A_view = A_buf[:, :K]
                np.take(geno, bi, axis=1, out=A_view)
                A_view *= scales[k]

                # B_view: (K × Vt), gather Z rows
                B_view = B_buf[:K, :Vt]
                np.take(Z, bi, axis=0, out=B_view)

                # Local scratch for this (k, v0:v1): compute with beta=0, then add once under lock
                c_view = _g_Xz2d[:, k*self.nvecs + v0 : k*self.nvecs + v1]   # (N × Vt)
                C_loc  = np.empty_like(c_view, order='F')
                gemm(1.0, A_view, B_view, c=C_loc, beta=0.0, overwrite_c=1)

                with _g_xz_locks[k]:
                    c_view += C_loc

        return 1

    
    # ------------------ Phase 2 worker: multiply (MG)^T (MB) ------------------
    def _compute_XtXz_blk(self, blk_idx):
        """
        Phase 2 (chunked V): Y = M·G (or G); left-normalize to partial correlations,
        then for each bin accumulate sum_j ( (Y^T @ MB_k)_j^2 ) across V **in chunks**.
        Writes mean-of-squares into shared _g_meansq[blk_start:blk_end, k].
        """
        import numpy as _np
        from scipy.linalg import blas as _blas

        blk_start, blk_end = blk_idx
        L = blk_end - blk_start
        if L <= 0:
            return 1

        vchunk = getattr(self, "v_chunk_xtxz", None)
        if not vchunk:
            vchunk = self._auto_vchunk('XtXz')

        row_sel = self.row_sel if self.row_sel is not None else slice(None)
        gemm = _blas.sgemm if self.dtype is _np.float32 else _blas.dgemm

        # Read & standardize (raw-space)
        geno = self.G.read(index=_np.s_[row_sel, blk_start:blk_end], dtype=self.dtype)
        means = _np.nanmean(geno, axis=0, dtype=self.dtype)
        stds  = _np.nanstd(geno, axis=0, dtype=self.dtype, ddof=self.ddof)
        stds[stds == 0] = 1.0
        geno  = (geno - means) / stds
        _np.nan_to_num(geno, copy=False)
        geno  = _np.asarray(geno, dtype=self.dtype, order='F')  # (N × L)

        # Project: Y = (I - C R) G  (C = Q, R = Q^T)
        if self.C is not None:
            tmpG = self.cov_R @ geno       # (p × L)
            Y    = geno - (self.C @ tmpG)  # (N × L)
            N_denom = float(self.N_eff)
        else:
            Y = geno
            N_denom = float(self.nsamp)

        # Left normalization to partial correlations
        inv_sqrt_left = self.inv_sqrt_resvar_all[blk_start:blk_end]

        # Allocate an accumulator for this block & per-bin mean-of-squares across V
        # We'll accumulate sum over V of squared entries, then divide by V once.
        # Layout note: acc is (L,), we'll reuse for each bin.
        work = _np.empty((L, vchunk), dtype=self.dtype, order='F')  # reused
        scale_cols = np.array(1.0 / float(N_denom - 1), dtype=self.dtype).item()

        for k in range(self.nbins):
            acc = _np.zeros(L, dtype=self.dtype, order='F')  # sum of squares across V

            # Process V in chunks so work stays (L × vchunk)
            for c0 in range(0, self.nvecs, vchunk):
                c1 = min(self.nvecs, c0 + vchunk)
                v  = c1 - c0

                MB_k = _g_Xz2d[:, k*self.nvecs + c0 : k*self.nvecs + c1]  # (N × v), already right-normalized & projected

                # work = Y^T @ MB_k  → (L × v)
                gemm(1.0, Y, MB_k, c=work[:, :v], beta=0.0, trans_a=True, overwrite_c=1)
                # left normalization + divide by N_eff-1
                work[:, :v] *= inv_sqrt_left[:, None]
                work[:, :v] *= scale_cols

                # accumulate squared values along V
                acc += _np.sum(work[:, :v] * work[:, :v], axis=1, dtype=self.dtype)

            # write mean across V
            _g_meansq[blk_start:blk_end, k] = (acc / float(self.nvecs)).astype(_g_meansq.dtype, copy=False)
        return 1


    # ------------------ I/O helpers ------------------
    def _read_annot(self, annot_path):
        """
        Read annotation for gw_ldscore:
        • LDSC-style full .annot(.gz): columns [CHR, BP, SNP, CM, <bins...>]
        • or 'thin' matrix (no base cols), via utils._read_with_optional_header.
        Supports binary OR continuous annotations. Values must be >= 0.
        Final row order MUST match .bim SNP order and length (no dropping).
        """
        if annot_path is None:
            self.l2cols = None
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)  # ∑ weights (here = M)
            self.log._log("Calculating genome-wide (non-partitioned) LD score")
            self.l2cols = [f"L2_{i}" for i in range(self.nbins)]
            self.log._log(f"Number of samples: {self.nsamp}")
            self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
            return

        parsed_ldsc = False
        try:
            df = pd.read_csv(annot_path, sep=r'\s+', compression='infer',
                            dtype={'CHR':str, 'BP':np.int64, 'SNP':str, 'CM':float})
            base_cols = ['CHR', 'BP', 'SNP', 'CM']
            if all(c in df.columns[:4].tolist() for c in base_cols) and 'SNP' in df.columns:
                annot_cols = [c for c in df.columns if c not in base_cols]
                if len(annot_cols) == 0:
                    raise ValueError("No annotation columns found after [CHR,BP,SNP,CM].")

                bim_snps = self.snplist.iloc[:, 1].astype(str).tolist()
                ann_snps = df['SNP'].astype(str).tolist()

                if ann_snps == bim_snps:
                    ann_mat = df[annot_cols].to_numpy(dtype=np.float64, copy=False)
                else:
                    ann_set = set(ann_snps); bim_set = set(bim_snps)
                    missing_in_annot = len(bim_set - ann_set)
                    extra_in_annot   = len(ann_set - bim_set)
                    if missing_in_annot > 0:
                        raise ValueError(
                            f"Annotation SNP set is missing {missing_in_annot} BIM SNP(s); "
                            f"prepare a matching .annot or regenerate it to the .bim."
                        )
                    if extra_in_annot > 0:
                        self.log._log(f"[info] Annotation contains {extra_in_annot} extra SNP(s) not in BIM; "
                                    f"keeping BIM SNPs only and reordering to BIM.")
                    ann_mat = df.set_index('SNP').loc[bim_snps, annot_cols].to_numpy(dtype=np.float64, copy=False)

                # --- sanitize & flag continuous ---
                np.nan_to_num(ann_mat, copy=False)  # replace NaN/±inf with finite values (0 by default)
                if (ann_mat < 0).any():
                    self.log._log("[warn] Negative annotation values found; clipping to 0.")
                    ann_mat[ann_mat < 0] = 0.0
                uniq = np.unique(ann_mat)
                is_binary = np.all(np.isin(uniq, [0.0, 1.0]))
                self.is_continuous = (not is_binary)
                if self.is_continuous:
                    self.log._log("[info] Detected continuous annotations (non 0/1 values).")
                else:
                    self.log._log("[info] Detected binary annotations (0/1).")

                self.annot = ann_mat
                self.nbins = self.annot.shape[1]
                self.l2cols = annot_cols
                parsed_ldsc = True
                self.log._log(f"Read LDSC-style annotation with shape {self.annot.shape}")
        except Exception:
            parsed_ldsc = False

        if not parsed_ldsc:
            self.l2cols, arr = utils._read_with_optional_header(annot_path)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = arr.astype(np.float64, copy=False)
            np.nan_to_num(arr, copy=False)
            if (arr < 0).any():
                self.log._log("[warn] Negative annotation values found; clipping to 0.")
                arr[arr < 0] = 0.0
            uniq = np.unique(arr)
            is_binary = np.all(np.isin(uniq, [0.0, 1.0]))
            self.is_continuous = (not is_binary)
            self.annot = arr
            if self.l2cols is None:
                self.l2cols = [f"L2_{i}" for i in range(self.annot.shape[1])]
            self.nbins = self.annot.shape[1]
            self.log._log(f"Read thin annotation matrix with shape {self.annot.shape}")

        if self.annot.shape[0] != self.nsnps:
            self.log._log(f"!!! number of SNPs in annotation ({self.annot.shape[0]}) "
                        f"does not match the input genotype file ({self.nsnps}) !!!")
            sys.exit(1)

        # For continuous: this is ∑_j a_{j,k}; for binary: #SNPs in bin.
        self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)

        self.log._log(f"Number of samples: {self.nsamp}")
        self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
        self.log._log(f"Nbins: {self.nbins}")


    def _read_bim(self, bim_path):
        if (bim_path is None):
            self.log._log("No .bim file is provided; all (anonymous) SNPs will be used")
            self.snplist = None
        else:
            self.log._log(f"Reading {bim_path} for SNPs")
            self.snplist = pd.read_csv(bim_path, header=None, sep=r'\s+')
            self.snplist.columns = ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2']
        if (len(self.snplist) != self.nsnps):
            self.log._log(f"!!! The number of SNPs in the .bed file ({self.nsnps}) does not match the .bim file ({len(self.snplist)}) !!!")
            sys.exit(1)
    
    def _partition_index(self, snpidx, annot) -> list[np.ndarray]:
        return [snpidx[(annot[:, c] != 0)] for c in range(self.nbins)]

    # ------------------ main compute ------------------
    def _compute_ldscore(self):
        """
        Phase 0: precompute Var(M x_m) for all SNPs (or ones if no covariates) and cache 1/sqrt.
        Phase 1: build X_k z from raw-standardized genotypes, scaling right columns by 1/sqrt(Var(M x_m)).
                Parent then projects: Xz2d ← (I - QQ^T) Xz2d in-place (if covariates).
        Phase 2: for each block, Y = M·G; scale left rows by 1/sqrt(Var(M g_j));
                compute Y^T @ (MB_k), divide by N_eff, square, and average across V.
        Baseline: subtract M_k / N_eff per bin (correlation null).
        """
        self.log._log(f"num_vecs: {self.nvecs}, num_workers: {self.nworkers}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        if self.C is not None:
            self.log._log(f"Covariate-adjusted partial correlations (N_eff={self.N_eff}, p={self.p_eff}).")
        else:
            self.log._log("No covariates: standard LD scores (squared correlations).")
        
        # Pick chunk sizes (you can also make them CLI flags)
        self.v_chunk_xz   = self._auto_vchunk('Xz')
        self.v_chunk_xtxz = self._auto_vchunk('XtXz')
        self.log._log(f"Using V-chunk sizes: Phase1 (Xz)={self.v_chunk_xz}, Phase2 (XtXz)={self.v_chunk_xtxz}")

        # ---- Phase 0: per-SNP residual variances and inverse sqrt (right side) ----
        self.inv_sqrt_resvar_all = self._precompute_residual_variances()

        self.nblks = len(np.arange(self.nsnps)[::self.step_size])
        self._print_expected_mem('Xz')
        _rss_snapshot("pre-alloc", self.log)

        Xz_input = []
        XtXz_input = []
        for j in range(self.nblks):
            idx_start = self.step_size*j
            idx_end = self.nsnps if j==self.nblks-1 else self.step_size*(j+1)
            annot_blk = self.annot[idx_start:idx_end]
            Xz_input.append((j, idx_start, idx_end))
            XtXz_input.append((idx_start, idx_end))

        shm_xz = shm_ms = None
        try:
            # ---- allocate shared arrays ----
            itemsize = np.dtype(self.dtype).itemsize
            xz_shape2d = (self.nsamp, self.nvecs * self.nbins)
            ms_shape   = (self.nsnps, self.nbins)

            shm_xz = shared_memory.SharedMemory(create=True, size=int(np.prod(xz_shape2d)) * itemsize)
            self.Xz2d = np.frombuffer(shm_xz.buf, dtype=self.dtype,
                                    count=xz_shape2d[0]*xz_shape2d[1]).reshape(xz_shape2d, order='F')
            self.Xz2d.fill(0)

            shm_ms = shared_memory.SharedMemory(create=True, size=int(np.prod(ms_shape)) * itemsize)
            self.meansq = np.ndarray(ms_shape, dtype=self.dtype, buffer=shm_ms.buf)
            self.meansq.fill(0)

            xz_locks = [mp.Lock() for _ in range(self.nbins)]
            _rss_snapshot("after SHM alloc", self.log)

            # ---- Phase 1: build Xz ----
            self._print_expected_mem('Xz')
            with mp.Pool(self.nworkers, maxtasksperchild=4,
                        initializer=_init_shared,
                        initargs=(shm_xz.name, xz_shape2d, None, None,
                                np.dtype(self.dtype).str, xz_locks)) as pool:
                with tqdm(total=self.nblks, desc='Calculating Xz') as pbar:
                    for _ in pool.imap_unordered(self._compute_Xz_blk, Xz_input):
                        pbar.update()
            
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            _rss_snapshot("after Xz", self.log)
            self.Xz_time = utils._get_time()
            self.log._log("Calculation of Xz (for each partition) completed. Runtime: "+format(self.Xz_time - self.start_time, '.3f')+" s")

            # ---- project right: Xz2d ← (I - QQ^T) Xz2d ----
            if self.C is not None:
                self.log._log("Projecting Xz onto covariate-orthogonal space: Xz ← (I - QQ^T) Xz (in-place)")
                chunk_bins = 4
                for b0 in range(0, self.nbins, chunk_bins):
                    b1 = min(self.nbins, b0 + chunk_bins)
                    c0 = b0 * self.nvecs
                    c1 = b1 * self.nvecs
                    X = self.Xz2d[:, c0:c1]           # (N × chunk)
                    tmp = self.cov_R @ X              # (p × chunk)
                    X  -= self.C @ tmp                # in-place
                    del tmp, X
                _rss_snapshot("after in-place projection of Xz", self.log)

            # ---- Phase 2: fill meansq ----
            self._print_expected_mem('XtXz')
            with mp.Pool(self.nworkers, maxtasksperchild=4,
                        initializer=_init_shared,
                        initargs=(shm_xz.name, xz_shape2d, shm_ms.name, ms_shape,
                                np.dtype(self.dtype).str, xz_locks)) as pool:
                with tqdm(total=self.nblks, desc='Calculating XtXz') as pbar:
                    for _ in pool.imap_unordered(self._compute_XtXz_blk, XtXz_input):
                        pbar.update()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass


            _rss_snapshot("after XtXz", self.log)
            self.XtXz_time = utils._get_time()
            self.log._log("Calculation of XtXz (for each partition) completed. Runtime: "+format(self.XtXz_time - self.Xz_time, '.3f')+" s")

            # ---- Baseline subtraction: classic correlation null M_k / N_eff ----
            N_denom = float(self.N_eff-1.0 if self.C is not None else self.nsamp-self.ddof)
            self.log._log("Applying correlation null: subtracting M_k / N_denom per bin.")
            self.meansq -= (self.nsnps_bin / N_denom).astype(self.meansq.dtype, copy=False)[None, :]

            # ---- Save outputs ----
            self.gwldscore = self.meansq.astype(np.float64, copy=False)
            self.log._log(f"Saving the genome-wide (partitioned) LD scores into: {self.outpath}.gw.ldscore.gz")
            snpcols = ['CHR', 'SNP', 'BP']
            if (self.snplist is None):
                self.snpdf = pd.DataFrame(np.nan*np.ones((self.nsnps, 3)), columns=snpcols)
            else:
                self.snpdf = self.snplist[['CHR','SNP','BP']].copy()
                self.snpdf.columns = snpcols
            
            scores_df = pd.DataFrame(self.gwldscore, columns=self.l2cols)
            out_df = pd.concat([self.snpdf, scores_df], axis=1)
            out_df.to_csv(f'{self.outpath}.gw.ldscore.gz', index=False, compression='gzip', sep='\t', float_format='%.6f')

            # ---- Post-run stats ----
            try:
                # Per-bin summaries
                desc = scores_df.describe(percentiles=[0.25, 0.5, 0.75]).loc[['count','mean','std','min','25%','50%','75%','max']]
                self.log._log("Per-bin LD score summary (count/mean/std/min/25%/50%/75%/max):")
                with pd.option_context('display.width', 140, 'display.max_columns', None, 'display.float_format', '{:.6f}'.format):
                    self.log._log(desc.to_string() + "\n")

                corr = scores_df.corr(method='pearson')
                self.log._log("Correlation matrix across bins (Pearson):")
                with pd.option_context('display.width', 140, 'display.max_columns', None, 'display.float_format', '{:.4f}'.format):
                    self.log._log("\n" + corr.to_string())

                col_sums = pd.Series(self.nsnps_bin, index=self.l2cols)
                lines = ["Annotation Column Sums"] + [f"{k:<35} {v:.6f}" for k, v in col_sums.items()]
                self.log._log("\n" + "\n".join(lines))

                row_sums = self.annot.sum(axis=1, dtype=np.float64)
                desc = pd.Series(row_sums).describe(percentiles=[0.25, 0.5, 0.75])
                self.log._log("\nSummary of Annotation Matrix Row Sums")
                with pd.option_context('display.float_format', '{:.4f}'.format):
                    ordered = ['count','mean','std','min','25%','50%','75%','max']
                    lines = [f"{k:<6} {desc[k]:.4f}" for k in ordered]
                    self.log._log("\n".join(lines))

            except Exception as e:
                self.log._log(f"[warn] Failed to compute summary stats / correlation: {e}")
            
        # -------------------------------------------------------------------

            self.end_time = utils._get_time()
            self.log._log(f"Calculation of genome-wide LD score ended at "+utils._get_timestr(self.end_time))
            self.runtime=self.end_time - self.start_time
            self.log._log("Runtime: "+format(self.runtime, '.3f')+f" s ({self.runtime//3600} hr {(self.runtime%3600)//60} m {(self.runtime%60):.3f} s)")
            self.log._save_log(self.outpath+".gw.log")
        except KeyboardInterrupt:
            self.log._log("KeyboardInterrupt received — terminating workers and cleaning shared memory.")
            raise
        finally:
            # ensure NO numpy views remain before closing SHM
            try:
                self.Xz2d = None
                self.meansq = None
                gc.collect()
            finally:
                if shm_xz is not None:
                    try:
                        shm_xz.close()
                    except BufferError as e:
                        self.log._log(f"[warn] shm_xz.close BufferError: {e}. Unlinking anyway.")
                    finally:
                        try: shm_xz.unlink()
                        except FileNotFoundError: pass
                        except Exception as e: self.log._log(f"[warn] shm_xz.unlink: {e}")
                if shm_ms is not None:
                    try:
                        shm_ms.close()
                    except BufferError as e:
                        self.log._log(f"[warn] shm_ms.close BufferError: {e}. Unlinking anyway.")
                    finally:
                        try: shm_ms.unlink()
                        except FileNotFoundError: pass
                        except Exception as e: self.log._log(f"[warn] shm_ms.unlink: {e}")
            _rss_snapshot("post-cleanup", self.log)
    
    def _print_expected_mem(self, phase, block_len=None, k_max=None):
        b = np.dtype(self.dtype).itemsize
        B, N, M, V, S, W = self.nbins, self.nsamp, self.nsnps, self.nvecs, self.step_size, self.nworkers
        L = min(S, M)

        xz_bytes = N * (V * B) * b
        ms_bytes = M * B * b

        vchunk = (self.v_chunk_xz if phase == 'Xz'
                else getattr(self, 'v_chunk_xtxz', self.nvecs))
        geno_blk = N * L * b
        per_worker = geno_blk + (L * min(vchunk, V) * b)  # chunked temp

        total_est = xz_bytes + ms_bytes + W * per_worker
        self.log._log(
            f"[expected {phase}] dtype={self.dtype}, B={B}, N={N}, M={M}, V={V}, S={S}, W={W} "
            f"→ parent(Xz2d+meansq)≈{_bytes_human(xz_bytes + ms_bytes)}, "
            f"per-worker temps≈{_bytes_human(per_worker)}, "
            f"total≈{_bytes_human(total_est)}"
        )

