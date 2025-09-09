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
    # Best-effort runtime setters (safe if unavailable)
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
    rss = None
    total = None
    num_children = 0
    try:
        import psutil
        p = psutil.Process()
        rss = p.memory_info().rss
        total = rss
        if include_children:
            kids = p.children(recursive=True)
            num_children = len(kids)
            total += sum(c.memory_info().rss for c in kids if c.is_running())
    except Exception:
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF)
            ru = r.ru_maxrss
            rss = ru * 1024 if ru < 10**9 else ru
            total = rss
        except Exception:
            pass

    msg = (f"[mem] {label}: parent RSS={_bytes_human(rss)}; "
           f"parent+children≈{_bytes_human(total)}; children={num_children}")
    if logger:
        try: logger._log(msg)
        except Exception: print(msg, file=sys.stderr, flush=True)
    else:
        print(msg, file=sys.stderr, flush=True)

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
    verbose=False
):
    fam = pd.read_csv(fam_filename, sep=r'\s+', header=None, usecols=[0,1], names=['FID','IID'])
    cov = pd.read_csv(cov_filename, sep=r'\s+')

    merged = fam.merge(cov, on=['FID','IID'], how='left', indicator=True)
    n_missing_in_cov = (merged['_merge'] != 'both').sum()
    if n_missing_in_cov:
        raise ValueError(f"{n_missing_in_cov} .fam samples not found in covariate file (FID/IID mismatch).")
    merged.drop(columns=['_merge'], inplace=True)

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
        df = (df - df.mean()) / df.std(ddof=1)
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

    keep_idx = np.flatnonzero(keep_mask.values) if isinstance(keep_mask, pd.Series) else np.flatnonzero(keep_mask)

    if logger:
        logger._log(f"Read {cov_filename}: kept {C.shape[0]} samples, {C.shape[1]} effective covariates. "
                    f"C shape={C.shape}, R shape=({R.shape[0]},{R.shape[1]}).")

    return C, R, keep_idx

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
                num_threads: int = 4):        # <-- NEW: cap BLAS threads
        # Limit BLAS threads up front (before we spawn any Pools)
        self.num_threads = num_threads
        limit_blas_threads(self.num_threads)

        self.G = open_bed(bed_path + ".bed")
        self.nsamp, self.nsnps = self.G.shape
        self.nvecs = num_vecs
        self.nworkers = num_workers
        self.step_size = step_size
        self.log = log
        self.verbose = verbose
        self.dtype = np.float32 if dtype in (np.float32, 'float32', 'f4') else np.float64
        self.rand_dist = rand_dist

        # read .bim and annotation
        self._read_bim(bed_path + ".bim")
        if annot_path is not None:
            self._read_annot(annot_path)
        else:
            self._read_annot(None)

        # covariates → orthonormal Q (C) and Q^T (cov_R)
        if covar_path is not None:
            fam_file = bed_path + ".fam"
            self.C, self.cov_R, self.keep_rows = read_cov(
                cov_filename=covar_path,
                fam_filename=fam_file,
                std=True,
                cov_impute_method="ignore",
                one_hot_conversion=False,
                categorical_threshold=100,
                logger=self.log,
                verbose=self.verbose
            )
            self.nsamp = self.C.shape[0]
            self.log._log(f"Final sample count after covariate filtering: {self.nsamp}")
            self.C = np.asarray(self.C, dtype=self.dtype, order='F')
            self.cov_R = np.asarray(self.cov_R, dtype=self.dtype, order='F')
        else:
            self.keep_rows = None
            self.C = None
            self.cov_R = None
            self.log._log("No covariate correction will be applied.")

        self.p_eff = self.C.shape[1] if self.C is not None else 0
        self.N_eff = self.nsamp - self.p_eff
        if self.N_eff <= 1:
            raise ValueError(f"N_eff={self.N_eff} is too small after covariate projection.")

        self.root_seed = seed
        self.outpath = out_path


    # ------------------ Phase 1 worker: build X_k z (raw-space) ------------------
    def _compute_Xz_blk(self, blk_idxs):
        """
        For block [blk_start:blk_end), build per-bin X_k z and accumulate into shared _g_Xz2d.
        NO covariate residualization here; standardize genotypes in RAW space only.
        """
        j, blk_start, blk_end, idxs = blk_idxs
        nsnps = sum(len(binidx) for binidx in idxs)

        rng = np.random.default_rng([j, self.root_seed] if self.root_seed is not None else None)
        if self.rand_dist == "normal":
            Zs = rng.standard_normal(size=(nsnps, self.nvecs))
        elif self.rand_dist == "rademacher":
            Zs = rng.integers(0, 2, size=(nsnps, self.nvecs)) * 2 - 1
        elif self.rand_dist == "spherical":
            Zs = rng.standard_normal(size=(nsnps, self.nvecs))
            norms = np.linalg.norm(Zs, axis=0)
            Zs = Zs / norms[None, :] * np.sqrt(nsnps)

        row_sel = self.keep_rows if self.keep_rows is not None else slice(None)
        geno = self.G.read(index=np.s_[row_sel, blk_start:blk_end], dtype=self.dtype)

        # standardize in raw space
        means = np.nanmean(geno, axis=0, dtype=self.dtype)
        stds  = np.nanstd(geno, axis=0, dtype=self.dtype)
        stds[stds == 0] = 1.0
        geno  = (geno - means) / stds
        np.nan_to_num(geno, copy=False)
        geno = np.asarray(geno, dtype=self.dtype, order='F')

        Zs = np.asarray(Zs, order='F', dtype=self.dtype)
        gemm = fblas.sgemm if self.dtype is np.float32 else fblas.dgemm

        for k, binidx in enumerate(idxs):
            if len(binidx) == 0:
                continue
            A = np.asfortranarray(geno[:, binidx], dtype=self.dtype)   # (N × K)
            B = np.asfortranarray(Zs[binidx, :],  dtype=self.dtype)    # (K × V)
            c_view = _g_Xz2d[:, k*self.nvecs:(k+1)*self.nvecs]         # (N × V)
            with _g_xz_locks[k]:
                gemm(1.0, A, B, c=c_view, beta=1.0, overwrite_c=1)
        return 1

    # ------------------ Phase 2 worker: multiply (MG)^T (MB) ------------------
    def _compute_XtXz_blk(self, blk_idx):
        """
        For genotype block G (columns blk_start:blk_end), let Y = M·G if covariates else G.
        Multiply for each bin k: work = Y^T @ (MB_k), where _g_Xz2d already holds MB_k.
        Scale by 1/N_denom, write mean of squares across V into _g_meansq.
        ALSO return:
        - resvar_block: residual variances of left SNPs in this block (length L),
        - sum_by_bin_block: per-bin sum of residual variances contributed by RIGHT SNPs
            from this block (length B), to build the data-adaptive baseline.
        """
        blk_start, blk_end = blk_idx
        gemm = fblas.sgemm if self.dtype is np.float32 else fblas.dgemm

        row_sel = self.keep_rows if self.keep_rows is not None else slice(None)
        geno = self.G.read(index=np.s_[row_sel, blk_start:blk_end], dtype=self.dtype)

        # raw-space standardization
        means = np.nanmean(geno, axis=0, dtype=self.dtype)
        stds  = np.nanstd(geno, axis=0, dtype=self.dtype)
        stds[stds == 0] = 1.0
        geno  = (geno - means) / stds
        np.nan_to_num(geno, copy=False)
        geno  = np.asarray(geno, dtype=self.dtype, order='F')

        block_len = blk_end - blk_start
        work = np.empty((block_len, self.nvecs), dtype=self.dtype, order='F')

        # Left projection once per block if covariates: Y = G - PG, else Y = G
        if self.C is not None:
            tmpG = self.cov_R @ geno   # (p × L)
            PG   = self.C @ tmpG       # (N × L)
            Y    = geno - PG
            N_denom = self.N_eff
        else:
            Y = geno
            N_denom = self.nsamp

        # residual variances for left SNPs in this block (float64 for stability)
        resvar_block = (np.sum(Y * Y, axis=0, dtype=np.float64) / float(N_denom))  # shape (L,)

        # For each bin, multiply and write means directly into shared meansq
        for k in range(self.nbins):
            MB_k = _g_Xz2d[:, k*self.nvecs:(k+1)*self.nvecs]  # (N × V), already M·X_k z if covariates, else X_k z
            gemm(1.0, Y, MB_k, c=work, beta=0.0, trans_a=True, overwrite_c=1)  # Y^T @ MB_k
            work *= (1.0 / float(N_denom))
            _g_meansq[blk_start:blk_end, k] = np.mean(work * work, axis=1)

        # RIGHT-side residual-variance sums for baseline, from this block:
        # sum_by_bin_block[b] = sum_{m in block ∩ bin b} Var(M x_m)
        # We can compute Var(M x_m) for RIGHT SNPs in this block as the diagonal of (1/N_denom) * (M G)ᵀ (M G).
        # But we already computed Y = M·G; its columnwise squared norms / N_denom give us exactly those vars.
        annot_blk = (self.annot[blk_start:blk_end] != 0)          # (L × B) bool
        # (B,) using bool→float multiply: sum over L
        sum_by_bin_block = annot_blk.T.dot(resvar_block)          # (B,)
        # Return to parent for accumulation
        return (blk_start, resvar_block, sum_by_bin_block)


    # ------------------ I/O helpers ------------------
    def _read_annot(self, annot_path):
        if (annot_path is None):
            self.l2cols = None
            self.annot = np.ones((self.nsnps, 1))
            self.log._log("Calculating genome-wide (non-partitioned) LD score")
        else:
            self.l2cols, self.annot = utils._read_with_optional_header(annot_path)
            if (self.annot.ndim == 1):
                self.annot = self.annot.reshape(-1, 1)
            self.log._log(f"Read SNP partition annotation in {annot_path}")

        self.log._log(f"Number of samples: {self.nsamp}")
        self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
        
        if (self.nsnps != self.annot.shape[0]):
            self.log._log(f"!!! number of SNPs in annotation ({self.annot.shape[0]}) does not match the input genotype file ({self.nsnps}) !!!")
            sys.exit(1)
        self.nbins = self.annot.shape[1]
        if (self.l2cols is None):
            self.l2cols = ['L2_'+str(i) for i in range(self.annot.shape[1])]
        else:
            self.l2cols = [i + 'L2' for i in self.l2cols]
        self.log._log(f"Nbins: {self.nbins}")
        self.nsnps_bin = self.annot.sum(axis=0)

    def _read_bim(self, bim_path):
        if (bim_path is None):
            self.log._log("No .bim file is provided; all (anonymous) SNPs will be used")
            self.snplist = None
        else:
            self.log._log(f"Reading {bim_path} for SNPs")
            self.snplist = pd.read_csv(bim_path, header=None, sep='\t')
        if (len(self.snplist) != self.nsnps):
            self.log._log(f"!!! The number of SNPs in the .bed file ({self.nsnps}) does not match the .bim file ({len(self.snplist)}) !!!")
            sys.exit(1)
    
    def _partition_index(self, snpidx, annot) -> list[np.ndarray]:
        return [snpidx[(annot[:, c] != 0)] for c in range(self.nbins)]

    # ------------------ main compute ------------------
    def _compute_ldscore(self):
        """
        Phase 1: build X_k z (raw standardization).  (No covariate work here.)
        If covariates present, parent converts Xz2d → M·Xz2d in-place after Phase 1.
        Phase 2: for each block, Y = G or M·G; compute Y^T @ (M·X_k z) with 1 GEMM per bin.
        After Phase 2, subtract the data-adaptive baseline for squared partial covariances:
            meansq[j,k] -= Var(M g_j) * (sum_{m in bin k} Var(M x_m)) / N_denom
        and write results.
        """
        self.start_time = utils._get_time()
        self.log._log("Genome-wide LD score calculation started at: "+utils._get_timestr(self.start_time))
        self.log._log(f"num_vecs: {self.nvecs}, num_workers: {self.nworkers}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        if self.C is not None:
            self.log._log(f"Both-side covariate correction enabled (N_eff={self.N_eff}, p={self.p_eff}).")
        else:
            self.log._log("No covariates: using raw-space estimator (original speed/memory).")

        self.nblks = len(np.arange(self.nsnps)[::self.step_size])
        self._print_expected_mem('Xz')
        _rss_snapshot("pre-alloc", self.log)
        
        Xz_input = []
        XtXz_input = []
        for j in range(self.nblks):
            idx_start = self.step_size*j
            idx_end = self.nsnps if j==self.nblks-1 else self.step_size*(j+1)
            annot_blk = self.annot[idx_start:idx_end]
            Xz_input.append((j, idx_start, idx_end, self._partition_index(np.arange(len(annot_blk)), annot_blk)))
            XtXz_input.append((idx_start, idx_end))
        
        shm_xz = shm_ms = None
        try:
            # ----------------- allocate shared Xz (2D Fortran) and meansq -----------------
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

            # ----------------- Phase 1: build Xz in shared memory -----------------
            self._print_expected_mem('Xz')
            with mp.Pool(self.nworkers,
                        initializer=_init_shared,
                        initargs=(shm_xz.name, xz_shape2d, None, None,
                                np.dtype(self.dtype).str, xz_locks)) as pool:
                with tqdm(total=self.nblks, desc='Calculating Xz') as pbar:
                    for _ in pool.imap_unordered(self._compute_Xz_blk, Xz_input):
                        pbar.update()

            _rss_snapshot("after Xz", self.log)
            self.Xz_time = utils._get_time()
            self.log._log("Calculation of Xz (for each partition) completed. Runtime: "+format(self.Xz_time - self.start_time, '.3f')+" s")

            # ----------------- Covariate step (parent only): Xz2d ← M·Xz2d (in-place) -----------------
            if self.C is not None:
                self.log._log("Projecting Xz onto covariate-orthogonal space: Xz ← (I - QQ^T) Xz (in-place)")
                # Project in bin-chunks to give BLAS fatter blocks (tune chunk_bins if desired)
                chunk_bins = 4
                for b0 in range(0, self.nbins, chunk_bins):
                    b1 = min(self.nbins, b0 + chunk_bins)
                    c0 = b0 * self.nvecs
                    c1 = b1 * self.nvecs
                    X = self.Xz2d[:, c0:c1]           # (N × chunk)
                    tmp = self.cov_R @ X              # (p × chunk)
                    X  -= self.C @ tmp                # in-place: now M·X_k z
                    del tmp, X
                _rss_snapshot("after in-place projection of Xz", self.log)

            # ----------------- Phase 2: fill meansq in shared memory -----------------
            self._print_expected_mem('XtXz')

            # Buffers for adaptive baseline
            N_denom = self.N_eff if self.C is not None else self.nsamp
            self.resvar_left = np.empty(self.nsnps, dtype=np.float64)
            self.sum_resvar_by_bin = np.zeros(self.nbins, dtype=np.float64)

            with mp.Pool(self.nworkers,
                        initializer=_init_shared,
                        initargs=(shm_xz.name, xz_shape2d, shm_ms.name, ms_shape,
                                np.dtype(self.dtype).str, xz_locks)) as pool:
                with tqdm(total=self.nblks, desc='Calculating XtXz') as pbar:
                    for result in pool.imap_unordered(self._compute_XtXz_blk, XtXz_input):
                        blk_start, resvar_block, sum_by_bin_block = result
                        L = resvar_block.shape[0]
                        self.resvar_left[blk_start:blk_start+L] = resvar_block
                        self.sum_resvar_by_bin += sum_by_bin_block
                        pbar.update()

            _rss_snapshot("after XtXz", self.log)
            self.XtXz_time = utils._get_time()
            self.log._log("Calculation of XtXz (for each partition) completed. Runtime: "+format(self.XtXz_time - self.Xz_time, '.3f')+" s")

            # ----------------- Adaptive baseline subtraction (covariance null) -----------------
            self.log._log("Applying data-adaptive baseline for squared partial covariances.")
            baseline_cols = (self.sum_resvar_by_bin / float(N_denom)).astype(self.meansq.dtype, copy=False)  # (B,)
            # Subtract in blocks to limit peak memory
            for s in range(0, self.nsnps, self.step_size):
                e = min(self.nsnps, s + self.step_size)
                left = self.resvar_left[s:e].astype(self.meansq.dtype, copy=False)[:, None]                  # (L×1)
                self.meansq[s:e, :] -= left * baseline_cols[None, :]                                         # (L×B)
            del baseline_cols

            # ----------------- finish / save -----------------
            # After adaptive baseline, 'meansq' is already the LD score estimate in covariance geometry.
            self.gwldscore = self.meansq.astype(np.float64, copy=False)  # make a non-SHM copy for pandas/IO

            self.log._log(f"Saving the genome-wide (partitioned) LD scores into: {self.outpath}.gw.ldscore.gz")
            snpcols = ['CHR', 'SNP', 'BP']
            if (self.snplist is None):
                self.snpdf = pd.DataFrame(np.nan*np.ones((self.nsnps, 3)), columns=snpcols)
            else:
                self.snpdf = self.snplist.iloc[:, :3]
                self.snpdf.columns = snpcols
            
            scores_df = pd.DataFrame(self.gwldscore, columns=self.l2cols)
            out_df = pd.concat([self.snpdf, scores_df], axis=1)
            out_df.to_csv(f'{self.outpath}.gw.ldscore.gz', index=False, compression='gzip', sep='\t', float_format='%.6f')

            # ----------------- post-run statistics -----------------
            try:
                desc = scores_df.describe(percentiles=[0.25, 0.5, 0.75]).loc[['count','mean','std','min','25%','50%','75%','max']]
                self.log._log("Per-bin LD score summary (count/mean/std/min/25%/50%/max):")
                with pd.option_context('display.width', 140, 'display.max_columns', None, 'display.float_format', '{:.6g}'.format):
                    self.log._log("\n" + desc.to_string())

                corr = scores_df.corr(method='pearson')
                self.log._log("Correlation matrix across bins (Pearson, over SNP-wise LD scores):")
                with pd.option_context('display.width', 140, 'display.max_columns', None, 'display.float_format', '{:.4f}'.format):
                    self.log._log("\n" + corr.to_string())
            except Exception as e:
                self.log._log(f"[warn] Failed to compute summary stats / correlation: {e}")

            self.end_time = utils._get_time()
            self.log._log(f"Calculation of genome-wide LD score ended at "+utils._get_timestr(self.end_time))
            self.log._log("Runtime: "+format(self.end_time - self.start_time, '.3f')+" s")
            self.log._save_log(self.outpath+".gw.log")
        except KeyboardInterrupt:
            self.log._log("KeyboardInterrupt received — terminating workers and cleaning shared memory.")
            raise
        finally:
            # Ensure NO numpy views remain before closing SHM
            try:
                self.Xz2d = None
                self.meansq = None
                gc.collect()
            finally:
                # Close/unlink with BufferError tolerance
                if shm_xz is not None:
                    try:
                        shm_xz.close()
                    except BufferError as e:
                        self.log._log(f"[warn] shm_xz.close raised BufferError: {e}. Proceeding to unlink; OS will free when mappings are gone.")
                    finally:
                        try: shm_xz.unlink()
                        except FileNotFoundError: pass
                        except Exception as e: self.log._log(f"[warn] shm_xz.unlink: {e}")
                if shm_ms is not None:
                    try:
                        shm_ms.close()
                    except BufferError as e:
                        self.log._log(f"[warn] shm_ms.close raised BufferError: {e}. Proceeding to unlink; OS will free when mappings are gone.")
                    finally:
                        try: shm_ms.unlink()
                        except FileNotFoundError: pass
                        except Exception as e: self.log._log(f"[warn] shm_ms.unlink: {e}")
            _rss_snapshot("post-cleanup", self.log)

    
    def _print_expected_mem(self, phase, block_len=None, k_max=None):
        """
        Rough upper-bound memory accounting for this run.
        phase: 'Xz' or 'XtXz'
        """
        b = np.dtype(self.dtype).itemsize
        B, N, M, V, S, W = self.nbins, self.nsamp, self.nsnps, self.nvecs, self.step_size, self.nworkers
        L = min(S, M)

        # Shared (parent) arrays
        xz_bytes = N * (V * B) * b           # Xz2d shape (N, V*B)
        ms_bytes = M * B * b                 # meansq (M, B)

        # Per-worker temps
        geno_blk = N * L * b
        if phase == 'Xz':
            per_worker = geno_blk
        else:
            work = L * V * b
            per_worker = geno_blk + work

        total_est = xz_bytes + ms_bytes + W * per_worker
        self.log._log(
            f"[expected {phase}] dtype={self.dtype}, B={B}, N={N}, M={M}, V={V}, S={S}, W={W} "
            f"→ parent(Xz2d+meansq)≈{_bytes_human(xz_bytes + ms_bytes)}, "
            f"per-worker temps≈{_bytes_human(per_worker)}, "
            f"total≈{_bytes_human(total_est)}"
        )
