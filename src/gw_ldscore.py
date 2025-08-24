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
import math
import os

def _bytes_human(n):
    # human-readable bytes
    if n is None: return "n/a"
    if n < 1024: return f"{n} B"
    for unit in ["KB","MB","GB","TB","PB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:,.2f} {unit}"
    return f"{n:,.2f} EB"

def _rss_snapshot(label, logger=None, include_children=True):
    """Log current RSS (resident memory). Tries psutil, then resource (Linux/Mac)."""
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
        # Fallback: resource (ru_maxrss is kB on Linux, bytes on macOS)
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF)
            # Linux reports kB, macOS bytes; detect via magnitude
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
_g_Xz2d = None
_g_meansq = None
_g_xz_locks = None
_g_shm_xz = None
_g_shm_ms = None

def _worker_cleanup():
    # drop array views first
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

    # ensure clean shutdown in the worker
    atexit.register(_worker_cleanup)


def read_cov(
        cov_filename: str,
        fam_filename: str,
        std: bool = True,
        cov_impute_method: str = "ignore",
        one_hot_conversion: bool = False,
        categorical_threshold: int = 100,
        logger = None,
        verbose = False
    ):
    """
    1) Read PLINK .fam to get FID/IID sample order.
    2) Read covariate file, merge on FID/IID (error if mismatch).
    3) Drop FID, IID, handle missingness/imputation.
    4) Optionally one-hot encode categoricals.
    5) If std=True, center & scale each covariate column.
    6) Return:
         C : (n_samples x n_covariates) array,
         R : (n_covariates x n_samples) regression matrix = (C^T C)^{-1} C^T
    """
    # 1) load .fam
    fam = pd.read_csv(
        fam_filename,
        sep=r'\s+',
        header=None,
        usecols=[0,1],
        names=['FID','IID']
    )

    # 2) load covariate file
    cov = pd.read_csv(cov_filename, sep=r'\s+')
    merged = fam.merge(cov, on=['FID','IID'], how='right', indicator=True)
    missing = merged.loc[merged['_merge'] != 'both', ['FID','IID']]
    if not missing.empty:
        if verbose:
            raise ValueError(
                f"Samples in {fam_filename} not found in {cov_filename}:\n"
                f"{missing.to_string(index=False)}"
            )
        else:
            raise ValueError(
                f"!!! {len(missing)} Samples are not found in {cov_filename} !!!"
            )
    merged = merged.drop(columns=['_merge'])

    # 3) drop IDs, handle missingness
    df = merged.drop(columns=['FID','IID']).copy()
    n_covariates = len(df.columns)
    is_na = df.isin(['NA', -9]).any(axis=1)
    if cov_impute_method == "ignore":
        if is_na.any():
            idx = np.where(is_na)[0].tolist()
            raise ValueError(f"Missing covariate entries at rows: {idx}")
    else:
        df.replace({'NA': np.nan, -9: np.nan}, inplace=True)
        for col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    # 4) one-hot encode if requested
    if one_hot_conversion:
        for col in df.columns:
            if df[col].nunique() <= categorical_threshold:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = df.drop(columns=[col]).join(dummies)

    # 5) standardize if requested
    if std:
        df = (df - df.mean()) / df.std(ddof=1)

    C = df.values  # shape (n_samples, n_cov)

    # 6) build regression matrix R = (C^T C)^{-1} C^T
    CtC = C.T @ C
    inv_CtC = np.linalg.inv(CtC)
    R = inv_CtC @ C.T  # shape (n_cov, n_samples)
    
    if logger:
        logger._log(
            f"Read {cov_filename} for {n_covariates} covariates (samples merged with {fam_filename}).\n"
            f"C shape={C.shape}, R shape={R.shape}, std={std}, one_hot={one_hot_conversion}, verbose={verbose}"
        )

    return C, R


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
                 dtype='float32'):
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

        # read & build covariate residualizer
        if covar_path is not None:
            fam_file = bed_path + ".fam"
            self.C, self.cov_R = read_cov(
                cov_filename       = covar_path,
                fam_filename       = fam_file,
                std                = True,
                cov_impute_method  = "ignore",
                one_hot_conversion = False,
                categorical_threshold = 100,
                logger             = self.log,
                verbose            = self.verbose
            )
            self.C = np.asarray(self.C, dtype=self.dtype, order='F')
            self.cov_R = np.asarray(self.cov_R, dtype=self.dtype, order='F')
        else:
            self.C = None
            self.cov_R = None
            self.log._log("No covariate correction will be applied.")

        self.root_seed = seed
        self.outpath = out_path

    def _compute_Xz_blk(self, blk_idxs):
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

        # read + standardize
        geno = self.G.read(index=np.s_[:, blk_start:blk_end], dtype=self.dtype)
        means = np.nanmean(geno, axis=0, dtype=self.dtype)
        stds  = np.nanstd(geno, axis=0, dtype=self.dtype)
        stds[stds == 0] = 1.0
        geno  = (geno - means) / stds
        np.nan_to_num(geno, copy=False)
        geno = np.asarray(geno, dtype=self.dtype, order='F')

        # regress out covariates if present
        if self.C is not None:
            geno -= self.C.dot(self.cov_R.dot(geno))

        Zs = np.asarray(Zs, order='F', dtype=self.dtype)
        gemm = fblas.sgemm if self.dtype is np.float32 else fblas.dgemm

        for k, binidx in enumerate(idxs):
            if len(binidx) == 0:
                continue
            A = np.asfortranarray(geno[:, binidx], dtype=self.dtype)   # (N × K)
            B = np.asfortranarray(Zs[binidx, :],  dtype=self.dtype)    # (K × V)
            c_view = _g_Xz2d[:, k*self.nvecs:(k+1)*self.nvecs]         # (N × V), Fortran view
            with _g_xz_locks[k]:
                # c := 1.0*A@B + 1.0*c   (accumulate)
                gemm(1.0, A, B, c=c_view, beta=1.0, overwrite_c=1)
        return 1
   

    def _compute_XtXz_blk(self, blk_idx):
        """
        For blk genotype, multiply with X_k z to get XtXkz.
        """
        blk_start, blk_end = blk_idx
        gemm = fblas.sgemm if self.dtype is np.float32 else fblas.dgemm

        geno = self.G.read(index=np.s_[:, blk_start:blk_end], dtype=self.dtype)
        means = np.nanmean(geno, axis=0, dtype=self.dtype)
        stds = np.nanstd(geno, axis=0, dtype=self.dtype)
        stds[stds == 0] = 1.0
        geno = (geno-means)/stds
        np.nan_to_num(geno, copy=False)
        geno  = np.asarray(geno, dtype=self.dtype, order='F')

        if self.C is not None:
            geno = geno - self.C.dot(self.cov_R.dot(geno))

        block_len = blk_end - blk_start
        work = np.empty((block_len, self.nvecs), dtype=self.dtype, order='F')

        # For each bin, multiply and write means directly into shared meansq
        for k in range(self.nbins):
            Xz_k = _g_Xz2d[:, k*self.nvecs:(k+1)*self.nvecs]  # (N × V), Fortran view
            gemm(1.0, geno, Xz_k, c=work, beta=0.0, trans_a=True, overwrite_c=1)  # geno^T @ Xz_k
            work *= (1.0 / self.nsamp)
            _g_meansq[blk_start:blk_end, k] = np.mean(work * work, axis=1)
        return 1

    def _read_annot(self, annot_path):
        """
        Read in the annotation. If the file includes a header, save it as the names for the annotations.
        If not, then have dummy names and read in the annotation.
        """
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
        """
        partition snp indices by annotation
        """
        return [snpidx[annot[:, c] == 1] for c in range(self.nbins)]


    def _compute_ldscore(self):
        """
        Use multi-processing to calculate the X_j^T X_k Z.
        General sketch: read in each block of genotype, calculate X_k Z for that blk. Aggregate X_k through all blks.
        Then re-read each blk from the start, multiply by the previous result (loop over k) to get X_j ^ T X_k (no need for agg this time).
        """
        self.start_time = utils._get_time()
        self.log._log("Genome-wide LD score calculation started at: "+utils._get_timestr(self.start_time))
        self.log._log(f"num_vecs: {self.nvecs}, num_workers: {self.nworkers}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
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
        
        # ----------------- allocate shared Xz (2D Fortran) and meansq -----------------
        itemsize = np.dtype(self.dtype).itemsize
        xz_shape2d = (self.nsamp, self.nvecs * self.nbins)   # Fortran 2D
        ms_shape   = (self.nsnps, self.nbins)

        shm_xz = shared_memory.SharedMemory(create=True, size=int(np.prod(xz_shape2d)) * itemsize)
        # Parent's view (Fortran)
        self.Xz2d = np.frombuffer(shm_xz.buf, dtype=self.dtype, count=xz_shape2d[0]*xz_shape2d[1]).reshape(xz_shape2d, order='F')
        self.Xz2d.fill(0)

        shm_ms = shared_memory.SharedMemory(create=True, size=int(np.prod(ms_shape)) * itemsize)
        self.meansq = np.ndarray(ms_shape, dtype=self.dtype, buffer=shm_ms.buf)
        self.meansq.fill(0)

        # per-bin locks
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

        # ----------------- Phase 2: fill meansq in shared memory -----------------
        self._print_expected_mem('XtXz')
        with mp.Pool(self.nworkers,
                     initializer=_init_shared,
                     initargs=(shm_xz.name, xz_shape2d, shm_ms.name, ms_shape,
                               np.dtype(self.dtype).str, xz_locks)) as pool:
            with tqdm(total=self.nblks, desc='Calculating XtXz') as pbar:
                for _ in pool.imap_unordered(self._compute_XtXz_blk, XtXz_input):
                    pbar.update()

        _rss_snapshot("after XtXz", self.log)
        self.XtXz_time = utils._get_time()
        self.log._log("Calculation of XtXz (for each partition) completed. Runtime: "+format(self.XtXz_time - self.Xz_time, '.3f')+" s")

        # ----------------- finish / save -----------------
        self.log._log("Converting XtXz into genome-wide (partitioned) LD scores.")
        self.gwldscore = self.nsamp/(self.nsamp+1) * (self.meansq - self.nsnps_bin / self.nsamp)
        self.gwldscore = self.gwldscore.astype(np.float64, copy=False)

        self.log._log(f"Saving the genome-wide (partitioned) LD scores into: {self.outpath}.gw.ldscore.gz")
        snpcols = ['CHR', 'SNP', 'BP']
        if (self.snplist is None):
            self.snpdf = pd.DataFrame(np.nan*np.ones((self.nsnps, 3)), columns=snpcols)
        else:
            self.snpdf = self.snplist.iloc[:, :3]
            self.snpdf.columns = snpcols
        
        self.gwldscore = pd.DataFrame(self.gwldscore, columns = self.l2cols)
        self.gwldscore = pd.concat([self.snpdf, self.gwldscore], axis=1)
        self.gwldscore.to_csv(f'{self.outpath}.gw.ldscore.gz', index=False, compression='gzip', sep='\t', float_format='%.3f')

        self.end_time = utils._get_time()
        self.log._log(f"Calculation of genome-wide LD score ended at "+utils._get_timestr(self.end_time))
        self.log._log("Runtime: "+format(self.end_time - self.start_time, '.3f')+" s")
        self.log._save_log(self.outpath+".gw.log")

        # ----------------- free shared memory -----------------
        self.Xz2d = None
        self.meansq = None
        gc.collect()
        shm_xz.close(); shm_xz.unlink()
        shm_ms.close(); shm_ms.unlink()
        _rss_snapshot("post-cleanup", self.log)
    
    def _print_expected_mem(self, phase, block_len=None, k_max=None):
        """
        Rough upper-bound memory accounting for this run.
        phase: 'Xz' or 'XtXz'
        block_len: defaults to min(step_size, nsnps) for estimates
        k_max: optional per-bin SNPs in block; if None we ignore A/B temps
        """
        b = np.dtype(self.dtype).itemsize
        B, N, M, V, S, W = self.nbins, self.nsamp, self.nsnps, self.nvecs, self.step_size, self.nworkers
        L = block_len if block_len is not None else min(S, M)

        # Shared (parent) arrays
        xz_bytes = N * (V * B) * b           # Xz2d shape (N, V*B)
        ms_bytes = M * B * b                 # meansq (M, B)

        # Per-worker temps
        geno_blk = N * L * b                 # geno block (N × L)
        if phase == 'Xz':
            # we accumulate in-place; A/B temps depend on per-bin K.
            # If you pass k_max, include a pessimistic bound; else omit.
            per_worker = geno_blk
            if k_max is not None:
                A = N * k_max * b            # N × K
                Btmp = k_max * V * b         # K × V
                per_worker += max(A, 0) + max(Btmp, 0)
        else:  # XtXz
            work = L * V * b                 # (L × V) buffer
            per_worker = geno_blk + work

        total_est = xz_bytes + ms_bytes + W * per_worker

        self.log._log(
            f"[expected {phase}] dtype={self.dtype}, B={B}, N={N}, M={M}, V={V}, S={S}, W={W} "
            f"→ parent(Xz2d+meansq)≈{_bytes_human(xz_bytes + ms_bytes)}, "
            f"per-worker temps≈{_bytes_human(per_worker)}, "
            f"total≈{_bytes_human(total_est)}"
        )
