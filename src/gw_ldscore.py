"""
Stochastically estimate (partitioned) genome-wide LD scores. Some part of the code is modified from Eric Liu's script
"""
import utils
import numpy as np
import pandas as pd
from bed_reader import open_bed
from tqdm import tqdm
import sys
import gc
import os
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import gwldcore

from contextlib import contextmanager

_THREAD_LOCAL = threading.local()

def apply_env(cfg: dict) -> int:
    """
    Apply low-level environment knobs from a plain dict and enforce BLAS/OMP threads.

    Recognized keys (all optional):
      - num_threads: int               # single knob used for BOTH OMP and BLAS
      - ctile: int                     # direct CTILE override (cols)
      - ctile_mb: int                  # CTILE memory budget (MiB)
      - ctile_l3pct: float             # 0.0–1.0 fraction of per-socket L3 cache
      - malloc_arena_max: int          # glibc arena hygiene (default is 2)
      - malloc_trim_threshold: int     # trim threshold bytes (default is 131072)
      - malloc_mmap_threshold: int     # mmap threshold bytes (default is 131072)

    Returns:
      actual_blas_threads (int): detected BLAS threads after enforcement.
    """

    def _cpu_count_affinity() -> int:
        try:
            return len(os.sched_getaffinity(0))  # respects cgroup/pin
        except Exception:
            return os.cpu_count() or 1

    # Probe BLAS threads after we set things, using threadpoolctl if available.
    def _detect_blas_threads_fallback() -> int:
        for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
            v = os.environ.get(k)
            if v and v.isdigit():
                return max(1, int(v))
        return _cpu_count_affinity()

    def _detect_blas_threads() -> int:
        try:
            from threadpoolctl import threadpool_info  # type: ignore
            tot = 0
            for lib in threadpool_info():
                api = str(lib.get("internal_api", "")).lower()
                path = str(lib.get("filepath", "")).lower()
                if any(k in api for k in ("openblas","mkl","blis","accelerate")) or \
                   any(k in path for k in ("openblas","mkl","blis","veclib","accelerate")):
                    tot += int(lib.get("num_threads", 0))
            return tot if tot > 0 else _detect_blas_threads_fallback()
        except Exception:
            return _detect_blas_threads_fallback()

    # -------- 1) Threads (single knob for both OMP & BLAS) --------
    n_threads = cfg.get("num_threads")
    if n_threads is None or int(n_threads) <= 0:
        n_threads = _cpu_count_affinity()
    n_threads = max(1, int(n_threads))

    # OpenMP (phase 1) & BLAS (phase 2) will be re-tuned around calls,
    # but we set sane process-wide defaults here:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    # BLAS vendors
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_DYNAMIC"] = "0"
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["BLIS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)

    # Also try to enforce into already loaded libraries
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n_threads)
    except Exception:
        pass
    # OpenBLAS direct calls (attempt several sonames/symbols)
    try:
        import ctypes
        for soname in ("libopenblas.so", "libopenblas.so.0", "libopenblas64_.so", "libopenblas64_.so.0"):
            try:
                lib = ctypes.CDLL(soname)
                for sym in ("openblas_set_num_threads", "openblas_set_num_threads64_"):
                    try:
                        getattr(lib, sym)(int(n_threads))
                        break
                    except AttributeError:
                        continue
                break
            except OSError:
                continue
    except Exception:
        pass
    # threadpoolctl as a final clamp
    tpctx = None
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        tpctx = threadpool_limits(limits=n_threads, user_api="blas")
    except Exception:
        tpctx = None
        
    actual = _detect_blas_threads()
    os.environ["SUMMIT_BLAS_THREADS"] = str(actual)

    # -------- 2) C++ tiler/env knobs --------
    if cfg.get("ctile") is not None:
        os.environ["SUMMIT_CTILE"] = str(int(cfg["ctile"]))
    else:
        os.environ.pop("SUMMIT_CTILE", None)

    if cfg.get("ctile_mb") is not None:
        os.environ["SUMMIT_CTILE_MB"] = str(int(cfg["ctile_mb"]))
    else:
        os.environ.pop("SUMMIT_CTILE_MB", None)

    if cfg.get("ctile_l3pct") is not None:
        os.environ["SUMMIT_CTILE_L3PCT"] = f"{float(cfg['ctile_l3pct']):.3f}"
    else:
        os.environ.pop("SUMMIT_CTILE_L3PCT", None)

    sockets = cfg.get("sockets")
    if sockets is not None:
        os.environ["SUMMIT_SOCKETS"] = str(int(sockets))
    
    # -------- 3) Malloc hygiene (optional) --------
    if cfg.get("malloc_arena_max") is not None:
        os.environ["MALLOC_ARENA_MAX"] = str(int(cfg["malloc_arena_max"]))
    if cfg.get("malloc_trim_threshold") is not None:
        os.environ["MALLOC_TRIM_THRESHOLD_"] = str(int(cfg["malloc_trim_threshold"]))
    if cfg.get("malloc_mmap_threshold") is not None:
        os.environ["MALLOC_MMAP_THRESHOLD_"] = str(int(cfg["malloc_mmap_threshold"]))

    return actual


def set_parallelism(omp_threads: int | None = None, blas_threads: int | None = None):
    """
    Set OpenMP and BLAS vendor threads independently.
    Call early (before spawning Pools) and around phases to avoid nested teams.
    """
    import os
    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))
        os.environ["OMP_DYNAMIC"] = "FALSE"

    if blas_threads is not None:
        n = max(1, int(blas_threads))
        os.environ["MKL_DYNAMIC"] = "FALSE"
        os.environ["MKL_NUM_THREADS"] = str(n)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n)
        os.environ["BLIS_NUM_THREADS"] = str(n)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
        try:
            import mkl  # type: ignore
            mkl.set_num_threads(n)
        except Exception:
            pass
        try:
            # OpenBLAS (if accessible)
            import ctypes
            for so in ("libopenblas.so", "libopenblas64_.so"):
                try:
                    ctypes.CDLL(so).openblas_set_num_threads(n)
                    break
                except OSError:
                    continue
        except Exception:
            pass

def _round_up_to(x, gran):
    return int(((x + gran - 1) // gran) * gran)

def _build_balanced_vtiles(V, vmax, gran=64, max_tiles=4):
    """
    Split V vectors into ~equal tiles, each ≤ vmax, rounded to `gran` (e.g., 64),
    minimizing the worst tail. Returns list of (v_start, v_count).
    """
    vmax = max(gran, (vmax // gran) * gran)
    if vmax >= V:
        return [(0, V)]
    # minimal number of tiles that respects vmax
    tiles = int(math.ceil(V / vmax))
    tiles = min(max_tiles, max(2, tiles))  # cap to avoid too many passes

    # base and remainder for balanced split
    q, r = divmod(V, tiles)
    # round each tile to granularity, but never exceed vmax
    sizes = []
    for t in range(tiles):
        sz = q + (1 if t < r else 0)
        sz = min(vmax, _round_up_to(sz, gran))
        sizes.append(sz)

    # adjust if rounding up overshot total by a lot
    total = sum(sizes)
    # if total > V, bleed off the excess from the last tiles
    over = total - V
    t = len(sizes) - 1
    while over > 0 and t >= 0:
        bleed = min(over, sizes[t] - max(gran, q))  # don't shrink below ~base
        bleed = (bleed // gran) * gran
        if bleed > 0:
            sizes[t] -= bleed
            over -= bleed
        t -= 1

    # build (start,count)
    vtiles, v0 = [], 0
    for sz in sizes:
        if sz <= 0: continue
        if v0 + sz > V: sz = V - v0
        if sz <= 0: break
        vtiles.append((v0, sz))
        v0 += sz
    # final guard
    if v0 < V:
        vtiles.append((v0, V - v0))
    return vtiles

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

def _canonical_bfile_prefix(x: str) -> str:
    """Return PLINK bfile prefix: strip only trailing .bed/.bim/.fam if present; otherwise leave as-is."""
    s = str(x)
    for ext in (".bed", ".bim", ".fam"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


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
                low_level,
                covar_path=None,
                num_vecs=10,
                step_size=1000,
                seed=None,
                verbose=False,
                dtype='float32',
                num_threads: int = 4,
                eps_var: float = 1e-10,
                rand_samp=None, # float in (0,1] or int in [100, N]
                ddof = 1):
        
        self.tune = apply_env(low_level)
        
        # Cap BLAS threads before any Pools spawn
        self.num_threads = int(num_threads)

        self.eps_var = float(eps_var)
        prefix = _canonical_bfile_prefix(bed_path)
        self.bed_prefix = os.path.abspath(prefix)     # optional: make absolute for stability
        self.fam_path   = self.bed_prefix + ".fam"
        self.bim_path   = self.bed_prefix + ".bim"

        self.G = open_bed(self.bed_prefix + ".bed")
        self.nsamp, self.nsnps = self.G.shape
        self.nvecs = num_vecs
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
        self._read_bim(self.bim_path)
        if annot_path is not None:
            self._read_annot(annot_path)
        else:
            self._read_annot(None)

        # covariates → orthonormal Q (C) and Q^T (cov_R); drop NA rows
        if covar_path is not None:
            C, R, keep_idx_global = read_cov(
                cov_filename=covar_path,
                fam_filename=self.fam_path,
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
    
    def _precompute_residual_variances(self):
        """
        Precompute inv sqrt residual variances per SNP, consistent with XtXz.
        Parallelized over SNP chunks using a thread pool. Each worker:
        - Opens its own .bed handle (thread-local)
        - Reads & standardizes the block
        - Optionally projects with covariates
        - Returns 1/sqrt(Var) for [s:e)

        We also cap BLAS threads to 1 within the parallel region to avoid
        oversubscription (NumPy GEMMs in the worker), and choose the number
        of workers conservatively based on available memory.
        """

        row_sel = self.row_sel if self.row_sel is not None else slice(None)
        inv = np.empty(self.nsnps, dtype=self.dtype)

        # Form SNP blocks
        chunks = [(s, min(self.nsnps, s + self.step_size))
                for s in range(0, self.nsnps, self.step_size)]
        if not chunks:
            self.log._log("[warn] No SNP chunks formed; returning zeros.")
            return np.zeros(self.nsnps, dtype=self.dtype)

        # Canonicalized bed/fam prefix (set in __init__)
        bed_prefix = getattr(self, "bed_prefix", None)
        if bed_prefix is None:
            # Fallback (shouldn't happen if __init__ set it)
            bp = Path(str(getattr(self.G, "filename", None)
                        or getattr(self.G, "filepath", None) or ""))
            if bp.suffix == ".bed":
                bp = bp.with_suffix("")
            bed_prefix = str(bp)

        # Worker args (read-only)
        dtype = self.dtype
        ddof  = int(self.ddof)
        C     = self.C if self.C is not None else None
        R     = self.cov_R if self.C is not None else None
        N_eff = float(self.N_eff if self.C is not None else self.nsamp)
        eps   = float(self.eps_var)

        # Choose number of workers conservatively to avoid RAM spikes
        # Rough per-chunk footprint ≈ N * L * itemsize * 3 (G, tmp/proj, Y)
        try:
            import psutil
            avail = int(psutil.virtual_memory().available)
        except Exception:
            avail = None

        b = int(np.dtype(dtype).itemsize)
        L = int(min(self.step_size, self.nsnps))
        est_per_chunk = max(1, self.nsamp * L * b * 3)
        nominal = max(1, int(self.num_threads))
        if avail is not None:
            max_by_mem = max(1, int(avail // est_per_chunk))
        else:
            max_by_mem = nominal
        n_workers = max(1, min(nominal, max_by_mem, os.cpu_count() or 1))
        # Be extra safe: don't spin more workers than chunks
        n_workers = min(n_workers, len(chunks))

        self.log._log(f"[resvar] Using {n_workers} workers "
                    f"(~{_bytes_human(est_per_chunk)} per task; avail={_bytes_human(avail)})")

        # Limit BLAS threads inside the pool to 1 to avoid oversubscription
        # (NumPy/MKL/OpenBLAS will otherwise multi-thread inside each worker)
        limiter = None
        try:
            from threadpoolctl import threadpool_limits  # type: ignore
            limiter = threadpool_limits(limits=1)
        except Exception:
            limiter = None  # OK if unavailable

        # Execute in parallel
        try:
            if limiter is None:
                # simple context manager that does nothing
                from contextlib import contextmanager
                @contextmanager
                def _nullctx():
                    yield
                ctx = _nullctx()
            else:
                ctx = limiter

            with ctx:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [
                        ex.submit(
                            _resvar_worker_thread,
                            span,            # (s, e)
                            bed_prefix,
                            row_sel,
                            dtype,
                            ddof,
                            C, R,
                            N_eff,
                            eps
                        )
                        for span in chunks
                    ]

                    # Fill results as workers complete
                    for fut in as_completed(futures):
                        s, e, inv_part = fut.result()
                        inv[s:e] = inv_part

        except Exception as e:
            # Fallback to serial path on any failure
            self.log._log(f"[resvar] Parallel precompute failed ({e}); falling back to serial.")
            for s, e in chunks:
                G = self.G.read(index=np.s_[row_sel, s:e], dtype=dtype)  # (N × L)
                means = np.nanmean(G, axis=0, dtype=dtype)
                stds  = np.nanstd( G, axis=0, dtype=dtype, ddof=ddof)
                stds[stds == 0] = 1.0
                G = (G - means) / stds
                np.nan_to_num(G, copy=False)

                if C is not None and R is not None:
                    tmp = R @ G
                    G   = G - (C @ tmp)
                    del tmp
                    denom = float(self.N_eff)
                else:
                    denom = float(self.nsamp)

                var = np.sum(G * G, axis=0, dtype=dtype) / max(denom - 1.0, 1.0)
                inv[s:e] = (1.0 / np.sqrt(np.maximum(var, self.eps_var))).astype(dtype, copy=False)

        return inv


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
            base_cols = {'CHR', 'BP', 'SNP', 'CM'}
            if base_cols.issubset(set(df.columns)) and 'SNP' in df.columns:
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
        Streamed, balanced V-tiling pipeline with a single progress bar over (vtiles × blocks).

        Steps:
        0) Precompute per-SNP inv sqrt residual variances (right side).
        1) Build SNP blocks; precompute Kmax per block from annotation.
        2) Choose an initial V-tile from a memory budget, then BALANCE tiles evenly.
        3) For each balanced V-tile:
            Phase 1: build Xz_chunk (N × B·Vt) across all blocks (skip if Kmax==0)
            Phase 2: consume Xz_chunk across all blocks (skip if Kmax==0)
            Accumulate weighted by Vt
        4) Finalize: divide by total V, subtract baseline, save; print summaries.
        """
        import os, time, ctypes, numpy as np, pandas as pd
        from tqdm import tqdm

        self.log._log(f"num_vecs: {self.nvecs}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        if self.C is not None:
            self.log._log(f"Covariate-adjusted partial correlations (N_eff={self.N_eff}, p={self.p_eff}).")
        else:
            self.log._log("No covariates: standard LD scores (squared correlations).")

        # -------------------- Phase 0: per-SNP residual variances --------------------
        set_parallelism(omp_threads=self.num_threads, blas_threads=1)
        self.inv_sqrt_resvar_all = self._precompute_residual_variances()

        # -------------------- Build SNP blocks --------------------
        blocks = []
        for j in range(0, self.nsnps, self.step_size):
            s = j
            e = min(self.nsnps, j + self.step_size)
            blocks.append((s, e))
        self.nblks = len(blocks)

        # -------------------- Precompute Kmax per block (trust hints) ----------------
        kmax_per_block: list[int] = []
        for (s, e) in blocks:
            blk = self.annot[s:e]  # (L x B)
            Kmax = int((blk != 0).sum(axis=0).max())
            kmax_per_block.append(Kmax)
        if any(k == 0 for k in kmax_per_block):
            zc = sum(1 for k in kmax_per_block if k == 0)
            self.log._log(f"[info] {zc} block(s) have Kmax=0 (will be skipped).")
        if kmax_per_block:
            self.log._log(f"Kmax per block (min/median/max): "
                        f"{min(kmax_per_block)}/{int(np.median(kmax_per_block))}/{max(kmax_per_block)}")

        # -------------------- Choose initial V-chunk by memory budget ----------------
        # Target memory for Xz panel (N x (B*Vt)) in GiB; you can set self.target_xz_gib
        target_gib = float(getattr(self, "target_xz_gib", 16.0)) ## TODO: make an argparse option for this
        itemsize = np.dtype(self.dtype).itemsize
        denom = max(1, int(self.nsamp) * int(self.nbins) * itemsize)
        vtile_guess = int((target_gib * (1024**3)) // denom)
        # Clamp to [min, nvecs]; keep it reasonable for big runs
        vtile_guess = max(256, min(self.nvecs, vtile_guess))
        # If guess is too small/large, fall back to something sane
        if vtile_guess <= 0:
            vtile_guess = min(self.nvecs, 4096)

        # BALANCED TILING: number of tiles then even split
        ntiles = int(np.ceil(self.nvecs / vtile_guess))
        ntiles = max(1, ntiles)
        base = (self.nvecs // ntiles)
        rem  = (self.nvecs %  ntiles)
        vtiles = [((base + (1 if i < rem else 0) + 63)//64)*64 for i in range(ntiles)]
        # fix last tile to hit sum(V)
        vtiles[-1] = self.nvecs - sum(vtiles[:-1])
        vtiles = [v for v in vtiles if v > 0]
        assert sum(vtiles) == self.nvecs

        if len(vtiles) == 1:
            self.log._log(
                f"[auto_vchunk] dtype={self.dtype} target≈{target_gib:.1f} GiB, "
                f"v_tiles=[{vtiles[0]}] → Xz≈{(self.nsamp*self.nbins*vtiles[0]*itemsize)/(1024**3):.2f} GiB"
            )
        else:
            self.log._log(
                f"[auto_vchunk] dtype={self.dtype} target≈{target_gib:.1f} GiB, "
                f"v_tiles={vtiles[0]}×{(len(vtiles)-1)}+{vtiles[-1]} → Xz≈{(self.nsamp*self.nbins*vtiles[0]*itemsize)/(1024**3):.2f} GiB"
            )
        self.log._log(f"Streaming with BALANCED V-tiles: {vtiles} (total V = {self.nvecs})")

        # -------------------- Paths / common args --------------------
        bed_prefix = self.bed_prefix
        fam_path   = self.fam_path
        row_sel = self.row_sel if self.row_sel is not None else None
        ddof    = int(self.ddof)
        B       = int(self.nbins)

        # -------------------- Accumulators & scratch --------------------
        meansq_accum = np.zeros((self.nsnps, self.nbins), dtype=self.dtype, order='C')

        # Preallocate Xz_chunk once at max tile width = B * max(vtiles)
        Vmax      = max(vtiles) if vtiles else 0
        Xz_chunk  = np.zeros((self.nsamp, int(self.nbins) * int(Vmax)),
                            dtype=self.dtype, order='F')
        meansq_chunk = np.zeros_like(meansq_accum, dtype=self.dtype, order='C')

        # -------------------- Progress bar over vtiles × blocks ----------------------
        total_units = len(vtiles) * len(blocks)
        bar = tqdm(total=total_units, desc="GW-LD progress", unit="task", smoothing=0.2, miniters=1)

        # EMA for phase weight (fraction for Phase-1)
        ema_p1 = 0.0
        ema_p2 = 0.0
        w1 = 0.5  # neutral start

        try:
            v_start = 0
            for vt_idx, Vt in enumerate(vtiles):
                used_cols = B * Vt
                Xz_view = Xz_chunk[:, :used_cols]
                Xz_view.fill(0)

                # ---------------------- Phase 1 (chunked) ----------------------
                # OMP = many; BLAS = 1
                set_parallelism(omp_threads=self.num_threads, blas_threads=1)
                t1_total = 0.0
                
                for blk_idx, (s, e) in enumerate(blocks):
                    kmax_hint = int(kmax_per_block[blk_idx])

                    if kmax_hint == 0:
                        bar.update(1.0)
                        continue

                    annot_blk = np.ascontiguousarray(self.annot[s:e].astype(self.dtype, copy=False))
                    inv_right = np.ascontiguousarray(self.inv_sqrt_resvar_all[s:e].astype(self.dtype, copy=False))

                    t0 = time.perf_counter()
                    gwldcore.phase1_compute_Xz_bed_chunk(
                        bed_prefix=bed_prefix,
                        fam_path=fam_path,
                        blk_start=int(s), blk_end=int(e),
                        row_sel=row_sel,
                        ddof=ddof,
                        annot_blk=annot_blk,              # (L x B)
                        inv_right=inv_right,              # (L,)
                        v_start=int(v_start),             # seed offset
                        v_count=int(Vt),
                        kmax_hint=kmax_hint,              # TRUST
                        rand_dist=self.rand_dist,
                        seed=self.root_seed,
                        Xz2d_chunk=Xz_view,               # (N x (B*Vt)) Fortran, V-major
                        project_right=False,
                        C=(self.C if self.C is not None else None),
                        R=(self.cov_R if self.C is not None else None)
                    )
                    t1_total += (time.perf_counter() - t0)

                    bar.update(w1)

                # ---------------------- Phase 2 ----------------------
                meansq_chunk.fill(0)
                t2_total = 0.0

                # OMP = 1; BLAS = many
                set_parallelism(omp_threads=1, blas_threads=self.num_threads)
                for blk_idx, (s, e) in enumerate(blocks):
                    kmax_hint = int(kmax_per_block[blk_idx])
                    if kmax_hint == 0:
                        continue

                    inv_left = np.ascontiguousarray(self.inv_sqrt_resvar_all[s:e].astype(self.dtype, copy=False))
                    if self.C is not None:
                        N_denom = int(self.N_eff)             # C++ uses N_eff - 1 inside
                    else:
                        N_denom = int(self.nsamp - self.ddof + 1)

                    t0 = time.perf_counter()
                    gwldcore.phase2_compute_XtXz_bed(
                        bed_prefix=bed_prefix,
                        fam_path=fam_path,
                        blk_start=int(s), blk_end=int(e),
                        row_sel=row_sel,
                        ddof=ddof,
                        inv_left=inv_left,            # (L,)
                        nvecs=int(Vt),                # mean over THIS tile only
                        vchunk=int(Vt),               # (unused in C++; keep for signature)
                        Xz2d=Xz_view,                 # (N x (B*Vt))
                        meansq=meansq_chunk,          # (M x B), per-block rows overwritten
                        C=(self.C if self.C is not None else None),
                        R=(self.cov_R if self.C is not None else None),
                        N_denom=int(N_denom)
                    )
                    t2_total += (time.perf_counter() - t0)

                    bar.update(1.0 - w1)

                # Weighted combine across tiles
                meansq_accum += (meansq_chunk * Vt)

                # Adapt phase weight (EMA)
                ema_p1 = 0.85 * ema_p1 + 0.15 * max(t1_total, 1e-9)
                ema_p2 = 0.85 * ema_p2 + 0.15 * max(t2_total, 1e-9)
                w1 = float(ema_p1 / (ema_p1 + ema_p2))
                bar.set_postfix_str(f"tile {vt_idx+1}/{len(vtiles)} | w1={w1:.2f} | P1={t1_total:.1f}s P2={t2_total:.1f}s")

                # Housekeeping
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                v_start += Vt  # advance seed offset

        finally:
            try:
                bar.close()
            except Exception:
                pass

        # ---------------------- Finalize & save ----------------------
        meansq = (meansq_accum / float(self.nvecs)).astype(self.dtype, copy=False)

        # Baseline subtraction: classic correlation null M_k / N_denom (ddof-aware)
        N_denom = float(self.N_eff - 1.0 if self.C is not None else self.nsamp - self.ddof)
        self.log._log("Applying correlation null: subtracting M_k / N_denom per bin.")
        meansq -= (self.nsnps_bin / N_denom).astype(meansq.dtype, copy=False)[None, :]

        self.gwldscore = meansq.astype(np.float64, copy=False)

        self.log._log(f"Saving the genome-wide (partitioned) LD scores into: {self.outpath}.gw.ldscore.gz")
        snpcols = ['CHR','SNP','BP']
        if self.snplist is None:
            self.snpdf = pd.DataFrame(np.nan*np.ones((self.nsnps, 3)), columns=snpcols)
        else:
            self.snpdf = self.snplist[['CHR','SNP','BP']].copy()
            self.snpdf.columns = snpcols

        scores_df = pd.DataFrame(self.gwldscore, columns=self.l2cols)
        out_df = pd.concat([self.snpdf, scores_df], axis=1)
        out_df.to_csv(f'{self.outpath}.gw.ldscore.gz', index=False, compression='gzip', sep='\t', float_format='%.6f')


        # Summaries (best-effort)
        try:
            desc = scores_df.describe(percentiles=[0.25,0.5,0.75]).loc[['count','mean','std','min','25%','50%','75%','max']]
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
            desc2 = pd.Series(row_sums).describe(percentiles=[0.25, 0.5, 0.75])
            self.log._log("\nSummary of Annotation Matrix Row Sums")
            with pd.option_context('display.float_format', '{:.4f}'.format):
                ordered = ['count','mean','std','min','25%','50%','75%','max']
                lines = [f"{k:<6} {desc2[k]:.4f}" for k in ordered]
                self.log._log("\n".join(lines))
        except Exception as e:
            self.log._log(f"[warn] Failed to compute summary stats / correlation: {e}")

        self.end_time = utils._get_time()
        self.log._log(f"Calculation of genome-wide LD score ended at "+utils._get_timestr(self.end_time))
        self.runtime = self.end_time - self.start_time
        self.log._log("Runtime: "+format(self.runtime, '.3f')+
                    f" s ({self.runtime//3600} hr {(self.runtime%3600)//60} m {(self.runtime%60):.3f} s)")
        self.log._save_log(self.outpath+".gw.log")