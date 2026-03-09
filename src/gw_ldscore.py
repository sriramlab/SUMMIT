"""
Stochastically estimate (partitioned) genome-wide LD scores. Some part of the code is modified from Eric Liu's script
"""
import utils
import math
import numpy as np
import pandas as pd
from bed_reader import open_bed
from tqdm import tqdm
import sys, shutil
import gc
import os, psutil
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import gwldcore

from contextlib import contextmanager, nullcontext
from threadpoolctl import threadpool_limits

_THREAD_LOCAL = threading.local()


## Device helpers
def _parse_device_str(s: str) -> tuple[str, int | None]:
    s = (s or "cpu").strip().lower()
    if s == "cpu":
        return "cpu", None
    if s.startswith("cuda"):
        if ":" in s:
            try:
                return "cuda", int(s.split(":", 1)[1])
            except ValueError:
                return "cuda", None
        return "cuda", None
    return "cpu", None


def _pick_cuda_index_auto(gcu):
    gpus = gcu.list_gpus()
    if not gpus:
        raise RuntimeError("CUDA requested but no GPUs are visible.")
    best = max(gpus, key=lambda d: int(d["free_bytes"]))
    return int(best["id"]), int(best["free_bytes"]), int(best["total_bytes"])


def apply_env(cfg: dict) -> int:
    import os, sys, shutil, ctypes

    def maybe_wrap_with_numactl(mode: str | None, nodes: str = "all") -> None:
        if not mode or os.name != "posix":
            return
        if os.environ.get("SUMMIT_NUMACTL_WRAPPED") == "1":
            return
        exe = shutil.which("numactl")
        if not exe:
            return
        flag = {
            "interleave":  "--interleave",
            "membind":     "--membind",
            "cpunodebind": "--cpunodebind",
            "preferred":   "--preferred",
        }.get(str(mode).lower())
        if not flag:
            return
        os.environ["SUMMIT_NUMACTL_WRAPPED"] = "1"
        args = [exe, f"{flag}={nodes}", sys.executable, *sys.argv]
        os.execv(exe, args)

    def _cpu_set_allowed():
        try:
            return set(os.sched_getaffinity(0))
        except Exception:
            return set(range(os.cpu_count() or 1))

    def _expand_affinity_to_all_allowed():
        if os.name != "posix" or not hasattr(os, "sched_setaffinity"):
            return
        if not bool(cfg.get("force_affinity_all", False)):
            return

        target = None
        try:
            with open("/sys/devices/system/cpu/online", "r") as f:
                s = f.read().strip()
            cpus = set()
            for part in s.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    a, b = int(a), int(b)
                    cpus.update(range(a, b + 1))
                else:
                    cpus.add(int(part))
            target = cpus
        except Exception:
            target = set(range(os.cpu_count() or 1))

        try:
            os.sched_setaffinity(0, target)
        except Exception:
            pass

    def _cpu_count_affinity() -> int:
        return len(_cpu_set_allowed())

    def _detect_blas_threads_fallback() -> int:
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
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
                if any(k in api for k in ("openblas", "mkl", "blis", "accelerate")) or \
                   any(k in path for k in ("openblas", "mkl", "blis", "veclib", "accelerate")):
                    tot += int(lib.get("num_threads", 0))
            return tot if tot > 0 else _detect_blas_threads_fallback()
        except Exception:
            return _detect_blas_threads_fallback()

    maybe_wrap_with_numactl(mode=cfg.get("numa_mode"), nodes=str(cfg.get("numa_nodes", "all")))
    _expand_affinity_to_all_allowed()

    n_aff = _cpu_count_affinity()
    if cfg.get("num_threads") is not None:
        n_threads = max(1, min(int(cfg["num_threads"]), n_aff))
    else:
        n_threads = max(1, n_aff)

    if "decode_threads" in cfg and cfg["decode_threads"] is not None:
        dec = int(cfg["decode_threads"])
        dec = max(1, min(dec, n_aff))
    else:
        cap = int(cfg.get("decode_threads_cap", 16))
        cap = max(1, cap)
        dec = min(cap, n_aff)

    decode_mem_cap_mb = int(cfg.get("decode_mem_cap_mb", 2048))
    if decode_mem_cap_mb < 64:
        decode_mem_cap_mb = 64
    os.environ["SUMMIT_DECODE_THREADS"] = str(dec)
    os.environ["SUMMIT_DECODE_MEM_CAP_MB"] = str(decode_mem_cap_mb)

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ.setdefault("OMP_PROC_BIND", str(cfg.get("omp_proc_bind", "spread")))
    os.environ.setdefault("OMP_PLACES",    str(cfg.get("omp_places", "threads")))
    os.environ.setdefault("KMP_BLOCKTIME", str(cfg.get("kmp_blocktime", 0)))
    os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX512")

    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n_threads)
    os.environ["OPENBLAS_DYNAMIC"] = "0"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n_threads)
    except Exception:
        pass

    try:
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

    actual = _detect_blas_threads()
    os.environ["SUMMIT_BLAS_THREADS"] = str(actual)

    if cfg.get("q_panel") is not None:
        os.environ["SUMMIT_P2_QPANEL"] = str(int(cfg["q_panel"]))
    else:
        os.environ.pop("SUMMIT_P2_QPANEL", None)

    if cfg.get("reduce_blk") is not None:
        os.environ["SUMMIT_P2_IBLK"] = str(int(cfg["reduce_blk"]))
    else:
        os.environ.pop("SUMMIT_P2_IBLK", None)

    if cfg.get("reduce_threads") is not None:
        os.environ["SUMMIT_P2_REDUCE_THREADS"] = str(int(cfg["reduce_threads"]))
    else:
        os.environ.pop("SUMMIT_P2_REDUCE_THREADS", None)

    sockets = cfg.get("sockets")
    if sockets is not None:
        os.environ["SUMMIT_SOCKETS"] = str(int(sockets))

    if cfg.get("malloc_arena_max") is not None:
        os.environ["MALLOC_ARENA_MAX"] = str(int(cfg["malloc_arena_max"]))
    if cfg.get("malloc_trim_threshold") is not None:
        os.environ["MALLOC_TRIM_THRESHOLD_"] = str(int(cfg["malloc_trim_threshold"]))
    if cfg.get("malloc_mmap_threshold") is not None:
        os.environ["MALLOC_MMAP_THRESHOLD_"] = str(int(cfg["malloc_mmap_threshold"]))

    return actual


def set_parallelism(omp_threads: int | None = None, blas_threads: int | None = None):
    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))
        os.environ["OMP_DYNAMIC"] = "FALSE"

    b = None
    if blas_threads is not None:
        b = max(1, int(blas_threads))
        os.environ["OPENBLAS_NUM_THREADS"] = str(b)
        os.environ["OPENBLAS_DYNAMIC"] = "0"
        os.environ["MKL_NUM_THREADS"] = str(b)
        os.environ["MKL_DYNAMIC"] = "FALSE"
        os.environ["BLIS_NUM_THREADS"] = str(b)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(b)
        try:
            import mkl
            mkl.set_num_threads(b)
        except Exception:
            pass

    if b is not None and threadpool_limits is not None:
        return threadpool_limits(limits=b, user_api="blas")
    else:
        return nullcontext()


def _round_up_to(x, gran):
    return int(((x + gran - 1) // gran) * gran)

def _build_balanced_vtiles(V, vmax, gran=64, max_tiles=4):
    vmax = max(gran, (vmax // gran) * gran)
    if vmax >= V:
        return [(0, V)]
    tiles = int(math.ceil(V / vmax))
    tiles = min(max_tiles, max(2, tiles))
    q, r = divmod(V, tiles)
    sizes = []
    for t in range(tiles):
        sz = q + (1 if t < r else 0)
        sz = min(vmax, _round_up_to(sz, gran))
        sizes.append(sz)

    total = sum(sizes)
    over = total - V
    t = len(sizes) - 1
    while over > 0 and t >= 0:
        bleed = min(over, sizes[t] - max(gran, q))
        bleed = (bleed // gran) * gran
        if bleed > 0:
            sizes[t] -= bleed
            over -= bleed
        t -= 1

    vtiles, v0 = [], 0
    for sz in sizes:
        if sz <= 0:
            continue
        if v0 + sz > V:
            sz = V - v0
        if sz <= 0:
            break
        vtiles.append((v0, sz))
        v0 += sz
    if v0 < V:
        vtiles.append((v0, V - v0))
    return vtiles

def _bytes_human(n):
    if n is None:
        return "n/a"
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB", "PB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:,.2f} {unit}"
    return f"{n:,.2f} EB"

def _rss_snapshot(label, logger=None, include_children=True):
    rss = pss = None
    try:
        pid = os.getpid()
        with open(f"/proc/{pid}/smaps_rollup", "r") as f:
            for line in f:
                if line.startswith("Pss:"):
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
        try:
            logger._log(msg)
        except Exception:
            print(msg, file=sys.stderr, flush=True)
    else:
        print(msg, file=sys.stderr, flush=True)

def _canonical_bfile_prefix(x: str) -> str:
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
    s, e = span

    G = getattr(_THREAD_LOCAL, "G", None)
    Gid = getattr(_THREAD_LOCAL, "bed_prefix", None)
    if (G is None) or (Gid != bed_prefix):
        G = open_bed(bed_prefix + ".bed")
        _THREAD_LOCAL.G = G
        _THREAD_LOCAL.bed_prefix = bed_prefix

    rows = row_sel if row_sel is not None else slice(None)

    geno = G.read(index=np.s_[rows, s:e], dtype=dtype)
    means = np.nanmean(geno, axis=0, dtype=dtype)
    stds  = np.nanstd(geno, axis=0, dtype=dtype, ddof=ddof)
    stds[stds == 0] = 1.0
    np.subtract(geno, means, out=geno)
    np.divide(geno, stds, out=geno)
    np.nan_to_num(geno, copy=False)
    geno = np.asfortranarray(geno, dtype=dtype)

    if C is not None and R is not None:
        tmp = R @ geno
        Y   = geno - (C @ tmp)
        del tmp
    else:
        Y = geno

    resvar = np.sum(Y * Y, axis=0, dtype=dtype) / float(N_eff - 1)
    inv_sqrt_resvar = (1.0 / np.sqrt(np.maximum(resvar, eps))).astype(dtype, copy=False)
    return (s, e, inv_sqrt_resvar)

# -------------------- covariate reader → orthonormal Q --------------------
def read_cov(
    cov_filename: str,
    fam_filename: str,
    std: bool = True,
    cov_impute_method: str = "ignore",
    one_hot_conversion: bool = False,
    categorical_threshold: int = 100,
    logger=None,
    verbose=False,
    sample_idx=None,
    ddof=1
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
        if logger:
            logger._log(f"Dropping {dropped} samples due to missing covariates.")
        df = df.loc[keep_mask].reset_index(drop=True)
    else:
        df = df.apply(lambda s: s.fillna(s.mean()), axis=0)
        keep_mask = np.ones(len(df), dtype=bool)

    zvc = df.std(ddof=0) == 0
    if zvc.any():
        drop_cols = zvc.index[zvc].tolist()
        if logger:
            logger._log(f"Dropping {len(drop_cols)} constant covariates: {drop_cols[:10]}{'...' if len(drop_cols)>10 else ''}")
        df.drop(columns=drop_cols, inplace=True)

    if std and not df.empty:
        df = (df - df.mean()) / df.std(ddof=ddof)
        bad_cols = [c for c in df.columns if df[c].isna().all()]
        if bad_cols:
            if logger:
                logger._log(f"Dropping malformed covariate columns after standardization: {bad_cols}")
            df.drop(columns=bad_cols, inplace=True)

    if df.empty:
        raise ValueError("After cleaning, no usable covariates remain.")

    C64 = df.to_numpy(dtype=np.float64)
    Q, _ = np.linalg.qr(C64, mode='reduced')
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
                num_threads: int | None = None,
                eps_var: float = 1e-10,
                rand_samp=None,
                ddof=1,
                target_xz_mem=16.0,
                target_mem=None,
                device='cpu',
                use_tp32=False,
                correct_skew: bool = False,
                hybrid: bool = False,
                hybrid_window_kb: float = 20000.0):

        self.eps_var = float(eps_var)
        prefix = _canonical_bfile_prefix(bed_path)
        self.bed_prefix = os.path.abspath(prefix)
        self.fam_path   = self.bed_prefix + ".fam"
        self.bim_path   = self.bed_prefix + ".bim"

        self.G = open_bed(self.bed_prefix + ".bed")
        self.nsamp, self.nsnps = self.G.shape
        self.nvecs = int(num_vecs)
        self.step_size = int(step_size)
        self.log = log
        self.verbose = verbose
        gwldcore.set_verbose(bool(self.verbose))
        self.rand_dist = rand_dist
        self.ddof = int(ddof)
        self.target_mem = target_mem
        self.target_xz_mem = target_xz_mem if target_mem is None else target_mem

        # Hybrid controls
        self.hybrid = bool(hybrid)
        self.hybrid_window_kb = float(hybrid_window_kb)
        self.hybrid_window_bp = int(round(1000.0 * self.hybrid_window_kb))
        if self.hybrid:
            if self.hybrid_window_bp <= 0:
                raise ValueError("--hybrid-window-kb must be > 0 when --hybrid is enabled.")
            self.log._log(f"[hybrid] enabled with exact local window = {self.hybrid_window_kb:.3f} kb")

        self.correct_skew = bool(correct_skew)
        if self.correct_skew:
            self.log._log(f"[fs-corr] Fourth-moment correction enabled: {self.correct_skew}")

        # Always resolve a concrete root seed so hybrid local-RP can reproduce the same probes.
        if seed is None:
            self.root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            self.log._log(f"[seed] No seed provided; using generated root seed {self.root_seed}")
        else:
            self.root_seed = int(seed)

        rng = np.random.default_rng(self.root_seed)

        explicit_threads = num_threads is not None and int(num_threads) > 0
        if explicit_threads:
            low_level["num_threads"] = int(num_threads)

        actual_blas_threads = apply_env(low_level)

        if explicit_threads:
            self.num_threads = int(num_threads)
        else:
            self.num_threads = max(1, int(actual_blas_threads))

        try:
            gwldcore.set_num_threads(self.num_threads)
            self.log._log(f"[gwldcore] OpenMP threads set to {self.num_threads}")
        except Exception as e:
            self.log._log(f"[gwldcore] set_num_threads failed (non-fatal): {e}")

        self.device_raw = device
        self.device_kind, self.device_index = _parse_device_str(device)

        self.dtype = np.float32 if dtype in (np.float32, 'float32', 'f4') else np.float64
        self.use_tp32 = bool(use_tp32)
        if self.use_tp32 and self.dtype is np.float64:
            self.log._log("[warn] --use-tp32 only affects float32; ignored for float64.")
            self.use_tp32 = False

        if self.device_kind == "cuda":
            try:
                import gwldcore_cuda as _gcu
                gpus = _gcu.list_gpus()
                if not gpus:
                    raise RuntimeError("CUDA requested but no GPUs are visible.")
                if self.device_index is None:
                    self.device_index = int(max(gpus, key=lambda d: int(d["free_bytes"]))["id"])
                else:
                    ids = {int(d["id"]) for d in gpus}
                    if self.device_index not in ids:
                        raise RuntimeError(f"Requested cuda:{self.device_index} not visible; available: {sorted(ids)}")
                self.use_cuda = True
                self.log._log(f"GPU backend enabled on cuda:{self.device_index} (TP32={'on' if self.use_tp32 else 'off'})")
            except Exception as e:
                self.use_cuda = False
                self.device_kind = "cpu"
                self.device_index = None
                self.log._log(f"[GPU] disabled: {e} (falling back to CPU)")
        else:
            self.use_cuda = False

        self.device = "cpu" if self.device_kind == "cpu" else f"cuda:{self.device_index}"

        self.start_time = utils._get_time()
        self.log._log("Genome-wide LD score calculation started at: " + utils._get_timestr(self.start_time))

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

        self._read_bim(self.bim_path)
        if self.hybrid:
            if self.snplist is None:
                raise ValueError("--hybrid requires a .bim file with CHR/BP columns.")
            chr_codes, _ = pd.factorize(self.snplist["CHR"].astype(str), sort=False)
            self.hybrid_chr_codes = np.ascontiguousarray(chr_codes.astype(np.int32, copy=False))
            self.hybrid_bp = np.ascontiguousarray(
                self.snplist["BP"].to_numpy(dtype=np.int64, copy=True)
            )

            # Validate within-chromosome monotonicity
            p = 0
            while p < self.nsnps:
                c = self.hybrid_chr_codes[p]
                q = p + 1
                while q < self.nsnps and self.hybrid_chr_codes[q] == c:
                    q += 1
                if q - p > 1 and np.any(np.diff(self.hybrid_bp[p:q]) < 0):
                    raise ValueError(
                        f"Hybrid mode requires nondecreasing BP within chromosome; "
                        f"failed on CHR={self.snplist.loc[p, 'CHR']}"
                    )
                p = q
                
        if annot_path is not None:
            self._read_annot(annot_path)
        else:
            self._read_annot(None)

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
                ddof=self.ddof
            )
            self.row_sel = np.asarray(keep_idx_global, dtype=int)
            self.C = np.asarray(C, dtype=self.dtype, order='F')
            self.cov_R = np.asarray(R, dtype=self.dtype, order='F')
            self.nsamp = self.C.shape[0]
            self.log._log(f"Final sample count after covariate filtering/subsample: {self.nsamp}")
        else:
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
        self.inv_sqrt_resvar_all = None

        # Hybrid metadata populated lazily in _compute_ldscore()
        self._hyb_chr = None
        self._hyb_bp = None
        self._hyb_src_starts = None
        self._hyb_src_ends = None
        self._hyb_tgt_blocks = None
        self._hyb_src_starts_by_tgt = None
        self._hyb_src_ends_by_tgt = None

    def _precompute_residual_variances(self):
        row_sel = self.row_sel if self.row_sel is not None else slice(None)
        inv = np.empty(self.nsnps, dtype=self.dtype)

        chunks = [(s, min(self.nsnps, s + self.step_size))
                  for s in range(0, self.nsnps, self.step_size)]
        if not chunks:
            self.log._log("[warn] No SNP chunks formed; returning zeros.")
            return np.zeros(self.nsnps, dtype=self.dtype)

        bed_prefix = getattr(self, "bed_prefix", None)
        if bed_prefix is None:
            bp = Path(str(getattr(self.G, "filename", None)
                          or getattr(self.G, "filepath", None) or ""))
            if bp.suffix == ".bed":
                bp = bp.with_suffix("")
            bed_prefix = str(bp)

        dtype = self.dtype
        ddof  = int(self.ddof)
        C     = self.C if self.C is not None else None
        R     = self.cov_R if self.C is not None else None
        N_eff = float(self.N_eff if self.C is not None else self.nsamp)
        eps   = float(self.eps_var)

        try:
            avail = int(psutil.virtual_memory().available) if self.target_mem is None else int((self.target_mem * (1024**3)))
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
        n_workers = min(min(n_workers, len(chunks)), 16)

        self.log._log(f"[resvar] Using {n_workers} workers "
                      f"(~{_bytes_human(est_per_chunk)} per task; avail={_bytes_human(avail)})")

        try:
            with set_parallelism(omp_threads=1, blas_threads=1):
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [
                        ex.submit(
                            _resvar_worker_thread,
                            span,
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

                    for fut in as_completed(futures):
                        s, e, inv_part = fut.result()
                        inv[s:e] = inv_part

        except Exception as e:
            self.log._log(f"[resvar] Parallel precompute failed ({e}); falling back to serial.")
            for s, e in chunks:
                G = self.G.read(index=np.s_[row_sel, s:e], dtype=dtype)
                means = np.nanmean(G, axis=0, dtype=dtype)
                stds  = np.nanstd(G, axis=0, dtype=dtype, ddof=ddof)
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

    def _estimate_mu22_bins(self, blocks):
        N = int(self.nsamp)
        B = int(self.nbins)
        if N <= 0 or B <= 0:
            raise RuntimeError("Invalid N or nbins for μ22 estimation.")

        row_sel = self.row_sel if self.row_sel is not None else slice(None)
        S_mat = np.zeros((N, B), dtype=np.float64)

        self.log._log("[mu22] Estimating bin-by-bin 4th moments μ̄_{22,ab} via streaming over genotype.")
        for (s, e) in tqdm(blocks, desc="mu22 blocks", unit="blk", smoothing=0.2, miniters=1):
            L = e - s
            if L <= 0:
                continue

            Gblk = self.G.read(index=np.s_[row_sel, s:e], dtype=self.dtype)
            means = np.nanmean(Gblk, axis=0, dtype=self.dtype)
            stds  = np.nanstd(Gblk, axis=0, dtype=self.dtype, ddof=int(self.ddof))
            stds[stds == 0] = 1.0
            np.subtract(Gblk, means, out=Gblk)
            np.divide(Gblk, stds, out=Gblk)
            np.nan_to_num(Gblk, copy=False)

            if self.C is not None and self.cov_R is not None:
                tmp = self.cov_R @ Gblk
                Y   = Gblk - (self.C @ tmp)
                del tmp
            else:
                Y = Gblk

            inv_slice = self.inv_sqrt_resvar_all[s:e].astype(self.dtype, copy=False)
            Y *= inv_slice

            X2 = np.asarray(Y, dtype=np.float64)**2
            ann_blk = np.asarray(self.annot[s:e], dtype=np.float64)
            S_mat += X2 @ ann_blk

            del Gblk, Y, X2, ann_blk
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        S_sum = S_mat.T @ S_mat
        M = np.asarray(self.nsnps_bin, dtype=np.float64)
        M_outer = M[:, None] * M[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mu22 = S_sum / (float(N) * M_outer)
            mu22[~np.isfinite(mu22)] = 0.0

        self.log._log("[mu22] Finished μ̄_{22,ab} estimation.")
        return mu22

    def _compute_block_corrections(self, meansq_raw, mu22, bin_idx):
        B = int(self.nbins)
        N = float(self.nsamp)

        if self.C is not None:
            d = float(self.N_eff)
        else:
            d = float(self.nsamp - self.ddof)

        mu22_block = np.asarray(mu22, dtype=np.float64)
        M_a = np.array([len(idx) for idx in bin_idx], dtype=np.float64)

        R2_block   = np.zeros((B, B), dtype=np.float64)
        rho2_block = np.zeros((B, B), dtype=np.float64)
        bias_block = np.zeros((B, B), dtype=np.float64)

        for a in range(B):
            Ma = M_a[a]
            if Ma <= 0:
                continue
            idx_a = bin_idx[a]
            for b in range(B):
                Mb = M_a[b]
                if Mb <= 0:
                    continue

                S_ab = float(np.sum(meansq_raw[idx_a, b], dtype=np.float64))
                if S_ab == 0.0:
                    continue

                R2_ab = S_ab / (Ma * Mb)
                mu_ab = float(mu22_block[a, b])
                rho2_ab = (d**2 * R2_ab - N * mu_ab) / (N * (N - 1.0))

                R2_block[a, b]   = R2_ab
                rho2_block[a, b] = rho2_ab
                sum_rho2 = Ma * Mb * rho2_ab
                bias_block[a, b] = S_ab - sum_rho2

        self.r2_block   = R2_block
        self.rho2_block = rho2_block
        self.bias_block = bias_block
        self.mu22_block = mu22_block

        delta_block = mu22_block - (1.0 + 2.0 * rho2_block)
        self.delta_block = delta_block

        self.log._log("[fs-corr] Estimated block-level R2, ρ2, bias, μ22 and δ (μ̄22 - (1 + 2ρ²)).")

        try:
            df_delta = pd.DataFrame(
                delta_block,
                index=self.l2cols,
                columns=self.l2cols,
            )
            with pd.option_context('display.width', 140,
                                   'display.max_columns', None,
                                   'display.float_format', '{:.6e}'.format):
                self.log._log("[fs-corr] Block-level δ matrix (rows/cols = annotation bins):")
                self.log._log("\n" + df_delta.to_string())
        except Exception as e:
            self.log._log(f"[fs-corr] Failed to pretty-print δ matrix via pandas ({e}); using numpy.")
            self.log._log(repr(delta_block))

    def _read_annot(self, annot_path):
        if annot_path is None:
            self.l2cols = None
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.l2cols = [f"L2_{i}" for i in range(self.nbins)]
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
            # Keep annotation in the compute dtype/contiguity so pybind does not create
            # a hidden temporary copy on every hybrid call.
            self.annot = np.ascontiguousarray(self.annot.astype(self.dtype, copy=False))
            self.log._log("Calculating genome-wide (non-partitioned) LD score")
            self.log._log(f"Number of samples: {self.nsamp}")
            self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
            return

        parsed_ldsc = False
        try:
            df = pd.read_csv(annot_path, sep=r'\s+', compression='infer',
                             dtype={'CHR': str, 'BP': np.int64, 'SNP': str, 'CM': float})
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
                    ann_set = set(ann_snps)
                    bim_set = set(bim_snps)
                    missing_in_annot = len(bim_set - ann_set)
                    extra_in_annot = len(ann_set - bim_set)
                    if missing_in_annot > 0:
                        raise ValueError(
                            f"Annotation SNP set is missing {missing_in_annot} BIM SNP(s); "
                            f"prepare a matching .annot or regenerate it to the .bim."
                        )
                    if extra_in_annot > 0:
                        self.log._log(f"[info] Annotation contains {extra_in_annot} extra SNP(s) not in BIM; "
                                      f"keeping BIM SNPs only and reordering to BIM.")
                    ann_mat = df.set_index('SNP').loc[bim_snps, annot_cols].to_numpy(dtype=np.float64, copy=False)

                np.nan_to_num(ann_mat, copy=False)
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

        self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
        self.annot = np.ascontiguousarray(self.annot.astype(self.dtype, copy=False))

        self.log._log(f"Number of samples: {self.nsamp}")
        self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
        self.log._log(f"Nbins: {self.nbins}")

    def _read_bim(self, bim_path):
        if bim_path is None:
            self.log._log("No .bim file is provided; all (anonymous) SNPs will be used")
            self.snplist = None
        else:
            self.log._log(f"Reading {bim_path} for SNPs")
            self.snplist = pd.read_csv(bim_path, header=None, sep=r'\s+')
            self.snplist.columns = ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2']
        if len(self.snplist) != self.nsnps:
            self.log._log(f"!!! The number of SNPs in the .bed file ({self.nsnps}) does not match the .bim file ({len(self.snplist)}) !!!")
            sys.exit(1)

    def _partition_index(self, snpidx, annot) -> list[np.ndarray]:
        return [snpidx[(annot[:, c] != 0)] for c in range(self.nbins)]

    def _prepare_hybrid_plan(self, src_blocks):
        if not self.hybrid:
            return
        if self.snplist is None:
            raise ValueError("Hybrid mode requires a .bim file with CHR/BP columns.")

        chr_codes, _ = pd.factorize(self.snplist["CHR"].astype(str), sort=False)
        chr_codes = np.asarray(chr_codes, dtype=np.int32)
        bp = self.snplist["BP"].to_numpy(dtype=np.int64, copy=True)

        # Validate monotone BP within chromosome.
        p = 0
        while p < self.nsnps:
            c = chr_codes[p]
            q = p + 1
            while q < self.nsnps and chr_codes[q] == c:
                q += 1
            if q - p > 1:
                if np.any(np.diff(bp[p:q]) < 0):
                    raise ValueError(f"Hybrid mode requires nondecreasing BP within chromosome block; failed on CHR={self.snplist.loc[p, 'CHR']}.")
            p = q

        src_starts = np.asarray([s for s, _ in src_blocks], dtype=np.int32)
        src_ends   = np.asarray([e for _, e in src_blocks], dtype=np.int32)

        src_first_chr = chr_codes[src_starts]
        src_last_chr  = chr_codes[src_ends - 1]
        src_single_chr = (src_first_chr == src_last_chr)
        src_min_bp = bp[src_starts]
        src_max_bp = bp[src_ends - 1]

        src_by_chr = {}
        for bi in range(len(src_blocks)):
            c0 = int(src_first_chr[bi])
            c1 = int(src_last_chr[bi])
            src_by_chr.setdefault(c0, []).append(bi)
            if c1 != c0:
                src_by_chr.setdefault(c1, []).append(bi)

        W = int(self.hybrid_window_bp)
        tgt_blocks = []
        src_starts_by_tgt = []
        src_ends_by_tgt = []

        p = 0
        while p < self.nsnps:
            c = int(chr_codes[p])
            q = p + 1
            while q < self.nsnps and chr_codes[q] == c:
                q += 1

            for s in range(p, q, self.step_size):
                e = min(q, s + self.step_size)
                tgt_blocks.append((s, e))
                tmin = int(bp[s])
                tmax = int(bp[e - 1])

                cand = []
                for bi in src_by_chr.get(c, []):
                    if src_single_chr[bi]:
                        if src_max_bp[bi] < (tmin - W):
                            continue
                        if src_min_bp[bi] > (tmax + W):
                            continue
                    cand.append(bi)

                src_starts_by_tgt.append(np.ascontiguousarray(src_starts[np.asarray(cand, dtype=np.int32)], dtype=np.int32))
                src_ends_by_tgt.append(np.ascontiguousarray(src_ends[np.asarray(cand, dtype=np.int32)], dtype=np.int32))

            p = q

        self._hyb_chr = np.ascontiguousarray(chr_codes, dtype=np.int32)
        self._hyb_bp = np.ascontiguousarray(bp, dtype=np.int64)
        self._hyb_src_starts = src_starts
        self._hyb_src_ends = src_ends
        self._hyb_tgt_blocks = tgt_blocks
        self._hyb_src_starts_by_tgt = src_starts_by_tgt
        self._hyb_src_ends_by_tgt = src_ends_by_tgt

        nn = np.asarray([len(x) for x in src_starts_by_tgt], dtype=np.int32)
        self.log._log(
            f"[hybrid] prepared {len(tgt_blocks)} target block(s); "
            f"candidate source blocks per target min/median/max = "
            f"{int(nn.min()) if len(nn) else 0}/{int(np.median(nn)) if len(nn) else 0}/{int(nn.max()) if len(nn) else 0}"
        )

    def _compute_ldscore(self):
        self.log._log(f"num_vecs: {self.nvecs}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        if self.C is not None:
            self.log._log(f"Covariate-adjusted partial correlations (N_eff={self.N_eff}, p={self.p_eff}).")
        else:
            self.log._log("No covariates: standard LD scores (squared correlations).")
        if self.hybrid:
            self.log._log(f"[hybrid] using optimized banded local pass "
                f"(window={self.hybrid_window_kb:.3f} kb)")

        H = self.num_threads
        t_blas1 = min(4, max(1, H // 4))
        t_omp1 = max(1, H // t_blas1)

        with set_parallelism(omp_threads=min(self.num_threads, 16), blas_threads=t_blas1):
            self.inv_sqrt_resvar_all = np.ascontiguousarray(self._precompute_residual_variances().astype(self.dtype, copy=False))

        blocks = []
        for j in range(0, self.nsnps, self.step_size):
            s = j
            e = min(self.nsnps, j + self.step_size)
            blocks.append((s, e))
        self.nblks = len(blocks)
        block_starts = np.asarray([s for (s, _) in blocks], dtype=np.int32)
        block_ends   = np.asarray([e for (_, e) in blocks], dtype=np.int32)

        kmax_per_block = []
        for (s, e) in blocks:
            blk = self.annot[s:e]
            Kmax = int((blk != 0).sum(axis=0).max())
            kmax_per_block.append(Kmax)
        if any(k == 0 for k in kmax_per_block):
            zc = sum(1 for k in kmax_per_block if k == 0)
            self.log._log(f"[info] {zc} block(s) have Kmax=0 (will be skipped).")
        if kmax_per_block:
            self.log._log(f"Kmax per block (min/median/max): "
                          f"{min(kmax_per_block)}/{int(np.median(kmax_per_block))}/{max(kmax_per_block)}")

        if self.hybrid:
            self._prepare_hybrid_plan(blocks)
            hyb_maxL = max(e - s for (s, e) in self._hyb_tgt_blocks) if self._hyb_tgt_blocks else 0
            hyb_local_buf = np.zeros((hyb_maxL, self.nbins), dtype=self.dtype, order='C')
            hyb_exact_buf = np.zeros((hyb_maxL, self.nbins), dtype=self.dtype, order='C')
        else:
            hyb_local_buf = None
            hyb_exact_buf = None

        target_gib = float(getattr(self, "target_xz_mem", 16.0))
        itemsize = np.dtype(self.dtype).itemsize
        denom = max(1, int(self.nsamp) * int(self.nbins) * itemsize)
        vtile_guess = int((target_gib * (1024**3)) // denom)
        vtile_guess = max(256, min(self.nvecs, vtile_guess))
        if vtile_guess <= 0:
            vtile_guess = min(self.nvecs, 4096)

        ntiles = int(np.ceil(self.nvecs / vtile_guess))
        ntiles = max(1, ntiles)
        base = (self.nvecs // ntiles)
        rem  = (self.nvecs % ntiles)
        vtiles = [((base + (1 if i < rem else 0) + 63)//64)*64 for i in range(ntiles)]
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

        bed_prefix = self.bed_prefix
        fam_path   = self.fam_path
        row_sel = self.row_sel if self.row_sel is not None else None
        ddof    = int(self.ddof)
        B       = int(self.nbins)

        meansq_accum = np.zeros((self.nsnps, self.nbins), dtype=self.dtype, order='C')
        Vmax = max(vtiles) if vtiles else 0
        Xz_chunk = np.zeros((self.nsamp, int(self.nbins) * int(Vmax)),
                            dtype=self.dtype, order='F')
        meansq_chunk = np.zeros_like(meansq_accum, dtype=self.dtype, order='C')

        ann_blocks = [np.ascontiguousarray(self.annot[s:e]) for (s, e) in blocks]
        inv_blocks = [np.ascontiguousarray(self.inv_sqrt_resvar_all[s:e]) for (s, e) in blocks]

        total_units = len(vtiles) * (len(blocks) + (1 if self.hybrid else 0))
        bar = tqdm(total=total_units, desc="GW-LD progress", unit="task", smoothing=0.2, miniters=1)

        ema_p1 = 0.0
        ema_p2 = 0.0
        w1 = 0.5

        try:
            v_start = 0
            for vt_idx, Vt in enumerate(vtiles):
                used_cols = B * Vt
                Xz_view = Xz_chunk[:, :used_cols]
                Xz_view.fill(0)

                t1_total = 0.0
                with set_parallelism(omp_threads=t_omp1, blas_threads=t_blas1):
                    for blk_idx, (s, e) in enumerate(blocks):
                        kmax_hint = int(kmax_per_block[blk_idx])
                        if kmax_hint == 0:
                            bar.update(1.0)
                            continue

                        annot_blk = ann_blocks[blk_idx]
                        inv_right = inv_blocks[blk_idx]

                        t0 = time.perf_counter()
                        gwldcore.phase1_compute_Xz_bed_chunk(
                            bed_prefix=bed_prefix,
                            fam_path=fam_path,
                            blk_start=int(s), blk_end=int(e),
                            row_sel=row_sel,
                            ddof=ddof,
                            annot_blk=annot_blk,
                            inv_right=inv_right,
                            v_start=int(v_start),
                            v_count=int(Vt),
                            kmax_hint=kmax_hint,
                            rand_dist=self.rand_dist,
                            seed=self.root_seed,
                            Xz2d_chunk=Xz_view,
                            project_right=False,
                            C=(self.C if self.C is not None else None),
                            R=(self.cov_R if self.C is not None else None)
                        )
                        t1_total += (time.perf_counter() - t0)
                        bar.update(w1)

                pref_ex = ThreadPoolExecutor(max_workers=1)
                meansq_chunk.fill(0)
                t2_total = 0.0

                dev_kind, dev_idx_req = _parse_device_str(getattr(self, "device", "cpu"))
                use_tf32 = bool(getattr(self, "use_tp32", False))
                use_cuda_backend = False
                gcu = None
                if dev_kind == "cuda":
                    try:
                        import gwldcore_cuda as gcu
                        if dev_idx_req is None:
                            dev_idx, free_b, tot_b = _pick_cuda_index_auto(gcu)
                        else:
                            gl = gcu.list_gpus()
                            ids = {int(g["id"]) for g in gl}
                            if dev_idx_req not in ids:
                                raise RuntimeError(f"Requested cuda:{dev_idx_req} not visible among {sorted(ids)}")
                            dev_idx = dev_idx_req
                        if self.verbose:
                            self.log._log(f"[GPU] Using cuda:{dev_idx} (TF32={'on' if use_tf32 else 'off'})")
                        use_cuda_backend = True
                    except Exception as e:
                        self.log._log(f"[GPU] Falling back to CPU: {e}")
                        use_cuda_backend = False

                if self.C is not None:
                    N_denom = int(self.N_eff)
                else:
                    N_denom = int(self.nsamp - self.ddof + 1)

                if use_cuda_backend:
                    with set_parallelism(omp_threads=1, blas_threads=1):
                        for blk_idx, (s, e) in enumerate(blocks):
                            kmax_hint = int(kmax_per_block[blk_idx])
                            if kmax_hint == 0:
                                continue

                            inv_left = inv_blocks[blk_idx]
                            t0 = time.perf_counter()
                            gcu.phase2_compute_XtXz_bed(
                                bed_prefix=bed_prefix,
                                fam_path=fam_path,
                                blk_start=int(s), blk_end=int(e),
                                row_sel=row_sel,
                                ddof=ddof,
                                inv_left=inv_left,
                                nvecs=int(Vt),
                                vchunk=int(Vt),
                                Xz2d=Xz_view,
                                meansq=meansq_chunk,
                                C=(self.C if self.C is not None else None),
                                R=(self.cov_R if self.C is not None else None),
                                N_denom=int(N_denom),
                                use_tf32=use_tf32,
                                device_index=int(dev_idx),
                            )
                            t2_total += (time.perf_counter() - t0)
                            bar.update(1.0 - w1)
                else:
                    with set_parallelism(omp_threads=1, blas_threads=self.num_threads):
                        for blk_idx, (s, e) in enumerate(blocks):
                            kmax_hint = int(kmax_per_block[blk_idx])
                            if kmax_hint == 0:
                                continue
                            inv_left = inv_blocks[blk_idx]

                            if blk_idx + 1 < len(blocks):
                                s2, e2 = blocks[blk_idx + 1]
                                pref_ex.submit(
                                    gwldcore.prefetch_bed_block,
                                    bed_prefix, fam_path,
                                    int(s2), int(e2),
                                    1,
                                )

                            t0 = time.perf_counter()
                            gwldcore.phase2_compute_XtXz_bed(
                                bed_prefix=bed_prefix,
                                fam_path=fam_path,
                                blk_start=int(s), blk_end=int(e),
                                row_sel=row_sel,
                                ddof=ddof,
                                inv_left=inv_left,
                                nvecs=int(Vt),
                                vchunk=int(Vt),
                                Xz2d=Xz_view,
                                meansq=meansq_chunk,
                                C=(self.C if self.C is not None else None),
                                R=(self.cov_R if self.C is not None else None),
                                N_denom=int(N_denom),
                            )
                            t2_total += (time.perf_counter() - t0)
                            bar.update(1.0 - w1)

                pref_ex.shutdown(wait=True)
                meansq_accum += (meansq_chunk * Vt)
                # ---------------------- Hybrid local pass (one C++ banded sweep per tile) ----------------------
                if self.hybrid:
                    with set_parallelism(omp_threads=1, blas_threads=self.num_threads):
                        gwldcore.phase2_compute_local_hybrid_banded_bed(
                            bed_prefix=bed_prefix,
                            fam_path=fam_path,
                            block_starts=block_starts,
                            block_ends=block_ends,
                            row_sel=row_sel,
                            ddof=ddof,
                            inv_all=self.inv_sqrt_resvar_all,
                            annot_all=self.annot,
                            chr_code_all=self.hybrid_chr_codes,
                            bp_all=self.hybrid_bp,
                            window_bp=int(self.hybrid_window_bp),
                            v_start=int(v_start),
                            v_count=int(Vt),
                            rand_dist=self.rand_dist,
                            seed=self.root_seed,
                            meansq_accum=meansq_accum,
                            rp_scale=float(-Vt),                     # weighted the same way as meansq_chunk * Vt
                            add_exact=bool(vt_idx == 0),            # add exact local only once
                            exact_scale=float(self.nvecs),          # so final divide by nvecs yields + exact_local
                            C=(self.C if self.C is not None else None),
                            R=(self.cov_R if self.C is not None else None),
                            N_denom=int(N_denom),
                        )
                    bar.update(1.0)

                
                ema_p1 = 0.85 * ema_p1 + 0.15 * max(t1_total, 1e-9)
                ema_p2 = 0.85 * ema_p2 + 0.15 * max(t2_total, 1e-9)
                w1 = float(ema_p1 / (ema_p1 + ema_p2))
                bar.set_postfix_str(f"tile {vt_idx+1}/{len(vtiles)} | w1={w1:.2f} | P1={t1_total:.1f}s P2={t2_total:.1f}s")

                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                v_start += Vt

        finally:
            try:
                bar.close()
            except Exception:
                pass

        meansq = (meansq_accum / float(self.nvecs)).astype(self.dtype, copy=False)

        if not self.correct_skew:
            self.mu22_block = None
            self.delta_block = None
            self.r2_block = None
            self.rho2_block = None
            self.bias_block = None
        else:
            meansq_raw = np.asarray(meansq, dtype=np.float64, order="C")
            try:
                mu22 = self._estimate_mu22_bins(blocks)

                if not getattr(self, "is_continuous", False):
                    snpidx = np.arange(self.nsnps, dtype=int)
                    bin_idx = self._partition_index(snpidx, self.annot)
                    self._compute_block_corrections(meansq_raw, mu22, bin_idx)
                else:
                    self.mu22_block = mu22
                    self.log._log(
                        "[fs-corr] Continuous / overlapping annotations detected; "
                        "stored μ22_block but skipped block-level bias correction."
                    )
            except Exception as e:
                self.log._log(f"[fs-corr] Failed to compute 4th-moment-based corrections: {e}")
                self.mu22_block = None
                self.delta_block = None
                self.r2_block = None
                self.rho2_block = None
                self.bias_block = None

        N_denom = float(self.N_eff - 1.0 if self.C is not None else self.nsamp - self.ddof)
        self.log._log("Applying correlation null: subtracting M_k / N_denom per bin.")
        meansq -= (self.nsnps_bin / N_denom).astype(meansq.dtype, copy=False)[None, :]

        self.gwldscore = meansq.astype(np.float64, copy=False)

        self.log._log(f"Saving the genome-wide (partitioned) LD scores into: {self.outpath}.gw.ldscore.gz")
        snpcols = ['CHR', 'SNP', 'BP']
        if self.snplist is None:
            self.snpdf = pd.DataFrame(np.nan * np.ones((self.nsnps, 3)), columns=snpcols)
        else:
            self.snpdf = self.snplist[['CHR', 'SNP', 'BP']].copy()
            self.snpdf.columns = snpcols

        scores_df = pd.DataFrame(self.gwldscore, columns=self.l2cols)
        out_df = pd.concat([self.snpdf, scores_df], axis=1)
        out_df.to_csv(f'{self.outpath}.gw.ldscore.gz', index=False, compression='gzip', sep='\t', float_format='%.6f')

        if self.correct_skew:
            try:
                if getattr(self, "delta_block", None) is not None:
                    delta_df = pd.DataFrame(
                        self.delta_block,
                        index=self.l2cols,
                        columns=self.l2cols,
                    )
                    delta_out = f"{self.outpath}.gw.delta"
                    delta_df.to_csv(delta_out, sep='\t', float_format='%.8e')
                    self.log._log(f"[fs-corr] Saved block-level δ matrix to: {delta_out}")
                else:
                    self.log._log("[fs-corr] delta_block not available; skipping .gw.delta write.")
            except Exception as e:
                self.log._log(f"[fs-corr] Failed to save δ matrix (.gw.delta): {e}")

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
        self.log._log(f"Calculation of genome-wide LD score ended at " + utils._get_timestr(self.end_time))
        self.runtime = self.end_time - self.start_time
        self.log._log("Runtime: " + format(self.runtime, '.3f') +
                      f" s ({self.runtime//3600} hr {(self.runtime%3600)//60} m {(self.runtime%60):.3f} s)")
        self.log._save_log(self.outpath + ".gw.log")