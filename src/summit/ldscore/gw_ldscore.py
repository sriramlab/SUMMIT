# gw_ldscore.py
from .. import utils
import math
import numpy as np
import pandas as pd
from bed_reader import open_bed
from tqdm import tqdm
import sys, shutil
import gc
import os, psutil
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
try:
    from .. import gwldcore
except Exception:
    import gwldcore

from contextlib import contextmanager, nullcontext
from threadpoolctl import threadpool_limits


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


def _set_openmp_threads_runtime(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    try:
        gwldcore.set_num_threads(n)
    except Exception:
        pass


def _get_openmp_threads_runtime() -> int | None:
    try:
        return int(gwldcore.get_max_threads())
    except Exception:
        v = os.environ.get("OMP_NUM_THREADS")
        return int(v) if (v and v.isdigit()) else None


def _set_blas_env_vars(n: int) -> None:
    n = max(1, int(n))
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_DYNAMIC"] = "0"
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["BLIS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)


def _set_blas_threads_runtime(n: int) -> None:
    n = max(1, int(n))
    _set_blas_env_vars(n)
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n)
    except Exception:
        pass
    try:
        for soname in ("libopenblas.so", "libopenblas.so.0", "libopenblas64_.so", "libopenblas64_.so.0"):
            try:
                lib = ctypes.CDLL(soname)
                for sym in ("openblas_set_num_threads", "openblas_set_num_threads64_"):
                    try:
                        getattr(lib, sym)(int(n))
                        break
                    except AttributeError:
                        continue
                break
            except OSError:
                continue
    except Exception:
        pass


@contextmanager
def set_parallelism(omp_threads: int | None = None, blas_threads: int | None = None):
    prev_env = {}
    for key in (
        "OMP_NUM_THREADS", "OMP_DYNAMIC",
        "OPENBLAS_NUM_THREADS", "OPENBLAS_DYNAMIC",
        "MKL_NUM_THREADS", "MKL_DYNAMIC",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        prev_env[key] = os.environ.get(key)

    prev_omp = _get_openmp_threads_runtime() if omp_threads is not None else None
    b = None
    if omp_threads is not None:
        _set_openmp_threads_runtime(int(omp_threads))
    if blas_threads is not None:
        b = max(1, int(blas_threads))
        _set_blas_env_vars(b)
        if threadpool_limits is None:
            _set_blas_threads_runtime(b)

    ctl = threadpool_limits(limits=b, user_api="blas") if (b is not None and threadpool_limits is not None) else nullcontext()
    try:
        with ctl:
            yield
    finally:
        if prev_omp is not None and omp_threads is not None:
            _set_openmp_threads_runtime(prev_omp)
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _clear_phase1_native_scratch() -> None:
    try:
        gwldcore.clear_phase1_native_scratch()
    except AttributeError:
        pass


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


def _zero_colmajor_bin_runs(X: np.ndarray, zero_bins: np.ndarray, vcount: int) -> None:
    zero_bins = np.asarray(zero_bins, dtype=np.int64)
    if zero_bins.size == 0 or vcount <= 0:
        return
    start = prev = int(zero_bins[0])
    for k in zero_bins[1:]:
        k = int(k)
        if k == prev + 1:
            prev = k
            continue
        X[:, start * vcount:(prev + 1) * vcount].fill(0)
        start = prev = k
    X[:, start * vcount:(prev + 1) * vcount].fill(0)


def _orthonormal_covariate_basis(C64: np.ndarray):
    C64 = np.asarray(C64, dtype=np.float64)
    if C64.ndim != 2 or C64.shape[0] == 0 or C64.shape[1] == 0:
        raise ValueError("Covariate matrix must be non-empty.")

    gram = C64.T @ C64
    gram = (gram + gram.T) * 0.5
    evals, evecs = np.linalg.eigh(gram)
    if evals.size == 0:
        raise ValueError("Covariate matrix must contain at least one column.")

    lam_max = float(max(evals[-1], 0.0))
    tol = float(max(C64.shape) * np.finfo(np.float64).eps * lam_max) if lam_max > 0.0 else 0.0
    keep = evals > tol
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise ValueError("After cleaning, covariates have numerical rank zero.")

    basis = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    Q = C64 @ basis

    # Remove roundoff from the Gram eigensolve without changing the selected
    # covariate column space.
    Q, _ = np.linalg.qr(Q, mode='reduced')
    return np.asfortranarray(Q), rank, tol, float(evals[0]), lam_max


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
    C, cov_rank, rank_tol, min_eval, max_eval = _orthonormal_covariate_basis(C64)
    R = np.asfortranarray(C.T)

    km = np.flatnonzero(keep_mask.values) if isinstance(keep_mask, pd.Series) else np.flatnonzero(keep_mask)
    if sample_idx is not None:
        keep_idx_global = np.asarray(sample_idx, dtype=int)[km]
    else:
        keep_idx_global = km

    if logger:
        if cov_rank < C64.shape[1]:
            logger._log(
                f"Dropped {C64.shape[1] - cov_rank} linearly dependent covariate direction(s) "
                f"by Gram eigendecomposition (tol={rank_tol:.6e}, min_eval={min_eval:.6e}, max_eval={max_eval:.6e})."
            )
        logger._log(f"Read {cov_filename}: kept {C.shape[0]} samples, {C.shape[1]} effective covariates. C shape={C.shape}, R shape=({R.shape[0]},{R.shape[1]}).")

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
                write_kmoments: bool = False,
                skip_kmoments=None,
                use_mailman: bool = True,
                impute_method: str = 'mean'):

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

        self._mu22_precomputed = None

        self.correct_skew = bool(correct_skew)
        if skip_kmoments is not None:
            write_kmoments = bool(write_kmoments) and not bool(skip_kmoments)
        self.write_kmoments = bool(write_kmoments)
        if self.correct_skew:
            self.log._log(f"[fs-corr] Fourth-moment correction enabled: {self.correct_skew}")
        if self.write_kmoments:
            self.log._log("[kmom] Higher-order GRM moment estimation enabled.")
        else:
            self.log._log("[kmom] Higher-order GRM moment estimation disabled.")

        if seed is None:
            self.root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            self.log._log(f"[seed] No seed provided; using generated root seed {self.root_seed}")
        else:
            self.root_seed = int(seed)

        self.impute_method = str(impute_method).strip().lower()
        if self.impute_method not in ("hwe", "mean"):
            raise ValueError("impute_method must be 'hwe' or 'mean'.")
        self.use_mailman = bool(use_mailman)
        self.impute_seed = int((np.uint64(self.root_seed) ^ np.uint64(0xA24BAED4963EE407)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        if self.impute_method != "hwe" and self.use_mailman:
            self.log._log("[mailman] Disabled because Mailman requires discrete HWE-imputed hard calls.")
            self.use_mailman = False
        self.log._log(f"[impute] method={self.impute_method} (seed={self.impute_seed})")

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

        if self.use_cuda and (self.impute_method == "hwe" or self.use_mailman):
            self.log._log("[GPU] HWE-imputation / Mailman path is CPU-only; falling back to CPU.")
            self.use_cuda = False
            self.device_kind = "cpu"
            self.device_index = None

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
            self.cov_gram = np.ascontiguousarray(
                (np.asarray(self.C, dtype=np.float64).T @ np.asarray(self.C, dtype=np.float64)).astype(self.dtype, copy=False)
            )
            self.nsamp = self.C.shape[0]
            self.log._log(f"Final sample count after covariate filtering/subsample: {self.nsamp}")
        else:
            self.row_sel = sel_idx if sel_idx is not None else None
            self.C = None
            self.cov_R = None
            self.cov_gram = None
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

    def _precompute_residual_variances(self):
        self.log._log(
            "[resvar] Using fused gwldcore C++ kernel for projected residual variances"
            + (" + μ22." if self.correct_skew else ".")
        )

        inv_all, mu22 = gwldcore.precompute_residual_variances_bed(
            bed_prefix=self.bed_prefix,
            fam_path=self.fam_path,
            nsnps=int(self.nsnps),
            step_size=int(self.step_size),
            row_sel=(self.row_sel if self.row_sel is not None else None),
            ddof=int(self.ddof),
            eps=float(self.eps_var),
            compute_mu22=bool(self.correct_skew),
            annot_all=(self.annot if self.correct_skew else None),
            C=(self.C if self.C is not None else None),
            R=(self.cov_R if self.C is not None else None),
            impute_mode=self.impute_method,
            impute_seed=int(self.impute_seed),
        )

        self._mu22_precomputed = None if mu22 is None else np.asarray(mu22, dtype=np.float64, order="C")
        return np.ascontiguousarray(inv_all.astype(self.dtype, copy=False))

    def _estimate_mu22_bins(self, blocks=None):
        mu22 = getattr(self, "_mu22_precomputed", None)
        if mu22 is None:
            raise RuntimeError("μ22 was not precomputed in the fused residual-variance pass.")
        self.log._log("[mu22] Using μ̄22 from the fused gwldcore residual-variance pass.")
        return np.asarray(mu22, dtype=np.float64, order="C")

    def _compute_block_corrections(self, meansq_raw, mu22, bin_idx=None):
        B = int(self.nbins)
        N = int(self.nsamp)
        d = float(self.N_eff) if self.C is not None else float(self.nsamp - self.ddof)

        (R2_block,
         rho2_block,
         bias_block,
         delta_block) = gwldcore.compute_block_corrections_binary(
            meansq_raw=np.asarray(meansq_raw, dtype=np.float64, order="C"),
            mu22=np.asarray(mu22, dtype=np.float64, order="C"),
            annot_all=self.annot,
            N=int(N),
            d=float(d),
        )

        self.r2_block    = np.asarray(R2_block,   dtype=np.float64, order="C").reshape(B, B)
        self.rho2_block  = np.asarray(rho2_block, dtype=np.float64, order="C").reshape(B, B)
        self.bias_block  = np.asarray(bias_block, dtype=np.float64, order="C").reshape(B, B)
        self.mu22_block  = np.asarray(mu22,       dtype=np.float64, order="C").reshape(B, B)
        self.delta_block = np.asarray(delta_block, dtype=np.float64, order="C").reshape(B, B)

        self.log._log("[fs-corr] Estimated block-level R2, ρ2, bias, μ22 and δ (μ̄22 - (1 + 2ρ²)).")

        try:
            df_delta = pd.DataFrame(self.delta_block, index=self.l2cols, columns=self.l2cols)
            with pd.option_context('display.width', 140, 'display.max_columns', None, 'display.float_format', '{:.6e}'.format):
                self.log._log("[fs-corr] Block-level δ matrix (rows/cols = annotation bins):")
                self.log._log("\n" + df_delta.to_string())
        except Exception as e:
            self.log._log(f"[fs-corr] Failed to pretty-print δ matrix via pandas ({e}); using numpy.")
            self.log._log(repr(self.delta_block))

    def _make_compute_blocks(self):
        return [(s, min(self.nsnps, s + self.step_size)) for s in range(0, self.nsnps, self.step_size)]

    def _read_annot(self, annot_path):
        if annot_path is None:
            self.l2cols = None
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.l2cols = [f"L2_{i}" for i in range(self.nbins)]
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
            self.annot = np.ascontiguousarray(self.annot.astype(self.dtype, copy=False))
            self.log._log("Calculating genome-wide (non-partitioned) LD score")
            self.log._log(f"Number of samples: {self.nsamp}")
            self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
            return

        parsed_ldsc = False
        try:
            df = pd.read_csv(annot_path, sep=r'\s+', compression='infer', dtype={'CHR': str, 'BP': np.int64, 'SNP': str, 'CM': float})
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
                            f"Annotation SNP set is missing {missing_in_annot} BIM SNP(s); prepare a matching .annot or regenerate it to the .bim."
                        )
                    if extra_in_annot > 0:
                        self.log._log(f"[info] Annotation contains {extra_in_annot} extra SNP(s) not in BIM; keeping BIM SNPs only and reordering to BIM.")
                    ann_mat = df.set_index('SNP').loc[bim_snps, annot_cols].to_numpy(dtype=np.float64, copy=False)

                np.nan_to_num(ann_mat, copy=False)
                if (ann_mat < 0).any():
                    self.log._log("[warn] Negative annotation values found; clipping to 0.")
                    ann_mat[ann_mat < 0] = 0.0
                uniq = np.unique(ann_mat)
                is_binary = np.all(np.isin(uniq, [0.0, 1.0]))
                self.is_continuous = (not is_binary)
                self.log._log("[info] Detected continuous annotations (non 0/1 values)." if self.is_continuous else "[info] Detected binary annotations (0/1).")

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
            self.log._log(f"!!! number of SNPs in annotation ({self.annot.shape[0]}) does not match the input genotype file ({self.nsnps}) !!!")
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

    def _compute_ldscore(self):
        self.log._log(f"num_vecs: {self.nvecs}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        self.log._log(f"[backend] Mailman={'on' if self.use_mailman else 'off'} ; impute={self.impute_method}")
        if self.C is not None:
            self.log._log(f"Covariate-adjusted partial correlations (N_eff={self.N_eff}, p={self.p_eff}).")
        else:
            self.log._log("No covariates: standard LD scores (squared correlations).")

        H = self.num_threads
        t_blas1 = min(4, max(1, H // 4))
        t_omp1 = max(1, H // t_blas1)

        self._mu22_precomputed = None
        gwldcore.clear_phase1_csr_cache()

        fuse_resvar = (
            not self.correct_skew
            and str(os.environ.get("SUMMIT_FUSE_RESVAR", "1")).strip().lower() not in {"0", "false", "no", "off"}
        )
        if fuse_resvar:
            self.log._log("[resvar] Fusing residual-variance estimation into the first phase-1 genotype pass.")
            self.inv_sqrt_resvar_all = np.empty(int(self.nsnps), dtype=self.dtype, order="C")
        else:
            with set_parallelism(omp_threads=min(self.num_threads, 16), blas_threads=t_blas1):
                inv_all = self._precompute_residual_variances()
                self.inv_sqrt_resvar_all = np.ascontiguousarray(inv_all.astype(self.dtype, copy=False))

        blocks = self._make_compute_blocks()
        self.nblks = len(blocks)

        kmax_per_block = []
        for (s, e) in blocks:
            blk = self.annot[s:e]
            Kmax = int((blk != 0).sum(axis=0).max())
            kmax_per_block.append(Kmax)

        if any(k == 0 for k in kmax_per_block):
            zc = sum(1 for k in kmax_per_block if k == 0)
            self.log._log(f"[info] {zc} block(s) have Kmax=0 (phase-1 only skip).")
        if kmax_per_block:
            self.log._log(f"Kmax per block (min/median/max): {min(kmax_per_block)}/{int(np.median(kmax_per_block))}/{max(kmax_per_block)}")

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
        vtiles = [((base + (1 if i < rem else 0) + 63) // 64) * 64 for i in range(ntiles)]
        vtiles[-1] = self.nvecs - sum(vtiles[:-1])
        vtiles = [v for v in vtiles if v > 0]
        assert sum(vtiles) == self.nvecs

        if len(vtiles) == 1:
            self.log._log(f"[auto_vchunk] dtype={self.dtype} target≈{target_gib:.1f} GiB, v_tiles=[{vtiles[0]}] → Xz≈{(self.nsamp * self.nbins * vtiles[0] * itemsize) / (1024**3):.2f} GiB")
        else:
            self.log._log(f"[auto_vchunk] dtype={self.dtype} target≈{target_gib:.1f} GiB, v_tiles={vtiles[0]}×{(len(vtiles) - 1)}+{vtiles[-1]} → Xz≈{(self.nsamp * self.nbins * vtiles[0] * itemsize) / (1024**3):.2f} GiB")
        self.log._log(f"Streaming with BALANCED V-tiles: {vtiles} (total V = {self.nvecs})")

        bed_prefix = self.bed_prefix
        fam_path   = self.fam_path
        row_sel    = self.row_sel if self.row_sel is not None else None
        ddof       = int(self.ddof)
        B          = int(self.nbins)

        meansq_accum = np.zeros((self.nsnps, self.nbins), dtype=self.dtype, order='C')
        Vmax = max(vtiles) if vtiles else 0
        Xz_chunk = None

        ann_blocks = [self.annot[s:e] for (s, e) in blocks]
        inv_blocks = [self.inv_sqrt_resvar_all[s:e] for (s, e) in blocks]

        dev_kind, dev_idx_req = _parse_device_str(getattr(self, "device", "cpu"))
        use_tf32 = bool(getattr(self, "use_tp32", False))
        use_cuda_backend = False
        gcu = None
        dev_idx = None
        if dev_kind == "cuda":
            try:
                import gwldcore_cuda as gcu
                if dev_idx_req is None:
                    dev_idx, _, _ = _pick_cuda_index_auto(gcu)
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

        meansq_chunk = np.zeros_like(meansq_accum, dtype=self.dtype, order='C') if use_cuda_backend else None
        use_mailman_backend = bool((not use_cuda_backend) and self.use_mailman and self.impute_method == "hwe")
        phase1_init = None
        if not use_mailman_backend:
            Xz_chunk = np.empty((self.nsamp, int(self.nbins) * int(Vmax)), dtype=self.dtype, order='F')
            phase1_init = np.empty(B, dtype=np.uint8)

        if self.C is not None:
            N_denom = int(self.N_eff)
        else:
            N_denom = int(self.nsamp - self.ddof + 1)

        total_units = len(vtiles) * len(blocks)
        bar = tqdm(total=total_units, desc="GW-LD progress", unit="task", smoothing=0.2, miniters=1)
        trim_period = int(os.environ.get("SUMMIT_TRIM_PERIOD", "0") or "0")

        ema_p1 = 0.0
        ema_p2 = 0.0
        w1 = 0.5

        try:
            v_start = 0
            for vt_idx, Vt in enumerate(vtiles):
                fuse_resvar_tile = bool(fuse_resvar and vt_idx == 0)
                used_cols = B * Vt
                if use_mailman_backend:
                    Xz_view = np.zeros((self.nsamp, used_cols), dtype=self.dtype, order='C')
                else:
                    Xz_view = Xz_chunk[:, :used_cols]
                    phase1_init.fill(0)

                t1_total = 0.0
                phase1_pref_ex = ThreadPoolExecutor(max_workers=1)
                try:
                    with set_parallelism(omp_threads=t_omp1, blas_threads=t_blas1):
                        try:
                            for blk_idx, (s, e) in enumerate(blocks):
                                if blk_idx + 1 < len(blocks):
                                    s2, e2 = blocks[blk_idx + 1]
                                    phase1_pref_ex.submit(gwldcore.prefetch_bed_block, bed_prefix, fam_path, int(s2), int(e2), 1)
                                kmax_hint = int(kmax_per_block[blk_idx])
                                if kmax_hint == 0 and not fuse_resvar_tile:
                                    bar.update(w1)
                                    continue
                                annot_blk = ann_blocks[blk_idx]
                                inv_right = inv_blocks[blk_idx]
                                inv_out = inv_right if fuse_resvar_tile else None
                                resvar_C = self.C if (fuse_resvar_tile and self.C is not None) else None
                                resvar_R = self.cov_R if (fuse_resvar_tile and self.C is not None) else None
                                resvar_gram = self.cov_gram if (fuse_resvar_tile and self.C is not None) else None
                                t0 = time.perf_counter()
                                if use_mailman_backend:
                                    gwldcore.phase1_compute_Xz_bed_chunk_rowmajor(
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
                                        C=resvar_C,
                                        R=resvar_R,
                                        impute_mode=self.impute_method,
                                        impute_seed=int(self.impute_seed),
                                        resvar_gram=resvar_gram,
                                        inv_out=inv_out,
                                        resvar_eps=float(self.eps_var),
                                    )
                                else:
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
                                        bin_init_mask=phase1_init,
                                        project_right=False,
                                        C=resvar_C,
                                        R=resvar_R,
                                        impute_mode=self.impute_method,
                                        impute_seed=int(self.impute_seed),
                                        resvar_gram=resvar_gram,
                                        inv_out=inv_out,
                                        resvar_eps=float(self.eps_var),
                                    )
                                t1_total += (time.perf_counter() - t0)
                                bar.update(w1)
                        finally:
                            _clear_phase1_native_scratch()
                finally:
                    phase1_pref_ex.shutdown(wait=True)

                if use_mailman_backend:
                    with set_parallelism(omp_threads=self.num_threads, blas_threads=1):
                        if self.C is not None:
                            sum_Xz_view = gwldcore.project_rowmajor_inplace_and_col_sums(Xz_view, self.C, self.cov_R)
                        else:
                            sum_Xz_view = gwldcore.compute_col_sums_rowmajor(Xz_view)
                else:
                    zero_bins = np.flatnonzero(phase1_init == 0)
                    if zero_bins.size:
                        _zero_colmajor_bin_runs(Xz_view, zero_bins, int(Vt))
                    sum_Xz_view = None
                    if self.C is not None:
                        with set_parallelism(omp_threads=1, blas_threads=self.num_threads):
                            gwldcore.project_colmajor_inplace(Xz_view, self.C, self.cov_R)

                t2_total = 0.0
                if use_cuda_backend:
                    meansq_chunk.fill(0)
                    with set_parallelism(omp_threads=1, blas_threads=1):
                        for blk_idx, (s, e) in enumerate(blocks):
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
                                C=None,
                                R=None,
                                N_denom=int(N_denom),
                                use_tf32=use_tf32,
                                device_index=int(dev_idx),
                            )
                            t2_total += (time.perf_counter() - t0)
                            bar.update(1.0 - w1)
                    meansq_accum += (meansq_chunk * Vt)
                else:
                    if use_mailman_backend:
                        pref_ex = ThreadPoolExecutor(max_workers=1)
                        try:
                            with set_parallelism(omp_threads=self.num_threads, blas_threads=1):
                                for blk_idx, (s, e) in enumerate(blocks):
                                    inv_left = inv_blocks[blk_idx]
                                    if blk_idx + 1 < len(blocks):
                                        s2, e2 = blocks[blk_idx + 1]
                                        pref_ex.submit(gwldcore.prefetch_bed_block, bed_prefix, fam_path, int(s2), int(e2), 1)
                                    t0 = time.perf_counter()
                                    gwldcore.phase2_accum_XtXz_bed_mailman_rowmajor(
                                        bed_prefix=bed_prefix,
                                        fam_path=fam_path,
                                        blk_start=int(s),
                                        blk_end=int(e),
                                        row_sel=row_sel,
                                        ddof=ddof,
                                        inv_left=inv_left,
                                        tile_nvecs=int(Vt),
                                        Xz2d=Xz_view,
                                        meansq_accum=meansq_accum,
                                        sum_Xz=sum_Xz_view,
                                        N_denom=int(N_denom),
                                        impute_seed=int(self.impute_seed),
                                    )
                                    t2_total += (time.perf_counter() - t0)
                                    bar.update(1.0 - w1)
                        finally:
                            pref_ex.shutdown(wait=True)
                    else:
                        pref_ex = ThreadPoolExecutor(max_workers=1)
                        try:
                            with set_parallelism(omp_threads=1, blas_threads=self.num_threads):
                                for blk_idx, (s, e) in enumerate(blocks):
                                    inv_left = inv_blocks[blk_idx]
                                    if blk_idx + 1 < len(blocks):
                                        s2, e2 = blocks[blk_idx + 1]
                                        pref_ex.submit(gwldcore.prefetch_bed_block, bed_prefix, fam_path, int(s2), int(e2), 1)

                                    t0 = time.perf_counter()
                                    gwldcore.phase2_accum_XtXz_bed(
                                        bed_prefix=bed_prefix,
                                        fam_path=fam_path,
                                        blk_start=int(s),
                                        blk_end=int(e),
                                        row_sel=row_sel,
                                        ddof=ddof,
                                        inv_left=inv_left,
                                        tile_nvecs=int(Vt),
                                        Xz2d=Xz_view,
                                        meansq_accum=meansq_accum,
                                        C=None,
                                        R=None,
                                        N_denom=int(N_denom),
                                        impute_mode=self.impute_method,
                                        impute_seed=int(self.impute_seed),
                                    )
                                    t2_total += (time.perf_counter() - t0)
                                    bar.update(1.0 - w1)
                        finally:
                            pref_ex.shutdown(wait=True)

                ema_p1 = 0.85 * ema_p1 + 0.15 * max(t1_total, 1e-9)
                ema_p2 = 0.85 * ema_p2 + 0.15 * max(t2_total, 1e-9)
                w1 = float(ema_p1 / (ema_p1 + ema_p2))
                bar.set_postfix_str(f"tile {vt_idx+1}/{len(vtiles)} | w1={w1:.2f} | P1={t1_total:.1f}s P2={t2_total:.1f}s")

                if trim_period > 0 and ((vt_idx + 1) % trim_period == 0):
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
            try:
                gwldcore.clear_phase1_csr_cache()
            except Exception:
                pass

        meansq = (meansq_accum / float(self.nvecs)).astype(self.dtype, copy=False)

        trace_k2_from_ldscore = None
        is_unpartitioned = (self.nbins == 1 and np.isclose(float(self.nsnps_bin[0]), float(self.nsnps)))
        if is_unpartitioned:
            proj_rank = float(self.N_eff - 1.0)
            trace_k2_from_ldscore = (proj_rank * proj_rank) * float(meansq[:, 0].sum(dtype=np.float64)) / float(self.nsnps * self.nsnps)

        if not self.correct_skew:
            self.mu22_block = self.delta_block = self.r2_block = self.rho2_block = self.bias_block = None
        else:
            meansq_raw = np.asarray(meansq, dtype=np.float64, order="C")
            try:
                mu22 = self._estimate_mu22_bins(None)
                if not getattr(self, "is_continuous", False):
                    self._compute_block_corrections(meansq_raw, mu22, None)
                else:
                    self.mu22_block = mu22
                    self.log._log("[fs-corr] Continuous / overlapping annotations detected; stored μ22_block but skipped block-level bias correction.")
            except Exception as e:
                self.log._log(f"[fs-corr] Failed to compute 4th-moment-based corrections: {e}")
                self.mu22_block = self.delta_block = self.r2_block = self.rho2_block = self.bias_block = None

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

        if trace_k2_from_ldscore is not None and self.write_kmoments:
            self._estimate_unpartitioned_kmoments(trace_k2_from_ldscore=trace_k2_from_ldscore, num_probes=256)

        if self.correct_skew:
            try:
                if getattr(self, "delta_block", None) is not None:
                    delta_df = pd.DataFrame(self.delta_block, index=self.l2cols, columns=self.l2cols)
                    delta_out = f"{self.outpath}.gw.delta"
                    delta_df.to_csv(delta_out, sep='\t', float_format='%.8e')
                    self.log._log(f"[fs-corr] Saved block-level δ matrix to: {delta_out}")
                else:
                    self.log._log("[fs-corr] delta_block not available; skipping .gw.delta write.")
            except Exception as e:
                self.log._log(f"[fs-corr] Failed to save δ matrix (.gw.delta): {e}")

        try:
            desc = scores_df.describe(percentiles=[0.25, 0.5, 0.75]).loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
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
                ordered = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                lines = [f"{k:<6} {desc2[k]:.4f}" for k in ordered]
                self.log._log("\n".join(lines))
        except Exception as e:
            self.log._log(f"[warn] Failed to compute summary stats / correlation: {e}")

        self.end_time = utils._get_time()
        self.log._log(f"Calculation of genome-wide LD score ended at " + utils._get_timestr(self.end_time))
        self.runtime = self.end_time - self.start_time
        self.log._log("Runtime: " + format(self.runtime, '.3f') + f" s ({self.runtime // 3600} hr {(self.runtime % 3600) // 60} m {(self.runtime % 60):.3f} s)")
        self.log._save_log(self.outpath + ".gw.log")

    def _estimate_unpartitioned_kmoments(self, trace_k2_from_ldscore, num_probes: int = 128):
        if trace_k2_from_ldscore is None:
            raise ValueError("trace_k2_from_ldscore must be provided for K-moment estimation.")
        if not np.isfinite(trace_k2_from_ldscore):
            raise ValueError(f"trace_k2_from_ldscore is not finite: {trace_k2_from_ldscore}")

        q = int(num_probes)
        if q <= 0:
            raise ValueError("num_probes must be positive.")

        proj_rank = float(self.N_eff - 1.0)
        if proj_rank <= 0.0:
            raise ValueError(f"Projected rank must be positive; got {proj_rank}")

        N = int(self.nsamp)
        mom_dtype = np.float64

        probe_seed = int((np.uint64(self.root_seed) ^ np.uint64(0xD1B54A32D192ED03)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        rng = np.random.default_rng(probe_seed)

        if self.rand_dist == "rademacher":
            Z = rng.integers(0, 2, size=(N, q), dtype=np.int8).astype(mom_dtype, copy=False)
            Z = np.asfortranarray(2.0 * Z - 1.0)
        elif self.rand_dist in ("gaussian", "normal"):
            Z = np.asfortranarray(rng.standard_normal(size=(N, q)).astype(mom_dtype, copy=False))
        elif self.rand_dist == "spherical":
            Z = np.asfortranarray(rng.standard_normal(size=(N, q)).astype(mom_dtype, copy=False))
            norms = np.linalg.norm(Z, axis=0)
            good = norms > 0
            Z[:, good] *= (np.sqrt(N) / norms[good])
        else:
            raise ValueError(f"Unsupported rand_dist for K-moment estimation: {self.rand_dist}")

        Y1 = np.zeros((N, q), dtype=mom_dtype, order="F")
        Y2 = np.zeros((N, q), dtype=mom_dtype, order="F")

        inv_all = np.ascontiguousarray(self.inv_sqrt_resvar_all.astype(mom_dtype, copy=False))
        C_mom = None if self.C is None else np.asfortranarray(self.C.astype(mom_dtype, copy=False))
        R_mom = None if self.cov_R is None else np.asfortranarray(self.cov_R.astype(mom_dtype, copy=False))

        self.log._log(f"[kmom] Estimating unpartitioned GRM moments with {q} sample-space probes (CPU, dtype=float64, seed={probe_seed}, dist={self.rand_dist}).")

        with set_parallelism(omp_threads=(self.num_threads if (self.use_mailman and self.impute_method == "hwe") else 1),
                             blas_threads=(1 if (self.use_mailman and self.impute_method == "hwe") else self.num_threads)):
            if self.use_mailman and self.impute_method == "hwe":
                gwldcore.apply_grm_bed_panel_mailman(
                    bed_prefix=self.bed_prefix,
                    fam_path=self.fam_path,
                    nsnps=int(self.nsnps),
                    step_size=int(self.step_size),
                    row_sel=(self.row_sel if self.row_sel is not None else None),
                    ddof=int(self.ddof),
                    inv_all=inv_all,
                    panel_in=Z,
                    panel_out=Y1,
                    C=(C_mom if C_mom is not None else None),
                    R=(R_mom if R_mom is not None else None),
                    impute_seed=int(self.impute_seed),
                )
                gwldcore.apply_grm_bed_panel_mailman(
                    bed_prefix=self.bed_prefix,
                    fam_path=self.fam_path,
                    nsnps=int(self.nsnps),
                    step_size=int(self.step_size),
                    row_sel=(self.row_sel if self.row_sel is not None else None),
                    ddof=int(self.ddof),
                    inv_all=inv_all,
                    panel_in=Y1,
                    panel_out=Y2,
                    C=(C_mom if C_mom is not None else None),
                    R=(R_mom if R_mom is not None else None),
                    impute_seed=int(self.impute_seed),
                )
            else:
                gwldcore.apply_grm_bed_panel(
                    bed_prefix=self.bed_prefix,
                    fam_path=self.fam_path,
                    nsnps=int(self.nsnps),
                    step_size=int(self.step_size),
                    row_sel=(self.row_sel if self.row_sel is not None else None),
                    ddof=int(self.ddof),
                    inv_all=inv_all,
                    panel_in=Z,
                    panel_out=Y1,
                    C=(C_mom if C_mom is not None else None),
                    R=(R_mom if R_mom is not None else None),
                    impute_mode=self.impute_method,
                    impute_seed=int(self.impute_seed),
                )
                gwldcore.apply_grm_bed_panel(
                    bed_prefix=self.bed_prefix,
                    fam_path=self.fam_path,
                    nsnps=int(self.nsnps),
                    step_size=int(self.step_size),
                    row_sel=(self.row_sel if self.row_sel is not None else None),
                    ddof=int(self.ddof),
                    inv_all=inv_all,
                    panel_in=Y1,
                    panel_out=Y2,
                    C=(C_mom if C_mom is not None else None),
                    R=(R_mom if R_mom is not None else None),
                    impute_mode=self.impute_method,
                    impute_seed=int(self.impute_seed),
                )

        k1_each = np.sum(Z * Y1, axis=0, dtype=np.float64)
        k2_each = np.sum(Y1 * Y1, axis=0, dtype=np.float64)
        k3_each = np.sum(Y1 * Y2, axis=0, dtype=np.float64)
        k4_each = np.sum(Y2 * Y2, axis=0, dtype=np.float64)

        trace_K_probe = float(k1_each.mean())
        trace_K2_probe = float(k2_each.mean())
        trace_K3_probe = float(k3_each.mean())
        trace_K4_probe = float(k4_each.mean())

        trace_K_probe_se = float(k1_each.std(ddof=1) / np.sqrt(q)) if q > 1 else 0.0
        trace_K2_probe_se = float(k2_each.std(ddof=1) / np.sqrt(q)) if q > 1 else 0.0
        trace_K3_probe_se = float(k3_each.std(ddof=1) / np.sqrt(q)) if q > 1 else 0.0
        trace_K4_probe_se = float(k4_each.std(ddof=1) / np.sqrt(q)) if q > 1 else 0.0

        trace_K2_used = float(trace_k2_from_ldscore)
        t0_rank = float(trace_K2_used - proj_rank)
        t1_rank = float(trace_K3_probe - 2.0 * trace_K2_used + proj_rank)
        t2_rank = float(trace_K4_probe - 2.0 * trace_K3_probe + trace_K2_used)

        alpha_probe = float(trace_K_probe / proj_rank)
        t0_probealpha = float(trace_K2_used - (trace_K_probe * trace_K_probe) / proj_rank)
        t1_probealpha = float(trace_K3_probe - 2.0 * alpha_probe * trace_K2_used + (alpha_probe * alpha_probe) * trace_K_probe)
        t2_probealpha = float(trace_K4_probe - 2.0 * alpha_probe * trace_K3_probe + (alpha_probe * alpha_probe) * trace_K2_used)

        alpha_probe_err = float(abs(alpha_probe - 1.0))
        k2_probe_relerr = float(abs(trace_K2_probe - trace_K2_used) / max(abs(trace_K2_used), 1e-12))
        s4_rank = float(t2_rank - 2.0 * t1_rank + t0_rank)
        delta_reff_rank = float((t0_rank * t0_rank) / s4_rank) if np.isfinite(s4_rank) and s4_rank > 0.0 else np.nan

        alpha_flag = "OK" if alpha_probe_err <= 5e-3 else ("WARN" if alpha_probe_err <= 1e-2 else "BAD")
        mc_flag = "OK" if k2_probe_relerr <= 5e-2 else ("WARN" if k2_probe_relerr <= 1e-1 else "BAD")
        if np.isfinite(delta_reff_rank):
            spike_flag = "LOW_EFFECTIVE_RANK" if delta_reff_rank < 50.0 else ("MODERATE" if delta_reff_rank < 200.0 else "DIFFUSE")
        else:
            spike_flag = "NA"

        out_df = pd.DataFrame([{
            "nsnps": int(self.nsnps), "proj_rank": proj_rank, "num_probes": q, "probe_seed": probe_seed,
            "trace_K_probe": trace_K_probe, "trace_K_probe_se": trace_K_probe_se,
            "trace_K_target_rank": proj_rank, "trace_K2_from_ldscore": trace_K2_used,
            "trace_K2_probe": trace_K2_probe, "trace_K2_probe_se": trace_K2_probe_se,
            "trace_K3_probe": trace_K3_probe, "trace_K3_probe_se": trace_K3_probe_se,
            "trace_K4_probe": trace_K4_probe, "trace_K4_probe_se": trace_K4_probe_se,
            "alpha_rank": 1.0, "alpha_probe": alpha_probe,
            "t0_rank": t0_rank, "t1_rank": t1_rank, "t2_rank": t2_rank,
            "t0_probealpha": t0_probealpha, "t1_probealpha": t1_probealpha, "t2_probealpha": t2_probealpha,
            "alpha_probe_err": alpha_probe_err, "k2_probe_relerr": k2_probe_relerr,
            "delta_reff_rank": delta_reff_rank, "alpha_flag": alpha_flag, "mc_flag": mc_flag, "spike_flag": spike_flag,
        }])

        kmom_out = f"{self.outpath}.gw.kmoments"
        out_df.to_csv(kmom_out, sep="\t", index=False, float_format="%.10e")
        row = out_df.iloc[0]
        self.log._log(f"[kmom] Saved higher-order GRM moments to: {kmom_out}")
        self.log._log(
            "[kmom] "
            f"tr(K) probe={row['trace_K_probe']:.8e} (mcse {row['trace_K_probe_se']:.3e}), "
            f"target rank={row['trace_K_target_rank']:.8e}, "
            f"tr(K^2) used={row['trace_K2_from_ldscore']:.8e}, "
            f"tr(K^2) probe={row['trace_K2_probe']:.8e} (mcse {row['trace_K2_probe_se']:.3e}), "
            f"tr(K^3)={row['trace_K3_probe']:.8e} (mcse {row['trace_K3_probe_se']:.3e}), "
            f"tr(K^4)={row['trace_K4_probe']:.8e} (mcse {row['trace_K4_probe_se']:.3e})"
        )
        self.log._log(
            "[kmom] "
            f"alpha_rank=1.00000000e+00, alpha_probe={row['alpha_probe']:.8e}, "
            f"t0_rank={row['t0_rank']:.8e}, t1_rank={row['t1_rank']:.8e}, t2_rank={row['t2_rank']:.8e}, "
            f"t0_probealpha={row['t0_probealpha']:.8e}, t1_probealpha={row['t1_probealpha']:.8e}, t2_probealpha={row['t2_probealpha']:.8e}"
        )
        self.log._log(
            "[kmom:diag] "
            f"alpha_probe_err={alpha_probe_err:.3e} [{alpha_flag}] ; "
            f"k2_probe_relerr={k2_probe_relerr:.3%} [{mc_flag}] ; "
            f"delta_reff_rank={delta_reff_rank:.3g} [{spike_flag}]"
        )

        return out_df.iloc[0].to_dict()
