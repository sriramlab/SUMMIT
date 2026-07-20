from __future__ import annotations

import os
import sys
import time
import ctypes
import threading
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from bed_reader import open_bed

from .. import utils
from .genotype_source import (
    PgenBlockReader,
    read_aligned_annotations,
    read_fam_sample_ids,
    read_psam_sample_ids,
    read_pvar_variants,
    resolve_genotype_input,
    validate_variant_metadata,
)

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

try:
    from .. import winldcore
    _WINLDCORE_IMPORT_ERROR = None
except Exception as package_error:  # pragma: no cover
    try:
        import winldcore
        _WINLDCORE_IMPORT_ERROR = None
    except Exception as e:  # pragma: no cover
        winldcore = None
        _WINLDCORE_IMPORT_ERROR = e


# -------------------- env / perf helpers --------------------


def _canonical_bfile_prefix(x: str) -> str:
    """Return PLINK bfile prefix: strip trailing .bed/.bim/.fam if present; otherwise leave as-is."""
    s = str(x)
    for ext in (".bed", ".bim", ".fam"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s



def _trim_malloc_best_effort():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass



def _set_openmp_threads_runtime(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    try:
        winldcore.set_num_threads(n)
    except Exception:
        pass



def _get_openmp_threads_runtime() -> Optional[int]:
    try:
        return int(winldcore.get_max_threads())
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
def _set_parallelism(
    omp_threads: Optional[int] = None,
    blas_threads: Optional[int] = None,
    decode_threads_cap: Optional[int] = None,
):
    """
    Coordinate OpenMP, BLAS, and decoder thread caps for the C++ backend.

    BLAS threads are controlled via threadpoolctl when available. OpenMP is driven
    explicitly through winldcore.set_num_threads(...) so phase-level changes take
    effect reliably at runtime.
    """
    prev_env = {}
    for key in (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DYNAMIC",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "SUMMIT_DECODE_THREADS_CAP",
    ):
        prev_env[key] = os.environ.get(key)

    prev_omp = _get_openmp_threads_runtime() if omp_threads is not None else None
    if omp_threads is not None:
        _set_openmp_threads_runtime(int(omp_threads))

    if decode_threads_cap is not None:
        os.environ["SUMMIT_DECODE_THREADS_CAP"] = str(max(1, int(decode_threads_cap)))

    b = None
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



def _parse_rand_samp(rand_samp, n: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    SUMMIT semantics:
      - float in (0, 1] -> ratio of individuals
      - int >= 100 -> exact number of individuals
    Returns sorted row indices or None.
    """
    if rand_samp is None:
        return None

    if isinstance(rand_samp, (float, np.floating)):
        r = float(rand_samp)
        if not (0.0 < r <= 1.0):
            raise ValueError("--rand-samp float must be in (0, 1].")
        k = int(np.floor(r * n))
        k = max(1, min(k, n))
    else:
        k = int(rand_samp)
        if not (100 <= k <= n):
            raise ValueError("--rand-samp int must be in [100, N].")

    idx = np.sort(rng.choice(np.arange(n, dtype=int), size=k, replace=False))
    return idx


# -------------------- covariates (QR projection) --------------------


def _read_cov_qr(
    cov_path: str,
    fam_path: Optional[str],
    log,
    sample_idx: Optional[np.ndarray] = None,
    add_intercept: bool = True,
    sample_ids: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read covariates aligned to .fam order, optional subselect rows, drop rows with any NA,
    then reduced-QR on [1, covs] (optional intercept) returning Q and Q^T, plus kept GLOBAL indices.

    Returns
    -------
    C : (N_kept, p_eff) float64  # orthonormal columns (Q)
    R : (p_eff, N_kept) float64  # Q^T
    keep_idx_global : (N_kept,) int64 indices into original .fam/.bed rows
    """
    if sample_ids is None:
        if fam_path is None:
            raise ValueError("fam_path or sample_ids must be provided.")
        fam = read_fam_sample_ids(fam_path)
    else:
        fam = pd.DataFrame(sample_ids)[["FID", "IID"]].copy()
    fam[["FID", "IID"]] = fam[["FID", "IID"]].astype(str)
    cov = pd.read_csv(
        cov_path,
        sep=r"\s+",
        dtype={"FID": str, "IID": str},
        keep_default_na=False,
    )

    if ("FID" not in cov.columns) or ("IID" not in cov.columns):
        raise ValueError("Covariate file must contain 'FID' and 'IID' columns.")
    if cov.duplicated(subset=["FID", "IID"]).any():
        raise ValueError("Covariate file contains duplicate FID/IID rows.")

    merged = fam.merge(cov, on=["FID", "IID"], how="left", indicator=True)
    miss = int((merged["_merge"] != "both").sum())
    if miss:
        raise ValueError(f"{miss} .fam samples not found in covariate file (FID/IID mismatch).")
    merged.drop(columns=["_merge"], inplace=True)

    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx, dtype=int)
        merged = merged.iloc[sample_idx].reset_index(drop=True)

    df = merged.drop(columns=["FID", "IID"]).copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep_mask = ~df.isna().any(axis=1)
    dropped = int((~keep_mask).sum())
    log._log(f"[win][cov] Dropping {dropped} samples due to missing covariates.")
    df = df.loc[keep_mask].reset_index(drop=True)

    if df.shape[1] == 0:
        raise ValueError("After parsing covariates, no covariate columns remain.")

    zvc = (df.std(ddof=0) == 0)
    if zvc.any():
        drop_cols = zvc.index[zvc].tolist()
        log._log(f"[win][cov] Dropping {len(drop_cols)} constant covariates.")
        df.drop(columns=drop_cols, inplace=True)

    if df.shape[1] == 0:
        raise ValueError("After dropping constant covariates, none remain.")

    X = df.to_numpy(dtype=np.float64, copy=False)
    if add_intercept:
        X = np.column_stack([np.ones((X.shape[0], 1), dtype=np.float64), X])

    Q, _ = np.linalg.qr(X, mode="reduced")
    C = np.asfortranarray(Q, dtype=np.float64)
    R = np.asfortranarray(Q.T, dtype=np.float64)

    km = np.flatnonzero(keep_mask.values if isinstance(keep_mask, pd.Series) else keep_mask).astype(np.int64, copy=False)
    if sample_idx is not None:
        keep_idx_global = np.asarray(sample_idx, dtype=int)[km]
    else:
        keep_idx_global = km

    log._log(f"[win][cov] Kept {C.shape[0]} samples; p_eff={C.shape[1]} (add_intercept={add_intercept}).")
    return C, R, keep_idx_global


def _window_cache_bytes(cache_mb: int, target_mem_gb: Optional[float] = None) -> int:
    if int(cache_mb) == 0:
        return 0
    if int(cache_mb) > 0:
        return int(cache_mb) * 1024 * 1024
    env_cache_mb = os.environ.get("SUMMIT_WIN_CACHE_MB")
    if env_cache_mb is not None:
        try:
            value = int(env_cache_mb)
        except ValueError as exc:
            raise ValueError("SUMMIT_WIN_CACHE_MB must be an integer number of MiB.") from exc
        return max(0, value) * 1024 * 1024
    if target_mem_gb is not None:
        target_mem_gb = float(target_mem_gb)
        if not np.isfinite(target_mem_gb) or target_mem_gb <= 0.0:
            raise ValueError("target_mem must be finite and positive when specified.")
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:  # pragma: no cover - conservative platform fallback
        available = 8 * 1024**3
    one_gib = 1024**3
    cap = min(4 * one_gib, max(0, available // 8))
    if target_mem_gb is not None:
        cap = min(cap, int(target_mem_gb * one_gib) // 4)
    return max(0, cap)


def _window_panel_cols(n_rows: int, chunk_size: int, panel_cols: int, cache_bytes: int) -> int:
    if int(panel_cols) > 0:
        return max(1, min(int(panel_cols), int(chunk_size)))
    if cache_bytes > 0:
        target = max(64 * 1024**2, min(2 * 1024**3, cache_bytes // 8))
    else:
        target = 512 * 1024**2
    cols = max(1, min(int(chunk_size), 4096, target // max(8, 8 * int(n_rows))))
    cross_cap = int(np.sqrt((1024**3) / 8.0))
    cols = min(cols, max(64, cross_cap))
    if cols >= 32:
        cols = max(32, (cols // 32) * 32)
    return max(1, min(cols, int(chunk_size)))


class _PgenPreparedPanelCache:
    """Bounded cache of projected, unit-variance PGEN dosage panels."""

    def __init__(
        self,
        reader: PgenBlockReader,
        n_rows: int,
        C: Optional[np.ndarray],
        R: Optional[np.ndarray],
        capacity_bytes: int,
    ) -> None:
        self.reader = reader
        self.n_rows = int(n_rows)
        self.C = None if C is None else np.asarray(C, dtype=np.float64, order="F")
        self.R = None if R is None else np.asarray(R, dtype=np.float64, order="F")
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.current_bytes = 0
        self._items: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def _prepare(self, start: int, end: int) -> np.ndarray:
        # The reader returns a reusable view, so every cached panel needs one
        # owned copy. float64 matches the established deterministic BED path.
        G = np.array(
            self.reader.read_standardized_block(start, end),
            dtype=np.float64,
            order="F",
            copy=True,
        )
        if self.C is not None and self.R is not None and self.C.shape[1] > 0:
            G -= self.C @ (self.R @ G)

        # PGEN standardization is over observed dosages. Re-standardizing after
        # zero/mean imputation and projection puts every nonconstant column on
        # the exact N-row correlation scale used by winldcore's BED path.
        G -= G.mean(axis=0, keepdims=True)
        ss = np.einsum("ij,ij->j", G, G, dtype=np.float64)
        good = np.isfinite(ss) & (ss > 0.0)
        if np.any(good):
            G[:, good] *= np.sqrt(float(self.n_rows) / ss[good]).reshape(1, -1)
        if np.any(~good):
            G[:, ~good] = 0.0
        return np.asfortranarray(G)

    def get(self, start: int, end: int) -> np.ndarray:
        key = (int(start), int(end))
        cached = self._items.pop(key, None)
        if cached is not None:
            self._items[key] = cached
            return cached

        panel = self._prepare(*key)
        need = int(panel.nbytes)
        if self.capacity_bytes <= 0 or need > self.capacity_bytes:
            return panel
        while self._items and self.current_bytes + need > self.capacity_bytes:
            _, old = self._items.popitem(last=False)
            self.current_bytes -= int(old.nbytes)
        self._items[key] = panel
        self.current_bytes += need
        return panel


class WindowedLDScore:
    """
    Windowed LD score computation for the deterministic sliding-window path.

    BED computation is delegated to the C++ winldcore module; PGEN dosage
    panels use the same mathematical preparation and accumulation in NumPy:
      - PLINK BED decoding and imputation (mean / HWE)
      - optional QR-covariate projection
      - post-projection re-standardization
      - deterministic tile/window LD-score accumulation
      - MAF pass for .win.M_5_50
    """

    def __init__(
        self,
        bed_path: str,
        annot_path: Optional[str],
        out_path: str,
        covar_path: Optional[str] = None,
        ld_wind_kb: float = 20000.0,
        log=None,
        verbose: bool = False,
        dtype: str = "float64",
        rand_samp=None,
        ddof: int = 1,
        num_threads: Optional[int] = None,
        seed: Optional[int] = None,
        step_size: Optional[int] = None,
        impute_method: str = "mean",
        panel_cols: Optional[int] = None,
        cache_mb: int = -1,
        target_mem: Optional[float] = None,
    ):
        if log is None:
            raise ValueError("WindowedLDScore requires a Logger instance (log=...).")
        if winldcore is None:
            raise ImportError(
                "winldcore could not be imported. Build the C++ extension before running the windowed LD path."
            ) from _WINLDCORE_IMPORT_ERROR

        self.log = log
        self.verbose = bool(verbose)

        raw_genotype_path = str(bed_path)
        if "@" in raw_genotype_path:
            raise ValueError(
                "WindowedLDScore expects a single genome-wide genotype prefix (no '@')."
            )
        self.genotype_input = resolve_genotype_input(raw_genotype_path)
        self.genotype_format = self.genotype_input.format
        self.genotype_prefix = self.genotype_input.prefix
        self.bed_prefix = self.genotype_prefix if self.genotype_format == "bed" else None
        self.fam_path = self.genotype_input.sample_path if self.genotype_format == "bed" else None
        self.bim_path = self.genotype_input.variant_path if self.genotype_format == "bed" else None
        self.pgen_path = self.genotype_input.genotype_path if self.genotype_format == "pgen" else None
        self.pvar_path = self.genotype_input.variant_path if self.genotype_format == "pgen" else None
        self.psam_path = self.genotype_input.sample_path if self.genotype_format == "pgen" else None
        self._pgen_reader = None

        if self.genotype_format == "bed":
            self.bed_file = self.genotype_input.genotype_path
            self.G = open_bed(self.bed_file)
            self.nsamp0, self.nsnps = self.G.shape
            self.sample_ids = read_fam_sample_ids(self.fam_path)
            self.snplist = None
        else:
            self.bed_file = None
            self.G = None
            self.sample_ids = read_psam_sample_ids(self.psam_path)
            self.snplist = validate_variant_metadata(
                read_pvar_variants(self.pvar_path), source=f"PVAR '{self.pvar_path}'"
            )
            self.nsamp0 = int(len(self.sample_ids))
            self.nsnps = int(len(self.snplist))
            self.bp_all = self.snplist["BP"].to_numpy(dtype=np.int64, copy=False)

        self.ld_wind_kb = float(ld_wind_kb)
        if not np.isfinite(self.ld_wind_kb) or self.ld_wind_kb <= 0:
            raise ValueError("--ld-wind-kb must be finite and positive.")

        self.dtype = str(dtype)
        if self.dtype not in ("float32", "float64", "f4", "f8") and verbose:
            self.log._log(f"[win][note] Unrecognized dtype='{self.dtype}'. The C++ backend computes in float64.")
        elif self.dtype not in ("float64", "f8") and verbose:
            self.log._log(f"[win][note] dtype={self.dtype} requested, but the C++ backend computes in float64 for exactness.")

        self.ddof = int(ddof)
        if self.ddof != 0 and self.verbose:
            self.log._log(f"[win][note] ddof={self.ddof} passed, but windowed LD uses ddof=0 for genotype standardization.")

        self.chunk_size = int(step_size) if step_size is not None else 10000
        if self.chunk_size <= 0:
            raise ValueError("Internal chunk_size must be positive.")

        if num_threads is None or int(num_threads) <= 0:
            self.num_threads = max(1, os.cpu_count() or 1)
        else:
            self.num_threads = int(num_threads)

        self.outpath = out_path
        self.panel_cols = 0 if panel_cols is None else int(panel_cols)
        self.cache_mb = int(cache_mb)
        if self.panel_cols < 0:
            raise ValueError("panel_cols must be positive when specified.")
        if self.cache_mb < -1:
            raise ValueError("cache_mb must be -1 (automatic), 0, or a positive MiB value.")
        self.target_mem = None if target_mem is None else float(target_mem)
        self.cache_bytes = _window_cache_bytes(self.cache_mb, self.target_mem)
        # Always pass an explicit bounded value to the native backend so its
        # cache does not infer node-wide free memory inside a scheduler job.
        self.effective_cache_mb = int(self.cache_bytes // (1024 * 1024))
        self._pgen_io_config_logged = False

        self.impute_method = str(impute_method).strip().lower()
        if self.impute_method not in ("mean", "hwe"):
            raise ValueError("impute_method must be 'mean' or 'hwe'.")
        if self.genotype_format == "pgen" and self.impute_method != "mean":
            raise ValueError("PGEN windowed LD scores support dosage mean imputation only.")

        if seed is None:
            self.root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            if self.verbose:
                self.log._log(f"[seed] No seed provided; using generated root seed {self.root_seed}")
        else:
            self.root_seed = int(seed)
        self.impute_seed = int((np.uint64(self.root_seed) ^ np.uint64(0xA24BAED4963EE407)) & np.uint64(0xFFFFFFFFFFFFFFFF))

        self.start_time = utils._get_time()
        self.log._log("Windowed LD score calculation started at: " + utils._get_timestr(self.start_time))
        backend = "native C++ BED" if self.genotype_format == "bed" else "streamed PGEN/NumPy"
        self.log._log(
            f"[win][backend] {backend} (impute={self.impute_method}, "
            f"impute_seed={self.impute_seed})"
        )

        rng = np.random.default_rng(self.root_seed)
        self.row_sel = _parse_rand_samp(rand_samp, self.nsamp0, rng)
        if self.row_sel is not None:
            k = int(len(self.row_sel))
            self.log._log(f"Randomly subsampling individuals: {k}/{self.nsamp0} ({k/self.nsamp0:.1%})")

        if self.genotype_format == "bed":
            self._read_bim(self.bim_path)
        else:
            self.log._log(f"[win] Reading PVAR metadata: {self.pvar_path}")
        self._validate_variant_order()
        self._read_annot(annot_path)

        self.log._log(f"Number of samples (pre-filter): {self.nsamp0}")
        self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {tuple(self.annot.shape)}")
        self.log._log(f"Nbins: {self.nbins}")

        self.C = None
        self.cov_R = None
        self.p_eff = 0

        if covar_path is not None:
            C, R, keep_idx_global = _read_cov_qr(
                cov_path=covar_path,
                fam_path=self.fam_path,
                log=self.log,
                sample_idx=self.row_sel,
                add_intercept=True,
                sample_ids=self.sample_ids,
            )
            self.C = C
            self.cov_R = R
            self.row_sel = np.asarray(keep_idx_global, dtype=int)
            self.p_eff = int(self.C.shape[1])
            self.nsamp = int(self.C.shape[0])
            self.log._log(f"Final sample count after covariate filtering/subsample: {self.nsamp}")
            self.log._log(f"Covariate-adjusted partial correlations (N_rows={self.nsamp}, p={self.p_eff}).")
        else:
            self.nsamp = int(len(self.row_sel)) if self.row_sel is not None else int(self.nsamp0)

        # Genotypes are centered even without an explicit covariate matrix.  If
        # C is present, it already includes the intercept.  Thus correlations
        # live in an N-p dimensional residual subspace (N-1 without C), and the
        # finite-sample r^2 correction denominator is corr_dim - 1.
        self.corr_dim = self.nsamp - (self.p_eff if self.C is not None else 1)

        if self.genotype_format == "pgen":
            # Panel sizing is finalized lazily from the selected sample count.
            self.log._log(
                "[win][pgen] Using streamed REF-dosage panels with bounded LRU caching; "
                "no chromosome-wide genotype matrix will be materialized."
            )

        if self.corr_dim <= 1:
            raise ValueError(
                f"[win] Too few residual dimensions for windowed LD scores: "
                f"n={self.nsamp}, p_eff={self.p_eff}, corr_dim={self.corr_dim}."
            )

        try:
            winldcore.set_verbose(bool(self.verbose))
            winldcore.set_num_threads(int(self.num_threads))
        except Exception as e:
            if self.verbose:
                self.log._log(f"[win][warn] Failed to set C++ verbosity / threads: {e}")

        if self.verbose:
            cache_msg = (
                f"auto->{self.effective_cache_mb}"
                if self.cache_mb < 0 else str(self.cache_mb)
            )
            panel_msg = "auto" if self.panel_cols <= 0 else str(self.panel_cols)
            self.log._log(
                f"[win] ld_wind_kb={self.ld_wind_kb}, chunk_size={self.chunk_size}, "
                f"panel_cols={panel_msg}, cache_mb={cache_msg}, threads={self.num_threads}"
            )

        self.win_ldscore: Optional[np.ndarray] = None

    def close(self) -> None:
        reader = getattr(self, "_pgen_reader", None)
        if reader is not None:
            reader.close()
            self._pgen_reader = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------ BIM / annotation ------------------

    def _read_bim(self, bim_path: str):
        self.log._log(f"[win] Reading BIM: {bim_path}")
        snplist = pd.read_csv(bim_path, header=None, sep=r"\s+")
        snplist.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        if len(snplist) != self.nsnps:
            raise ValueError(f"[win] .bed SNPs ({self.nsnps}) != .bim rows ({len(snplist)})")
        self.snplist = validate_variant_metadata(snplist, source=f"BIM '{bim_path}'")
        self.bp_all = self.snplist["BP"].to_numpy(dtype=np.int64, copy=False)

    def _validate_variant_order(self) -> None:
        chrom = self.snplist["CHR"].to_numpy(dtype=str, copy=False)
        bp = self.snplist["BP"].to_numpy(dtype=np.int64, copy=False)
        seen: set[str] = set()
        start = 0
        while start < len(chrom):
            label = str(chrom[start])
            if label in seen:
                raise ValueError(
                    f"Chromosome {label!r} occurs in multiple noncontiguous blocks; "
                    "sort genotype variants by chromosome and position."
                )
            seen.add(label)
            end = start + 1
            while end < len(chrom) and chrom[end] == label:
                end += 1
            if np.any(bp[start + 1:end] < bp[start:end - 1]):
                raise ValueError(
                    f"BP is not sorted within chromosome {label!r}; "
                    "sort genotype variants by chromosome and position."
                )
            start = end

    def _read_annot(self, annot_path: Optional[str]):
        if annot_path is None:
            self.l2cols = ["L2_0"]
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.is_continuous = False
            self.log._log("[win] No annotation: using single-bin (all SNPs).")
            return

        self.l2cols, self.annot, self.is_continuous = read_aligned_annotations(
            annot_path,
            self.snplist,
            log=self.log,
            source_label=("BIM" if self.genotype_format == "bed" else "PVAR"),
        )
        self.nbins = int(self.annot.shape[1])
        self.log._log(f"[win] Read aligned annotation matrix: shape={self.annot.shape}")

        self.log._log(
            f"[win] nsamp0={self.nsamp0}, nsnps={self.nsnps}, nbins={self.nbins}, continuous={bool(self.is_continuous)}"
        )

    # ------------------ C++ core wrappers ------------------

    def _ensure_pgen_reader(self, panel_cols: int) -> PgenBlockReader:
        if self.genotype_format != "pgen":
            raise RuntimeError("PGEN reader requested for non-PGEN input.")
        required_capacity = min(int(panel_cols), int(self.nsnps))
        reader = self._pgen_reader
        if reader is None:
            reader = PgenBlockReader(
                pgen_path=self.pgen_path,
                raw_sample_ct=int(self.nsamp0),
                variant_ct=int(self.nsnps),
                sample_subset=self.row_sel,
                step_size=max(1, required_capacity),
                dtype=np.float64,
                ddof=0,
                standardize_threads=self.num_threads,
            )
            self._pgen_reader = reader
        elif reader.block_capacity < required_capacity:
            raise RuntimeError("Existing PGEN decode buffer is smaller than the requested panel.")
        return reader

    def _compute_maf(self) -> np.ndarray:
        if self.genotype_format == "pgen":
            self.log._log("[win][pgen] Computing dosage MAF for .win.M_5_50.")
            cache_bytes = self.cache_bytes
            panel_cols = _window_panel_cols(
                self.nsamp, self.chunk_size, self.panel_cols, cache_bytes
            )
            reader = self._ensure_pgen_reader(panel_cols)
            maf = np.zeros(self.nsnps, dtype=np.float64)
            for s in range(0, self.nsnps, panel_cols):
                e = min(self.nsnps, s + panel_cols)
                dosage = reader.read_dosage_block(s, e)
                valid = np.isfinite(dosage) & (dosage >= 0.0) & (dosage <= 2.0)
                nobs = valid.sum(axis=0, dtype=np.int64)
                sums = np.where(valid, dosage, 0.0).sum(axis=0, dtype=np.float64)
                freq = np.divide(
                    sums,
                    2.0 * nobs,
                    out=np.zeros(e - s, dtype=np.float64),
                    where=nobs > 0,
                )
                np.clip(freq, 0.0, 1.0, out=freq)
                maf[s:e] = np.minimum(freq, 1.0 - freq)
            return maf

        self.log._log("[win] Computing MAF for .win.M_5_50 via C++ core.")
        step = int(max(1024, min(self.nsnps, self.chunk_size)))
        with _set_parallelism(omp_threads=self.num_threads, blas_threads=1, decode_threads_cap=self.num_threads):
            maf = winldcore.compute_maf_bed(
                bed_prefix=self.bed_prefix,
                fam_path=self.fam_path,
                nsnps=int(self.nsnps),
                step_size=step,
                row_sel=(self.row_sel if self.row_sel is not None else None),
            )
        return np.asarray(maf, dtype=np.float64, order="C")

    @staticmethod
    def _accumulate_pgen_self(
        X: np.ndarray,
        annot: np.ndarray,
        out: np.ndarray,
        n_rows: int,
        corr_dim: int,
        bp: np.ndarray,
        window_bp: float,
    ) -> None:
        corr = (X.T @ X) / float(n_rows)
        r2 = corr * corr
        r2 -= (1.0 - r2) / float(corr_dim - 1)
        bp = np.asarray(bp, dtype=np.int64)
        for j, right_bp in enumerate(bp):
            outside = np.abs(bp - int(right_bp)) > float(window_bp)
            r2[outside, j] = 0.0
        out += r2 @ annot

    @staticmethod
    def _accumulate_pgen_cross(
        X_left: np.ndarray,
        X_right: np.ndarray,
        annot_left: np.ndarray,
        annot_right: np.ndarray,
        out_left: np.ndarray,
        out_right: np.ndarray,
        n_rows: int,
        corr_dim: int,
        bp_left: np.ndarray,
        bp_right: np.ndarray,
        window_bp: float,
    ) -> None:
        corr = (X_left.T @ X_right) / float(n_rows)
        r2 = corr * corr
        r2 -= (1.0 - r2) / float(corr_dim - 1)
        bp_left = np.asarray(bp_left, dtype=np.int64)
        bp_right = np.asarray(bp_right, dtype=np.int64)
        for j, right_bp in enumerate(bp_right):
            outside = np.abs(bp_left - int(right_bp)) > float(window_bp)
            r2[outside, j] = 0.0
        out_left += r2 @ annot_right
        out_right += r2.T @ annot_left

    def _compute_chrom_ldscores_pgen(self, s: int, e: int, pbar=None) -> np.ndarray:
        bp = self.bp_all[s:e]
        if np.any(bp[1:] < bp[:-1]):
            raise ValueError("BP not sorted within chromosome block. Sort the PVAR by CHR+POS.")
        m = int(e - s)
        cache_bytes = self.cache_bytes
        panel_cols = _window_panel_cols(
            self.nsamp, self.chunk_size, self.panel_cols, cache_bytes
        )
        reader = self._ensure_pgen_reader(panel_cols)
        cache = _PgenPreparedPanelCache(
            reader=reader,
            n_rows=self.nsamp,
            C=self.C,
            R=self.cov_R,
            capacity_bytes=cache_bytes,
        )
        if not self._pgen_io_config_logged:
            self.log._log(
                f"[win][pgen] Effective panel_cols={panel_cols}; prepared-panel cache="
                f"{cache_bytes / 1024**2:.0f} MiB; corr_dim={self.corr_dim}."
            )
            self._pgen_io_config_logged = True
        annot = np.asarray(self.annot[s:e, :], dtype=np.float64, order="C")
        out = np.zeros((m, self.nbins), dtype=np.float64)
        ntiles = (m + self.chunk_size - 1) // self.chunk_size

        # Select a chunk-rounded candidate superset for efficient panel GEMMs.
        # Each panel product is then masked by the exact BP distance, so
        # step_size and panel boundaries cannot change the mathematical window.
        window_bp = float(self.ld_wind_kb) * 1000.0
        left = np.searchsorted(bp, bp.astype(np.float64) - window_bp, side="left")
        shifted = np.flatnonzero(left > 0)
        first_pos = int(shifted[0]) if shifted.size else -1
        b0 = ((first_pos if first_pos >= 0 else m) + self.chunk_size - 1) // self.chunk_size
        b0 *= self.chunk_size
        prefix_tiles = b0 // self.chunk_size

        def panels(local_start: int, local_end: int):
            return [
                (p0, min(local_end, p0 + panel_cols))
                for p0 in range(local_start, local_end, panel_cols)
            ]

        with _set_parallelism(
            omp_threads=self.num_threads,
            blas_threads=self.num_threads,
            decode_threads_cap=self.num_threads,
        ):
            for t in range(ntiles):
                t0 = t * self.chunk_size
                t1 = min(m, t0 + self.chunk_size)
                target_panels = panels(t0, t1)
                # Keep the target tile alive throughout all crossings.  This
                # bounds active memory by the two tiles being multiplied and
                # prevents a small LRU cache from repeatedly decoding targets.
                target_loaded = [
                    (p0, p1, cache.get(s + p0, s + p1))
                    for p0, p1 in target_panels
                ]

                if t < prefix_tiles:
                    a_start = 0
                else:
                    span = int(t0 - left[t0])
                    left_tiles = (span + self.chunk_size - 1) // self.chunk_size
                    a_start = max(0, t - left_tiles)

                for a in range(a_start, t):
                    a0 = a * self.chunk_size
                    a1 = min(m, a0 + self.chunk_size)
                    left_loaded = [
                        (p0, p1, cache.get(s + p0, s + p1))
                        for p0, p1 in panels(a0, a1)
                    ]
                    for lp0, lp1, X_left in left_loaded:
                        for rp0, rp1, X_right in target_loaded:
                            self._accumulate_pgen_cross(
                                X_left,
                                X_right,
                                annot[lp0:lp1],
                                annot[rp0:rp1],
                                out[lp0:lp1],
                                out[rp0:rp1],
                                self.nsamp,
                                self.corr_dim,
                                bp[lp0:lp1],
                                bp[rp0:rp1],
                                window_bp,
                            )

                for rp, (rp0, rp1, X_right) in enumerate(target_loaded):
                    self._accumulate_pgen_self(
                        X_right,
                        annot[rp0:rp1],
                        out[rp0:rp1],
                        self.nsamp,
                        self.corr_dim,
                        bp[rp0:rp1],
                        window_bp,
                    )
                    for lp0, lp1, X_left in target_loaded[:rp]:
                        self._accumulate_pgen_cross(
                            X_left,
                            X_right,
                            annot[lp0:lp1],
                            annot[rp0:rp1],
                            out[lp0:lp1],
                            out[rp0:rp1],
                            self.nsamp,
                            self.corr_dim,
                            bp[lp0:lp1],
                            bp[rp0:rp1],
                            window_bp,
                        )
                if pbar is not None:
                    pbar.update(1)

        if self.verbose:
            self.log._log(
                f"[win][pgen] chr panel_cols={panel_cols}, cache={cache_bytes / 1024**3:.2f} GiB, "
                f"cumulative decoded blocks={reader.blocks_read}."
            )
        return out

    def _compute_chrom_ldscores(self, s: int, e: int, pbar=None) -> np.ndarray:
        if self.genotype_format == "pgen":
            return self._compute_chrom_ldscores_pgen(s, e, pbar=pbar)
        bp = self.bp_all[s:e]
        ann_chr = np.asfortranarray(self.annot[s:e, :], dtype=np.float64)

        result = {}
        error = {}
        done_evt = threading.Event()

        def _worker():
            try:
                result["ld_chr"] = winldcore.compute_windowed_ld_chr(
                    bed_prefix=self.bed_prefix,
                    fam_path=self.fam_path,
                    chr_start=int(s),
                    chr_end=int(e),
                    bp=bp,
                    annot_chr=ann_chr,
                    ld_wind_kb=float(self.ld_wind_kb),
                    chunk_size=int(self.chunk_size),
                    row_sel=(self.row_sel if self.row_sel is not None else None),
                    C=(self.C if self.C is not None else None),
                    R=(self.cov_R if self.cov_R is not None else None),
                    impute_mode=self.impute_method,
                    impute_seed=int(self.impute_seed),
                    panel_cols=int(self.panel_cols),
                    cache_mb=int(self.effective_cache_mb),
                )
            except BaseException as ex:
                error["ex"] = ex
            finally:
                done_evt.set()

        with _set_parallelism(omp_threads=self.num_threads, blas_threads=self.num_threads, decode_threads_cap=self.num_threads):
            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            last_done = 0
            while not done_evt.wait(0.10):
                if pbar is not None:
                    done = int(winldcore.get_progress_done())
                    total = int(pbar.total) if pbar.total is not None else done
                    done = min(done, total)
                    if done > last_done:
                        pbar.update(done - last_done)
                        last_done = done

            t.join()

            if pbar is not None:
                done = int(winldcore.get_progress_done())
                total = int(pbar.total) if pbar.total is not None else done
                done = min(done, total)
                if done > last_done:
                    pbar.update(done - last_done)

        if "ex" in error:
            raise error["ex"]

        if os.environ.get("SUMMIT_TRIM_AFTER_CHR", "0") not in ("", "0", "false", "False", "FALSE"):
            _trim_malloc_best_effort()
        return np.asarray(result["ld_chr"], dtype=np.float64, order="C")

    # ------------------ public entrypoint ------------------

    def _compute_ldscore(self):
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            raise ImportError("tqdm is required for progress display. Install with `pip install tqdm`.")

        chr_arr = self.snplist["CHR"].astype(str).to_numpy()
        blocks: List[Tuple[str, int, int]] = []
        i = 0
        while i < self.nsnps:
            c = chr_arr[i]
            j = i + 1
            while j < self.nsnps and chr_arr[j] == c:
                j += 1
            blocks.append((c, i, j))
            i = j

        ld_all = np.zeros((self.nsnps, self.nbins), dtype=np.float64)

        for chrom, s, e in blocks:
            n_chunks = max(1, (int(e) - int(s) + self.chunk_size - 1) // self.chunk_size)
            pbar = tqdm(
                total=n_chunks,
                desc=f"chr {chrom}",
                unit="chunk",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=True,
            )

            try:
                t0 = time.time()
                ld_chr = self._compute_chrom_ldscores(s, e, pbar=pbar)
                ld_all[s:e, :] = ld_chr
                pbar.set_postfix_str("done")

                if self.verbose:
                    bp0 = int(self.snplist["BP"].iloc[s])
                    bp1 = int(self.snplist["BP"].iloc[e - 1])
                    dt = time.time() - t0
                    self.log._log(f"[win] chr {chrom}: m={e - s}, BP=[{bp0},{bp1}], runtime={dt:.2f}s")
            finally:
                pbar.close()

        self.win_ldscore = ld_all

        out_ld = f"{self.outpath}.win.ldscore.gz"
        if self.nbins > 1:
            self.log._log(f"Saving the windowed (partitioned) LD scores into: {out_ld}")
        else:
            self.log._log(f"Saving the windowed LD scores into: {out_ld}")

        snpdf = self.snplist[["CHR", "SNP", "BP"]].copy()
        scores_df = pd.DataFrame(self.win_ldscore, columns=self.l2cols)
        out_df = pd.concat([snpdf, scores_df], axis=1)
        out_df.to_csv(out_ld, index=False, compression="gzip", sep="\t", float_format="%.6f")

        self._log_basic_stats(scores_df)

        annot64 = np.ascontiguousarray(self.annot, dtype=np.float64)
        M = annot64.sum(axis=0, dtype=np.float64)

        maf = self._compute_maf()
        mask_5 = maf > 0.05
        if mask_5.any():
            M_5 = annot64[mask_5, :].sum(axis=0, dtype=np.float64)
        else:
            M_5 = np.zeros_like(M)

        out_M = f"{self.outpath}.win.M"
        out_M5 = f"{self.outpath}.win.M_5_50"
        with open(out_M, "w") as f:
            f.write("\t".join(f"{x:.6f}" for x in M) + "\n")
        with open(out_M5, "w") as f:
            f.write("\t".join(f"{x:.6f}" for x in M_5) + "\n")

        if self.genotype_format == "pgen" and self._pgen_reader is not None:
            seconds = float(self._pgen_reader.decode_seconds)
            rate = self._pgen_reader.variants_read / seconds if seconds > 0.0 else float("nan")
            self.log._log(
                f"[win][pgen] Decode/standardize totals (including MAF pass): "
                f"blocks={self._pgen_reader.blocks_read}, "
                f"variant-records={self._pgen_reader.variants_read}, "
                f"missing-values={self._pgen_reader.missing_values}, seconds={seconds:.3f}, "
                f"rate={rate:.1f} variant-records/s."
            )

        end_time = utils._get_time()
        runtime = end_time - self.start_time
        self.log._log("Calculation of windowed LD score ended at " + utils._get_timestr(end_time))
        self.log._log(
            "Runtime: " + format(runtime, ".3f") +
            f" s ({runtime//3600} hr {(runtime%3600)//60} m {(runtime%60):.3f} s)"
        )

        try:
            self.log._save_log(self.outpath + ".win.log")
        except Exception:
            pass

    def _log_basic_stats(self, scores_df: pd.DataFrame):
        try:
            desc = scores_df.describe(percentiles=[0.25, 0.5, 0.75]).loc[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ]
            self.log._log("Per-bin LD score summary (count/mean/std/min/25%/50%/75%/max):")
            with pd.option_context(
                "display.width", 140,
                "display.max_columns", None,
                "display.float_format", "{:.6f}".format,
            ):
                self.log._log(desc.to_string() + "\n")

            if scores_df.shape[1] >= 2:
                corr = scores_df.corr(method="pearson")
                self.log._log("Correlation matrix across bins (Pearson):\n")
                with pd.option_context(
                    "display.width", 140,
                    "display.max_columns", None,
                    "display.float_format", "{:.4f}".format,
                ):
                    self.log._log(corr.to_string() + "\n")

            self.log._log("Annotation Column Sums")
            col_sums = self.annot.sum(axis=0, dtype=np.float64)
            for name, val in zip(self.l2cols, col_sums):
                self.log._log(f"{name:<35} {val:.6f}")
            self.log._log("")

            row_sums = np.asarray(self.annot, dtype=np.float64).sum(axis=1)
            rs = pd.Series(row_sums).describe(percentiles=[0.25, 0.5, 0.75])
            rs = rs.loc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            self.log._log("Summary of Annotation Matrix Row Sums")
            with pd.option_context("display.float_format", "{:.4f}".format):
                self.log._log(rs.to_string() + "\n")

        except Exception as e:
            self.log._log(f"[win][warn] Failed to print basic statistics: {e}")
