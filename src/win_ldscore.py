from __future__ import annotations

import os
import sys
import time
import ctypes
from contextlib import nullcontext
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from bed_reader import open_bed

import utils

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

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


def _set_parallelism(blas_threads: Optional[int]):
    """
    Cap BLAS threads for C++ BLAS calls.
    Returns a context manager if threadpoolctl is available; otherwise no-op.
    """
    if blas_threads is None:
        return nullcontext()

    b = max(1, int(blas_threads))
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(b)
    os.environ["OPENBLAS_DYNAMIC"] = "0"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    if threadpool_limits is not None:
        return threadpool_limits(limits=b, user_api="blas")
    return nullcontext()


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
    fam_path: str,
    log,
    sample_idx: Optional[np.ndarray] = None,
    add_intercept: bool = True,
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
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[0, 1], names=["FID", "IID"])
    cov = pd.read_csv(cov_path, sep=r"\s+")

    if ("FID" not in cov.columns) or ("IID" not in cov.columns):
        raise ValueError("Covariate file must contain 'FID' and 'IID' columns.")

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


class WindowedLDScore:
    """
    Windowed LD score computation for the deterministic sliding-window path.

    Heavy computation is delegated to the C++ winldcore module:
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
    ):
        if log is None:
            raise ValueError("WindowedLDScore requires a Logger instance (log=...).")
        if winldcore is None:
            raise ImportError(
                "winldcore could not be imported. Build the C++ extension before running the windowed LD path."
            ) from _WINLDCORE_IMPORT_ERROR

        self.log = log
        self.verbose = bool(verbose)

        prefix = _canonical_bfile_prefix(bed_path)
        if "@" in prefix:
            raise ValueError(
                "WindowedLDScore expects a single genome-wide PLINK prefix (no '@'). "
                "Use genome-wide PLINK files for SUMMIT windowed LD scores."
            )

        self.bed_prefix = os.path.abspath(prefix)
        self.fam_path = self.bed_prefix + ".fam"
        self.bim_path = self.bed_prefix + ".bim"
        self.bed_file = self.bed_prefix + ".bed"

        if not os.path.exists(self.bed_file):
            raise FileNotFoundError(f"Missing .bed: {self.bed_file}")
        if not os.path.exists(self.bim_path):
            raise FileNotFoundError(f"Missing .bim: {self.bim_path}")
        if not os.path.exists(self.fam_path):
            raise FileNotFoundError(f"Missing .fam: {self.fam_path}")

        self.G = open_bed(self.bed_file)
        self.nsamp0, self.nsnps = self.G.shape

        self.ld_wind_kb = float(ld_wind_kb)
        if self.ld_wind_kb <= 0:
            raise ValueError("--ld-wind-kb must be positive.")

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

        self.impute_method = str(impute_method).strip().lower()
        if self.impute_method not in ("mean", "hwe"):
            raise ValueError("impute_method must be 'mean' or 'hwe'.")

        if seed is None:
            self.root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            if self.verbose:
                self.log._log(f"[seed] No seed provided; using generated root seed {self.root_seed}")
        else:
            self.root_seed = int(seed)
        self.impute_seed = int((np.uint64(self.root_seed) ^ np.uint64(0xA24BAED4963EE407)) & np.uint64(0xFFFFFFFFFFFFFFFF))

        self.start_time = utils._get_time()
        self.log._log("Windowed LD score calculation started at: " + utils._get_timestr(self.start_time))
        self.log._log(f"[win][backend] C++ core enabled (impute={self.impute_method}, impute_seed={self.impute_seed})")

        rng = np.random.default_rng(self.root_seed)
        self.row_sel = _parse_rand_samp(rand_samp, self.nsamp0, rng)
        if self.row_sel is not None:
            k = int(len(self.row_sel))
            self.log._log(f"Randomly subsampling individuals: {k}/{self.nsamp0} ({k/self.nsamp0:.1%})")

        self._read_bim(self.bim_path)
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

        if self.nsamp <= 2:
            raise ValueError(f"[win] Too few samples (n={self.nsamp}) for windowed LD score computation.")

        try:
            winldcore.set_verbose(bool(self.verbose))
            winldcore.set_num_threads(int(self.num_threads))
        except Exception as e:
            if self.verbose:
                self.log._log(f"[win][warn] Failed to set C++ verbosity / threads: {e}")

        if self.verbose:
            self.log._log(
                f"[win] ld_wind_kb={self.ld_wind_kb}, chunk_size={self.chunk_size}, "
                f"panel_cols={self.panel_cols or 'auto'}, cache_mb={self.cache_mb}, threads={self.num_threads}"
            )

        self.win_ldscore: Optional[np.ndarray] = None

    # ------------------ BIM / annotation ------------------

    def _read_bim(self, bim_path: str):
        self.log._log(f"[win] Reading BIM: {bim_path}")
        snplist = pd.read_csv(bim_path, header=None, sep=r"\s+")
        snplist.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        if len(snplist) != self.nsnps:
            raise ValueError(f"[win] .bed SNPs ({self.nsnps}) != .bim rows ({len(snplist)})")
        self.snplist = snplist

    def _read_annot(self, annot_path: Optional[str]):
        if annot_path is None:
            self.l2cols = ["L2_0"]
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.is_continuous = False
            self.log._log("[win] No annotation: using single-bin (all SNPs).")
            return

        parsed_ldsc = False
        try:
            df = pd.read_csv(
                annot_path,
                sep=r"\s+",
                compression="infer",
                dtype={"CHR": str, "BP": np.int64, "SNP": str, "CM": float},
            )
            base_cols = {"CHR", "BP", "SNP", "CM"}
            if base_cols.issubset(set(df.columns)) and "SNP" in df.columns:
                annot_cols = [c for c in df.columns if c not in base_cols]
                if len(annot_cols) == 0:
                    raise ValueError("No annotation columns found after [CHR,BP,SNP,CM].")

                bim_snps = self.snplist["SNP"].astype(str).tolist()
                ann_snps = df["SNP"].astype(str).tolist()

                if ann_snps == bim_snps:
                    ann_mat = df[annot_cols].to_numpy(dtype=np.float64, copy=False)
                else:
                    ann_set = set(ann_snps)
                    bim_set = set(bim_snps)
                    missing_in_annot = len(bim_set - ann_set)
                    if missing_in_annot > 0:
                        raise ValueError(
                            f"Annotation SNP set is missing {missing_in_annot} BIM SNP(s); "
                            "regenerate the annotation to match the .bim."
                        )
                    ann_mat = df.set_index("SNP").loc[bim_snps, annot_cols].to_numpy(dtype=np.float64, copy=False)

                np.nan_to_num(ann_mat, copy=False)
                if (ann_mat < 0).any():
                    self.log._log("[win][warn] Negative annotation values found; clipping to 0.")
                    ann_mat[ann_mat < 0] = 0.0

                uniq = np.unique(ann_mat)
                is_binary = np.all(np.isin(uniq, [0.0, 1.0]))
                self.is_continuous = (not is_binary)

                self.annot = ann_mat
                self.nbins = int(self.annot.shape[1])
                self.l2cols = annot_cols
                parsed_ldsc = True
                self.log._log(f"[win] Read LDSC-style annotation: shape={self.annot.shape}")
        except Exception:
            parsed_ldsc = False

        if not parsed_ldsc:
            self.l2cols, arr = utils._read_with_optional_header(annot_path)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = arr.astype(np.float64, copy=False)
            np.nan_to_num(arr, copy=False)
            if (arr < 0).any():
                self.log._log("[win][warn] Negative annotation values found; clipping to 0.")
                arr[arr < 0] = 0.0

            uniq = np.unique(arr)
            is_binary = np.all(np.isin(uniq, [0.0, 1.0]))
            self.is_continuous = (not is_binary)

            self.annot = arr
            if self.l2cols is None:
                self.l2cols = [f"L2_{i}" for i in range(self.annot.shape[1])]
            self.nbins = int(self.annot.shape[1])
            self.log._log(f"[win] Read thin annotation matrix: shape={self.annot.shape}")

        if self.annot.shape[0] != self.nsnps:
            raise ValueError(
                f"[win] #SNPs in annotation ({self.annot.shape[0]}) != #SNPs in genotype ({self.nsnps})"
            )

        self.log._log(
            f"[win] nsamp0={self.nsamp0}, nsnps={self.nsnps}, nbins={self.nbins}, continuous={bool(self.is_continuous)}"
        )

    # ------------------ C++ core wrappers ------------------

    def _compute_maf(self) -> np.ndarray:
        self.log._log("[win] Computing MAF for .win.M_5_50 via C++ core.")
        step = int(max(1024, min(self.nsnps, self.chunk_size)))
        with _set_parallelism(blas_threads=1):
            maf = winldcore.compute_maf_bed(
                bed_prefix=self.bed_prefix,
                fam_path=self.fam_path,
                nsnps=int(self.nsnps),
                step_size=step,
                row_sel=(self.row_sel if self.row_sel is not None else None),
            )
        return np.asarray(maf, dtype=np.float64, order="C")

    def _compute_chrom_ldscores(self, s: int, e: int) -> np.ndarray:
        bp = self.snplist["BP"].to_numpy(dtype=np.int64, copy=False)[s:e]
        ann_chr = np.asfortranarray(self.annot[s:e, :], dtype=np.float64)

        with _set_parallelism(blas_threads=self.num_threads):
            ld_chr = winldcore.compute_windowed_ld_chr(
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
                cache_mb=int(self.cache_mb),
            )
        _trim_malloc_best_effort()
        return np.asarray(ld_chr, dtype=np.float64, order="C")

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

        pbar = tqdm(
            blocks,
            total=len(blocks),
            desc="WIN-LD progress",
            unit="chr",
            file=sys.stderr,
            dynamic_ncols=True,
        )

        for chrom, s, e in pbar:
            t0 = time.time()
            ld_chr = self._compute_chrom_ldscores(s, e)
            ld_all[s:e, :] = ld_chr

            if self.verbose:
                bp0 = int(self.snplist["BP"].iloc[s])
                bp1 = int(self.snplist["BP"].iloc[e - 1])
                dt = time.time() - t0
                self.log._log(f"[win] chr {chrom}: m={e - s}, BP=[{bp0},{bp1}], runtime={dt:.2f}s")
            pbar.set_postfix_str(f"chr {chrom}")

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
