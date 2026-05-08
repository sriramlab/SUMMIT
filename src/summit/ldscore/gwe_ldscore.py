from __future__ import annotations

import gc
import math
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from bed_reader import open_bed
from tqdm import tqdm

from .. import utils
from .gw_ldscore import apply_env



def _canonical_bfile_prefix(x: str) -> str:
    s = str(x)
    for ext in (".bed", ".bim", ".fam"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s



def _round_up_to(x: int, gran: int) -> int:
    return int(((x + gran - 1) // gran) * gran)



def _build_balanced_vtiles(V: int, vmax: int, gran: int = 64, max_tiles: int = 8) -> list[tuple[int, int]]:
    if V <= 0:
        return []
    vmax = max(gran, (int(vmax) // gran) * gran)
    if vmax <= 0:
        vmax = gran
    if vmax >= V:
        return [(0, V)]

    tiles = int(math.ceil(V / vmax))
    tiles = min(max_tiles, max(2, tiles))
    q, r = divmod(V, tiles)
    sizes: list[int] = []
    for t in range(tiles):
        sz = q + (1 if t < r else 0)
        sz = min(vmax, _round_up_to(sz, gran))
        sizes.append(sz)

    total = sum(sizes)
    over = total - V
    t = len(sizes) - 1
    while over > 0 and t >= 0:
        bleed = min(over, max(0, sizes[t] - max(gran, q)))
        bleed = (bleed // gran) * gran
        if bleed > 0:
            sizes[t] -= bleed
            over -= bleed
        t -= 1

    out: list[tuple[int, int]] = []
    v0 = 0
    for sz in sizes:
        if sz <= 0:
            continue
        if v0 + sz > V:
            sz = V - v0
        if sz <= 0:
            break
        out.append((v0, sz))
        v0 += sz
    if v0 < V:
        out.append((v0, V - v0))
    return out



def _mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF



def _make_seed(root: int, block: int, v0: int) -> int:
    s = 0x1234ABCD
    s ^= _mix64(int(root))
    s ^= _mix64(int(block) + 0x9E37)
    s ^= _mix64(int(v0) + 0x85EB)
    return int(s & 0xFFFFFFFFFFFFFFFF)



def _orthonormalize_columns(X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if X.shape[1] == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64, order="F")
    Q, R = np.linalg.qr(np.asarray(X, dtype=np.float64, order="F"), mode="reduced")
    if R.size == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64, order="F")
    keep = np.abs(np.diag(R)) > tol
    if keep.sum() == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64, order="F")
    return np.asfortranarray(Q[:, keep])



def read_env_and_cov(
    env_filename: str,
    fam_filename: str,
    cov_filename: str | None = None,
    std: bool = True,
    cov_impute_method: str = "ignore",
    logger=None,
    verbose: bool = False,
    sample_idx=None,
    ddof: int = 1,
):
    fam = pd.read_csv(fam_filename, sep=r"\s+", header=None, usecols=[0, 1], names=["FID", "IID"])
    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx, dtype=int)
        fam = fam.iloc[sample_idx].reset_index(drop=True)

    env = pd.read_csv(env_filename, sep=r"\s+")
    if "FID" not in env.columns or "IID" not in env.columns:
        raise ValueError("Environment file must contain FID and IID columns.")
    if env.duplicated(subset=["FID", "IID"]).any():
        raise ValueError("Environment file contains duplicate FID/IID rows.")
    env_cols = [c for c in env.columns if c not in ("FID", "IID")]
    if len(env_cols) != 1:
        raise ValueError(
            "Environment file must contain exactly one environment column in addition to FID and IID. "
            f"Found {len(env_cols)} column(s): {env_cols}."
        )
    env_name = env_cols[0]

    merged = fam.merge(env[["FID", "IID", env_name]], on=["FID", "IID"], how="left", indicator=True)
    n_missing_env = int((merged["_merge"] != "both").sum())
    if n_missing_env:
        raise ValueError(f"{n_missing_env} .fam samples not found in environment file (FID/IID mismatch).")
    merged.drop(columns=["_merge"], inplace=True)

    cov_cols: list[str] = []
    if cov_filename is not None:
        cov = pd.read_csv(cov_filename, sep=r"\s+")
        if "FID" not in cov.columns or "IID" not in cov.columns:
            raise ValueError("Covariate file must contain FID and IID columns.")
        if cov.duplicated(subset=["FID", "IID"]).any():
            raise ValueError("Covariate file contains duplicate FID/IID rows.")
        merged = merged.merge(cov, on=["FID", "IID"], how="left", indicator=True)
        n_missing_cov = int((merged["_merge"] != "both").sum())
        if n_missing_cov:
            raise ValueError(f"{n_missing_cov} .fam samples not found in covariate file (FID/IID mismatch).")
        merged.drop(columns=["_merge"], inplace=True)
        cov_cols = [c for c in merged.columns if c not in ("FID", "IID", env_name)]

    merged[env_name] = pd.to_numeric(merged[env_name], errors="coerce")
    for c in cov_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if cov_impute_method == "ignore":
        keep_mask = merged[env_name].notna()
        if cov_cols:
            keep_mask &= ~merged[cov_cols].isna().any(axis=1)
    else:
        if cov_cols:
            merged[cov_cols] = merged[cov_cols].apply(lambda s: s.fillna(s.mean()), axis=0)
        keep_mask = merged[env_name].notna()

    dropped = int((~keep_mask).sum())
    if logger is not None:
        logger._log(f"[env] Dropping {dropped} samples due to missing environment/covariates.")

    merged = merged.loc[keep_mask].reset_index(drop=True)
    if merged.shape[0] == 0:
        raise ValueError("After filtering, no samples remain for the environment-specific LD-score calculation.")

    env_vec = merged[env_name].to_numpy(dtype=np.float64)
    env_vec = env_vec - float(env_vec.mean())
    env_std = float(env_vec.std(ddof=ddof))
    if not np.isfinite(env_std) or env_std <= 0.0:
        raise ValueError(f"Environment '{env_name}' has zero or invalid variance after filtering.")
    if std:
        env_vec = env_vec / env_std

    cov_base = np.empty((merged.shape[0], 0), dtype=np.float64)
    kept_cov_cols: list[str] = []
    if cov_cols:
        cov_df = merged[cov_cols].copy()
        zvc = cov_df.std(ddof=0) == 0
        if zvc.any():
            drop_cols = zvc.index[zvc].tolist()
            if logger is not None:
                logger._log(
                    f"[env] Dropping {len(drop_cols)} constant covariates: "
                    f"{drop_cols[:10]}{'...' if len(drop_cols) > 10 else ''}"
                )
            cov_df.drop(columns=drop_cols, inplace=True)
        if not cov_df.empty and std:
            cov_df = (cov_df - cov_df.mean()) / cov_df.std(ddof=ddof)
            bad_cols = [c for c in cov_df.columns if cov_df[c].isna().all()]
            if bad_cols:
                if logger is not None:
                    logger._log(f"[env] Dropping malformed covariate columns after standardization: {bad_cols}")
                cov_df.drop(columns=bad_cols, inplace=True)
        if not cov_df.empty:
            kept_cov_cols = list(cov_df.columns)
            cov_base = cov_df.to_numpy(dtype=np.float64, copy=False)

    # Interaction / GWIS covariate space: user covariates + environment main effect.
    # We use the same space for the additive X side in the XW cross-score so that the
    # resulting summary objects match the score-scale derivation.
    design_base = np.column_stack([cov_base, env_vec.reshape(-1, 1)])
    C = _orthonormalize_columns(design_base)
    R = np.asfortranarray(C.T)

    km = np.flatnonzero(keep_mask.to_numpy())
    if sample_idx is not None:
        keep_idx_global = np.asarray(sample_idx, dtype=int)[km]
    else:
        keep_idx_global = km

    if logger is not None:
        logger._log(
            f"[env] Read {env_filename}: kept {C.shape[0]} samples; environment='{env_name}'; "
            f"effective covariate rank={C.shape[1]} ({len(kept_cov_cols)} user covariate(s) + environment main effect)."
        )

    return (
        np.asarray(env_vec, dtype=np.float64),
        str(env_name),
        np.asfortranarray(C),
        np.asfortranarray(R),
        np.asarray(keep_idx_global, dtype=int),
        kept_cov_cols,
    )


class GenomewideEnvLDScore:
    r"""
    Estimate both genome-wide interaction-interaction and additive-interaction
    LD scores using randomized sketches.

    For each annotation bin k, the module estimates
        ell^{WW}_{jk} = sum_{j' in S_k} (r^{WW}_{jj'})^2,
        ell^{XW}_{jk} = sum_{j' in S_k} (r^{XW}_{jj'})^2,
    where both X and W are standardized after projection onto the same
    covariate space that includes the environment main effect.

    Output files:
        {out}.gee.ldscore.gz   -> WW / interaction-interaction LD scores
        {out}.gxe.ldscore.gz   -> XW / additive-interaction cross-LD scores
    """

    def __init__(
        self,
        bed_path,
        env_path,
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
        dtype="float32",
        num_threads: int | None = None,
        eps_var: float = 1e-10,
        rand_samp=None,
        ddof=1,
        target_xz_mem=16.0,
        target_mem=None,
        device="cpu",
        impute_method: str = "mean",
    ):
        self.eps_var = float(eps_var)
        prefix = _canonical_bfile_prefix(bed_path)
        self.bed_prefix = os.path.abspath(prefix)
        self.fam_path = self.bed_prefix + ".fam"
        self.bim_path = self.bed_prefix + ".bim"
        self.env_path = str(env_path)
        self.covar_path = covar_path

        self.G = open_bed(self.bed_prefix + ".bed")
        self.nsamp_total, self.nsnps = self.G.shape
        self.nvecs = int(num_vecs)
        self.step_size = int(step_size)
        self.log = log
        self.verbose = bool(verbose)
        self.rand_dist = str(rand_dist).strip().lower()
        self.ddof = int(ddof)
        self.target_mem = target_mem
        self.target_xz_mem = target_xz_mem if target_mem is None else target_mem
        self.device = str(device).strip().lower()
        self.impute_method = str(impute_method).strip().lower()
        if self.impute_method != "mean":
            raise ValueError("The current Python GxE LD-score implementation supports only impute_method='mean'.")
        if self.device != "cpu":
            self.log._log(f"[gxe] device='{device}' requested, but this Python implementation is CPU-only. Falling back to CPU.")
            self.device = "cpu"

        if seed is None:
            self.root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            self.log._log(f"[seed] No seed provided; using generated root seed {self.root_seed}")
        else:
            self.root_seed = int(seed)
        self.rng = np.random.default_rng(self.root_seed)

        if low_level is not None:
            try:
                actual_threads = apply_env(low_level)
            except Exception as e:
                self.log._log(f"[threads] apply_env failed in GxE LD-score setup (non-fatal): {e}")
                actual_threads = None
        else:
            actual_threads = None
        if num_threads is not None and int(num_threads) > 0:
            self.num_threads = int(num_threads)
        elif actual_threads is not None:
            self.num_threads = int(actual_threads)
        else:
            self.num_threads = max(1, os.cpu_count() or 1)

        self.dtype = np.float32 if dtype in (np.float32, "float32", "f4") else np.float64
        self.start_time = utils._get_time()
        self.log._log("Genome-wide GxE LD-score calculation started at: " + utils._get_timestr(self.start_time))

        base_idx = np.arange(self.nsamp_total, dtype=int)
        sel_idx = None
        if rand_samp is not None:
            if isinstance(rand_samp, (float, np.floating)):
                if not (0.0 < rand_samp <= 1.0):
                    raise ValueError("--rand-samp float must be in (0,1].")
                k = int(np.floor(float(rand_samp) * self.nsamp_total))
                k = max(1, min(k, self.nsamp_total))
            else:
                k = int(rand_samp)
                if not (100 <= k <= self.nsamp_total):
                    raise ValueError("--rand-samp int must be in [100, N].")
            sel_idx = np.sort(self.rng.choice(base_idx, size=k, replace=False))
            self.log._log(f"Randomly subsampling individuals: {k}/{self.nsamp_total} ({k / self.nsamp_total:.1%})")

        self._read_bim(self.bim_path)
        self._read_annot(annot_path)

        env_vec, env_name, C_int, cov_R_int, keep_idx_global, cov_cols = read_env_and_cov(
            env_filename=self.env_path,
            fam_filename=self.fam_path,
            cov_filename=self.covar_path,
            std=True,
            cov_impute_method="ignore",
            logger=self.log,
            verbose=self.verbose,
            sample_idx=sel_idx if sel_idx is not None else None,
            ddof=self.ddof,
        )
        self.row_sel = np.asarray(keep_idx_global, dtype=int)
        self.env = np.asarray(env_vec, dtype=np.float64)
        self.env_name = env_name
        self.C_int = np.asarray(C_int, dtype=np.float64, order="F")
        self.cov_R_int = np.asarray(cov_R_int, dtype=np.float64, order="F")
        self.cov_cols = list(cov_cols)

        self.nsamp = int(self.row_sel.shape[0])
        self.p_eff = int(self.C_int.shape[1])
        self.N_eff = self.nsamp - self.p_eff
        self.df_corr = self.N_eff - 1
        if self.df_corr <= 0:
            raise ValueError(
                f"Residual correlation degrees of freedom are non-positive: nsamp={self.nsamp}, "
                f"p_eff={self.p_eff}, df={self.df_corr}."
            )

        self.outpath = out_path
        self.inv_sqrt_resvar_x_all: np.ndarray | None = None
        self.inv_sqrt_resvar_w_all: np.ndarray | None = None
        self.log._log(
            f"[env] Using environment '{self.env_name}' with {self.nsamp} samples; "
            f"effective covariate rank={self.p_eff}; correlation df={self.df_corr}."
        )
        if self.covar_path is None:
            self.log._log("[env] No additional user covariates supplied; projecting on the environment main effect only.")
        else:
            self.log._log(
                f"[env] Included {len(self.cov_cols)} user covariate column(s) together with the environment main effect in the projection."
            )

    def _read_bim(self, bim_path):
        if bim_path is None:
            self.log._log("No .bim file is provided; all (anonymous) SNPs will be used")
            self.snplist = None
            return
        self.log._log(f"Reading {bim_path} for SNPs")
        self.snplist = pd.read_csv(bim_path, header=None, sep=r"\s+")
        self.snplist.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        if len(self.snplist) != self.nsnps:
            self.log._log(f"!!! The number of SNPs in the .bed file ({self.nsnps}) does not match the .bim file ({len(self.snplist)}) !!!")
            sys.exit(1)

    def _read_annot(self, annot_path):
        if annot_path is None:
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.l2cols = ["L2_0"]
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
            self.is_continuous = False
            self.annot = np.ascontiguousarray(self.annot.astype(self.dtype, copy=False))
            self.log._log("Calculating genome-wide (non-partitioned) GxE LD scores")
            self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
            return

        parsed_ldsc = False
        try:
            df = pd.read_csv(annot_path, sep=r"\s+", compression="infer", dtype={"CHR": str, "BP": np.int64, "SNP": str, "CM": float})
            base_cols = {"CHR", "BP", "SNP", "CM"}
            if base_cols.issubset(set(df.columns)) and "SNP" in df.columns:
                annot_cols = [c for c in df.columns if c not in base_cols]
                if len(annot_cols) == 0:
                    raise ValueError("No annotation columns found after [CHR,BP,SNP,CM].")
                bim_snps = self.snplist.iloc[:, 1].astype(str).tolist()
                ann_snps = df["SNP"].astype(str).tolist()
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
                    ann_mat = df.set_index("SNP").loc[bim_snps, annot_cols].to_numpy(dtype=np.float64, copy=False)
                np.nan_to_num(ann_mat, copy=False)
                if (ann_mat < 0).any():
                    self.log._log("[warn] Negative annotation values found; clipping to 0.")
                    ann_mat[ann_mat < 0] = 0.0
                uniq = np.unique(ann_mat)
                self.is_continuous = not np.all(np.isin(uniq, [0.0, 1.0]))
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
            self.is_continuous = not np.all(np.isin(uniq, [0.0, 1.0]))
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
        self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
        self.log._log(f"Nbins: {self.nbins}")

    def _make_compute_blocks(self):
        return [(s, min(self.nsnps, s + self.step_size)) for s in range(0, self.nsnps, self.step_size)]

    def _read_genotype_block(self, blk_start: int, blk_end: int) -> np.ndarray:
        indexer = np.s_[self.row_sel, blk_start:blk_end]
        try:
            G = self.G.read(index=indexer, dtype=np.float64)
        except TypeError:
            try:
                G = self.G.read(index=indexer)
            except TypeError:
                G = self.G.read(indexer)
            G = np.asarray(G, dtype=np.float64)

        G = np.asarray(G, dtype=np.float64)
        if G.shape == (blk_end - blk_start, self.nsamp):
            G = G.T
        if G.shape != (self.nsamp, blk_end - blk_start):
            raise RuntimeError(
                f"Unexpected bed-reader block shape {G.shape} for block [{blk_start}:{blk_end}); expected ({self.nsamp}, {blk_end - blk_start})."
            )

        col_means = np.nanmean(G, axis=0)
        bad_means = ~np.isfinite(col_means)
        if np.any(bad_means):
            col_means[bad_means] = 0.0
        mask = np.isnan(G)
        if mask.any():
            rr, cc = np.where(mask)
            G[rr, cc] = col_means[cc]
        G -= col_means
        col_std = G.std(axis=0, ddof=self.ddof)
        good = np.isfinite(col_std) & (col_std > self.eps_var)
        if np.any(good):
            G[:, good] /= col_std[good]
        if np.any(~good):
            G[:, ~good] = 0.0
        return np.asarray(G, dtype=np.float64, order="F")

    def _project_and_center_inplace(self, M: np.ndarray) -> np.ndarray:
        if self.p_eff > 0:
            M -= self.C_int @ (self.cov_R_int @ M)
        M -= M.mean(axis=0, keepdims=True)
        return M

    def _prepare_additive_block(self, blk_start: int, blk_end: int, G: np.ndarray | None = None, apply_scale: bool = True, out_dtype=None) -> np.ndarray:
        if G is None:
            G = self._read_genotype_block(blk_start, blk_end)
        X = np.array(G, copy=True, dtype=np.float64, order="F")
        self._project_and_center_inplace(X)
        if apply_scale:
            if self.inv_sqrt_resvar_x_all is None:
                raise RuntimeError("Additive residual variances have not been precomputed.")
            X *= self.inv_sqrt_resvar_x_all[blk_start:blk_end].reshape(1, -1)
        return np.asarray(X, dtype=(out_dtype or self.dtype), order="F")

    def _prepare_interaction_block(self, blk_start: int, blk_end: int, G: np.ndarray | None = None, apply_scale: bool = True, out_dtype=None) -> np.ndarray:
        if G is None:
            G = self._read_genotype_block(blk_start, blk_end)
        W = np.asarray(G * self.env[:, None], dtype=np.float64, order="F")
        self._project_and_center_inplace(W)
        if apply_scale:
            if self.inv_sqrt_resvar_w_all is None:
                raise RuntimeError("Interaction residual variances have not been precomputed.")
            W *= self.inv_sqrt_resvar_w_all[blk_start:blk_end].reshape(1, -1)
        return np.asarray(W, dtype=(out_dtype or self.dtype), order="F")

    def _precompute_residual_variances(self) -> tuple[np.ndarray, np.ndarray]:
        self.log._log("[gxe] Precomputing additive and interaction residual variances in a single pass.")
        inv_x = np.zeros(self.nsnps, dtype=np.float64)
        inv_w = np.zeros(self.nsnps, dtype=np.float64)
        bad_x = 0
        bad_w = 0
        blocks = self._make_compute_blocks()
        for s, e in tqdm(blocks, desc="GxE var", unit="block", disable=(not self.verbose)):
            G = self._read_genotype_block(s, e)

            X = self._prepare_additive_block(s, e, G=G, apply_scale=False, out_dtype=np.float64)
            ssx = np.sum(X * X, axis=0, dtype=np.float64)
            varx = ssx / float(self.df_corr)
            good_x = np.isfinite(varx) & (varx > self.eps_var)
            inv_x[s:e][good_x] = 1.0 / np.sqrt(varx[good_x])
            inv_x[s:e][~good_x] = 0.0
            bad_x += int((~good_x).sum())

            W = self._prepare_interaction_block(s, e, G=G, apply_scale=False, out_dtype=np.float64)
            ssw = np.sum(W * W, axis=0, dtype=np.float64)
            varw = ssw / float(self.df_corr)
            good_w = np.isfinite(varw) & (varw > self.eps_var)
            inv_w[s:e][good_w] = 1.0 / np.sqrt(varw[good_w])
            inv_w[s:e][~good_w] = 0.0
            bad_w += int((~good_w).sum())

            del G, X, W, ssx, ssw, varx, varw, good_x, good_w

        if bad_x > 0:
            self.log._log(f"[gxe] {bad_x} additive column(s) had zero or invalid residual variance and were set to zero.")
        if bad_w > 0:
            self.log._log(f"[gxe] {bad_w} interaction column(s) had zero or invalid residual variance and were set to zero.")
        return np.asarray(inv_x, dtype=np.float64), np.asarray(inv_w, dtype=np.float64)

    def _generate_random_block(self, L: int, v_count: int, blk_start: int, v_start: int) -> np.ndarray:
        rng = np.random.default_rng(_make_seed(self.root_seed, blk_start, v_start))
        if self.rand_dist == "rademacher":
            Z = rng.integers(0, 2, size=(L, v_count), dtype=np.int8).astype(np.float64, copy=False)
            Z = 2.0 * Z - 1.0
        elif self.rand_dist in ("gaussian", "normal"):
            Z = rng.standard_normal(size=(L, v_count)).astype(np.float64, copy=False)
        elif self.rand_dist == "spherical":
            Z = rng.standard_normal(size=(L, v_count)).astype(np.float64, copy=False)
            norms = np.linalg.norm(Z, axis=0)
            good = norms > 0.0
            if np.any(good):
                Z[:, good] *= math.sqrt(float(L)) / norms[good]
            if np.any(~good):
                Z[:, ~good] = 0.0
        else:
            raise ValueError(f"Unsupported rand_dist: {self.rand_dist}")
        return np.asarray(Z, dtype=self.dtype, order="F")

    def _auto_vtiles(self) -> list[tuple[int, int]]:
        target_gib = float(self.target_xz_mem)
        itemsize = np.dtype(self.dtype).itemsize
        denom = max(1, int(self.nsamp) * int(self.nbins) * itemsize)
        vmax = int((target_gib * (1024 ** 3)) // denom)
        vmax = max(1, min(self.nvecs, vmax))
        tiles = _build_balanced_vtiles(self.nvecs, vmax=vmax, gran=64, max_tiles=8)
        if not tiles:
            tiles = [(0, self.nvecs)]
        vdesc = ",".join(str(v) for _, v in tiles)
        approx = (self.nsamp * self.nbins * max(v for _, v in tiles) * itemsize) / (1024 ** 3)
        self.log._log(f"[gxe:auto_vchunk] dtype={self.dtype} target≈{target_gib:.1f} GiB, v_tiles=[{vdesc}] -> max sketch≈{approx:.2f} GiB")
        return tiles

    def _accumulate_sketch_block(self, U_chunk: np.ndarray, W: np.ndarray, Z: np.ndarray, annot_blk: np.ndarray) -> None:
        if self.nbins == 1:
            U_chunk[:, : Z.shape[1]] += W @ Z
            return
        sqrt_annot = np.sqrt(np.maximum(annot_blk, 0))
        for k in range(self.nbins):
            wk = sqrt_annot[:, k]
            if not np.any(wk):
                continue
            seg = slice(k * Z.shape[1], (k + 1) * Z.shape[1])
            U_chunk[:, seg] += (W * wk.reshape(1, -1)) @ Z

    def _accumulate_left_scores(self, Work: np.ndarray, meansq_accum: np.ndarray, blk_start: int, blk_end: int, Vt: int) -> None:
        scale = 1.0 / float(self.df_corr ** 2)
        for k in range(self.nbins):
            seg = Work[:, k * Vt:(k + 1) * Vt]
            meansq_accum[blk_start:blk_end, k] += np.sum(seg * seg, axis=1, dtype=np.float64) * scale

    def _save_score_file(self, path: str, score: np.ndarray) -> None:
        snpcols = ["CHR", "SNP", "BP"]
        if self.snplist is None:
            snpdf = pd.DataFrame(np.nan * np.ones((self.nsnps, 3)), columns=snpcols)
        else:
            snpdf = self.snplist[["CHR", "SNP", "BP"]].copy()
            snpdf.columns = snpcols
        scores_df = pd.DataFrame(score, columns=self.l2cols)
        out_df = pd.concat([snpdf, scores_df], axis=1)
        out_df.to_csv(path, index=False, compression="gzip", sep="\t", float_format="%.6f")

    def _log_score_summary(self, label: str, score: np.ndarray) -> None:
        try:
            scores_df = pd.DataFrame(score, columns=self.l2cols)
            desc = scores_df.describe(percentiles=[0.25, 0.5, 0.75]).loc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            self.log._log(f"Per-bin {label} summary (count/mean/std/min/25%/50%/75%/max):")
            with pd.option_context("display.width", 140, "display.max_columns", None, "display.float_format", "{:.6f}".format):
                self.log._log(desc.to_string() + "\n")

            corr = scores_df.corr(method="pearson")
            self.log._log(f"{label} correlation matrix across bins (Pearson):")
            with pd.option_context("display.width", 140, "display.max_columns", None, "display.float_format", "{:.4f}".format):
                self.log._log("\n" + corr.to_string())
        except Exception as e:
            self.log._log(f"[warn] Failed to compute summary stats / correlation for {label}: {e}")

    def _compute_ldscore(self):
        self.log._log(f"num_vecs: {self.nvecs}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        self.log._log("[backend] Python / NumPy only (non-Mailman path).")
        self.log._log(
            f"[target] Estimating additive-interaction cross-LD (XW -> .gxe) and interaction-interaction LD (WW -> .gee) "
            f"for environment '{self.env_name}'."
        )

        self.inv_sqrt_resvar_x_all, self.inv_sqrt_resvar_w_all = self._precompute_residual_variances()
        blocks = self._make_compute_blocks()
        vtiles = self._auto_vtiles()

        meansq_xw_accum = np.zeros((self.nsnps, self.nbins), dtype=np.float64)
        meansq_ww_accum = np.zeros((self.nsnps, self.nbins), dtype=np.float64)

        total_units = max(1, len(vtiles) * 2 * len(blocks))
        bar = tqdm(total=total_units, desc="GxE-LD progress", unit="task", smoothing=0.2, disable=(not self.verbose))
        try:
            for v0, Vt in vtiles:
                U_chunk = np.zeros((self.nsamp, self.nbins * Vt), dtype=self.dtype, order="F")

                # Phase 1: build the interaction sketch U = W_k Z_k once.
                for s, e in blocks:
                    G = self._read_genotype_block(s, e)
                    W = self._prepare_interaction_block(s, e, G=G, apply_scale=True, out_dtype=self.dtype)
                    Z = self._generate_random_block(L=(e - s), v_count=Vt, blk_start=s, v_start=v0)
                    annot_blk = np.asarray(self.annot[s:e], dtype=self.dtype)
                    self._accumulate_sketch_block(U_chunk, W, Z, annot_blk)
                    del G, W, Z, annot_blk
                    bar.update(1)

                # Phase 2: use the same U for both left designs X and W.
                for s, e in blocks:
                    G = self._read_genotype_block(s, e)

                    X = self._prepare_additive_block(s, e, G=G, apply_scale=True, out_dtype=self.dtype)
                    WorkX = np.asarray(X.T @ U_chunk, dtype=np.float64)
                    self._accumulate_left_scores(WorkX, meansq_xw_accum, s, e, Vt)
                    del X, WorkX

                    W = self._prepare_interaction_block(s, e, G=G, apply_scale=True, out_dtype=self.dtype)
                    WorkW = np.asarray(W.T @ U_chunk, dtype=np.float64)
                    self._accumulate_left_scores(WorkW, meansq_ww_accum, s, e, Vt)
                    del G, W, WorkW

                    bar.update(1)

                del U_chunk
                gc.collect()
        finally:
            try:
                bar.close()
            except Exception:
                pass

        meansq_xw = meansq_xw_accum / float(self.nvecs)
        meansq_ww = meansq_ww_accum / float(self.nvecs)

        # Match the current GW-LD implementation style: subtract the correlation null M_k / N_eff.
        null_corr = self.nsnps_bin.reshape(1, -1) / float(self.N_eff)
        self.log._log(f"Applying correlation null subtraction M_k / N_eff with N_eff={self.N_eff} to both XW and WW sketches.")
        meansq_xw -= null_corr
        meansq_ww -= null_corr

        self.gxe_ldscore = np.asarray(meansq_xw, dtype=np.float64)
        self.gee_ldscore = np.asarray(meansq_ww, dtype=np.float64)

        gxe_path = f"{self.outpath}.gxe.ldscore.gz"
        gee_path = f"{self.outpath}.gee.ldscore.gz"
        self.log._log(f"Saving additive-interaction cross-LD scores (XW) into: {gxe_path}")
        self._save_score_file(gxe_path, self.gxe_ldscore)
        self.log._log(f"Saving interaction-interaction LD scores (WW) into: {gee_path}")
        self._save_score_file(gee_path, self.gee_ldscore)

        self._log_score_summary("XW / .gxe LD scores", self.gxe_ldscore)
        self._log_score_summary("WW / .gee LD scores", self.gee_ldscore)

        try:
            col_sums = pd.Series(self.nsnps_bin, index=self.l2cols)
            lines = ["Annotation Column Sums"] + [f"{k:<35} {v:.6f}" for k, v in col_sums.items()]
            self.log._log("\n" + "\n".join(lines))
        except Exception as e:
            self.log._log(f"[warn] Failed to report annotation column sums: {e}")

        self.end_time = utils._get_time()
        self.log._log("Calculation of genome-wide GxE LD scores ended at " + utils._get_timestr(self.end_time))
        self.runtime = self.end_time - self.start_time
        self.log._log(
            "Runtime: " + format(self.runtime, ".3f") +
            f" s ({self.runtime // 3600} hr {(self.runtime % 3600) // 60} m {(self.runtime % 60):.3f} s)"
        )
        # Save identical run logs to both output-specific log names for convenience.
        self.log._save_log(self.outpath + ".gxe.log")
        self.log._save_log(self.outpath + ".gee.log")


GenomewideGxELDScore = GenomewideEnvLDScore
