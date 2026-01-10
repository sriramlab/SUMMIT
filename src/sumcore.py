#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd

from sumstats import Sumstats
from trace import Trace
import utils


class Sumcore:
    def __init__(
        self,
        bim_path=None,
        save_path=None,
        rg=None,
        ldscores=None,
        log=None,
        verbose=False,
        chisq_threshold=0,
        annot=None,
        njack=None,
        out=None,
        # allele alignment
        align_alleles=True,
        drop_ambiguous=True,
    ):
        self.log = log
        self.verbose = verbose
        self.out = out
        self.align_alleles = bool(align_alleles)
        self.drop_ambiguous = bool(drop_ambiguous)

        self.chisq_threshold = chisq_threshold
        self.start_time = utils._get_time()
        self.log._log("Analysis started at: " + utils._get_timestr(self.start_time))

        # parse --rg argument: two comma-separated sumstat paths
        try:
            self.phen_dir = utils._parse_rgdir(rg)
        except ValueError as e:
            self.log._log(f"Error reading sumstat file pair: {e}")
            sys.exit(1)
        if len(self.phen_dir) != 2:
            self.log._log("ERROR: --rg must be exactly two comma-separated sumstat paths.")
            sys.exit(1)

        self.phen_names = [os.path.basename(p)[:-8] for p in self.phen_dir]

        # Trace (must be LD-score input)
        self.tr = Trace(
            bimpath=bim_path,
            sumpath=None,
            savepath=save_path,
            ldscores=ldscores,
            log=self.log,
            nblks=njack,
            annot=annot,
            verbose=verbose,
        )
        if self.tr.ldscores is None:
            raise RuntimeError("SUM-CORE requires LD-score input (Trace.ldscores is None).")

        self.nblks = self.tr.nblks
        self.nbins = self.tr.nbins
        self.annot_header = self.tr.annot_header

        # outputs/containers
        B, K = self.nblks, self.nbins
        self.nsamp = [np.nan, np.nan]

        self.sigmas = np.zeros((2, B + 1, K + 2), dtype=np.float64)  # [:, :, :K] genetic, [:, :, K] noise, [:, :, -1] sum genetic
        self.gamma_g = np.zeros((B + 1, K), dtype=np.float64)
        self.c_opt = np.zeros((B + 1,), dtype=np.float64)

        self.gamma_se = np.full((K,), np.nan, dtype=np.float64)
        self.rg = np.full((B + 1, K), np.nan, dtype=np.float64)
        self.rg_se = np.full((K,), np.nan, dtype=np.float64)

        # univariate h2 outputs (NEW)
        self.h2 = np.full((2, B + 1, K), np.nan, dtype=np.float64)
        self.h2_se = np.full((2, K), np.nan, dtype=np.float64)
        self.h2_total = np.full((2, B + 1), np.nan, dtype=np.float64)
        self.h2_total_se = np.full((2,), np.nan, dtype=np.float64)

    def _make_sumstats(self, annot_df):
        return Sumstats(
            nblks=self.nblks,
            chisq_threshold=self.chisq_threshold,
            log=self.log,
            annot_df=annot_df,
            nbins=self.nbins,
        )

    @staticmethod
    def _alleles_to_int(a):
        # maps A,C,G,T -> 0,1,2,3; else -1
        a = np.asarray(a, dtype=str)
        a = np.char.upper(a)
        out = np.full(a.shape, -1, dtype=np.int8)
        out[a == "A"] = 0
        out[a == "C"] = 1
        out[a == "G"] = 2
        out[a == "T"] = 3
        return out

    def _allele_masks(self, a1_ref, a2_ref, a1, a2):
        """
        Returns (keep_mask, flip_mask) to align (a1,a2) to (a1_ref,a2_ref).
        flip_mask means: z should be multiplied by -1 for the SECOND trait.
        """
        a1r = self._alleles_to_int(a1_ref)
        a2r = self._alleles_to_int(a2_ref)
        a1 = self._alleles_to_int(a1)
        a2 = self._alleles_to_int(a2)

        valid = (a1r >= 0) & (a2r >= 0) & (a1 >= 0) & (a2 >= 0)

        # ambiguous (A/T or C/G) in reference alleles (dropping is safest without freq)
        amb = (
            ((a1r == 0) & (a2r == 3)) | ((a1r == 3) & (a2r == 0)) |
            ((a1r == 1) & (a2r == 2)) | ((a1r == 2) & (a2r == 1))
        )

        comp_map = np.array([3, 2, 1, 0], dtype=np.int8)
        comp_a1 = np.where(a1 >= 0, comp_map[a1], -1)
        comp_a2 = np.where(a2 >= 0, comp_map[a2], -1)

        direct = (a1r == a1) & (a2r == a2)
        strand = (a1r == comp_a1) & (a2r == comp_a2)

        swapped = (a1r == a2) & (a2r == a1)
        swapped_strand = (a1r == comp_a2) & (a2r == comp_a1)

        flip = swapped | swapped_strand
        match = direct | strand | swapped | swapped_strand

        keep = valid & match
        if self.drop_ambiguous:
            keep = keep & (~amb)

        return keep, flip

    # --------------------------------------------------------------------- #
    # Fast SNP-set prep: avoid repeated re-processing / set/isin bottlenecks #
    # --------------------------------------------------------------------- #

    def _prepare_final_snpset_and_filter_trace(self):
        """
        1) Read both sumstats once (cache arrays + Index; chi^2 filter applied once).
        2) Build the common SNP set directly in Trace/annotation order via Index.get_indexer.
        3) Allele align on the common set (drop mismatches/ambiguous; compute flip mask).
        4) Filter Trace to final SNP list.
        5) Rematch both sumstats ONCE to annot_df_final (no reread), compute RHS.
        6) ASSERT: Trace order == Sumstats matched order == annot_df_final order.
        """
        annot_df_full = self.tr.annot_df
        if annot_df_full is None or "SNP" not in annot_df_full.columns:
            raise RuntimeError("Trace.annot_df missing or malformed.")

        # Base SNP universe / order for everything (this is what Trace uses)
        # Prefer Trace's own base SNPs if present.
        base_snps = np.asarray(getattr(self.tr, "_base_snps", annot_df_full["SNP"].astype(str).to_numpy()), dtype=str)
        if base_snps.size != annot_df_full.shape[0]:
            # fallback to annot_df_full order if something odd happened
            base_snps = annot_df_full["SNP"].astype(str).to_numpy()

        # 1) Read both sumstats ONCE (no matching yet)
        ss0 = self._make_sumstats(annot_df_full)
        ss1 = self._make_sumstats(annot_df_full)
        ss0.read_only(self.phen_dir[0], self.phen_names[0])
        ss1.read_only(self.phen_dir[1], self.phen_names[1])
        self.names = [ss0.name, ss1.name]

        self.nsamp[0] = float(ss0.nsamp)
        self.nsamp[1] = float(ss1.nsamp)

        # 2) Compute common SNPs in *base/annotation order*
        # indexer gives position in sumstats cache arrays; -1 => missing
        idx0 = ss0._sum_index.get_indexer(base_snps)
        idx1 = ss1._sum_index.get_indexer(base_snps)

        common_mask = (idx0 >= 0) & (idx1 >= 0)
        n_common = int(common_mask.sum())
        if n_common == 0:
            raise RuntimeError("No overlapping SNPs between the two sumstats after filtering.")

        common_pos = np.flatnonzero(common_mask)          # positions in base_snps / annot_df_full
        p0 = idx0[common_pos]                              # positions into ss0 cached arrays
        p1 = idx1[common_pos]                              # positions into ss1 cached arrays

        # 3) Allele alignment on this common set (trait0 as reference)
        flip_final = None
        keep2 = np.ones(common_pos.size, dtype=bool)

        if self.align_alleles:
            keep2, flip2 = self._allele_masks(
                ss0._sum_a1[p0], ss0._sum_a2[p0],
                ss1._sum_a1[p1], ss1._sum_a2[p1],
            )

            n_drop = int((~keep2).sum())
            n_flip = int(flip2[keep2].sum())
            if n_drop > 0:
                self.log._log(
                    f"Allele alignment: dropping {n_drop} SNPs "
                    f"(drop_ambiguous={self.drop_ambiguous}). "
                    f"Will flip trait2 z for {n_flip} SNPs among those kept."
                )

            final_pos = common_pos[keep2]
            flip_final = flip2[keep2]   # aligned with final_pos order

        else:
            final_pos = common_pos

        if final_pos.size == 0:
            raise RuntimeError("All overlapping SNPs were removed during allele alignment / ambiguity filtering.")

        # Final SNP list in base/annotation order
        keep_snps_final = base_snps[final_pos]

        # Final annotation df in that same order
        annot_df_final = annot_df_full.iloc[final_pos].reset_index(drop=True)

        # 4) Filter Trace ONCE to the final SNP list
        if hasattr(self.tr, "_filter_keep_snps"):
            self.tr._filter_keep_snps(keep_snps_final)
        else:
            # fallback (shouldn't happen in your codebase)
            removes = np.setdiff1d(base_snps, keep_snps_final, assume_unique=False).tolist()
            self.tr._filter_snps(removes)

        self.log._log(
            f"Final SNP set for SUM-CORE: keeping {keep_snps_final.size} SNPs out of {base_snps.size} annotated SNPs."
        )

        # HARD ASSERT: Trace order == keep_snps_final
        tr_snps = np.asarray(self.tr.snplist, dtype=str)
        if tr_snps.shape[0] != keep_snps_final.shape[0]:
            raise RuntimeError(
                f"Trace filtering mismatch: Trace kept {tr_snps.shape[0]} SNPs but keep_snps_final has {keep_snps_final.shape[0]}."
            )
        if not np.array_equal(tr_snps, keep_snps_final):
            raise RuntimeError(
                "SNP order mismatch after Trace filtering. Trace.snplist != keep_snps_final "
                "(should be identical in annotation order)."
            )

        # 5) Rematch sumstats to annot_df_final using cached arrays (NO reread), then compute RHS
        ss0.rematch_and_recompute(annot_df_final)
        ss1.rematch_and_recompute(annot_df_final)

        # 6) HARD ASSERT: sumstats matched order == annot_df_final order exactly
        snps_final_df = annot_df_final["SNP"].astype(str).to_numpy()
        s0 = np.asarray(ss0.matched_snps, dtype=str)
        s1 = np.asarray(ss1.matched_snps, dtype=str)

        if s0.shape[0] != snps_final_df.shape[0] or s1.shape[0] != snps_final_df.shape[0]:
            raise RuntimeError(
                "Post-filter mismatch: after final matching, one or both sumstats did not match all final SNPs."
            )
        if not np.array_equal(s0, snps_final_df) or not np.array_equal(s1, snps_final_df):
            raise RuntimeError(
                "Order mismatch: Sumstats.matched_snps != annot_df_final['SNP'] order. "
                "This would misalign LD-scores vs z-scores / blocks."
            )

        # sanity: M matches everywhere
        M = int(self.tr.nsnps)
        if ss0.zscores.size != M or ss1.zscores.size != M:
            raise RuntimeError(
                f"Post-filter mismatch: Trace M={M}, zscore lens {ss0.zscores.size} and {ss1.zscores.size}."
            )

        # Apply flip mask to trait2 zscores (aligned with final SNP order)
        if self.align_alleles and (flip_final is not None):
            if flip_final.shape[0] != M:
                raise RuntimeError(f"Internal error: flip mask length {flip_final.shape[0]} != final SNP count {M}.")
            ss1.zscores = ss1.zscores.copy()
            ss1.zscores[flip_final] *= -1.0

        # store final sumstats objects
        self.sums = [ss0, ss1]



    # ---------------------------- estimation ---------------------------- #

    def _estimate_univariate_sigmas(self):
        """
        Solve the same univariate normal equations as SUMRHE (sigma_g per bin + sigma_e),
        used to normalize rg. (These are the SUMRHE-identical variance component estimates.)
        """
        K = self.nbins
        for t in range(2):
            rhs = self.sums[t].rhs
            pred_tr = self.tr._calc_trace(self.nsamp[t])
            sig_est = utils._solve_linear_equation(pred_tr, rhs)  # (B+1, K+1)
            self.sigmas[t, :, :K + 1] = sig_est
            self.sigmas[t, :, -1] = sig_est[:, :K].sum(axis=1)

    def _precompute_h2_overlap_terms(self):
        """
        Precompute overlap/count terms needed by SUMRHE-style h2 calculation.
        In SUMCORE, Trace has already been filtered to the final shared SNP set,
        so these terms are shared by both phenotypes.
        """
        A = np.asarray(self.tr.annot, dtype=np.float32, order="C")  # (M, K)
        M, K = A.shape
        B = self.nblks

        # Full overlaps and counts
        overlap_full = A.T @ A              # (K, K)
        counts_full = A.sum(axis=0)         # (K,)

        # Block boundaries in the *filtered* SNP order
        bs = self.tr.blk_size
        starts = bs * np.arange(B, dtype=np.int64)
        ends = starts + bs
        if B > 0:
            ends[-1] = self.tr.nsnps

        overlap_blk = np.empty((B, K, K), dtype=np.float32)
        counts_blk = np.empty((B, K), dtype=np.float32)

        for j in range(B):
            ab = A[starts[j]:ends[j], :]
            overlap_blk[j] = ab.T @ ab
            counts_blk[j] = ab.sum(axis=0)

        self._h2_overlap_full = overlap_full
        self._h2_counts_full = counts_full
        self._h2_overlap_blk = overlap_blk
        self._h2_counts_blk = counts_blk

    def _estimate_univariate_h2(self):
        """
        SUMRHE-identical h2 calculation:
        - per-category h2 uses overlap-aware mapping (A^T A / counts) @ sigma_g
        - total h2 is sum_k sigma_gk (no sigma_e normalization)
        - SEs computed via jackknife on (B+1, K+1) matrix
        """
        if not hasattr(self, "_h2_overlap_full"):
            self._precompute_h2_overlap_terms()

        overlap_full = self._h2_overlap_full      # (K, K)
        counts_full = self._h2_counts_full        # (K,)
        overlap_blk = self._h2_overlap_blk        # (B, K, K)
        counts_blk = self._h2_counts_blk          # (B, K)

        B = self.nblks
        K = self.nbins

        # allocate like SUMRHE: (trait, B+1, K+1) where last column is total h2
        if not hasattr(self, "herits"):
            self.herits = np.full((2, B + 1, K + 1), np.nan, dtype=np.float64)
            self.hersums = np.full((2, K + 1, 2), np.nan, dtype=np.float64)  # [point, SE]

        # match SUMRHE nan_policy logic
        clip_nonfinite = bool(getattr(self, "clip_nonfinite_vals", False))
        nan_policy = "propagate" if clip_nonfinite else "omit"

        # LOO overlaps/counts
        overlap_minus = overlap_full[None, :, :] - overlap_blk          # (B, K, K)
        counts_minus = counts_full[None, :] - counts_blk                # (B, K)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_minus = overlap_minus / counts_minus[:, None, :]      # (B, K, K)
            ratio_full = overlap_full / counts_full[None, :]            # (K, K)

        for t in range(2):
            # LOO category h2: einsum over k
            sigma_g_minus = self.sigmas[t, :B, :K]                      # (B, K)
            h2_cat_minus = np.einsum("bck,bk->bc", ratio_minus, sigma_g_minus, optimize=True)

            bad_minus = ~np.isfinite(h2_cat_minus)
            if clip_nonfinite:
                h2_cat_minus[bad_minus] = 0.0  # SUMRHE legacy option
            # else: keep NaNs; jackknife will omit

            self.herits[t, :B, :K] = h2_cat_minus

            # Full category h2
            h2_cat_full = ratio_full @ self.sigmas[t, B, :K]            # (K,)
            bad_full = ~np.isfinite(h2_cat_full)
            if clip_nonfinite:
                h2_cat_full[bad_full] = 0.0

            self.herits[t, B, :K] = h2_cat_full

            # Total h2 = sum sigma_g
            self.herits[t, :, -1] = self.sigmas[t, :, :K].sum(axis=1)

            # Jackknife summaries (point + SE), identical to SUMRHE
            est_full, se_jk = utils._calc_jackknife_se(self.herits[t], axis=0, center="full", nan_policy=nan_policy)
            self.hersums[t, :, 0] = est_full
            self.hersums[t, :, 1] = se_jk


    def _estimate_gamma_and_rg(self):
        z1 = self.sums[0].zscores
        z2 = self.sums[1].zscores
        y = z1 * z2

        l2_bins = self.tr.ldscores          # (M, K)
        nsnps_blk = self.tr.nsnps_blk       # (B+1, K)

        # Step 1: robust intercept via smoothed-weight regression (ignore slopes)
        _, c_all = utils.bivariate_regression_partitioned_jn(
            l2_bins=l2_bins,
            y=y,
            nblks=self.nblks,
            n1=self.nsamp[0],
            n2=self.nsamp[1],
            nsnps_blk=nsnps_blk,
            blk_idx=self.tr.blk_idx,
            segment_ids=getattr(self.tr, "chr", None),          # smooth within chr if available
            positions_bp=getattr(self.tr, "bp", None),          # bp window smoothing if available
            smooth_bp_window=getattr(self, "smooth_bp_window", None),  # set on Sumcore if you want
            smooth_window=getattr(self, "smooth_window", 501),         # fallback
            weight_floor=getattr(self, "weight_floor", None),
        )
        self.c_opt = c_all

        # Step 2: plug intercept into SCORE normal equations (partitioned)
        ld_sum_all = self.tr.get_ldsum_all(use_cache=True)  # (B+1, K, K)

        t1_all = utils.compute_t1_all_jn(
            annot=self.tr.annot,          # (M, K)
            y=y,                          # (M,)
            blk_idx=self.tr.blk_idx,      # exact Trace blocks
            nblks=self.nblks,
        )                                 # (B+1, K)

        self.gamma_g = utils.solve_score_gamma_from_intercept_jn(
            ld_sum_all=ld_sum_all,
            t1_all=t1_all,
            nsnps_blk=nsnps_blk,
            c_all=c_all,
            n1=self.nsamp[0],
            n2=self.nsamp[1],
            ridge_rel=getattr(self, "ridge_rel", 1e-12),
        )

        # SE for gamma via jackknife
        _, se = utils._calc_jackknife_se(self.gamma_g, axis=0, center="full", nan_policy="propagate")
        self.gamma_se = se

        # rg per bin (same as your current normalization)
        K = self.nbins
        v1 = self.sigmas[0, :, :K]
        v2 = self.sigmas[1, :, :K]
        denom = v1 * v2
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rg = self.gamma_g / np.sqrt(denom)
        self.rg[~np.isfinite(self.rg)] = np.nan

        _, se_rg = utils._calc_jackknife_se(self.rg, axis=0, center="full", nan_policy="omit")
        self.rg_se = se_rg

    # ---------------------------- run / logging ---------------------------- #

    def _run(self):
        self._prepare_final_snpset_and_filter_trace()
        self._estimate_univariate_sigmas()
        self._estimate_univariate_h2()      # NEW
        self._estimate_gamma_and_rg()
        self._log_results()

    def _log_results(self):
        K = self.nbins

        # ---------- enrichment denominator: SNP proportions (from Trace) ----------
        Mbin = np.asarray(getattr(self.tr, "nsnps_bin", None), dtype=np.float64)
        if Mbin is None or Mbin.shape[0] != K:
            # fallback: use full row of nsnps_blk if needed
            Mbin = np.asarray(self.tr.nsnps_blk[-1], dtype=np.float64)
        Mtot = float(np.nansum(Mbin))

        def _enrichment(h2_cat, h2_tot):
            with np.errstate(divide="ignore", invalid="ignore"):
                prop_h2 = h2_cat / h2_tot
                prop_m = Mbin / Mtot
                enr = prop_h2 / prop_m
            enr[~np.isfinite(enr)] = np.nan
            return enr

        # ---------- sigma_g^2 SEs from jackknife ----------
        # sigmas[t] has shape (B+1, K+2); genetic bins are [:,:K]
        sigma_se = np.full((2, K), np.nan, dtype=np.float64)
        for t in range(2):
            _, se = utils._calc_jackknife_se(self.sigmas[t, :, :K], axis=0, center="full", nan_policy="propagate")
            sigma_se[t] = se

        # ---------- print per-trait blocks (SUMRHE style) ----------
        for t in range(2):
            phen_name = self.phen_names[t]

            # per-bin point estimates
            sigma_full = self.sigmas[t, -1, :K]
            h2_full = self.hersums[t, :K, 0]
            h2_se = self.hersums[t, :K, 1]
            h2_tot = float(self.hersums[t, -1, 0])
            h2_tot_se = float(self.hersums[t, -1, 1])

            enr = _enrichment(h2_full, h2_tot)

            for j, header in enumerate(self.annot_header):
                self.log._log(
                    f"^^^ Phenotype [{self.names[t]}] Bin [{header}] "
                    f"sigma_g^2: {sigma_full[j]:.6g} (SE: {sigma_se[t, j]:.6g}) "
                    f"h^2_cat: {h2_full[j]:.6g} (SE: {h2_se[j]:.6g}) "
                    f"Enrichment: {enr[j]:.6g}"
                )

            self.log._log(
                f"^^^ Phenotype [{self.names[t]}] Total SNP heritability (h^2): {h2_tot:.6g} SE: {h2_tot_se:.6g}"
            )

        # ---------- cross-trait block ----------
        # intercept c + SE
        c_full, c_se = utils._calc_jackknife_se(self.c_opt, axis=0, center="full", nan_policy="propagate")
        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] Intercept (c): {float(c_full):.6g} (SE: {float(c_se):.6g})"
        )

        for j, header in enumerate(self.annot_header):
            self.log._log(
                f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] Bin [{header}] "
                f"gamma_g: {self.gamma_g[-1, j]:.6g} (SE: {self.gamma_se[j]:.6g}) "
                f"rg: {self.rg[-1, j]:.6g} (SE: {self.rg_se[j]:.6g})"
            )

        # Optional totals (nice in practice)
        gamma_tot = np.nansum(self.gamma_g, axis=1)  # (B+1,)
        g_full, g_se = utils._calc_jackknife_se(gamma_tot, axis=0, center="full", nan_policy="propagate")

        # total h2 per replicate (we already store per-rep total h2 in herits)
        h2_0 = np.asarray(self.herits[0, :, -1], dtype=np.float64)
        h2_1 = np.asarray(self.herits[1, :, -1], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rg_tot = gamma_tot / np.sqrt(h2_0 * h2_1)
        rg_tot[~np.isfinite(rg_tot)] = np.nan
        r_full, r_se = utils._calc_jackknife_se(rg_tot, axis=0, center="full", nan_policy="omit")

        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] Total genetic covariance (gamma_g): {float(g_full):.6g} (SE: {float(g_se):.6g})"
        )
        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] Total genetic correlation (rg): {float(r_full):.6g} (SE: {float(r_se):.6g})"
        )


    def _logoff(self):
        self.end_time = utils._get_time()
        self.log._log("Analysis ended at: " + utils._get_timestr(self.end_time))
        self.log._log("run time: " + format(self.end_time - self.start_time, ".3f") + " s")
        if self.out is not None:
            self.log._log("Saved log in " + self.out + ".log")
            self.log._save_log(self.out + ".log")
        return

    # ---------------------------- optional overlap cov ---------------------------- #

    def estimate_overlap_cov(self, pheno_paths: str):
        """
        Compute sample covariance of overlapping individuals between two phenotype files.
        pheno_paths: "path1,path2" where each file has columns: FID IID PHENO
        """
        paths = pheno_paths.split(",")
        if len(paths) != 2:
            self.log._log("ERROR: pheno_paths must be exactly two comma-separated phenotype files.")
            sys.exit(1)

        df1 = pd.read_csv(paths[0], sep=r"\s+", header=0)
        df2 = pd.read_csv(paths[1], sep=r"\s+", header=0)

        if (len(df1.columns) != 3 or len(df2.columns) != 3):
            self.log._log("ERROR: The phenotype files must have exactly 3 columns: FID IID PHENO")
            sys.exit(1)

        pheno1 = df1.columns[2]
        pheno2 = df2.columns[2]

        df1 = df1.rename(columns={pheno1: "pheno1"}).dropna()
        df2 = df2.rename(columns={pheno2: "pheno2"}).dropna()

        merged = pd.merge(
            df1[["FID", "IID", "pheno1"]],
            df2[["FID", "IID", "pheno2"]],
            on=["FID", "IID"],
            how="inner",
        )

        self.n_overlap = int(merged.shape[0])

        if self.n_overlap < 2:
            self.log._log("NOTE: There are no (or only 1) overlapping individuals in the provided phenotypes.")
            self.cov = 0.0
        else:
            self.cov = float(np.cov(merged["pheno1"], merged["pheno2"], ddof=1)[0, 1])

        self.log._log(
            f"Sample covariance (y1^T y2) of overlapping ({self.n_overlap}) individuals: {self.cov:.5f}"
        )
        return self.cov
