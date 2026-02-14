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
        ldscores_reg=None,   # LD scores used for regression (e.g., windowed LD scores)
        log=None,
        verbose=False,
        chisq_threshold=0,
        annot=None,
        njack=None,
        out=None,
        # allele alignment
        align_alleles=False,
        drop_ambiguous=True,
        collapse_reg_ld=False,
        enrich_mode: str = "auto"
    ):
        self.log = log
        self.verbose = verbose
        self.out = out
        self.align_alleles = bool(align_alleles)
        self.drop_ambiguous = bool(drop_ambiguous)
        self.enrich_mode = self._normalize_enrich_mode(enrich_mode)
        self._enrich_mode_used = ["", ""]  # resolved mode per trait in logs


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

        # Trace (LD-score input)
        self.tr = Trace(
            bimpath=bim_path,
            sumpath=None,
            savepath=save_path,
            ldscores=ldscores,
            ldscores_reg=ldscores_reg,
            log=self.log,
            nblks=njack,
            annot=annot,
            verbose=verbose,
        )
        if self.tr.ldscores is None:
            raise RuntimeError("SUM-CORE requires LD-score input (Trace.ldscores is None).")

        if getattr(self.tr, "ldscores_reg", None) is not None:
            self.log._log("[rg] Using ldscores_reg for intercept regression; using ldscores for SCORE normal equation.")
        else:
            self.log._log("[rg] ldscores_reg not provided; using ldscores for both intercept regression and SCORE solve.")

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

        # univariate h2 outputs
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
    def _normalize_enrich_mode(mode: str) -> str:
        if mode is None:
            return "auto"
        m = str(mode).strip().lower().replace("_", "-").replace(" ", "-")
        if m == "auto":
            return "auto"
        if m in ("overlap", "overlapping"):
            return "overlap"
        if m in ("non-overlap", "nonoverlap", "nonoverlapping", "component", "components"):
            return "non-overlap"
        if m in ("both", "all"):
            return "both"
        raise ValueError(f"Invalid enrich_mode={mode!r}. Choose from: auto, overlap, non-overlap, both.")

    @staticmethod
    def _has_overlapping_annotations(A: np.ndarray) -> bool:
        """
        True if any SNP has >1 nonzero annotation entry.
        Works for binary or continuous weights (exact-zero test).
        """
        return bool(np.any(np.count_nonzero(A, axis=1) > 1))


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

        # 5) Rematch sumstats to annot_df_final using cached arrays (w/o reread), then compute RHS
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
        Precompute overlap/mass terms needed by SUMRHE-style h2 calculation.
        For continuous/overlapping annotations, "counts" here are bin masses:
            A_k = sum_j a_{j,k}
        not necessarily integer SNP counts.
        """
        A = np.asarray(self.tr.annot, dtype=np.float32, order="C")  # (M, K)
        M, K = A.shape
        B = self.nblks

        # Full overlaps and bin masses
        overlap_full = A.T @ A              # (K, K)
        mass_full = A.sum(axis=0)           # (K,)

        # Prefer Trace's own block bounds if present
        if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            starts = np.asarray(self.tr._blk_starts, dtype=np.int64)
            ends   = np.asarray(self.tr._blk_ends,   dtype=np.int64)
            if starts.size != B or ends.size != B:
                raise RuntimeError("Trace._blk_starts/_blk_ends shape mismatch with nblks.")
        else:
            # Derive contiguous bounds from blk_idx (recommended)
            blk_idx = np.asarray(self.tr.blk_idx, dtype=np.int64).ravel()
            if blk_idx.size != M:
                raise RuntimeError("Trace.blk_idx length mismatch with Trace.annot.")
            if blk_idx.min() < 0 or blk_idx.max() >= B:
                raise RuntimeError("Trace.blk_idx out of range [0, nblks-1].")

            # Check contiguity: blk_idx should be nondecreasing if blocks are contiguous slices
            if np.any(blk_idx[1:] < blk_idx[:-1]):
                raise RuntimeError("Trace.blk_idx is not nondecreasing; cannot form contiguous block slices safely.")

            # starts/ends per block id
            starts = np.full(B, -1, dtype=np.int64)
            ends   = np.full(B, -1, dtype=np.int64)
            # first occurrence
            changes = np.flatnonzero(np.r_[True, blk_idx[1:] != blk_idx[:-1]])
            for i, s in enumerate(changes):
                b = blk_idx[s]
                if starts[b] != -1:
                    raise RuntimeError(f"Block id {b} appears in multiple segments; blk_idx is not contiguous.")
                starts[b] = s
                # end is next change or M
                e = changes[i + 1] if (i + 1) < changes.size else M
                ends[b] = e

            if np.any(starts < 0) or np.any(ends < 0):
                raise RuntimeError("Some jackknife blocks are empty or missing in blk_idx.")
            
        # Cache JK block slicing so enrichment can compute replicate proportions fast
        self._jk_starts = starts.copy()
        self._jk_ends   = ends.copy()
        self._jk_m_blk  = (ends - starts).astype(np.int64, copy=False)
        self._jk_M_full = int(M)
           

        overlap_blk = np.empty((B, K, K), dtype=np.float32)
        mass_blk    = np.empty((B, K),     dtype=np.float32)

        for b in range(B):
            ab = A[int(starts[b]):int(ends[b]), :]
            overlap_blk[b] = ab.T @ ab
            mass_blk[b]    = ab.sum(axis=0)

        self._h2_overlap_full = overlap_full
        self._h2_mass_full    = mass_full
        self._h2_overlap_blk  = overlap_blk
        self._h2_mass_blk     = mass_blk


    def _estimate_univariate_h2(self):
        """
        SUMRHE category h2 calculation using overlap/mass mapping:

        h2_cat = ( (A^T A) / (sum_j a_{j,k}) ) @ sigma_g

        Works for overlapping and continuous annotations as long as annotation weights are
        nonnegative and Trace has been filtered to the final SNP set.
        """
        if not hasattr(self, "_h2_overlap_full"):
            self._precompute_h2_overlap_terms()

        overlap_full = self._h2_overlap_full      # (K, K)
        mass_full    = self._h2_mass_full         # (K,)
        overlap_blk  = self._h2_overlap_blk       # (B, K, K)
        mass_blk     = self._h2_mass_blk          # (B, K)

        B = self.nblks
        K = self.nbins

        if not hasattr(self, "herits"):
            self.herits  = np.full((2, B + 1, K + 1), np.nan, dtype=np.float64)
            self.hersums = np.full((2, K + 1, 2), np.nan, dtype=np.float64)  # [point, SE]

        clip_nonfinite = bool(getattr(self, "clip_nonfinite_vals", False))
        nan_policy = "propagate" if clip_nonfinite else "omit"

        # LOO overlaps/masses
        overlap_minus = overlap_full[None, :, :] - overlap_blk          # (B, K, K)
        mass_minus    = mass_full[None, :]       - mass_blk             # (B, K)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_minus = overlap_minus / mass_minus[:, None, :]        # (B, K, K)
            ratio_full  = overlap_full  / mass_full[None, :]            # (K, K)

        for t in range(2):
            sigma_g_minus = self.sigmas[t, :B, :K]                      # (B, K)
            h2_cat_minus = np.einsum("bck,bk->bc", ratio_minus, sigma_g_minus, optimize=True)

            bad_minus = ~np.isfinite(h2_cat_minus)
            if clip_nonfinite:
                h2_cat_minus[bad_minus] = 0.0

            self.herits[t, :B, :K] = h2_cat_minus

            h2_cat_full = ratio_full @ self.sigmas[t, B, :K]            # (K,)
            bad_full = ~np.isfinite(h2_cat_full)
            if clip_nonfinite:
                h2_cat_full[bad_full] = 0.0

            self.herits[t, B, :K] = h2_cat_full

            # total h2 in your convention = sum sigma_g
            self.herits[t, :, -1] = self.sigmas[t, :, :K].sum(axis=1)

            est_full, se_jk = utils._calc_jackknife_se(
                self.herits[t], axis=0, center="full", nan_policy=nan_policy
            )
            self.hersums[t, :, 0] = est_full
            self.hersums[t, :, 1] = se_jk

    def _get_prop_rep(self) -> np.ndarray:
        """
        prop_rep[r, k] = Ak_rep[r, k] / M_rep[r]
        where Ak_rep is bin mass (sum of annotation weights) and M_rep is SNP count,
        for r=0..B-1 leave-one-block-out, and r=B full.
        Cached since it's the same for both traits.
        """
        if hasattr(self, "_prop_rep") and (self._prop_rep is not None):
            return self._prop_rep

        if not hasattr(self, "_h2_mass_full"):
            self._precompute_h2_overlap_terms()

        mass_full = np.asarray(self._h2_mass_full, dtype=np.float64)      # (K,)
        mass_blk  = np.asarray(self._h2_mass_blk,  dtype=np.float64)      # (B,K)

        B = self.nblks
        K = self.nbins

        if not hasattr(self, "_jk_m_blk") or not hasattr(self, "_jk_M_full"):
            raise RuntimeError("Missing cached JK block sizes; _precompute_h2_overlap_terms must run first.")

        m_blk = np.asarray(self._jk_m_blk, dtype=np.float64)              # (B,)
        M_full = float(self._jk_M_full)

        Ak_rep = np.empty((B + 1, K), dtype=np.float64)
        Ak_rep[:B, :] = mass_full[None, :] - mass_blk
        Ak_rep[B,  :] = mass_full

        M_rep = np.empty((B + 1,), dtype=np.float64)
        M_rep[:B] = M_full - m_blk
        M_rep[B]  = M_full

        with np.errstate(divide="ignore", invalid="ignore"):
            prop = Ak_rep / M_rep[:, None]

        self._prop_rep = prop
        return prop


    def _compute_enrich_reps_and_se(self, t: int, mode: str):
        """
        Compute enrichment across all jackknife replicates and return (enr_full, enr_se).
        mode:
          - "overlap": uses SUMRHE-style SNP-set h2 (self.herits[t,:, :K])
          - "non-overlap": uses component share (self.sigmas[t,:, :K])
        """
        mode = self._normalize_enrich_mode(mode)
        if mode not in ("overlap", "non-overlap"):
            raise ValueError("mode must be 'overlap' or 'non-overlap'.")

        K = self.nbins
        prop = self._get_prop_rep()                                   # (B+1, K)

        # totals consistent with your convention (sum sigma_g)
        h2_tot = np.asarray(self.herits[t, :, -1], dtype=np.float64)  # (B+1,)

        if mode == "overlap":
            h2_cat = np.asarray(self.herits[t, :, :K], dtype=np.float64)   # (B+1,K)
        else:
            h2_cat = np.asarray(self.sigmas[t, :, :K], dtype=np.float64)   # (B+1,K)

        with np.errstate(divide="ignore", invalid="ignore"):
            enr = (h2_cat / h2_tot[:, None]) / prop

            invalid = (~np.isfinite(enr)) | (~np.isfinite(prop)) | (prop <= 0.0) | (h2_tot[:, None] <= 0.0)
            enr[invalid] = np.nan

        clip_nonfinite = bool(getattr(self, "clip_nonfinite_vals", False))
        nan_policy = "propagate" if clip_nonfinite else "omit"

        enr_full, enr_se = utils._calc_jackknife_se(enr, axis=0, center="full", nan_policy=nan_policy)
        return enr_full, enr_se




    def _estimate_gamma_and_rg(self):
        z1 = self.sums[0].zscores
        z2 = self.sums[1].zscores
        y = z1 * z2

        # Estimation LD scores (must match annotation bins)
        l2_bins_score = np.asarray(self.tr.ldscores, dtype=np.float64, order="C")  # (M, K_est)
        nsnps_blk_est = np.asarray(self.tr.nsnps_blk, dtype=np.float64)            # (B+1, K_est)

        M, K_est = l2_bins_score.shape
        if y.size != M:
            raise RuntimeError(f"Internal mismatch: y has {y.size} SNPs but Trace has M={M}.")

        # ----------------------------
        # 1D LD regressor for intercept regression
        # ----------------------------
        Lreg = getattr(self.tr, "ldscores_reg", None)

        if Lreg is None:
            # If user didn't provide ldscores_reg, only allow intercept regression if primary is already 1D
            if K_est == 1:
                L1 = l2_bins_score  # (M,1)
                self.log._log("[rg] ldscores_reg not provided; using primary ldscores (already 1D) for intercept regression.")
            else:
                raise RuntimeError(
                    "Intercept regression requires 1D --ldscores-reg by default.\n"
                    "You provided no --ldscores-reg and primary ldscores has multiple columns.\n"
                    "Provide a 1D ldscores_reg (recommended) OR pass --collapse-reg-ld to collapse ldscores_reg/primary to 1D."
                )
        else:
            Lreg = np.asarray(Lreg, dtype=np.float64, order="C")
            if Lreg.ndim == 1:
                if Lreg.shape[0] != M:
                    raise RuntimeError(f"ldscores_reg length {Lreg.shape[0]} != M={M}")
                L1 = Lreg.reshape(M, 1)
                self.log._log("[rg] Using provided 1D ldscores_reg for intercept regression.")
            elif Lreg.ndim == 2:
                if Lreg.shape[0] != M:
                    raise RuntimeError(f"ldscores_reg has M={Lreg.shape[0]} SNPs but Trace has M={M}")
                K_reg = Lreg.shape[1]
                if K_reg == 1:
                    L1 = Lreg
                    self.log._log("[rg] Using provided 1-column ldscores_reg for intercept regression.")
                else:
                    if not self.collapse_reg_ld:
                        raise RuntimeError(
                            f"--ldscores-reg must be 1D by default, but got {K_reg} columns.\n"
                            "If you really want to collapse multi-column regression LD scores to 1D total LD, pass --collapse-reg-ld."
                        )
                    # "Stable" collapse: sum in float64, keep shape (M,1)
                    L1 = np.sum(Lreg, axis=1, dtype=np.float64, keepdims=True)
                    self.log._log(
                        f"[rg] Collapsing ldscores_reg from {K_reg} columns to 1D total LD for intercept regression (--collapse-reg-ld)."
                    )
            else:
                raise RuntimeError("ldscores_reg must be a vector (M,) or matrix (M,K).")

        # nsnps_blk for regression step is only needed for shape scaling (gamma output ignored).
        # Use total SNP mass per block as a (B+1,1) placeholder.
        nsnps_blk_reg = np.sum(nsnps_blk_est, axis=1, keepdims=True)  # (B+1,1)

        # chisq filter only for intercept regression
        intercept_chisq_thr = getattr(self, "intercept_chisq_threshold", 30.0)
        if intercept_chisq_thr is not None:
            chisq1 = z1 * z1
            chisq2 = z2 * z2
        else:
            chisq1 = None
            chisq2 = None

        # Step 1: estimate intercept (LOO), using 1D LD regressor
        _, c_all = utils.bivariate_regression_partitioned_jn(
            l2_bins=L1,                     # (M,1)
            y=y,
            nblks=self.nblks,
            n1=self.nsamp[0],
            n2=self.nsamp[1],
            nsnps_blk=nsnps_blk_reg,        # (B+1,1)
            blk_idx=self.tr.blk_idx,
            weight_floor=getattr(self, "weight_floor", None),
            weight_cap_quantile=getattr(self, "weight_cap_quantile", None),
            chisq1=chisq1,
            chisq2=chisq2,
            chisq_threshold=intercept_chisq_thr,
            chisq_mode="either",
        )
        self.c_opt = c_all

        # Step 2: SCORE normal equations (ALL SNPs) with summary stats, using estimation bins
        ld_sum_all = self.tr.get_ldsum_all(use_cache=True)  # from self.tr.ldscores (B+1, K_est, K_est)

        t1_all = utils.compute_t1_all_jn(
            annot=self.tr.annot,          # (M, K_est)
            y=y,                          # (M,)
            blk_idx=self.tr.blk_idx,
            nblks=self.nblks,
        )                                 # (B+1, K_est)

        self.gamma_g = utils.solve_score_gamma_from_intercept_jn(
            ld_sum_all=ld_sum_all,
            t1_all=t1_all,
            nsnps_blk=nsnps_blk_est,      # (B+1, K_est)
            c_all=c_all,
            n1=self.nsamp[0],
            n2=self.nsamp[1],
            ridge_rel=getattr(self, "ridge_rel", 1e-12),
        )

        # SE for gamma via jackknife
        _, se = utils._calc_jackknife_se(self.gamma_g, axis=0, center="full", nan_policy="propagate")
        self.gamma_se = se

        # rg per bin
        v1 = self.sigmas[0, :, :K_est]
        v2 = self.sigmas[1, :, :K_est]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rg = self.gamma_g / np.sqrt(v1 * v2)
        self.rg[~np.isfinite(self.rg)] = np.nan

        _, se_rg = utils._calc_jackknife_se(self.rg, axis=0, center="full", nan_policy="omit")
        self.rg_se = se_rg


    # ---------------------------- run / logging ---------------------------- #

    def _run(self):
        self._prepare_final_snpset_and_filter_trace()
        self._estimate_univariate_sigmas()
        self._estimate_univariate_h2()
        self._estimate_gamma_and_rg()
        self._log_results()

    def _log_results(self):
        K = self.nbins
        M = float(self.tr.nsnps)

        A = np.asarray(self.tr.annot, dtype=np.float64, order="C")
        has_ov = self._has_overlapping_annotations(A)

        req = self.enrich_mode
        if req == "auto":
            mode_used = "overlap" if has_ov else "non-overlap"
            modes_to_compute = (mode_used,)
        elif req == "both":
            mode_used = "overlap" if has_ov else "non-overlap"
            modes_to_compute = ("non-overlap", "overlap")
        else:
            mode_used = req
            modes_to_compute = (req,)
        
        self.log._log(f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] enrichment_mode_used: {mode_used}")

        if self.verbose and req in ("auto", "both"):
            self.log._log(f"[enrichment] enrich_mode={req} resolved={mode_used} has_overlap={has_ov}")


        # sigma_g^2 SEs from jackknife
        sigma_se = np.full((2, K), np.nan, dtype=np.float64)
        for t in range(2):
            _, se = utils._calc_jackknife_se(self.sigmas[t, :, :K], axis=0, center="full", nan_policy="propagate")
            sigma_se[t] = se

        # per-trait blocks
        for t in range(2):
            sigma_full = self.sigmas[t, -1, :K]
            h2_full    = self.hersums[t, :K, 0]
            h2_se      = self.hersums[t, :K, 1]
            h2_tot     = float(self.hersums[t, -1, 0])
            h2_tot_se  = float(self.hersums[t, -1, 1])

            # Enrichment(s) + SE(s) from jackknife replicates
            enr_main_full, enr_main_se = self._compute_enrich_reps_and_se(t, mode_used)
            self._enrich_mode_used[t] = mode_used

            enr_no_full = enr_no_se = None
            enr_ov_full = enr_ov_se = None
            if req == "both":
                if "non-overlap" in modes_to_compute:
                    enr_no_full, enr_no_se = self._compute_enrich_reps_and_se(t, "non-overlap")
                if "overlap" in modes_to_compute:
                    enr_ov_full, enr_ov_se = self._compute_enrich_reps_and_se(t, "overlap")

            for j, header in enumerate(self.annot_header):
                enr_str = f"Enrichment: {enr_main_full[j]:.6g} (SE: {enr_main_se[j]:.6g})"
                if req == "both" and (enr_no_full is not None) and (enr_ov_full is not None):
                    enr_str = f"Enrichment_nonoverlap: {enr_no_full[j]:.6g} (SE: {enr_no_se[j]:.6g}) " + \
                                "& Enrichment_overlap: {enr_ov_full[j]:.6g} (SE: {enr_ov_se[j]:.6g})"
                self.log._log(
                    f"^^^ Phenotype [{self.names[t]}] Bin [{header}] "
                    f"sigma_g^2: {sigma_full[j]:.6g} (SE: {sigma_se[t, j]:.6g}) "
                    f"h^2_cat: {h2_full[j]:.6g} (SE: {h2_se[j]:.6g}) "+ enr_str                    
                )

            self.log._log(
                f"^^^ Phenotype [{self.names[t]}] Total SNP heritability (h^2): {h2_tot:.6g} SE: {h2_tot_se:.6g}"
            )

        # cross-trait
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

        # Totals
        gamma_tot = np.nansum(self.gamma_g, axis=1)  # (B+1,)
        g_full, g_se = utils._calc_jackknife_se(gamma_tot, axis=0, center="full", nan_policy="propagate")

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
