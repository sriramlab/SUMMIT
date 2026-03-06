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
        ldscores_reg=None,  # LD scores used for regression (e.g., windowed LD scores)
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
        enrich_mode: str = "auto",
        jack_mode: str = "mean",
        clip_nonfinite_vals=False,
        jackknife_weighted: bool = True,
        rg_se_method: str = "delta",
        intercept_chisq_thr=None,
    ):
        self.log = log
        self.verbose = verbose
        self.out = out
        self.align_alleles = bool(align_alleles)
        self.drop_ambiguous = bool(drop_ambiguous)
        self.enrich_mode = utils._normalize_enrich_mode(enrich_mode)
        self._enrich_mode_used = ["", ""]  # resolved mode per trait in logs
        self.jack_mode = jack_mode
        self.collapse_reg_ld = bool(collapse_reg_ld)
        self.clip_nonfinite_vals = clip_nonfinite_vals
        self.nan_policy = "propagate" if self.clip_nonfinite_vals else "omit"
        self._jackknife_weighted = bool(jackknife_weighted)
        self.rg_se_method = str(rg_se_method).strip().lower()
        if self.rg_se_method not in {"jackknife", "delta"}:
            raise ValueError("rg_se_method must be one of {'jackknife','delta'}")

        self.chisq_threshold = chisq_threshold
        self.intercept_chisq_thr = intercept_chisq_thr
        self._intercept_chisq_info = None

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
            nblks=njack,  # can be int-like or 'chr' now
            annot=annot,
            verbose=verbose,
        )

        if self.tr.ldscores is None:
            raise RuntimeError("SUM-CORE requires LD-score input (Trace.ldscores is None).")

        if getattr(self.tr, "ldscores_reg", None) is not None:
            self.log._log(
                "[rg] Using ldscores_reg for intercept regression; "
                "using ldscores for SCORE normal equation."
            )
        else:
            self.log._log(
                "[rg] ldscores_reg not provided; using ldscores for both "
                "intercept regression and SCORE solve."
            )

        # propagate jackknife partitioning info to Sumstats
        self._jackknife_partition_mode = getattr(self.tr, "jackknife_mode", "block")
        self._jackknife_chrs = getattr(self.tr, "jackknife_chrs", None)

        self.nblks = self.tr.nblks
        self.nbins = self.tr.nbins
        self.annot_header = self.tr.annot_header

        # outputs/containers
        B, K = self.nblks, self.nbins
        self.nsamp = [np.nan, np.nan]

        # [:, :, :K] genetic, [:, :, K] noise, [:, :, -1] sum genetic
        self.sigmas = np.zeros((2, B + 1, K + 2), dtype=np.float64)
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
            chisq_action="drop",  # explicit: trait-local filtering for h2/SCORE inputs
            log=self.log,
            annot_df=annot_df,
            nbins=self.nbins,
            jackknife_mode=getattr(self.tr, "jackknife_mode", "block"),
            jackknife_chrs=getattr(self.tr, "jackknife_chrs", None),
            jackknife_delete=int(getattr(self.tr, "jackknife_delete", 1)),
            jackknife_delete_sets=getattr(self.tr, "jackknife_delete_sets", None),
            jackknife_seed=getattr(self.tr, "jackknife_seed", None),
        )

    def _is_delete_d_jackknife(self) -> bool:
        """
        True if we are in chr-mode and the replicates are NOT the full delete-1 LOCO set.
        That includes:
        - delete_d > 1
        - or delete_d == 1 but random subset (R != U)
        """
        if getattr(self.tr, "jackknife_mode", "block") != "chr":
            return False
        d = int(getattr(self.tr, "jackknife_delete", 1))
        U = int(getattr(self.tr, "jackknife_units", 0))
        R = int(self.nblks)
        return (d > 1) or (U > 0 and R != U)

    def _jk_se(
        self,
        arr,
        axis=0,
        center=None,
        nan_policy=None,
        weights=None,
        use_pseudovalues=None,
    ):
        if center is None:
            center = self.jack_mode
        if nan_policy is None:
            nan_policy = self.nan_policy

        if getattr(self.tr, "jackknife_mode", "block") == "chr":
            D = self.tr.get_jackknife_delete_matrix(dtype=np.float64)
            if D is None:
                raise RuntimeError("chr-mode requested but Trace did not provide delete matrix D.")
            unit_sizes = self.tr.get_jackknife_unit_sizes(dtype=np.float64)
            if unit_sizes is None:
                raise RuntimeError("chr-mode requested but Trace did not provide unit sizes.")
            return utils._calc_jackknife_se_from_delete_sets(
                arr,
                D=D,
                unit_sizes=unit_sizes,
                axis=axis,
                center=center,
                nan_policy=nan_policy,
            )

        # block-mode
        if weights is None or use_pseudovalues is None:
            w, use_pv = self._get_jackknife_weights()
            if weights is None:
                weights = w
            if use_pseudovalues is None:
                use_pseudovalues = use_pv

        return utils._calc_jackknife_se(
            arr,
            axis=axis,
            center=center,
            nan_policy=nan_policy,
            weights=weights,
            use_pseudovalues=bool(use_pseudovalues),
        )

    def _get_jackknife_weights(self):
        if self._is_delete_d_jackknife():
            return None, False
        if not getattr(self, "_jackknife_weighted", False):
            return None, False
        if not (hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends")):
            return None, False

        starts = np.asarray(self.tr._blk_starts, dtype=np.float64)
        ends = np.asarray(self.tr._blk_ends, dtype=np.float64)
        if starts.size != self.nblks or ends.size != self.nblks:
            return None, False

        m = ends - starts
        return m, True

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
            ((a1r == 0) & (a2r == 3))
            | ((a1r == 3) & (a2r == 0))
            | ((a1r == 1) & (a2r == 2))
            | ((a1r == 2) & (a2r == 1))
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

    # ---------------------------------------------------------------------
    #
    # Fast SNP-set prep: avoid repeated re-processing / set/isin bottlenecks
    #
    # ---------------------------------------------------------------------
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
        base_snps = np.asarray(
            getattr(self.tr, "_base_snps", annot_df_full["SNP"].astype(str).to_numpy()),
            dtype=str,
        )
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

        # 2) Compute common SNPs in base/annotation order
        idx0 = ss0._sum_index.get_indexer(base_snps)
        idx1 = ss1._sum_index.get_indexer(base_snps)
        common_mask = (idx0 >= 0) & (idx1 >= 0)
        n_common = int(common_mask.sum())
        if n_common == 0:
            raise RuntimeError("No overlapping SNPs between the two sumstats after filtering.")

        common_pos = np.flatnonzero(common_mask)  # positions in base_snps / annot_df_full
        p0 = idx0[common_pos]  # positions into ss0 cached arrays
        p1 = idx1[common_pos]  # positions into ss1 cached arrays

        # 3) Allele alignment on this common set (trait0 as reference)
        flip_final = None
        if self.align_alleles:
            keep2, flip2 = self._allele_masks(
                ss0._sum_a1[p0],
                ss0._sum_a2[p0],
                ss1._sum_a1[p1],
                ss1._sum_a2[p1],
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
            flip_final = flip2[keep2]  # aligned with final_pos order
        else:
            final_pos = common_pos

        if final_pos.size == 0:
            raise RuntimeError(
                "All overlapping SNPs were removed during allele alignment / ambiguity filtering."
            )

        # Final SNP list in base/annotation order
        keep_snps_final = base_snps[final_pos]
        # Final annotation df in that same order
        annot_df_final = annot_df_full.iloc[final_pos].reset_index(drop=True)

        # 4) Filter Trace ONCE to the final SNP list
        if hasattr(self.tr, "_filter_keep_snps"):
            self.tr._filter_keep_snps(keep_snps_final)
        else:
            removes = np.setdiff1d(base_snps, keep_snps_final, assume_unique=False).tolist()
            self.tr._filter_snps(removes)

        self.log._log(
            f"Final SNP set for SUM-CORE: keeping {keep_snps_final.size} SNPs "
            f"out of {base_snps.size} annotated SNPs."
        )

        # HARD ASSERT: Trace order == keep_snps_final
        tr_snps = np.asarray(self.tr.snplist, dtype=str)
        if tr_snps.shape[0] != keep_snps_final.shape[0]:
            raise RuntimeError(
                f"Trace filtering mismatch: Trace kept {tr_snps.shape[0]} SNPs but "
                f"keep_snps_final has {keep_snps_final.shape[0]}."
            )
        if not np.array_equal(tr_snps, keep_snps_final):
            raise RuntimeError(
                "SNP order mismatch after Trace filtering. Trace.snplist != keep_snps_final "
                "(should be identical in annotation order)."
            )

        # 5) Rematch sumstats to annot_df_final using cached arrays, then compute RHS
        ss0.rematch_and_recompute(annot_df_final)
        ss1.rematch_and_recompute(annot_df_final)

        # 6) HARD ASSERT: sumstats matched order == annot_df_final order exactly
        snps_final_df = annot_df_final["SNP"].astype(str).to_numpy()
        s0 = np.asarray(ss0.matched_snps, dtype=str)
        s1 = np.asarray(ss1.matched_snps, dtype=str)

        if s0.shape[0] != snps_final_df.shape[0] or s1.shape[0] != snps_final_df.shape[0]:
            raise RuntimeError(
                "Post-filter mismatch: after final matching, one or both sumstats "
                "did not match all final SNPs."
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
                f"Post-filter mismatch: Trace M={M}, zscore lens {ss0.zscores.size} "
                f"and {ss1.zscores.size}."
            )

        # Apply flip mask to trait2 zscores (aligned with final SNP order)
        if self.align_alleles and (flip_final is not None):
            if flip_final.shape[0] != M:
                raise RuntimeError(
                    f"Internal error: flip mask length {flip_final.shape[0]} != final SNP count {M}."
                )
            ss1.zscores = ss1.zscores.copy()
            ss1.zscores[flip_final] *= -1.0

        # store final sumstats objects
        self.sums = [ss0, ss1]

    # ---------------------------- estimation ---------------------------- #
    def _estimate_univariate_sigmas(self):
        """
        Solve the same univariate normal equations as SUMRHE
        (sigma_g per bin + sigma_e), used to normalize rg.
        (These are the SUMRHE-identical variance component estimates.)
        """
        K = self.nbins
        for t in range(2):
            rhs = self.sums[t].rhs
            pred_tr = self.tr._calc_trace(self.nsamp[t])
            sig_est = utils._solve_linear_equation(pred_tr, rhs)  # (B+1, K+1)
            self.sigmas[t, :, : K + 1] = sig_est
            self.sigmas[t, :, -1] = sig_est[:, :K].sum(axis=1)

    def _precompute_h2_overlap_terms(self):
        A = np.asarray(self.tr.annot, dtype=np.float32, order="C")  # (M, K)
        M, K = A.shape
        B = int(self.nblks)

        overlap_full = A.T @ A
        mass_full = A.sum(axis=0)

        if getattr(self.tr, "jackknife_mode", "block") == "chr":
            # delete-* replicates defined by D over chromosome units
            D = self.tr.get_jackknife_delete_matrix(dtype=np.float32)  # (B,U)
            if D is None:
                raise RuntimeError("chr-mode requires Trace delete matrix D.")
            U = int(D.shape[1])

            if hasattr(self.tr, "_unit_starts") and hasattr(self.tr, "_unit_ends"):
                starts_u = np.asarray(self.tr._unit_starts, dtype=np.int64)
                ends_u = np.asarray(self.tr._unit_ends, dtype=np.int64)
            elif hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
                starts_u = np.asarray(self.tr._blk_starts, dtype=np.int64)
                ends_u = np.asarray(self.tr._blk_ends, dtype=np.int64)
            else:
                raise RuntimeError("chr-mode requires unit bounds (_unit_starts/_unit_ends).")

            if starts_u.size != U or ends_u.size != U:
                raise RuntimeError("Unit bounds length mismatch with D.")

            m_unit = (ends_u - starts_u).astype(np.float64)

            unit_mass = np.zeros((U, K), dtype=np.float32)
            unit_ov_flat = np.zeros((U, K * K), dtype=np.float32)
            for u in range(U):
                s = int(starts_u[u])
                e = int(ends_u[u])
                if e <= s:
                    continue
                Au = A[s:e, :]
                unit_mass[u] = Au.sum(axis=0, dtype=np.float32)
                unit_ov_flat[u] = (Au.T @ Au).reshape(-1)

            del_mass = (D @ unit_mass).astype(np.float32, copy=False)  # (B,K)
            del_ov_flat = (D @ unit_ov_flat).astype(np.float32, copy=False)  # (B,K*K)

            overlap_blk = del_ov_flat.reshape(B, K, K)  # deleted overlap per replicate
            mass_blk = del_mass  # deleted mass per replicate
            m_blk = D.astype(np.float64) @ m_unit  # deleted SNP count per replicate

            self._jk_m_blk = m_blk.astype(np.float64, copy=False)
            self._jk_M_full = int(M)

        else:
            if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
                starts = np.asarray(self.tr._blk_starts, dtype=np.int64)
                ends = np.asarray(self.tr._blk_ends, dtype=np.int64)
                if starts.size != B or ends.size != B:
                    raise RuntimeError("Trace._blk_starts/_blk_ends shape mismatch with nblks.")
            else:
                blk_idx = np.asarray(self.tr.blk_idx, dtype=np.int64).ravel()
                if blk_idx.size != M:
                    raise RuntimeError("Trace.blk_idx length mismatch with Trace.annot.")
                if blk_idx.min() < 0 or blk_idx.max() >= B:
                    raise RuntimeError("Trace.blk_idx out of range [0, nblks-1].")
                if np.any(blk_idx[1:] < blk_idx[:-1]):
                    raise RuntimeError(
                        "Trace.blk_idx is not nondecreasing; cannot form contiguous slices."
                    )

                starts = np.full(B, -1, dtype=np.int64)
                ends = np.full(B, -1, dtype=np.int64)
                changes = np.flatnonzero(np.r_[True, blk_idx[1:] != blk_idx[:-1]])
                for i, s in enumerate(changes):
                    b = blk_idx[s]
                    if starts[b] != -1:
                        raise RuntimeError(f"Block id {b} appears in multiple segments.")
                    starts[b] = s
                    e = changes[i + 1] if (i + 1) < changes.size else M
                    ends[b] = e

                if np.any(starts < 0) or np.any(ends < 0):
                    raise RuntimeError("Some jackknife blocks are empty or missing in blk_idx.")

            m_blk = (ends - starts).astype(np.float64, copy=False)
            self._jk_m_blk = m_blk
            self._jk_M_full = int(M)

            overlap_blk = np.empty((B, K, K), dtype=np.float32)
            mass_blk = np.empty((B, K), dtype=np.float32)
            for b in range(B):
                ab = A[int(starts[b]) : int(ends[b]), :]
                overlap_blk[b] = ab.T @ ab
                mass_blk[b] = ab.sum(axis=0)

        self._h2_overlap_full = overlap_full
        self._h2_mass_full = mass_full
        self._h2_overlap_blk = overlap_blk
        self._h2_mass_blk = mass_blk

    def _estimate_univariate_h2(self):
        if not hasattr(self, "_h2_overlap_full"):
            self._precompute_h2_overlap_terms()

        overlap_full = self._h2_overlap_full
        mass_full = self._h2_mass_full
        overlap_blk = self._h2_overlap_blk
        mass_blk = self._h2_mass_blk

        B = int(self.nblks)
        K = int(self.nbins)

        if not hasattr(self, "herits"):
            self.herits = np.full((2, B + 1, K + 1), np.nan, dtype=np.float64)
            self.hersums = np.full((2, K + 1, 2), np.nan, dtype=np.float64)

        overlap_minus = overlap_full[None, :, :] - overlap_blk
        mass_minus = mass_full[None, :] - mass_blk

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_minus = overlap_minus / mass_minus[:, None, :]
            ratio_full = overlap_full / mass_full[None, :]

        for t in range(2):
            sigma_g_minus = self.sigmas[t, :B, :K]
            h2_cat_minus = np.einsum("bck,bk->bc", ratio_minus, sigma_g_minus, optimize=True)

            bad_minus = ~np.isfinite(h2_cat_minus)
            if self.clip_nonfinite_vals:
                h2_cat_minus[bad_minus] = 0.0
            else:
                h2_cat_minus[bad_minus] = np.nan
            self.herits[t, :B, :K] = h2_cat_minus

            h2_cat_full = ratio_full @ self.sigmas[t, B, :K]
            bad_full = ~np.isfinite(h2_cat_full)
            if self.clip_nonfinite_vals:
                h2_cat_full[bad_full] = 0.0
            else:
                h2_cat_full[bad_full] = np.nan
            self.herits[t, B, :K] = h2_cat_full

            self.herits[t, :, -1] = self.sigmas[t, :, :K].sum(axis=1)

            est_full, se_jk = self._jk_se(self.herits[t], axis=0)
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

        mass_full = np.asarray(self._h2_mass_full, dtype=np.float64)  # (K,)
        mass_blk = np.asarray(self._h2_mass_blk, dtype=np.float64)  # (B,K)

        B = self.nblks
        K = self.nbins

        if not hasattr(self, "_jk_m_blk") or not hasattr(self, "_jk_M_full"):
            raise RuntimeError(
                "Missing cached JK block sizes; _precompute_h2_overlap_terms must run first."
            )
        m_blk = np.asarray(self._jk_m_blk, dtype=np.float64)  # (B,)
        M_full = float(self._jk_M_full)

        Ak_rep = np.empty((B + 1, K), dtype=np.float64)
        Ak_rep[:B, :] = mass_full[None, :] - mass_blk
        Ak_rep[B, :] = mass_full

        M_rep = np.empty((B + 1,), dtype=np.float64)
        M_rep[:B] = M_full - m_blk
        M_rep[B] = M_full

        with np.errstate(divide="ignore", invalid="ignore"):
            prop = Ak_rep / M_rep[:, None]

        self._prop_rep = prop
        return prop

    def _compute_enrich_reps_and_se(self, t: int, mode: str):
        K = int(self.nbins)
        prop = self._get_prop_rep()  # (B+1, K)
        h2_tot = np.asarray(self.herits[t, :, -1], dtype=np.float64)  # (B+1,)

        if mode == "overlap":
            h2_cat = np.asarray(self.herits[t, :, :K], dtype=np.float64)
        elif mode in ("non-overlap", "nonoverlap"):
            h2_cat = np.asarray(self.sigmas[t, :, :K], dtype=np.float64)
        else:
            raise ValueError(f"Unknown enrichment mode {mode!r}")

        with np.errstate(divide="ignore", invalid="ignore"):
            enr = (h2_cat / h2_tot[:, None]) / prop

        invalid = (
            (~np.isfinite(enr))
            | (~np.isfinite(prop))
            | (prop <= 0.0)
            | (h2_tot[:, None] <= 0.0)
        )
        enr[invalid] = np.nan

        enr_full, enr_se = self._jk_se(enr, axis=0)
        return enr_full, enr_se

    def _estimate_total_rg_delta_se(self):
        gamma_tot = np.asarray(np.nansum(self.gamma_g, axis=1), dtype=np.float64)  # (B+1,)
        h2_0 = np.asarray(self.herits[0, :, -1], dtype=np.float64)  # (B+1,)
        h2_1 = np.asarray(self.herits[1, :, -1], dtype=np.float64)  # (B+1,)

        g = float(gamma_tot[-1])
        v1 = float(h2_0[-1])
        v2 = float(h2_1[-1])

        if not (np.isfinite(g) and np.isfinite(v1) and np.isfinite(v2) and v1 > 0.0 and v2 > 0.0):
            return np.nan, np.nan

        rg_full = g / np.sqrt(v1 * v2)

        # Gradient of f(g, v1, v2) = g / sqrt(v1*v2) at the full-sample estimate
        d_g = 1.0 / np.sqrt(v1 * v2)
        d_v1 = -0.5 * rg_full / v1
        d_v2 = -0.5 * rg_full / v2

        lin_rg = (
            rg_full
            + d_g * (gamma_tot - g)
            + d_v1 * (h2_0 - v1)
            + d_v2 * (h2_1 - v2)
        ).astype(np.float64, copy=False)
        lin_rg[-1] = rg_full

        _, rg_se = self._jk_se(
            lin_rg,
            axis=0,
            center=self.jack_mode,
            nan_policy=self.nan_policy,
        )
        return rg_full, float(rg_se)

    def _get_jackknife_unit_partition(self):
        """
        Return contiguous unit bounds plus a deletion-incidence matrix D.
        Units are:
        - chr-mode: chromosome units provided by Trace
        - block-mode: contiguous jackknife blocks

        Returns
        -------
        starts : (U,) int64
        ends : (U,) int64
        D : (R, U) float64
            D[r, u] = 1 if replicate r deletes unit u.
        """
        jm = getattr(self.tr, "jackknife_mode", "block")

        if jm == "chr":
            D = self.tr.get_jackknife_delete_matrix(dtype=np.float64)
            if D is None:
                raise RuntimeError("chr-mode requested but Trace did not provide delete matrix D.")
            D = np.asarray(D, dtype=np.float64, order="C")
            R, U = D.shape

            if R != int(self.nblks):
                raise RuntimeError(
                    f"Delete matrix row mismatch: D has {R} rows, expected {self.nblks}."
                )

            if hasattr(self.tr, "_unit_starts") and hasattr(self.tr, "_unit_ends"):
                starts = np.asarray(self.tr._unit_starts, dtype=np.int64)
                ends = np.asarray(self.tr._unit_ends, dtype=np.int64)
            elif hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
                starts = np.asarray(self.tr._blk_starts, dtype=np.int64)
                ends = np.asarray(self.tr._blk_ends, dtype=np.int64)
            else:
                raise RuntimeError("chr-mode requires Trace unit bounds (_unit_starts/_unit_ends).")

            if starts.size != U or ends.size != U:
                raise RuntimeError(
                    f"Unit-bound mismatch: got starts/ends of length {starts.size}/{ends.size}, "
                    f"expected U={U}."
                )

            return starts, ends, D

        # ---------------- block mode ----------------
        B = int(self.nblks)

        if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            starts = np.asarray(self.tr._blk_starts, dtype=np.int64)
            ends = np.asarray(self.tr._blk_ends, dtype=np.int64)
            if starts.size != B or ends.size != B:
                raise RuntimeError("Trace._blk_starts/_blk_ends shape mismatch with nblks.")
        else:
            if not hasattr(self.tr, "blk_idx"):
                raise RuntimeError(
                    "block-mode requires Trace.blk_idx or Trace._blk_starts/_blk_ends."
                )

            blk_idx = np.asarray(self.tr.blk_idx, dtype=np.int64).ravel()
            M = int(self.tr.nsnps)

            if blk_idx.size != M:
                raise RuntimeError("Trace.blk_idx length mismatch with Trace.nsnps.")
            if blk_idx.min() < 0 or blk_idx.max() >= B:
                raise RuntimeError("Trace.blk_idx out of range [0, nblks-1].")
            if np.any(blk_idx[1:] < blk_idx[:-1]):
                raise RuntimeError("Trace.blk_idx must be nondecreasing.")

            starts = np.full(B, -1, dtype=np.int64)
            ends = np.full(B, -1, dtype=np.int64)
            changes = np.flatnonzero(np.r_[True, blk_idx[1:] != blk_idx[:-1]])
            for i, s in enumerate(changes):
                b = int(blk_idx[s])
                if starts[b] != -1:
                    raise RuntimeError(f"Block id {b} appears in multiple disjoint segments.")
                starts[b] = s
                ends[b] = changes[i + 1] if (i + 1) < changes.size else M

            if np.any(starts < 0) or np.any(ends < 0):
                raise RuntimeError("Some jackknife blocks are empty or missing in blk_idx.")

        D = np.eye(B, dtype=np.float64)
        return starts, ends, D

    def _make_intercept_keep_mask(self, z1, z2, threshold=None, chisq_mode="either"):
        """
        Keep mask for the cross-trait intercept regression only.

        This is intentionally orthogonal to the trait-local Sumstats chi^2 filtering:
        it is applied only when estimating c_all, and does not alter the univariate
        RHS / sigma / h2 pieces or the SCORE sufficient statistics.
        """
        z1 = np.asarray(z1, dtype=np.float64).ravel()
        z2 = np.asarray(z2, dtype=np.float64).ravel()
        if z1.size != z2.size:
            raise ValueError("z1/z2 length mismatch in intercept keep-mask construction.")

        keep = np.isfinite(z1) & np.isfinite(z2)

        n_auto = float(np.nanmax(np.asarray(self.nsamp, dtype=np.float64)))
        thr, thr_mode = utils._resolve_chisq_threshold(n_auto, threshold)

        mode = str(chisq_mode).strip().lower()
        thr_used = None

        if thr is not None:
            thr = float(thr)
            if np.isfinite(thr) and thr > 0.0:
                c1 = z1 * z1
                c2 = z2 * z2

                # "either" and "max" are equivalent here; keep alias for compatibility.
                if mode in ("either", "max"):
                    keep &= (c1 <= thr) & (c2 <= thr)
                elif mode == "both":
                    keep &= ~((c1 > thr) & (c2 > thr))
                else:
                    raise ValueError("chisq_mode must be one of {'either','both','max'}")
                thr_used = thr

        self._intercept_chisq_info = {
            "threshold": thr_used,
            "threshold_mode": thr_mode,
            "chisq_mode": mode,
            "n_total": int(z1.size),
            "n_kept": int(keep.sum()),
            "n_removed": int((~keep).sum()),
        }
        return keep

    def _build_intercept_weights(self, L, keep):
        """
        Build per-SNP regression weights for the intercept regression.
        Only rows with keep=True are active.
        """
        L = np.asarray(L, dtype=np.float64, order="C")
        keep = np.asarray(keep, dtype=bool).ravel()

        if L.ndim != 2:
            raise ValueError("L must be 2D (M, Kreg).")
        if L.shape[0] != keep.size:
            raise ValueError("L/keep length mismatch.")

        ltot = np.sum(L, axis=1, dtype=np.float64)

        wf = getattr(self, "weight_floor", None)
        if wf is None:
            bad = keep & ((~np.isfinite(ltot)) | (ltot <= 0.0))
            if np.any(bad):
                nb = int(bad.sum())
                mn = float(np.nanmin(ltot)) if np.isfinite(ltot).any() else np.nan
                raise ValueError(
                    f"Total LD must be finite and >0 for kept SNPs; found {nb} bad kept SNPs "
                    f"(min total LD={mn}). Pass weight_floor to clamp."
                )
            ltot_safe = np.ones_like(ltot, dtype=np.float64)
            ltot_safe[keep] = ltot[keep]
        else:
            eps = float(wf)
            if not (np.isfinite(eps) and eps > 0.0):
                raise ValueError("weight_floor must be positive finite.")
            ltot_safe = np.where(np.isfinite(ltot), ltot, eps)
            ltot_safe = np.maximum(ltot_safe, eps)

        w = np.zeros(L.shape[0], dtype=np.float64)
        w[keep] = 1.0 / ltot_safe[keep]

        qcap = getattr(self, "weight_cap_quantile", None)
        if qcap is not None:
            q = float(qcap)
            if not (0.0 < q < 1.0):
                raise ValueError("weight_cap_quantile must be in (0,1).")
            wpos = w[w > 0.0]
            if wpos.size > 0:
                cap = float(np.quantile(wpos, q))
                if np.isfinite(cap) and cap > 0.0:
                    w = np.minimum(w, cap)

        if not np.isfinite(w).all():
            raise ValueError("Non-finite intercept regression weights encountered.")
        if w.sum() <= 0.0:
            raise ValueError("No SNPs remain for intercept regression after filtering (sum(w)=0).")

        return w

    @staticmethod
    def _solve_raw_wls_beta(Wv, sv, Qv, tv, uv):
        """
        Solve the raw weighted normal equations for [intercept, slopes].
        Used only as a hard fallback when the centered solve fails for the full sample.
        """
        Kreg = int(np.asarray(sv).size)
        p = Kreg + 1

        SXX = np.empty((p, p), dtype=np.float64)
        SXY = np.empty((p,), dtype=np.float64)

        SXX[0, 0] = Wv
        SXX[0, 1:] = sv
        SXX[1:, 0] = sv
        SXX[1:, 1:] = Qv

        SXY[0] = tv
        SXY[1:] = uv

        try:
            return np.linalg.solve(SXX, SXY)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(SXX, SXY, rcond=None)[0]

    def _solve_centered_wls_beta(self, Wv, sv, Qv, tv, uv, global_scale):
        """
        Stable centered weighted-LS solve for [c, beta].
        Returns (beta, status), where beta has shape (Kreg+1,).
        """
        sv = np.asarray(sv, dtype=np.float64).ravel()
        uv = np.asarray(uv, dtype=np.float64).ravel()
        Qv = np.asarray(Qv, dtype=np.float64)
        Kreg = int(sv.size)

        beta = np.full((Kreg + 1,), np.nan, dtype=np.float64)

        if not (
            np.isfinite(Wv)
            and Wv > 0.0
            and np.isfinite(tv)
            and np.isfinite(sv).all()
            and np.isfinite(Qv).all()
            and np.isfinite(uv).all()
        ):
            return beta, "invalid"

        ridge_rel = float(getattr(self, "intercept_ridge_rel", 1e-10))
        var_floor_rel = float(getattr(self, "intercept_var_floor_rel", 1e-12))
        cond_max = float(getattr(self, "intercept_cond_max", 1e12))
        floor_abs = max(var_floor_rel * float(global_scale), 0.0)

        muL = sv / Wv
        muy = tv / Wv
        C = Qv - np.outer(sv, sv) / Wv
        C = 0.5 * (C + C.T)
        r = uv - sv * (tv / Wv)

        if not (
            np.isfinite(C).all()
            and np.isfinite(r).all()
            and np.isfinite(muL).all()
            and np.isfinite(muy)
        ):
            return beta, "invalid"

        # ---- scalar case ----
        if Kreg == 1:
            denom = float(C[0, 0])
            numer = float(r[0])
            if not (np.isfinite(denom) and np.isfinite(numer)):
                return beta, "invalid"

            reg_used = False
            lam = max(ridge_rel * float(global_scale), 0.0)
            if denom <= floor_abs:
                lam = max(lam, floor_abs - denom)
                reg_used = True

            denom_reg = denom + lam
            if not (np.isfinite(denom_reg) and denom_reg > 0.0):
                return beta, "invalid"

            b = numer / denom_reg
            c = muy - muL[0] * b
            if not (np.isfinite(b) and np.isfinite(c)):
                return beta, "invalid"

            beta[0] = c
            beta[1] = b
            return beta, ("ridge" if reg_used else "ok")

        # ---- multivariate case ----
        diagC = np.diag(C).astype(np.float64)
        dscale = np.sqrt(np.maximum(np.where(np.isfinite(diagC), diagC, 0.0), floor_abs))
        if not np.isfinite(dscale).all():
            return beta, "invalid"

        A = C / dscale[:, None] / dscale[None, :]
        rhs = r / dscale

        if not (np.isfinite(A).all() and np.isfinite(rhs).all()):
            return beta, "invalid"

        reg_used = False
        A_solve = A
        try:
            condA = float(np.linalg.cond(A))
        except np.linalg.LinAlgError:
            condA = np.inf

        if (not np.isfinite(condA)) or (condA > cond_max):
            lam_std = max(ridge_rel, var_floor_rel)
            A_solve = A + lam_std * np.eye(Kreg, dtype=np.float64)
            reg_used = True
        elif ridge_rel > 0.0:
            A_solve = A + ridge_rel * np.eye(Kreg, dtype=np.float64)
            reg_used = True

        try:
            b_std = np.linalg.solve(A_solve, rhs)
        except np.linalg.LinAlgError:
            lam_std = max(ridge_rel, var_floor_rel)
            try:
                b_std = np.linalg.solve(A + lam_std * np.eye(Kreg, dtype=np.float64), rhs)
                reg_used = True
            except np.linalg.LinAlgError:
                b_std = np.linalg.lstsq(A_solve, rhs, rcond=None)[0]
                reg_used = True

        b = b_std / dscale
        c = muy - muL @ b
        if not (np.isfinite(b).all() and np.isfinite(c)):
            return beta, "invalid"

        beta[0] = c
        beta[1:] = b
        return beta, ("ridge" if reg_used else "ok")

    def _estimate_intercept_all(
        self,
        L1,
        y,
        z1,
        z2,
        intercept_chisq_threshold=None,
        chisq_mode="either",
    ):
        """
        Estimate cross-trait intercept c for all jackknife replicates + full sample.

        This is the only place where the pairwise intercept chi^2 filter is applied.
        It does NOT alter the univariate RHS or the SCORE sufficient statistics.
        """
        L = np.asarray(L1, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64).ravel()

        if L.ndim != 2:
            raise ValueError("L1 must be 2D (M, Kreg).")
        M, Kreg = L.shape
        if y.size != M:
            raise ValueError(f"y must have length M={M}, got {y.size}.")

        starts_u, ends_u, D = self._get_jackknife_unit_partition()
        starts_u = np.asarray(starts_u, dtype=np.int64)
        ends_u = np.asarray(ends_u, dtype=np.int64)
        D = np.asarray(D, dtype=np.float64, order="C")
        R, U = D.shape

        if starts_u.size != U or ends_u.size != U:
            raise RuntimeError("Unit bounds mismatch with delete matrix D.")
        if np.any(starts_u < 0) or np.any(ends_u < starts_u) or np.any(ends_u > M):
            raise RuntimeError("Invalid jackknife unit bounds for intercept regression.")

        # Pairwise chi^2 filter (intercept only)
        keep = self._make_intercept_keep_mask(
            z1=z1,
            z2=z2,
            threshold=intercept_chisq_threshold,
            chisq_mode=chisq_mode,
        )

        # Also require finite regression inputs
        finite_reg = np.isfinite(y) & np.isfinite(L).all(axis=1)
        if not finite_reg.all():
            keep &= finite_reg

        if hasattr(self, "_intercept_chisq_info") and (self._intercept_chisq_info is not None):
            self._intercept_chisq_info["n_removed_nonfinite_reg"] = int((~finite_reg).sum())
            self._intercept_chisq_info["n_kept"] = int(keep.sum())
            self._intercept_chisq_info["n_removed"] = int(M - keep.sum())

        info = getattr(self, "_intercept_chisq_info", None)
        if info is not None and info.get("threshold") is not None:
            tag = " (auto)" if info.get("threshold_mode") == "auto" else ""
            self.log._log(
                f"[rg:c] intercept chi^2 filter: threshold={info['threshold']:.3f}{tag}, "
                f"mode={info['chisq_mode']}, removed={info['n_removed']} SNPs, "
                f"kept={info['n_kept']}."
            )

        w = self._build_intercept_weights(L, keep)

        # Zero out inactive rows so 0 * NaN never contaminates sufficient stats
        L_use = np.where(keep[:, None], L, 0.0)
        y_use = np.where(keep, y, 0.0)
        wy = w * y_use

        # Full sufficient stats
        W_tot = float(w.sum())
        s_tot = L_use.T @ w
        Q_tot = (L_use * w[:, None]).T @ L_use
        t_tot = float(wy.sum())
        u_tot = L_use.T @ wy

        # Per-unit sufficient stats
        W_u = np.zeros(U, dtype=np.float64)
        s_u = np.zeros((U, Kreg), dtype=np.float64)
        Q_u_flat = np.zeros((U, Kreg * Kreg), dtype=np.float64)
        t_u = np.zeros(U, dtype=np.float64)
        u_u = np.zeros((U, Kreg), dtype=np.float64)

        for u in range(U):
            s = int(starts_u[u])
            e = int(ends_u[u])
            if e <= s:
                continue

            wu = w[s:e]
            if wu.sum() <= 0.0:
                continue

            Lu = L_use[s:e, :]
            wyu = wy[s:e]

            W_u[u] = float(wu.sum())
            s_u[u] = Lu.T @ wu
            Q_u_flat[u] = ((Lu * wu[:, None]).T @ Lu).reshape(-1)
            t_u[u] = float(wyu.sum())
            u_u[u] = Lu.T @ wyu

        # Replicate sufficient stats
        del_W = D @ W_u
        del_s = D @ s_u
        del_Q_flat = D @ Q_u_flat
        del_t = D @ t_u
        del_u = D @ u_u

        W_rep = W_tot - del_W
        s_rep = s_tot[None, :] - del_s
        Q_rep = Q_tot.reshape(1, -1) - del_Q_flat
        Q_rep = Q_rep.reshape(R, Kreg, Kreg)
        t_rep = t_tot - del_t
        u_rep = u_tot[None, :] - del_u

        # Global stabilization scale from full centered covariance
        C_full = Q_tot - np.outer(s_tot, s_tot) / W_tot
        C_full = 0.5 * (C_full + C_full.T)
        diag_full = np.diag(C_full).astype(np.float64)
        pos_diag_full = diag_full[np.isfinite(diag_full) & (diag_full > 0.0)]
        if pos_diag_full.size > 0:
            global_scale = float(np.median(pos_diag_full))
        else:
            trf = float(np.trace(C_full))
            global_scale = float(trf / max(Kreg, 1)) if (np.isfinite(trf) and trf > 0.0) else 1.0
        global_scale = max(global_scale, 1.0)

        # Solve full + replicate systems
        beta_full, full_status = self._solve_centered_wls_beta(
            W_tot, s_tot, Q_tot, t_tot, u_tot, global_scale
        )
        if not np.isfinite(beta_full).all():
            beta_full = self._solve_raw_wls_beta(W_tot, s_tot, Q_tot, t_tot, u_tot)
            full_status = "raw_fallback"

        beta_rep = np.full((R, Kreg + 1), np.nan, dtype=np.float64)
        n_ok = 0
        n_ridge = 0
        n_invalid = 0

        for r in range(R):
            br, status = self._solve_centered_wls_beta(
                W_rep[r],
                s_rep[r],
                Q_rep[r],
                t_rep[r],
                u_rep[r],
                global_scale,
            )
            beta_rep[r] = br
            if status == "ok":
                n_ok += 1
            elif status == "ridge":
                n_ridge += 1
            else:
                n_invalid += 1

        if getattr(self, "verbose", False):
            self.log._log(
                f"[rg:c] intercept regression: ok={n_ok}/{R}, ridge={n_ridge}/{R}, "
                f"invalid={n_invalid}/{R}, full_status={full_status}"
            )

        beta_all = np.vstack([beta_rep, beta_full[None, :]])  # (R+1, 1+Kreg)
        return beta_all[:, 0]

    def _compute_t1_all_from_units(self, y):
        """
        Construct replicate-level t1 = A^T y.

        In block-mode, use the historical blk_idx-based construction exactly,
        so behavior matches utils.compute_t1_all_jn.
        In chr/delete-d mode, use the generic delete-matrix construction.
        """
        A = np.asarray(self.tr.annot, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64).ravel()

        M, K = A.shape
        if y.size != M:
            raise ValueError(f"y must have length M={M}, got {y.size}.")
        if not np.isfinite(y).all():
            raise ValueError("Non-finite values in y for t1 construction.")

        jm = getattr(self.tr, "jackknife_mode", "block")

        # Exact backward-compatible path for ordinary block jackknife
        if jm == "block":
            if not hasattr(self.tr, "blk_idx"):
                raise RuntimeError("block-mode requires Trace.blk_idx for exact t1 construction.")
            return utils.compute_t1_all_jn(A, y, self.tr.blk_idx, self.nblks)

        # Generic chr/delete-d path
        starts_u, ends_u, D = self._get_jackknife_unit_partition()
        starts_u = np.asarray(starts_u, dtype=np.int64)
        ends_u = np.asarray(ends_u, dtype=np.int64)
        D = np.asarray(D, dtype=np.float64, order="C")
        R, U = D.shape

        if starts_u.size != U or ends_u.size != U:
            raise RuntimeError("Unit bounds mismatch with delete matrix D.")

        t1_full = A.T @ y
        t1_unit = np.zeros((U, K), dtype=np.float64)

        for u in range(U):
            s = int(starts_u[u])
            e = int(ends_u[u])
            if e <= s:
                continue
            t1_unit[u] = A[s:e, :].T @ y[s:e]

        del_t1 = D @ t1_unit
        t1_rep = t1_full[None, :] - del_t1

        t1_all = np.empty((R + 1, K), dtype=np.float64)
        t1_all[:R] = t1_rep
        t1_all[R] = t1_full
        return t1_all

    def _get_intercept_regression_ld(self):
        """
        Return the LD-score design matrix used for the cross-trait intercept regression.
        """
        l2_bins_score = np.asarray(self.tr.ldscores, dtype=np.float64, order="C")
        if l2_bins_score.ndim == 1:
            l2_bins_score = l2_bins_score.reshape(-1, 1)
        if l2_bins_score.ndim != 2:
            raise RuntimeError("Primary ldscores must be 1D or 2D.")

        M, K_est = l2_bins_score.shape

        Lreg = getattr(self.tr, "ldscores_reg", None)
        if Lreg is None:
            if K_est == 1:
                self.log._log(
                    "[rg] ldscores_reg not provided; using primary ldscores "
                    "(already 1D) for intercept regression."
                )
                return l2_bins_score

            raise RuntimeError(
                "Intercept regression requires 1D --ldscores-reg by default.\n"
                "Primary ldscores has multiple columns; provide --ldscores-reg "
                "or pass --collapse-reg-ld."
            )

        Lreg = np.asarray(Lreg, dtype=np.float64, order="C")
        if Lreg.ndim == 1:
            if Lreg.shape[0] != M:
                raise RuntimeError(f"ldscores_reg length {Lreg.shape[0]} != M={M}")
            return Lreg.reshape(M, 1)

        if Lreg.ndim != 2:
            raise RuntimeError("ldscores_reg must be a vector (M,) or matrix (M,K).")
        if Lreg.shape[0] != M:
            raise RuntimeError(f"ldscores_reg has M={Lreg.shape[0]} SNPs but Trace has M={M}")
        if Lreg.shape[1] == 1:
            return Lreg

        if not self.collapse_reg_ld:
            raise RuntimeError(
                f"--ldscores-reg must be 1D by default; got {Lreg.shape[1]} columns.\n"
                "Pass --collapse-reg-ld to collapse to total LD."
            )

        self.log._log(
            f"[rg] Collapsing {Lreg.shape[1]}-column ldscores_reg to total LD "
            "for intercept regression."
        )
        return np.sum(Lreg, axis=1, dtype=np.float64, keepdims=True)

    def _estimate_gamma_and_rg(self):
        z1 = np.asarray(self.sums[0].zscores, dtype=np.float64)
        z2 = np.asarray(self.sums[1].zscores, dtype=np.float64)
        if z1.shape != z2.shape:
            raise RuntimeError(f"Internal mismatch: z1 has shape {z1.shape}, z2 has shape {z2.shape}.")

        y = z1 * z2
        M = z1.size
        K_est = int(self.nbins)

        # Ensure Trace replicate-level SCORE caches exist
        if (getattr(self.tr, "nsnps_blk", None) is None) or (
            np.asarray(self.tr.nsnps_blk).shape[0] != (self.nblks + 1)
        ):
            _ = self.tr._calc_trace(float(self.nsamp[0]))
        nsnps_blk_est = np.asarray(self.tr.nsnps_blk, dtype=np.float64)

        # ---- intercept regression (with its own orthogonal chi^2 filter) ----
        L1 = self._get_intercept_regression_ld()
        if L1.shape[0] != M:
            raise RuntimeError(f"Intercept regression LD has {L1.shape[0]} SNPs, expected {M}.")

        self.c_opt = self._estimate_intercept_all(
            L1=L1,
            y=y,
            z1=z1,
            z2=z2,
            intercept_chisq_threshold=self.intercept_chisq_thr,
            chisq_mode="either",
        )

        # ---- SCORE normal-equation pieces (no intercept-specific chi^2 filter here) ----
        ld_sum_all = np.asarray(self.tr.get_ldsum_all(use_cache=True), dtype=np.float64, order="C")
        expected = (self.nblks + 1, self.nbins, self.nbins)
        if ld_sum_all.shape != expected:
            raise RuntimeError(
                f"Trace.get_ldsum_all returned shape {ld_sum_all.shape}, expected {expected}."
            )

        t1_all = self._compute_t1_all_from_units(y)

        self.gamma_g = utils.solve_score_gamma_from_intercept_jn(
            ld_sum_all=ld_sum_all,
            t1_all=t1_all,
            nsnps_blk=nsnps_blk_est,
            c_all=self.c_opt,
            n1=self.nsamp[0],
            n2=self.nsamp[1],
            ridge_rel=getattr(self, "ridge_rel", 0.0),
        )

        # gamma SE
        _, se = self._jk_se(self.gamma_g, axis=0)
        self.gamma_se = se

        # per-bin rg
        v1 = self.sigmas[0, :, :K_est]
        v2 = self.sigmas[1, :, :K_est]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rg = self.gamma_g / np.sqrt(v1 * v2)
        self.rg[~np.isfinite(self.rg)] = np.nan

        _, se_rg = self._jk_se(self.rg, axis=0)
        self.rg_se = se_rg

        # single-component override so per-bin and total rg SE agree under delta mode
        if self.rg_se_method == "delta" and K_est == 1:
            rg_full_delta, rg_se_delta = self._estimate_total_rg_delta_se()
            if np.isfinite(rg_full_delta):
                self.rg[-1, 0] = rg_full_delta
                self.rg_se[0] = rg_se_delta

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
        has_ov = utils._has_overlapping_annotations(A)

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

        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] "
            f"enrichment_mode_used: {mode_used}"
        )
        if self.verbose and req in ("auto", "both"):
            self.log._log(
                f"[enrichment] enrich_mode={req} resolved={mode_used} has_overlap={has_ov}"
            )

        jk_w, jk_use_pv = self._get_jackknife_weights()

        # sigma_g^2 SEs from jackknife
        sigma_se = np.full((2, K), np.nan, dtype=np.float64)
        for t in range(2):
            _, se = self._jk_se(
                self.sigmas[t, :, :K],
                axis=0,
                center=self.jack_mode,
                nan_policy=self.nan_policy,
                weights=jk_w,
                use_pseudovalues=jk_use_pv,
            )
            sigma_se[t] = se

        # per-trait blocks
        for t in range(2):
            sigma_full = self.sigmas[t, -1, :K]
            h2_full = self.hersums[t, :K, 0]
            h2_se = self.hersums[t, :K, 1]
            h2_tot = float(self.hersums[t, -1, 0])
            h2_tot_se = float(self.hersums[t, -1, 1])

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
                    enr_str = (
                        f"Enrichment_nonoverlap: {enr_no_full[j]:.6g} "
                        f"(SE: {enr_no_se[j]:.6g}) "
                        f"& Enrichment_overlap: {enr_ov_full[j]:.6g} "
                        f"(SE: {enr_ov_se[j]:.6g})"
                    )

                self.log._log(
                    f"^^^ Phenotype [{self.names[t]}] Bin [{header}] "
                    f"sigma_g^2: {sigma_full[j]:.6g} (SE: {sigma_se[t, j]:.6g}) "
                    f"h^2_cat: {h2_full[j]:.6g} (SE: {h2_se[j]:.6g}) "
                    + enr_str
                )

            self.log._log(
                f"^^^ Phenotype [{self.names[t]}] Total SNP heritability (h^2): "
                f"{h2_tot:.6g} SE: {h2_tot_se:.6g}"
            )

        # cross-trait intercept
        c_full, c_se = self._jk_se(
            self.c_opt,
            axis=0,
            center=self.jack_mode,
            nan_policy=self.nan_policy,
            weights=jk_w,
            use_pseudovalues=jk_use_pv,
        )
        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] "
            f"Intercept (c): {float(c_full):.6g} (SE: {float(c_se):.6g})"
        )

        for j, header in enumerate(self.annot_header):
            self.log._log(
                f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] Bin [{header}] "
                f"gamma_g: {self.gamma_g[-1, j]:.6g} (SE: {self.gamma_se[j]:.6g}) "
                f"rg: {self.rg[-1, j]:.6g} (SE: {self.rg_se[j]:.6g})"
            )

        # Totals
        gamma_tot = np.nansum(self.gamma_g, axis=1)  # (B+1,)
        g_full, g_se = self._jk_se(
            gamma_tot,
            axis=0,
            center=self.jack_mode,
            nan_policy=self.nan_policy,
            weights=jk_w,
            use_pseudovalues=jk_use_pv,
        )

        if self.rg_se_method == "delta":
            r_full, r_se = self._estimate_total_rg_delta_se()
        else:
            h2_0 = np.asarray(self.herits[0, :, -1], dtype=np.float64)
            h2_1 = np.asarray(self.herits[1, :, -1], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                rg_tot = gamma_tot / np.sqrt(h2_0 * h2_1)
            rg_tot[~np.isfinite(rg_tot)] = np.nan

            r_full, r_se = self._jk_se(
                rg_tot,
                axis=0,
                center=self.jack_mode,
                nan_policy=self.nan_policy,
                weights=jk_w,
                use_pseudovalues=jk_use_pv,
            )

        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] "
            f"Total genetic covariance (gamma_g): {float(g_full):.6g} "
            f"(SE: {float(g_se):.6g})"
        )
        self.log._log(
            f"^^^ Phenotype [{self.names[0]}] & [{self.names[1]}] "
            f"Total genetic correlation (rg): {float(r_full):.6g} "
            f"(SE: {float(r_se):.6g})"
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
