from sumstats import Sumstats
from trace import Trace
import utils

import numpy as np
import os
import sys

class Sumrhe:
    def __init__(self, bim_path=None, sum_path=None, save_path=None, h2_path=None, out=None, chisq_threshold=0,
            log=None, mem=False, verbose=False, ldscores=None, njack=None, annot=None, chisq_action='drop',
            report_tau: bool = True, allow_neg_enr: bool = False, clip_nonfinite_vals: bool = False,
            adjust_delta: bool = False, enrich_mode: str = "auto", jack_mode: str = "mean",
            jackknife_weighted: bool = True):

        self.mem = mem
        self.log = log
        self.start_time = utils._get_time()
        self.log._log("Analysis started at: "+utils._get_timestr(self.start_time))

        self.tr = Trace(
            bimpath=bim_path, sumpath=sum_path, savepath=save_path,
            ldscores=ldscores, log=self.log, nblks=njack, annot=annot,
            verbose=verbose, adjust_delta=adjust_delta
        )

        self.nblks = self.tr.nblks  # replicate count R
        self.jack_mode = jack_mode
        self.annot_header = self.tr.annot_header
        self.nbins = self.tr.nbins

        self._jackknife_weighted = bool(jackknife_weighted)
        self._jk_mode = getattr(self.tr, "jackknife_mode", "block")
        self._jk_delete = int(getattr(self.tr, "jackknife_delete", 1))
        self._jk_units = int(getattr(self.tr, "jackknife_units", self.nblks))

        self.sums = Sumstats(
            nblks=self.nblks,
            chisq_threshold=chisq_threshold,
            log=self.log,
            annot_df=self.tr.annot_df,
            nbins=self.nbins,
            chisq_action=chisq_action,
            jackknife_mode=self._jk_mode,
            jackknife_chrs=getattr(self.tr, "jackknife_chrs", None),
            jackknife_delete=self._jk_delete,
            jackknife_delete_sets=getattr(self.tr, "jackknife_delete_sets", None),
            jackknife_seed=getattr(self.tr, "jackknife_seed", None),
        )

        try:
            self.h2_dir = utils._parse_sumdir(h2_path)
        except ValueError as e:
            self.log._log(f"Error reading sumstat files: {e}")
            sys.exit(1)

        self.npheno = len(self.h2_dir)
        self.nsamp = []
        self.phen_names = [os.path.basename(name)[:-8] for name in self.h2_dir]

        self.sigmas = np.zeros((self.npheno, self.nblks+1, self.nbins+2))
        self.sigsums = np.zeros((self.npheno, self.nbins+2, 2))

        self.herits  = np.zeros((self.npheno, self.nblks+1, self.nbins+1))
        self.hersums = np.zeros((self.npheno, self.nbins+1, 2))

        self.enrich_mode = utils._normalize_enrich_mode(enrich_mode)
        self._enrich_mode_used = [""] * self.npheno

        self.enrich = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
        self.enrich_sums = np.zeros((self.npheno, self.nbins, 2), dtype=np.float64)

        self.enrich_overlap = None
        self.enrich_overlap_sums = None
        self.enrich_nonoverlap = None
        self.enrich_nonoverlap_sums = None

        if self.enrich_mode == "both":
            self.enrich_overlap = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
            self.enrich_overlap_sums = np.zeros((self.npheno, self.nbins, 2), dtype=np.float64)
            self.enrich_nonoverlap = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
            self.enrich_nonoverlap_sums = np.zeros((self.npheno, self.nbins, 2), dtype=np.float64)

        self.out = out
        self.verbose = verbose
        self.report_tau = bool(report_tau)
        self.allow_neg_enr = bool(allow_neg_enr)
        self.clip_nonfinite_vals = clip_nonfinite_vals
        self.nan_policy = "propagate" if self.clip_nonfinite_vals else "omit"
        self.adjust_delta = adjust_delta

        if self.report_tau:
            self.tau         = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
            self.tau_star    = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
            self.tau_sums    = np.zeros((self.npheno, self.nbins, 2), dtype=np.float64)
            self.tau_star_sums = np.zeros((self.npheno, self.nbins, 2), dtype=np.float64)

    def _calc_sigmas(self, idx):
        rhs = self.sums.rhs                              # (nblks+1, p)
        pred_tr = self.tr._calc_trace(self.nsamp[idx])       # (nblks+1, p, p)

        sig_est = utils._solve_linear_equation(pred_tr, rhs) # (nblks+1, p)

        # store [sigma_g1^2,...,sigma_gK^2, sigma_e^2, sum_g]
        self.sigmas[idx, :, :self.nbins+1] = sig_est
        self.sigmas[idx, :, -1] = sig_est[:, :self.nbins].sum(axis=1)

        if self.verbose:
            names = [f"sigma^2_g{t}" for t in range(self.nbins)] + ["sigma^2_e"]
            self.log._log("Normal equation:\n"+np.array2string(pred_tr[self.nblks], precision=2, separator=', ')+"\n\t\t*\n"\
                +"["+", ".join(names)+"]\n\t\t=\n"+np.array2string(rhs[self.nblks], precision=2, separator=', '))
            self.log._log("Sigma solution:\n"+np.array2string(self.sigmas[idx, self.nblks, :self.nbins], precision=3, separator=', '))
        return self.sigmas[idx]

    def _calc_h2(self, idx):
        """
        Vectorized, filter-aware h^2 calculation.
        Supports:
        - block/LOO (contiguous chunk jackknife)
        - chr-mode delete-1 LOCO
        - chr-mode delete-d LOCO (including random subsets)

        For continuous annotations:
        Ak = sum_j a_{j,k} (weight mass), not necessarily integer SNP count.
        """
        import numpy as np

        A = self.tr.annot
        M, K = A.shape
        B = int(self.nblks)  # number of replicates (R in chr delete-d)
        if np.nanmin(A) < 0:
            self.log._log("[WARNING] Detected negative annotation weights; h2/enrichment/tau may be ill-defined.")

        # ---------- chr-mode (delete-1 or delete-d) ----------
        if getattr(self.tr, "jackknife_mode", "block") == "chr":
            cache = self._get_chr_unit_cache(A, need_overlap=True, need_Ak2=False)
            D = cache["D"]                     # (B,U)
            Ak_full = cache["Ak_full"]         # (K,)
            Ak_unit = cache["Ak_unit"]         # (U,K)
            ov_full = cache["ov_full"]         # (K,K)
            ov_unit_flat = cache["ov_unit_flat"]  # (U,K*K)

            # replicate remaining masses: Ak_minus = Ak_full - D@Ak_unit
            del_Ak = (D @ Ak_unit).astype(np.float64, copy=False)     # (B,K)
            Ak_minus = Ak_full[None, :].astype(np.float64) - del_Ak   # (B,K)

            # replicate remaining overlaps: ov_minus = ov_full - D@ov_unit
            full_flat = ov_full.reshape(-1).astype(np.float64, copy=False)  # (K*K,)
            del_ov_flat = (D @ ov_unit_flat).astype(np.float64, copy=False) # (B,K*K)
            ov_minus = (full_flat[None, :] - del_ov_flat).reshape(B, K, K)  # (B,K,K)

            # weights per replicate: w_bk = sigma_g[b,k] / Ak_minus[b,k]
            sigma_g_minus = self.sigmas[idx, :B, :K].astype(np.float64, copy=False)  # (B,K)
            with np.errstate(divide="ignore", invalid="ignore"):
                w = sigma_g_minus / Ak_minus  # (B,K)

            # h2_cat_minus[b,c] = sum_k ov_minus[b,c,k] * w[b,k]
            h2_cat_minus = np.einsum("bck,bk->bc", ov_minus, w, optimize=True)

            bad = ~np.isfinite(h2_cat_minus)
            if self.clip_nonfinite_vals:
                h2_cat_minus[bad] = 0.0
            else:
                h2_cat_minus[bad] = np.nan

            self.herits[idx, :B, :K] = h2_cat_minus

            # full-sample
            sigma_g_full = self.sigmas[idx, B, :K].astype(np.float64, copy=False)  # (K,)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_full = sigma_g_full / Ak_full.astype(np.float64, copy=False)     # (K,)
                h2_cat_full = ov_full.astype(np.float64, copy=False) @ w_full      # (K,)

            bad_full = ~np.isfinite(h2_cat_full)
            if self.clip_nonfinite_vals:
                h2_cat_full[bad_full] = 0.0
            else:
                h2_cat_full[bad_full] = np.nan

            self.herits[idx, B, :K] = h2_cat_full

            # total h2 convention: sum sigma_g
            self.herits[idx, :, -1] = self.sigmas[idx, :, :K].sum(axis=1)
            return

        # ---------- block-mode (legacy contiguous LOO) ----------
        A32 = np.asarray(A, dtype=np.float32, order='C')
        overlap_full = A32.T @ A32
        Ak_full = A32.sum(axis=0)

        if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            starts = self.tr._blk_starts
            ends = self.tr._blk_ends
        else:
            bs = self.tr.blk_size
            starts = bs * np.arange(B)
            ends = starts + bs
            ends[-1] = self.tr.nsnps

        overlap_blk = np.empty((B, K, K), dtype=np.float32)
        Ak_blk = np.empty((B, K), dtype=np.float32)
        for b in range(B):
            ab = A32[int(starts[b]):int(ends[b]), :]
            overlap_blk[b] = ab.T @ ab
            Ak_blk[b] = ab.sum(axis=0)

        overlap_minus = overlap_full[None, :, :] - overlap_blk
        Ak_minus = Ak_full[None, :] - Ak_blk

        sigma_g_minus = self.sigmas[idx, :B, :K].astype(np.float64, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = sigma_g_minus / Ak_minus.astype(np.float64, copy=False)
        h2_cat_minus = np.einsum("bck,bk->bc", overlap_minus.astype(np.float64, copy=False), w, optimize=True)

        bad_minus = ~np.isfinite(h2_cat_minus)
        if self.clip_nonfinite_vals:
            h2_cat_minus[bad_minus] = 0.0
        else:
            h2_cat_minus[bad_minus] = np.nan

        self.herits[idx, :B, :K] = h2_cat_minus

        sigma_g_full = self.sigmas[idx, B, :K].astype(np.float64, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_full = sigma_g_full / Ak_full.astype(np.float64, copy=False)
            h2_cat_full = overlap_full.astype(np.float64, copy=False) @ w_full

        bad_full = ~np.isfinite(h2_cat_full)
        if self.clip_nonfinite_vals:
            h2_cat_full[bad_full] = 0.0
        else:
            h2_cat_full[bad_full] = np.nan

        self.herits[idx, B, :K] = h2_cat_full
        self.herits[idx, :, -1] = self.sigmas[idx, :, :K].sum(axis=1)

    
    def _calc_enrich(self, idx):
        """
        Enrichment = (h2_cat / h2_tot) / prop, where prop[k] = Ak_rep[k] / M_rep.

        Supports chr-mode delete-d: replicate masses built via D @ per-chr stats.
        """
        import numpy as np
        import utils

        A = np.asarray(self.tr.annot, dtype=np.float64, order='C')
        M_full, K = A.shape
        B = int(self.nblks)

        if np.nanmin(A) < 0:
            self.log._log("[WARNING] Detected negative annotation weights; enrichment may be ill-defined.")

        # ---------- chr-mode ----------
        if getattr(self.tr, "jackknife_mode", "block") == "chr":
            cache = self._get_chr_unit_cache(A, need_overlap=False, need_Ak2=False)
            D = cache["D"].astype(np.float64, copy=False)      # (B,U)
            Ak_full = cache["Ak_full"].astype(np.float64)      # (K,)
            Ak_unit = cache["Ak_unit"].astype(np.float64)      # (U,K)
            m_unit  = cache["m_unit"].astype(np.float64)       # (U,)

            del_Ak = D @ Ak_unit                               # (B,K)
            Ak_rep = np.empty((B + 1, K), dtype=np.float64)
            Ak_rep[:B] = Ak_full[None, :] - del_Ak
            Ak_rep[B]  = Ak_full

            del_m = D @ m_unit                                 # (B,)
            M_rep = np.empty(B + 1, dtype=np.float64)
            M_rep[:B] = float(M_full) - del_m
            M_rep[B]  = float(M_full)

            with np.errstate(divide='ignore', invalid='ignore'):
                prop = Ak_rep / M_rep[:, None]

        else:
            # ---------- block-mode (legacy) ----------
            if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
                starts = self.tr._blk_starts
                ends   = self.tr._blk_ends
            else:
                bs = self.tr.blk_size
                starts = bs * np.arange(B)
                ends   = starts + bs
                ends[-1] = self.tr.nsnps

            Ak_full = A.sum(axis=0)
            Ak_blk = np.empty((B, K), dtype=np.float64)
            m_blk = np.empty(B, dtype=np.int64)
            for b in range(B):
                ab = A[int(starts[b]):int(ends[b]), :]
                Ak_blk[b] = ab.sum(axis=0)
                m_blk[b] = ab.shape[0]

            Ak_rep = np.empty((B + 1, K), dtype=np.float64)
            Ak_rep[:B] = Ak_full[None, :] - Ak_blk
            Ak_rep[B]  = Ak_full

            M_rep = np.empty(B + 1, dtype=np.float64)
            M_rep[:B] = float(M_full) - m_blk
            M_rep[B]  = float(M_full)

            with np.errstate(divide='ignore', invalid='ignore'):
                prop = Ak_rep / M_rep[:, None]

        # Total h2 (your convention)
        h2_tot = self.herits[idx, :, -1].astype(np.float64, copy=False)

        requested = self.enrich_mode
        has_ov = utils._has_overlapping_annotations(A)

        if requested == "auto":
            mode_used = "overlap" if has_ov else "non-overlap"
            modes_to_compute = (mode_used,)
        elif requested == "both":
            mode_used = "overlap" if has_ov else "non-overlap"
            modes_to_compute = ("non-overlap", "overlap")
        else:
            mode_used = requested
            modes_to_compute = (requested,)

        self._enrich_mode_used[idx] = mode_used
        if self.verbose and requested in ("auto", "both"):
            self.log._log(f"[enrichment] phenotype={idx} enrich_mode={requested} resolved={mode_used} has_overlap={has_ov}")

        def _compute_enr_from_h2cat(h2_cat: np.ndarray) -> np.ndarray:
            with np.errstate(divide='ignore', invalid='ignore'):
                enr = (h2_cat / h2_tot[:, None]) / prop
                invalid = (~np.isfinite(enr)) | (~np.isfinite(prop)) | (prop <= 0.0)
                if not self.allow_neg_enr:
                    invalid |= (h2_tot[:, None] <= 0.0)
                enr[invalid] = np.nan
            return enr

        enr_ov = None
        enr_no = None

        if "overlap" in modes_to_compute:
            h2_cat_ov = self.herits[idx, :, :K].astype(np.float64, copy=False)
            enr_ov = _compute_enr_from_h2cat(h2_cat_ov)

        if "non-overlap" in modes_to_compute:
            h2_cat_no = self.sigmas[idx, :, :K].astype(np.float64, copy=False)
            enr_no = _compute_enr_from_h2cat(h2_cat_no)

        if requested == "both":
            self.enrich_overlap[idx] = enr_ov
            self.enrich_nonoverlap[idx] = enr_no
            self.enrich[idx] = enr_ov if mode_used == "overlap" else enr_no
        else:
            self.enrich[idx] = enr_ov if mode_used == "overlap" else enr_no
      
    def _calc_tau(self, idx):
        """
        Compute LDSC τ_k and τ*_k across all jackknife replicates (B delete-* + full).

        chr-mode delete-d:
        Ak_rep  = Ak_full  - D @ Ak_unit
        Ak2_rep = Ak2_full - D @ Ak2_unit
        M_rep   = M_full   - D @ m_unit
        """
        import numpy as np

        A = np.asarray(self.tr.annot, dtype=np.float64, order='C')
        M_full, K = A.shape
        B = int(self.nblks)

        if np.nanmin(A) < 0:
            self.log._log("[WARNING] Detected negative annotation weights; tau/tau* may be ill-defined.")

        # ---------- chr-mode ----------
        if getattr(self.tr, "jackknife_mode", "block") == "chr":
            cache = self._get_chr_unit_cache(A, need_overlap=False, need_Ak2=True)
            D = cache["D"].astype(np.float64, copy=False)          # (B,U)
            Ak_full  = cache["Ak_full"].astype(np.float64)         # (K,)
            Ak_unit  = cache["Ak_unit"].astype(np.float64)         # (U,K)
            Ak2_full = cache["Ak2_full"].astype(np.float64)        # (K,)
            Ak2_unit = cache["Ak2_unit"].astype(np.float64)        # (U,K)
            m_unit   = cache["m_unit"].astype(np.float64)          # (U,)

            del_Ak  = D @ Ak_unit                                  # (B,K)
            del_Ak2 = D @ Ak2_unit                                 # (B,K)

            Ak_rep = np.empty((B + 1, K), dtype=np.float64)
            Ak2_rep = np.empty((B + 1, K), dtype=np.float64)
            Ak_rep[:B]  = Ak_full[None, :]  - del_Ak
            Ak2_rep[:B] = Ak2_full[None, :] - del_Ak2
            Ak_rep[B]   = Ak_full
            Ak2_rep[B]  = Ak2_full

            del_m = D @ m_unit                                     # (B,)
            M_rep = np.empty(B + 1, dtype=np.float64)
            M_rep[:B] = float(M_full) - del_m
            M_rep[B]  = float(M_full)

        else:
            # ---------- block-mode (legacy) ----------
            if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
                starts = self.tr._blk_starts
                ends   = self.tr._blk_ends
            else:
                bs = self.tr.blk_size
                starts = bs * np.arange(B)
                ends   = starts + bs
                ends[-1] = self.tr.nsnps

            Ak_full  = A.sum(axis=0)
            Ak2_full = (A * A).sum(axis=0)

            Ak_blk  = np.empty((B, K), dtype=np.float64)
            Ak2_blk = np.empty((B, K), dtype=np.float64)
            m_blk   = np.empty(B, dtype=np.int64)
            for b in range(B):
                ab = A[int(starts[b]):int(ends[b]), :]
                Ak_blk[b]  = ab.sum(axis=0)
                Ak2_blk[b] = (ab * ab).sum(axis=0)
                m_blk[b]   = ab.shape[0]

            M_rep = np.empty(B + 1, dtype=np.float64)
            M_rep[:B] = float(M_full) - m_blk
            M_rep[B]  = float(M_full)

            Ak_rep  = np.empty((B + 1, K), dtype=np.float64)
            Ak2_rep = np.empty((B + 1, K), dtype=np.float64)
            Ak_rep[:B]  = Ak_full[None, :]  - Ak_blk
            Ak2_rep[:B] = Ak2_full[None, :] - Ak2_blk
            Ak_rep[B]   = Ak_full
            Ak2_rep[B]  = Ak2_full

        sigma_g_rep = self.sigmas[idx, :, :K].astype(np.float64, copy=False)  # (B+1,K)
        h2_tot_rep  = self.herits[idx, :, -1].astype(np.float64, copy=False)  # (B+1,)

        # τ
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = sigma_g_rep / Ak_rep

        bad_tau = ~np.isfinite(tau)
        if self.clip_nonfinite_vals:
            tau[bad_tau] = 0.0
        else:
            tau[bad_tau] = np.nan

        # τ*
        with np.errstate(divide='ignore', invalid='ignore'):
            meanA  = Ak_rep / M_rep[:, None]
            meanA2 = Ak2_rep / M_rep[:, None]
            varA   = np.maximum(meanA2 - meanA * meanA, 0.0)
            sdA    = np.sqrt(varA, dtype=np.float64)

            denom = h2_tot_rep / M_rep
            tau_star = tau * (sdA / denom[:, None])

        bad_ts = ~np.isfinite(tau_star)
        if self.clip_nonfinite_vals:
            tau_star[bad_ts] = 0.0
        else:
            tau_star[bad_ts] = np.nan

        self.tau[idx]      = tau
        self.tau_star[idx] = tau_star


    def _run_jackknife(self, idx):
        import numpy as np
        import utils

        R = int(self.nblks)

        # ------------------------------------------------------------
        # chr-mode: unified LOCO/delete-d via unit-level pseudovalues
        # ------------------------------------------------------------
        if getattr(self.tr, "jackknife_mode", "block") == "chr":
            D = self.tr.get_jackknife_delete_matrix(dtype=np.float64)  # (R,U)
            unit_sizes = self.tr.get_jackknife_unit_sizes(dtype=np.float64)  # (U,)

            # Sigma components
            est_full, se = utils._calc_jackknife_se_from_delete_sets(
                self.sigmas[idx], D=D, unit_sizes=unit_sizes,
                axis=0, center=self.jack_mode, nan_policy=self.nan_policy
            )
            self.sigsums[idx, :, 0] = est_full
            self.sigsums[idx, :, 1] = se

            # Heritabilities
            est_full, se = utils._calc_jackknife_se_from_delete_sets(
                self.herits[idx], D=D, unit_sizes=unit_sizes,
                axis=0, center=self.jack_mode, nan_policy=self.nan_policy
            )
            self.hersums[idx, :, 0] = est_full
            self.hersums[idx, :, 1] = se

            # Enrichment
            est_full, se = utils._calc_jackknife_se_from_delete_sets(
                self.enrich[idx], D=D, unit_sizes=unit_sizes,
                axis=0, center=self.jack_mode, nan_policy=self.nan_policy
            )
            self.enrich_sums[idx, :, 0] = est_full
            self.enrich_sums[idx, :, 1] = se

            # Optional enrichment outputs (both-mode)
            if self.enrich_overlap is not None:
                est_full, se = utils._calc_jackknife_se_from_delete_sets(
                    self.enrich_overlap[idx], D=D, unit_sizes=unit_sizes,
                    axis=0, center=self.jack_mode, nan_policy=self.nan_policy
                )
                self.enrich_overlap_sums[idx, :, 0] = est_full
                self.enrich_overlap_sums[idx, :, 1] = se

            if self.enrich_nonoverlap is not None:
                est_full, se = utils._calc_jackknife_se_from_delete_sets(
                    self.enrich_nonoverlap[idx], D=D, unit_sizes=unit_sizes,
                    axis=0, center=self.jack_mode, nan_policy=self.nan_policy
                )
                self.enrich_nonoverlap_sums[idx, :, 0] = est_full
                self.enrich_nonoverlap_sums[idx, :, 1] = se

            # τ / τ* if requested
            if self.report_tau:
                est_full, se = utils._calc_jackknife_se_from_delete_sets(
                    self.tau[idx], D=D, unit_sizes=unit_sizes,
                    axis=0, center=self.jack_mode, nan_policy=self.nan_policy
                )
                self.tau_sums[idx, :, 0] = est_full
                self.tau_sums[idx, :, 1] = se

                est_full, se = utils._calc_jackknife_se_from_delete_sets(
                    self.tau_star[idx], D=D, unit_sizes=unit_sizes,
                    axis=0, center=self.jack_mode, nan_policy=self.nan_policy
                )
                self.tau_star_sums[idx, :, 0] = est_full
                self.tau_star_sums[idx, :, 1] = se

            return

        # ------------------------------------------------------------
        # block-mode: keep your existing (weighted) pseudovalue jackknife
        # ------------------------------------------------------------
        weights = None
        use_pv = False
        if self._jackknife_weighted and hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            m = (np.asarray(self.tr._blk_ends, dtype=np.float64) - np.asarray(self.tr._blk_starts, dtype=np.float64))
            weights = m
            use_pv = True

        est_full, se = utils._calc_jackknife_se(
            self.sigmas[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
            weights=weights, use_pseudovalues=use_pv
        )
        self.sigsums[idx, :, 0] = est_full
        self.sigsums[idx, :, 1] = se

        est_full, se = utils._calc_jackknife_se(
            self.herits[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
            weights=weights, use_pseudovalues=use_pv
        )
        self.hersums[idx, :, 0] = est_full
        self.hersums[idx, :, 1] = se

        est_full, se = utils._calc_jackknife_se(
            self.enrich[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
            weights=weights, use_pseudovalues=use_pv
        )
        self.enrich_sums[idx, :, 0] = est_full
        self.enrich_sums[idx, :, 1] = se

        if self.enrich_overlap is not None:
            est_full, se = utils._calc_jackknife_se(
                self.enrich_overlap[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
                weights=weights, use_pseudovalues=use_pv
            )
            self.enrich_overlap_sums[idx, :, 0] = est_full
            self.enrich_overlap_sums[idx, :, 1] = se

        if self.enrich_nonoverlap is not None:
            est_full, se = utils._calc_jackknife_se(
                self.enrich_nonoverlap[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
                weights=weights, use_pseudovalues=use_pv
            )
            self.enrich_nonoverlap_sums[idx, :, 0] = est_full
            self.enrich_nonoverlap_sums[idx, :, 1] = se

        if self.report_tau:
            est_full, se = utils._calc_jackknife_se(
                self.tau[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
                weights=weights, use_pseudovalues=use_pv
            )
            self.tau_sums[idx, :, 0] = est_full
            self.tau_sums[idx, :, 1] = se

            est_full, se = utils._calc_jackknife_se(
                self.tau_star[idx], axis=0, center=self.jack_mode, nan_policy=self.nan_policy,
                weights=weights, use_pseudovalues=use_pv
            )
            self.tau_star_sums[idx, :, 0] = est_full
            self.tau_star_sums[idx, :, 1] = se

    def _run(self):
        for i in range(self.npheno):
            h2_path = self.h2_dir[i]
            removesnps = self.sums._process(h2_path, self.phen_names[i])

            # report chi^2 distribution diagnostics (report-only; no estimator changes)
            if hasattr(self.sums, "log_chisq_diagnostics"):
                self.sums.log_chisq_diagnostics(
                    topk=10,
                    warn_min_count=10,
                    warn_min_frac=1e-5,
                    verbose=bool(self.verbose),
                    include_read_when_verbose=True,
                )


            self.tr._filter_snps(removesnps)  # always try to filter both sides
            self.nsamp.append(self.sums.nsamp)

            self._calc_sigmas(i)
            self._calc_h2(i)
            if self.report_tau:
                self._calc_tau(i)
            self._calc_enrich(i)
            self._run_jackknife(i)

    
    def _logoff(self):
        for i in range(self.npheno):
            if self._enrich_mode_used[i]:
                self.log._log(f"^^^ Phenotype {i} enrichment_mode_used: {self._enrich_mode_used[i]}")

            if (self.nbins > 1):
                for j in range(self.nbins):
                    sig   = self.sigsums[i, j, 0]
                    sigse = self.sigsums[i, j, 1]
                    h2    = self.hersums[i, j, 0]
                    h2se  = self.hersums[i, j, 1]
                    enr   = self.enrich_sums[i, j, 0]
                    ense  = self.enrich_sums[i, j, 1]

                    # If both-mode was requested, report both enrichments explicitly
                    enr_ov = ense_ov = None
                    enr_no = ense_no = None
                    if self.enrich_overlap_sums is not None:
                        enr_ov  = self.enrich_overlap_sums[i, j, 0]
                        ense_ov = self.enrich_overlap_sums[i, j, 1]
                    if self.enrich_nonoverlap_sums is not None:
                        enr_no  = self.enrich_nonoverlap_sums[i, j, 0]
                        ense_no = self.enrich_nonoverlap_sums[i, j, 1]

                    line = (f"^^^ Phenotype {i} Bin [{self.annot_header[j]}] "
                            f"sigma_g^2: {sig:.5f} (SE: {sigse:.5f}) "
                            f"h^2_cat: {h2:.5f} (SE: {h2se:.5f}) "
                            f"Enrichment: {enr:.5f} (SE: {ense:.5f})")

                    if (enr_no is not None) and (enr_ov is not None):
                        line += (f" Enrichment_nonoverlap: {enr_no:.5f} (SE: {ense_no:.5f})"
                                 f" Enrichment_overlap: {enr_ov:.5f} (SE: {ense_ov:.5f})")

                    if self.report_tau:
                        t    = self.tau_sums[i, j, 0]
                        tse  = self.tau_sums[i, j, 1]
                        ts   = self.tau_star_sums[i, j, 0]
                        tsse = self.tau_star_sums[i, j, 1]
                        line += f" tau: {t:.6g} (SE: {tse:.6g}) tau_*: {ts:.6g} (SE: {tsse:.6g})"
                    self.log._log(line)

            # total SNP h2
            h2tot   = self.hersums[i, -1, 0]
            h2totse = self.hersums[i, -1, 1]
            self.log._log("^^^ Phenotype "+str(i)+" Total SNP heritability (h^2): "
                        +format(h2tot, '.5f')+" SE: "+format(h2totse, '.5f'))

        self.end_time = utils._get_time()
        self.log._log("Analysis ended at: "+utils._get_timestr(self.end_time))
        self.log._log("run time: "+format(self.end_time - self.start_time, '.3f')+" s")
        if (self.out is not None):
            self.log._log("Saved log in "+ self.out + ".log")
            self.log._save_log(self.out+".log")
        return

    def _get_chr_unit_cache(self, A, need_overlap: bool = False, need_Ak2: bool = False):
        """
        Build/cache chromosome-unit sufficient statistics for chr-mode (delete-1 or delete-d).

        Returns a dict with:
        D:          (R, U) float32 deletion incidence matrix (replicate r deletes units where D[r,u]=1)
        starts/ends:(U,) int64 chromosome bounds in the CURRENT SNP order
        m_unit:     (U,) float64 SNP counts per chromosome unit
        Ak_unit:    (U, K) float32 sum of weights per bin within unit
        Ak_full:    (K,) float32
        (optional) Ak2_unit/Ak2_full
        (optional) ov_unit_flat: (U, K*K) float32  with ov_u = A_u^T A_u flattened
        (optional) ov_full:      (K, K) float32
        """
        import numpy as np

        if getattr(self.tr, "jackknife_mode", "block") != "chr":
            return None

        K = A.shape[1]
        R = int(self.nblks)

        # deletion matrix D: (R,U)
        D = self.tr.get_jackknife_delete_matrix(dtype=np.float32)
        if D is None:
            raise RuntimeError("chr-mode requested but Trace did not provide a delete matrix (D).")
        if D.shape[0] != R:
            raise RuntimeError(f"Delete matrix has R={D.shape[0]} rows but self.nblks={R}.")
        U = int(D.shape[1])

        # cache key (same A object across h2/enrich/tau within phenotype)
        key = (id(A), A.shape, int(need_overlap), int(need_Ak2))
        cache = getattr(self, "_chr_unit_cache", None)
        if cache is not None and cache.get("_key", None) == key:
            return cache

        A32 = np.asarray(A, dtype=np.float32, order="C")

        # unit bounds: prefer explicit unit bounds; otherwise reconstruct from chr labels
        if hasattr(self.tr, "_unit_starts") and hasattr(self.tr, "_unit_ends"):
            starts = np.asarray(self.tr._unit_starts, dtype=np.int64)
            ends   = np.asarray(self.tr._unit_ends,   dtype=np.int64)
        else:
            # fallback: use blk bounds if they correspond to units (delete-1 LOCO) or reconstruct from chr
            starts = np.asarray(getattr(self.tr, "_blk_starts", None), dtype=np.int64) if hasattr(self.tr, "_blk_starts") else None
            ends   = np.asarray(getattr(self.tr, "_blk_ends", None),   dtype=np.int64) if hasattr(self.tr, "_blk_ends") else None

            if starts is None or ends is None or starts.size != U:
                # reconstruct from chr array + jackknife_chrs (requires chr sorted)
                if not hasattr(self.tr, "chr") or self.tr.chr is None:
                    raise RuntimeError("chr-mode requires Trace.chr to reconstruct unit bounds.")
                if getattr(self.tr, "jackknife_chrs", None) is None:
                    raise RuntimeError("chr-mode requires Trace.jackknife_chrs to reconstruct unit bounds.")
                chr_arr = np.asarray(self.tr.chr, dtype=np.int32).ravel()
                chrs = np.asarray(self.tr.jackknife_chrs, dtype=np.int32).ravel()
                if chrs.size != U:
                    raise RuntimeError("Mismatch between D.shape[1] and number of jackknife_chrs.")
                if chr_arr.size > 1 and np.any(chr_arr[1:] < chr_arr[:-1]):
                    raise RuntimeError("chr-mode requires SNPs sorted by CHR (and BP).")
                starts = np.searchsorted(chr_arr, chrs, side="left").astype(np.int64)
                ends   = np.searchsorted(chr_arr, chrs, side="right").astype(np.int64)

        if starts.size != U or ends.size != U:
            raise RuntimeError(f"Unit bounds mismatch: starts/ends size {starts.size}/{ends.size}, expected U={U}.")

        m_unit = (ends - starts).astype(np.float64)
        if np.any(m_unit < 0):
            raise RuntimeError("Negative unit sizes encountered in chr unit bounds.")

        # unit sufficient stats
        Ak_unit = np.zeros((U, K), dtype=np.float32)

        Ak2_unit = None
        if need_Ak2:
            Ak2_unit = np.zeros((U, K), dtype=np.float32)

        ov_unit_flat = None
        ov_full = None
        if need_overlap:
            ov_unit_flat = np.zeros((U, K * K), dtype=np.float32)
            ov_full = np.zeros((K, K), dtype=np.float32)

        for u in range(U):
            s = int(starts[u]); e = int(ends[u])
            if e <= s:
                continue
            Au = A32[s:e, :]                # (m_u, K)
            Ak_unit[u] = Au.sum(axis=0, dtype=np.float32)

            if need_Ak2:
                Ak2_unit[u] = (Au * Au).sum(axis=0, dtype=np.float32)

            if need_overlap:
                ov = Au.T @ Au              # (K,K)
                ov_full += ov
                ov_unit_flat[u] = ov.reshape(-1)

        Ak_full = Ak_unit.sum(axis=0)

        out = {
            "_key": key,
            "D": D,
            "starts": starts,
            "ends": ends,
            "m_unit": m_unit,
            "Ak_unit": Ak_unit,
            "Ak_full": Ak_full,
        }
        if need_Ak2:
            out["Ak2_unit"] = Ak2_unit
            out["Ak2_full"] = Ak2_unit.sum(axis=0)

        if need_overlap:
            out["ov_unit_flat"] = ov_unit_flat
            out["ov_full"] = ov_full

        self._chr_unit_cache = out
        return out
