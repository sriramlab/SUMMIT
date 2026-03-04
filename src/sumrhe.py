from sumstats import Sumstats
from trace import Trace
import utils

import numpy as np
import os
import sys

class Sumrhe:
    def __init__(
        self,
        bim_path=None,
        sum_path=None,
        save_path=None,
        h2_path=None,
        out=None,
        chisq_threshold=0,
        log=None,
        mem=False,
        verbose=False,
        ldscores=None,
        njack=None,
        annot=None,
        chisq_action='drop',
        report_tau: bool = True,
        allow_neg_enr: bool = False,
        clip_nonfinite_vals: bool = False,
        adjust_delta: bool = False,
        enrich_mode: str = "auto",
        jack_mode: str = "median",
        jackknife_weighted: bool = True,   # NEW (default ON)
    ):
        self.mem = mem
        self.log = log
        self.start_time = utils._get_time()
        self.log._log("Analysis started at: "+utils._get_timestr(self.start_time))

        self.tr = Trace(
            bimpath=bim_path,
            sumpath=sum_path,
            savepath=save_path,
            ldscores=ldscores,
            log=self.log,
            nblks=njack,                 # can be int or 'chr' now
            annot=annot,
            verbose=verbose,
            adjust_delta=adjust_delta,
        )

        self.nblks = self.tr.nblks
        self.jack_mode = jack_mode
        self.annot_header = self.tr.annot_header
        self.nbins = self.tr.nbins

        # NEW: propagate jackknife partitioning so RHS blocks match LHS blocks
        self._jackknife_weighted = bool(jackknife_weighted)
        self._jackknife_partition_mode = getattr(self.tr, "jackknife_mode", "block")
        self._jackknife_chrs = getattr(self.tr, "jackknife_chrs", None)

        self.sums = Sumstats(
            nblks=self.nblks,
            chisq_threshold=chisq_threshold,
            log=self.log,
            annot_df=self.tr.annot_df,
            nbins=self.nbins,
            chisq_action=chisq_action,
            jackknife_mode=self._jackknife_partition_mode,
            jackknife_chrs=self._jackknife_chrs,
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
        Vectorized, filter-aware h^2 calculation (supports overlapping + continuous annotations).
        Assumes Trace._filter_snps() has already been called for this phenotype,
        so self.tr.annot/self.tr.nsnps/self.tr.blk_size reflect the finalized SNP set.

        IMPORTANT semantic note:
        For continuous annotations, "bin size" is Ak = sum_j a_{j,k} (sum of weights),
        not necessarily an integer SNP count.
        """
        A = self.tr.annot                    # (M, K), can be 0/1, overlapping, or continuous >=0
        M, K = A.shape
        B = self.nblks

        # optional sanity: continuous annotations should usually be >=0
        if np.nanmin(A) < 0:
            self.log._log("[WARNING] Detected negative annotation weights; h2/enrichment/tau may be ill-defined.")

        # Use float32 for fast BLAS
        A32 = np.asarray(A, dtype=np.float32, order='C')

        # Full overlaps and "counts" (weight sums)
        overlap_full = A32.T @ A32           # (K, K)
        Ak_full      = A32.sum(axis=0)       # (K,)

        # Block bounds: prefer Trace's cached bounds
        if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            starts = self.tr._blk_starts
            ends   = self.tr._blk_ends
        else:
            bs = self.tr.blk_size
            starts = bs * np.arange(B)
            ends   = starts + bs
            ends[-1] = self.tr.nsnps

        # Per-block overlaps and Ak (in-block)
        overlap_blk = np.empty((B, K, K), dtype=np.float32)
        Ak_blk      = np.empty((B, K),     dtype=np.float32)

        for b in range(B):
            ab = A32[int(starts[b]):int(ends[b]), :]   # (m_b, K)
            overlap_blk[b] = ab.T @ ab                 # (K, K)
            Ak_blk[b]      = ab.sum(axis=0)            # (K,)

        # LOO overlaps / Ak (weight sums)
        overlap_minus = overlap_full[None, :, :] - overlap_blk      # (B, K, K)
        Ak_minus      = Ak_full[None, :]       - Ak_blk             # (B, K)

        # ratio_minus[b, c, k] = overlap_minus[b, c, k] / Ak_minus[b, k]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_minus = overlap_minus / Ak_minus[:, None, :]      # (B, K, K)

        # Contract: h2_cat_minus[b, c] = sum_k ratio_minus[b, c, k] * sigma_g[b, k]
        sigma_g_minus = self.sigmas[idx, :B, :K].astype(np.float64, copy=False)  # (B, K)
        h2_cat_minus  = np.einsum('bck,bk->bc', ratio_minus, sigma_g_minus, optimize=True)

        # invalidate bad replicates (Ak_minus <= 0 yields inf/nan)
        bad_minus = (~np.isfinite(h2_cat_minus))
        if self.clip_nonfinite_vals:
            h2_cat_minus[bad_minus] = 0.0
        else:
            h2_cat_minus[bad_minus] = np.nan

        self.herits[idx, :B, :K] = h2_cat_minus

        # ----- Full-sample estimate -----
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_full  = overlap_full / Ak_full[None, :]          # (K, K)
            h2_cat_full = ratio_full @ self.sigmas[idx, B, :K]     # (K,)

        bad_full = ~np.isfinite(h2_cat_full)
        if self.clip_nonfinite_vals:
            h2_cat_full[bad_full] = 0.0
        else:
            h2_cat_full[bad_full] = np.nan

        self.herits[idx, B, :K] = h2_cat_full

        # Total h2 across categories is sum of sigma_g^2 (same convention as your current code)
        self.herits[idx, :, -1] = self.sigmas[idx, :, :K].sum(axis=1)

    
    def _calc_enrich(self, idx):
        """
        Enrichment = (h2_cat / h2_tot) / prop

        prop is computed as:
            prop[k] = Ak_rep[k] / M_rep
        where Ak_rep[k] is sum of weights (or SNP count if binary) in replicate,
        and M_rep is the number of SNPs in replicate.

        Enrichment modes:
          - "overlap": uses overlap/SNP-set h2_cat (your self.herits[..., :K])
          - "non-overlap": uses component h2_cat from variance components (sigma_g)
          - "auto": picks "non-overlap" if annotations are non-overlapping, else "overlap"
          - "both": computes both and stores in self.enrich_overlap/nonoverlap (and also sets self.enrich to auto-picked one)
        """
        A = np.asarray(self.tr.annot, dtype=np.float64, order='C')  # (M, K)
        M_full, K = A.shape
        B = self.nblks

        if np.nanmin(A) < 0:
            self.log._log("[WARNING] Detected negative annotation weights; enrichment may be ill-defined.")

        # Block bounds
        if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            starts = self.tr._blk_starts
            ends   = self.tr._blk_ends
        else:
            bs = self.tr.blk_size
            starts = bs * np.arange(B)
            ends   = starts + bs
            ends[-1] = self.tr.nsnps

        # Full bin mass (sum of weights)
        Ak_full = A.sum(axis=0)  # (K,)

        # In-block bin mass and block sizes
        Ak_blk = np.empty((B, K), dtype=np.float64)
        m_blk  = np.empty(B, dtype=np.int64)
        for b in range(B):
            ab = A[int(starts[b]):int(ends[b]), :]
            Ak_blk[b] = ab.sum(axis=0)
            m_blk[b]  = ab.shape[0]

        # Replicate masses (LOO + full)
        Ak_rep = np.empty((B + 1, K), dtype=np.float64)
        Ak_rep[:B, :] = Ak_full[None, :] - Ak_blk
        Ak_rep[B,  :] = Ak_full

        # Replicate SNP counts (NOT sum of weights; bins can overlap)
        M_rep = np.empty(B + 1, dtype=np.float64)
        M_rep[:B] = float(M_full) - m_blk
        M_rep[B]  = float(M_full)

        # prop (relative to mean weight per SNP)
        with np.errstate(divide='ignore', invalid='ignore'):
            prop = Ak_rep / M_rep[:, None]  # (B+1, K)

        # Total SNP h2 (your convention: sum of sigma_g)
        # Keep consistent with your herits[:, -1] = sum sigma_g.
        h2_tot = self.herits[idx, :, -1].astype(np.float64, copy=False)  # (B+1,)

        # Decide which mode(s) to compute
        requested = self.enrich_mode
        has_ov = utils._has_overlapping_annotations(A)

        if requested == "auto":
            mode_used = "overlap" if has_ov else "non-overlap"
            modes_to_compute = (mode_used,)
        elif requested == "both":
            # compute both; choose what to put in self.enrich using the auto rule
            mode_used = "overlap" if has_ov else "non-overlap"
            modes_to_compute = ("non-overlap", "overlap")
        else:
            mode_used = requested
            modes_to_compute = (requested,)

        self._enrich_mode_used[idx] = mode_used
        if self.verbose and requested in ("auto", "both"):
            self.log._log(f"[enrichment] phenotype={idx} enrich_mode={requested} resolved={mode_used} has_overlap={has_ov}")

        def _compute_enr_from_h2cat(h2_cat: np.ndarray) -> np.ndarray:
            # h2_cat: (B+1, K)
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
            # overlap/SNP-set enrichment (paper's "overlapping enrichment" notion)
            h2_cat_ov = self.herits[idx, :, :K].astype(np.float64, copy=False)  # (B+1, K)
            enr_ov = _compute_enr_from_h2cat(h2_cat_ov)

        if "non-overlap" in modes_to_compute:
            # component enrichment (variance-component share)
            h2_cat_no = self.sigmas[idx, :, :K].astype(np.float64, copy=False)  # (B+1, K) == sigma_g,k
            enr_no = _compute_enr_from_h2cat(h2_cat_no)

        # Store results
        if requested == "both":
            self.enrich_overlap[idx] = enr_ov
            self.enrich_nonoverlap[idx] = enr_no
            # keep backward-compatible self.enrich as the auto-resolved one
            self.enrich[idx] = enr_ov if mode_used == "overlap" else enr_no
        else:
            self.enrich[idx] = enr_ov if mode_used == "overlap" else enr_no
      
    def _calc_tau(self, idx):
        """
        Compute LDSC τ_k and τ*_k across all jackknife replicates (B LOO + full).

        For continuous/overlapping annotations:
        Ak_rep  = ∑ a_{j,k}          (sum of weights)
        Ak2_rep = ∑ a_{j,k}^2
        sdA     from sums and sumsqs over replicate SNPs.

        τ_k(b)      = σ^2_{g,k}(b) / Ak_rep(b)
        τ*_k(b)     = τ_k(b) * [ sd(a_k)(b) / ( h2_tot(b) / M_rep(b) ) ]
        """
        A = np.asarray(self.tr.annot, dtype=np.float64, order='C')  # (M, K)
        M_full, K = A.shape
        B = self.nblks

        if np.nanmin(A) < 0:
            self.log._log("[WARNING] Detected negative annotation weights; tau/tau* may be ill-defined.")

        # Block bounds
        if hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            starts = self.tr._blk_starts
            ends   = self.tr._blk_ends
        else:
            bs = self.tr.blk_size
            starts = bs * np.arange(B)
            ends   = starts + bs
            ends[-1] = self.tr.nsnps

        # Full sums
        Ak_full  = A.sum(axis=0)           # ∑ a
        Ak2_full = (A * A).sum(axis=0)     # ∑ a^2

        # In-block sums
        Ak_blk  = np.empty((B, K), dtype=np.float64)
        Ak2_blk = np.empty((B, K), dtype=np.float64)
        m_blk   = np.empty(B, dtype=np.int64)
        for b in range(B):
            ab = A[int(starts[b]):int(ends[b]), :]
            Ak_blk[b]  = ab.sum(axis=0)
            Ak2_blk[b] = (ab * ab).sum(axis=0)
            m_blk[b]   = ab.shape[0]

        # Replicate SNP counts
        M_rep = np.empty(B + 1, dtype=np.float64)
        M_rep[:B] = float(M_full) - m_blk
        M_rep[B]  = float(M_full)

        # Replicate sums
        Ak_rep  = np.empty((B + 1, K), dtype=np.float64)
        Ak2_rep = np.empty((B + 1, K), dtype=np.float64)
        Ak_rep[:B, :]  = Ak_full[None, :]  - Ak_blk
        Ak2_rep[:B, :] = Ak2_full[None, :] - Ak2_blk
        Ak_rep[B, :]   = Ak_full
        Ak2_rep[B, :]  = Ak2_full

        sigma_g_rep = self.sigmas[idx, :, :K]       # (B+1, K)
        h2_tot_rep  = self.herits[idx, :, -1]       # (B+1,)

        # τ
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = sigma_g_rep / Ak_rep

        bad_tau = ~np.isfinite(tau)
        if self.clip_nonfinite_vals:
            tau[bad_tau] = 0.0
        else:
            tau[bad_tau] = np.nan

        # sd(a_k) per replicate (population-style)
        with np.errstate(divide='ignore', invalid='ignore'):
            meanA  = Ak_rep / M_rep[:, None]
            meanA2 = Ak2_rep / M_rep[:, None]
            varA   = np.maximum(meanA2 - meanA * meanA, 0.0)
            sdA    = np.sqrt(varA, dtype=np.float64)

            denom = h2_tot_rep / M_rep   # (B+1,)
            tau_star = tau * (sdA / denom[:, None])

        bad_ts = ~np.isfinite(tau_star)
        if self.clip_nonfinite_vals:
            tau_star[bad_ts] = 0.0
        else:
            tau_star[bad_ts] = np.nan

        self.tau[idx]      = tau
        self.tau_star[idx] = tau_star


    def _run_jackknife(self, idx):
        """Run SNP-level block jackknife for this phenotype."""
        # Block sizes = number of SNPs deleted for replicate b (used as delete-m weights)
        weights = None
        use_pv = False
        if self._jackknife_weighted and hasattr(self.tr, "_blk_starts") and hasattr(self.tr, "_blk_ends"):
            m = (np.asarray(self.tr._blk_ends, dtype=np.float64) - np.asarray(self.tr._blk_starts, dtype=np.float64))
            weights = m
            use_pv = True

        # Sigma components
        est_full, se_jk = utils._calc_jackknife_se(
            self.sigmas[idx],
            axis=0,
            center=self.jack_mode,
            nan_policy=self.nan_policy,
            weights=weights,
            use_pseudovalues=use_pv,
        )
        self.sigsums[idx, :, 0] = est_full
        self.sigsums[idx, :, 1] = se_jk

        # Heritabilities (per bin + total)
        est_full_h2, se_jk_h2 = utils._calc_jackknife_se(
            self.herits[idx],
            axis=0,
            center=self.jack_mode,
            nan_policy=self.nan_policy,
            weights=weights,
            use_pseudovalues=use_pv,
        )
        self.hersums[idx, :, 0] = est_full_h2
        self.hersums[idx, :, 1] = se_jk_h2

        # Enrichment
        est_full_enr, se_jk_enr = utils._calc_jackknife_se(
            self.enrich[idx],
            axis=0,
            center=self.jack_mode,
            nan_policy=self.nan_policy,
            weights=weights,
            use_pseudovalues=use_pv,
        )
        self.enrich_sums[idx, :, 0] = est_full_enr
        self.enrich_sums[idx, :, 1] = se_jk_enr

        # Optional enrichment outputs (both-mode)
        if self.enrich_overlap is not None:
            est_full_eov, se_jk_eov = utils._calc_jackknife_se(
                self.enrich_overlap[idx],
                axis=0,
                center=self.jack_mode,
                nan_policy=self.nan_policy,
                weights=weights,
                use_pseudovalues=use_pv,
            )
            self.enrich_overlap_sums[idx, :, 0] = est_full_eov
            self.enrich_overlap_sums[idx, :, 1] = se_jk_eov

        if self.enrich_nonoverlap is not None:
            est_full_eno, se_jk_eno = utils._calc_jackknife_se(
                self.enrich_nonoverlap[idx],
                axis=0,
                center=self.jack_mode,
                nan_policy=self.nan_policy,
                weights=weights,
                use_pseudovalues=use_pv,
            )
            self.enrich_nonoverlap_sums[idx, :, 0] = est_full_eno
            self.enrich_nonoverlap_sums[idx, :, 1] = se_jk_eno

        # τ / τ* if requested
        if self.report_tau:
            est_full_tau, se_jk_tau = utils._calc_jackknife_se(
                self.tau[idx],
                axis=0,
                center=self.jack_mode,
                nan_policy=self.nan_policy,
                weights=weights,
                use_pseudovalues=use_pv,
            )
            self.tau_sums[idx, :, 0] = est_full_tau
            self.tau_sums[idx, :, 1] = se_jk_tau

            est_full_ts, se_jk_ts = utils._calc_jackknife_se(
                self.tau_star[idx],
                axis=0,
                center=self.jack_mode,
                nan_policy=self.nan_policy,
                weights=weights,
                use_pseudovalues=use_pv,
            )
            self.tau_star_sums[idx, :, 0] = est_full_ts
            self.tau_star_sums[idx, :, 1] = se_jk_ts

        if self.verbose:
            self.log._log("Sigma solution & jackknife SE:\n"+ np.array2string(self.sigsums[idx], precision=5, separator=', '))
            self.log._log("Heritability (category) & jackknife SE:\n"+ np.array2string(self.hersums[idx], precision=5, separator=', '))
            self.log._log("Enrichment & jackknife SE:\n"+ np.array2string(self.enrich_sums[idx], precision=5, separator=', '))
            if self.report_tau:
                self.log._log("Tau & Tau* (point, SE):\n"+ np.array2string(np.stack([self.tau_sums[idx,:,0], self.tau_sums[idx,:,1]], axis=1), precision=5, separator=', '))

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
