from sumstats import Sumstats
from trace import Trace
import utils

import numpy as np
import os
import sys

class Sumrhe:
    def __init__(self, bim_path=None, sum_path=None, save_path=None, h2_path=None, out=None, chisq_threshold=0, \
            log=None, mem=False, verbose=False, ldscores=None, njack=None, annot=None,
            report_tau: bool = True, allow_neg_enr: bool = False, clip_nonfinite_vals: bool = False, adjust_delta: bool = False):
        self.mem = mem
        self.log = log
        self.start_time = utils._get_time()
        self.log._log("Analysis started at: "+utils._get_timestr(self.start_time))
        self.tr = Trace(bimpath=bim_path, sumpath=sum_path, savepath=save_path, ldscores=ldscores, log=self.log, nblks=njack, annot=annot, verbose=verbose, adjust_delta=adjust_delta)
        self.nblks = self.tr.nblks
        self.annot_header = self.tr.annot_header
        self.nbins = self.tr.nbins
        self.sums = Sumstats(nblks=self.nblks, chisq_threshold=chisq_threshold, log=self.log, annot_df=self.tr.annot_df, nbins=self.nbins)
        
        try:
            self.h2_dir = utils._parse_sumdir(h2_path)
        except ValueError as e:
            self.log._log(f"Error reading sumstat files: {e}")
            sys.exit(1)

        self.npheno = len(self.h2_dir)
        self.nsamp = []
        self.phen_names = [os.path.basename(name)[:-8] for name in self.h2_dir]

        self.sigmas = np.zeros((self.npheno, self.nblks+1, self.nbins+2)) # jackknife subsampled variance components
        self.sigsums = np.zeros((self.npheno, self.nbins+2, 2)) # variance components + total variance components

        self.herits  = np.zeros((self.npheno, self.nblks+1, self.nbins+1)) # jackknife subsampled h2
        self.hersums = np.zeros((self.npheno, self.nbins+1, 2)) # total h2 + se
        
        self.enrich = np.zeros((self.npheno, self.nblks+1, self.nbins))
        self.enrich_sums = np.zeros((self.npheno, self.nbins, 2))

        self.out = out
        self.verbose = verbose
        self.report_tau = bool(report_tau)
        self.allow_neg_enr = bool(allow_neg_enr)
        self.clip_nonfinite_vals = clip_nonfinite_vals
        self.adjust_delta = adjust_delta
        
        if self.report_tau:
            self.tau         = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
            self.tau_star    = np.zeros((self.npheno, self.nblks+1, self.nbins), dtype=np.float64)
            self.tau_sums    = np.zeros((self.npheno, self.nbins, 2), dtype=np.float64)  # [point, SE]
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

        clip_nonfinite = bool(getattr(self, "clip_nonfinite_vals", False))

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
        if clip_nonfinite:
            h2_cat_minus[bad_minus] = 0.0
        else:
            h2_cat_minus[bad_minus] = np.nan

        self.herits[idx, :B, :K] = h2_cat_minus

        # ----- Full-sample estimate -----
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_full  = overlap_full / Ak_full[None, :]          # (K, K)
            h2_cat_full = ratio_full @ self.sigmas[idx, B, :K]     # (K,)

        bad_full = ~np.isfinite(h2_cat_full)
        if clip_nonfinite:
            h2_cat_full[bad_full] = 0.0
        else:
            h2_cat_full[bad_full] = np.nan

        self.herits[idx, B, :K] = h2_cat_full

        # Total h2 across categories is sum of sigma_g^2 (same convention as your current code)
        self.herits[idx, :, -1] = self.sigmas[idx, :, :K].sum(axis=1)

    
    def _calc_enrich(self, idx):
        """
        Enrichment = (h2_cat / h2_tot) / prop

        For continuous/overlapping annotations:
        Ak_rep[k] = sum_j a_{j,k}  (sum of weights in replicate)
        prop[k]   = Ak_rep[k] / M_rep, where M_rep is #SNPs in replicate (not sum_k Ak).

        This matches your previous overlapping convention and generalizes to continuous
        as "relative to mean weight per SNP".
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

        h2_cat = self.herits[idx, :, :K]   # (B+1, K)
        h2_tot = self.herits[idx, :, -1]   # (B+1,)

        with np.errstate(divide='ignore', invalid='ignore'):
            prop = Ak_rep / M_rep[:, None]                    # (B+1, K)
            enr  = (h2_cat / h2_tot[:, None]) / prop

            invalid = (~np.isfinite(enr)) | (~np.isfinite(prop)) | (prop <= 0.0)

            if not self.allow_neg_enr:
                invalid |= (h2_tot[:, None] <= 0.0)

            enr[invalid] = np.nan

        self.enrich[idx] = enr

        
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

        clip_nonfinite = bool(getattr(self, "clip_nonfinite_vals", False))

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
        if clip_nonfinite:
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
        if clip_nonfinite:
            tau_star[bad_ts] = 0.0
        else:
            tau_star[bad_ts] = np.nan

        self.tau[idx]      = tau
        self.tau_star[idx] = tau_star




    def _run_jackknife(self, idx):
        """Run SNP-level block jackknife for this phenotype."""
        clip_nonfinite = bool(getattr(self, "clip_nonfinite_vals", False))
        nan_policy = 'propagate' if clip_nonfinite else 'omit'

        # Sigma components
        est_full, se_jk = utils._calc_jackknife_se(self.sigmas[idx], axis=0, center='full', nan_policy=nan_policy)
        self.sigsums[idx, :, 0] = est_full
        self.sigsums[idx, :, 1] = se_jk

        # Heritabilities (per bin + total)
        est_full_h2, se_jk_h2 = utils._calc_jackknife_se(self.herits[idx], axis=0, center='full', nan_policy=nan_policy)
        self.hersums[idx, :, 0] = est_full_h2
        self.hersums[idx, :, 1] = se_jk_h2

        # Enrichment
        est_full_enr, se_jk_enr = utils._calc_jackknife_se(self.enrich[idx], axis=0, center='full', nan_policy=nan_policy)
        self.enrich_sums[idx, :, 0] = est_full_enr
        self.enrich_sums[idx, :, 1] = se_jk_enr

        # τ / τ* if requested
        if self.report_tau:
            est_full_tau, se_jk_tau = utils._calc_jackknife_se(self.tau[idx], axis=0, center='full', nan_policy=nan_policy)
            self.tau_sums[idx, :, 0] = est_full_tau
            self.tau_sums[idx, :, 1] = se_jk_tau

            est_full_ts, se_jk_ts = utils._calc_jackknife_se(self.tau_star[idx], axis=0, center='full', nan_policy=nan_policy)
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
            if (self.nbins > 1):
                for j in range(self.nbins):
                    sig   = self.sigsums[i, j, 0]
                    sigse = self.sigsums[i, j, 1]
                    h2    = self.hersums[i, j, 0]
                    h2se  = self.hersums[i, j, 1]
                    enr   = self.enrich_sums[i, j, 0]
                    ense  = self.enrich_sums[i, j, 1]

                    line = (f"^^^ Phenotype {i} Bin [{self.annot_header[j]}] "
                            f"sigma_g^2: {sig:.5f} (SE: {sigse:.5f}) "
                            f"h^2_cat: {h2:.5f} (SE: {h2se:.5f}) "
                            f"Enrichment: {enr:.5f} (SE: {ense:.5f})")
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
