from sumstats import Sumstats
from trace import Trace
import utils

import numpy as np
import os
import sys

class Sumrhe:
    def __init__(self, bim_path=None, sum_path=None, save_path=None, h2_path=None, out=None, chisq_threshold=0, \
            log=None, mem=False, verbose=False, ldscores=None, njack=None, annot=None,
            report_tau: bool = True, allow_neg_enr: bool = False):
        self.mem = mem
        self.log = log
        self.start_time = utils._get_time()
        self.log._log("Analysis started at: "+utils._get_timestr(self.start_time))
        self.tr = Trace(bimpath=bim_path, sumpath=sum_path, savepath=save_path, ldscores=ldscores, log=self.log, nblks=njack, annot=annot, verbose=verbose)
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
        self.allow_neg_enr = bool(allow_neg_enr)  # <--- NEW FLAG
        
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
        Vectorized, filter-aware h^2 calculation (supports overlapping annotations).
        Assumes Trace._filter_snps() has already been called for this phenotype,
        so self.tr.annot/self.tr.nsnps/self.tr.blk_size reflect the finalized SNP set.
        """
        A = self.tr.annot                    # shape (M, K), typically {0,1} but can be floats
        M, K = A.shape
        B = self.nblks

        # ----- One-time per-phenotype precompute (after filtering) -----
        # Use float32 to leverage fast sgemm; cast once.
        A32 = np.asarray(A, dtype=np.float32, order='C')

        # Full overlaps and counts
        overlap_full = A32.T @ A32           # (K, K)
        counts_full  = A32.sum(axis=0)       # (K, )

        # Per-block overlaps and counts
        overlap_blk = np.empty((B, K, K), dtype=np.float32)
        counts_blk  = np.empty((B, K),     dtype=np.float32)

        # Vectorized block boundaries
        # (equal-sized except possibly the last in your current setup; filtering may unbalance counts but the slices are still valid)
        bs = self.tr.blk_size
        starts = bs * np.arange(B)
        ends   = starts + bs
        ends[-1] = self.tr.nsnps  # ensure last block reaches end

        # Compute ab.T @ ab once per block (K is small; BLAS makes this fast)
        for j in range(B):
            ab = A32[starts[j]:ends[j], :]          # (m_j, K), contiguous slice
            overlap_blk[j] = ab.T @ ab              # (K, K)
            counts_blk[j]  = ab.sum(axis=0)         # (K, )

        # ----- Jackknife LOO numerator/denominator (batched over blocks) -----
        # LOO overlaps/counts = full - block
        overlap_minus = overlap_full[None, :, :] - overlap_blk     # (B, K, K)
        counts_minus  = counts_full[None, :]    - counts_blk       # (B, K)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_minus = overlap_minus / counts_minus[:, None, :] # (B, K, K)

        # ----- Batched contraction to get all LOO h2 per category -----
        # sigma_g (LOO) per block/category
        sigma_g_minus = self.sigmas[idx, :B, :K]                   # (B, K)
        # h2_cat_minus[b, c] = sum_k ratio_minus[b, c, k] * sigma_g_minus[b, k]
        h2_cat_minus  = np.einsum('bck,bk->bc', ratio_minus, sigma_g_minus, optimize=True)
        h2_cat_minus[~np.isfinite(h2_cat_minus)] = 0.0
        self.herits[idx, :B, :K] = h2_cat_minus

        # ----- Point estimate (full) -----
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_full   = overlap_full / counts_full[None, :]      # (K, K)
            h2_cat_full  = ratio_full @ self.sigmas[idx, B, :K]     # (K, )
            h2_cat_full[~np.isfinite(h2_cat_full)] = 0.0
        self.herits[idx, B, :K] = h2_cat_full

        # Total h2 across categories is just the sum of sigma_g^2
        self.herits[idx, :, -1] = self.sigmas[idx, :, :K].sum(axis=1)

    
    def _calc_enrich(self, idx):
        """
        Enrichment = (h2_cat / h2_tot) / prop, with prop built from LOO bin counts.
        Handles overlapping annotations. Vectorized over jackknife blocks.
        """
        A = np.asarray(self.tr.annot, dtype=np.float32)   # (M, K)
        M, K = A.shape
        B    = self.nblks

        # --- Full counts per bin (overlaps allowed) ---
        counts_full = A.sum(axis=0)                       # (K,)

        # --- Per-block bin counts (in-block) ---
        bs = self.tr.blk_size
        starts = bs * np.arange(B)
        ends   = starts + bs
        ends[-1] = self.tr.nsnps

        counts_blk = np.empty((B, K), dtype=np.float32)
        blk_sizes  = np.empty(B,    dtype=np.int64)
        for j in range(B):
            ab = A[starts[j]:ends[j], :]                  # (m_j, K)
            counts_blk[j] = ab.sum(axis=0)
            blk_sizes[j]  = ab.shape[0]

        # --- LOO bin counts & totals ---
        # For j-th LOO replicate: remaining SNPs in bin k = counts_full[k] - counts_blk[j, k]
        M_bin = np.empty((B + 1, K), dtype=np.float64)
        M_bin[:B, :] = counts_full[None, :] - counts_blk
        M_bin[B,  :] = counts_full

        # Total SNPs per replicate (not "sum of bin counts" since bins can overlap)
        M_tot = np.empty(B + 1, dtype=np.float64)
        M_tot[:B] = float(self.tr.nsnps) - blk_sizes
        M_tot[B]  = float(self.tr.nsnps)

        # --- Build prop and enrichment ---
        h2_cat = self.herits[idx, :, :K]                  # (B+1, K)
        h2_tot = self.herits[idx, :, -1]                  # (B+1,)

        with np.errstate(divide='ignore', invalid='ignore'):
            prop = M_bin / M_tot[:, None]                 # (B+1, K)
            enr  = (h2_cat / h2_tot[:, None]) / prop

            # base invalid mask (NaN/inf, zero/negative prop)
            invalid = (~np.isfinite(enr)) | (prop <= 0.0)

            # only treat h2_tot <= 0 as invalid when NOT allowing negative enrichment
            if not self.allow_neg_enr:
                invalid |= (h2_tot[:, None] <= 0.0)

            enr[invalid] = np.nan

        self.enrich[idx] = enr
        
    def _calc_tau(self, idx):
        """
        Compute LDSC τ_k and τ*_k across all jackknife replicates (B LOO + full).
        τ_k(b)      = σ^2_{g,k}(b) / sum_j a_{j,k}(b)
        τ*_k(b)     = τ_k(b) * [ sd(a_k)(b) / ( h2_tot(b) / M(b) ) ]
        Uses LOO sums and sumsq to get per-replicate sd(a_k).
        """
        A = np.asarray(self.tr.annot, dtype=np.float64)   # (M, K) after filtering for this phenotype
        M_full, K = A.shape
        B = self.nblks

        # Per-column sums and sums of squares (full)
        Ak_full   = A.sum(axis=0)                         # ∑ a_{j,k}
        Ak2_full  = (A*A).sum(axis=0)                     # ∑ a_{j,k}^2

        # Per-block (in-block) sums, sums of squares, and block sizes
        bs = self.tr.blk_size
        starts = bs * np.arange(B)
        ends   = starts + bs
        ends[-1] = self.tr.nsnps

        Ak_blk  = np.empty((B, K), dtype=np.float64)
        Ak2_blk = np.empty((B, K), dtype=np.float64)
        m_blk   = np.empty(B,      dtype=np.int64)
        for b in range(B):
            ab = A[starts[b]:ends[b], :]
            Ak_blk[b]  = ab.sum(axis=0)
            Ak2_blk[b] = (ab*ab).sum(axis=0)
            m_blk[b]   = ab.shape[0]

        # Build per-replicate totals (B LOO + full)
        M_rep   = np.empty(B+1, dtype=np.float64)
        M_rep[:B] = M_full - m_blk
        M_rep[B]  = M_full

        Ak_rep   = np.empty((B+1, K), dtype=np.float64)   # ∑ a over replicate
        Ak2_rep  = np.empty((B+1, K), dtype=np.float64)   # ∑ a^2 over replicate
        Ak_rep[:B, :]  = Ak_full[None, :] - Ak_blk
        Ak2_rep[:B, :] = Ak2_full[None, :] - Ak2_blk
        Ak_rep[B,  :]  = Ak_full
        Ak2_rep[B, :]  = Ak2_full

        # σ_g^2 per replicate/category and total h^2 per replicate
        sigma_g_rep = self.sigmas[idx, :, :K]            # (B+1, K)
        h2_tot_rep  = self.herits[idx, :, -1]            # (B+1,)

        # τ: safe divide (0 where Ak==0 or not finite)
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = sigma_g_rep / Ak_rep
            tau[~np.isfinite(tau)] = 0.0

        # sd(a_k) per replicate from sums and sums of squares (population-style sd)
        # var = E[a^2] - (E[a])^2, with E[…] over replicate SNPs
        with np.errstate(divide='ignore', invalid='ignore'):
            meanA   = Ak_rep / M_rep[:, None]
            meanA2  = Ak2_rep / M_rep[:, None]
            varA    = np.maximum(meanA2 - meanA*meanA, 0.0)
            sdA     = np.sqrt(varA, dtype=np.float64)

            denom   = h2_tot_rep / M_rep                   # (B+1,)
            tau_star = tau * (sdA / denom[:, None])
            tau_star[~np.isfinite(tau_star)] = 0.0

        self.tau[idx]      = tau
        self.tau_star[idx] = tau_star



    def _run_jackknife(self, idx):
        ''' run snp-level block jackknife '''
        self.sigsums[idx, :, 0] = self.sigmas[idx, self.nblks]
        self.sigsums[idx, :, 1] = utils._calc_jackknife_se(self.sigmas[idx])[1]
        
        self.hersums[idx, :, 0] = self.herits[idx, self.nblks]
        self.hersums[idx, :, 1] = utils._calc_jackknife_se(self.herits[idx])[1]
    
        self.enrich_sums[idx, :, 0] = self.enrich[idx, self.nblks]
        self.enrich_sums[idx, :, 1] = utils._calc_jackknife_se(self.enrich[idx])[1]
        
        if self.report_tau:
            self.tau_sums[idx, :, 0]      = self.tau[idx, self.nblks]
            self.tau_sums[idx, :, 1]      = utils._calc_jackknife_se(self.tau[idx])[1]
            self.tau_star_sums[idx, :, 0] = self.tau_star[idx, self.nblks]
            self.tau_star_sums[idx, :, 1] = utils._calc_jackknife_se(self.tau_star[idx])[1]

        if (self.verbose):
            self.log._log("Sigma solution & jackknife SE:\n"+np.array2string(self.sigsums, precision=5, separator=', '))
            self.log._log("Heritability (category) & jackknife SE:\n"+np.array2string(self.hersums, precision=5, separator=', '))
            self.log._log("Enrichment & jackknife SE:\n"+np.array2string(self.enrich_sums, precision=5, separator=', '))
            if self.report_tau:
                self.log._log("Tau & Tau* (point, SE):\n" + np.array2string(np.stack([self.tau_sums[idx,:,0], self.tau_sums[idx,:,1]], axis=1), precision=5, separator=', '))

    def _run(self):
        for i in range(self.npheno):
            h2_path=self.h2_dir[i]
            removesnps = self.sums._process(h2_path, self.phen_names[i])
            self.tr._filter_snps(removesnps) # always try to filter both sides
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
