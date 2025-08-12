from sumstats import Sumstats
from trace import Trace
import utils

import numpy as np
import os
import sys

class Sumrhe:
    def __init__(self, bim_path=None, sum_path=None, save_path=None, h2_path=None, out=None, chisq_threshold=0, \
            log=None, mem=False, verbose=False, ldscores=None, njack=None, annot=None):
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

    def _calc_sigmas(self, idx):
        rhs = self.sums.rhs
        pred_tr = self.tr._calc_trace(self.nsamp[idx])
        for i in range(self.nblks+1):
            sig_est = utils._solve_linear_equation(pred_tr[i], rhs[i])
            self.sigmas[idx][i] = np.append(sig_est, sig_est[:-1].sum())
        if self.verbose:
            names = [f"sigma^2_g{t}" for t in range(self.nbins)] + ["sigma^2_e"]
            self.log._log("Normal equation:\n"+np.array2string(pred_tr[self.nblks], precision=2, separator=', ')+"\n\t\t*\n"\
                +"["+", ".join(names)+"]\n\t\t=\n"+np.array2string(rhs[self.nblks], precision=2, separator=', '))
            self.log._log("Sigma solution:\n"+np.array2string(self.sigmas[idx, self.nblks, :self.nbins], precision=3, separator=', '))
        return self.sigmas[idx]

    def _calc_h2(self, idx):
        '''
        calculate heritabilities from variance components (supports overlapping annotations)
        '''
        full_overlap = self.tr.annot.T @ self.tr.annot # (K, K): |S_k \cap S_c|
        full_counts  = self.tr.annot.sum(axis=0) # (K, ): M_c

        for j in range(self.nblks):
            start = self.tr.blk_size * j
            end   = self.tr.blk_size*(j+1) if (j < self.nblks-1) else self.tr.nsnps
            ab = self.tr.annot[start:end]
            overlap_j = full_overlap - (ab.T @ ab)
            counts_j = full_counts - ab.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                h2_cat_j = (overlap_j / counts_j[None, :]) @ self.sigmas[idx, j, :self.nbins]
                h2_cat_j[~np.isfinite(h2_cat_j)] = 0.0
            self.herits[idx, j, :self.nbins] = h2_cat_j

        # point estimate
        with np.errstate(divide='ignore', invalid='ignore'):
            h2_cat_full = (full_overlap / full_counts[None, :]) @ self.sigmas[idx, self.nblks, :self.nbins]
            h2_cat_full[~np.isfinite(h2_cat_full)] = 0.0
        self.herits[idx, self.nblks, :self.nbins] = h2_cat_full
        self.herits[idx, :, -1] = self.sigmas[idx, :, :self.nbins].sum(axis=1)
    
    def _calc_enrich(self, idx):
        h2_cat = self.herits[idx, :, :self.nbins]
        h2_tot = self.herits[idx, :, -1]

        M_bin = self.tr.nsnps_blk.astype(float)  # (nblks+1, nbins)

        M_tot = np.zeros(self.nblks+1, dtype=float)
        for j in range(self.nblks):
            start = self.tr.blk_size * j
            end   = self.tr.blk_size*(j+1) if (j < self.nblks-1) else self.tr.nsnps
            M_tot[j] = float(self.tr.nsnps - (end - start))
        M_tot[self.nblks] = float(self.tr.nsnps)

        with np.errstate(divide='ignore', invalid='ignore'):
            prop = M_bin / M_tot[:, None]
            enr  = (h2_cat / h2_tot[:, None]) / prop
            enr[~np.isfinite(enr)] = np.nan

        self.enrich[idx] = enr

    def _run_jackknife(self, idx):
        ''' run snp-level block jackknife '''
        self.sigsums[idx, :, 0] = self.sigmas[idx, self.nblks]
        self.sigsums[idx, :, 1] = utils._calc_jackknife_se(self.sigmas[idx])[1]
        
        self.hersums[idx, :, 0] = self.herits[idx, self.nblks]
        self.hersums[idx, :, 1] = utils._calc_jackknife_se(self.herits[idx])[1]
    
        self.enrich_sums[idx, :, 0] = self.enrich[idx, self.nblks]
        self.enrich_sums[idx, :, 1] = utils._calc_jackknife_se(self.enrich[idx])[1]

        if (self.verbose):
            self.log._log("Sigma solution & jackknife SE:\n"+np.array2string(self.sigsums, precision=5, separator=', '))
            self.log._log("Heritability (category) & jackknife SE:\n"+np.array2string(self.hersums, precision=5, separator=', '))
            self.log._log("Enrichment & jackknife SE:\n"+np.array2string(self.enrich_sums, precision=5, separator=', '))

    def _run(self):
        for i in range(self.npheno):
            h2_path=self.h2_dir[i]
            removesnps = self.sums._process(h2_path, self.phen_names[i])
            self.tr._filter_snps(removesnps) # always try to filter both sides
            self.nsamp.append(self.sums.nsamp)

            self._calc_sigmas(i)
            self._calc_h2(i)
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
                    self.log._log(
                        f"^^^ Phenotype {i} Bin [{self.annot_header[j]}] "
                        f"sigma_g^2: {sig:.5f} (SE: {sigse:.5f}) "
                        f"h^2_cat: {h2:.5f} (SE: {h2se:.5f}) "
                        f"Enrichment: {enr:.5f} (SE: {ense:.5f})"
                    )
            # total SNP h2 (sum sigma_g^2)
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
