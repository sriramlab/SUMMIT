#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from sumstats import Sumstats
from trace import Trace
from sumrhe import Sumrhe
import utils

class Sumcore(Sumrhe):
    def __init__(self, bim_path=None, save_path=None, rg=None, ldscores=None, log=None, verbose=False,\
                chisq_threshold=0, filter_both=False, annot=None, njack=None, out=None, intercept=None, phenos=None):
        self.mem = False
        self.log = log
        self.verbose = verbose
        self.start_time = utils._get_time()
        self.log._log("SUM-CORE started at: " + utils._get_timestr(self.start_time))
        self.n_overlap = None
        self.intercept = intercept # constrain the intercept term
        self.phenos = phenos
        if (self.phenos is not None):
            self.estimate_overlap_cov()

        # initialize Trace
        self.tr = Trace(bimpath=bim_path, sumpath=None, savepath=save_path,
                        ldscores=ldscores, log=self.log, nblks=njack,
                        annot=annot, verbose=verbose)
        self.nblks = self.tr.nblks
        self.annot_header = self.tr.annot_header
        self.nbins = self.tr.nbins

        # parameter containers
        self.gamma_g = np.zeros(self.nblks+1)
        self.c_opt = np.zeros(self.nblks+1)
        self.rg = np.zeros(self.nblks+1)

        # parse --rg argument: must be two comma‐separated .sumstat paths
        try:
            self.phen_dir = utils._parse_rgdir(rg)
        except ValueError as e:
            self.log._log(f"Error reading sumstat file pair: {e}")
            sys.exit(1)
        self.npheno = 2
        self.phen_names = [os.path.basename(name)[:-8] for name in self.phen_dir]

        self.sums = []
        for pth, name in zip(self.phen_dir, self.phen_names):
            ss = Sumstats(nblks=self.nblks,
                          chisq_threshold=chisq_threshold,
                          log=self.log,
                          annot_df = self.tr.annot_df,
                          nbins=self.nbins)
            self.sums.append(ss)
        
        self.nsnps        = self.tr.nsnps

        # storage for heritabilities
        self.nsamp  = []
        self.herits = np.zeros((2, self.nblks+1, self.nbins+2))
        self.hsums  = np.zeros((2, self.nbins+2, 2))

        self.out         = out
        self.filter_both = filter_both
    
    def _run(self):
        # TODO: Implement allele alignment
        # Sketch: align the SNPs first. Calculate heritability 
        for i in range(2):
            pheno_path = self.phen_dir[i]
            removesnps  = self.sums[i]._process(pheno_path, self.phen_names[i])
            self.tr._filter_snps(removesnps) # always try to filter both sides
            self.nsamp.append(self.sums[i].nsamp)

            rhs     = self.sums[i].rhs
            pred_tr = self.tr._calc_trace(self.nsamp[i])

            for b in range(self.nblks+1):
                h2_est = utils._solve_linear_equation(pred_tr[b], rhs[b])
                self.herits[i][b] = np.append(h2_est, h2_est[:-1].sum())
            
            # jackknife SE
            self.hsums[i, :, 0] = self.herits[i, self.nblks]
            self.hsums[i, :, 1] = utils._calc_jackknife_se(self.herits[i])[1]

        z1 = self.sums[0].zscores
        z2 = self.sums[1].zscores
        l2 = self.tr.ldscores
        n1, n2 = float(self.nsamp[0]), float(self.nsamp[1])
        nsnps_bin = self.tr.nsnps_bin
        
        if self.intercept is None and self.phenos is None:
            self.gamma_g, self.c_opt = utils.bivariate_regression_partitioned_jn(l2, z1 * z2, 1.0 / l2, self.nblks, n1, n2, self.tr.nsnps_blk)
            print(self.gamma_g)
        else:
            # additional information to constrain moments
            if self.intercept is not None:
                self.log._log(" --intercept-rg: Constraining the intercept N*gamma_e / sqrt(N1*N2) [WARNING: This may bias the genetic correlation estimate!]")
            elif self.phenos is not None:
                self.log._log(" --pheno-rg: Using overlapping sample covariance [WARNING: This may bias the genetic correlation estimate!]")
            
            pred_tr = self.tr._calc_trace_rg(self.nsamp[0], self.nsamp[1], self.n_overlap)
            rhs = self._calc_rhs_rg(self.nsamp[0], self.nsamp[1])
            if self.verbose:
                labels = [f"gamma_g{i}" for i in range(self.nbins)] + ["c_opt"]
                labels_str = "[" + ", ".join(labels) + "]\n"
                self.log._log("Normal equation:\n" + np.array2string(pred_tr[self.nblks], precision=2, separator=', ') + "\n\t*\n" + labels_str + "\t=\n" 
                    + np.array2string(rhs[self.nblks], precision=2, separator=', '))
            for i in range(self.nblks + 1):
                h2_est = utils._solve_linear_equation(pred_tr[i], rhs[i])
                self.gamma_g[i] = h2_est[0]

        self.gamma_g_se = utils._calc_jackknife_se(self.gamma_g)[1]
        print(self.herits[0, :, 0].shape)
        self.rg = self.gamma_g / np.sqrt(self.herits[0, :, 0]*self.herits[1, :, 0])
        self.rg_se = utils._calc_jackknife_se(self.rg)[1]

        self.log._log("\n=== SUM-CORE genetic covariance & correlation ===")
        if self.nbins > 1:
            for j, header in enumerate(self.annot_header):
                self.log._log(
                    f"^^^ [{self.phen_names[0]} & {self.phen_names[1]}] "
                    f"Estimated genetic cov. bin {header} "
                    rf"(γ_g_{header}): {self.gamma_g[-1, j]:.6f} "
                    rf"SE: {self.gamma_g_se[j]:.6f}"
                )
            for j, header in enumerate(self.annot_header):
                self.log._log(
                    f"^^^ [{self.phen_names[0]} & {self.phen_names[1]}] "
                    f"Estimated genetic cor. bin {header} "
                    rf"(r_g_{header}): {self.rg[-1, j]:.6f} "
                    rf"SE: {self.rg_se[j]:.6f}"
                )
        else:
            self.log._log(
                f"^^^ [{self.phen_names[0]} & {self.phen_names[1]}] "
                rf"Estimated genetic cov. (γ_g): {np.squeeze(self.gamma_g[-1]):.6f} "
                rf"SE: {np.squeeze(self.gamma_g_se):.6f}"
            )
            self.log._log(
                f"^^^ [{self.phen_names[0]} & {self.phen_names[1]}] "
                rf"Estimated genetic cor. (r_g): {np.squeeze(self.rg[-1]):.6f} "
                rf"SE: {np.squeeze(self.rg_se):.6f}"
            )


    def _logoff(self):
        for i, name in enumerate(self.phen_names):
            if self.nbins > 1:
                for j, header in enumerate(self.annot_header):
                    self.log._log(
                        f"^^^ [{name}] Estimated partitioned heritability bin {header} "
                        f"(h^2_{header}): {self.hsums[i, j, 0]:.5f} SE: {self.hsums[i, j, 1]:.5f}"
                    )
            self.log._log(
                f"^^^ [{name}] Estimated total heritability (h^2): {self.hsums[i, -1, 0]:.5f} SE: {self.hsums[i, -1, 1]:.5f}"
            )

        self.end_time = utils._get_time()
        self.log._log("Analysis ended at: " + utils._get_timestr(self.end_time))
        self.log._log("run time: " + format(self.end_time - self.start_time, '.3f') + " s")
        if self.out is not None:
            self.log._log("Saved log in " + self.out + ".log")
            self.log._save_log(self.out + ".log")
        return
    
    def _calc_rhs_rg(self, n1, n2):
        """
        Return the RHS of the normal equation for genetic cov estimation. Only used when intercept is specified or phenos are provided.
        """
        a = np.zeros(self.nblks + 1) if self.intercept is None else self.intercept * np.ones(self.nblks + 1)*np.sqrt(n1 * n2)
        b = np.zeros(self.nblks + 1)

        for j in range(self.nblks+1):
            if (j < self.nblks):
                mask = np.ones(self.nsnps, bool)
                start = j*self.tr.blk_size
                end   = start + self.tr.blk_size
                mask[start:end] = False
                b[j] = np.dot(self.sums[0].zscores[mask], self.sums[1].zscores[mask]) * np.sqrt(n1 * n2) / self.tr.nsnps_blk[j]
            else:
                b[j] = np.dot(self.sums[0].zscores, self.sums[1].zscores) * np.sqrt(n1 * n2) / self.tr.nsnps_blk[j]
            if (self.phenos is None):
                b[j] -= a[j]
        
        ncols = self.nbins+1 if self.phenos is not None else self.nbins
        if self.phenos is not None:
            rhs = np.full((self.nblks+1, ncols), self.cov*self.n_overlap) # solve the SCORE MoM, since we can do it
            rhs[:, 0] = b
        else:
            rhs = b.reshape(self.nblks+1, 1)
        return rhs
    
    def estimate_overlap_cov(self):
        paths = self.phenos.split(",")
        if len(paths) != 2:
            self.log._log("ERROR: --pheno-rg must be exactly two comma‐separated phenotype files.")
            exit(1)

        df1 = pd.read_csv(paths[0], sep=r'\s+', header=0)
        df2 = pd.read_csv(paths[1], sep=r'\s+', header=0)
        
        if (len(df1.columns) != 3 or len(df2.columns) != 3):
            self.log._log("ERROR: The phenotype files must have FID IID (pheno) as columns")
            exit(1)

        pheno1 = df1.columns[2]
        pheno2 = df2.columns[2]

        df1 = df1.rename(columns={pheno1: 'pheno1'}).dropna()
        df2 = df2.rename(columns={pheno2: 'pheno2'}).dropna()
        
        merged = pd.merge(df1[['FID', 'IID', 'pheno1']],
                        df2[['FID', 'IID', 'pheno2']],
                        on=['FID', 'IID'], how='inner')
        
        self.n_overlap = merged.shape[0]
        
        if self.n_overlap < 2:
            self.log._log("NOTE: There are no (or only 1) overlapping individuals in the provided phenotypes.")
            self.cov = .0
        else:
            self.cov = np.cov(merged['pheno1'], merged['pheno2'], ddof=1)[0, 1]
        
        self.log._log(f"Sample covariance (y1^T y2) of overlapping ({self.n_overlap}) individuals: {self.cov:.5f}")