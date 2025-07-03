'''
Read in the trace (or LD) summary statistics to estimate the trace for the summary stats.
For .tr file, there should be (njackknife+1) rows excluding the header, where
the last row is the trace summary (i.e., sum of the LD scores) from the entire genotype.
The SNP sets should be the same for all the outputs -- it is recommended to input .bim for the list of SNPs (read in once)

For .ldscore.gz, standard LDSC ld score format works.
'''
import utils

import numpy as np
import pandas as pd
from os import listdir, path
import sys

class Trace:
    def __init__(self, bimpath = None, sumpath=None, savepath=None, log=None, ldscores=None, nblks=100, annot=None, verbose=False):
        self.log = log
        self.sumpath = sumpath
        self.savepath = savepath
        self.ldscorespath = ldscores
        self.ldscores = None
        self.sums = []
        self.nblks = nblks # nblks specified only if using ld proj; otherwise it'll be overwritten by trace summaries.
        self.ntrace = 0
        self.K = []
        self.snplist = None
        self.nsamp = [] # number of samples used for trace summaries; can be of varying size
        self.nsnps = 0 # number of SNPs is fixed
        self.nsnps_blk = None # array for keeping track of number of SNPs in each leave-one-out blk
        self.nsnps_bin = None
        self.verbose = verbose
        if (bimpath is None) or (bimpath == ""):
            if (self.ldscorespath is None):
                self.log._log("!!! SNP list (.bim) is required if using trace summaries (.tr) !!!")
        elif (bimpath.endswith(".bim")):
            with open(bimpath, 'r') as fd:
                for line in fd:
                    self.snplist.append(line.split()[1])
        else:
            self.log._log(f'!!! {bimpath} is not a .bim file! !!!')
        
        if (sumpath is not None):
            self._read_all_trace()
            if (savepath is not None):
                self._save_trace()
        elif (ldscores is not None):
            self._read_ldscores()
        
        self._read_annot(annot)
    
    def _read_annot(self, annot_path):
        # use single bin
        if (annot_path is None):
            header = np.array(['L2'])
            annot = np.ones((self.nsnps, 1))
            self.log._log("Running with single component annotation...")
        else:
            try: # try reading full annotation dataframe (.annot or .annot.gz)
                df = pd.read_csv(annot_path, sep=r'\s+', compression='infer')
                if 'SNP' not in df.columns:
                    raise ValueError("!!! Input annotation file is not in correct format !!!")
                # annotation bins are all cols after the first three metadata columns ('CHR', 'SNP', 'BP')
                annot_cols = df.columns.tolist()[3:]
                self.annot_header = np.array(annot_cols)
                annot_df = df[['SNP'] + annot_cols].copy()
                
                overlap = [snp for snp in self.snplist if snp in annot_df['SNP'].values]
                missing = set(self.snplist) - set(overlap)
                if missing:
                    self.log._log(f"Dropping {len(missing)} SNPs from annotation as they are missing LD information.")
                    
                annot_df = (annot_df.set_index('SNP').loc[overlap].reset_index())
                self.annot_df = annot_df
                self.annot = annot_df[annot_cols].values
                self.log._log("Read full annotation of shape " + str(self.annot.shape))
                
                # prune LD scores if present
                if getattr(self, 'ldscores', None) is not None:
                    ld_df = (self.ldscores_df.set_index('SNP').loc[overlap].reset_index())
                    self.ldscores_df = ld_df
                    self.ldscores = ld_df.iloc[:, 3:].to_numpy()
                    self.nsnps = len(overlap)
                    self.snplist = overlap
                    self.log._log(f"Pruned LD‐score to {self.nsnps} SNPs that match with the annotation file.")
            
            except ValueError: # try reading thin annot
                if (self.snplist is None):
                    raise ValueError("!!! Thin annotation requires a BIM/snplist when using trace-summaries !!!")
                header, annot = utils._read_with_optional_header(annot_path)
                if annot.ndim == 1: # single bin
                    annot = annot.reshape(-1, 1)
                else:
                    annot = annot.reshape(-1, annot.shape[-1])
                if header is None:
                    header = np.array([f'bin_{i}' for i in range(annot.shape[1])])
                cols = header.tolist()
                
                if (self.nbins != annot.shape[1]) or (self.nsnps != annot.shape[0]):
                    self.log._log("!!! number of components in annotation does not match the input trace/LD summary !!!")
                    sys.exit(1)
                
                annot_df = pd.DataFrame(annot, index=self.snplist, columns=cols)
                annot_df.reset_index(inplace=True)
                annot_df.rename(columns={'index':'SNP'}, inplace=True)
                self.annot_header = np.array(cols)
                self.annot_df = annot_df
                self.annot = annot_df[cols].values
                self.log._log("Read thin annotation matrix of shape " + str(self.annot.shape))      
    
    def _save_trace(self):
        ''' Save trace summaries as a file'''
        with open(self.savepath+".MN", 'w') as fd:
            fd.write("NSAMPLE,NSNPS,NBLKS,NBINS,K\n")
            fd.write(f"{self.nsamp:.0f},{self.nsnps:.0f},{self.nblks:.0f},{self.nbins:.0f},{self.effective_K:.0f}")

        with open(self.savepath+".tr", 'w') as fd:
            header_str = ','.join(f'LD_SUM_{i:d}' for i in range(self.nbins))
            fd.write(header_str+",NSNPS_JACKKNIFE\n")
            for j in range(self.nblks+1):
                for k in range(self.nbins):
                    row_str = ','.join(f'{self.sums[j,k,l]:.3f}' for l in range(self.nbins))
                    row_str += f',{self.nsnps_blk[j, k]:.0f}\n'
                    fd.write(row_str)
        self.log._log(f"Saved trace summary into {self.savepath}(.tr/.MN)")
    
    def _read_trace(self, filename, idx):
        '''
        Read trace summaries (block-wise LD scores)
        '''
        # read in metadata
        nsamp, nsnps, nblks, nbins = 0, 0, 0, 0
        with open(filename+".MN", 'r') as fd:
            next(fd)
            nsamp, nsnps, nblks, nbins, K = map(int, fd.readline().split(','))
        self.nsamp.append(nsamp)
        self.K.append(K)

        if not idx:
            self.nsnps = nsnps
            self.nblks = nblks
            self.nbins = nbins
        elif (nblks != self.nblks):
            self.log._log("!!! Trace summary "+filename+" has incorrect number of jackknife blocks !!!")
            return
        elif (nbins != self.nbins):
            self.log._log("!!! Trace summary "+filename+" has incorrect number of annotation bins !!!")
            return

        sums = np.zeros((self.nblks+1, self.nbins, self.nbins))
        nsnps_blk = np.zeros((self.nblks+1, self.nbins))
        
        # read in trace values
        for cnt, vals in enumerate(utils._read_multiple_lines(filename+".tr", self.nbins)):
            sums[cnt] = vals[:, :-1]
            nsnps_blk[cnt] = vals[:, -1].transpose()

        if not idx:
            self.nsnps_blk = nsnps_blk
        elif not (np.array_equal(self.nsnps_blk, nsnps_blk)):
            self.log._log("!!! Trace summary "+filename+" has different annotations !!!")
            sys.exit(1)
        
        self.sums.append(sums)
        self.ntrace += 1

        self.log._log(f"Read in trace summaries from {filename} generated with {K} random vectors")
        if (self.verbose):
            self.log._log("-- avg. jackknife LDscore sum:\n"+np.array2string(sums[:-1].mean(axis=0), precision=3, separator=', ')\
                +"\n-- number of jackknife blocks:\t"+str(self.nblks)\
                +"\n-- genome-wide LDscore sum:\n"+np.array2string(sums[-1], precision=3, separator=', '))
        return self.sums
        
    def _read_all_trace(self):
        trace_files = [self.sumpath]
        if path.isdir(self.sumpath):
            prefix = self.sumpath.rstrip('/') + "/"
            trace_files = sorted([prefix + f.rstrip('.tr') for f in listdir(self.sumpath) if f.endswith('.tr')])
        for i, f in enumerate(trace_files):
            self._read_trace(f, i)
        self.log._log("Finished reading "+str(len(trace_files))+" trace summaries.")
        if (np.std(self.sums, axis=0).sum() == .0) and (self.ntrace > 1):
            self.log._log("!!! Duplicate trace summaries are used -- effective number of random vectors remains unchanged. !!!")
            self.effective_K = np.min(self.K)
        else:
            self.effective_K = np.sum(self.K)
        ## TODO: is it correct to take a weighted average by sample size?
        self.sums = np.average(self.sums, axis=0, weights=self.nsamp)
        self.nsamp = np.mean(self.nsamp)

    def _read_ldscores(self):
        '''
        Read the LD score matrix (X^T Xz) instead of trace summaries. Works with either the (truncated) LDSC LD scores (.l2.ldscore.gz) or
        the genome-wide LD scores (.gw.ldscore.gz)
        '''
        self.ldscores_df = pd.read_csv(self.ldscorespath, compression='gzip', sep=r'\s+', index_col=False)
        self.ldscores = self.ldscores_df.iloc[:, 3:].to_numpy()
        self.snplist = self.ldscores_df['SNP'].to_numpy()
        self.nsnps = self.ldscores.shape[0]
        self.nbins = self.ldscores.shape[1]
        self.log._log("Loaded the LD score matrix with "+str(self.nsnps)+" SNPs and "+\
                        str(self.nbins)+" bins")

    def _calc_trace(self, nsample):
        self.log._log("Calculating trace...")
        ## ensure that annotation dimension matches that of the trace summaries or LD scores
        if (self.ldscores is not None):
            return self._calc_trace_from_ldscores(nsample)
        else:
            return self._calc_trace_from_sums(nsample)
    
    def _calc_trace_from_sums(self, N):
        trace = np.full((self.nblks+1, self.nbins+1, self.nbins+1), N)
        for k in range(self.nbins):
            for l in range(self.nbins):
                for j in range(self.nblks+1):
                    trace[j, k, l] = utils._calc_trace_from_ld(self.sums[j, k, l], N, self.nsnps_blk[j, k], self.nsnps_blk[j, l])
        return trace
    
    def _calc_trace_rg(self, nsample1, nsample2, noverlap):
        self.log._log("Calculating trace for genetic correlation...")
        ## ensure that annotation dimension matches that of the trace summaries or LD scores
        # TODO: for now only take in snp-level ld score as input, but also allow for block-level trace summaries as input
        ncols = self.nbins+1 if noverlap is not None else self.nbins
        fills = noverlap if noverlap is not None else np.sqrt(nsample1*nsample2)
        trace = np.full((self.nblks+1, ncols, ncols), fills)
        for k in range(self.nbins):
            for l in range(self.nbins):
                ld_sum = self.ldscores[self.annot[:, k]==1][:, l].sum()
                for j in range(self.nblks+1):
                    idx_start = self.blk_size*j
                    idx_end = self.nsnps if j==self.nblks-1 else self.blk_size*(j+1)
                    annot_jn = self.annot[idx_start:idx_end]
                    ldscores_jn = self.ldscores[idx_start:idx_end]
                    ld_sum_jn = ld_sum - ldscores_jn[annot_jn[:, k]==1][:, l].sum()
                    if (j == self.nblks):
                        ld_sum_jn = ld_sum
                    trace[j, k, l] = utils._calc_rg_trace_from_ld(ld_sum_jn, nsample1, nsample2, self.nsnps_blk[j, k], self.nsnps_blk[j, l])
                    if (noverlap is not None): # constrained version with N is available
                        trace[j, k, l] += noverlap
        return trace

    def _filter_snps(self, removesnps):
        '''
        Remove the SNPs in the removesnps from trace calculation. Only possible when LD scores are used as input.
        '''
        if self.ldscores is None:
            return
        
        mask = ~self.annot_df['SNP'].isin(removesnps)
        
        new_snps = self.annot_df.loc[mask, 'SNP'].tolist()
        self.nsnps = len(new_snps)

        self.annot = self.annot_df.loc[mask, list(self.annot_header)].to_numpy()
        self.ldscores = self.ldscores_df.loc[mask, self.ldscores_df.columns[3:]].to_numpy()

        self.blk_size = self.nsnps // self.nblks
        self.nsnps_bin = self.annot.sum(axis=0)

        self.nsnps_blk = np.full((self.nblks+1, self.nbins), self.nsnps_bin)
        for j in range(self.nblks):
            start = self.blk_size * j
            end   = self.blk_size*(j+1) if (j < self.nblks-1) else self.nsnps
            self.nsnps_blk[j] -= self.annot[start:end].sum(axis=0)

        n_removed = len(self.annot_df) - mask.sum()
        self.log._log(f"Filtered {n_removed} SNPs from the Trace module. Shape of final annotation used for analysis: {self.annot.shape}")
        if (n_removed/len(self.annot_df) > 0.01):
            self.log._log(f"[WARNING: Removing too many Trace SNPs will result in under-estimated heritability!]\n"+\
                "[We recommend using a better curated reference LD score panel with a more similar SNP set to the summary statistics SNPs.]")

            

    def _calc_trace_from_ldscores(self, N):
        trace = np.full((self.nblks+1, self.nbins+1, self.nbins+1), N)
        for k in range(self.nbins):
            for l in range(self.nbins):
                ld_sum = self.ldscores[self.annot[:, k]==1][:, l].sum()
                for j in range(self.nblks+1):
                    idx_start = self.blk_size*j
                    idx_end = self.nsnps if j==self.nblks-1 else self.blk_size*(j+1)
                    annot_jn = self.annot[idx_start:idx_end]
                    ldscores_jn = self.ldscores[idx_start:idx_end]
                    ld_sum_jn = ld_sum - ldscores_jn[annot_jn[:, k]==1][:, l].sum()
                    if (j == self.nblks):
                        ld_sum_jn = ld_sum
                    trace[j, k, l] = utils._calc_trace_from_ld(ld_sum_jn, N, self.nsnps_blk[j, k], self.nsnps_blk[j, l])
        return trace