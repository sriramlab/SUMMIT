'''
Read the phenotype-specific summary statistics (generally PLINK)
The standard LDSC summary statistics format (.sumstat) works.
Format should be:
    SNPID, NMISS (or OBS_CT), Z
'''
import utils

import numpy as np
import pandas as pd
from os import listdir
import sys

class Sumstats:
    '''
    Sumstats class is called in the sums (sums.py) module. It's called after the trace module, which will then call this module
    in order to read in the phenotype sumstats. Current version can store only a single phenotype sumstats at a time (to save memory)
    '''
    def __init__(self, nblks=100, chisq_threshold=0, log=None, both_side=False, annot_df=None, nbins=1):
        self.log = log
        self.nblks = nblks
        self.nbins = nbins
        # self.snplist = snplist
        # if snplist is None:
        #     self.log._log("!!! Missing the list of SNPs used in trace calculation."+\
        #           " All SNPs in the phenotype sumstats will be used. !!!")
        # else:
        #     self.nsnps_trace = len(snplist)
        self.annot_df = annot_df
        self.snpids = len(annot_df)
        self.annot = None # annotation for partitioned heritability (if None, assume single-bin)
        self.zscores = []
        self.zscores_bin = [] # list of list, where zscores are partitioned by bin assignment
        self.zscores_blk = [] # list of list of list, where zscores are partitioned by blk & bin assignment
        self.RHS = None # array of RHS for each pheno
        self.nsamp = 0
        self.nsnps = 0
        self.nsnps_blk = None # array for keeping track of number of SNPs in each leave-one-out blk
        self.nsnps_bin = None # total number of SNPs for each bin
        self.chisq_threshold = chisq_threshold # if positive value, then remove snps with chisq above it
        self.matched_snps = None # array of matched SNPs
        self.name = None
        self.removesnps = [] ## store any SNPs that are removed from estimation

    def _read_sumstats(self, path, name):
        ''' Read in summary statistics for a single phenotype '''
        self.name = name
        sumdf = pd.read_csv(path, sep=r'\s+')

        # drop any row with missing N or Z
        ncol = utils._parse_column_name(sumdf, ['N','n'], 3)
        zcol = utils._parse_column_name(sumdf, ['Z','z'], 3)
        idcol = utils._parse_column_name(sumdf, ['ID','id','snp','SNP'], 0)
        drop_mask = sumdf[ncol].isna() | sumdf[zcol].isna()

        # keep those SNP IDs for both-sides filtering
        self.removesnps = sumdf.loc[drop_mask, idcol].dropna().astype(str).tolist()

        self.log._log(f"Dropping {len(self.removesnps)} SNPs with NA values.")

        sumdf = sumdf.loc[~drop_mask]

        sumdf = sumdf.rename(columns = {idcol: 'SNP', zcol: 'Z', ncol: 'N'})

        sumdf['Z'] = sumdf['Z']*np.sqrt(sumdf['N']/sumdf['N'].max())

        self.sumdf = sumdf[['SNP','Z']].copy()
        self.nsamp = float(sumdf['N'].max())

    def _match_snps(self):
        ''' 
        match SNPs used for trace calculations & summary statistics
        '''
        matched_zscores = []
        nsnps_blk = np.zeros((self.nblks, self.nbins))

        if (self.chisq_threshold is not None):
            self.log._log(f"Filtering SNPs with chi-sq greater than {self.chisq_threshold}")
            chisq = self.sumdf['Z']**2
            keep  = (chisq <= self.chisq_threshold) & (~chisq.isna())
            chisq_snps = self.sumdf.loc[~keep, 'SNP'].tolist()
            self.removesnps += chisq_snps
            self.sumdf = self.sumdf.loc[keep]
            self.log._log(f"Removed {len(chisq_snps)} SNPs with chi-sq above the threshold {self.chisq_threshold} ({len(self.sumdf)} SNPs remaining)")

        # if (self.snplist is None):
        #     # using all the SNPs
        #     for i in range(self.nblks):
        #         blk_size = self.nsnps//self.nblks
        #         blk_zscores = np.array(self.zscores[blk_size*i: blk_size*(i+1)] if (i < self.nblks-1) else self.zscores[blk_size*i:])
        #         blk_annot = self.annot[blk_size*i:blk_size*(i+1)] if (i < self.nblks-1) else self.annot[blk_size*i:]
        #         partition, nsnps_partition = utils._partition_bin_non_overlapping(blk_zscores, blk_annot, self.nbins)
        #         matched_zscores.append(partition)
        #         nsnps_blk[i] = nsnps_partition

        #     self.log._log("Using " + str(self.nsnps) + " SNPs to calculate the RHS of the normal equation...")
        #     # overall histogram on the full set
        #     self.zscores_bin, self.nsnps_bin = utils._partition_bin_non_overlapping(self.zscores, self.annot, self.nbins)

        # else: # FIXME: snplist is now always provided!
        
        # match summary statistics with annot_df
        df = self.sumdf.merge(self.annot_df, how='inner', on='SNP')
        missing = set(self.annot_df['SNP']) - set(df['SNP'])
        if missing:
            self.removesnps.extend(list(missing))
        self.matched_snps = np.array(df['SNP'])
        self.log._log(f"Matched {len(df)} SNPs in phenotype {self.name} out of {len(self.annot_df)} annotated SNPs ({len(missing)} missing or filtered)")
    
        # align z-scores & annotations to the filtered list
        all_z    = df['Z'].values
        ann_cols = df.columns[-self.nbins:]
        all_ann  = df[ann_cols].values
        total = len(df)
        
        blk_size = total//self.nblks
        
        for i in range(self.nblks):
            start = blk_size*i
            end = blk_size*(i+1) if (i < self.nblks-1) else total

            blk_zscores = all_z[start:end]
            blk_annot = all_ann[start:end]

            partition, nsnps_partition = utils._partition_bin_non_overlapping(blk_zscores, blk_annot, self.nbins)
            matched_zscores.append(partition)
            nsnps_blk[i] = nsnps_partition

        self.zscores_bin, self.nsnps_bin = utils._partition_bin_non_overlapping(all_z, all_ann, self.nbins)

        # check if any bins are empty
        for b in range(self.nbins):
            if (self.nsnps_bin[b] == 0):
                self.log._log(f"!!! Bin {b} contains zero SNPs after matching. Please check your annotation and SNP lists. !!!")
                sys.exit(1)

        # store block‐wise results
        self.nsnps_blk = nsnps_blk
        self.zscores_blk = matched_zscores            
        
    def _calc_rhs_h2(self):
        ''' calculate the RHS of the normal equation in h2 calculation '''
        self.rhs = np.full((self.nblks+1, self.nbins+1), self.nsamp)
        for i in range(self.nbins):
            total_zTz = np.dot(self.zscores_bin[i], self.zscores_bin[i])
            for j in range(self.nblks+1):
                if (j < self.nblks):
                    blk_zTz = np.dot(self.zscores_blk[j][i], self.zscores_blk[j][i])
                    self.rhs[j, i] = (total_zTz - blk_zTz)*self.nsamp/(self.nsnps_bin[i] - self.nsnps_blk[j][i])
                else:
                    self.rhs[j, i] = total_zTz*self.nsamp/self.nsnps_bin[i]
        self.log._log("Calculated the RHS for phenotype "+self.name)
        
    #def _calc_RHS_rg(self):
        ''' calculate the RHS of the normal equation in rg calculation '''
        
    def _process(self, path, name):
        ''' 
        process a sumstat provided in the path
        '''
        self._read_sumstats(path, name)
        self._match_snps()
        self._calc_rhs_h2()
        return self.removesnps