"""
Stochastically estimate (partitioned) genome-wide LD scores. Some part of the code is modified from Eric Liu's script
"""
import utils
import numpy as np
import pandas as pd
from bed_reader import open_bed
import multiprocessing as mp
from tqdm import tqdm
import scipy
import sys
import gc

def read_cov(
        cov_filename: str,
        fam_filename: str,
        std: bool = True,
        cov_impute_method: str = "ignore",
        one_hot_conversion: bool = False,
        categorical_threshold: int = 100,
        logger = None,
        verbose = False
    ):
    """
    1) Read PLINK .fam to get FID/IID sample order.
    2) Read covariate file, merge on FID/IID (error if mismatch).
    3) Drop FID, IID, handle missingness/imputation.
    4) Optionally one-hot encode categoricals.
    5) If std=True, center & scale each covariate column.
    6) Return:
         C : (n_samples x n_covariates) array,
         R : (n_covariates x n_samples) regression matrix = (C^T C)^{-1} C^T
    """
    # 1) load .fam
    fam = pd.read_csv(
        fam_filename,
        sep=r'\s+',
        header=None,
        usecols=[0,1],
        names=['FID','IID']
    )

    # 2) load covariate file
    cov = pd.read_csv(cov_filename, sep=r'\s+')
    merged = fam.merge(cov, on=['FID','IID'], how='right', indicator=True)
    missing = merged.loc[merged['_merge'] != 'both', ['FID','IID']]
    if not missing.empty:
        if verbose:
            raise ValueError(
                f"Samples in {fam_filename} not found in {cov_filename}:\n"
                f"{missing.to_string(index=False)}"
            )
        else:
            raise ValueError(
                f"!!! {len(missing)} Samples are not found in {cov_filename} !!!"
            )
    merged = merged.drop(columns=['_merge'])

    # 3) drop IDs, handle missingness
    df = merged.drop(columns=['FID','IID']).copy()
    n_covariates = len(df.columns)
    is_na = df.isin(['NA', -9]).any(axis=1)
    if cov_impute_method == "ignore":
        if is_na.any():
            idx = np.where(is_na)[0].tolist()
            raise ValueError(f"Missing covariate entries at rows: {idx}")
    else:
        df.replace({'NA': np.nan, -9: np.nan}, inplace=True)
        for col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    # 4) one-hot encode if requested
    if one_hot_conversion:
        for col in df.columns:
            if df[col].nunique() <= categorical_threshold:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = df.drop(columns=[col]).join(dummies)

    # 5) standardize if requested
    if std:
        df = (df - df.mean()) / df.std(ddof=1)

    C = df.values  # shape (n_samples, n_cov)

    # 6) build regression matrix R = (C^T C)^{-1} C^T
    CtC = C.T @ C
    inv_CtC = np.linalg.inv(CtC)
    R = inv_CtC @ C.T  # shape (n_cov, n_samples)
    
    if logger:
        logger._log(
            f"Read {cov_filename} for {n_covariates} covariates (samples merged with {fam_filename}).\n"
            f"C shape={C.shape}, R shape={R.shape}, std={std}, one_hot={one_hot_conversion}, verbose={verbose}"
        )

    return C, R


class GenomewideLDScore:
    def __init__(self,
                 bed_path,
                 annot_path,
                 out_path,
                 log,
                 covar_path=None,
                 num_vecs=10,
                 num_workers=4,
                 step_size=1000,
                 seed=None,
                 verbose=False):
        self.G = open_bed(bed_path + ".bed")
        self.nsamp, self.nsnps = self.G.shape
        self.nvecs = num_vecs
        self.nworkers = num_workers
        self.step_size = step_size
        self.log = log
        self.verbose = verbose

        # read .bim and annotation
        self._read_bim(bed_path + ".bim")
        if annot_path is not None:
            self._read_annot(annot_path)
        else:
            self._read_annot(None)

        # read & build covariate residualizer
        if covar_path is not None:
            fam_file = bed_path + ".fam"
            self.C, self.cov_R = read_cov(
                cov_filename       = covar_path,
                fam_filename       = fam_file,
                std                = True,
                cov_impute_method  = "ignore",
                one_hot_conversion = False,
                categorical_threshold = 100,
                logger             = self.log,
                verbose            = self.verbose
            )
        else:
            self.C = None
            self.cov_R = None
            self.log._log("No covariate correction will be applied.")

        self.root_seed = seed
        self.outpath = out_path

    def _compute_Xz_blk(self, blk_idxs):
        j, blk_start, blk_end, idxs = blk_idxs
        nsnps = sum(len(binidx) for binidx in idxs)
        Xz = np.zeros((self.nbins, self.nsamp, self.nvecs))

        rng = np.random.default_rng([j, self.root_seed] if self.root_seed is not None else None)
        Zs = rng.standard_normal(size=(nsnps, self.nvecs))

        # read + standardize
        geno = self.G.read(index=np.s_[:, blk_start:blk_end])
        means = np.nanmean(geno, axis=0)
        stds  = np.nanstd(geno, axis=0)
        geno  = (geno - means) / stds
        geno[np.isnan(geno)] = 0
        geno = np.array(geno, order='F')

        # regress out covariates if present
        if self.C is not None:
            geno = geno - self.C.dot(self.cov_R.dot(geno))

        Zs = np.array(Zs, order='F')
        for k, binidx in enumerate(idxs):
            Xz[k, :, :] = scipy.linalg.blas.sgemm(1.0, geno[:, binidx], Zs[binidx, :])

        return Xz
   

    def _compute_XtXz_blk(self, blk_idx):
        """
        For blk genotype, multiply with X_k z to get XtXkz.
        """
        blk_start, blk_end = blk_idx

        geno = self.G.read(index=np.s_[:, blk_start:blk_end])
        means = np.nanmean(geno, axis=0)
        stds = np.nanstd(geno, axis=0)

        geno = (geno-means)/stds
        geno[np.isnan(geno)] = 0

        if self.C is not None:
            geno = geno - self.C.dot(self.cov_R.dot(geno))

        ## TODO: benchmark sgemm vs. np broadcasting - in small scale, looks like sgemm is faster (could be b/c sgemm is used in Xz estimation)
        geno_t = np.array(geno.T, order='F')
        XtXz = np.zeros((blk_end - blk_start, self.nbins, self.nvecs))
        for k in range(self.nbins):
            XtXz[:, k, :] = scipy.linalg.blas.sgemm(1.0, geno_t, self.Xz[k])

        #XtXz = np.einsum('nm,knb->mkb', geno, self.Xz)
        return (blk_start, blk_end, XtXz)


    def _read_annot(self, annot_path):
        """
        Read in the annotation. If the file includes a header, save it as the names for the annotations.
        If not, then have dummy names and read in the annotation.
        """
        if (annot_path is None):
            self.l2cols = None
            self.annot = np.ones((self.nsnps, 1))
            self.log._log("Calculating genome-wide (non-partitioned) LD score")
        else:
            self.l2cols, self.annot = utils._read_with_optional_header(annot_path)
            if (self.annot.ndim == 1):
                self.annot = self.annot.reshape(-1, 1)
            self.log._log("Read SNP partition annotation of dimensions "+str(self.annot.shape))
        
        if (self.nsnps != self.annot.shape[0]):
            self.log._log(f"!!! number of SNPs in annotation ({self.annot.shape[0]}) does not match the input genotype file ({self.nsnps}) !!!")
            sys.exit(1)
        self.nbins = self.annot.shape[1]
        if (self.l2cols is None):
            self.l2cols = ['L2_'+str(i) for i in range(self.annot.shape[1])]
        else:
            self.l2cols = [i + 'L2' for i in self.l2cols]
        self.log._log(f"Nbins: {self.nbins}")
        self.nsnps_bin = self.annot.sum(axis=0)

    def _read_bim(self, bim_path):
        if (bim_path is None):
            self.log._log("No .bim file is provided; all (anonymous) SNPs will be used")
            self.snplist = None
        else:
            self.log._log(f"Reading {bim_path} for SNPs")
            self.snplist = pd.read_csv(bim_path, header=None, sep='\t')
        if (len(self.snplist) != self.nsnps):
            self.log._log(f"!!! The number of SNPs in the .bed file ({self.nsnps}) does not match the .bim file ({len(self.snplist)}) !!!")
            sys.exit(1)
    
    def _partition_index(self, snpidx, annot) -> list[np.ndarray]:
        """
        partition snp indices by annotation
        """
        return [snpidx[annot[:, c] == 1] for c in range(self.nbins)]


    def _compute_ldscore(self):
        """
        Use multi-processing to calculate the X_j^T X_k Z.
        General sketch: read in each block of genotype, calculate X_k Z for that blk. Aggregate X_k through all blks.
        Then re-read each blk from the start, multiply by the previous result (loop over k) to get X_j ^ T X_k (no need for agg this time).
        """
        self.start_time = utils._get_time()
        self.log._log("Genome-wide LD score calculation started at: "+utils._get_timestr(self.start_time))
        self.log._log(f"num_vecs: {self.nvecs}, num_workers: {self.nworkers}, step_size: {self.step_size}, seed: {self.root_seed}")

        self.nblks = len(np.arange(self.nsnps)[::self.step_size])
        Xz_input = []
        XtXz_input = []
        for j in range(self.nblks):
            idx_start = self.step_size*j
            idx_end = self.nsnps if j==self.nblks-1 else self.step_size*(j+1)
            annot_blk = self.annot[idx_start:idx_end]
            Xz_input.append((j, idx_start, idx_end, self._partition_index(np.arange(len(annot_blk)), annot_blk)))
            XtXz_input.append((idx_start, idx_end))
        
        self.Xz = np.zeros((self.nbins, self.nsamp, self.nvecs))

        with mp.Pool(self.nworkers) as pool:
            with tqdm(total=self.nblks) as pbar:
                pbar.set_description('Calculating Xz')
                for result in pool.imap_unordered(self._compute_Xz_blk, Xz_input):
                    self.Xz += result
                    gc.collect()
                    pbar.update()
        pool.join()

        self.Xz_time = utils._get_time()
        self.log._log("Calculation of Xz (for each partition) completed. Runtime: "+format(self.Xz_time - self.start_time, '.3f')+" s")

        self.XtXz = np.zeros((self.nsnps, self.nbins, self.nvecs))

        with mp.Pool(self.nworkers) as pool:
            with tqdm(total=self.nblks) as pbar:
                pbar.set_description('Calculating XtXz')
                for result in pool.imap_unordered(self._compute_XtXz_blk, XtXz_input):
                    idx_start, idx_end, XtXz_blk = result
                    self.XtXz[idx_start:idx_end, :, :] = XtXz_blk
                    gc.collect()
                    pbar.update()
        pool.join()

        self.XtXz = self.XtXz / self.nsamp

        self.XtXz_time = utils._get_time()
        self.log._log("Calculation of XtXz (for each partition) completed. Runtime: "+format(self.XtXz_time - self.Xz_time, '.3f')+" s")

        self.log._log("Converting XtXz into genome-wide (partitioned) LD scores.")
        self.gwldscore = self.nsamp/(self.nsamp+1) * (np.square(self.XtXz).mean(axis=2) - self.nsnps_bin/self.nsamp)
        self.log._log(f"Saving the genome-wide (partitioned) LD scores into: {self.outpath}.gw.ldscore.gz")
        snpcols = ['CHR', 'SNP', 'BP']
        if (self.snplist is None):
            self.snpdf = pd.DataFrame(np.nan*np.ones((self.nsnps, 3)), columns=snpcols)
        else:
            self.snpdf = self.snplist.iloc[:, :3]
            self.snpdf.columns = snpcols
        
        self.gwldscore = pd.DataFrame(self.gwldscore, columns = self.l2cols)
        self.gwldscore = pd.concat([self.snpdf, self.gwldscore], axis=1)
        self.gwldscore.to_csv(f'{self.outpath}.gw.ldscore.gz', index=False, compression='gzip', sep='\t', float_format='%.3f')
        self.end_time = utils._get_time()
        self.log._log(f"Calculation of genome-wide LD score ended at "+utils._get_timestr(self.end_time))
        self.log._log("Runtime: "+format(self.end_time - self.start_time, '.3f')+" s")
        self.log._save_log(self.outpath+".gw.log")

        





        



        

            