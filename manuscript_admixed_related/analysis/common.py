"""Helpers shared by the scripts of this folder: shell commands, PLINK 2 GWAS to SUMMIT summary statistics, and transforms."""
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

GLM = ('--glm hide-covar no-firth cols=chrom,pos,ref,alt,omitted,nobs,beta,se --covar-variance-standardize '
       '--no-input-missing-phenotype')
ONE_THREAD = 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1'


def run(cmd, log=None):
    """Run a shell command; on failure raise with the log file to read."""
    with open(log or os.devnull, 'w') as fh:
        r = subprocess.run(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd[:200]}' + (f' (see {log})' if log else ''))


def run_all(cmds, threads):
    """Run independent commands side by side; raise if any fails."""
    cmds = [c for c in cmds if c]
    with ThreadPoolExecutor(threads) as ex:
        codes = list(ex.map(lambda c: subprocess.run(c, shell=True).returncode, cmds))
    failed = [c for c, r in zip(cmds, codes) if r]
    if failed:
        raise RuntimeError(f'{len(failed)} of {len(cmds)} commands failed, e.g. {failed[0][:200]}')


def read_fam(prefix):
    return pd.read_csv(prefix + '.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)


def gwas(plink2, bfile, pheno, names, covar, cov_rank, out, threads=8):
    """PLINK 2 linear GWAS of every phenotype in `names`, written as SUMMIT summary statistics OUT/<name>.tsv
    (SNP A1 A2 N BETA SE COV_RANK; COV_RANK counts the covariates without the intercept)."""
    tmp = f'{out}.glm'
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    run(f'{plink2} --bfile {bfile} --pheno {pheno} --pheno-name {",".join(names)} --covar {covar} {GLM} '
        f'--out {tmp}/g --threads {threads}', log=f'{out}.log')
    for n in names:
        g = pd.read_csv(f'{tmp}/g.{n}.glm.linear', sep='\t', dtype={'ID': str})
        g = g[g.BETA.notna() & g.SE.notna()]
        pd.DataFrame({'SNP': g.ID, 'A1': g.A1, 'A2': g.OMITTED, 'N': g.OBS_CT, 'BETA': g.BETA, 'SE': g.SE,
                      'COV_RANK': cov_rank}).to_csv(f'{out}/{n}.tsv', sep='\t', index=False, float_format='%.10g')
    shutil.rmtree(tmp)


def irnt(x):
    """Inverse rank-normal transform of the non-missing values (ties averaged)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x)
    out[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / ok.sum())
    return out


def liability_factor(prevalence):
    """Observed (0/1) to liability scale factor for a trait with the given sample prevalence."""
    k = float(prevalence)
    return k * (1 - k) / norm.pdf(norm.isf(k)) ** 2


def read_h2_results(path):
    """SUMMIT --h2 results table as phen, h2, se."""
    r = pd.read_csv(path, sep='\t', dtype={'phen': str})
    return r[['phen', 'h2', 'h2_se']].rename(columns={'h2_se': 'se'})
