"""Other methods that keep relatives, for one simulation task of simulate.py (everyone, related pairs kept in the data):

  reml       the two-kernel model of Zaitlen et al. (2013), Var(y) = s_g A + s_t B + s_e I, fitted by average-information REML
             without constraints, with the covariates as fixed effects. A is the GRM of the analysed SNPs; B keeps the GRM
             entries of the related pairs (relatedness above 0.0442) and the diagonal. (Zaitlen's GRM threshold of 0.05 is
             exceeded by many unrelated pairs in structured samples, where V is then not positive definite.)
             SNP heritability h2_g = s_g / (s_g + s_t + s_e), with a delta-method SE from the inverse AI matrix.
  covldsc    cov-LDSC (Luo et al. 2021): ldsc.py with a free intercept on the task's GWAS of everyone, with LD scores from
             cov.ldscore.py (in-sample, covariate-adjusted, 20-Mb windows; --ld window) or SUMMIT's genome-wide in-sample LD
             scores converted to the LDSC format (--ld genome); one component and the 8 MAF-LD bins.
REML costs about N^3 operations per iteration and a few N x N matrices of memory. Writes TASK/methods_<ld>.tsv.
usage: python compare_methods.py --cohort DIR --task DIR --covldsc DIR --ldsc-python PY [--ld window|genome] [--threads 8]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import scipy.linalg as sla
from bed_reader import open_bed
from scipy.linalg import lapack

from common import ONE_THREAD, read_fam, run, run_all


def reml_parts(V, X, y):
    """P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1, Py and the REML log-likelihood up to a constant (V is overwritten)."""
    c, info = lapack.dpotrf(V, lower=1, overwrite_a=1, clean=0)
    if info:
        raise np.linalg.LinAlgError('V is not positive definite')
    logdet = 2 * np.log(np.diag(c)).sum()
    P, info = lapack.dpotri(c, lower=1, overwrite_c=1)
    P = np.tril(P) + np.tril(P, -1).T
    PX = P @ X
    cx = sla.cho_factor(X.T @ PX, lower=True)
    P -= PX @ sla.cho_solve(cx, PX.T)
    Py = P @ y
    return P, Py, -0.5 * (logdet + 2 * np.log(np.diag(cx[0])).sum() + y @ Py)


def reml(y, X, kernels, start=None, maxit=60, tol=1e-6):
    """AI-REML for Var(y) = sum_k s_k K_k + s_e I. Steps that lower the likelihood or leave V indefinite are halved."""
    n, r = y.size, len(kernels) + 1
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    s = np.full(r, resid.var() / r) if start is None else np.array(start, dtype=float)

    def parts(sig):
        V = sum(sk * Kk for sk, Kk in zip(sig[:-1], kernels))
        V.flat[::n + 1] += sig[-1]
        return reml_parts(V, X, y)

    def ai_score(P, Py):
        KPy = np.column_stack([Kk @ Py for Kk in kernels] + [Py])
        trPK = np.array([np.einsum('ij,ij->', P, Kk) for Kk in kernels] + [np.trace(P)])
        return 0.5 * KPy.T @ (P @ KPy), -0.5 * (trPK - Py @ KPy)

    while True:
        try:
            P, Py, ll = parts(s)
            break
        except np.linalg.LinAlgError:
            s[-1] += s[:-1].sum() / 2
            s[:-1] /= 2
    converged = False
    for _ in range(maxit):
        AI, score = ai_score(P, Py)
        step, lam = np.linalg.solve(AI, score), 1.0
        while lam > 1e-4:
            try:
                P_new, Py_new, ll_new = parts(s + lam * step)
                if ll_new >= ll - 1e-8:
                    break
            except np.linalg.LinAlgError:
                pass
            lam /= 2
        else:
            break
        change = np.abs(lam * step).max() / np.abs(s + lam * step).sum()
        s, P, Py, ll_change, ll = s + lam * step, P_new, Py_new, ll_new - ll, ll_new
        if change < tol and abs(ll_change) < 1e-6:
            converged = True
            break
    AI, _ = ai_score(P, Py)
    return s, np.linalg.inv(AI), converged


def h2_share(s, cov, k):
    """Share of components k in the total variance and its delta-method SE."""
    k = np.atleast_1d(k)
    total = s.sum()
    h = s[k].sum() / total
    grad = np.full(s.size, -h / total)
    grad[k] += 1 / total
    return h, float(np.sqrt(max(grad @ cov @ grad, 0)))


def grm(bfile):
    bed = open_bed(bfile + '.bed', count_A1=True)
    A = np.zeros((bed.shape[0],) * 2)
    for s in range(0, bed.shape[1], 4096):
        X = bed.read(index=np.s_[:, s:s + 4096], dtype='float64')
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu, X) - mu
        sd = X.std(0, ddof=1)
        X = X[:, sd > 0] / sd[sd > 0]
        A += X @ X.T
    return A / bed.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--task', required=True, help='output folder of simulate.py')
    ap.add_argument('--covldsc', required=True, help='folder with cov-LDSC (cov.ldscore.py, ldsc.py)')
    ap.add_argument('--ldsc-python', required=True, help='Python interpreter of the cov-LDSC environment')
    ap.add_argument('--ld', choices=['window', 'genome'], default='genome')
    ap.add_argument('--no-reml', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    C, T = a.cohort, a.task
    fam = read_fam(f'{C}/full')
    ph = pd.read_csv(f'{T}/pheno.tsv', sep='\t', dtype={'FID': str, 'IID': str})
    names = [c for c in ph.columns if c not in ('FID', 'IID')]
    rg = names[0].startswith('a')
    rows = []

    # cov-LDSC LD scores, once per cohort
    B = f'{C}/covldsc_{a.ld}'
    os.makedirs(B, exist_ok=True)
    for model, annot in (('single', ''), ('8bin', f'--annot {C}/annot8.tsv.gz')):
        if os.path.exists(f'{B}/{model}.l2.ldscore.gz'):
            continue
        if a.ld == 'window':
            run(f'{a.ldsc_python} {a.covldsc}/cov.ldscore.py --bfile-prefix {C}/full --cov {C}/covar_all.txt '
                f'--ld-wind-kb 20000 --n-parallel 1 {annot} --out-prefix {B}/{model}', log=f'{B}/{model}.out')
        else:                                         # SUMMIT's in-sample LD scores of everyone (related pairs included)
            src = f'{C}/ld/all.{"ld1" if model == "single" else "ld8"}.gw'
            L = pd.read_csv(f'{src}.ldscore.gz', sep='\t')
            cols = [c for c in L.columns if c not in ('CHR', 'SNP', 'BP', 'V') and not c.startswith('LD_REL')]
            L[['CHR', 'SNP', 'BP'] + cols].to_csv(f'{B}/{model}.l2.ldscore.gz', sep='\t', index=False, compression='gzip')
            m = open(f'{src}.M').read().split()
            for suffix in ('.l2.M', '.l2.M_5_50'):
                open(f'{B}/{model}{suffix}', 'w').write('\t'.join(m) + '\n')

    # LDSC-format summary statistics from the GWAS of everyone
    S = f'{T}/ldsc_{a.ld}'
    os.makedirs(S, exist_ok=True)
    for n in names:
        ss = pd.read_csv(f'{T}/ss_K/{n}.tsv', sep='\t', dtype={'SNP': str})
        if rg:                                        # ldsc --rg stops at strand-ambiguous SNPs
            ss = ss[~(ss.A1 + ss.A2).isin(['AT', 'TA', 'CG', 'GC'])]
        pd.DataFrame({'SNP': ss.SNP, 'A1': ss.A1, 'A2': ss.A2, 'N': ss.N, 'Z': ss.BETA / ss.SE}).to_csv(
            f'{S}/{n}.sumstats.gz', sep='\t', index=False, float_format='%.8g', compression='gzip')
    LDSC = f'{ONE_THREAD} {a.ldsc_python} {a.covldsc}/ldsc.py'
    common = f'--w-ld {B}/single --not-M-5-50 --chisq-max 99999'
    if rg:
        R = len(names) // 2
        run_all([f'{LDSC} --rg {S}/a{r}.sumstats.gz,{S}/b{r}.sumstats.gz --ref-ld {B}/single {common} --out {S}/rg.{r} '
                 f'> /dev/null 2>&1' for r in range(R)], a.threads)
    else:
        run_all([f'{LDSC} --h2 {S}/{n}.sumstats.gz --ref-ld {B}/{m} {common} --out {S}/{m}.{n} > /dev/null 2>&1'
                 for m in ('single', '8bin') for n in names], a.threads)
    NUM = r'(-?(?:[0-9.]+(?:e[-+]?[0-9]+)?|nan|inf))'
    for f in sorted(os.listdir(S)):
        if not f.endswith('.log'):
            continue
        text = open(f'{S}/{f}').read()
        if rg:
            gc = re.findall(rf'Total Observed scale gencov: {NUM} \({NUM}\)', text) or [('nan', 'nan')]
            r_ = re.findall(rf'Genetic Correlation: {NUM} \({NUM}\)', text) or [('nan', 'nan')]
            rows.append(dict(rep=int(f.split('.')[1]), arm=f'covLDSC_{a.ld}', model='single', est=float(gc[-1][0]),
                             se=float(gc[-1][1]), rg=float(r_[-1][0]), rg_se=float(r_[-1][1])))
        else:
            model, n = f.split('.')[:2]
            h = re.findall(rf'Total Observed scale h2: {NUM} \({NUM}\)', text)[-1]
            ic = re.findall(rf'Intercept: {NUM} \({NUM}\)', text)[-1]
            rows.append(dict(rep=int(n[1:]), arm=f'covLDSC_{a.ld}', model=model, est=float(h[0]), se=float(h[1]),
                             intercept=float(ic[0])))

    # two-kernel REML
    if not rg and not a.no_reml:
        A = grm(f'{C}/full')
        pairs = pd.read_csv(f'{C}/pairs.txt', sep=' ', dtype={'IID1': str, 'IID2': str})
        pos = pd.Series(np.arange(len(fam)), index=fam.IID)
        i, j = pos[pairs.IID1].to_numpy(), pos[pairs.IID2].to_numpy()
        Bk = np.diag(np.diag(A))
        Bk[i, j] = A[i, j]
        Bk[j, i] = A[j, i]
        cov = pd.read_csv(f'{C}/covar_all.txt', sep=' ', dtype={'FID': str, 'IID': str}).set_index('IID').loc[fam.IID]
        X = np.column_stack([np.ones(len(fam)), cov.drop(columns='FID').to_numpy(float)])
        start = None
        for n in names:
            s, Cinv, ok = reml(ph[n].to_numpy(float), X, [A, Bk], start=start)
            h, se = h2_share(s, Cinv, 0)
            rows.append(dict(rep=int(n[1:]), arm='REML', model='single', est=h, se=se, converged=ok))
            start = s if ok else None
    truth = pd.read_csv(f'{T}/truth.tsv', sep='\t')
    out = pd.DataFrame(rows).merge(truth, on='rep')
    meta = pd.read_csv(f'{T}/results.tsv', sep='\t', nrows=1)[['cohort', 'setting', 'design', 'label']]
    for c in reversed(meta.columns):
        out.insert(0, c, meta[c].iloc[0])
    out.to_csv(f'{T}/methods_{a.ld}.tsv', sep='\t', index=False, float_format='%.8g')


if __name__ == '__main__':
    main()
