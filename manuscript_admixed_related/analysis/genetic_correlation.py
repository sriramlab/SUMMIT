"""Genetic correlation among the 40 traits in one prepared cohort, and agreement with a reference cohort (Table 4, Fig. 4B, D and Fig. 5).

  fit      every pair of traits for U (unrelated subset) and XA (everyone, related pairs excluded with REL_M and REL_M12),
           with one component and 8 MAF-LD bins, the overlapping covariance in HE scale built from the phenotypes and
           covariates, strand-ambiguous SNPs kept, and the 200-block jackknife with the long-range LD correction. Run
           real_traits.py first: it writes the GWAS, covariates and long-range sums used here. Writes OUT/rg.tsv.
  compare  for each analysis, pairs with positive heritability in both analyses of the cohort; pairs significant at a false
           discovery rate of 5%; median absolute difference and median SE ratio against U; and against the reference
           (e.g. the unrelated white British fit of the same pipeline): inverse-variance weighted correlation, slope of the
           cohort's estimates on the reference's (weights from the cohort's SEs), and the robust SD (1.4826 x median
           absolute value) of standardized differences. Writes OUT/rg_compare.tsv.
  reference  the same fits for an unrelated reference sample from its GWAS and in-sample LD scores.
usage: python genetic_correlation.py fit --cohort DIR --real DIR --traits DIR --out DIR
       python genetic_correlation.py reference --sumstats DIR --pheno FILE --covar FILE --ldscores FILE --annot FILE --out DIR
       python genetic_correlation.py compare --rg OUT/rg.tsv --reference REF/rg.tsv --out DIR
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import ONE_THREAD, run, run_all


def fit(a):
    from summit.manifest.rg_manifest_builder import build_rg_manifest
    C, Rd, P, O = a.cohort, a.real, a.traits, a.out
    os.makedirs(O, exist_ok=True)
    traits = [c for c in pd.read_csv(f'{P}/pheno_all.tsv', sep='\t', nrows=0).columns if c not in ('FID', 'IID')]
    A8 = f'--annot {C}/annot8.tsv.gz'
    if not os.path.exists(f'{O}/rel.rel_m12.tsv'):
        run(f'{a.summit} relatives moments --geno {C}/full.bed --pheno {P}/pheno_all.tsv --covar {Rd}/covar_all.txt {A8} '
            f'--pairs {C}/pairs.txt --kinship-col K_grm --kinship-min 0.0442 --cross --out {O}/rel', log=f'{O}/rel.out')
    sample = {'U': ('unrel', 'ss_U'), 'XA': ('all', 'ss_XA')}
    jobs = []
    for arm, (s, ss) in sample.items():
        man = f'{O}/manifest_{arm}.tsv'
        if not os.path.exists(man):
            pd.DataFrame({'phen': traits, 'sumstats': [f'{Rd}/{ss}1/{t}.tsv' if arm == 'XA' else f'{Rd}/{ss}/{t}.tsv'
                                                       for t in traits]}).to_csv(f'{O}/sumstats_{arm}.tsv', sep='\t', index=False)
            open(f'{O}/traits.txt', 'w').write('\n'.join(traits) + '\n')
            build_rg_manifest(output_path=man, phen_source=f'{P}/pheno_{s}.tsv', sumstats_source=f'{O}/sumstats_{arm}.tsv',
                              cov_source=f'{Rd}/covar_{s}.txt', phen_list_path=f'{O}/traits.txt', all_pairwise=True,
                              pheno_missing_values=['-9', 'NA'])
        base = pd.read_csv(man, sep='\t')
        for model, ld, annot, k in (('single', 'ld1', '', '1'), ('8bin', 'ld8', A8, '8')):
            m = base.copy()
            ssd = f'{Rd}/ss_XA{k}' if arm == 'XA' else f'{Rd}/ss_U'
            m['sumstats1'] = [f'{ssd}/{t}.tsv' for t in m.phen1]
            m['sumstats2'] = [f'{ssd}/{t}.tsv' for t in m.phen2]
            cols = ['phen1', 'phen2', 'sumstats1', 'sumstats2', 'overlap_covariance']
            if arm == 'XA':                          # REL_M12 of each pair (per class for the 8-bin fit)
                d = f'{O}/m12_{model}'
                os.makedirs(d, exist_ok=True)
                m['rel_m12'] = [f'{d}/{p}.{q}.tsv' for p, q in zip(m.phen1, m.phen2)]
                run_all([f'{a.summit} relatives cross-m12 --rel {O}/rel.rel_m12.tsv --pheno1 {p} --pheno2 {q} {annot} --out {f}'
                         for p, q, f in zip(m.phen1, m.phen2, m.rel_m12) if not os.path.exists(f)], a.threads)
                cols.append('rel_m12')
            for t, rows in m.groupby('phen1', sort=False):     # one manifest per first trait, fitted side by side
                d = f'{O}/fit_{arm}.{model}/{t}'
                os.makedirs(d, exist_ok=True)
                rows[cols].to_csv(f'{d}/manifest.tsv', sep='\t', index=False)
                if not os.path.exists(f'{d}/manifest.results.tsv'):
                    jobs.append(f'{ONE_THREAD} {a.summit} --rg {d}/manifest.tsv --ldscores {C}/ld/{s}.{ld}.gw.ldscore.gz {annot} '
                                f'--njack 200 --keep-ambiguous --long-range-se {Rd}/longrange_{s} --out {d} > {d}/run.out 2>&1')
    run_all(jobs, a.threads)
    res = []
    for f in sorted(glob.glob(f'{O}/fit_*/*/manifest.results.tsv')):
        arm, model = f.split('/fit_')[1].split('/')[0].split('.')
        x = pd.read_csv(f, sep='\t')
        x.insert(0, 'model', model)
        x.insert(0, 'arm', arm)
        res.append(x)
    keep = ['arm', 'model', 'phen1', 'phen2', 'h2_trait1', 'h2_trait1_se', 'h2_trait2', 'h2_trait2_se', 'rg_total', 'rg_total_se']
    pd.concat(res)[keep].to_csv(f'{O}/rg.tsv', sep='\t', index=False, float_format='%.6g')


def reference(a):
    """rg of an unrelated reference sample from its own GWAS (one file per trait in --sumstats), phenotype and covariate
    tables and in-sample 8-bin LD scores, with the same settings as `fit` (written as analysis U, model 8bin)."""
    from summit.manifest.rg_manifest_builder import build_rg_manifest
    O = a.out
    os.makedirs(O, exist_ok=True)
    traits = [c for c in pd.read_csv(a.pheno, sep='\t', nrows=0).columns if c not in ('FID', 'IID')]
    pd.DataFrame({'phen': traits, 'sumstats': [f'{a.sumstats}/{t}.tsv' for t in traits]}).to_csv(f'{O}/sumstats.tsv', sep='\t', index=False)
    open(f'{O}/traits.txt', 'w').write('\n'.join(traits) + '\n')
    m = build_rg_manifest(output_path=f'{O}/manifest.tsv', phen_source=a.pheno, sumstats_source=f'{O}/sumstats.tsv',
                          cov_source=a.covar, phen_list_path=f'{O}/traits.txt', all_pairwise=True,
                          pheno_missing_values=['-9', 'NA'])
    m = pd.read_csv(f'{O}/manifest.tsv', sep='\t')
    jobs = []
    for t, rows in m.groupby('phen1', sort=False):
        d = f'{O}/fit_U.8bin/{t}'
        os.makedirs(d, exist_ok=True)
        rows.to_csv(f'{d}/manifest.tsv', sep='\t', index=False)
        jobs.append(f'{ONE_THREAD} {a.summit} --rg {d}/manifest.tsv --ldscores {a.ldscores} --annot {a.annot} --njack 200 '
                    f'--keep-ambiguous --out {d} > {d}/run.out 2>&1')
    run_all(jobs, a.threads)
    res = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(f'{O}/fit_U.8bin/*/manifest.results.tsv'))])
    res.insert(0, 'model', '8bin')
    res.insert(0, 'arm', 'U')
    res[['arm', 'model', 'phen1', 'phen2', 'h2_trait1', 'h2_trait1_se', 'h2_trait2', 'h2_trait2_se', 'rg_total',
         'rg_total_se']].to_csv(f'{O}/rg.tsv', sep='\t', index=False, float_format='%.6g')


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty_like(q)
    out[o] = np.minimum.accumulate(q[::-1])[::-1]
    return np.minimum(out, 1)


def compare(a):
    R = pd.read_csv(a.rg, sep='\t')
    ref = pd.read_csv(a.reference, sep='\t')
    key = lambda d: ['|'.join(sorted(p)) for p in zip(d.phen1, d.phen2)]
    R['key'], ref['key'] = key(R), key(ref)
    ref = ref[(ref.arm == 'U') & (ref.model == '8bin')].set_index('key')
    rows = []
    for model in ('single', '8bin'):
        u = R[(R.arm == 'U') & (R.model == model)].set_index('key')
        x = R[(R.arm == 'XA') & (R.model == model)].set_index('key')
        ok = [(k in u.index) and all(d.loc[k, c] > 0 for d in (u, x) for c in ('h2_trait1', 'h2_trait2')) for k in x.index]
        x, u = x[ok], u.loc[x.index[ok]]
        fdr = lambda d: int((bh(2 * stats.norm.sf(np.abs(d.rg_total / d.rg_total_se))) < 0.05).sum())
        both = x.index.intersection(ref.index)
        xs, rs = x.loc[both], ref.loc[both]
        w = 1 / xs.rg_total_se ** 2
        z = (xs.rg_total - rs.rg_total) / np.sqrt(xs.rg_total_se ** 2 + rs.rg_total_se ** 2)
        row = dict(model=model, pairs=len(x), fdr_xa=fdr(x) if len(x) else 0, fdr_u=fdr(u) if len(u) else 0,
                   median_abs_difference=float((x.rg_total - u.rg_total).abs().median()),
                   median_se_ratio=float((x.rg_total_se / u.rg_total_se).median()),
                   weighted_correlation=np.nan, slope=np.nan, robust_sd=np.nan)
        if len(both) >= 3:
            row.update(weighted_correlation=float(np.cov(xs.rg_total, rs.rg_total, aweights=w)[0, 1]
                                                  / np.sqrt(np.cov(xs.rg_total, aweights=w) * np.cov(rs.rg_total, aweights=w))),
                       slope=float(np.polyfit(rs.rg_total, xs.rg_total, 1, w=np.sqrt(w))[0]),
                       robust_sd=float(1.4826 * np.median(np.abs(z))))
        rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f'{a.out}/rg_compare.tsv', sep='\t', index=False, float_format='%.4g')
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['fit', 'reference', 'compare'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--cohort'); ap.add_argument('--real', help='output folder of real_traits.py'); ap.add_argument('--traits')
    ap.add_argument('--rg'); ap.add_argument('--reference')
    ap.add_argument('--sumstats'); ap.add_argument('--pheno'); ap.add_argument('--covar'); ap.add_argument('--ldscores')
    ap.add_argument('--annot')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--summit', default='summit')
    a = ap.parse_args()
    {'fit': fit, 'reference': reference, 'compare': compare}[a.step](a)


if __name__ == '__main__':
    main()
