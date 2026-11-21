"""Heritability of the 40 UK Biobank traits in one prepared cohort (prepare_cohort.py), for the analyses of simulate.py:
U (unrelated subset), K (everyone, related pairs kept) and XA (everyone, related pairs excluded), with one component and with
8 MAF-LD bins, and standard errors from the 200-block jackknife with the long-range LD correction. Covariates are the
cohort's (PCs and, for the wider African-ancestry group, the ancestry covariates), age and sex terms, and optionally the
origin covariates of traits.py. Binary traits are put on the liability scale with the prevalence of the analysed sample.
Inputs from traits.py: DIR/pheno_{all,unrel}.tsv, agesex_{all,unrel}.tsv, prevalence.tsv and, with --origin,
origin_{all,unrel}.tsv. Writes OUT/real_h2.tsv (trait, arm, model, n, h2, se).
usage: python real_traits.py --cohort DIR --traits DIR --out DIR [--origin] [--arms U,K,XA] [--threads 8]
"""
import argparse
import os

import pandas as pd

from common import ONE_THREAD, gwas, liability_factor, read_h2_results, run, run_all

ap = argparse.ArgumentParser()
ap.add_argument('--cohort', required=True)
ap.add_argument('--traits', required=True, help='output folder of traits.py for this cohort (samples all and unrel)')
ap.add_argument('--out', required=True)
ap.add_argument('--origin', action='store_true')
ap.add_argument('--arms', default='U,K,XA')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--summit', default='summit')
ap.add_argument('--plink2', default='plink2')
a = ap.parse_args()
C, P, W = a.cohort, a.traits, a.out
os.makedirs(W, exist_ok=True)
arms = a.arms.split(',')
A8 = f'--annot {C}/annot8.tsv.gz'
PAIRS = f'--pairs {C}/pairs.txt --kinship-col K_grm --kinship-min 0.0442'
traits = [c for c in pd.read_csv(f'{P}/pheno_all.tsv', sep='\t', nrows=0).columns if c not in ('FID', 'IID')]

# covariates of each sample: cohort covariates + age/sex (+ origin)
for s in ('all', 'unrel'):
    cov = pd.read_csv(f'{C}/covar_{s}.txt', sep=' ', dtype={'FID': str, 'IID': str})
    extra = [pd.read_csv(f'{P}/agesex_{s}.tsv', sep='\t', dtype={'FID': str, 'IID': str}).drop(columns='FID')]
    if a.origin:
        extra.append(pd.read_csv(f'{P}/origin_{s}.tsv', sep='\t', dtype={'FID': str, 'IID': str}).drop(columns='FID'))
    for e in extra:
        cov = cov.merge(e, on='IID', how='left')
    assert cov.notna().all().all(), f'missing covariates in the {s} sample'
    cov.to_csv(f'{W}/covar_{s}.txt', sep=' ', index=False, float_format='%.10g')
rank = {s: pd.read_csv(f'{W}/covar_{s}.txt', sep=' ', nrows=0).shape[1] - 2 for s in ('all', 'unrel')}

if 'U' in arms and not os.path.exists(f'{W}/ss_U/{traits[-1]}.tsv'):
    gwas(a.plink2, f'{C}/unrel', f'{P}/pheno_unrel.tsv', traits, f'{W}/covar_unrel.txt', rank['unrel'], f'{W}/ss_U', a.threads)
if {'K', 'XA'} & set(arms) and not os.path.exists(f'{W}/ss_K/{traits[-1]}.tsv'):
    gwas(a.plink2, f'{C}/full', f'{P}/pheno_all.tsv', traits, f'{W}/covar_all.txt', rank['all'], f'{W}/ss_K', a.threads)
if 'XA' in arms:
    if not os.path.exists(f'{W}/rel.rel_m.tsv'):
        run(f'{a.summit} relatives moments --geno {C}/full.bed --pheno {P}/pheno_all.tsv --covar {W}/covar_all.txt {A8} {PAIRS} '
            f'--out {W}/rel', log=f'{W}/rel.out')
    for tag, annot in (('XA1', ''), ('XA8', A8)):
        os.makedirs(f'{W}/ss_{tag}', exist_ok=True)
        run_all([f'{a.summit} relatives append --sumstats {W}/ss_K/{t}.tsv --rel {W}/rel.rel_m.tsv --pheno {t} {annot} '
                 f'--out {W}/ss_{tag}/{t}.tsv > /dev/null' for t in traits if not os.path.exists(f'{W}/ss_{tag}/{t}.tsv')], a.threads)

# long-range LD sums of each analysed sample with its covariates
for s, geno, pairs in (('all', 'full', PAIRS), ('unrel', 'unrel', '')):
    if not os.path.exists(f'{W}/longrange_{s}.longrange.json'):
        run(f'{a.summit} relatives longrange --geno {C}/{geno}.bed --covar {W}/covar_{s}.txt {pairs} {A8} --out {W}/longrange_{s}')

cmds = []
SAMPLE = {'U': 'unrel', 'K': 'all', 'XA': 'all'}
for arm in arms:
    s = SAMPLE[arm]
    for model, ld, annot in (('single', 'ld1', ''), ('8bin', 'ld8', A8)):
        ss = f'ss_XA{"1" if model == "single" else "8"}' if arm == 'XA' else f'ss_{arm}'
        pre = f'{W}/fit_{arm}.{model}'
        if not os.path.exists(pre + '.results.tsv'):
            cmds.append(f'{ONE_THREAD} {a.summit} --h2 {W}/{ss} --ldscores {C}/ld/{s}.{ld}.gw.ldscore.gz {annot} --njack 200 '
                        f'--long-range-se {W}/longrange_{s} --num-threads 1 --out {pre} > {pre}.out 2>&1')
run_all(cmds, a.threads)

prev = pd.read_csv(f'{P}/prevalence.tsv', sep='\t').set_index(['sample', 'trait']).prevalence
rows = []
for arm in arms:
    for model in ('single', '8bin'):
        for x in read_h2_results(f'{W}/fit_{arm}.{model}.results.tsv').itertuples():
            ss = pd.read_csv(f'{W}/ss_{"U" if arm == "U" else "K"}/{x.phen}.tsv', sep='\t', usecols=['N'])
            f = liability_factor(prev[(SAMPLE[arm], x.phen)]) if (SAMPLE[arm], x.phen) in prev.index else 1.0
            rows.append(dict(cohort=os.path.basename(os.path.normpath(C)), origin=a.origin, trait=x.phen, arm=arm, model=model,
                             n=int(ss.N.max()), h2=x.h2 * f, se=x.se * f))
pd.DataFrame(rows).to_csv(f'{W}/real_h2.tsv', sep='\t', index=False, float_format='%.6g')
print(f'{len(rows)} fits written to {W}/real_h2.tsv')
