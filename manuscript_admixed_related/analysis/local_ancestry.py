"""Joint SNP and local-ancestry heritability in an admixed cohort (prepared by prepare_cohort.py).

  build      local-ancestry markers (dosage of African ancestry at one calling site per 250 kb) and global ancestry THETA
             from the calls (`summit local-ancestry markers`), a second marker set at 600 kb spacing for simulating untagged
             variants, the SNPs and markers merged with the 8 MAF-LD bins plus a class LA (`summit local-ancestry merge`),
             and in-sample LD scores of the merged set with covariates that include THETA. --fill gives participants
             without calls (unadmixed members of the wider group) African ancestry on both haplotypes.
  simulate   replicate phenotypes y = sqrt(0.25) g + sqrt(s2_la) a + sqrt(s2_u) u + e, with g on the SNPs, a on the
             standardized local-ancestry markers and u on untagged variants whose genotypes are drawn, haplotype by haplotype,
             with the allele frequency of the local ancestry (Balding-Nichols, F = 0.15 per ancestry); settings null
             (0, 0), model (0.05, 0), untagged (0, 0.25) and untagged2 (0, 0.5). GWAS of all markers (SNPs and admixture
             mapping), related pairs excluded, and fits of the SNPs alone (8 bins) and of SNPs and local ancestry (8 bins + LA),
             with 200 SNP blocks and with chromosomes as jackknife units.
  real       the same fits for the real traits of traits.py (covariates: cohort covariates, age and sex, THETA, optionally
             origin), and `summit local-ancestry test` for the test of the local-ancestry component and its interval.
Writes OUT/la_results.tsv (setting, rep, model, h2_snp, h2_snp_se, h2_la, h2_la_se) and, for real traits, OUT/la_test.tsv.
usage: python local_ancestry.py build --cohort DIR --calls PATTERN --sites PATTERN --out DIR [--fill] [--strip-id-suffix]
       python local_ancestry.py simulate --cohort DIR --la DIR --setting null --out DIR
       python local_ancestry.py real --cohort DIR --la DIR --traits DIR --out DIR [--origin]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from bed_reader import open_bed

from common import ONE_THREAD, gwas, read_fam, run, run_all

SETTINGS = {'null': (0.0, 0.0), 'model': (0.05, 0.0), 'untagged': (0.0, 0.25), 'untagged2': (0.0, 0.5)}
REL = '--kinship-col K_grm --kinship-min 0.0442'


def build(a):
    C, L = a.cohort, a.out
    os.makedirs(L, exist_ok=True)
    fill = '--fill 1' if a.fill else ''
    samples = ''
    if a.strip_id_suffix:                               # .psam IDs written as <id>_<suffix>: keep the part before '_'
        psam = pd.read_csv(os.path.splitext(a.sites.format(chr=1))[0] + '.psam', sep='\t', dtype=str)
        psam['#IID'].str.split('_').str[0].to_csv(f'{L}/samples.txt', index=False, header=False)
        samples = f'--samples {L}/samples.txt'
    for name, spacing in (('markers', 250000), ('untagged', 600000)):
        if not os.path.exists(f'{L}/{name}.bed'):
            run(f'{a.summit} local-ancestry markers --calls {a.calls} --format lanc --sites {a.sites} {samples} '
                f'--fam {C}/full.fam --ancestry 1 --spacing {spacing} {fill} --out {L}/{name}')
    if not os.path.exists(f'{L}/merged.bed'):
        run(f'{a.summit} local-ancestry merge --geno {C}/full --markers {L}/markers --annot {C}/annot8.tsv.gz --out {L}/merged')
    cov = pd.read_csv(f'{C}/covar_all.txt', sep=' ', dtype={'FID': str, 'IID': str})
    theta = pd.read_csv(f'{L}/markers.theta.tsv', sep='\t', dtype={'FID': str, 'IID': str})[['IID', 'THETA']]
    cov.merge(theta, on='IID', how='left').to_csv(f'{L}/covar_all.txt', sep=' ', index=False, float_format='%.10g')
    common = f'--nvecs 20000 --block-size 16384 --dtype float64 --num-threads {a.threads} --relative-pairs {C}/pairs.txt {REL}'
    variants = [('', f'{L}/covar_all.txt')] + ([('_pcs', f'{C}/covar_all.txt')] if a.without_theta else [])
    for suffix, cov_ in variants:
        for name, geno, annot in (('la', f'{L}/merged.bed', f'{L}/merged.annot.tsv.gz'),
                                  ('snp', f'{C}/full.bed', f'{C}/annot8.tsv.gz')):
            if not os.path.exists(f'{L}/ld_{name}{suffix}.gw.ldscore.gz'):
                run(f'{a.summit} --geno {geno} --annot {annot} --covar {cov_} {common} --seed 5 --out {L}/ld_{name}{suffix}')


def standardize(X):
    X = X - X.mean(0)
    sd = X.std(0, ddof=1)
    return np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)


def simulate_phenotypes(a, names):
    L, R = a.la, a.nrep
    s2_la, s2_u = SETTINGS[a.setting]
    bed = open_bed(f'{L}/merged.bed', count_A1=True)
    is_la = np.char.startswith(np.asarray(bed.sid, dtype=str), 'LA:')
    N = bed.shape[0]
    G = {'snp': np.zeros((N, R)), 'la': np.zeros((N, R))}
    for kind, cols in (('snp', np.flatnonzero(~is_la)), ('la', np.flatnonzero(is_la))):
        B = np.zeros((cols.size, R))
        for r in range(R):
            b = np.random.default_rng([a.seed, 1 if kind == 'snp' else 2, r]).standard_normal(cols.size)
            B[:, r] = b / np.linalg.norm(b)
        for s in range(0, cols.size, 4096):
            X = bed.read(index=np.s_[:, cols[s:s + 4096]], dtype='float64')
            X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
            G[kind] += standardize(X) @ B[s:s + 4096]
    U = np.zeros((N, R))
    if s2_u > 0:
        dosage = open_bed(f'{L}/untagged.bed', count_A1=True).read(dtype='float64')     # African haplotypes 0-2 per site
        for r in range(R):
            rng = np.random.default_rng([a.seed, 3, r])
            pick = rng.choice(dosage.shape[1], min(1000, dosage.shape[1]), replace=False)
            p0, F = rng.uniform(0.05, 0.95, pick.size), 0.15
            p_eur = rng.beta(p0 * (1 - F) / F, (1 - p0) * (1 - F) / F)
            p_afr = rng.beta(p0 * (1 - F) / F, (1 - p0) * (1 - F) / F)
            d = dosage[:, pick].astype(int)
            x = rng.binomial(d, p_afr) + rng.binomial(2 - d, p_eur)
            w = rng.standard_normal(pick.size)
            U[:, r] = standardize(x.astype(float)) @ (w / np.linalg.norm(w))
    Y = np.zeros((N, R))
    for r in range(R):
        e = np.random.default_rng([a.seed, 4, r]).standard_normal(N)
        Y[:, r] = (np.sqrt(0.25) * G['snp'][:, r] + np.sqrt(s2_la) * G['la'][:, r] + np.sqrt(s2_u) * U[:, r]
                   + np.sqrt(1 - 0.25 - s2_la - s2_u) * e)
    ph = read_fam(f'{L}/merged')
    ph[names] = Y
    return ph


def fit(a, names, pheno, covar, cov_rank, W, ld_suffix=''):
    C, L = a.cohort, a.la
    if not os.path.exists(f'{W}/ss/{names[-1]}.tsv'):
        gwas(a.plink2, f'{L}/merged', pheno, names, covar, cov_rank, f'{W}/ss', a.threads)
    snps = set(pd.read_csv(f'{C}/full.bim', sep=r'\s+', header=None, usecols=[1], dtype=str)[1])
    os.makedirs(f'{W}/ss_snp', exist_ok=True)
    for n in names:
        s = pd.read_csv(f'{W}/ss/{n}.tsv', sep='\t', dtype={'SNP': str})
        s[s.SNP.isin(snps)].to_csv(f'{W}/ss_snp/{n}.tsv', sep='\t', index=False, float_format='%.10g')
    for tag, geno, annot in (('snp', f'{C}/full.bed', f'{C}/annot8.tsv.gz'), ('la', f'{L}/merged.bed', f'{L}/merged.annot.tsv.gz')):
        if not os.path.exists(f'{W}/rel_{tag}.rel_m.tsv'):
            run(f'{a.summit} relatives moments --geno {geno} --pheno {pheno} --covar {covar} --annot {annot} '
                f'--pairs {C}/pairs.txt {REL} --out {W}/rel_{tag}')
        src = 'ss_snp' if tag == 'snp' else 'ss'
        os.makedirs(f'{W}/ss_{tag}8', exist_ok=True)
        run_all([f'{a.summit} relatives append --sumstats {W}/{src}/{n}.tsv --rel {W}/rel_{tag}.rel_m.tsv --pheno {n} '
                 f'--annot {annot} --out {W}/ss_{tag}8/{n}.tsv > /dev/null' for n in names], a.threads)
    cmds = []
    for model, ss, ld, annot in (('SNP8', 'ss_snp8', 'ld_snp' + ld_suffix, f'{C}/annot8.tsv.gz'),
                                 ('LA9', 'ss_la8', 'ld_la' + ld_suffix, f'{L}/merged.annot.tsv.gz')):
        for nj, suffix in (('200', ''), ('chr', 'c')):
            pre = f'{W}/fit_{model}{suffix}'
            if not os.path.exists(pre + '.results.tsv'):
                cmds.append(f'{ONE_THREAD} {a.summit} --h2 {W}/{ss} --ldscores {L}/{ld}.gw.ldscore.gz --annot {annot} --njack {nj} '
                            f'--write-jack --num-threads 1 --out {pre} > {pre}.out 2>&1')
    run_all(cmds, a.threads)
    rows = []
    for f in sorted(glob.glob(f'{W}/fit_*.results.tsv')):
        model = os.path.basename(f)[4:-len('.results.tsv')]
        for _, x in pd.read_csv(f, sep='\t', dtype={'phen': str}).iterrows():
            nb = int(x.num_bins)
            row = dict(setting=a.setting, rep=x.phen, model=model, h2_snp=float(x.h2), h2_snp_se=float(x.h2_se),
                       h2_la=np.nan, h2_la_se=np.nan)
            if model.startswith('LA'):                  # last class is LA; SE of the SNP classes from the delete-one values
                J = pd.read_csv(f'{W}/fit_{model}.{x.phen}.jack', sep='\t')
                J = J[J['# replicate'].astype(str) != 'full']
                v = J[[f'h2_cat_{k}' for k in range(nb - 1)]].sum(1).to_numpy()
                row.update(h2_snp=sum(float(x[f'h2bin_{k}']) for k in range(nb - 1)),
                           h2_snp_se=float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum())),
                           h2_la=float(x[f'h2bin_{nb - 1}']), h2_la_se=float(x[f'h2bin_se_{nb - 1}']))
            rows.append(row)
    pd.DataFrame(rows).to_csv(f'{W}/la_results.tsv', sep='\t', index=False, float_format='%.6g')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['build', 'simulate', 'real'])
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--la', help='output folder of the build step')
    ap.add_argument('--calls', help='admix-kit .lanc files with {chr}')
    ap.add_argument('--sites', help='matching .pvar files with {chr}')
    ap.add_argument('--fill', action='store_true')
    ap.add_argument('--strip-id-suffix', action='store_true', help="sample IDs of the .psam are <id>_<suffix>")
    ap.add_argument('--setting', default='real', choices=list(SETTINGS) + ['real'])
    ap.add_argument('--traits', help='output folder of traits.py (real step)')
    ap.add_argument('--origin', action='store_true')
    ap.add_argument('--without-theta', action='store_true',
                    help='build: also LD scores with the cohort covariates alone; simulate: use them (as in Fig. 2D)')
    ap.add_argument('--nrep', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--summit', default='summit')
    ap.add_argument('--plink2', default='plink2')
    a = ap.parse_args()
    if a.step == 'build':
        return build(a)
    W = a.out
    os.makedirs(W, exist_ok=True)
    if a.step == 'simulate':
        names = [f'y{r}' for r in range(a.nrep)]
        if not os.path.exists(f'{W}/pheno.tsv'):
            simulate_phenotypes(a, names).to_csv(f'{W}/pheno.tsv', sep='\t', index=False, float_format='%.8g')
        # cohort covariates and THETA, as for the LD scores; --without-theta uses the cohort covariates alone (Fig. 2D)
        covar = f'{a.cohort}/covar_all.txt' if a.without_theta else f'{a.la}/covar_all.txt'
        fit(a, names, f'{W}/pheno.tsv', covar, pd.read_csv(covar, sep=' ', nrows=0).shape[1] - 2, W,
            ld_suffix='_pcs' if a.without_theta else '')
        return
    pheno = f'{a.traits}/pheno_all.tsv'
    names = [c for c in pd.read_csv(pheno, sep='\t', nrows=0).columns if c not in ('FID', 'IID')]
    cov = pd.read_csv(f'{a.la}/covar_all.txt', sep=' ', dtype={'FID': str, 'IID': str})
    for f in [f'{a.traits}/agesex_all.tsv'] + ([f'{a.traits}/origin_all.tsv'] if a.origin else []):
        cov = cov.merge(pd.read_csv(f, sep='\t', dtype={'FID': str, 'IID': str}).drop(columns='FID'), on='IID', how='left')
    cov.to_csv(f'{W}/covar_all.txt', sep=' ', index=False, float_format='%.10g')
    fit(a, names, pheno, f'{W}/covar_all.txt', cov.shape[1] - 2, W)
    if not os.path.exists(f'{W}/la.la_test.tsv'):
        run(f'{a.summit} local-ancestry test --geno {a.la}/merged --annot {a.la}/merged.annot.tsv.gz --pheno {pheno} '
            f'--covar {W}/covar_all.txt --pairs {a.cohort}/pairs.txt {REL} --out {W}/la')


if __name__ == '__main__':
    main()
