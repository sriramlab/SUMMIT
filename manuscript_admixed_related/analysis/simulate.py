"""One simulation task: 100 replicate phenotypes for one setting of designs.py on the genotypes of one prepared cohort
(prepare_cohort.py), PLINK 2 GWAS of everyone and of the unrelated subset, the related-pair parts with
`summit relatives moments`, and SUMMIT fits with one component and with 8 MAF-LD bins for three analyses:
  U   the unrelated subset (its GWAS and LD scores)
  K   everyone, related pairs kept (GWAS of everyone, in-sample LD scores)
  XA  everyone, related pairs excluded (the same GWAS with REL_M; in-sample LD scores, whose related-pair parts are
      subtracted)
For the genetic-correlation designs the fits are `summit --rg` with the overlapping covariance in HE scale and REL_M12.
The 'ancestry' designs need --ancestry (IID, AFR, admixed). Steps whose outputs exist are skipped, so a task can be rerun.
Writes OUT/pheno.tsv, truth.tsv, ss_*/ (summary statistics, kept for compare_methods.py) and results.tsv.
usage: python simulate.py --cohort DIR --setting K --out DIR [--nrep 100] [--threads 8] [--long-range]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from bed_reader import open_bed

import designs
from common import ONE_THREAD, gwas, read_fam, read_h2_results, run, run_all

ap = argparse.ArgumentParser()
ap.add_argument('--cohort', required=True, help='folder written by prepare_cohort.py')
ap.add_argument('--setting', type=int, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--nrep', type=int, default=designs.NREP)
ap.add_argument('--seed', type=int, default=20261026)
ap.add_argument('--ancestry', default=None, help='IID AFR admixed (ancestry designs)')
ap.add_argument('--long-range', action='store_true', help='long-range LD correction of the standard errors')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--summit', default='summit')
ap.add_argument('--plink2', default='plink2')
a = ap.parse_args()
S = designs.settings()[a.setting]
C, W, R = a.cohort, a.out, a.nrep
os.makedirs(W, exist_ok=True)
fam = read_fam(f'{C}/full')
bim = pd.read_csv(f'{C}/full.bim', sep=r'\s+', header=None, usecols=[1], names=['SNP'], dtype={'SNP': str})
N, M = len(fam), len(bim)
RG = S['kind'] == 'rg'
names = [f'a{r}' for r in range(R)] + [f'b{r}' for r in range(R)] if RG else [f'y{r}' for r in range(R)]
npc = pd.read_csv(f'{C}/covar_all.txt', sep=' ', nrows=0).shape[1] - 2


def standardized_blocks(path, cols=None):
    """Genotype blocks standardized in the cohort (mean imputation, sample SD)."""
    bed = open_bed(path, count_A1=True)
    cols = np.arange(bed.shape[1]) if cols is None else cols
    for s in range(0, len(cols), 4096):
        c = cols[s:s + 4096]
        X = bed.read(index=np.s_[:, c], dtype='float64')
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu, X) - mu
        sd = X.std(0, ddof=1)
        X = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)
        yield c, X


def phenotypes():
    rng0 = lambda r, k: np.random.default_rng([a.seed, a.setting, r, k])
    maf = pd.read_csv(f'{C}/maf.tsv', sep='\t', dtype={'SNP': str}).set_index('SNP').MAF.loc[bim.SNP].to_numpy()
    ld = pd.read_csv(f'{C}/ld/all.ld1.gw.ldscore.gz', sep='\t', dtype={'SNP': str}).set_index('SNP').L2_0.loc[bim.SNP].to_numpy()
    eligible = {'all': np.arange(M), 'low': np.flatnonzero(maf < 0.1), 'high': np.flatnonzero(maf >= 0.1)}[S['mafbin']]
    weight = np.ones(M) if S['arch'] == 'gcta' else (maf * (1 - maf)) ** 0.75 / np.maximum(ld, 1.0)
    B = np.zeros((M, len(names)), dtype=np.float32)
    truth = []
    for r in range(R):
        rng = rng0(r, 1)
        causal = rng.choice(eligible, max(1, round(S['p'] * eligible.size)), replace=False)
        sd = np.sqrt(weight[causal])
        b1 = rng.standard_normal(causal.size) * sd
        B[causal, r] = b1 / np.linalg.norm(b1)
        row = dict(rep=r, truth=S['h2'])
        if RG:
            z = rng.standard_normal(causal.size) * sd
            b2 = S['rg'] * B[causal, r] + np.sqrt(1 - S['rg'] ** 2) * z / np.linalg.norm(z)
            B[causal, R + r] = b2 / np.linalg.norm(b2)
            row.update(truth=S['h2'] * float(B[causal, r] @ B[causal, R + r]), rg_truth=float(B[causal, r] @ B[causal, R + r]))
        truth.append(row)
    G = np.zeros((N, len(names)))
    for c, X in standardized_blocks(f'{C}/full.bed'):
        G += X @ B[c]
    # effects shared by relatives: untagged genetics within families (covariance s2u x GRM) and households
    fams = pd.read_csv(f'{C}/families.tsv.gz', sep='\t', dtype={'IID': str})
    blocks = pd.read_csv(f'{C}/family_blocks.tsv.gz', sep='\t')
    roots = []
    by_family = dict(tuple(blocks.groupby(fams.family.to_numpy()[blocks.i.to_numpy()])))
    for f, idx in pd.Series(np.arange(N)).groupby(fams.family.to_numpy()):
        idx = idx.to_numpy()
        if idx.size > 1:
            b = by_family[f]
            pos = {x: k for k, x in enumerate(idx)}
            Kf = np.zeros((idx.size, idx.size))
            Kf[[pos[x] for x in b.i], [pos[x] for x in b.j]] = b.K_grm
            Kf = np.maximum(Kf, Kf.T)
            w, v = np.linalg.eigh(Kf)
            roots.append((idx, v * np.sqrt(np.clip(w, 0, None))))
    single = np.flatnonzero(np.bincount(fams.family)[fams.family] == 1)
    _, house = np.unique(fams.household, return_inverse=True)

    def family_draw(rng):
        out = np.zeros(N)
        for idx, L in roots:
            out[idx] = L @ rng.standard_normal(idx.size)
        out[single] = rng.standard_normal(single.size)
        return out

    env = np.zeros((N, 2))
    if S['s2g'] or S['s2q']:
        anc = pd.read_csv(a.ancestry, sep='\t', dtype={'IID': str}).set_index('IID').loc[fam.IID]
        z = lambda v: (v - v.mean()) / v.std()
        env = np.column_stack([z((anc.admixed.to_numpy() == 0).astype(float)), z((anc.AFR.to_numpy() - anc.AFR.mean()) ** 2)])
    s2e = 1 - S['h2'] - S['s2u'] - S['s2c'] - S['s2g'] - S['s2q']
    Y = np.zeros((N, len(names)))
    mix = lambda x1, x2, rho: rho * x1 + np.sqrt(1 - rho ** 2) * x2
    for r in range(R):
        rng = rng0(r, 2)
        u1, c1, e1 = family_draw(rng), rng.standard_normal(house.max() + 1)[house], rng.standard_normal(N)
        Y[:, r] = (np.sqrt(S['h2']) * G[:, r] + np.sqrt(S['s2u']) * u1 + np.sqrt(S['s2c']) * c1 + np.sqrt(S['s2g']) * env[:, 0]
                   + np.sqrt(S['s2q']) * env[:, 1] + np.sqrt(s2e) * e1)
        if RG:
            u2, c2, e2 = family_draw(rng), rng.standard_normal(house.max() + 1)[house], rng.standard_normal(N)
            Y[:, R + r] = (np.sqrt(S['h2']) * G[:, R + r] + np.sqrt(S['s2u']) * mix(u1, u2, S['ru'])
                           + np.sqrt(S['s2c']) * mix(c1, c2, S['rc']) + np.sqrt(s2e) * mix(e1, e2, S['re']))
    ph = fam.copy()
    ph[names] = Y
    ph.to_csv(f'{W}/pheno.tsv', sep='\t', index=False, float_format='%.8g')
    pd.DataFrame(truth).to_csv(f'{W}/truth.tsv', sep='\t', index=False)


if not os.path.exists(f'{W}/truth.tsv'):
    phenotypes()

# GWAS of everyone and of the unrelated subset
unrel = set(read_fam(f'{C}/unrel').IID)
if not os.path.exists(f'{W}/pheno_unrel.tsv'):
    ph = pd.read_csv(f'{W}/pheno.tsv', sep='\t', dtype={'FID': str, 'IID': str})
    ph[ph.IID.isin(unrel)].to_csv(f'{W}/pheno_unrel.tsv', sep='\t', index=False, float_format='%.8g')
for arm, bfile, pheno, cov in (('K', 'full', 'pheno.tsv', 'covar_all.txt'), ('U', 'unrel', 'pheno_unrel.tsv', 'covar_unrel.txt')):
    if not os.path.exists(f'{W}/ss_{arm}/{names[-1]}.tsv'):
        gwas(a.plink2, f'{C}/{bfile}', f'{W}/{pheno}', names, f'{C}/{cov}', npc, f'{W}/ss_{arm}', a.threads)

# related-pair parts: one REL_M value per trait and MAF-LD bin (REL_M12 per pair of traits for rg)
A8 = f'--annot {C}/annot8.tsv.gz'
if not os.path.exists(f'{W}/rel.rel_m.tsv'):
    run(f'{a.summit} relatives moments --geno {C}/full.bed --pheno {W}/pheno.tsv --covar {C}/covar_all.txt {A8} '
        f'--pairs {C}/pairs.txt --kinship-col K_grm --kinship-min 0.0442 {"--cross" if RG else ""} --out {W}/rel', log=f'{W}/rel.out')
for tag, annot in (('XA1', ''), ('XA8', A8)):
    os.makedirs(f'{W}/ss_{tag}', exist_ok=True)
    run_all([f'{a.summit} relatives append --sumstats {W}/ss_K/{n}.tsv --rel {W}/rel.rel_m.tsv --pheno {n} {annot} '
             f'--out {W}/ss_{tag}/{n}.tsv > /dev/null' for n in names if not os.path.exists(f'{W}/ss_{tag}/{n}.tsv')], a.threads)

# long-range LD sums of each analysed sample (once per cohort)
lr = {}
if a.long_range:
    for arm, geno, cov, pairs in (('all', 'full', 'covar_all.txt', f'--pairs {C}/pairs.txt --kinship-col K_grm --kinship-min 0.0442'),
                                  ('unrel', 'unrel', 'covar_unrel.txt', '')):
        lr[arm] = f'{C}/ld/longrange_{arm}'
        if not os.path.exists(lr[arm] + '.longrange.json'):
            run(f'{a.summit} relatives longrange --geno {C}/{geno}.bed --covar {C}/{cov} {pairs} {A8} --out {lr[arm]}')
LR = lambda arm: f'--long-range-se {lr[arm]}' if lr else ''

# fits
L = f'{C}/ld'
os.makedirs(f'{W}/fit', exist_ok=True)
ARMS = (('U', 'ss_U', 'unrel'), ('K', 'ss_K', 'all'), ('XA', 'ss_XA', 'all'))
cmds, rows = [], []
for arm, ss, sample in ARMS:
    for model, ld, annot in (('single', 'ld1', ''), ('8bin', 'ld8', A8)):
        ssd = ss + ('1' if model == 'single' else '8') if arm == 'XA' else ss
        pre = f'{W}/fit/{arm}.{model}'
        if not RG and not os.path.exists(pre + '.results.tsv'):
            cmds.append(f'{ONE_THREAD} {a.summit} --h2 {W}/{ssd} --ldscores {L}/{sample}.{ld}.gw.ldscore.gz {annot} --njack 200 '
                        f'{LR(sample)} --num-threads 1 --out {pre} > {pre}.out 2>&1')
        if RG:
            for r in range(R):
                p1, p2 = (f'{W}/ph/{sample}_{x}{r}.txt' for x in 'ab')
                m12 = ''
                if arm == 'XA':
                    m12 = f'{W}/fit/m12.{model}.{r}.tsv'
                    run(f'{a.summit} relatives cross-m12 --rel {W}/rel.rel_m12.tsv --pheno1 a{r} --pheno2 b{r} {annot} --out {m12}')
                    m12 = f'--rel-m12 {m12}'
                cov = f'{C}/covar_{sample}.txt'
                pre_r = f'{pre}.{r}'
                if not os.path.exists(pre_r + '.log'):
                    cmds.append(f'{ONE_THREAD} {a.summit} --rg {W}/{ssd}/a{r}.tsv,{W}/{ssd}/b{r}.tsv '
                                f'--ldscores {L}/{sample}.{ld}.gw.ldscore.gz {annot} --pheno-rg {p1},{p2} --pheno-rg-cov {cov},{cov} '
                                f'{m12} --keep-ambiguous --njack 200 {LR(sample)} --num-threads 1 --out {pre_r} > /dev/null 2>&1')
if RG:                                                         # one phenotype per file for --pheno-rg
    os.makedirs(f'{W}/ph', exist_ok=True)
    ph = pd.read_csv(f'{W}/pheno.tsv', sep='\t', dtype={'FID': str, 'IID': str})
    for sample, sel in (('all', np.ones(len(ph), bool)), ('unrel', ph.IID.isin(unrel).to_numpy())):
        for n in names:
            ph.loc[sel, ['FID', 'IID', n]].rename(columns={n: 'PHENO'}).to_csv(f'{W}/ph/{sample}_{n}.txt', sep=' ', index=False)
run_all(cmds, a.threads)

# results
truth = pd.read_csv(f'{W}/truth.tsv', sep='\t')
NUM = r'(-?(?:[0-9.]+(?:e[-+]?[0-9]+)?|nan|inf))'
for arm, *_ in ARMS:
    for model in ('single', '8bin'):
        if not RG:
            for x in read_h2_results(f'{W}/fit/{arm}.{model}.results.tsv').itertuples():
                rows.append(dict(rep=int(x.phen[1:]), arm=arm, model=model, est=x.h2, se=x.se))
            continue
        for r in range(R):
            text = open(f'{W}/fit/{arm}.{model}.{r}.log').read()
            gc = re.findall(rf'Total genetic covariance \(gamma_g\): {NUM} \(SE: {NUM}\)', text)[-1]
            rg = re.findall(rf'Total genetic correlation \(rg\): {NUM} \(SE: {NUM}\)', text)[-1]
            rows.append(dict(rep=r, arm=arm, model=model, est=float(gc[0]), se=float(gc[1]), rg=float(rg[0]), rg_se=float(rg[1])))
out = pd.DataFrame(rows).merge(truth, on='rep')
if S['design'] == 'ancestry':                                   # target after covariates that absorb the environment terms
    out['truth_adjusted'] = S['h2'] / (1 - S['s2g'] - S['s2q'])
out.insert(0, 'label', designs.label(S))
out.insert(0, 'design', S['design'])
out.insert(0, 'setting', a.setting)
out.insert(0, 'cohort', os.path.basename(os.path.normpath(C)))
out.to_csv(f'{W}/results.tsv', sep='\t', index=False, float_format='%.8g')
print(f'{len(out)} rows written to {W}/results.tsv')
