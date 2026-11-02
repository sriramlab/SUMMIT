"""Heritability due to direct genetic effects from sibships (`summit relatives direct`).

  sibships  full siblings are pairs with kinship in (0.177, 0.354] and IBS0 > 0.0012 (KING's first-degree range; parents
            and offspring share no IBS0 segment); sibships are connected components of sibling pairs and extended families
            connected components of all pairs with kinship above 0.0442. Writes OUT/sibships.tsv (IID SIBSHIP FAMILY) for
            members of sibships with at least two genotyped people, and OUT/sibs.keep.
  simulate  replicate phenotypes on the siblings' genotypes with direct heritability 0.4 and one family-level term per
            setting: base (none), sibenv (sibling environment 0.2), extfam (sibling environment 0.1, extended-family
            environment 0.05 and untagged effects shared by relatedness 0.1), nurture (the sibship's mean of a second genetic
            score correlated 0.5 with the direct one, 0.1), ancestry (the sibship's mean of a genotype PC that is not adjusted
            for, 0.05) and sibeffect (each sibling's genetic value acting on the others, 0.02; target
            (sqrt(0.4) - sqrt(0.02))^2). Then the direct and population estimates of every replicate.
  real      direct and population estimates of the real traits of traits.py, with age and sex terms regressed out before
            centering within sibships and PCs added for the population estimate.
Class GRMs of the siblings are cached in OUT/grm. Writes OUT/direct.direct.tsv (and OUT/truth.tsv for simulate).
usage: python direct_effects.py sibships --kinship KIN --pairs PAIRS --fam FAM --out DIR
       python direct_effects.py simulate --geno PREFIX --sibships FILE --annot ANNOT --pairs PAIRS --pcs PCS --out DIR
       python direct_effects.py real --geno PREFIX --sibships FILE --annot ANNOT --traits DIR --covar COV --out DIR
"""
import argparse
import os

import numpy as np
import pandas as pd
from bed_reader import open_bed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import read_fam, run

SETTINGS = {'base': {}, 'sibenv': dict(c=0.2), 'extfam': dict(c=0.1, x=0.05, u=0.1), 'nurture': dict(n=0.1),
            'ancestry': dict(a=0.05), 'sibeffect': dict(s=0.02)}
H2 = 0.4


def components(i, j, n):
    return connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[1]


def sibships(a):
    fam = pd.read_csv(a.fam, sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)
    pos = pd.Series(np.arange(len(fam)), index=fam.IID)
    kin = pd.read_csv(a.kinship, sep=r'\s+', dtype={'ID1': str, 'ID2': str})
    kin = kin[kin.ID1.isin(pos.index) & kin.ID2.isin(pos.index)]
    sib = kin[(kin.Kinship > 0.177) & (kin.Kinship <= 0.354) & (kin.IBS0 > 0.0012)]
    rel = kin[kin.Kinship > 0.0442]
    s = components(pos[sib.ID1].to_numpy(), pos[sib.ID2].to_numpy(), len(fam))
    f = components(pos[rel.ID1].to_numpy(), pos[rel.ID2].to_numpy(), len(fam))
    keep = np.bincount(s)[s] >= 2
    out = pd.DataFrame({'IID': fam.IID[keep], 'SIBSHIP': s[keep], 'FAMILY': f[keep]})
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(f'{a.out}/sibships.tsv', sep='\t', index=False)
    fam[keep].to_csv(f'{a.out}/sibs.keep', sep='\t', index=False, header=False)
    print(f'{keep.sum()} siblings in {out.SIBSHIP.nunique()} sibships and {out.FAMILY.nunique()} extended families')


def sib_mean(v, sib):
    """Mean of each sibship, returned per person (v: people x replicates)."""
    sums = np.zeros((sib.max() + 1, v.shape[1]))
    np.add.at(sums, sib, v)
    return (sums / np.bincount(sib)[:, None])[sib]


def simulate(a):
    fam = read_fam(a.geno)
    sibs = pd.read_csv(a.sibships, sep='\t', dtype={'IID': str}).set_index('IID').loc[fam.IID]
    sib, ext = pd.factorize(sibs.SIBSHIP)[0], pd.factorize(sibs.FAMILY)[0]
    N, R = len(fam), a.nrep
    rng = np.random.default_rng([a.seed, 1])
    bed = open_bed(a.geno + '.bed', count_A1=True)
    Gd, Gn = np.zeros((N, R)), np.zeros((N, R))
    for s in range(0, bed.shape[1], 4096):
        X = bed.read(index=np.s_[:, s:s + 4096], dtype='float64')
        X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        X -= X.mean(0)
        sd = X.std(0, ddof=1)
        X = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)
        b1 = rng.standard_normal((X.shape[1], R))
        Gd += X @ b1
        Gn += X @ (a.rho * b1 + np.sqrt(1 - a.rho ** 2) * rng.standard_normal((X.shape[1], R)))
    Gd, Gn = Gd / np.sqrt(bed.shape[1]), Gn / np.sqrt(bed.shape[1])
    # untagged effects shared by relatedness within extended families: covariance 2 x kinship of the listed pairs
    pairs = pd.read_csv(a.pairs, sep=r'\s+', dtype={'ID1': str, 'ID2': str})
    pos = pd.Series(np.arange(N), index=fam.IID)
    pairs = pairs[pairs.ID1.isin(pos.index) & pairs.ID2.isin(pos.index)]
    pi, pj, rel = pos[pairs.ID1].to_numpy(), pos[pairs.ID2].to_numpy(), 2 * pairs.Kinship.to_numpy()
    roots = []
    for e in np.unique(ext):
        idx = np.flatnonzero(ext == e)
        if idx.size > 1:
            Phi = np.eye(idx.size)
            loc = {x: k for k, x in enumerate(idx)}
            inside = np.isin(pi, idx)
            Phi[[loc[x] for x in pi[inside]], [loc[x] for x in pj[inside]]] = rel[inside]
            Phi = np.maximum(Phi, Phi.T)
            w, v = np.linalg.eigh(Phi)
            roots.append((idx, v * np.sqrt(np.clip(w, 0, None))))
    hidden = pd.read_csv(a.pcs, sep=' ', dtype={'IID': str}).set_index('IID').loc[fam.IID, a.hidden_pc].to_numpy()
    hb = sib_mean(((hidden - hidden.mean()) / hidden.std())[:, None], sib)[:, 0]
    nsib = np.bincount(sib)[sib][:, None]
    columns, truth = {}, []
    for k, (name, S) in enumerate(SETTINGS.items()):
        g = np.random.default_rng([a.seed, 2, k])
        Y = np.sqrt(H2) * Gd
        if 'c' in S:
            Y += np.sqrt(S['c']) * g.standard_normal((sib.max() + 1, R))[sib]
        if 'x' in S:
            Y += np.sqrt(S['x']) * g.standard_normal((ext.max() + 1, R))[ext]
        if 'u' in S:
            U = g.standard_normal((N, R))
            for idx, L in roots:
                U[idx] = L @ U[idx]
            Y += np.sqrt(S['u']) * U
        if 'n' in S:
            nb = sib_mean(Gn, sib)
            Y += np.sqrt(S['n']) * nb / nb.std(0)
        if 'a' in S:
            Y += np.sqrt(S['a']) * (hb / hb.std())[:, None]
        if 's' in S:
            Y += np.sqrt(S['s']) * (sib_mean(Gd, sib) * nsib - Gd)
        Y += np.sqrt(1 - H2 - sum(S.values())) * g.standard_normal((N, R))
        target = (np.sqrt(H2) - np.sqrt(S['s'])) ** 2 if 's' in S else H2
        for r in range(R):
            columns[f'{name}_{r}'] = Y[:, r]
            truth.append(dict(trait=f'{name}_{r}', setting=name, rep=r, truth=target))
    os.makedirs(a.out, exist_ok=True)
    ph = pd.concat([fam, pd.DataFrame(columns)], axis=1)
    ph.to_csv(f'{a.out}/pheno.tsv', sep='\t', index=False, float_format='%.8g')
    pd.DataFrame(truth).to_csv(f'{a.out}/truth.tsv', sep='\t', index=False)
    cov = pd.read_csv(a.pcs, sep=' ', dtype={'FID': str, 'IID': str})[['FID', 'IID'] + [f'PC{j}' for j in range(1, 11)]]
    cov.to_csv(f'{a.out}/covar.txt', sep=' ', index=False, float_format='%.10g')
    fit(a, f'{a.out}/pheno.tsv', f'{a.out}/covar.txt', '')


def fit(a, pheno, covar, within):
    os.makedirs(f'{a.out}/grm', exist_ok=True)
    run(f'{a.summit} relatives direct --geno {a.geno} --sibships {a.sibships} --annot {a.annot} --pheno {pheno} '
        f'--covar {covar} {within} --grm {a.out}/grm --nblocks 200 --out {a.out}/direct', log=f'{a.out}/direct.out')


def real(a):
    P = a.traits
    cov = pd.read_csv(a.covar, sep=r'\s+', dtype={'FID': str, 'IID': str})
    ag = pd.read_csv(f'{P}/agesex_all.tsv', sep='\t', dtype={'FID': str, 'IID': str}).drop(columns='FID')
    cov = cov.merge(ag, on='IID', how='inner')
    os.makedirs(a.out, exist_ok=True)
    cov.to_csv(f'{a.out}/covar.txt', sep=' ', index=False, float_format='%.10g')
    fit(a, f'{P}/pheno_all.tsv', f'{a.out}/covar.txt', '--within-covar ' + ','.join(c for c in ag.columns if c != 'IID'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['sibships', 'simulate', 'real'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--kinship', help='KING kinship table (ID1 ID2 ... IBS0 Kinship)')
    ap.add_argument('--pairs', help='the same table, for the simulated untagged effects')
    ap.add_argument('--fam')
    ap.add_argument('--geno')
    ap.add_argument('--sibships')
    ap.add_argument('--annot')
    ap.add_argument('--pcs', help='FID IID PC1..PC20 of the siblings (prepare_cohort.py pcs20.txt)')
    ap.add_argument('--hidden-pc', default='PC15')
    ap.add_argument('--traits')
    ap.add_argument('--covar', help='FID IID PC1..PC10 of the siblings (real step)')
    ap.add_argument('--nrep', type=int, default=200)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--summit', default='summit')
    a = ap.parse_args()
    {'sibships': sibships, 'simulate': simulate, 'real': real}[a.step](a)


if __name__ == '__main__':
    main()
