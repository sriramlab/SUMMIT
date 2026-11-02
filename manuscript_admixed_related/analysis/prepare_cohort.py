"""Inputs of one analysis cohort (participants with their relatives) from a quality-controlled PLINK fileset.

Steps: SNPs with MAF >= 0.01 of the participants in --keep; LD-pruned PCA SNPs (MAF >= 0.05, missingness <= 0.01, long-range
LD regions excluded, r2 < 0.1 in 500 kb); PCs fitted in everyone, relatedness from the PC-adjusted GRM of the pruned SNPs, an
unrelated subset (no pair above 0.0442), PCs refitted in that subset with everyone projected, and relatedness and the
unrelated subset again; families (connected components of pairs above 0.0442) and households (components of first-degree
pairs, above 0.354); covariates; MAF; and in-sample genome-wide LD scores of everyone (with the related-pair parts) and of
the unrelated subset, for one component and for 8 MAF-LD bins (MAF below or above 0.1 crossed with quartiles of the
single-component LD score, computed with a different random seed so that the bins do not select on its Monte Carlo noise).
With --ancestry (IID, AFR = African proportion, admixed = 0/1), two covariates are added: an indicator of unadmixed
participants and the squared centred African proportion.

Writes to OUT: full.{bed,bim,fam}, unrel.{bed,bim,fam}, covar_all.txt, covar_unrel.txt, pcs20.txt, pairs.txt (IID1 IID2
K_grm, pairs above 0.0442), pairs_all.tsv.gz (pairs above 0.02), families.tsv.gz (IID family household), family_blocks.tsv.gz
(within-family GRM entries i j K_grm, diagonal included), maf.tsv, annot8.tsv.gz and ld/{all,unrel}.{ld1,ld8}.gw.ldscore.gz.
usage: python prepare_cohort.py --bfile QC --keep KEEP --long-range-ld REGIONS --out OUT [--npc 10] [--ancestry FILE]
"""
import argparse
import heapq
import os
import subprocess

import numpy as np
import pandas as pd
from bed_reader import open_bed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ap = argparse.ArgumentParser()
ap.add_argument('--bfile', required=True)
ap.add_argument('--keep', required=True, help='FID IID of the cohort')
ap.add_argument('--long-range-ld', required=True, help='plink --exclude range file (MHC and other long-range LD regions)')
ap.add_argument('--out', required=True)
ap.add_argument('--npc', type=int, default=10)
ap.add_argument('--ancestry', default=None)
ap.add_argument('--plink2', default='plink2')
ap.add_argument('--summit', default='summit')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--nvecs', type=int, default=20000, help='random vectors for the LD scores')
a = ap.parse_args()
O = a.out
W = f'{O}/build'
os.makedirs(f'{O}/ld', exist_ok=True)
os.makedirs(W, exist_ok=True)
REL, FIRST = 0.0442, 0.354


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def plink(args):
    run(f'{a.plink2} {args} --threads {a.threads}')


def read_fam(prefix):
    return pd.read_csv(prefix + '.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)


def pcs_from(eigen_prefix, sample_prefix, keep=None):
    """Fit PCs (in `keep`, or everyone) and project every participant; returns FID IID PC1..PC20 (unit variance)."""
    k = f'--keep {keep}' if keep else ''
    plink(f'--bfile {sample_prefix} {k} --freq --out {eigen_prefix}')
    plink(f'--bfile {sample_prefix} {k} --pca 20 allele-wts --out {eigen_prefix}')
    plink(f'--bfile {sample_prefix} --read-freq {eigen_prefix}.afreq --score {eigen_prefix}.eigenvec.allele 2 6 header-read '
          f'no-mean-imputation variance-standardize --score-col-nums 7-26 --out {eigen_prefix}.proj')
    s = pd.read_csv(f'{eigen_prefix}.proj.sscore', sep='\t', dtype={'#FID': str, 'IID': str})
    out = pd.DataFrame({'FID': s['#FID'], 'IID': s.IID})
    for j in range(1, 21):
        out[f'PC{j}'] = s[f'PC{j}_AVG'] / s[f'PC{j}_AVG'].std()
    return out


def grm_pairs(prefix, pcs):
    """PC-adjusted GRM of the pruned SNPs (mean imputation, standardization, projection on [1, PCs]); pairs above 0.02."""
    bed = open_bed(prefix + '.bed', count_A1=True, num_threads=a.threads)
    fam = read_fam(prefix)
    V = np.column_stack([np.ones(len(fam)), pcs.set_index('IID').loc[fam.IID, [f'PC{j}' for j in range(1, a.npc + 1)]]])
    Q = np.linalg.qr(V)[0]
    N, M = bed.shape
    K = np.zeros((N, N))
    used = 0
    for s in range(0, M, 4096):
        X = bed.read(index=np.s_[:, s:s + 4096], dtype='float64')
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu, X) - mu
        sd = X.std(0, ddof=1)
        X = X[:, sd > 0] / sd[sd > 0]
        X -= Q @ (Q.T @ X)
        X *= np.sqrt((N - Q.shape[1]) / (X * X).sum(0))
        K += X @ X.T
        used += X.shape[1]
    K /= used
    i, j = np.triu_indices(N, 1)
    sel = K[i, j] > 0.02
    pairs = pd.DataFrame({'i': i[sel], 'j': j[sel], 'IID1': fam.IID.to_numpy()[i[sel]], 'IID2': fam.IID.to_numpy()[j[sel]],
                          'K_grm': K[i, j][sel]})
    return pairs, K


def unrelated(pairs, iids, thr=REL):
    """Greedy maximal set with no pair above thr: repeatedly drop the participant with most remaining relatives."""
    pos = {x: k for k, x in enumerate(iids)}
    adj = [set() for _ in iids]
    for x, y in pairs.loc[pairs.K_grm > thr, ['IID1', 'IID2']].itertuples(index=False):
        adj[pos[x]].add(pos[y])
        adj[pos[y]].add(pos[x])
    alive = np.ones(len(iids), bool)
    heap = [(-len(adj[k]), -k, k) for k in range(len(iids)) if adj[k]]
    heapq.heapify(heap)
    while heap:
        d, _, k = heapq.heappop(heap)
        if not alive[k] or not adj[k]:
            continue
        if -d != len(adj[k]):
            heapq.heappush(heap, (-len(adj[k]), -k, k))
            continue
        alive[k] = False
        for m in adj[k]:
            adj[m].discard(k)
        adj[k] = set()
    return np.asarray(iids)[alive]


def components(pairs, n, thr):
    sel = pairs.K_grm > thr
    g = coo_matrix((np.ones(sel.sum()), (pairs.i[sel], pairs.j[sel])), shape=(n, n))
    return connected_components(g, directed=False)[1]


# genotypes and PCA SNPs
plink(f'--bfile {a.bfile} --keep {a.keep} --maf 0.01 --make-bed --out {O}/full')
plink(f'--bfile {O}/full --maf 0.05 --geno 0.01 --exclude range {a.long_range_ld} --indep-pairwise 500kb 1 0.1 --out {W}/prune')
plink(f'--bfile {O}/full --extract {W}/prune.prune.in --make-bed --out {W}/pruned')
fam = read_fam(f'{O}/full')
N = len(fam)

# relatedness with PCs of everyone, then PCs of the unrelated subset and relatedness again
pairs, _ = grm_pairs(f'{W}/pruned', pcs_from(f'{W}/pca1', f'{W}/pruned'))
keep1 = unrelated(pairs, fam.IID.tolist())
pd.DataFrame({'FID': fam.set_index('IID').loc[keep1, 'FID'], 'IID': keep1}).to_csv(f'{W}/unrel1.keep', sep='\t', index=False,
                                                                                   header=False)
pcs = pcs_from(f'{W}/pca2', f'{W}/pruned', keep=f'{W}/unrel1.keep')
pairs, K = grm_pairs(f'{W}/pruned', pcs)
keep = unrelated(pairs, fam.IID.tolist())
pd.DataFrame({'FID': fam.set_index('IID').loc[keep, 'FID'], 'IID': keep}).to_csv(f'{W}/unrel.keep', sep='\t', index=False,
                                                                                 header=False)
plink(f'--bfile {O}/full --keep {W}/unrel.keep --make-bed --out {O}/unrel')
pcs.to_csv(f'{O}/pcs20.txt', sep=' ', index=False, float_format='%.10g')

# relatives: pairs, families, households and the within-family GRM blocks used to simulate shared untagged effects
pairs.to_csv(f'{O}/pairs_all.tsv.gz', sep='\t', index=False, float_format='%.6g')
pairs.loc[pairs.K_grm > REL, ['IID1', 'IID2', 'K_grm']].to_csv(f'{O}/pairs.txt', sep=' ', index=False, float_format='%.6g')
family, household = components(pairs, N, REL), components(pairs, N, FIRST)
pd.DataFrame({'IID': fam.IID, 'family': family, 'household': household}).to_csv(f'{O}/families.tsv.gz', sep='\t', index=False)
blocks = []
size = np.bincount(family)
for f in np.flatnonzero(size > 1):
    idx = np.flatnonzero(family == f)
    i, j = np.triu_indices(len(idx))
    blocks.append(pd.DataFrame({'i': idx[i], 'j': idx[j], 'K_grm': K[idx[i], idx[j]]}))
pd.concat(blocks).to_csv(f'{O}/family_blocks.tsv.gz', sep='\t', index=False, float_format='%.6g')

# covariates
cov = fam.merge(pcs[['IID'] + [f'PC{j}' for j in range(1, a.npc + 1)]], on='IID', how='left')
if a.ancestry:
    anc = pd.read_csv(a.ancestry, sep='\t', dtype={'IID': str}).set_index('IID').loc[cov.IID]
    cov['UNADMIXED'] = (anc.admixed.to_numpy() == 0).astype(float)
    cov['AFR2'] = (anc.AFR.to_numpy() - anc.AFR.mean()) ** 2
assert cov.notna().all().all()
cov.to_csv(f'{O}/covar_all.txt', sep=' ', index=False, float_format='%.10g')
cov[cov.IID.isin(set(keep))].to_csv(f'{O}/covar_unrel.txt', sep=' ', index=False, float_format='%.10g')

# MAF, LD scores and the 8 MAF-LD bins
bim = pd.read_csv(f'{O}/full.bim', sep=r'\s+', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], dtype={'SNP': str})
bed = open_bed(f'{O}/full.bed', count_A1=True, num_threads=a.threads)
af = np.concatenate([np.nanmean(bed.read(index=np.s_[:, s:s + 4096], dtype='float64'), axis=0) / 2
                     for s in range(0, bed.shape[1], 4096)])
maf = np.minimum(af, 1 - af)
pd.DataFrame({'SNP': bim.SNP, 'MAF': maf}).to_csv(f'{O}/maf.tsv', sep='\t', index=False, float_format='%.8g')
common = f'--nvecs {a.nvecs} --block-size 16384 --dtype float64 --num-threads {a.threads}'
rel = f'--relative-pairs {O}/pairs.txt --kinship-col K_grm --kinship-min {REL}'
run(f'{a.summit} --geno {O}/full.bed --covar {O}/covar_all.txt {rel} {common} --seed 1 --out {O}/ld/all.ld1')
ld = pd.read_csv(f'{O}/ld/all.ld1.gw.ldscore.gz', sep='\t', dtype={'SNP': str}).set_index('SNP').L2_0.loc[bim.SNP].to_numpy()
annot = bim[['CHR', 'BP', 'SNP', 'CM']].copy()
for cls, sel in (('A', maf < 0.1), ('B', maf >= 0.1)):
    q = pd.qcut(pd.Series(ld[sel]).rank(method='first'), 4, labels=False).to_numpy() + 1
    for b in range(1, 5):
        col = np.zeros(len(bim), dtype=int)
        col[np.flatnonzero(sel)[q == b]] = 1
        annot[f'maf{cls}_ld{b}'] = col
annot.to_csv(f'{O}/annot8.tsv.gz', sep='\t', index=False, compression='gzip')
A8 = f'--annot {O}/annot8.tsv.gz'
run(f'{a.summit} --geno {O}/full.bed --covar {O}/covar_all.txt {A8} {rel} {common} --seed 2 --out {O}/ld/all.ld8')
run(f'{a.summit} --geno {O}/unrel.bed --covar {O}/covar_unrel.txt {common} --seed 3 --out {O}/ld/unrel.ld1')
run(f'{a.summit} --geno {O}/unrel.bed --covar {O}/covar_unrel.txt {A8} {common} --seed 4 --out {O}/ld/unrel.ld8')
print(f'{N} participants, {len(keep)} unrelated, {(pairs.K_grm > REL).sum()} related pairs, {(size > 1).sum()} families')
