"""Helpers for the All of Us cells "SUMMIT with related individuals" (relatives_cells.py).

Pure numpy/pandas/bed_reader; nothing person-level leaves the Workbench. Imported by the cells from the kit folder, and run as a
script (`python relatives_helpers.py grm ...`) in the kit's own environment for the GRM step.

Related pairs (as in the manuscript): KING (plink2 --make-king-table) on each group's LD-pruned SNPs lists candidate pairs above a
loose kinship. The relatedness of each candidate is then taken from the genetic relatedness matrix of the pruned SNPs after adjustment
for the within-group PCs, on the scale of twice the kinship. Pairs with relatedness above 0.0442 (kinship 0.0221, the lower bound of
fourth-degree relatives) are joined into families, and SUMMIT excludes every pair within a family.
"""
import collections
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------- related pairs
def read_fam(prefix):
    return pd.read_csv(prefix + '.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)


def covariate_basis(X):
    """Orthonormal basis of [1, X] (N x q) by QR; X has no missing values."""
    Z = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    Q, Rr = np.linalg.qr(Z)
    rank = int(np.sum(np.abs(np.diag(Rr)) > 1e-8 * np.abs(Rr).max()))
    assert rank == Z.shape[1], f'covariates are not of full rank ({rank} < {Z.shape[1]})'
    return Q


def grm_relatedness(bfile, snp_ids, pcs, cand, block=1000, pair_chunk=200_000, log=print):
    """Relatedness of candidate pairs from the GRM of `snp_ids` after adjustment for [1, pcs].

    bfile: PLINK 1 prefix; snp_ids: SNP IDs to use (the pruned SNPs); pcs: DataFrame with IID and PC columns for every person in the
    fam (no missing values); cand: DataFrame with IID1, IID2. Genotypes are mean-imputed, centered, projected off the covariates and
    scaled so that x'x = N - q, and the relatedness is x_i'x_i' / M over the M SNPs used (the kernel of SUMMIT's LD scores, without the
    V weights and the other covariates). Returns (cand with a column GRM, M)."""
    from bed_reader import open_bed
    fam = read_fam(bfile)
    bim = pd.read_csv(bfile + '.bim', sep=r'\s+', header=None, usecols=[1], names=['SNP'], dtype=str)
    cols = np.flatnonzero(bim.SNP.isin(set(snp_ids)).to_numpy())
    assert len(cols) > 0, 'none of the pruned SNPs is in the bim'
    P = fam[['IID']].merge(pcs, on='IID', how='left')
    pc_cols = [c for c in pcs.columns if c != 'IID']
    assert len(P) == len(fam) and P[pc_cols].notna().all().all(), 'PCs missing for some group members'
    Q = covariate_basis(P[pc_cols].to_numpy())
    N, q = Q.shape
    nu = N - q
    pos = pd.Index(fam.IID)
    a = pos.get_indexer(cand.IID1.astype(str)); b = pos.get_indexer(cand.IID2.astype(str))
    assert (a >= 0).all() and (b >= 0).all(), 'candidate pair members missing from the fam'
    acc = np.zeros(len(cand))
    used = 0
    bed = open_bed(bfile + '.bed', count_A1=True, num_threads=None)
    for s in range(0, len(cols), block):
        idx = cols[s:s + block]
        X = bed.read(index=np.s_[:, idx], dtype='float32').astype(np.float64)
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
        X -= Q @ (Q.T @ X)
        ss = np.einsum('ij,ij->j', X, X)
        ok = ss > 1e-8
        X = X[:, ok] * np.sqrt(nu / ss[ok])[None, :]
        used += int(ok.sum())
        for c in range(0, len(cand), pair_chunk):
            sl = slice(c, c + pair_chunk)
            acc[sl] += np.einsum('ij,ij->i', X[a[sl]], X[b[sl]])
        if (s // block) % 25 == 0:
            log(f'  GRM: {min(s + block, len(cols))} of {len(cols)} SNPs', flush=True)
    out = cand.copy()
    out['GRM'] = acc / used
    return out, used


def families(pairs, thr, col='GRM'):
    """Connected components of pairs with col > thr; returns {IID: family_id} for people in families and the family sizes."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in pairs.loc[pairs[col] > thr, ['IID1', 'IID2']].itertuples(index=False):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    fam = {x: find(x) for x in parent}
    return fam, collections.Counter(fam.values())


DEGREES = (('4th', 0.0442, 0.0884), ('3rd', 0.0884, 0.177), ('2nd', 0.177, 0.354), ('1st', 0.354, 0.707),
           ('duplicate_or_MZ', 0.707, np.inf))           # relatedness scale (twice kinship), Manichaikul et al. 2010 boundaries


def degree_counts(r):
    """Pairs by degree from relatedness values r."""
    r = np.asarray(r, dtype=float)
    return {lab: int(((r > lo) & (r <= hi)).sum()) for lab, lo, hi in DEGREES}


def king_grm_table(pairs):
    """Candidate pairs by KING kinship class (rows) and GRM relatedness class (columns), as counts."""
    kb = pd.cut(pairs.KING, [-np.inf, 0.0221, 0.0442, 0.0884, 0.177, 0.354, np.inf],
                labels=['KING<=0.0221', '0.0221-0.0442', '0.0442-0.0884', '0.0884-0.177', '0.177-0.354', '>0.354'])
    gb = pd.cut(pairs.GRM, [-np.inf, 0.0442, 0.0884, 0.177, 0.354, 0.707, np.inf],
                labels=['GRM<=0.0442', '4th', '3rd', '2nd', '1st', 'dup/MZ'])
    return pd.crosstab(kb, gb, dropna=False)


# ---------------------------------------------------------------------------------------------------------------- disclosure
def mask_count(n):
    """All of Us rule: counts of 1-20 are shown as '<=20'."""
    n = int(n)
    return '<=20' if 1 <= n <= 20 else n


def mask_parts(total, parts):
    """Masked components and total; the total is withheld when any component is masked, so that it cannot be recovered."""
    m = {k: mask_count(v) for k, v in parts.items()}
    return ('withheld' if any(v == '<=20' for v in m.values()) else mask_count(total)), m


# ---------------------------------------------------------------------------------------------------------------- simulation
def write_score_file(bim_prefix, nrep, out, seed):
    """Random effects for every SNP and `nrep` replicates (standard normal); columns SNP A1 W1..Wnrep. Returns their sums of squares."""
    bim = pd.read_csv(bim_prefix + '.bim', sep=r'\s+', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], dtype=str)
    assert bim.SNP.is_unique, 'duplicate SNP IDs in the bim'
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((len(bim), nrep))
    df = pd.DataFrame(W, columns=[f'W{r + 1}' for r in range(nrep)])
    df.insert(0, 'A1', bim.A1.values); df.insert(0, 'SNP', bim.SNP.values)
    df.to_csv(out, sep='\t', index=False, float_format='%.8g')
    return pd.read_csv(out, sep='\t', usecols=[f'W{r + 1}' for r in range(nrep)]).pow(2).sum().to_numpy()   # as written (rounded)


def read_scores(sscore, nrep):
    """Genetic values (plink2 --score ... variance-standardize cols=scoresums) for every person, one column per replicate."""
    s = pd.read_csv(sscore, sep='\t', dtype={'#FID': str, 'IID': str, 'FID': str, '#IID': str})
    s.columns = [c.lstrip('#') for c in s.columns]
    cols = [f'W{r + 1}_SUM' for r in range(nrep)]
    missing = [c for c in cols if c not in s.columns]
    assert not missing, f'missing score columns in {sscore}: {missing[:3]} (columns: {list(s.columns)[:8]})'
    return s[['IID'] + cols].rename(columns={f'W{r + 1}_SUM': f'g{r + 1}' for r in range(nrep)}).set_index('IID')


def family_effects(ids, pairs, s2u, s2c, rel_min, thr_first, rng, col='GRM'):
    """Shared effects of relatives for people `ids` (the pairs are restricted to them).

    u: variance s2u, covariance s2u * relatedness within families (connected components of pairs with col > rel_min; relatedness of
    the listed pairs, 0 for other pairs of the family, 1 on the diagonal, nearest positive semidefinite matrix), independent between
    families. d: variance s2c, shared within households (connected components of first-degree pairs, col > thr_first). Returns (u, d)."""
    ids = list(map(str, ids)); pos = {x: i for i, x in enumerate(ids)}
    pairs = pairs[pairs.IID1.isin(pos) & pairs.IID2.isin(pos)]
    fam_map, _ = families(pairs, rel_min, col)
    u = rng.standard_normal(len(ids)) * np.sqrt(s2u)
    members = collections.defaultdict(list)
    for x, f in fam_map.items():
        members[f].append(x)
    lookup = {}
    for x, y, k in pairs[['IID1', 'IID2', col]].itertuples(index=False):
        lookup[(x, y)] = lookup[(y, x)] = k
    for f in sorted(members, key=str):
        mem = sorted(members[f])
        Phi = np.eye(len(mem))
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                Phi[i, j] = Phi[j, i] = lookup.get((mem[i], mem[j]), 0.0)
        w, v = np.linalg.eigh(Phi)
        L = v * np.sqrt(np.clip(w, 0, None))
        u[[pos[x] for x in mem]] = L @ rng.standard_normal(len(mem)) * np.sqrt(s2u)
    hh, hh_sizes = families(pairs, thr_first, col)
    hh_effect = {h: rng.standard_normal() * np.sqrt(s2c) for h in sorted(hh_sizes, key=str)}
    d = np.array([hh_effect[hh[x]] if x in hh else rng.standard_normal() * np.sqrt(s2c) for x in ids])
    return u, d


# ---------------------------------------------------------------------------------------------------------------- statistics
def irnt(x):
    from scipy.stats import norm, rankdata
    x = pd.Series(x, dtype=float); ok = x.notna(); r = pd.Series(np.nan, index=x.index)
    r[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / ok.sum())
    return r


def glm_to_sumstats(glm, out, cov_rank):
    """plink2 --glm (cols=chrom,pos,ref,alt,omitted,nobs,beta,se) to SUMMIT's format with the covariate rank (excluding the intercept).
    Returns the largest per-SNP sample size."""
    r = pd.read_csv(glm, sep='\t'); r.columns = [c.lstrip('#') for c in r.columns]
    r = r[np.isfinite(r.BETA) & np.isfinite(r.SE) & (r.SE > 0)]
    a2 = r.OMITTED if 'OMITTED' in r else np.where(r.A1 == r.ALT, r.REF, r.ALT)
    pd.DataFrame({'SNP': r.ID, 'CHR': r.CHROM, 'BP': r.POS, 'A1': r.A1, 'A2': a2, 'N': r.OBS_CT, 'BETA': r.BETA, 'SE': r.SE,
                  'Z': r.BETA / r.SE, 'COV_RANK': int(cov_rank)}).to_csv(out, sep='\t', index=False)
    return int(r.OBS_CT.max())


def read_results(results_tsv):
    """SUMMIT --h2 results (one row per phenotype; directory input) as phen, h2, se."""
    r = pd.read_csv(results_tsv, sep='\t', dtype={'phen': str})
    return r[['phen', 'h2', 'h2_se']].rename(columns={'h2_se': 'se'})


def summarize_sims(df, truth_col='truth', est='h2', se='se', t=1.9720):
    """Mean, bias, Monte Carlo s.e. of the bias, SD, mean s.e., s.e./SD and coverage of t intervals (199 degrees of freedom)."""
    d = df
    return pd.Series({'n_rep': len(d), 'truth': float(d[truth_col].iloc[0]), 'mean': d[est].mean(),
                      'bias': d[est].mean() - d[truth_col].iloc[0], 'mcse': d[est].std(ddof=1) / np.sqrt(len(d)),
                      'sd': d[est].std(ddof=1), 'mean_se': d[se].mean(), 'se_over_sd': d[se].mean() / d[est].std(ddof=1),
                      'coverage': float(np.mean(np.abs(d[est] - d[truth_col]) <= t * d[se]))})


def write_json(obj, path):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


# ---------------------------------------------------------------------------------------------------------------- command line
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(prog='relatives_helpers.py')
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('grm', help='GRM relatedness (PC-adjusted, pruned SNPs) of the KING candidate pairs')
    g.add_argument('--bfile', required=True)
    g.add_argument('--snps', required=True, help='SNP IDs, one per line (the LD-pruned SNPs)')
    g.add_argument('--pcs', required=True, help='IID PC1 ... PCk for every person in the fam')
    g.add_argument('--kin0', required=True, help='plink2 --make-king-table output')
    g.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'grm':
        k = pd.read_csv(a.kin0, sep=r'\s+', dtype={'#FID1': str, 'FID1': str, 'IID1': str, 'FID2': str, 'IID2': str})
        k.columns = [c.lstrip('#') for c in k.columns]
        cand = k[['IID1', 'IID2', 'KINSHIP']].rename(columns={'KINSHIP': 'KING'}).reset_index(drop=True)
        snps = pd.read_csv(a.snps, header=None, dtype=str)[0]
        pcs = pd.read_csv(a.pcs, sep='\t', dtype={'IID': str})
        out, used = grm_relatedness(a.bfile, snps, pcs, cand)
        tmp = a.out + '.tmp'
        out.to_csv(tmp, sep='\t', index=False, float_format='%.6g')
        import os
        os.replace(tmp, a.out)
        print(f'GRM from {used} SNPs for the candidate pairs', flush=True)   # counts are printed, masked, by the cell
