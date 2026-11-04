"""European and African-ancestry participants in one analysis (Appendix G, Fig. 7): heritability within each group and the
genetic covariance between the groups, from all pairs of different, unrelated participants.

Groups: the European group (white British participants with their relatives) and the wider group of all participants of
African ancestry (admixed and unadmixed members, relatives included). Genotypes are standardized within each group and
then adjusted for that group's covariates (K = P Z Z' P / M). With SNP effects that may differ between the groups,

    E(y_i y_i') = s_gh K_ii' (+ s_LA L_ii' when i and i' are both African-ancestry participants)   for unrelated pairs,

with a residual variance per group. L is the local-ancestry kernel (dosages standardized, then adjusted for the
African-ancestry covariates). Related pairs within a group (pairs.txt, PC-adjusted GRM above 0.0442), pairs across groups
with K above the same value and, with --closure, all other pairs of the families they form are left out of the
estimating equations, as is each individual paired with themselves (with the exact covariate term sum_i K_ii h_ii). The
equations separate by blocks of pairs: European (s_EE, residual), African ancestry (s_AA, s_LA, residual) and between the
groups (s_EA alone), so that rg = s_EA / sqrt(s_EE s_AA). The pooled fit uses one kernel for every pair.

Standard errors: jackknife over contiguous blocks of SNPs (local-ancestry markers go to the block that covers their
position). --check compares the fast delete-block quantities with the brute-force jackknife on a small subset.

Inputs (--inp): EUR/ and AFRW/ (full.{bed,bim,fam}, pairs.txt, real/agesex.tsv, real/real_all.tsv) from
prepare_cohort.py and traits.py; covariate folders EUR, EURO, AFRWX and AFRWO (covar_all.txt: PCs, with the unadmixed
indicator and squared African proportion for AFRWX/AFRWO and the origin covariates for EURO/AFRWO); afrw.la.{bed,bim,fam},
the local-ancestry dosages of the African-ancestry group from local_ancestry.py.

The paper's runs:
  real traits   --cov origin --closure --la-theta --drop-overlap
  simulations   --cov base --sim --shared kinship --closure --la-theta --drop-overlap
                --cov base --sim --shared kinship --closure --la-theta --drop-overlap --scenarios rho1_h2,rho05_h2 --tag _h2
usage: python ancestry_groups.py --inp DIR --out DIR [--cov base|origin] [--blocks 200] [--sim --reps 100] [--check]
       [--closure] [--la-theta] [--drop-overlap] [--shared family|kinship]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from bed_reader import open_bed
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--cov', choices=('base', 'origin'), default='base')
ap.add_argument('--blocks', type=int, default=200); ap.add_argument('--chunk', type=int, default=8192)
ap.add_argument('--sim', action='store_true'); ap.add_argument('--reps', type=int, default=100)
ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--check', action='store_true', help='small subset; brute-force jackknife comparison')
ap.add_argument('--no-la', action='store_true')
ap.add_argument('--scenarios', default='rho1,rho05,rho05_la,rho05_t3', help='comma-separated simulation scenarios')
ap.add_argument('--tag', default='', help='suffix of the simulation output files')
ap.add_argument('--la-theta', action='store_true',
                help='add global ancestry from the local-ancestry calls (mean dosage / 2) to the AFR covariates, as in the paper')
ap.add_argument('--shared', choices=('family', 'kinship'), default='family',
                help='simulated resemblance of relatives: family = one constant (variance 0.15) per family; kinship = as in '
                     'the main simulations, covariance 0.25 x GRM within families plus a household constant (0.10) among '
                     'first-degree relatives (GRM > 0.354)')
ap.add_argument('--drop-overlap', action='store_true',
                help='drop participants who are in both group files from the European group')
ap.add_argument('--closure', action='store_true',
                help='exclude every pair within a family (connected component of the related pairs), as in the paper')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
REL_MIN = 0.0442


def log(m):
    print(f'[{time.time() - T0:7.0f} s] {m}', flush=True)


# ---------------------------------------------------------------------------------------------------------- data
def bim(p):
    b = pd.read_csv(p, sep=r'\s+', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], dtype={'SNP': str})
    b['key'] = b.CHR.astype(str) + ':' + b.BP.astype(str)
    return b


def fam(p):
    return pd.read_csv(p, sep=r'\s+', header=None, dtype=str)[1].to_numpy()


I = a.inp
be, ba = bim(f'{I}/EUR/full.bim'), bim(f'{I}/AFRW/full.bim')
be = be[~be.key.duplicated(keep=False)]; ba = ba[~ba.key.duplicated(keep=False)]
m = be.reset_index().merge(ba.reset_index(), on='key', suffixes=('_e', '_a'))
same = (m.A1_e == m.A1_a) & (m.A2_e == m.A2_a)
swap = (m.A1_e == m.A2_a) & (m.A2_e == m.A1_a)
m = m[same | swap].copy(); m['flip'] = swap[same | swap].to_numpy()
m = m.sort_values(['CHR_e', 'BP_e']).reset_index(drop=True)
ide, ida = fam(f'{I}/EUR/full.fam'), fam(f'{I}/AFRW/full.fam')
if a.drop_overlap:
    # participants in both group files (UK Biobank white British draws who meet the African-ancestry definition) are kept
    # in the African-ancestry group only, so that each person appears once
    ide = ide[~np.isin(ide, ida)]
rng0 = np.random.default_rng(12345)
if a.check:
    m = m[m.CHR_e.isin([20, 21, 22])].reset_index(drop=True)
    ide = np.sort(rng0.choice(ide, 1500, replace=False)); ida = np.sort(rng0.choice(ida, 1000, replace=False))
M = len(m)


def covars(cohort, ids):
    ag = pd.read_csv(f'{I}/{cohort}/real/agesex.tsv', sep='\t', dtype={'IID': str}).set_index('IID').drop(columns=['FID'])
    cdir = {('EUR', 'base'): 'EUR', ('EUR', 'origin'): 'EURO', ('AFRW', 'base'): 'AFRWX', ('AFRW', 'origin'): 'AFRWO'}[
        (cohort, a.cov)]
    pc = pd.read_csv(f'{I}/{cdir}/covar_all.txt', sep=r'\s+', dtype={'IID': str}).set_index('IID').drop(columns=['FID'])
    C = pd.concat([ag, pc], axis=1).reindex(ids)
    if cohort == 'AFRW' and a.la_theta:
        lf_ = fam(f'{I}/afrw.la.fam')
        rl_ = pd.Series(np.arange(len(lf_)), index=lf_).reindex(ids).to_numpy()
        assert np.isfinite(rl_).all()
        with open_bed(f'{I}/afrw.la.bed', count_A1=True) as fl_:
            D_ = fl_.read(index=np.s_[rl_.astype(int), :], dtype='float64')
        C['LA_THETA'] = np.nanmean(D_, 1) / 2.0
        del D_
    C.insert(0, 'one', 1.0)
    X = C.to_numpy(float)
    assert np.isfinite(X).all(), f'{cohort}: missing covariates'
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > s[0] * 1e-8], X


Qe, Ce = covars('EUR', ide)
Qa, Ca = covars('AFRW', ida)
ne, na = len(ide), len(ida)
N = ne + na
grp = np.r_[np.zeros(ne, int), np.ones(na, int)]
GN = ['EUR', 'AFR']
log(f'N {N} (EUR {ne}, AFR {na}), M {M}, covariates {a.cov}: EUR {Qe.shape[1]}, AFR {Qa.shape[1]}')

X = np.empty((N, M), dtype=np.float32)
keep = np.ones(M, bool)
with open_bed(f'{I}/EUR/full.bed', count_A1=True) as fe, open_bed(f'{I}/AFRW/full.bed', count_A1=True) as fa:
    re_ = pd.Series(np.arange(len(fam(f'{I}/EUR/full.fam'))), index=fam(f'{I}/EUR/full.fam')).reindex(ide).to_numpy()
    ra_ = pd.Series(np.arange(len(fam(f'{I}/AFRW/full.fam'))), index=fam(f'{I}/AFRW/full.fam')).reindex(ida).to_numpy()
    for s0 in range(0, M, a.chunk):
        s1 = min(M, s0 + a.chunk)
        ge = fe.read(index=np.s_[re_, m['index_e'].to_numpy()[s0:s1]], dtype='float64')
        ga = fa.read(index=np.s_[ra_, m['index_a'].to_numpy()[s0:s1]], dtype='float64')
        ga = np.where(m.flip.to_numpy()[s0:s1][None, :], 2 - ga, ga)
        ok = np.ones(s1 - s0, bool)
        for G, Q in ((ge, Qe), (ga, Qa)):
            mu = np.nanmean(G, 0)
            G[:] = np.where(np.isnan(G), mu[None, :], G) - mu[None, :]
            sd = G.std(0)
            ok &= sd > 0
            G /= np.where(sd > 0, sd, 1.0)[None, :]
            G -= Q @ (Q.T @ G)
        keep[s0:s1] = ok
        X[:ne, s0:s1] = ge; X[ne:, s0:s1] = ga
if not keep.all():
    X = X[:, keep]; m = m[keep].reset_index(drop=True)
M = X.shape[1]
chrom, bp = m.CHR_e.to_numpy(), m.BP_e.to_numpy()
log(f'genotypes: {M} SNPs')

L = None
if not a.no_la:
    lb = bim(f'{I}/afrw.la.bim'); lf = fam(f'{I}/afrw.la.fam')
    rl = pd.Series(np.arange(len(lf)), index=lf).reindex(ida).to_numpy()
    assert np.isfinite(rl).all()
    sel = lb.CHR.isin(np.unique(chrom)).to_numpy() if a.check else np.ones(len(lb), bool)
    with open_bed(f'{I}/afrw.la.bed', count_A1=True) as fl:
        D = fl.read(index=np.s_[rl.astype(int), np.flatnonzero(sel)], dtype='float64')
    mu = np.nanmean(D, 0); D = np.where(np.isnan(D), mu[None, :], D) - mu[None, :]
    sd = D.std(0); okl = sd > 0
    D = D[:, okl] / sd[okl][None, :]
    D -= Qa @ (Qa.T @ D)
    L = D.astype(np.float32)
    lchr, lbp = lb.CHR.to_numpy()[sel][okl], lb.BP.to_numpy()[sel][okl]
    log(f'local ancestry: {L.shape[1]} markers')

# blocks: contiguous in SNP order; LA markers assigned by position
B = a.blocks
bounds = np.linspace(0, M, B + 1).astype(int)
sblock = np.repeat(np.arange(B), np.diff(bounds))
if L is not None:
    snp_key = chrom.astype(np.int64) * 10 ** 10 + bp.astype(np.int64)
    la_key = lchr.astype(np.int64) * 10 ** 10 + lbp.astype(np.int64)
    lblock = sblock[np.clip(np.searchsorted(snp_key, la_key, side='right') - 1, 0, M - 1)]

# related pairs (global row indices, i < i')
def read_pairs(cohort, ids, off):
    p = pd.read_csv(f'{I}/{cohort}/pairs.txt', sep=r'\s+', dtype={'IID1': str, 'IID2': str})
    p = p[p.K_grm >= REL_MIN]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    i, j = pos.reindex(p.IID1).to_numpy(), pos.reindex(p.IID2).to_numpy()
    ok = np.isfinite(i) & np.isfinite(j)
    return np.column_stack([i[ok], j[ok]]).astype(int) + off


pairs = np.vstack([read_pairs('EUR', ide, 0), read_pairs('AFRW', ida, ne)])
pairs = np.sort(pairs, 1)
pairs = np.unique(pairs, axis=0)

# ---------------------------------------------------------------------------------------------------------- kernels
def gram(A, Bm=None):
    Bm = A if Bm is None else Bm
    S = np.zeros((A.shape[0], Bm.shape[0]))
    for c0 in range(0, A.shape[1], 4096):
        S += (A[:, c0:c0 + 4096] @ Bm[:, c0:c0 + 4096].T).astype(np.float64)
    return S


S = gram(X)
log('S = X X\'')
cross = np.argwhere(S[:ne, ne:] / M > (0.2 if a.check else REL_MIN))
log(f'largest similarity across groups {S[:ne, ne:].max() / M:.4f}; 99.99th percentile {np.quantile(S[:ne, ne:], 0.9999) / M:.4f}')
if len(cross):
    pairs = np.unique(np.vstack([pairs, np.column_stack([cross[:, 0], cross[:, 1] + ne])]), axis=0)
if a.closure:
    # families = connected components of the related pairs (within and across groups); all pairs within a family
    parent_ = np.arange(N)

    def find_(x):
        while parent_[x] != x:
            parent_[x] = parent_[parent_[x]]; x = parent_[x]
        return x

    for i_, j_ in pairs:
        ri_, rj_ = find_(i_), find_(j_)
        if ri_ != rj_:
            parent_[ri_] = rj_
    root_ = np.array([find_(i_) for i_ in range(N)])
    fid_ = pd.factorize(root_)[0]
    n_listed = len(pairs)
    allp = [np.array([(u, v) for k_, u in enumerate(ix) for v in ix[k_ + 1:]])
            for ix in (np.flatnonzero(fid_ == f_) for f_ in np.flatnonzero(np.bincount(fid_) > 1))]
    pairs = np.unique(np.sort(np.vstack([pairs] + allp), 1), axis=0)
    log(f'closure: {n_listed} listed pairs -> {len(pairs)} pairs within families')
log(f'related pairs excluded: {len(pairs)} ({len(cross)} across groups)')
pi, pj = pairs[:, 0], pairs[:, 1]
Su = S.copy(); Su[pi, pj] = 0.0; Su[pj, pi] = 0.0
del S
LAu = None
if L is not None:
    LA_ = gram(L)
    LAu = LA_.copy(); ai, aj = pi[pi >= ne] - ne, pj[pi >= ne] - ne
    LAu[ai, aj] = 0.0; LAu[aj, ai] = 0.0
    del LA_
E_, A_ = slice(0, ne), slice(ne, N)
apairs = pi >= ne                                             # pairs within AFR (cross pairs have pi < ne <= pj)
epairs = pj < ne
xpairs = (pi < ne) & (pj >= ne)

# ---------------------------------------------------------------------------------------------------------- phenotypes
def rint(v):
    r = pd.Series(v).rank(method='average').to_numpy()
    return norm.ppf((r - 0.5) / len(v))


def process(Yraw, Wobs, transform=True):
    """Residualize on each group's covariates among the observed, rank-inverse-normal transform, residualize again,
    standardize. Returns Y (0 where missing), W, nu (observed n minus covariate rank) per group and trait, and the
    leverage of each person in the covariate projection of each trait."""
    K = Yraw.shape[1]
    Y = np.zeros((N, K)); nu = np.zeros((2, K)); Hlev = np.zeros((N, K))
    for g, (rows, C) in enumerate(((E_, Ce), (A_, Ca))):
        for k in range(K):
            ok = Wobs[rows, k] > 0
            Cq, s, _ = np.linalg.svd(C[ok], full_matrices=False)
            Cq = Cq[:, s > s[0] * 1e-8]
            v = Yraw[rows, k][ok]
            v = v - Cq @ (Cq.T @ v)
            if transform:
                v = rint(v); v = v - Cq @ (Cq.T @ v)
            v = v / v.std()
            y = np.zeros(rows.stop - rows.start); y[ok] = v
            Y[rows, k] = y
            h = np.zeros(rows.stop - rows.start); h[ok] = (Cq ** 2).sum(1)
            Hlev[rows, k] = h
            nu[g, k] = ok.sum() - Cq.shape[1]
    return Y, Wobs.astype(float), nu, Hlev


# ---------------------------------------------------------------------------------------------------------- moments
def trait_traces(Y, W, H):
    """Exact per-trait quantities on all SNPs over pairs of different unrelated individuals (self pairs left out, as in
    the paper): quadratic forms Q, sums of squared kernel entries T, the covariate term r = sum_i K_ii h_ii of the first
    equation (h = leverage of the trait's covariate projection), traces d and y'y for the residual equation. Tdiag is
    the self-pair part of T, used for the delete-block traces."""
    WE, WA = W[E_], W[A_]
    YE, YA = Y[E_], Y[A_]
    dS = np.diag(Su)
    q = {}
    q['Q_EE'] = (YE * (Su[E_, E_] @ YE)).sum(0) - (YE ** 2).T @ dS[E_]
    q['Q_AA'] = (YA * (Su[A_, A_] @ YA)).sum(0) - (YA ** 2).T @ dS[A_]
    q['Q_EA'] = (YE * (Su[E_, A_] @ YA)).sum(0)
    S2 = Su * Su
    q['Tdiag_EE'] = WE.T @ dS[E_] ** 2; q['Tdiag_AA'] = WA.T @ dS[A_] ** 2
    q['T_EE'] = (WE * (S2[E_, E_] @ WE)).sum(0) - q['Tdiag_EE']
    q['T_AA'] = (WA * (S2[A_, A_] @ WA)).sum(0) - q['Tdiag_AA']
    q['T_EA'] = (WE * (S2[E_, A_] @ WA)).sum(0)
    del S2
    q['d_E'] = WE.T @ dS[E_]; q['d_A'] = WA.T @ dS[A_]
    q['r_E'] = (WE * H[E_]).T @ dS[E_]; q['r_A'] = (WA * H[A_]).T @ dS[A_]
    q['yy_E'] = (YE ** 2).sum(0); q['yy_A'] = (YA ** 2).sum(0)
    if LAu is not None:
        dL = np.diag(LAu)
        q['Q_LL'] = (YA * (LAu @ YA)).sum(0) - (YA ** 2).T @ dL
        q['Tdiag_LL'] = WA.T @ dL ** 2
        q['T_LL'] = (WA * ((LAu * LAu) @ WA)).sum(0) - q['Tdiag_LL']
        q['Tdiag_KL'] = WA.T @ (dS[A_] * dL)
        q['T_KL'] = (WA * ((Su[A_, A_] * LAu) @ WA)).sum(0) - q['Tdiag_KL']
        q['d_L'] = WA.T @ dL
        q['r_L'] = (WA * H[A_]).T @ dL
    return q


def all_traces():
    """Traces over all participants (unrelated pairs), used to scale the delete-block traces to each trait."""
    one = np.ones((N, 1))
    return trait_traces(np.zeros((N, 1)), one, np.zeros((N, 1)))


_STATIC = {}


def block_static():
    """Trait-independent parts of the delete-block quantities (computed once): <S, S_b> and ||S_b||^2 over unrelated
    pairs for each block of pairs, the same for the local-ancestry kernel and the cross term, S_b of the related pairs,
    and each person's squared norm in each block."""
    if _STATIC:
        return _STATIC
    st = {k: np.zeros(B) for k in ('C_EE', 'C_AA', 'C_EA', 'D_EE', 'D_AA', 'D_EA', 'C_LL', 'D_LL', 'CKL_SbL', 'CKL_SLb',
                                   'D_KL')}
    st['sq'] = np.zeros((N, B)); st['sqL'] = np.zeros((na, B))
    st['pb'] = np.zeros((len(pi), B)); st['lpb'] = np.zeros((int(apairs.sum()), B))
    Suf = Su.astype(np.float32)
    LAuf = LAu.astype(np.float32) if LAu is not None else None
    cEE, cAA, cEA, cSbL = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    for c0 in range(0, M, 4096):
        c1 = min(M, c0 + 4096)
        XE, XA = X[E_, c0:c1], X[A_, c0:c1]
        cEE[c0:c1] = np.einsum('ij,ij->j', XE, Suf[E_, E_] @ XE, dtype=np.float64)
        cAA[c0:c1] = np.einsum('ij,ij->j', XA, Suf[A_, A_] @ XA, dtype=np.float64)
        cEA[c0:c1] = np.einsum('ij,ij->j', XE, Suf[E_, A_] @ XA, dtype=np.float64)
        if LAuf is not None:
            cSbL[c0:c1] = np.einsum('ij,ij->j', XA, LAuf @ XA, dtype=np.float64)
    if L is not None:
        cLL = np.einsum('ij,ij->j', L, LAuf @ L, dtype=np.float64)
        cSLb = np.einsum('ij,ij->j', L, Suf[A_, A_] @ L, dtype=np.float64)
    del Suf
    ai, aj = pi[apairs] - ne, pj[apairs] - ne
    for b in range(B):
        sl = slice(bounds[b], bounds[b + 1])
        Xb = np.ascontiguousarray(X[:, sl])
        st['C_EE'][b], st['C_AA'][b], st['C_EA'][b] = cEE[sl].sum(), cAA[sl].sum(), cEA[sl].sum()
        GE = (Xb[E_].T @ Xb[E_]).astype(np.float64); GA = (Xb[A_].T @ Xb[A_]).astype(np.float64)
        pb = np.einsum('ij,ij->i', Xb[pi], Xb[pj], dtype=np.float64)        # S_b of related pairs
        st['pb'][:, b] = pb
        st['D_EE'][b] = (GE ** 2).sum() - 2 * (pb[epairs] ** 2).sum()
        st['D_AA'][b] = (GA ** 2).sum() - 2 * (pb[apairs] ** 2).sum()
        st['D_EA'][b] = (GE * GA).sum() - (pb[xpairs] ** 2).sum()      # ||X_bE X_bA'||^2 = <X_bE'X_bE, X_bA'X_bA>
        st['sq'][:, b] = np.einsum('ij,ij->i', Xb, Xb, dtype=np.float64)
        if L is not None:
            lsel = lblock == b
            Lb = L[:, lsel]
            st['C_LL'][b] = cLL[lsel].sum()
            st['CKL_SbL'][b] = cSbL[sl].sum()
            st['CKL_SLb'][b] = cSLb[lsel].sum()
            GL = (Lb.T @ Lb).astype(np.float64)
            GKL = (Xb[A_].T @ Lb).astype(np.float64)
            lpb = np.einsum('ij,ij->i', Lb[ai], Lb[aj], dtype=np.float64)
            st['lpb'][:, b] = lpb
            st['D_LL'][b] = (GL ** 2).sum() - 2 * (lpb ** 2).sum()
            st['D_KL'][b] = (GKL ** 2).sum() - 2 * (pb[apairs] * lpb).sum()
            st['sqL'][:, b] = np.einsum('ij,ij->i', Lb, Lb, dtype=np.float64)
    _STATIC.update(st)
    log('static delete-block quantities')
    return _STATIC


def block_deletions(Y):
    """Per block b, the change of every quantity when block b is deleted: numerators per trait (exact, from the GWAS
    statistics of each group and the related pairs' S_b) plus the trait-independent parts."""
    st = block_static()
    K = Y.shape[1]
    out = dict(st)
    ME = (X[E_].T @ Y[E_]).astype(np.float64); MA = (X[A_].T @ Y[A_]).astype(np.float64)
    bsum = lambda V: np.add.reduceat(V, bounds[:-1], axis=0)
    out['Q_EE'] = bsum(ME ** 2); out['Q_AA'] = bsum(MA ** 2); out['Q_EA'] = bsum(ME * MA)
    pb = st['pb']
    out['Q_EE'] -= 2 * pb[epairs].T @ (Y[pi[epairs]] * Y[pj[epairs]])
    out['Q_AA'] -= 2 * pb[apairs].T @ (Y[pi[apairs]] * Y[pj[apairs]])
    out['Q_EA'] -= pb[xpairs].T @ (Y[pi[xpairs]] * Y[pj[xpairs]])
    if L is not None:
        YA = Y[A_]
        MLa = (L.T @ YA).astype(np.float64)
        QL = np.zeros((B, K))
        np.add.at(QL, lblock, MLa ** 2)
        ai, aj = pi[apairs] - ne, pj[apairs] - ne
        out['Q_LL'] = QL - 2 * st['lpb'].T @ (YA[ai] * YA[aj])
    return out


# ---------------------------------------------------------------------------------------------------------- estimates
def solve(q, mS, mL, nu, with_la=True):
    """Estimates per trait. Within a group the first equation fits the products of phenotypes of different unrelated
    individuals, E = s T/mS^2 + e r/mS (r from the covariate projection), and the second the sum of squares,
    E(y'y) = s d/mS + e nu. Between groups there is no residual term."""
    K = len(q['Q_EE'])
    r = {}
    sE = np.full(K, np.nan); sA = np.full(K, np.nan); sL = np.full(K, np.nan); sA1 = np.full(K, np.nan)
    sP = np.full(K, np.nan)
    for k in range(K):
        A2 = np.array([[q['T_EE'][k] / mS ** 2, q['r_E'][k] / mS], [q['d_E'][k] / mS, nu[0, k]]])
        sE[k] = np.linalg.solve(A2, [q['Q_EE'][k] / mS, q['yy_E'][k]])[0]
        A2 = np.array([[q['T_AA'][k] / mS ** 2, q['r_A'][k] / mS], [q['d_A'][k] / mS, nu[1, k]]])
        sA1[k] = np.linalg.solve(A2, [q['Q_AA'][k] / mS, q['yy_A'][k]])[0]
        if with_la and 'Q_LL' in q:
            A3 = np.array([[q['T_AA'][k] / mS ** 2, q['T_KL'][k] / (mS * mL), q['r_A'][k] / mS],
                           [q['T_KL'][k] / (mS * mL), q['T_LL'][k] / mL ** 2, q['r_L'][k] / mL],
                           [q['d_A'][k] / mS, q['d_L'][k] / mL, nu[1, k]]])
            sol = np.linalg.solve(A3, [q['Q_AA'][k] / mS, q['Q_LL'][k] / mL, q['yy_A'][k]])
            sA[k], sL[k] = sol[0], sol[1]
        else:
            sA[k] = sA1[k]
        # pooled: one SNP kernel for every pair of different unrelated individuals, no local-ancestry kernel
        T = q['T_EE'][k] + q['T_AA'][k] + 2 * q['T_EA'][k]
        A2 = np.array([[T / mS ** 2, (q['r_E'][k] + q['r_A'][k]) / mS],
                       [(q['d_E'][k] + q['d_A'][k]) / mS, nu[0, k] + nu[1, k]]])
        sP[k] = np.linalg.solve(A2, [(q['Q_EE'][k] + q['Q_AA'][k] + 2 * q['Q_EA'][k]) / mS,
                                     q['yy_E'][k] + q['yy_A'][k]])[0]
    sEA = (q['Q_EA'] / mS) / (q['T_EA'] / mS ** 2)
    nE, nA = q['nE'], q['nA']
    r['h2_EUR'] = sE * q['d_E'] / mS / nE
    r['h2_AFR'] = sA * q['d_A'] / mS / nA
    r['h2_AFR_snponly'] = sA1 * q['d_A'] / mS / nA
    if 'd_L' in q:
        r['h2_LA'] = sL * q['d_L'] / mL / nA
    r['gcov'] = sEA * np.sqrt(q['d_E'] / mS / nE * q['d_A'] / mS / nA)
    den = sE * sA
    r['rg'] = np.where(den > 0, sEA / np.sqrt(np.abs(den)), np.nan)
    den1 = sE * sA1
    r['rg_snponly'] = np.where(den1 > 0, sEA / np.sqrt(np.abs(den1)), np.nan)
    r['h2_pooled'] = sP * (q['d_E'] + q['d_A']) / mS / (nE + nA)
    r['nE'] = nE; r['nA'] = nA
    return pd.DataFrame(r)


def fit(Y, W, nu, H, names):
    q = trait_traces(Y, W, H)
    q['nE'], q['nA'] = W[E_].sum(0), W[A_].sum(0)
    mL = L.shape[1] if L is not None else 1
    full = solve(q, M, mL, nu)
    tall = all_traces()
    dl = block_deletions(Y)
    dS = np.diag(Su)
    dL = np.diag(LAu) if LAu is not None else None
    sq, sqL = dl['sq'], dl['sqL']
    Y2 = Y ** 2
    reps = []
    for b in range(B):
        mb = bounds[b + 1] - bounds[b]
        mlb = int((lblock == b).sum()) if L is not None else 0
        qb = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in q.items()}
        dSb = dS - sq[:, b]                                       # diagonal without block b
        for g, rows in (('EE', E_), ('AA', A_)):
            # numerators over pairs of different individuals: full deletion minus its self-pair part
            qb['Q_' + g] = q['Q_' + g] - (dl['Q_' + g][b] - Y2[rows].T @ sq[rows, b])
        qb['Q_EA'] = q['Q_EA'] - dl['Q_EA'][b]
        for g, rows in (('EE', E_), ('AA', A_)):
            t_full = tall['T_' + g][0] + tall['Tdiag_' + g][0]
            t_del_full = t_full - 2 * dl['C_' + g][b] + dl['D_' + g][b]
            t_del_off = t_del_full - (dSb[rows] ** 2).sum()
            qb['T_' + g] = q['T_' + g] * (t_del_off / tall['T_' + g][0])
        t_all = tall['T_EA'][0]
        qb['T_EA'] = q['T_EA'] * ((t_all - 2 * dl['C_EA'][b] + dl['D_EA'][b]) / t_all)
        qb['d_E'] = W[E_].T @ dSb[E_]; qb['d_A'] = W[A_].T @ dSb[A_]
        qb['r_E'] = (W[E_] * H[E_]).T @ dSb[E_]; qb['r_A'] = (W[A_] * H[A_]).T @ dSb[A_]
        if L is not None:
            dLb = dL - sqL[:, b]
            qb['Q_LL'] = q['Q_LL'] - (dl['Q_LL'][b] - Y2[A_].T @ sqL[:, b])
            t_full = tall['T_LL'][0] + tall['Tdiag_LL'][0]
            t_del_off = t_full - 2 * dl['C_LL'][b] + dl['D_LL'][b] - (dLb ** 2).sum()
            qb['T_LL'] = q['T_LL'] * (t_del_off / tall['T_LL'][0])
            t_full = tall['T_KL'][0] + tall['Tdiag_KL'][0]
            t_del_off = t_full - dl['CKL_SbL'][b] - dl['CKL_SLb'][b] + dl['D_KL'][b] - (dSb[A_] * dLb).sum()
            qb['T_KL'] = q['T_KL'] * (t_del_off / tall['T_KL'][0])
            qb['d_L'] = W[A_].T @ dLb
            qb['r_L'] = (W[A_] * H[A_]).T @ dLb
        reps.append(solve(qb, M - mb, mL - mlb, nu).to_numpy())
    R = np.stack(reps)
    se = np.sqrt((B - 1) / B * np.nansum((R - np.nanmean(R, 0)) ** 2, 0))
    res = full.copy()
    for j, c in enumerate(full.columns):
        res[c + '_se'] = se[:, j]
    res.index = names
    return res, R


def brute(Y, W, nu, H):
    """Jackknife by recomputing every kernel without each block (for --check)."""
    global X, L, Su, LAu, M
    X0, L0, Su0, LAu0, M0 = X, L, Su, LAu, M
    reps = []
    for b in range(B):
        ks = np.ones(M0, bool); ks[bounds[b]:bounds[b + 1]] = False
        X = X0[:, ks]; M = int(ks.sum())
        S = gram(X); S[pi, pj] = 0; S[pj, pi] = 0; Su = S
        if L0 is not None:
            kl = lblock != b
            L = L0[:, kl]
            LA = gram(L); ai, aj = pi[apairs] - ne, pj[apairs] - ne; LA[ai, aj] = 0; LA[aj, ai] = 0; LAu = LA
        q = trait_traces(Y, W, H)
        q['nE'], q['nA'] = W[E_].sum(0), W[A_].sum(0)
        reps.append(solve(q, M, L.shape[1] if L is not None else 1, nu).to_numpy())
    X, L, Su, LAu, M = X0, L0, Su0, LAu0, M0
    return np.stack(reps)


def summarize(name, res):
    res.to_csv(f'{a.out}/{name}.tsv', sep='\t', float_format='%.6g')
    log(f'wrote {name}.tsv')


# ---------------------------------------------------------------------------------------------------------- real traits
pe = pd.read_csv(f'{I}/EUR/real/real_all.tsv', sep='\t', dtype={'IID': str}).set_index('IID').reindex(ide)
pa = pd.read_csv(f'{I}/AFRW/real/real_all.tsv', sep='\t', dtype={'IID': str}).set_index('IID').reindex(ida)
traits = [t for t in pe.columns if t in pa.columns and t != 'FID']
Yraw = np.vstack([pe[traits].to_numpy(float), pa[traits].to_numpy(float)])
Wobs = np.isfinite(Yraw).astype(float)
Yraw = np.where(np.isfinite(Yraw), Yraw, 0.0)
if a.check:
    traits = traits[:6]; Yraw = Yraw[:, :6]; Wobs = Wobs[:, :6]
    Wobs[:, 0] = 1.0                                              # one complete trait: the fast jackknife is exact
Y, W, nu, H = process(Yraw, Wobs)
res, R = fit(Y, W, nu, H, traits)
summarize(f'real_{a.cov}', res)
if a.check:
    Rb = brute(Y, W, nu, H)
    se_b = np.sqrt((B - 1) / B * np.nansum((Rb - np.nanmean(Rb, 0)) ** 2, 0))
    cols = res.columns[:R.shape[2]]
    diff = pd.DataFrame(np.abs(R - Rb).max(0), index=traits, columns=cols)
    ser = pd.DataFrame(se_b, index=traits, columns=[c + '_se_brute' for c in cols])
    out = pd.concat([res[[c + '_se' for c in cols]], ser, diff.add_prefix('maxdiff_')], axis=1)
    out.to_csv(f'{a.out}/check.tsv', sep='\t', float_format='%.6g')
    log('check written')

# ---------------------------------------------------------------------------------------------------------- simulations
if a.sim:
    rng = np.random.default_rng(a.seed)
    # families: connected components of related pairs, for a shared environment
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    root = np.array([find(i) for i in range(N)])
    fam_id = pd.factorize(root)[0]
    in_fam = np.bincount(fam_id)[fam_id] > 1
    fam_list = [np.flatnonzero(fam_id == f_) for f_ in np.flatnonzero(np.bincount(fam_id) > 1)]
    if a.shared == 'kinship':
        # covariance of the shared component within each family = GRM block (PSD), Cholesky factors; households =
        # connected components of first-degree pairs within a family
        chol, hh_id = [], np.full(N, -1)
        nh = 0
        for ix in fam_list:
            Cf = (X[ix].astype(np.float64) @ X[ix].T.astype(np.float64)) / M
            w_, V_ = np.linalg.eigh(Cf)
            chol.append(V_ * np.sqrt(np.clip(w_, 0, None))[None, :])
            first = Cf > 0.354
            par = list(range(len(ix)))
            def fh(x):
                while par[x] != x:
                    par[x] = par[par[x]]; x = par[x]
                return x
            for u_ in range(len(ix)):
                for v_ in range(u_ + 1, len(ix)):
                    if first[u_, v_]:
                        par[fh(u_)] = fh(v_)
            rt = np.array([fh(u_) for u_ in range(len(ix))])
            for r_ in np.unique(rt):
                memb = ix[rt == r_]
                if len(memb) > 1:
                    hh_id[memb] = nh; nh += 1
        in_hh = hh_id >= 0
        shared_var = 0.25 * in_fam + 0.10 * in_hh
        log(f'kinship-scaled shared effects: {len(fam_list)} families, {nh} households ({int(in_hh.sum())} people)')
    else:
        shared_var = 0.15 * in_fam
    SC_ALL = {'rho1': dict(rho=1.0), 'rho05': dict(rho=0.5), 'rho05_la': dict(rho=0.5, la=0.05),
              'rho05_t3': dict(rho=0.5, t3=True), 'rho1_h2': dict(rho=1.0, h2=(0.40, 0.15)),
              'rho05_h2': dict(rho=0.5, h2=(0.40, 0.15))}
    SC = {k: SC_ALL[k] for k in a.scenarios.split(',')}
    cols, meta = [], []
    for name, sp in SC.items():
        nc = int(0.1 * M)
        Bf = np.zeros((M, 2 * a.reps), dtype=np.float32)
        for r in range(a.reps):
            C = rng.choice(M, nc, replace=False)
            z0 = rng.standard_normal(nc)
            for g in range(2):
                Bf[C, 2 * r + g] = np.sqrt(sp['rho']) * z0 + np.sqrt(1 - sp['rho']) * rng.standard_normal(nc)
        GV = (X @ Bf).astype(np.float64)
        del Bf
        H2 = sp.get('h2', (0.25, 0.25))
        for r in range(a.reps):
            y = np.zeros(N)
            for g, rows in enumerate((E_, A_)):
                v = GV[rows, 2 * r + g]
                y[rows] = v / v.std() * np.sqrt(H2[g])
            if sp.get('la') and L is not None:
                u = L @ rng.standard_normal(L.shape[1]).astype(np.float32)
                y[A_] += u / u.std() * np.sqrt(sp['la'])
            if a.shared == 'kinship':
                for ix, Lf in zip(fam_list, chol):
                    y[ix] += np.sqrt(0.25) * (Lf @ rng.standard_normal(Lf.shape[1]))
                hv = rng.standard_normal(max(nh, 1)) * np.sqrt(0.10)
                y += np.where(in_hh, hv[np.clip(hh_id, 0, None)], 0.0)
            else:
                fe = rng.standard_normal(fam_id.max() + 1) * np.sqrt(0.15)
                y += np.where(in_fam, fe[fam_id], 0.0)
            if sp.get('t3'):
                e = rng.standard_t(3, N)
                out_i = rng.random(N) < 0.01
                e[out_i] *= 5
            else:
                e = rng.standard_normal(N)
            for g, rows in enumerate((E_, A_)):
                ee = e[rows]
                y[rows] += ee / ee.std() * np.sqrt(max(1 - H2[g] - shared_var[rows].mean() - (sp.get('la', 0) if g else 0), 0.1))
            cols.append(y); meta.append(dict(scenario=name, rep=r, rho=sp['rho'], h2E=H2[0], h2A=H2[1]))
    Ysim = np.column_stack(cols)
    Wsim = np.ones_like(Ysim)
    for proc in (True, False):
        Yp, Wp, nup, Hp = process(Ysim, Wsim, transform=proc)
        if not proc:
            keepc = [i for i, mt in enumerate(meta) if mt['scenario'] == 'rho05_t3']
            if not keepc:
                continue
            Yp, Wp, nup, Hp = Yp[:, keepc], Wp[:, keepc], nup[:, keepc], Hp[:, keepc]
            mt = [dict(meta[i], scenario='rho05_t3_raw') for i in keepc]
        else:
            mt = meta
        rs, _ = fit(Yp, Wp, nup, Hp, [f"{x['scenario']}_{x['rep']}" for x in mt])
        rs = pd.concat([pd.DataFrame(mt, index=rs.index), rs], axis=1)
        summarize(('sims' if proc else 'sims_t3_raw') + a.tag, rs)

json.dump(dict(N=int(N), M=int(M), blocks=B, cov=a.cov, pairs=int(len(pairs)), cross_pairs=int(len(cross)),
               la_markers=int(L.shape[1]) if L is not None else 0, seconds=round(time.time() - T0)),
          open(f'{a.out}/run_{a.cov}{"_sim" if a.sim else ""}.json', 'w'), indent=1)
log('done')
