"""Real-genotype simulation benchmark: SNP heritability with related individuals (SUMMIT pair exclusion and family GLS).

Phenotypes are simulated on the full related-inclusive cohort:
    y = sqrt(h2g) * g + u + c + e
  g : standardized-genotype genetic value on the analysis SNPs (dense: all SNPs causal; sparse: 1% causal), Var ~ 1
  u : untagged genetic value shared through realized IBD, u_f ~ N(0, s2u * K_f) within families (K_f = pruned-SNP GRM block,
      families = connected components of GRM relatedness > 0.0442); singletons u_i ~ N(0, s2u)
  c : household environment shared within first-degree components, Var s2c
  e : independent noise, Var 1 - h2g - s2u - s2c

Arms (all with the cohort's PCs as GWAS covariates and the paper's PC-adjusted genome-wide LD scores):
  U      : the paper's unrelated panel (GWAS on panel members only)
  U2     : one member per GRM family (proper unrelated subset)
  F      : everyone, standard SUMMIT (no correction)
  FREL   : everyone, per-SNP related-pair moments removed, P = all pairs within GRM-0.0442 families
  FREL3  : as FREL with P = pairs within GRM-0.0884 families (4th-degree relatives left in)
  GLS    : everyone, family GLS with an estimated working within-family covariance (own implementation)
Per-SNP HE-scale moments follow SUMMIT's h2 convention (z* from OLS beta/SE with cov_rank 0); the SUMMIT estimator
(prepare_h2_reference_axis + fit_h2, chromosome jackknife, the CLI defaults) is called on summary_y = m_j - m_rel_j.
Per-chromosome sums of every per-SNP quantity are saved so all single-component estimators can be recomputed.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bedio import bed_codes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cohort', required=True)
ap.add_argument('--bfile', required=True)
ap.add_argument('--pcs', required=True)
ap.add_argument('--npc', type=int, required=True)
ap.add_argument('--panel-fam', required=True)
ap.add_argument('--rel-dir', required=True)
ap.add_argument('--ldscore', required=True)
ap.add_argument('--summit-src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--nrep', type=int, default=100)
ap.add_argument('--seed', type=int, default=20261023)
ap.add_argument('--block', type=int, default=10000)
ap.add_argument('--stride', type=int, default=1, help='debug: use every k-th SNP genome-wide')
ap.add_argument('--dump-y', action='store_true', help='save simulated phenotypes (N x R) and stop before pass 2')
ap.add_argument('--scenario-set', default='relatives', choices=['relatives', 'structure'])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.summit_src)
T0 = time.time()
def log(msg):
    print(f'[{time.time()-T0:7.0f}s] {msg}', flush=True)

SCEN_SETS = {
    'relatives': [  # name, h2g, s2u, s2c, architecture
        ('A', 0.25, 0.00, 0.00, 'dense'),
        ('B', 0.25, 0.25, 0.00, 'dense'),
        ('C', 0.25, 0.25, 0.10, 'dense'),
        ('D', 0.10, 0.20, 0.10, 'dense'),
        ('E', 0.25, 0.25, 0.10, 'sparse'),
        ('O', 0.25, 0.00, 0.00, 'outlier'),   # dense h2 0.25 plus one SNP explaining V_OUTLIER (random location per rep)
    ],
    'structure': [  # population-structure check: polygenic vs sparse, no family effects
        ('P', 0.50, 0.00, 0.00, 'dense'),
        ('S', 0.50, 0.00, 0.00, 'sparse'),
    ],
}
SCEN = SCEN_SETS[a.scenario_set]
V_OUTLIER = 0.01
N_FINE = 1000                                  # equal-count SNP blocks; unions give SUMMIT's 100/200-block partitions
R = a.nrep * len(SCEN)

# ------------------------------------------------------------------ inputs
fam = pd.read_csv(a.bfile + '.fam', sep=r'\s+', header=None, usecols=[1], names=['IID'])
bim = pd.read_csv(a.bfile + '.bim', sep=r'\s+', header=None, names=['chr', 'snp', 'cm', 'bp', 'a1', 'a2'])
N = len(fam); M_all = len(bim)
snp_idx = np.arange(0, M_all, a.stride)
M = snp_idx.size
bim = bim.iloc[snp_idx].reset_index(drop=True)
chr_of = bim.chr.to_numpy()
chroms = np.unique(chr_of)
pcs = fam.merge(pd.read_csv(a.pcs, sep=' '), on='IID', how='left')
PC = pcs[[f'PC{k}' for k in range(1, a.npc + 1)]].to_numpy(np.float64)
assert np.isfinite(PC).all()
panel = set(pd.read_csv(a.panel_fam, sep=r'\s+', header=None)[1])
in_panel = fam.IID.isin(panel).to_numpy()
fams = pd.read_csv(os.path.join(a.rel_dir, 'families.tsv.gz'), sep='\t')
assert (fams.IID.to_numpy() == fam.IID.to_numpy()).all()
famid = fams['fam_grm_0.0442'].to_numpy()
famid3 = fams['fam_grm_0.0884'].to_numpy()
house = fams['household'].to_numpy()
blocks = pd.read_csv(os.path.join(a.rel_dir, 'family_blocks_grm0.0442.tsv.gz'), sep='\t')
log(f'{a.cohort}: N={N} M={M} R={R} panel={in_panel.sum()}')

# unrelated subsets
U_rows = np.flatnonzero(in_panel)
order = np.lexsort((~in_panel, famid))           # within family prefer panel members
first = np.zeros(N, bool); first[order[0]] = True; first[order[1:]] = famid[order[1:]] != famid[order[:-1]]
U2_rows = np.flatnonzero(first)                 # first[i]: i is the first member of its family in `order`

# within-family unordered pairs (main P) and the 3rd-degree subset
fam_members = {}
for i, f in enumerate(famid):
    fam_members.setdefault(f, []).append(i)
pi, pj = [], []
for f, mem in fam_members.items():
    for x in range(len(mem)):
        for y in range(x + 1, len(mem)):
            pi.append(mem[x]); pj.append(mem[y])
pi = np.array(pi, dtype=np.int64); pj = np.array(pj, dtype=np.int64)
P = pi.size
in3 = famid3[pi] == famid3[pj]
same_house = (house[pi] == house[pj]).astype(np.float64)
log(f'pairs within GRM-0.0442 families: {P} (within GRM-0.0884 families: {in3.sum()}); U={U_rows.size} U2={U2_rows.size}')

def basis(rows):
    V = np.column_stack([np.ones(rows.size), PC[rows]])
    Q, _ = np.linalg.qr(V)
    return Q
Q_F, Q_U, Q_U2 = basis(np.arange(N)), basis(U_rows), basis(U2_rows)
p = a.npc
neff = {'F': N - p - 1, 'U': U_rows.size - p - 1, 'U2': U2_rows.size - p - 1}
nn = {'F': N, 'U': U_rows.size, 'U2': U2_rows.size}

LUT = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float32)
def decode(s, e):
    """analysis SNPs s..e-1 (positions in snp_idx) as N x Mb float32 with NaN"""
    lo, hi = snp_idx[s], snp_idx[e - 1] + 1
    codes = bed_codes(a.bfile, N, lo, hi)[snp_idx[s:e] - lo]
    return LUT[codes].T

def standardize(D):
    """mean-impute, center, scale to unit variance (ddof=1). Returns standardized copy and validity mask."""
    mu = np.nanmean(D, axis=0)
    X = np.where(np.isnan(D), mu[None, :], D) - mu[None, :]
    sd = X.std(axis=0, ddof=1)
    ok = sd > 0
    X[:, ok] /= sd[ok]; X[:, ~ok] = 0.0
    return X.astype(np.float32), ok

def project(Xs, Q, ne):
    X = Xs - (Q @ (Q.T @ Xs)).astype(np.float32)
    ss = (X.astype(np.float64) ** 2).sum(0)
    ok = ss > 1e-8
    X[:, ok] *= np.sqrt(ne / ss[ok]).astype(np.float32)[None, :]; X[:, ~ok] = 0.0
    return X, ok

blk_edges = list(range(0, M, a.block)) + [M]

# ------------------------------------------------------------------ pass 1: genetic values, K~ on pairs and diagonal
rng_cs = np.random.default_rng([a.seed, 7])
Mc = max(1, int(round(0.01 * M)))
causal = np.zeros((M, a.nrep), dtype=bool)
for r in range(a.nrep):
    causal[rng_cs.choice(M, Mc, replace=False), r] = True
G = np.zeros((N, 2 * a.nrep), dtype=np.float64)
rng_o = np.random.default_rng([a.seed, 23])
outlier_snp = np.sort(rng_o.choice(M, a.nrep, replace=False))     # candidates; non-polymorphic ones are replaced below
Xo = np.zeros((N, a.nrep), dtype=np.float64); outlier_ok = np.zeros(a.nrep, bool)
Kp = np.zeros((P, chroms.size)); Kd = np.zeros((N, chroms.size)); Mchr = np.zeros(chroms.size)
cidx = {c: k for k, c in enumerate(chroms)}
for b in range(len(blk_edges) - 1):
    s, e = blk_edges[b], blk_edges[b + 1]
    D = decode(s, e)
    Xs, ok = standardize(D)
    rb = np.random.default_rng([a.seed, 11, b])
    Bd = rb.standard_normal((e - s, a.nrep)).astype(np.float32) / np.sqrt(M)
    Bs = (rb.standard_normal((e - s, a.nrep)) / np.sqrt(Mc)).astype(np.float32) * causal[s:e]
    Bd[~ok] = 0; Bs[~ok] = 0
    G[:, :a.nrep] += Xs @ Bd
    G[:, a.nrep:] += Xs @ Bs
    for r in np.flatnonzero((outlier_snp >= s) & (outlier_snp < e)):
        j = outlier_snp[r] - s
        if ok[j]:
            Xo[:, r] = Xs[:, j]; outlier_ok[r] = True
    Xt, okt = project(Xs, Q_F, neff['F'])
    for c in np.unique(chr_of[s:e]):
        cm = chr_of[s:e] == c
        Xc = Xt[:, cm]
        Kp[:, cidx[c]] += np.einsum('pm,pm->p', Xc[pi].astype(np.float64), Xc[pj].astype(np.float64))
        Kd[:, cidx[c]] += (Xc.astype(np.float64) ** 2).sum(1)
        Mchr[cidx[c]] += int((okt & cm).sum())
    if b % 20 == 0:
        log(f'pass1 block {b+1}/{len(blk_edges)-1}')
Mtot = Mchr.sum()
Ktil_p = Kp.sum(1) / Mtot; Ktil_d = Kd.sum(1) / Mtot
log(f'pass1 done; mean var(g_dense)={G[:, :a.nrep].var(0).mean():.4f} var(g_sparse)={G[:, a.nrep:].var(0).mean():.4f}; '
    f'K~ pairs mean={Ktil_p.mean():.4f}; K~ diag mean={Ktil_d.mean():.4f}')

# ------------------------------------------------------------------ phenotypes
rng_y = np.random.default_rng([a.seed, 13])
blk_by_fam = {}
for i_, j_, k_ in blocks.itertuples(index=False):
    blk_by_fam.setdefault(famid[i_], {})[(i_, j_)] = k_
fam_sqrt = {}
for f, mem in fam_members.items():
    if len(mem) < 2:
        continue
    d = blk_by_fam[f]
    Kf = np.array([[d.get((min(x, y), max(x, y)), 0.0) for y in mem] for x in mem])
    w, v = np.linalg.eigh(Kf)
    fam_sqrt[f] = (np.array(mem), v * np.sqrt(np.clip(w, 0, None))[None, :])
house_ids, house_inv = np.unique(house, return_inverse=True)
Y = np.zeros((N, R)); meta = []
for si, (nm, h2, su, sc, arch) in enumerate(SCEN):
    for r in range(a.nrep):
        col = si * a.nrep + r
        g = G[:, a.nrep + r] if arch == 'sparse' else G[:, r]
        if arch == 'outlier':
            g = g + np.sqrt(V_OUTLIER / h2) * Xo[:, r]           # sqrt(h2) g then adds V_OUTLIER from one SNP
        u = rng_y.standard_normal(N) * np.sqrt(su)
        for f, (mem, Lf) in fam_sqrt.items():
            u[mem] = np.sqrt(su) * (Lf @ rng_y.standard_normal(Lf.shape[1]))
        c = rng_y.standard_normal(house_ids.size)[house_inv] * np.sqrt(sc)
        e_ = rng_y.standard_normal(N) * np.sqrt(1.0 - h2 - su - sc - (V_OUTLIER if arch == 'outlier' else 0.0))
        Y[:, col] = np.sqrt(h2) * g + u + c + e_
        meta.append((nm, r, h2 + (V_OUTLIER if arch == 'outlier' else 0.0), su, sc, arch))
meta = pd.DataFrame(meta, columns=['scenario', 'rep', 'h2g', 's2u', 's2c', 'arch'])
# PC-adjusted truth: the PCs absorb a share R2 of the simulated genetic value in each analysis sample, so the
# heritability on the PC-adjusted scale is h2 (1 - R2) / (1 - h2 R2) (other components are independent of the PCs).
for key, rows_, Q_ in (('F', np.arange(N), Q_F), ('U', U_rows, Q_U), ('U2', U2_rows, Q_U2)):
    gcols = np.array([a.nrep + r if ar == 'sparse' else r for r, ar in zip(meta.rep, meta.arch)])
    Gc = G[rows_][:, gcols]; Gc = Gc - Gc.mean(0)
    r2 = 1.0 - ((Gc - Q_ @ (Q_.T @ Gc)) ** 2).sum(0) / (Gc ** 2).sum(0)
    meta[f'r2pc_{key}'] = r2
    meta[f'h2proj_{key}'] = meta.h2g * (1 - r2) / (1 - meta.h2g * r2)
log('phenotypes simulated; mean R2(g on PCs) F/U/U2 = ' + '/'.join(f"{meta[f'r2pc_{k}'].mean():.4f}" for k in ('F', 'U', 'U2')))
np.save(os.path.join(a.out, 'Y.npy'), Y); np.save(os.path.join(a.out, 'G.npy'), G)
meta.to_csv(os.path.join(a.out, 'phenotypes_meta.tsv'), sep='\t', index=False)
if a.dump_y:
    log('dumped Y and G; stopping'); sys.exit(0)

def proj_pheno(Yr, Q, ne):
    Yp = Yr - Q @ (Q.T @ Yr)
    return (Yp * np.sqrt(ne / (Yp ** 2).sum(0))[None, :]).astype(np.float32)
Yt = {'F': proj_pheno(Y, Q_F, neff['F']), 'U': proj_pheno(Y[U_rows], Q_U, neff['U']), 'U2': proj_pheno(Y[U2_rows], Q_U2, neff['U2'])}

# ------------------------------------------------------------------ family GLS working covariance (estimated from within-family moments)
yF = Yt['F'].astype(np.float64)
Tpairs = yF[pi] * yF[pj]                                           # P x R
Xreg = np.column_stack([Ktil_p, same_house])
coef = np.linalg.lstsq(Xreg, Tpairs, rcond=None)[0]               # 2 x R : (psi, c) per phenotype
multi = [(f, np.array(mem)) for f, mem in fam_members.items() if len(mem) > 1]
pair_index = {(int(x), int(y)): k for k, (x, y) in enumerate(zip(pi, pj))}
YW = yF.copy()                                                     # singletons: W = 1
Wblocks = []                                                       # per family: (mem, W (R x n x n), Kf_chr (C x n x n))
for f, mem in multi:
    n_ = mem.size
    Kf_full = np.zeros((n_, n_)); Kf_chr = np.zeros((chroms.size, n_, n_)); H = np.zeros((n_, n_))
    for x in range(n_):
        Kf_full[x, x] = Ktil_d[mem[x]]; Kf_chr[:, x, x] = Kd[mem[x]]
        for y in range(x + 1, n_):
            k = pair_index[(mem[x], mem[y])]
            Kf_full[x, y] = Kf_full[y, x] = Ktil_p[k]
            Kf_chr[:, x, y] = Kf_chr[:, y, x] = Kp[k]
            H[x, y] = H[y, x] = same_house[k]
    off = Kf_full - np.diag(np.diag(Kf_full))
    V = coef[0][:, None, None] * off[None] + coef[1][:, None, None] * H[None] + np.eye(n_)[None]
    w, v = np.linalg.eigh(V)
    w = np.clip(w, 0.05, None)
    W = np.einsum('rij,rj,rkj->rik', v, 1.0 / w, v)
    YW[mem] = np.einsum('rij,jr->ir', W, yF[mem])
    Wblocks.append((mem, W, Kf_chr))
YW32 = YW.astype(np.float32)
log(f'GLS working covariance: median psi={np.median(coef[0]):.3f} median c={np.median(coef[1]):.3f}')

# ------------------------------------------------------------------ pass 2: per-SNP moments, accumulated per chromosome
NU = lambda key: nn[key]
def zstar2(r, key):
    n = nn[key]; nu_true = n - p - 2
    r2 = r * r
    return (n - 1) * r2 / (r2 + (n - 2) * (1 - r2) / nu_true)
kappa = (N - 1) * (N - p - 2) / ((N - 2) * neff['F'] ** 2)        # d m / d (x~'y~)^2 at r=0, SUMMIT convention
acc = {k: np.zeros((chroms.size, R)) for k in ['m_F', 'mrel', 'mrel3', 'm_U', 'm_U2', 'a_W']}
acc_cnt = {k: np.zeros(chroms.size) for k in ['F', 'U', 'U2']}
lrel_chr = np.zeros(chroms.size)
snp_ok_all = np.zeros(M, bool)
fine_edges = np.floor(np.linspace(0, M, N_FINE + 1)).astype(np.int64); fine_edges[-1] = M
fine_of = np.searchsorted(fine_edges, np.arange(M), side='right') - 1
accf = {k: np.zeros((N_FINE, R)) for k in ['m_F', 'mrel', 'm_U']}
accf_cnt = np.zeros(N_FINE); lrel_fine = np.zeros(N_FINE)
yU, yU2 = Yt['U'], Yt['U2']; yFf = Yt['F']
wp_all = (yFf[pi].astype(np.float64) * yFf[pj].astype(np.float64))  # P x R
for b in range(len(blk_edges) - 1):
    s, e = blk_edges[b], blk_edges[b + 1]
    D = decode(s, e)
    Xs, ok = standardize(D)
    XF, okF = project(Xs, Q_F, neff['F'])
    XsU, okU0 = standardize(D[U_rows]); XU, okU = project(XsU, Q_U, neff['U'])
    XsU2, okU20 = standardize(D[U2_rows]); XU2, okU2 = project(XsU2, Q_U2, neff['U2'])
    good = ok & okF & okU0 & okU & okU20 & okU2
    snp_ok_all[s:e] = good
    rF = (XF.T @ yFf).astype(np.float64) / neff['F']
    rU = (XU.T @ yU).astype(np.float64) / neff['U']
    rU2 = (XU2.T @ yU2).astype(np.float64) / neff['U2']
    aW = (XF.T @ YW32).astype(np.float64) ** 2
    G_p = XF[pi].astype(np.float64) * XF[pj].astype(np.float64)    # P x Mb
    srel = 2.0 * (G_p.T @ wp_all)                                   # ordered pairs
    srel3 = 2.0 * (G_p[in3].T @ wp_all[in3])
    lrel = (Mtot / neff['F'] ** 2) * 2.0 * (G_p.T @ Ktil_p)         # per-SNP relatedness LD score
    for c in np.unique(chr_of[s:e]):
        cm = (chr_of[s:e] == c) & good
        k = cidx[c]
        acc['m_F'][k] += zstar2(rF[cm], 'F').sum(0)
        acc['mrel'][k] += kappa * srel[cm].sum(0)
        acc['mrel3'][k] += kappa * srel3[cm].sum(0)
        acc['m_U'][k] += zstar2(rU[cm], 'U').sum(0)
        acc['m_U2'][k] += zstar2(rU2[cm], 'U2').sum(0)
        acc['a_W'][k] += aW[cm].sum(0)
        acc_cnt['F'][k] += cm.sum()
        lrel_chr[k] += lrel[cm].sum()
    fb = fine_of[s:e]
    zF, zU = zstar2(rF, 'F'), zstar2(rU, 'U')
    for f in np.unique(fb):
        fm = (fb == f) & good
        accf['m_F'][f] += zF[fm].sum(0); accf['m_U'][f] += zU[fm].sum(0)
        accf['mrel'][f] += kappa * srel[fm].sum(0)
        accf_cnt[f] += fm.sum(); lrel_fine[f] += lrel[fm].sum()
    if b % 20 == 0:
        log(f'pass2 block {b+1}/{len(blk_edges)-1}')
log(f'pass2 done; SNPs usable in all arms: {snp_ok_all.sum()} of {M}')
np.savez_compressed(os.path.join(a.out, 'per_chrom_sums.npz'), chroms=chroms, cnt=acc_cnt['F'], lrel_chr=lrel_chr,
                    **acc, Kp=Kp, Kd=Kd, Mchr=Mchr, coef=coef)
np.savez_compressed(os.path.join(a.out, 'per_fine_block_sums.npz'), fine_edges=fine_edges, cnt=accf_cnt, lrel=lrel_fine,
                    outlier_snp=outlier_snp, outlier_ok=outlier_ok, **accf)
meta.to_csv(os.path.join(a.out, 'phenotypes_meta.tsv'), sep='\t', index=False)
np.save(os.path.join(a.out, 'snp_ok.npy'), snp_ok_all)

# ------------------------------------------------------------------ estimation
from summit.inference.trace import Trace
from summit.inference.jackknife import JackknifeSpec, JackknifeDesign
from summit.inference.h2core import (H2MatchedMetadata, compute_h2_structural_unit_stats,
                                     prepare_h2_reference_axis, fit_h2)
trace = Trace(ldscores=a.ldscore)
pos = trace.index.get_indexer(bim.snp.to_numpy())
keep_mask = np.zeros(trace.nsnps, bool)
keep_mask[pos[(pos >= 0) & snp_ok_all]] = True
log(f'Trace SNPs {trace.nsnps}; matched usable {keep_mask.sum()}')
tv = trace.materialize_view()
jk = JackknifeDesign.from_trace_view(tv, JackknifeSpec.parse('chr'))
struct = compute_h2_structural_unit_stats(tv, jk, ld_kind='main')
L_chr = np.array([trace.ldscores[keep_mask & (trace.chr == c), 0].sum() for c in chroms])
M_chr = np.array([(keep_mask & (trace.chr == c)).sum() for c in chroms], dtype=float)
assert np.allclose(M_chr, acc_cnt['F']), (M_chr, acc_cnt['F'])

def summit_fit(y_per_snp_trace, nsamp):
    matched = H2MatchedMetadata(nsnps=int(keep_mask.sum()), nsamp=float(nsamp), n_scale=float(nsamp - 1), cov_rank=0,
                                cov_rank_source='forced0', name='sim', used_summary=None, used_top=None,
                                clip_count=0, clip_threshold=None)
    prep = prepare_h2_reference_axis(tv, matched, jk, keep_mask, summary_y=y_per_snp_trace,
                                     summary_y_info={'n_scale': float(nsamp - 1)}, full_struct=struct)
    fit = fit_h2(prep)
    return float(fit.h2[-1, 0]), float(fit.h2[-1, 1])

def own_ols(ysum_chr, nsamp):
    """SUMMIT single-component normal equations from per-chromosome sums; delete-one-chromosome jackknife."""
    ns = nsamp - 1.0
    def est(ys, Ms, Ls):
        T = ns + ns * ns * Ls / Ms ** 2
        rhs1 = ns * ys / Ms
        return (rhs1 - ns) / (T - ns)
    full = est(ysum_chr.sum(0), M_chr.sum(), L_chr.sum())
    reps = np.array([est(ysum_chr.sum(0) - ysum_chr[k], M_chr.sum() - M_chr[k], L_chr.sum() - L_chr[k]) for k in range(chroms.size)])
    G_ = chroms.size
    se = np.sqrt((G_ - 1) / G_ * ((reps - reps.mean(0)) ** 2).sum(0))
    return full, se

# GLS estimator
def gls_estimates():
    Sa = acc['a_W']                                                 # C x R
    singles = np.setdiff1d(np.arange(N), np.concatenate([m for m, _, _ in Wblocks]) if Wblocks else np.array([], int))
    yW = YW
    # Q_W and u per chromosome (raw sums over SNPs on that chromosome)
    Q_chr = np.zeros((chroms.size, R)); u_single_chr = Kd[singles]     # (ns x C)
    Q_chr += (Kd[singles][:, :, None] * (yW[singles][:, None, :] ** 2)).sum(0)
    u_fam_chr = []                                                   # list of (C x R)
    for mem, W, Kf_chr in Wblocks:
        z = yW[mem]                                                  # n x R  (= W y)
        Q_chr += np.einsum('ir,cij,jr->cr', z, Kf_chr, z)
        u_fam_chr.append(np.einsum('rij,cji->cr', W, Kf_chr))
    def est(keep):
        Mk = M_chr[keep].sum(); Lk = L_chr[keep].sum()
        num = Sa[keep].sum(0) / Mk - Q_chr[keep].sum(0) / Mk
        us = u_single_chr[:, keep].sum(1) / Mk                        # ns
        uf = np.array([u[keep].sum(0) / Mk for u in u_fam_chr]) if u_fam_chr else np.zeros((0, R))  # F x R
        S1 = us.sum() + uf.sum(0); S2 = (us ** 2).sum() + (uf ** 2).sum(0)
        den = (Lk / Mk ** 2) * (S1 ** 2 - S2)
        return num / den, S1
    full, neff_gls = est(np.ones(chroms.size, bool))
    reps = np.array([est(np.arange(chroms.size) != k)[0] for k in range(chroms.size)])
    G_ = chroms.size
    se = np.sqrt((G_ - 1) / G_ * ((reps - reps.mean(0)) ** 2).sum(0))
    return full, se, neff_gls

rows = []
yfull = {}
for key, arr, nsamp in [('U', acc['m_U'], nn['U']), ('U2', acc['m_U2'], nn['U2']), ('F', acc['m_F'], N),
                        ('FREL', acc['m_F'] - acc['mrel'], N), ('FREL3', acc['m_F'] - acc['mrel3'], N)]:
    h_own, se_own = own_ols(arr, nsamp)
    for col in range(R):
        rows.append(dict(arm=key, col=col, h2_own=h_own[col], se_own=se_own[col], nsamp=nsamp))
g_full, g_se, g_neff = gls_estimates()
for col in range(R):
    rows.append(dict(arm='GLS', col=col, h2_own=g_full[col], se_own=g_se[col], nsamp=N, neff_gls=g_neff[col]))
res = pd.DataFrame(rows)

# SUMMIT's own estimator. For a single component with the chromosome jackknife (no delta adjustment, the CLI default) the
# estimator depends on the per-SNP moments only through their per-chromosome sums, so a per-SNP vector carrying each
# chromosome's sum (spread evenly over its SNPs) gives SUMMIT's exact output.
log('own-implementation estimates done; running the SUMMIT estimator')
chr_trace = trace.chr
def y_from_chr_sums(sums):
    y = np.full(trace.nsnps, np.nan)
    for k, c in enumerate(chroms):
        m = keep_mask & (chr_trace == c)
        y[m] = sums[k] / m.sum()
    return y
summit_cols = []
t_s = time.time()
for key, arr, nsamp in [('U', acc['m_U'], nn['U']), ('U2', acc['m_U2'], nn['U2']), ('F', acc['m_F'], N),
                        ('FREL', acc['m_F'] - acc['mrel'], N), ('FREL3', acc['m_F'] - acc['mrel3'], N)]:
    for col in range(R):
        h, se_ = summit_fit(y_from_chr_sums(arr[:, col]), nsamp)
        summit_cols.append(dict(arm=key, col=col, h2_summit=h, se_summit=se_))
log(f'SUMMIT estimator: {len(summit_cols)} fits in {time.time()-t_s:.0f}s')
res = res.merge(pd.DataFrame(summit_cols), on=['arm', 'col'], how='left')
res = res.merge(meta.reset_index().rename(columns={'index': 'col'}), on='col')
res['c_rel_hat'] = np.nan
crel = (acc['mrel'].sum(0) / M_chr.sum())
res.loc[res.arm == 'F', 'c_rel_hat'] = crel[res.loc[res.arm == 'F', 'col'].to_numpy()]
res.to_csv(os.path.join(a.out, 'results.tsv.gz'), sep='\t', index=False)
res['h2'] = res['h2_summit'].fillna(res['h2_own']); res['se'] = res['se_summit'].fillna(res['se_own'])
print('max |SUMMIT - own| point estimate:', float(np.nanmax(np.abs(res.h2_summit - res.h2_own))),
      ' max |SE diff|:', float(np.nanmax(np.abs(res.se_summit - res.se_own))))
summ = res.groupby(['scenario', 'arm']).agg(h2g=('h2g', 'first'), mean=('h2', 'mean'), sd=('h2', 'std'),
                                          mean_se=('se', 'mean'), n=('h2', 'size')).reset_index()
summ['bias'] = summ['mean'] - summ['h2g']; summ['mc_se'] = summ['sd'] / np.sqrt(summ['n'])
res['h2proj'] = np.where(res.arm == 'U', res.h2proj_U, np.where(res.arm == 'U2', res.h2proj_U2, res.h2proj_F))
pj = res.groupby(['scenario', 'arm']).h2proj.mean().rename('h2proj').reset_index()
summ = summ.merge(pj, on=['scenario', 'arm']); summ['bias_proj'] = summ['mean'] - summ['h2proj']
summ.to_csv(os.path.join(a.out, 'summary.tsv'), sep='\t', index=False)
print(summ.to_string())
json.dump({'N': N, 'M': int(M), 'M_used': int(keep_mask.sum()), 'U': int(U_rows.size), 'U2': int(U2_rows.size), 'pairs': int(P),
           'pairs3': int(in3.sum()), 'L_sum': float(L_chr.sum()), 'lrel_sum': float(lrel_chr.sum()),
           'time_s': time.time() - T0}, open(os.path.join(a.out, 'run_info.json'), 'w'), indent=1)
log('done')
