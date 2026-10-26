"""Approximate related-pair parts for the final benchmark (veena, ~/summit_relatives).

The exact estimator subtracts REL_M_j (one value per SNP and trait). The approximations replace it by one number per trait,
S = sum_j V_j REL_M_j over the SNPs shared with the LD scores, spread over SNPs:
  A1 (uniform):           REL_M_j ~= S / sum_j V_j
  A2 (genotype-weighted): REL_M_j ~= S * kappa_j / sum_j V_j kappa_j,
where kappa_j = (1/nu) sum over ordered related pairs x_ij x_i'j is a genotype-only per-SNP column computed once per cohort
(the related-pair part of a constant phenotype). Both keep sum_j V_j REL_M_j, so the genome-wide estimate with one class of
SNPs is unchanged; they change the jackknife replicates and per-class sums. The same is done for REL_M12 (rg).

Steps (run from ~/summit_relatives):
  kappa POP          writes final/POP/kappa.tsv.gz and checks the code path against rel.tsv.gz for one phenotype
  h2 POP NPROC       writes final/POP/ss/frel_A1 and frel_A2 from final/POP/ss/frel
  rg POP NPROC       writes final/POP/rg/ss/frel_A{1,2} and final/POP/rg/m12_A{1,2}
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, 'summit_dev_install_v2')


def kappa(pop):
    from bed_reader import open_bed
    from summit.ldscore.genotype_source import read_fam_sample_ids, resolve_genotype_input
    from summit.ldscore.gw_ldscore import read_cov
    from summit.relatives.moments import _project_normalize, pairs_within_sample, read_relative_pairs
    O = f'final/{pop}'
    gi = resolve_genotype_input(f'panels/{pop}/{pop}_full')
    sample_ids = read_fam_sample_ids(gi.sample_path)
    bim = pd.read_csv(gi.variant_path, sep=r'\s+', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'],
                      dtype={'SNP': str})
    bed = open_bed(gi.genotype_path)
    n_all, M = bed.shape
    ph = pd.read_csv(f'{O}/pheno.txt', sep=r'\s+', dtype={'FID': str, 'IID': str})
    merged = sample_ids.merge(ph, on=['FID', 'IID'], how='left')
    col0 = [c for c in ph.columns if c not in ('FID', 'IID')][0]
    y = pd.to_numeric(merged[col0], errors='coerce').to_numpy()
    cv = pd.read_csv(f'{O}/covar_all.txt', sep=r'\s+', dtype={'FID': str, 'IID': str})
    cm = sample_ids.merge(cv, on=['FID', 'IID'], how='left').drop(columns=['FID', 'IID'])
    idx = np.flatnonzero(np.isfinite(y) & ~cm.isna().any(axis=1).to_numpy())
    assert idx.size == n_all, 'benchmark phenotypes and covariates are complete'
    C, _, keep = read_cov(cov_filename=f'{O}/covar_all.txt', fam_filename=None, std=True, cov_impute_method='ignore',
                          logger=None, sample_idx=idx, ddof=1, sample_ids=sample_ids)
    C = np.asarray(C, dtype=np.float64)
    n_eff = float(idx.size - C.shape[1] - 1)
    rp = read_relative_pairs(f'{O}/pairs.txt', sample_ids, kinship_col='K_grm', kinship_min=0.0442, closure=True)
    a, b = pairs_within_sample(rp, idx, n_all)
    Y, _ = _project_normalize(y[idx][:, None].copy(), C, n_eff)
    wp = Y[a, 0] * Y[b, 0]
    kap = np.zeros(M)
    relm0 = np.zeros(M)
    for s in range(0, M, 5000):
        e = min(M, s + 5000)
        X, _ = _project_normalize(bed.read(index=np.s_[idx, s:e], dtype='float64'), C, n_eff)
        G = X[a] * X[b]
        kap[s:e] = 2.0 / n_eff * G.sum(0)
        relm0[s:e] = 2.0 / n_eff * (G.T @ wp)
    ref = pd.read_csv(f'{O}/rel.tsv.gz', sep='\t', usecols=['SNP', f'REL_M_{col0}'], dtype={'SNP': str})
    err = np.max(np.abs(relm0 - ref[f'REL_M_{col0}'].to_numpy())) / np.max(np.abs(ref[f'REL_M_{col0}'].to_numpy()))
    print(f'{pop}: N={idx.size} q-1={C.shape[1]} pairs={a.size} M={M}; REL_M recomputed vs rel.tsv.gz max rel diff {err:.2e}; '
          f'kappa mean {kap.mean():.5f} sd {kap.std():.5f}', flush=True)
    assert err < 1e-6
    pd.DataFrame({'SNP': bim.SNP.to_numpy(), 'KAPPA': kap}).to_csv(f'{O}/kappa.tsv.gz', sep='\t', index=False,
                                                                   float_format='%.10g')


def _weights(pop):
    """V and kappa on the SNPs of the in-sample LD scores (the SNPs a fit can use)."""
    O = f'final/{pop}'
    ld = pd.read_csv(f'{O}/ld/all.gw.ldscore.gz', sep='\t', usecols=['SNP', 'V'], dtype={'SNP': str})
    kap = pd.read_csv(f'{O}/kappa.tsv.gz', sep='\t', dtype={'SNP': str})
    w = ld.merge(kap, on='SNP', how='inner')
    assert len(w) == len(ld)
    return w.set_index('SNP')


def approx_column(snps, relm, W):
    """Return (A1, A2) columns for SNP ids `snps` with exact values `relm`; SNPs absent from W get 0 (not used)."""
    v = W.V.reindex(snps).to_numpy()
    k = W.KAPPA.reindex(snps).to_numpy()
    ok = np.isfinite(v) & np.isfinite(relm)
    S = float(np.sum(v[ok] * relm[ok]))
    a1 = np.where(ok, S / np.sum(v[ok]), 0.0)
    a2 = np.where(ok, S * k / np.sum(v[ok] * k[ok]), 0.0)
    return a1, a2, S


W = None


def _init(pop):
    global W
    W = _weights(pop)


def _h2_one(args):
    src, dst1, dst2 = args
    ss = pd.read_csv(src, sep='\t', dtype={'SNP': str})
    a1, a2, _ = approx_column(ss.SNP.to_numpy(), ss.REL_M.to_numpy(), W)
    ss.assign(REL_M=a1).to_csv(dst1, sep='\t', index=False, float_format='%.10g')
    ss.assign(REL_M=a2).to_csv(dst2, sep='\t', index=False, float_format='%.10g')


def h2(pop, nproc):
    O = f'final/{pop}/ss'
    for d in ('frel_A1', 'frel_A2'):
        os.makedirs(f'{O}/{d}', exist_ok=True)
    jobs = [(f'{O}/frel/{f}', f'{O}/frel_A1/{f}', f'{O}/frel_A2/{f}') for f in sorted(os.listdir(f'{O}/frel'))]
    with Pool(nproc, initializer=_init, initargs=(pop,)) as p:
        p.map(_h2_one, jobs)
    print(f'{pop}: {len(jobs)} h2 sumstats written', flush=True)


def _rg_one(args):
    O, r = args
    for t in ('a', 'b'):
        _h2_one((f'{O}/ss/frel/{t}{r}.tsv.gz', f'{O}/ss/frel_A1/{t}{r}.tsv.gz', f'{O}/ss/frel_A2/{t}{r}.tsv.gz'))
    m = pd.read_csv(f'{O}/m12/p{r}.tsv', sep='\t', dtype={'SNP': str})
    a1, a2, _ = approx_column(m.SNP.to_numpy(), m.REL_M12.to_numpy(), W)
    m.assign(REL_M12=a1).to_csv(f'{O}/m12_A1/p{r}.tsv', sep='\t', index=False, float_format='%.10g')
    m.assign(REL_M12=a2).to_csv(f'{O}/m12_A2/p{r}.tsv', sep='\t', index=False, float_format='%.10g')


def rg(pop, nproc):
    O = f'final/{pop}/rg'
    for d in ('ss/frel_A1', 'ss/frel_A2', 'm12_A1', 'm12_A2'):
        os.makedirs(f'{O}/{d}', exist_ok=True)
    npairs = len([f for f in os.listdir(f'{O}/m12') if f.endswith('.tsv')])
    with Pool(nproc, initializer=_init, initargs=(pop,)) as p:
        p.map(_rg_one, [(O, r) for r in range(npairs)])
    print(f'{pop}: {npairs} rg pairs written', flush=True)


if __name__ == '__main__':
    step, pop = sys.argv[1], sys.argv[2]
    if step == 'kappa':
        kappa(pop)
    elif step == 'h2':
        h2(pop, int(sys.argv[3]))
    elif step == 'rg':
        rg(pop, int(sys.argv[3]))
