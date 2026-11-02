"""Helpers for the All of Us cells for direct genetic effects in siblings (direct_cells.py).

Pure numpy/pandas/scipy; nothing person-level leaves the Workbench. The estimators are those of sib_direct.py (a copy of the
validated UK Biobank script); this module only finds the sibships and extended families, checks resources and masks counts.

Sibships (as in the UK Biobank analysis, recomb/scripts/ukb_sibships.py): full siblings are pairs with KING kinship in
(0.177, 0.354] and IBS0 > 0.0012 (KING's first-degree range; parent-offspring pairs share no opposite homozygote apart from
genotyping errors), both members with every covariate. Sibships are the connected components of sibling edges. Extended families
are the connected components of the v1 kit's related pairs (GRM relatedness > 0.0442) over every QC'd member, with the sibling
edges added, so that every sibship lies in one extended family.
"""
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------- inputs
def read_fam(prefix):
    return pd.read_csv(prefix + '.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)


def read_king(path):
    """plink2 --make-king-table output; the columns IID1, IID2, KINSHIP and IBS0 must be there."""
    k = pd.read_csv(path, sep=r'\s+', dtype={'#FID1': str, 'FID1': str, 'IID1': str, 'FID2': str, 'IID2': str})
    k.columns = [c.lstrip('#') for c in k.columns]
    missing = [c for c in ('IID1', 'IID2', 'KINSHIP', 'IBS0') if c not in k.columns]
    assert not missing, f'{path} has no column(s) {missing} (columns: {list(k.columns)})'
    return k[['IID1', 'IID2', 'KINSHIP', 'IBS0']]


def components(n, i, j):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(i)), (np.asarray(i, int), np.asarray(j, int))), shape=(n, n))
    return connected_components(g, directed=False)[1]


IBS0_BINS = [0, 0.0003, 0.0006, 0.0012, 0.0025, 0.005, 0.01, 0.02, 0.05, 1.0]


def sibships(king, pairs, all_ids, comp_ids, kin=(0.177, 0.354), ibs0_min=0.0012, rel_min=0.0442):
    """Sibships and extended families. king: IID1, IID2, KINSHIP, IBS0; pairs: the v1 kit's IID1, IID2, KING, GRM (every KING
    candidate); all_ids: the QC'd members (fam order); comp_ids: members with every covariate.

    Returns (table, report). table: IID, sib, ext, n_sib for members of sibships with at least two people (all with every
    covariate). report: raw counts and checks (the caller masks the counts)."""
    all_ids = np.asarray(all_ids, dtype=object)
    N = len(all_ids)
    pos = pd.Series(np.arange(N), index=pd.Index(all_ids))
    k = king[king.IID1.isin(pos.index) & king.IID2.isin(pos.index)]
    first = k[(k.KINSHIP > kin[0]) & (k.KINSHIP <= kin[1])]
    sib_e = first[first.IBS0 > ibs0_min]
    po_e = first[first.IBS0 <= ibs0_min]
    comp = set(map(str, comp_ids))
    sib_c = sib_e[sib_e.IID1.isin(comp) & sib_e.IID2.isin(comp)]
    lab = components(N, pos[sib_c.IID1].to_numpy(), pos[sib_c.IID2].to_numpy())
    size = np.bincount(lab, minlength=N)[lab]
    insib = size >= 2
    rel = pairs[(pairs.GRM > rel_min) & pairs.IID1.isin(pos.index) & pairs.IID2.isin(pos.index)]
    ei = np.r_[pos[rel.IID1].to_numpy(), pos[sib_e.IID1].to_numpy()]
    ej = np.r_[pos[rel.IID2].to_numpy(), pos[sib_e.IID2].to_numpy()]
    ext = components(N, ei, ej)
    tab = pd.DataFrame({'IID': all_ids[insib], 'sib': lab[insib], 'ext': ext[insib], 'n_sib': size[insib]})
    assert tab.groupby('sib').ext.nunique().max() == 1 if len(tab) else True   # every sibship inside one extended family
    # checks
    key = lambda a, b: tuple(sorted((a, b)))
    gpair = {key(a, b): g for a, b, g in pairs[['IID1', 'IID2', 'GRM']].itertuples(index=False)}
    kpair = {key(a, b): (kk, z) for a, b, kk, z in k[['IID1', 'IID2', 'KINSHIP', 'IBS0']].itertuples(index=False)}
    sib_grm = np.array([gpair.get(key(a, b), np.nan) for a, b in sib_c[['IID1', 'IID2']].itertuples(index=False)])
    sibset = {key(a, b) for a, b in sib_c[['IID1', 'IID2']].itertuples(index=False)}
    nonsib = {'parent_offspring': 0, 'duplicate_or_MZ': 0, 'second_degree_or_less': 0}
    inconsistent = 0
    for _, d in tab.groupby('sib'):
        mem = sorted(d.IID)
        bad = False
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                p = key(mem[x], mem[y])
                if p in sibset:
                    continue
                kk, z = kpair.get(p, (np.nan, np.nan))
                if kk > kin[1]:
                    nonsib['duplicate_or_MZ'] += 1             # biologically consistent (twins); listed apart
                    continue
                bad = True
                if kin[0] < kk <= kin[1] and z <= ibs0_min:
                    nonsib['parent_offspring'] += 1
                else:
                    nonsib['second_degree_or_less'] += 1
        inconsistent += bad
    sizes = tab.groupby('sib').size()
    ext_n = tab.groupby('ext').size()
    hist = np.histogram(first.IBS0.clip(upper=1.0), bins=IBS0_BINS)[0]
    rep = dict(first_degree_pairs=len(first), sibling_pairs=len(sib_e), parent_offspring_pairs=len(po_e),
               sibling_pairs_complete=len(sib_c), sib_members=len(tab), sibships=int(sizes.size),
               sibship_sizes={('5+' if s >= 5 else str(s)): int(v) for s, v in
                              sizes.clip(upper=5).value_counts().sort_index().items()},
               nu=int(len(tab) - sizes.size),
               ext_families=int(ext_n.size), largest_ext_sib_members=int(ext_n.max()) if len(ext_n) else 0,
               S_pairs=int((ext_n.astype(np.int64) ** 2).sum()),
               ibs0_first_degree={f'{IBS0_BINS[i]}-{IBS0_BINS[i + 1]}': int(h) for i, h in enumerate(hist)},
               sibling_edges_grm_le_rel_min=int(np.sum(~(sib_grm > rel_min))),
               sibling_grm_quantiles={q: (float(np.nanquantile(sib_grm, float(q))) if len(sib_grm) else None)
                                      for q in ('0.01', '0.5', '0.99')},
               sibships_with_non_sibling_pairs=int(inconsistent), non_sibling_pairs_in_sibships=nonsib)
    return tab, rep


# ---------------------------------------------------------------------------------------------------------------- resources
def resources(N, S, M, C=8, nblocks=200):
    """Peak memory (GiB) and disk (GiB) of the kernel steps for N siblings, |S*| = S pairs, M SNPs and C classes, and the SNP
    chunk of `blocks` that keeps its S* products near 3 GB. Kernels are float32 on disk (C classes and one weighted sum)."""
    chunk = int(max(128, min(4096, 3e9 / (16 * max(S, 1)))))
    grm = N * N * 8 + 3 * N * 4096 * 8
    blocks = C * N * N * 4 + nblocks * C * S * 4 + 2 * S * chunk * 8 + 4 * N * chunk * 8 + C * S * 8
    sim = N * N * 8 + 2 * N * 4096 * 8 + 6 * N * 300 * 8
    fit = 3 * N * 4096 * 8 + N * N * 8 * 2
    ram = max(grm, blocks, sim, fit) / 2 ** 30
    disk = ((C + 1) * N * N * 4 + nblocks * C * S * 4 + N * M / 4 + N * 300 * 8 * 2) / 2 ** 30
    return dict(ram_gib=round(ram, 2), disk_gib=round(disk, 2), blocks_chunk=chunk)


# ---------------------------------------------------------------------------------------------------------------- disclosure
def mask_count(n):
    """All of Us rule: counts of 1-20 are shown as '<=20'."""
    n = int(n)
    return '<=20' if 1 <= n <= 20 else n


def mask_parts(total, parts):
    """Masked components and total; the total is withheld when any component is masked, so that it cannot be recovered."""
    m = {k: mask_count(v) for k, v in parts.items()}
    return ('withheld' if any(v == '<=20' for v in m.values()) else mask_count(total)), m


def mask_report(rep):
    """The sibship report with every count masked (the GRM quantiles are not counts)."""
    out = {}
    for k, v in rep.items():
        if k == 'sibling_grm_quantiles':
            out[k] = {q: (round(x, 4) if x is not None else None) for q, x in v.items()}
        elif isinstance(v, dict):
            out[k] = {kk: mask_count(vv) for kk, vv in v.items()}
        else:
            out[k] = mask_count(v)
    tot, parts = mask_parts(rep['first_degree_pairs'], {'sib': rep['sibling_pairs'], 'po': rep['parent_offspring_pairs']})
    out['first_degree_pairs'] = tot
    tot, _ = mask_parts(rep['sibships'], rep['sibship_sizes'])
    out['sibships'] = tot
    return out


def summarize(df, est='est', se='se', target='target', t=1.9720):
    """Mean, bias against the target, Monte Carlo s.e., SD, mean s.e., s.e./SD and coverage of t intervals (199 df)."""
    e = df[est].to_numpy(float); s = df[se].to_numpy(float); tg = float(df[target].iloc[0])
    return pd.Series({'n_rep': len(df), 'target': tg, 'mean': e.mean(), 'bias': e.mean() - tg, 'mcse': e.std(ddof=1) / np.sqrt(len(e)),
                      'sd': e.std(ddof=1), 'mean_se': s.mean(), 'se_over_sd': s.mean() / e.std(ddof=1),
                      'coverage': float(np.mean(np.abs(e - tg) <= t * s))})


def write_json(obj, path):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
