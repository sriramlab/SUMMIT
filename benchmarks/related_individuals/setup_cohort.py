"""Inputs for the final CLI benchmark of one cohort (run in ~/summit_relatives on veena).

Writes final/<pop>/: covar_all.txt and covar_unrel.txt (FID IID PCs), keep_unrel.txt, pheno.txt (FID IID y<col> for
scenarios A-E of runs/<pop>_v3, 100 replicates each), pairs.txt (IID1 IID2 K_grm), meta.tsv.
The unrelated set is a greedy maximal set with no pair above relatedness 0.0442 (repeatedly drop the person with
the most remaining relatives; ties keep paper-panel members, then the lower row), as PLINK's --king-cutoff does.
usage: python setup_cohort.py POP NPC
"""
import json, os, sys
import heapq
import numpy as np, pandas as pd

pop, npc = sys.argv[1], int(sys.argv[2])
O = f'final/{pop}'; os.makedirs(O, exist_ok=True)
fam = pd.read_csv(f'panels/{pop}/{pop}_full.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)
N = len(fam)
pcs = pd.read_csv(f'panels/{pop}/{pop}_full.pc{npc}.txt', sep=' ', dtype={'FID': str, 'IID': str})
cov = fam.merge(pcs, on=['FID', 'IID'], how='left')
assert cov[[f'PC{k}' for k in range(1, npc + 1)]].notna().all().all()
pairs = pd.read_csv(f'rel/{pop}/pairs.tsv.gz', sep='\t', dtype={'IID1': str, 'IID2': str})
rel = pairs[pairs.K_grm > 0.0442]
rel[['IID1', 'IID2', 'K_grm']].to_csv(f'{O}/pairs.txt', sep=' ', index=False)
panel = set(pd.read_csv(f'panels/{pop}/panel.fam', sep=r'\s+', header=None, dtype=str)[1])
pos = {iid: i for i, iid in enumerate(fam.IID)}
adj = [set() for _ in range(N)]
for a, b in zip(rel.IID1, rel.IID2):
    i, j = pos[a], pos[b]
    adj[i].add(j); adj[j].add(i)
in_panel = fam.IID.isin(panel).to_numpy()
alive = np.ones(N, bool)
heap = [(-len(adj[i]), int(in_panel[i]), -i, i) for i in range(N) if adj[i]]
heapq.heapify(heap)
while heap:
    d, pan, _, i = heapq.heappop(heap)
    if not alive[i] or -d != len(adj[i]):
        if alive[i] and adj[i]:
            heapq.heappush(heap, (-len(adj[i]), int(in_panel[i]), -i, i))
        continue
    if not adj[i]:
        continue
    alive[i] = False
    for j in adj[i]:
        adj[j].discard(i)
        if adj[j]:
            heapq.heappush(heap, (-len(adj[j]), int(in_panel[j]), -j, j))
    adj[i] = set()
keep = alive
for a, b in zip(rel.IID1, rel.IID2):
    assert not (keep[pos[a]] and keep[pos[b]])
cov.to_csv(f'{O}/covar_all.txt', sep=' ', index=False, float_format='%.10g')
cov[keep].to_csv(f'{O}/covar_unrel.txt', sep=' ', index=False, float_format='%.10g')
fam[keep].to_csv(f'{O}/keep_unrel.txt', sep=' ', index=False, header=False)
meta = pd.read_csv(f'runs/{pop}_v3/phenotypes_meta.tsv', sep='\t')
cols = meta.index[meta.scenario.isin(list('ABCDE'))].to_numpy()
Y = np.load(f'runs/{pop}_v3/Y.npy', mmap_mode='r')
ph = fam.copy()
for c in cols:
    ph[f'y{c}'] = np.asarray(Y[:, c])
ph.to_csv(f'{O}/pheno.txt', sep=' ', index=False, float_format='%.10g')
meta.loc[cols].assign(col=cols).to_csv(f'{O}/meta.tsv', sep='\t', index=False)
fams = pd.read_csv(f'rel/{pop}/families.tsv.gz', sep='\t', dtype={'IID': str})
info = {'pop': pop, 'N': int(N), 'npc': npc, 'n_unrelated': int(keep.sum()), 'n_relative_pairs_listed': int(len(rel)),
        'one_per_family': int(fams['fam_grm_0.0442'].nunique()), 'n_phenotypes': int(len(cols))}
json.dump(info, open(f'{O}/setup.json', 'w'), indent=1)
print(info)
