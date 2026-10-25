"""Bivariate phenotypes for the final benchmark's rg arm (run in ~/summit_relatives on veena).

y_k = sqrt(h2) g_k + u_k + c_k + e_k (k = a, b) with h2 0.25, rg 0.5 (g_b = 0.5 g_a + sqrt(0.75) g'), untagged values
u with covariance 0.25 K_f within families and cross-trait correlation 0.8, household effects (variance 0.10)
shared by both traits, and independent noise. g are the dense genetic values of runs/<pop>_v3 (columns r and r+50).
Writes final/<pop>/rg/pheno.txt (FID IID a<r> b<r>), one phenotype file per trait for --pheno-rg (all individuals and
the unrelated set), and truth.tsv.
usage: python sim_bivariate_final.py POP NPAIRS
"""
import os, sys
import numpy as np, pandas as pd

pop, npairs = sys.argv[1], int(sys.argv[2])
H2, S2U, S2C, RHO_G, RHO_U, RHO_C = 0.25, 0.25, 0.10, 0.5, 0.8, 1.0
O = f'final/{pop}/rg'; os.makedirs(f'{O}/ph_all', exist_ok=True); os.makedirs(f'{O}/ph_unrel', exist_ok=True)
fam = pd.read_csv(f'panels/{pop}/{pop}_full.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)
N = len(fam)
G = np.load(f'runs/{pop}_v3/G.npy', mmap_mode='r')
fams = pd.read_csv(f'rel/{pop}/families.tsv.gz', sep='\t')
famid, house = fams['fam_grm_0.0442'].to_numpy(), fams['household'].to_numpy()
blocks = pd.read_csv(f'rel/{pop}/family_blocks_grm0.0442.tsv.gz', sep='\t')
members = {}
for i, f in enumerate(famid):
    members.setdefault(f, []).append(i)
kb = {(int(i), int(j)): k for i, j, k in blocks.itertuples(index=False)}
roots = []
for f, mem in members.items():
    if len(mem) > 1:
        Kf = np.array([[kb.get((min(x, y), max(x, y)), 0.0) for y in mem] for x in mem])
        w, v = np.linalg.eigh(Kf)
        roots.append((np.array(mem), v * np.sqrt(np.clip(w, 0, None))[None, :]))
hid, hinv = np.unique(house, return_inverse=True)
rng = np.random.default_rng([20261024, 91])


def untagged():
    z = rng.standard_normal(N)
    for mem, L in roots:
        z[mem] = L @ rng.standard_normal(L.shape[1])
    return z


keep = set(pd.read_csv(f'final/{pop}/keep_unrel.txt', sep=' ', header=None, dtype=str)[1])
unrel = fam.IID.isin(keep).to_numpy()
cols = {}
for r in range(npairs):
    g1 = np.asarray(G[:, r]); g2 = RHO_G * g1 + np.sqrt(1 - RHO_G ** 2) * np.asarray(G[:, r + 50])
    za, zb = untagged(), untagged()
    u1 = np.sqrt(S2U) * za; u2 = np.sqrt(S2U) * (RHO_U * za + np.sqrt(1 - RHO_U ** 2) * zb)
    ca = rng.standard_normal(hid.size)[hinv]; cb = rng.standard_normal(hid.size)[hinv]
    c1 = np.sqrt(S2C) * ca; c2 = np.sqrt(S2C) * (RHO_C * ca + np.sqrt(1 - RHO_C ** 2) * cb)
    se = np.sqrt(1 - H2 - S2U - S2C)
    cols[f'a{r}'] = np.sqrt(H2) * g1 + u1 + c1 + se * rng.standard_normal(N)
    cols[f'b{r}'] = np.sqrt(H2) * g2 + u2 + c2 + se * rng.standard_normal(N)
ph = pd.concat([fam, pd.DataFrame(cols)], axis=1)
ph.to_csv(f'{O}/pheno.txt', sep=' ', index=False, float_format='%.10g')
for t in cols:
    one = ph[['FID', 'IID', t]].rename(columns={t: 'y'})
    one.to_csv(f'{O}/ph_all/{t}.txt', sep=' ', index=False, float_format='%.10g')
    one[unrel].to_csv(f'{O}/ph_unrel/{t}.txt', sep=' ', index=False, float_format='%.10g')
pd.DataFrame([dict(pair=r, h2=H2, gcov=H2 * RHO_G, rg=RHO_G) for r in range(npairs)]).to_csv(f'{O}/truth.tsv', sep='\t', index=False)
print(pop, 'wrote', npairs, 'pairs')
