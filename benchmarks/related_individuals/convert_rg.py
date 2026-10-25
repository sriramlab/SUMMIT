"""PLINK 2 output -> SUMMIT sumstats for the rg arm (SNP A1 A2 N BETA SE COV_RANK [REL_M]); raw files removed.
usage: python convert_rg.py POP NPC NPAIRS NPROC
"""
import os, sys
from multiprocessing import Pool
import pandas as pd

pop, npc, npairs, nproc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
O = f'final/{pop}/rg'


def convert(args):
    arm, t = args
    out = f'{O}/ss/{arm}/{t}.tsv.gz'
    if os.path.exists(out) and (arm == 'unrel' or os.path.exists(f'{O}/ss/frel/{t}.tsv.gz')):
        return
    src = f'{O}/gwas_{arm}/g.{t}.glm.linear'
    g = pd.read_csv(src, sep='\t', dtype={'ID': str})
    g = g[g.BETA.notna() & g.SE.notna()]
    ss = pd.DataFrame({'SNP': g.ID.to_numpy(), 'A1': g.A1.to_numpy(), 'A2': g.OMITTED.to_numpy(),
                       'N': g.OBS_CT.to_numpy(), 'BETA': g.BETA.to_numpy(), 'SE': g.SE.to_numpy(), 'COV_RANK': npc})
    ss.to_csv(out, sep='\t', index=False, float_format='%.10g')
    if arm == 'all':
        r = int(t[1:])
        rel = pd.read_csv(f'{O}/rel/b{r // 5 * 5}.tsv.gz', sep='\t', usecols=['SNP', f'REL_M_{t}'], dtype={'SNP': str})
        rel_m = rel.set_index('SNP')[f'REL_M_{t}'].reindex(ss.SNP).to_numpy()
        ss.assign(REL_M=rel_m).to_csv(f'{O}/ss/frel/{t}.tsv.gz', sep='\t', index=False, float_format='%.10g')
    os.remove(src)


if __name__ == '__main__':
    jobs = [(arm, f'{x}{r}') for arm in ('unrel', 'all') for r in range(npairs) for x in 'ab']
    with Pool(nproc) as pool:
        list(pool.imap_unordered(convert, jobs))
    print(pop, 'rg sumstats converted')
