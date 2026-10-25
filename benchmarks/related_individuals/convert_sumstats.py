"""PLINK 2 --glm output -> SUMMIT sumstats for the final benchmark (SNP A1 A2 N BETA SE COV_RANK [REL_M]).

Arms: unrel (GWAS on the unrelated set), all (everyone, no REL_M: related pairs kept), frel (everyone with REL_M
from `summit relatives moments`, matched by SNP ID as `summit relatives append` does). Raw PLINK files are removed
after conversion.
usage: python convert_sumstats.py POP NPC NPROC
"""
import os, sys
from multiprocessing import Pool
import pandas as pd

pop, npc, nproc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
O = f'final/{pop}'
cols = [c for c in pd.read_csv(f'{O}/pheno.txt', sep=' ', nrows=0).columns if c not in ('FID', 'IID')]
for arm in ('unrel', 'all', 'frel'):
    os.makedirs(f'{O}/ss/{arm}', exist_ok=True)
REL = None


def load_rel():
    global REL
    rel = pd.read_csv(f'{O}/rel.tsv.gz', sep='\t', dtype={'SNP': str})
    REL = rel.set_index('SNP')


def convert(args):
    arm, c = args
    src = f'{O}/gwas_{arm}/g.{c}.glm.linear'
    out = f'{O}/ss/{arm}/{c}.tsv.gz'
    if os.path.exists(out) and (arm == 'unrel' or os.path.exists(f'{O}/ss/frel/{c}.tsv.gz')):
        return
    g = pd.read_csv(src, sep='\t', dtype={'ID': str})
    g = g[g.BETA.notna() & g.SE.notna()]
    ss = pd.DataFrame({'SNP': g.ID.to_numpy(), 'A1': g.A1.to_numpy(), 'A2': g.OMITTED.to_numpy(),
                       'N': g.OBS_CT.to_numpy(), 'BETA': g.BETA.to_numpy(), 'SE': g.SE.to_numpy(), 'COV_RANK': npc})
    ss.to_csv(out, sep='\t', index=False, float_format='%.10g')
    if arm == 'all':
        rel_m = REL[f'REL_M_{c}'].reindex(ss.SNP).to_numpy()
        ss.assign(REL_M=rel_m).to_csv(f'{O}/ss/frel/{c}.tsv.gz', sep='\t', index=False, float_format='%.10g')
    os.remove(src)


if __name__ == '__main__':
    load_rel()
    jobs = [(arm, c) for arm in ('unrel', 'all') for c in cols]
    with Pool(nproc) as pool:
        for _ in pool.imap_unordered(convert, jobs, chunksize=4):
            pass
    print(pop, 'converted', len(jobs))
