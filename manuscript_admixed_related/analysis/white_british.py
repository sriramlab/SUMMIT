"""White British participants with their relatives against the unrelated subset (Fig. 3A; cov-LDSC in Appendix D).

  sample     participants of the with-relatives genotype release that pass UK Biobank sample QC (genetic sex equal to
             reported sex, 22001 = 31; no sex-chromosome aneuploidy, 22019; not a heterozygosity or missingness outlier,
             22027; included in kinship inference, 22021 != -1), belong to the white British grouping (22006 = 1) and have
             not withdrawn; the unrelated subset (--unrelated) must lie inside. Related pairs: the UK Biobank kinship table,
             kinship above 0.0442, both members in the sample. Writes rel.keep, unrel.keep and pairs.tsv.
  genotypes  one fileset of the sample at the SNPs of --snps, and the unrelated subset; PCs by projecting everyone onto a
             PCA fitted in the unrelated subset (--pca PREFIX: PREFIX.eigenvec.allele and PREFIX.afreq).
  fit        GWAS of everyone and of the unrelated subset (covariates: age and sex terms of traits.py and PC1-10), REL_M,
             in-sample LD scores of the unrelated subset (one component and 8 MAF-LD bins; LD scores of unrelated people need
             no related-pair parts), and fits of U (unrelated), K (related pairs kept) and XA (related pairs excluded).
             With --covldsc, also cov-LDSC on both GWAS with the same LD scores.
Writes OUT/wb_h2.tsv (trait, arm, model, n, h2, se on the liability scale for binary traits) and OUT/covldsc.tsv.
usage: python white_british.py sample --qc TABLE --fam FAM --withdrawn FILE --unrelated KEEP --kinship KIN --out DIR
       python white_british.py genotypes --bfiles PATTERN --snps FILE --pca PREFIX --out DIR
       python white_british.py fit --traits DIR --out DIR [--covldsc DIR --ldsc-python PY]
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import ONE_THREAD, gwas, liability_factor, read_fam, read_h2_results, run, run_all


def sample(a):
    ids = read_fam(a.fam).IID.astype(int)
    qc = pd.read_csv(a.qc, sep='\t').set_index('f.eid')
    current = ids.isin(qc.index).to_numpy()
    qc = qc.reindex(ids)
    withdrawn = set(pd.read_csv(a.withdrawn, header=None)[0].astype(int))
    ok = (current & ~ids.isin(withdrawn).to_numpy() & ~(qc['f.22001.0.0'].notna() & (qc['f.22001.0.0'] != qc['f.31.0.0']))
          & (qc['f.22019.0.0'] != 1) & (qc['f.22027.0.0'] != 1) & (qc['f.22021.0.0'] != -1) & (qc['f.22006.0.0'] == 1)).to_numpy()
    rel = ids[ok].astype(str)
    unrel = pd.read_csv(a.unrelated, sep=r'\s+', header=None)[1].astype(str)
    assert unrel.isin(set(rel)).all(), 'the unrelated subset is not inside the sample'
    kin = pd.read_csv(a.kinship, sep=r'\s+', dtype={'ID1': str, 'ID2': str})
    pairs = kin[kin.ID1.isin(set(rel)) & kin.ID2.isin(set(rel)) & (kin.Kinship > 0.0442)]
    os.makedirs(a.out, exist_ok=True)
    for name, x in (('rel', rel), ('unrel', unrel)):
        pd.DataFrame({'FID': x, 'IID': x}).to_csv(f'{a.out}/{name}.keep', sep='\t', index=False, header=False)
    pairs.rename(columns={'ID1': 'IID1', 'ID2': 'IID2'})[['IID1', 'IID2', 'Kinship']].to_csv(f'{a.out}/pairs.tsv', sep='\t', index=False)
    print(f'{len(rel)} participants, {len(unrel)} unrelated, {len(pairs)} related pairs')


def genotypes(a):
    O = a.out
    with open(f'{O}/merge_list.txt', 'w') as fh:
        fh.write(''.join(a.bfiles.format(chr=c) + '\n' for c in range(1, 23)))
    run(f'{a.plink2} --pmerge-list {O}/merge_list.txt bfile --keep {O}/rel.keep --make-pgen --out {O}/merged --threads {a.threads}')
    run(f'{a.plink2} --pfile {O}/merged --extract {a.snps} --make-bed --out {O}/rel --threads {a.threads}')
    run(f'{a.plink2} --bfile {O}/rel --keep {O}/unrel.keep --make-bed --out {O}/unrel --threads {a.threads}')
    hdr = pd.read_csv(f'{a.pca}.eigenvec.allele', sep='\t', nrows=0).columns.tolist()
    a1, pc1 = hdr.index('A1') + 1, hdr.index('PC1') + 1
    run(f'{a.plink2} --pfile {O}/merged --read-freq {a.pca}.afreq --score {a.pca}.eigenvec.allele 2 {a1} header-read '
        f'no-mean-imputation variance-standardize --score-col-nums {pc1}-{pc1 + 9} --out {O}/pcs --threads {a.threads}')
    s = pd.read_csv(f'{O}/pcs.sscore', sep='\t', dtype={'#FID': str, 'IID': str})
    pcs = pd.DataFrame({'FID': s['#FID'], 'IID': s.IID})
    for j in range(1, 11):
        pcs[f'PC{j}'] = s[f'PC{j}_AVG'] / s[f'PC{j}_AVG'].std()
    pcs.to_csv(f'{O}/pcs.txt', sep=' ', index=False, float_format='%.10g')


def fit(a):
    O, P = a.out, a.traits
    traits = [c for c in pd.read_csv(f'{P}/pheno_rel.tsv', sep='\t', nrows=0).columns if c not in ('FID', 'IID')]
    pcs = pd.read_csv(f'{O}/pcs.txt', sep=' ', dtype={'FID': str, 'IID': str}).drop(columns='FID')
    for s in ('rel', 'unrel'):
        ag = pd.read_csv(f'{P}/agesex_{s}.tsv', sep='\t', dtype={'FID': str, 'IID': str})
        ag.merge(pcs, on='IID', how='left').to_csv(f'{O}/covar_{s}.txt', sep=' ', index=False, float_format='%.10g')
        if not os.path.exists(f'{O}/ss_{s}/{traits[-1]}.tsv'):
            gwas(a.plink2, f'{O}/{s}', f'{P}/pheno_{s}.tsv', traits, f'{O}/covar_{s}.txt', 15, f'{O}/ss_{s}', a.threads)
    # LD scores of the unrelated subset and the 8 MAF-LD bins, unless supplied (the paper used existing panel LD scores)
    for given, name in ((a.ld1, 'ld1.gw.ldscore.gz'), (a.ld8, 'ld8.gw.ldscore.gz'), (a.annot8, 'annot8.tsv.gz')):
        if given and not os.path.exists(f'{O}/{name}'):
            os.symlink(os.path.abspath(given), f'{O}/{name}')
    common = f'--covar {O}/covar_unrel.txt --nvecs 1024 --block-size 16384 --num-threads {a.threads}'
    if not os.path.exists(f'{O}/ld1.gw.ldscore.gz'):
        run(f'{a.summit} --geno {O}/unrel.bed {common} --seed 1 --out {O}/ld1')
    if not os.path.exists(f'{O}/annot8.tsv.gz'):
        bim = pd.read_csv(f'{O}/unrel.bim', sep=r'\s+', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], dtype={'SNP': str})
        run(f'{a.plink2} --bfile {O}/unrel --freq --out {O}/unrel --threads {a.threads}')
        af = pd.read_csv(f'{O}/unrel.afreq', sep='\t', dtype={'ID': str}).set_index('ID').ALT_FREQS.loc[bim.SNP].to_numpy()
        maf = np.minimum(af, 1 - af)
        ld = pd.read_csv(f'{O}/ld1.gw.ldscore.gz', sep='\t', dtype={'SNP': str}).set_index('SNP').L2_0.loc[bim.SNP].to_numpy()
        annot = bim[['CHR', 'BP', 'SNP', 'CM']].copy()
        for cls, sel in (('A', maf < 0.1), ('B', maf >= 0.1)):
            q = pd.qcut(pd.Series(ld[sel]).rank(method='first'), 4, labels=False).to_numpy() + 1
            for b in range(1, 5):
                annot[f'maf{cls}_ld{b}'] = 0
                annot.loc[np.flatnonzero(sel)[q == b], f'maf{cls}_ld{b}'] = 1
        annot.to_csv(f'{O}/annot8.tsv.gz', sep='\t', index=False, compression='gzip')
    A8 = f'--annot {O}/annot8.tsv.gz'
    if not os.path.exists(f'{O}/ld8.gw.ldscore.gz'):
        run(f'{a.summit} --geno {O}/unrel.bed {A8} {common} --seed 2 --out {O}/ld8')
    if not os.path.exists(f'{O}/rel.rel_m.tsv'):
        run(f'{a.summit} relatives moments --geno {O}/rel.bed --pheno {P}/pheno_rel.tsv --covar {O}/covar_rel.txt {A8} '
            f'--pairs {O}/pairs.tsv --kinship-col Kinship --kinship-min 0.0442 --out {O}/rel', log=f'{O}/rel.out')
    for tag, annot in (('xa1', ''), ('xa8', A8)):
        os.makedirs(f'{O}/ss_{tag}', exist_ok=True)
        run_all([f'{a.summit} relatives append --sumstats {O}/ss_rel/{t}.tsv --rel {O}/rel.rel_m.tsv --pheno {t} {annot} '
                 f'--out {O}/ss_{tag}/{t}.tsv > /dev/null' for t in traits], a.threads)
    ARMS = {'U': ('ss_unrel', 'ss_unrel'), 'K': ('ss_rel', 'ss_rel'), 'XA': ('ss_xa1', 'ss_xa8')}
    cmds = []
    for arm, ss in ARMS.items():
        for k, (model, ld, annot) in enumerate((('single', 'ld1', ''), ('8bin', 'ld8', A8))):
            pre = f'{O}/fit_{arm}.{model}'
            if not os.path.exists(pre + '.results.tsv'):
                cmds.append(f'{ONE_THREAD} {a.summit} --h2 {O}/{ss[k]} --ldscores {O}/{ld}.gw.ldscore.gz {annot} --njack 200 '
                            f'--max-chisq auto --num-threads 1 --out {pre} > {pre}.out 2>&1')
    run_all(cmds, a.threads)
    prev = pd.read_csv(f'{P}/prevalence.tsv', sep='\t').set_index(['sample', 'trait']).prevalence
    rows = []
    for arm in ARMS:
        s = 'unrel' if arm == 'U' else 'rel'
        for model in ('single', '8bin'):
            for x in read_h2_results(f'{O}/fit_{arm}.{model}.results.tsv').itertuples():
                f = liability_factor(prev[(s, x.phen)]) if (s, x.phen) in prev.index else 1.0
                rows.append(dict(trait=x.phen, arm=arm, model=model, h2=x.h2 * f, se=x.se * f))
    pd.DataFrame(rows).to_csv(f'{O}/wb_h2.tsv', sep='\t', index=False, float_format='%.6g')
    if a.covldsc:
        covldsc(a, traits, prev)


def covldsc(a, traits, prev):
    """cov-LDSC with a free intercept on both GWAS, with SUMMIT's LD scores converted to the LDSC format."""
    O = a.out
    D = f'{O}/covldsc'
    os.makedirs(D, exist_ok=True)
    for model, src in (('single', 'ld1'), ('8bin', 'ld8')):
        L = pd.read_csv(f'{O}/{src}.gw.ldscore.gz', sep='\t')
        cols = [c for c in L.columns if c not in ('CHR', 'SNP', 'BP', 'V') and not c.startswith('LD_REL')]
        L[['CHR', 'SNP', 'BP'] + cols].to_csv(f'{D}/{model}.l2.ldscore.gz', sep='\t', index=False, compression='gzip')
        for suffix in ('.l2.M', '.l2.M_5_50'):
            open(f'{D}/{model}{suffix}', 'w').write('\t'.join(open(f'{O}/{src}.gw.M').read().split()) + '\n')
    cmds = []
    for s in ('rel', 'unrel'):
        for t in traits:
            ss = pd.read_csv(f'{O}/ss_{s}/{t}.tsv', sep='\t', dtype={'SNP': str})
            pd.DataFrame({'SNP': ss.SNP, 'A1': ss.A1, 'A2': ss.A2, 'N': ss.N, 'Z': ss.BETA / ss.SE}).to_csv(
                f'{D}/{s}.{t}.sumstats.gz', sep='\t', index=False, float_format='%.8g', compression='gzip')
            for model in ('single', '8bin'):
                cmds.append(f'{ONE_THREAD} {a.ldsc_python} {a.covldsc}/ldsc.py --h2 {D}/{s}.{t}.sumstats.gz --ref-ld {D}/{model} '
                            f'--w-ld {D}/single --not-M-5-50 --out {D}/{model}.{s}.{t} > /dev/null 2>&1')
    run_all(cmds, a.threads)
    rows = []
    for s in ('rel', 'unrel'):
        for t in traits:
            for model in ('single', '8bin'):
                text = open(f'{D}/{model}.{s}.{t}.log').read()
                h = text.split('Total Observed scale h2: ')[-1].split()
                ic = text.split('Intercept: ')[-1].split()
                f = liability_factor(prev[(s, t)]) if (s, t) in prev.index else 1.0
                rows.append(dict(trait=t, sample=s, model=model, h2=float(h[0]) * f, se=float(h[1].strip('()')) * f,
                                 intercept=float(ic[0])))
    pd.DataFrame(rows).to_csv(f'{O}/covldsc.tsv', sep='\t', index=False, float_format='%.6g')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['sample', 'genotypes', 'fit'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--qc'); ap.add_argument('--fam'); ap.add_argument('--withdrawn'); ap.add_argument('--unrelated')
    ap.add_argument('--kinship'); ap.add_argument('--bfiles', help='per-chromosome PLINK prefixes with {chr}')
    ap.add_argument('--snps'); ap.add_argument('--pca'); ap.add_argument('--traits')
    ap.add_argument('--covldsc'); ap.add_argument('--ldsc-python')
    ap.add_argument('--ld1', help='fit: single-component LD-score file to use instead of computing one')
    ap.add_argument('--ld8', help='fit: LD-score file of the 8 MAF-LD bins to use instead of computing one')
    ap.add_argument('--annot8', help='fit: annotation of the 8 MAF-LD bins matching --ld8')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--summit', default='summit')
    ap.add_argument('--plink2', default='plink2')
    a = ap.parse_args()
    {'sample': sample, 'genotypes': genotypes, 'fit': fit}[a.step](a)


if __name__ == '__main__':
    main()
