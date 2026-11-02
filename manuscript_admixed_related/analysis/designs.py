"""Simulation settings shared by simulate.py, compare_methods.py and summarize.py.

Phenotype: y = sqrt(h2) g + u + c + e (+ ancestry-linked environment terms in the 'ancestry' design).
  g  genetic value of the analysed SNPs, effects on a random fraction p of SNPs with equal variance (GCTA) or variance
     proportional to [MAF(1 - MAF)]^0.75 / LD score (LDAK), scaled so that the squared effects sum to one
  u  untagged effects shared by relatives: covariance s2u times the GRM within families (relatedness > 0.0442)
  c  household environment, variance s2c, shared within connected components of first-degree pairs
  e  independent noise; the variances add to one
Settings are numbered in the order of `settings()`; `python designs.py` prints them.
"""

NREP = 100


def _setting(design, **kw):
    s = dict(design=design, kind='h2', h2=0.25, p=1.0, s2u=0.0, s2c=0.0, arch='gcta', mafbin='all', rg=None, ru=None,
             rc=None, re=None, s2g=0.0, s2q=0.0)
    s.update(kw)
    return s


def settings():
    out = []
    for h2 in (0.10, 0.25, 0.40):                      # main grid, without and with effects shared by relatives
        for p in (1.0, 0.1, 0.01):
            for shared in (0, 1):
                out.append(_setting('grid', h2=h2, p=p, s2u=0.25 * shared, s2c=0.10 * shared))
    for arch in ('gcta', 'ldak'):                      # causal SNPs in one MAF class (split at 0.1)
        for mafbin in ('low', 'high'):
            for p in (1.0, 0.01):
                out.append(_setting('mafld', h2=0.40, p=p, s2u=0.25, s2c=0.10, arch=arch, mafbin=mafbin))
    for rg in (0.0, 0.5):                              # genetic correlation; relatives share correlated u and c
        out.append(_setting('rg', kind='rg', s2u=0.25, s2c=0.10, rg=rg, ru=0.8, rc=0.8, re=0.2))
    for s2g, s2q in ((0.05, 0.0), (0.0, 0.05), (0.05, 0.05)):   # environments that follow global ancestry nonlinearly
        out.append(_setting('ancestry', s2g=s2g, s2q=s2q))
    for k, s in enumerate(out):
        s['setting'] = k
    return out


def label(s):
    if s['kind'] == 'rg':
        return f"rg={s['rg']}"
    text = f"h2={s['h2']} p={s['p']}"
    if s['design'] == 'mafld':
        text += f" {s['arch']} {s['mafbin']}-MAF"
    if s['s2u'] or s['s2c']:
        text += ' shared'
    if s['s2g']:
        text += f" unadmixed-group environment {s['s2g']}"
    if s['s2q']:
        text += f" quadratic-ancestry environment {s['s2q']}"
    return text


# Settings of Fig. 2E (wider African-ancestry group and its admixed members)
WIDER_GROUP = ('h2=0.25 p=1.0', 'h2=0.25 p=1.0 shared', 'h2=0.25 p=0.01', 'h2=0.4 p=1.0', 'h2=0.4 p=1.0 shared',
               'h2=0.4 p=1.0 ldak low-MAF shared')

if __name__ == '__main__':
    for s in settings():
        print(s['setting'], s['design'], label(s))
