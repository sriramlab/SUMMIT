"""Outcome columns for scale-free focal-context x background epistasis passes.

An *analysis* pairs a context (genotype-defined classes, such as a focal SNP's
minor-allele count or a set-score tertile) with one outcome. On every requested
scale the outcome is residualized on the covariates within class x stratum, and
up to three columns are formed on the analysis rows:

    gy   standardized class value x residual   (dosage statistic, FAME-equivalent)
    lo   residual in the low classes            (pattern and selective statistics)
    hi   residual in the high classes
    y    residual                               (additive normalizer, optional)

Scales are ``raw``, ``pooled_rank`` (rank-inverse-normal over the analysis rows,
the FAME convention) and ``within_rank`` (rank-inverse-normal within class x
stratum, exactly null under an affine latent action of the context; see
docs/wiki/Epistasis-scale-free.md). Columns are ordered by kind (all gy, then
lo, then hi) so that the pass can form every G x G kernel from contiguous
batched views.

Individuals are assigned to G random groups. The pass reads them in group order
and keeps per-group scores, which is what the group-level Hoeffding variance
needs; the arithmetic is that of a single ungrouped product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import blom_rank_normal, residualize_within

SCALES = ('raw', 'pooled_rank', 'within_rank')
KINDS = ('gy', 'lo', 'hi', 'y')


@dataclass(frozen=True)
class Context:
    """Genotype-defined classes on the pass rows.

    labels: integer class per pass row, -1 where the context is undefined.
    values: class label -> numeric value for the dosage statistic (e.g. allele count).
    low/high: class labels forming the pattern contrast.
    exclude_chromosomes: background chromosomes removed for this context. A
    focal SNP excludes its own chromosome; a set excludes all its members'
    chromosomes. U-statistics are invalid when context and background share a
    chromosome, because the context is then not independent of the background.
    """
    name: str
    labels: np.ndarray
    values: dict
    low: tuple = (0,)
    high: tuple = (1, 2)
    exclude_chromosomes: tuple = ()


@dataclass(frozen=True)
class Outcome:
    name: str
    values: np.ndarray            # float per pass row; NaN = missing
    positive: bool = False        # also drop values <= 0 (e.g. laboratory concentrations)


@dataclass
class Plan:
    rows: np.ndarray              # sorted source rows of the pass (N)
    groups: np.ndarray            # group per pass row
    n_groups: int
    order: np.ndarray             # pass-row positions in group order
    bounds: np.ndarray            # (G + 1,) group offsets into order
    blocks: list                  # G F-order arrays (n_g x q): outcome columns in group order
    counts: np.ndarray            # (G, q) rows contributing to each column in each group
    columns: list                 # per column: dict(kind, analysis, scale)
    kinds: dict                   # kind -> slice of columns
    keys: dict                    # kind -> list of (analysis, scale) in column order
    analyses: list                # per analysis: context, outcome, n, classes, exclusions

    @property
    def n_columns(self):
        return len(self.columns)


def build_plan(rows, contexts, outcomes, covariates, *, analyses=None, strata=None,
               scales=('pooled_rank', 'within_rank'), pattern_scales=None, additive_scales=(), n_groups=200,
               seed=20261011, min_group_rows=5):
    """Form outcome columns for every requested (context, outcome) analysis.

    rows: strictly increasing source rows (N); every array below is aligned to them.
    covariates: (N, p); rows with any nonfinite covariate are dropped per analysis.
    strata: optional labels (e.g. sex) crossed with the context classes for
    ranking and residualization. analyses: (context, outcome) names, default all
    combinations. pattern_scales: scales that also receive lo/hi columns
    (default all scales). additive_scales: scales that also receive the plain
    residual column y, for theta_y = mean_j (E[x_j y])^2, the relative dosage
    statistic and the global (affine-modifier) prediction of the dosage statistic.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or np.any(np.diff(rows) <= 0):
        raise ValueError('pass rows must be nonempty and strictly increasing')
    n = len(rows)
    scales = tuple(scales)
    pattern_scales = scales if pattern_scales is None else tuple(pattern_scales)
    if not scales or len(set(scales)) != len(scales) or not set(scales) <= set(SCALES):
        raise ValueError(f'scales must be distinct members of {SCALES}')
    additive_scales = tuple(additive_scales)
    if not set(pattern_scales) <= set(scales) or not set(additive_scales) <= set(scales):
        raise ValueError('pattern and additive scales must be among the analysed scales')
    if type(n_groups) is not int or n_groups < 4:
        raise ValueError('at least four individual groups are required')
    contexts = {c.name: c for c in contexts}
    outcomes = {o.name: o for o in outcomes}
    pairs = (list(analyses) if analyses is not None else
             [(c, o) for c in contexts for o in outcomes])
    if not pairs or len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError('analyses must be nonempty and unique')
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim != 2 or len(cov) != n:
        raise ValueError('covariates must be an (N, p) matrix aligned to the pass rows')
    extra = (np.zeros(n, dtype=np.int64) if strata is None else
             np.unique(np.asarray(strata), return_inverse=True)[1].reshape(-1))
    if len(extra) != n:
        raise ValueError('strata must align to the pass rows')
    covok = np.all(np.isfinite(cov), axis=1)

    groups = np.random.default_rng(seed).integers(n_groups, size=n)
    order = np.argsort(groups, kind='stable')
    bounds = np.r_[0, np.cumsum(np.bincount(groups, minlength=n_groups))]

    cols = {kind: [] for kind in KINDS}
    members = {kind: [] for kind in KINDS}
    keys = {kind: [] for kind in KINDS}
    info = []
    for a, (cname, oname) in enumerate(pairs):
        if cname not in contexts or oname not in outcomes:
            raise ValueError(f'unknown context or outcome in analysis {cname}/{oname}')
        c, o = contexts[cname], outcomes[oname]
        labels = np.asarray(c.labels).astype(np.int64, copy=False)
        y = np.asarray(o.values, dtype=np.float64)
        if labels.shape != (n,) or y.shape != (n,):
            raise ValueError(f'{cname}/{oname}: labels and outcome must align to the pass rows')
        use = (labels >= 0) & np.isfinite(y) & covok
        if o.positive:
            use &= y > 0
        idx = np.flatnonzero(use)
        lab = labels[idx]
        present = np.unique(lab)
        if any(int(k) not in c.values for k in present):
            raise ValueError(f'{cname}: every observed class needs a numeric value')
        lo, hi = np.isin(lab, c.low), np.isin(lab, c.high)
        value = np.array([c.values[int(k)] for k in present], dtype=np.float64)[np.searchsorted(present, lab)]
        if value.std() == 0:
            raise ValueError(f'{cname}/{oname}: the context takes one value on the analysis rows')
        g = (value - value.mean()) / value.std()
        stratum = lab * (int(extra.max()) + 1) + extra[idx]
        stacked = []
        for scale in scales:
            if scale == 'raw':
                stacked.append(y[idx])
            elif scale == 'pooled_rank':
                stacked.append(blom_rank_normal(y[idx]))
            else:
                stacked.append(blom_rank_normal(y[idx], stratum))
        residual = residualize_within(np.column_stack(stacked), cov[idx], stratum)
        share = np.array([np.mean(lab == k) for k in present])
        level = np.array([g[lab == k][0] for k in present])
        spread = {scale: np.array([residual[lab == k, s].std() for k in present]) for s, scale in enumerate(scales)}
        info.append(dict(index=a, context=cname, outcome=oname, n=int(len(idx)),
                         class_sd={scale: {int(k): float(v) for k, v in zip(present, spread[scale])} for scale in scales},
                         global_ratio={scale: float((np.sum(share * level * spread[scale])
                                                     / np.sum(share * spread[scale])) ** 2) for scale in scales},
                         classes={int(k): int(np.sum(lab == k)) for k in present},
                         class_values={int(k): float(c.values[int(k)]) for k in present},
                         low=[int(k) for k in c.low], high=[int(k) for k in c.high],
                         n_low=int(lo.sum()), n_high=int(hi.sum()),
                         exclude_chromosomes=[str(x) for x in c.exclude_chromosomes]))
        for s, scale in enumerate(scales):
            r = residual[:, s]
            cols['gy'].append((idx, g * r))
            members['gy'].append(idx)
            keys['gy'].append((a, scale))
            if scale in pattern_scales:
                cols['lo'].append((idx[lo], r[lo]))
                members['lo'].append(idx[lo])
                keys['lo'].append((a, scale))
                cols['hi'].append((idx[hi], r[hi]))
                members['hi'].append(idx[hi])
                keys['hi'].append((a, scale))
            if scale in additive_scales:
                cols['y'].append((idx, r))
                members['y'].append(idx)
                keys['y'].append((a, scale))

    q = sum(len(cols[k]) for k in KINDS)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    ordered = np.zeros((n, q), dtype=np.float64, order='F')
    counts = np.zeros((n_groups, q), dtype=np.int64)
    columns, kinds, j = [], {}, 0
    for kind in KINDS:
        start = j
        for (where, vec), member, (a, scale) in zip(cols[kind], members[kind], keys[kind]):
            ordered[position[where], j] = vec
            counts[:, j] = np.bincount(groups[member], minlength=n_groups)
            columns.append(dict(kind=kind, analysis=a, scale=scale))
            j += 1
        kinds[kind] = slice(start, j)
    short = np.flatnonzero(counts.min(axis=0) < min_group_rows)
    if len(short):
        bad = columns[int(short[0])]
        name = info[bad['analysis']]
        raise ValueError(f"{name['context']}/{name['outcome']} ({bad['kind']}): some of the {n_groups} groups have "
                         f'fewer than {min_group_rows} rows; use fewer groups or larger classes')
    blocks = [np.asfortranarray(ordered[bounds[g]:bounds[g + 1]]) for g in range(n_groups)]
    return Plan(rows=rows, groups=groups, n_groups=n_groups, order=order, bounds=bounds, blocks=blocks,
                counts=counts, columns=columns, kinds=kinds, keys=keys, analyses=info)
