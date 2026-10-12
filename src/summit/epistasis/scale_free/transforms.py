"""Rank and covariate transforms applied separately within strata.

Within-stratum rank-inverse-normal values are exactly invariant to every
strictly increasing outcome transformation and to any affine latent action of
the stratifying genotype (docs/wiki/Epistasis-scale-free.md). Ties receive the
average rank, matching ``scipy.stats.rankdata``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def _stratum_codes(strata, n):
    if strata is None:
        return np.zeros(n, dtype=np.int64), 1
    codes, inverse = np.unique(np.asarray(strata), return_inverse=True)
    return inverse.astype(np.int64), len(codes)


def blom_rank_normal(values, strata=None):
    """Blom rank-inverse-normal transform, separately within each stratum.

    All values must be finite. Average ranks for ties; Phi^{-1}((r-3/8)/(n+1/4)).
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError('finite one-dimensional values required for rank transformation')
    code, k = _stratum_codes(strata, len(y))
    order = np.lexsort((y, code))
    ys, cs = y[order], code[order]
    # boundaries of strata and of tied runs within strata
    new_stratum = np.r_[True, cs[1:] != cs[:-1]]
    new_value = new_stratum | np.r_[True, ys[1:] != ys[:-1]]
    run_start = np.flatnonzero(new_value)
    run_end = np.r_[run_start[1:], len(ys)]
    stratum_start = np.flatnonzero(new_stratum)
    stratum_of_run = np.searchsorted(stratum_start, run_start, side='right') - 1
    first = stratum_start[stratum_of_run]
    average_rank = (run_start + run_end - 1) / 2 - first + 1
    counts = np.bincount(cs, minlength=k).astype(np.float64)
    size = counts[cs[run_start]]
    if np.any(size < 2):
        raise ValueError('each rank-transformation stratum needs at least two values')
    run_value = ndtri((average_rank - .375) / (size + .25))
    out = np.empty(len(y))
    out[order] = np.repeat(run_value, run_end - run_start)
    return out


def residualize_within(values, covariates, strata=None, *, rcond=1e-10):
    """Least-squares residuals of ``values`` on [1, covariates] within each stratum.

    ``values`` may be one- or two-dimensional (columns transformed jointly).
    Rank-deficient covariate columns within a stratum are handled by SVD
    truncation; strata need more rows than retained columns.
    """
    y = np.asarray(values, dtype=np.float64)
    one = y.ndim == 1
    y2 = y[:, None] if one else y
    n = len(y2)
    c = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, dtype=np.float64)
    if c.ndim != 2 or len(c) != n or not np.all(np.isfinite(c)) or not np.all(np.isfinite(y2)):
        raise ValueError('finite aligned values and covariates required')
    code, k = _stratum_codes(strata, n)
    out = np.empty_like(y2)
    for s in range(k):
        rows = np.flatnonzero(code == s)
        design = np.column_stack([np.ones(len(rows)), c[rows]])
        u, sv, _ = np.linalg.svd(design, full_matrices=False)
        keep = sv > rcond * sv[0]
        if keep.sum() >= len(rows):
            raise ValueError('a residualization stratum has no residual degrees of freedom')
        basis = u[:, keep]
        block = y2[rows]
        out[rows] = block - basis @ (basis.T @ block)
    return out[:, 0] if one else out
