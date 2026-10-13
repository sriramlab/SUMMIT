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
    Within a stratum the covariates are centred (the intercept) and scaled to
    unit length; columns that are constant there are dropped. The projection
    uses the eigendecomposition of the small correlation matrix of the
    remaining columns, keeping eigenvalues above ``rcond`` times the largest,
    so exactly collinear columns drop out. This needs one cache-blocked pass
    over the stratum's rows, where a QR or SVD of the tall design would sweep
    it once per column. Strata need more rows than retained columns.
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
        block = y2[rows]
        resid = block - block.mean(axis=0)
        retained = 1
        if c.shape[1]:
            x = c[rows]
            x = x - x.mean(axis=0)
            norm = np.sqrt(np.einsum('ij,ij->j', x, x))
            live = norm > 1e-12 * np.sqrt(len(rows)) * np.maximum(np.abs(c[rows]).max(axis=0), 1.)
            if live.any():
                z = x[:, live] / norm[live]
                eigenvalues, vectors = np.linalg.eigh(z.T @ z)
                keep = eigenvalues > rcond * eigenvalues[-1]
                basis = vectors[:, keep] / np.sqrt(eigenvalues[keep])
                resid -= (z @ basis) @ (basis.T @ (z.T @ block))
                retained += int(keep.sum())
        if retained >= len(rows):
            raise ValueError('a residualization stratum has no residual degrees of freedom')
        out[rows] = resid
    return out[:, 0] if one else out
