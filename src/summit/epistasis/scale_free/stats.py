"""Scale-free statistics, Hoeffding and SNP-block variances, from pass kernels.

For a statistic m = (c1, c2) the between-group U-statistic is

    theta_m = mean_{g != h} T_m[g, h],
    T_m[g, h] = sum_j w_j S~_j(c1, g) S~_j(c2, h) / W,

with S~_j(c, g) the class-normalized score sum_{i in g, c} x_ij y_i / n_{c,g}.
It estimates the background mean of E[x_j y | c1] E[x_j y | c2]. Group-level
Hoeffding variance treats the G random individual groups as i.i.d. units. The
SNP-block delete-one jackknife captures LD and SNP-level heterogeneity. Claims
use the larger of the two standard errors (max-SE rule). The Hoeffding
estimator is heavy-tailed for small G; G = 200 is the qualified default.

Statistics (lo = low classes, hi = high classes):
    dosage     theta(gy, gy) = mean_j (E[g x_j y])^2, FAME-equivalent; null 0 (one-sided)
    rho        theta(lo, hi) / sqrt(theta(lo, lo) theta(hi, hi)); pattern; null 1
    log_amp    log theta(hi, hi) / theta(lo, lo); genetic-share ratio; null 0
    delta2     theta(lo, lo) + theta(hi, hi) - 2 theta(lo, hi); null 0 (one-sided)
    sel_lam    log lambda_A - log lambda_rest, lambda = theta(lo, hi) / theta(lo, lo); null 0
    sel_amp    log alpha_A - log alpha_rest, alpha = theta(hi, hi) / theta(lo, lo); null 0
    dosage_diff  theta_A(gy, gy) - theta_rest(gy, gy); null 0
On the within-class rank scale every null holds exactly under an affine latent
action of the context. On other scales rho and sel_* are first-order
scale-free (proportional background), while the others are scale-dependent.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

NULLS = dict(dosage=0., dosage_diff=0., rho=1., log_amp=0., delta2=0., sel_lam=0., sel_amp=0.)
ONE_SIDED = ('dosage', 'delta2')


def hoeffding(kernels):
    """Mean of off-diagonal entries of symmetric G x G kernels and its covariance.

    kernels: (m, G, G) symmetric with zero diagonal. Uses unbiased estimates of
    zeta_1 and zeta_2. theta theta' is estimated from four distinct groups by
    inclusion-exclusion.
    """
    t = np.asarray(kernels, dtype=np.float64)
    g = t.shape[1]
    if g < 4:
        raise ValueError('Hoeffding variance needs at least four groups')
    flat = t.reshape(len(t), -1)
    theta = flat.sum(axis=1) / (g * (g - 1))
    e2 = flat @ flat.T / (g * (g - 1))
    rows = t.sum(axis=2)
    same = flat @ flat.T
    e1 = (rows @ rows.T - same) / (g * (g - 1) * (g - 2))
    total = flat.sum(axis=1)
    tt = (np.outer(total, total) - 4 * rows @ rows.T + 2 * same) / (g * (g - 1) * (g - 2) * (g - 3))
    cov = 4 * (g - 2) / (g * (g - 1)) * (e1 - tt) + 2 / (g * (g - 1)) * (e2 - tt)
    return theta, cov


def _kernel(raw, weight):
    t = raw / weight
    t = (t + t.T) / 2
    np.fill_diagonal(t, 0.)
    return t


def _gradient(function, theta):
    base = function(theta)
    grads = {k: np.zeros(len(theta)) for k in base}
    for i in range(len(theta)):
        step = 1e-6 * max(abs(theta[i]), 1e-12)
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        hi, lo = function(up), function(down)
        for k in base:
            grads[k][i] = (hi[k] - lo[k]) / (2 * step)
    return base, grads


def _log_ratio(a, b):
    return np.log(a / b) if a > 0 and b > 0 else np.nan


def dosage_statistics(theta, annotations, weights):
    """theta: (1 + K,) for all, A_1..A_K; weights: matching background weights."""
    out = dict(dosage=theta[0])
    for k, name in enumerate(annotations, start=1):
        rest = (weights[0] * theta[0] - weights[k] * theta[k]) / (weights[0] - weights[k])
        out[f'dosage[{name}]'] = theta[k]
        out[f'dosage_diff[{name}]'] = theta[k] - rest
    return out


def pattern_statistics(theta, annotations, weights):
    """theta: (3 (1 + K),) ordered (ll, hh, lh) for all, A_1..A_K."""
    ll, hh, lh = theta[0], theta[1], theta[2]
    out = dict(rho=lh / np.sqrt(ll * hh) if ll > 0 and hh > 0 else np.nan,
               log_amp=_log_ratio(hh, ll), delta2=ll + hh - 2 * lh)
    for k, name in enumerate(annotations, start=1):
        a = theta[3 * k:3 * k + 3]
        rest = (weights[0] * theta[:3] - weights[k] * a) / (weights[0] - weights[k])
        out[f'sel_lam[{name}]'] = _log_ratio(a[2], a[0]) - _log_ratio(rest[2], rest[0])
        out[f'sel_amp[{name}]'] = _log_ratio(a[1], a[0]) - _log_ratio(rest[1], rest[0])
    return out


def _family(raw, between, weights, block_weights, function):
    """Estimates with Hoeffding (delta method) and SNP-block jackknife SEs.

    raw: (m, G, G) unnormalized kernels; between: (m, n_blocks) between-group
    sums; weights: (m,) background weights; block_weights: (m, n_blocks).
    """
    kernels = np.stack([_kernel(r, w) for r, w in zip(raw, weights)])
    theta, cov = hoeffding(kernels)
    est, grads = _gradient(function, theta)
    g = raw.shape[1]
    pairs = g * (g - 1)
    total = between.sum(axis=1)
    used = block_weights.sum(axis=0) > 0
    drops = []
    for b in np.flatnonzero(used):
        w = weights - block_weights[:, b]
        drops.append(function((total - between[:, b]) / (w * pairs)))
    nb = len(drops)
    result = {}
    for key, value in est.items():
        var = float(grads[key] @ cov @ grads[key])
        se_u = np.sqrt(var) if np.isfinite(var) and var > 0 else np.nan
        values = np.array([d[key] for d in drops], dtype=np.float64)
        finite = np.isfinite(values)
        se_s = (np.sqrt((nb - 1) / nb * np.sum((values - values.mean()) ** 2))
                if np.isfinite(value) and finite.all() and nb > 1 else np.nan)
        result[key] = (float(value), float(se_u), float(se_s))
    return result


def _null(name):
    return NULLS[name.split('[')[0]]


def _record(base, statistic, value, se_u, se_s):
    se = np.nanmax([se_u, se_s]) if np.isfinite([se_u, se_s]).any() else np.nan
    null = _null(statistic)
    z = (value - null) / se if np.isfinite(se) and se > 0 else np.nan
    one_sided = statistic.split('[')[0] in ONE_SIDED
    p = (norm.sf(z) if one_sided else 2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    zu = (value - null) / se_u if np.isfinite(se_u) and se_u > 0 else np.nan
    zs = (value - null) / se_s if np.isfinite(se_s) and se_s > 0 else np.nan
    return dict(base, statistic=statistic, estimate=value, null=null, se_hoeffding=se_u, se_jackknife=se_s,
                se=se, z=z, z_hoeffding=zu, z_jackknife=zs, p=p, sided='upper' if one_sided else 'two')


def summarize(plan, accumulation):
    """One record per analysis, scale and statistic.

    Analyses with an empty background (or an empty annotation) are omitted.
    """
    names = accumulation.names[1:]
    bw = accumulation.block_weight
    records = []

    def weights_for(pair, p):
        keep = accumulation.keep[pair][p]
        blocks = bw * keep
        return blocks.sum(axis=1), blocks

    def base(a, scale):
        info = plan.analyses[a]
        return dict(analysis=a, context=info['context'], outcome=info['outcome'], scale=scale, n=info['n'],
                    n_low=info['n_low'], n_high=info['n_high'])

    if 'gy' in accumulation.kernels:
        for p, (a, scale) in enumerate(plan.keys['gy']):
            w, blocks = weights_for('gy', p)
            if np.any(w <= 0):
                continue
            stats = _family(accumulation.kernels['gy'][:, p], accumulation.between['gy'][:, p], w, blocks,
                            lambda t, w=w: dosage_statistics(t, names, w))
            records += [_record(base(a, scale), k, *v) for k, v in stats.items()]
    if 'll' in accumulation.kernels:
        for p, (a, scale) in enumerate(plan.keys['lo']):
            w, blocks = weights_for('ll', p)
            if np.any(w <= 0):
                continue
            raw = np.stack([accumulation.kernels[k][i, p] for i in range(len(accumulation.names))
                            for k in ('ll', 'hh', 'lh')])
            between = np.stack([accumulation.between[k][i, p] for i in range(len(accumulation.names))
                                for k in ('ll', 'hh', 'lh')])
            ww, bb = np.repeat(w, 3), np.repeat(blocks, 3, axis=0)
            stats = _family(raw, between, ww, bb, lambda t, w=w: pattern_statistics(t, names, w))
            records += [_record(base(a, scale), k, *v) for k, v in stats.items()]
    return records
