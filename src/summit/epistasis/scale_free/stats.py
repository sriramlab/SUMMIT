"""Scale-free statistics, Hoeffding and SNP-block variances, from pass kernels.

For a statistic m = (c1, c2) the between-group U-statistic is

    theta_m = mean_{g != h} T_m[g, h],
    T_m[g, h] = sum_j w_j S~_j(c1, g) S~_j(c2, h) / W,

with S~_j(c, g) the class-normalized score sum_{i in g, c} x_ij y_i / n_{c,g}.
It estimates the background mean of E[x_j y | c1] E[x_j y | c2]. Group-level
Hoeffding variance treats the G random individual groups as i.i.d. units. The
SNP-block delete-one jackknife captures LD and SNP-level heterogeneity.

Tests of the degenerate statistics (dosage, dosage[A], delta2) use the
null-restricted variance. Their kernel is a product of two group scores whose
expectation is zero under the null, so E[T_ab | group a] = 0 and the T_ab with
a != b are uncorrelated. Then Var0(theta) = 2 E[T_ab^2] / (G (G - 1)) exactly,
whatever the LD, the phenotype distribution or the group sizes, and E[T_ab^2]
is estimated by the mean of T_ab^2 over a != b. This estimate is far less
noisy than the full Hoeffding variance, whose first-order term is zero under
the null but estimated with noise. It omits that first-order term under
alternatives, so intervals use the Hoeffding and jackknife standard errors.
relative_dosage tests the same null as dosage and uses the same variance.

The p-values of these tests come from sign flips of the group scores. Under the
null the score vectors of different groups are independent with mean zero and,
being sums over about n / G people, symmetric, so
theta(eps) = sum_{a != b} eps_a eps_b T_ab / (G (G - 1)) with Rademacher eps has
the null distribution of theta given the scores. Its variance is the
null-restricted variance, but its skewness, 2 sqrt(2) sum nu^3 / (sum nu^2)^1.5
over the eigenvalues nu of T, is not negligible when the background is a set
of a few hundred loci (0.1-0.3 in UK Biobank trait loci; 0.02 for all
variants), and the normal tail is then several-fold too light near z = 4.
Monte Carlo flips are adaptive (more flips for small p) and seeded from the
kernel, so results are reproducible. p_normal keeps the normal-tail value.
The ratio statistics keep the larger of the Hoeffding and jackknife standard
errors (max-SE rule). rho = 1 lies on the Cauchy-Schwarz boundary, where the
first-order delta-method term vanishes; its Hoeffding SE alone is then
unreliable, and the max-SE rule was checked by simulation (ITEM_23b: nominal
in all null scenarios).

Statistics (lo = low classes, hi = high classes):
    dosage     theta(gy, gy) = mean_j (E[g x_j y])^2, FAME-equivalent; null 0 (one-sided)
    rho        theta(lo, hi) / sqrt(theta(lo, lo) theta(hi, hi)); pattern; null 1
    log_amp    log theta(hi, hi) / theta(lo, lo); genetic-share ratio; null 0
    delta2     theta(lo, lo) + theta(hi, hi) - 2 theta(lo, hi); null 0 (one-sided)
    sel_lam    log lambda_A - log lambda_rest, lambda = theta(lo, hi) / theta(lo, lo); null 0
    sel_amp    log alpha_A - log alpha_rest, alpha = theta(hi, hi) / theta(lo, lo); null 0
    dosage_diff  theta_A(gy, gy) - theta_rest(gy, gy); null 0
    additive   theta(y, y) = mean_j (E[x_j y])^2 (with an additive column); descriptive
    relative_dosage  theta(gy, gy) / theta(y, y): marginal epistasis relative to additive signal
    global_excess    theta(gy, gy) - R theta(y, y), R = (sum_k pi_k g_k s_k / sum_k pi_k s_k)^2 with
               s_k the class SD of the residual: the dosage statistic that an affine focal action
               (class-specific location and scale of one common liability) would produce on this
               scale, subtracted. Approximate (first order) on pooled scales; descriptive
On the within-class rank scale every null holds exactly under an affine latent
action of the context. On other scales rho and sel_* are first-order
scale-free (proportional background), while the others are scale-dependent.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.stats import norm

NULLS = dict(dosage=0., dosage_diff=0., rho=1., log_amp=0., delta2=0., sel_lam=0., sel_amp=0., additive=0.,
             relative_dosage=0., global_excess=0.)
ONE_SIDED = ('dosage', 'delta2', 'additive', 'relative_dosage')
DEGENERATE = ('dosage', 'delta2', 'relative_dosage')
LINEAR = ('dosage', 'delta2')       # degenerate statistics that are linear in the kernels
FLIPS = (20_000, 2_000_000, 2e-3)  # first flips, flips when the first p is below the third value


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


def null_covariance(kernels):
    """Covariance of the off-diagonal means when every kernel is degenerate.

    kernels: (m, G, G) symmetric with zero diagonal. If E[T_m[a, b] | group a] = 0
    for all m, Cov0(theta_m, theta_m') = 2 E[T_m[a, b] T_m'[a, b]] / (G (G - 1)),
    estimated without bias by the mean product over a != b.
    """
    t = np.asarray(kernels, dtype=np.float64)
    g = t.shape[1]
    flat = t.reshape(len(t), -1)
    return 2 * (flat @ flat.T) / (g * (g - 1)) ** 2


def signflip_sf(kernel, flips=FLIPS, seed=None):
    """Upper-tail sign-flip p-value of the off-diagonal sum of a symmetric G x G kernel.

    Returns (p, flips used, null skewness). p = (exceedances + 1) / (flips + 1).
    The default seed is a checksum of the kernel, so equal inputs give equal p.
    """
    t = np.asarray(kernel, dtype=np.float64)
    t = (t + t.T) / 2
    np.fill_diagonal(t, 0.)
    obs = t.sum()
    nu = np.linalg.eigvalsh(t)
    power = np.sum(nu ** 2)
    if not np.isfinite(obs) or power <= 0:
        return np.nan, 0, np.nan
    skew = float(2 * np.sqrt(2) * np.sum(nu ** 3) / power ** 1.5)
    rng = np.random.default_rng(zlib.crc32(t.tobytes()) if seed is None else seed)
    first, most, refine = flips

    def tail(count):
        hits, done = 0, 0
        while done < count:
            m = min(100_000, count - done)
            e = rng.integers(0, 2, size=(m, len(t))).astype(np.float64) * 2 - 1
            hits += int(np.sum(np.einsum('ij,ij->i', e @ t, e) >= obs * (1 - 1e-12)))
            done += m
        return (hits + 1) / (count + 1)

    p, used = tail(first), first
    if p < refine and most > first:
        p, used = tail(most), most
    return float(p), used, skew


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


def _rest(weights, k, total, part):
    """Kernel mean over the complement of annotation k (NaN when the annotation covers everything)."""
    left = weights[0] - weights[k]
    return (weights[0] * total - weights[k] * part) / left if left > 0 else total * np.nan


def dosage_statistics(theta, annotations, weights, global_ratio=None):
    """theta: (1 + K,) dosage kernels for all, A_1..A_K, optionally followed by the
    (1 + K,) additive kernels; weights: background weights per annotation."""
    k_count = len(annotations) + 1
    out = dict(dosage=theta[0])
    for k, name in enumerate(annotations, start=1):
        rest = _rest(weights, k, theta[0], theta[k])
        out[f'dosage[{name}]'] = theta[k]
        out[f'dosage_diff[{name}]'] = theta[k] - rest
    if len(theta) == 2 * k_count:
        additive = theta[k_count:]
        out['additive'] = additive[0]
        out['relative_dosage'] = theta[0] / additive[0] if additive[0] > 0 else np.nan
        if global_ratio is not None:
            out['global_excess'] = theta[0] - global_ratio * additive[0]
        for k, name in enumerate(annotations, start=1):
            out[f'additive[{name}]'] = additive[k]
            out[f'relative_dosage[{name}]'] = theta[k] / additive[k] if additive[k] > 0 else np.nan
    return out


def pattern_statistics(theta, annotations, weights):
    """theta: (3 (1 + K),) ordered (ll, hh, lh) for all, A_1..A_K."""
    ll, hh, lh = theta[0], theta[1], theta[2]
    out = dict(rho=lh / np.sqrt(ll * hh) if ll > 0 and hh > 0 else np.nan,
               log_amp=_log_ratio(hh, ll), delta2=ll + hh - 2 * lh)
    for k, name in enumerate(annotations, start=1):
        a = theta[3 * k:3 * k + 3]
        rest = _rest(weights, k, theta[:3], a)
        out[f'sel_lam[{name}]'] = _log_ratio(a[2], a[0]) - _log_ratio(rest[2], rest[0])
        out[f'sel_amp[{name}]'] = _log_ratio(a[1], a[0]) - _log_ratio(rest[1], rest[0])
    return out


def _family(raw, between, weights, block_weights, function, annotation_weights, annotation_blocks, flips=FLIPS):
    """Estimates with Hoeffding (delta method) and SNP-block jackknife SEs.

    raw: (m, G, G) unnormalized kernels; between: (m, n_blocks) between-group
    sums; weights: (m,) background weight of each kernel; block_weights:
    (m, n_blocks). function(theta, annotation_weights) gives the statistics;
    annotation_weights: (A,) and annotation_blocks: (A, n_blocks), so that each
    delete-one-block estimate uses its own complement weights. Values are
    (estimate, se_hoeffding, se_jackknife, se_null, p_signflip, flips); the last two
    are set for dosage-type statistics when flips is not None (relative_dosage
    takes the p of its dosage statistic, which tests the same null).
    """
    kernels = np.stack([_kernel(r, w) for r, w in zip(raw, weights)])
    theta, cov = hoeffding(kernels)
    cov0 = null_covariance(kernels)
    est, grads = _gradient(lambda t: function(t, annotation_weights), theta)
    g = raw.shape[1]
    pairs = g * (g - 1)
    total = between.sum(axis=1)
    used = block_weights.sum(axis=0) > 0
    drops = []
    for b in np.flatnonzero(used):
        w = weights - block_weights[:, b]
        drops.append(function((total - between[:, b]) / (w * pairs), annotation_weights - annotation_blocks[:, b]))
    nb = len(drops)
    result = {}
    for key, value in est.items():
        var = float(grads[key] @ cov @ grads[key])
        se_u = np.sqrt(var) if np.isfinite(var) and var > 0 else np.nan
        values = np.array([d[key] for d in drops], dtype=np.float64)
        finite = np.isfinite(values)
        se_s = (np.sqrt((nb - 1) / nb * np.sum((values - values.mean()) ** 2))
                if np.isfinite(value) and finite.all() and nb > 1 else np.nan)
        se_0 = np.nan
        if key.split('[')[0] in DEGENERATE:
            var0 = float(grads[key] @ cov0 @ grads[key])
            se_0 = np.sqrt(var0) if np.isfinite(var0) and var0 > 0 else np.nan
        p_flip, used = np.nan, 0
        if flips is not None and key.split('[')[0] in LINEAR and np.isfinite(value):
            p_flip, used, _ = signflip_sf(np.tensordot(grads[key], kernels, axes=1), flips)
        result[key] = (float(value), float(se_u), float(se_s), float(se_0), p_flip, used)
    for key in result:
        if key.split('[')[0] == 'relative_dosage':
            source = result.get(key.replace('relative_dosage', 'dosage', 1))
            if source is not None:
                result[key] = result[key][:4] + source[4:]
    return result


def _null(name):
    return NULLS[name.split('[')[0]]


def _record(base, statistic, value, se_u, se_s, se_0, p_flip=np.nan, flips=0):
    """The test uses se_0 (null-restricted) for degenerate statistics, else the max-SE rule.

    p is the sign-flip p-value when one was computed, else the normal tail of z."""
    if statistic.split('[')[0] in DEGENERATE:
        se, rule = se_0, 'null_restricted'
    else:
        se = np.nanmax([se_u, se_s]) if np.isfinite([se_u, se_s]).any() else np.nan
        rule = 'max'
    null = _null(statistic)
    z = (value - null) / se if np.isfinite(se) and se > 0 else np.nan
    one_sided = statistic.split('[')[0] in ONE_SIDED
    p_normal = (norm.sf(z) if one_sided else 2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    p = p_flip if np.isfinite(p_flip) else p_normal
    zu = (value - null) / se_u if np.isfinite(se_u) and se_u > 0 else np.nan
    zs = (value - null) / se_s if np.isfinite(se_s) and se_s > 0 else np.nan
    return dict(base, statistic=statistic, estimate=value, null=null, se_hoeffding=se_u, se_jackknife=se_s,
                se_null=se_0, se=se, se_rule=rule, z=z, z_hoeffding=zu, z_jackknife=zs, p=p, p_normal=p_normal,
                p_method='signflip' if np.isfinite(p_flip) else 'normal', flips=int(flips),
                sided='upper' if one_sided else 'two')


def summarize(plan, accumulation, flips=FLIPS):
    """One record per analysis, scale and statistic.

    flips: (first, most, refine) for the sign-flip p-values of dosage-type
    statistics, or None for normal-tail p-values only.

    Analyses with an empty background are omitted. An annotation with no kept
    weight for an analysis (for example one confined to the context's own
    chromosome) is dropped from that analysis only.
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

    def annotations_used(w):
        return [0] + [k for k in range(1, len(w)) if w[k] > 0]

    additive = {key: i for i, key in enumerate(plan.keys.get('y', ()))}
    if 'gy' in accumulation.kernels:
        for p, (a, scale) in enumerate(plan.keys['gy']):
            w, blocks = weights_for('gy', p)
            if w[0] <= 0:
                continue
            sel = annotations_used(w)
            sub = [names[k - 1] for k in sel[1:]]
            w, blocks = w[sel], blocks[sel]
            raw, between = accumulation.kernels['gy'][sel, p], accumulation.between['gy'][sel, p]
            ww, bb = w, blocks
            ratio = None
            if (a, scale) in additive and 'yy' in accumulation.kernels:
                i = additive[(a, scale)]
                raw = np.concatenate([raw, accumulation.kernels['yy'][sel, i]])
                between = np.concatenate([between, accumulation.between['yy'][sel, i]])
                ww, bb = np.r_[w, w], np.concatenate([blocks, blocks])
                ratio = plan.analyses[a].get('global_ratio', {}).get(scale)
            stats = _family(raw, between, ww, bb, lambda t, aw, s=sub, r=ratio: dosage_statistics(t, s, aw, r),
                            w, blocks, flips)
            records += [_record(base(a, scale), k, *v) for k, v in stats.items()]
    if 'll' in accumulation.kernels:
        for p, (a, scale) in enumerate(plan.keys['lo']):
            w, blocks = weights_for('ll', p)
            if w[0] <= 0:
                continue
            sel = annotations_used(w)
            sub = [names[k - 1] for k in sel[1:]]
            w, blocks = w[sel], blocks[sel]
            raw = np.stack([accumulation.kernels[k][i, p] for i in sel for k in ('ll', 'hh', 'lh')])
            between = np.stack([accumulation.between[k][i, p] for i in sel for k in ('ll', 'hh', 'lh')])
            ww, bb = np.repeat(w, 3), np.repeat(blocks, 3, axis=0)
            stats = _family(raw, between, ww, bb, lambda t, aw, s=sub: pattern_statistics(t, s, aw), w, blocks,
                            flips)
            records += [_record(base(a, scale), k, *v) for k, v in stats.items()]
    return records
