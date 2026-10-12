"""Scale-robust tests for one pair of genotype classes (x, z) on one outcome.

Every statistic except the fixed-scale baselines depends on the outcome only
through ranks, or is exact under a stated scale null:

    cpm_fit              cumulative-probability (ordinal) model on rank bins, four
                         links; the x*z coefficient is invariant to increasing
                         transforms of y (analytic Newton, sandwich SE)
    cpm_location_scale   location-scale CPM: interaction net of genotype-specific
                         dispersion (L-BFGS, numerical Hessian)
    stratified_rank_test rank-inverse-normal within z (x sex) strata; tests whether
                         x effects differ across z strata. Exact when z acts on
                         the latent scale only by an affine map.
    auc_contrasts        Mann-Whitney contrasts of x within z strata (DeLong SEs)
    robustness_radius    R* = max(P/N, N/P) of the CDF interaction D(t) on the
                         pooled-rank scale, with a sup-t band lower bound; R*_lo =
                         inf establishes a qualitative (one-signed) interaction
    hc3_product_test     OLS x*z product with HC3 on a fixed scale (baseline)

See docs/wiki/Epistasis-scale-free.md for the scale-null hierarchy.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm, rankdata

from .transforms import blom_rank_normal

LINKS = ('logit', 'probit', 'cloglog', 'loglog')


def _link(name):
    if name == 'logit':
        def f(u):
            p = expit(u)
            return p * (1 - p)

        def fp(u):
            p = expit(u)
            return p * (1 - p) * (1 - 2 * p)
        return expit, f, fp, lambda q: np.log(q / (1 - q))
    if name == 'probit':
        return norm.cdf, norm.pdf, lambda u: -u * norm.pdf(u), norm.ppf
    if name == 'cloglog':      # left-skewed latent noise (Gumbel minimum)
        def F(u):
            return -np.expm1(-np.exp(np.clip(u, -40, 40)))

        def f(u):
            e = np.exp(np.clip(u, -40, 40))
            return e * np.exp(-e)

        def fp(u):
            e = np.exp(np.clip(u, -40, 40))
            return e * np.exp(-e) * (1 - e)
        return F, f, fp, lambda q: np.log(-np.log1p(-q))
    if name == 'loglog':       # right-skewed latent noise (Gumbel maximum)
        def F(u):
            return np.exp(-np.exp(-np.clip(u, -40, 40)))

        def f(u):
            e = np.exp(-np.clip(u, -40, 40))
            return e * np.exp(-e)

        def fp(u):
            e = np.exp(-np.clip(u, -40, 40))
            return e * np.exp(-e) * (e - 1)
        return F, f, fp, lambda q: -np.log(-np.log(q))
    raise ValueError(f'unknown link {name}')


def rank_bins(y, bins=40):
    """Rank-based categories: identical for every increasing transform of y."""
    q = np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1])
    c = np.searchsorted(np.unique(q), y, side='right')
    return np.unique(c, return_inverse=True)[1]


def _bin_matrix(index, mask, m):
    rows = np.flatnonzero(mask)
    return sparse.csr_matrix((np.ones(len(rows)), (index[rows], rows)), shape=(m, len(index)))


def _cell_terms(a, eta, c, k, F, f, fp):
    up = np.where(c < k - 1, a[np.minimum(c, k - 2)], np.inf)
    lo = np.where(c > 0, a[np.maximum(c - 1, 0)], -np.inf)
    upper, lower = up - eta, lo - eta
    fin_u, fin_l = np.isfinite(upper), np.isfinite(lower)
    su, sl = np.where(fin_u, upper, 0), np.where(fin_l, lower, 0)
    FU, FL = np.where(fin_u, F(su), 1.), np.where(fin_l, F(sl), 0.)
    fU, fL = np.where(fin_u, f(su), 0.), np.where(fin_l, f(sl), 0.)
    dU, dL = np.where(fin_u, fp(su), 0.), np.where(fin_l, fp(sl), 0.)
    return np.clip(FU - FL, 1e-300, None), fU, fL, dU, dL


def cpm_fit(c, X, link='logit', maxit=100, tol=1e-9):
    """Proportional CPM P(Y <= k | X) = F(a_k - X b). c: categories 0..K-1; X: n x p, no intercept.

    Returns beta, model and sandwich covariances, log-likelihood and convergence.
    A failed line search ends the fit with converged=False rather than raising.
    """
    F, f, fp, Finv = _link(link)
    n, p = X.shape
    k = int(c.max()) + 1
    m = k - 1
    a = Finv(np.clip(np.cumsum(np.bincount(c, minlength=k))[:-1] / n, 1e-6, 1 - 1e-6))
    b = np.zeros(p)
    has_u, has_l = c < k - 1, c > 0
    bin_u = _bin_matrix(np.minimum(c, m - 1), has_u, m)
    bin_l = _bin_matrix(np.maximum(c - 1, 0), has_l, m)
    both = has_u & has_l
    bin_ul = _bin_matrix(np.maximum(c - 1, 0), both, m)

    def loglik(a, b):
        return np.log(_cell_terms(a, X @ b, c, k, F, f, fp)[0]).sum()

    def derivatives(a, b):
        pr, fU, fL, dU, dL = _cell_terms(a, X @ b, c, k, F, f, fp)
        g = np.r_[bin_u @ (fU / pr) - bin_l @ (fL / pr), -X.T @ ((fU - fL) / pr)]
        h = np.zeros((m + p, m + p))
        idx = np.arange(m)
        h[idx, idx] = bin_u @ (dU / pr - fU ** 2 / pr ** 2) + bin_l @ (-dL / pr - fL ** 2 / pr ** 2)
        off = (bin_ul @ (fU * fL / pr ** 2))[:m - 1]
        h[idx[1:], idx[:-1]] += off
        h[idx[:-1], idx[1:]] += off
        h[m:, m:] = X.T @ (X * ((dU - dL) / pr - (fU - fL) ** 2 / pr ** 2)[:, None])
        cross = (bin_u @ (X * (-(dU / pr - fU * (fU - fL) / pr ** 2))[:, None])
                 + bin_l @ (X * (dL / pr - fL * (fU - fL) / pr ** 2)[:, None]))
        h[:m, m:], h[m:, :m] = cross, cross.T
        return g, h, (pr, fU, fL)

    ll, converged, iterations = loglik(a, b), False, 0
    for iterations in range(1, maxit + 1):
        g, h, _ = derivatives(a, b)
        step = np.linalg.solve(h, g)
        t, accepted = 1., False
        while t >= 1e-8:
            a2, b2 = a - t * step[:m], b - t * step[m:]
            if np.all(np.diff(a2) > 0):
                ll2 = loglik(a2, b2)
                if np.isfinite(ll2) and ll2 >= ll - 1e-10 * max(1., abs(ll)):
                    accepted = True
                    break
            t /= 2
        if not accepted:
            break
        a, b, change, ll = a2, b2, ll2 - ll, ll2
        if abs(change) < tol and np.max(np.abs(t * step)) < 1e-7:
            converged = True
            break
    _, h, (pr, fU, fL) = derivatives(a, b)
    cov = np.linalg.inv(-h)
    scores = np.zeros((n, m + p))
    rows = np.arange(n)
    scores[rows[has_u], c[has_u]] += (fU / pr)[has_u]
    scores[rows[has_l], c[has_l] - 1] -= (fL / pr)[has_l]
    scores[:, m:] = -X * ((fU - fL) / pr)[:, None]
    sandwich = cov @ (scores.T @ scores) @ cov
    return dict(beta=b, cov=cov[m:, m:], cov_sandwich=sandwich[m:, m:], loglik=float(ll), iterations=iterations,
                cutpoints=a, converged=converged)


def cpm_location_scale(c, X, W, link='probit'):
    """Location-scale CPM P(Y <= k) = F((a_k - X b) / exp(W g)); interaction net of dispersion."""
    F, f, _, Finv = _link(link)
    n, p = X.shape
    q = W.shape[1]
    k = int(c.max()) + 1
    m = k - 1
    a0 = Finv(np.clip(np.cumsum(np.bincount(c, minlength=k))[:-1] / n, 1e-6, 1 - 1e-6))
    has_u, has_l = c < k - 1, c > 0
    bin_u = _bin_matrix(np.minimum(c, m - 1), has_u, m)
    bin_l = _bin_matrix(np.maximum(c - 1, 0), has_l, m)

    def objective(th):
        a = np.cumsum(np.r_[th[0], np.exp(th[1:m])])
        b, gm = th[m:m + p], th[m + p:]
        eta, tau = X @ b, np.exp(W @ gm)
        up = np.where(has_u, a[np.minimum(c, m - 1)], 0.)
        lo = np.where(has_l, a[np.maximum(c - 1, 0)], 0.)
        U, L = (up - eta) / tau, (lo - eta) / tau
        FU, FL = np.where(has_u, F(U), 1.), np.where(has_l, F(L), 0.)
        fU, fL = np.where(has_u, f(U), 0.), np.where(has_l, f(L), 0.)
        pr = np.clip(FU - FL, 1e-300, None)
        ga = bin_u @ (fU / pr / tau) - bin_l @ (fL / pr / tau)
        rev = np.cumsum(ga[::-1])[::-1]
        gth = np.r_[rev[0], rev[1:] * np.exp(th[1:m])]
        gb = -X.T @ ((fU - fL) / pr / tau)
        gg = -W.T @ ((fU * np.where(has_u, U, 0.) - fL * np.where(has_l, L, 0.)) / pr)
        return -np.log(pr).sum(), -np.r_[gth, gb, gg]

    start = np.r_[a0[0], np.log(np.diff(a0)), np.zeros(p + q)]
    fit = minimize(objective, start, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=5000, maxfun=20000, ftol=1e-13, gtol=1e-7))
    th, size = fit.x, len(fit.x)
    hessian = np.zeros((size, size))
    for i in range(size):
        e = np.zeros(size)
        e[i] = 1e-5 * max(1, abs(th[i]))
        hessian[:, i] = (objective(th + e)[1] - objective(th - e)[1]) / (2 * e[i])
    cov = np.linalg.inv((hessian + hessian.T) / 2)
    return dict(beta=th[m:m + p], cov=cov[m:m + p, m:m + p], gamma=th[m + p:], cov_gamma=cov[m + p:, m + p:],
                loglik=float(-fit.fun), converged=bool(fit.success))


def hc3_wald(y, X, test):
    """OLS with HC3 covariance; Wald test of coefficients `test` (X includes the intercept)."""
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - X @ beta
    leverage = np.sum(Q * Q, axis=1)
    rinv = np.linalg.inv(R)
    bread = rinv @ rinv.T
    scaled = X * (resid / (1 - leverage))[:, None]
    cov = bread @ (scaled.T @ scaled) @ bread
    bt, vt = beta[test], cov[np.ix_(test, test)]
    w, U = np.linalg.eigh((vt + vt.T) / 2)
    keep = w > w.max() * 1e-10
    stat = float(np.sum((U[:, keep].T @ bt) ** 2 / w[keep]))
    df = int(keep.sum())
    return dict(beta=beta, cov=cov, stat=stat, df=df, p=float(chi2.sf(stat, df)))


def stratified_rank_test(y, x, z, covariates=None, strata=None, min_cell=50):
    """Rank-normalize within z (x strata) and test x effects across z strata (HC3).

    Exact under an affine latent action of z. Returns a 2(S-1)-df omnibus test
    and a 1-df trend test of x_add x z. z strata lacking support in at least
    two x classes (min_cell each) are dropped.
    """
    supported = [s for s in np.unique(z) if sum(np.sum((z == s) & (x == v)) >= min_cell for v in (0, 1, 2)) >= 2]
    keep = np.isin(z, supported)
    y, x, z = y[keep], x[keep], z[keep]
    covariates = None if covariates is None else covariates[keep]
    stratum = z.astype(np.int64) if strata is None else (
        z.astype(np.int64) * (int(np.max(strata)) + 1) + np.asarray(strata)[keep].astype(np.int64))
    ranked = blom_rank_normal(y, stratum)
    levels = np.unique(z)
    xa = x - x.mean()
    xh = (x == 1).astype(float)
    xh -= xh.mean()
    base = [np.ones(len(y))] + [(z == s).astype(float) for s in levels[1:]] + [xa, xh]
    cols, test = list(base), []
    for s in levels[1:]:
        cols.append(xa * (z == s))
        test.append(len(cols) - 1)
        if np.sum((z == s) & (x == 1)) >= min_cell and np.sum((z == s) & (x != 1)) >= min_cell:
            cols.append(xh * (z == s))
            test.append(len(cols) - 1)
    extra = [] if covariates is None else list(np.asarray(covariates).T)
    omnibus = hc3_wald(ranked, np.column_stack(cols + extra), test)
    trend_cols = base + [xa * (z - z.mean())] + extra
    position = len(base)
    trend = hc3_wald(ranked, np.column_stack(trend_cols), [position])
    beta = float(trend['beta'][position])
    return dict(omnibus_p=omnibus['p'], omnibus_stat=omnibus['stat'], omnibus_df=omnibus['df'], trend_beta=beta,
                trend_z=beta / float(np.sqrt(trend['cov'][position, position])), trend_p=trend['p'],
                strata=[int(s) for s in levels])


def _auc(a, b):
    n1, n0 = len(a), len(b)
    r = rankdata(np.r_[a, b])
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    v10 = (r[:n1] - rankdata(a)) / n0
    v01 = 1 - (r[n1:] - rankdata(b)) / n1
    return auc, np.sqrt(v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0)


def auc_contrasts(y, x, z, high=2, low=0, min_cell=30):
    """AUC(x = high vs x = low) within each z stratum; first-last difference and IVW trend."""
    out = []
    for s in np.unique(z):
        a, b = y[(z == s) & (x == high)], y[(z == s) & (x == low)]
        if len(a) >= min_cell and len(b) >= min_cell:
            auc, se = _auc(a, b)
            out.append(dict(stratum=int(s), auc=auc, se=se, n_high=len(a), n_low=len(b)))
    if len(out) < 2:
        return dict(strata=out)
    diff = out[0]['auc'] - out[-1]['auc']
    zs = np.array([o['stratum'] for o in out], float)
    aucs = np.array([o['auc'] for o in out])
    w = 1 / np.array([o['se'] for o in out]) ** 2
    centred = zs - np.sum(w * zs) / w.sum()
    slope = np.sum(w * centred * aucs) / np.sum(w * centred ** 2)
    return dict(strata=out, first_last_diff=diff, first_last_z=diff / np.hypot(out[0]['se'], out[-1]['se']),
                trend=slope, trend_z=slope * np.sqrt(np.sum(w * centred ** 2)))


def _cdf_interaction(t, x, z, grid, xs, zs):
    F = {}
    for xv in xs:
        for zv in zs:
            v = np.sort(t[(x == xv) & (z == zv)])
            F[(xv, zv)] = np.searchsorted(v, grid, side='right') / len(v)
    return F[(xs[1], zs[1])] - F[(xs[0], zs[1])] - F[(xs[1], zs[0])] + F[(xs[0], zs[0])]


def robustness_radius(y, x, z, *, reference='rank', xs=(0, 2), zs=(0, 2), boot=200, seed=0, grid_size=400,
                      alpha=.05, min_cell=100):
    """Scale-distortion robustness radius of a 2 x 2 genotype interaction.

    On reference scale t (pooled ranks by default), D(t) = F11 - F10 - F01 + F00.
    For transforms whose slope relative to t varies at most R-fold, the mean
    interaction ranges over [P - R N, R P - N]. R* = max(P/N, N/P) is the
    smallest distortion that removes it. A simultaneous one-sided (1 - alpha/2)
    sup-t bootstrap band L <= D gives the conservative R*_lo = int L+ / int L-
    (mirrored for a negative interaction); R*_lo = inf establishes a
    qualitative interaction.
    """
    n = len(y)
    cell_n = [int(np.sum((x == xv) & (z == zv))) for xv in xs for zv in zs]
    if min(cell_n) < min_cell:
        return dict(reference=reference, R_star=np.nan, P=np.nan, N=np.nan, sign=None, lower=np.nan, alpha=alpha,
                    cell_n=cell_n, note=f'2 x 2 cells need at least {min_cell} individuals')
    t = (rankdata(y) / (n + 1) if reference == 'rank' else np.log(y) if reference == 'log' else
         np.asarray(y, dtype=np.float64))
    grid = np.quantile(t, np.linspace(.5 / grid_size, 1 - .5 / grid_size, grid_size))
    w = np.gradient(grid)

    def masses(d):
        return float(np.sum(np.clip(d, 0, None) * w)), float(np.sum(np.clip(-d, 0, None) * w))

    d = _cdf_interaction(t, x, z, grid, xs, zs)
    P, N = masses(d)
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero((x == xv) & (z == zv)) for xv in xs for zv in zs]
    draws = np.empty((boot, grid_size))
    for b in range(boot):
        idx = np.concatenate([rng.choice(cell, len(cell)) for cell in cells])
        draws[b] = _cdf_interaction(t[idx], x[idx], z[idx], grid, xs, zs)
    se = np.maximum(draws.std(axis=0, ddof=1), 1e-12)
    dev = (draws - d) / se
    c_lo, c_hi = np.quantile(dev.max(axis=1), 1 - alpha / 2), np.quantile((-dev).max(axis=1), 1 - alpha / 2)
    lp, ln = masses(d - c_lo * se)
    up, un = masses(d + c_hi * se)
    positive = P >= N
    estimate = (P / N if N > 0 else np.inf) if positive else (N / P if P > 0 else np.inf)
    lower = (lp / ln if ln > 0 else np.inf) if positive else (un / up if up > 0 else np.inf)
    return dict(reference=reference, R_star=float(estimate), P=P, N=N, sign='positive' if positive else 'negative',
                lower=float(lower), alpha=alpha, band_c=(float(c_lo), float(c_hi)), cell_n=cell_n)


def hc3_product_test(y, x, z, covariates=None):
    """Fixed-scale baseline: y ~ x_add + x_het + z_add + z_het + x_add z_add (+ covariates), HC3."""
    xa, za = x - x.mean(), z - z.mean()
    cols = [np.ones(len(y)), xa, (x == 1) - np.mean(x == 1), za, (z == 1) - np.mean(z == 1), xa * za]
    if covariates is not None:
        cols += list(np.asarray(covariates).T)
    r = hc3_wald(y, np.column_stack(cols), [5])
    return dict(beta=float(r['beta'][5]), z=float(r['beta'][5] / np.sqrt(r['cov'][5, 5])), p=r['p'])


def pair_battery(y, x, z, covariates, *, strata=None, links=LINKS, bins=40, boot=200, seed=7,
                 min_homozygote_cell=500):
    """All pair statistics for minor-allele counts x, z (0/1/2) and a positive outcome y.

    covariates: (n, p) without intercept; they are centred here. strata (e.g.
    sex) enter the stratified rank tests and the dispersion model. Returns a
    flat dict.
    """
    y, x, z = np.asarray(y, float), np.asarray(x, np.int64), np.asarray(z, np.int64)
    cov = np.asarray(covariates, float)
    cov = (cov - cov.mean(axis=0)) / np.where(cov.std(axis=0) > 0, cov.std(axis=0), 1)

    def centred(v):
        v = np.asarray(v, float)
        return v - v.mean()

    X = np.column_stack([centred(x), centred(x == 1), centred(z), centred(z == 1), cov, centred(x) * centred(z)])
    c = rank_bins(y, bins)
    out = dict(n=int(len(y)), maf_x=float(x.mean() / 2), maf_z=float(z.mean() / 2))
    for link in links:
        r = cpm_fit(c, X, link)
        out[f'cpm_{link}_beta'] = float(r['beta'][-1])
        out[f'cpm_{link}_z'] = float(r['beta'][-1] / np.sqrt(r['cov_sandwich'][-1, -1])) if r['converged'] else np.nan
        out[f'cpm_{link}_converged'] = bool(r['converged'])
    dispersion = [centred(x == 1), centred(x == 2), centred(z == 1), centred(z == 2)]
    if strata is not None:
        dispersion.append(centred(strata))
    r = cpm_location_scale(c, X, np.column_stack(dispersion), 'probit')
    out.update(ls_probit_beta=float(r['beta'][-1]), ls_probit_z=float(r['beta'][-1] / np.sqrt(r['cov'][-1, -1])),
               ls_converged=r['converged'], ls_scale_gamma=[float(v) for v in r['gamma']])
    by_z = stratified_rank_test(y, x, z, covariates=cov, strata=strata)
    by_x = stratified_rank_test(y, z, x, covariates=cov, strata=strata)
    out.update(strat_byz_trend_z=by_z['trend_z'], strat_byz_omnibus_p=by_z['omnibus_p'],
               strat_byx_trend_z=by_x['trend_z'], strat_byx_omnibus_p=by_x['omnibus_p'])
    a = auc_contrasts(y, x, z, high=2 if np.sum(x == 2) >= min_homozygote_cell else 1, low=0)
    out['auc_trend_z'] = float(a.get('trend_z', np.nan))
    homozygous = all(np.sum((x == xv) & (z == zv)) >= min_homozygote_cell for xv in (0, 2) for zv in (0, 2))
    xx, zz = (x, z) if homozygous else (np.minimum(x, 1), np.minimum(z, 1))
    levels = (0, 2) if homozygous else (0, 1)
    rr = robustness_radius(y, xx, zz, xs=levels, zs=levels, boot=boot, seed=seed)
    out.update(R_star=rr['R_star'], R_star_lower=rr['lower'], R_star_sign=rr['sign'],
               R_star_cells='homozygote' if homozygous else 'carrier')
    out['hc3_raw_z'] = hc3_product_test(y, x, z, cov)['z']
    if np.all(y > 0):
        out['hc3_log_z'] = hc3_product_test(np.log(y), x, z, cov)['z']
    return out
