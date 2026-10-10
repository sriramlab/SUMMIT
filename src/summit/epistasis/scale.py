"""Continuous Box--Cox union-null tests for finite interaction mean effects.

The reported p upper bound dominates every pointwise HC3 Wald p in the
declared compact interval. Adaptive searches cannot substitute their sampled
maximum for this bound. Statistical validity inherits pointwise HC3 validity;
this is not an arbitrary-monotone or causal epistasis test.
"""
from math import factorial

import numpy as np
from scipy.stats import chi2

from .robust import _inverse


def boxcox_derivatives(log_values, power, *, order=3):
    """Outcome and first ``order`` power derivatives, including power zero.

    d^k/dlambda^k BoxCox(exp(t),lambda) =
        t^(k+1) integral_0^1 u^k exp(lambda*t*u) du.
    A convergent series near zero avoids cancellation in the recurrence.
    """
    t = np.asarray(log_values, float)
    if (t.ndim != 1 or not np.all(np.isfinite(t)) or not np.isfinite(power)
            or type(order) is not int or not 0 <= order <= 4):
        raise ValueError('finite log outcomes/power and derivative order 0..4 required')
    x = float(power) * t
    if np.max(x, initial=0.) > 600:
        raise ValueError('Box-Cox range exceeds the supported numerical dynamic range')
    small = abs(x) <= .5
    values = np.empty((len(t), order+1))
    integrals = np.empty_like(values)
    xs = x[small]
    for k in range(order+1):
        term = np.ones(len(xs))
        total = term / (k+1)
        for j in range(1, 32):
            term = term * xs / j
            total += term / (j+k+1)
        integrals[small, k] = total
    large = ~small
    xl = x[large]
    exponential = np.exp(xl)
    integrals[large, 0] = np.expm1(xl) / xl
    for k in range(1, order+1):
        integrals[large, k] = (exponential-k*integrals[large, k-1]) / xl
    for k in range(order+1):
        values[:, k] = t**(k+1) * integrals[:, k]
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite Box-Cox derivative')
    return values


def boxcox_scale_test(geometry, phenotype, *, groups=None, bounds=(-2., 2.),
                      alpha=.05, max_evaluations=129, batch_size=8, order=3):
    """Bound the supremum HC3 p over a continuous Box--Cox power interval.

    ``geometry`` is ``prepare_robust_geometry`` on fixed/frozen features and
    nuisance terms. All tested coefficients must be identifiable. ``groups``
    maps names to coefficient indices; every test is conditional on the other
    columns. The default tests all columns together. ``alpha`` is the declared
    per-test decision threshold, including the caller's multiplicity factor.

    Reuse one QR and protected NN/TN products. Each transformed outcome and
    derivative is refit, including the nuisance mean and HC3 covariance.
    Unresolved intervals retain a conservative upper bound (possibly one).
    """
    y = np.asarray(phenotype, float)
    lower, upper = map(float, bounds)
    if (y.shape != (len(geometry.r),) or not np.all(np.isfinite(y)) or np.any(y <= 0)
            or not np.isfinite(lower+upper) or lower >= upper
            or not np.isfinite(alpha) or not 0 < alpha < 1
            or type(max_evaluations) is not int or max_evaluations < 3
            or type(batch_size) is not int or batch_size < 2
            or type(order) is not int or not 1 <= order <= 4):
        raise ValueError('positive aligned outcome, finite ordered interval and valid search settings required')
    if not np.all(geometry.estimable):
        raise ValueError('scale tests require identifiable supplied feature coefficients')
    one = np.ones((len(y), 1))
    constant_residual = one-geometry.nn(geometry.u, geometry.tn(geometry.u, one))
    if np.linalg.norm(constant_residual) > 1e-9*np.sqrt(len(y)):
        raise ValueError('scale tests require an intercept in the nuisance span')
    p = geometry.r.shape[1]
    groups = {'joint': list(range(p))} if groups is None else dict(groups)
    if not groups:
        raise ValueError('at least one coefficient group required')
    indices = {}
    for name, selected in groups.items():
        index = np.asarray(selected)
        if (not isinstance(name, str) or not name or index.ndim != 1 or not len(index)
                or not np.issubdtype(index.dtype, np.integer) or len(np.unique(index)) != len(index)
                or np.any(index < 0) or np.any(index >= p)):
            raise ValueError('distinct valid integer coefficient indices required per named group')
        indices[name] = index
    names = list(indices)
    # This outcome-dependent unit choice is exactly irrelevant to every Wald
    # test: changing units adds an intercept and multiplies the outcome by a
    # positive lambda-dependent constant. It does not fit a transformation.
    log_y = np.log(y)
    log_reference = float(log_y.mean())
    t = log_y-log_reference
    if max(abs(lower), abs(upper))*np.max(abs(t)) > 600:
        raise ValueError('Box-Cox interval exceeds supported numerical dynamic range')
    weights = geometry.nn(geometry.r, geometry.inverse.T)
    weights -= geometry.nn(geometry.u, geometry.tn(geometry.u, weights))
    weights[geometry.saturated] = 0
    cache = {}

    def evaluate(powers):
        powers = [float(v) for v in powers if float(v) not in cache]
        for start in range(0, len(powers), batch_size):
            batch = powers[start:start+batch_size]
            values = np.column_stack([boxcox_derivatives(t, v, order=order) for v in batch])
            _, beta, _, meat, _ = geometry.fit(values)
            covariances = np.stack([geometry.inverse @ v @ geometry.inverse.T for v in meat])
            for j, power in enumerate(batch):
                sl = slice(j*(order+1), (j+1)*(order+1))
                b, cov = beta[:, sl].copy(), covariances[sl].copy()
                records = {}
                for name, index in indices.items():
                    v = cov[0][np.ix_(index, index)]
                    try:
                        inverse, _ = _inverse((v+v.T)/2)
                        direction = inverse @ b[index, 0]
                        statistic = float(b[index, 0] @ direction)
                        if not np.isfinite(statistic) or statistic < 0:
                            raise ValueError('unresolved scale covariance')
                        direction = direction/np.sqrt(statistic) if statistic > 0 else np.zeros(len(index))
                        records[name] = dict(p=float(chi2.sf(statistic, len(index))),
                            statistic=statistic, direction=direction, inverse=inverse, resolved=True)
                    except (ValueError, np.linalg.LinAlgError):
                        records[name] = dict(p=1., statistic=0., direction=np.zeros(len(index)), resolved=False)
                cache[power] = dict(beta=b, covariance=cov, records=records,
                    outcome_norm=float(np.linalg.norm(values[:, j*(order+1)])))

    def interval(a, b):
        center, radius = (a+b)/2, (b-a)/2
        value = cache[center]
        # Taylor's integral remainder: |f^(r+1)| <=
        # |t|^(r+2) exp(max(0,center*t+radius*|t|))/(r+2).
        remainder = (radius**(order+1)/factorial(order+1)/(order+2)
            * abs(t)**(order+2)*np.exp(np.maximum(0., center*t+radius*abs(t))))
        remainder_norm = float(np.linalg.norm(remainder))
        limits = {}
        for name, index in indices.items():
            record = value['records'][name]
            if not record['resolved'] or record['statistic'] == 0:
                limits[name] = 1.
                continue
            direction = record['direction']
            influence = geometry.nn(weights[:, index], direction[:, None])[:, 0]
            numerator_change, denominator_change = 0., 0.
            covariance_radius = 0.
            for k in range(1, order+1):
                factor = radius**k/factorial(k)
                numerator_change += factor*abs(float(direction @ value['beta'][index, k]))
                variance = float(direction @ value['covariance'][k][np.ix_(index, index)] @ direction)
                denominator_change += factor*np.sqrt(max(0., variance))
                covariance_radius += factor*np.sqrt(max(0., float(np.trace(
                    record['inverse'] @ value['covariance'][k][np.ix_(index,index)]))))
            whitened_rows = np.sum(weights[:,index]*geometry.nn(weights[:,index],record['inverse']),axis=1)
            covariance_radius += (1+1e-8)*float(np.max(
                np.sqrt(np.maximum(0.,whitened_rows))/geometry.denominator))*remainder_norm
            # In the center covariance metric, the influence matrix starts
            # with all singular values one. A perturbation norm below one
            # proves full rank throughout this interval. Otherwise retain the
            # safe bound rather than assuming an unseen covariance is regular.
            if not np.isfinite(covariance_radius) or covariance_radius >= 1-1e-8:
                limits[name] = 1.
                continue
            numerator_change += float(abs(influence) @ remainder)
            # The full OLS residual map is an orthogonal contraction. HC3 adds
            # the diagonal influence/remaining-leverage factor, whose operator
            # norm is its largest absolute entry. No outcome model enters this
            # deterministic envelope.
            denominator_change += (1+1e-8)*float(np.max(abs(influence/geometry.denominator)))*remainder_norm
            numerical_slack = 1e-8*(1+np.sqrt(record['statistic'])+numerator_change
                +np.linalg.norm(influence)*value['outcome_norm'])
            numerator = max(0., np.sqrt(record['statistic'])-numerator_change-numerical_slack)
            denominator = 1+denominator_change+numerical_slack
            bound = float(chi2.sf((numerator/denominator)**2, len(index)))
            limits[name] = min(1., max(bound, record['p'])+1e-12)
        return dict(lower=a, upper=b, center=center, p_upper=limits)

    evaluate([lower, (lower+upper)/2, upper])
    leaves = [interval(lower, upper)]
    while True:
        sampled = {name: max(v['records'][name]['p'] for v in cache.values()) for name in names}
        envelope = {name: max(v['p_upper'][name] for v in leaves) for name in names}
        unresolved = [name for name in names if sampled[name] < alpha <= envelope[name]]
        remaining = (max_evaluations-len(cache))//2
        if not unresolved or remaining < 1:
            break
        candidates = [j for j, leaf in enumerate(leaves)
            if any(leaf['p_upper'][name] >= alpha for name in unresolved)]
        candidates.sort(key=lambda j: max(leaves[j]['p_upper'][name] for name in unresolved), reverse=True)
        selected = candidates[:min(remaining, batch_size//2)]
        if not selected:
            break
        children = []
        for j in selected:
            leaf = leaves[j]
            children.extend([(leaf['lower'], leaf['center']), (leaf['center'], leaf['upper'])])
        centers = [(a+b)/2 for a,b in children]
        if any(c in cache for c in centers):
            break  # floating-point interval resolution exhausted: keep bound
        evaluate(centers)
        leaves = [leaf for j,leaf in enumerate(leaves) if j not in selected]
        leaves.extend(interval(a,b) for a,b in children)
    tests = {}
    for name in names:
        best = max(cache, key=lambda x: cache[x]['records'][name]['p'])
        lo = cache[best]['records'][name]['p']
        hi = max(lo, max(leaf['p_upper'][name] for leaf in leaves))
        status = ('unresolved_covariance' if not cache[best]['records'][name]['resolved'] else
                  'rejected_specified_scale_family' if hi < alpha else
                  'compatible_scale_found' if lo >= alpha else 'unresolved_search_bound')
        tests[name] = dict(p_sup_lower=lo, p_upper=hi, status=status,
            sampled_maximizer=best, df=len(indices[name]), threshold=alpha)
    support = []
    normalized = geometry.r/np.sqrt(np.diag(geometry.h))
    effective = float(np.min(1/np.sum(normalized**4,axis=0)))
    for failed, reason in [(len(y)<1000,'N below 1000'),
            ((geometry.u.shape[1]+geometry.rank-geometry.saturated.sum())/geometry.active.sum()>.05,
             'fitted rank exceeds 5% of N'),
            (geometry.leverage[geometry.active].max()>.1,'maximum leverage exceeds .1'),
            (effective<100,'feature effective support below 100'),
            (geometry.condition>1e6,'normalized information condition exceeds 1e6')]:
        if failed:
            support.append(reason)
    return dict(method='continuous_boxcox_HC3_union_envelope_v1', bounds=[lower,upper],
        log_reference=log_reference, tests=tests, evaluations=len(cache),
        derivative_order=order, max_evaluations=max_evaluations,
        points=[dict(power=power,p={name:v['records'][name]['p'] for name in names},
                     covariance_resolved={name:v['records'][name]['resolved'] for name in names})
            for power,v in sorted(cache.items())], intervals=sorted(leaves,key=lambda v:v['lower']),
        diagnostics=dict(n=len(y),fixed_rank=geometry.u.shape[1],feature_rank=geometry.rank,
            max_leverage=float(geometry.leverage.max()),minimum_feature_effective_support=effective,
            information_condition=geometry.condition,outside_confirmation_design=support),
        inference='Asymptotic HC3 pointwise inference; continuous union-null envelope with float64 numerical slack.',
        null='Some common power in the declared interval makes the tested finite-projection coefficients zero.',
        scope='Declared Box-Cox family and supplied finite feature span only; neither arbitrary monotone invariance '
              'nor biological causality. A compatible scale is a failure to reject, not proof of latent additivity. '
              'Support flags remain applicable; these are not interval-arithmetic roundoff certificates.')
