"""Bivariate liability simulations with explicitly preserved marginal sampling."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from validate_pcgc import ar_parameters, draw_markers
from summit.pcgc.cross import QuantitativeResponse
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk


def fitted_or_failure(y, prevalence, covariates):
    try:
        return fit_binary_risk(y, prevalence, covariates)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return "risk fit: "+str(exc)


def joint_cells(threshold, gamma, correlation):
    def cdf(t):
        return quad(lambda x: norm.pdf(x)*norm.cdf((t-correlation*x)/np.sqrt(1-correlation**2)),
                    -np.inf, t, epsabs=2e-12)[0]
    p00 = .5*(cdf(threshold-gamma)+cdf(threshold+gamma))
    K = .5*(norm.sf(threshold-gamma)+norm.sf(threshold+gamma))
    return np.array([p00, 1-K-p00, 1-K-p00, 2*K-1+p00])


def generate(seed, scenario, n=4000, m=4000, block_size=40):
    if scenario not in ("BB0", "BB_shared", "BQ") or n % 4:
        raise ValueError("use BB0, BB_shared or BQ with a sample count divisible by four")
    rng = np.random.default_rng(seed)
    rho, R = ar_parameters(m, block_size)
    beta1 = rng.normal(size=m)
    beta1 *= np.sqrt(.25/(beta1 @ R @ beta1))
    beta2 = rng.normal(size=m)
    beta2 -= beta1*(beta1 @ R @ beta2)/.25
    beta2 *= np.sqrt(.25/(beta2 @ R @ beta2))
    rg = 0 if scenario == "BB0" else .5
    beta2 = rg*beta1+np.sqrt(1-rg*rg)*beta2
    gamma, K = .5, .1
    threshold = brentq(lambda t: .5*(norm.sf(t-gamma)+norm.sf(t+gamma))-K, -6, 6)
    cells = joint_cells(threshold, gamma, beta1 @ R @ beta2)
    shared = n//4 if scenario == "BB_shared" else 0
    # Each label owns a distinct participant pool except the explicit shared
    # controls. Private controls compensate for joint eligibility so that
    # each cohort still has its proper marginal control distribution.
    requests = []
    if shared:
        requests.append(("shared", 0, shared))
    for side in (0, 1):
        ncontrol, ncase = n//2, n//2
        control_other = 1 if side == 0 else 2
        private = ncontrol-shared
        p00 = (ncontrol*cells[0]/(1-K)-shared)/private
        counts = rng.multinomial(private, [p00, 1-p00])
        requests.extend([(str(side), 0, counts[0]), (str(side), control_other, counts[1])])
        own_only = 2 if side == 0 else 1
        counts = rng.multinomial(ncase, [cells[own_only]/K, cells[3]/K])
        requests.extend([(str(side), own_only, counts[0]), (str(side), 3, counts[1])])
    if scenario == "BQ":
        requests = [r for r in requests if r[0] == "0"]
    remaining = np.array([r[2] for r in requests])
    collected = [[] for _ in requests]
    diagnostics = np.zeros(4, dtype=int)
    while remaining.sum():
        x = draw_markers(rng, 2048, rho)
        c = rng.choice([-1., 1.], len(x))
        latent = np.column_stack([x @ beta1, x @ beta2])+gamma*c[:, None]+rng.normal(size=(len(x), 2))*np.sqrt(.75)
        y = latent > threshold
        cell = 2*y[:, 0]+y[:, 1]
        diagnostics += np.bincount(cell, minlength=4)
        for value in range(4):
            candidates = np.flatnonzero(cell == value)
            offset = 0
            for r, (_, needed_cell, _) in enumerate(requests):
                if needed_cell != value or not remaining[r]:
                    continue
                take = candidates[offset:offset+remaining[r]]
                offset += len(take)
                if len(take):
                    collected[r].append((x[take], c[take], y[take]))
                    remaining[r] -= len(take)
    groups, genotypes, covariates, outcomes = [], [], [], []
    for request, chunks in zip(requests, collected):
        for x, c, y in chunks:
            groups.extend([request[0]]*len(x))
            genotypes.append(x)
            covariates.append(c)
            outcomes.append(y)
    x, c, y = np.vstack(genotypes), np.concatenate(covariates), np.vstack(outcomes)
    groups = np.asarray(groups)
    rows1 = np.flatnonzero(np.isin(groups, ["0", "shared"]))
    rows2 = np.flatnonzero(np.isin(groups, ["1", "shared"]))
    if scenario == "BQ":
        x2 = draw_markers(rng, n, rho)
        c2 = rng.choice([-1., 1.], n)
        z2 = x2 @ beta2+rng.normal(size=n)*np.sqrt(.75)
        rows2 = np.arange(len(x), len(x)+n)
        x = np.vstack([x, x2])
        right = QuantitativeResponse(z2, gamma**2)
        fitted_right = right
    else:
        right = prepare_binary_risk(y[rows2, 1], K, population_risk=norm.sf(threshold-gamma*c[rows2]), covariate_variance=gamma**2)
        fitted_right = fitted_or_failure(y[rows2, 1], K, c[rows2, None])
    left = prepare_binary_risk(y[rows1, 0], K, population_risk=norm.sf(threshold-gamma*c[rows1]), covariate_variance=gamma**2)
    fitted_left = fitted_or_failure(y[rows1, 0], K, c[rows1, None])
    return dict(x=x, left=left, right=right, fitted_left=fitted_left, fitted_right=fitted_right,
                left_rows=rows1, right_rows=rows2, truth_covariance=float(beta1 @ R @ beta2), truth_rg=rg,
                marginal_covariance=float(beta1 @ R @ beta2/(1+gamma**2)),
                diagnostics=dict(population_cell_probabilities=cells.tolist(), observed_population_cells=diagnostics.tolist(),
                                 shared_controls=shared, selection="private controls compensate joint eligibility"))
