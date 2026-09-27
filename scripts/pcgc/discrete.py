"""Independent diploid Markov-haplotype generator for PCGC qualification.

Allele counts have exact population covariance rho**distance within each block.
The liability CDF is obtained by characteristic-function inversion, so the
prevalence threshold does not rely on fitting a Gaussian to generated scores.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_legendre


def marker_calls(rng, n, rho, frequency=.3):
    calls = np.zeros((n, len(rho)), dtype=np.int8)
    for _ in range(2):
        previous = np.zeros(n, dtype=bool)
        for j, r in enumerate(rho):
            new = rng.random(n) < frequency
            if r:
                previous = np.where(rng.random(n) < r, previous, new)
            else:
                previous = new
            calls[:, j] += previous
    return calls


class LiabilityDistribution:
    def __init__(self, beta, rho, residual_variance, *, frequency=.3, nodes=384):
        if not 0 < residual_variance <= 1:
            raise ValueError("liability inversion requires positive Gaussian residual variance")
        p = frequency
        b = beta / np.sqrt(2*p*(1-p))
        points, weights = roots_legendre(nodes)
        upper = 12/np.sqrt(residual_variance)
        self.u = (points+1)*upper/2
        self.weights = weights*upper/2/(np.pi*self.u)
        characteristic = np.ones(nodes, dtype=complex)
        starts = np.flatnonzero(rho == 0)
        for start, end in zip(starts, np.r_[starts[1:], len(rho)]):
            s0 = np.full(nodes, 1-p, dtype=complex)
            s1 = p*np.exp(1j*self.u*b[start])
            for j in range(start+1, end):
                r = rho[j]
                t0 = s0*(1-(1-r)*p) + s1*(1-r)*(1-p)
                t1 = (s0*(1-r)*p + s1*(1-(1-r)*(1-p)))*np.exp(1j*self.u*b[j])
                s0, s1 = t0, t1
            characteristic *= (s0+s1)**2
        self.phi = characteristic * np.exp(-2j*p*self.u*b.sum() - .5*residual_variance*self.u**2)

    def survival(self, cut):
        cut = np.asarray(cut)
        # Integration error is checked against independent rejection sampling.
        result = .5 + np.sum(self.weights * np.imag(np.exp(-1j*cut[..., None]*self.u)*self.phi), axis=-1)
        if np.any((result < -1e-10) | (result > 1+1e-10)):
            raise ValueError("liability CDF integration failed")
        return result


def generate(seed, scenario, n, m, nref, block_size):
    from validate_pcgc import SCENARIOS, ar_parameters
    if scenario not in ("S2", "S5"):
        raise ValueError("discrete qualification currently supports S2 and S5")
    rng = np.random.default_rng(seed)
    spec = SCENARIOS[scenario]
    rho, R = ar_parameters(m, block_size)
    a = np.ones((m, len(spec["h"])))
    if a.shape[1] == 2:
        labels = np.arange(m)//block_size % 2
        a = np.column_stack([labels == j for j in range(2)]).astype(float)
    beta = rng.normal(size=m)
    for j, h in enumerate(spec["h"]):
        b = beta*a[:, j]
        beta[a[:, j] > 0] *= np.sqrt(h/(b @ R @ b))
    truth = np.array([(beta*col) @ R @ (beta*col) for col in a.T])
    gamma, p = .5, .3
    distribution = LiabilityDistribution(beta, rho, 1-truth.sum())
    threshold = brentq(lambda t: .5*(distribution.survival(t-gamma)+distribution.survival(t+gamma))-spec["K"], -6, 6, xtol=1e-12)
    risks = distribution.survival(threshold-gamma*np.array([-1., 1.]))
    target = np.array([n-int(round(n*spec["P"])), int(round(n*spec["P"]))])
    observed = np.zeros(2, dtype=int)
    sampled = np.zeros(2, dtype=int)
    genotypes, covariates, outcomes = [], [], []
    # Fixed quota sampling from independently generated population chunks.
    while np.any(sampled < target):
        calls = marker_calls(rng, 2048, rho, p)
        x = (calls-2*p)/np.sqrt(2*p*(1-p))
        cov = rng.choice([-1., 1.], len(x))
        liability = x @ beta + gamma*cov + rng.normal(size=len(x))*np.sqrt(1-truth.sum())
        y = (liability > threshold).astype(int)
        observed += np.bincount(y, minlength=2)
        for state in (0, 1):
            selected = np.flatnonzero(y == state)[:max(0, target[state]-sampled[state])]
            if selected.size:
                genotypes.append(x[selected])
                covariates.append(cov[selected])
                outcomes.append(y[selected])
                sampled[state] += selected.size
    order = rng.permutation(n)
    x = np.vstack(genotypes)[order]
    cov = np.concatenate(covariates)[order]
    y = np.concatenate(outcomes)[order]
    ref = (marker_calls(rng, nref, rho, p)-2*p)/np.sqrt(2*p*(1-p))
    return dict(x=x, y=y, cov=cov, k=risks[(cov > 0).astype(int)], Vc=gamma**2,
                annotations=a, reference=ref, truth=truth,
                random_effect_parameter=np.array([np.sum((beta*col)**2) for col in a.T]),
                beta=beta, rho=rho, R=R, threshold=threshold, gamma=gamma,
                generator_diagnostics=dict(population_draws=int(observed.sum()),
                    observed_prevalence=float(observed[1]/observed.sum()), exact_stratum_risks=risks.tolist(),
                    reference_standardized_mean=float(ref.mean()), reference_standardized_second_moment=float(np.mean(ref**2))))
