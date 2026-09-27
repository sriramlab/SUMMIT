"""Ascertainment-aware risk preparation on a fixed residual-liability scale.

These quantities are not OLS residuals or logistic GWAS z statistics. Sampling
must depend on case status alone; population prevalence is an external input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit, log_ndtr, ndtri


def finite_array(name, value, ndim=None):
    array = np.asarray(value, dtype=np.float64)
    if (ndim is not None and array.ndim != ndim) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim or ''}-dimensional array")
    return array


def readonly(value):
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def probability(name, value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def binary_phenotype(value):
    y = finite_array("phenotype", value, 1)
    if y.size < 3 or not np.all((y == 0) | (y == 1)) or np.unique(y).size != 2:
        raise ValueError("binary phenotype must contain both 0 and 1 and at least three samples")
    return y


def sampled_log_probabilities(eta, log_sampling_ratio):
    """Stable log probabilities and derivative of sampled log odds."""
    lk, l0 = log_ndtr(eta), log_ndtr(-eta)
    odds = lk - l0 - log_sampling_ratio
    lp, lq = -np.logaddexp(0.0, -odds), -np.logaddexp(0.0, odds)
    log_phi = -0.5 * np.asarray(eta)**2 - 0.5 * np.log(2 * np.pi)
    slope = np.exp(log_phi - lk - l0)
    return lp, lq, slope


def risk_objective(coefficients, design, y, log_sampling_ratio):
    lp, lq, slope = sampled_log_probabilities(design @ coefficients, log_sampling_ratio)
    loss = -np.mean(y * lp + (1 - y) * lq)
    gradient = design.T @ ((np.exp(lp) - y) * slope) / y.size
    return float(loss), gradient


@dataclass(frozen=True)
class BinaryRisk:
    population_prevalence: float
    sample_prevalence: float
    population_risk: np.ndarray
    sample_risk: np.ndarray
    z: np.ndarray
    sensitivity: np.ndarray
    covariate_variance: float
    source: str
    coefficients: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "population_prevalence", probability("population prevalence", self.population_prevalence))
        object.__setattr__(self, "sample_prevalence", probability("sample prevalence", self.sample_prevalence))
        n = np.asarray(self.z).size
        for name in ("population_risk", "sample_risk", "z", "sensitivity"):
            value = finite_array(name, getattr(self, name), 1)
            if value.size != n or n < 3:
                raise ValueError("risk vectors must share a nonempty sample axis")
            if name.endswith("risk") and np.any((value <= 0) | (value >= 1)):
                raise ValueError("risk probabilities must lie strictly between zero and one")
            if name == "sensitivity" and np.any(value <= 0):
                raise ValueError("risk sensitivity must be positive")
            object.__setattr__(self, name, readonly(value))
        if not np.isfinite(self.covariate_variance) or self.covariate_variance < 0:
            raise ValueError("covariate variance must be finite and nonnegative")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("risk source must be a nonempty label")
        K, P = self.population_prevalence, self.sample_prevalence
        log_a = np.log(K) + np.log1p(-P) - np.log(P) - np.log1p(-K)
        lp, lq, slope = sampled_log_probabilities(ndtri(self.population_risk), log_a)
        if not np.allclose(self.sample_risk, np.exp(lp), rtol=2e-13, atol=0):
            raise ValueError("sample risks disagree with prevalence and ascertainment")
        if not np.allclose(self.sensitivity, slope*np.exp(.5*(lp+lq)), rtol=2e-13, atol=0):
            raise ValueError("sensitivity disagrees with the risk model")
        phenotype = np.where(self.z > 0, 1., 0.)
        expected_z = np.where(phenotype == 1, np.exp(.5*(lq-lp)), -np.exp(.5*(lp-lq)))
        if not np.allclose(self.z, expected_z, rtol=2e-13, atol=0) or phenotype.mean() != P:
            raise ValueError("standardized response disagrees with binary ascertainment")
        object.__setattr__(self, "coefficients", readonly(finite_array("coefficients", self.coefficients, 1)))

    @property
    def n_samples(self):
        return self.z.size

    def diagnostics(self):
        return {
            "source": self.source,
            "population_prevalence": self.population_prevalence,
            "sample_prevalence": self.sample_prevalence,
            "covariate_variance": self.covariate_variance,
            "population_risk_range": [float(self.population_risk.min()), float(self.population_risk.max())],
            "sensitivity_range": [float(self.sensitivity.min()), float(self.sensitivity.max())],
            "liability_residual_variance": 1.0,
        }


def prepare_binary_risk(y, prevalence, *, population_risk=None,
                        covariate_variance=None, source="supplied", coefficients=()):
    y = binary_phenotype(y)
    K = probability("population prevalence", prevalence)
    P = float(y.mean())
    log_a = np.log(K) + np.log1p(-P) - np.log(P) - np.log1p(-K)
    k = np.full(y.size, K) if population_risk is None else finite_array("population risk", population_risk, 1)
    if k.shape != y.shape or np.any((k <= 0) | (k >= 1)):
        raise ValueError("population risks must match the sample axis and lie in (0,1)")
    eta = ndtri(k)
    lp, lq, slope = sampled_log_probabilities(eta, log_a)
    p = np.exp(lp)
    z = np.where(y == 1, np.exp(0.5 * (lq - lp)), -np.exp(0.5 * (lp - lq)))
    d = slope * np.exp(0.5 * (lp + lq))
    if covariate_variance is None:
        population_weights = np.where(y == 1, K / P, (1 - K) / (1 - P))
        eta_mean = np.average(eta, weights=population_weights)
        covariate_variance = float(np.average((eta - eta_mean)**2, weights=population_weights))
    return BinaryRisk(K, P, k, p, z, d, float(covariate_variance), source, np.asarray(coefficients))


def _risk_design(y, covariates):
    cov = finite_array("covariates", covariates, 2)
    if cov.shape[0] != y.size:
        raise ValueError("covariates must match the phenotype sample axis")
    if cov.shape[1] == 0:
        return cov, np.ones(0), np.ones((len(y), 1))
    scale = cov.std(axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("covariates must exclude constant columns and the intercept")
    design = np.column_stack((np.ones(y.size), (cov - cov.mean(axis=0)) / scale))
    if design.shape[1] >= y.size or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("risk design is rank deficient")
    # A finite unpenalized MLE does not exist under complete or quasi separation.
    # A small gradient alone cannot diagnose this: it also tends to zero as a
    # separating coefficient diverges. The bounded LP tests both cases.
    signed_design = (2 * y - 1)[:, None] * design
    separation = linprog(-signed_design.mean(axis=0), A_ub=-signed_design,
                         b_ub=np.zeros(y.size), bounds=[(-1, 1)] * design.shape[1],
                         method="highs")
    if not separation.success:
        raise RuntimeError(f"risk separation check failed: {separation.message}")
    if -separation.fun > 1e-8:
        raise ValueError("risk fit is separated; an unpenalized finite MLE does not exist")
    return cov, scale, design


def fit_binary_risk(y, prevalence, covariates=None):
    """Fit a probit population risk through the ascertained Bernoulli likelihood.

    Covariate input excludes the intercept. Covariate variance is estimated by
    inverse ascertainment weighting and presumes the stated liability model.
    No SNP predictors, implicit regularization or probability clipping are used.
    """
    y = binary_phenotype(y)
    K = probability("population prevalence", prevalence)
    if covariates is None:
        return prepare_binary_risk(y, K, covariate_variance=0.0, source="intercept_only")
    cov, scale, design = _risk_design(y, covariates)
    if cov.shape[1] == 0:
        return fit_binary_risk(y, K)
    P = float(y.mean())
    log_a = np.log(K) + np.log1p(-P) - np.log(P) - np.log1p(-K)
    start = np.r_[ndtri(K), np.zeros(cov.shape[1])]
    result = minimize(risk_objective, start, args=(design, y, log_a), jac=True,
                      method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    _, gradient = risk_objective(result.x, design, y, log_a)
    if not np.all(np.isfinite(result.x)) or np.max(np.abs(gradient)) > 1e-7:
        raise RuntimeError(f"ascertainment-aware risk fit failed: {result.message}")
    eta = design @ result.x
    lp, lq, slope = sampled_log_probabilities(eta, log_a)
    information = design.T @ ((np.exp(lp + lq) * slope**2)[:, None] * design)
    if np.linalg.cond(information) > 1e10 or np.max(np.abs(result.x)) > 30:
        raise ValueError("risk fit is separated or ill-conditioned")
    k = np.exp(log_ndtr(eta))
    return prepare_binary_risk(y, K, population_risk=k, source="ascertained_probit",
                               coefficients=np.r_[result.x[0] - (cov.mean(axis=0)/scale) @ result.x[1:], result.x[1:]/scale])


def fit_binary_risk_logistic(y, prevalence, covariates=None):
    """Research compatibility comparison: sampled logistic MLE, then back-transform.

    This generally misspecifies a population-probit generator with continuous
    covariates. Its coefficients are sample log odds, not liability effects.
    """
    y = binary_phenotype(y)
    K = probability("population prevalence", prevalence)
    if covariates is None:
        return fit_binary_risk(y, K)
    cov, scale, design = _risk_design(y, covariates)
    if cov.shape[1] == 0:
        return fit_binary_risk(y, K)
    P = float(y.mean())
    log_a = np.log(K)+np.log1p(-P)-np.log(P)-np.log1p(-K)

    def objective(coef):
        eta = design @ coef
        loss = np.mean(y*np.logaddexp(0., -eta)+(1-y)*np.logaddexp(0., eta))
        return float(loss), design.T @ (expit(eta)-y)/len(y)

    start = np.r_[np.log(P)-np.log1p(-P), np.zeros(cov.shape[1])]
    result = minimize(objective, start, jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    if not np.all(np.isfinite(result.x)) or np.max(np.abs(objective(result.x)[1])) > 1e-7:
        raise RuntimeError(f"sample-logistic risk fit failed: {result.message}")
    eta = design @ result.x
    variance = np.exp(-np.logaddexp(0., -eta)-np.logaddexp(0., eta))
    if np.linalg.cond(design.T @ (variance[:, None]*design)) > 1e10 or np.max(np.abs(result.x)) > 30:
        raise ValueError("sample-logistic risk fit is ill-conditioned")
    return prepare_binary_risk(y, K, population_risk=expit(eta+log_a), source="sample_logistic_backtransform",
        coefficients=np.r_[result.x[0]-(cov.mean(axis=0)/scale) @ result.x[1:], result.x[1:]/scale])
