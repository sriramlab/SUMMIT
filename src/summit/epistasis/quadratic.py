"""Central Gaussian quadratic forms, including indefinite nuisance contrasts.

Characteristic-function inversion with an explicit absolute truncation bound.
This is an independent Imhof/Gil-Pelaez implementation, not Davies code.
Quadrature error is an adaptive numerical estimate; the omitted tail bound is
analytic. Unresolved accuracy raises an error, never a silent approximation.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq
from scipy.stats import chi2, f as f_dist


def quadratic_sf(
    value, eigenvalues, *, multiplicities=None, atol=1e-9, max_intervals=2048
):
    """P(sum lambda_i Z_i² >= value), independent standard Gaussian Z.

    The tail envelope is |phi(t)| <= prod(2 |lambda_i| t)^(-1/2)
    for any chosen nonzero subset. Integrating |phi(t)|/t from T yields
    2 prod(2|lambda_i|)^(-1/2) T^(-k/2)/k. The best subset bound is used.
    All nonzero eigenvalues remain in the actual characteristic function.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    if lam.ndim != 1 or not np.all(np.isfinite(lam)) or not np.isfinite(value):
        raise ValueError("quadratic form requires finite scalar and eigenvalue vector")
    if not 0 < atol < 0.01:
        raise ValueError("absolute tail tolerance must lie in (0, .01)")
    df = (
        np.ones(len(lam))
        if multiplicities is None
        else np.asarray(multiplicities, dtype=float)
    )
    if df.shape != lam.shape or np.any(df <= 0) or not np.all(np.isfinite(df)):
        raise ValueError("multiplicities must be positive finite degrees of freedom")
    df, lam = df[lam != 0], lam[lam != 0]
    if not len(lam):
        return dict(
            p=float(value <= 0), absolute_error=0.0, method="point_mass", evaluations=0
        )
    scale = np.max(abs(lam))
    lam, q = lam / scale, float(value / scale)
    if np.all(lam > 0) and q <= 0:
        return dict(p=1.0, absolute_error=0.0, method="support", evaluations=0)
    if np.all(lam < 0) and q >= 0:
        return dict(p=0.0, absolute_error=0.0, method="support", evaluations=0)
    if np.all(lam == lam[0]):
        p = (
            chi2.sf(q / lam[0], df.sum())
            if lam[0] > 0
            else chi2.cdf(q / lam[0], df.sum())
        )
        return dict(
            p=float(p), absolute_error=0.0, method="scaled_chi_square", evaluations=0
        )
    if len(lam) == 2 and q == 0 and lam[0] * lam[1] < 0:
        positive = int(lam[1] > 0)
        negative = 1 - positive
        threshold = -lam[negative] * df[negative] / (lam[positive] * df[positive])
        return dict(
            p=float(f_dist.sf(threshold, df[positive], df[negative])),
            absolute_error=0.0,
            method="exact_chi_square_ratio_F",
            evaluations=0,
        )
    if np.all(lam > 0) and lam.max() / lam.min() <= 200:
        # Positive gamma-mixture expansion. If b=min(lambda), Laplace
        # factorization gives a mixture of b*chi2_(sum(df)+2k), with positive
        # coefficients. The remaining coefficient mass bounds absolute tail
        # error. This avoids the enormous oscillatory range for rank 3-4.
        # Numerical fallback below changes integration only, never the test.
        b = float(lam.min())
        ratio = 1 - b / lam
        alpha = df / 2
        coefficient = np.zeros(8193)
        recurrence = np.zeros(8193)
        coefficient[0] = np.exp(np.dot(alpha, np.log(b / lam)))
        mass = coefficient[0]
        powers = np.ones(len(lam))
        terms = 0
        while 1 - mass > atol / 4 and terms < 8192:
            terms += 1
            powers *= ratio
            recurrence[terms] = np.dot(alpha, powers)
            coefficient[terms] = (
                np.dot(recurrence[1 : terms + 1], coefficient[terms - 1 :: -1]) / terms
            )
            mass += coefficient[terms]
        if 1 - mass <= atol / 4:
            probability = float(
                coefficient[: terms + 1]
                @ chi2.sf(q / b, df.sum() + 2 * np.arange(terms + 1))
            )
            error = float(max(0.0, 1 - mass) + 8 * np.finfo(float).eps * (terms + 1))
            if error <= atol and -error <= probability <= 1 + error:
                return dict(
                    p=float(np.clip(probability, 0, 1)),
                    absolute_error=error,
                    truncation_bound=float(max(0.0, 1 - mass)),
                    method="positive_gamma_mixture",
                    evaluations=terms + 1,
                )
    if len(lam) == 2 and np.array_equal(df, [1.0, 1.0]):
        # Polar coordinates: radius² ~ chi²_2, angle uniform. This avoids the
        # slowly decaying characteristic-function bound for a rank-two form.
        def angular(angle):
            a = lam[0] * np.cos(angle) ** 2 + lam[1] * np.sin(angle) ** 2
            if a > 0:
                return 1.0 if q <= 0 else np.exp(-q / (2 * a))
            if a < 0:
                return 0.0 if q >= 0 else -np.expm1(-q / (2 * a))
            return float(q <= 0)

        points = None
        if lam[0] * lam[1] < 0:
            points = [float(np.arccos(np.sqrt(-lam[1] / (lam[0] - lam[1]))))]
        value, error = quad(
            angular,
            0,
            np.pi / 2,
            epsabs=atol * np.pi / 4,
            epsrel=0,
            points=points,
            limit=max_intervals,
        )
        if error * 2 / np.pi > atol:
            raise ArithmeticError("rank-two angular integration unresolved")
        return dict(
            p=float(value * 2 / np.pi),
            absolute_error=float(error * 2 / np.pi),
            method="rank_two_polar_integration",
            evaluations=None,
        )
    order = np.argsort(abs(lam))[::-1]
    k = np.cumsum(df[order])
    logs = np.cumsum(df[order] * np.log(2 * abs(lam[order])))
    log_bound = np.log(2 / (np.pi * k)) - logs / 2
    # Bound each subset separately; the tightest bound certifies the tail.
    log_t = np.min(2 * (log_bound - np.log(atol / 4)) / k)
    truncation = max(1.0, float(np.exp(min(log_t, 700))))
    if not np.isfinite(truncation) or truncation > 1e7:
        raise ArithmeticError(
            "quadratic-form tail bound needs excessive integration range"
        )
    bound = float(np.exp(np.min(log_bound - k * np.log(truncation) / 2)))
    evaluations = 0

    def integrand(t):
        nonlocal evaluations
        evaluations += 1
        if t == 0:
            return float(lam @ df - q)
        v = 2 * t * lam
        log_amplitude = -0.25 * np.dot(df, np.log1p(v * v))
        phase = 0.5 * np.dot(df, np.arctan(v)) - q * t
        return float(np.exp(log_amplitude) * np.sin(phase) / t)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        integral, error = quad(
            integrand,
            0.0,
            truncation,
            epsabs=np.pi * atol / 4,
            epsrel=0.0,
            limit=max_intervals,
        )
    error = error / np.pi + bound
    if caught or error > atol:
        raise ArithmeticError(
            f"quadratic-form integration unresolved: error estimate {error:g}"
        )
    p = 0.5 + integral / np.pi
    if p < -error or p > 1 + error:
        raise ArithmeticError(
            "quadratic-form probability is outside its numerical bounds"
        )
    return dict(
        p=float(np.clip(p, 0, 1)),
        absolute_error=float(error),
        truncation_bound=bound,
        method="characteristic_function_inversion",
        evaluations=evaluations,
        integration_limit=truncation,
    )


def quadratic_quantile(probability, eigenvalues, *, atol=1e-9):
    if not 0 < probability < 1:
        raise ValueError("probability must lie in (0,1)")
    lam = np.asarray(eigenvalues, dtype=float)
    mean, sd = lam.sum(), np.sqrt(2 * np.dot(lam, lam))
    if sd == 0:
        return 0.0
    lo, hi = mean - 8 * sd, mean + 8 * sd
    for _ in range(30):
        if quadratic_sf(lo, lam, atol=atol)["p"] >= 1 - probability:
            break
        lo -= 2 * (hi - lo)
    for _ in range(30):
        if quadratic_sf(hi, lam, atol=atol)["p"] <= 1 - probability:
            break
        hi += 2 * (hi - lo)
    return float(
        brentq(
            lambda x: quadratic_sf(x, lam, atol=atol)["p"] - (1 - probability), lo, hi
        )
    )


def moment_spectrum(kernels, normal_matrix, covariance, component):
    """Bounded oracle: theta_hat_a=y' A_a y, A_a=sum_b T^-1_ab K_b."""
    kernels = np.asarray(kernels, dtype=float)
    if (
        kernels.ndim != 3
        or kernels.shape[1] != kernels.shape[2]
        or kernels.shape[1] > 2048
    ):
        raise ValueError("moment spectrum is a bounded square-kernel reference")
    row = np.linalg.solve(
        np.asarray(normal_matrix).T, np.eye(len(kernels))[:, component]
    )
    a = np.einsum("b,bij->ij", row, kernels)
    v = np.asarray(covariance, dtype=float)
    eig, vectors = np.linalg.eigh((v + v.T) / 2)
    if eig[0] < -1e-10 * max(eig[-1], 1):
        raise ValueError("invalid Gaussian covariance")
    factor = vectors * np.sqrt(np.maximum(eig, 0))
    spectrum = np.linalg.eigvalsh(factor.T @ a @ factor)
    return dict(
        contrast=a,
        eigenvalues=spectrum,
        mean=float(spectrum.sum()),
        variance=float(2 * spectrum @ spectrum),
        skewness=float(8 * np.sum(spectrum**3) / (2 * spectrum @ spectrum) ** 1.5),
    )
