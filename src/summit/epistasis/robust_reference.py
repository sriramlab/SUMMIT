"""Bounded independent validators; never used to supply production variances."""
import numpy as np
from scipy.linalg import null_space
from .quadratic import quadratic_sf


def hc3_ratio_reference(feature, fixed, variance, threshold, *, atol=1e-9):
    """Actual random-denominator HC3 rejection under a specified Gaussian null.

    Dense, N<=1024. Known generating variances are validation inputs only.
    Independent SVD/null-space algebra checks the production QR implementation.
    """
    f = np.asarray(feature, float).reshape(-1)
    c = np.asarray(fixed, float)
    omega = np.asarray(variance, float)
    n = len(f)
    if (
        n > 1024
        or c.ndim != 2
        or len(c) != n
        or omega.shape != (n,)
        or np.any(omega <= 0)
    ):
        raise ValueError(
            "bounded ratio reference needs N<=1024 and positive diagonal variance"
        )
    if not all(np.isfinite(x).all() for x in (f, c, omega)) or threshold <= 0:
        raise ValueError("invalid ratio reference inputs")
    rc = null_space(c.T)
    r = rc @ (rc.T @ f)
    if r @ r <= 1e-20 * (f @ f):
        raise ValueError("absorbed scalar feature")
    a = r / (r @ r)
    residual = null_space(np.column_stack([c, f]).T)
    m = residual @ residual.T
    diagonal = np.diag(m)
    tol = 64 * n * np.finfo(float).eps
    saturated = (diagonal <= tol) & (abs(a) <= tol * np.linalg.norm(a))
    if np.any((diagonal <= 1e-8) & ~saturated):
        raise ValueError("essential unit leverage")
    denominator = diagonal.copy()
    denominator[saturated] = 1
    a[saturated] = 0
    b = (m * (a / denominator) ** 2) @ m
    contrast = np.outer(a, a) - threshold**2 * b
    contrast *= np.sqrt(omega[:, None] * omega[None, :])
    lam = np.linalg.eigvalsh((contrast + contrast.T) / 2)
    delta = 64 * n * np.finfo(float).eps * max(abs(lam))
    tail = quadratic_sf(0, lam, atol=atol)
    lower = quadratic_sf(0, lam - delta, atol=atol)
    upper = quadratic_sf(0, lam + delta, atol=atol)
    return dict(
        tail,
        probability_bracket=[
            max(0, lower["p"] - lower["absolute_error"]),
            min(1, upper["p"] + upper["absolute_error"]),
        ],
        spectral_roundoff_allowance=delta,
        max_leverage=float(1 - diagonal.min()),
        residual_rank=residual.shape[1],
        coefficient_influence=a,
        denominator_form=b,
        eigenvalues=lam,
    )


def many_covariate_reference(feature, y, fixed):
    """Independent bounded HC3, leave-out and Hadamard variance comparison.

    The last two are unbiased under a correct linear mean/independent errors
    when defined, but can be negative in finite samples. No clipping occurs.
    """
    f, y, c = np.asarray(feature).reshape(-1), np.asarray(y), np.asarray(fixed)
    n = len(f)
    if n > 1024:
        raise ValueError("many-covariate dense reference is bounded at N=1024")
    rc = null_space(c.T)
    r = rc @ (rc.T @ f)
    a = r / (r @ r)
    residual = null_space(np.column_stack([c, f]).T)
    m = residual @ residual.T
    d = np.diag(m)
    if d.min() <= 1e-8:
        raise ValueError("leave-out fit unidentified")
    e = m @ y
    if y.ndim == 1:
        e = e[:, None]
        y = y[:, None]
    hadamard = m * m
    kappa = np.linalg.cond(hadamard)
    identified = np.isfinite(kappa) and kappa < 1 / np.sqrt(np.finfo(float).eps)
    corrected = np.linalg.solve(hadamard, a * a) if identified else np.full(n, np.nan)
    return dict(
        beta=a @ y,
        HC3=(a * a / d**2) @ (e * e),
        leave_out=(a * a / d) @ (y * e),
        hadamard=corrected @ (e * e),
        hadamard_condition=float(kappa),
        hadamard_identified=identified,
        max_leverage=float(1 - d.min()),
    )
