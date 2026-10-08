"""Reference-probe error, separate from phenotype sampling and pair sketches."""
import numpy as np


def uniform_reference_bound(
    matrix, *, genetic_count, probes, failure_probability=0.01, look=None
):
    """Hutchinson entry bounds followed by a simultaneous inverse perturbation bound.

    Every selected-component T_ab probe is a PSD quadratic form on the common
    variant Rademacher vector (the two directions are symmetrized). Theorem 1
    of Roosta-Khorasani & Ascher, arXiv:1308.2475, and a union bound give
    epsilon=sqrt(6 log(2*entries/delta)/B). For epsilon<1, absolute entry
    errors are at most epsilon/(1-epsilon) times their estimated entries.
    Residual cross-traces here are exact. The bound may be very conservative.
    An optional numbered look spends delta/[look*(look+1)] for repeated
    prespecified precision checks; the union budget is then at most delta.
    """
    t = np.asarray(matrix, dtype=float)
    c = genetic_count
    if (
        t.ndim != 2
        or t.shape[0] != t.shape[1]
        or not 1 <= c <= len(t)
        or probes < 1
        or not 0 < failure_probability < 1
    ):
        raise ValueError("invalid reference precision inputs")
    delta = failure_probability
    if look is not None:
        if type(look) is not int or look < 1:
            raise ValueError("precision look must be a positive integer")
        delta /= look * (look + 1)
    entries = c * (c + 1) // 2
    epsilon = float(np.sqrt(6 * np.log(2 * entries / delta) / probes))
    result = dict(
        method="hutchinson_uniform_inverse_perturbation_bound_v1",
        failure_probability=delta,
        total_failure_budget=failure_probability,
        look=look,
        entry_relative_bound=epsilon,
        assumptions="independent Rademacher columns; selected PSD kernels; exact residual cross-traces",
        informative=False,
    )
    if epsilon >= 1:
        return dict(result, reason="probe count gives a vacuous relative entry bound")
    try:
        l = np.linalg.cholesky(t)
    except np.linalg.LinAlgError:
        return dict(result, reason="reference Gram is not positive definite")
    if np.any(t[:c, :c] < -1e-12):
        raise ValueError("selected PSD cross-traces must be nonnegative")
    envelope = np.zeros_like(t)
    envelope[:c, :c] = epsilon / (1 - epsilon) * np.maximum(t[:c, :c], 0)
    inverse = np.abs(np.linalg.solve(l, np.eye(len(t))))
    rho = float(np.linalg.norm(inverse @ envelope @ inverse.T, 2))
    result.update(normal_energy_error_bound=rho, informative=rho < 1)
    if rho < 1:
        result["relative_coefficient_energy_bound"] = rho / (1 - rho)
        result[
            "interpretation"
        ] = "simultaneous for every phenotype q in the estimated-T energy norm; not a phenotype confidence interval"
    return result


def probe_diagnostics(summary, coefficients, *, failure_probability=0.01):
    matrices = summary.probe_matrices
    if matrices is None:
        return dict(status="not_exported", included_in_fame_se=False)
    b = len(matrices)
    if b < 2:
        return dict(status="insufficient_probes", included_in_fame_se=False)
    t = summary.matrix
    inverse = np.linalg.inv(t)
    centered = matrices - t
    influence = -np.einsum("bij,j->bi", centered, coefficients) @ inverse.T
    covariance = influence.T @ influence / (b * (b - 1))
    bias = -np.einsum("bij,bj->i", centered, influence) @ inverse.T / (b * (b - 1))
    result = dict(
        status="first_order_probe_calculation",
        covariance=covariance,
        standard_errors=np.sqrt(np.diag(covariance)),
        leading_order_bias=bias,
        probes=b,
        included_in_fame_se=False,
        interpretation="delta calculation conditional on fixed phenotype; inverse fitting is nonlinear, so this is an approximation, not a calibrated tail correction",
    )
    ref = summary.metadata.get("reference", {})
    if ref.get("probe_spec", {}).get("distribution") == "rademacher":
        result["uniform_precision_bound"] = uniform_reference_bound(
            t,
            genetic_count=summary.metadata["genetic_count"],
            probes=b,
            failure_probability=failure_probability,
        )
    return result
