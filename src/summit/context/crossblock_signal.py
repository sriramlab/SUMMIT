"""Between-block signal covariance of general G×E moments (quantitative traits).

The SNP-block jackknife treats the genetic moments of different variant
blocks as independent. They are not. Block b's genetic value enters the
phenotype, so the moment of block b contains products of block b's realized
genetic values with those of every other block b'. The product for a pair of
blocks appears in the moments of both blocks, and deleting block b removes
only the copy carried by block b's own kernel. Summed over blocks, these
products form the participants' realized covariance between each context
and their squared genetic value, which looks like G–G×E covariance
(amplification). The jackknife therefore underestimates the variance of the
G–G×E covariances; G×E variances and additive variance are barely affected.

For the genetic moments d_u = y' H_u y, H_qr = P (D_q A D_r + D_r A D_q) P,
the missed between-block covariance is, to leading order,

    C[(a,u),(c,v)] = 4 sum_{b,d} G[a,b,c,d] sum_i w_u^(b)(i) w_v^(d)(i),
    G[a,b,c,d]     = sum_{x != y} abar_x[a,b] abar_y[c,d],

with abar_x[a,b] block x's different-person additive trace between
annotations a and b divided by the study size,
w_qr^(a)(i) = phi_iq z_r^(a)(i) + phi_ir z_q^(a)(i) (q != r),
w_qq^(a)(i) = phi_iq z_q^(a)(i) and z^(a)(i) = phi_i' Omega^(a) Sigma_phi,
Sigma_phi = sum_i phi_i phi_i' / n. The corrected covariance of the
coefficients is V_jk + B/(B-1) T^-1 [C 0; 0 0] T^-1; its genetic block is
B/(B-1) S^-1 C S^-1 with S the residual-profiled genetic information.

Every input is a person-summed aggregate that a fit already has: the
reference's per-block intercept LD scalars (``intercept_ld_scalars``), the
annotation masses, the study's context second and fourth moments
(``context_second_moments``, ``context_fourth_moments``), the normal matrix
and the fitted coefficients. No genotype pass is needed. Derivation and exact
checks: docs/wiki/Methods.md#between-block-signal-covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np

from .spec import ContextComponentIndex

METHOD = "snp_block_jackknife_plus_crossblock_signal_v1"
JACKKNIFE_ONLY_METHOD = "snp_block_jackknife_v1"
STANDARD_ERROR_CHOICES = ("crossblock", "jackknife")
DEFAULT_STANDARD_ERRORS = "crossblock"


class CrossBlockSignalUnavailableWarning(UserWarning):
    """The between-block correction could not be formed; jackknife only."""


def resolve_standard_errors(value: object) -> str:
    """Validate a ``standard_errors`` choice (None selects the default)."""
    if value is None:
        return DEFAULT_STANDARD_ERRORS
    if not isinstance(value, str) or value not in STANDARD_ERROR_CHOICES:
        raise ValueError(f"standard_errors must be one of {STANDARD_ERROR_CHOICES}")
    return value


def _finite(name: str, value: object, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


@dataclass(frozen=True)
class CrossBlockSignalInputs:
    """Aggregates needed by the correction for one trait.

    ``block_traces[x, a, b]`` is abar_x[a, b]: block x's different-person
    additive (intercept-pair) trace between annotations a and b in normal-
    matrix units, divided by the study size. ``study_second_moments`` is
    ``sum_i phi_i phi_i'`` (Q x Q) and ``study_fourth_moments`` the saved
    ``P x P`` matrix ``sum_i u(i) u(i)'`` with ``u_p = kappa_p phi_q phi_r``,
    both over the trait's people.
    """

    block_traces: np.ndarray
    study_n: int
    study_second_moments: np.ndarray
    study_fourth_moments: np.ndarray

    def __post_init__(self) -> None:
        traces = _finite("block_traces", self.block_traces, 3)
        if traces.shape[0] < 2 or traces.shape[1] != traces.shape[2]:
            raise ValueError("block_traces must be blocks x K x K with at least two blocks")
        if isinstance(self.study_n, bool) or not isinstance(self.study_n, (int, np.integer)) or self.study_n < 2:
            raise ValueError("study_n must be an integer >= 2")
        second = _finite("study_second_moments", self.study_second_moments, 2)
        fourth = _finite("study_fourth_moments", self.study_fourth_moments, 2)
        q = second.shape[0]
        if second.shape != (q, q) or fourth.shape != (q * (q + 1) // 2,) * 2:
            raise ValueError("context moment shapes disagree with the basis size")
        for name, value in (("block_traces", 0.5 * (traces + np.swapaxes(traces, 1, 2))),
                            ("study_second_moments", 0.5 * (second + second.T)),
                            ("study_fourth_moments", 0.5 * (fourth + fourth.T))):
            value = np.array(value, dtype=np.float64)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "study_n", int(self.study_n))

    @property
    def block_count(self) -> int:
        return int(self.block_traces.shape[0])


def block_traces_from_ld_scalars(
    ld_scalars: object,
    annotation_masses: object,
    study_n: int,
    *,
    block_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert raw intercept LD scalars to abar (blocks x K x K).

    ``ld_scalars`` are :func:`summit.context.oracle.intercept_ld_scalars`
    (raw, per pair of distinct people). The study's different-person additive
    trace of block x is ``n_S (n_S - 1) ell_x / (M M')`` with the full
    annotation masses M, so abar_x = ``(n_S - 1) ell_x / (M M')``. With
    ``block_ids`` (one per row), rows sharing a jackknife block are summed.
    """
    scalars = _finite("ld_scalars", ld_scalars, 3)
    masses = _finite("annotation_masses", annotation_masses, 1)
    if scalars.shape[1:] != (masses.size, masses.size) or np.any(masses <= 0.0):
        raise ValueError("ld_scalars and annotation_masses disagree or masses are not positive")
    if block_ids is not None:
        ids = np.asarray(block_ids)
        if ids.shape != (scalars.shape[0],):
            raise ValueError("block_ids must give one block per LD-scalar row")
        labels, inverse = np.unique(ids, return_inverse=True)
        summed = np.zeros((len(labels),) + scalars.shape[1:])
        np.add.at(summed, inverse, scalars)
        scalars = summed
    return (float(study_n) - 1.0) * scalars / np.outer(masses, masses)[None]


def _w_maps(component_index: ContextComponentIndex, omegas: np.ndarray,
            sigma_phi: np.ndarray) -> np.ndarray:
    """L[a, p, u]: w_u^(a)(i) = sum_p u_p(i) L[a, p, u] for saved products u_p."""
    pairs = component_index.pair_index.entries
    q = component_index.pair_index.num_basis
    position = {}
    for pair in pairs:
        position[(pair.q, pair.r)] = position[(pair.r, pair.q)] = pair.index
    kappa = np.array([float(pair.kernel_factor) for pair in pairs])
    out = np.zeros((omegas.shape[0], len(pairs), len(pairs)))
    for a in range(omegas.shape[0]):
        m = omegas[a] @ sigma_phi                      # z_r(i) = sum_s phi_is m[s, r]
        for pair in pairs:
            terms = ((pair.q, pair.r),) if pair.q == pair.r else ((pair.q, pair.r), (pair.r, pair.q))
            for left, col in terms:                    # phi_left z_col
                for s in range(q):
                    out[a, position[(left, s)], pair.index] += m[s, col]
        out[a] /= kappa[:, None]
    return out


def crossblock_moment_covariance(
    component_index: ContextComponentIndex,
    omegas: object,
    inputs: CrossBlockSignalInputs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Between-block covariance C (c x c) of the genetic moments."""
    if not isinstance(component_index, ContextComponentIndex):
        raise ValueError("component_index must be a ContextComponentIndex")
    k = len(component_index.annotation_names)
    q = component_index.pair_index.num_basis
    p = len(component_index.pair_index)
    om = _finite("omegas", omegas, 3)
    if om.shape != (k, q, q) or inputs.block_traces.shape[1:] != (k, k):
        raise ValueError("omegas, block traces and the component index disagree")
    if inputs.study_second_moments.shape != (q, q):
        raise ValueError("study context moments disagree with the basis size")
    om = 0.5 * (om + np.swapaxes(om, 1, 2))
    sigma_phi = inputs.study_second_moments / float(inputs.study_n)
    traces = inputs.block_traces
    total = np.sum(traces, axis=0)
    g = np.einsum("ab,cd->abcd", total, total) - np.einsum("xab,xcd->abcd", traces, traces)
    maps = _w_maps(component_index, om, sigma_phi)
    gram = np.einsum("apu,pr,crv->acuv", maps, inputs.study_fourth_moments, maps)
    block = 4.0 * np.einsum("abcd,bduv->aucv", g, gram)        # (annotation, pair) x (annotation, pair)
    index = np.empty((k, p), dtype=np.int64)
    for entry in component_index.entries:
        index[entry.annotation_index, entry.pair_index] = entry.index
    order = index.reshape(-1)
    covariance = np.zeros((k * p, k * p))
    covariance[np.ix_(order, order)] = block.reshape(k * p, k * p)
    covariance = 0.5 * (covariance + covariance.T)
    shares = traces / np.where(total == 0.0, 1.0, total)[None]
    diagnostics = {
        "method": METHOD,
        "jackknife_blocks": inputs.block_count,
        "study_n": inputs.study_n,
        "additive_trace_total": total.tolist(),
        "one_minus_sum_block_share_squares": (1.0 - np.sum(shares**2, axis=0)).tolist(),
    }
    return covariance, diagnostics


def _genetic_positions(keep: np.ndarray, genetic_count: int) -> np.ndarray:
    return np.flatnonzero(keep < genetic_count)


def _keep_indices(keep: object | None, total: int) -> np.ndarray:
    if keep is None:
        return np.arange(total)
    indices = np.asarray(keep, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= total):
        raise ValueError("keep must be unique coefficient indices")
    return np.sort(indices)


def _model_omegas(component_index: ContextComponentIndex, coefficients: np.ndarray,
                  keep: np.ndarray) -> np.ndarray:
    from .oracle import coefficients_to_omegas
    c = len(component_index)
    theta = np.zeros(c)
    genetic = keep[keep < c]
    theta[genetic] = coefficients[genetic]
    return coefficients_to_omegas(theta, component_index)


def _inverse_columns(matrix: np.ndarray, keep: np.ndarray, genetic_count: int) -> tuple[np.ndarray, np.ndarray]:
    sub = matrix[np.ix_(keep, keep)]
    positions = _genetic_positions(keep, genetic_count)
    unit = np.zeros((len(keep), len(positions)))
    unit[positions, np.arange(len(positions))] = 1.0
    return np.linalg.solve(sub, unit), keep[positions]


def added_covariance_from_moments(
    normal_matrix: object,
    moment_covariance: object,
    genetic_count: int,
    block_count: int,
    *,
    keep: object | None = None,
    keep_b: object | None = None,
) -> np.ndarray:
    """``B/(B-1) T_a^-1 [C 0; 0 0] T_b^-1`` for kept index sets a and b.

    ``T_a`` solves ``T[keep, keep]``; with ``keep_b`` None the result is the
    added covariance of one model (``len(keep)`` square), otherwise the
    cross-covariance between two models fitted to the same moments.
    """
    matrix = _finite("normal_matrix", normal_matrix, 2)
    moment = _finite("moment_covariance", moment_covariance, 2)
    if matrix.shape[0] != matrix.shape[1] or moment.shape != (genetic_count, genetic_count):
        raise ValueError("normal_matrix and moment_covariance disagree")
    if isinstance(block_count, bool) or block_count < 2:
        raise ValueError("block_count must be at least 2")
    kept_a = _keep_indices(keep, matrix.shape[0])
    cols_a, gen_a = _inverse_columns(matrix, kept_a, genetic_count)
    factor = block_count / (block_count - 1.0)
    if keep_b is None:
        added = factor * cols_a @ moment[np.ix_(gen_a, gen_a)] @ cols_a.T
        return 0.5 * (added + added.T)
    kept_b = _keep_indices(keep_b, matrix.shape[0])
    cols_b, gen_b = _inverse_columns(matrix, kept_b, genetic_count)
    return factor * cols_a @ moment[np.ix_(gen_a, gen_b)] @ cols_b.T


def crossblock_signal_covariance(
    normal_matrix: object,
    coefficients: object,
    component_index: ContextComponentIndex,
    inputs: CrossBlockSignalInputs,
    *,
    keep: object | None = None,
    omegas: object | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Added covariance ``B/(B-1) T^-1 [C 0; 0 0] T^-1`` of one model.

    ``coefficients`` is the full coefficient vector (genetic first); ``keep``
    selects the model's coefficients (all when None), as for a restricted
    fit that solves ``T[keep, keep]``. C is evaluated at ``omegas`` when
    given (for example the full model's estimate when several nested models
    are fitted to one trait), otherwise at the model's own estimate. Returns
    a ``len(keep)`` square matrix on the kept axis and diagnostics.
    """
    matrix = _finite("normal_matrix", normal_matrix, 2)
    theta = _finite("coefficients", coefficients, 1)
    c = len(component_index)
    if matrix.shape != (theta.size, theta.size) or theta.size < c:
        raise ValueError("normal_matrix and coefficients disagree")
    kept = _keep_indices(keep, theta.size)
    if omegas is None:
        omegas = _model_omegas(component_index, theta, kept)
    moment, diagnostics = crossblock_moment_covariance(component_index, omegas, inputs)
    added = added_covariance_from_moments(matrix, moment, c, inputs.block_count, keep=kept)
    return added, diagnostics


def cross_model_crossblock_covariance(
    normal_matrix: object,
    component_index: ContextComponentIndex,
    inputs: CrossBlockSignalInputs,
    *,
    omegas: object,
    keep_a: object | None,
    keep_b: object | None,
) -> np.ndarray:
    """Added cross-covariance between two models fitted to the same moments.

    For paired contrasts such as joint-minus-single-exposure estimates:
    ``B/(B-1) T_a^-1 [C 0; 0 0] T_b^-1``, with one C evaluated at ``omegas``
    (the estimate of the most general model) for both models. Returns
    ``len(keep_a) x len(keep_b)``.
    """
    moment, _ = crossblock_moment_covariance(component_index, omegas, inputs)
    return added_covariance_from_moments(normal_matrix, moment, len(component_index),
                                         inputs.block_count, keep=keep_a, keep_b=keep_b)


def numerical_gradient(function: Callable[[np.ndarray], float], coefficients: object,
                       *, relative_step: float = 1.0e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function of the coefficients."""
    theta = _finite("coefficients", coefficients, 1).copy()
    step = relative_step * max(float(np.max(np.abs(theta))), np.finfo(np.float64).tiny)
    gradient = np.empty_like(theta)
    for j in range(theta.size):
        up = theta.copy(); down = theta.copy()
        up[j] += step; down[j] -= step
        gradient[j] = (float(function(up)) - float(function(down))) / (2.0 * step)
    return gradient


def derived_variance(
    function: Callable[[np.ndarray], float],
    coefficients: object,
    loo_coefficients: object,
    added_covariance: object | None,
) -> dict[str, float]:
    """Variance of a scalar derived quantity f(theta).

    The jackknife part is the delete-one-block variance of f over the stored
    deletion coefficients (no linearization); the between-block part is
    ``grad f' A grad f`` with A the added covariance on the full coefficient
    axis (None or zeros for a jackknife-only fit).
    """
    theta = _finite("coefficients", coefficients, 1)
    loo = _finite("loo_coefficients", loo_coefficients, 2)
    if loo.shape[1] != theta.size or loo.shape[0] < 2:
        raise ValueError("loo_coefficients must be deletions x coefficients")
    values = np.array([float(function(row)) for row in loo])
    b = values.size
    jackknife = float((b - 1.0) / b * np.sum((values - values.mean()) ** 2))
    added = 0.0
    if added_covariance is not None:
        matrix = _finite("added_covariance", added_covariance, 2)
        if matrix.shape != (theta.size, theta.size):
            raise ValueError("added_covariance must match the coefficient axis")
        if np.any(matrix != 0.0):
            gradient = numerical_gradient(function, theta)
            added = float(gradient @ matrix @ gradient)
    return {"estimate": float(function(theta)), "jackknife_variance": jackknife,
            "crossblock_variance": added, "variance": jackknife + added,
            "standard_error": float(np.sqrt(max(jackknife + added, 0.0)))}


def warn_unavailable(reason: str, *, stacklevel: int = 3) -> str:
    """Warn that the correction was not formed; return the status label."""
    warnings.warn(
        f"Between-block signal covariance not applied ({reason}); standard errors are "
        "the SNP-block jackknife only, which understates the variance of G-GxE "
        "covariances (amplification).",
        CrossBlockSignalUnavailableWarning, stacklevel=stacklevel)
    return f"not_applied_{reason}"


def uncertainty_record(requested: str, status: str,
                       diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Manifest entry describing how the coefficient covariance was formed."""
    record = {"requested_standard_errors": requested,
              "method": METHOD if status == "applied" else JACKKNIFE_ONLY_METHOD,
              "crossblock_status": status,
              "covariance": ("raw_jackknife_covariance + raw_crossblock_covariance"
                             if status == "applied" else "raw_jackknife_covariance")}
    if diagnostics:
        record["crossblock_diagnostics"] = dict(diagnostics)
    return record
