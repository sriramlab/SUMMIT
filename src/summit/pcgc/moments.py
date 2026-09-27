"""PCGC sufficient statistics and small normal equations (no variance row)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from summit.context.oracle import symmetric_rank_diagnostics
from summit.context.annotations import _jackknife_covariance
from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
from summit.sumstats.binary import finite_array, readonly

FROZEN_JACKKNIFE = "pcgc_frozen_offdiagonal_estimating_equations_v2"
OFFDIAGONAL = "offdiagonal_variant_rows_v2"
LEGACY_DIAGONAL = "full_diagonal_variant_rows_v1"
METHODS = ("liability", "pcgc", "pcgc-inverse", "pcgc-basis", "pcgc-ld")


def annotations_array(value, num_variants):
    a = finite_array("annotations", value, 2)
    if a.shape[0] != num_variants or a.shape[1] < 1 or np.any(a < 0) or np.any(a.sum(axis=0) <= 0):
        raise ValueError("annotations must be nonnegative M by K with positive column masses")
    return a


def method_vectors(risk, method):
    """Return the sample feature multiplier and raw SNP score response."""
    d = risk.sensitivity
    if method == "liability":
        if not np.allclose(risk.population_risk, risk.population_prevalence, rtol=1e-12, atol=0):
            raise ValueError("liability requires constant population risk; use pcgc for covariates")
        return d, d * risk.z
    if method in ("pcgc", "pcgc-ld"):
        return d, d * risk.z
    if method == "pcgc-inverse":
        return np.ones_like(d), risk.z / d
    raise ValueError("basis weights must be explicitly contracted before scoring")


@dataclass(frozen=True)
class BinaryMoments:
    """One aligned study/reference contract, with exact diagonal removal per SNP.

    ``ldscores[j,b]`` is the directional kernel-product numerator divided by
    reference N squared, before annotation-mass normalization. In v2 the
    same-person contribution is already removed from each LD row; the small
    ``same_person`` matrix is retained only as a diagnostic. Legacy v1 moments
    support point estimates but cannot support corrected row deletions.
    """
    annotations: np.ndarray
    ldscores: np.ndarray
    same_person: np.ndarray
    rhs_rows: np.ndarray
    n_samples: int
    method: str
    covariate_variance: float = 0.0
    ldscore_contract: str = OFFDIAGONAL

    def __post_init__(self):
        rhs = finite_array("rhs_rows", self.rhs_rows, 1)
        a = annotations_array(self.annotations, len(rhs))
        ld = finite_array("ldscores", self.ldscores, 2)
        sp = finite_array("same_person", self.same_person, 2)
        if ld.shape != a.shape or sp.shape != (a.shape[1], a.shape[1]):
            raise ValueError("PCGC sufficient-statistic axes do not match")
        if type(self.n_samples) is not int or self.n_samples < 3 or self.method not in METHODS:
            raise ValueError("invalid PCGC sample count or method")
        if not np.allclose(sp, sp.T, rtol=1e-12, atol=1e-12):
            raise ValueError("same-person matrix must be symmetric")
        tolerance = 1e-10 * max(1., float(np.max(np.abs(sp))))
        if np.min(sp) < -tolerance or np.linalg.eigvalsh(sp)[0] < -tolerance:
            raise ValueError("same-person matrix must be nonnegative and positive semidefinite")
        if not np.isfinite(self.covariate_variance) or self.covariate_variance < 0:
            raise ValueError("invalid population covariate variance")
        if self.ldscore_contract not in (OFFDIAGONAL, LEGACY_DIAGONAL):
            raise ValueError("unsupported PCGC LD row contract")
        for name, value in (("annotations", a), ("ldscores", ld), ("same_person", sp), ("rhs_rows", rhs)):
            object.__setattr__(self, name, readonly(value))

    def equations(self, retained=None):
        if retained is not None:
            retained = np.asarray(retained)
            if retained.dtype != bool or retained.shape != self.rhs_rows.shape:
                raise ValueError("retained must be a Boolean mask on the variant axis")
            if self.ldscore_contract == LEGACY_DIAGONAL:
                raise ValueError("legacy PCGC moments lack per-SNP diagonal removal; regenerate for jackknife")
        selection = slice(None) if retained is None else retained
        a = self.annotations[selection]
        mass = a.sum(axis=0)
        if np.any(mass <= 0):
            raise ValueError("variant deletion empties an annotation")
        directed = a.T @ self.ldscores[selection]
        # Only the target axis is deleted. Source kernels, masses and probes
        # remain fixed. Row normalization cancels in the solve; columns must
        # retain FULL masses to estimate the original genome-wide components.
        matrix = self.n_samples**2 * directed / np.outer(mass, self.annotations.sum(axis=0))
        if self.ldscore_contract == LEGACY_DIAGONAL:
            matrix = (matrix + matrix.T)/2 - self.same_person
        rhs = a.T @ self.rhs_rows[selection] / mass
        return matrix, rhs


def solve(matrix, rhs, *, deletion=False):
    # Directional probe estimates and one-sided deletions need not be
    # symmetric. Diagnose the symmetric part of the full Gram, but solve the
    # actual estimating equations. Symmetrizing a deletion changes its target.
    diagnostics = symmetric_rank_diagnostics((matrix + matrix.T)/2, rtol=1e-10)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition >= 1e10 or (not deletion and diagnostics.eigenvalues[0] <= 0):
        raise ValueError("PCGC normal matrix is nonpositive or rank deficient; check annotations/reference probes")
    return np.linalg.solve(matrix, rhs), diagnostics


def risk_pair_factor(risk):
    """Mean d_i^2 d_j^2 over distinct people, conditional on supplied risks."""
    squares = risk.sensitivity**2
    n = risk.n_samples
    return float((squares.sum()**2-squares @ squares)/(n*(n-1)))


def external_ld_moments(rhs_rows, annotations, risk, population_ld):
    """Explicit risk-independent external-LD approximation, conditional scale.

    E[H_ab | d] ~ sum_{i!=j}(d_i^2 d_j^2) tr(R A_a R A_b)/(M_a M_b).
    Finite-reference same-person terms must already have been removed per SNP.
    Ascertainment-induced risk/genotype dependence violates this factorization.
    """
    n = risk.n_samples
    ld = finite_array("external offdiagonal LD", population_ld, 2) * (n-1)/n * risk_pair_factor(risk)
    k = np.asarray(annotations).shape[1]
    return BinaryMoments(annotations, ld, np.zeros((k, k)), rhs_rows, n, "pcgc-ld", risk.covariate_variance)


def fit_moments(moments, *, block_ids=None):
    matrix, rhs = moments.equations()
    theta, diagnostics = solve(matrix, rhs)
    result = {
        "method": moments.method,
        "conditional_components": theta.tolist(),
        "marginal_components": (theta / (1 + moments.covariate_variance)).tolist(),
        "conditional_total": float(theta.sum()),
        "marginal_total": float(theta.sum() / (1 + moments.covariate_variance)),
        "normal_condition": float(np.linalg.cond(matrix)),
        "normal_relative_asymmetry": float(np.linalg.norm(matrix-matrix.T)/max(np.linalg.norm(matrix), np.finfo(float).tiny)),
        "minimum_eigenvalue": float(diagnostics.eigenvalues[0]),
        "relative_solve_residual": float(np.linalg.norm(matrix @ theta - rhs) / max(1., np.linalg.norm(rhs))),
        "uncertainty_status": "not_requested",
    }
    if block_ids is not None:
        if moments.ldscore_contract != OFFDIAGONAL:
            raise ValueError("legacy PCGC moments lack per-SNP diagonal removal; regenerate for jackknife")
        ids = np.asarray(block_ids)
        if ids.dtype.kind not in "iu" or ids.shape != moments.rhs_rows.shape:
            raise ValueError("jackknife block IDs must be integers on the variant axis")
        labels, ids, sizes = np.unique(ids, return_inverse=True, return_counts=True)
        count = len(labels)
        if count < 2:
            raise ValueError("jackknife needs at least two nonempty blocks")
        mass = moments.annotations.sum(axis=0)
        # Recover the already-computed full sums from their normalization.
        directed = matrix*np.outer(mass, mass)/moments.n_samples**2
        raw_rhs = rhs*mass
        k = len(theta)
        if k == 1:
            # The common single-component path needs only three linear scans.
            aj = moments.annotations[:, 0]
            block_mass = np.bincount(ids, weights=aj, minlength=count)[:, None]
            block_directed = np.bincount(ids, weights=aj*moments.ldscores[:, 0], minlength=count)[:, None, None]
        else:
            # Preserve SUMMIT's contiguous-block GEMM and sparse annotation
            # fast paths rather than streaming the whole M axis K squared times.
            block_directed, block_mass, _ = reduce_generalized_gxe_reference_for_inference(
                directional_ldscores=moments.ldscores[:, None, :], annotations=moments.annotations,
                variant_block_ids=ids, block_labels=tuple(map(str, labels)),
                expected_directed_numerator=directed)
        block_rhs = np.empty((count, k))
        for j in range(k):
            aj = moments.annotations[:, j]
            block_rhs[:, j] = np.bincount(ids, weights=aj*moments.rhs_rows, minlength=count)
        loo = []
        for bm, bd, br in zip(block_mass, block_directed, block_rhs):
            retained = mass - bm
            if np.any(retained <= 0):
                raise ValueError("jackknife deletion empties an annotation")
            dr = directed - bd
            H = moments.n_samples**2 * dr / np.outer(retained, mass)
            loo.append(solve(H, (raw_rhs - br) / retained, deletion=True)[0])
        loo = np.asarray(loo)
        covariance = _jackknife_covariance(loo)
        result.update({
            "uncertainty_status": "estimated",
            "jackknife_method": FROZEN_JACKKNIFE,
            "jackknife_blocks": count,
            "conditional_jackknife_covariance": covariance.tolist(),
            "marginal_jackknife_covariance": (covariance/(1+moments.covariate_variance)**2).tolist(),
            "conditional_standard_errors": np.sqrt(np.maximum(0, covariance.diagonal())).tolist(),
            "conditional_total_standard_error": float(np.sqrt(max(0, covariance.sum()))),
            "marginal_standard_errors": (np.sqrt(np.maximum(0, covariance.diagonal()))/(1+moments.covariate_variance)).tolist(),
            "marginal_total_standard_error": float(np.sqrt(max(0, covariance.sum()))/(1+moments.covariate_variance)),
            "jackknife_replicates": loo.tolist(),
            "jackknife_block_sizes": sizes.tolist(),
            "jackknife_risk_refit": False,
            "uncertainty_conditioning": "fixed_risk_fit_covariate_variance_genotype_scale_and_reference_probes",
        })
    return result
