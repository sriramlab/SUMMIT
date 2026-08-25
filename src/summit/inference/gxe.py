"""Summary-statistics method-of-moments estimation for G + GxE + NxE.

The implementation in this module is deliberately based on raw marginal score
cross-products, not on conditional interaction test statistics.  It reconstructs
the same normal equations as an individual-level GENIE analysis after a fixed
projection has been chosen.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..ldscore.gwe_ldscore import _validate_gxe_annotation_names
from .jackknife import JackknifeSpec


_REFERENCE_KIND = "summit.gxe.reference"
_MOMENTS_KIND = "summit.gxe.phenotype_moments"
_SCHEMA_VERSION = 4
_FIT_BATCH_KIND = "summit.gxe.fit_batch"
_FIT_BATCH_SCHEMA_VERSION = 1
_FIT_BATCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_POSTHOC_JACKKNIFE_METHOD = "frozen_full_genome_variant_ldscore_delete_block_v1"
_MATCHED_REFERENCE_MODE = "matched"
_POPULATION_REFERENCE_MODE = "population"


def _as_float_array(name: str, value: Any, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _require_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}; expected {shape}.")


@dataclass(frozen=True)
class GxENormalEquations:
    matrix: np.ndarray
    rhs: np.ndarray
    traces: np.ndarray
    component_names: tuple[str, ...]
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class GxEFitResult:
    component_names: tuple[str, ...]
    coefficients: np.ndarray
    contributions: np.ndarray
    proportions: np.ndarray
    normal_matrix: np.ndarray
    rhs: np.ndarray
    singular_values: np.ndarray
    normal_eigenvalues: np.ndarray
    rank: int
    condition_number: float
    relative_residual: float
    solve_method: str = "residual_eliminated_svd"
    normal_symmetry_error: float = 0.0
    cauchy_schwarz_max_violation: float = 0.0
    component_influence: np.ndarray | None = None
    identifiable: bool = True
    normal_equation_diagnostics: dict[str, Any] | None = None
    coefficient_standard_errors: np.ndarray | None = None
    proportion_standard_errors: np.ndarray | None = None
    jackknife_coefficients: np.ndarray | None = None
    jackknife_proportions: np.ndarray | None = None
    jackknife_block_labels: tuple[str, ...] = ()
    phenotype_residual_variance_fraction: float | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component": self.component_names,
                "coefficient": self.coefficients,
                "coefficient_se": (
                    self.coefficient_standard_errors
                    if self.coefficient_standard_errors is not None
                    else np.full(len(self.component_names), np.nan)
                ),
                "variance_contribution": self.contributions,
                "proportion": self.proportions,
                "proportion_se": (
                    self.proportion_standard_errors
                    if self.proportion_standard_errors is not None
                    else np.full(len(self.component_names), np.nan)
                ),
                "original_scale_proportion": (
                    self.proportions * self.phenotype_residual_variance_fraction
                    if self.phenotype_residual_variance_fraction is not None
                    else np.full(len(self.component_names), np.nan)
                ),
            }
        ).replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class GxEPhenotypeInput:
    """One phenotype summary triplet for a reusable GxE reference."""

    phenotype_moments: str | Path
    gwas_scores: str | Path
    gwis_scores: str | Path


@dataclass(frozen=True)
class GxEFitBatchEntry:
    """Validated input/output record from a batch-fit manifest."""

    name: str
    phenotype_input: GxEPhenotypeInput
    output_prefix: Path


@dataclass(frozen=True)
class _PreparedGxEReference:
    """Validated in-memory reference sufficient statistics shared by traits."""

    path: Path
    payload: dict[str, Any]
    schema_version: int
    n_variants: int
    variant_axis: dict[str, np.ndarray]
    annotations: np.ndarray
    annotation_masses: np.ndarray
    annotation_names: tuple[str, ...]
    residual_rank: int
    component_names: tuple[str, ...]
    normal_matrix: np.ndarray
    traces: np.ndarray
    block_values: np.ndarray | None
    block_labels: tuple[str, ...]
    block_masses: np.ndarray | None
    deleted_matrices: np.ndarray | None
    deleted_traces: np.ndarray | None
    population_same_individual_products: np.ndarray | None


def _population_same_individual_products(
    payload: Mapping[str, Any], annotation_names: Sequence[str]
) -> np.ndarray:
    declaration = payload.get("population_trace")
    if not isinstance(declaration, Mapping):
        raise ValueError(
            "Population-reference inference requires a reference with population_trace moments."
        )
    if (
        declaration.get("method") != "independent_probe_u_statistic_v1"
        or declaration.get("sampling_axis") != "individual"
    ):
        raise ValueError("Reference population_trace uses an unsupported estimator.")
    probes = declaration.get("num_vectors")
    if isinstance(probes, bool) or not isinstance(probes, int) or probes < 2:
        raise ValueError("Reference population_trace requires at least two probes.")
    randomization = payload.get("randomization")
    if (
        not isinstance(randomization, Mapping)
        or randomization.get("num_vectors") != probes
    ):
        raise ValueError(
            "Reference population_trace probe count disagrees with its randomization."
        )
    expected_order = [
        *[f"G:{name}" for name in annotation_names],
        *[f"GxE:{name}" for name in annotation_names],
    ]
    if declaration.get("feature_order") != expected_order:
        raise ValueError("Reference population_trace feature order is inconsistent.")
    products = _as_float_array(
        "population same-individual kernel products",
        declaration.get("same_individual_kernel_products"),
        ndim=2,
    )
    expected_shape = (2 * len(annotation_names), 2 * len(annotation_names))
    _require_shape("population same-individual kernel products", products, expected_shape)
    # The estimand is entrywise non-negative, but the unbiased order-two
    # finite-probe U-statistic need not be.  Do not truncate or reject a valid
    # Monte Carlo realization; its uncertainty is controlled by B.
    scale = max(1.0, float(np.max(np.abs(products))))
    if float(np.max(np.abs(products - products.T))) > 2.0e-10 * scale:
        raise ValueError("Reference population same-individual products are not symmetric.")
    return 0.5 * (products + products.T)


def _study_population_design_moments(
    payload: Mapping[str, Any],
    annotation_names: Sequence[str],
) -> np.ndarray:
    """Validate exact full-study genetic-by-NxE trace summaries."""
    declaration = payload.get("population_design")
    if not isinstance(declaration, Mapping):
        raise ValueError(
            "Population-reference inference requires exact study population_design moments."
        )
    if declaration.get("method") != "exact_projected_feature_nxe_v1":
        raise ValueError("Study population_design uses an unsupported estimator.")
    expected_order = [
        *[f"G:{name}" for name in annotation_names],
        *[f"GxE:{name}" for name in annotation_names],
    ]
    if declaration.get("feature_order") != expected_order:
        raise ValueError("Study population_design feature order is inconsistent.")
    full = _as_float_array(
        "study genetic-by-NxE traces",
        declaration.get("genetic_nxe_traces"),
        ndim=1,
    )
    _require_shape("study genetic-by-NxE traces", full, (len(expected_order),))
    if np.any(full < 0.0):
        raise ValueError("Study genetic-by-NxE traces must be non-negative.")

    if (
        declaration.get("jackknife_genetic_nxe_traces") is not None
        or declaration.get("jackknife_block_labels") is not None
    ):
        raise ValueError(
            "Reference-bound population jackknife moments are obsolete; "
            "delete-block traces are formed from per-SNP DNXE rows at inference."
        )
    return full


def transfer_reference_normal_equations(
    reference_equations: GxENormalEquations,
    *,
    reference_n_samples: int,
    study_n_samples: int,
    reference_residual_rank: int,
    study_residual_rank: int,
    same_individual_products: np.ndarray,
    genetic_nxe_traces: np.ndarray,
    q_nxe: float,
    q_residual: float,
    trace_nxe: float,
    trace_nxe_sq: float,
) -> GxENormalEquations:
    """Transfer phenotype-independent kernel moments to a study cohort.

    The reference genetic block is decomposed into same-person and
    different-person parts. Under the declared reference-population
    approximation, those parts scale as N and N(N-1), respectively. The
    study's genetic-by-NxE and low-dimensional NxE moments are exact and are
    never borrowed from the reference.
    """
    n_ref = int(reference_n_samples)
    n_study = int(study_n_samples)
    r_ref = int(reference_residual_rank)
    r_study = int(study_residual_rank)
    if (
        n_ref < 2
        or n_study < 2
        or r_ref < 2
        or r_study < 2
        or r_ref >= n_ref
        or r_study >= n_study
    ):
        raise ValueError("Population trace transfer has inconsistent sample sizes/ranks.")
    reference_matrix = _as_float_array(
        "reference normal matrix", reference_equations.matrix, ndim=2
    )
    reference_traces = _as_float_array(
        "reference kernel traces", reference_equations.traces, ndim=1
    )
    p = reference_matrix.shape[0]
    if reference_matrix.shape != (p, p) or reference_traces.shape != (p,):
        raise ValueError("Reference equations have inconsistent dimensions.")
    if (p - 2) % 2:
        raise ValueError("Reference equations do not have a G/GxE/NxE/residual layout.")
    genetic_count = p - 2
    diagonal = _as_float_array(
        "same_individual_products", same_individual_products, ndim=2
    )
    _require_shape(
        "same_individual_products", diagonal, (genetic_count, genetic_count)
    )
    diagonal_scale = max(1.0, float(np.max(np.abs(diagonal))))
    diagonal_symmetry_error = float(np.max(np.abs(diagonal - diagonal.T)))
    if diagonal_symmetry_error > 2.0e-10 * diagonal_scale:
        raise ValueError("same_individual_products are materially asymmetric.")
    diagonal = 0.5 * (diagonal + diagonal.T)
    diagonal_eigenvalues = np.linalg.eigvalsh(diagonal)
    genetic_nxe = _as_float_array(
        "genetic_nxe_traces", genetic_nxe_traces, ndim=1
    )
    _require_shape("genetic_nxe_traces", genetic_nxe, (genetic_count,))
    if np.any(genetic_nxe < 0.0):
        raise ValueError("genetic_nxe_traces must be non-negative.")
    if not np.allclose(
        reference_traces[:genetic_count],
        float(r_ref),
        rtol=1.0e-9,
        atol=1.0e-8,
    ):
        raise ValueError(
            "Population transfer requires standardized reference genetic kernels."
        )

    matrix = np.zeros_like(reference_matrix)
    same_scale = float(n_study) / float(n_ref)
    different_scale = (
        float(n_study * (n_study - 1)) / float(n_ref * (n_ref - 1))
    )
    reference_genetic = reference_matrix[:genetic_count, :genetic_count]
    matrix[:genetic_count, :genetic_count] = (
        same_scale * diagonal
        + different_scale * (reference_genetic - diagonal)
    )
    matrix[:genetic_count, genetic_count] = genetic_nxe
    matrix[genetic_count, :genetic_count] = matrix[
        :genetic_count, genetic_count
    ]
    traces = np.zeros_like(reference_traces)
    traces[:genetic_count] = float(r_study)
    matrix[:genetic_count, genetic_count + 1] = traces[:genetic_count]
    matrix[genetic_count + 1, :genetic_count] = traces[:genetic_count]

    q_nxe = float(q_nxe)
    q_residual = float(q_residual)
    trace_nxe = float(trace_nxe)
    trace_nxe_sq = float(trace_nxe_sq)
    scalars = (q_nxe, q_residual, trace_nxe, trace_nxe_sq)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("Study NxE/residual moments must be finite.")
    if q_nxe < 0.0 or trace_nxe < 0.0 or trace_nxe_sq < 0.0:
        raise ValueError("Study NxE moments must be non-negative.")
    if not np.isclose(q_residual, float(r_study), rtol=1.0e-10, atol=1.0e-8):
        raise ValueError(
            "Study q_residual is inconsistent with its residual rank: "
            f"expected {r_study}, got {q_residual:.16g}."
        )
    traces[genetic_count] = trace_nxe
    traces[genetic_count + 1] = float(r_study)
    matrix[genetic_count, genetic_count] = trace_nxe_sq
    matrix[genetic_count, genetic_count + 1] = trace_nxe
    matrix[genetic_count + 1, genetic_count] = trace_nxe
    matrix[genetic_count + 1, genetic_count + 1] = float(r_study)
    return GxENormalEquations(
        0.5 * (matrix + matrix.T),
        np.zeros(p, dtype=np.float64),
        traces,
        reference_equations.component_names,
        {
            "population_transfer": "same_and_different_individual_moment_scaling_v1",
            "reference_n_samples": n_ref,
            "study_n_samples": n_study,
            "reference_residual_rank": r_ref,
            "study_residual_rank": r_study,
            "same_individual_scale": same_scale,
            "different_individual_scale": different_scale,
            "extrapolation": bool(n_study > n_ref),
            "same_individual_symmetry_error": diagonal_symmetry_error,
            "same_individual_minimum_eigenvalue": float(diagonal_eigenvalues[0]),
            "same_individual_estimator_psd_expected": False,
        },
    )


def assemble_normal_equations(
    *,
    annotations: np.ndarray,
    score_x: np.ndarray,
    score_w: np.ndarray,
    ld_xx: np.ndarray,
    ld_xw: np.ndarray,
    ld_wx: np.ndarray,
    ld_ww: np.ndarray,
    norm_x: np.ndarray,
    norm_w: np.ndarray,
    diag_nxe_x: np.ndarray,
    diag_nxe_w: np.ndarray,
    residual_rank: int,
    q_nxe: float,
    q_residual: float,
    trace_nxe: float,
    trace_nxe_sq: float,
    annotation_names: Sequence[str] | None = None,
    ld_scale: str = "cross_product_over_rank_squared",
) -> GxENormalEquations:
    """Assemble exact GENIE normal equations from aligned summary arrays.

    ``score_x`` and ``score_w`` must equal F' y / sqrt(r), where r is the
    residual-maker rank.  LD panels contain source-annotation sums of
    (f'g/r)^2.  ``norm_*`` and ``diag_nxe_*`` contain f'f/r and
    f'diag(e^2)f/r, respectively.
    """
    annot = _as_float_array("annotations", annotations, ndim=2)
    m, k = annot.shape
    if m == 0 or k == 0:
        raise ValueError("annotations must have at least one row and one column.")
    if np.any(annot < 0.0):
        raise ValueError("annotations must be non-negative.")

    vectors = {
        "score_x": score_x,
        "score_w": score_w,
        "norm_x": norm_x,
        "norm_w": norm_w,
        "diag_nxe_x": diag_nxe_x,
        "diag_nxe_w": diag_nxe_w,
    }
    vec: dict[str, np.ndarray] = {}
    for name, value in vectors.items():
        arr = _as_float_array(name, value, ndim=1)
        _require_shape(name, arr, (m,))
        vec[name] = arr
    for name in ("norm_x", "norm_w", "diag_nxe_x", "diag_nxe_w"):
        if np.any(vec[name] < 0.0):
            bad = int(np.flatnonzero(vec[name] < 0.0)[0])
            raise ValueError(f"{name} must be non-negative; first invalid row is {bad}.")

    panels: dict[str, np.ndarray] = {}
    for name, value in {
        "ld_xx": ld_xx,
        "ld_xw": ld_xw,
        "ld_wx": ld_wx,
        "ld_ww": ld_ww,
    }.items():
        arr = _as_float_array(name, value, ndim=2)
        _require_shape(name, arr, (m, k))
        if np.any(arr < 0.0):
            bad = np.argwhere(arr < 0.0)[0].tolist()
            raise ValueError(
                f"{name} must be non-negative squared-sketch contributions; "
                f"first invalid index is {bad}."
            )
        arr = np.array(arr, dtype=np.float64, copy=True)
        arr[arr == 0.0] = 0.0
        panels[name] = arr

    r = int(residual_rank)
    if r <= 0:
        raise ValueError(f"residual_rank must be positive; got {r}.")
    scalars = {
        "q_nxe": float(q_nxe),
        "q_residual": float(q_residual),
        "trace_nxe": float(trace_nxe),
        "trace_nxe_sq": float(trace_nxe_sq),
    }
    if not all(np.isfinite(v) for v in scalars.values()):
        raise ValueError("NxE/residual moments must all be finite.")
    if scalars["q_nxe"] < 0.0 or scalars["q_residual"] <= 0.0:
        raise ValueError("Phenotype quadratic moments require q_nxe >= 0 and q_residual > 0.")
    if not np.isclose(scalars["q_residual"], float(r), rtol=1.0e-10, atol=1.0e-8):
        raise ValueError(
            "q_residual is inconsistent with the declared normalized marginal-score scale: "
            f"expected residual_rank={r}, got {scalars['q_residual']:.16g}."
        )
    if scalars["trace_nxe"] < 0.0 or scalars["trace_nxe_sq"] < 0.0:
        raise ValueError("NxE traces must be non-negative.")

    masses = annot.sum(axis=0, dtype=np.float64)
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        bad = np.flatnonzero((~np.isfinite(masses)) | (masses <= 0.0)).tolist()
        raise ValueError(f"Annotation columns must have positive finite mass; invalid columns: {bad}.")

    if annotation_names is None:
        names = tuple(f"bin{i}" for i in range(k))
    else:
        names = tuple(str(x) for x in annotation_names)
        if len(names) != k or len(set(names)) != k:
            raise ValueError("annotation_names must contain one unique name per annotation column.")

    if ld_scale != "cross_product_over_rank_squared":
        raise ValueError(
            "Unsupported LD scale. Expected 'cross_product_over_rank_squared'; "
            f"got {ld_scale!r}."
        )

    component_names = (
        tuple(f"G:{name}" for name in names)
        + tuple(f"GxE:{name}" for name in names)
        + ("NxE", "residual")
    )
    p = 2 * k + 2
    lhs = np.zeros((p, p), dtype=np.float64)
    rhs = np.zeros(p, dtype=np.float64)
    traces = np.zeros(p, dtype=np.float64)

    # Genetic RHS and traces against the residual and NxE kernels.
    for a in range(k):
        wa = annot[:, a]
        ma = masses[a]
        gi = a
        wi = k + a
        rhs[gi] = r * np.dot(wa, vec["score_x"] ** 2) / ma
        rhs[wi] = r * np.dot(wa, vec["score_w"] ** 2) / ma
        traces[gi] = r * np.dot(wa, vec["norm_x"]) / ma
        traces[wi] = r * np.dot(wa, vec["norm_w"]) / ma
        lhs[gi, 2 * k] = lhs[2 * k, gi] = r * np.dot(wa, vec["diag_nxe_x"]) / ma
        lhs[wi, 2 * k] = lhs[2 * k, wi] = r * np.dot(wa, vec["diag_nxe_w"]) / ma
        lhs[gi, 2 * k + 1] = lhs[2 * k + 1, gi] = traces[gi]
        lhs[wi, 2 * k + 1] = lhs[2 * k + 1, wi] = traces[wi]

    scale = float(r * r)

    def directed(panel: np.ndarray, left_bin: int, source_bin: int) -> float:
        value = scale * np.dot(annot[:, left_bin], panel[:, source_bin])
        value /= masses[left_bin] * masses[source_bin]
        return float(value)

    for a in range(k):
        for b in range(k):
            # Averaging the two orientations preserves exact values and reduces
            # Monte-Carlo asymmetry for stochastic LD panels.
            gg = 0.5 * (directed(panels["ld_xx"], a, b) + directed(panels["ld_xx"], b, a))
            ii = 0.5 * (directed(panels["ld_ww"], a, b) + directed(panels["ld_ww"], b, a))
            gi = 0.5 * (directed(panels["ld_xw"], a, b) + directed(panels["ld_wx"], b, a))
            lhs[a, b] = gg
            lhs[k + a, k + b] = ii
            lhs[a, k + b] = lhs[k + b, a] = gi

    rhs[2 * k] = scalars["q_nxe"]
    rhs[2 * k + 1] = scalars["q_residual"]
    traces[2 * k] = scalars["trace_nxe"]
    traces[2 * k + 1] = float(r)
    lhs[2 * k, 2 * k] = scalars["trace_nxe_sq"]
    lhs[2 * k, 2 * k + 1] = lhs[2 * k + 1, 2 * k] = scalars["trace_nxe"]
    lhs[2 * k + 1, 2 * k + 1] = float(r)

    # Remove only roundoff-level asymmetry.  Material orientation disagreement
    # has already been retained in the average above and should be diagnosed by
    # the caller from the raw panels if desired.
    lhs = 0.5 * (lhs + lhs.T)
    return GxENormalEquations(lhs, rhs, traces, component_names)


def solve_normal_equations(
    equations: GxENormalEquations,
    *,
    rcond: float | None = None,
    max_condition: float = 1.0e12,
    allow_ill_conditioned: bool = False,
) -> GxEFitResult:
    """Solve the unconstrained MoM system after eliminating the residual row."""
    max_condition = float(max_condition)
    if not np.isfinite(max_condition) or max_condition <= 0.0:
        raise ValueError(f"max_condition must be positive and finite; got {max_condition!r}.")
    if rcond is not None:
        rcond = float(rcond)
        if not np.isfinite(rcond) or rcond <= 0.0:
            raise ValueError(f"rcond must be positive and finite when supplied; got {rcond!r}.")
    lhs_raw = _as_float_array("normal matrix", equations.matrix, ndim=2)
    rhs = _as_float_array("normal RHS", equations.rhs, ndim=1)
    if (
        lhs_raw.shape[0] != lhs_raw.shape[1]
        or lhs_raw.shape[0] != rhs.shape[0]
        or lhs_raw.shape[0] < 2
    ):
        raise ValueError("Normal matrix must be square and match the RHS length.")
    traces = _as_float_array("traces", equations.traces, ndim=1)
    _require_shape("traces", traces, (rhs.shape[0],))

    matrix_scale = max(float(np.max(np.abs(lhs_raw))), 1.0)
    symmetry_error = float(np.max(np.abs(lhs_raw - lhs_raw.T)))
    symmetry_tolerance = 2.0e-10 * matrix_scale
    if symmetry_error > symmetry_tolerance:
        raise ValueError(
            "GxE/NxE normal matrix has material asymmetry: "
            f"maximum={symmetry_error:.6g}, tolerance={symmetry_tolerance:.6g}."
        )
    lhs = 0.5 * (lhs_raw + lhs_raw.T)

    diagonal = np.diag(lhs)
    diagonal_tolerance = 2.0e-10 * matrix_scale
    if float(np.min(diagonal)) < -diagonal_tolerance:
        raise ValueError(
            "GxE/NxE normal matrix has a materially negative kernel norm: "
            f"minimum diagonal={float(np.min(diagonal)):.6g}."
        )
    cauchy_bound = np.sqrt(
        np.maximum(diagonal, 0.0)[:, None]
        * np.maximum(diagonal, 0.0)[None, :]
    )
    cauchy_violation = np.abs(lhs) - cauchy_bound
    cauchy_max = float(np.max(cauchy_violation))
    cauchy_tolerance = 2.0e-10 * max(
        matrix_scale, float(np.max(cauchy_bound)), 1.0
    )
    if cauchy_max > cauchy_tolerance:
        location = np.unravel_index(
            int(np.argmax(cauchy_violation)), cauchy_violation.shape
        )
        raise ValueError(
            "GxE/NxE kernel Gram matrix is not positive semidefinite because it "
            "violates a Cauchy--Schwarz bound: "
            f"entry={location}, excess={cauchy_max:.6g}, "
            f"tolerance={cauchy_tolerance:.6g}."
        )

    eigenvalues = np.linalg.eigvalsh(lhs)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    psd_tolerance = 1.0e-10 * eigen_scale
    if eigenvalues[0] < -psd_tolerance:
        raise ValueError(
            "GxE/NxE normal matrix is not positive semidefinite: "
            f"minimum eigenvalue={eigenvalues[0]:.6g}, tolerance={psd_tolerance:.6g}. "
            "The randomized trace estimate is too noisy or the summary artifacts are inconsistent; "
            "increase --nvecs and regenerate the complete reference bundle."
        )

    residual_trace = float(traces[-1])
    residual_tolerance = 2.0e-10 * max(
        matrix_scale, abs(residual_trace), 1.0
    )
    if residual_trace <= 0.0:
        raise ValueError("Residual kernel trace must be positive.")
    if (
        abs(lhs[-1, -1] - residual_trace) > residual_tolerance
        or float(np.max(np.abs(lhs[:-1, -1] - traces[:-1])))
        > residual_tolerance
        or abs(rhs[-1] - residual_trace) > residual_tolerance
    ):
        raise ValueError(
            "GxE/NxE residual row is incompatible with the declared kernel "
            "traces or normalized phenotype RHS."
        )

    trace_nonresidual = traces[:-1]
    reduced = (
        lhs[:-1, :-1]
        - np.outer(trace_nonresidual, trace_nonresidual) / residual_trace
    )
    reduced = 0.5 * (reduced + reduced.T)
    reduced_rhs = rhs[:-1] - trace_nonresidual
    u, singular, vh = np.linalg.svd(reduced, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("Residual-eliminated normal matrix has no positive singular values.")
    tol = (
        max(reduced.shape) * np.finfo(np.float64).eps * singular[0]
        if rcond is None
        else float(rcond) * singular[0]
    )
    reduced_rank = int(np.sum(singular > tol))
    rank = reduced_rank + 1
    condition = float(np.inf if singular[-1] <= 0.0 else singular[0] / singular[-1])
    if not allow_ill_conditioned and (
        reduced_rank < reduced.shape[0] or condition > max_condition
    ):
        raise ValueError(
            "Residual-eliminated GxE/NxE normal equations are not identifiable "
            "at the requested tolerance: "
            f"rank={rank}/{lhs.shape[0]}, condition={condition:.6g}. "
            "This commonly occurs when E^2 is constant (NxE equals residual noise), "
            "or when annotations/kernels are redundant."
        )

    inverse_singular = np.zeros_like(singular)
    inverse_singular[singular > tol] = 1.0 / singular[singular > tol]
    reduced_pseudoinverse = (
        vh.T * inverse_singular.reshape(1, -1)
    ) @ u.T
    coef_nonresidual = reduced_pseudoinverse @ reduced_rhs
    coef_residual = (
        1.0
        - float(trace_nonresidual @ coef_nonresidual) / residual_trace
    )
    coef = np.concatenate(
        [np.asarray(coef_nonresidual, dtype=np.float64), [coef_residual]]
    )
    resid_denom = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
    rel_resid = float(np.linalg.norm(lhs @ coef - rhs) / resid_denom)
    component_influence = np.empty(lhs.shape[0], dtype=np.float64)
    component_influence[:-1] = np.linalg.norm(
        reduced_pseudoinverse, axis=1
    )
    component_influence[-1] = math.sqrt(
        1.0
        + float(
            np.linalg.norm(
                trace_nonresidual @ reduced_pseudoinverse
                / residual_trace
            )
            ** 2
        )
    )
    contributions = coef * traces
    total = float(contributions.sum())
    proportions = np.full_like(contributions, np.nan)
    if np.isfinite(total) and abs(total) > np.finfo(np.float64).tiny:
        proportions = contributions / total
    return GxEFitResult(
        component_names=equations.component_names,
        coefficients=np.asarray(coef, dtype=np.float64),
        contributions=np.asarray(contributions, dtype=np.float64),
        proportions=np.asarray(proportions, dtype=np.float64),
        normal_matrix=lhs,
        rhs=rhs,
        singular_values=np.asarray(singular, dtype=np.float64),
        normal_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        rank=rank,
        condition_number=condition,
        relative_residual=rel_resid,
        solve_method="residual_eliminated_svd",
        normal_symmetry_error=symmetry_error,
        cauchy_schwarz_max_violation=max(0.0, cauchy_max),
        component_influence=component_influence,
        identifiable=bool(
            reduced_rank == reduced.shape[0] and condition <= max_condition
        ),
        normal_equation_diagnostics=(
            None
            if equations.diagnostics is None
            else dict(equations.diagnostics)
        ),
    )


def _block_weighted_sums(
    blocks: np.ndarray,
    values: np.ndarray,
    nblock: int,
    *,
    validate_blocks: bool = True,
) -> np.ndarray:
    """Sum one or more value columns by contiguous integer block ID.

    The work is O(MC), where C is the number of value columns, rather than
    O(JM).  ``np.bincount`` also handles valid non-contiguous SNP ordering
    without allocating one Boolean mask per block.
    """
    block_ids = np.asarray(blocks, dtype=np.int64)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != block_ids.shape[0]:
        raise ValueError("Block aggregation values are not aligned to block IDs.")
    if nblock < 1 or (
        validate_blocks
        and set(np.unique(block_ids).tolist()) != set(range(nblock))
    ):
        raise ValueError("Block IDs must be contiguous integers starting at zero.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Block aggregation values contain non-finite entries.")
    result = np.empty((nblock, matrix.shape[1]), dtype=np.float64)
    for column in range(matrix.shape[1]):
        result[:, column] = np.bincount(
            block_ids,
            weights=matrix[:, column],
            minlength=nblock,
        )
    return result


def _posthoc_inference_blocks(
    variants: pd.DataFrame,
    njack: str | int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assign fixed per-SNP rows to delete-one inference blocks.

    This partition is deliberately created after reference LD scores and trait
    summary rows have been estimated.  It never changes a retained SNP's
    full-genome LD score.
    """
    spec = JackknifeSpec.parse(njack)
    m = len(variants)
    if m < 2:
        raise ValueError("GxE jackknife inference requires at least two SNPs.")
    if spec.mode == "block":
        assert spec.nblocks is not None
        nblock = int(spec.nblocks)
        if not 2 <= nblock <= m:
            raise ValueError(
                f"GxE --njack requires 2 <= blocks <= SNPs; got {nblock} for {m} SNPs."
            )
        edges = np.floor(np.linspace(0, m, nblock + 1)).astype(np.int64)
        blocks = np.empty(m, dtype=np.int64)
        for block_id, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            blocks[start:stop] = block_id
        labels = tuple(f"block_{block_id + 1:04d}" for block_id in range(nblock))
        return blocks, labels

    if spec.delete != 1 or spec.nrep is not None or spec.seed is not None:
        raise ValueError(
            "GxE inference currently supports delete-one chromosome jackknifing "
            "(--njack chr), not chr delete-d/random replicate specifications."
        )
    chromosome = variants["CHR"].astype(str).to_numpy()
    labels = tuple(dict.fromkeys(chromosome.tolist()))
    if len(labels) < 2:
        raise ValueError("GxE --njack chr requires SNPs on at least two chromosomes.")
    lookup = {label: index for index, label in enumerate(labels)}
    blocks = np.asarray([lookup[value] for value in chromosome], dtype=np.int64)
    return blocks, tuple(f"chr_{label}" for label in labels)


def _annotation_vector_aggregates(
    annotations: np.ndarray,
    blocks: np.ndarray,
    values: np.ndarray,
    nblock: int,
    *,
    validate_blocks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full and per-block annotation-weighted vector sums."""
    annot = np.asarray(annotations, dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    if annot.ndim != 2 or vector.shape != (annot.shape[0],):
        raise ValueError("Annotation/vector aggregates received inconsistent shapes.")
    weighted = annot * vector[:, None]
    full = np.asarray(
        [np.dot(annot[:, column], vector) for column in range(annot.shape[1])],
        dtype=np.float64,
    )
    return full, _block_weighted_sums(
        blocks,
        weighted,
        nblock,
        validate_blocks=validate_blocks,
    )


def _posthoc_population_design_moments(
    annotations: np.ndarray,
    d_nxe_x: np.ndarray,
    d_nxe_w: np.ndarray,
    residual_rank: int,
    blocks: np.ndarray,
    block_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-SNP study design rows to full/delete-block traces."""
    annot = np.asarray(annotations, dtype=np.float64)
    x = _as_float_array("GWAS DNXE", d_nxe_x, ndim=1)
    w = _as_float_array("GWIS DNXE", d_nxe_w, ndim=1)
    if x.shape != (annot.shape[0],) or w.shape != (annot.shape[0],):
        raise ValueError("Population DNXE rows are not aligned to annotations.")
    if np.any(x < 0.0) or np.any(w < 0.0):
        raise ValueError("Population DNXE rows must be non-negative.")
    nblock = len(block_labels)
    masses = annot.sum(axis=0, dtype=np.float64)
    block_masses = _block_weighted_sums(blocks, annot, nblock)
    remain = masses[None, :] - block_masses
    if np.any(remain <= 0.0):
        raise ValueError("A GxE jackknife deletion empties an annotation column.")
    full_x, block_x = _annotation_vector_aggregates(annot, blocks, x, nblock)
    full_w, block_w = _annotation_vector_aggregates(annot, blocks, w, nblock)
    rank = float(residual_rank)
    full = rank * np.concatenate([full_x / masses, full_w / masses])
    deleted = rank * np.concatenate(
        [
            (full_x[None, :] - block_x) / remain,
            (full_w[None, :] - block_w) / remain,
        ],
        axis=1,
    )
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(deleted)):
        raise ValueError("Population genetic-by-NxE traces are non-finite.")
    return full, deleted


def _prepare_reference_sufficient_statistics(
    *,
    path: Path,
    payload: dict[str, Any],
    schema_version: int,
    variants: pd.DataFrame,
    annotations: np.ndarray,
    annotation_names: Sequence[str],
    panels: Mapping[str, np.ndarray],
    norm_x: np.ndarray,
    norm_w: np.ndarray,
    diag_x: np.ndarray,
    diag_w: np.ndarray,
    equations: GxENormalEquations,
    block_values: np.ndarray | None = None,
    block_labels: Sequence[str] = (),
) -> _PreparedGxEReference:
    """Collapse fixed per-SNP reference rows to full/delete-block templates."""
    annot = np.asarray(annotations, dtype=np.float64)
    m, k = annot.shape
    masses = annot.sum(axis=0, dtype=np.float64)
    r = float(payload["residual_rank"])
    labels = tuple(str(value) for value in block_labels)

    prepared_blocks: np.ndarray | None = None
    block_masses: np.ndarray | None = None
    deleted_matrices: np.ndarray | None = None
    deleted_traces: np.ndarray | None = None
    has_jackknife = block_values is not None or bool(labels)
    if has_jackknife:
        if block_values is None or len(labels) < 2:
            raise ValueError("Incomplete jackknife inputs for prepared GxE reference.")
        prepared_blocks = np.asarray(block_values, dtype=np.int64)
        if prepared_blocks.shape != (m,):
            raise ValueError("Jackknife block IDs are not aligned to annotations.")
        nblock = len(labels)
        block_masses = _block_weighted_sums(prepared_blocks, annot, nblock)
        remain = masses[None, :] - block_masses
        if np.any(remain <= 0.0):
            block_id, annotation = np.argwhere(remain <= 0.0)[0].tolist()
            raise ValueError(
                f"Deleting block {block_id} empties annotation column {annotation}."
            )

        vector_full: dict[str, np.ndarray] = {}
        vector_block: dict[str, np.ndarray] = {}
        for key, values in {
            "norm_x": norm_x,
            "norm_w": norm_w,
            "diag_x": diag_x,
            "diag_w": diag_w,
        }.items():
            vector_full[key], vector_block[key] = _annotation_vector_aggregates(
                annot,
                prepared_blocks,
                np.asarray(values, dtype=np.float64),
                nblock,
                validate_blocks=False,
            )

        panel_full: dict[str, np.ndarray] = {}
        panel_block: dict[str, np.ndarray] = {}
        for key in ("xx", "xw", "wx", "ww"):
            panel = np.asarray(panels[key], dtype=np.float64)
            if panel.shape != (m, k):
                raise ValueError(f"Reference panel {key!r} has inconsistent shape {panel.shape}.")
            panel_full[key] = np.asarray(
                [
                    [np.dot(annot[:, left], panel[:, source]) for source in range(k)]
                    for left in range(k)
                ],
                dtype=np.float64,
            )
            grouped = np.empty((nblock, k, k), dtype=np.float64)
            for left in range(k):
                grouped[:, left, :] = _block_weighted_sums(
                    prepared_blocks,
                    annot[:, left, None] * panel,
                    nblock,
                    validate_blocks=False,
                )
            panel_block[key] = grouped

        p = 2 * k + 2
        deleted_matrices = np.zeros((nblock, p, p), dtype=np.float64)
        deleted_traces = np.zeros((nblock, p), dtype=np.float64)

        def retained_directed(
            block_id: int,
            forward: str,
            left: int,
            source: int,
        ) -> float:
            cross_sum = (
                panel_full[forward][left, source]
                - panel_block[forward][block_id, left, source]
            )
            return float(
                r
                * r
                * cross_sum
                / (remain[block_id, left] * remain[block_id, source])
            )

        for block_id in range(nblock):
            lhs = deleted_matrices[block_id]
            traces = deleted_traces[block_id]
            for annotation in range(k):
                mass = remain[block_id, annotation]
                gi, wi = annotation, k + annotation
                traces[gi] = (
                    r
                    * (vector_full["norm_x"][annotation] - vector_block["norm_x"][block_id, annotation])
                    / mass
                )
                traces[wi] = (
                    r
                    * (vector_full["norm_w"][annotation] - vector_block["norm_w"][block_id, annotation])
                    / mass
                )
                lhs[gi, 2 * k] = lhs[2 * k, gi] = (
                    r
                    * (vector_full["diag_x"][annotation] - vector_block["diag_x"][block_id, annotation])
                    / mass
                )
                lhs[wi, 2 * k] = lhs[2 * k, wi] = (
                    r
                    * (vector_full["diag_w"][annotation] - vector_block["diag_w"][block_id, annotation])
                    / mass
                )
                lhs[gi, 2 * k + 1] = lhs[2 * k + 1, gi] = traces[gi]
                lhs[wi, 2 * k + 1] = lhs[2 * k + 1, wi] = traces[wi]

            for left in range(k):
                for source in range(k):
                    lhs[left, source] = 0.5 * (
                        retained_directed(block_id, "xx", left, source)
                        + retained_directed(block_id, "xx", source, left)
                    )
                    lhs[k + left, k + source] = 0.5 * (
                        retained_directed(block_id, "ww", left, source)
                        + retained_directed(block_id, "ww", source, left)
                    )
                    cross = 0.5 * (
                        retained_directed(block_id, "xw", left, source)
                        + retained_directed(block_id, "wx", source, left)
                    )
                    lhs[left, k + source] = lhs[k + source, left] = cross

            traces[2 * k] = float(payload["trace_nxe"])
            traces[2 * k + 1] = r
            lhs[2 * k, 2 * k] = float(payload["trace_nxe_sq"])
            lhs[2 * k, 2 * k + 1] = lhs[2 * k + 1, 2 * k] = float(
                payload["trace_nxe"]
            )
            lhs[2 * k + 1, 2 * k + 1] = r
            lhs[:] = 0.5 * (lhs + lhs.T)

    return _PreparedGxEReference(
        path=path,
        payload=dict(payload),
        schema_version=int(schema_version),
        n_variants=len(variants),
        variant_axis={
            column: variants[column].astype(str).to_numpy()
            for column in ("CHR", "SNP", "BP", "A1", "A2")
        },
        annotations=np.array(annot, copy=True),
        annotation_masses=np.array(masses, copy=True),
        annotation_names=tuple(str(value) for value in annotation_names),
        residual_rank=int(payload["residual_rank"]),
        component_names=tuple(equations.component_names),
        normal_matrix=np.array(equations.matrix, copy=True),
        traces=np.array(equations.traces, copy=True),
        block_values=(None if prepared_blocks is None else np.array(prepared_blocks, copy=True)),
        block_labels=labels,
        block_masses=(None if block_masses is None else np.array(block_masses, copy=True)),
        deleted_matrices=(
            None if deleted_matrices is None else np.array(deleted_matrices, copy=True)
        ),
        deleted_traces=(
            None if deleted_traces is None else np.array(deleted_traces, copy=True)
        ),
        population_same_individual_products=(
            None
            if payload.get("population_trace") is None
            else _population_same_individual_products(payload, annotation_names)
        ),
    )


def _equations_from_prepared_scores(
    prepared: _PreparedGxEReference,
    score_x: np.ndarray,
    score_w: np.ndarray,
    *,
    q_nxe: float,
    q_residual: float,
    reference_mode: str = _MATCHED_REFERENCE_MODE,
    study_n_samples: int | None = None,
    study_residual_rank: int | None = None,
    genetic_nxe_traces: np.ndarray | None = None,
    jackknife_genetic_nxe_traces: np.ndarray | None = None,
    trace_nxe: float | None = None,
    trace_nxe_sq: float | None = None,
) -> tuple[GxENormalEquations, tuple[GxENormalEquations, ...]]:
    """Insert phenotype score moments into prepared reference templates."""
    annot = prepared.annotations
    m, k = annot.shape
    sx = _as_float_array("score_x", score_x, ndim=1)
    sw = _as_float_array("score_w", score_w, ndim=1)
    _require_shape("score_x", sx, (m,))
    _require_shape("score_w", sw, (m,))
    if reference_mode not in {
        _MATCHED_REFERENCE_MODE,
        _POPULATION_REFERENCE_MODE,
    }:
        raise ValueError(f"Unsupported GxE reference_mode={reference_mode!r}.")
    population_transfer = reference_mode == _POPULATION_REFERENCE_MODE
    if population_transfer:
        if prepared.population_same_individual_products is None:
            raise ValueError(
                "Reference does not contain population same-individual trace moments."
            )
        if (
            study_n_samples is None
            or study_residual_rank is None
            or genetic_nxe_traces is None
            or trace_nxe is None
            or trace_nxe_sq is None
        ):
            raise ValueError(
                "Population-reference fitting requires study size and exact design traces."
            )
        r = float(int(study_residual_rank))
    else:
        r = float(prepared.residual_rank)
    q_nxe = float(q_nxe)
    q_residual = float(q_residual)
    if not np.isfinite(q_nxe) or not np.isfinite(q_residual):
        raise ValueError("NxE/residual phenotype moments must be finite.")
    if q_nxe < 0.0 or q_residual <= 0.0:
        raise ValueError("Phenotype quadratic moments require q_nxe >= 0 and q_residual > 0.")
    if not np.isclose(q_residual, r, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError(
            "q_residual is inconsistent with the declared normalized marginal-score scale: "
            f"expected residual_rank={int(r)}, got {q_residual:.16g}."
        )

    score_full: dict[str, np.ndarray] = {}
    score_block: dict[str, np.ndarray] = {}
    for key, score in {"x": sx, "w": sw}.items():
        squared = score * score
        if not np.all(np.isfinite(squared)):
            raise ValueError(f"score_{key} squared contains non-finite values.")
        score_full[key] = np.asarray(
            [np.dot(annot[:, column], squared) for column in range(k)],
            dtype=np.float64,
        )
        if prepared.block_values is not None:
            score_block[key] = _block_weighted_sums(
                prepared.block_values,
                annot * squared[:, None],
                len(prepared.block_labels),
                validate_blocks=False,
            )

    rhs = np.zeros(2 * k + 2, dtype=np.float64)
    rhs[:k] = r * score_full["x"] / prepared.annotation_masses
    rhs[k : 2 * k] = r * score_full["w"] / prepared.annotation_masses
    rhs[2 * k] = q_nxe
    rhs[2 * k + 1] = q_residual
    if population_transfer:
        transferred = transfer_reference_normal_equations(
            GxENormalEquations(
                np.array(prepared.normal_matrix, copy=True),
                np.zeros(2 * k + 2, dtype=np.float64),
                np.array(prepared.traces, copy=True),
                prepared.component_names,
            ),
            reference_n_samples=int(prepared.payload["n_samples"]),
            study_n_samples=int(study_n_samples),
            reference_residual_rank=prepared.residual_rank,
            study_residual_rank=int(r),
            same_individual_products=prepared.population_same_individual_products,
            genetic_nxe_traces=np.asarray(genetic_nxe_traces, dtype=np.float64),
            q_nxe=q_nxe,
            q_residual=q_residual,
            trace_nxe=float(trace_nxe),
            trace_nxe_sq=float(trace_nxe_sq),
        )
        full = replace(transferred, rhs=rhs)
    else:
        full = GxENormalEquations(
            np.array(prepared.normal_matrix, copy=True),
            rhs,
            np.array(prepared.traces, copy=True),
            prepared.component_names,
        )

    deleted: list[GxENormalEquations] = []
    if prepared.block_values is not None:
        if (
            prepared.block_masses is None
            or prepared.deleted_matrices is None
            or prepared.deleted_traces is None
        ):
            raise RuntimeError("Prepared jackknife reference is incomplete.")
        remain = prepared.annotation_masses[None, :] - prepared.block_masses
        for block_id in range(len(prepared.block_labels)):
            deleted_rhs = np.zeros(2 * k + 2, dtype=np.float64)
            deleted_rhs[:k] = (
                r
                * (score_full["x"] - score_block["x"][block_id])
                / remain[block_id]
            )
            deleted_rhs[k : 2 * k] = (
                r
                * (score_full["w"] - score_block["w"][block_id])
                / remain[block_id]
            )
            deleted_rhs[2 * k] = q_nxe
            deleted_rhs[2 * k + 1] = q_residual
            deleted_reference = GxENormalEquations(
                np.array(prepared.deleted_matrices[block_id], copy=True),
                np.zeros(2 * k + 2, dtype=np.float64),
                np.array(prepared.deleted_traces[block_id], copy=True),
                prepared.component_names,
            )
            if population_transfer:
                if jackknife_genetic_nxe_traces is None:
                    raise ValueError(
                        "Population-reference jackknife requires study delete-block design traces."
                    )
                deleted_template = transfer_reference_normal_equations(
                    deleted_reference,
                    reference_n_samples=int(prepared.payload["n_samples"]),
                    study_n_samples=int(study_n_samples),
                    reference_residual_rank=prepared.residual_rank,
                    study_residual_rank=int(r),
                    same_individual_products=(
                        prepared.population_same_individual_products
                    ),
                    genetic_nxe_traces=np.asarray(
                        jackknife_genetic_nxe_traces[block_id],
                        dtype=np.float64,
                    ),
                    q_nxe=q_nxe,
                    q_residual=q_residual,
                    trace_nxe=float(trace_nxe),
                    trace_nxe_sq=float(trace_nxe_sq),
                )
                deleted.append(replace(deleted_template, rhs=deleted_rhs))
            else:
                deleted.append(replace(deleted_reference, rhs=deleted_rhs))
    return full, tuple(deleted)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=r"\s+",
        compression="infer",
        dtype={"CHR": str, "SNP": str, "A1": str, "A2": str},
    )


def _aligned_panel(
    path: Path,
    variants: pd.DataFrame,
    value_columns: Sequence[str],
) -> np.ndarray:
    panel = _read_table(path)
    expected = ["CHR", "SNP", "BP", *value_columns]
    observed = panel.columns.astype(str).tolist()
    if observed != expected or len(set(observed)) != len(observed):
        raise ValueError(
            f"{path} must contain exactly the ordered columns {expected}; observed {observed}."
        )
    if len(panel) != len(variants):
        raise ValueError(f"{path} has {len(panel)} rows; expected {len(variants)}.")
    for column in ("CHR", "SNP", "BP"):
        if not np.array_equal(
            panel[column].astype(str).to_numpy(),
            variants[column].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"{path} is not in the reference variant order ({column} mismatch)."
            )
    values = panel.loc[:, list(value_columns)].to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains a non-finite panel value.")
    return values


def _validate_reference_design_diagnostics(
    payload: Mapping[str, Any],
    diagonal: pd.DataFrame,
    annotations: np.ndarray,
    masses: np.ndarray,
    annotation_count: int,
    residual_rank: int,
) -> None:
    transform = payload["environment_transform"]
    expected_ss = float(payload["n_samples"] - int(transform["ddof"]))
    if not np.isclose(
        float(transform["analysis_mean"]), 0.0, rtol=0.0, atol=2.0e-10
    ):
        raise ValueError("Reference standardized environment has nonzero analysis mean.")
    if not np.isclose(
        float(transform["analysis_sum_squares"]),
        expected_ss,
        rtol=2.0e-10,
        atol=2.0e-8,
    ):
        raise ValueError("Reference environment sum of squares is inconsistent.")
    diagnostics = payload.get("feature_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("Reference is missing feature_diagnostics.")
    m = len(diagonal)
    if (
        diagnostics.get("valid_additive_columns") != m
        or diagnostics.get("valid_interaction_columns") != m
    ):
        raise ValueError("Reference feature diagnostics do not cover every SNP.")
    for key in (
        "max_projection_leakage_additive",
        "max_projection_leakage_interaction",
    ):
        value = diagnostics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
            or value > 1.0e-9
        ):
            raise ValueError(f"Reference feature diagnostic {key} is invalid.")
    expected_traces = {
        "kernel_traces_additive": (
            float(residual_rank)
            * annotations.T
            @ diagonal["NORM_X"].to_numpy(dtype=np.float64)
            / masses
        ),
        "kernel_traces_interaction": (
            float(residual_rank)
            * annotations.T
            @ diagonal["NORM_W"].to_numpy(dtype=np.float64)
            / masses
        ),
    }
    for key, expected in expected_traces.items():
        observed = _as_float_array(
            f"feature_diagnostics[{key}]", diagnostics.get(key), ndim=1
        )
        _require_shape(f"feature_diagnostics[{key}]", observed, (annotation_count,))
        if not np.allclose(observed, expected, rtol=5.0e-10, atol=1.0e-8):
            raise ValueError(f"Reference feature diagnostic {key} disagrees with per-SNP rows.")


def _validate_reference_for_fit(
    reference_manifest: str | Path,
    njack: str | int,
) -> _PreparedGxEReference:
    path = Path(reference_manifest).expanduser().resolve()
    ref = _load_json(path)
    if ref.get("kind") != _REFERENCE_KIND or ref.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Only the current schema-v4 SUMMIT GxE reference is supported.")
    for key in ("n_samples", "fixed_effect_rank_excluding_intercept", "residual_rank"):
        value = ref.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Reference {key} must be a JSON integer.")
    n = int(ref["n_samples"])
    fixed_rank = int(ref["fixed_effect_rank_excluding_intercept"])
    residual_rank = int(ref["residual_rank"])
    if n <= 0 or fixed_rank < 0 or residual_rank != n - fixed_rank - 1:
        raise ValueError("Reference sample size and fixed-effect rank are inconsistent.")
    if ref.get("ld_scale") != "cross_product_over_rank_squared":
        raise ValueError("Reference has an unsupported LD-score scale.")
    if ref.get("null_corrected") is not False:
        raise ValueError("Current GxE references must store raw, non-offset per-SNP scores.")
    feature_convention = ref.get("feature_convention")
    if feature_convention not in {"standardized_projected", "raw_projected"}:
        raise ValueError("Reference has an unsupported feature convention.")
    if ref.get("kernel_mode") != feature_convention:
        raise ValueError("Reference kernel_mode must equal its canonical feature convention.")
    if ref.get("genotype_scale") not in {"sample", "hwe"}:
        raise ValueError("Reference has an unsupported genotype scale.")
    if ref.get("annotation_value_dtype") != "float64":
        raise ValueError("Reference annotations must use the schema-v4 float64 contract.")
    transform = ref.get("environment_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Reference is missing environment_transform.")
    if (
        transform.get("standardized") is not True
        or transform.get("units") != "per_environment_sd"
        or transform.get("ddof") not in (0, 1)
    ):
        raise ValueError("Reference environment transform is unsupported.")
    for key in ("raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares"):
        value = transform.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Reference environment_transform[{key!r}] is invalid.")
    if float(transform["raw_sd"]) <= 0.0:
        raise ValueError("Reference environment standard deviation must be positive.")

    names_raw = ref.get("annotation_names")
    if not isinstance(names_raw, list):
        raise ValueError("Reference annotation_names must be a JSON list.")
    names = tuple(_validate_gxe_annotation_names(names_raw))
    files = ref.get("files")
    required_files = {"xx", "xw", "wx", "ww", "diagonal"}
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise ValueError(
            f"Reference files must contain exactly {sorted(required_files)}."
        )
    diagonal_path = _resolve_path(path, str(files["diagonal"]))
    diagonal = _read_table(diagonal_path)
    weight_columns = [f"ANNOT_{index}" for index in range(len(names))]
    expected_columns = [
        "CHR", "SNP", "BP", "A1", "A2",
        "NORM_X", "NORM_W", "SCALE_X", "SCALE_W",
        "DNXE_X", "DNXE_W", "CORR_XW", *weight_columns,
    ]
    observed_columns = diagonal.columns.astype(str).tolist()
    if observed_columns != expected_columns or len(set(observed_columns)) != len(observed_columns):
        raise ValueError(
            f"{diagonal_path} must contain exactly the ordered columns {expected_columns}."
        )
    if len(diagonal) < 2 or diagonal["SNP"].duplicated().any():
        raise ValueError("Reference diagonal is empty or contains duplicate SNP IDs.")
    variants = diagonal.loc[:, ["CHR", "SNP", "BP", "A1", "A2"]].copy()
    annotations = diagonal.loc[:, weight_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(annotations)) or np.any(annotations < 0.0):
        raise ValueError("Reference annotations must be finite and non-negative.")
    masses = annotations.sum(axis=0, dtype=np.float64)
    declared_masses = _as_float_array(
        "reference annotation_masses", ref.get("annotation_masses"), ndim=1
    )
    _require_shape("reference annotation_masses", declared_masses, masses.shape)
    if np.any(masses <= 0.0) or not np.allclose(
        masses, declared_masses, rtol=5.0e-10, atol=1.0e-8
    ):
        raise ValueError("Reference annotation masses disagree with the per-SNP rows.")
    numeric_columns = [
        "NORM_X", "NORM_W", "SCALE_X", "SCALE_W", "DNXE_X", "DNXE_W", "CORR_XW"
    ]
    numeric = diagonal.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Reference diagonal contains non-finite values.")
    for column in ("NORM_X", "NORM_W", "SCALE_X", "SCALE_W"):
        if np.any(diagonal[column].to_numpy(dtype=np.float64) <= 0.0):
            raise ValueError(f"Reference {column} values must be positive.")
    norm_bound = np.sqrt(
        diagonal["NORM_X"].to_numpy(dtype=np.float64)
        * diagonal["NORM_W"].to_numpy(dtype=np.float64)
    )
    if np.any(np.abs(diagonal["CORR_XW"].to_numpy(dtype=np.float64)) > norm_bound + 1.0e-8):
        raise ValueError("Reference additive/interaction diagonals violate Cauchy-Schwarz.")
    if feature_convention == "standardized_projected":
        for column in ("NORM_X", "NORM_W"):
            if not np.allclose(
                diagonal[column].to_numpy(dtype=np.float64),
                1.0,
                rtol=1.0e-9,
                atol=1.0e-9,
            ):
                raise ValueError(f"Standardized reference violates {column}=1.")
    elif not (
        np.allclose(diagonal["SCALE_X"], 1.0, rtol=0.0, atol=1.0e-12)
        and np.allclose(diagonal["SCALE_W"], 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Raw projected references must store unit post-projection scales.")
    _validate_reference_design_diagnostics(
        ref, diagonal, annotations, masses, len(names), residual_rank
    )

    panels = {
        key: _aligned_panel(_resolve_path(path, str(files[key])), variants, names)
        for key in ("xx", "xw", "wx", "ww")
    }
    trace_nxe = float(ref.get("trace_nxe"))
    trace_nxe_sq = float(ref.get("trace_nxe_sq"))
    if not np.isfinite(trace_nxe) or not np.isfinite(trace_nxe_sq):
        raise ValueError("Reference NxE traces must be finite.")
    if trace_nxe < 0.0 or trace_nxe_sq < 0.0:
        raise ValueError("Reference NxE traces must be non-negative.")
    template = assemble_normal_equations(
        annotations=annotations,
        score_x=np.zeros(len(variants), dtype=np.float64),
        score_w=np.zeros(len(variants), dtype=np.float64),
        ld_xx=panels["xx"],
        ld_xw=panels["xw"],
        ld_wx=panels["wx"],
        ld_ww=panels["ww"],
        norm_x=diagonal["NORM_X"].to_numpy(dtype=np.float64),
        norm_w=diagonal["NORM_W"].to_numpy(dtype=np.float64),
        diag_nxe_x=diagonal["DNXE_X"].to_numpy(dtype=np.float64),
        diag_nxe_w=diagonal["DNXE_W"].to_numpy(dtype=np.float64),
        residual_rank=residual_rank,
        q_nxe=0.0,
        q_residual=float(residual_rank),
        trace_nxe=trace_nxe,
        trace_nxe_sq=trace_nxe_sq,
        annotation_names=names,
        ld_scale=str(ref["ld_scale"]),
    )
    blocks, labels = _posthoc_inference_blocks(variants, njack)
    return _prepare_reference_sufficient_statistics(
        path=path,
        payload=ref,
        schema_version=_SCHEMA_VERSION,
        variants=variants,
        annotations=annotations,
        annotation_names=names,
        panels=panels,
        norm_x=diagonal["NORM_X"].to_numpy(dtype=np.float64),
        norm_w=diagonal["NORM_W"].to_numpy(dtype=np.float64),
        diag_x=diagonal["DNXE_X"].to_numpy(dtype=np.float64),
        diag_w=diagonal["DNXE_W"].to_numpy(dtype=np.float64),
        equations=template,
        block_values=blocks,
        block_labels=labels,
    )


def _load_score_table(
    path: Path,
    prepared: _PreparedGxEReference,
    *,
    label: str,
    n_samples: int,
    residual_rank: int,
    population_transfer: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    table = _read_table(path)
    expected_columns = [
        "CHR", "SNP", "BP", "A1", "A2", "N", "DF", "SCORE_MODE", "SCORE"
    ]
    if population_transfer:
        expected_columns.append("DNXE")
    observed_columns = table.columns.astype(str).tolist()
    if observed_columns != expected_columns or len(set(observed_columns)) != len(observed_columns):
        raise ValueError(
            f"{label} must contain exactly the ordered columns {expected_columns}; "
            f"observed {observed_columns}."
        )
    if len(table) != prepared.n_variants:
        raise ValueError(
            f"{label} has {len(table)} rows; expected {prepared.n_variants}."
        )
    for column in ("CHR", "SNP", "BP", "A1", "A2"):
        if not np.array_equal(
            table[column].astype(str).to_numpy(), prepared.variant_axis[column]
        ):
            raise ValueError(
                f"{label} is not aligned to the reference ({column} mismatch)."
            )
    if not bool(
        (table["SCORE_MODE"].astype("string").str.lower() == "marginal_cross_product").all()
    ):
        raise ValueError(
            f"{label} must contain marginal cross-products, not conditional interaction statistics."
        )
    if not np.all(pd.to_numeric(table["N"], errors="raise") == n_samples):
        raise ValueError(f"{label} N does not match phenotype moments.")
    if not np.all(pd.to_numeric(table["DF"], errors="raise") == residual_rank):
        raise ValueError(f"{label} DF does not match phenotype moments.")
    score = _as_float_array(f"{label} SCORE", table["SCORE"], ndim=1)
    d_nxe = (
        _as_float_array(f"{label} DNXE", table["DNXE"], ndim=1)
        if population_transfer
        else None
    )
    return score, d_nxe


def _fit_prepared_trait(
    prepared: _PreparedGxEReference,
    phenotype_input: GxEPhenotypeInput,
    *,
    allow_ill_conditioned: bool,
    max_condition: float,
) -> tuple[GxEFitResult, GxENormalEquations]:
    moments_path = Path(phenotype_input.phenotype_moments).expanduser().resolve()
    moments = _load_json(moments_path)
    if moments.get("kind") != _MOMENTS_KIND or moments.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Only current schema-v4 GxE phenotype moments are supported.")
    if moments.get("feature_convention") != prepared.payload.get("feature_convention"):
        raise ValueError("Reference and phenotype feature conventions disagree.")
    if moments.get("score_definition") != (
        "feature_transpose_residualized_y_over_sqrt_residual_rank"
    ):
        raise ValueError("Phenotype moments use an unsupported score definition.")
    reference_mode = moments.get("reference_mode", _MATCHED_REFERENCE_MODE)
    if reference_mode not in {_MATCHED_REFERENCE_MODE, _POPULATION_REFERENCE_MODE}:
        raise ValueError(f"Unsupported phenotype reference_mode={reference_mode!r}.")
    population_transfer = reference_mode == _POPULATION_REFERENCE_MODE
    study_n = moments.get("n_samples")
    study_rank = moments.get("residual_rank")
    if (
        isinstance(study_n, bool)
        or not isinstance(study_n, int)
        or isinstance(study_rank, bool)
        or not isinstance(study_rank, int)
        or study_n <= 0
        or study_rank <= 0
        or study_rank >= study_n
    ):
        raise ValueError("Phenotype sample size and residual rank are inconsistent.")
    if not population_transfer and (
        study_n != prepared.payload["n_samples"]
        or study_rank != prepared.residual_rank
    ):
        raise ValueError("Matched-cohort phenotype size/rank disagree with the reference.")
    if population_transfer:
        fixed_rank = moments.get("fixed_effect_rank_excluding_intercept")
        if (
            isinstance(fixed_rank, bool)
            or not isinstance(fixed_rank, int)
            or fixed_rank < 0
            or study_rank != study_n - fixed_rank - 1
        ):
            raise ValueError("Population phenotype fixed-effect rank is inconsistent.")
        if prepared.population_same_individual_products is None:
            raise ValueError("Reference lacks population trace-transfer moments.")

    files = moments.get("files")
    if not isinstance(files, Mapping) or set(files) != {"gwas", "gwis"}:
        raise ValueError("Phenotype moments files must contain exactly gwas and gwis.")
    supplied = {
        "gwas": Path(phenotype_input.gwas_scores).expanduser().resolve(),
        "gwis": Path(phenotype_input.gwis_scores).expanduser().resolve(),
    }
    for key, path in supplied.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    score_x, d_nxe_x = _load_score_table(
        supplied["gwas"],
        prepared,
        label="GWAS",
        n_samples=study_n,
        residual_rank=study_rank,
        population_transfer=population_transfer,
    )
    score_w, d_nxe_w = _load_score_table(
        supplied["gwis"],
        prepared,
        label="GWIS",
        n_samples=study_n,
        residual_rank=study_rank,
        population_transfer=population_transfer,
    )

    genetic_nxe: np.ndarray | None = None
    deleted_genetic_nxe: np.ndarray | None = None
    trace_nxe: float | None = None
    trace_nxe_sq: float | None = None
    if population_transfer:
        assert d_nxe_x is not None and d_nxe_w is not None
        genetic_nxe = _study_population_design_moments(
            moments, prepared.annotation_names
        )
        if prepared.block_values is None:
            raise RuntimeError("Prepared GxE reference has no inference blocks.")
        reduced_full, deleted_genetic_nxe = _posthoc_population_design_moments(
            prepared.annotations,
            d_nxe_x,
            d_nxe_w,
            study_rank,
            prepared.block_values,
            prepared.block_labels,
        )
        if not np.allclose(
            reduced_full, genetic_nxe, rtol=5.0e-10, atol=1.0e-8
        ):
            raise ValueError(
                "Population per-SNP DNXE rows disagree with full design traces."
            )
        trace_nxe = float(moments.get("trace_nxe"))
        trace_nxe_sq = float(moments.get("trace_nxe_sq"))
        if not np.isfinite(trace_nxe) or not np.isfinite(trace_nxe_sq):
            raise ValueError("Population phenotype NxE traces must be finite.")

    equations, deleted_equations = _equations_from_prepared_scores(
        prepared,
        score_x,
        score_w,
        q_nxe=float(moments.get("q_nxe")),
        q_residual=float(moments.get("q_residual")),
        reference_mode=str(reference_mode),
        study_n_samples=study_n,
        study_residual_rank=study_rank,
        genetic_nxe_traces=genetic_nxe,
        jackknife_genetic_nxe_traces=deleted_genetic_nxe,
        trace_nxe=trace_nxe,
        trace_nxe_sq=trace_nxe_sq,
    )
    fit = solve_normal_equations(
        equations,
        allow_ill_conditioned=allow_ill_conditioned,
        max_condition=max_condition,
    )
    residual_fraction_raw = moments.get("phenotype_residual_variance_fraction")
    if residual_fraction_raw is not None:
        residual_fraction = float(residual_fraction_raw)
        if not np.isfinite(residual_fraction) or not (
            0.0 < residual_fraction <= 1.0 + 1.0e-8
        ):
            raise ValueError("Invalid phenotype_residual_variance_fraction.")
        fit = replace(
            fit, phenotype_residual_variance_fraction=residual_fraction
        )
    replicate_fits = [
        solve_normal_equations(
            deleted,
            allow_ill_conditioned=allow_ill_conditioned,
            max_condition=max_condition,
        )
        for deleted in deleted_equations
    ]
    coefficient_replicates = np.asarray(
        [replicate.coefficients for replicate in replicate_fits], dtype=np.float64
    )
    proportion_replicates = np.asarray(
        [replicate.proportions for replicate in replicate_fits], dtype=np.float64
    )
    nblock = len(deleted_equations)
    def jackknife_se(replicates: np.ndarray) -> np.ndarray:
        center = replicates.mean(axis=0)
        return np.sqrt(
            (nblock - 1.0)
            / nblock
            * np.sum((replicates - center) ** 2, axis=0)
        )
    fit = replace(
        fit,
        coefficient_standard_errors=jackknife_se(coefficient_replicates),
        proportion_standard_errors=jackknife_se(proportion_replicates),
        jackknife_coefficients=coefficient_replicates,
        jackknife_proportions=proportion_replicates,
        jackknife_block_labels=prepared.block_labels,
    )
    return fit, equations


def fit_from_files(
    reference_manifest: str | Path,
    phenotype_moments: str | Path,
    gwas_scores: str | Path,
    gwis_scores: str | Path,
    *,
    njack: str | int = "chr",
    allow_ill_conditioned: bool = False,
    max_condition: float = 1.0e12,
) -> tuple[GxEFitResult, GxENormalEquations]:
    """Fit one phenotype from fixed per-SNP reference and trait summaries."""
    return fit_many_from_files(
        reference_manifest,
        {
            "phenotype": GxEPhenotypeInput(
                phenotype_moments=phenotype_moments,
                gwas_scores=gwas_scores,
                gwis_scores=gwis_scores,
            )
        },
        njack=njack,
        allow_ill_conditioned=allow_ill_conditioned,
        max_condition=max_condition,
    )["phenotype"]


def fit_many_from_files(
    reference_manifest: str | Path,
    phenotype_inputs: Mapping[str, GxEPhenotypeInput],
    *,
    njack: str | int = "chr",
    allow_ill_conditioned: bool = False,
    max_condition: float = 1.0e12,
) -> dict[str, tuple[GxEFitResult, GxENormalEquations]]:
    """Fit several phenotype triplets after preparing one reference once."""
    if not isinstance(phenotype_inputs, Mapping) or not phenotype_inputs:
        raise ValueError("phenotype_inputs must be a non-empty mapping.")
    normalized: list[tuple[str, GxEPhenotypeInput]] = []
    triplets: set[tuple[Path, Path, Path]] = set()
    for raw_name, phenotype_input in phenotype_inputs.items():
        name = str(raw_name)
        if not _FIT_BATCH_NAME.fullmatch(name):
            raise ValueError(f"Invalid GxE batch phenotype name {name!r}.")
        if not isinstance(phenotype_input, GxEPhenotypeInput):
            raise TypeError(
                "Every phenotype_inputs value must be a GxEPhenotypeInput."
            )
        triplet = (
            Path(phenotype_input.phenotype_moments).expanduser().resolve(),
            Path(phenotype_input.gwas_scores).expanduser().resolve(),
            Path(phenotype_input.gwis_scores).expanduser().resolve(),
        )
        if triplet in triplets:
            raise ValueError("GxE phenotype input triplets must be unique.")
        triplets.add(triplet)
        normalized.append((name, phenotype_input))
    prepared = _validate_reference_for_fit(reference_manifest, njack)
    return {
        name: _fit_prepared_trait(
            prepared,
            phenotype_input,
            allow_ill_conditioned=allow_ill_conditioned,
            max_condition=max_condition,
        )
        for name, phenotype_input in normalized
    }


def load_fit_batch_manifest(
    manifest_path: str | Path,
) -> tuple[Path, tuple[GxEFitBatchEntry, ...]]:
    """Validate a no-overwrite batch-fit manifest."""
    path = Path(manifest_path).expanduser().resolve()
    payload = _load_json(path)
    expected_root = {"kind", "schema_version", "reference", "traits"}
    if set(payload) != expected_root:
        raise ValueError(
            f"GxE fit-batch manifest must contain exactly {sorted(expected_root)}."
        )
    version = payload.get("schema_version")
    if (
        payload.get("kind") != _FIT_BATCH_KIND
        or isinstance(version, bool)
        or version != _FIT_BATCH_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported GxE fit-batch manifest schema_version.")
    reference_value = payload.get("reference")
    if not isinstance(reference_value, str) or not reference_value:
        raise ValueError("GxE fit-batch reference must be a non-empty path.")
    traits = payload.get("traits")
    if not isinstance(traits, list) or not traits:
        raise ValueError("GxE fit-batch traits must be a non-empty list.")

    def resolve(value: str) -> Path:
        return _resolve_path(path, value)

    entries: list[GxEFitBatchEntry] = []
    names: set[str] = set()
    outputs: set[Path] = set()
    inputs: set[tuple[Path, Path, Path]] = set()
    expected_trait = {"name", "moments", "gwas", "gwis", "out"}
    for index, record in enumerate(traits):
        if not isinstance(record, dict) or set(record) != expected_trait:
            raise ValueError(
                f"GxE fit-batch trait {index} must contain exactly {sorted(expected_trait)}."
            )
        if any(
            not isinstance(record[key], str) or not record[key]
            for key in expected_trait
        ):
            raise ValueError(
                f"GxE fit-batch trait {index} values must be non-empty strings."
            )
        name = record["name"]
        if not _FIT_BATCH_NAME.fullmatch(name) or name in names:
            raise ValueError(f"Invalid or duplicate GxE trait name {name!r}.")
        output = resolve(record["out"])
        triplet = (
            resolve(record["moments"]),
            resolve(record["gwas"]),
            resolve(record["gwis"]),
        )
        if output in outputs:
            raise ValueError(f"Duplicate GxE batch output prefix {output}.")
        if triplet in inputs:
            raise ValueError("Duplicate GxE batch phenotype input triplet.")
        names.add(name)
        outputs.add(output)
        inputs.add(triplet)
        entries.append(
            GxEFitBatchEntry(
                name=name,
                phenotype_input=GxEPhenotypeInput(
                    phenotype_moments=triplet[0],
                    gwas_scores=triplet[1],
                    gwis_scores=triplet[2],
                ),
                output_prefix=output,
            )
        )
    if len({entry.output_prefix.parent for entry in entries}) != 1:
        raise ValueError("All GxE batch outputs must share one parent directory.")
    planned = [
        output
        for entry in entries
        for output in (
            Path(str(entry.output_prefix) + ".gxe.results.tsv"),
            Path(str(entry.output_prefix) + ".gxe.fit.json"),
        )
    ]
    existing = [output for output in planned if os.path.lexists(output)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing GxE batch output(s): "
            + ", ".join(str(output) for output in existing)
        )
    return resolve(reference_value), tuple(entries)

def write_fit(
    prefix: str | Path,
    fit: GxEFitResult,
    equations: GxENormalEquations,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    prefix = Path(prefix)
    table_path = Path(str(prefix) + ".gxe.results.tsv")
    json_path = Path(str(prefix) + ".gxe.fit.json")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in (table_path, json_path) if os.path.lexists(path)
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing GxE fit output(s); choose a new --out prefix or pass "
            f"--gxe-overwrite explicitly: {', '.join(str(path) for path in existing)}"
        )
    frame = pd.DataFrame(
        {
            "component": fit.component_names,
            "coefficient": fit.coefficients,
            "coefficient_se": (
                fit.coefficient_standard_errors
                if fit.coefficient_standard_errors is not None
                else np.full(len(fit.component_names), np.nan)
            ),
            "kernel_trace": equations.traces,
            "variance_contribution": fit.contributions,
            "proportion": fit.proportions,
            "proportion_se": (
                fit.proportion_standard_errors
                if fit.proportion_standard_errors is not None
                else np.full(len(fit.component_names), np.nan)
            ),
            "original_scale_proportion": (
                fit.proportions * fit.phenotype_residual_variance_fraction
                if fit.phenotype_residual_variance_fraction is not None
                else np.full(len(fit.component_names), np.nan)
            ),
        }
    )
    frame["coefficient_z"] = frame["coefficient"] / frame["coefficient_se"]
    frame["coefficient_p"] = [
        math.erfc(abs(value) / math.sqrt(2.0)) if np.isfinite(value) else np.nan
        for value in frame["coefficient_z"]
    ]
    frame["proportion_z"] = frame["proportion"] / frame["proportion_se"]
    frame["original_scale_se"] = (
        frame["proportion_se"] * fit.phenotype_residual_variance_fraction
        if fit.phenotype_residual_variance_fraction is not None
        else np.nan
    )
    diag_scale = np.sqrt(np.maximum(np.diag(equations.matrix), 0.0))
    denom = np.outer(diag_scale, diag_scale)
    kernel_correlations = np.full_like(equations.matrix, np.nan, dtype=np.float64)
    np.divide(
        equations.matrix,
        denom,
        out=kernel_correlations,
        where=denom > 0.0,
    )
    nxe_index = len(fit.component_names) - 2
    residual_index = len(fit.component_names) - 1
    nxe_residual_correlation = float(kernel_correlations[nxe_index, residual_index])
    payload: dict[str, Any] = {
        "kind": "summit.gxe.fit",
        "schema_version": _SCHEMA_VERSION,
        "component_names": list(fit.component_names),
        "rank": fit.rank,
        "condition_number": fit.condition_number,
        "relative_residual": fit.relative_residual,
        "solve_method": fit.solve_method,
        "normal_symmetry_error": fit.normal_symmetry_error,
        "cauchy_schwarz_max_violation": fit.cauchy_schwarz_max_violation,
        "component_influence": (
            None
            if fit.component_influence is None
            else fit.component_influence.tolist()
        ),
        "identifiable": fit.identifiable,
        "normal_equation_diagnostics": fit.normal_equation_diagnostics,
        "singular_values": fit.singular_values.tolist(),
        "normal_eigenvalues": fit.normal_eigenvalues.tolist(),
        "normal_matrix": fit.normal_matrix.tolist(),
        "kernel_correlation_matrix": kernel_correlations.tolist(),
        "nxe_residual_kernel_correlation": nxe_residual_correlation,
        "rhs": fit.rhs.tolist(),
        "kernel_traces": equations.traces.tolist(),
        "coefficients": fit.coefficients.tolist(),
        "variance_contributions": fit.contributions.tolist(),
        "proportions": fit.proportions.tolist(),
        "original_scale_proportions": (
            (fit.proportions * fit.phenotype_residual_variance_fraction).tolist()
            if fit.phenotype_residual_variance_fraction is not None
            else None
        ),
        "jackknife_block_labels": list(fit.jackknife_block_labels),
        "jackknife_method": _POSTHOC_JACKKNIFE_METHOD,
        "phenotype_residual_variance_fraction": fit.phenotype_residual_variance_fraction,
    }
    if fit.coefficient_standard_errors is not None:
        payload["coefficient_standard_errors"] = (
            fit.coefficient_standard_errors.tolist()
        )
    if fit.proportion_standard_errors is not None:
        payload["proportion_standard_errors"] = (
            fit.proportion_standard_errors.tolist()
        )
        payload["original_scale_proportion_standard_errors"] = (
            (fit.proportion_standard_errors * fit.phenotype_residual_variance_fraction).tolist()
            if fit.phenotype_residual_variance_fraction is not None
            else None
        )
    if fit.jackknife_coefficients is not None:
        payload["jackknife_coefficients"] = fit.jackknife_coefficients.tolist()
    if fit.jackknife_proportions is not None:
        payload["jackknife_proportions"] = fit.jackknife_proportions.tolist()
    def json_finite(value: Any) -> Any:
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, dict):
            return {key: json_finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_finite(item) for item in value]
        return value

    payload = json_finite(payload)

    # Build the complete two-file fit in a private staging directory.  For the
    # default no-overwrite contract, publish with hard-link no-replace and put
    # the JSON manifest last.  Rollback removes only the exact inodes linked by
    # this call, never a competitor which appeared after preflight.
    published: list[tuple[Path, int, int]] = []
    with tempfile.TemporaryDirectory(
        prefix=".gxe-fit-stage-", dir=table_path.parent
    ) as stage_name:
        stage_dir = Path(stage_name)
        os.chmod(stage_dir, 0o700)
        staged_table = stage_dir / table_path.name
        staged_json = stage_dir / json_path.name
        table_fd = os.open(
            staged_table,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(table_fd, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep="\t", index=False, float_format="%.12g")
        json_fd = os.open(
            staged_json,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(json_fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        def verify_published() -> None:
            for path, device, inode in published:
                try:
                    observed = path.stat(follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"A published GxE fit output disappeared before commit: {path}."
                    ) from exc
                if observed.st_dev != device or observed.st_ino != inode:
                    raise RuntimeError(
                        f"A published GxE fit output was concurrently replaced: {path}."
                    )

        try:
            for staged, final in (
                (staged_table, table_path),
                (staged_json, json_path),
            ):
                if overwrite:
                    os.replace(staged, final)
                    continue
                if final == json_path:
                    verify_published()
                staged_stat = staged.stat()
                try:
                    os.link(staged, final)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Refusing to overwrite concurrently created GxE fit output: {final}."
                    ) from exc
                published.append((final, staged_stat.st_dev, staged_stat.st_ino))
            if not overwrite:
                verify_published()
        except Exception:
            if not overwrite:
                for path, device, inode in reversed(published):
                    try:
                        observed = path.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if observed.st_dev == device and observed.st_ino == inode:
                        path.unlink(missing_ok=True)
            raise
    return table_path, json_path


def write_fits(
    outputs: Mapping[
        str,
        tuple[str | Path, GxEFitResult, GxENormalEquations],
    ],
) -> dict[str, tuple[Path, Path]]:
    """Publish a complete multi-trait fit as one no-overwrite transaction."""
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("outputs must be a non-empty mapping.")
    normalized: list[
        tuple[str, Path, GxEFitResult, GxENormalEquations, Path, Path]
    ] = []
    final_paths: set[Path] = set()
    parents: set[Path] = set()
    for raw_name, value in outputs.items():
        name = str(raw_name)
        if not _FIT_BATCH_NAME.fullmatch(name):
            raise ValueError(f"Invalid GxE batch phenotype name {name!r}.")
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError("Every batch output must be (prefix, fit, equations).")
        prefix_raw, fit, equations = value
        if not isinstance(fit, GxEFitResult) or not isinstance(
            equations, GxENormalEquations
        ):
            raise TypeError("Every batch output must contain GxEFitResult/GxENormalEquations.")
        prefix = Path(prefix_raw).expanduser().resolve()
        table = Path(str(prefix) + ".gxe.results.tsv")
        manifest = Path(str(prefix) + ".gxe.fit.json")
        if table in final_paths or manifest in final_paths:
            raise ValueError("GxE batch output paths must be unique.")
        final_paths.update((table, manifest))
        parents.add(table.parent)
        normalized.append((name, prefix, fit, equations, table, manifest))
    if len(parents) != 1:
        raise ValueError("All GxE batch output prefixes must share one parent directory.")
    parent = next(iter(parents))
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in final_paths if os.path.lexists(path))
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing GxE batch fit output(s): "
            + ", ".join(str(path) for path in existing)
        )

    published: list[tuple[Path, int, int]] = []
    result: dict[str, tuple[Path, Path]] = {}
    with tempfile.TemporaryDirectory(prefix=".gxe-fit-batch-stage-", dir=parent) as stage_name:
        stage_dir = Path(stage_name)
        os.chmod(stage_dir, 0o700)
        staged_pairs: list[tuple[str, Path, Path, Path, Path]] = []
        for index, (name, _, fit, equations, table, manifest) in enumerate(normalized):
            staged_table, staged_manifest = write_fit(
                stage_dir / f"{index:06d}",
                fit,
                equations,
            )
            staged_pairs.append(
                (name, staged_table, staged_manifest, table, manifest)
            )
        def verify_published() -> None:
            for path, device, inode in published:
                try:
                    observed = path.stat(follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"A published GxE batch output disappeared before commit: {path}."
                    ) from exc
                if observed.st_dev != device or observed.st_ino != inode:
                    raise RuntimeError(
                        f"A published GxE batch output was concurrently replaced: {path}."
                    )

        publication_order = [
            (staged_table, table)
            for _, staged_table, _, table, _ in staged_pairs
        ] + [
            (staged_manifest, manifest)
            for _, _, staged_manifest, _, manifest in staged_pairs
        ]
        try:
            for staged, final in publication_order:
                verify_published()
                staged_stat = staged.stat()
                try:
                    os.link(staged, final)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Refusing to overwrite concurrently created GxE batch output: {final}."
                    ) from exc
                published.append((final, staged_stat.st_dev, staged_stat.st_ino))
            verify_published()
        except Exception:
            for path, device, inode in reversed(published):
                try:
                    observed = path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if observed.st_dev == device and observed.st_ino == inode:
                    path.unlink(missing_ok=True)
            raise
        for name, _, _, table, manifest in staged_pairs:
            result[name] = (table, manifest)
    return result
