"""Summary-statistics method-of-moments estimation for G + GxE + NxE.

The implementation in this module is deliberately based on raw marginal score
cross-products, not on conditional interaction test statistics.  It reconstructs
the same normal equations as an individual-level GENIE analysis after a fixed
projection has been chosen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..ldscore.gwe_ldscore import (
    _FEATURE_CACHE_ARRAY_DTYPES,
    _validate_backend_provenance,
    _validate_feature_cache_semantics,
    _validate_gxe_annotation_names,
)


_REFERENCE_KIND = "summit.gxe.reference"
_MOMENTS_KIND = "summit.gxe.phenotype_moments"
_SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = frozenset({2, 3})
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_FIT_BATCH_KIND = "summit.gxe.fit_batch"
_FIT_BATCH_SCHEMA_VERSION = 1
_FIT_BATCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_EXACT_JACKKNIFE_METHOD = "two_sided_snp_kernel_deletion"
_BLOCK_LOCAL_JACKKNIFE_METHOD = "block_local_ldscore_deletion"


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_COPY_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


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


def ordered_variant_digest(frame: pd.DataFrame) -> str:
    """Return a stable digest without serializing sample-level information."""
    required = ("CHR", "SNP", "BP", "A1", "A2")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Variant table is missing columns: {missing}.")
    digest = hashlib.sha256()
    for row in frame.loc[:, required].itertuples(index=False, name=None):
        digest.update("\x1f".join(str(x) for x in row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class GxENormalEquations:
    matrix: np.ndarray
    rhs: np.ndarray
    traces: np.ndarray
    component_names: tuple[str, ...]


@dataclass(frozen=True)
class GxEInputProvenance:
    """Identity of one canonical input streamed into a private fit snapshot."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class GxEConsumedInputProvenance:
    """Exact input snapshots consumed by one phenotype fit."""

    reference_manifest: GxEInputProvenance
    feature_cache: GxEInputProvenance | None
    phenotype_moments: GxEInputProvenance
    gwas: GxEInputProvenance
    gwis: GxEInputProvenance


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
    standard_errors: np.ndarray | None = None
    jackknife_estimates: np.ndarray | None = None
    jackknife_block_labels: tuple[str, ...] = ()
    phenotype_residual_variance_fraction: float | None = None
    consumed_input_provenance: GxEConsumedInputProvenance | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component": self.component_names,
                "coefficient": self.coefficients,
                "variance_contribution": self.contributions,
                "proportion": self.proportions,
                "proportion_se": (
                    self.standard_errors
                    if self.standard_errors is not None
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
    """One hash-bound phenotype summary triplet for a reusable GxE reference."""

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
    manifest_sha256: str
    schema_version: int
    feature_cache_sha256: str | None
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
    reference_provenance: GxEInputProvenance
    feature_cache_provenance: GxEInputProvenance | None


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
    null_corrected: bool = False,
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
        if not null_corrected and np.any(arr < 0.0):
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

    if null_corrected:
        # Legacy schema-v2 bundles stored each source-bin panel as
        #   raw[j, b] - annotation_mass[b] / residual_rank.
        # Stored values may therefore be negative even though the recovered
        # squared-sketch contribution is non-negative.  Validate that raw
        # quantity here, before the aggregate offset is restored in directed().
        offset = masses[None, :] / float(r)
        tolerance = 5.0e-9 * np.maximum(1.0, offset)
        for name, panel in panels.items():
            recovered = panel + offset
            invalid = recovered < -tolerance
            if np.any(invalid):
                bad = np.argwhere(invalid)[0].tolist()
                raise ValueError(
                    f"{name} has a materially negative squared-sketch contribution "
                    f"after reversing its legacy storage offset; first invalid index is {bad}."
                )

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
        if null_corrected:
            # The writer subtracted m_source/r from every row.  This is an
            # algebraic storage offset, not a population-LD approximation.
            value += float(r)
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
    """Solve the unconstrained MoM system with explicit identifiability checks."""
    max_condition = float(max_condition)
    if not np.isfinite(max_condition) or max_condition <= 0.0:
        raise ValueError(f"max_condition must be positive and finite; got {max_condition!r}.")
    if rcond is not None:
        rcond = float(rcond)
        if not np.isfinite(rcond) or rcond <= 0.0:
            raise ValueError(f"rcond must be positive and finite when supplied; got {rcond!r}.")
    lhs = _as_float_array("normal matrix", equations.matrix, ndim=2)
    rhs = _as_float_array("normal RHS", equations.rhs, ndim=1)
    if lhs.shape[0] != lhs.shape[1] or lhs.shape[0] != rhs.shape[0]:
        raise ValueError("Normal matrix must be square and match the RHS length.")
    _require_shape("traces", np.asarray(equations.traces), (rhs.shape[0],))

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

    _, singular, _ = np.linalg.svd(lhs, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("Normal matrix has no positive singular values.")
    tol = (max(lhs.shape) * np.finfo(np.float64).eps * singular[0]) if rcond is None else float(rcond) * singular[0]
    rank = int(np.sum(singular > tol))
    condition = float(np.inf if singular[-1] <= 0.0 else singular[0] / singular[-1])
    if not allow_ill_conditioned and (rank < lhs.shape[0] or condition > max_condition):
        raise ValueError(
            "GxE/NxE normal equations are not identifiable at the requested tolerance: "
            f"rank={rank}/{lhs.shape[0]}, condition={condition:.6g}. "
            "This commonly occurs when E^2 is constant (NxE equals residual noise), "
            "or when annotations/kernels are redundant."
        )

    coef = np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]
    resid_denom = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
    rel_resid = float(np.linalg.norm(lhs @ coef - rhs) / resid_denom)
    contributions = coef * np.asarray(equations.traces, dtype=np.float64)
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
    )


def _assemble_deleted_normal_equations(
    *,
    block_id: int,
    annotations: np.ndarray,
    blocks: np.ndarray,
    score_x: np.ndarray,
    score_w: np.ndarray,
    panels: Mapping[str, np.ndarray],
    within: Mapping[str, np.ndarray],
    norm_x: np.ndarray,
    norm_w: np.ndarray,
    diag_x: np.ndarray,
    diag_w: np.ndarray,
    residual_rank: int,
    q_nxe: float,
    q_residual: float,
    trace_nxe: float,
    trace_nxe_sq: float,
    annotation_names: Sequence[str],
    null_corrected: bool,
) -> GxENormalEquations:
    """Construct an exact two-sided leave-SNP-block-out kernel system."""
    annot = np.asarray(annotations, dtype=np.float64)
    m, k = annot.shape
    mask = np.asarray(blocks == int(block_id), dtype=bool)
    if mask.shape != (m,) or not np.any(mask):
        raise ValueError(f"Jackknife block {block_id} is empty or misaligned.")
    masses = annot.sum(axis=0, dtype=np.float64)
    deleted_masses = annot[mask].sum(axis=0, dtype=np.float64)
    remain = masses - deleted_masses
    if np.any(remain <= 0.0):
        bad = np.flatnonzero(remain <= 0.0).tolist()
        raise ValueError(f"Deleting block {block_id} empties annotation columns {bad}.")
    keep = ~mask
    r = float(residual_rank)
    p = 2 * k + 2
    lhs = np.zeros((p, p), dtype=np.float64)
    rhs = np.zeros(p, dtype=np.float64)
    traces = np.zeros(p, dtype=np.float64)
    names = tuple(f"G:{x}" for x in annotation_names) + tuple(
        f"GxE:{x}" for x in annotation_names
    ) + ("NxE", "residual")

    for a in range(k):
        wa = annot[:, a]
        ma = remain[a]
        gi, wi = a, k + a
        rhs[gi] = r * np.dot(wa[keep], score_x[keep] ** 2) / ma
        rhs[wi] = r * np.dot(wa[keep], score_w[keep] ** 2) / ma
        traces[gi] = r * np.dot(wa[keep], norm_x[keep]) / ma
        traces[wi] = r * np.dot(wa[keep], norm_w[keep]) / ma
        lhs[gi, 2 * k] = lhs[2 * k, gi] = r * np.dot(wa[keep], diag_x[keep]) / ma
        lhs[wi, 2 * k] = lhs[2 * k, wi] = r * np.dot(wa[keep], diag_w[keep]) / ma
        lhs[gi, 2 * k + 1] = lhs[2 * k + 1, gi] = traces[gi]
        lhs[wi, 2 * k + 1] = lhs[2 * k + 1, wi] = traces[wi]

    def raw_column(panel_name: str, source_bin: int) -> np.ndarray:
        col = np.asarray(panels[panel_name][:, source_bin], dtype=np.float64)
        if null_corrected:
            col = col + masses[source_bin] / r
        return col

    def deleted_directed(
        forward: str,
        reverse_name: str,
        left_bin: int,
        source_bin: int,
    ) -> float:
        forward_col = raw_column(forward, source_bin)
        reverse_col = raw_column(reverse_name, left_bin)
        total = np.dot(annot[:, left_bin], forward_col)
        rows = np.dot(annot[mask, left_bin], forward_col[mask])
        columns = np.dot(annot[mask, source_bin], reverse_col[mask])
        intersection = float(within[forward][block_id, left_bin, source_bin])
        cross_sum = total - rows - columns + intersection
        return r * r * cross_sum / (remain[left_bin] * remain[source_bin])

    for a in range(k):
        for b in range(k):
            lhs[a, b] = 0.5 * (
                deleted_directed("xx", "xx", a, b)
                + deleted_directed("xx", "xx", b, a)
            )
            lhs[k + a, k + b] = 0.5 * (
                deleted_directed("ww", "ww", a, b)
                + deleted_directed("ww", "ww", b, a)
            )
            cross = 0.5 * (
                deleted_directed("xw", "wx", a, b)
                + deleted_directed("wx", "xw", b, a)
            )
            lhs[a, k + b] = lhs[k + b, a] = cross

    rhs[2 * k] = float(q_nxe)
    rhs[2 * k + 1] = float(q_residual)
    traces[2 * k] = float(trace_nxe)
    traces[2 * k + 1] = r
    lhs[2 * k, 2 * k] = float(trace_nxe_sq)
    lhs[2 * k, 2 * k + 1] = lhs[2 * k + 1, 2 * k] = float(trace_nxe)
    lhs[2 * k + 1, 2 * k + 1] = r
    return GxENormalEquations(0.5 * (lhs + lhs.T), rhs, traces, names)


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


def _prepare_reference_sufficient_statistics(
    *,
    path: Path,
    payload: dict[str, Any],
    manifest_sha256: str,
    schema_version: int,
    feature_cache_sha256: str | None,
    reference_provenance: GxEInputProvenance,
    feature_cache_provenance: GxEInputProvenance | None,
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
    within: Mapping[str, np.ndarray] | None = None,
    jackknife_method: str | None = None,
    null_corrected: bool,
) -> _PreparedGxEReference:
    """Collapse a validated reference to reusable full/deletion templates."""
    annot = np.asarray(annotations, dtype=np.float64)
    m, k = annot.shape
    masses = annot.sum(axis=0, dtype=np.float64)
    r = float(payload["residual_rank"])
    labels = tuple(str(value) for value in block_labels)

    prepared_blocks: np.ndarray | None = None
    block_masses: np.ndarray | None = None
    deleted_matrices: np.ndarray | None = None
    deleted_traces: np.ndarray | None = None
    if jackknife_method is None and within is not None:
        jackknife_method = _EXACT_JACKKNIFE_METHOD
    has_jackknife = (
        block_values is not None
        or bool(labels)
        or within is not None
        or jackknife_method is not None
    )
    if has_jackknife:
        if block_values is None or len(labels) < 2:
            raise ValueError("Incomplete jackknife inputs for prepared GxE reference.")
        if jackknife_method not in {
            _EXACT_JACKKNIFE_METHOD,
            _BLOCK_LOCAL_JACKKNIFE_METHOD,
        }:
            raise ValueError(f"Unsupported GxE jackknife method {jackknife_method!r}.")
        if jackknife_method == _EXACT_JACKKNIFE_METHOD and within is None:
            raise ValueError("Exact two-sided GxE deletion requires within-block traces.")
        if jackknife_method == _BLOCK_LOCAL_JACKKNIFE_METHOD and within is not None:
            raise ValueError(
                "Block-local GxE deletion must not include exact within-block traces."
            )
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
            if null_corrected:
                panel = panel + masses[None, :] / r
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
            reverse: str,
            left: int,
            source: int,
        ) -> float:
            cross_sum = (
                panel_full[forward][left, source]
                - panel_block[forward][block_id, left, source]
            )
            if jackknife_method == _EXACT_JACKKNIFE_METHOD:
                assert within is not None
                cross_sum += (
                    -panel_block[reverse][block_id, source, left]
                    + float(within[forward][block_id, left, source])
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
                        retained_directed(block_id, "xx", "xx", left, source)
                        + retained_directed(block_id, "xx", "xx", source, left)
                    )
                    lhs[k + left, k + source] = 0.5 * (
                        retained_directed(block_id, "ww", "ww", left, source)
                        + retained_directed(block_id, "ww", "ww", source, left)
                    )
                    cross = 0.5 * (
                        retained_directed(block_id, "xw", "wx", left, source)
                        + retained_directed(block_id, "wx", "xw", source, left)
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
        manifest_sha256=str(manifest_sha256),
        schema_version=int(schema_version),
        feature_cache_sha256=feature_cache_sha256,
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
        reference_provenance=reference_provenance,
        feature_cache_provenance=feature_cache_provenance,
    )


def _equations_from_prepared_scores(
    prepared: _PreparedGxEReference,
    score_x: np.ndarray,
    score_w: np.ndarray,
    *,
    q_nxe: float,
    q_residual: float,
) -> tuple[GxENormalEquations, tuple[GxENormalEquations, ...]]:
    """Insert phenotype score moments into prepared reference templates."""
    annot = prepared.annotations
    m, k = annot.shape
    sx = _as_float_array("score_x", score_x, ndim=1)
    sw = _as_float_array("score_w", score_w, ndim=1)
    _require_shape("score_x", sx, (m,))
    _require_shape("score_w", sw, (m,))
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
            deleted.append(
                GxENormalEquations(
                    np.array(prepared.deleted_matrices[block_id], copy=True),
                    deleted_rhs,
                    np.array(prepared.deleted_traces[block_id], copy=True),
                    prepared.component_names,
                )
            )
    return full, tuple(deleted)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "rt", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return obj


@dataclass(frozen=True)
class _InputSnapshot:
    original_path: Path
    snapshot_path: Path
    sha256: str
    size: int


def _snapshot_provenance(snapshot: _InputSnapshot) -> GxEInputProvenance:
    """Freeze provenance from the snapshot copy operation, not the live path."""
    return GxEInputProvenance(
        path=str(snapshot.original_path),
        bytes=int(snapshot.size),
        sha256=str(snapshot.sha256),
    )


def _consumed_input_provenance(
    *,
    reference_manifest: _InputSnapshot,
    feature_cache: _InputSnapshot | None,
    phenotype_moments: _InputSnapshot,
    gwas: _InputSnapshot,
    gwis: _InputSnapshot,
) -> GxEConsumedInputProvenance:
    return GxEConsumedInputProvenance(
        reference_manifest=_snapshot_provenance(reference_manifest),
        feature_cache=(
            None if feature_cache is None else _snapshot_provenance(feature_cache)
        ),
        phenotype_moments=_snapshot_provenance(phenotype_moments),
        gwas=_snapshot_provenance(gwas),
        gwis=_snapshot_provenance(gwis),
    )


def _serialize_consumed_input_provenance(
    provenance: GxEConsumedInputProvenance | None,
) -> dict[str, dict[str, Any] | None] | None:
    """Emit null for in-memory fits; strictly validate every supplied record."""
    if provenance is None:
        return None
    if not isinstance(provenance, GxEConsumedInputProvenance):
        raise ValueError(
            "Refusing to write malformed consumed-input provenance."
        )

    def serialize_record(
        role: str, record: GxEInputProvenance | None, *, optional: bool = False
    ) -> dict[str, Any] | None:
        if record is None and optional:
            return None
        if not isinstance(record, GxEInputProvenance):
            raise ValueError(f"Consumed-input provenance for {role} is incomplete.")
        if (
            not isinstance(record.path, str)
            or not record.path
            or "\x00" in record.path
            or not Path(record.path).is_absolute()
            or str(Path(record.path)) != record.path
        ):
            raise ValueError(
                f"Consumed-input provenance for {role} lacks a canonical absolute path."
            )
        if (
            not isinstance(record.bytes, int)
            or isinstance(record.bytes, bool)
            or record.bytes < 0
        ):
            raise ValueError(f"Consumed-input provenance for {role} has invalid bytes.")
        if not _is_sha256(record.sha256) or record.sha256 != record.sha256.lower():
            raise ValueError(f"Consumed-input provenance for {role} has invalid SHA-256.")
        return {
            "path": record.path,
            "bytes": record.bytes,
            "sha256": record.sha256,
        }

    return {
        "reference_manifest": serialize_record(
            "reference_manifest", provenance.reference_manifest
        ),
        "feature_cache": serialize_record(
            "feature_cache", provenance.feature_cache, optional=True
        ),
        "phenotype_moments": serialize_record(
            "phenotype_moments", provenance.phenotype_moments
        ),
        "gwas": serialize_record("gwas", provenance.gwas),
        "gwis": serialize_record("gwis", provenance.gwis),
    }


class _InputSnapshotStore:
    """Stream fit inputs into private read-only files before validation/parsing."""

    def __init__(self, scratch_dir: str | Path) -> None:
        scratch_root = Path(scratch_dir).expanduser().resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)
        if not scratch_root.is_dir():
            raise ValueError(f"GxE fit scratch path is not a directory: {scratch_root}.")
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".summit-gxe-fit-inputs-", dir=scratch_root
        )
        self.directory = Path(self._temporary.name)
        os.chmod(self.directory, 0o700)
        self._snapshots: dict[Path, _InputSnapshot] = {}

    def __enter__(self) -> "_InputSnapshotStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._temporary.cleanup()

    def capture(self, path: str | Path) -> _InputSnapshot:
        try:
            original = Path(path).expanduser().resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(Path(path).expanduser().resolve()) from None
        existing = self._snapshots.get(original)
        if existing is not None:
            return existing

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        source_fd = os.open(original, flags)
        destination: Path | None = None
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"GxE fit input is not a regular file: {original}.")
            suffixes = original.suffixes
            suffix = "".join(suffixes[-2:]) if suffixes else ".bin"
            destination = self.directory / f"{len(self._snapshots):06d}{suffix}"
            destination_fd = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            digest = hashlib.sha256()
            copied = 0
            with os.fdopen(source_fd, "rb", closefd=True) as source, os.fdopen(
                destination_fd, "wb", closefd=True
            ) as sink:
                source_fd = -1
                while True:
                    block = source.read(_COPY_CHUNK_BYTES)
                    if not block:
                        break
                    digest.update(block)
                    sink.write(block)
                    copied += len(block)
                sink.flush()
                os.fsync(sink.fileno())
            os.chmod(destination, 0o400)
            snapshot = _InputSnapshot(
                original_path=original,
                snapshot_path=destination,
                sha256=digest.hexdigest(),
                size=copied,
            )
            self._snapshots[original] = snapshot
            return snapshot
        except Exception:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise
        finally:
            if source_fd >= 0:
                os.close(source_fd)


def _load_feature_cache_bundle(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and semantically validate an exact schema-v2 feature cache."""
    with np.load(path, allow_pickle=False) as bundle:
        expected_members = {*_FEATURE_CACHE_ARRAY_DTYPES, "metadata_json"}
        if len(bundle.files) != len(expected_members) or set(bundle.files) != expected_members:
            raise ValueError(
                f"GxE feature cache has unexpected, duplicate, or missing arrays: {path}."
            )
        try:
            metadata = json.loads(str(bundle["metadata_json"].item()))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"GxE feature cache has invalid metadata_json: {path}.") from exc
        arrays = {
            name: np.asarray(bundle[name]).copy() for name in _FEATURE_CACHE_ARRAY_DTYPES
        }
    _validate_feature_cache_semantics(metadata, arrays)
    return metadata, arrays


def _validate_reference_feature_cache_contract(
    reference: Mapping[str, Any],
    diagonal: pd.DataFrame,
    annotation_names: Sequence[str],
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Prove that a reusable-score cache defines this exact reference."""
    exact_fields = (
        "analysis_fingerprint",
        "variant_digest",
        "n_samples",
        "fixed_effect_rank_excluding_intercept",
        "residual_rank",
        "kernel_mode",
        "genotype_scale",
        "environment",
        "environment_transform",
        "covariates",
        "genotype_files",
    )
    mismatched = [
        field for field in exact_fields if reference.get(field) != metadata.get(field)
    ]
    if (
        reference.get("feature_backend_provenance") is not None
        or metadata.get("backend_provenance") is not None
    ) and reference.get("feature_backend_provenance") != metadata.get(
        "backend_provenance"
    ):
        mismatched.append("feature_backend_provenance")
    if list(annotation_names) != metadata.get("annotation_names"):
        mismatched.append("annotation_names")
    if mismatched:
        raise ValueError(
            "Reference manifest disagrees with its bound feature cache for fields "
            f"{sorted(set(mismatched))}."
        )

    cached_diagnostics = metadata.get("feature_diagnostics")
    reference_diagnostics = reference.get("feature_diagnostics")
    if not isinstance(cached_diagnostics, Mapping) or not isinstance(
        reference_diagnostics, Mapping
    ):
        raise ValueError("Reference/cache feature diagnostics are invalid.")
    changed_diagnostics = [
        key
        for key, value in cached_diagnostics.items()
        if reference_diagnostics.get(key) != value
    ]
    allowed_reference_only = {
        "max_xw_wx_trace_asymmetry_absolute",
        "max_xw_wx_trace_asymmetry_relative",
    }
    unexpected_diagnostics = set(reference_diagnostics) - set(cached_diagnostics)
    if changed_diagnostics or not unexpected_diagnostics.issubset(allowed_reference_only):
        raise ValueError("Reference feature diagnostics disagree with its feature cache.")
    for key in unexpected_diagnostics:
        value = reference_diagnostics[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Reference feature diagnostic {key!r} is invalid.")

    variants = {
        "CHR": np.asarray(arrays["variant_chr"]).astype(str),
        "SNP": np.asarray(arrays["variant_snp"]).astype(str),
        "BP": np.asarray(arrays["variant_bp"]).astype(str),
        "A1": np.asarray(arrays["variant_a1"]).astype(str),
        "A2": np.asarray(arrays["variant_a2"]).astype(str),
    }
    for column, expected in variants.items():
        observed = diagonal[column].astype(str).to_numpy()
        if not np.array_equal(observed, expected):
            raise ValueError(
                f"Reference diagonal {column} axis disagrees with its feature cache."
            )

    weight_columns = [f"ANNOT_{index}" for index in range(len(annotation_names))]
    comparisons = {
        "annotations": (
            diagonal.loc[:, weight_columns].to_numpy(dtype=np.float64),
            np.asarray(arrays["annotations"], dtype=np.float64),
        ),
        "NORM_X": (
            diagonal["NORM_X"].to_numpy(dtype=np.float64),
            np.asarray(arrays["norm_x"], dtype=np.float64),
        ),
        "NORM_W": (
            diagonal["NORM_W"].to_numpy(dtype=np.float64),
            np.asarray(arrays["norm_w"], dtype=np.float64),
        ),
        "SCALE_X": (
            diagonal["SCALE_X"].to_numpy(dtype=np.float64),
            np.asarray(arrays["scale_x"], dtype=np.float64),
        ),
        "SCALE_W": (
            diagonal["SCALE_W"].to_numpy(dtype=np.float64),
            np.asarray(arrays["scale_w"], dtype=np.float64),
        ),
        "DNXE_X": (
            diagonal["DNXE_X"].to_numpy(dtype=np.float64),
            np.asarray(arrays["diag_nxe_x"], dtype=np.float64),
        ),
        "DNXE_W": (
            diagonal["DNXE_W"].to_numpy(dtype=np.float64),
            np.asarray(arrays["diag_nxe_w"], dtype=np.float64),
        ),
        "CORR_XW": (
            diagonal["CORR_XW"].to_numpy(dtype=np.float64),
            np.asarray(arrays["corr_xw"], dtype=np.float64),
        ),
    }
    for label, (observed, expected) in comparisons.items():
        if observed.shape != expected.shape or not np.allclose(
            observed, expected, rtol=5.0e-10, atol=1.0e-10, equal_nan=False
        ):
            raise ValueError(
                f"Reference diagonal {label} values disagree with its feature cache."
            )

    declared_masses = np.asarray(reference.get("annotation_masses"), dtype=np.float64)
    cached_masses = np.asarray(metadata.get("annotation_masses"), dtype=np.float64)
    if declared_masses.shape != cached_masses.shape or not np.allclose(
        declared_masses, cached_masses, rtol=0.0, atol=0.0
    ):
        raise ValueError("Reference annotation masses disagree with its feature cache.")
    for field in ("trace_nxe", "trace_nxe_sq"):
        if not np.isclose(
            float(reference.get(field)),
            float(metadata.get(field)),
            rtol=2.0e-12,
            atol=2.0e-10,
        ):
            raise ValueError(f"Reference {field} disagrees with its feature cache.")

    cached_labels = metadata.get("jackknife_labels")
    if cached_labels is not None:
        if "BLOCK" not in diagonal:
            raise ValueError("Reference diagonal omits cache-bound jackknife block IDs.")
        observed_blocks = pd.to_numeric(
            diagonal["BLOCK"], errors="raise"
        ).to_numpy(dtype=np.int32)
        if not np.array_equal(observed_blocks, arrays["jackknife_ids"]):
            raise ValueError("Reference jackknife block IDs disagree with its feature cache.")
        jackknife = reference.get("jackknife")
        if not isinstance(jackknife, Mapping) or jackknife.get("block_labels") != cached_labels:
            raise ValueError("Reference jackknife labels disagree with its feature cache.")


def _resolve_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


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
    *,
    allow_negative_storage: bool = False,
) -> np.ndarray:
    panel = _read_table(path)
    required = ["CHR", "SNP", "BP", *value_columns]
    observed_columns = panel.columns.astype(str).tolist()
    if observed_columns != required or len(set(observed_columns)) != len(observed_columns):
        raise ValueError(
            f"{path} must contain exactly the ordered columns {required}; "
            f"observed {observed_columns}."
        )
    if len(panel) != len(variants):
        raise ValueError(f"{path} has {len(panel)} rows; expected {len(variants)}.")
    for col in ("CHR", "SNP", "BP"):
        if not np.array_equal(panel[col].astype(str).to_numpy(), variants[col].astype(str).to_numpy()):
            raise ValueError(f"{path} is not in the reference variant order ({col} mismatch).")
    values = panel.loc[:, list(value_columns)].to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains a non-finite panel value.")
    if not allow_negative_storage and np.any(values < 0.0):
        raise ValueError(f"{path} contains a negative squared-sketch panel value.")
    values[values == 0.0] = 0.0
    return values


def fit_from_files(
    reference_manifest: str | Path,
    phenotype_moments: str | Path,
    gwas_scores: str | Path,
    gwis_scores: str | Path,
    *,
    allow_ill_conditioned: bool = False,
    max_condition: float = 1.0e12,
    scratch_dir: str | Path | None = None,
) -> tuple[GxEFitResult, GxENormalEquations]:
    """Load an immutable snapshot of a SUMMIT bundle and fit the full model."""
    result = fit_many_from_files(
        reference_manifest,
        {
            "phenotype": GxEPhenotypeInput(
                phenotype_moments=phenotype_moments,
                gwas_scores=gwas_scores,
                gwis_scores=gwis_scores,
            )
        },
        allow_ill_conditioned=allow_ill_conditioned,
        max_condition=max_condition,
        scratch_dir=scratch_dir,
    )
    return result["phenotype"]


def _fit_from_input_snapshots(
    reference_manifest: str | Path,
    phenotype_moments: str | Path,
    gwas_scores: str | Path,
    gwis_scores: str | Path,
    *,
    snapshots: _InputSnapshotStore,
    allow_ill_conditioned: bool,
    max_condition: float,
    prepared_out: list[_PreparedGxEReference] | None = None,
) -> tuple[GxEFitResult, GxENormalEquations]:
    ref_path = Path(reference_manifest).resolve()
    mom_path = Path(phenotype_moments).resolve()
    ref_snapshot = snapshots.capture(ref_path)
    moments_snapshot = snapshots.capture(mom_path)
    ref = _load_json(ref_snapshot.snapshot_path)
    moments = _load_json(moments_snapshot.snapshot_path)
    ref_version_raw = ref.get("schema_version")
    moments_version_raw = moments.get("schema_version")
    if not isinstance(ref_version_raw, int) or isinstance(ref_version_raw, bool):
        raise ValueError(f"Reference schema_version must be a JSON integer: {ref_path}.")
    if not isinstance(moments_version_raw, int) or isinstance(moments_version_raw, bool):
        raise ValueError(f"Phenotype schema_version must be a JSON integer: {mom_path}.")
    ref_version = int(ref_version_raw)
    moments_version = int(moments_version_raw)
    if ref.get("kind") != _REFERENCE_KIND or ref_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported GxE reference manifest: {ref_path}.")
    if ref.get("backend_provenance") is not None:
        _validate_backend_provenance(
            ref["backend_provenance"], expected_stage="reference"
        )
    shard_backends = ref.get("shard_backend_provenance")
    if shard_backends is not None:
        if not isinstance(shard_backends, list) or not shard_backends:
            raise ValueError("Merged GxE reference has invalid shard backend provenance.")
        for backend in shard_backends:
            _validate_backend_provenance(
                backend, expected_stage="reference_shard"
            )
    if moments.get("kind") != _MOMENTS_KIND or moments_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported GxE phenotype moments file: {mom_path}.")
    if ref_version != moments_version:
        raise ValueError(
            "Reference and phenotype bundles use different schema versions: "
            f"reference={ref_version}, moments={moments_version}."
        )
    kernel_mode = ref.get("kernel_mode")
    if kernel_mode not in {"genie", "standardized"}:
        raise ValueError(f"Reference manifest has unsupported kernel_mode={kernel_mode!r}.")
    genotype_scale = ref.get("genotype_scale")
    if genotype_scale not in {"hwe", "sample"}:
        raise ValueError(f"Reference manifest has unsupported genotype_scale={genotype_scale!r}.")
    null_corrected = ref.get("null_corrected")
    if not isinstance(null_corrected, bool):
        raise ValueError("Reference manifest null_corrected must be a JSON boolean.")
    for key in ("analysis_fingerprint", "variant_digest", "residual_rank"):
        if str(ref.get(key)) != str(moments.get(key)):
            raise ValueError(f"Reference and phenotype bundles disagree on {key}.")
    n_samples = ref.get("n_samples")
    fixed_rank = ref.get("fixed_effect_rank_excluding_intercept")
    residual_rank = ref.get("residual_rank")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (n_samples, fixed_rank, residual_rank)
    ):
        raise ValueError("Reference sample size and fixed/residual ranks must be JSON integers.")
    if n_samples <= 0 or fixed_rank < 0 or residual_rank <= 0:
        raise ValueError("Reference sample size and fixed/residual ranks are not positive/valid.")
    if residual_rank != n_samples - fixed_rank - 1:
        raise ValueError(
            "Reference residual rank is inconsistent with its sample size and fixed-effect rank."
        )
    if moments.get("score_definition") != "feature_transpose_residualized_y_over_sqrt_residual_rank":
        raise ValueError("Phenotype moments use an unsupported marginal-score scale.")
    expected_reference_hash = moments.get("reference_manifest_sha256")
    if not _is_sha256(expected_reference_hash):
        raise ValueError("Phenotype moments do not bind the exact GxE reference manifest.")
    observed_reference_hash = ref_snapshot.sha256
    ref_cache = ref.get("feature_cache")
    ref_cache_hash = None
    ref_cache_path = None
    ref_cache_metadata = None
    ref_cache_arrays = None
    ref_cache_snapshot: _InputSnapshot | None = None
    validated_cache_by_hash: dict[
        str, tuple[dict[str, Any], dict[str, np.ndarray]]
    ] = {}
    if ref_cache is not None:
        if (
            not isinstance(ref_cache, dict)
            or not isinstance(ref_cache.get("path"), str)
            or not ref_cache["path"]
            or not _is_sha256(ref_cache.get("sha256"))
        ):
            raise ValueError("Reference manifest has an invalid feature-cache binding.")
        ref_cache_hash = str(ref_cache["sha256"])
        ref_cache_path = _resolve_path(ref_path, ref_cache["path"]).resolve()
        ref_cache_snapshot = snapshots.capture(ref_cache_path)
        if ref_cache_snapshot.sha256 != ref_cache_hash:
            raise ValueError("Reference feature cache failed its SHA-256 check.")
        ref_cache_metadata, ref_cache_arrays = _load_feature_cache_bundle(
            ref_cache_snapshot.snapshot_path
        )
        validated_cache_by_hash[ref_cache_hash] = (
            ref_cache_metadata,
            ref_cache_arrays,
        )
    moments_cache_hash = moments.get("feature_cache_sha256")
    if moments_cache_hash is not None and not _is_sha256(moments_cache_hash):
        raise ValueError("Phenotype moments contain an invalid feature-cache SHA-256 binding.")
    moments_cache = moments.get("feature_cache")
    verified_moments_cache_hash = None
    moments_cache_snapshot: _InputSnapshot | None = None
    if moments_cache is not None:
        if (
            not isinstance(moments_cache, dict)
            or not isinstance(moments_cache.get("path"), str)
            or not moments_cache["path"]
            or not _is_sha256(moments_cache.get("sha256"))
        ):
            raise ValueError("Phenotype moments have an invalid feature-cache binding.")
        moments_cache_path = _resolve_path(mom_path, moments_cache["path"]).resolve()
        verified_moments_cache_hash = str(moments_cache["sha256"])
        moments_cache_snapshot = snapshots.capture(moments_cache_path)
        if moments_cache_snapshot.sha256 != verified_moments_cache_hash:
            raise ValueError("Phenotype-moment feature cache failed its SHA-256 check.")
        if (
            ref_cache_hash is not None
            and verified_moments_cache_hash != ref_cache_hash
        ):
            raise ValueError(
                "Phenotype moments and reference point to different feature caches."
            )
        if verified_moments_cache_hash in validated_cache_by_hash:
            moments_cache_metadata, moments_cache_arrays = validated_cache_by_hash[
                verified_moments_cache_hash
            ]
        else:
            moments_cache_metadata, moments_cache_arrays = _load_feature_cache_bundle(
                moments_cache_snapshot.snapshot_path
            )
            validated_cache_by_hash[verified_moments_cache_hash] = (
                moments_cache_metadata,
                moments_cache_arrays,
            )
        if (
            moments_cache_hash is not None
            and moments_cache_hash != verified_moments_cache_hash
        ):
            raise ValueError("Phenotype moments contain inconsistent feature-cache hashes.")
    if observed_reference_hash != expected_reference_hash:
        # Phenotype scores do not depend on the randomized trace realization.
        # They may therefore be reused across B10/B100 references only when
        # both manifests are cryptographically bound to the exact same feature
        # cache (genotype, samples, design, scales, annotations, and SNP axis).
        if (
            ref_cache_hash is None
            or verified_moments_cache_hash is None
            or verified_moments_cache_hash != ref_cache_hash
        ):
            raise ValueError(
                "Phenotype moments were generated for a different GxE reference/feature definition."
            )
    elif moments_cache_hash is not None and moments_cache_hash != ref_cache_hash:
        raise ValueError("Phenotype moments and reference declare different feature caches.")

    declared_score_files = moments.get("files")
    if not isinstance(declared_score_files, dict) or not {"gwas", "gwis"}.issubset(declared_score_files):
        raise ValueError("Phenotype moments must declare the exact GWAS and GWIS score artifacts.")
    supplied_score_paths = (Path(gwas_scores).resolve(), Path(gwis_scores).resolve())
    score_snapshots: dict[str, _InputSnapshot] = {}
    for label, supplied in zip(("gwas", "gwis"), supplied_score_paths):
        declared = _resolve_path(mom_path, str(declared_score_files[label])).resolve()
        if supplied != declared:
            raise ValueError(
                f"Supplied {label.upper()} file {supplied} is not the phenotype-bound artifact {declared}."
            )
        score_hashes = moments.get("score_sha256")
        if not isinstance(score_hashes, dict) or label not in score_hashes:
            raise ValueError("Phenotype moments do not cryptographically bind both score artifacts.")
        expected_hash = score_hashes[label]
        if not _is_sha256(expected_hash):
            raise ValueError(f"Phenotype moments contain an invalid {label.upper()} SHA-256.")
        score_snapshot = snapshots.capture(supplied)
        if score_snapshot.sha256 != str(expected_hash):
            raise ValueError(f"{label.upper()} score file SHA-256 does not match phenotype moments.")
        score_snapshots[label] = score_snapshot

    artifact_hashes = ref.get("artifact_sha256")
    required_artifacts = {"xx", "xw", "wx", "ww", "diagonal"}
    if "jackknife" in ref.get("files", {}):
        required_artifacts.add("jackknife")
    if ref.get("jackknife") is not None:
        randomization = ref.get("randomization")
        if not isinstance(randomization, dict) or "num_vectors" not in randomization:
            raise ValueError("Jackknife reference is missing its random-probe count.")
        probe_count = randomization["num_vectors"]
        if (
            isinstance(probe_count, bool)
            or not isinstance(probe_count, int)
            or probe_count <= 0
            or probe_count > 2**64
        ):
            raise ValueError(
                "Jackknife reference num_vectors must be a positive uint64-range JSON integer."
            )
        low_probe_override = randomization.get("low_probe_jackknife_override", False)
        if not isinstance(low_probe_override, bool):
            raise ValueError(
                "Jackknife reference low_probe_jackknife_override must be a JSON boolean."
            )
        if probe_count < 100 and not low_probe_override:
            raise ValueError(
                "Jackknife reference has fewer than 100 probes without an explicit diagnostic override."
            )
    if not isinstance(artifact_hashes, dict) or not required_artifacts.issubset(artifact_hashes):
        raise ValueError("Reference manifest does not cryptographically bind every declared artifact.")
    artifact_snapshots: dict[str, _InputSnapshot] = {}
    for label in sorted(required_artifacts):
        artifact = _resolve_path(ref_path, str(ref["files"][label])).resolve()
        if not _is_sha256(artifact_hashes[label]):
            raise ValueError(f"Reference artifact {label!r} has an invalid SHA-256 declaration.")
        artifact_snapshot = snapshots.capture(artifact)
        if artifact_snapshot.sha256 != str(artifact_hashes[label]):
            raise ValueError(f"Reference artifact {label!r} failed its SHA-256 check: {artifact}.")
        artifact_snapshots[label] = artifact_snapshot

    diag_path = _resolve_path(ref_path, str(ref["files"]["diagonal"]))
    diag = _read_table(artifact_snapshots["diagonal"].snapshot_path)
    annotation_names_raw = ref.get("annotation_names")
    if not isinstance(annotation_names_raw, list):
        raise ValueError("Reference annotation_names must be a JSON list.")
    names = tuple(_validate_gxe_annotation_names(annotation_names_raw))
    weight_cols = [f"ANNOT_{i}" for i in range(len(names))]
    required_diag = ["CHR", "SNP", "BP", "A1", "A2", "NORM_X", "NORM_W", "DNXE_X", "DNXE_W", *weight_cols]
    if ref_version >= 3:
        required_diag = [
            "CHR", "SNP", "BP", "A1", "A2", "NORM_X", "NORM_W",
            "SCALE_X", "SCALE_W", "DNXE_X", "DNXE_W", "CORR_XW",
            *weight_cols,
        ]
        if ref.get("jackknife") is not None:
            required_diag.append("BLOCK")
        observed_diag_columns = diag.columns.astype(str).tolist()
        if (
            observed_diag_columns != required_diag
            or len(set(observed_diag_columns)) != len(observed_diag_columns)
        ):
            raise ValueError(
                f"{diag_path} must contain exactly the canonical ordered columns "
                f"{required_diag}; observed {observed_diag_columns}."
            )
    missing = [c for c in required_diag if c not in diag.columns]
    if missing:
        raise ValueError(f"{diag_path} is missing columns {missing}.")
    if diag["SNP"].duplicated().any():
        raise ValueError(f"{diag_path} contains duplicate SNP identifiers.")
    if ordered_variant_digest(diag) != str(ref["variant_digest"]):
        raise ValueError("Diagonal file variant digest does not match its manifest.")
    variants = diag.loc[:, ["CHR", "SNP", "BP", "A1", "A2"]].copy()
    annotations = diag.loc[:, weight_cols].to_numpy(dtype=np.float64)
    observed_masses = annotations.sum(axis=0, dtype=np.float64)
    declared_masses = _as_float_array(
        "reference annotation_masses", ref.get("annotation_masses"), ndim=1
    )
    _require_shape("reference annotation_masses", declared_masses, observed_masses.shape)
    if not np.allclose(declared_masses, observed_masses, rtol=5.0e-10, atol=1.0e-8):
        raise ValueError(
            "Reference annotation_masses disagree with the canonical diagonal annotation weights."
        )
    if ref_cache_metadata is not None:
        if ref_version < 3 or ref_cache_arrays is None:
            raise ValueError(
                "Cross-reference feature-cache reuse requires a schema-v3 reference "
                "and a validated schema-v2 cache."
            )
        _validate_reference_feature_cache_contract(
            ref,
            diag,
            names,
            ref_cache_metadata,
            ref_cache_arrays,
        )
        for _, cached_arrays in validated_cache_by_hash.values():
            cached_arrays.clear()
        validated_cache_by_hash.clear()
        ref_cache_arrays = None
    if validated_cache_by_hash:
        for _, cached_arrays in validated_cache_by_hash.values():
            cached_arrays.clear()
        validated_cache_by_hash.clear()
    if ref_version >= 3:
        genotype_files = ref.get("genotype_files")
        valid_genotype_trios = (
            {".bed", ".bim", ".fam"},
            {".pgen", ".pvar", ".psam"},
        )
        if (
            not isinstance(genotype_files, dict)
            or set(genotype_files) not in valid_genotype_trios
        ):
            raise ValueError("Schema-v3 reference is missing genotype file provenance.")
        for extension in sorted(genotype_files):
            entry = genotype_files[extension]
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("bytes"), int)
                or int(entry["bytes"]) <= 0
                or not _is_sha256(entry.get("sha256"))
            ):
                raise ValueError(f"Invalid genotype provenance for {extension} in reference manifest.")
        env_transform = ref.get("environment_transform")
        if not isinstance(env_transform, dict) or env_transform.get("standardized") is not True:
            raise ValueError(
                "Schema-v3 reference must declare a standardized per-SD environment transform."
            )
        for key in ("raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares"):
            value = env_transform.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                raise ValueError(f"Invalid environment_transform[{key!r}] in reference manifest.")
        if not _is_sha256(env_transform.get("fixed_effect_design_sha256")):
            raise ValueError(
                "Schema-v3 environment transform lacks a fixed-effect design digest."
            )
        ddof = env_transform.get("ddof")
        if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof not in (0, 1):
            raise ValueError("Schema-v3 environment transform has an invalid ddof declaration.")
        if float(env_transform["raw_sd"]) <= 0.0 or env_transform.get("units") != "per_environment_sd":
            raise ValueError("Schema-v3 environment transform has an invalid scale or units declaration.")
        if abs(float(env_transform["analysis_mean"])) > 1.0e-10:
            raise ValueError("Schema-v3 standardized environment is not centered.")
        expected_environment_ss = float(n_samples - ddof)
        if not np.isclose(
            float(env_transform["analysis_sum_squares"]),
            expected_environment_ss,
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise ValueError(
                "Schema-v3 standardized environment sum of squares is inconsistent with N and ddof."
            )
        diagnostics = ref.get("feature_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Schema-v3 reference is missing feature_diagnostics.")
        required_diagnostics = {
            "valid_additive_columns",
            "valid_interaction_columns",
            "max_projection_leakage_additive",
            "max_projection_leakage_interaction",
            "kernel_traces_additive",
            "kernel_traces_interaction",
        }
        if not required_diagnostics.issubset(diagnostics):
            raise ValueError("Schema-v3 reference has incomplete feature_diagnostics.")
        if (
            diagnostics["valid_additive_columns"] != len(diag)
            or diagnostics["valid_interaction_columns"] != len(diag)
        ):
            raise ValueError("Schema-v3 feature diagnostics do not cover every reference variant.")
        for key in ("max_projection_leakage_additive", "max_projection_leakage_interaction"):
            value = diagnostics[key]
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Schema-v3 feature diagnostic {key!r} is invalid.")
            if float(value) > 1.0e-9:
                raise ValueError(f"Schema-v3 feature diagnostic {key!r} exceeds its projection tolerance.")
        for key in ("kernel_traces_additive", "kernel_traces_interaction"):
            values = _as_float_array(f"feature_diagnostics[{key}]", diagnostics[key], ndim=1)
            _require_shape(f"feature_diagnostics[{key}]", values, (len(names),))
        for scale_column in ("SCALE_X", "SCALE_W"):
            values = diag[scale_column].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"Schema-v3 reference contains an invalid {scale_column} value.")
        corr_xw = diag["CORR_XW"].to_numpy(dtype=np.float64)
        norm_bound = np.sqrt(
            diag["NORM_X"].to_numpy(dtype=np.float64)
            * diag["NORM_W"].to_numpy(dtype=np.float64)
        )
        if not np.all(np.isfinite(corr_xw)) or np.any(np.abs(corr_xw) > norm_bound + 1.0e-8):
            raise ValueError("Schema-v3 additive/interaction correlations violate Cauchy-Schwarz.")
        if kernel_mode == "standardized":
            for norm_column in ("NORM_X", "NORM_W"):
                values = diag[norm_column].to_numpy(dtype=np.float64)
                if not np.allclose(values, 1.0, rtol=1.0e-9, atol=1.0e-9):
                    raise ValueError(
                        f"Schema-v3 standardized reference violates the {norm_column}=1 invariant."
                    )
        trace_from_diag = {
            "kernel_traces_additive": (
                float(residual_rank) * annotations.T @ diag["NORM_X"].to_numpy(dtype=np.float64)
                / observed_masses
            ),
            "kernel_traces_interaction": (
                float(residual_rank) * annotations.T @ diag["NORM_W"].to_numpy(dtype=np.float64)
                / observed_masses
            ),
        }
        for key, observed in trace_from_diag.items():
            declared = np.asarray(diagnostics[key], dtype=np.float64)
            if not np.allclose(declared, observed, rtol=5.0e-10, atol=1.0e-8):
                raise ValueError(f"Schema-v3 feature diagnostic {key!r} disagrees with the diagonal table.")

    reference_variant_axis = {
        column: variants[column].astype(str).to_numpy()
        for column in ("CHR", "SNP", "BP", "A1", "A2")
    }
    score_arrays = []
    for label, score_key, path in zip(
        ("GWAS", "GWIS"), ("gwas", "gwis"), supplied_score_paths
    ):
        table = _read_table(score_snapshots[score_key].snapshot_path)
        required = ["CHR", "SNP", "BP", "A1", "A2", "SCORE", "SCORE_MODE"]
        if ref_version >= 3:
            canonical_score_columns = [
                "CHR", "SNP", "BP", "A1", "A2", "N", "DF", "SCORE_MODE", "SCORE"
            ]
            observed_score_columns = table.columns.astype(str).tolist()
            if (
                observed_score_columns != canonical_score_columns
                or len(set(observed_score_columns)) != len(observed_score_columns)
            ):
                raise ValueError(
                    f"{label} score file must contain exactly the canonical ordered columns "
                    f"{canonical_score_columns}; observed {observed_score_columns}."
                )
        missing = [c for c in required if c not in table.columns]
        if missing:
            raise ValueError(f"{label} score file {path} is missing columns {missing}.")
        if len(table) != len(variants):
            raise ValueError(f"{label} score file has {len(table)} rows; expected {len(variants)}.")
        for col in ("CHR", "SNP", "BP", "A1", "A2"):
            if not np.array_equal(
                table[col].astype(str).to_numpy(), reference_variant_axis[col]
            ):
                raise ValueError(f"{label} score file is not aligned to the reference ({col} mismatch).")
        modes = table["SCORE_MODE"].astype("string").str.lower()
        mode_values = set(modes.dropna().astype(str))
        if modes.isna().any() or not bool((modes == "marginal_cross_product").all()):
            raise ValueError(
                f"{label} uses unsupported SCORE_MODE={sorted(mode_values)}. "
                "Conditional PLINK interaction statistics are not GENIE marginal scores."
            )
        n_values = pd.to_numeric(table["N"], errors="raise").to_numpy(dtype=np.float64) if "N" in table else None
        df_values = pd.to_numeric(table["DF"], errors="raise").to_numpy(dtype=np.float64) if "DF" in table else None
        if n_values is None or df_values is None:
            raise ValueError(f"{label} score file must contain N and DF columns.")
        if not np.all(n_values == float(ref["n_samples"])):
            raise ValueError(f"{label} score file N does not match the reference sample count.")
        if not np.all(df_values == float(ref["residual_rank"])):
            raise ValueError(f"{label} score file DF does not match the reference residual rank.")
        score_arrays.append(table["SCORE"].to_numpy(dtype=np.float64))

    shared_cache_snapshot = (
        ref_cache_snapshot
        if ref_cache_snapshot is not None
        else moments_cache_snapshot
    )
    reference_provenance = _snapshot_provenance(ref_snapshot)
    feature_cache_provenance = (
        None
        if shared_cache_snapshot is None
        else _snapshot_provenance(shared_cache_snapshot)
    )

    panels = {}
    for key in ("xx", "xw", "wx", "ww"):
        panels[key] = _aligned_panel(
            artifact_snapshots[key].snapshot_path,
            variants,
            names,
            allow_negative_storage=null_corrected,
        )

    eq = assemble_normal_equations(
        annotations=annotations,
        score_x=score_arrays[0],
        score_w=score_arrays[1],
        ld_xx=panels["xx"],
        ld_xw=panels["xw"],
        ld_wx=panels["wx"],
        ld_ww=panels["ww"],
        norm_x=diag["NORM_X"].to_numpy(dtype=np.float64),
        norm_w=diag["NORM_W"].to_numpy(dtype=np.float64),
        diag_nxe_x=diag["DNXE_X"].to_numpy(dtype=np.float64),
        diag_nxe_w=diag["DNXE_W"].to_numpy(dtype=np.float64),
        residual_rank=int(ref["residual_rank"]),
        q_nxe=float(moments["q_nxe"]),
        q_residual=float(moments["q_residual"]),
        trace_nxe=float(ref["trace_nxe"]),
        trace_nxe_sq=float(ref["trace_nxe_sq"]),
        annotation_names=names,
        ld_scale=str(ref["ld_scale"]),
        null_corrected=null_corrected,
    )
    fit = solve_normal_equations(
        eq,
        allow_ill_conditioned=allow_ill_conditioned,
        max_condition=max_condition,
    )
    residual_fraction_raw = moments.get("phenotype_residual_variance_fraction")
    if residual_fraction_raw is not None:
        residual_fraction = float(residual_fraction_raw)
        if not np.isfinite(residual_fraction) or not (0.0 < residual_fraction <= 1.0 + 1e-8):
            raise ValueError("Invalid phenotype_residual_variance_fraction in phenotype moments.")
        fit = replace(fit, phenotype_residual_variance_fraction=residual_fraction)
    jackknife_file = ref.get("files", {}).get("jackknife")
    jackknife_declaration = ref.get("jackknife")
    prepared: _PreparedGxEReference
    if jackknife_declaration is not None:
        if not isinstance(jackknife_declaration, dict):
            raise ValueError("Reference jackknife declaration must be an object.")
        jackknife_method = jackknife_declaration.get("method")
        if jackknife_method not in {
            _EXACT_JACKKNIFE_METHOD,
            _BLOCK_LOCAL_JACKKNIFE_METHOD,
        }:
            raise ValueError(f"Unsupported GxE jackknife method {jackknife_method!r}.")
        if "BLOCK" not in diag.columns:
            raise ValueError("Reference declares a jackknife but its diagonal table has no BLOCK column.")
        block_values = pd.to_numeric(diag["BLOCK"], errors="raise").to_numpy(dtype=np.int64)
        declared_labels = jackknife_declaration.get("block_labels")
        if not isinstance(declared_labels, list):
            raise ValueError("Reference jackknife block_labels must be a JSON list.")
        labels = tuple(str(value) for value in declared_labels)
        within = None
        if jackknife_method == _EXACT_JACKKNIFE_METHOD:
            if jackknife_file is None:
                raise ValueError("Exact two-sided GxE jackknife is missing its trace bundle.")
            with np.load(artifact_snapshots["jackknife"].snapshot_path, allow_pickle=False) as bundle:
                expected_members = {
                    "block_labels", "within_xx", "within_xw", "within_wx", "within_ww"
                }
                if len(bundle.files) != len(expected_members) or set(bundle.files) != expected_members:
                    raise ValueError(
                        "Reference jackknife contains unexpected, duplicate, or missing arrays."
                    )
                label_array = np.asarray(bundle["block_labels"])
                if label_array.ndim != 1 or label_array.dtype.kind != "U":
                    raise ValueError("Reference jackknife block_labels must be a one-dimensional Unicode array.")
                if tuple(label_array.tolist()) != labels:
                    raise ValueError("Jackknife trace labels disagree with the reference manifest.")
                within = {}
                for key in ("xx", "xw", "wx", "ww"):
                    raw = np.asarray(bundle[f"within_{key}"])
                    if raw.dtype != np.dtype(np.float64):
                        raise ValueError(f"Reference jackknife within_{key} must use float64.")
                    within[key] = np.array(raw, copy=True)
        elif jackknife_file is not None:
            raise ValueError("Block-local GxE jackknife must not declare an exact trace bundle.")
        nblock = len(labels)
        if (
            nblock < 2
            or any(not label for label in labels)
            or len(set(labels)) != nblock
            or set(np.unique(block_values).tolist()) != set(range(nblock))
        ):
            raise ValueError("Jackknife block IDs are not contiguous or do not match block_labels.")
        if jackknife_declaration.get("num_blocks") != nblock:
            raise ValueError("Jackknife block count disagrees with its labels.")
        if within is not None:
            expected_shape = (nblock, len(names), len(names))
            for key, value in within.items():
                if (
                    value.shape != expected_shape
                    or not np.all(np.isfinite(value))
                    or np.any(value < 0.0)
                ):
                    raise ValueError(
                        f"Jackknife within_{key} has shape {value.shape}; expected finite, "
                        f"non-negative float64 {expected_shape}."
                    )
                value[value == 0.0] = 0.0
        prepared = _prepare_reference_sufficient_statistics(
            path=ref_path,
            payload=ref,
            manifest_sha256=observed_reference_hash,
            schema_version=ref_version,
            feature_cache_sha256=ref_cache_hash,
            reference_provenance=reference_provenance,
            feature_cache_provenance=feature_cache_provenance,
            variants=variants,
            annotations=annotations,
            annotation_names=names,
            panels=panels,
            norm_x=diag["NORM_X"].to_numpy(dtype=np.float64),
            norm_w=diag["NORM_W"].to_numpy(dtype=np.float64),
            diag_x=diag["DNXE_X"].to_numpy(dtype=np.float64),
            diag_w=diag["DNXE_W"].to_numpy(dtype=np.float64),
            equations=eq,
            block_values=block_values,
            block_labels=labels,
            within=within,
            jackknife_method=str(jackknife_method),
            null_corrected=null_corrected,
        )
        _, deleted_equations = _equations_from_prepared_scores(
            prepared,
            score_arrays[0],
            score_arrays[1],
            q_nxe=float(moments["q_nxe"]),
            q_residual=float(moments["q_residual"]),
        )
        replicate_proportions = []
        for deleted in deleted_equations:
            deleted_fit = solve_normal_equations(
                deleted,
                allow_ill_conditioned=allow_ill_conditioned,
                max_condition=max_condition,
            )
            replicate_proportions.append(deleted_fit.proportions)
        replicate_array = np.asarray(replicate_proportions, dtype=np.float64)
        center = replicate_array.mean(axis=0)
        se = np.sqrt((nblock - 1.0) / nblock * np.sum((replicate_array - center) ** 2, axis=0))
        fit = replace(
            fit,
            standard_errors=se,
            jackknife_estimates=replicate_array,
            jackknife_block_labels=labels,
        )
    else:
        if jackknife_file is not None:
            raise ValueError("Reference declares a jackknife artifact without jackknife metadata.")
        prepared = _prepare_reference_sufficient_statistics(
            path=ref_path,
            payload=ref,
            manifest_sha256=observed_reference_hash,
            schema_version=ref_version,
            feature_cache_sha256=ref_cache_hash,
            reference_provenance=reference_provenance,
            feature_cache_provenance=feature_cache_provenance,
            variants=variants,
            annotations=annotations,
            annotation_names=names,
            panels=panels,
            norm_x=diag["NORM_X"].to_numpy(dtype=np.float64),
            norm_w=diag["NORM_W"].to_numpy(dtype=np.float64),
            diag_x=diag["DNXE_X"].to_numpy(dtype=np.float64),
            diag_w=diag["DNXE_W"].to_numpy(dtype=np.float64),
            equations=eq,
            null_corrected=null_corrected,
        )
    fit = replace(
        fit,
        consumed_input_provenance=_consumed_input_provenance(
            reference_manifest=ref_snapshot,
            feature_cache=shared_cache_snapshot,
            phenotype_moments=moments_snapshot,
            gwas=score_snapshots["gwas"],
            gwis=score_snapshots["gwis"],
        ),
    )
    if prepared_out is not None:
        prepared_out.append(prepared)
    return fit, eq


def _fit_prepared_from_input_snapshots(
    prepared: _PreparedGxEReference,
    phenotype_input: GxEPhenotypeInput,
    *,
    snapshots: _InputSnapshotStore,
    validated_cache_hashes: set[str],
    allow_ill_conditioned: bool,
    max_condition: float,
) -> tuple[GxEFitResult, GxENormalEquations]:
    """Validate one phenotype triplet and fit it to an in-memory reference."""
    ref = prepared.payload
    mom_path = Path(phenotype_input.phenotype_moments).expanduser().resolve()
    moments_snapshot = snapshots.capture(mom_path)
    moments = _load_json(moments_snapshot.snapshot_path)
    moments_version_raw = moments.get("schema_version")
    if not isinstance(moments_version_raw, int) or isinstance(moments_version_raw, bool):
        raise ValueError(f"Phenotype schema_version must be a JSON integer: {mom_path}.")
    moments_version = int(moments_version_raw)
    if (
        moments.get("kind") != _MOMENTS_KIND
        or moments_version not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError(f"Unsupported GxE phenotype moments file: {mom_path}.")
    if moments_version != prepared.schema_version:
        raise ValueError(
            "Reference and phenotype bundles use different schema versions: "
            f"reference={prepared.schema_version}, moments={moments_version}."
        )
    for key in ("analysis_fingerprint", "variant_digest", "residual_rank"):
        if str(ref.get(key)) != str(moments.get(key)):
            raise ValueError(f"Reference and phenotype bundles disagree on {key}.")
    if moments.get("score_definition") != (
        "feature_transpose_residualized_y_over_sqrt_residual_rank"
    ):
        raise ValueError("Phenotype moments use an unsupported marginal-score scale.")

    expected_reference_hash = moments.get("reference_manifest_sha256")
    if not _is_sha256(expected_reference_hash):
        raise ValueError("Phenotype moments do not bind the exact GxE reference manifest.")
    ref_cache_hash = prepared.feature_cache_sha256
    moments_cache_hash = moments.get("feature_cache_sha256")
    if moments_cache_hash is not None and not _is_sha256(moments_cache_hash):
        raise ValueError("Phenotype moments contain an invalid feature-cache SHA-256 binding.")
    moments_cache = moments.get("feature_cache")
    verified_moments_cache_hash = None
    moments_cache_snapshot: _InputSnapshot | None = None
    if moments_cache is not None:
        if (
            not isinstance(moments_cache, dict)
            or not isinstance(moments_cache.get("path"), str)
            or not moments_cache["path"]
            or not _is_sha256(moments_cache.get("sha256"))
        ):
            raise ValueError("Phenotype moments have an invalid feature-cache binding.")
        moments_cache_path = _resolve_path(mom_path, moments_cache["path"]).resolve()
        verified_moments_cache_hash = str(moments_cache["sha256"])
        moments_cache_snapshot = snapshots.capture(moments_cache_path)
        if moments_cache_snapshot.sha256 != verified_moments_cache_hash:
            raise ValueError("Phenotype-moment feature cache failed its SHA-256 check.")
        if ref_cache_hash is not None and verified_moments_cache_hash != ref_cache_hash:
            raise ValueError("Phenotype moments and reference point to different feature caches.")
        if verified_moments_cache_hash not in validated_cache_hashes:
            _, arrays = _load_feature_cache_bundle(
                moments_cache_snapshot.snapshot_path
            )
            arrays.clear()
            validated_cache_hashes.add(verified_moments_cache_hash)
        if (
            moments_cache_hash is not None
            and moments_cache_hash != verified_moments_cache_hash
        ):
            raise ValueError("Phenotype moments contain inconsistent feature-cache hashes.")
    if prepared.manifest_sha256 != expected_reference_hash:
        if (
            ref_cache_hash is None
            or verified_moments_cache_hash is None
            or verified_moments_cache_hash != ref_cache_hash
        ):
            raise ValueError(
                "Phenotype moments were generated for a different GxE reference/feature definition."
            )
    elif moments_cache_hash is not None and moments_cache_hash != ref_cache_hash:
        raise ValueError("Phenotype moments and reference declare different feature caches.")

    declared_score_files = moments.get("files")
    if not isinstance(declared_score_files, dict) or not {"gwas", "gwis"}.issubset(
        declared_score_files
    ):
        raise ValueError("Phenotype moments must declare the exact GWAS and GWIS score artifacts.")
    supplied_score_paths = (
        Path(phenotype_input.gwas_scores).expanduser().resolve(),
        Path(phenotype_input.gwis_scores).expanduser().resolve(),
    )
    score_snapshots: dict[str, _InputSnapshot] = {}
    for label, supplied in zip(("gwas", "gwis"), supplied_score_paths):
        declared = _resolve_path(mom_path, str(declared_score_files[label])).resolve()
        if supplied != declared:
            raise ValueError(
                f"Supplied {label.upper()} file {supplied} is not the phenotype-bound artifact {declared}."
            )
        score_hashes = moments.get("score_sha256")
        if not isinstance(score_hashes, dict) or label not in score_hashes:
            raise ValueError("Phenotype moments do not cryptographically bind both score artifacts.")
        expected_hash = score_hashes[label]
        if not _is_sha256(expected_hash):
            raise ValueError(f"Phenotype moments contain an invalid {label.upper()} SHA-256.")
        score_snapshot = snapshots.capture(supplied)
        if score_snapshot.sha256 != str(expected_hash):
            raise ValueError(f"{label.upper()} score file SHA-256 does not match phenotype moments.")
        score_snapshots[label] = score_snapshot

    score_arrays: list[np.ndarray] = []
    for label, score_key, path in zip(
        ("GWAS", "GWIS"), ("gwas", "gwis"), supplied_score_paths
    ):
        table = _read_table(score_snapshots[score_key].snapshot_path)
        required = ["CHR", "SNP", "BP", "A1", "A2", "SCORE", "SCORE_MODE"]
        if prepared.schema_version >= 3:
            canonical_score_columns = [
                "CHR",
                "SNP",
                "BP",
                "A1",
                "A2",
                "N",
                "DF",
                "SCORE_MODE",
                "SCORE",
            ]
            observed_score_columns = table.columns.astype(str).tolist()
            if (
                observed_score_columns != canonical_score_columns
                or len(set(observed_score_columns)) != len(observed_score_columns)
            ):
                raise ValueError(
                    f"{label} score file must contain exactly the canonical ordered columns "
                    f"{canonical_score_columns}; observed {observed_score_columns}."
                )
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"{label} score file {path} is missing columns {missing}.")
        if len(table) != prepared.n_variants:
            raise ValueError(
                f"{label} score file has {len(table)} rows; expected {prepared.n_variants}."
            )
        for column in ("CHR", "SNP", "BP", "A1", "A2"):
            if not np.array_equal(
                table[column].astype(str).to_numpy(),
                prepared.variant_axis[column],
            ):
                raise ValueError(
                    f"{label} score file is not aligned to the reference ({column} mismatch)."
                )
        modes = table["SCORE_MODE"].astype("string").str.lower()
        mode_values = set(modes.dropna().astype(str))
        if modes.isna().any() or not bool((modes == "marginal_cross_product").all()):
            raise ValueError(
                f"{label} uses unsupported SCORE_MODE={sorted(mode_values)}. "
                "Conditional PLINK interaction statistics are not GENIE marginal scores."
            )
        n_values = (
            pd.to_numeric(table["N"], errors="raise").to_numpy(dtype=np.float64)
            if "N" in table
            else None
        )
        df_values = (
            pd.to_numeric(table["DF"], errors="raise").to_numpy(dtype=np.float64)
            if "DF" in table
            else None
        )
        if n_values is None or df_values is None:
            raise ValueError(f"{label} score file must contain N and DF columns.")
        if not np.all(n_values == float(ref["n_samples"])):
            raise ValueError(f"{label} score file N does not match the reference sample count.")
        if not np.all(df_values == float(prepared.residual_rank)):
            raise ValueError(f"{label} score file DF does not match the reference residual rank.")
        score_arrays.append(table["SCORE"].to_numpy(dtype=np.float64))

    equations, deleted_equations = _equations_from_prepared_scores(
        prepared,
        score_arrays[0],
        score_arrays[1],
        q_nxe=float(moments["q_nxe"]),
        q_residual=float(moments["q_residual"]),
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
            raise ValueError("Invalid phenotype_residual_variance_fraction in phenotype moments.")
        fit = replace(fit, phenotype_residual_variance_fraction=residual_fraction)

    if deleted_equations:
        replicate_array = np.asarray(
            [
                solve_normal_equations(
                    deleted,
                    allow_ill_conditioned=allow_ill_conditioned,
                    max_condition=max_condition,
                ).proportions
                for deleted in deleted_equations
            ],
            dtype=np.float64,
        )
        nblock = len(deleted_equations)
        center = replicate_array.mean(axis=0)
        se = np.sqrt(
            (nblock - 1.0)
            / nblock
            * np.sum((replicate_array - center) ** 2, axis=0)
        )
        fit = replace(
            fit,
            standard_errors=se,
            jackknife_estimates=replicate_array,
            jackknife_block_labels=prepared.block_labels,
        )
    feature_cache_provenance = prepared.feature_cache_provenance
    if (
        prepared.feature_cache_sha256 is None
        and moments_cache_snapshot is not None
    ):
        feature_cache_provenance = _snapshot_provenance(moments_cache_snapshot)
    fit = replace(
        fit,
        consumed_input_provenance=GxEConsumedInputProvenance(
            reference_manifest=prepared.reference_provenance,
            feature_cache=feature_cache_provenance,
            phenotype_moments=_snapshot_provenance(moments_snapshot),
            gwas=_snapshot_provenance(score_snapshots["gwas"]),
            gwis=_snapshot_provenance(score_snapshots["gwis"]),
        ),
    )
    return fit, equations


def fit_many_from_files(
    reference_manifest: str | Path,
    phenotype_inputs: Mapping[str, GxEPhenotypeInput],
    *,
    allow_ill_conditioned: bool = False,
    max_condition: float = 1.0e12,
    scratch_dir: str | Path | None = None,
) -> dict[str, tuple[GxEFitResult, GxENormalEquations]]:
    """Fit several phenotype triplets after validating their reference once."""
    if not isinstance(phenotype_inputs, Mapping) or not phenotype_inputs:
        raise ValueError("phenotype_inputs must be a non-empty mapping.")
    normalized: list[tuple[str, GxEPhenotypeInput]] = []
    for raw_name, phenotype_input in phenotype_inputs.items():
        name = str(raw_name)
        if not _FIT_BATCH_NAME.fullmatch(name):
            raise ValueError(f"Invalid GxE batch phenotype name {name!r}.")
        if not isinstance(phenotype_input, GxEPhenotypeInput):
            raise TypeError(
                "Every phenotype_inputs value must be a GxEPhenotypeInput instance."
            )
        normalized.append((name, phenotype_input))
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("GxE batch phenotype names must be unique.")
    normalized_triplets = [
        (
            Path(value.phenotype_moments).expanduser().resolve(),
            Path(value.gwas_scores).expanduser().resolve(),
            Path(value.gwis_scores).expanduser().resolve(),
        )
        for _, value in normalized
    ]
    if len(set(normalized_triplets)) != len(normalized_triplets):
        raise ValueError("GxE batch phenotype input triplets must be unique.")
    if scratch_dir is None:
        scratch_dir = (
            Path(normalized[0][1].phenotype_moments).expanduser().resolve().parent
        )

    results: dict[str, tuple[GxEFitResult, GxENormalEquations]] = {}
    with _InputSnapshotStore(scratch_dir) as snapshots:
        # Capture every explicitly supplied file before parsing any manifest.
        # Manifest-declared cache/reference artifacts are subsequently captured
        # before their own validation and parsing, as in the singleton path.
        snapshots.capture(reference_manifest)
        for _, phenotype_input in normalized:
            snapshots.capture(phenotype_input.phenotype_moments)
            snapshots.capture(phenotype_input.gwas_scores)
            snapshots.capture(phenotype_input.gwis_scores)

        first_name, first_input = normalized[0]
        prepared_out: list[_PreparedGxEReference] = []
        results[first_name] = _fit_from_input_snapshots(
            reference_manifest,
            first_input.phenotype_moments,
            first_input.gwas_scores,
            first_input.gwis_scores,
            snapshots=snapshots,
            allow_ill_conditioned=allow_ill_conditioned,
            max_condition=max_condition,
            prepared_out=prepared_out,
        )
        if len(prepared_out) != 1:
            raise RuntimeError("GxE reference preparation did not produce one reusable state.")
        prepared = prepared_out[0]
        validated_cache_hashes = (
            {prepared.feature_cache_sha256}
            if prepared.feature_cache_sha256 is not None
            else set()
        )
        for name, phenotype_input in normalized[1:]:
            results[name] = _fit_prepared_from_input_snapshots(
                prepared,
                phenotype_input,
                snapshots=snapshots,
                validated_cache_hashes=validated_cache_hashes,
                allow_ill_conditioned=allow_ill_conditioned,
                max_condition=max_condition,
            )
    return results


def load_fit_batch_manifest(
    manifest_path: str | Path,
    *,
    scratch_dir: str | Path | None = None,
) -> tuple[Path, tuple[GxEFitBatchEntry, ...]]:
    """Snapshot and validate a strict no-overwrite batch-fit manifest."""
    path = Path(manifest_path).expanduser().resolve()
    if scratch_dir is None:
        scratch_dir = path.parent
    with _InputSnapshotStore(scratch_dir) as snapshots:
        snapshot = snapshots.capture(path)
        payload = _load_json(snapshot.snapshot_path)
    expected_root = {"kind", "schema_version", "reference", "traits"}
    if set(payload) != expected_root:
        raise ValueError(
            f"GxE fit-batch manifest must contain exactly {sorted(expected_root)}."
        )
    if payload.get("kind") != _FIT_BATCH_KIND:
        raise ValueError("Unsupported GxE fit-batch manifest kind.")
    version = payload.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != _FIT_BATCH_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported GxE fit-batch manifest schema_version.")
    reference_value = payload.get("reference")
    if not isinstance(reference_value, str) or not reference_value:
        raise ValueError("GxE fit-batch reference must be a non-empty path string.")
    traits = payload.get("traits")
    if not isinstance(traits, list) or not traits:
        raise ValueError("GxE fit-batch traits must be a non-empty list.")

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate.resolve()

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
        if any(not isinstance(record[key], str) or not record[key] for key in expected_trait):
            raise ValueError(f"GxE fit-batch trait {index} values must be non-empty strings.")
        name = record["name"]
        if not _FIT_BATCH_NAME.fullmatch(name) or name in names:
            raise ValueError(f"Invalid or duplicate GxE fit-batch trait name {name!r}.")
        output = resolve(record["out"])
        if output in outputs:
            raise ValueError(f"Duplicate GxE fit-batch output prefix {output}.")
        input_triplet = (
            resolve(record["moments"]),
            resolve(record["gwas"]),
            resolve(record["gwis"]),
        )
        if input_triplet in inputs:
            raise ValueError("Duplicate GxE fit-batch phenotype input triplet.")
        names.add(name)
        outputs.add(output)
        inputs.add(input_triplet)
        entries.append(
            GxEFitBatchEntry(
                name=name,
                phenotype_input=GxEPhenotypeInput(
                    phenotype_moments=input_triplet[0],
                    gwas_scores=input_triplet[1],
                    gwis_scores=input_triplet[2],
                ),
                output_prefix=output,
            )
        )
    output_parents = {entry.output_prefix.parent for entry in entries}
    if len(output_parents) != 1:
        raise ValueError(
            "All GxE fit-batch output prefixes must share one parent directory."
        )
    planned_outputs = sorted(
        path
        for entry in entries
        for path in (
            Path(str(entry.output_prefix) + ".gxe.results.tsv"),
            Path(str(entry.output_prefix) + ".gxe.fit.json"),
        )
    )
    existing = [path for path in planned_outputs if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing GxE fit-batch output(s): "
            + ", ".join(str(path) for path in existing)
        )
    return resolve(reference_value), tuple(entries)


def write_fit(
    prefix: str | Path,
    fit: GxEFitResult,
    equations: GxENormalEquations,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    consumed_input_provenance = _serialize_consumed_input_provenance(
        fit.consumed_input_provenance
    )
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
            "kernel_trace": equations.traces,
            "variance_contribution": fit.contributions,
            "proportion": fit.proportions,
            "proportion_se": (
                fit.standard_errors
                if fit.standard_errors is not None
                else np.full(len(fit.component_names), np.nan)
            ),
            "original_scale_proportion": (
                fit.proportions * fit.phenotype_residual_variance_fraction
                if fit.phenotype_residual_variance_fraction is not None
                else np.full(len(fit.component_names), np.nan)
            ),
        }
    )
    frame["z"] = frame["proportion"] / frame["proportion_se"]
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
        "consumed_input_provenance": consumed_input_provenance,
        "component_names": list(fit.component_names),
        "rank": fit.rank,
        "condition_number": fit.condition_number,
        "relative_residual": fit.relative_residual,
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
        "phenotype_residual_variance_fraction": fit.phenotype_residual_variance_fraction,
    }
    if fit.standard_errors is not None:
        payload["standard_errors"] = fit.standard_errors.tolist()
        payload["original_scale_standard_errors"] = (
            (fit.standard_errors * fit.phenotype_residual_variance_fraction).tolist()
            if fit.phenotype_residual_variance_fraction is not None
            else None
        )
    if fit.jackknife_estimates is not None:
        payload["jackknife_estimates"] = fit.jackknife_estimates.tolist()
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
        expected_hashes = {
            table_path: _sha256_file(staged_table),
            json_path: _sha256_file(staged_json),
        }

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
                if _sha256_file(path) != expected_hashes[path]:
                    raise RuntimeError(
                        f"A published GxE fit output was modified in place: {path}."
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
        expected_hashes = {
            final: _sha256_file(staged)
            for _, staged_table, staged_manifest, table, manifest in staged_pairs
            for staged, final in (
                (staged_table, table),
                (staged_manifest, manifest),
            )
        }

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
                if _sha256_file(path) != expected_hashes[path]:
                    raise RuntimeError(
                        f"A published GxE batch output was modified in place: {path}."
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
