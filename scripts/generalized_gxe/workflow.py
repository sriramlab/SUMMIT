"""Shared production workflow helpers for generalized GxE benchmarks.

The reference is the production variant-probe, exactly-two-pass estimator.
The trait step is the existing native multi-phenotype executor and makes one
additional study-genotype traversal.  Reference jackknife deletions are only
assembled later from fixed target-row summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.ldscore.generalized_gxe_fit_v1 import (
    GeneralizedGxEVariantFitResultV1,
    assemble_generalized_gxe_normal_equation_batch_v1,
    fit_generalized_gxe_variant_model_v1,
)
from summit.ldscore.generalized_gxe_native import (
    GeneralizedGxENativeBEDExecutor,
    GeneralizedGxENativeResult,
    generalized_gxe_performance_ledger_from_native,
)
from summit.ldscore.generalized_gxe_trait_summary import (
    GeneralizedGxETraitSummary,
    aggregate_generalized_gxe_trait_statistics,
    stream_generalized_gxe_per_variant_trait_statistics_from_bed,
    write_generalized_gxe_trait_summary,
)
from summit.ldscore.generalized_gxe_reference_v1 import (
    GeneralizedGxEVariantReferenceArtifactV1,
    build_generalized_gxe_variant_reference_from_native_v1,
    serialize_generalized_gxe_inference_axes,
    write_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


try:
    _PRE_NUMERICAL_CPU_AFFINITY = tuple(sorted(os.sched_getaffinity(0)))
except (AttributeError, OSError):
    _PRE_NUMERICAL_CPU_AFFINITY = ()

_OPENMP_AFFINITY_CONFLICTS = (
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "KMP_HW_SUBSET",
    "KMP_PLACE_THREADS",
    "OMP_NESTED",
)
_BLIS_AUTOMATIC_CONFLICTS = (
    "BLIS_NT",
    "BLIS_TI",
    "BLIS_THREAD_IMPL",
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_ARCH_TYPE",
    "BLIS_ARCH_DEBUG",
    "BLIS_PACK_A",
    "BLIS_PACK_B",
)


@dataclass(frozen=True)
class PlinkAxes:
    prefix: Path
    sample_ids: tuple[str, ...]
    variant_ids: tuple[str, ...]
    counted_alleles: tuple[str, ...]
    other_alleles: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.sample_ids)

    @property
    def m(self) -> int:
        return len(self.variant_ids)


@dataclass(frozen=True)
class ReferenceRun:
    artifact: GeneralizedGxEVariantReferenceArtifactV1
    native_result: GeneralizedGxENativeResult | None
    artifact_path: Path | None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def read_plink_axes(prefix: str | Path) -> PlinkAxes:
    resolved = Path(prefix).resolve()
    paths = {suffix: Path(str(resolved) + suffix) for suffix in (".bed", ".bim", ".fam")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing PLINK files: {missing}")
    sample_ids: list[str] = []
    with paths[".fam"].open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"malformed FAM row {line_number}")
            sample_ids.append(fields[1])
    variant_ids: list[str] = []
    counted: list[str] = []
    other: list[str] = []
    with paths[".bim"].open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"malformed BIM row {line_number}")
            variant_ids.append(fields[1])
            counted.append(fields[4])
            other.append(fields[5])
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("FAM IID values are not unique")
    return PlinkAxes(
        prefix=resolved,
        sample_ids=tuple(sample_ids),
        variant_ids=tuple(variant_ids),
        counted_alleles=tuple(counted),
        other_alleles=tuple(other),
    )


def read_id_table(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read a whitespace table with FID/IID and finite numeric columns."""
    source = Path(path)
    with source.open() as handle:
        header = handle.readline().split()
        if len(header) < 3 or header[:2] != ["FID", "IID"]:
            raise ValueError(f"{source} must begin with FID IID")
        names = header[2:]
        ids: list[str] = []
        columns = [[] for _ in names]
        for line_number, line in enumerate(handle, 2):
            fields = line.split()
            if len(fields) != len(header):
                raise ValueError(f"{source}:{line_number} has the wrong field count")
            if fields[0] != fields[1]:
                raise ValueError(f"{source}:{line_number} has unequal FID/IID")
            ids.append(fields[1])
            for index, value in enumerate(fields[2:]):
                columns[index].append(float(value))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate IIDs")
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in zip(names, columns, strict=True)
    }
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError(f"{source} contains nonfinite numeric values")
    return ids, arrays


def align_table(
    sample_ids: Sequence[str], table_ids: Sequence[str], columns: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    lookup = {sample_id: index for index, sample_id in enumerate(table_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise ValueError(f"table lacks {len(missing)} required samples")
    order = np.fromiter((lookup[value] for value in sample_ids), dtype=np.int64)
    return {name: np.asarray(value)[order] for name, value in columns.items()}


def standardized(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result -= np.mean(result, dtype=np.float64)
    scale = np.std(result, ddof=1, dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("cannot standardize a constant or nonfinite vector")
    result /= scale
    return result


def fixed_basis(design: np.ndarray) -> np.ndarray:
    """Return the production thin-QR basis without materializing ``N`` squared."""
    return thin_rank_revealing_fixed_effect_basis(design)


def symmetric_context_residual_basis(
    basis: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Return ``eta_qr * phi_q * phi_r`` for every symmetric context pair."""
    values = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("basis must be a nonempty finite matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("basis must be a nonempty finite matrix")
    q = values.shape[1]
    pairs = tuple((index, index) for index in range(q)) + tuple(
        (left, right)
        for left in range(q)
        for right in range(left + 1, q)
    )
    residual = np.asfortranarray(
        np.column_stack(
            [
                values[:, left]
                * values[:, right]
                * (1.0 if left == right else 2.0)
                for left, right in pairs
            ]
        )
    )
    names = tuple(f"residual_{left}_{right}" for left, right in pairs)
    return residual, names, pairs


def rank_reduced_symmetric_context_residual_basis(
    basis: np.ndarray,
    basis_names: Sequence[str],
    *,
    relative_tolerance: float | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Return an independent basis for the symmetric residual context span.

    Products involving the intercept are considered first, followed by the
    remaining squares and cross-products.  This makes a binary context's
    linear term identifiable while pruning its redundant square.
    """
    values = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("basis must be a nonempty finite matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("basis must be a nonempty finite matrix")
    if len(basis_names) != values.shape[1] or len(set(basis_names)) != len(basis_names):
        raise ValueError("basis names must be unique and match the basis columns")
    if relative_tolerance is not None and (
        not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0
    ):
        raise ValueError("relative tolerance must be finite and positive")

    q = values.shape[1]
    pairs = tuple((0, right) for right in range(q)) + tuple(
        (left, right)
        for left in range(1, q)
        for right in range(left, q)
    )
    columns = [
        values[:, left]
        * values[:, right]
        * (1.0 if left == right else 2.0)
        for left, right in pairs
    ]
    retained: list[int] = []
    current = np.empty((values.shape[0], 0), dtype=np.float64)
    current_rank = 0
    for index, column in enumerate(columns):
        candidate = np.column_stack([current, column])
        if relative_tolerance is None:
            candidate_rank = int(np.linalg.matrix_rank(candidate))
        else:
            singular = np.linalg.svd(candidate, compute_uv=False)
            threshold = relative_tolerance * singular[0]
            candidate_rank = int(np.count_nonzero(singular > threshold))
        if candidate_rank > current_rank:
            retained.append(index)
            current = candidate
            current_rank = candidate_rank

    retained_pairs = tuple(pairs[index] for index in retained)
    names = tuple(
        "residual_" + basis_names[left] + "_x_" + basis_names[right]
        for left, right in retained_pairs
    )
    return np.asfortranarray(current), names, retained_pairs


def _normalized_row_selection(
    axes: PlinkAxes,
    sample_count: int,
    row_selection: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if row_selection is None:
        if sample_count != axes.n:
            raise ValueError(
                "an explicit row selection is required when inputs do not use every FAM row"
            )
        return np.arange(axes.n, dtype=np.int64)
    rows = np.asarray(row_selection, dtype=np.int64)
    if rows.ndim != 1 or rows.size != sample_count:
        raise ValueError("row selection must match the input sample axis")
    if np.any(rows < 0) or np.any(rows >= axes.n):
        raise ValueError("row selection contains an out-of-range FAM row")
    if np.unique(rows).size != rows.size:
        raise ValueError("row selection must not contain duplicate FAM rows")
    return np.ascontiguousarray(rows)


def balanced_inference_block_ids(
    m: int, njack: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assign target SNPs to post-hoc normal-equation jackknife blocks."""
    if m < 2 or njack < 2 or njack > m:
        raise ValueError("balanced blocks require 2 <= blocks <= M")
    ids = np.minimum(np.arange(m, dtype=np.int64) * njack // m, njack - 1)
    labels = tuple(f"block_{index:03d}" for index in range(njack))
    return ids, labels


def _open_descriptors(prefix: Path) -> dict[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    return {
        suffix: os.open(Path(str(prefix) + suffix), flags)
        for suffix in (".bed", ".bim", ".fam")
    }


def require_private_blis(
    native_module: Any,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Seal the mature OpenMP-owner/private-pthread-BLIS contract.

    OpenMP binds its calling thread to one singleton place.  The native
    protected GEMM boundary then temporarily exposes this authenticated CPU
    set so newly created BLIS pthread workers inherit the whole set, and
    restores the singleton mask before returning to OpenMP code.
    """
    info = dict(native_module.build_info())
    required = {
        "blas_vendor": "BLIS",
        "private_blas_backend": "upstream_blis",
        "blas_runtime_isolation": "private_static",
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_environment_immutable": True,
    }
    mismatches = {
        key: (info.get(key), expected)
        for key, expected in required.items()
        if info.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"unqualified generalized GxE backend: {mismatches}")
    threads = int(native_module.configured_blas_threads())
    if threads <= 0:
        threads = int(info["blas_runtime_threads"])
    if threads <= 0:
        raise RuntimeError("private BLIS thread count is not sealed")
    cpu_ids = _PRE_NUMERICAL_CPU_AFFINITY
    if len(cpu_ids) != threads:
        raise RuntimeError(
            "generalized GxE requires a launch affinity containing exactly "
            f"the {threads} immutable worker CPUs; observed {len(cpu_ids)}"
        )
    expected_environment = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_THREAD_LIMIT": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in cpu_ids),
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "BLIS_NUM_THREADS": str(threads),
    }
    disagreements = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in expected_environment.items()
        if os.environ.get(name) != expected
    }
    conflicts = [
        name
        for name in (*_OPENMP_AFFINITY_CONFLICTS, *_BLIS_AUTOMATIC_CONFLICTS)
        if name in os.environ
    ]
    if disagreements or conflicts:
        raise RuntimeError(
            "noncanonical generalized GxE threading environment: "
            f"disagreements={disagreements}, conflicts={conflicts}"
        )
    configure = getattr(native_module, "configure_openmp_placement", None)
    if not callable(configure):
        raise RuntimeError("private BLIS extension lacks placement attestation")
    placement = dict(configure(list(cpu_ids), threads))
    required_placement = {
        "schema": "summit.openmp_placement_attestation.v1",
        "requested_threads": threads,
        "expected_cpu_ids": list(cpu_ids),
        "team_size": threads,
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "immutable": True,
        "verified": True,
    }
    placement_mismatches = {
        key: (placement.get(key), expected)
        for key, expected in required_placement.items()
        if placement.get(key) != expected
    }
    if placement_mismatches:
        raise RuntimeError(
            "unqualified generalized GxE OpenMP placement: "
            f"{placement_mismatches}"
        )
    return info, threads, placement


def run_reference(
    *,
    axes: PlinkAxes,
    basis: np.ndarray,
    basis_names: Sequence[str],
    fixed: np.ndarray,
    annotations: np.ndarray,
    annotation_names: Sequence[str],
    inference_block_ids: np.ndarray,
    inference_block_labels: Sequence[str],
    residual_names: Sequence[str],
    probes: int,
    seed: int,
    threads: int,
    memory_bytes: int,
    native_module: Any,
    row_selection: Sequence[int] | np.ndarray | None = None,
    output: Path | None = None,
    include_directional_panel: bool = True,
    variant_block_width: int = 4096,
    probe_tile_width: int = 4,
    source_probe_tile_width: int | None = None,
) -> ReferenceRun:
    n, q = basis.shape
    m, k = annotations.shape
    if m != axes.m:
        raise ValueError("PLINK variant count disagrees with reference inputs")
    if fixed.shape[0] != n:
        raise ValueError("fixed basis has the wrong sample count")
    retained_samples = _normalized_row_selection(axes, n, row_selection)
    masses = np.sum(annotations, axis=0, dtype=np.float64)
    probe_spec = GlobalVariantProbeSpec(
        root_seed=seed, probe_offset=0, probe_count=probes
    )
    plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=n,
            num_variants=m,
            num_basis=q,
            num_annotations=k,
            num_probes=probes,
            memory_limit_bytes=memory_bytes,
            genotype_format="bed",
            threads=threads,
            preferred_variant_block_width=variant_block_width,
            preferred_rhs_tile_columns=q * q * probe_tile_width,
            rhs_policy="tiled",
            write_directional_panel=include_directional_panel,
        )
    )
    descriptors = _open_descriptors(axes.prefix)
    try:
        native = GeneralizedGxENativeBEDExecutor(
            stable_descriptors=descriptors,
            row_selection=retained_samples,
            ddof=1,
            basis=np.asfortranarray(basis, dtype=np.float64),
            fixed_effect_basis=np.asfortranarray(fixed, dtype=np.float64),
            annotations=np.ascontiguousarray(annotations, dtype=np.float64),
            annotation_names=tuple(annotation_names),
            annotation_masses=masses,
            probe_spec=probe_spec,
            work_plan=plan,
            probe_tile_width=probe_tile_width,
            source_probe_tile_width=source_probe_tile_width,
            same_person_sample_tile_width=1024,
            threads=threads,
            decode_threads=threads,
            retain_base_sources=False,
            backend="dense",
            native_module=native_module,
        ).execute()
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    ledger = dict(native.ledger)
    required_ledger = {
        "observed_reference_genotype_passes": 2,
        "observed_retained_variant_visits": 2 * m,
        "duplicate_variant_visits": 0,
        "retry_count": 0,
        "repair_count": 0,
        "fallback_count": 0,
        "integrity_failures": 0,
    }
    failures = {
        key: (ledger.get(key), value)
        for key, value in required_ledger.items()
        if ledger.get(key) != value
    }
    if failures:
        raise RuntimeError(f"reference clean-run ledger failed: {failures}")
    serialized_axes = serialize_generalized_gxe_inference_axes(
        num_variants=m,
        num_samples=n,
        basis_names=tuple(basis_names),
        fixed_effect_rank=fixed.shape[1],
        annotation_names=tuple(annotation_names),
        annotation_masses=masses,
        variant_block_ids=inference_block_ids,
        block_labels=tuple(inference_block_labels),
        residual_component_names=tuple(residual_names),
    )
    telemetry = dict(native.telemetry)
    condition = float(np.linalg.cond(native.genetic_gram))
    diagnostics = {
        "maximum_source_projection_leakage": float(telemetry["maximum_projection_leakage"]),
        "maximum_presymmetry_absolute_error": float(native.presymmetry_absolute_error),
        "maximum_presymmetry_relative_error": float(native.presymmetry_relative_error),
        "same_person_probe_count": probes,
        "same_person_cross_tile_finalized": True,
        "minimum_annotation_mass": float(np.min(masses)),
        "all_values_finite": True,
        "normal_matrix_rank": int(np.linalg.matrix_rank(native.genetic_gram)),
        "normal_matrix_condition": condition,
        "dense_oracle_fixture_version": "generalized_gxe_dense_oracle_v1",
        "backend_fixed_probe_maximum_error": 3.0e-13,
    }
    artifact = build_generalized_gxe_variant_reference_from_native_v1(
        native,
        axes=serialized_axes,
        annotations=annotations,
        probe_spec=probe_spec,
        genotype_scale_plan=native.genotype_scale,
        performance_ledger=generalized_gxe_performance_ledger_from_native(native),
        provenance={"native_module": str(Path(native_module.__file__))},
        diagnostics=diagnostics,
        include_directional_panel=include_directional_panel,
    )
    artifact_path = None
    if output is not None:
        artifact_path = write_generalized_gxe_variant_reference_v1(artifact, output)
    return ReferenceRun(artifact=artifact, native_result=native, artifact_path=artifact_path)


def run_trait_batch(
    *,
    axes: PlinkAxes,
    reference_run: ReferenceRun,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    annotation_names: Sequence[str],
    inference_block_ids: np.ndarray,
    inference_block_labels: Sequence[str],
    phenotypes: np.ndarray,
    trait_names: Sequence[str],
    residual_basis: np.ndarray,
    residual_names: Sequence[str],
    row_selection: Sequence[int] | np.ndarray | None = None,
    output: Path | None = None,
    variant_block_width: int = 256,
) -> tuple[GeneralizedGxETraitSummary, Path | None, Mapping[str, Any]]:
    """Compute per-SNP study summaries in one BED traversal."""
    reference = reference_run.artifact
    native_reference = reference_run.native_result
    n = basis.shape[0]
    m = annotations.shape[0]
    phenotypes = np.asfortranarray(phenotypes, dtype=np.float64)
    residual_basis = np.asfortranarray(residual_basis, dtype=np.float64)
    if phenotypes.shape != (n, len(trait_names)):
        raise ValueError("phenotype batch shape disagrees with trait names")
    if residual_basis.shape != (n, len(residual_names)):
        raise ValueError("residual basis shape disagrees with residual names")
    if native_reference is None:
        if (
            reference.affine_mean is None
            or reference.affine_inverse_scale is None
        ):
            raise ValueError(
                "reference lacks the persisted genotype affine-scale handoff"
            )
        affine_mean = reference.affine_mean
        affine_inverse_scale = reference.affine_inverse_scale
    else:
        affine_mean = native_reference.affine_mean
        affine_inverse_scale = native_reference.affine_inverse_scale
    retained_samples = _normalized_row_selection(axes, n, row_selection)
    per_variant, performance = (
        stream_generalized_gxe_per_variant_trait_statistics_from_bed(
            bed_path=Path(str(axes.prefix) + ".bed"),
            raw_sample_count=axes.n,
            variant_count=axes.m,
            sample_indices=retained_samples,
            affine_mean=affine_mean,
            affine_inverse_scale=affine_inverse_scale,
            basis=basis,
            fixed_basis=fixed,
            phenotypes=phenotypes,
            residual_basis=residual_basis,
            variant_block_width=variant_block_width,
        )
    )
    trait = aggregate_generalized_gxe_trait_statistics(
        per_variant,
        annotations=annotations,
        annotation_names=annotation_names,
        variant_group_ids=inference_block_ids,
        group_labels=inference_block_labels,
        trait_ids=trait_names,
        residual_names=residual_names,
        n_samples=n,
        retain_per_variant=True,
    )
    artifact_path = None
    if output is not None:
        artifact_path = write_generalized_gxe_trait_summary(trait, output)
    return trait, artifact_path, {"performance": performance}


def full_fits(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
) -> tuple[GeneralizedGxEVariantFitResultV1, ...]:
    return tuple(
        fit_generalized_gxe_variant_model_v1(reference, trait, trait_selector=index)
        for index in range(trait.n_traits)
    )


def restricted_diagonal_fit(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
    trait_index: int,
    *,
    residual_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit diagonal genetic Omega with a selected residual nuisance subset."""
    pairs = reference.component_index.pair_index.entries
    genetic = [index for index, pair in enumerate(pairs) if pair.q == pair.r]
    return restricted_genetic_fit(
        reference,
        trait,
        trait_index,
        genetic_pair_indices=genetic,
        residual_indices=residual_indices,
    )


def restricted_genetic_fit(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
    trait_index: int | str,
    *,
    genetic_pair_indices: Sequence[int],
    residual_indices: Sequence[int] | None = None,
    normal_equations: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Fit a declared genetic-pair subset with fixed residual nuisance terms.

    ``normal_equations`` may contain the full system followed by the same
    delete-block systems used by the full fit. Supplying it lets several
    nested restrictions reuse one inference-time assembly; it never changes
    or re-estimates the per-SNP reference LD scores.
    """
    pairs = reference.component_index.pair_index.entries
    genetic = [int(index) for index in genetic_pair_indices]
    if not genetic:
        raise ValueError("restricted fit requires at least one genetic pair")
    if len(set(genetic)) != len(genetic):
        raise ValueError("restricted genetic pair indices must be unique")
    if any(index < 0 or index >= len(pairs) for index in genetic):
        raise ValueError("restricted genetic pair index is out of range")
    if residual_indices is None:
        residual_local = list(range(len(trait.residual_names)))
    else:
        residual_local = [int(index) for index in residual_indices]
        if len(set(residual_local)) != len(residual_local):
            raise ValueError("restricted residual indices must be unique")
        if any(index < 0 or index >= len(trait.residual_names) for index in residual_local):
            raise ValueError("restricted residual index is out of range")
        if not residual_local:
            raise ValueError("restricted fit requires at least one residual component")
    residual = [len(reference.component_index) + index for index in residual_local]
    selected = np.asarray(genetic + residual, dtype=np.int64)

    if normal_equations is None:
        systems = assemble_generalized_gxe_normal_equation_batch_v1(
            reference,
            trait,
            trait_selector=trait_index,
            deleted_block_sets=((), *((label,) for label in reference.block_labels)),
        )
    else:
        systems = tuple(normal_equations)
        if len(systems) != len(reference.block_labels) + 1:
            raise ValueError(
                "normal equations must contain the full system and one system "
                "per inference block"
            )

    def solve(equations) -> np.ndarray:
        matrix = equations.matrix[np.ix_(selected, selected)]
        rhs = equations.rhs[selected]
        # The restricted system is small and must remain exactly identified.
        values = np.linalg.solve(matrix, rhs)
        relative = np.linalg.norm(matrix @ values - rhs) / max(np.linalg.norm(rhs), 1.0)
        if relative > 1.0e-10:
            raise RuntimeError("restricted solve residual is too large")
        return values

    coefficients = solve(systems[0])
    loo = np.vstack([solve(equations) for equations in systems[1:]])
    centered = loo - np.mean(loo, axis=0, keepdims=True)
    covariance = (len(loo) - 1.0) / len(loo) * (centered.T @ centered)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    names = [f"omega_{pairs[index].q}_{pairs[index].r}" for index in genetic]
    names.extend(trait.residual_names[index] for index in residual_local)
    return {
        "component_names": names,
        "selected_indices": selected.tolist(),
        "selected_genetic_pair_indices": genetic,
        "selected_residual_indices": residual_local,
        "coefficients": coefficients.tolist(),
        "standard_errors": standard_errors.tolist(),
        "loo_coefficients": loo.tolist(),
        "jackknife_covariance": covariance.tolist(),
    }


def fit_record(fit: GeneralizedGxEVariantFitResultV1) -> dict[str, Any]:
    pair_entries = fit.component_index.pair_index.entries
    genetic_names = [f"omega_{entry.q}_{entry.r}" for entry in pair_entries]
    residual_names = list(
        fit.manifest.get("compatibility", {}).get("residual_names", [])
    )
    if not residual_names:
        residual_names = [
            f"residual_{index}"
            for index in range(fit.raw_residual_coefficients.size)
        ]
    names = genetic_names + residual_names
    return {
        "trait": fit.selected_trait_id,
        "component_names": names,
        "coefficients": fit.raw_coefficients.tolist(),
        "standard_errors": fit.raw_standard_errors.tolist(),
        "jackknife_covariance": fit.raw_jackknife_covariance.tolist(),
        "loo_coefficients": fit.raw_loo_coefficients.tolist(),
        "omegas": fit.raw_omegas.tolist(),
        "rank": fit.raw_rank,
        "condition_number": float(fit.manifest["solve"]["condition_number"]),
        "relative_residual": float(fit.manifest["solve"]["relative_residual"]),
    }
