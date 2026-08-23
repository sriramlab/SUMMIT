"""Dimension-only resource planning for disjoint contextual annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .oracle import symmetric_rank_diagnostics


@dataclass(frozen=True)
class AnnotationResourceRequest:
    n_reference: int
    n_study: int
    n_variants: int
    k: int
    q: int
    h: int
    loo_groups: int
    gram_probes: int = 64
    same_person_probes: int = 64
    probe_tile_size: int = 32
    memory_cap_bytes: int = 8 * 1024**3
    reference_method: str = "hutchinson"
    dtype_bytes: int = 8


@dataclass(frozen=True)
class AnnotationResourceEstimate:
    request: AnnotationResourceRequest
    p_genetic: int
    p_total: int
    selected_probe_tile_size: int
    buffer_bytes: dict[str, int]
    storage_bytes: dict[str, int]
    operation_counts: dict[str, int]
    dominant_flop_proxies: dict[str, int]
    operator_peak_bytes: int
    current_python_peak_bytes: int
    within_memory_cap: bool
    current_python_within_memory_cap: bool
    condition_status: str
    pilot_rank: int | None
    pilot_condition_number: float | None
    pilot_eigenvalues: np.ndarray | None
    verdict: str
    manifest: dict[str, Any]


def _positive(name: str, value: int, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def estimate_annotation_resources(
    request: AnnotationResourceRequest,
    pilot_normal_matrix: object | None = None,
) -> AnnotationResourceEstimate:
    """Estimate buffers without allocating arrays of the requested dimensions.

    The operator estimate describes the intended block-streamed implementation.
    The separate Python-development estimate includes the currently materialized
    Q x N x M features and Pg x Q x M x tile Hutchinson buffer.
    """
    if not isinstance(request, AnnotationResourceRequest):
        raise TypeError("request must be an AnnotationResourceRequest.")
    n_reference = _positive("n_reference", request.n_reference)
    n_study = _positive("n_study", request.n_study)
    n_variants = _positive("n_variants", request.n_variants)
    k_count = _positive("k", request.k)
    q_count = _positive("q", request.q)
    h_count = _positive("h", request.h, allow_zero=True)
    group_count = _positive("loo_groups", request.loo_groups)
    gram_probes = _positive("gram_probes", request.gram_probes)
    same_probes = _positive("same_person_probes", request.same_person_probes)
    requested_tile = _positive("probe_tile_size", request.probe_tile_size)
    cap = _positive("memory_cap_bytes", request.memory_cap_bytes)
    scalar = _positive("dtype_bytes", request.dtype_bytes)
    if q_count > 4:
        raise ValueError("The correctness framework limits Q to at most 4.")
    if request.reference_method not in {"exact", "hutchinson"}:
        raise ValueError("reference_method must be 'exact' or 'hutchinson'.")
    pair_count = q_count * (q_count + 1) // 2
    p_genetic = k_count * pair_count
    p_total = p_genetic + h_count

    # Planned streamed operator buffers.  This is the cap-controlled path.
    operator_fixed = scalar * (
        p_total * p_total
        + 2 * p_genetic * p_genetic
        + n_reference * p_genetic
        + p_genetic * p_genetic
    )
    # Keep all Q source sketches U_q (M x b), then stream one source q at a
    # time through its K annotation-weighted targets (N x b).  The complete
    # action tile (N x b x Pg) is retained for the Gram reduction.  KQ target
    # products are performed over a tile, but only K targets are resident at
    # once; operation count and peak storage are deliberately distinct.
    operator_per_tile = scalar * (
        n_variants * q_count + n_reference * k_count + n_reference * p_genetic
    )
    if operator_fixed >= cap:
        selected_tile = 0
    else:
        selected_tile = min(
            requested_tile,
            gram_probes,
            max(0, (cap - operator_fixed) // max(operator_per_tile, 1)),
        )
    action_tile = scalar * n_reference * selected_tile * p_genetic
    source_tile = scalar * n_variants * selected_tile * q_count
    annotation_target_tile = scalar * n_reference * selected_tile * k_count
    operator_peak = operator_fixed + action_tile + source_tile + annotation_target_tile

    reference_group_contribution = scalar * group_count * p_genetic * p_genetic
    reference_group_mass = scalar * group_count * k_count
    reference_snp_contribution = scalar * n_variants * p_genetic * p_genetic
    reference_snp_weights = scalar * n_variants * k_count
    trait_group_contribution = scalar * group_count * p_genetic * (h_count + 2)
    trait_snp_contribution = scalar * n_variants * p_genetic * (h_count + 2)
    grouped_reference_total = (
        reference_group_contribution
        + reference_group_mass
        + scalar * group_count
        + scalar * (2 * p_genetic * p_genetic + k_count)
    )
    snp_reference_total = (
        reference_snp_contribution
        + reference_snp_weights
        + scalar * (2 * p_genetic * p_genetic + k_count)
    )
    grouped_trait_total = (
        trait_group_contribution
        + scalar * group_count * k_count
        + scalar * group_count
        + scalar
        * (p_genetic * (h_count + 2) + h_count * h_count + 2 * h_count + k_count)
    )
    snp_trait_total = (
        trait_snp_contribution
        + scalar * n_variants * k_count
        + scalar
        * (p_genetic * (h_count + 2) + h_count * h_count + 2 * h_count + k_count)
    )

    python_features = scalar * q_count * n_reference * n_variants
    python_exact_kernels = scalar * p_genetic * n_reference * n_reference
    python_exact_raw_targets = scalar * p_genetic * q_count * n_variants
    python_hutch_feature_action = (
        scalar * p_genetic * q_count * n_variants * max(selected_tile, 1)
    )
    full_gram_probe_panel = scalar * n_reference * gram_probes
    full_variant_probe_panel = scalar * n_variants * same_probes
    if request.reference_method == "exact":
        python_peak = (
            python_features
            + python_exact_kernels
            + python_exact_raw_targets
            + reference_group_contribution
        )
    else:
        python_peak = (
            python_features
            + python_hutch_feature_action
            + action_tile
            + full_gram_probe_panel
            + full_variant_probe_panel
            + reference_group_contribution
        )

    condition_status = "unknown_without_pilot"
    pilot_rank: int | None = None
    pilot_condition: float | None = None
    pilot_eigenvalues: np.ndarray | None = None
    if pilot_normal_matrix is not None:
        matrix = np.asarray(pilot_normal_matrix, dtype=np.float64)
        if matrix.shape != (p_total, p_total):
            raise ValueError(
                "pilot_normal_matrix must have the estimated P_total square shape."
            )
        diagnostics = symmetric_rank_diagnostics(matrix, rtol=1.0e-10)
        pilot_rank = diagnostics.rank
        pilot_condition = diagnostics.condition_number
        pilot_eigenvalues = diagnostics.eigenvalues.copy()
        condition_status = (
            "pilot_full_rank" if diagnostics.rank == p_total else "pilot_rank_deficient"
        )

    within_cap = bool(selected_tile >= 1 and operator_peak <= cap)
    python_within_cap = bool(python_peak <= cap)
    if not within_cap:
        verdict = "operator_memory_cap_exceeded"
    elif condition_status == "pilot_rank_deficient":
        verdict = "pilot_rank_deficient_more_probes_will_not_fix_design"
    elif python_within_cap:
        verdict = "python_development_and_operator_plan_fit_cap"
    else:
        verdict = "operator_plan_fits_cap_python_development_backend_does_not"

    tiles = (
        0 if selected_tile == 0 else (gram_probes + selected_tile - 1) // selected_tile
    )
    buffer_bytes = {
        "normal_matrix": scalar * p_total * p_total,
        "reference_gram_and_same_person": scalar * 2 * p_genetic * p_genetic,
        "operator_action_tile": action_tile,
        "operator_source_tile": source_tile,
        "operator_annotation_target_tile": annotation_target_tile,
        "same_person_persistent": scalar
        * (n_reference * p_genetic + p_genetic * p_genetic),
        "current_python_features": python_features,
        "current_python_exact_kernels": python_exact_kernels,
        "current_python_exact_raw_targets": python_exact_raw_targets,
        "current_python_hutch_feature_action": python_hutch_feature_action,
        "full_gram_probe_panel": full_gram_probe_panel,
        "full_variant_probe_panel": full_variant_probe_panel,
    }
    storage_bytes = {
        "grouped_reference_contribution_bytes": reference_group_contribution,
        "grouped_reference_total_bytes": grouped_reference_total,
        "snp_reference_contribution_bytes": reference_snp_contribution,
        "snp_reference_total_bytes": snp_reference_total,
        "grouped_trait_contribution_bytes": trait_group_contribution,
        "grouped_trait_total_bytes": grouped_trait_total,
        "snp_trait_contribution_bytes": trait_snp_contribution,
        "snp_trait_total_bytes": snp_trait_total,
    }
    operation_counts = {
        "probe_tiles": tiles,
        "source_genotype_products": q_count * tiles,
        "annotation_target_genotype_products": k_count * q_count * tiles,
        "grouped_numerator_source_products": q_count * p_genetic * tiles,
        "same_person_genotype_products": k_count
        * ((same_probes + max(selected_tile, 1) - 1) // max(selected_tile, 1)),
        "trait_decode_passes": 1,
        "trait_rhs_columns_per_transform": q_count,
        "normal_dimension": p_total,
    }
    dominant_flops = {
        "operator_source_products": 2
        * n_reference
        * n_variants
        * q_count
        * gram_probes,
        "operator_target_products": 2
        * n_reference
        * n_variants
        * k_count
        * q_count
        * gram_probes,
        "grouped_numerator_source_products": 2
        * n_reference
        * n_variants
        * q_count
        * p_genetic
        * gram_probes,
        "grouped_numerator_bilinear_reductions": n_variants
        * p_genetic
        * p_genetic
        * gram_probes,
        "action_gram_reductions": 2 * n_reference * p_genetic * p_genetic * gram_probes,
        "trait_score_products": 2 * n_study * n_variants * q_count,
    }
    manifest = {
        "kind": "summit.context.annotation_resource_estimate",
        "schema_version": 1,
        "dtype": f"float{8 * scalar}",
        "dimensions_only": pilot_normal_matrix is None,
        "condition_status": condition_status,
        "memory_cap_bytes": cap,
        "selected_probe_tile_size": selected_tile,
        "current_backend": "python_dense_development",
        "planned_backend": "block_streamed_operator",
        "planned_operator_scope": (
            "base_gram_and_same_person_peak_with_grouped_numerator_work_counted_"
            "but_not_yet_native_implemented"
        ),
        "operator_tile_schedule": (
            "retain_Q_Mxb_sources_stream_one_source_through_K_Nxb_targets_"
            "retain_Pg_Nxb_actions"
        ),
        "more_probes_fix_structural_collinearity": False,
    }
    return AnnotationResourceEstimate(
        request=request,
        p_genetic=p_genetic,
        p_total=p_total,
        selected_probe_tile_size=selected_tile,
        buffer_bytes=buffer_bytes,
        storage_bytes=storage_bytes,
        operation_counts=operation_counts,
        dominant_flop_proxies=dominant_flops,
        operator_peak_bytes=int(operator_peak),
        current_python_peak_bytes=int(python_peak),
        within_memory_cap=within_cap,
        current_python_within_memory_cap=python_within_cap,
        condition_status=condition_status,
        pilot_rank=pilot_rank,
        pilot_condition_number=pilot_condition,
        pilot_eigenvalues=pilot_eigenvalues,
        verdict=verdict,
        manifest=manifest,
    )
