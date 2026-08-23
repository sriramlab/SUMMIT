from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from summit.context.performance_v1 import (
    BACKEND_IDENTITY_KEYS_V1,
    BUILD_IDENTITY_KEYS_V1,
    CONTEXTUAL_TABLA_BENCHMARK_RECORD_V1_SCHEMA,
    CONTEXTUAL_TABLA_DIMENSIONS_V1_SCHEMA,
    CONTEXTUAL_TABLA_PLAN_V1_SCHEMA,
    DESCRIPTOR_PHASE_KEYS_V1,
    HOST_IDENTITY_KEYS_V1,
    MAX_U128,
    MEMORY_CATEGORY_KEYS_V1,
    MMAP_PHYSICAL_READ_BYTES_UNAVAILABLE_SOURCE_V1,
    PERFORMANCE_CATEGORY_KEYS_V1,
    PERFORMANCE_METRIC_KEYS_V1,
    PERFORMANCE_METRIC_UNITS_V1,
    PERFORMANCE_PHASE_KEYS_V1,
    PROTECTED_OPERATION_KEYS_V1,
    PROTECTED_SHAPE_HISTOGRAM_ENTRY_KEYS_V1,
    PROTECTED_SHAPE_KEYS_V1,
    TABLA_TARGET_B128_FIRST_ORDER_ESTIMATE_V1,
    TABLA_TARGET_B1024_FIRST_ORDER_ESTIMATE_V1,
    ContextualPerformanceEvidenceV1,
    ContextualTablaAcceptanceV1,
    ContextualTablaBenchmarkRecordV1,
    ContextualTablaDimensionsV1,
    ContextualTablaPlanV1,
    checked_u128_product_v1,
    checked_u128_sum_v1,
    contextual_tabla_cache_key_v1,
    estimate_contextual_tabla_plan_v1,
    select_contextual_tabla_plan_v1,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _dimensions(**changes: object) -> ContextualTablaDimensionsV1:
    values = {
        "n_samples": 512,
        "n_variants": 2_048,
        "context_count": 4,
        "annotation_count": 2,
        "pair_count": 10,
        "component_count": 20,
        "group_count": 5,
        "sample_probe_count": 128,
        "variant_probe_count": 128,
        "trait_count": 3,
        "residual_basis_count": 2,
        "fixed_rank": 4,
    }
    values.update(changes)
    return ContextualTablaDimensionsV1(**values)


def _host() -> dict[str, object]:
    return {
        "host_name": "tabla",
        "machine": "x86_64",
        "kernel_release": "4.19.0-21-amd64",
        "cpu_model": "AMD EPYC 7501",
        "physical_core_count": 64,
        "logical_cpu_count": 128,
        "socket_count": 2,
        "numa_node_count": 8,
        "memory_bytes": 1_099_511_627_776,
    }


def _backend(plan: ContextualTablaPlanV1) -> dict[str, str]:
    return {
        "native_backend": "gwldcore.contextual_streamed_reference_v1",
        "native_execution_backend": plan.integrity_backend,
        "blas_vendor": "OpenBLAS",
        "blas_version": "0.3.34",
        "openmp_runtime": "libgomp.so.1",
        "numeric_policy": plan.numeric_policy,
    }


def _build() -> dict[str, str]:
    return {
        "source_commit": "37317a85d5cbada9b06921c39642e164e24613d9",
        "source_tree_sha256": SHA_A,
        "native_build_provenance_sha256": SHA_B,
        "python_runtime_sha256": SHA_C,
        "dependency_runtime_sha256": SHA_D,
        "compiler_id": "GNU",
        "compiler_version": "12.2.0",
        "build_type": "Release",
        "sanitizer_mode": "none",
        "effective_optimization": "-O3",
        "architecture_tuning": "-march=native",
    }


def _evidence(value: object, unit: str = "count") -> ContextualPerformanceEvidenceV1:
    return ContextualPerformanceEvidenceV1(
        evidence_kind="measured", value=value, unit=unit, source="native_report_v1"
    )


def _metrics(wall_seconds: float) -> dict[str, ContextualPerformanceEvidenceV1]:
    result = {
        key: ContextualPerformanceEvidenceV1(
            evidence_kind="unavailable",
            value=None,
            unit=PERFORMANCE_METRIC_UNITS_V1[key],
            source=(
                MMAP_PHYSICAL_READ_BYTES_UNAVAILABLE_SOURCE_V1
                if key == "physical_read_bytes"
                else "tool_unavailable"
            ),
        )
        for key in PERFORMANCE_METRIC_KEYS_V1
    }
    phase_wall = {key: 0.0 for key in PERFORMANCE_PHASE_KEYS_V1}
    phase_wall.update(
        {
            "source": wall_seconds / 2,
            "trait": wall_seconds / 2,
            "run_total": wall_seconds,
        }
    )
    phase_cpu = {key: 0.0 for key in PERFORMANCE_PHASE_KEYS_V1}
    phase_cpu.update({"source": wall_seconds, "trait": wall_seconds})
    category_wall = {
        key: float(index + 1) for index, key in enumerate(PERFORMANCE_CATEGORY_KEYS_V1)
    }
    category_cpu = {
        key: float(2 * (index + 1))
        for index, key in enumerate(PERFORMANCE_CATEGORY_KEYS_V1)
    }
    descriptor_passes = {key: 1 for key in DESCRIPTOR_PHASE_KEYS_V1}
    descriptor_passes["same_person"] = 4
    decoded_blocks = {key: 8 for key in DESCRIPTOR_PHASE_KEYS_V1}
    decoded_blocks["same_person"] = 32
    protected_counts = {key: 1 for key in PROTECTED_OPERATION_KEYS_V1}
    protected_shapes = {
        operation: {
            "minimum_rows": 32,
            "maximum_rows": 32,
            "minimum_columns": 128,
            "maximum_columns": 128,
            "minimum_reduction": 512,
            "maximum_reduction": 512,
            "shape_histogram": [
                {
                    "transpose_left": operation.endswith("_tn"),
                    "rows": 32,
                    "columns": 128,
                    "reduction": 512,
                    "left_stride": 512 if operation.endswith("_tn") else 32,
                    "right_stride": 512,
                    "output_stride": 32,
                    "calls": 1,
                }
            ],
        }
        for operation in PROTECTED_OPERATION_KEYS_V1
    }
    protected_throughput = {key: 1.0 for key in PROTECTED_OPERATION_KEYS_V1}
    result.update(
        {
            "end_to_end_wall_seconds": _evidence(wall_seconds, "seconds"),
            "phase_wall_seconds": _evidence(phase_wall, "seconds"),
            "phase_cpu_seconds": _evidence(phase_cpu, "seconds"),
            "category_wall_seconds": _evidence(category_wall, "seconds"),
            "category_cpu_seconds": _evidence(category_cpu, "seconds"),
            "logical_descriptor_passes": _evidence(descriptor_passes),
            "decoded_blocks": _evidence(decoded_blocks),
            "physical_record_visits": _evidence(10_240),
            "duplicate_decodes": _evidence(8_192),
            "protected_call_shapes": _evidence(protected_shapes, "elements"),
            "protected_call_count": _evidence(protected_counts),
            "protected_effective_gflops": _evidence(
                protected_throughput, "GFLOP/second"
            ),
            "peak_rss_bytes": _evidence(1_000_000, "bytes"),
            "admitted_peak_bytes": _evidence(1_100_000, "bytes"),
            "integrity_overhead_seconds": _evidence(5.0, "seconds"),
            "integrity_overhead_fraction": _evidence(0.05, "fraction"),
            "artifact_write_load_fit_seconds": _evidence(6.0, "seconds"),
        }
    )
    return result


def _acceptance(
    *, discrepancy: float = 0.0, all_passed: bool = True
) -> ContextualTablaAcceptanceV1:
    return ContextualTablaAcceptanceV1(
        fixed_probe_evidence_kind="measured",
        fixed_probe_discrepancies={
            "reference_gram_max_abs": discrepancy,
            "trait_score_max_abs": 0.0,
            "fit_beta_max_abs": 0.0,
        },
        every_deletion_science_unchanged=all_passed,
        rank_estimability_identical=all_passed,
        integrity_fault_coverage_identical=all_passed,
        descriptor_passes_explained=all_passed,
        admitted_measured_memory_agrees=all_passed,
        reference_trait_end_to_end_complete=all_passed,
        artifact_round_trip_passed=all_passed,
    )


def _record(
    run_id: str,
    wall_seconds: float,
    *,
    dimensions: ContextualTablaDimensionsV1 | None = None,
    plan: ContextualTablaPlanV1 | None = None,
    host: dict[str, object] | None = None,
    build: dict[str, str] | None = None,
    science_identity: str = SHA_A,
    integrity_identity: str = SHA_B,
    rank_identity: str = SHA_C,
    acceptance: ContextualTablaAcceptanceV1 | None = None,
    terminal_status: str = "accepted",
    production_qualified: bool = True,
    rejection_reasons: tuple[str, ...] = (),
    threshold: float = 1.05,
    metrics: dict[str, ContextualPerformanceEvidenceV1] | None = None,
    comparison_baseline: ContextualTablaBenchmarkRecordV1 | None = None,
) -> ContextualTablaBenchmarkRecordV1:
    dimensions = _dimensions() if dimensions is None else dimensions
    plan = ContextualTablaPlanV1() if plan is None else plan
    host = _host() if host is None else host
    backend = _backend(plan)
    build = _build() if build is None else build
    cache_key = contextual_tabla_cache_key_v1(
        host_identity=host,
        backend_identity=backend,
        build_identity=build,
        dimensions=dimensions,
    )
    return ContextualTablaBenchmarkRecordV1(
        run_id=run_id,
        host_identity=host,
        backend_identity=backend,
        build_identity=build,
        cache_key=cache_key,
        dimensions=dimensions,
        plan=plan,
        science_identity_sha256=science_identity,
        integrity_identity_sha256=integrity_identity,
        rank_identity_sha256=rank_identity,
        metrics=_metrics(wall_seconds) if metrics is None else metrics,
        acceptance=_acceptance() if acceptance is None else acceptance,
        terminal_status=terminal_status,
        production_qualified=production_qualified,
        rejection_reasons=rejection_reasons,
        benchmark_speedup_threshold=threshold,
        comparison_baseline_run_id=(
            None if comparison_baseline is None else comparison_baseline.run_id
        ),
        comparison_baseline_record_sha256=(
            None if comparison_baseline is None else comparison_baseline.record_sha256
        ),
        declared_parent_source_commit=(
            None
            if comparison_baseline is None
            else comparison_baseline.build_identity["source_commit"]
        ),
    )


def test_dimension_and_plan_schema_are_exact_and_immutable() -> None:
    dimensions = _dimensions()
    plan = ContextualTablaPlanV1()
    assert dimensions.to_dict()["schema"] == CONTEXTUAL_TABLA_DIMENSIONS_V1_SCHEMA
    assert plan.to_dict()["schema"] == CONTEXTUAL_TABLA_PLAN_V1_SCHEMA
    assert dimensions.pair_count == dimensions.context_count * 5 // 2
    assert plan.production_nonqualification_reasons == ()
    with pytest.raises(FrozenInstanceError):
        plan.action_tile = 10  # type: ignore[misc]
    with pytest.raises(ValueError, match="pair_count"):
        _dimensions(pair_count=9)
    with pytest.raises(ValueError, match="component_count"):
        _dimensions(component_count=19)
    with pytest.raises(ValueError, match="native range"):
        _dimensions(fixed_rank=0)
    with pytest.raises(ValueError, match="native range"):
        _dimensions(fixed_rank=512)
    with pytest.raises(ValueError, match="disagree"):
        ContextualTablaPlanV1(integrity_policy="algebraic_checksum_v1")
    with pytest.raises(ValueError, match="decode_threads"):
        ContextualTablaPlanV1(decode_threads=65)
    with pytest.raises(ValueError, match="blas_threads"):
        ContextualTablaPlanV1(blas_threads=65)
    with pytest.raises(ValueError, match="output_numa_node=-1"):
        ContextualTablaPlanV1(output_numa_node=0)
    with pytest.raises(ValueError, match="explicit output node"):
        ContextualTablaPlanV1(numa_policy="experimental_single_socket_bound_v1")


def test_evidence_is_finite_owned_and_kind_explicit() -> None:
    source = {"source": [1.0, 2.0]}
    evidence = _evidence(source)
    source["source"][0] = 99.0
    assert evidence.to_dict()["value"] == {"source": [1.0, 2.0]}
    with pytest.raises(TypeError):
        evidence.value["source"] = (3.0,)  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        _evidence(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _evidence({"phase": float("inf")})
    with pytest.raises(ValueError, match="value=None"):
        ContextualPerformanceEvidenceV1("unavailable", 0, "count", "none")


def test_checked_unsigned_128_bit_arithmetic_rejects_overflow() -> None:
    assert checked_u128_sum_v1(MAX_U128, 0) == MAX_U128
    assert checked_u128_product_v1(MAX_U128, 1) == MAX_U128
    with pytest.raises(OverflowError, match="addition"):
        checked_u128_sum_v1(MAX_U128, 1)
    with pytest.raises(OverflowError, match="multiplication"):
        checked_u128_product_v1(MAX_U128, 2)
    with pytest.raises(ValueError):
        checked_u128_product_v1(True)


def test_target_b128_and_b1024_first_order_estimates_match_templates() -> None:
    b128 = TABLA_TARGET_B128_FIRST_ORDER_ESTIMATE_V1
    assert b128["resident_objects_bytes"]["source_scores"] == 4_096_000_000
    assert b128["resident_objects_bytes"]["full_raw_actions"] == 24_576_000_000
    assert b128["core_phase_lower_bound_bytes"] == {
        "source_core": 5_324_800_000,
        "action_core": 38_502_400_000,
        "group_restricted_core": 40_970_240_000,
        "same_person_core": 34_598_400_000,
    }
    b1024 = TABLA_TARGET_B1024_FIRST_ORDER_ESTIMATE_V1
    assert b1024["resident_objects_bytes"]["source_scores"] == 32_768_000_000
    assert b1024["resident_objects_bytes"]["full_raw_actions"] == 196_608_000_000
    assert b1024["core_phase_lower_bound_bytes"] == {
        "source_core": 42_598_400_000,
        "action_core": 308_019_200_000,
        "group_restricted_core": 327_690_240_000,
        "same_person_core": 275_443_200_000,
    }


def test_cache_key_covers_every_host_backend_build_and_dimension_axis() -> None:
    dimensions = _dimensions()
    plan = ContextualTablaPlanV1()
    host = _host()
    backend = _backend(plan)
    build = _build()

    def key(
        host_value: dict[str, object] = host,
        backend_value: dict[str, str] = backend,
        build_value: dict[str, str] = build,
        dimension_value: ContextualTablaDimensionsV1 = dimensions,
    ) -> str:
        return contextual_tabla_cache_key_v1(
            host_identity=host_value,
            backend_identity=backend_value,
            build_identity=build_value,
            dimensions=dimension_value,
        )

    baseline = key()
    for name in HOST_IDENTITY_KEYS_V1:
        changed = dict(host)
        changed[name] = (
            changed[name] + 1
            if isinstance(changed[name], int)
            else f"{changed[name]}-x"
        )
        assert key(host_value=changed) != baseline
    for name in BACKEND_IDENTITY_KEYS_V1:
        changed = dict(backend)
        changed[name] += "-x"
        assert key(backend_value=changed) != baseline
    for name in BUILD_IDENTITY_KEYS_V1:
        changed = dict(build)
        if name.endswith("sha256"):
            changed[name] = SHA_D if changed[name] != SHA_D else SHA_A
        elif name == "source_commit":
            changed[name] = "4" * 40
        else:
            changed[name] += "-x"
        assert key(build_value=changed) != baseline
    assert key(dimension_value=replace(dimensions, n_samples=513)) != baseline
    missing = dict(host)
    missing.pop("machine")
    with pytest.raises(ValueError, match="missing keys"):
        key(host_value=missing)


@pytest.mark.parametrize(
    "field", ("python_runtime_sha256", "dependency_runtime_sha256")
)
def test_cache_identity_requires_canonical_runtime_environment_digests(
    field: str,
) -> None:
    dimensions = _dimensions()
    plan = ContextualTablaPlanV1()
    build = _build()
    build.pop(field)
    with pytest.raises(ValueError, match="missing keys"):
        contextual_tabla_cache_key_v1(
            host_identity=_host(),
            backend_identity=_backend(plan),
            build_identity=build,
            dimensions=dimensions,
        )
    build = _build()
    build[field] = "/path-only-is-not-a-runtime-identity"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        contextual_tabla_cache_key_v1(
            host_identity=_host(),
            backend_identity=_backend(plan),
            build_identity=build,
            dimensions=dimensions,
        )


def test_checked_plan_estimate_reports_every_category_and_admission() -> None:
    dimensions = _dimensions()
    plan = ContextualTablaPlanV1(
        integrity_reserve_bytes=4096,
        telemetry_capacity_bytes=2048,
        fixed_headroom_bytes=1024,
    )
    estimate = estimate_contextual_tabla_plan_v1(
        dimensions, plan, workspace_cap_bytes=MAX_U128
    )
    assert tuple(estimate.memory_categories) == MEMORY_CATEGORY_KEYS_V1
    assert estimate.evidence_kind == "modeled"
    assert estimate.admitted
    assert estimate.memory_categories["bytes_integrity"] > 4096
    assert estimate.memory_categories["bytes_telemetry"] > 2048
    assert estimate.required_peak_bytes > max(estimate.phase_lower_bound_bytes.values())
    rejected = estimate_contextual_tabla_plan_v1(
        dimensions, plan, workspace_cap_bytes=estimate.required_peak_bytes - 1
    )
    assert not rejected.admitted
    with pytest.raises(ValueError, match="sample_probe_resident_count"):
        estimate_contextual_tabla_plan_v1(
            dimensions,
            replace(plan, sample_probe_resident_count=256),
            workspace_cap_bytes=MAX_U128,
        )
    with pytest.raises(ValueError, match="source_variant_block"):
        estimate_contextual_tabla_plan_v1(
            _dimensions(n_variants=128),
            plan,
            workspace_cap_bytes=MAX_U128,
        )


def test_b1024_resident_memory_pass_call_and_group_cost_formulas() -> None:
    dimensions = ContextualTablaDimensionsV1(
        n_samples=300_000,
        n_variants=1_000_000,
        context_count=4,
        annotation_count=8,
        pair_count=10,
        component_count=80,
        group_count=200,
        sample_probe_count=1_024,
        variant_probe_count=1_024,
        fixed_rank=4,
    )
    plan = ContextualTablaPlanV1(sample_probe_resident_count=128)
    estimate = estimate_contextual_tabla_plan_v1(
        dimensions, plan, workspace_cap_bytes=MAX_U128
    )
    assert estimate.memory_categories["bytes_source_scores"] == 32_768_000_000
    assert estimate.memory_categories["bytes_actions"] == 196_608_000_000
    assert estimate.memory_categories["bytes_targets"] == 9_830_400_000
    assert estimate.memory_categories["bytes_group_targets"] == 78_643_200_000
    assert estimate.memory_categories["bytes_group_action_tile"] == 20_889_600_000
    assert estimate.memory_categories["bytes_same_sketch"] == 6_400_524_288
    assert estimate.memory_categories["bytes_group_cross"] == 8 * 1 * 8 * 80
    assert estimate.phase_lower_bound_bytes["source"] == 402_413_204_000
    assert estimate.required_peak_bytes == 482_895_844_800
    assert estimate.logical_descriptor_passes["action"] == 8
    assert estimate.protected_call_lower_bounds == {
        "source": 125_088,
        "action": 125_672,
        "group": 127_024,
        "same_person": 125_121,
        "trait": 0,
    }
    assert estimate.flop_lower_bounds["group"] == 2 * 300_000 * 1_000_000 * 4 * 1_024


def test_b128_current_executor_resident_arenas_and_derived_reserves() -> None:
    dimensions = ContextualTablaDimensionsV1(
        n_samples=300_000,
        n_variants=1_000_000,
        context_count=4,
        annotation_count=8,
        pair_count=10,
        component_count=80,
        group_count=200,
        sample_probe_count=128,
        variant_probe_count=128,
        fixed_rank=4,
    )
    estimate = estimate_contextual_tabla_plan_v1(
        dimensions, ContextualTablaPlanV1(), workspace_cap_bytes=MAX_U128
    )
    categories = estimate.memory_categories
    assert categories["bytes_source_scores"] == 4_096_000_000
    assert categories["bytes_actions"] == 24_576_000_000
    assert categories["bytes_targets"] == 9_830_400_000
    assert categories["bytes_group_targets"] == 9_830_400_000
    assert categories["bytes_group_action_tile"] == 3_686_400_000
    assert categories["bytes_same_sketch"] == 6_400_524_288
    assert categories["bytes_integrity"] == 9_830_400_512
    assert categories["bytes_telemetry"] == 1_187_061_760
    assert estimate.phase_lower_bound_bytes["source"] == 107_441_230_368
    assert estimate.required_peak_bytes == 128_929_476_442


def test_modeled_peak_covers_native_tracked_admission_for_matching_plan(
    tmp_path: Path,
) -> None:
    try:
        __import__("summit.gxeldcore")
    except ImportError:
        pytest.skip("native extension is not available in this source-only run")
    from test_context_stage2_streamed_reference import _make_case, _philox_probes
    from test_context_stage3_complete_reference_native import _stage3_executor
    from test_context_stage4_trait_native import _trait_executor, _trait_inputs

    def native_phase_calls(preflight: object) -> dict[str, int]:
        ledger = {
            str(key): int(value)
            for key, value in dict(preflight["semantic_call_ledger"]).items()
        }

        def total(*operations: str) -> int:
            return sum(ledger[operation] for operation in operations)

        return {
            "source": total(
                "sample_probe_projection_tn",
                "sample_probe_projection_nn",
                "source_tn",
            ),
            "action": total(
                "full_target_nn",
                "action_projection_tn",
                "action_projection_nn",
                "action_gram_tn",
            ),
            "group": total(
                "group_target_nn", "group_cross_gram_tn", "direct_grouped_tn"
            ),
            "same_person": total(
                "same_person_target_nn",
                "same_person_projection_tn",
                "same_person_projection_nn",
                "same_person_gram_tn",
            ),
            "trait": total(
                "trait_score_tn",
                "trait_feature_projection_tn",
                "trait_feature_projection_nn",
            ),
        }

    case = _make_case(tmp_path, q_count=3, name="stage6-planner-admission")
    sample_keys, _ = _philox_probes(case.fixed_basis.shape[0], 5)
    variant_keys, variant_probes = _philox_probes(
        case.retained_variant_rows.size, 3, root_seed=741_921
    )
    executor_options = {
        "sample_probes": None,
        "philox_keys": sample_keys,
        "sample_probe_count": 5,
        "variant_probes": None,
        "variant_philox_keys": variant_keys,
        "variant_probe_count": 3,
        "variant_block": 3,
        "sample_probe_resident": 2,
        "sample_probe_tile": 1,
        "action_tile": 2,
        "annotation_tile": 1,
        "context_tile": 2,
        "variant_probe_tile": 2,
        "group_tile": 2,
        "enable_differential_snapshot": False,
        "enable_grouped_differential": False,
    }
    executor = _stage3_executor(case, variant_probes, **executor_options)
    preflight = dict(executor.preflight())
    native_required = int(preflight["required_workspace_bytes"])
    q = case.phi.shape[1]
    annotation_count = case.annotations.shape[1]
    pair_count = q * (q + 1) // 2
    dimensions = ContextualTablaDimensionsV1(
        n_samples=case.fixed_basis.shape[0],
        n_variants=case.retained_variant_rows.size,
        context_count=q,
        annotation_count=annotation_count,
        pair_count=pair_count,
        component_count=annotation_count * pair_count,
        group_count=len(case.group_names),
        sample_probe_count=5,
        variant_probe_count=3,
        fixed_rank=case.fixed_basis.shape[1],
    )
    plan = ContextualTablaPlanV1(
        source_variant_block=3,
        target_variant_block=3,
        grouped_variant_block=3,
        same_person_variant_block=3,
        trait_variant_block=3,
        sample_probe_resident_count=2,
        sample_probe_tile=1,
        variant_probe_tile=2,
        action_tile=2,
        annotation_tile=1,
        context_tile=2,
        group_tile=2,
        annotation_mode=case.annotation_mode,
    )
    estimate = estimate_contextual_tabla_plan_v1(
        dimensions, plan, workspace_cap_bytes=MAX_U128
    )
    assert estimate.phase_lower_bound_bytes["source"] >= native_required
    assert estimate.memory_categories["bytes_integrity"] > 0
    assert estimate.memory_categories["bytes_telemetry"] > 0
    actual_calls = native_phase_calls(preflight)
    for phase, lower_bound in estimate.protected_call_lower_bounds.items():
        assert lower_bound <= actual_calls[phase]
    for phase in ("source", "action", "same_person", "trait"):
        assert estimate.protected_call_lower_bounds[phase] == actual_calls[phase]

    direct_preflight = dict(
        _stage3_executor(
            case,
            variant_probes,
            **{
                **executor_options,
                "grouped_algorithm": "direct_grouped_tn_v1",
                "direct_grouped_scaling": "action_scaled_v1",
            },
        ).preflight()
    )
    direct_plan = replace(
        plan,
        grouped_attribution_algorithm="direct_grouped_tn_v1",
        direct_grouped_scaling_policy="action_scaled_v1",
    )
    direct_estimate = estimate_contextual_tabla_plan_v1(
        dimensions, direct_plan, workspace_cap_bytes=MAX_U128
    )
    direct_actual_calls = native_phase_calls(direct_preflight)
    for phase, lower_bound in direct_estimate.protected_call_lower_bounds.items():
        assert lower_bound == direct_actual_calls[phase]

    phenotypes, residual_basis = _trait_inputs(case)
    trait_preflight = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=3,
            trait_feature_tile=3,
        ).preflight()
    )
    trait_dimensions = replace(
        dimensions,
        trait_count=phenotypes.shape[1],
        residual_basis_count=residual_basis.shape[1],
    )
    trait_estimate = estimate_contextual_tabla_plan_v1(
        trait_dimensions, plan, workspace_cap_bytes=MAX_U128
    )
    trait_actual_calls = native_phase_calls(trait_preflight)
    assert trait_estimate.protected_call_lower_bounds["trait"] == (
        trait_actual_calls["trait"]
    )


def test_two_process_estimate_conservatively_scales_memory_calls_and_cost() -> None:
    dimensions = _dimensions()
    single = ContextualTablaPlanV1()
    two_process = replace(
        single,
        process_count=2,
        decode_threads=32,
        blas_threads=32,
        affinity_cpu_count=32,
        numa_policy="experimental_two_socket_split_v1",
        output_numa_node=0,
    )
    single_estimate = estimate_contextual_tabla_plan_v1(
        dimensions, single, workspace_cap_bytes=MAX_U128
    )
    two_estimate = estimate_contextual_tabla_plan_v1(
        dimensions, two_process, workspace_cap_bytes=MAX_U128
    )
    for category in MEMORY_CATEGORY_KEYS_V1:
        assert (
            two_estimate.memory_categories[category]
            == 2 * single_estimate.memory_categories[category]
        )
    assert two_estimate.required_peak_bytes == 2 * single_estimate.required_peak_bytes
    for phase in single_estimate.logical_descriptor_passes:
        assert (
            two_estimate.logical_descriptor_passes[phase]
            == 2 * single_estimate.logical_descriptor_passes[phase]
        )
        assert (
            two_estimate.decoded_blocks[phase]
            == 2 * single_estimate.decoded_blocks[phase]
        )
        assert (
            two_estimate.protected_call_lower_bounds[phase]
            == 2 * single_estimate.protected_call_lower_bounds[phase]
        )
        assert (
            two_estimate.flop_lower_bounds[phase]
            == 2 * single_estimate.flop_lower_bounds[phase]
        )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"multiplication_backend": "legacy_mailman_v1"}, "legacy_mailman"),
        (
            {"process_count": 2, "decode_threads": 32, "blas_threads": 32},
            "multi_process",
        ),
        (
            {
                "numa_policy": "experimental_single_socket_bound_v1",
                "output_numa_node": 0,
            },
            "numa_policy",
        ),
        (
            {
                "numa_policy": "experimental_single_socket_bound_v1",
                "output_numa_node": 0,
            },
            "output_placement",
        ),
        ({"numeric_policy": "fp32_storage_fp64_compute_v1"}, "mixed_precision"),
        ({"huge_page_policy": "transparent_host_default_v1"}, "huge_page"),
    ],
)
def test_experimental_plan_axes_cannot_be_labeled_production(
    changes: dict[str, object], reason: str
) -> None:
    plan = replace(ContextualTablaPlanV1(), **changes)
    assert any(reason in item for item in plan.production_nonqualification_reasons)
    with pytest.raises(ValueError, match="inconsistent"):
        _record("invalid-production-label", 90.0, plan=plan)


def test_benchmark_record_has_exact_schema_and_rejects_incomplete_metrics() -> None:
    record = _record("baseline", 100.0)
    payload = record.to_dict()
    assert payload["schema"] == CONTEXTUAL_TABLA_BENCHMARK_RECORD_V1_SCHEMA
    assert set(payload["metrics"]) == set(PERFORMANCE_METRIC_KEYS_V1)
    assert payload["acceptance"]["fixed_probe_discrepancy_sha256"]
    assert record.record_sha256 == record.record_sha256
    incomplete = _metrics(100.0)
    incomplete.pop("decoded_blocks")
    with pytest.raises(ValueError, match="missing keys"):
        _record("missing-metric", 100.0, metrics=incomplete)
    unavailable = _metrics(100.0)
    unavailable["phase_wall_seconds"] = ContextualPerformanceEvidenceV1(
        "unavailable", None, "seconds", "not_recorded"
    )
    with pytest.raises(ValueError, match="inconsistent"):
        _record("unmeasured-required", 100.0, metrics=unavailable)


@pytest.mark.parametrize(
    ("metric", "required_keys"),
    [
        ("phase_wall_seconds", PERFORMANCE_PHASE_KEYS_V1),
        ("category_wall_seconds", PERFORMANCE_CATEGORY_KEYS_V1),
        ("logical_descriptor_passes", DESCRIPTOR_PHASE_KEYS_V1),
        ("protected_call_count", PROTECTED_OPERATION_KEYS_V1),
    ],
)
def test_measured_mapping_metrics_require_exact_keys(
    metric: str, required_keys: tuple[str, ...]
) -> None:
    metrics = _metrics(100.0)
    value = metrics[metric].to_dict()["value"]
    value.pop(required_keys[-1])
    metrics[metric] = _evidence(value, PERFORMANCE_METRIC_UNITS_V1[metric])
    with pytest.raises(ValueError, match="missing keys"):
        _record(f"bad-{metric}", 100.0, metrics=metrics)


def test_protected_shape_histogram_is_exact_typed_and_call_bound() -> None:
    metrics = _metrics(100.0)
    shapes = metrics["protected_call_shapes"].to_dict()["value"]
    assert set(shapes[PROTECTED_OPERATION_KEYS_V1[0]]) == set(PROTECTED_SHAPE_KEYS_V1)
    assert set(shapes[PROTECTED_OPERATION_KEYS_V1[0]]["shape_histogram"][0]) == set(
        PROTECTED_SHAPE_HISTOGRAM_ENTRY_KEYS_V1
    )
    shapes["source_tn"]["shape_histogram"][0].pop("right_stride")
    metrics["protected_call_shapes"] = _evidence(shapes, "elements")
    with pytest.raises(ValueError, match="missing keys"):
        _record("missing-shape-entry-key", 100.0, metrics=metrics)

    metrics = _metrics(100.0)
    shapes = metrics["protected_call_shapes"].to_dict()["value"]
    shapes["source_tn"]["shape_histogram"][0]["calls"] = 2
    metrics["protected_call_shapes"] = _evidence(shapes, "elements")
    with pytest.raises(ValueError, match="do not match protected_call_count"):
        _record("shape-call-mismatch", 100.0, metrics=metrics)

    metrics = _metrics(100.0)
    shapes = metrics["protected_call_shapes"].to_dict()["value"]
    shapes["source_tn"]["shape_histogram"][0]["transpose_left"] = False
    metrics["protected_call_shapes"] = _evidence(shapes, "elements")
    with pytest.raises(ValueError, match="transpose_left"):
        _record("shape-transpose-mismatch", 100.0, metrics=metrics)


def test_selection_requires_measured_visit_and_duplicate_counts() -> None:
    for metric in ("physical_record_visits", "duplicate_decodes"):
        metrics = _metrics(100.0)
        metrics[metric] = ContextualPerformanceEvidenceV1(
            "unavailable",
            None,
            PERFORMANCE_METRIC_UNITS_V1[metric],
            "native_counter_missing",
        )
        with pytest.raises(ValueError, match="inconsistent"):
            _record(f"unmeasured-{metric}", 100.0, metrics=metrics)

    metrics = _metrics(100.0)
    metrics["physical_record_visits"] = _evidence(1.5)
    with pytest.raises(ValueError, match="closed interval"):
        _record("noninteger-visits", 100.0, metrics=metrics)


def test_physical_read_bytes_unavailability_requires_explicit_mmap_reason() -> None:
    assert _record("explicit-mmap-unavailable", 100.0).production_qualified
    metrics = _metrics(100.0)
    metrics["physical_read_bytes"] = ContextualPerformanceEvidenceV1(
        "unavailable", None, "bytes", "generic_tool_unavailable"
    )
    with pytest.raises(ValueError, match="explicit mmap"):
        _record("ambiguous-read-unavailable", 100.0, metrics=metrics)
    metrics = _metrics(100.0)
    metrics["physical_read_bytes"] = ContextualPerformanceEvidenceV1(
        "modeled", 1234, "bytes", "logical_visit_estimate"
    )
    with pytest.raises(ValueError, match="inconsistent"):
        _record("modeled-physical-reads", 100.0, metrics=metrics)


@pytest.mark.parametrize(
    ("identity_field", "replacement", "expected_reason"),
    [
        ("science_identity", SHA_D, "science_identity_mismatch"),
        ("integrity_identity", SHA_D, "integrity_identity_mismatch"),
        ("rank_identity", SHA_D, "rank_identity_mismatch"),
    ],
)
def test_selection_rejects_science_integrity_and_rank_identity_mismatches(
    identity_field: str, replacement: str, expected_reason: str
) -> None:
    baseline = _record("baseline", 100.0)
    candidate = _record("candidate", 50.0, **{identity_field: replacement})
    result = select_contextual_tabla_plan_v1(baseline, [candidate])
    assert result.selected_is_baseline
    assert expected_reason in result.candidate_decisions[0].rejection_reasons


def test_selection_accepts_within_contract_fixed_probe_and_requires_full_evidence() -> (
    None
):
    baseline = _record("baseline", 100.0)
    discrepancy = _record(
        "different-fixed-probe",
        50.0,
        acceptance=_acceptance(discrepancy=1e-12),
    )
    incomplete = _record(
        "incomplete-acceptance",
        40.0,
        acceptance=_acceptance(all_passed=False),
        production_qualified=False,
    )
    result = select_contextual_tabla_plan_v1(baseline, [incomplete, discrepancy])
    decisions = {item.run_id: item for item in result.candidate_decisions}
    assert decisions["different-fixed-probe"].selected
    assert decisions["different-fixed-probe"].rejection_reasons == ()
    assert (
        "full_acceptance_evidence_missing"
        in decisions["incomplete-acceptance"].rejection_reasons
    )
    assert result.selected_run_id == "different-fixed-probe"


def test_integrity_policy_backend_candidate_can_be_selected_with_full_evidence() -> (
    None
):
    baseline = _record("scalar-baseline", 100.0)
    checksum_plan = replace(
        baseline.plan,
        integrity_policy="algebraic_checksum_v1",
        integrity_backend="deterministic_tiled_fp64_with_algebraic_checksum_v1",
    )
    candidate = _record("checksum-candidate", 70.0, plan=checksum_plan)
    assert candidate.cache_key != baseline.cache_key
    result = select_contextual_tabla_plan_v1(baseline, [candidate])
    assert not result.selected_is_baseline
    assert result.selected_run_id == "checksum-candidate"
    assert result.selected_plan.integrity_policy == "algebraic_checksum_v1"


def test_controlled_cross_build_comparison_requires_baseline_relationship() -> None:
    baseline = _record("baseline-build", 100.0)
    candidate_build = dict(_build())
    candidate_build.update(
        {
            "source_commit": "4" * 40,
            "source_tree_sha256": SHA_C,
            "native_build_provenance_sha256": SHA_D,
        }
    )
    undeclared = _record("undeclared-cross-build", 60.0, build=candidate_build)
    rejected = select_contextual_tabla_plan_v1(baseline, [undeclared])
    assert rejected.selected_is_baseline
    assert set(rejected.candidate_decisions[0].rejection_reasons) >= {
        "cross_build_baseline_run_id_mismatch",
        "cross_build_baseline_record_sha256_mismatch",
        "cross_build_parent_source_commit_mismatch",
    }
    declared = _record(
        "declared-cross-build",
        60.0,
        build=candidate_build,
        comparison_baseline=baseline,
    )
    selected = select_contextual_tabla_plan_v1(baseline, [declared])
    assert selected.selected_run_id == "declared-cross-build"
    assert selected.selected_record_sha256 == declared.record_sha256
    assert declared.cache_key != baseline.cache_key


def test_selection_is_order_independent_and_records_every_rejected_alternative() -> (
    None
):
    baseline = _record("baseline", 100.0)
    first_plan = replace(baseline.plan, source_variant_block=128)
    second_plan = replace(baseline.plan, source_variant_block=512)
    first = _record("first", 70.0, plan=first_plan)
    second = _record("second", 70.0, plan=second_plan)
    selected_id = min(
        (first.plan.plan_sha256, first.run_id),
        (second.plan.plan_sha256, second.run_id),
    )[1]
    forward = select_contextual_tabla_plan_v1(baseline, [first, second])
    reverse = select_contextual_tabla_plan_v1(baseline, [second, first])
    assert forward.to_dict() == reverse.to_dict()
    assert forward.selected_run_id == selected_id
    rejected = [item for item in forward.candidate_decisions if not item.selected]
    assert len(rejected) == 1
    assert rejected[0].rejection_reasons == ("deterministic_tie_break_lost",)


def test_no_material_speedup_retains_baseline_and_threshold_has_hard_floor() -> None:
    baseline = _record("baseline", 100.0)
    too_small = _record("too-small", 96.0)
    result = select_contextual_tabla_plan_v1(baseline, [too_small])
    assert result.selected_is_baseline
    assert result.selected_speedup == 1.0
    assert result.candidate_decisions[0].rejection_reasons == (
        "material_end_to_end_speedup_not_met",
    )
    with pytest.raises(ValueError, match="benchmark_speedup_threshold"):
        _record("invalid-low-threshold", 99.0, threshold=1.049)
    explicit = _record("explicit-stricter-threshold", 90.0, threshold=1.10)
    selected = select_contextual_tabla_plan_v1(baseline, [explicit])
    assert selected.selected_run_id == "explicit-stricter-threshold"


def test_nonaccepted_records_preserve_reasons_and_are_never_selected() -> None:
    baseline = _record("baseline", 100.0)
    failed = _record(
        "failed",
        10.0,
        terminal_status="failed",
        production_qualified=False,
        rejection_reasons=("native_integrity_gate_failed",),
    )
    result = select_contextual_tabla_plan_v1(baseline, [failed])
    assert result.selected_is_baseline
    assert (
        "terminal_status_not_accepted"
        in result.candidate_decisions[0].rejection_reasons
    )
    assert (
        "record_rejection_reason:native_integrity_gate_failed"
        in result.candidate_decisions[0].rejection_reasons
    )
    assert failed.to_dict()["rejection_reasons"] == ["native_integrity_gate_failed"]
