from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import math
import sys
from typing import Any

import numpy as np
import pytest

from summit.context.reference_v1 import (
    _NATIVE_RESULT_KEYS as REFERENCE_RESULT_KEYS,
)
from summit.context.trait_v1 import _NATIVE_RESULT_KEYS as TRAIT_RESULT_KEYS

from test_context_stage2_streamed_reference import (
    STAGE2_OPERATIONS,
    _assert_close,
    _executor,
    _make_case,
)
from test_context_stage3_complete_reference_native import (
    STAGE3_OPERATIONS,
    _fault_executor,
    _stage3_executor,
    _variant_probes,
)
from test_context_stage4_trait_native import (
    SCIENCE_ARRAYS,
    TRAIT_OPERATIONS,
    _trait_executor,
    _trait_inputs,
)


PHASE_ORDER = (
    "source",
    "action",
    "gram",
    "group",
    "same_person",
    "residual_moments",
    "trait",
    "finalization",
    "publication",
    "run_total",
)
CATEGORY_ORDER = ("decode", "projection", "row_scale", "packing", "reduction")
OPERATION_ORDER = STAGE2_OPERATIONS + STAGE3_OPERATIONS + TRAIT_OPERATIONS

REPORT_KEYS = {
    "schema",
    "schema_version",
    "metadata_only",
    "contextual_native_api_version",
    "contextual_backend_version",
    "contextual_backend",
    "contextual_execution_backend",
    "contextual_build_id",
    "source_tree_sha256",
    "execution_plan_sha256",
    "lifecycle",
    "execution_mode",
    "units",
    "dimensions",
    "selected_plan",
    "runtime_policy",
    "admission_bytes",
    "phase_order",
    "phases",
    "category_order",
    "categories",
    "operation_order",
    "operations",
    "accounting",
    "totals",
    "resource_capabilities",
    "invariants",
    "categories_are_nonoverlapping",
    "categories_cover_entire_run",
}
UNITS_KEYS = {"time", "memory", "work", "flop_count", "throughput", "count"}
DIMENSION_KEYS = {
    "samples",
    "variants",
    "fixed_effect_rank",
    "contexts",
    "annotations",
    "context_pairs",
    "components",
    "groups",
    "sample_probes",
    "variant_probes",
    "traits",
    "residual_components",
}
SELECTED_PLAN_KEYS = {
    "variant_block",
    "sample_probe_resident",
    "sample_probe_tile",
    "action_tile",
    "annotation_tile",
    "context_tile",
    "variant_probe_tile",
    "group_tile",
    "trait_feature_tile",
    "grouped_algorithm",
    "direct_grouped_scaling",
    "strict_disjoint_indexed_routing",
}
RUNTIME_POLICY_KEYS = {
    "decode_threads",
    "blas_threads",
    "numa_policy",
    "output_numa_node",
    "protected_dispatch_serialized",
    "deterministic_non_vendor_backend",
}
ADMISSION_KEYS = {
    "permanent",
    "source_phase",
    "action_phase",
    "group_phase",
    "same_person_phase",
    "residual_phase",
    "trait_phase",
    "compact_output",
    "integrity_reserve",
    "telemetry",
    "required_workspace",
    "workspace_cap",
}
PHASE_KEYS = {
    "calls",
    "wall_ns",
    "process_cpu_ns",
    "rss_begin_bytes",
    "rss_end_bytes",
    "rss_high_water_begin_bytes",
    "rss_high_water_end_bytes",
    "rss_high_water_delta_bytes",
    "process_cpu_available",
    "proc_status_available",
    "rusage_available",
    "proc_io_available",
    "minor_page_faults",
    "major_page_faults",
    "input_block_operations",
    "output_block_operations",
    "io_read_characters",
    "io_write_characters",
    "io_read_syscalls",
    "io_write_syscalls",
    "io_read_bytes",
    "io_write_bytes",
}
CATEGORY_KEYS = {"calls", "wall_ns", "process_cpu_ns", "work_items", "logical_bytes"}
OPERATION_KEYS = {
    "calls",
    "logical_flop_count",
    "output_elements",
    "transpose_left",
    "minimum_rows",
    "maximum_rows",
    "minimum_columns",
    "maximum_columns",
    "minimum_reduction",
    "maximum_reduction",
    "shape_histogram",
    "primary_wall_ns",
    "witness_wall_ns",
    "retry_calls",
    "fallback_calls",
    "recovery_wall_ns",
    "protected_total_wall_ns",
    "integrity_noncompute_wall_ns",
    "integrity_overhead_wall_ns",
    "primary_effective_gflops",
    "integrity_overhead_fraction",
}
SHAPE_HISTOGRAM_KEYS = {
    "transpose_left",
    "rows",
    "columns",
    "reduction",
    "left_stride",
    "right_stride",
    "output_stride",
    "calls",
}
ACCOUNTING_KEYS = {
    "expected_descriptor_passes",
    "observed_descriptor_passes",
    "expected_decoded_blocks",
    "observed_decoded_blocks",
    "expected_variant_record_visits",
    "observed_variant_record_visits",
    "unique_variant_records",
    "duplicate_decodes",
    "bed_record_bytes_per_variant",
    "expected_physical_record_bytes_visited",
    "observed_physical_record_bytes_visited",
    "expected_group_execution_variant_visits",
    "observed_group_execution_variant_visits",
    "expected_protected_calls",
    "observed_protected_calls",
    "descriptor_accounting_exact",
    "protected_call_accounting_exact",
}
TOTAL_KEYS = {
    "run_wall_ns",
    "run_process_cpu_ns",
    "active_phase_wall_ns",
    "active_phase_process_cpu_ns",
    "category_wall_ns",
    "category_process_cpu_ns",
    "protected_calls",
    "protected_logical_flop_count",
    "primary_wall_ns",
    "witness_wall_ns",
    "recovery_wall_ns",
    "protected_total_wall_ns",
    "integrity_noncompute_wall_ns",
    "integrity_overhead_wall_ns",
    "integrity_overhead_fraction",
    "process_peak_rss_bytes",
}
RESOURCE_KEYS = {
    "process_cpu_clock",
    "linux_proc_status",
    "posix_getrusage",
    "linux_proc_io",
}
INVARIANT_KEYS = {
    "phase_wall_within_run",
    "phase_process_cpu_within_run",
    "category_wall_within_active_phases",
    "category_process_cpu_within_active_phases",
    "descriptor_accounting_exact",
    "protected_calls_match_semantic_ledger",
    "scientific_state_unchanged_verified",
    "report_contains_scientific_ndarray",
    "instrumentation_changes_execution_plan",
}


def _assert_metadata_only(value: Any, path: str = "report") -> None:
    assert not isinstance(value, np.ndarray), path
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert isinstance(key, str), f"{path} key"
            _assert_metadata_only(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_metadata_only(item, f"{path}[{index}]")
    else:
        assert type(value) in {bool, int, float, str}, (path, type(value))


def _assert_closed_report(
    report: Mapping[str, Any],
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    trait_mode: bool,
) -> None:
    assert set(report) == REPORT_KEYS
    assert report["schema"] == "contextual_native_performance_report_v1"
    assert report["schema_version"] == 1
    assert report["metadata_only"] is True
    assert report["contextual_native_api_version"] == 1
    for name in (
        "contextual_backend_version",
        "contextual_backend",
        "contextual_execution_backend",
        "contextual_build_id",
        "source_tree_sha256",
        "execution_plan_sha256",
        "lifecycle",
    ):
        assert report[name] == result[name], name
    assert set(report["units"]) == UNITS_KEYS
    assert set(report["dimensions"]) == DIMENSION_KEYS
    assert set(report["selected_plan"]) == SELECTED_PLAN_KEYS
    assert set(report["runtime_policy"]) == RUNTIME_POLICY_KEYS
    assert set(report["admission_bytes"]) == ADMISSION_KEYS
    assert tuple(report["phase_order"]) == PHASE_ORDER
    assert tuple(report["category_order"]) == CATEGORY_ORDER
    assert tuple(report["operation_order"]) == OPERATION_ORDER
    assert set(report["phases"]) == set(PHASE_ORDER)
    assert set(report["categories"]) == set(CATEGORY_ORDER)
    assert set(report["operations"]) == set(OPERATION_ORDER)
    assert set(report["accounting"]) == ACCOUNTING_KEYS
    assert set(report["totals"]) == TOTAL_KEYS
    assert set(report["resource_capabilities"]) == RESOURCE_KEYS
    assert set(report["invariants"]) == INVARIANT_KEYS
    assert report["categories_are_nonoverlapping"] is True
    assert report["categories_cover_entire_run"] is False
    _assert_metadata_only(report)

    phases = {name: dict(item) for name, item in report["phases"].items()}
    expected_active = (
        {"residual_moments", "trait", "finalization", "publication", "run_total"}
        if trait_mode
        else {
            "source",
            "action",
            "gram",
            "group",
            "same_person",
            "finalization",
            "publication",
            "run_total",
        }
    )
    for name, phase in phases.items():
        assert set(phase) == PHASE_KEYS
        assert phase["calls"] == (1 if name in expected_active else 0)
        for key, value in phase.items():
            if key.endswith("_available"):
                assert type(value) is bool
            else:
                assert type(value) is int and value >= 0, (name, key, value)

    categories = {name: dict(item) for name, item in report["categories"].items()}
    for name, category in categories.items():
        assert set(category) == CATEGORY_KEYS
        assert category["calls"] > 0, name
        for key, value in category.items():
            assert type(value) is int and value >= 0, (name, key, value)

    operations = {name: dict(item) for name, item in report["operations"].items()}
    semantic_call_ledger = dict(preflight["semantic_call_ledger"])
    expected_shape_histograms = {name: Counter() for name in OPERATION_ORDER}
    for event in result["telemetry"]["events"]:
        if event["event_class"] != "protected_call":
            continue
        expected_shape_histograms[event["operation"]][
            (
                bool(event["transpose_left"]),
                int(event["rows"]),
                int(event["columns"]),
                int(event["reduction"]),
                int(event["left_stride"]),
                int(event["right_stride"]),
                int(event["output_stride"]),
            )
        ] += 1
    for name, operation in operations.items():
        assert set(operation) == OPERATION_KEYS
        assert operation["calls"] == int(semantic_call_ledger[name])
        assert type(operation["transpose_left"]) is bool
        for key, value in operation.items():
            if key in {"transpose_left", "shape_histogram"}:
                continue
            if key in {
                "logical_flop_count",
                "primary_effective_gflops",
                "integrity_overhead_fraction",
            }:
                assert type(value) is float and math.isfinite(value) and value >= 0.0
            else:
                assert type(value) is int and value >= 0
        observed_shape_histogram: Counter[
            tuple[bool, int, int, int, int, int, int]
        ] = Counter()
        observed_shape_order = []
        for shape_value in operation["shape_histogram"]:
            shape = dict(shape_value)
            assert set(shape) == SHAPE_HISTOGRAM_KEYS
            assert type(shape["transpose_left"]) is bool
            for key in SHAPE_HISTOGRAM_KEYS - {"transpose_left"}:
                assert type(shape[key]) is int and shape[key] > 0
            shape_key = (
                shape["transpose_left"],
                shape["rows"],
                shape["columns"],
                shape["reduction"],
                shape["left_stride"],
                shape["right_stride"],
                shape["output_stride"],
            )
            observed_shape_order.append(shape_key)
            observed_shape_histogram[shape_key] = shape["calls"]
        assert observed_shape_order == sorted(observed_shape_order)
        assert observed_shape_histogram == expected_shape_histograms[name]
        assert sum(observed_shape_histogram.values()) == operation["calls"]
        assert 0.0 <= operation["integrity_overhead_fraction"] <= 1.0
        if operation["calls"]:
            assert operation["logical_flop_count"] > 0.0
            assert operation["minimum_rows"] > 0
            assert operation["minimum_columns"] > 0
            assert operation["minimum_reduction"] > 0
            assert operation["maximum_rows"] >= operation["minimum_rows"]
            assert operation["maximum_columns"] >= operation["minimum_columns"]
            assert operation["maximum_reduction"] >= operation["minimum_reduction"]
            shape_keys = tuple(observed_shape_histogram)
            assert operation["minimum_rows"] == min(item[1] for item in shape_keys)
            assert operation["maximum_rows"] == max(item[1] for item in shape_keys)
            assert operation["minimum_columns"] == min(item[2] for item in shape_keys)
            assert operation["maximum_columns"] == max(item[2] for item in shape_keys)
            assert operation["minimum_reduction"] == min(item[3] for item in shape_keys)
            assert operation["maximum_reduction"] == max(item[3] for item in shape_keys)
            assert operation["primary_wall_ns"] > 0
            assert operation["witness_wall_ns"] > 0
            assert operation["protected_total_wall_ns"] >= (
                operation["primary_wall_ns"]
                + operation["witness_wall_ns"]
                + operation["recovery_wall_ns"]
            )

    accounting = dict(report["accounting"])
    assert accounting["expected_descriptor_passes"] == int(
        preflight["total_descriptor_passes"]
    )
    assert accounting["observed_descriptor_passes"] == int(
        preflight["total_descriptor_passes"]
    )
    assert accounting["expected_decoded_blocks"] == int(
        preflight["total_decoded_blocks"]
    )
    assert accounting["observed_decoded_blocks"] == int(
        preflight["total_decoded_blocks"]
    )
    assert accounting["expected_variant_record_visits"] == int(
        preflight["total_variant_record_visits"]
    )
    assert accounting["observed_variant_record_visits"] == int(
        preflight["total_variant_record_visits"]
    )
    assert accounting["expected_protected_calls"] == int(
        preflight["total_protected_calls"]
    )
    assert accounting["observed_protected_calls"] == int(
        preflight["total_protected_calls"]
    )
    assert accounting["descriptor_accounting_exact"] is True
    assert accounting["protected_call_accounting_exact"] is True
    assert categories["decode"]["calls"] == accounting["observed_decoded_blocks"]
    assert categories["decode"]["work_items"] == (
        accounting["observed_variant_record_visits"] * report["dimensions"]["samples"]
    )
    assert (
        categories["decode"]["logical_bytes"]
        == accounting["observed_physical_record_bytes_visited"]
    )
    assert categories["projection"]["calls"] == sum(
        operations[name]["calls"]
        for name in OPERATION_ORDER
        if name.endswith("projection_tn")
    )

    totals = dict(report["totals"])
    for key, value in totals.items():
        if key in {"protected_logical_flop_count", "integrity_overhead_fraction"}:
            assert type(value) is float and math.isfinite(value) and value >= 0.0
        else:
            assert type(value) is int and value >= 0
    assert 0.0 <= totals["integrity_overhead_fraction"] <= 1.0
    assert totals["run_wall_ns"] == phases["run_total"]["wall_ns"]
    assert totals["run_process_cpu_ns"] == phases["run_total"]["process_cpu_ns"]
    assert totals["active_phase_wall_ns"] == sum(
        phases[name]["wall_ns"] for name in PHASE_ORDER if name != "run_total"
    )
    assert totals["active_phase_process_cpu_ns"] == sum(
        phases[name]["process_cpu_ns"] for name in PHASE_ORDER if name != "run_total"
    )
    assert totals["category_wall_ns"] == sum(
        item["wall_ns"] for item in categories.values()
    )
    assert totals["category_process_cpu_ns"] == sum(
        item["process_cpu_ns"] for item in categories.values()
    )
    assert totals["protected_calls"] == sum(
        item["calls"] for item in operations.values()
    )
    assert totals["protected_logical_flop_count"] == pytest.approx(
        sum(item["logical_flop_count"] for item in operations.values()),
        rel=0.0,
        abs=0.0,
    )
    assert totals["primary_wall_ns"] == sum(
        item["primary_wall_ns"] for item in operations.values()
    )
    assert totals["witness_wall_ns"] == sum(
        item["witness_wall_ns"] for item in operations.values()
    )
    assert totals["recovery_wall_ns"] == sum(
        item["recovery_wall_ns"] for item in operations.values()
    )
    assert totals["protected_total_wall_ns"] == sum(
        item["protected_total_wall_ns"] for item in operations.values()
    )
    assert totals["integrity_noncompute_wall_ns"] == sum(
        item["integrity_noncompute_wall_ns"] for item in operations.values()
    )
    assert totals["integrity_overhead_wall_ns"] == sum(
        item["integrity_overhead_wall_ns"] for item in operations.values()
    )

    capabilities = dict(report["resource_capabilities"])
    assert all(type(value) is bool for value in capabilities.values())
    if sys.platform.startswith("linux"):
        assert capabilities["linux_proc_status"] is True
        assert phases["run_total"]["rss_begin_bytes"] > 0
        assert phases["run_total"]["rss_end_bytes"] > 0
        assert phases["run_total"]["rss_high_water_begin_bytes"] > 0
        assert phases["run_total"]["rss_high_water_end_bytes"] > 0

    invariants = dict(report["invariants"])
    assert all(
        invariants[name] is True
        for name in INVARIANT_KEYS
        - {
            "report_contains_scientific_ndarray",
            "instrumentation_changes_execution_plan",
        }
    )
    assert invariants["report_contains_scientific_ndarray"] is False
    assert invariants["instrumentation_changes_execution_plan"] is False


def test_native_performance_report_is_closed_exact_and_post_success_only(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=3, name="stage6-performance-reference")
    probes = _variant_probes(case)
    executor = _stage3_executor(
        case,
        probes,
        variant_block=3,
        sample_probe_resident=2,
        sample_probe_tile=1,
        action_tile=2,
        annotation_tile=1,
        context_tile=2,
        variant_probe_tile=2,
        group_tile=2,
    )
    preflight = dict(executor.preflight())
    with pytest.raises(RuntimeError, match="requires successful run"):
        executor.performance_report()
    result = dict(executor.run())
    assert set(result) == REFERENCE_RESULT_KEYS
    assert "performance_report" not in result
    report = dict(executor.performance_report())
    _assert_closed_report(report, result, preflight, trait_mode=False)
    assert report == dict(executor.performance_report())
    with pytest.raises(RuntimeError, match="one-shot"):
        executor.run()


def test_native_performance_group_execution_visits_track_restricted_path_only(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=2,
        name="stage6-performance-group-accounting",
    )
    probes = _variant_probes(case)

    stage2_executor = _executor(case, variant_block=3)
    stage2_executor.run()
    stage2_report = dict(stage2_executor.performance_report())

    direct_executor = _stage3_executor(
        case,
        probes,
        variant_block=3,
        grouped_algorithm="direct_grouped_tn_v1",
        direct_grouped_scaling="action_scaled_v1",
        enable_grouped_differential=False,
    )
    direct_preflight = dict(direct_executor.preflight())
    direct_result = dict(direct_executor.run())
    direct_report = dict(direct_executor.performance_report())
    _assert_closed_report(
        direct_report, direct_result, direct_preflight, trait_mode=False
    )
    assert int(direct_preflight["group_descriptor_passes"]) == 1
    assert direct_report["accounting"]["expected_group_execution_variant_visits"] == 0
    assert direct_report["accounting"]["observed_group_execution_variant_visits"] == 0

    differential_executor = _stage3_executor(
        case,
        probes,
        variant_block=3,
        grouped_algorithm="direct_grouped_tn_v1",
        direct_grouped_scaling="action_scaled_v1",
        enable_grouped_differential=True,
    )
    differential_preflight = dict(differential_executor.preflight())
    differential_result = dict(differential_executor.run())
    differential_report = dict(differential_executor.performance_report())
    _assert_closed_report(
        differential_report,
        differential_result,
        differential_preflight,
        trait_mode=False,
    )
    expected_restricted_visits = case.retained_variant_rows.size
    assert int(differential_preflight["group_descriptor_passes"]) == 3
    assert (
        differential_report["accounting"]["expected_group_execution_variant_visits"]
        == expected_restricted_visits
    )
    assert (
        differential_report["accounting"]["observed_group_execution_variant_visits"]
        == expected_restricted_visits
    )

    stage2_categories = stage2_report["categories"]
    direct_categories = direct_report["categories"]
    differential_categories = differential_report["categories"]
    for category in ("packing", "row_scale", "reduction"):
        assert (
            direct_categories[category]["calls"] > stage2_categories[category]["calls"]
        )
        assert (
            differential_categories[category]["calls"]
            > direct_categories[category]["calls"]
        )
        assert direct_categories[category]["wall_ns"] > 0
        assert differential_categories[category]["wall_ns"] > 0
    assert direct_report["invariants"]["category_wall_within_active_phases"] is True
    assert (
        differential_report["invariants"]["category_wall_within_active_phases"] is True
    )


def test_native_performance_report_trait_tiling_preserves_science(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=4, name="stage6-performance-trait")
    phenotypes, residual_basis = _trait_inputs(case)
    baseline_executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=case.retained_variant_rows.size,
        trait_feature_tile=case.retained_variant_rows.size,
    )
    baseline_preflight = dict(baseline_executor.preflight())
    baseline = dict(baseline_executor.run())
    baseline_report = dict(baseline_executor.performance_report())
    tiled_executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=3,
        trait_feature_tile=2,
    )
    tiled_preflight = dict(tiled_executor.preflight())
    tiled = dict(tiled_executor.run())
    tiled_report = dict(tiled_executor.performance_report())
    assert set(baseline) == TRAIT_RESULT_KEYS
    assert set(tiled) == TRAIT_RESULT_KEYS
    for name in SCIENCE_ARRAYS:
        _assert_close(tiled[name], baseline[name], f"instrumented tiled {name}")
    _assert_closed_report(
        baseline_report, baseline, baseline_preflight, trait_mode=True
    )
    _assert_closed_report(tiled_report, tiled, tiled_preflight, trait_mode=True)
    assert (
        baseline_report["execution_plan_sha256"]
        != tiled_report["execution_plan_sha256"]
    )
    assert baseline_report["accounting"]["duplicate_decodes"] == 0
    assert tiled_report["accounting"]["duplicate_decodes"] == 0
    assert baseline_report["accounting"]["observed_variant_record_visits"] == (
        tiled_report["accounting"]["observed_variant_record_visits"]
    )


def test_native_performance_report_rejects_failed_lifecycle(tmp_path: Path) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage6-performance-failed")
    probes = _variant_probes(case)
    executor = _fault_executor(case, probes, "source_tn", "fallback_failure")
    with pytest.raises(RuntimeError, match="trusted fallback failure"):
        executor.run()
    with pytest.raises(RuntimeError, match="requires successful run"):
        executor.performance_report()
