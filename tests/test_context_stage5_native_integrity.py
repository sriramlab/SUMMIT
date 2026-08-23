from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np
import pytest

from summit.context import array_sha256
from summit.context.probe_merge_v1 import (
    adapt_native_contextual_variant_probe_partial_v1,
    contextual_variant_probe_plan_sha256_v1,
)

from test_context_stage2_streamed_reference import (
    _make_case,
    _run_checkpoint_failure,
)
from test_context_stage3_complete_reference_native import (
    _stage3_executor,
    _variant_probes,
)
from test_context_stage4_trait_native import (
    SCIENCE_ARRAYS,
    TRAIT_OPERATIONS,
    _trait_executor,
    _trait_inputs,
)
from test_context_stage5_probe_merge import _identity, _sha


SOURCE = Path(__file__).parents[1] / "src/native/contextual_streamed_reference_v1.inc"

REFERENCE_ARRAYS = {
    "gram",
    "raw_gram_numerator",
    "annotation_masses",
    "same_person",
    "group_gram_unnormalized_num",
    "group_annotation_masses",
    "group_variant_counts",
    "pair_q",
    "pair_r",
    "pair_eta",
    "component_annotation",
    "component_pair",
}
TRAIT_ARRAYS = set(SCIENCE_ARRAYS) | {
    "pair_q",
    "pair_r",
    "pair_eta",
    "component_annotation",
    "component_pair",
}
RECOVERABLE_TRAIT_FAULTS = (
    "one_shot",
    "repeated",
    "repair_corruption",
    "force_fallback",
    "nan",
    "inf",
    "canary",
)
TERMINAL_TRAIT_FAULTS = (
    "fallback_corruption",
    "fallback_failure",
    "operand_mutation",
    "runtime_mutation",
)
ATTEMPT_EVIDENCE = {
    "sequence",
    "process_id",
    "thread_id",
    "elapsed_ns",
    "operand_fingerprint_fnv64",
    "witness_fingerprint_fnv64",
    "accepted_output_fingerprint_fnv64",
    "runtime_before_fingerprint_fnv64",
    "runtime_after_fingerprint_fnv64",
    "prefix_canary_verified",
    "suffix_canary_verified",
    "finiteness_verified",
    "serialized_entry_verified",
    "deterministic_non_vendor_backend",
    "semantic_phase",
    "semantic_role",
    "semantic_placement",
    "canonical_begin",
    "canonical_end",
    "variant_coordinate_mode",
    "variant_range_is_exact",
    "variant_membership_count",
    "variant_membership_sha256",
}
BUILD_PROVENANCE_KEYS = {
    "schema",
    "source_commit",
    "source_tree_sha256",
    "compiler_id",
    "compiler_version",
    "cxx_standard",
    "build_type",
    "sanitizer_mode",
    "asan_enabled",
    "ubsan_enabled",
    "effective_optimization",
    "architecture_tuning",
    "configured_compiler_flags",
    "blas_vendor",
    "gemm_integrity_enabled",
    "gemm_checksum_enabled",
    "private_blas_enabled",
    "private_blas_backend",
    "private_blas_sha256",
    "private_blas_source_commit",
    "private_blas_source_tree_sha256",
    "private_blas_config_family",
    "private_blas_header_sha256",
    "private_blas_cblas_header_sha256",
    "private_openblas_enabled",
    "private_openblas_sha256",
    "native_arch_optimization_enabled",
    "openmp_enabled",
    "contextual_dispatch_backend",
    "contextual_dispatch_vendor_calls",
}


def _assert_sha256(value: object) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")


def _assert_array_digests(result: Mapping[str, Any], expected_names: set[str]) -> None:
    observed = dict(result["scientific_array_sha256"])
    assert set(observed) == expected_names
    for name, digest in observed.items():
        assert digest == array_sha256(result[name]), name


def _assert_success_ledger(result: Mapping[str, Any]) -> None:
    telemetry = dict(result["telemetry"])
    events = [dict(value) for value in telemetry["events"]]
    assert [int(value["sequence"]) for value in events] == list(
        range(1, len(events) + 1)
    )
    assert int(telemetry["observed_events"]) == len(events)
    assert bool(telemetry["complete_without_drop"])
    protected = [value for value in events if value["event_class"] == "protected_call"]
    verified = [
        value
        for value in events
        if value["event_class"] == "semantic_verification"
        and value["operation"] != "none"
    ]
    assert len(protected) == int(result["admission"]["total_protected_calls"])
    assert len(verified) == len(protected)
    for event in protected + verified:
        assert ATTEMPT_EVIDENCE <= set(event)
        assert int(event["rows"]) > 0
        assert int(event["columns"]) > 0
        assert int(event["reduction"]) > 0
        assert int(event["left_stride"]) > 0
        assert int(event["right_stride"]) > 0
        assert int(event["output_stride"]) >= int(event["rows"])
        assert bool(event["prefix_canary_verified"])
        assert bool(event["suffix_canary_verified"])
        assert bool(event["finiteness_verified"])
        assert bool(event["serialized_entry_verified"])
        assert bool(event["deterministic_non_vendor_backend"])
        assert event["runtime_before_fingerprint_fnv64"] == (
            event["runtime_after_fingerprint_fnv64"]
        )


def _membership_sha256(variants: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"contextual_exact_logical_variant_membership_v1")
    digest.update(len(variants).to_bytes(8, "little", signed=False))
    for variant in variants:
        digest.update(variant.to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _event_membership(
    event: Mapping[str, Any], candidates: tuple[int, ...]
) -> tuple[int, ...]:
    digest = str(event["variant_membership_sha256"])
    matches = [
        candidates[begin:end]
        for begin in range(len(candidates))
        for end in range(begin + 1, len(candidates) + 1)
        if _membership_sha256(candidates[begin:end]) == digest
    ]
    assert len(matches) == 1
    membership = matches[0]
    assert int(event["variant_membership_count"]) == len(membership)
    assert int(event["variant_begin"]) == min(membership)
    assert int(event["variant_end"]) == max(membership) + 1
    assert bool(event["variant_range_is_exact"]) == (
        membership == tuple(range(min(membership), max(membership) + 1))
    )
    return membership


def _runtime_mutation_worker(
    case: Any,
    phenotypes: np.ndarray,
    residual_basis: np.ndarray,
    queue: Any,
) -> None:
    before = tuple(sorted(os.sched_getaffinity(0)))
    executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=4,
        trait_feature_tile=5,
        fault_operation="trait_score_tn",
        fault_mode="runtime_mutation",
        fault_semantic_anchor=_trait_anchor(
            case, phenotypes, residual_basis, "trait_score_tn"
        ),
    )
    try:
        executor.run()
    except RuntimeError as error:
        report = dict(executor.failure_report())
        detections = [
            dict(value)
            for value in report["events"]
            if value["event_class"] == "fault_detection"
        ]
        queue.put(
            {
                "before": before,
                "after": tuple(sorted(os.sched_getaffinity(0))),
                "message": str(error),
                "report": {
                    name: report[name]
                    for name in (
                        "event_sequence_contiguous",
                        "event_ownership_complete",
                        "event_evidence_complete",
                        "event_resolution_complete",
                        "event_counts_consistent",
                        "terminal_event_unique",
                        "terminal_event_last",
                        "event_ledger_complete_without_drop",
                    )
                },
                "detections": detections,
            }
        )
        return
    queue.put({"unexpected_success": True})


def _trait_anchor(
    case: Any,
    phenotypes: np.ndarray,
    residual_basis: np.ndarray,
    operation: str,
) -> str:
    executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=4,
        trait_feature_tile=5,
    )
    return str(
        next(
            value["semantic_anchor"]
            for value in executor.semantic_anchors()
            if value["operation"] == operation
        )
    )


def test_stage5_reference_schema_digests_file_identity_and_raw_snapshot(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=1, name="stage5-reference-schema")
    probes = _variant_probes(case)
    executor = _stage3_executor(case, probes, enable_differential_snapshot=True)
    result = dict(executor.run())
    snapshot = dict(executor.differential_snapshot())

    assert result["contextual_native_api_version"] == 1
    assert result["contextual_backend_version"] == 2
    assert result["contextual_backend"] == ("plink_bed_descriptor_stream_stage2_v1")
    assert result["contextual_execution_backend"] == (
        "deterministic_tiled_fp64_with_scalar_witness_v1"
    )
    _assert_array_digests(result, REFERENCE_ARRAYS)
    _assert_sha256(result["execution_plan_sha256"])
    assert set(result["build_provenance"]) == BUILD_PROVENANCE_KEYS
    assert result["build_provenance"]["contextual_dispatch_vendor_calls"] is False
    assert set(result["phase_evidence_sha256"]) == {"post_gram"}
    _assert_sha256(result["phase_evidence_sha256"]["post_gram"])
    _assert_success_ledger(result)

    identity = dict(result["file_content_identity"])
    assert (
        identity["bim_full_sha256"]
        == hashlib.sha256(Path(f"{case.prefix}.bim").read_bytes()).hexdigest()
    )
    assert (
        identity["fam_full_sha256"]
        == hashlib.sha256(Path(f"{case.prefix}.fam").read_bytes()).hexdigest()
    )
    assert int(identity["retained_record_sha256_count"]) == int(
        identity["retained_record_count"]
    )
    _assert_sha256(identity["bed_header_sha256"])
    _assert_sha256(identity["retained_bed_logical_record_stream_sha256"])
    assert identity["full_bed_file_sha256_claimed"] is False
    assert identity["absolute_snapshot_or_lease_claimed"] is False
    assert identity["toctou_closed"] is False

    sums = np.asarray(snapshot["same_probe_sums"])
    cross = np.asarray(snapshot["same_probe_cross"])
    assert sums.shape == (case.fixed_basis.shape[0], result["gram"].shape[0])
    assert cross.shape == result["gram"].shape
    assert sums.flags.c_contiguous and cross.flags.c_contiguous
    assert snapshot["same_probe_sums_layout"] == "sample_component_c_v1"
    assert snapshot["same_probe_cross_layout"] == "component_component_c_v1"
    assert snapshot["same_probe_sums_sha256"] == array_sha256(sums)
    assert snapshot["same_probe_cross_sha256"] == array_sha256(cross)
    native_identity = dict(snapshot["same_probe_native_identity"])
    assert native_identity["schema"] == (
        "contextual_native_variant_probe_full_range_identity_v1"
    )
    assert native_identity["source_commit"] == result["contextual_build_id"]
    assert native_identity["source_tree_sha256"] == result["source_tree_sha256"]
    assert native_identity["native_api_version"] == 1
    assert native_identity["native_backend_version"] == 2
    assert native_identity["native_backend"] == result["contextual_backend"]
    assert native_identity["native_execution_backend"] == (
        result["contextual_execution_backend"]
    )
    assert native_identity["execution_plan_sha256"] == (result["execution_plan_sha256"])
    assert native_identity["sealed_plan_sha256"] == result["sealed_plan_sha256"]
    assert native_identity["global_probe_begin"] == 0
    assert native_identity["global_probe_end"] == probes.shape[1]
    assert native_identity["global_probe_count"] == probes.shape[1]
    assert native_identity["sample_count"] == sums.shape[0]
    assert native_identity["component_count"] == sums.shape[1]
    assert native_identity["full_executor_range_only"] is True
    assert native_identity["partitioned_native_finalizer_claimed"] is False
    ordered_leaves = tuple(native_identity["ordered_probe_sha256s"])
    assert ordered_leaves == tuple(
        array_sha256(np.asarray(probes[:, probe])) for probe in range(probes.shape[1])
    )
    assert native_identity["global_probe_plan_sha256"] == (
        contextual_variant_probe_plan_sha256_v1(ordered_leaves)
    )
    science = dict(native_identity["science_identity_inputs"])
    for name in (
        "annotation_map_sha256",
        "evaluated_phi_sha256",
        "fixed_basis_sha256",
        "group_map_sha256",
        "missingness_sha256",
        "retained_sample_map_sha256",
        "retained_variant_order_sha256",
        "scale_plan_sha256",
        "variant_order_allele_sha256",
    ):
        assert science[name] == result[name]
    assert tuple(science["annotation_names"]) == tuple(result["annotation_names"])
    assert tuple(science["group_names"]) == tuple(result["group_names"])
    for name in (
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
    ):
        np.testing.assert_array_equal(science[name], result[name])
    for forbidden in ("same_probe_sums", "same_probe_cross"):
        assert forbidden not in result

    leaves = tuple(_sha(f"native-probe-{index}") for index in range(probes.shape[1]))
    with pytest.raises(ValueError, match="full-range commitment mismatch"):
        adapt_native_contextual_variant_probe_partial_v1(
            snapshot,
            identity=_identity(leaves, policy="explicit_rademacher_v1"),
            global_probe_begin=0,
            global_probe_end=probes.shape[1],
            global_probe_count=probes.shape[1],
            ordered_probe_sha256s=leaves,
            process_slot=0,
            process_start_method="in_process_native_stage5_test",
        )


def test_stage5_trait_schema_and_all_array_digests(tmp_path: Path) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage5-trait-schema")
    phenotypes, residual_basis = _trait_inputs(case)
    result = dict(_trait_executor(case, phenotypes, residual_basis).run())
    assert result["contextual_native_api_version"] == 1
    assert result["contextual_backend_version"] == 2
    assert result["contextual_backend"] == ("plink_bed_descriptor_stream_trait_v1")
    _assert_array_digests(result, TRAIT_ARRAYS)
    assert set(result["phase_evidence_sha256"]) == {
        "residual_derived_state",
        "post_trait_outputs",
    }
    for value in result["phase_evidence_sha256"].values():
        _assert_sha256(value)
    _assert_success_ledger(result)


def test_stage5_direct_scaling_anchors_target_each_placement(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage5-direct-anchors")
    probes = _variant_probes(case)
    options = {
        "direct_grouped_scaling": "both_differential_v1",
        "enable_grouped_differential": True,
    }
    anchors = [
        dict(value)
        for value in _stage3_executor(case, probes, **options).semantic_anchors()
        if value["operation"] == "direct_grouped_tn"
    ]
    assert {value["semantic_placement"] for value in anchors} == {
        "action_scaled",
        "genotype_scaled",
    }
    for selected in anchors:
        result = dict(
            _stage3_executor(
                case,
                probes,
                **options,
                fault_operation="direct_grouped_tn",
                fault_mode="one_shot",
                fault_semantic_anchor=selected["semantic_anchor"],
            ).run()
        )
        injected = [
            dict(value)
            for value in result["telemetry"]["events"]
            if value["event_class"] == "fault_injection"
        ]
        assert len(injected) == 1
        assert injected[0]["semantic_placement"] == selected["semantic_placement"]


def test_stage5_same_person_anchors_target_tile_and_global_merge(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage5-same-person-anchors")
    probes = _variant_probes(case)
    anchors = [
        dict(value)
        for value in _stage3_executor(case, probes).semantic_anchors()
        if value["operation"] == "same_person_gram_tn"
    ]
    assert {value["semantic_role"] for value in anchors} == {
        "tile",
        "global_merge",
    }
    for selected in anchors:
        result = dict(
            _stage3_executor(
                case,
                probes,
                fault_operation="same_person_gram_tn",
                fault_mode="one_shot",
                fault_semantic_anchor=selected["semantic_anchor"],
            ).run()
        )
        injected = [
            dict(value)
            for value in result["telemetry"]["events"]
            if value["event_class"] == "fault_injection"
        ]
        assert len(injected) == 1
        assert injected[0]["semantic_role"] == selected["semantic_role"]


def test_stage5_grouped_semantic_anchors_choose_valid_permuted_group_points(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=2,
        annotation_mode="strict_disjoint_binary_v1",
        name="stage5-permuted-group-anchors",
    )
    probes = _variant_probes(case)
    group_index = np.asarray([2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    restricted_options = {"group_index": group_index}
    restricted_anchors = {
        str(value["operation"]): str(value["semantic_anchor"])
        for value in _stage3_executor(
            case, probes, **restricted_options
        ).semantic_anchors()
    }
    assert "variant=1|annotation=1" in restricted_anchors["group_target_nn"]
    assert "|group=0" in restricted_anchors["group_target_nn"]
    assert "variant=1|annotation=0" in restricted_anchors["group_cross_gram_tn"]
    assert "|group=0" in restricted_anchors["group_cross_gram_tn"]
    for operation in ("group_target_nn", "group_cross_gram_tn"):
        result = dict(
            _stage3_executor(
                case,
                probes,
                **restricted_options,
                fault_operation=operation,
                fault_mode="one_shot",
                fault_semantic_anchor=restricted_anchors[operation],
            ).run()
        )
        injected = [
            dict(value)
            for value in result["telemetry"]["events"]
            if value["event_class"] == "fault_injection"
        ]
        assert len(injected) == 1
        assert int(injected[0]["group_begin"]) == 0
        candidates = (1, 7) if operation == "group_target_nn" else (1, 4, 7, 10)
        assert _event_membership(injected[0], candidates)[0] == 1

    direct_options = {
        "group_index": group_index,
        "grouped_algorithm": "direct_grouped_tn_v1",
    }
    direct_anchor = str(
        next(
            value["semantic_anchor"]
            for value in _stage3_executor(
                case, probes, **direct_options
            ).semantic_anchors()
            if value["operation"] == "direct_grouped_tn"
        )
    )
    assert "variant=0|annotation=0" in direct_anchor
    assert "|group=2|" in direct_anchor
    direct = dict(
        _stage3_executor(
            case,
            probes,
            **direct_options,
            fault_operation="direct_grouped_tn",
            fault_mode="one_shot",
            fault_semantic_anchor=direct_anchor,
        ).run()
    )
    injected = [
        dict(value)
        for value in direct["telemetry"]["events"]
        if value["event_class"] == "fault_injection"
    ]
    assert len(injected) == 1
    assert int(group_index[0]) == 2


@pytest.mark.parametrize(
    "annotation_mode",
    ("generic_nonnegative_weights_v1", "strict_disjoint_binary_v1"),
)
def test_stage5_group_target_coordinates_are_global_and_membership_exact_across_tiles(
    tmp_path: Path,
    annotation_mode: str,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=2,
        annotation_mode=annotation_mode,
        name=f"stage5-group-coordinate-{annotation_mode}",
    )
    probes = _variant_probes(case)
    results = [
        dict(
            _stage3_executor(
                case,
                probes,
                variant_block=variant_block,
                sample_probe_tile=sample_probe_tile,
                annotation_tile=annotation_tile,
                context_tile=context_tile,
            ).run()
        )
        for variant_block, sample_probe_tile, annotation_tile, context_tile in (
            (2, 2, 1, 1),
            (5, 3, 2, 2),
        )
    ]
    for name in REFERENCE_ARRAYS:
        np.testing.assert_allclose(
            results[0][name], results[1][name], rtol=2e-12, atol=2e-12
        )

    for result in results:
        events = [
            dict(value)
            for value in result["telemetry"]["events"]
            if value["event_class"] == "protected_call"
            and value["operation"] == "group_target_nn"
        ]
        assert events
        for event in events:
            group = int(event["group_begin"])
            annotation = int(event["annotation_begin"])
            candidates = tuple(
                variant
                for variant, observed_group in enumerate(case.group_index)
                if int(observed_group) == group
                and (
                    annotation_mode != "strict_disjoint_binary_v1"
                    or variant % len(case.annotation_names) == annotation
                )
            )
            membership = _event_membership(event, candidates)
            assert all(
                int(case.group_index[variant]) == group for variant in membership
            )
            assert event["variant_coordinate_mode"] == ("exact_membership_sha256_v1")
            assert "variant_sha256=" in str(event["semantic_anchor"])
        cross_events = [
            dict(value)
            for value in result["telemetry"]["events"]
            if value["event_class"] == "protected_call"
            and value["operation"] == "group_cross_gram_tn"
        ]
        assert cross_events
        for event in cross_events:
            group = int(event["group_begin"])
            candidates = tuple(
                variant
                for variant, observed_group in enumerate(case.group_index)
                if int(observed_group) == group
            )
            assert _event_membership(event, candidates) == candidates


def test_stage5_strict_target_selectors_reject_nonmembers_and_use_global_points(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=2,
        annotation_mode="strict_disjoint_binary_v1",
        name="stage5-strict-membership-selector",
    )
    probes = _variant_probes(case)
    absent = {
        "full_target_nn": (
            "full_target_nn|resident=0|probe=0|variant=1|annotation=0|"
            "context=0|action=0"
        ),
        "group_target_nn": (
            "group_target_nn|resident=0|probe=0|variant=1|annotation=1|"
            "context=0|action=0|group=0"
        ),
    }
    for operation, anchor in absent.items():
        executor = _stage3_executor(
            case,
            probes,
            variant_block=4,
            annotation_tile=1,
            context_tile=1,
            fault_operation=operation,
            fault_mode="one_shot",
            fault_semantic_anchor=anchor,
        )
        with pytest.raises(RuntimeError, match="matched no protected call"):
            executor.run()

    positive_anchor = (
        "group_target_nn|resident=0|probe=0|variant=9|annotation=1|"
        "context=0|action=0|group=0"
    )
    positive = dict(
        _stage3_executor(
            case,
            probes,
            variant_block=2,
            annotation_tile=1,
            context_tile=1,
            fault_operation="group_target_nn",
            fault_mode="one_shot",
            fault_semantic_anchor=positive_anchor,
        ).run()
    )
    injected = [
        dict(value)
        for value in positive["telemetry"]["events"]
        if value["event_class"] == "fault_injection"
    ]
    assert len(injected) == 1
    assert _event_membership(injected[0], (3, 9)) == (9,)

    for operation, variant, group, should_match in (
        ("group_cross_gram_tn", 1, 0, False),
        ("group_cross_gram_tn", 9, 0, True),
        ("direct_grouped_tn", 1, 0, False),
        ("direct_grouped_tn", 9, 0, True),
    ):
        anchor = (
            f"{operation}|resident=0|probe=0|variant={variant}|annotation=0|"
            f"context=0|action=0|group={group}"
        )
        options: dict[str, object] = {}
        if operation == "direct_grouped_tn":
            options["grouped_algorithm"] = "direct_grouped_tn_v1"
        executor = _stage3_executor(
            case,
            probes,
            variant_block=4,
            fault_operation=operation,
            fault_mode="one_shot",
            fault_semantic_anchor=anchor,
            **options,
        )
        if not should_match:
            with pytest.raises(RuntimeError, match="matched no protected call"):
                executor.run()
            continue
        result = dict(executor.run())
        selected = [
            dict(value)
            for value in result["telemetry"]["events"]
            if value["event_class"] == "fault_injection"
        ]
        assert len(selected) == 1
        assert int(case.group_index[variant]) == int(selected[0]["group_begin"])


def test_stage5_trait_projection_coordinates_cover_global_features_across_tiles(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=3, name="stage5-trait-global-features")
    phenotypes, residual_basis = _trait_inputs(case)
    results = [
        dict(
            _trait_executor(
                case,
                phenotypes,
                residual_basis,
                variant_block=variant_block,
                trait_feature_tile=feature_tile,
            ).run()
        )
        for variant_block, feature_tile in ((4, 5), (3, 2))
    ]
    for name in SCIENCE_ARRAYS:
        np.testing.assert_allclose(
            results[0][name], results[1][name], rtol=2e-12, atol=2e-12
        )
    expected = list(range(case.phi.shape[1] * case.retained_variant_rows.size))
    for result in results:
        for operation in (
            "trait_feature_projection_tn",
            "trait_feature_projection_nn",
        ):
            observed: list[int] = []
            events = [
                dict(value)
                for value in result["telemetry"]["events"]
                if value["event_class"] == "protected_call"
                and value["operation"] == operation
            ]
            assert events
            for event in events:
                context = int(event["context_begin"])
                variant_begin = int(event["variant_begin"])
                variant_end = int(event["variant_end"])
                assert int(event["context_end"]) == context + 1
                assert int(event["canonical_begin"]) == (
                    context * case.retained_variant_rows.size + variant_begin
                )
                assert int(event["canonical_end"]) == (
                    context * case.retained_variant_rows.size + variant_end
                )
                assert int(event["action_begin"]) == int(event["canonical_begin"])
                assert int(event["action_end"]) == int(event["canonical_end"])
                observed.extend(
                    range(
                        int(event["canonical_begin"]),
                        int(event["canonical_end"]),
                    )
                )
            assert sorted(observed) == expected


@pytest.mark.parametrize("operation", TRAIT_OPERATIONS)
@pytest.mark.parametrize("mode", RECOVERABLE_TRAIT_FAULTS)
def test_stage5_all_recoverable_trait_faults_preserve_science(
    tmp_path: Path,
    operation: str,
    mode: str,
) -> None:
    case = _make_case(tmp_path, q_count=2, name=f"trait-recover-{operation}-{mode}")
    phenotypes, residual_basis = _trait_inputs(case)
    baseline = dict(
        _trait_executor(
            case, phenotypes, residual_basis, variant_block=4, trait_feature_tile=5
        ).run()
    )
    result = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=4,
            trait_feature_tile=5,
            fault_operation=operation,
            fault_mode=mode,
            fault_semantic_anchor=_trait_anchor(
                case, phenotypes, residual_basis, operation
            ),
        ).run()
    )
    for name in SCIENCE_ARRAYS:
        np.testing.assert_allclose(result[name], baseline[name], rtol=2e-12, atol=2e-12)
    events = [dict(value) for value in result["telemetry"]["events"]]
    for event in events:
        if event["event_class"] in {"retry", "trusted_fallback"}:
            assert int(event["rows"]) > 0
            assert int(event["columns"]) > 0
            assert int(event["reduction"]) > 0
            assert ATTEMPT_EVIDENCE <= set(event)
    _assert_success_ledger(result)


@pytest.mark.parametrize("operation", TRAIT_OPERATIONS)
@pytest.mark.parametrize("mode", TERMINAL_TRAIT_FAULTS)
def test_stage5_all_terminal_trait_faults_report_without_publication(
    tmp_path: Path,
    operation: str,
    mode: str,
) -> None:
    case = _make_case(tmp_path, q_count=2, name=f"trait-terminal-{operation}-{mode}")
    phenotypes, residual_basis = _trait_inputs(case)
    executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=4,
        trait_feature_tile=5,
        fault_operation=operation,
        fault_mode=mode,
        fault_semantic_anchor=_trait_anchor(
            case, phenotypes, residual_basis, operation
        ),
    )
    with pytest.raises(RuntimeError):
        executor.run()
    report = dict(executor.failure_report())
    assert report["metadata_only"] is True
    assert report["last_phase"] == "trait"
    assert report["terminal_event_recorded"] is True
    assert report["event_ledger_complete_without_drop"] is True
    for name in (
        "event_sequence_contiguous",
        "event_ownership_complete",
        "event_evidence_complete",
        "event_resolution_complete",
        "event_counts_consistent",
        "event_capacity_complete",
        "terminal_event_unique",
        "terminal_event_last",
    ):
        assert report[name] is True
    assert int(report["terminal_event_count"]) == 1
    events = [dict(value) for value in report["events"]]
    assert events[-1]["event_class"] == "terminal_failure"
    assert [int(value["sequence"]) for value in events] == list(
        range(1, len(events) + 1)
    )
    assert not any(isinstance(value, np.ndarray) for value in report.values())
    detections = [
        value for value in events if value["event_class"] == "fault_detection"
    ]
    assert detections
    assert int(report["fault_detection_count"]) == len(detections)
    assert all(str(value["resolution"]) for value in detections)
    with pytest.raises(RuntimeError, match="one-shot|lifecycle"):
        executor.run()


def test_stage5_runtime_mutation_uses_actual_affinity_and_restores_in_spawn(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 2:
        pytest.skip("actual runtime affinity mutation needs two admitted CPUs")
    case = _make_case(tmp_path, q_count=2, name="stage5-runtime-affinity-spawn")
    phenotypes, residual_basis = _trait_inputs(case)
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_runtime_mutation_worker,
        args=(case, phenotypes, residual_basis, queue),
    )
    process.start()
    process.join(timeout=30.0)
    assert process.exitcode == 0
    observed = queue.get(timeout=5.0)
    assert "unexpected_success" not in observed
    assert observed["before"] == observed["after"]
    assert "detected and restored" in observed["message"]
    assert all(observed["report"].values())
    detections = observed["detections"]
    assert len(detections) == 1
    assert detections[0]["resolution"] == "runtime_fingerprint_mismatch"
    assert detections[0]["runtime_before_fingerprint_fnv64"] != (
        detections[0]["runtime_after_fingerprint_fnv64"]
    )


def test_stage5_exact_and_minus_one_telemetry_capacity(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=1, name="stage5-capacity")
    probes = _variant_probes(case)
    required = int(
        _stage3_executor(case, probes).preflight()["required_telemetry_capacity"]
    )
    result = dict(_stage3_executor(case, probes, telemetry_capacity=required).run())
    _assert_success_ledger(result)
    with pytest.raises(RuntimeError, match="telemetry capacity"):
        _stage3_executor(case, probes, telemetry_capacity=required - 1)

    phenotypes, residual_basis = _trait_inputs(case)
    trait_required = int(
        _trait_executor(case, phenotypes, residual_basis).preflight()[
            "required_telemetry_capacity"
        ]
    )
    trait = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            telemetry_capacity=trait_required,
        ).run()
    )
    _assert_success_ledger(trait)
    with pytest.raises(RuntimeError, match="telemetry capacity"):
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            telemetry_capacity=trait_required - 1,
        )


def test_stage5_retained_bed_record_private_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage5-bed-content")
    executor = _stage3_executor(
        case,
        _variant_probes(case),
        test_checkpoint="post_source",
        test_mutation_target="retained_bed_record",
    )
    error = _run_checkpoint_failure(executor, "post_source")
    assert "retained bed record content mutation" in str(error).lower()
    report = dict(executor.failure_report())
    assert report["last_phase"] == "action"
    assert report["terminal_event_recorded"] is True
    assert report["event_ledger_complete_without_drop"] is True


def test_stage5_post_gram_seal_detects_output_mutation(tmp_path: Path) -> None:
    case = _make_case(tmp_path, q_count=1, name="stage5-post-gram")
    executor = _stage3_executor(
        case,
        _variant_probes(case),
        test_checkpoint="post_gram",
        test_mutation_target="output_buffers",
    )
    error = _run_checkpoint_failure(executor, "post_gram")
    assert "post-gram phase seal mutation" in str(error).lower()


def test_stage5_process_wide_dispatch_is_observed_serialized_across_executors(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage5-dispatch-threads")
    probes = _variant_probes(case)
    executors = [
        _stage3_executor(case, probes),
        _stage3_executor(case, probes),
    ]
    start = threading.Barrier(2)

    def run(executor: Any) -> Mapping[str, Any]:
        start.wait()
        return executor.run()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, executor) for executor in executors]
        results = [future.result(timeout=30.0) for future in futures]

    protected_events = [
        event
        for result in results
        for event in result["telemetry"]["events"]
        if event["event_class"]
        in {
            "protected_call",
            "retry",
            "trusted_fallback",
            "semantic_verification",
        }
        and event["operation"] != "none"
    ]
    assert protected_events
    assert len({event["thread_id"] for event in protected_events}) == 2
    assert all(event["serialized_entry_verified"] for event in protected_events)
    for key in REFERENCE_ARRAYS:
        np.testing.assert_array_equal(results[0][key], results[1][key])


def test_stage5_native_static_no_vendor_or_unprotected_contextual_bypass() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "contextual_dispatcher_mutex()" in source
    assert "ContextualDispatcherObservationV1" in source
    assert "process_maximum.compare_exchange_weak" in source
    assert "singleton_process_entry_observed()" in source
    assert "kProtectedPrefixCanariesV1" in source
    assert "kProtectedSuffixCanariesV1" in source
    assert "primary_buffer_.data() + prefix" in source
    assert "retry_buffer_.data() + prefix" in source
    assert "fallback_buffer_.data() + prefix" in source
    for forbidden in (
        "cblas_dgemm(",
        "dgemm_nn_raw(",
        "dgemm_tn_raw(",
        "DirectContext::",
        "MultiEnvironmentKernel(",
    ):
        assert forbidden not in source
    assert source.count("protected_tn(") > 7
    assert source.count("protected_nn(") > 5
    assert source.count("ContextualDenseSemanticV1::") >= 17
