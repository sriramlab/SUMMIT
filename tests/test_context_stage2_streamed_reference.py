from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pytest
from bed_reader import to_bed

from summit import gxeldcore
from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
    array_sha256,
    context_kernel_actions,
    hutchinson_gram,
)
from summit.ldscore.gwe_ldscore import _make_seed


ATOL = 1.0e-9
RTOL = 1.0e-11

STAGE2_OPERATIONS = (
    "sample_probe_projection_tn",
    "sample_probe_projection_nn",
    "source_tn",
    "full_target_nn",
    "action_projection_tn",
    "action_projection_nn",
    "action_gram_tn",
)

CONTEXTUAL_STREAMED_SOURCE = (
    Path(__file__).parents[1]
    / "src"
    / "native"
    / "contextual_streamed_reference_v1.inc"
)


@dataclass(frozen=True)
class StreamedCase:
    prefix: Path
    raw_a1: np.ndarray
    retained_sample_rows: np.ndarray
    retained_variant_rows: np.ndarray
    expected_variant_ids: tuple[str, ...]
    expected_counted_alleles: tuple[str, ...]
    expected_other_alleles: tuple[str, ...]
    affine_mean: np.ndarray
    affine_inverse_scale: np.ndarray
    fixed_basis: np.ndarray
    phi: np.ndarray
    annotations: np.ndarray
    group_index: np.ndarray
    sample_probes: np.ndarray
    annotation_names: tuple[str, ...]
    group_names: tuple[str, ...]
    annotation_mode: str
    counted_allele_mode: str
    centering_source: str
    retained_variant_order_sha256: str
    affine_mean_sha256: str
    affine_inverse_scale_sha256: str
    scale_plan_sha256: str
    missingness_sha256: str
    scaled_genotype: np.ndarray


def _assert_close(actual: object, expected: object, label: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=RTOL,
        atol=ATOL,
        err_msg=label,
    )


def _walk_arrays(value: Any, path: str = "result") -> Iterator[tuple[str, np.ndarray]]:
    if isinstance(value, np.ndarray):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_arrays(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_arrays(item, f"{path}[{index}]")


def _scale_plan(
    *,
    retained_variant_order_sha256: str,
    counted_allele_mode: str,
    centering_source: str,
    affine_mean_sha256: str,
    affine_inverse_scale_sha256: str,
) -> GenotypeScalePlanV1:
    return GenotypeScalePlanV1(
        policy=GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1,
        retained_variant_order_sha256=retained_variant_order_sha256,
        allele_orientation=counted_allele_mode,
        allele_coding="plink_bed_snp_major_diploid_hardcall_v1",
        centering_source=centering_source,
        centering_formula="provided_variant_affine_mean_v1",
        scaling_formula="dosage_minus_mean_times_inverse_scale_v1",
        missing_imputation="sealed_mean_v1",
        ploidy_policy="diploid_v1",
        affine_mean_sha256=affine_mean_sha256,
        affine_inverse_scale_sha256=affine_inverse_scale_sha256,
    )


def _make_case(
    tmp_path: Path,
    *,
    q_count: int = 3,
    retained_sample_count: int = 13,
    annotation_count: int = 2,
    annotation_mode: str = "generic_nonnegative_weights_v1",
    counted_allele_mode: str = "bim_a1_counted_v1",
    name: str = "streamed",
) -> StreamedCase:
    rng = np.random.default_rng(284_000 + q_count)
    n_total, m_total = 19, 17
    raw_a1 = rng.integers(0, 3, size=(n_total, m_total)).astype(np.float64)
    raw_a1[0, :] = 0.0
    raw_a1[1, :] = 1.0
    raw_a1[2, :] = 2.0
    raw_a1[2, 3] = np.nan
    raw_a1[7, 8] = np.nan
    raw_a1[15, 14] = np.nan
    raw_a1[0, 16] = np.nan

    prefix = tmp_path / name
    variant_ids = tuple(f"rs{7000 + index}" for index in range(m_total))
    allele_pairs = (
        ("A", "C"),
        ("C", "G"),
        ("G", "T"),
        ("T", "A"),
    )
    allele_1 = tuple(allele_pairs[index % len(allele_pairs)][0] for index in range(m_total))
    allele_2 = tuple(allele_pairs[index % len(allele_pairs)][1] for index in range(m_total))
    properties = {
        "fid": [f"F{index:02d}" for index in range(n_total)],
        "iid": [f"I{index:02d}" for index in range(n_total)],
        "chromosome": [str(1 + index // 9) for index in range(m_total)],
        "sid": list(variant_ids),
        "bp_position": [1000 + 7 * index for index in range(m_total)],
        "allele_1": list(allele_1),
        "allele_2": list(allele_2),
    }
    to_bed(
        str(prefix) + ".bed",
        raw_a1,
        properties=properties,
        count_A1=True,
    )

    all_sample_rows = [18, 2, 7, 0, 12, 4, 9, 15, 1, 6, 11, 3, 17]
    if retained_sample_count < 3 or retained_sample_count > len(all_sample_rows):
        raise AssertionError("retained_sample_count is outside the fixture envelope")
    if annotation_count not in (1, 2):
        raise AssertionError("annotation_count is outside the fixture envelope")
    sample_rows = np.asarray(all_sample_rows[:retained_sample_count], dtype=np.int64)
    variant_rows = np.asarray(
        [16, 3, 11, 0, 8, 5, 14, 1, 9, 6, 12], dtype=np.int64
    )
    selected_a1 = raw_a1[np.ix_(sample_rows, variant_rows)]
    base_mean = np.linspace(0.35, 1.65, variant_rows.size, dtype=np.float64)
    inverse_scale = np.linspace(0.55, 1.45, variant_rows.size, dtype=np.float64)
    if counted_allele_mode == "bim_a1_counted_v1":
        counted = selected_a1
        mean = base_mean
        expected_counted = tuple(allele_1[index] for index in variant_rows)
        expected_other = tuple(allele_2[index] for index in variant_rows)
    elif counted_allele_mode == "bim_a2_counted_v1":
        counted = np.where(np.isnan(selected_a1), np.nan, 2.0 - selected_a1)
        mean = 2.0 - base_mean
        expected_counted = tuple(allele_2[index] for index in variant_rows)
        expected_other = tuple(allele_1[index] for index in variant_rows)
    else:  # pragma: no cover - helper misuse
        raise AssertionError(counted_allele_mode)
    scaled = (
        np.where(np.isnan(counted), mean[None, :], counted) - mean[None, :]
    ) * inverse_scale[None, :]

    fixed_design = np.column_stack(
        [np.ones(sample_rows.size), np.linspace(-1.0, 1.0, sample_rows.size)]
    )
    fixed_basis = np.linalg.qr(fixed_design, mode="reduced")[0]
    coordinate = np.linspace(-1.3, 1.1, sample_rows.size)
    phi_all = np.column_stack(
        [
            np.ones(sample_rows.size),
            coordinate,
            np.sin(1.7 * coordinate) + 0.2 * coordinate,
            np.cos(0.8 * coordinate) - 0.15 * coordinate,
        ]
    )
    phi = phi_all[:, :q_count]
    if annotation_mode == "strict_disjoint_binary_v1":
        annotations = np.zeros((variant_rows.size, annotation_count), dtype=np.float64)
        annotations[
            np.arange(variant_rows.size),
            np.arange(variant_rows.size) % annotation_count,
        ] = 1.0
    elif annotation_mode == "generic_nonnegative_weights_v1":
        index = np.arange(variant_rows.size, dtype=np.float64)
        columns = [0.2 + 0.07 * index, 1.35 - 0.035 * index]
        annotations = np.column_stack(columns[:annotation_count])
    else:  # pragma: no cover - helper misuse
        raise AssertionError(annotation_mode)
    group_index = np.arange(variant_rows.size, dtype=np.int64) % 3
    sample_probes = (
        2 * rng.integers(0, 2, size=(sample_rows.size, 5)) - 1
    ).astype(np.float64)

    order_sha256 = array_sha256(variant_rows)
    mean_sha256 = array_sha256(mean)
    inverse_sha256 = array_sha256(inverse_scale)
    centering_source = "stage2_test_known_affine_v1"
    plan = _scale_plan(
        retained_variant_order_sha256=order_sha256,
        counted_allele_mode=counted_allele_mode,
        centering_source=centering_source,
        affine_mean_sha256=mean_sha256,
        affine_inverse_scale_sha256=inverse_sha256,
    )
    return StreamedCase(
        prefix=prefix,
        raw_a1=raw_a1,
        retained_sample_rows=sample_rows,
        retained_variant_rows=variant_rows,
        expected_variant_ids=tuple(variant_ids[index] for index in variant_rows),
        expected_counted_alleles=expected_counted,
        expected_other_alleles=expected_other,
        affine_mean=np.asarray(mean, dtype=np.float64),
        affine_inverse_scale=inverse_scale,
        fixed_basis=np.asfortranarray(fixed_basis),
        phi=np.asfortranarray(phi),
        annotations=np.asfortranarray(annotations),
        group_index=group_index,
        sample_probes=np.asfortranarray(sample_probes),
        annotation_names=tuple(f"a{index}" for index in range(annotation_count)),
        group_names=("g0", "g1", "g2"),
        annotation_mode=annotation_mode,
        counted_allele_mode=counted_allele_mode,
        centering_source=centering_source,
        retained_variant_order_sha256=order_sha256,
        affine_mean_sha256=mean_sha256,
        affine_inverse_scale_sha256=inverse_sha256,
        scale_plan_sha256=plan.digest,
        missingness_sha256=array_sha256(
            np.ascontiguousarray(np.isnan(selected_a1).T, dtype=np.uint8)
        ),
        scaled_genotype=np.asfortranarray(scaled),
    )


def _components(case: StreamedCase) -> ContextComponentIndex:
    return ContextComponentIndex(
        case.annotation_names,
        ContextPairIndex(case.phi.shape[1]),
    )


def _dense_expected(case: StreamedCase) -> dict[str, np.ndarray]:
    components = _components(case)
    projector = np.eye(case.fixed_basis.shape[0]) - case.fixed_basis @ case.fixed_basis.T
    projected_probes = projector @ case.sample_probes
    source = np.stack(
        [
            case.scaled_genotype.T
            @ (case.phi[:, context, None] * projected_probes)
            for context in range(case.phi.shape[1])
        ],
        axis=1,
    )
    normalized_component_first = context_kernel_actions(
        case.scaled_genotype,
        case.phi,
        projector,
        case.annotations,
        components,
        case.sample_probes,
    )
    component_annotation = np.asarray(
        [entry.annotation_index for entry in components.entries], dtype=np.int64
    )
    annotation_masses = np.sum(case.annotations, axis=0, dtype=np.float64)
    component_masses = annotation_masses[component_annotation]
    raw_component_first = (
        normalized_component_first * component_masses[:, None, None]
    )
    return {
        "source_scores": source,
        "normalized_actions": np.transpose(normalized_component_first, (1, 0, 2)),
        "raw_actions": np.transpose(raw_component_first, (1, 0, 2)),
        "gram": hutchinson_gram(normalized_component_first),
        "raw_gram_numerator": hutchinson_gram(raw_component_first),
        "annotation_masses": annotation_masses,
    }


def _dense_stage1_gram(case: StreamedCase) -> np.ndarray:
    rng = np.random.default_rng(481_000 + case.phi.shape[1])
    variant_probes = (
        2 * rng.integers(0, 2, size=(case.scaled_genotype.shape[1], 3)) - 1
    ).astype(np.float64)
    executor = gxeldcore.ContextualBlockExecutorV1(
        case.scaled_genotype,
        case.fixed_basis,
        case.phi,
        case.annotations,
        case.group_index,
        case.sample_probes,
        np.asfortranarray(variant_probes),
        np.ascontiguousarray(rng.normal(size=case.scaled_genotype.shape[0])),
        np.asfortranarray(np.ones((case.scaled_genotype.shape[0], 1))),
        list(case.annotation_names),
        list(case.group_names),
        annotation_mode=case.annotation_mode,
    )
    return np.asarray(executor.run()["gram"])


def _open_descriptors(prefix: Path) -> list[int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    return [
        os.open(str(prefix) + extension, flags)
        for extension in (".bed", ".bim", ".fam")
    ]


def _expected_missing_counts(case: StreamedCase) -> np.ndarray:
    selected = case.raw_a1[
        np.ix_(case.retained_sample_rows, case.retained_variant_rows)
    ]
    return np.sum(np.isnan(selected), axis=0, dtype=np.int64)


def _philox_probes(
    rows: int, probes: int, *, root_seed: int = 92_711
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.empty((probes, 2), dtype=np.uint64)
    values = np.empty((rows, probes), dtype=np.float64, order="F")
    for probe in range(probes):
        seed = _make_seed(root_seed, 0, probe)
        bit_generator = np.random.Philox(seed)
        keys[probe] = np.asarray(
            bit_generator.state["state"]["key"], dtype=np.uint64
        )
        generator = np.random.Generator(np.random.Philox(seed))
        values[:, probe] = (
            2.0 * generator.integers(0, 2, size=rows, dtype=np.int8) - 1.0
        )
    return np.ascontiguousarray(keys), values


def _executor(case: StreamedCase, **changes: Any) -> Any:
    flags = np.full(
        case.retained_variant_rows.size,
        case.counted_allele_mode == "bim_a1_counted_v1",
        dtype=np.uint8,
    )
    values: dict[str, Any] = {
        "retained_sample_indices": case.retained_sample_rows,
        "retained_variant_indices": case.retained_variant_rows,
        "expected_variant_ids": list(case.expected_variant_ids),
        "counted_alleles": list(case.expected_counted_alleles),
        "other_alleles": list(case.expected_other_alleles),
        "counted_allele_is_a1": flags,
        "affine_mean": case.affine_mean,
        "affine_inverse_scale": case.affine_inverse_scale,
        "expected_missing_counts": _expected_missing_counts(case),
        "fixed_basis": case.fixed_basis,
        "context_basis": case.phi,
        "annotation_weights": case.annotations,
        "group_index": case.group_index,
        "sample_probes": case.sample_probes,
        "philox_keys": None,
        "annotation_names": list(case.annotation_names),
        "group_names": list(case.group_names),
        "annotation_mode": case.annotation_mode,
        "retained_variant_order_sha256": case.retained_variant_order_sha256,
        "affine_mean_sha256": case.affine_mean_sha256,
        "affine_inverse_scale_sha256": case.affine_inverse_scale_sha256,
        "missingness_sha256": case.missingness_sha256,
        "scale_plan_sha256": case.scale_plan_sha256,
        "centering_source": case.centering_source,
        "sample_probe_count": case.sample_probes.shape[1],
        "enable_differential_snapshot": True,
    }
    values.update(changes)
    descriptors = _open_descriptors(case.prefix)
    try:
        return gxeldcore.ContextualReferenceExecutorV1(
            descriptors[0],
            descriptors[1],
            descriptors[2],
            values.pop("retained_sample_indices"),
            values.pop("retained_variant_indices"),
            values.pop("expected_variant_ids"),
            values.pop("counted_alleles"),
            values.pop("other_alleles"),
            values.pop("counted_allele_is_a1"),
            values.pop("affine_mean"),
            values.pop("affine_inverse_scale"),
            values.pop("expected_missing_counts"),
            values.pop("fixed_basis"),
            values.pop("context_basis"),
            values.pop("annotation_weights"),
            values.pop("group_index"),
            values.pop("sample_probes"),
            values.pop("philox_keys"),
            values.pop("annotation_names"),
            values.pop("group_names"),
            **values,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _run_with_snapshot(executor: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    result = dict(executor.run())
    snapshot = dict(executor.differential_snapshot())
    return result, snapshot


def _assert_maps(result: Mapping[str, Any], case: StreamedCase) -> None:
    components = _components(case)
    expected = {
        "pair_q": [entry.q for entry in components.pair_index.entries],
        "pair_r": [entry.r for entry in components.pair_index.entries],
        "pair_eta": [entry.kernel_factor for entry in components.pair_index.entries],
        "component_annotation": [
            entry.annotation_index for entry in components.entries
        ],
        "component_pair": [entry.pair_index for entry in components.entries],
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(result[name], values, err_msg=name)


def _assert_science(
    result: Mapping[str, Any], snapshot: Mapping[str, Any], case: StreamedCase
) -> None:
    expected = _dense_expected(case)
    for name in (
        "source_scores",
        "raw_actions",
        "normalized_actions",
    ):
        _assert_close(snapshot[name], expected[name], name)
    for name in ("gram", "raw_gram_numerator", "annotation_masses"):
        _assert_close(result[name], expected[name], name)
    _assert_close(result["gram"], _dense_stage1_gram(case), "Stage 1 Gram")


def _assert_execution_diagnostics(
    result: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    diagnostics = dict(result["diagnostics"])
    for path, value in _walk_arrays(result):
        if value.dtype.kind in "fc":
            assert np.isfinite(value).all(), path
    assert float(diagnostics["maximum_projection_leakage"]) <= 2.0e-9
    assert float(diagnostics["gram_pre_symmetry_max_abs"]) <= ATOL
    assert int(diagnostics["observed_descriptor_passes"]) == int(
        preflight["total_descriptor_passes"]
    )
    assert int(diagnostics["observed_decoded_blocks"]) == int(
        preflight["total_decoded_blocks"]
    )
    assert int(diagnostics["observed_variant_record_visits"]) == int(
        preflight["total_variant_record_visits"]
    )
    assert dict(diagnostics["protected_call_counts"]) == dict(
        preflight["semantic_call_ledger"]
    )
    assert bool(diagnostics["descriptor_accounting_verified"])
    assert bool(diagnostics["semantic_call_ledger_exact"])
    assert bool(diagnostics["files_unchanged_at_all_checkpoints"])
    assert bool(diagnostics["inputs_unchanged_at_all_checkpoints"])
    assert bool(diagnostics["scratch_released"])
    assert bool(diagnostics["all_large_buffers_preallocated_before_decode"])
    assert int(diagnostics["runtime_large_allocations_after_decode"]) == 0
    assert int(diagnostics["tracked_high_water_bytes"]) <= int(
        preflight["required_workspace_bytes"]
    )
    assert bool(diagnostics["tracked_high_water_within_admission"])


def _assert_protected_telemetry_records(
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    case: StreamedCase,
) -> None:
    telemetry = dict(result["telemetry"])
    events = [dict(event) for event in telemetry["events"]]
    assert int(telemetry["observed_events"]) == len(events)
    assert int(telemetry["required_capacity"]) == int(
        preflight["required_telemetry_capacity"]
    )
    assert len(events) <= int(telemetry["required_capacity"])
    assert dict(telemetry["protected_call_counts"]) == dict(
        preflight["semantic_call_ledger"]
    )

    protected = [event for event in events if event["event_class"] == "protected_call"]
    observed_counts = {operation: 0 for operation in STAGE2_OPERATIONS}
    limits = {
        "resident": case.sample_probes.shape[1],
        "probe": case.sample_probes.shape[1],
        "variant": case.scaled_genotype.shape[1],
        "annotation": case.annotations.shape[1],
        "context": case.phi.shape[1],
        "action": len(_components(case)),
    }
    for event in protected:
        operation = str(event["operation"])
        assert operation in observed_counts
        observed_counts[operation] += 1
        assert int(event["attempt"]) == 1
        rows = int(event["rows"])
        columns = int(event["columns"])
        reduction = int(event["reduction"])
        assert rows > 0 and columns > 0 and reduction > 0
        transpose_left = bool(event["transpose_left"])
        assert transpose_left == operation.endswith("_tn")
        assert int(event["left_stride"]) >= (reduction if transpose_left else rows)
        assert int(event["right_stride"]) >= reduction
        assert int(event["output_stride"]) >= rows
        assert str(event["semantic_anchor"]).startswith(operation + "|")
        assert "call_ordinal" not in event and "occurrence" not in event
        for coordinate, limit in limits.items():
            begin = int(event[f"{coordinate}_begin"])
            end = int(event[f"{coordinate}_end"])
            assert 0 <= begin <= end <= limit
    expected_counts = {
        operation: int(dict(preflight["semantic_call_ledger"])[operation])
        for operation in STAGE2_OPERATIONS
    }
    assert observed_counts == expected_counts


@pytest.mark.parametrize("q_count", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "counted_allele_mode", ["bim_a1_counted_v1", "bim_a2_counted_v1"]
)
def test_generated_bed_missing_affine_source_actions_and_gram_match_dense_stage1(
    tmp_path: Path, q_count: int, counted_allele_mode: str
) -> None:
    case = _make_case(
        tmp_path,
        q_count=q_count,
        counted_allele_mode=counted_allele_mode,
        name=f"science-q{q_count}-{counted_allele_mode}",
    )
    executor = _executor(case, variant_block=4, sample_probe_tile=2, action_tile=3)
    preflight = dict(executor.preflight())
    result, snapshot = _run_with_snapshot(executor)
    _assert_science(result, snapshot, case)
    _assert_maps(result, case)
    _assert_execution_diagnostics(result, preflight)
    _assert_protected_telemetry_records(result, preflight, case)

    assert bool(result["complete_reference_artifact"]) is False
    assert result["internal_result_kind"] == "stage2_source_action_full_gram_v1"
    assert result["genotype_scale_policy"] == "sealed_variant_affine_v1"
    assert result["probe_policy"] == "explicit_rademacher_v1"
    assert result["retained_variant_order_sha256"] == (
        case.retained_variant_order_sha256
    )
    assert result["affine_mean_sha256"] == case.affine_mean_sha256
    assert result["affine_inverse_scale_sha256"] == (
        case.affine_inverse_scale_sha256
    )
    assert result["scale_plan_sha256"] == case.scale_plan_sha256
    assert int(dict(result["diagnostics"])["missing_genotype_calls"]) == int(
        np.sum(_expected_missing_counts(case))
    )
    assert dict(result["admission"]) == preflight


def test_a1_a2_orientation_changes_source_sign_but_not_kernel_actions(
    tmp_path: Path,
) -> None:
    a1 = _make_case(tmp_path, counted_allele_mode="bim_a1_counted_v1", name="orientation")
    a2 = _make_case(tmp_path, counted_allele_mode="bim_a2_counted_v1", name="orientation")
    a1_result, a1_snapshot = _run_with_snapshot(_executor(a1))
    a2_result, a2_snapshot = _run_with_snapshot(_executor(a2))
    _assert_close(a2_snapshot["source_scores"], -np.asarray(a1_snapshot["source_scores"]), "A2 source sign")
    _assert_close(a2_snapshot["raw_actions"], a1_snapshot["raw_actions"], "A1/A2 raw actions")
    _assert_close(a2_result["gram"], a1_result["gram"], "A1/A2 Gram")


def _ceil_div(value: int, width: int) -> int:
    return (value + width - 1) // width


def _expected_stage2_ledger(
    case: StreamedCase,
    *,
    variant_block: int = 0,
    sample_probe_resident: int = 0,
    sample_probe_tile: int = 0,
    action_tile: int = 0,
    annotation_tile: int = 0,
    context_tile: int = 0,
) -> tuple[dict[str, int], dict[str, int]]:
    n_variants = case.scaled_genotype.shape[1]
    probe_count = case.sample_probes.shape[1]
    q_count = case.phi.shape[1]
    k_count = case.annotations.shape[1]
    component_count = len(_components(case))
    variant_block = n_variants if variant_block == 0 else min(variant_block, n_variants)
    resident = probe_count if sample_probe_resident == 0 else min(
        sample_probe_resident, probe_count
    )
    probe_tile = probe_count if sample_probe_tile == 0 else sample_probe_tile
    action_tile = component_count if action_tile == 0 else action_tile
    annotation_tile = k_count if annotation_tile == 0 else annotation_tile
    context_tile = q_count if context_tile == 0 else context_tile
    variant_blocks = _ceil_div(n_variants, variant_block)
    context_tiles = _ceil_div(q_count, context_tile)
    annotation_tiles = _ceil_div(k_count, annotation_tile)
    action_tiles = _ceil_div(component_count, action_tile)
    resident_widths = [
        min(resident, probe_count - start)
        for start in range(0, probe_count, resident)
    ]
    probe_tiles = [
        _ceil_div(width, min(probe_tile, width)) for width in resident_widths
    ]
    total_probe_tiles = sum(probe_tiles)
    calls = {
        "sample_probe_projection_tn": total_probe_tiles,
        "sample_probe_projection_nn": total_probe_tiles,
        "source_tn": total_probe_tiles * variant_blocks * context_tiles,
        "full_target_nn": (
            total_probe_tiles
            * variant_blocks
            * annotation_tiles
            * context_tiles
        ),
        "action_projection_tn": total_probe_tiles * action_tiles,
        "action_projection_nn": total_probe_tiles * action_tiles,
        "action_gram_tn": len(resident_widths),
    }
    passes = {
        "total_descriptor_passes": 2 * len(resident_widths),
        "total_decoded_blocks": 2 * len(resident_widths) * variant_blocks,
        "total_variant_record_visits": 2 * len(resident_widths) * n_variants,
    }
    return calls, passes


def _assert_exact_ledger(
    preflight: Mapping[str, Any], case: StreamedCase, **tiles: int
) -> None:
    expected_calls, expected_passes = _expected_stage2_ledger(case, **tiles)
    observed_calls = {
        str(key): int(value)
        for key, value in dict(preflight["semantic_call_ledger"]).items()
    }
    for operation, count in expected_calls.items():
        assert observed_calls[operation] == count, operation
    assert sum(observed_calls.values()) == sum(expected_calls.values())
    for name, value in expected_passes.items():
        assert int(preflight[name]) == value, name


@pytest.mark.parametrize(
    ("policy", "value"),
    [
        ("variant_block", 1),
        ("sample_probe_resident", 1),
        ("sample_probe_tile", 1),
        ("action_tile", 1),
        ("annotation_tile", 1),
        ("context_tile", 1),
    ],
)
def test_each_streaming_tile_is_independently_scientific_and_ledger_invariant(
    tmp_path: Path, policy: str, value: int
) -> None:
    case = _make_case(tmp_path, q_count=3, name=f"tile-{policy}")
    baseline_result, baseline_snapshot = _run_with_snapshot(_executor(case))
    executor = _executor(case, **{policy: value})
    preflight = dict(executor.preflight())
    tiled_result, tiled_snapshot = _run_with_snapshot(executor)
    _assert_exact_ledger(preflight, case, **{policy: value})
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(tiled_snapshot[name], baseline_snapshot[name], f"{policy} {name}")
    for name in ("gram", "raw_gram_numerator", "annotation_masses"):
        _assert_close(tiled_result[name], baseline_result[name], f"{policy} {name}")
    diagnostics = dict(tiled_result["diagnostics"])
    assert dict(diagnostics["protected_call_counts"]) == dict(
        preflight["semantic_call_ledger"]
    )
    assert bool(diagnostics["descriptor_accounting_verified"])


def test_all_one_tiles_and_decode_threads_are_invariant(tmp_path: Path) -> None:
    case = _make_case(tmp_path, q_count=4, name="all-one")
    baseline_result, baseline_snapshot = _run_with_snapshot(_executor(case))
    policies = {
        "variant_block": 1,
        "sample_probe_resident": 1,
        "sample_probe_tile": 1,
        "action_tile": 1,
        "annotation_tile": 1,
        "context_tile": 1,
    }
    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2:
        policies["decode_threads"] = 2
    executor = _executor(case, **policies)
    preflight = dict(executor.preflight())
    tiled_result, tiled_snapshot = _run_with_snapshot(executor)
    _assert_exact_ledger(
        preflight,
        case,
        **{key: value for key, value in policies.items() if key != "decode_threads"},
    )
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(tiled_snapshot[name], baseline_snapshot[name], name)
    for name in ("gram", "raw_gram_numerator"):
        _assert_close(tiled_result[name], baseline_result[name], name)


def test_source_output_scratch_handles_n_less_than_v_with_one_annotation(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=3,
        retained_sample_count=7,
        annotation_count=1,
        name="source-scratch-n-less-than-v",
    )
    n_samples, n_variants = case.scaled_genotype.shape
    assert n_samples < n_variants
    assert case.annotations.shape[1] == 1

    executor = _executor(
        case,
        variant_block=n_variants,
        sample_probe_resident=case.sample_probes.shape[1],
        sample_probe_tile=case.sample_probes.shape[1],
        annotation_tile=1,
        context_tile=case.phi.shape[1],
    )
    preflight = dict(executor.preflight())
    result, snapshot = _run_with_snapshot(executor)
    _assert_science(result, snapshot, case)
    _assert_exact_ledger(
        preflight,
        case,
        variant_block=n_variants,
        sample_probe_resident=case.sample_probes.shape[1],
        sample_probe_tile=case.sample_probes.shape[1],
        annotation_tile=1,
        context_tile=case.phi.shape[1],
    )
    _assert_execution_diagnostics(result, preflight)


def test_strict_disjoint_route_matches_generic_binary_and_eliminates_zero_work(
    tmp_path: Path,
) -> None:
    strict = _make_case(
        tmp_path,
        q_count=3,
        annotation_mode="strict_disjoint_binary_v1",
        name="strict",
    )
    generic = replace(strict, annotation_mode="generic_nonnegative_weights_v1")
    strict_result, strict_snapshot = _run_with_snapshot(_executor(strict))
    generic_result, generic_snapshot = _run_with_snapshot(_executor(generic))
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(strict_snapshot[name], generic_snapshot[name], name)
    for name in ("gram", "raw_gram_numerator", "annotation_masses"):
        _assert_close(strict_result[name], generic_result[name], name)
    strict_diagnostics = dict(strict_result["diagnostics"])
    generic_diagnostics = dict(generic_result["diagnostics"])
    assert bool(strict_diagnostics["strict_disjoint_optimized_path"])
    assert not bool(generic_diagnostics["strict_disjoint_optimized_path"])
    assert int(strict_diagnostics["strict_zero_weight_columns_eliminated"]) > 0
    assert int(generic_diagnostics["strict_zero_weight_columns_eliminated"]) == 0


def test_explicit_and_native_philox_probes_match_and_are_schedule_invariant(
    tmp_path: Path,
) -> None:
    base = _make_case(tmp_path, q_count=3, name="philox")
    keys, probes = _philox_probes(base.fixed_basis.shape[0], 7)
    explicit = replace(base, sample_probes=probes)
    explicit_result, explicit_snapshot = _run_with_snapshot(_executor(explicit))

    counter_options = {
        "sample_probes": None,
        "philox_keys": keys,
        "sample_probe_count": probes.shape[1],
    }
    counter_result, counter_snapshot = _run_with_snapshot(
        _executor(base, **counter_options)
    )
    tiled_result, tiled_snapshot = _run_with_snapshot(
        _executor(
            base,
            **counter_options,
            variant_block=1,
            sample_probe_resident=2,
            sample_probe_tile=1,
            action_tile=1,
            annotation_tile=1,
            context_tile=1,
            decode_threads=(
                2
                if hasattr(os, "sched_getaffinity")
                and len(os.sched_getaffinity(0)) >= 2
                else 1
            ),
        )
    )
    assert counter_result["probe_policy"] == "numpy_philox_per_probe_key_v1"
    for observed in (counter_snapshot, tiled_snapshot):
        for name in ("source_scores", "raw_actions", "normalized_actions"):
            _assert_close(observed[name], explicit_snapshot[name], f"Philox {name}")
    for observed in (counter_result, tiled_result):
        for name in ("gram", "raw_gram_numerator"):
            _assert_close(observed[name], explicit_result[name], f"Philox {name}")


def test_philox_probe_and_science_are_fresh_process_invariant(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "local"
    child_root = tmp_path / "child"
    local_root.mkdir()
    child_root.mkdir()
    case = _make_case(local_root, q_count=3, name="process")
    keys, probes = _philox_probes(case.fixed_basis.shape[0], 7)
    options = {
        "sample_probes": None,
        "philox_keys": keys,
        "sample_probe_count": probes.shape[1],
        "variant_block": 3,
        "sample_probe_resident": 3,
        "sample_probe_tile": 2,
        "action_tile": 2,
        "annotation_tile": 1,
        "context_tile": 2,
    }
    local_result, local_snapshot = _run_with_snapshot(_executor(case, **options))

    code = r"""
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import test_context_stage2_streamed_reference as helper

case = helper._make_case(Path(sys.argv[2]), q_count=3, name="process")
keys, probes = helper._philox_probes(case.fixed_basis.shape[0], 7)
executor = helper._executor(
    case,
    sample_probes=None,
    philox_keys=keys,
    sample_probe_count=probes.shape[1],
    variant_block=1,
    sample_probe_resident=2,
    sample_probe_tile=1,
    action_tile=1,
    annotation_tile=1,
    context_tile=1,
)
result, snapshot = helper._run_with_snapshot(executor)
payload = {
    "gram": result["gram"].tolist(),
    "raw_gram_numerator": result["raw_gram_numerator"].tolist(),
    "source_scores": snapshot["source_scores"].tolist(),
    "raw_actions": snapshot["raw_actions"].tolist(),
    "normalized_actions": snapshot["normalized_actions"].tolist(),
    "probe_identity_sha256": result["probe_identity_sha256"],
}
print("STAGE2_PROCESS_RESULT=" + json.dumps(payload, sort_keys=True))
"""
    environment = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).parent),
            str(child_root),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("STAGE2_PROCESS_RESULT=")
    ]
    assert len(lines) == 1, completed.stdout + completed.stderr
    child = json.loads(lines[0].split("=", 1)[1])
    assert child["probe_identity_sha256"] == local_result["probe_identity_sha256"]
    for name in ("gram", "raw_gram_numerator"):
        _assert_close(child[name], local_result[name], f"process {name}")
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(child[name], local_snapshot[name], f"process {name}")


def _mutate_file(prefix: Path, extension: str) -> None:
    path = Path(str(prefix) + extension)
    with path.open("r+b") as handle:
        offset = 3 if extension == ".bed" else 0
        handle.seek(offset)
        value = handle.read(1)
        if not value:
            raise AssertionError(f"cannot mutate empty {path}")
        handle.seek(offset)
        handle.write(bytes([value[0] ^ (0x03 if extension == ".bed" else 0x01)]))
        handle.flush()
        os.fsync(handle.fileno())


def _swap_bed_calls(
    prefix: Path, *, n_total: int, variant: int, row_a: int, row_b: int
) -> None:
    path = Path(str(prefix) + ".bed")
    stride = (n_total + 3) // 4
    data = bytearray(path.read_bytes())

    def code(row: int) -> int:
        position = 3 + variant * stride + row // 4
        return (data[position] >> (2 * (row % 4))) & 0x03

    code_a, code_b = code(row_a), code(row_b)
    assert code_a != code_b
    for row, value in ((row_a, code_b), (row_b, code_a)):
        position = 3 + variant * stride + row // 4
        shift = 2 * (row % 4)
        data[position] = (data[position] & ~(0x03 << shift)) | (value << shift)
    path.write_bytes(data)


def _run_checkpoint_failure(
    executor: Any,
    checkpoint: str,
    mutate: Any | None = None,
) -> BaseException:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = executor.run()
        except BaseException as exc:  # deliberately transfer worker failure
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    reached = str(executor.wait_for_test_checkpoint(5000))
    try:
        assert reached == checkpoint
        if mutate is not None:
            mutate()
    finally:
        executor.release_test_checkpoint()
    worker.join(timeout=10)
    assert not worker.is_alive(), "checkpoint worker failed to terminate"
    assert "result" not in outcome
    error = outcome.get("error")
    assert isinstance(error, BaseException)
    return error


def test_executor_owns_descriptors_and_caller_arrays_and_is_one_shot(
    tmp_path: Path,
) -> None:
    baseline = _make_case(tmp_path, name="owned-baseline")
    baseline_result, baseline_snapshot = _run_with_snapshot(_executor(baseline))

    case = _make_case(tmp_path, name="owned-mutated-callers")
    executor = _executor(case)
    for value in (
        case.retained_sample_rows,
        case.retained_variant_rows,
        case.affine_mean,
        case.affine_inverse_scale,
        case.fixed_basis,
        case.phi,
        case.annotations,
        case.group_index,
        case.sample_probes,
    ):
        value[...] = 0
    result, snapshot = _run_with_snapshot(executor)
    for name in ("gram", "raw_gram_numerator", "annotation_masses"):
        _assert_close(result[name], baseline_result[name], f"caller copy {name}")
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(snapshot[name], baseline_snapshot[name], f"caller copy {name}")
    with pytest.raises(RuntimeError, match="one-shot|lifecycle"):
        executor.run()

    assert not np.asarray(snapshot["source_scores"]).flags.writeable
    second_snapshot = dict(executor.differential_snapshot())
    _assert_close(
        second_snapshot["source_scores"],
        baseline_snapshot["source_scores"],
        "defensive differential snapshot",
    )


@pytest.mark.parametrize(
    "checkpoint",
    [
        "post_seal",
        "post_admission",
        "post_source",
        "post_action",
        "pre_finalization",
        "pre_publication",
    ],
)
def test_real_bed_mutation_at_every_phase_checkpoint_fails_closed(
    tmp_path: Path, checkpoint: str
) -> None:
    case = _make_case(tmp_path, name=f"bed-mutation-{checkpoint}")
    executor = _executor(case, test_checkpoint=checkpoint)
    error = _run_checkpoint_failure(
        executor, checkpoint, lambda: _mutate_file(case.prefix, ".bed")
    )
    assert "mutation" in str(error).lower() or "changed" in str(error).lower()


@pytest.mark.parametrize(
    "checkpoint",
    [
        "post_seal",
        "post_admission",
        "post_source",
        "post_action",
        "pre_finalization",
        "pre_publication",
    ],
)
@pytest.mark.parametrize("extension", [".bim", ".fam"])
def test_real_bim_fam_mutation_at_every_phase_checkpoint_fails_closed(
    tmp_path: Path, extension: str, checkpoint: str
) -> None:
    case = _make_case(
        tmp_path,
        name=f"metadata-mutation-{extension[1:]}-{checkpoint}",
    )
    executor = _executor(case, test_checkpoint=checkpoint)
    error = _run_checkpoint_failure(
        executor, checkpoint, lambda: _mutate_file(case.prefix, extension)
    )
    assert "mutation" in str(error).lower() or "changed" in str(error).lower()


@pytest.mark.parametrize(
    "checkpoint",
    [
        "post_seal",
        "post_admission",
        "post_source",
        "post_action",
        "pre_finalization",
        "pre_publication",
    ],
)
def test_owned_array_mutation_at_every_checkpoint_fails_closed(
    tmp_path: Path, checkpoint: str
) -> None:
    case = _make_case(tmp_path, name=f"owned-checkpoint-{checkpoint}")
    executor = _executor(
        case,
        test_checkpoint=checkpoint,
        test_mutation_target="context_basis",
    )
    error = _run_checkpoint_failure(executor, checkpoint)
    assert "mutation" in str(error).lower()


@pytest.mark.parametrize(
    "target",
    [
        "fixed_basis",
        "context_basis",
        "annotations",
        "affine_mean",
        "affine_inverse_scale",
        "sample_probes",
        "counted_allele_flags",
        "variant_metadata",
        "expected_missing_counts",
        "retained_samples",
        "retained_variants",
        "groups",
        "annotation_names",
        "group_names",
        "pair_component_maps",
        "output_buffers",
    ],
)
def test_each_owned_plan_input_is_fingerprinted(
    tmp_path: Path, target: str
) -> None:
    case = _make_case(tmp_path, name=f"owned-target-{target}")
    executor = _executor(
        case,
        test_checkpoint="pre_publication",
        test_mutation_target=target,
    )
    error = _run_checkpoint_failure(executor, "pre_publication")
    assert "mutation" in str(error).lower()


def test_owned_philox_key_mutation_is_fingerprinted(tmp_path: Path) -> None:
    case = _make_case(tmp_path, name="owned-philox-keys")
    keys, probes = _philox_probes(case.fixed_basis.shape[0], 5)
    executor = _executor(
        case,
        sample_probes=None,
        philox_keys=keys,
        sample_probe_count=probes.shape[1],
        test_checkpoint="pre_publication",
        test_mutation_target="probe_keys",
    )
    error = _run_checkpoint_failure(executor, "pre_publication")
    assert "mutation" in str(error).lower()


@pytest.mark.parametrize(
    ("changes", "label"),
    [
        ({"retained_variant_order_sha256": "0" * 64}, "order"),
        ({"affine_mean_sha256": "0" * 64}, "mean"),
        ({"affine_inverse_scale_sha256": "0" * 64}, "scale"),
        ({"scale_plan_sha256": "0" * 64}, "scale"),
    ],
)
def test_declared_order_and_scale_digest_failures_reject_before_run(
    tmp_path: Path, changes: Mapping[str, Any], label: str
) -> None:
    case = _make_case(tmp_path, name=f"bad-digest-{label}-{len(changes)}")
    with pytest.raises((RuntimeError, ValueError), match=f"(?i){label}|digest|sha256"):
        _executor(case, **changes)


def test_variant_order_and_allele_metadata_mismatch_rejects(tmp_path: Path) -> None:
    case = _make_case(tmp_path, name="bad-map")
    wrong_ids = list(case.expected_variant_ids)
    wrong_ids[0], wrong_ids[1] = wrong_ids[1], wrong_ids[0]
    with pytest.raises((RuntimeError, ValueError), match="(?i)variant|order|BIM"):
        _executor(case, expected_variant_ids=wrong_ids)

    wrong_alleles = list(case.expected_counted_alleles)
    wrong_alleles[0] = "N"
    with pytest.raises((RuntimeError, ValueError), match="(?i)allele|BIM"):
        _executor(case, counted_alleles=wrong_alleles)


def test_scale_array_content_must_match_its_digest(tmp_path: Path) -> None:
    case = _make_case(tmp_path, name="bad-scale-content")
    changed = case.affine_mean.copy()
    changed[0] += 0.125
    with pytest.raises((RuntimeError, ValueError), match="(?i)mean|digest|sha256"):
        _executor(case, affine_mean=changed)


def test_missing_count_and_position_digest_fail_closed(tmp_path: Path) -> None:
    count_case = _make_case(tmp_path, name="bad-missing-count")
    wrong_counts = _expected_missing_counts(count_case)
    wrong_counts[0] += 1
    with pytest.raises((RuntimeError, ValueError), match="(?i)missing"):
        _executor(count_case, expected_missing_counts=wrong_counts).run()

    position_case = _make_case(tmp_path, name="bad-missing-position")
    # Variant 3 has one missing retained call at total row 2.  Swap that BED
    # code with retained row 3, preserving the per-variant count while changing
    # the exact retained logical missingness identity.
    _swap_bed_calls(
        position_case.prefix,
        n_total=position_case.raw_a1.shape[0],
        variant=3,
        row_a=2,
        row_b=3,
    )
    with pytest.raises((RuntimeError, ValueError), match="(?i)missing|digest|sha256"):
        _executor(position_case).run()


def test_exact_and_cap_minus_one_workspace_and_telemetry_admission(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, name="capacity")
    preflight = dict(_executor(case).preflight())
    workspace = int(preflight["required_workspace_bytes"])
    telemetry = int(preflight["required_telemetry_capacity"])
    assert workspace > 0
    assert telemetry > 0
    categories = {
        name: int(preflight[name])
        for name in (
            "permanent_bytes",
            "source_phase_bytes",
            "action_phase_bytes",
            "compact_output_bytes",
            "integrity_reserve_bytes",
            "telemetry_bytes",
        )
    }
    assert workspace == (
        categories["permanent_bytes"]
        + max(categories["source_phase_bytes"], categories["action_phase_bytes"])
        + categories["compact_output_bytes"]
        + categories["integrity_reserve_bytes"]
        + categories["telemetry_bytes"]
    )
    memory_ledger = {
        str(key): int(value)
        for key, value in dict(preflight["memory_ledger"]).items()
    }
    assert memory_ledger == categories

    exact = _executor(
        case,
        workspace_cap_bytes=workspace,
        telemetry_capacity=telemetry,
    )
    result, snapshot = _run_with_snapshot(exact)
    _assert_science(result, snapshot, case)
    _assert_execution_diagnostics(result, preflight)
    with pytest.raises(RuntimeError, match="admission: workspace cap"):
        _executor(case, workspace_cap_bytes=workspace - 1)
    with pytest.raises(RuntimeError, match="admission: telemetry capacity"):
        _executor(case, telemetry_capacity=telemetry - 1)


def test_compact_result_is_explicitly_incomplete_and_snapshot_is_nonproduction(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=4, name="compact")
    executor = _executor(case)
    result, snapshot = _run_with_snapshot(executor)
    n_samples, n_variants = case.scaled_genotype.shape
    assert result["complete_reference_artifact"] is False
    assert result["internal_result_kind"] == "stage2_source_action_full_gram_v1"
    for forbidden in (
        "source_scores",
        "raw_actions",
        "normalized_actions",
        "same_person",
        "group_gram_numerator",
        "deletion",
        "artifact_family",
        "logical_schema_version",
        "canonical_manifest_json",
    ):
        assert forbidden not in result
    for path, value in _walk_arrays(result):
        assert n_samples not in value.shape, f"compact result leaks N at {path}"
        assert n_variants not in value.shape, f"compact result leaks M at {path}"
        assert value.flags.owndata, path
        assert not value.flags.writeable, path
    assert np.asarray(snapshot["source_scores"]).shape == (
        n_variants,
        case.phi.shape[1],
        case.sample_probes.shape[1],
    )
    assert np.asarray(snapshot["raw_actions"]).shape == (
        n_samples,
        len(_components(case)),
        case.sample_probes.shape[1],
    )
    for path, value in _walk_arrays(snapshot, "snapshot"):
        assert value.flags.owndata, path
        assert not value.flags.writeable, path
    ownership = dict(result["output_ownership"])
    assert ownership == {"owns_data": True, "read_only": True}
    numa = dict(result["numa"])
    assert numa["policy"] == "unbound_first_touch_v1"
    assert int(numa["output_numa_node"]) == -1
    assert not bool(numa["numa_applicable"])
    assert not bool(numa["numa_verified"])
    assert str(numa["reason"])
    assert dict(executor.info())["lifecycle"] == "stage2_incomplete_ready"

    without_snapshot = _executor(case, enable_differential_snapshot=False)
    compact = dict(without_snapshot.run())
    assert compact["complete_reference_artifact"] is False
    with pytest.raises(RuntimeError, match="differential|snapshot|disabled"):
        without_snapshot.differential_snapshot()


def test_contextual_streamed_source_uses_only_the_approved_reuse_boundary() -> None:
    assert tuple(gxeldcore.contextual_streamed_semantic_operations_v1()) == (
        STAGE2_OPERATIONS
    )
    source = CONTEXTUAL_STREAMED_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "dgemm_nn_raw(",
        "dgemm_tn_raw(",
        "cblas_dgemm(",
        "DirectContext::",
        "MultiEnvironmentKernel(",
        "MultiEnvironmentDirectContext(",
        "ContextualBlockExecutorV1(",
        "scale_x",
        "scale_w",
    ):
        assert forbidden not in source
    for required in (
        "duplicate_cloexec(",
        "validate_regular_fd(",
        "protected_nn(",
        "protected_tn(",
        "source_phase(",
        "action_phase(",
        "gram_phase(",
    ):
        assert required in source


def _semantic_anchor(case: StreamedCase, operation: str) -> str:
    anchors = [dict(item) for item in _executor(case).semantic_anchors()]
    matches = [
        str(item["semantic_anchor"])
        for item in anchors
        if str(item["operation"]) == operation
    ]
    assert len(matches) == 1
    anchor = matches[0]
    assert anchor.startswith(operation + "|")
    return anchor


@pytest.mark.parametrize("operation", STAGE2_OPERATIONS)
def test_stable_coordinate_one_shot_corruption_repairs_every_stage2_operation(
    tmp_path: Path, operation: str
) -> None:
    case = _make_case(tmp_path, q_count=2, name=f"fault-{operation}")
    anchor = _semantic_anchor(case, operation)
    executor = _executor(
        case,
        fault_operation=operation,
        fault_mode="one_shot",
        fault_semantic_anchor=anchor,
    )
    result, snapshot = _run_with_snapshot(executor)
    _assert_science(result, snapshot, case)
    telemetry = dict(result["telemetry"])
    assert int(telemetry["injection_count"]) == 1
    assert int(telemetry["retry_count"]) >= 1
    assert telemetry["fault_semantic_anchor"] == anchor
    injections = [
        dict(event)
        for event in telemetry["events"]
        if dict(event)["event_class"] == "fault_injection"
    ]
    assert len(injections) == 1
    assert injections[0]["operation"] == operation
    assert int(injections[0]["attempt"]) == 1


@pytest.mark.parametrize("operation", STAGE2_OPERATIONS)
@pytest.mark.parametrize(
    "mode",
    [
        "repeated",
        "repair_corruption",
        "force_fallback",
        "nan",
        "inf",
        "canary",
    ],
)
def test_detected_fault_modes_recover_with_admitted_buffers(
    tmp_path: Path, mode: str, operation: str
) -> None:
    case = _make_case(tmp_path, q_count=1, name=f"recover-{operation}-{mode}")
    anchor = _semantic_anchor(case, operation)
    result, snapshot = _run_with_snapshot(
        _executor(
            case,
            fault_operation=operation,
            fault_mode=mode,
            fault_semantic_anchor=anchor,
        )
    )
    _assert_science(result, snapshot, case)
    telemetry = dict(result["telemetry"])
    assert int(telemetry["injection_count"]) == 1
    if mode in {"repeated", "repair_corruption", "force_fallback"}:
        assert int(telemetry["fallback_count"]) == 1
    if mode == "repair_corruption":
        assert int(telemetry["retry_count"]) == 1
        assert int(telemetry["repair_count"]) == 0
    assert bool(telemetry["complete_without_drop"])


@pytest.mark.parametrize("operation", STAGE2_OPERATIONS)
@pytest.mark.parametrize(
    "mode",
    [
        "fallback_corruption",
        "fallback_failure",
        "operand_mutation",
        "runtime_mutation",
    ],
)
def test_terminal_integrity_faults_fail_closed_without_result(
    tmp_path: Path, mode: str, operation: str
) -> None:
    case = _make_case(tmp_path, q_count=1, name=f"terminal-{operation}-{mode}")
    anchor = _semantic_anchor(case, operation)
    executor = _executor(
        case,
        fault_operation=operation,
        fault_mode=mode,
        fault_semantic_anchor=anchor,
    )
    with pytest.raises(RuntimeError, match="(?i)integrity|mutation|fallback|repair"):
        executor.run()
    assert dict(executor.info())["lifecycle"] == "failed"
    with pytest.raises(RuntimeError, match="(?i)failed|snapshot|complete"):
        executor.differential_snapshot()


def test_fault_selector_targets_same_point_under_full_and_all_one_tiles(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=3, name="fault-tiling")
    operation = "full_target_nn"
    anchor = _semantic_anchor(case, operation)
    common = {
        "fault_operation": operation,
        "fault_mode": "one_shot",
        "fault_semantic_anchor": anchor,
    }
    full_result, full_snapshot = _run_with_snapshot(_executor(case, **common))
    tiled_result, tiled_snapshot = _run_with_snapshot(
        _executor(
            case,
            **common,
            variant_block=1,
            sample_probe_resident=1,
            sample_probe_tile=1,
            action_tile=1,
            annotation_tile=1,
            context_tile=1,
        )
    )
    for result in (full_result, tiled_result):
        telemetry = dict(result["telemetry"])
        assert int(telemetry["injection_count"]) == 1
        assert telemetry["fault_semantic_anchor"] == anchor
    for name in ("gram", "raw_gram_numerator"):
        _assert_close(tiled_result[name], full_result[name], name)
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(tiled_snapshot[name], full_snapshot[name], name)


def test_strict_target_fault_anchor_tracks_variant_annotation_membership(
    tmp_path: Path,
) -> None:
    base = _make_case(
        tmp_path,
        q_count=2,
        annotation_mode="strict_disjoint_binary_v1",
        name="strict-fault-membership",
    )
    case = replace(
        base,
        annotations=np.asfortranarray(base.annotations[:, ::-1]),
    )
    assert case.annotations[0, 1] == 1.0
    operation = "full_target_nn"
    anchor = (
        f"{operation}|resident=0|probe=0|variant=0|annotation=1|"
        "context=0|action=0"
    )
    common = {
        "fault_operation": operation,
        "fault_mode": "one_shot",
        "fault_semantic_anchor": anchor,
    }
    full_result, full_snapshot = _run_with_snapshot(_executor(case, **common))
    tiled_result, tiled_snapshot = _run_with_snapshot(
        _executor(
            case,
            **common,
            variant_block=1,
            sample_probe_resident=1,
            sample_probe_tile=1,
            action_tile=1,
            annotation_tile=1,
            context_tile=1,
        )
    )
    for result in (full_result, tiled_result):
        telemetry = dict(result["telemetry"])
        assert int(telemetry["injection_count"]) == 1
        assert telemetry["fault_semantic_anchor"] == anchor
    for name in ("gram", "raw_gram_numerator"):
        _assert_close(tiled_result[name], full_result[name], name)
    for name in ("source_scores", "raw_actions", "normalized_actions"):
        _assert_close(tiled_snapshot[name], full_snapshot[name], name)


def test_recovery_path_uses_exact_admitted_workspace_and_telemetry(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="recovery-capacity")
    operation = "source_tn"
    fault = {
        "fault_operation": operation,
        "fault_mode": "repeated",
        "fault_semantic_anchor": _semantic_anchor(case, operation),
    }
    preflight = dict(_executor(case, **fault).preflight())
    workspace = int(preflight["required_workspace_bytes"])
    telemetry = int(preflight["required_telemetry_capacity"])
    exact_result, exact_snapshot = _run_with_snapshot(
        _executor(
            case,
            **fault,
            workspace_cap_bytes=workspace,
            telemetry_capacity=telemetry,
        )
    )
    _assert_science(exact_result, exact_snapshot, case)
    observed = dict(exact_result["telemetry"])
    assert int(observed["fallback_count"]) == 1
    assert int(observed["observed_events"]) <= telemetry
    assert bool(observed["complete_without_drop"])


def test_unknown_or_unmatched_semantic_fault_selector_fails_closed(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="bad-fault-selector")
    with pytest.raises((RuntimeError, ValueError), match="(?i)fault|operation"):
        _executor(
            case,
            fault_operation="same_person_target_nn",
            fault_mode="one_shot",
            fault_semantic_anchor=(
                "same_person_target_nn|resident=0|probe=0|variant=0|"
                "annotation=0|context=0|action=0"
            ),
        )
    with pytest.raises(RuntimeError, match="(?i)fault|anchor|consumed"):
        _executor(
            case,
            fault_operation="source_tn",
            fault_mode="one_shot",
            fault_semantic_anchor=(
                "source_tn|resident=999|probe=999|variant=999|"
                "annotation=999|context=999|action=999"
            ),
        ).run()


def test_semantic_fault_requires_explicit_stable_anchor(tmp_path: Path) -> None:
    case = _make_case(tmp_path, q_count=2, name="missing-fault-anchor")
    with pytest.raises(RuntimeError, match="semantic point anchor"):
        _executor(
            case,
            fault_operation="source_tn",
            fault_mode="one_shot",
            fault_semantic_anchor="",
        )
