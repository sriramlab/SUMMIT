from __future__ import annotations

import os

import numpy as np
import pytest

from summit.ldscore.generalized_gxe_reference_v1 import (
    GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
    build_generalized_gxe_variant_reference_v1,
    serialize_generalized_gxe_inference_axes,
)
from summit.ldscore.generalized_gxe_variant import (
    GENERALIZED_GXE_VARIANT_PROBE_ALGORITHM,
    GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
    GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    TwoPassLedger,
    generate_global_variant_probes,
    native_global_variant_probes,
    plan_generalized_gxe_variant_work,
)


def _completed_ledger(num_variants: int) -> dict[str, int]:
    ledger = TwoPassLedger(num_variants)
    for pass_number in (1, 2):
        ledger.begin_pass(pass_number)
        ledger.record_block(0, 2)
        ledger.record_block(2, num_variants)
        ledger.finish_pass()
    ledger.validate_clean_completion()
    return ledger.to_dict()


def _performance_ledger() -> dict:
    phases = {name: 0.0 for name in ("pass1", "barrier", "pass2", "finalize")}
    return {
        "backend": "fixture",
        "threads": 1,
        "affinity": {},
        "numa_evidence": {},
        "phase_wall_seconds": dict(phases),
        "phase_cpu_seconds": dict(phases),
        "bytes_read": 0,
        "gemm_dimensions": [],
        "peak_rss_bytes": 0,
        "output_bytes": 0,
    }


def _valid_artifact() -> tuple[dict, dict[str, np.ndarray]]:
    num_variants, num_samples = 5, 7
    basis_names = ("intercept", "environment", "environment_sq")
    annotation_names = ("baseline", "coding")
    annotation_masses = np.asarray([4.0, 5.0])
    block_ids = np.asarray([0, 0, 1, 1, 1])
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=num_variants,
        num_samples=num_samples,
        basis_names=basis_names,
        fixed_effect_rank=2,
        annotation_names=annotation_names,
        annotation_masses=annotation_masses,
        variant_block_ids=block_ids,
        block_labels=("left", "right"),
        residual_component_names=("identity",),
    )
    component_count = len(axes["components"]["table"])
    pair_count = len(axes["pairs"]["table"])
    rng = np.random.default_rng(20260822)
    block_directed = rng.normal(size=(2, component_count, component_count))
    directed = np.sum(block_directed, axis=0)
    symmetric = 0.5 * (directed + directed.T)
    component_annotations = np.asarray(
        [entry[0] for entry in axes["components"]["table"]]
    )
    denominator = (
        annotation_masses[component_annotations, None]
        * annotation_masses[component_annotations][None, :]
    )
    residual_rank = axes["fixed_effects"]["residual_rank"]
    gram = float(residual_rank**2) * symmetric / denominator
    block_masses = np.asarray([[1.5, 2.0], [2.5, 3.0]])
    deleted = np.empty_like(block_directed)
    for block in range(2):
        retained_directed = directed - block_directed[block]
        retained_symmetric = 0.5 * (
            retained_directed + retained_directed.T
        )
        retained_masses = annotation_masses - block_masses[block]
        retained_denominator = (
            retained_masses[component_annotations, None]
            * retained_masses[component_annotations][None, :]
        )
        deleted[block] = (
            float(residual_rank**2)
            * retained_symmetric
            / retained_denominator
        )
    same_seed = rng.normal(size=(component_count, component_count))
    arrays = {
        "directed_numerator": np.asarray(directed, dtype=np.float64),
        "symmetric_numerator": np.asarray(symmetric, dtype=np.float64),
        "genetic_gram": np.asarray(gram, dtype=np.float64),
        "block_directed_numerator": np.asarray(
            block_directed, dtype=np.float64
        ),
        "block_annotation_mass": np.asarray(block_masses, dtype=np.float64),
        "deleted_genetic_gram": np.asarray(deleted, dtype=np.float64),
        "same_person": np.asarray(
            0.5 * (same_seed + same_seed.T), dtype=np.float64
        ),
        "directional_ldscores": np.asarray(
            rng.normal(size=(num_variants, pair_count, component_count)),
            dtype=np.float64,
        ),
    }
    probe_spec = GlobalVariantProbeSpec(
        root_seed=20260822,
        probe_offset=5,
        probe_count=11,
    )
    diagnostics = {
        "maximum_source_projection_leakage": 0.0,
        "maximum_presymmetry_absolute_error": 0.0,
        "maximum_presymmetry_relative_error": 0.0,
        "block_reconstruction_error": 0.0,
        "same_person_probe_count": probe_spec.probe_count,
        "same_person_cross_tile_finalized": True,
        "minimum_annotation_mass": 4.0,
        "minimum_deleted_annotation_mass": 1.5,
        "all_values_finite": True,
        "normal_matrix_rank": component_count,
        "normal_matrix_condition": 2.0,
        "dense_oracle_fixture_version": "stage02_v1",
        "backend_fixed_probe_maximum_error": 0.0,
    }
    artifact = build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe_spec,
        genotype_scale_plan={
            "genotype_scale_policy": "sealed_variant_affine_v1",
            "allele_orientation": "a1_count",
            "allele_coding": "plink_a1_dosage_0_1_2",
            "centering_source": "reference_retained_samples",
            "centering_formula": "observed_mean_after_mean_imputation",
            "scaling_formula": "inverse_sample_sd_ddof1",
            "missing_imputation": "observed_mean",
            "ploidy_policy": "diploid_autosome",
        },
        arrays=arrays,
        pass_ledger=_completed_ledger(num_variants),
        performance_ledger=_performance_ledger(),
        provenance={"fixture": "generalized-gxe-contract-test"},
        diagnostics=diagnostics,
    )
    return dict(artifact.manifest), arrays


def test_canonical_identity_constants_are_distinct() -> None:
    assert (
        GENERALIZED_GXE_VARIANT_REFERENCE_KIND
        == "summit.generalized_gxe.variant_ldscore_reference"
    )
    assert (
        GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT
        == "generalized_gxe_variant_ldscore_v1"
    )
    assert (
        GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD
        == "frozen_full_genome_variant_ldscore_delete_block_v1"
    )
    assert (
        GENERALIZED_GXE_VARIANT_PROBE_ALGORITHM
        == "counter_global_variant_global_probe_v1"
    )
    assert (
        "block_local_ldscore_deletion"
        != GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD
    )


def test_inference_artifact_has_concrete_ordered_axes() -> None:
    manifest, arrays = _valid_artifact()
    assert manifest["axes"]["pairs"]["table"] == [
        [0, 0],
        [1, 1],
        [2, 2],
        [0, 1],
        [0, 2],
        [1, 2],
    ]
    assert manifest["axes"]["components"]["table"][:7] == [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 0],
    ]
    assert arrays["directional_ldscores"].shape[0] == 5

    assert "identity" not in manifest


def test_global_probe_stream_has_frozen_values() -> None:
    variants = np.asarray([0, 1, 7, 19])
    probes = np.asarray([3, 4, 11])
    observed = generate_global_variant_probes(
        variants,
        probes,
        root_seed=20260822,
    )
    expected = np.asarray(
        [
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ]
    )
    np.testing.assert_array_equal(observed, expected)


def test_global_probe_stream_is_invariant_to_blocks_chunks_and_row_order() -> None:
    variants = np.arange(37, dtype=np.int64)
    probes = np.arange(9, 26, dtype=np.int64)
    full = generate_global_variant_probes(variants, probes, root_seed=8817)
    by_variant_blocks = np.vstack(
        [
            generate_global_variant_probes(
                variants[start:stop], probes, root_seed=8817
            )
            for start, stop in ((0, 3), (3, 14), (14, 29), (29, 37))
        ]
    )
    by_probe_chunks = np.column_stack(
        [
            generate_global_variant_probes(
                variants, probes[start:stop], root_seed=8817
            )
            for start, stop in ((0, 1), (1, 8), (8, 13), (13, 17))
        ]
    )
    order = np.asarray([19, 0, 36, 7, 7, 3, 25])
    reordered = generate_global_variant_probes(order, probes, root_seed=8817)
    np.testing.assert_array_equal(by_variant_blocks, full)
    np.testing.assert_array_equal(by_probe_chunks, full)
    np.testing.assert_array_equal(reordered, full[order])


def test_python_and_native_global_probes_match_across_threads() -> None:
    try:
        from summit import gxeldcore
    except ImportError:
        pytest.skip("native extension is unavailable in the source-only test run")

    if not callable(getattr(gxeldcore, "global_variant_rademacher", None)):
        pytest.skip("loaded native extension predates Stage 03 global probes")
    variants = np.asarray([31, 0, 8, 999, 8, 4], dtype=np.int64)
    probes = np.asarray([17, 3, 4097, 18], dtype=np.int64)
    info = dict(gxeldcore.build_info())
    available_threads = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else max(1, os.cpu_count() or 1)
    )
    if info.get("blas_runtime_environment_immutable") is True:
        available_threads = min(
            available_threads, int(info["blas_runtime_threads"])
        )
    thread_counts = tuple(
        threads for threads in (1, 2, 4) if threads <= available_threads
    )
    assert thread_counts
    expected = generate_global_variant_probes(
        variants,
        probes,
        root_seed=2**63 + 41,
    )
    for threads in thread_counts:
        observed = native_global_variant_probes(
            variants,
            probes,
            root_seed=2**63 + 41,
            threads=threads,
            native_module=gxeldcore,
        )
        assert observed.flags.f_contiguous
        np.testing.assert_array_equal(observed, expected)
    assert info["global_variant_probe_supported"] is True
    assert (
        info["global_variant_probe_algorithm"]
        == GENERALIZED_GXE_VARIANT_PROBE_ALGORITHM
    )


@pytest.mark.parametrize(
    ("num_basis", "num_annotations", "num_probes", "expected"),
    (
        (3, 1, 128, (76_800_000_000_000, 691_200_000_000_000)),
        (3, 1, 1024, (614_400_000_000_000, 5_529_600_000_000_000)),
        (4, 8, 128, (614_400_000_000_000, 9_830_400_000_000_000)),
        (4, 8, 1024, (4_915_200_000_000_000, 78_643_200_000_000_000)),
    ),
)
def test_target_planner_work_and_memory_arithmetic(
    num_basis: int,
    num_annotations: int,
    num_probes: int,
    expected: tuple[int, int],
) -> None:
    inputs = GeneralizedGxEPlanInputs(
        num_samples=300_000,
        num_variants=1_000_000,
        num_basis=num_basis,
        num_annotations=num_annotations,
        num_probes=num_probes,
        memory_limit_bytes=1024**4,
        genotype_format="bed",
        threads=64,
    )
    plan = plan_generalized_gxe_variant_work(inputs)
    p = num_basis * (num_basis + 1) // 2
    c = num_annotations * p
    total_rhs = num_annotations * num_basis**2 * num_probes
    assert plan.work["pass1_flops"] == expected[0]
    assert plan.work["pass2_flops"] == expected[1]
    assert plan.work["total_leading_flops"] == sum(expected)
    assert (
        plan.memory["base_sources"]
        == 8 * 300_000 * num_annotations * num_probes
    )
    assert (
        plan.memory["contextual_sources"]
        == 8 * 300_000 * num_annotations * num_basis * num_probes
    )
    assert plan.memory["pass2_rhs"] == 8 * 300_000 * total_rhs
    assert plan.memory["decoded_genotype_block"] == 8 * 300_000 * 4096
    assert plan.memory["cross_sketch_block"] == 8 * 4096 * total_rhs
    assert plan.memory["pair_reduction_scratch"] == 8 * 4096 * p * p
    assert plan.memory["same_person_sample_accumulator"] == 8 * c * 300_000
    assert "block_directed_numerator" not in plan.memory
    assert plan.output_size_bytes == 8 * 1_000_000 * num_annotations * p * p
    assert plan.tiling["rhs_precomputed"] is True
    assert plan.ledger["planned_reference_genotype_passes"] == 2
    assert plan.ledger["planned_retained_variant_visits"] == 2_000_000


def test_constrained_memory_reduces_tiles_without_adding_passes() -> None:
    inputs = GeneralizedGxEPlanInputs(
        num_samples=300_000,
        num_variants=1_000_000,
        num_basis=3,
        num_annotations=1,
        num_probes=128,
        memory_limit_bytes=8 * 1024**3,
        genotype_format="pgen",
        threads=8,
        rhs_policy="auto",
    )
    plan = plan_generalized_gxe_variant_work(inputs)
    assert plan.peak_resident_bytes <= inputs.memory_limit_bytes
    assert plan.tiling["rhs_precomputed"] is False
    assert plan.tiling["variant_block_width"] == 2048
    assert plan.tiling["rhs_tile_columns"] == 128
    assert plan.tiling["rhs_tile_columns"] < plan.tiling["total_rhs_columns"]
    assert plan.descriptor["planned_complete_passes"] == 2
    assert plan.ledger["planned_reference_genotype_passes"] == 2
    assert plan.descriptor["format"] == "pgen"
    assert plan.descriptor["estimated_bytes_per_pass"] is None


def test_planner_rejects_memory_below_fixed_source_state() -> None:
    inputs = GeneralizedGxEPlanInputs(
        num_samples=300_000,
        num_variants=1_000_000,
        num_basis=3,
        num_annotations=1,
        num_probes=128,
        memory_limit_bytes=1 * 1024**3,
        genotype_format="bed",
        threads=8,
        rhs_policy="tiled",
    )
    with pytest.raises(MemoryError, match="fixed global sources"):
        plan_generalized_gxe_variant_work(inputs)


def test_bed_and_pgen_plans_change_metadata_not_work_or_passes() -> None:
    common = dict(
        num_samples=2000,
        num_variants=10_000,
        num_basis=3,
        num_annotations=1,
        num_probes=32,
        memory_limit_bytes=16 * 1024**3,
        threads=4,
    )
    bed = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(**common, genotype_format="bed")
    )
    pgen = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(**common, genotype_format="pgen")
    )
    assert bed.work == pgen.work
    assert bed.peak_resident_bytes == pgen.peak_resident_bytes
    assert bed.descriptor["planned_complete_passes"] == 2
    assert pgen.descriptor["planned_complete_passes"] == 2
    assert bed.descriptor["estimated_bytes_per_pass"] == 5_000_000
    assert pgen.descriptor["estimated_bytes_per_pass"] is None


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_samples": -1},
        {"num_basis": 0},
        {"memory_limit_bytes": 0},
        {"genotype_format": "bgen"},
        {"headroom_fraction": -0.1},
    ),
)
def test_planner_rejects_invalid_inputs(kwargs: dict) -> None:
    values = dict(
        num_samples=100,
        num_variants=200,
        num_basis=3,
        num_annotations=1,
        num_probes=8,
        memory_limit_bytes=1024**3,
        genotype_format="bed",
    )
    values.update(kwargs)
    with pytest.raises((ValueError, OverflowError)):
        GeneralizedGxEPlanInputs(**values)


def test_planner_rejects_signed_64bit_arithmetic_overflow() -> None:
    inputs = GeneralizedGxEPlanInputs(
        num_samples=2**31,
        num_variants=2**31,
        num_basis=3,
        num_annotations=1,
        num_probes=8,
        memory_limit_bytes=2**62,
        genotype_format="bed",
    )
    with pytest.raises(OverflowError, match="signed 64-bit"):
        plan_generalized_gxe_variant_work(inputs)


def test_two_pass_ledger_accepts_clean_execution_and_rejects_duplicates() -> None:
    clean = TwoPassLedger(9)
    for pass_number in (1, 2):
        clean.begin_pass(pass_number)
        for start, stop in ((0, 4), (4, 7), (7, 9)):
            clean.record_block(start, stop)
        clean.finish_pass()
    clean.validate_clean_completion()
    assert clean.to_dict() == {
        "planned_reference_genotype_passes": 2,
        "observed_reference_genotype_passes": 2,
        "planned_retained_variant_visits": 18,
        "observed_retained_variant_visits": 18,
        "duplicate_retained_variant_visits": 0,
        "pass1_decoded_blocks": 3,
        "pass2_decoded_blocks": 3,
        "retry_count": 0,
        "repair_count": 0,
        "fallback_count": 0,
        "integrity_failure_count": 0,
    }

    duplicate = TwoPassLedger(5)
    duplicate.begin_pass(1)
    duplicate.record_block(0, 3)
    duplicate.record_block(2, 5)
    duplicate.finish_pass()
    duplicate.begin_pass(2)
    duplicate.record_block(0, 5)
    duplicate.finish_pass()
    assert duplicate.duplicate_retained_variant_visits == 1
    with pytest.raises(RuntimeError, match="duplicate_retained_variant_visits"):
        duplicate.validate_clean_completion()
