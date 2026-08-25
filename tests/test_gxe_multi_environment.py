from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit import gxeldcore
from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _non_blas_fp64_inner_product,
)
from summit.ldscore import gxe_multi
from summit.ldscore.gxe_multi import (
    combine_multi_environment_reference_batches,
    generate_multi_environment_references,
)
from summit.logger import Logger


def test_final_cross_trace_inner_product_does_not_enter_numpy_dot(monkeypatch):
    left = np.asarray([1.25, -2.0, 3.5, 0.125], dtype=np.float64)
    right = np.asarray([4.0, 0.5, -1.0, 8.0], dtype=np.float64)

    def reject_dot(*_args, **_kwargs):
        raise AssertionError("the final cross-trace diagnostic entered np.dot")

    monkeypatch.setattr(np, "dot", reject_dot)
    observed = _non_blas_fp64_inner_product(left, right)
    assert observed == 1.5
    assert _non_blas_fp64_inner_product(left, right) == observed

    cancellation = np.asarray([1.0e16, 1.0, -1.0e16], dtype=np.float64)
    ones = np.ones(3, dtype=np.float64)
    assert _non_blas_fp64_inner_product(ones, cancellation) == 1.0
    assert _non_blas_fp64_inner_product(ones[::-1], cancellation[::-1]) == 1.0

    with pytest.raises(ValueError, match="equal length"):
        _non_blas_fp64_inner_product(left, right[:-1])


def test_reference_score_text_round_trips_binary64(tmp_path):
    estimator = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    estimator.nsnps = 2
    estimator.snplist = pd.DataFrame(
        {"CHR": [1, 1], "SNP": ["rs1", "rs2"], "BP": [11, 12]}
    )
    estimator.l2cols = ["L2_0"]
    estimator._performance_phase = lambda _name: nullcontext()
    values = np.asarray(
        [[np.nextafter(1.0, 2.0)], [np.nextafter(-1.0, -2.0)]],
        dtype=np.float64,
    )
    reference = tmp_path / "reference.ldscore.gz"
    repeated = tmp_path / "repeated.ldscore.gz"
    candidate = tmp_path / "candidate.ldscore.gz"
    estimator._save_score_file(str(reference), values)
    estimator._save_score_file(str(repeated), values)
    assert reference.read_bytes() == repeated.read_bytes()
    changed = values.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    estimator._save_score_file(str(candidate), changed)
    observed = pd.read_csv(reference, sep="\t", float_precision="round_trip")[
        ["L2_0"]
    ].to_numpy()
    np.testing.assert_array_equal(observed, values)
    changed_observed = pd.read_csv(
        candidate, sep="\t", float_precision="round_trip"
    )[["L2_0"]].to_numpy()
    maximum_absolute_error = float(np.max(np.abs(changed_observed - observed)))
    assert maximum_absolute_error == abs(changed[0, 0] - values[0, 0])
    assert maximum_absolute_error > 0.0


def _inputs(
    tmp_path: Path,
    *,
    missing_second: bool = False,
    missing_genotype: bool = False,
):
    rng = np.random.default_rng(314159)
    n, m = 31, 13
    raw = rng.binomial(2, rng.uniform(0.16, 0.43, size=m), size=(n, m)).astype(float)
    if missing_genotype:
        raw[0, 1] = np.nan
        raw[3, 1] = np.nan
        raw[5, 7] = np.nan
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    env1 = rng.normal(size=n)
    env2 = 0.35 * env1 + rng.normal(size=n)
    if missing_second:
        env2[0] = np.nan
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "age": env1,
            "bmi": env2,
        }
    ).to_csv(environment, sep="\t", index=False, na_rep="NA")
    return prefix, environment, m


def _estimator(
    prefix: Path,
    environment: Path,
    output: Path,
    column: str,
    *,
    num_vectors: int = 17,
    target_xz_mem: float = 0.01,
    rand_dist: str = "rademacher",
):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(output),
        log=Logger(suppress=True),
        rand_dist=rand_dist,
        low_level=None,
        num_vecs=num_vectors,
        step_size=5,
        seed=2718,
        dtype="float64",
        num_threads=2,
        target_xz_mem=target_xz_mem,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend="python",
    )


def _score(prefix: Path, suffix: str) -> np.ndarray:
    frame = pd.read_csv(f"{prefix}.{suffix}.ldscore.gz", sep="\t")
    return frame[["L2_0"]].to_numpy()


def test_shared_multi_environment_matches_independent_references(tmp_path):
    prefix, environment, variants = _inputs(tmp_path)
    independent = {}
    for column in ("age", "bmi"):
        estimator = _estimator(
            prefix, environment, tmp_path / f"independent.{column}", column,
            num_vectors=64,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"independent.{column}"

    estimators = [
        _estimator(
            prefix, environment, tmp_path / f"multi.{column}", column,
            num_vectors=64,
        )
        for column in ("age", "bmi")
    ]
    read_counts = [0, 0]
    for index, estimator in enumerate(estimators):
        original = estimator._read_genotype_block

        def counted(self, start, stop, *, _index=index, _original=original):
            read_counts[_index] += 1
            return _original(start, stop)

        estimator._read_genotype_block = types.MethodType(counted, estimator)
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "multi.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()

    # The descriptor-owned context duplicates the opened PLINK descriptors and
    # performs both native BED passes without materializing a genotype block in
    # Python. Jackknife replicates are formed later from completed score rows.
    assert read_counts == [0, 0]
    payload = json.loads(batch.read_text(encoding="utf-8"))
    assert payload["kind"] == "summit.gxe.multi_environment_reference_batch"
    assert payload["num_environments"] == 2
    assert payload["shared_genotype_passes"] == 2
    assert payload["protected_native_gemm"] is True
    assert (
        payload["execution"]
        == "descriptor_owned_native_multi_environment_end_to_end"
    )
    assert payload["multi_environment_native_end_to_end"] is True
    assert payload["multi_environment_descriptor_owned_end_to_end"] is True
    assert (
        payload["multi_environment_direct_context_schema"]
        == "summit.multi_environment_direct_context.v3"
    )
    assert (
        payload["multi_environment_native_kernel_schema"]
        == "summit.multi_environment_native_kernel.v1"
    )
    assert payload["native_gemm_integrity_enabled"] is bool(
        gxeldcore.build_info()["gemm_integrity_enabled"]
    )
    build_info = dict(gxeldcore.build_info())
    assert payload["native_blas_runtime_isolation"] == build_info[
        "blas_runtime_isolation"
    ]
    assert (
        payload["native_blas_runtime_isolation"] == "private_static"
        or build_info["gemm_integrity_enabled"] is True
    )
    assert payload["repaired_gemm_output_columns"] == 0
    assert payload["checksum_recomputed_gemm_output_columns"] == 0
    assert payload["roundoff_only_gemm_output_columns"] == 0
    assert [item["environment"] for item in payload["references"]] == ["age", "bmi"]
    performance = payload["performance_telemetry"]
    assert performance["multi_environment_native_kernel_required"] is True
    assert performance["multi_environment_native_kernel_complete"] is True
    assert performance["multi_environment_direct_context_required"] is True
    assert performance["multi_environment_direct_context_complete"] is True
    direct_context = performance["multi_environment_direct_context"]
    assert direct_context["descriptor_owned_bed"] is True
    assert direct_context["native_probe_generation"] is True
    assert direct_context["contracted_numa_decode"] is False
    assert direct_context["contracted_packed_source_panel_numa"] is False
    assert direct_context["completed"] is True
    assert direct_context["probe_tile_count"] == 1
    vendor_probe_chunk = build_info[
        "multi_environment_direct_context_max_vendor_probe_chunk"
    ]
    expected_execution_chunks = (
        math.ceil(64 / vendor_probe_chunk) if vendor_probe_chunk > 0 else 1
    )
    expected_execution_width = min(64, vendor_probe_chunk or 64)
    assert (
        direct_context["execution_probe_chunk_count"]
        == expected_execution_chunks
    )
    assert (
        direct_context["maximum_execution_probe_chunk_width"]
        == expected_execution_width
    )
    assert direct_context["fused_two_pass_execution"] is True
    assert direct_context["execution_kernel"] == "dense_private_blas_streamed_pair"
    assert direct_context["mailman_maximum_probe_count"] == 10
    assert direct_context["mailman_probe_count_eligible"] is False
    assert direct_context["packed_genotype_mailman"] is False
    assert direct_context["packed_feature_moments"] is False
    assert direct_context["packed_source_direct_accumulation"] is False
    assert direct_context["virtual_environment_weighted_target_rhs"] is False
    assert direct_context["materialized_target_rhs"] is True
    assert direct_context["dense_genotype_feature_decode"] is True
    assert direct_context["dense_genotype_target_decode"] is True
    assert direct_context["source_panel_released_before_target"] is False
    assert (
        direct_context["source_panel_reused_as_protected_pair_first_half"]
        is False
    )
    native_kernel = performance["multi_environment_native_kernel"]
    assert native_kernel["schema"] == "summit.multi_environment_native_kernel.v1"
    assert native_kernel["environment_count"] == 2
    assert native_kernel["feature_calls"] == math.ceil(variants / 5)
    assert native_kernel["packed_feature_calls"] == 0
    assert native_kernel["source_calls"] == (
        expected_execution_chunks * math.ceil(variants / 5)
    )
    assert native_kernel["projection_calls"] == expected_execution_chunks
    assert native_kernel["target_calls"] == (
        expected_execution_chunks * math.ceil(variants / 5)
    )
    assert native_kernel["normalization_calls"] == 1
    assert native_kernel["persistent_scratch_output_allocations"] > 0
    assert native_kernel["persistent_scratch_output_reuses"] > 0
    assert native_kernel["packed_source_calls"] == 0
    assert native_kernel["packed_target_calls"] == 0
    assert native_kernel["source_weight_scratch_capacity_bytes"] == 0
    assert native_kernel["source_annotation_scratch_capacity_bytes"] == 0
    output_evidence = performance["native_gemm_output_numa_records"]
    output_status = performance["native_gemm_output_numa_evidence_status"]
    assert performance["native_gemm_output_numa_contract_supported"] is True
    assert performance["native_gemm_output_numa_contract_required"] is False
    assert performance["native_gemm_output_numa_evidence_available"] is True
    assert performance["native_gemm_output_numa_evidence_complete"] is True
    assert performance["native_gemm_output_numa_record_count"] == len(output_evidence)
    expected_native_outputs = direct_context["planned_output_calls"]
    assert len(output_evidence) == expected_native_outputs
    assert performance["dropped_native_gemm_output_numa_records"] == 0
    assert output_status["buffered_records"] == 0
    assert output_status["captured_records"] == len(output_evidence)
    assert output_status["dropped_records"] == 0
    assert output_status["attempted_calls"] == len(output_evidence)
    assert output_status["verified_calls"] + output_status["legacy_calls"] == len(
        output_evidence
    )
    assert output_status["failed_calls"] == 0
    assert [record["call_id"] for record in output_evidence] == list(
        range(
            output_status["next_call_id"] - len(output_evidence),
            output_status["next_call_id"],
        )
    )

    for column in ("age", "bmi"):
        observed = tmp_path / f"multi.{column}"
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=2e-12,
                atol=2e-12,
            )
        assert not Path(f"{observed}.gxe.jackknife.npz").exists()
        assert not Path(f"{independent[column]}.gxe.jackknife.npz").exists()
        manifest = json.loads(
            Path(f"{observed}.gxe.ref.json").read_text(encoding="utf-8")
        )
        assert "jackknife" not in manifest
        assert "BLOCK" not in pd.read_csv(
            Path(f"{observed}.gxe.diag.tsv.gz"), sep="\t"
        ).columns
        resources = manifest["resource_estimates"]
        assert resources["multi_environment_shared_decode"] == 1
        assert resources["multi_environment_protected_gemm"] == 1
        assert resources["multi_environment_native_end_to_end"] == 1
        assert resources["multi_environment_descriptor_owned_end_to_end"] == 1
        assert (
            resources["multi_environment_native_kernel_schema"]
            == "summit.multi_environment_native_kernel.v1"
        )
        assert resources["multi_environment_count"] == 2
        assert resources["shared_genotype_passes"] == 2
        assert resources["native_repaired_gemm_output_columns"] == 0
        assert resources["native_checksum_recomputed_gemm_output_columns"] == 0
        assert resources["native_roundoff_only_gemm_output_columns"] == 0

def test_dense_direct_context_tiles_without_mailman_and_matches_reference(tmp_path):
    prefix, environment, _ = _inputs(tmp_path, missing_genotype=True)
    independent = {}
    independent_missing = {}
    for column in ("age", "bmi"):
        estimator = _estimator(
            prefix,
            environment,
            tmp_path / f"dense-independent.{column}",
            column,
            num_vectors=11,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"dense-independent.{column}"
        independent_missing[column] = (
            np.array(estimator.genotype_missing_call_count, copy=True),
            np.array(
                estimator.genotype_missing_environment_correlation, copy=True
            ),
        )

    estimators = [
        _estimator(
            prefix,
            environment,
            tmp_path / f"dense-multi.{column}",
            column,
            num_vectors=11,
        )
        for column in ("age", "bmi")
    ]
    # Full two-environment panel liveness is 21,824 bytes. This admits one
    # environment at a time and proves the dense non-fused tile path.
    for estimator in estimators:
        estimator.target_xz_mem = 11_000 / 1024**3
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "dense-multi.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()

    payload = json.loads(batch.read_text(encoding="utf-8"))
    context = payload["performance_telemetry"][
        "multi_environment_direct_context"
    ]
    plan = payload["complete_process_memory_plan"]
    assert plan["direct_kernel_mode"] == "dense_blas_hybrid"
    assert plan["mailman_probe_count_eligible"] is False
    assert context["execution_kernel"] == "dense_private_blas_streamed_pair"
    assert context["packed_genotype_mailman"] is False
    assert context["fused_two_pass_execution"] is False
    assert context["environment_tile_count"] == 2
    assert context["planned_genotype_passes"] == 5
    for column, estimator in zip(("age", "bmi"), estimators, strict=True):
        observed = tmp_path / f"dense-multi.{column}"
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=2e-12,
                atol=2e-12,
            )
        np.testing.assert_array_equal(
            estimator.genotype_missing_call_count,
            independent_missing[column][0],
        )
        np.testing.assert_allclose(
            estimator.genotype_missing_environment_correlation,
            independent_missing[column][1],
            rtol=2e-14,
            atol=2e-14,
        )


def test_descriptor_owned_pipeline_supports_one_environment_and_missing_calls(
    tmp_path,
):
    prefix, environment, _ = _inputs(tmp_path, missing_genotype=True)
    independent = _estimator(
        prefix,
        environment,
        tmp_path / "independent.single",
        "age",
        num_vectors=3,
    )
    unified = _estimator(
        prefix,
        environment,
        tmp_path / "unified.single",
        "age",
        num_vectors=3,
    )
    reads = 0
    original = unified._read_genotype_block

    def counted(self, start, stop):
        nonlocal reads
        reads += 1
        return original(start, stop)

    unified._read_genotype_block = types.MethodType(counted, unified)
    try:
        independent._compute_ldscore()
        batch = generate_multi_environment_references(
            [unified],
            batch_manifest=tmp_path / "unified.single.gxe.multi.json",
            requested_backend="direct",
        )
        np.testing.assert_array_equal(
            unified.genotype_missing_call_count,
            independent.genotype_missing_call_count,
        )
        np.testing.assert_allclose(
            unified.genotype_missing_environment_correlation,
            independent.genotype_missing_environment_correlation,
            rtol=2e-14,
            atol=2e-14,
        )
    finally:
        independent.close()
        unified.close()

    assert reads == 0
    payload = json.loads(batch.read_text(encoding="utf-8"))
    assert payload["num_environments"] == 1
    assert payload["multi_environment_descriptor_owned_end_to_end"] is True
    for suffix in ("gxx", "gxe", "exg", "gee"):
        np.testing.assert_allclose(
            _score(tmp_path / "unified.single", suffix),
            _score(tmp_path / "independent.single", suffix),
            rtol=2e-12,
            atol=2e-12,
        )


def test_direct_pipeline_rejects_probe_distributions_without_native_generation(
    tmp_path,
):
    prefix, environment, _ = _inputs(tmp_path)
    estimator = _estimator(
        prefix,
        environment,
        tmp_path / "gaussian.direct",
        "age",
        num_vectors=3,
        rand_dist="gaussian",
    )
    try:
        with pytest.raises(ValueError, match="requires --rand-dist rademacher"):
            generate_multi_environment_references(
                [estimator],
                batch_manifest=tmp_path / "gaussian.direct.gxe.multi.json",
                requested_backend="direct",
            )
    finally:
        estimator.close()
    assert not (tmp_path / "gaussian.direct.gxe.multi.json").exists()


def test_multi_environment_rejects_different_complete_case_cohorts(tmp_path):
    prefix, environment, _ = _inputs(tmp_path, missing_second=True)
    estimators = [
        _estimator(prefix, environment, tmp_path / f"different.{column}", column)
        for column in ("age", "bmi")
    ]
    try:
        try:
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "different.gxe.multi.json",
            )
        except ValueError as error:
            assert "same complete-case cohort" in str(error)
        else:
            raise AssertionError("different complete-case cohorts were accepted")
    finally:
        for estimator in estimators:
            estimator.close()


def test_shared_multi_environment_no_jackknife_accounts_for_probe_tiles(tmp_path):
    prefix, environment, variants = _inputs(tmp_path)
    # The packed path retains one additive+interaction source panel and forms
    # environment weighting virtually. Under this tiny budget, two complete
    # one-environment panels are pass-optimal.
    target_xz_mem = 6000 / 1024**3
    independent = {}
    for column in ("age", "bmi"):
        estimator = _estimator(
            prefix,
            environment,
            tmp_path / f"independent.nojk.{column}",
            column,
            num_vectors=7,
            target_xz_mem=target_xz_mem,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"independent.nojk.{column}"

    estimators = [
        _estimator(
            prefix,
            environment,
            tmp_path / f"multi.nojk.{column}",
            column,
            num_vectors=7,
            target_xz_mem=target_xz_mem,
        )
        for column in ("age", "bmi")
    ]
    reads = 0
    original = estimators[0]._read_genotype_block

    def counted(self, start, stop):
        nonlocal reads
        reads += 1
        return original(start, stop)

    estimators[0]._read_genotype_block = types.MethodType(counted, estimators[0])
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "multi.nojk.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()

    payload = json.loads(batch.read_text(encoding="utf-8"))
    assert payload["environment_tiles"] == [[0, 1], [1, 2]]
    assert payload["randomization"]["probe_tiles"] == [[0, 7]]
    assert payload["shared_genotype_passes"] == 5
    assert reads == 0
    for column in ("age", "bmi"):
        observed = tmp_path / f"multi.nojk.{column}"
        assert not Path(f"{observed}.gxe.jackknife.npz").exists()
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=2e-12,
                atol=2e-12,
            )


def test_complete_memory_planner_charges_both_source_families_and_optimizes_passes():
    rows = 101
    source_panel = rows * 2 * 1 * 2 * 4 * 8
    candidate = gxe_multi._descriptor_memory_candidate(
        rows=rows,
        variants=37,
        annotation_bins=1,
        environments=2,
        probes=7,
        block_width=5,
        block_count=8,
        feature_columns=6,
        common_rank=1,
        reader_rank=2,
        threads=2,
        environment_tile_count=1,
        probe_tile_count=2,
        protected=True,
        integrity_enabled=False,
        native_workspace_gib=1.0,
        baseline_rss_bytes=64 * 1024**2,
    )
    assert candidate["component_bytes"]["source_panel"] == source_panel
    assert candidate["panel_live_peak_bytes"] == source_panel
    assert candidate["component_bytes"]["target_virtual_rhs"] == 0
    assert candidate["component_bytes"]["vendor_workspace_allowance"] == 0
    assert candidate["component_bytes"]["packed_missing_index_upper_bound"] == (
        rows * 5 * 4 + 5 * 24
    )
    assert candidate["phase_peak_bytes"]["target"] == (
        candidate["component_bytes"]["context_copies"]
        + candidate["component_bytes"]["persistent_outputs"]
        + source_panel
        + candidate["component_bytes"]["packed_genotype_block"]
        + candidate["component_bytes"]["target_virtual_rhs"]
        + candidate["component_bytes"]["target_output"]
        + candidate["component_bytes"]["source_block_scales"]
        + candidate["component_bytes"]["mailman_worker_scratch"]
    )

    wide_unbounded = gxe_multi._descriptor_memory_candidate(
        rows=rows,
        variants=37,
        annotation_bins=1,
        environments=3,
        probes=256,
        block_width=5,
        block_count=8,
        feature_columns=9,
        common_rank=1,
        reader_rank=2,
        threads=2,
        environment_tile_count=1,
        probe_tile_count=1,
        protected=True,
        integrity_enabled=True,
        native_workspace_gib=1.0,
        baseline_rss_bytes=64 * 1024**2,
        vendor_probe_chunk_width=None,
        direct_kernel_mode="dense_blas_hybrid",
    )
    assert wide_unbounded["probe_tile_count"] == 1
    assert wide_unbounded["execution_probe_chunk_count"] == 1
    assert wide_unbounded["maximum_execution_probe_chunk_width"] == 256
    assert wide_unbounded["fused_two_pass_execution"] is True
    assert wide_unbounded["planned_genotype_passes"] == 2
    assert wide_unbounded["direct_kernel_mode"] == "dense_blas_hybrid"
    assert wide_unbounded["component_bytes"]["vendor_workspace_allowance"] == 0
    assert wide_unbounded["panel_live_peak_bytes"] == (
        2 * wide_unbounded["component_bytes"]["source_panel"]
    )
    assert wide_unbounded["planned_native_output_calls"] == 25

    python_fallback = gxe_multi._descriptor_memory_candidate(
        rows=rows,
        variants=37,
        annotation_bins=1,
        environments=2,
        probes=7,
        block_width=5,
        block_count=8,
        feature_columns=6,
        common_rank=1,
        reader_rank=2,
        threads=2,
        environment_tile_count=1,
        probe_tile_count=2,
        protected=False,
        integrity_enabled=False,
        native_workspace_gib=1.0,
        baseline_rss_bytes=64 * 1024**2,
    )
    assert python_fallback["component_bytes"]["decoded_genotype_block"] == (
        rows * 5 * 8
    )
    assert python_fallback["component_bytes"]["packed_genotype_block"] == 0
    assert python_fallback["panel_live_peak_bytes"] == 2 * source_panel
    assert python_fallback["component_bytes"]["vendor_workspace_allowance"] == 0

    large_python_fallback = gxe_multi._descriptor_memory_candidate(
        rows=10_000,
        variants=20_000,
        annotation_bins=1,
        environments=2,
        probes=64,
        block_width=2_000,
        block_count=10,
        feature_columns=6,
        common_rank=1,
        reader_rank=2,
        threads=2,
        environment_tile_count=1,
        probe_tile_count=1,
        protected=False,
        integrity_enabled=False,
        native_workspace_gib=1.0,
        baseline_rss_bytes=64 * 1024**2,
    )
    assert large_python_fallback["component_bytes"][
        "vendor_workspace_allowance"
    ] == 1024**3

    panel_limit = 4 * rows * 4 * 8
    estimators = tuple(
        SimpleNamespace(
            target_xz_mem=panel_limit / 1024**3,
            gxe_total_memory_request=10.0,
            nsamp=rows,
            nsnps=37,
            nbins=1,
            nvecs=7,
            p_eff=1,
            num_threads=2,
            native_workspace_gib=1.0,
        )
        for _ in range(2)
    )
    environment_tiles, probe_tiles, plan = gxe_multi._shared_execution_tiles(
        estimators,
        np.dtype(np.float64),
        protected=True,
        blocks=[(0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
                (25, 30), (30, 35), (35, 37)],
        feature_plan={"basis": np.empty((rows, 6), dtype=np.float64)},
        common_rank=1,
        integrity_enabled=False,
        baseline_rss_bytes=64 * 1024**2,
        available_memory=(20 * 1024**3, {"limiting_source": "test"}),
    )
    assert environment_tiles == [(0, 1), (1, 2)]
    assert probe_tiles == [(0, 7)]
    assert plan["planned_genotype_passes"] == 5
    assert plan["direct_kernel_mode"] == "packed_mailman"
    assert plan["mailman_maximum_probe_count"] == 10
    assert plan["mailman_probe_count_eligible"] is True
    assert plan["panel_live_peak_bytes"] == rows * 2 * 7 * 8
    assert plan["memory_contract_satisfied"] is True

    with pytest.raises(ValueError, match="at most 10 probes"):
        gxe_multi._descriptor_memory_candidate(
            rows=rows,
            variants=37,
            annotation_bins=1,
            environments=2,
            probes=11,
            block_width=5,
            block_count=8,
            feature_columns=6,
            common_rank=1,
            reader_rank=2,
            threads=2,
            environment_tile_count=1,
            probe_tile_count=2,
            protected=True,
            integrity_enabled=False,
            native_workspace_gib=1.0,
            baseline_rss_bytes=64 * 1024**2,
            direct_kernel_mode="packed_mailman",
        )

    for estimator in estimators:
        estimator.target_xz_mem = (2 * rows * 8 - 1) / 1024**3
    with pytest.raises(RuntimeError, match="No GxE descriptor tile plan"):
        gxe_multi._shared_execution_tiles(
            estimators,
            np.dtype(np.float64),
            protected=True,
            blocks=[(0, 5)],
            feature_plan={"basis": np.empty((rows, 6), dtype=np.float64)},
            common_rank=1,
            integrity_enabled=False,
            baseline_rss_bytes=64 * 1024**2,
            available_memory=(20 * 1024**3, {"limiting_source": "test"}),
        )


def test_total_process_budget_has_exact_baseline_and_available_boundaries():
    gib = 1024**3
    accepted = gxe_multi._resolve_total_process_budget(
        2.0,
        baseline_rss_bytes=gib,
        available_memory=(gib, {"limiting_source": "test"}),
    )
    assert accepted["resolved_bytes"] == 2 * gib
    assert accepted["resolved_increment_bytes"] == gib
    with pytest.raises(RuntimeError, match="does not exceed"):
        gxe_multi._resolve_total_process_budget(
            1.0,
            baseline_rss_bytes=gib,
            available_memory=(gib, {"limiting_source": "test"}),
        )
    with pytest.raises(RuntimeError, match="exceeds currently available"):
        gxe_multi._resolve_total_process_budget(
            2.0 + 1.0 / gib,
            baseline_rss_bytes=gib,
            available_memory=(gib, {"limiting_source": "test"}),
        )


def test_multi_environment_batch_manifest_failure_rolls_back_bundles(
    tmp_path, monkeypatch
):
    prefix, environment, _ = _inputs(tmp_path)
    estimators = [
        _estimator(prefix, environment, tmp_path / f"rollback.{column}", column)
        for column in ("age", "bmi")
    ]
    monkeypatch.setattr(
        gxe_multi,
        "_publish_json_no_replace",
        lambda *_: (_ for _ in ()).throw(OSError("injected batch seal failure")),
    )
    try:
        with pytest.raises(OSError, match="injected batch seal failure"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "rollback.gxe.multi.json",
            )
    finally:
        for estimator in estimators:
            estimator.close()

    for column in ("age", "bmi"):
        output = tmp_path / f"rollback.{column}"
        assert not Path(f"{output}.gxe.ref.json").exists()
        assert not Path(f"{output}.gxx.ldscore.gz").exists()
        assert not Path(f"{output}.gxe.jackknife.npz").exists()


def test_incomplete_final_telemetry_rolls_back_before_batch_publication(
    tmp_path, monkeypatch
):
    prefix, environment, _ = _inputs(tmp_path)
    estimators = [
        _estimator(prefix, environment, tmp_path / f"telemetry.{column}", column)
        for column in ("age", "bmi")
    ]
    original_executor = gxe_multi._MultiEnvironmentGemm

    class FinalStatusFailureExecutor(original_executor):
        def performance_report(self, **kwargs):
            report = super().performance_report(**kwargs)
            if kwargs.get("capture_boundary") == (
                "post_output_artifact_publication_pre_batch_manifest"
            ):
                report["telemetry_complete"] = False
            return report

    monkeypatch.setattr(
        gxe_multi, "_MultiEnvironmentGemm", FinalStatusFailureExecutor
    )
    batch = tmp_path / "telemetry.gxe.multi.json"
    try:
        with pytest.raises(RuntimeError, match="final performance telemetry"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=batch,
                requested_backend="direct",
            )
    finally:
        for estimator in estimators:
            estimator.close()

    assert not batch.exists()
    for column in ("age", "bmi"):
        output = tmp_path / f"telemetry.{column}"
        assert not Path(f"{output}.gxe.ref.json").exists()
        assert not Path(f"{output}.gxx.ldscore.gz").exists()
        assert not Path(f"{output}.gxe.jackknife.npz").exists()


def test_incomplete_output_numa_evidence_stops_before_bundle_staging(
    tmp_path, monkeypatch
):
    prefix, environment, _ = _inputs(tmp_path)
    estimators = [
        _estimator(prefix, environment, tmp_path / f"preoutput.{column}", column)
        for column in ("age", "bmi")
    ]
    original_executor = gxe_multi._MultiEnvironmentGemm

    class PreOutputStatusFailureExecutor(original_executor):
        def performance_report(self, **kwargs):
            report = super().performance_report(**kwargs)
            if kwargs.get("capture_boundary") == "pre_output_bundle_staging":
                report["native_gemm_output_numa_evidence_complete"] = False
            return report

    monkeypatch.setattr(
        gxe_multi, "_MultiEnvironmentGemm", PreOutputStatusFailureExecutor
    )
    batch = tmp_path / "preoutput.gxe.multi.json"
    try:
        with pytest.raises(RuntimeError, match="performance telemetry"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=batch,
                requested_backend="direct",
            )
    finally:
        for estimator in estimators:
            estimator.close()

    assert not batch.exists()
    for column in ("age", "bmi"):
        output = tmp_path / f"preoutput.{column}"
        assert not Path(f"{output}.gxe.ref.json").exists()
        assert not Path(f"{output}.gxx.ldscore.gz").exists()
        assert not Path(f"{output}.gxe.jackknife.npz").exists()


def test_native_output_evidence_joins_vendor_and_deterministic_fallback_records():
    estimator = SimpleNamespace(num_threads=1, dtype=np.dtype("float64"))
    executor = gxe_multi._MultiEnvironmentGemm("python", estimator)
    vendor_records: list[dict] = []
    output_records: list[dict] = []

    def evidence(call_id):
        return {
            "schema": "summit.native_gemm_output_numa.v1",
            "schema_version": 1,
            "applicable": True,
            "call_id": call_id,
            "contract_required": True,
            "complete": True,
            "operand_role": "protected_gemm_output",
        }

    class FakeNative:
        emit_vendor = True
        call_id = 0

        @classmethod
        def protected_matmul_nn(cls, left, right, threads):
            assert threads == 1
            cls.call_id += 1
            observed = evidence(cls.call_id)
            output_records.append(dict(observed))
            if cls.emit_vendor:
                vendor_records.append(
                    {
                        "operation": "dgemm_nn",
                        "m": left.shape[0],
                        "n": right.shape[1],
                        "k": left.shape[1],
                        "wall_seconds": 0.1,
                        "process_cpu_seconds": 0.1,
                        "native_gemm_output_numa": dict(observed),
                    }
                )
            return left @ right, 0

    def consume(records):
        drained = list(records)
        records.clear()
        return drained

    executor.protected = True
    executor._module = FakeNative
    executor.integrity_enabled = True
    executor.integrity_minimum_vendor_flops = 10**9
    executor._native_telemetry_consumer = lambda: consume(vendor_records)
    executor._native_telemetry_available = True
    executor._native_gemm_output_numa_contract_supported = True
    executor._native_gemm_output_numa_contract_required = True
    executor._native_gemm_output_numa_consumer = lambda: consume(output_records)
    executor._native_gemm_output_numa_available = True
    left = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(4, 3))
    right = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(3, 2))

    first = executor.nn(left, right)
    FakeNative.emit_vendor = False
    second = executor.nn(left, right)
    np.testing.assert_array_equal(first, left @ right)
    np.testing.assert_array_equal(second, left @ right)
    assert [
        item["native_gemm_output_numa"]["call_id"]
        for item in executor._gemm_records
    ] == [1, 2]
    assert executor._gemm_records[0]["telemetry_scope"] == "vendor_call"
    assert executor._gemm_records[1]["telemetry_scope"] == (
        "deterministic_tiled_call_boundary"
    )
    assert [
        item["call_id"] for item in executor._native_gemm_output_numa_records
    ] == [1, 2]

    # The bounded feature and target shapes that intentionally bypass BLIS
    # remain observed deterministic calls even though they exceed the generic
    # one-billion-FLOP integrity threshold.
    bounded_fallback_shapes = (
        ("multi_feature", 112, 2000, 9982),
        ("multi_target_score", 2000, 40, 9982),
    )
    for call_id, (operation, m, n, k) in enumerate(
        bounded_fallback_shapes, start=3
    ):
        output_records.append(evidence(call_id))
        executor._record_gemm(
            executor._native_kernel_fallback(
                operation,
                m=m,
                n=n,
                k=k,
                transpose_a="T",
                lda=k,
                ldb=k,
                ldc=m,
                wall_seconds=0.1,
                process_cpu_seconds=0.1,
            ),
            protected_output_expected=True,
        )
        assert executor._gemm_records[-1]["telemetry_scope"] == (
            "deterministic_tiled_call_boundary"
        )

    malformed = evidence(5)
    malformed["call_id"] = np.int64(5)
    output_records.append(malformed)
    with pytest.raises(RuntimeError, match="call ID is not exact"):
        executor._consume_native_gemm_output_numa(output_expected=True)

    executor._native_gemm_output_numa_records = [
        {} for _ in range(gxe_multi._NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY)
    ]
    output_records.append(evidence(6))
    with pytest.raises(RuntimeError, match="evidence buffer overflowed"):
        executor._consume_native_gemm_output_numa(output_expected=True)
    assert len(executor._native_gemm_output_numa_records) == 16384
    assert executor._dropped_native_gemm_output_numa_records == 1


def test_no_placement_protected_output_preserves_uncontracted_evidence():
    estimator = SimpleNamespace(num_threads=1, dtype=np.dtype("float64"))
    executor = gxe_multi._MultiEnvironmentGemm("python", estimator)
    drained: list[dict] = []

    class FakeNative:
        @staticmethod
        def protected_matmul_nn(left, right, threads):
            assert threads == 1
            drained.append(
                {
                    "schema": "summit.native_gemm_output_numa.v1",
                    "schema_version": 1,
                    "applicable": True,
                    "operand_role": "protected_gemm_output",
                    "contract_required": False,
                    "complete": False,
                    "call_id": 7,
                    "logical_rows": left.shape[0],
                    "logical_columns": right.shape[1],
                    "storage_layout": "column_major",
                    "logical_byte_count": left.shape[0] * right.shape[1] * 8,
                    "allocation_mode": "legacy_posix_memalign",
                }
            )
            return left @ right, 0

    executor.protected = True
    executor._module = FakeNative
    executor.integrity_enabled = True
    executor.integrity_minimum_vendor_flops = 10**9
    executor._native_gemm_output_numa_contract_supported = True
    assert executor._native_gemm_output_numa_contract_required is False
    def consume_uncontracted():
        records = list(drained)
        drained.clear()
        return records

    executor._native_gemm_output_numa_consumer = consume_uncontracted
    executor._native_gemm_output_numa_available = True
    left = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(4, 3))
    right = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(3, 2))
    observed = executor.nn(left, right)
    np.testing.assert_array_equal(observed, left @ right)
    evidence = executor._gemm_records[0]["native_gemm_output_numa"]
    assert evidence["contract_required"] is False
    assert evidence["complete"] is False
    assert evidence["allocation_mode"] == "legacy_posix_memalign"

    drained.append(
        {
            "schema": "summit.native_gemm_output_numa.v1",
            "schema_version": 1,
            "applicable": True,
            "call_id": 8,
            "contract_required": True,
            "complete": True,
            "operand_role": "protected_gemm_output",
        }
    )
    contracted = executor._consume_native_gemm_output_numa(output_expected=True)
    assert contracted["contract_required"] is True
    assert contracted["complete"] is True


def test_cli_parallel_environment_groups_publish_canonical_batch(tmp_path):
    from summit import cli, gxeldcore

    sockets = cli._socket_local_core_groups()
    if shutil.which("taskset") is None or len(sockets) < 2:
        pytest.skip("socket-isolated CLI integration requires Linux taskset and two sockets")
    native_build = gxeldcore.build_info()
    placement_backend_supported = (
        native_build.get("blas_vendor") == "BLIS"
        and native_build.get("blas_runtime_threading_layer") == "pthreads"
        and native_build.get("openmp_enabled") is True
    ) or native_build.get("blas_runtime_threading_layer") == "openmp"
    if not placement_backend_supported:
        pytest.skip(
            "explicit placement requires pthread-BLIS or an OpenMP BLAS"
        )
    prefix, environment, _ = _inputs(tmp_path)
    frame = pd.read_csv(environment, sep="\t")
    frame["alcohol"] = 0.25 * frame["age"] + np.linspace(-1.0, 1.0, len(frame))
    frame["smoking"] = 0.15 * frame["bmi"] + np.linspace(1.0, -1.0, len(frame))
    frame.to_csv(environment, sep="\t", index=False)
    output = tmp_path / "parallel"
    environment_vars = os.environ.copy()
    environment_vars["OMP_WAIT_POLICY"] = "PASSIVE"
    environment_vars["GOMP_SPINCOUNT"] = "0"
    environment_vars["OMP_PROC_BIND"] = "FALSE"
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "summit.cli",
            "--geno", str(prefix),
            "--env", str(environment),
            "--gxe-env-cols", "age,bmi,alcohol,smoking",
            "--gxe-native-backend", "direct",
            "--rand-dist", "rademacher",
            "--gxe-parallel-environment-groups", "2",
            "--gxe-kernel-mode", "standardized",
            "--gxe-genotype-scale", "sample",
            "--nvecs", "11",
            "--step_size", "5",
            "--target-xz-mem", "0.01",
            "--num-threads", "32",
            "--force_affinity_all", "false",
            "--seed", "2718",
            "--suppress",
            "--out", str(output),
        ],
        env=environment_vars,
        check=True,
    )
    payload = json.loads(
        Path(f"{output}.gxe.multi.json").read_text(encoding="utf-8")
    )
    assert payload["execution"] == "parallel_isolated_environment_groups"
    assert [record["environment"] for record in payload["references"]] == [
        "age", "bmi", "alcohol", "smoking"
    ]
    assert len(payload["environment_groups"]) == 2
    assert payload["numa_bound_bed_decode_complete"] is True
    decode_groups = payload["numa_bound_bed_decode_groups"]
    assert len(decode_groups) == 2
    assert all(
        report["schema"] == "summit.numa_bound_bed_decode.v1"
        and report["complete"] is True
        and report["records_included"] is False
        and report["observed_block_read_count"] > 0
        for report in decode_groups
    )
    assert payload["packed_source_panel_numa_complete"] is True
    source_groups = payload["packed_source_panel_numa_groups"]
    assert len(source_groups) == 2
    assert all(
        report["schema"] == "summit.packed_source_panel_numa.v1"
        and report["complete"] is True
        and report["records_included"] is False
        and report["record_count"] > 0
        for report in source_groups
    )
    for name in ("age", "bmi", "alcohol", "smoking"):
        assert Path(f"{output}.{name}.gxe.ref.json").is_file()

    serial = tmp_path / "serial"
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "summit.cli",
            "--geno", str(prefix),
            "--env", str(environment),
            "--gxe-env-cols", "age,bmi,alcohol,smoking",
            "--gxe-native-backend", "direct",
            "--rand-dist", "rademacher",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-kernel-mode", "standardized",
            "--gxe-genotype-scale", "sample",
            "--nvecs", "11",
            "--step_size", "5",
            "--target-xz-mem", "0.01",
            "--num-threads", "4",
            "--force_affinity_all", "false",
            "--seed", "2718",
            "--suppress",
            "--out", str(serial),
        ],
        env=environment_vars,
        check=True,
    )
    for name in ("age", "bmi", "alcohol", "smoking"):
        for score_name in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(Path(f"{output}.{name}"), score_name),
                _score(Path(f"{serial}.{name}"), score_name),
                rtol=3e-12,
                atol=3e-12,
            )
