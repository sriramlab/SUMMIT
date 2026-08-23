from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit import gxeldcore
from summit.cli import build_parser
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.gxe_multi import (
    _MultiEnvironmentGemm,
    generate_multi_environment_references,
)
from summit.logger import Logger


def test_native_fp64_tt_row_layout_diagnostic_primitives_match_dense_oracles():
    """Keep the non-production row-target API covered as a diagnostic primitive."""
    assert gxeldcore.build_info()["api_version"] >= 8
    rng = np.random.default_rng(81073)
    n, block, panel = 37, 11, 12

    genotype_f = np.asfortranarray(rng.normal(size=(n, block)))
    weights_f = np.asfortranarray(rng.normal(size=(block, panel)))
    observed_source, repaired = gxeldcore.protected_matmul_tt_row_major_output(
        weights_f, genotype_f, 2
    )
    observed_source = np.asarray(observed_source)
    assert repaired == 0
    assert observed_source.flags.c_contiguous
    np.testing.assert_allclose(
        observed_source, genotype_f @ weights_f, rtol=2e-13, atol=2e-13
    )

    raw = rng.normal(size=(n, block))
    raw[3, 2] = np.nan
    raw[19, 7] = np.nan
    raw[5, 4] = np.inf
    raw[7, 5] = -np.inf
    targets = np.asfortranarray(rng.normal(size=(n, 3)))
    targets -= targets.mean(axis=0, keepdims=True)
    genotype_column = np.asfortranarray(raw)
    genotype_row = np.ascontiguousarray(raw)
    counts_column, correlations_column = gxeldcore.standardize_genotype_block(
        genotype_column, targets, 1, False, 1e-12, 2
    )
    counts_row, correlations_row = gxeldcore.standardize_genotype_block_row_major(
        genotype_row, targets, 1, False, 1e-12, 2
    )
    assert genotype_row.flags.c_contiguous
    np.testing.assert_array_equal(counts_row, counts_column)
    np.testing.assert_allclose(
        correlations_row, correlations_column, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        genotype_row, genotype_column, rtol=2e-15, atol=2e-15
    )

    common, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    common = np.asfortranarray(common)
    directions = np.asfortranarray(rng.normal(size=(n, 2)))
    columns_per_environment = 3
    source_panel = np.ascontiguousarray(rng.normal(size=(n, 12)))
    expected_panel = source_panel.copy()
    expected_panel -= common @ (common.T @ expected_panel)
    for group in range(2):
        segment = slice(
            group * 2 * columns_per_environment,
            (group + 1) * 2 * columns_per_environment,
        )
        direction = directions[:, group]
        expected_panel[:, segment] -= direction[:, None] * (
            direction @ expected_panel[:, segment]
        )
    expected_panel -= expected_panel.mean(axis=0, keepdims=True)
    gxeldcore.project_protected_row_major_sources(
        source_panel, common, directions, columns_per_environment, 2
    )
    np.testing.assert_allclose(
        source_panel, expected_panel, rtol=3e-13, atol=3e-13
    )

    environments = np.asfortranarray(rng.normal(size=(n, 2)))
    sealed_panel = source_panel.copy()
    sealed_environments = environments.copy()
    pair = gxeldcore.prepare_protected_row_major_weighted_pair(
        source_panel, environments, 2
    )
    source_panel.fill(np.nan)
    environments.fill(np.nan)
    del source_panel, environments
    gc.collect()
    genotype_target = np.ascontiguousarray(rng.normal(size=(n, block)))
    observed_target, repaired = gxeldcore.protected_matmul_row_major_tn_pair(
        genotype_target, pair, 2
    )
    observed_target = np.asarray(observed_target)
    assert repaired == 0
    assert observed_target.flags.c_contiguous
    weighted = np.empty_like(sealed_panel)
    for group in range(2):
        segment = slice(
            group * 2 * columns_per_environment,
            (group + 1) * 2 * columns_per_environment,
        )
        weighted[:, segment] = (
            sealed_environments[:, group, None] * sealed_panel[:, segment]
        )
    expected_target = np.concatenate(
        [genotype_target.T @ sealed_panel, genotype_target.T @ weighted], axis=1
    )
    np.testing.assert_allclose(
        observed_target, expected_target, rtol=3e-13, atol=3e-13
    )


def _small_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(99181)
    n, variants = 31, 13
    raw = rng.binomial(
        2, rng.uniform(0.14, 0.42, size=variants), size=(n, variants)
    ).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    age = rng.normal(size=n)
    bmi = 0.4 * age + rng.normal(size=n)
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "age": age,
            "bmi": bmi,
        }
    ).to_csv(environment, sep="\t", index=False)
    return prefix, environment


def _estimator(
    prefix: Path, environment: Path, output: Path, column: str
) -> GenomewideEnvLDScore:
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(output),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=17,
        step_size=5,
        seed=2718,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized",
        genotype_scale="sample",
        native_backend="python",
    )


def _run_reference(
    prefix: Path,
    environment: Path,
    output: Path,
    *,
    layout: str | None,
) -> dict:
    estimators = [
        _estimator(prefix, environment, output.with_name(f"{output.name}.{name}"), name)
        for name in ("age", "bmi")
    ]
    try:
        options = {}
        if layout is not None:
            options["full_precision_layout"] = layout
        manifest = generate_multi_environment_references(
            estimators,
            batch_manifest=output.with_suffix(".gxe.multi.json"),
            requested_backend="direct",
            **options,
        )
    finally:
        for estimator in estimators:
            estimator.close()
    return json.loads(manifest.read_text(encoding="utf-8"))


def _scores(prefix: Path, family: str) -> np.ndarray:
    frame = pd.read_csv(f"{prefix}.{family}.ldscore.gz", sep="\t")
    return frame[["L2_0"]].to_numpy()


def _reference_payload(prefix: Path) -> tuple[Path, dict]:
    manifest = Path(f"{prefix}.gxe.ref.json")
    return manifest, json.loads(manifest.read_text(encoding="utf-8"))


def _population_and_diagonal(prefix: Path) -> tuple[np.ndarray, pd.DataFrame]:
    manifest, payload = _reference_payload(prefix)
    population_trace = payload["population_trace"]
    assert "jackknife_diagonal_method" not in population_trace
    diagonal = manifest.parent / payload["files"]["diagonal"]
    digest = hashlib.sha256(diagonal.read_bytes()).hexdigest()
    assert digest == payload["artifact_sha256"]["diagonal"]
    return (
        np.asarray(
            population_trace["same_individual_kernel_products"],
            dtype=np.float64,
        ),
        pd.read_csv(diagonal, sep="\t"),
    )


def test_explicit_current_layout_matches_default_four_family_oracle(tmp_path):
    prefix, environment = _small_inputs(tmp_path)
    default_prefix = tmp_path / "default"
    explicit_prefix = tmp_path / "explicit"
    current = _run_reference(
        prefix, environment, default_prefix, layout=None
    )
    explicit = _run_reference(
        prefix,
        environment,
        explicit_prefix,
        layout="current",
    )
    assert current["full_precision_layout"] == "current"
    assert explicit["full_precision_layout"] == "current"
    assert explicit["shared_genotype_passes"] == current["shared_genotype_passes"]
    current_performance = current["performance_telemetry"]
    explicit_performance = explicit["performance_telemetry"]
    for performance in (current_performance, explicit_performance):
        assert performance["telemetry_complete"] is True
        assert performance["full_precision_layout"] == "current"
        assert performance["optimized_fp64_layout_telemetry_complete"] is True
        assert performance["optimized_fp64_layout_telemetry"]["required"] is False
        records = performance["gemm_records"]
        assert any(
            record.get("phase") == "source_gemm"
            and record.get("transpose_a") == "N"
            and record.get("transpose_b") == "N"
            and record.get("layout") == "column_major"
            for record in records
        )
        assert any(
            record.get("phase") == "target_gemm"
            and record.get("transpose_a") == "T"
            and record.get("transpose_b") == "N"
            and record.get("layout") == "column_major"
            for record in records
        )
    for reference in explicit["references"]:
        _, payload = _reference_payload(
            explicit_prefix.with_name(
                f"explicit.{reference['environment']}"
            )
        )
        resources = payload["resource_estimates"]
        assert resources["source_panel_memory_order"] == "F"
        assert resources["target_panel_memory_order"] == "F"
        assert resources["target_genotype_memory_order"] == "F"
        assert resources["source_to_target_layout_transition"] == "none"
        assert resources["modeled_target_pair_sealing_live_peak_gib"] == (
            pytest.approx(2 * resources["modeled_packed_source_panel_gib"])
        )
    for environment_name in ("age", "bmi"):
        default_environment_prefix = default_prefix.with_name(
            f"default.{environment_name}"
        )
        explicit_environment_prefix = explicit_prefix.with_name(
            f"explicit.{environment_name}"
        )
        for family in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _scores(default_environment_prefix, family),
                _scores(explicit_environment_prefix, family),
                rtol=0.0,
                atol=0.0,
            )
        default_population, default_diagonal = _population_and_diagonal(
            default_environment_prefix
        )
        explicit_population, explicit_diagonal = _population_and_diagonal(
            explicit_environment_prefix
        )
        np.testing.assert_array_equal(
            explicit_population,
            default_population,
        )
        pd.testing.assert_frame_equal(
            explicit_diagonal,
            default_diagonal,
            check_exact=True,
        )


def test_only_current_production_layout_is_accepted():
    parser = build_parser()
    assert parser.parse_args([]).gxe_fp64_layout == "current"
    assert parser.parse_args(
        ["--gxe-fp64-layout", "current"]
    ).gxe_fp64_layout == "current"
    for rejected in ("source-tt-target-current", "source-tt-target-row"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--gxe-fp64-layout", rejected])

    fp64 = SimpleNamespace(num_threads=1, dtype=np.dtype(np.float64))
    for rejected in ("source_tt_target_current", "source_tt_target_row"):
        with pytest.raises(ValueError, match="only supported production"):
            _MultiEnvironmentGemm(
                "python", fp64, full_precision_layout=rejected
            )


def test_current_layout_telemetry_remains_nonrequired_and_observable():
    executor = object.__new__(_MultiEnvironmentGemm)
    executor.full_precision_layout = "current"
    executor._hot_gemm_phase_counts = {
        "source_gemm": {"logical_calls": 1},
        "target_gemm": {"logical_calls": 1},
    }
    executor._gemm_records = [
        {
            "sequence": 1,
            "phase": "source_gemm",
            "telemetry_scope": "vendor_call",
            "operation": "dgemm_nn",
            "layout": "column_major",
            "transpose_a": "N",
            "transpose_b": "N",
            "m": 3,
            "n": 5,
            "k": 7,
            "lda": 3,
            "ldb": 7,
            "ldc": 3,
        },
        {
            "sequence": 2,
            "phase": "target_gemm",
            "telemetry_scope": "vendor_call",
            "operation": "dgemm_tn",
            "layout": "column_major",
            "transpose_a": "T",
            "transpose_b": "N",
            "m": 3,
            "n": 5,
            "k": 7,
            "lda": 7,
            "ldb": 7,
            "ldc": 3,
        },
    ]
    valid = executor._optimized_layout_telemetry_status()
    assert valid["required"] is False
    assert valid["complete"] is True
    assert valid["violation_count"] == 0

    executor._gemm_records[1]["ldb"] = 5
    invalid = executor._optimized_layout_telemetry_status()
    assert invalid["complete"] is True
    assert invalid["violation_count"] == 1
    assert invalid["violations"][0]["phase"] == "target_gemm"


@pytest.mark.parametrize(
    "layout", ("source_tt_target_current", "source_tt_target_row")
)
def test_generator_rejects_noncurrent_layout_before_publication(tmp_path, layout):
    prefix, environment = _small_inputs(tmp_path)
    output = tmp_path / layout
    estimators = [
        _estimator(
            prefix,
            environment,
            output.with_name(f"{output.name}.{name}"),
            name,
        )
        for name in ("age", "bmi")
    ]
    batch = output.with_suffix(".gxe.multi.json")
    try:
        with pytest.raises(ValueError, match="only supported production"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=batch,
                requested_backend="direct",
                full_precision_layout=layout,
            )
    finally:
        for estimator in estimators:
            estimator.close()
    assert not batch.exists()
    for name in ("age", "bmi"):
        assert not Path(f"{output}.{name}.gxe.ref.json").exists()
        assert not Path(f"{output}.{name}.gxx.ldscore.gz").exists()


def _configured_native_threads() -> int:
    desired = min(2, len(os.sched_getaffinity(0)))
    try:
        return int(gxeldcore.configure_blas_threads(desired))
    except RuntimeError as exc:
        assert "different thread count" in str(exc)
        configured = int(gxeldcore.build_info()["blas_runtime_threads"])
        assert gxeldcore.configure_blas_threads(configured) == configured
        return configured


@pytest.mark.xfail(
    strict=False,
    reason=(
        "API8 source TT is a rejected diagnostic primitive: fresh-process "
        "integrity stress observed gross wrong cells and repairs"
    ),
)
def test_diagnostic_source_tt_vendor_telemetry_and_dense_oracle():
    """Retain direct API8 evidence without making TT a production gate."""
    threads = _configured_native_threads()
    rng = np.random.default_rng(52091)
    m, n, k = 256, 512, 4096

    weights = np.asfortranarray(rng.normal(size=(k, m)))
    source_genotype = np.asfortranarray(rng.normal(size=(n, k)))
    target_genotype = np.asfortranarray(rng.normal(size=(k, m)))
    source_panel = np.asfortranarray(rng.normal(size=(k, n // 2)))
    row_weights = np.asfortranarray(np.ones((k, 1), dtype=np.float64))
    pair = gxeldcore.prepare_protected_row_weighted_pair(
        source_panel, row_weights, threads
    )

    gxeldcore.reset_gemm_telemetry()
    observed_source, repaired_source = (
        gxeldcore.protected_matmul_tt_row_major_output(
            weights, source_genotype, threads
        )
    )
    observed_target, repaired_target = (
        gxeldcore.protected_matmul_tn_pair(
            target_genotype, pair, threads
        )
    )
    assert repaired_source == 0
    assert repaired_target == 0
    np.testing.assert_allclose(
        observed_source, source_genotype @ weights, rtol=3e-13, atol=3e-12
    )
    expected_target_half = target_genotype.T @ source_panel
    np.testing.assert_allclose(
        observed_target,
        np.concatenate([expected_target_half, expected_target_half], axis=1),
        rtol=3e-13,
        atol=3e-12,
    )

    records = [dict(record) for record in gxeldcore.consume_gemm_telemetry()]
    assert len(records) == 2
    source_record, target_record = records
    assert source_record["operation"] == "dgemm_tt"
    assert source_record["layout"] == "column_major"
    assert (source_record["transpose_a"], source_record["transpose_b"]) == (
        "T", "T"
    )
    assert (source_record["m"], source_record["n"], source_record["k"]) == (
        m, n, k
    )
    assert (
        source_record["lda"], source_record["ldb"], source_record["ldc"]
    ) == (k, n, m)
    assert target_record["operation"] == "dgemm_tn"
    assert target_record["layout"] == "column_major"
    assert (target_record["transpose_a"], target_record["transpose_b"]) == (
        "T", "N"
    )
    assert (target_record["m"], target_record["n"], target_record["k"]) == (
        m, n, k
    )
    assert (
        target_record["lda"], target_record["ldb"], target_record["ldc"]
    ) == (k, k, m)
    for record in records:
        assert record["omp_in_parallel"] is False
        assert record["omp_level"] == 0
        assert record["requested_threads"] == threads
        assert record["configured_threads"] == threads
        assert record["backend_threads"] == threads
