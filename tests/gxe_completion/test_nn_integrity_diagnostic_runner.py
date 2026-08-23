from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.gxe import run_nn_integrity_diagnostic as runner


def _fake_diagnostic(
    *, classification: str, flagged: bool, m: int = 3, n: int = 4,
) -> tuple[dict, np.ndarray]:
    output = np.asfortranarray(np.arange(m * n, dtype=np.float64).reshape((m, n)))
    if flagged:
        column = 1
        raw = np.asfortranarray((output[:, [column]] + 1.0e-13))
        tiled = np.asfortranarray(output[:, [column]])
        reference = np.asfortranarray(output[:, [column]])
        flagged_columns = [column]
        expected_value = 1.0
        observed_value = 1.0 + 1.0e-13
        difference = expected_value - observed_value
        epsilon = float(np.finfo(np.float64).eps)
        gamma_m = (m * epsilon) / (1.0 - m * epsilon)
        gamma_k = (7 * epsilon) / (1.0 - 7 * epsilon)
        relative_bound = float(32.0 * (2.0 * gamma_m + gamma_k + epsilon))
        current_tolerance = float(relative_bound * max(
            1.0, abs(expected_value), abs(observed_value)
        ))
        checks = [
            {
                "column": column,
                "check": 2,
                "expected": expected_value,
                "observed": observed_value,
                "difference": difference,
                "absolute_difference": abs(difference),
                "current_relative_bound": relative_bound,
                "current_tolerance": current_tolerance,
                "direct_absolute_product_sum": 2.0,
                "factored_expected_absolute_sum": 2.0,
                "observed_checksum_absolute_sum": 2.0,
                "projection_roundoff_component": 1.0e-12,
                "expected_reduction_roundoff_component": 1.0e-12,
                "vendor_product_roundoff_component": 1.0e-12,
                "observed_reduction_roundoff_component": 1.0e-12,
                "cancellation_aware_bound": float(
                    np.longdouble(32.0)
                    * (
                        np.longdouble(4.0e-12)
                        + np.longdouble(epsilon) * np.longdouble(2.0)
                    )
                ),
                "cancellation_aware_disagrees": False,
            }
        ]
        comparisons = [
            {
                "column": column,
                "classification": classification,
                "raw_vendor_nonfinite_count": 0,
                "deterministic_tiled_nonfinite_count": 0,
                "long_double_reference_nonfinite_count": 0,
                "vendor_tiled_unequal_count": m,
                "vendor_reference_unequal_count": m,
                "tiled_reference_unequal_count": 0,
                "vendor_rows_outside_forward_error_bound": (
                    1 if classification == "vendor_result_outside_forward_error_bound" else 0
                ),
                "tiled_rows_outside_forward_error_bound": 0,
                "max_abs_vendor_minus_tiled": float(
                    np.max(np.abs(raw - tiled))
                ),
                "max_abs_vendor_minus_long_double": float(
                    np.max(np.abs(raw - reference))
                ),
                "max_abs_tiled_minus_long_double": float(
                    np.max(np.abs(tiled - reference))
                ),
                "max_vendor_forward_error_ratio": 0.5,
                "max_tiled_forward_error_ratio": 0.0,
            }
        ]
        decimals = [[str(value) for value in reference[:, 0]]]
    else:
        raw = np.empty((m, 0), dtype=np.float64, order="F")
        tiled = np.empty((m, 0), dtype=np.float64, order="F")
        reference = np.empty((m, 0), dtype=np.float64, order="F")
        flagged_columns = []
        checks = []
        comparisons = []
        decimals = []
    count = len(flagged_columns)
    fingerprint = {"xor_hash_uint64": "1", "sum_hash_uint64": "2"}
    original_fingerprints = {
        name: dict(fingerprint) for name in runner.ORIGINAL_FINGERPRINT_KEYS
    }
    protected_fingerprints = {
        name: dict(fingerprint) for name in runner.PROTECTED_FINGERPRINT_KEYS
    }
    diagnostic = {
        "schema_version": 1,
        "integrity_check_eligible": True,
        "diagnostic_executed": True,
        "classification": classification,
        "m": m,
        "n": n,
        "k": 7,
        "minimum_vendor_flops": runner.INTEGRITY_MINIMUM_VENDOR_FLOPS,
        "long_double_reference_accumulator": "C++ long double",
        "long_double_reference_array_storage": (
            "binary64_cast_with_full_precision_decimal_companion"
        ),
        "floating_point_capabilities": {
            "long_double": {
                "sizeof_bytes": 16,
                "digits": 64,
                "digits10": 18,
                "max_digits10": 21,
                "epsilon": 1.0842021724855044e-19,
            },
            "double": {
                "sizeof_bytes": 8,
                "digits": 53,
                "digits10": 15,
                "max_digits10": 17,
                "epsilon": 2.220446049250313e-16,
            },
        },
        "classification_changes_production_decision": False,
        "detailed_column_limit": 16,
        "flagged_column_count": count,
        "captured_flagged_column_count": count,
        "dropped_flagged_column_count": 0,
        "flagged_check_count": len(checks),
        "captured_flagged_check_count": len(checks),
        "dropped_flagged_check_count": 0,
        "first_flagged_column": flagged_columns[0] if flagged else None,
        "first_flagged_check": 2 if flagged else None,
        "flagged_columns": flagged_columns,
        "captured_flagged_columns": flagged_columns,
        "flagged_check_ids": [
            {"column": item["column"], "check": item["check"]} for item in checks
        ],
        "flagged_checks": checks,
        "column_comparisons": comparisons,
        "raw_vendor_columns": raw,
        "deterministic_tiled_columns": tiled,
        "long_double_reference_columns": reference,
        "long_double_reference_decimal_columns": decimals,
        "fingerprints": {
            "original_b": original_fingerprints,
            "protected_b_snapshot": protected_fingerprints,
        },
        "fingerprint_equalities": {
            name: True for name in runner.FINGERPRINT_EQUALITY_KEYS
        },
        "fault_injection": {"enabled": False},
    }
    return diagnostic, output


def test_exact_case_and_runtime_caps_are_fixed_and_fail_closed(tmp_path):
    assert (runner.SEED, runner.M, runner.N, runner.K, runner.THREADS) == (
        70131,
        512,
        512,
        2048,
        2,
    )
    assert runner.LEFT_PAGE_OFFSET == 8
    assert runner.FLOPS == 1_073_741_824
    assert runner.DEFAULT_WORKERS == runner.MAX_WORKERS == 20
    assert runner.MAX_WORKER_TIMEOUT_SECONDS == 40.0
    assert runner.MAX_CONTROLLER_TIMEOUT_SECONDS == 1200.0
    parser = runner._parser()
    base = [
        "--install-prefix", str(tmp_path),
        "--native-module", str(tmp_path / "gxeldcore.fake.so"),
        "--expected-native-sha256", "a" * 64,
        "--output", str(tmp_path / "report.json"),
    ]
    for extra in (
        ["--workers", "21"],
        ["--worker-timeout-seconds", "40.1"],
        ["--controller-timeout-seconds", "1200.1"],
        ["--workers", "20", "--worker-timeout-seconds", "40",
         "--controller-timeout-seconds", "799"],
    ):
        with pytest.raises(SystemExit):
            runner._validate_args(parser, parser.parse_args([*base, *extra]))


def test_native_module_must_be_directly_inside_installed_summit(tmp_path):
    install = tmp_path / "install"
    nested = install / "nested" / "summit"
    nested.mkdir(parents=True)
    native = nested / "gxeldcore.fake.so"
    native.write_bytes(b"nested native")
    expected = hashlib.sha256(native.read_bytes()).hexdigest()
    parser = runner._parser()
    args = parser.parse_args(
        [
            "--install-prefix", str(install),
            "--native-module", str(native),
            "--expected-native-sha256", expected,
            "--dependency-path", str(tmp_path),
            "--workers", "1",
            "--worker-timeout-seconds", "1",
            "--controller-timeout-seconds", "1",
            "--output", str(tmp_path / "new.json"),
        ]
    )
    with pytest.raises(SystemExit):
        runner._validate_args(parser, args)


def test_linkage_rejects_unresolved_dynamic_blas_symbols(tmp_path, monkeypatch):
    module = tmp_path / "gxeldcore.fake.so"
    module.write_bytes(b"placeholder")
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, timeout=20.0):
        del timeout
        if "readelf" in command[0]:
            return " 0x1 (NEEDED) Shared library: [libc.so.6]\n"
        return "                 U cblas_dgemm\n"

    monkeypatch.setattr(runner, "_run_checked", fake_run)
    with pytest.raises(RuntimeError, match="exported or unresolved"):
        runner._linkage_evidence(module)


@pytest.mark.parametrize(
    ("dynamic_output", "symbol_output"),
    [
        (
            " 0x1 (NEEDED) Shared library: [libblis.so.4]\n",
            "0000000000000000 T harmless_symbol\n",
        ),
        (
            " 0x1 (NEEDED) Shared library: [libc.so.6]\n",
            "                 U bli_gemm_ex\n",
        ),
        *(
            (
                " 0x1 (NEEDED) Shared library: [libc.so.6]\n",
                f"                 U {symbol}\n",
            )
            for symbol in (
                "dgemm_", "sgemm_", "xerbla_", "CBLAS_CallFromC", "RowMajorStrg"
            )
        ),
    ],
)
def test_linkage_rejects_dynamic_blis_dependencies_and_symbols(
    tmp_path, monkeypatch, dynamic_output, symbol_output
):
    module = tmp_path / "gxeldcore.fake.so"
    module.write_bytes(b"placeholder")
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, timeout=20.0):
        del timeout
        return dynamic_output if "readelf" in command[0] else symbol_output

    monkeypatch.setattr(runner, "_run_checked", fake_run)
    with pytest.raises(RuntimeError, match="dynamic BLAS|exported or unresolved"):
        runner._linkage_evidence(module)


def _fake_native_module(build_info):
    module = SimpleNamespace(build_info=lambda: dict(build_info))
    setattr(module, runner.DIAGNOSTIC_API, lambda *_args: None)
    return module


def _common_build_info():
    return {
        "api_version": 8,
        "backend_version": "1.5",
        "blas_runtime_isolation": "private_static",
        "gemm_integrity_enabled": True,
        "gemm_vendor_entry_outer_openmp_guard": True,
        "source_tree_sha256": "1" * 64,
        "gemm_integrity_minimum_vendor_flops": (
            runner.INTEGRITY_MINIMUM_VENDOR_FLOPS
        ),
    }


def _blis_build_info(archive_sha256):
    return {
        **_common_build_info(),
        "blas_vendor": "BLIS",
        "gemm_execution_mode": "serialized_fixed_private_blis",
        "blas_runtime_config": "BLIS 2.0 config=zen",
        "blas_runtime_corename": "zen",
        "blas_runtime_threads": 2,
        "blas_runtime_threading_layer": "pthreads",
        "blas_runtime_worker_affinity_policy": (
            "inherit_authenticated_selected_cpu_set_per_call"
        ),
        "private_openblas_archive_sha256": "none",
        "private_blas_backend": "upstream_blis",
        "private_blas_archive_sha256": archive_sha256,
        "private_blas_source_commit": runner.BLIS_SOURCE_COMMIT,
        "private_blas_source_tree_sha256": runner.BLIS_SOURCE_TREE_SHA256,
        "private_blas_config_family": "zen",
        "private_blas_header_sha256": runner.BLIS_HEADER_SHA256,
        "private_blas_cblas_header_sha256": runner.BLIS_CBLAS_HEADER_SHA256,
        "blas_runtime_thread_strategy": "automatic",
        "blas_runtime_thread_ways": {
            "jc": 1, "pc": 1, "ic": 1, "jr": 1, "ir": 1,
        },
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_owner_thread_configured": True,
        "blas_runtime_environment_immutable": True,
        "blas_runtime_environment_contract": "blis_process_start_v1",
        "blas_runtime_tls_enabled": True,
    }


def _placement_attestation(cpus=(2, 3)):
    return {
        "schema": runner.OPENMP_PLACEMENT_SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": 2,
        "expected_cpu_ids": list(cpus),
        "omp_dynamic": False,
        "omp_thread_limit": 2,
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": 2,
        "effective_openmp_capacity": 2,
        "place_cpu_ids": [[cpu] for cpu in cpus],
        "team_size": 2,
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "workers": [
            {
                "thread_num": index,
                "place_num": index,
                "place_cpu_ids": [cpu],
                "sched_affinity_cpu_ids": [cpu],
                "current_cpu": cpu,
                "verified": True,
            }
            for index, cpu in enumerate(cpus)
        ],
        "vendor_calls": 0,
    }


def _placement_build_info(build, attestation):
    return {
        **build,
        "api_version": 9,
        "backend_version": "1.9",
        "openmp_effective_capacity_policy": (
            "bound_places_else_sched_affinity_v1"
        ),
        "openmp_placement_contract_supported": True,
        "openmp_placement_contract_schema": runner.OPENMP_PLACEMENT_SCHEMA,
        "openmp_placement_contract_configured": True,
        "openmp_placement_contract_immutable": True,
        "openmp_placement_probe_vendor_calls": 0,
        "openmp_placement_contract_evidence": attestation,
    }


def test_openblas_build_contract_remains_the_default(tmp_path):
    module_path = tmp_path / "gxeldcore.fake.so"
    module_path.write_bytes(b"openblas module")
    archive_sha256 = "a" * 64
    build = {
        **_common_build_info(),
        "private_openblas_archive_sha256": archive_sha256,
    }
    evidence = runner._validate_build_info(
        _fake_native_module(build), module_path, archive_sha256
    )
    assert evidence["build_info"] == build
    assert runner._parser().get_default("expected_backend") == "openblas"
    assert runner._parser().get_default("require_openmp_placement") is False


def test_explicit_placement_requires_exact_api9_build_and_stored_evidence(
    tmp_path,
):
    module_path = tmp_path / "gxeldcore.fake.so"
    module_path.write_bytes(b"placement module")
    archive_sha256 = "a" * 64
    placement = _placement_attestation()
    build = _placement_build_info(
        {
            **_common_build_info(),
            "private_openblas_archive_sha256": archive_sha256,
        },
        placement,
    )
    evidence = runner._validate_build_info(
        _fake_native_module(build),
        module_path,
        archive_sha256,
        placement_attestation=placement,
    )
    assert evidence["build_info"] == build

    with pytest.raises(RuntimeError, match="native build contract"):
        runner._validate_build_info(
            _fake_native_module(build), module_path, archive_sha256
        )
    with pytest.raises(RuntimeError, match="native build contract"):
        runner._validate_build_info(
            _fake_native_module(
                {
                    **build,
                    "api_version": 8,
                    "backend_version": "1.5",
                }
            ),
            module_path,
            archive_sha256,
            placement_attestation=placement,
        )

    changed_evidence = _placement_attestation()
    changed_evidence["workers"][0]["current_cpu"] = 3
    with pytest.raises(RuntimeError, match="OpenMP placement"):
        runner._validate_build_info(
            _fake_native_module(
                {**build, "openmp_placement_contract_evidence": changed_evidence}
            ),
            module_path,
            archive_sha256,
            placement_attestation=placement,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("omp_max_active_levels", 2),
        ("omp_max_active_levels", True),
        ("omp_proc_bind", "true"),
        ("effective_openmp_capacity", 1),
        ("verified", 1),
        ("vendor_calls", False),
        ("expected_cpu_ids", [3, 2]),
    ],
)
def test_placement_attestation_fails_closed_on_value_and_type_drift(
    field, invalid
):
    placement = _placement_attestation()
    placement[field] = invalid
    with pytest.raises(RuntimeError, match="OpenMP placement"):
        runner._validate_openmp_placement_attestation(placement, [2, 3])


def test_blis_build_contract_requires_exact_provenance_and_threading(tmp_path):
    module_path = tmp_path / "gxeldcore.fake.so"
    module_path.write_bytes(b"blis module")
    archive_sha256 = "a" * 64
    build = _blis_build_info(archive_sha256)
    evidence = runner._validate_build_info(
        _fake_native_module(build), module_path, archive_sha256, "blis"
    )
    assert evidence["build_info"] == build


def test_blis_api9_build_contract_accepts_exact_explicit_placement(tmp_path):
    module_path = tmp_path / "gxeldcore.fake.so"
    module_path.write_bytes(b"blis placement module")
    archive_sha256 = "a" * 64
    placement = _placement_attestation()
    build = _placement_build_info(
        _blis_build_info(archive_sha256), placement
    )
    evidence = runner._validate_build_info(
        _fake_native_module(build), module_path, archive_sha256, "blis",
        placement,
    )
    assert evidence["build_info"] == build


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("private_blas_archive_sha256", "f" * 64),
        ("blas_runtime_config", "BLIS 2.1 config=zen"),
        ("blas_runtime_corename", "generic"),
        ("blas_runtime_threads", 1),
        ("blas_runtime_threads", True),
        ("blas_runtime_threading_layer", "openmp"),
        ("private_blas_config_family", "generic"),
        ("blas_runtime_thread_strategy", "manual"),
        ("blas_runtime_thread_ways", {"jc": 1, "pc": 1}),
        (
            "blas_runtime_thread_ways",
            {"jc": 2, "pc": 1, "ic": 1, "jr": 1, "ir": 1},
        ),
        (
            "blas_runtime_thread_ways",
            {"jc": True, "pc": 1, "ic": 1, "jr": 1, "ir": 1},
        ),
        ("blas_runtime_owner_thread_enforced", False),
        ("blas_runtime_owner_thread_enforced", 1),
        ("blas_runtime_owner_thread_configured", False),
        ("blas_runtime_owner_thread_configured", "yes"),
        ("blas_runtime_environment_immutable", False),
        ("blas_runtime_environment_immutable", 1),
        ("blas_runtime_tls_enabled", False),
        ("blas_runtime_tls_enabled", 1),
        ("gemm_integrity_enabled", 1),
        ("api_version", 8.0),
        ("private_blas_source_commit", "not-a-commit"),
        ("private_blas_source_commit", "a" * 40),
        ("private_blas_source_tree_sha256", "b" * 64),
        ("private_blas_header_sha256", "not-a-sha256"),
        ("private_blas_header_sha256", "c" * 64),
        ("private_blas_cblas_header_sha256", "d" * 64),
    ],
)
def test_blis_build_contract_fails_closed_on_backend_drift(
    tmp_path, field, invalid
):
    module_path = tmp_path / "gxeldcore.fake.so"
    module_path.write_bytes(b"blis module")
    archive_sha256 = "a" * 64
    build = _blis_build_info(archive_sha256)
    build[field] = invalid
    with pytest.raises(RuntimeError, match="BLIS|native build contract"):
        runner._validate_build_info(
            _fake_native_module(build), module_path, archive_sha256, "blis"
        )


def test_blis_worker_environment_is_fixed_before_import(monkeypatch):
    for name in runner.BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR:
        monkeypatch.setenv(name, "uncontrolled")
    monkeypatch.setenv("BLIS_NUM_THREADS", "99")

    openblas_environment, openblas_settings = runner._worker_environment()
    assert "BLIS_NUM_THREADS" not in openblas_settings
    assert openblas_environment["BLIS_NUM_THREADS"] == "99"
    assert all(
        openblas_environment[name] == "uncontrolled"
        for name in runner.BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR
    )

    blis_environment, blis_settings = runner._worker_environment("blis")
    assert blis_settings["BLIS_NUM_THREADS"] == "2"
    assert blis_environment["BLIS_NUM_THREADS"] == "2"
    assert all(
        name not in blis_environment
        for name in runner.BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR
    )
    runner._validate_blis_process_environment(blis_environment)

    with pytest.raises(RuntimeError, match="BLIS_NUM_THREADS"):
        runner._validate_blis_process_environment(
            {**blis_environment, "BLIS_NUM_THREADS": "1"}
        )
    with pytest.raises(RuntimeError, match="process-start overrides"):
        runner._validate_blis_process_environment(
            {**blis_environment, "BLIS_JC_NT": "2"}
        )


def test_explicit_placement_environment_is_process_start_fixed(monkeypatch):
    for name in (
        "OMP_PLACES", "OMP_NESTED", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
        "KMP_HW_SUBSET", "KMP_PLACE_THREADS",
    ):
        monkeypatch.setenv(name, "uncontrolled")
    environment, settings = runner._worker_environment("blis", [2, 3])
    assert settings["OMP_PROC_BIND"] == "SPREAD"
    assert settings["OMP_PLACES"] == "{2},{3}"
    assert environment["OMP_THREAD_LIMIT"] == "2"
    assert environment["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert all(
        name not in environment
        for name in (
            "OMP_NESTED", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
            "KMP_HW_SUBSET", "KMP_PLACE_THREADS",
        )
    )
    runner._validate_openmp_placement_process_environment(environment, [2, 3])
    with pytest.raises(RuntimeError, match="process-start environment"):
        runner._validate_openmp_placement_process_environment(
            {**environment, "OMP_PROC_BIND": "TRUE"}, [2, 3]
        )


def test_history_schedule_is_declared_deterministic_and_never_expands_workers():
    modes = ("none", "allocation-churn", "subthreshold-nn")
    schedule = runner._schedule(7, modes)
    assert [item["worker_index"] for item in schedule] == list(range(7))
    assert [item["history_mode"] for item in schedule] == [
        "none", "allocation-churn", "subthreshold-nn",
        "none", "allocation-churn", "subthreshold-nn", "none",
    ]


def test_misaligned_fortran_fixture_matches_original_seed_and_page_offset(monkeypatch):
    monkeypatch.setattr(runner, "LEFT_PAGE_OFFSET", 8)
    rng_observed = np.random.default_rng(70131)
    observed = runner._misaligned_fortran_normal(rng_observed, (7, 11), np)
    rng_expected = np.random.default_rng(70131)
    expected = rng_expected.normal(size=(7, 11))
    assert observed.flags.f_contiguous
    assert observed.flags.aligned
    assert int(observed.ctypes.data) % os.sysconf("SC_PAGE_SIZE") == 8
    np.testing.assert_array_equal(observed, expected)


class _HistoryModule:
    def __init__(self):
        self.records = []

    def reset_gemm_telemetry(self):
        self.records.clear()

    def consume_gemm_telemetry(self):
        records = list(self.records)
        self.records.clear()
        return records

    def gemm_telemetry_status(self):
        return {"buffered_records": len(self.records), "dropped_records": 0}

    @staticmethod
    def build_info():
        return {"gemm_integrity_minimum_vendor_flops": 1_000_000_000}

    @staticmethod
    def protected_matmul_nn(left, right, threads):
        assert threads == 2
        return np.empty((left.shape[0], right.shape[1]), dtype=np.float64, order="F"), 0

    @staticmethod
    def protected_matmul_tn(left, right, threads):
        assert threads == 2
        return np.empty((left.shape[1], right.shape[1]), dtype=np.float64, order="F"), 0


@pytest.mark.parametrize("mode", ["subthreshold-nn", "subthreshold-tn"])
def test_subthreshold_history_records_one_logical_call_and_zero_vendor_calls(mode):
    record = runner._run_history(mode, _HistoryModule(), np)
    assert record["completed"] is True
    assert record["protected_call_count"] == 1
    assert record["vendor_call_count"] == 0
    assert record["integrity_threshold_crossed"] is False
    assert record["repaired_columns"] == 0


def test_compaction_preserves_classification_evidence_without_numeric_columns(monkeypatch):
    monkeypatch.setattr(runner, "M", 3)
    monkeypatch.setattr(runner, "N", 4)
    monkeypatch.setattr(runner, "K", 7)
    diagnostic, output = _fake_diagnostic(
        classification=(
            "checksum_tolerance_false_positive_under_cancellation_aware_bound"
        ),
        flagged=True,
    )
    compact = runner._compact_diagnostic(diagnostic, output, np)
    assert compact["classification_contract_valid"] is True
    assert compact["all_repairs_classified"] is True
    assert compact["safe_diagnostic_classification"] is True
    assert compact["raw_numeric_columns_persisted"] is True
    assert compact["column_arrays"]["raw_vendor_columns"]["shape"] == [3, 1]
    assert compact["captured_numeric_column_witnesses"][
        "raw_vendor_columns"
    ]["columns"] == [diagnostic["raw_vendor_columns"][:, 0].tolist()]
    assert len(
        compact["long_double_reference_decimal_column_summaries"][0][
            "canonical_json_sha256"
        ]
    ) == 64
    assert compact["long_double_reference_decimal_columns"] == diagnostic[
        "long_double_reference_decimal_columns"
    ]
    assert not any(isinstance(value, np.ndarray) for value in compact.values())


def test_compaction_rejects_inconsistent_detail_caps(monkeypatch):
    monkeypatch.setattr(runner, "M", 3)
    monkeypatch.setattr(runner, "N", 4)
    monkeypatch.setattr(runner, "K", 7)
    diagnostic, output = _fake_diagnostic(
        classification="unclassified_detail_cap_exceeded", flagged=True
    )
    diagnostic["flagged_column_count"] = 2
    diagnostic["flagged_columns"] = [1, 2]
    diagnostic["dropped_flagged_column_count"] = 0
    with pytest.raises(RuntimeError, match="captured-column count"):
        runner._compact_diagnostic(diagnostic, output, np)


def test_safe_checksum_label_cannot_override_forward_error_evidence(monkeypatch):
    monkeypatch.setattr(runner, "M", 3)
    monkeypatch.setattr(runner, "N", 4)
    monkeypatch.setattr(runner, "K", 7)
    diagnostic, output = _fake_diagnostic(
        classification=(
            "checksum_tolerance_false_positive_under_cancellation_aware_bound"
        ),
        flagged=True,
    )
    diagnostic["column_comparisons"][0][
        "vendor_rows_outside_forward_error_bound"
    ] = 1
    compact = runner._compact_diagnostic(diagnostic, output, np)
    assert compact["native_classification"] == (
        "checksum_tolerance_false_positive_under_cancellation_aware_bound"
    )
    assert compact["classification"] == "unclassified"
    assert compact["classification_contract_valid"] is False
    assert compact["safe_diagnostic_classification"] is False


def test_classification_requires_true_extended_long_double_precision(monkeypatch):
    monkeypatch.setattr(runner, "M", 3)
    monkeypatch.setattr(runner, "N", 4)
    monkeypatch.setattr(runner, "K", 7)
    diagnostic, output = _fake_diagnostic(
        classification="no_current_gate_flags", flagged=False
    )
    diagnostic["floating_point_capabilities"]["long_double"]["digits"] = 53
    with pytest.raises(RuntimeError, match="more precision bits"):
        runner._compact_diagnostic(diagnostic, output, np)


def test_safe_classification_reconstructs_current_and_cancellation_bounds(monkeypatch):
    monkeypatch.setattr(runner, "M", 3)
    monkeypatch.setattr(runner, "N", 4)
    monkeypatch.setattr(runner, "K", 7)
    diagnostic, output = _fake_diagnostic(
        classification=(
            "checksum_tolerance_false_positive_under_cancellation_aware_bound"
        ),
        flagged=True,
    )
    diagnostic["flagged_checks"][0]["current_relative_bound"] *= 2.0
    compact = runner._compact_diagnostic(diagnostic, output, np)
    assert compact["check_numeric_evidence_consistent"] is False
    assert compact["classification"] == "unclassified"
    assert compact["safe_diagnostic_classification"] is False


def test_dense_oracle_gate_uses_current_exact_test_tolerance():
    left = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])
    right = np.asfortranarray([[5.0, 6.0], [7.0, 8.0]])
    exact = np.asfortranarray(left @ right)
    passed = runner._dense_oracle_comparison(left, right, exact, np)
    assert passed["passed"] is True
    assert passed["rtol"] == 3.0e-14
    assert passed["atol"] == 3.0e-12
    damaged = exact.copy(order="F")
    damaged[0, 0] += 1.0e-6
    failed = runner._dense_oracle_comparison(left, right, damaged, np)
    assert failed["passed"] is False
    assert failed["maximum_tolerance_normalized_error"] > 1.0


def test_telemetry_gate_requires_active_cores_and_fully_local_numa_pages():
    affinity = {"allowed_cpus": [2, 3], "numa_nodes": [0]}
    operand = {
        "query_status": "queried",
        "selected_sample_pages": 8,
        "resolved_sample_pages": 8,
        "page_query_error_pages": 0,
        "node_histogram": {"0": 8},
    }
    record = {
        "wall_seconds": 0.1,
        "process_cpu_seconds": 0.19,
        "gflops_per_second": 10.0,
        "process_cpu_to_wall_ratio": 1.9,
        "active_core_equivalents": 1.9,
        "entry_cpu": 2,
        "exit_cpu": 3,
        "operand_numa_page_samples": {
            "sampling_method": "move_pages_query_no_migration",
            "operands": {name: dict(operand) for name in ("a", "b", "c")},
        },
    }
    assert runner._validate_telemetry_gate(record, affinity)["passed"] is True
    record["active_core_equivalents"] = 1.0
    record["operand_numa_page_samples"]["operands"]["b"][
        "node_histogram"
    ] = {"1": 8}
    rejected = runner._validate_telemetry_gate(record, affinity)
    assert rejected["passed"] is False
    assert "active_core_equivalents_below_75_percent_of_threads" in rejected[
        "reasons"
    ]
    assert "numa_operand_b_not_fully_local_and_resolved" in rejected["reasons"]


def _controller_args(tmp_path: Path, *, workers: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        install_prefix=tmp_path,
        native_module=tmp_path / "gxeldcore.fake.so",
        expected_native_sha256="a" * 64,
        expected_archive_sha256="b" * 64,
        expected_backend="openblas",
        require_openmp_placement=False,
        python_executable=Path(sys.executable),
        dependency_path=[tmp_path],
        cpus=[0, 1],
        workers=workers,
        history_modes=("none",),
        worker_timeout_seconds=1.0,
        controller_timeout_seconds=10.0,
        output=tmp_path / "report.json",
        dry_run=False,
        _worker=False,
        _worker_index=None,
        _history_mode=None,
    )


def test_worker_command_preserves_default_openblas_bytes_and_opts_blis_in(
    tmp_path,
):
    args = _controller_args(tmp_path, workers=1)
    dependencies = [tmp_path / "dependency"]
    openblas = runner._worker_command(args, 0, "none", dependencies)
    assert "--expected-backend" not in openblas

    del args.expected_backend
    assert runner._worker_command(args, 0, "none", dependencies) == openblas

    args.expected_backend = "blis"
    blis = runner._worker_command(args, 0, "none", dependencies)
    assert blis[:-2] == openblas
    assert blis[-2:] == ["--expected-backend", "blis"]

    args.expected_backend = "openblas"
    args.require_openmp_placement = True
    placed = runner._worker_command(args, 0, "none", dependencies)
    assert placed[:-1] == openblas
    assert placed[-1] == "--require-openmp-placement"


def _fake_worker(index: int, classification: str, repairs: int) -> dict:
    return {
        "status": "ok",
        "worker_index": index,
        "pid": 1000 + index,
        "history_mode": "none",
        "runtime_attestation": {
            "native": {"build_info_canonical_json_sha256": "c" * 64},
            "numpy_version": "2.3.2",
            "numpy_module": {
                "path": "/declared/numpy/__init__.py",
                "sha256": "f" * 64,
            },
            "python_executable": "/declared/python",
        },
        "repair": {
            "repaired_columns": repairs,
            "classification": classification,
            "all_repairs_classified": classification not in {
                "unclassified", "unclassified_detail_cap_exceeded"
            },
            "safe_diagnostic_classification": (
                classification in runner.SAFE_CLASSIFICATIONS
            ),
        },
        "telemetry": {"gate": {"passed": True}},
        "correctness": {"dense_numpy_oracle": {"passed": True}},
        "diagnostic": {
            "classification": classification,
            "classification_contract_valid": True,
            "flagged_column_count": repairs,
            "captured_flagged_column_count": min(repairs, 16),
            "all_repairs_classified": classification not in {
                "unclassified", "unclassified_detail_cap_exceeded"
            },
            "safe_diagnostic_classification": (
                classification in runner.SAFE_CLASSIFICATIONS
            ),
            "raw_numeric_columns_persisted": repairs > 0,
        },
    }


def test_run_one_worker_propagates_placement_command_and_environment(
    tmp_path, monkeypatch
):
    args = _controller_args(tmp_path, workers=1)
    args.require_openmp_placement = True
    cpus = [2, 3]
    placement = _placement_attestation(cpus)
    result = _fake_worker(0, "no_current_gate_flags", 0)
    result["pid"] = 4321
    result["runtime_attestation"].update(
        {
            "python_no_site": True,
            "python_executable": str(args.python_executable.resolve()),
            "native": {
                "module_path": str(args.native_module.resolve()),
                "module_sha256": args.expected_native_sha256,
                "build_info_canonical_json_sha256": "c" * 64,
                "build_info": {
                    "openmp_placement_contract_evidence": placement,
                },
            },
            "affinity": {"allowed_cpus": cpus},
            "openmp_placement": placement,
            "post_configuration_calling_thread_affinity": {
                "allowed_cpus": [cpus[0]],
            },
        }
    )
    captured = {}

    class FakeProcess:
        pid = 4321
        returncode = 0

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return (
                runner.RESULT_PREFIX
                + json.dumps(result, sort_keys=True, allow_nan=False)
                + "\n",
                "",
            )

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    observed = runner._run_one_worker(
        args, 0, "none", [tmp_path], cpus, 1.0
    )
    assert observed["status"] == "ok"
    assert "--require-openmp-placement" in captured["command"]
    assert captured["environment"]["OMP_PROC_BIND"] == "SPREAD"
    assert captured["environment"]["OMP_PLACES"] == "{2},{3}"
    assert captured["environment"]["OMP_THREAD_LIMIT"] == "2"
    monkeypatch.setattr(runner, "_physical_core_key", lambda cpu: (0, cpu))
    report = runner._base_report(args, cpus, [tmp_path], {})
    assert report["worker_thread_environment"]["OMP_PROC_BIND"] == "SPREAD"
    assert report["worker_thread_environment"]["OMP_PLACES"] == "{2},{3}"
    assert report["placement"]["ordered_singleton_places"] == [[2], [3]]


def test_controller_never_marks_diagnostic_as_production_accepted(tmp_path, monkeypatch):
    args = _controller_args(tmp_path)
    provenance = {
        "installed_package": {"manifest_sha256": "d" * 64},
        "runner": {"sha256": "e" * 64},
    }
    monkeypatch.setattr(runner, "_resolve_cpus", lambda _value: [0, 1])
    monkeypatch.setattr(runner, "_dependency_paths", lambda _args: [tmp_path])
    monkeypatch.setattr(runner, "_static_provenance", lambda *_args: provenance)
    monkeypatch.setattr(
        runner, "_package_identity", lambda _path: provenance["installed_package"]
    )
    monkeypatch.setattr(runner, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(runner, "_file_identity", lambda _path: provenance["runner"])
    monkeypatch.setattr(
        runner,
        "_run_one_worker",
        lambda _args, index, _history, _deps, _cpus, _timeout: _fake_worker(
            index, "no_current_gate_flags", 0
        ),
    )
    report = runner._controller(args)
    assert report["protocol_complete"] is True
    assert report["event_reproduced"] is False
    assert report["event_classified"] is False
    assert report["root_cause_resolved"] is False
    assert report["accepted"] is False
    assert report["status"] == "completed_no_repair_observed"


def test_controller_fails_closed_on_unclassified_repair(tmp_path, monkeypatch):
    args = _controller_args(tmp_path, workers=1)
    provenance = {
        "installed_package": {"manifest_sha256": "d" * 64},
        "runner": {"sha256": "e" * 64},
    }
    monkeypatch.setattr(runner, "_resolve_cpus", lambda _value: [0, 1])
    monkeypatch.setattr(runner, "_dependency_paths", lambda _args: [tmp_path])
    monkeypatch.setattr(runner, "_static_provenance", lambda *_args: provenance)
    monkeypatch.setattr(
        runner, "_package_identity", lambda _path: provenance["installed_package"]
    )
    monkeypatch.setattr(runner, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(runner, "_file_identity", lambda _path: provenance["runner"])
    monkeypatch.setattr(
        runner,
        "_run_one_worker",
        lambda _args, index, _history, _deps, _cpus, _timeout: _fake_worker(
            index, "unclassified", 2
        ),
    )
    report = runner._controller(args)
    assert report["protocol_complete"] is True
    assert report["event_reproduced"] is True
    assert report["event_classified"] is False
    assert report["unclassified_repairs"] == 2
    assert report["diagnostic_gate_passed"] is False
    assert report["accepted"] is False
    assert report["status"] == "completed_rejected_integrity_event"


def test_vendor_corruption_can_be_classified_but_never_passes_gate(
    tmp_path, monkeypatch
):
    args = _controller_args(tmp_path, workers=1)
    provenance = {
        "installed_package": {"manifest_sha256": "d" * 64},
        "runner": {"sha256": "e" * 64},
    }
    monkeypatch.setattr(runner, "_resolve_cpus", lambda _value: [0, 1])
    monkeypatch.setattr(runner, "_dependency_paths", lambda _args: [tmp_path])
    monkeypatch.setattr(runner, "_static_provenance", lambda *_args: provenance)
    monkeypatch.setattr(
        runner, "_package_identity", lambda _path: provenance["installed_package"]
    )
    monkeypatch.setattr(runner, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(runner, "_file_identity", lambda _path: provenance["runner"])
    monkeypatch.setattr(
        runner,
        "_run_one_worker",
        lambda _args, index, _history, _deps, _cpus, _timeout: _fake_worker(
            index, "vendor_result_outside_forward_error_bound", 1
        ),
    )
    report = runner._controller(args)
    assert report["protocol_complete"] is True
    assert report["event_reproduced"] is True
    assert report["event_classified"] is True
    assert report["root_cause_evidence_complete"] is True
    assert report["root_cause_resolved"] is True
    assert report["diagnostic_gate_passed"] is False
    assert report["accepted"] is False


def test_main_returns_nonzero_after_publishing_completed_rejected_evidence(
    tmp_path, monkeypatch
):
    published = []
    monkeypatch.setattr(runner, "_validate_args", lambda _parser, _args: None)
    monkeypatch.setattr(
        runner,
        "_controller",
        lambda _args: {
            "accepted": False,
            "protocol_complete": True,
            "diagnostic_gate_passed": False,
        },
    )
    monkeypatch.setattr(
        runner,
        "_atomic_json_no_replace",
        lambda payload, output: published.append((payload, output)),
    )
    output = tmp_path / "rejected.json"
    code = runner.main(
        [
            "--install-prefix", str(tmp_path),
            "--native-module", str(tmp_path / "gxeldcore.fake.so"),
            "--expected-native-sha256", "a" * 64,
            "--output", str(output),
        ]
    )
    assert code == 2
    assert len(published) == 1
    assert published[0][0]["protocol_complete"] is True


def test_main_deadline_covers_preflight_execution_and_publication(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "_validate_args", lambda _parser, _args: None)
    monkeypatch.setattr(runner, "_controller", lambda _args: time.sleep(0.2))
    published = []
    monkeypatch.setattr(
        runner,
        "_atomic_json_no_replace",
        lambda payload, output: published.append((payload, output)),
    )
    with pytest.raises(runner._ControllerDeadlineExpired):
        runner.main(
            [
                "--install-prefix", str(tmp_path),
                "--native-module", str(tmp_path / "gxeldcore.fake.so"),
                "--expected-native-sha256", "a" * 64,
                "--worker-timeout-seconds", "0.001",
                "--controller-timeout-seconds", "0.02",
                "--output", str(tmp_path / "deadline.json"),
            ]
        )
    assert published == []
    assert not (tmp_path / "deadline.json").exists()


def test_dry_run_publishes_one_no_replace_report_without_execution(tmp_path, monkeypatch):
    install = tmp_path / "install"
    package = install / "summit"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    native = package / "gxeldcore.fake.so"
    native.write_bytes(b"dry-run placeholder")
    native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
    output = tmp_path / "diagnostic-dry-run.json"
    cpus = runner._default_physical_cpus()
    if len(cpus) < 2:
        pytest.skip("two physical CPUs are unavailable")
    monkeypatch.setattr(
        runner,
        "_linkage_evidence",
        lambda _path: {
            "dynamic_blas_dependencies": [],
            "exported_blas_symbol_count": 0,
            "private_static_linkage_verified": True,
        },
    )
    arguments = [
        "--install-prefix", str(install),
        "--native-module", str(native),
        "--expected-native-sha256", native_sha,
        "--python-executable", sys.executable,
        "--dependency-path", str(tmp_path),
        "--cpus", ",".join(map(str, cpus)),
        "--workers", "2",
        "--history-modes", "none,allocation-churn",
        "--worker-timeout-seconds", "1",
        "--controller-timeout-seconds", "2",
        "--output", str(output),
        "--dry-run",
    ]
    assert runner.main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "validated_dry_run"
    assert report["accepted"] is False
    assert report["scientific_case_executed"] is False
    assert report["history_schedule"] == [
        {"worker_index": 0, "history_mode": "none"},
        {"worker_index": 1, "history_mode": "allocation-churn"},
    ]
    assert report["provenance"][
        "expected_private_openblas_archive_sha256"
    ] == runner.DEFAULT_ARCHIVE_SHA256
    assert "expected_private_blas_backend" not in report["provenance"]
    assert "BLIS_NUM_THREADS" not in report["worker_thread_environment"]
    assert report["worker_thread_environment"]["OMP_PROC_BIND"] == "FALSE"
    assert "OMP_PLACES" not in report["worker_thread_environment"]
    assert set(report["placement"]) == {
        "taskset_required", "selected_cpus", "physical_core_keys",
        "smt_siblings_excluded",
    }
    command = report["planned_worker_command_template"]
    assert command[1:3] == ["-c", runner._compress_cpu_list(cpus)]
    assert command[3:5] == [str(Path(sys.executable).resolve()), "-S"]
    assert "--expected-backend" not in command
    assert output.stat().st_mode & 0o777 == 0o600

    blis_output = tmp_path / "diagnostic-blis-dry-run.json"
    blis_arguments = list(arguments)
    blis_arguments[blis_arguments.index(str(output))] = str(blis_output)
    blis_arguments.extend(
        ["--expected-backend", "blis", "--expected-archive-sha256", "c" * 64]
    )
    assert runner.main(blis_arguments) == 0
    blis_report = json.loads(blis_output.read_text(encoding="utf-8"))
    assert blis_report["provenance"]["expected_private_blas_backend"] == (
        "upstream_blis"
    )
    assert blis_report["provenance"][
        "expected_private_blas_archive_sha256"
    ] == "c" * 64
    assert "expected_private_openblas_archive_sha256" not in blis_report[
        "provenance"
    ]
    assert blis_report["planned_worker_command_template"][-2:] == [
        "--expected-backend", "blis",
    ]
    assert blis_report["worker_thread_environment"]["BLIS_NUM_THREADS"] == "2"
    with pytest.raises(SystemExit):
        runner.main(arguments)


def test_no_replace_publication_rejects_dangling_output_symlink(tmp_path):
    target = tmp_path / "escaped-target.json"
    output = tmp_path / "report.json"
    output.symlink_to(target)
    with pytest.raises(FileExistsError, match="refusing existing report"):
        runner._atomic_json_no_replace({"accepted": False}, output)
    assert output.is_symlink()
    assert not target.exists()
