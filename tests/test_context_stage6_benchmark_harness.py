from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "context"
    / "benchmark_context_native_stage6.py"
)
SPEC = importlib.util.spec_from_file_location("stage6_benchmark_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


INVARIANTS = {
    "phase_wall_within_run": True,
    "phase_process_cpu_within_run": True,
    "category_wall_within_active_phases": True,
    "category_process_cpu_within_active_phases": True,
    "descriptor_accounting_exact": True,
    "protected_calls_match_semantic_ledger": True,
    "scientific_state_unchanged_verified": True,
    "report_contains_scientific_ndarray": False,
    "instrumentation_changes_execution_plan": False,
}
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


def _report() -> dict[str, object]:
    report = {key: {} for key in REPORT_KEYS}
    report.update(
        {
            "schema": "contextual_native_performance_report_v1",
            "schema_version": 1,
            "metadata_only": True,
            "execution_mode": "reference_v1",
            "invariants": dict(INVARIANTS),
            "categories_are_nonoverlapping": True,
            "categories_cover_entire_run": False,
        }
    )
    return report


def _qualification(
    *, source_commit: str, source_tree: str, native_sha256: str
) -> dict[str, object]:
    planner = HARNESS._planner_module()
    return {
        "schema": HARNESS.FAULT_QUALIFICATION_SCHEMA,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree,
        "native_module_sha256": native_sha256,
        "integrity_policy": "full_scalar_witness_v1",
        "integrity_backend": "deterministic_tiled_fp64_with_scalar_witness_v1",
        "fault_coverage_contract": HARNESS.FAULT_COVERAGE_CONTRACT,
        "covered_operations": list(planner.PROTECTED_OPERATION_KEYS_V1),
        "recoverable_fault_modes": list(HARNESS.RECOVERABLE_FAULT_MODES),
        "terminal_fault_modes": list(HARNESS.TERMINAL_FAULT_MODES),
        "release_fault_matrix_passed": True,
        "focused_test_count": 72,
        "full_regression_test_count": 350,
        "asan_ubsan_status": "passed",
        "ubsan_status": "passed",
    }


def test_performance_report_boundary_is_closed_and_accepts_false_invariants() -> None:
    report = _report()
    HARNESS._performance_report_required(report, "reference_v1")

    missing = dict(report)
    missing.pop("units")
    with pytest.raises(ValueError, match="top-level keys mismatch"):
        HARNESS._performance_report_required(missing, "reference_v1")

    extra = {**report, "unexpected": 1}
    with pytest.raises(ValueError, match="top-level keys mismatch"):
        HARNESS._performance_report_required(extra, "reference_v1")

    wrong_invariant = dict(report)
    wrong_invariant["invariants"] = {
        **INVARIANTS,
        "report_contains_scientific_ndarray": True,
    }
    with pytest.raises(ValueError, match="invariant mapping mismatch"):
        HARNESS._performance_report_required(wrong_invariant, "reference_v1")


def test_exact_trace_mode_rejects_non_bitwise_science_within_tolerance(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.npz"
    actual = tmp_path / "actual.npz"
    np.savez(baseline, value=np.asarray([1.0], dtype=np.float64))
    np.savez(actual, value=np.asarray([1.0 + 1.0e-12], dtype=np.float64))
    comparison = HARNESS._compare_capsules(actual, baseline)
    assert comparison["all_arrays_exact"] is False
    assert comparison["all_arrays_within_tolerance"] is True
    assert not HARNESS._comparison_passes(comparison, "exact_trace_v1", True)
    assert HARNESS._comparison_passes(comparison, "science_tolerance_v1", False)


def test_protected_shape_histogram_is_combined_without_losing_strides() -> None:
    planner = HARNESS._planner_module()

    def operations(active: bool) -> dict[str, dict[str, object]]:
        result = {}
        for name in planner.PROTECTED_OPERATION_KEYS_V1:
            item: dict[str, object] = {
                "calls": 0,
                "logical_flop_count": 0.0,
                "primary_wall_ns": 0,
                "shape_histogram": [],
            }
            if active and name == "source_tn":
                item = {
                    "calls": 3,
                    "logical_flop_count": 120.0,
                    "primary_wall_ns": 12,
                    "shape_histogram": [
                        {
                            "transpose_left": True,
                            "rows": 2,
                            "columns": 5,
                            "reduction": 7,
                            "left_stride": 9,
                            "right_stride": 8,
                            "output_stride": 6,
                            "calls": 3,
                        }
                    ],
                }
            result[name] = item
        return result

    worker = {
        "reference": {"performance_report": {"operations": operations(True)}},
        "trait": {"performance_report": {"operations": operations(False)}},
    }
    counts, shapes, throughput = HARNESS._protected_sample(worker)
    assert counts["source_tn"] == 3
    assert shapes["source_tn"]["shape_histogram"] == [
        {
            "transpose_left": True,
            "rows": 2,
            "columns": 5,
            "reduction": 7,
            "left_stride": 9,
            "right_stride": 8,
            "output_stride": 6,
            "calls": 3,
        }
    ]
    assert shapes["source_tn"]["minimum_reduction"] == 7
    assert throughput["source_tn"] == 10.0


def test_build_provenance_ignores_runtime_threads_but_binds_compile_identity() -> None:
    native_sha256 = "a" * 64
    package_sha256 = "e" * 64
    build_info = {
        key: f"fixture-{key}" for key in HARNESS.IMMUTABLE_NATIVE_BUILD_INFO_KEYS
    }
    build_info.update(
        {
            "api_version": 9,
            "source_commit": "b" * 40,
            "source_tree_sha256": "c" * 64,
            "cxx_standard": 17,
        }
    )
    build_info.update(
        {
            "blas_runtime_threads": 1,
            "openmp_placement_contract_configured": False,
            "openmp_placement_contract_evidence": None,
        }
    )
    baseline = HARNESS._native_build_provenance_sha256(
        build_info, native_sha256, package_sha256
    )
    runtime_variant = {
        **build_info,
        "blas_runtime_threads": 8,
        "openmp_placement_contract_configured": True,
        "openmp_placement_contract_evidence": {"requested_threads": 8},
    }
    assert (
        HARNESS._native_build_provenance_sha256(
            runtime_variant, native_sha256, package_sha256
        )
        == baseline
    )
    assert (
        HARNESS._native_build_provenance_sha256(
            {**build_info, "compiler_version": "13.0.0"},
            native_sha256,
            package_sha256,
        )
        != baseline
    )
    assert (
        HARNESS._native_build_provenance_sha256(build_info, "d" * 64, package_sha256)
        != baseline
    )
    assert (
        HARNESS._native_build_provenance_sha256(build_info, native_sha256, "f" * 64)
        != baseline
    )
    missing = dict(build_info)
    missing.pop("compiler_id")
    with pytest.raises(ValueError, match="missing immutable keys.*compiler_id"):
        HARNESS._native_build_provenance_sha256(missing, native_sha256, package_sha256)


def test_random_visits_are_zero_for_contiguous_and_measured_for_indexed() -> None:
    workers = [
        {
            "reference": {
                "performance_report": {
                    "accounting": {"observed_group_execution_variant_visits": 17}
                }
            }
        }
    ]
    assert HARNESS._random_indexed_visits(workers, "contiguous_sealed_v1") == 0
    assert HARNESS._random_indexed_visits(workers, "indexed_group_major_v1") == 17


def test_remaining_child_timeout_refuses_exhausted_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HARNESS.time, "monotonic", lambda: 20.0)
    with pytest.raises(TimeoutError, match="exhausted"):
        HARNESS._remaining_child_timeout(
            sweep_started=10.0, sweep_timeout=5.0, worker_timeout=100.0
        )
    assert (
        HARNESS._remaining_child_timeout(
            sweep_started=19.0, sweep_timeout=5.0, worker_timeout=100.0
        )
        == 4.0
    )


def test_calibration_key_binds_host_and_binary_package_identity(tmp_path: Path) -> None:
    planner = HARNESS._planner_module()
    install = tmp_path / "install"
    install.mkdir()
    native = install / "gxeldcore.so"
    native.write_bytes(b"native")
    source = tmp_path / "build.json"
    source.write_text("{}", encoding="utf-8")
    build = HARNESS.BuildSpec(
        label="fixture",
        install_prefix=install,
        native_module=native,
        python_executable=Path(sys.executable).resolve(),
        dependency_paths=(),
        environment={},
        expected_native_sha256="a" * 64,
        expected_package_sha256="b" * 64,
        expected_python_runtime_sha256="e" * 64,
        expected_dependency_runtime_sha256="f" * 64,
        expected_source_commit="c" * 40,
        expected_source_tree_sha256="d" * 64,
        qualification_evidence_path=None,
        qualification_evidence_sha256=None,
        declared_parent_source_commit=None,
        source_path=source,
    )
    dimensions = planner.ContextualTablaDimensionsV1(
        n_samples=32,
        n_variants=64,
        context_count=2,
        annotation_count=2,
        pair_count=3,
        component_count=6,
        group_count=3,
        sample_probe_count=4,
        variant_probe_count=4,
        trait_count=1,
        residual_basis_count=1,
        fixed_rank=2,
    )
    plan = planner.ContextualTablaPlanV1()
    args = SimpleNamespace(
        input_kind="synthetic",
        annotation_mode="strict_disjoint_binary_v1",
        group_layout="contiguous",
    )
    host = {"host_name": "tabla", "logical_cpu_count": 128}
    key = HARNESS._calibration_key(
        build=build,
        plan=plan,
        dimensions=dimensions,
        args=args,
        host_identity=host,
    )
    assert key != HARNESS._calibration_key(
        build=replace(build, expected_package_sha256="e" * 64),
        plan=plan,
        dimensions=dimensions,
        args=args,
        host_identity=host,
    )
    assert key != HARNESS._calibration_key(
        build=replace(build, expected_dependency_runtime_sha256="0" * 64),
        plan=plan,
        dimensions=dimensions,
        args=args,
        host_identity=host,
    )
    assert key != HARNESS._calibration_key(
        build=build,
        plan=plan,
        dimensions=dimensions,
        args=args,
        host_identity={**host, "host_name": "other"},
    )


def test_build_spec_resolves_python_and_binds_qualification_content(
    tmp_path: Path,
) -> None:
    install = tmp_path / "install"
    install.mkdir()
    native = install / "gxeldcore.so"
    native.write_bytes(b"native")
    qualification = tmp_path / "qualification.json"
    qualification.write_text('{"fault_matrix_passed":true}\n', encoding="utf-8")
    spec_path = tmp_path / "build.json"
    value = {
        "schema": HARNESS.BUILD_SPEC_SCHEMA,
        "label": "fixture",
        "install_prefix": str(install),
        "native_module": str(native),
        "python_executable": sys.executable,
        "dependency_paths": [],
        "environment": {},
        "expected_native_sha256": HARNESS._file_sha256(native),
        "expected_package_sha256": HARNESS._package_sha256(install),
        "expected_python_runtime_sha256": "c" * 64,
        "expected_dependency_runtime_sha256": "d" * 64,
        "expected_source_commit": "a" * 40,
        "expected_source_tree_sha256": "b" * 64,
        "qualification_evidence_path": str(qualification),
        "qualification_evidence_sha256": HARNESS._file_sha256(qualification),
        "declared_parent_source_commit": None,
    }
    spec_path.write_text(json.dumps(value), encoding="utf-8")
    build = HARNESS.BuildSpec.load(spec_path)
    assert build.python_executable == Path(sys.executable).resolve()
    plan = HARNESS._planner_module().ContextualTablaPlanV1()
    assert not HARNESS._has_content_bound_qualification(build, plan)
    valid = _qualification(
        source_commit=build.expected_source_commit,
        source_tree=build.expected_source_tree_sha256,
        native_sha256=build.expected_native_sha256,
    )
    qualification.write_text(json.dumps(valid), encoding="utf-8")
    valid_build = replace(
        build,
        qualification_evidence_sha256=HARNESS._file_sha256(qualification),
    )
    assert HARNESS._has_content_bound_qualification(valid_build, plan)
    assert HARNESS._integrity_identity_sha256(plan) == HARNESS._canonical_sha256(
        {
            "integrity_policy": valid["integrity_policy"],
            "integrity_backend": valid["integrity_backend"],
            "fault_coverage_contract": valid["fault_coverage_contract"],
        }
    )
    tampered = {**valid, "fault_coverage_contract": "tampered_contract_v0"}
    qualification.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_build = replace(
        build,
        qualification_evidence_sha256=HARNESS._file_sha256(qualification),
    )
    assert not HARNESS._has_content_bound_qualification(tampered_build, plan)


def test_qualification_preload_keeps_asan_first_and_seals_exact_worker_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asan = "/usr/lib/gcc/x86_64-linux-gnu/12/libasan.so"
    libstdcxx = "/opt/conda/lib/libstdc++.so.6"
    unrelated = "/controller/lib/libtrace.so"
    monkeypatch.setenv("LD_PRELOAD", f"{unrelated}:{asan}:{libstdcxx}:{asan}")

    sealed = HARNESS._qualification_build_preload(
        os.environ.get("LD_PRELOAD"), f"{libstdcxx} {libstdcxx}"
    )
    assert sealed == f"{asan}:{libstdcxx}"
    child = HARNESS._child_environment(
        SimpleNamespace(
            environment={
                "ASAN_OPTIONS": "halt_on_error=1:detect_leaks=0",
                "LD_PRELOAD": sealed,
                "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
            }
        ),
        tmp_path,
    )
    assert child["LD_PRELOAD"] == sealed
    assert child["ASAN_OPTIONS"] == "halt_on_error=1:detect_leaks=0"
    assert child["UBSAN_OPTIONS"] == "halt_on_error=1:print_stacktrace=1"
    assert unrelated not in child["LD_PRELOAD"]


def test_qualification_preload_does_not_inherit_nonsanitizer_controller_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    libstdcxx = "/opt/conda/lib/libstdc++.so.6"
    unrelated = "/controller/lib/libtrace.so"
    monkeypatch.setenv("LD_PRELOAD", unrelated)

    sealed = HARNESS._qualification_build_preload(
        os.environ.get("LD_PRELOAD"), libstdcxx
    )
    assert sealed == libstdcxx
    child = HARNESS._child_environment(
        SimpleNamespace(environment={"LD_PRELOAD": sealed}), tmp_path
    )
    assert child["LD_PRELOAD"] == libstdcxx
    with pytest.raises(ValueError, match="ASan runtime not inherited"):
        HARNESS._qualification_build_preload(
            None, f"{libstdcxx}:/unexpected/libasan.so"
        )


def test_case_file_identity_mutation_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {}
    for extension in ("bed", "bim", "fam"):
        path = tmp_path / f"case.{extension}"
        path.write_bytes(extension.encode("ascii"))
        files[extension] = HARNESS._file_identity(path, hash_content=True)
    case = {"files": files}
    HARNESS._verify_case_file_identities(case, require_full_hashes=True)
    monkeypatch.setattr(
        HARNESS,
        "_file_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected full hash")),
    )
    HARNESS._verify_case_file_identities(
        case, require_full_hashes=True, verify_content_hashes=False
    )
    monkeypatch.undo()
    (tmp_path / "case.bed").write_bytes(b"mutated-bed")
    with pytest.raises(RuntimeError, match="bed input identity changed"):
        HARNESS._verify_case_file_identities(case, require_full_hashes=True)


def test_case_science_identity_is_workspace_independent_and_self_verifying() -> None:
    science = {
        "schema": "context_native_stage6_science_identity_v1",
        "dimensions": {"n": 8, "m": 12},
        "arrays": {"selected_genotype": "a" * 64},
    }
    digest = HARNESS._canonical_sha256(science)
    cases = [
        {
            "science_identity": science,
            "case_identity_sha256": digest,
            "bundle": f"/tmp/workspace-{index}/case_arrays.npz",
            "files": {"bed": {"inode": index + 1}},
        }
        for index in range(2)
    ]
    assert [HARNESS._verify_case_science_identity(case) for case in cases] == [
        digest,
        digest,
    ]
    cases[1]["science_identity"] = {**science, "dimensions": {"n": 9, "m": 12}}
    with pytest.raises(RuntimeError, match="digest mismatch"):
        HARNESS._verify_case_science_identity(cases[1])


def test_global_duplicate_decodes_counts_trait_pass() -> None:
    def report(visits: int, local_duplicates: int) -> dict[str, object]:
        return {
            "dimensions": {"variants": 10},
            "accounting": {
                "observed_variant_record_visits": visits,
                "duplicate_decodes": local_duplicates,
            },
        }

    worker = {
        "reference": {"performance_report": report(40, 30)},
        "trait": {"performance_report": report(10, 0)},
    }
    assert HARNESS._global_duplicate_decodes([worker]) == 40


@pytest.mark.skipif(
    "SUMMIT_STAGE6_TEST_INSTALL" not in os.environ,
    reason="set SUMMIT_STAGE6_TEST_INSTALL for the native smoke test",
)
def test_tiny_fresh_process_end_to_end(tmp_path: Path) -> None:
    install = Path(os.environ["SUMMIT_STAGE6_TEST_INSTALL"]).resolve(strict=True)
    native = next((install / "summit").glob("gxeldcore*.so"))
    library = Path(sys.executable).resolve().parents[1] / "lib"
    sealed_preload = HARNESS._qualification_build_preload(
        os.environ.get("LD_PRELOAD"), str(library / "libstdc++.so.6")
    )
    child_environment = {
        "LD_LIBRARY_PATH": str(library),
        "LD_PRELOAD": sealed_preload,
    }
    for sanitizer_option in ("ASAN_OPTIONS", "UBSAN_OPTIONS"):
        if sanitizer_option in os.environ:
            child_environment[sanitizer_option] = os.environ[sanitizer_option]
    info_environment = {**os.environ, **child_environment, "PYTHONPATH": str(install)}
    build_info = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-S",
                "-c",
                "import json; from summit import gxeldcore; "
                "print(json.dumps(dict(gxeldcore.build_info()), default=list))",
            ],
            env=info_environment,
            text=True,
        )
    )
    runtime_identity = HARNESS._runtime_dependency_identity()
    qualification = tmp_path / "fault-qualification.json"
    qualification.write_text(
        json.dumps(
            _qualification(
                source_commit=build_info["source_commit"],
                source_tree=build_info["source_tree_sha256"],
                native_sha256=HARNESS._file_sha256(native),
            )
        ),
        encoding="utf-8",
    )
    build_path = tmp_path / "build.json"
    build_path.write_text(
        json.dumps(
            {
                "schema": HARNESS.BUILD_SPEC_SCHEMA,
                "label": "instrumented",
                "install_prefix": str(install),
                "native_module": str(native),
                "python_executable": sys.executable,
                "dependency_paths": [
                    str(
                        library
                        / f"python{sys.version_info.major}.{sys.version_info.minor}"
                        / "site-packages"
                    )
                ],
                "environment": child_environment,
                "expected_native_sha256": HARNESS._file_sha256(native),
                "expected_package_sha256": HARNESS._package_sha256(install),
                "expected_python_runtime_sha256": runtime_identity[
                    "python_runtime_sha256"
                ],
                "expected_dependency_runtime_sha256": runtime_identity[
                    "dependency_runtime_sha256"
                ],
                "expected_source_commit": build_info["source_commit"],
                "expected_source_tree_sha256": build_info["source_tree_sha256"],
                "qualification_evidence_path": str(qualification),
                "qualification_evidence_sha256": HARNESS._file_sha256(qualification),
                "declared_parent_source_commit": None,
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": HARNESS.PLAN_SPEC_SCHEMA,
                "plan_id": "tiny-t1",
                "build_label": "instrumented",
                "comparison_mode": "exact_trace_v1",
                "plan": {
                    "source_variant_block": 8,
                    "target_variant_block": 8,
                    "grouped_variant_block": 8,
                    "same_person_variant_block": 8,
                    "trait_variant_block": 8,
                    "sample_probe_resident_count": 2,
                    "sample_probe_tile": 2,
                    "variant_probe_tile": 2,
                    "action_tile": 2,
                    "annotation_tile": 1,
                    "source_coordinate_tile": 2,
                    "context_tile": 2,
                    "group_tile": 1,
                    "resident_source_scores": True,
                    "resident_actions": True,
                    "grouped_attribution_algorithm": "group_restricted_action_v1",
                    "direct_grouped_scaling_policy": "not_applicable",
                    "group_execution_order": "contiguous_sealed_v1",
                    "annotation_mode": "strict_disjoint_binary_v1",
                    "multiplication_backend": "protected_dense_gemm_v1",
                    "integrity_policy": "full_scalar_witness_v1",
                    "integrity_backend": "deterministic_tiled_fp64_with_scalar_witness_v1",
                    "process_count": 1,
                    "decode_threads": 1,
                    "blas_threads": 1,
                    "affinity_cpu_count": "auto",
                    "affinity_identity_sha256": "auto",
                    "numa_policy": "unbound_first_touch_v1",
                    "output_numa_node": -1,
                    "huge_page_policy": "disabled_v1",
                    "allocation_reuse": True,
                    "deterministic_reductions": True,
                    "numeric_policy": "fp64_v1",
                    "reduction_copy_count": 1,
                    "integrity_reserve_bytes": 0,
                    "telemetry_capacity_bytes": 0,
                    "fixed_headroom_bytes": 0,
                    "headroom_basis_points": 2000,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "benchmark.json"
    cpu = min(os.sched_getaffinity(0))
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--build-spec",
            str(build_path),
            "--baseline-plan",
            str(plan_path),
            "--cpu-list",
            str(cpu),
            "--retained-samples",
            "24",
            "--retained-variants",
            "24",
            "--q",
            "2",
            "--k",
            "2",
            "--groups",
            "3",
            "--sample-probes",
            "2",
            "--variant-probes",
            "2",
            "--traits",
            "1",
            "--residual-bases",
            "1",
            "--fixed-rank",
            "2",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["terminal_status"] == "complete"
    record = result["benchmark_runs"][0]["record"]
    if record["terminal_status"] == "accepted":
        assert record["rejection_reasons"] == []
    elif build_info["sanitizer_mode"] == "asan_ubsan":
        assert record["terminal_status"] == "rejected"
        assert record["rejection_reasons"] == [
            "measured_rss_exceeds_admission_margin",
            "full_acceptance_evidence_failed",
        ]
        acceptance = record["acceptance"]
        assert acceptance["admitted_measured_memory_agrees"] is False
        assert all(
            value is True
            for name, value in acceptance.items()
            if name
            not in {
                "admitted_measured_memory_agrees",
                "fixed_probe_discrepancies",
                "fixed_probe_discrepancy_sha256",
                "fixed_probe_evidence_kind",
                "schema",
            }
        )
        assert all(
            discrepancy == 0.0
            for discrepancy in acceptance["fixed_probe_discrepancies"].values()
        )
    else:
        raise AssertionError(record["rejection_reasons"])
