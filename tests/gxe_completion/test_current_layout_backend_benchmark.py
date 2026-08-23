from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.gxe import benchmark_current_layout_backends as benchmark


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT_A = "a" * 40


def _args(
    tmp_path: Path,
    *,
    backend: str = "openblas",
    strategy: str = "automatic",
    ways: dict[str, int] | None = None,
) -> argparse.Namespace:
    module = tmp_path / "gxeldcore.test.so"
    module.write_bytes(b"native-module")
    return argparse.Namespace(
        expected_backend=backend,
        expected_archive_sha256=SHA_A,
        expected_source_commit=COMMIT_A,
        expected_source_tree_sha256=SHA_B,
        expected_private_source_commit=COMMIT_A if backend == "blis" else None,
        expected_private_source_tree_sha256=SHA_B if backend == "blis" else None,
        threads=2,
        cpus=(2, 4),
        blis_thread_strategy=strategy,
        blis_thread_ways=ways,
        native_module=module,
    )


def _placement(cpus: tuple[int, ...] = (2, 4)) -> dict:
    threads = len(cpus)
    return {
        "schema": benchmark.PLACEMENT_SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": threads,
        "expected_cpu_ids": list(cpus),
        "omp_dynamic": False,
        "omp_thread_limit": threads,
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": threads,
        "effective_openmp_capacity": threads,
        "place_cpu_ids": [[cpu] for cpu in cpus],
        "team_size": threads,
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


def _build(args: argparse.Namespace, placement: dict) -> dict:
    result = benchmark._exact_build_required(args)
    result["openmp_placement_contract_evidence"] = copy.deepcopy(placement)
    if args.expected_backend == "openblas":
        result.update(
            {
                "blas_runtime_config": (
                    "OpenBLAS 0.3.34 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP Zen"
                ),
                "blas_runtime_corename": "Zen",
            }
        )
    else:
        result.update(
            {
                "private_blas_header_sha256": "c" * 64,
                "private_blas_cblas_header_sha256": "d" * 64,
            }
        )
    return result


def _operand_numa(record: dict, name: str, node: int) -> dict:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    byte_count = benchmark._operand_byte_count(record, name)
    start = 0
    full_pages = byte_count // page_size
    selected = min(full_pages, 8)
    samples = []
    for ordinal in range(selected):
        page_index = (
            0 if selected == 1 else ordinal * (full_pages - 1) // (selected - 1)
        )
        samples.append(
            {
                "sample_ordinal": ordinal,
                "full_page_index": page_index,
                "byte_offset_from_operand_start": page_index * page_size,
                "status_kind": "numa_node",
                "raw_move_pages_status": node,
                "numa_node": node,
                "page_query_errno": None,
            }
        )
    return {
        "query_status": "queried",
        "storage_span_pages": (byte_count - 1) // page_size + 1,
        "fully_contained_pages": full_pages,
        "operand_byte_count": byte_count,
        "operand_start_address_page_offset": start,
        "operand_end_exclusive_address_page_offset": byte_count % page_size,
        "selected_sample_pages": selected,
        "resolved_sample_pages": selected,
        "page_query_error_pages": 0,
        "node_histogram": {str(node): selected},
        "page_error_errno_histogram": {},
        "ordered_samples": samples,
    }


def _telemetry(case: benchmark.Case, *, active: float = 2.0) -> dict:
    record = {
        **case.cblas,
        "schema_version": 1,
        "sequence": 1,
        "arithmetic_dtype": "float64",
        "alpha": 1.0,
        "beta": 0.0,
        "flop_count": float(case.flops),
        "wall_seconds": 1.0,
        "process_cpu_seconds": active,
        "gflops_per_second": float(case.flops) / 1.0e9,
        "process_cpu_to_wall_ratio": active,
        "active_core_equivalents": active,
        "requested_threads": case.threads,
        "configured_threads": case.threads,
        "backend_threads": case.threads,
        "omp_in_parallel": False,
        "omp_level": 0,
        "omp_active_level": 0,
        "omp_max_active_levels": 1,
        "omp_max_threads": case.threads,
        "omp_num_threads": 1,
        "omp_thread_num": 0,
        "entry_cpu": 2,
        "exit_cpu": 2,
        "cpu_affinity_count": 1,
        "cpu_affinity_list": "2",
        "backend": "OpenBLAS",
        "backend_config": "OpenBLAS 0.3.34 USE_OPENMP Zen",
        "backend_corename": "Zen",
        "completed": True,
    }
    record["operand_numa_page_samples"] = {
        "schema_version": 1,
        "sampling_method": "move_pages_query_no_migration",
        "sampling_timing": "after_vendor_call_outside_timed_interval",
        "sample_limit_per_operand": 8,
        "address_selection_schema_version": 1,
        "address_selection_policy": benchmark.NUMA_ADDRESS_SELECTION_POLICY,
        "selected_addresses_are_page_bases": True,
        "partial_boundary_pages_included": False,
        "first_and_last_fully_contained_pages_selected": True,
        "virtual_addresses_exposed": False,
        "operand_byte_range_semantics": "[start_address,end_exclusive_address)",
        "address_evidence": (
            "ordered_samples_with_operand_relative_byte_offsets_and_full_page_indices"
        ),
        "system_page_size": int(os.sysconf("SC_PAGE_SIZE")),
        "syscall_result": 0,
        "syscall_errno": 0,
        "operands": {name: _operand_numa(record, name, 0) for name in ("a", "b", "c")},
    }
    return record


def test_exact_current_layout_cases_and_protocol_are_fixed():
    source = benchmark.Case("source", 289_111, 2_000, 32, 3, 32)
    target = benchmark.Case("target", 289_111, 2_000, 32, 3, 32)
    assert source.panel_columns == 192
    assert source.cblas == {
        "operation": "dgemm_nn",
        "layout": "column_major",
        "transpose_a": "N",
        "transpose_b": "N",
        "m": 289_111,
        "n": 192,
        "k": 2_000,
        "lda": 289_111,
        "ldb": 2_000,
        "ldc": 289_111,
    }
    assert target.panel_columns == 384
    assert target.cblas == {
        "operation": "dgemm_tn",
        "layout": "column_major",
        "transpose_a": "T",
        "transpose_b": "N",
        "m": 2_000,
        "n": 384,
        "k": 289_111,
        "lda": 289_111,
        "ldb": 289_111,
        "ldc": 2_000,
    }
    assert (benchmark.WARMUPS, benchmark.MEASURED_REPEATS) == (3, 5)
    assert benchmark.MAX_CASE_SECONDS == 90.0
    assert benchmark.MAX_SWEEP_SECONDS == 1_200.0
    for case in (source, target):
        memory = benchmark._estimated_memory(case)
        assert memory["uncertified_vendor_workspace_fallback_bytes"] == 16 * 1024**3
        assert memory["headroom_fraction"] == 0.20
        assert memory["conservative_peak_bytes"] > memory["input_bytes"]
        assert memory["integrity_right_snapshot_bytes"] == memory["right_input_bytes"]
        assert memory["integrity_checksum_and_repair_scratch_bytes"] > 0
    source_memory = benchmark._estimated_memory(source)
    assert source_memory["production_sealed_pair_target_peak"] is None
    target_memory = benchmark._estimated_memory(target)
    assert (
        target_memory["production_sealed_pair_target_peak"][
            "per_call_integrity_right_snapshot_bytes"
        ]
        == 0
    )
    assert target_memory["integrity_right_snapshot_bytes"] == 289_111 * 384 * 8
    assert (
        target_memory["production_sealed_pair_target_peak"]["conservative_peak_bytes"]
        < target_memory["conservative_peak_bytes"]
    )


def test_placement_contract_requires_exact_ordered_singletons_and_types():
    value = _placement()
    assert benchmark._validate_placement_attestation(value, (2, 4), 2) == value
    damaged = copy.deepcopy(value)
    damaged["workers"][1]["current_cpu"] = 2
    with pytest.raises(RuntimeError, match="worker 1 mismatch"):
        benchmark._validate_placement_attestation(damaged, (2, 4), 2)
    damaged = copy.deepcopy(value)
    damaged["verified"] = 1
    with pytest.raises(RuntimeError, match="boolean fields"):
        benchmark._validate_placement_attestation(damaged, (2, 4), 2)
    damaged = copy.deepcopy(value)
    damaged["omp_max_active_levels"] = True
    with pytest.raises(RuntimeError, match="integer fields"):
        benchmark._validate_placement_attestation(damaged, (2, 4), 2)
    damaged = copy.deepcopy(value)
    damaged["schema_version"] = np.int64(1)
    with pytest.raises(RuntimeError, match="integer fields"):
        benchmark._validate_placement_attestation(damaged, (2, 4), 2)
    damaged = copy.deepcopy(value)
    damaged["extra"] = "not accepted"
    with pytest.raises(RuntimeError, match="unexpected schema"):
        benchmark._validate_placement_attestation(damaged, (2, 4), 2)


def test_early_numa_attestation_is_exact_and_pre_numeric():
    value = {
        "schema": benchmark.NUMA_SCHEMA,
        "mode": "membind",
        "requested_nodes": "0-1",
        "effective_nodes": [0, 1],
        "task_count_at_application": 1,
        "applied_before_numeric_import": True,
        "verified": True,
        "source": "libnuma",
        "applied_policy": "libnuma:membind:0,1",
        "static_nodes": True,
        "pid": os.getpid(),
    }
    assert benchmark._validate_early_numa_attestation(value, (0, 1)) == value
    damaged = dict(value, applied_before_numeric_import=1)
    with pytest.raises(RuntimeError, match="early NUMA attestation mismatch"):
        benchmark._validate_early_numa_attestation(damaged, (0, 1))
    damaged = dict(value, extra=True)
    with pytest.raises(RuntimeError, match="unexpected schema"):
        benchmark._validate_early_numa_attestation(damaged, (0, 1))
    damaged = dict(value)
    damaged.pop("static_nodes")
    with pytest.raises(RuntimeError, match="unexpected schema"):
        benchmark._validate_early_numa_attestation(damaged, (0, 1))
    damaged = dict(value, static_nodes=False)
    with pytest.raises(RuntimeError, match="early NUMA attestation mismatch"):
        benchmark._validate_early_numa_attestation(damaged, (0, 1))
    damaged = dict(value, static_nodes=1)
    with pytest.raises(RuntimeError, match="early NUMA attestation mismatch"):
        benchmark._validate_early_numa_attestation(damaged, (0, 1))


@pytest.mark.parametrize("backend", ["openblas", "blis"])
def test_build_contract_accepts_only_exact_api9_backend16_provenance(
    tmp_path: Path,
    backend: str,
):
    args = _args(tmp_path, backend=backend)
    placement = _placement()
    build = _build(args, placement)
    module = SimpleNamespace(build_info=lambda: build)
    result = benchmark._validate_build_info(module, args.native_module, args, placement)
    assert result["module_sha256"] == hashlib.sha256(b"native-module").hexdigest()
    assert result["build_info"] == build

    wrong_evidence = copy.deepcopy(build)
    wrong_evidence["openmp_placement_contract_evidence"]["workers"].reverse()
    with pytest.raises(RuntimeError, match="placement evidence"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: wrong_evidence),
            args.native_module,
            args,
            placement,
        )

    wrong_api = dict(build, api_version=8)
    with pytest.raises(RuntimeError, match="build contract mismatch"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: wrong_api),
            args.native_module,
            args,
            placement,
        )
    wrong_api_type = dict(build, api_version=np.int64(9))
    with pytest.raises(RuntimeError, match="integer contract"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: wrong_api_type),
            args.native_module,
            args,
            placement,
        )


def test_blis_build_contract_requires_tls_and_strict_automatic_ways(tmp_path: Path):
    args = _args(tmp_path, backend="blis")
    placement = _placement()
    build = _build(args, placement)

    no_tls = dict(build, blas_runtime_tls_enabled=False)
    with pytest.raises(RuntimeError, match="build contract mismatch"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: no_tls),
            args.native_module,
            args,
            placement,
        )

    bool_way = copy.deepcopy(build)
    bool_way["blas_runtime_thread_ways"]["jc"] = True
    with pytest.raises(RuntimeError, match="thread ways"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: bool_way),
            args.native_module,
            args,
            placement,
        )


def test_blis_build_contract_requires_exact_native_numa_fields_and_types(
    tmp_path: Path,
):
    class StringSubclass(str):
        pass

    args = _args(tmp_path, backend="blis")
    placement = _placement()
    build = _build(args, placement)
    assert benchmark._validate_native_numa_build_contract(build)[
        "native_gemm_output_numa_evidence_capacity"
    ] == 16_384

    for field, replacement in (
        ("gemm_integrity_minimum_vendor_flops", np.int64(1_000_000_000)),
        ("native_integrity_snapshot_numa_contract_supported", 1),
        ("native_gemm_output_numa_contract_supported", 1),
        ("native_integrity_snapshot_numa_query_chunk_page_limit", np.int64(65_536)),
        ("native_gemm_output_numa_query_chunk_page_limit", np.int64(65_536)),
        ("native_gemm_output_numa_evidence_capacity", np.int64(16_384)),
        (
            "native_integrity_snapshot_numa_contract_schema",
            StringSubclass(benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA),
        ),
        (
            "native_gemm_output_numa_contract_schema",
            StringSubclass(benchmark.NATIVE_GEMM_OUTPUT_NUMA_SCHEMA),
        ),
    ):
        damaged = copy.deepcopy(build)
        damaged[field] = replacement
        with pytest.raises(RuntimeError, match="native NUMA build contract"):
            benchmark._validate_build_info(
                SimpleNamespace(build_info=lambda damaged=damaged: damaged),
                args.native_module,
                args,
                placement,
            )

    missing = copy.deepcopy(build)
    missing.pop("native_gemm_output_numa_contract_schema")
    with pytest.raises(RuntimeError, match="build contract mismatch"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: missing),
            args.native_module,
            args,
            placement,
        )

    wrong_threshold = copy.deepcopy(build)
    wrong_threshold["gemm_integrity_minimum_vendor_flops"] -= 1
    with pytest.raises(RuntimeError, match="build contract mismatch"):
        benchmark._validate_build_info(
            SimpleNamespace(build_info=lambda: wrong_threshold),
            args.native_module,
            args,
            placement,
        )


def test_private_static_linkage_inventory_and_fail_closed_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = tmp_path / "gxeldcore.test.so"
    module.write_bytes(b"x")
    archive = tmp_path / "libprivateblas.a"
    archive.write_bytes(b"archive")

    def outputs(command, timeout=20.0):
        del timeout
        if command[0].endswith("readelf"):
            return " 0x1 (NEEDED) Shared library: [libgomp.so.1]\n"
        if command[1:2] == ["-D"]:
            return "000000 T summit_native_entry\n"
        if command[1:2] == ["-u"]:
            return "                 U PyExc_RuntimeError\n"
        return (
            "000000 T cblas_dgemm\n"
            "000010 T cblas_sgemm\n"
            "000020 T bli_info_get_version_str\n"
            "000030 T bli_info_get_enable_tls\n"
        )

    monkeypatch.setattr(benchmark, "_run_checked", outputs)
    evidence = benchmark._linkage_evidence(module, archive, "blis")
    assert evidence["private_static_linkage_verified"] is True
    assert evidence["defined_private_blas_global_count"] == 4
    assert evidence["dynamic_blas_global_count"] == 0
    assert evidence["defined_private_blas_inventory_source"] == (
        "hash_pinned_static_archive"
    )

    def openblas_outputs(command, timeout=20.0):
        if command[1:3] == ["-g", "--defined-only"]:
            return "000000 T cblas_dgemm\n000010 T openblas_get_config\n"
        return outputs(command, timeout)

    monkeypatch.setattr(benchmark, "_run_checked", openblas_outputs)
    openblas = benchmark._linkage_evidence(module, archive, "openblas")
    assert openblas["required_private_blas_globals"] == [
        "cblas_dgemm",
        "openblas_get_config",
    ]

    def dynamic_dependency(command, timeout=20.0):
        if command[0].endswith("readelf"):
            return " 0x1 (NEEDED) Shared library: [libblis.so.4]\n"
        return outputs(command, timeout)

    monkeypatch.setattr(benchmark, "_run_checked", dynamic_dependency)
    with pytest.raises(RuntimeError, match="dynamic BLAS dependency"):
        benchmark._linkage_evidence(module, archive, "blis")

    def dynamic_symbol(command, timeout=20.0):
        if command[1:2] == ["-D"]:
            return "000000 T bli_gemm_ex\n"
        return outputs(command, timeout)

    monkeypatch.setattr(benchmark, "_run_checked", dynamic_symbol)
    with pytest.raises(RuntimeError, match="exported or unresolved dynamically"):
        benchmark._linkage_evidence(module, archive, "blis")

    def unresolved_symbol(command, timeout=20.0):
        if command[1:2] == ["-u"]:
            return "                 U dgemm_64_\n"
        return outputs(command, timeout)

    monkeypatch.setattr(benchmark, "_run_checked", unresolved_symbol)
    with pytest.raises(RuntimeError, match="unresolved BLAS globals"):
        benchmark._linkage_evidence(module, archive, "blis")


def test_fresh_worker_environment_is_backend_exact_and_drops_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-escape")
    monkeypatch.setenv("GOMP_CPU_AFFINITY", "0-99")
    monkeypatch.setenv("BLIS_NT", "99")
    for name in benchmark.BLIS_UNCONTROLLED_OVERRIDES:
        monkeypatch.setenv(name, "uncontrolled")
    automatic = _args(tmp_path, backend="blis")
    environment, settings = benchmark._worker_environment(automatic)
    assert environment["BLIS_NUM_THREADS"] == "2"
    assert environment["OMP_PLACES"] == "{2},{4}"
    assert environment["OMP_PROC_BIND"] == "SPREAD"
    assert "GITHUB_TOKEN" not in environment
    assert "GOMP_CPU_AFFINITY" not in environment
    assert "BLIS_NT" not in environment
    assert all(
        name not in environment for name in benchmark.BLIS_UNCONTROLLED_OVERRIDES
    )
    assert all(name not in environment for name in benchmark.BLIS_WAY_ENV.values())
    assert set(environment).issubset(
        set(benchmark.SAFE_ENV_PASSTHROUGH) | set(settings)
    )
    assert all(environment[name] == value for name, value in settings.items())

    manual_ways = {"jc": 2, "pc": 1, "ic": 1, "jr": 1, "ir": 1}
    manual = _args(tmp_path, backend="blis", strategy="manual", ways=manual_ways)
    environment, _ = benchmark._worker_environment(manual)
    assert environment["BLIS_THREAD_IMPL"] == "openmp"
    assert {
        name: environment[env_name] for name, env_name in benchmark.BLIS_WAY_ENV.items()
    } == {name: str(value) for name, value in manual_ways.items()}


def test_exact_numa_telemetry_and_measured_active_core_gate():
    case = benchmark.Case("source", 512, 512, 1, 1, 2)
    record = _telemetry(case)
    build = {
        "blas_vendor": "OpenBLAS",
        "blas_runtime_config": "OpenBLAS 0.3.34 USE_OPENMP Zen",
        "blas_runtime_corename": "Zen",
    }
    result = benchmark._validate_telemetry(
        record, case, (2, 4), (0,), build, measured=True
    )
    assert result["passed"] is True
    assert result["active_core_equivalents"] == 2.0
    assert all(
        operand["selected_sample_pages"] > 0
        for operand in result["numa"]["operands"].values()
    )

    underactive = _telemetry(case, active=1.49)
    with pytest.raises(RuntimeError, match="fewer than 75%"):
        benchmark._validate_telemetry(
            underactive, case, (2, 4), (0,), build, measured=True
        )
    # Warmups retain every integrity/NUMA gate but do not claim utilization.
    benchmark._validate_telemetry(
        underactive, case, (2, 4), (0,), build, measured=False
    )

    nonlocal_page = copy.deepcopy(record)
    nonlocal_page["operand_numa_page_samples"]["operands"]["b"]["ordered_samples"][0][
        "numa_node"
    ] = 1
    with pytest.raises(RuntimeError, match="unresolved or nonlocal"):
        benchmark._validate_telemetry(
            nonlocal_page, case, (2, 4), (0,), build, measured=True
        )

    wrong_affinity = copy.deepcopy(record)
    wrong_affinity["cpu_affinity_count"] = 2
    wrong_affinity["cpu_affinity_list"] = "2,4"
    with pytest.raises(RuntimeError, match="vendor telemetry mismatch"):
        benchmark._validate_telemetry(
            wrong_affinity, case, (2, 4), (0,), build, measured=True
        )


def test_telemetry_status_is_exact_and_lossless():
    status = {
        "schema_version": 1,
        "capacity": 4096,
        "buffered_records": 0,
        "captured_records": 1,
        "dropped_records": 0,
        "next_sequence": 8,
        "operand_numa_sampling_method": "move_pages_query_no_migration",
        "operand_numa_sample_limit_per_operand": 8,
        "operand_numa_address_selection_schema_version": 1,
        "operand_numa_address_selection_policy": (
            benchmark.NUMA_ADDRESS_SELECTION_POLICY
        ),
        "operand_numa_partial_boundary_pages_included": False,
    }
    assert benchmark._validate_telemetry_status(status, observed_sequence=7) == status
    damaged = dict(status, dropped_records=1)
    with pytest.raises(RuntimeError, match="status contract mismatch"):
        benchmark._validate_telemetry_status(damaged, observed_sequence=7)
    damaged = dict(status, schema_version=True)
    with pytest.raises(RuntimeError, match="status contract mismatch"):
        benchmark._validate_telemetry_status(damaged, observed_sequence=7)


@pytest.mark.parametrize("operation", ["source", "target"])
def test_full_robust_oracle_accepts_exact_and_rejects_damage(operation: str):
    case = benchmark.Case(operation, 7, 5, 1, 1, 1)
    rng = np.random.default_rng(17)
    left = np.asfortranarray(rng.standard_normal((7, 5)))
    right_shape = (5, 2) if operation == "source" else (7, 4)
    right = np.asfortranarray(rng.standard_normal(right_shape))
    exact = np.asfortranarray(left @ right if operation == "source" else left.T @ right)
    result = benchmark._full_oracle(case, left, right, exact, np)
    assert result["passed"] is True
    assert result["full_output_compared"] is True
    assert len(result["long_double_witnesses"]) >= 3

    damaged = exact.copy(order="F")
    damaged[0, 0] += 1.0e-5
    with pytest.raises(RuntimeError, match="forward-error bound"):
        benchmark._full_oracle(case, left, right, damaged, np)


def _physical_cpus(count: int) -> tuple[int, ...]:
    selected = []
    physical = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                int((topology / "physical_package_id").read_text().strip()),
                int((topology / "core_id").read_text().strip()),
            )
        except (OSError, ValueError):
            continue
        if key in physical or benchmark._cpu_numa_node(cpu) is None:
            continue
        physical.add(key)
        selected.append(cpu)
        if len(selected) == count:
            break
    return tuple(selected)


def test_dry_run_emits_fresh_python_commands_and_refuses_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cpus = _physical_cpus(2)
    if len(cpus) != 2:
        pytest.skip("two allowed physical CPUs with NUMA records are unavailable")
    prefix = tmp_path / "install"
    package = prefix / "summit"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# exact test package\n")
    module = package / "gxeldcore.test.so"
    module.write_bytes(b"native-test")
    archive = tmp_path / "libopenblas.a"
    archive.write_bytes(b"archive-test")
    output = tmp_path / "report.json"
    package_hash = benchmark._package_identity(prefix)["manifest_sha256"]

    def linkage(_module: Path, _archive: Path, _backend: str) -> dict:
        return {
            "private_static_linkage_verified": True,
            "defined_private_blas_globals": ["cblas_dgemm", "openblas_get_config"],
        }

    monkeypatch.setattr(benchmark, "_linkage_evidence", linkage)
    argv = [
        "--install-prefix",
        str(prefix),
        "--native-module",
        str(module),
        "--expected-native-sha256",
        benchmark._sha256(module),
        "--expected-package-manifest-sha256",
        package_hash,
        "--private-archive",
        str(archive),
        "--expected-archive-sha256",
        benchmark._sha256(archive),
        "--expected-source-commit",
        COMMIT_A,
        "--expected-source-tree-sha256",
        SHA_B,
        "--expected-backend",
        "openblas",
        "--python-executable",
        sys.executable,
        "--dependency-path",
        str(tmp_path),
        "--cpus",
        ",".join(map(str, cpus)),
        "--threads",
        "2",
        "--case-timeout-seconds",
        "10",
        "--sweep-timeout-seconds",
        "20",
        "--output",
        str(output),
        "--dry-run",
    ]
    assert benchmark.main(argv) == 0
    report = json.loads(output.read_text())
    assert report["accepted"] is False
    assert report["status"] == "validated_dry_run"
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        command = case["command"]
        assert "-S" in command
        assert "--_worker" in command
        assert command[0].endswith("taskset")
    assert output.stat().st_mode & 0o777 == 0o600

    with pytest.raises(SystemExit):
        benchmark.main(argv)
