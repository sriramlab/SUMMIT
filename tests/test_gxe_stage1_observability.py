from __future__ import annotations

import os
import mmap
import types

import numpy as np

from summit import gxeldcore
from summit.ldscore import gxe_multi
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.gxe_multi import _MultiEnvironmentGemm


_NUMA_ADDRESS_SELECTION_POLICY = (
    "evenly_spaced_fully_contained_page_bases"
)


def _affinity_list(cpus: set[int]) -> str:
    ranges: list[str] = []
    ordered = sorted(cpus)
    index = 0
    while index < len(ordered):
        first = ordered[index]
        last = first
        while index + 1 < len(ordered) and ordered[index + 1] == last + 1:
            index += 1
            last = ordered[index]
        ranges.append(str(first) if first == last else f"{first}-{last}")
        index += 1
    return ",".join(ranges)


def _misaligned_fortran_normal(
    rng: np.random.Generator, shape: tuple[int, int]
) -> np.ndarray:
    """Return a page-misaligned, double-aligned Fortran array."""
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    byte_count = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float64).itemsize
    backing = np.empty(byte_count + page_size, dtype=np.uint8)
    desired_page_offset = np.dtype(np.float64).itemsize
    offset = (
        desired_page_offset - int(backing.ctypes.data) % page_size
    ) % page_size
    result = np.ndarray(
        shape, dtype=np.float64, buffer=backing, offset=offset, order="F"
    )
    result[...] = rng.normal(size=shape)
    assert result.flags.f_contiguous
    assert result.flags.aligned
    assert int(result.ctypes.data) % page_size == desired_page_offset
    return result


def _configured_native_threads() -> int:
    desired = min(2, len(os.sched_getaffinity(0)))
    assert desired > 0
    try:
        configured = int(gxeldcore.configure_blas_threads(desired))
    except RuntimeError as exc:
        # The private runtime is immutable after its first configuration.  A
        # wider test session may therefore have fixed it before this test.
        assert "different thread count" in str(exc)
        configured = int(gxeldcore.build_info()["blas_runtime_threads"])
        assert gxeldcore.configure_blas_threads(configured) == configured
    assert configured > 0
    return configured


def test_executor_persists_in_process_early_numa_attestation(monkeypatch) -> None:
    attestation = {
        "schema": "summit.numa_policy_attestation.v1",
        "mode": "membind",
        "requested_nodes": "0",
        "effective_nodes": [0],
        "task_count_at_application": 1,
        "applied_before_numeric_import": True,
        "verified": True,
        "source": "libnuma",
        "applied_policy": "libnuma:membind:0",
        "pid": os.getpid(),
    }
    monkeypatch.setattr(
        gxe_multi,
        "current_numa_policy_attestation",
        lambda: dict(attestation),
    )
    executor = _MultiEnvironmentGemm(
        "python", types.SimpleNamespace(num_threads=1, dtype=np.float64)
    )
    report = executor.performance_report(
        include_records=False, phase_telemetry_complete=False
    )

    assert report["early_numa_attestation"] == attestation
    assert executor._numa_process_evidence["early_numa_attestation"] == attestation


def _assert_vendor_record(
    record: dict,
    *,
    operation: str,
    transpose_a: str,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    threads: int,
    expected_a_start_page_offset: int | None = None,
) -> None:
    assert record["schema_version"] == 1
    assert record["operation"] == operation
    assert record["arithmetic_dtype"] == "float64"
    assert record["layout"] == "column_major"
    assert record["transpose_a"] == transpose_a
    assert record["transpose_b"] == "N"
    assert (record["m"], record["n"], record["k"]) == (m, n, k)
    assert (record["lda"], record["ldb"], record["ldc"]) == (lda, ldb, ldc)
    assert record["alpha"] == 1.0
    assert record["beta"] == 0.0
    assert record["flop_count"] == float(2 * m * n * k)
    assert record["wall_seconds"] > 0.0
    assert record["process_cpu_seconds"] > 0.0
    assert record["gflops_per_second"] > 0.0
    assert record["process_cpu_to_wall_ratio"] > 0.0
    assert record["active_core_equivalents"] > 0.0
    assert record["requested_threads"] == threads
    assert record["configured_threads"] == threads
    assert record["backend_threads"] == threads
    assert record["omp_in_parallel"] is False
    assert record["omp_level"] == 0
    assert record["omp_active_level"] == 0
    assert record["omp_num_threads"] == 1
    assert record["omp_thread_num"] == 0
    assert record["omp_max_active_levels"] >= 1
    assert record["omp_max_threads"] >= 1
    affinity = set(os.sched_getaffinity(0))
    assert record["cpu_affinity_count"] == len(affinity)
    assert record["cpu_affinity_list"] == _affinity_list(affinity)
    assert record["entry_cpu"] in affinity
    assert record["exit_cpu"] in affinity
    assert record["backend"]
    assert record["backend_corename"]
    assert record["backend_config"]
    snapshot = dict(record["native_integrity_snapshot_numa"])
    common_snapshot = {
        "schema": "summit.native_integrity_snapshot_numa.v1",
        "schema_version": 1,
        "operand_role": "native_integrity_snapshot_of_logical_b",
        "contract_required": False,
        "complete": False,
    }
    for key, value in common_snapshot.items():
        assert snapshot[key] == value
    checksum_enabled = gxeldcore.build_info()["gemm_checksum_enabled"] is True
    if operation == "dgemm_nn" and checksum_enabled:
        assert snapshot == {
            **common_snapshot,
            "integrity_check_shape_eligible": True,
            "integrity_check_executed": True,
            "snapshot_available": True,
            "sealed_read_only_before_vendor": True,
        }
    else:
        assert snapshot == {
            **common_snapshot,
            "integrity_check_shape_eligible": False,
            "integrity_check_executed": False,
            "snapshot_available": False,
        }
    numa = dict(record["operand_numa_page_samples"])
    assert numa["schema_version"] == 1
    assert numa["sampling_method"] == "move_pages_query_no_migration"
    assert numa["sampling_timing"] == "after_vendor_call_outside_timed_interval"
    assert numa["sample_limit_per_operand"] == 8
    assert numa["address_selection_schema_version"] == 1
    assert numa["address_selection_policy"] == _NUMA_ADDRESS_SELECTION_POLICY
    assert numa["selected_addresses_are_page_bases"] is True
    assert numa["partial_boundary_pages_included"] is False
    assert numa["first_and_last_fully_contained_pages_selected"] is True
    assert numa["virtual_addresses_exposed"] is False
    assert numa["operand_byte_range_semantics"] == (
        "[start_address,end_exclusive_address)"
    )
    assert numa["address_evidence"] == (
        "ordered_samples_with_operand_relative_byte_offsets_and_full_page_indices"
    )
    page_size = int(numa["system_page_size"])
    assert page_size > 0
    assert isinstance(numa["syscall_result"], int)
    assert isinstance(numa["syscall_errno"], int)
    operands = {key: dict(value) for key, value in dict(numa["operands"]).items()}
    assert set(operands) == {"a", "b", "c"}
    element_size = np.dtype(np.float64).itemsize
    expected_byte_counts = {
        "a": element_size * (
            ((k - 1) * lda + m)
            if transpose_a == "N"
            else ((m - 1) * lda + k)
        ),
        "b": element_size * ((n - 1) * ldb + k),
        "c": element_size * ((n - 1) * ldc + m),
    }
    permitted_statuses = {
        "queried",
        "partial",
        "permission_denied",
        "unsupported",
        "syscall_error",
        "page_query_failed",
    }
    for operand_name, operand in operands.items():
        assert operand["query_status"] in permitted_statuses
        byte_count = int(operand["operand_byte_count"])
        assert byte_count == expected_byte_counts[operand_name]
        start_page_offset = int(operand["operand_start_address_page_offset"])
        end_page_offset = int(
            operand["operand_end_exclusive_address_page_offset"]
        )
        assert 0 <= start_page_offset < page_size
        assert end_page_offset == (start_page_offset + byte_count) % page_size
        if operand_name == "a" and expected_a_start_page_offset is not None:
            assert start_page_offset == expected_a_start_page_offset
        expected_storage_pages = (
            start_page_offset + byte_count + page_size - 1
        ) // page_size
        assert operand["storage_span_pages"] == expected_storage_pages
        bytes_to_first_full_page = (-start_page_offset) % page_size
        expected_full_pages = (
            (byte_count - bytes_to_first_full_page) // page_size
            if bytes_to_first_full_page <= byte_count
            else 0
        )
        assert operand["fully_contained_pages"] == expected_full_pages
        expected_sample_pages = min(expected_full_pages, 8)
        assert operand["selected_sample_pages"] == expected_sample_pages
        assert expected_sample_pages == 8
        assert 0 <= operand["resolved_sample_pages"] <= operand[
            "selected_sample_pages"
        ]
        assert 0 <= operand["page_query_error_pages"] <= operand[
            "selected_sample_pages"
        ]
        node_histogram = dict(operand["node_histogram"])
        error_histogram = dict(operand["page_error_errno_histogram"])
        assert all(int(node) >= 0 and int(count) > 0
                   for node, count in node_histogram.items())
        assert all(int(error) > 0 and int(count) > 0
                   for error, count in error_histogram.items())
        assert sum(node_histogram.values()) == operand["resolved_sample_pages"]
        assert sum(error_histogram.values()) == operand["page_query_error_pages"]
        evidence = [dict(item) for item in operand["ordered_samples"]]
        assert len(evidence) == expected_sample_pages
        expected_indices = [
            sample * (expected_full_pages - 1) // (expected_sample_pages - 1)
            for sample in range(expected_sample_pages)
        ]
        assert [item["sample_ordinal"] for item in evidence] == list(
            range(expected_sample_pages)
        )
        assert [item["full_page_index"] for item in evidence] == expected_indices
        assert [item["byte_offset_from_operand_start"] for item in evidence] == [
            bytes_to_first_full_page + page_index * page_size
            for page_index in expected_indices
        ]
        assert expected_indices[0] == 0
        assert expected_indices[-1] == expected_full_pages - 1
        for item in evidence:
            byte_offset = int(item["byte_offset_from_operand_start"])
            assert byte_offset >= 0
            assert (start_page_offset + byte_offset) % page_size == 0
            assert byte_offset + page_size <= byte_count
            if item["status_kind"] == "numa_node":
                assert item["raw_move_pages_status"] == item["numa_node"]
                assert item["numa_node"] >= 0
                assert item["page_query_errno"] is None
            elif item["status_kind"] == "page_query_error":
                assert item["raw_move_pages_status"] < 0
                assert item["numa_node"] is None
                assert item["page_query_errno"] == -item["raw_move_pages_status"]
            else:
                assert item["status_kind"] == "unavailable"
                assert item["raw_move_pages_status"] is None
                assert item["numa_node"] is None
                assert item["page_query_errno"] is None
        if operand["query_status"] == "queried":
            assert operand["resolved_sample_pages"] == operand[
                "selected_sample_pages"
            ]
            assert operand["page_query_error_pages"] == 0
            assert all(item["status_kind"] == "numa_node" for item in evidence)
            assert all(
                item["raw_move_pages_status"] == item["numa_node"]
                and item["page_query_errno"] is None
                for item in evidence
            )
        elif operand["query_status"] == "partial":
            assert 0 < operand["resolved_sample_pages"] < operand[
                "selected_sample_pages"
            ]
            assert {item["status_kind"] for item in evidence} == {
                "numa_node",
                "page_query_error",
            }
        elif operand["query_status"] == "page_query_failed":
            assert operand["resolved_sample_pages"] == 0
            assert operand["page_query_error_pages"] == operand[
                "selected_sample_pages"
            ]
            assert all(
                item["status_kind"] == "page_query_error" for item in evidence
            )
        else:
            # Linux security policy may deny move_pages even for the current
            # process.  Denial/unavailability is valid evidence, not a test
            # failure and never changes the GEMM result.
            assert operand["resolved_sample_pages"] == 0
            assert operand["page_query_error_pages"] == 0
            assert all(item["status_kind"] == "unavailable" for item in evidence)
            assert all(
                item["raw_move_pages_status"] is None
                and item["numa_node"] is None
                and item["page_query_errno"] is None
                for item in evidence
            )
    assert record["completed"] is True


def test_api9_numa_selector_reports_operand_without_a_full_page() -> None:
    operand = np.array([1.0], dtype=np.float64)
    selection = dict(gxeldcore._test_operand_numa_page_selection(operand))
    page_size = int(selection["system_page_size"])
    start_page_offset = int(selection["operand_start_address_page_offset"])

    assert selection["address_selection_schema_version"] == 1
    assert selection["address_selection_policy"] == _NUMA_ADDRESS_SELECTION_POLICY
    assert selection["partial_boundary_pages_included"] is False
    assert selection["query_status"] == "no_full_pages"
    assert selection["operand_byte_count"] == np.dtype(np.float64).itemsize
    assert selection["storage_span_pages"] == 1
    assert selection["fully_contained_pages"] == 0
    assert selection["selected_sample_pages"] == 0
    assert selection["resolved_sample_pages"] == 0
    assert selection["page_query_error_pages"] == 0
    assert list(selection["ordered_samples"]) == []
    assert list(selection["selected_page_base_relative_offsets"]) == []
    assert selection["selected_page_bases_match_ordered_offsets"] is True
    assert dict(selection["node_histogram"]) == {}
    assert dict(selection["page_error_errno_histogram"]) == {}
    assert selection["operand_end_exclusive_address_page_offset"] == (
        start_page_offset + np.dtype(np.float64).itemsize
    ) % page_size

    large = np.arange(page_size * 10 // 8 + 3, dtype=np.float64)[1:]
    large_selection = dict(gxeldcore._test_operand_numa_page_selection(large))
    ordered_offsets = [
        int(item["byte_offset_from_operand_start"])
        for item in large_selection["ordered_samples"]
    ]
    assert list(large_selection["selected_page_base_relative_offsets"]) == (
        ordered_offsets
    )
    assert large_selection["selected_page_bases_match_ordered_offsets"] is True


def test_native_integrity_snapshot_legacy_mode_seals_without_locality_claims(
    monkeypatch,
) -> None:
    import pytest

    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED", raising=False)
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_PROVENANCE", raising=False)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    evidence = dict(
        gxeldcore._test_native_integrity_snapshot_numa(page_size + 17)
    )
    assert evidence == {
        "schema": "summit.native_integrity_snapshot_numa.v1",
        "schema_version": 1,
        "operand_role": "native_integrity_snapshot_of_logical_b",
        "integrity_check_shape_eligible": True,
        "integrity_check_executed": True,
        "snapshot_available": True,
        "contract_required": False,
        "complete": False,
        "sealed_read_only_before_vendor": True,
    }

    # Legacy gw_ldscore policy modes publish APPLIED without the immutable
    # pre-import provenance marker.  That one-marker state remains explicitly
    # uncontracted and must not acquire locality claims.
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "libnuma:interleave:all")
    applied_only = dict(
        gxeldcore._test_native_integrity_snapshot_numa(page_size)
    )
    assert applied_only == {
        **evidence,
        "sealed_read_only_before_vendor": True,
    }

    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED")
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_PROVENANCE", "pre_numeric_import")
    with pytest.raises(RuntimeError, match="Partial .* markers are forbidden"):
        gxeldcore._test_native_integrity_snapshot_numa(page_size)
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "forged")
    with pytest.raises(RuntimeError, match="not a libnuma membind"):
        gxeldcore._test_native_integrity_snapshot_numa(page_size)
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "libnuma:membind:0")
    assert gxeldcore._test_native_integrity_snapshot_request_match(0) is True
    with pytest.raises(RuntimeError, match="NUMA nodes changed before vendor"):
        gxeldcore._test_native_integrity_snapshot_request_match(1)


def test_native_integrity_snapshot_contract_queries_every_page_in_fresh_process(
    tmp_path,
) -> None:
    import ctypes.util
    import pathlib
    import subprocess
    import sys
    import textwrap

    import pytest
    from summit import _early_numa

    if sys.platform != "linux" or ctypes.util.find_library("numa") is None:
        pytest.skip("native integrity snapshot contract requires Linux libnuma")

    selected_node = _early_numa._read_mems_allowed()[0]
    package_root = pathlib.Path(_early_numa.__file__).resolve().parent.parent
    native_directory = pathlib.Path(gxeldcore.__file__).resolve().parent
    dependency_root = pathlib.Path(np.__file__).resolve().parent.parent
    code = textwrap.dedent(
        """
        import hashlib
        import os
        import re
        import struct

        from summit._early_numa import apply_early_numa_membind

        node = int(os.environ["SUMMIT_TEST_NUMA_NODE"])
        try:
            apply_early_numa_membind(str(node))
        except OSError as error:
            print(f"NUMA_BIND_UNAVAILABLE:{error}")
            raise SystemExit(77)
        import summit
        summit.__path__.insert(0, os.environ["SUMMIT_TEST_NATIVE_DIRECTORY"])
        from summit import gxeldcore

        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = 33
        logical_byte_count = page_count * page_size - 17
        evidence = dict(
            gxeldcore._test_native_integrity_snapshot_numa(logical_byte_count)
        )
        assert evidence["schema"] == "summit.native_integrity_snapshot_numa.v1"
        assert evidence["schema_version"] == 1
        assert evidence["operand_role"] == (
            "native_integrity_snapshot_of_logical_b"
        )
        assert evidence["contract_required"] is True
        assert evidence["integrity_check_shape_eligible"] is True
        assert evidence["integrity_check_executed"] is True
        assert evidence["snapshot_available"] is True
        assert evidence["logical_byte_count"] == logical_byte_count
        assert evidence["mapping_bytes"] == page_count * page_size
        assert evidence["page_size"] == page_size
        assert evidence["mapping_page_count"] == page_count
        assert list(evidence["selected_nodes"]) == [node]
        assert evidence["policy_mode"] == "bind_static_nodes"
        assert evidence["policy_mode_value"] == 32770
        assert evidence["anonymous_private_mapping"] is True
        assert evidence["page_aligned_mapping"] is True
        assert evidence["bound_before_first_touch"] is True
        assert evidence["live_owner_policy_verified"] is True
        assert evidence["pre_touch_range_policy_verified"] is True
        assert evidence["pre_vendor_complete_page_query"] is True
        assert evidence["queried_pages"] == page_count
        assert evidence["resolved_pages"] == page_count
        assert evidence["query_chunks"] == 1
        assert evidence["query_chunk_page_limit"] == 65536
        assert dict(evidence["node_histogram"]) == {str(node): page_count}
        assert evidence["ordered_status_encoding"] == (
            "signed_int32_little_endian"
        )
        expected_digest = hashlib.sha256(
            b"".join(struct.pack("<i", node) for _ in range(page_count))
        ).hexdigest()
        assert evidence["ordered_status_sha256"] == expected_digest
        assert re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        assert evidence["pre_vendor_strict_policy_verified"] is True
        assert evidence["strict_policy_check"] == (
            "MPOL_MF_STRICT_without_MPOL_MF_MOVE"
        )
        assert evidence["sealed_read_only_before_vendor"] is True
        assert evidence["page_query_method"] == (
            "move_pages_query_no_migration"
        )
        assert evidence["page_migration_requested"] is False
        assert evidence["placement_repair_performed"] is False
        assert evidence["complete"] is True
        print("NATIVE_INTEGRITY_SNAPSHOT_NUMA_PASS")
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment.pop("SUMMIT_NUMA_POLICY_APPLIED", None)
    environment.pop("SUMMIT_NUMA_POLICY_PROVENANCE", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(tmp_path / "empty-pycache"),
            "PYTHONPATH": os.pathsep.join(
                (str(package_root), str(dependency_root))
            ),
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_PROC_BIND": "FALSE",
            "SUMMIT_TEST_NUMA_NODE": str(selected_node),
            "SUMMIT_TEST_NATIVE_DIRECTORY": str(native_directory),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-B", "-c", code],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode == 77 and completed.stdout.startswith(
        "NUMA_BIND_UNAVAILABLE:"
    ):
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "NATIVE_INTEGRITY_SNAPSHOT_NUMA_PASS"


def test_api9_vendor_telemetry_records_protected_nn_and_tn(monkeypatch) -> None:
    """Cross the integrity threshold while keeping the test allocation bounded."""
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED", raising=False)
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_PROVENANCE", raising=False)
    info = dict(gxeldcore.build_info())
    assert info["api_version"] == 9
    threads = _configured_native_threads()
    rng = np.random.default_rng(70131)
    m, n, k = 512, 512, 2048

    gxeldcore.reset_gemm_telemetry()
    status = dict(gxeldcore.gemm_telemetry_status())
    assert status["schema_version"] == 1
    assert status["capacity"] >= 2
    assert status["buffered_records"] == 0
    assert status["captured_records"] == 0
    assert status["dropped_records"] == 0
    assert status["operand_numa_sampling_method"] == (
        "move_pages_query_no_migration"
    )
    assert status["operand_numa_sample_limit_per_operand"] == 8
    assert status["operand_numa_address_selection_schema_version"] == 1
    assert status["operand_numa_address_selection_policy"] == (
        _NUMA_ADDRESS_SELECTION_POLICY
    )
    assert status["operand_numa_partial_boundary_pages_included"] is False

    left_nn = _misaligned_fortran_normal(rng, (m, k))
    right_nn = np.asfortranarray(rng.normal(size=(k, n)))
    observed_nn, repaired_nn = gxeldcore.protected_matmul_nn(
        left_nn, right_nn, threads
    )
    np.testing.assert_allclose(
        observed_nn, left_nn @ right_nn, rtol=3e-14, atol=3e-12
    )
    assert repaired_nn == 0

    left_tn = _misaligned_fortran_normal(rng, (k, m))
    right_tn = np.asfortranarray(rng.normal(size=(k, n)))
    observed_tn, repaired_tn = gxeldcore.protected_matmul_tn(
        left_tn, right_tn, threads
    )
    np.testing.assert_allclose(
        observed_tn, left_tn.T @ right_tn, rtol=3e-14, atol=3e-12
    )
    assert repaired_tn == 0

    status = dict(gxeldcore.gemm_telemetry_status())
    assert status["buffered_records"] == 2
    assert status["captured_records"] == 2
    assert status["dropped_records"] == 0
    records = [dict(record) for record in gxeldcore.consume_gemm_telemetry()]
    assert len(records) == 2
    assert records[0]["sequence"] < records[1]["sequence"]
    _assert_vendor_record(
        records[0],
        operation="dgemm_nn",
        transpose_a="N",
        m=m,
        n=n,
        k=k,
        lda=m,
        ldb=k,
        ldc=m,
        threads=threads,
        expected_a_start_page_offset=np.dtype(np.float64).itemsize,
    )
    _assert_vendor_record(
        records[1],
        operation="dgemm_tn",
        transpose_a="T",
        m=m,
        n=n,
        k=k,
        lda=k,
        ldb=k,
        ldc=m,
        threads=threads,
        expected_a_start_page_offset=np.dtype(np.float64).itemsize,
    )
    status = dict(gxeldcore.gemm_telemetry_status())
    assert status["buffered_records"] == 0
    assert status["captured_records"] == 2
    assert status["dropped_records"] == 0


def test_api9_build_identity_and_vendor_entry_guard() -> None:
    info = dict(gxeldcore.build_info())
    assert info["api_version"] == 9
    assert info["backend_version"] == "1.9"
    assert info["openmp_effective_capacity_policy"] == (
        "bound_places_else_sched_affinity_v1"
    )
    assert info["openmp_placement_contract_supported"] is True
    assert info["openmp_placement_contract_schema"] == (
        "summit.openmp_placement_attestation.v1"
    )
    assert isinstance(info["openmp_placement_contract_configured"], bool)
    assert info["openmp_placement_contract_immutable"] is True
    assert info["openmp_placement_probe_vendor_calls"] == 0
    if info["openmp_placement_contract_configured"]:
        assert dict(info["openmp_placement_contract_evidence"])["verified"] is True
    else:
        assert info["openmp_placement_contract_evidence"] is None
    assert info["gemm_telemetry_schema_version"] == 1
    assert info["gemm_telemetry_capacity"] > 0
    assert info["gemm_vendor_entry_outer_openmp_guard"] is True
    assert info["gemm_integrity_minimum_vendor_flops"] == (
        1_000_000_000 if info["gemm_integrity_enabled"] else 0
    )
    assert info["gemm_operand_numa_sampling_method"] == (
        "move_pages_query_no_migration"
    )
    assert info["gemm_operand_numa_sample_limit_per_operand"] == 8
    assert info["gemm_operand_numa_address_selection_schema_version"] == 1
    assert info["gemm_operand_numa_address_selection_policy"] == (
        _NUMA_ADDRESS_SELECTION_POLICY
    )
    assert info["gemm_operand_numa_partial_boundary_pages_included"] is False
    assert info["native_integrity_snapshot_numa_contract_supported"] is True
    assert info["native_integrity_snapshot_numa_contract_schema"] == (
        "summit.native_integrity_snapshot_numa.v1"
    )
    assert info["native_integrity_snapshot_numa_query_chunk_page_limit"] == 65536
    assert info["openmp_enabled"] is True
    assert info["blas_runtime_corename"]
    assert info["blas_runtime_config"]

    guard = dict(gxeldcore._test_vendor_entry_guard())
    assert guard["openmp_enabled"] is True
    assert guard["outside_allowed"] is True
    assert guard["outside_omp_in_parallel"] is False
    assert guard["outside_omp_level"] == 0
    assert guard["nested_probe_executed"] is True
    assert guard["nested_checks"] > 0
    assert guard["nested_rejections"] == guard["nested_checks"]
    assert guard["all_nested_entries_rejected"] is True
    assert guard["production_boundary_checks"] == 1
    assert (
        guard["production_boundary_rejections"]
        == guard["production_boundary_checks"]
    )
    assert guard["production_boundary_output_changes"] == 0
    assert guard["all_production_boundary_entries_rejected"] is True
    assert guard["production_boundary_outputs_unchanged"] is True


def test_streamed_performance_phase_accumulation() -> None:
    estimator = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    estimator._record_performance_phase("bed_read_decode", 0.25, 0.50)
    estimator._record_performance_phase("bed_read_decode", 0.75, 1.25)
    estimator._record_performance_phase("genotype_standardization", 0.4, 0.6)

    assert estimator.performance_phase_timings == {
        "bed_read_decode": {
            "wall_seconds": 1.0,
            "process_cpu_seconds": 1.75,
            "calls": 2,
        },
        "genotype_standardization": {
            "wall_seconds": 0.4,
            "process_cpu_seconds": 0.6,
            "calls": 1,
        },
    }


def test_native_standardization_fans_out_shared_missingness_sinks(
    monkeypatch,
) -> None:
    samples, start, stop = 7, 2, 5
    raw = np.asfortranarray(np.arange(samples * (stop - start)).reshape(
        samples, stop - start
    ), dtype=np.float64)

    class FakeBed:
        def read(self, **_kwargs):
            return raw.copy(order="F")

    first = types.SimpleNamespace(
        genotype_missing_call_count=np.zeros(8, dtype=np.int64),
        genotype_missing_environment_correlation=np.zeros(8),
        genotype_missing_phenotype_correlation=np.zeros(8),
    )
    second = types.SimpleNamespace(
        genotype_missing_call_count=np.zeros(8, dtype=np.int64),
        genotype_missing_environment_correlation=np.zeros(8),
        genotype_missing_phenotype_correlation=np.zeros(8),
    )
    counts = np.array([1, 2, 3], dtype=np.int64)
    correlations = np.array(
        [
            [0.11, 0.12, 0.13],
            [0.21, 0.22, 0.23],
            [0.31, 0.32, 0.33],
            [0.41, 0.42, 0.43],
        ],
        dtype=np.float64,
        order="F",
    )

    def fake_standardize(genotype, targets, *_args):
        assert genotype.flags.f_contiguous
        assert targets.shape == (samples, 4)
        return counts, correlations

    monkeypatch.setattr(gxeldcore, "standardize_genotype_block", fake_standardize)
    reader = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    reader.genotype_format = "bed"
    reader.G = FakeBed()
    reader.row_sel = slice(None)
    reader.decode_threads = 1
    reader.nsamp = samples
    reader._native_parallel_standardization = True
    reader._native_missingness_targets = np.zeros((samples, 4), order="F")
    reader._native_missingness_sinks = (
        (first, "genotype_missing_environment_correlation"),
        (first, "genotype_missing_phenotype_correlation"),
        (second, "genotype_missing_environment_correlation"),
        (second, "genotype_missing_phenotype_correlation"),
    )
    reader.ddof = 1
    reader.genotype_scale = "sample"
    reader.eps_var = 1.0e-10
    reader.num_threads = 1

    observed = reader._read_genotype_block(start, stop)
    np.testing.assert_array_equal(observed, raw)
    for target in (first, second):
        np.testing.assert_array_equal(
            target.genotype_missing_call_count[start:stop], counts
        )
    np.testing.assert_allclose(
        first.genotype_missing_environment_correlation[start:stop], correlations[0]
    )
    np.testing.assert_allclose(
        first.genotype_missing_phenotype_correlation[start:stop], correlations[1]
    )
    np.testing.assert_allclose(
        second.genotype_missing_environment_correlation[start:stop], correlations[2]
    )
    np.testing.assert_allclose(
        second.genotype_missing_phenotype_correlation[start:stop], correlations[3]
    )
    assert reader.performance_phase_timings["bed_read_decode"]["calls"] == 1
    assert (
        reader.performance_phase_timings["genotype_standardization"]["calls"]
        == 1
    )


def test_numa_bound_bed_decode_uses_prebound_output_and_verifies_after_standardization(
    monkeypatch,
) -> None:
    from bed_reader import bed_reader
    from summit import _early_numa

    samples, variants, start, stop = 5, 9, 2, 6
    selected_rows = np.array([0, 2, 4, 6, 8], dtype=np.intp)
    raw = np.asfortranarray(
        np.arange(samples * (stop - start), dtype=np.float64).reshape(
            samples, stop - start
        )
    )
    events: list[str] = []
    owners: list[mmap.mmap] = []

    class FakeBed:
        iid_count = 10
        sid_count = variants
        count_A1 = True
        filepath = "/tmp/fake-bound-reader.bed"

        def read(self, **_kwargs):
            raise AssertionError("the allocating high-level BED path was used")

    def fake_allocate(byte_count, nodes):
        events.append("allocate")
        assert byte_count == raw.nbytes
        assert tuple(nodes) == (0, 1)
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        owner = mmap.mmap(-1, page_size)
        owners.append(owner)
        return owner, {
            "schema": "summit.numa_bound_anonymous_buffer.v1",
            "schema_version": 1,
            "byte_count": byte_count,
            "mapping_bytes": page_size,
            "page_count": 1,
            "selected_nodes": [0, 1],
            "bound_before_first_touch": True,
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }

    def fake_read(
        filename,
        iid_count,
        sid_count,
        is_a1_counted,
        iid_index,
        sid_index,
        val,
        num_threads,
    ):
        events.append("decode")
        assert filename == FakeBed.filepath
        assert iid_count == FakeBed.iid_count
        assert sid_count == variants
        assert is_a1_counted is True
        np.testing.assert_array_equal(iid_index, selected_rows)
        np.testing.assert_array_equal(sid_index, np.arange(start, stop))
        assert val.flags.f_contiguous
        assert num_threads == 2
        val[...] = raw

    def fake_standardize(genotype, targets, *_args):
        events.append("standardize")
        assert targets.shape == (samples, 2)
        genotype += 1.0
        return np.zeros(stop - start, dtype=np.int64), np.zeros(
            (2, stop - start), dtype=np.float64
        )

    def fake_verify(owner, byte_count, nodes):
        events.append("verify")
        assert owner is owners[0]
        assert byte_count == raw.nbytes
        assert tuple(nodes) == (0, 1)
        observed = np.ndarray(
            raw.shape, dtype=np.float64, buffer=owner, order="F"
        )
        np.testing.assert_array_equal(observed, raw + 1.0)
        return {
            "schema": "summit.numa_bound_anonymous_buffer.v1",
            "schema_version": 1,
            "byte_count": byte_count,
            "mapping_bytes": len(owner),
            "page_count": 1,
            "selected_nodes": [0, 1],
            "post_decode_complete_page_query": True,
            "queried_pages": 1,
            "resolved_pages": 1,
            "page_migration_requested": False,
            "placement_repair_performed": False,
            "node_histogram": {"0": 1},
            "complete": True,
        }

    monkeypatch.setattr(
        _early_numa, "allocate_numa_bound_anonymous_buffer", fake_allocate
    )
    monkeypatch.setattr(
        _early_numa, "verify_numa_bound_anonymous_buffer", fake_verify
    )
    monkeypatch.setattr(bed_reader, "read_f64", fake_read)
    monkeypatch.setattr(gxeldcore, "standardize_genotype_block", fake_standardize)

    reader = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    reader.genotype_format = "bed"
    reader.G = FakeBed()
    reader.row_sel = selected_rows
    reader.decode_threads = 2
    reader.nsamp = samples
    reader._native_parallel_standardization = True
    reader._native_numa_bound_decode_nodes = (0, 1)
    reader._native_numa_bound_decode_records = []
    reader._native_missingness_targets = np.zeros((samples, 2), order="F")
    reader.ddof = 1
    reader.genotype_scale = "sample"
    reader.eps_var = 1.0e-10
    reader.num_threads = 2

    observed = reader._read_genotype_block(start, stop)
    assert events == ["allocate", "decode", "standardize", "verify"]
    assert observed.flags.f_contiguous
    np.testing.assert_array_equal(observed, raw + 1.0)
    assert reader._native_numa_bound_iid_index is selected_rows
    assert reader._native_numa_bound_decode_records == [
        {
            "genotype_block": [start, stop],
            "memory_order": "F",
            "decoder": "bed_reader.read_f64_into_bound_mapping",
            "allocation": {
                "schema": "summit.numa_bound_anonymous_buffer.v1",
                "schema_version": 1,
                "byte_count": raw.nbytes,
                "mapping_bytes": len(owners[0]),
                "page_count": 1,
                "selected_nodes": [0, 1],
                "bound_before_first_touch": True,
                "page_migration_requested": False,
                "placement_repair_performed": False,
            },
            "bound_mapping_preserved_after_standardization": True,
            "verification_stage": "post_standardization_pre_return",
            "verification": {
                "schema": "summit.numa_bound_anonymous_buffer.v1",
                "schema_version": 1,
                "byte_count": raw.nbytes,
                "mapping_bytes": len(owners[0]),
                "page_count": 1,
                "selected_nodes": [0, 1],
                "post_decode_complete_page_query": True,
                "queried_pages": 1,
                "resolved_pages": 1,
                "page_migration_requested": False,
                "placement_repair_performed": False,
                "node_histogram": {"0": 1},
                "complete": True,
            },
        }
    ]
    del observed
    owners[0].close()


def test_real_bed_decode_retains_static_bound_mapping_without_gemm(
    tmp_path,
) -> None:
    """Exercise the installed BED/native ABI in a fresh pre-import process."""
    import ctypes.util
    import pathlib
    import subprocess
    import sys
    import textwrap

    import pytest
    import summit
    from bed_reader import to_bed
    from summit._early_numa import _read_mems_allowed

    if sys.platform != "linux" or ctypes.util.find_library("numa") is None:
        pytest.skip("real NUMA-bound BED smoke requires Linux libnuma")

    samples, variants = 257, 3
    raw = (
        np.arange(samples, dtype=np.int64).reshape(-1, 1)
        + 2 * np.arange(variants, dtype=np.int64).reshape(1, -1)
    ) % 3
    raw = raw.astype(np.float64)
    bed_path = tmp_path / "bound-smoke.bed"
    to_bed(bed_path, raw)
    selected_node = _read_mems_allowed()[0]
    package_root = pathlib.Path(summit.__file__).resolve().parent.parent
    native_directory = pathlib.Path(gxeldcore.__file__).resolve().parent
    dependency_root = pathlib.Path(np.__file__).resolve().parent.parent
    code = textwrap.dedent(
        """
        import mmap
        import os
        from pathlib import Path

        from summit._early_numa import (
            apply_early_numa_membind,
            verify_numa_bound_anonymous_buffer,
        )

        node = int(os.environ["SUMMIT_TEST_NUMA_NODE"])
        try:
            attestation = apply_early_numa_membind(str(node))
        except OSError as error:
            print(f"NUMA_BIND_UNAVAILABLE:{error}")
            raise SystemExit(77)
        import summit
        summit.__path__.append(os.environ["SUMMIT_TEST_NATIVE_DIRECTORY"])
        import numpy as np
        from bed_reader import open_bed
        from summit import gxeldcore
        from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore

        bed = open_bed(Path(os.environ["SUMMIT_TEST_BED_PATH"]))
        plain = np.asfortranarray(bed.read(dtype=np.float64), dtype=np.float64)
        targets = np.zeros((plain.shape[0], 2), dtype=np.float64, order="F")
        expected = plain.copy(order="F")
        gxeldcore.standardize_genotype_block(
            expected, targets, 1, False, 1.0e-10, 1
        )

        reader = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
        reader.genotype_format = "bed"
        reader.G = bed
        reader.row_sel = np.arange(plain.shape[0], dtype=np.intp)
        reader.decode_threads = 1
        reader.nsamp = plain.shape[0]
        reader._native_parallel_standardization = True
        reader._native_numa_bound_decode_nodes = (node,)
        reader._native_numa_bound_decode_records = []
        reader._native_missingness_targets = targets
        reader.ddof = 1
        reader.genotype_scale = "sample"
        reader.eps_var = 1.0e-10
        reader.num_threads = 1

        observed = reader._read_genotype_block(0, plain.shape[1])
        np.testing.assert_array_equal(observed, expected)
        record = reader._native_numa_bound_decode_records[0]
        verification = record["verification"]
        assert isinstance(observed.base, mmap.mmap)
        assert record["bound_mapping_preserved_after_standardization"] is True
        assert verification["complete"] is True
        assert verification["queried_pages"] == verification["page_count"] == 2
        assert verification["node_histogram"] == {str(node): 2}
        assert verification["page_migration_requested"] is False
        assert verification["placement_repair_performed"] is False
        repeated = verify_numa_bound_anonymous_buffer(
            observed.base, observed.nbytes, (node,), chunk_pages=1
        )
        assert repeated["query_chunks"] == repeated["page_count"] == 2
        assert repeated["node_histogram"] == {str(node): 2}
        assert attestation["static_nodes"] is True
        assert gxeldcore.gemm_telemetry_status()["captured_records"] == 0
        print("BOUND_BED_SMOKE_PASS")
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(tmp_path / "empty-pycache"),
            "PYTHONPATH": os.pathsep.join(
                (str(package_root), str(dependency_root))
            ),
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_PROC_BIND": "FALSE",
            "SUMMIT_TEST_NUMA_NODE": str(selected_node),
            "SUMMIT_TEST_NATIVE_DIRECTORY": str(native_directory),
            "SUMMIT_TEST_BED_PATH": str(bed_path),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-B", "-c", code],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode == 77 and completed.stdout.startswith(
        "NUMA_BIND_UNAVAILABLE:"
    ):
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "BOUND_BED_SMOKE_PASS"


def test_multi_environment_records_native_semantics_and_phase_context() -> None:
    estimator = types.SimpleNamespace(num_threads=3, dtype=np.dtype("float32"))
    executor = _MultiEnvironmentGemm("python", estimator)
    native_records: list[dict] = []

    class FakeNative:
        @staticmethod
        def protected_matmul_nn(left, right, threads):
            assert threads == 3
            native_records.append(
                {
                    "schema_version": 1,
                    "operation": "dgemm_nn",
                    "layout": "column_major",
                    "transpose_a": "N",
                    "transpose_b": "N",
                    "m": left.shape[0],
                    "n": right.shape[1],
                    "k": left.shape[1],
                    "lda": left.shape[0],
                    "ldb": right.shape[0],
                    "ldc": left.shape[0],
                    "wall_seconds": 0.25,
                    "process_cpu_seconds": 0.50,
                    "completed": True,
                }
            )
            return left @ right, 0

    executor.protected = True
    executor.compute_dtype = np.dtype("float64")
    executor._module = FakeNative()
    executor._native_telemetry_available = True

    def consume_native_records():
        consumed = list(native_records)
        native_records.clear()
        return consumed

    executor._native_telemetry_consumer = consume_native_records
    executor._native_telemetry_getter = None
    executor.backend_name = "gxeldcore_direct"
    executor.backend_version = "1.4"
    executor.backend_build_sha256 = "a" * 64
    executor.source_commit = "b" * 40
    executor.source_tree_sha256 = "c" * 64

    left = np.asfortranarray(np.arange(30, dtype=np.float64).reshape(6, 5))
    right = np.asfortranarray(np.arange(20, dtype=np.float64).reshape(5, 4))
    with executor.phase("source_gemm", {"environment_tile": [1, 3]}):
        with executor.semantic_context(
            {"block_index": 7, "genotype_block": [14000, 16000]}
        ):
            observed = executor.nn(left, right, semantic={"probe_tile": [32, 16]})
    np.testing.assert_array_equal(observed, left @ right)

    source_only_report = executor.performance_report(include_records=False)
    assert source_only_report["hot_gemm_telemetry_complete"] is False
    assert source_only_report["hot_gemm_phase_coverage"]["source_gemm"][
        "complete"
    ] is True
    assert source_only_report["hot_gemm_phase_coverage"]["target_gemm"][
        "complete"
    ] is False

    with executor.phase("target_gemm", {"environment_tile": [1, 3]}):
        with executor.semantic_context(
            {"block_index": 8, "genotype_block": [16000, 18000]}
        ):
            executor.nn(left, right, semantic={"probe_tile": [32, 16]})

    report = executor.performance_report(include_records=True)
    assert report["arithmetic_dtype"] == "float64"
    assert report["requested_storage_dtype"] == "float32"
    assert report["gemm_record_count"] == 2
    assert report["dropped_gemm_records"] == 0
    assert report["phase_totals"]["source_gemm"]["calls"] == 1
    assert report["expected_hot_gemm_call_count"] == 2
    assert report["hot_gemm_logical_call_count"] == 2
    assert report["hot_gemm_telemetry_complete"] is True
    assert report["hot_gemm_phase_coverage"]["source_gemm"]["complete"] is True
    assert report["hot_gemm_phase_coverage"]["target_gemm"]["complete"] is True
    record = report["gemm_records"][0]
    assert record["telemetry_scope"] == "vendor_call"
    assert record["backend"] == "gxeldcore_direct"
    assert record["backend_build_sha256"] == "a" * 64
    assert record["native_source_commit"] == "b" * 40
    assert record["native_source_tree_sha256"] == "c" * 64
    assert record["arithmetic_dtype"] == "float64"
    assert record["requested_storage_dtype"] == "float32"
    assert record["actual_left_storage_dtype"] == "float64"
    assert record["actual_right_storage_dtype"] == "float64"
    assert record["actual_output_storage_dtype"] == "float64"
    assert record["requested_blas_threads"] == 3
    assert record["phase"] == "source_gemm"
    assert record["environment_tile"] == [1, 3]
    assert record["block_index"] == 7
    assert record["genotype_block"] == [14000, 16000]
    assert record["probe_tile"] == [32, 16]
    assert (record["m"], record["n"], record["k"]) == (6, 4, 5)
    assert (record["lda"], record["ldb"], record["ldc"]) == (6, 5, 6)
    assert record["flops"] == 2 * 6 * 4 * 5
    assert record["achieved_gflops"] > 0.0
    assert record["average_active_cores"] == 2.0


def test_protected_report_rejects_vacuous_zero_hot_gemm_coverage() -> None:
    estimator = types.SimpleNamespace(num_threads=1, dtype=np.dtype("float64"))
    executor = _MultiEnvironmentGemm("python", estimator)
    executor.protected = True
    executor._native_telemetry_available = True
    executor._native_telemetry_status_getter = lambda: {
        "captured_records": 0,
        "dropped_records": 0,
    }
    report = executor.performance_report(include_records=True)
    assert report["expected_hot_gemm_call_count"] == 0
    assert report["hot_gemm_logical_call_count"] == 0
    assert report["hot_gemm_telemetry_complete"] is False
    assert report["hot_gemm_phase_coverage"]["source_gemm"]["complete"] is False
    assert report["hot_gemm_phase_coverage"]["target_gemm"]["complete"] is False


def test_python_missingness_fans_out_shared_environment_diagnostics() -> None:
    samples = 6
    width = 3

    def estimator(environment: np.ndarray) -> GenomewideEnvLDScore:
        observed = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
        observed.nsamp = samples
        observed.env = np.asarray(environment, dtype=np.float64)
        observed.pheno = None
        observed.genotype_missing_call_count = np.zeros(width, dtype=np.int64)
        observed.genotype_missing_environment_correlation = np.zeros(
            width, dtype=np.float64
        )
        observed.genotype_missing_phenotype_correlation = np.zeros(
            width, dtype=np.float64
        )
        return observed

    first = estimator(np.arange(samples, dtype=np.float64))
    second = estimator(np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float64))
    first._shared_missingness_estimators = (first, second)
    mask = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [True, True, False],
            [False, False, True],
            [False, False, True],
            [False, False, False],
        ],
        dtype=bool,
    )

    first._record_genotype_missingness(0, width, mask)

    expected_counts = mask.sum(axis=0, dtype=np.int64)
    np.testing.assert_array_equal(first.genotype_missing_call_count, expected_counts)
    np.testing.assert_array_equal(second.genotype_missing_call_count, expected_counts)
    assert not np.array_equal(
        first.genotype_missing_environment_correlation,
        second.genotype_missing_environment_correlation,
    )
    np.testing.assert_array_equal(
        first.genotype_missing_phenotype_correlation, np.zeros(width)
    )
    np.testing.assert_array_equal(
        second.genotype_missing_phenotype_correlation, np.zeros(width)
    )
