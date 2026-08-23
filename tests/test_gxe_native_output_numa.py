from __future__ import annotations

import ctypes.util
import gc
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from summit import gxeldcore


_SCHEMA = "summit.native_gemm_output_numa.v1"


def _configured_native_threads() -> int:
    desired = min(2, len(os.sched_getaffinity(0)))
    assert desired > 0
    try:
        return int(gxeldcore.configure_blas_threads(desired))
    except RuntimeError as exc:
        assert "different thread count" in str(exc)
        configured = int(gxeldcore.build_info()["blas_runtime_threads"])
        assert gxeldcore.configure_blas_threads(configured) == configured
        return configured


def test_native_gemm_output_build_capability_is_explicit() -> None:
    info = dict(gxeldcore.build_info())
    assert info["native_gemm_output_numa_contract_supported"] is True
    assert info["native_gemm_output_numa_contract_schema"] == _SCHEMA
    assert info["native_gemm_output_numa_query_chunk_page_limit"] == 65536
    assert info["native_gemm_output_numa_evidence_capacity"] > 0


def test_native_gemm_output_legacy_schema_queue_and_capsule_lifetime(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED", raising=False)
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_PROVENANCE", raising=False)
    gxeldcore.reset_gemm_telemetry()
    first_call_id = int(
        gxeldcore.native_gemm_output_numa_evidence_status()["next_call_id"]
    )

    left = np.asfortranarray(
        np.arange(55, dtype=np.float64).reshape(5, 11, order="F")
    )
    right = np.asfortranarray(
        np.arange(77, dtype=np.float64).reshape(11, 7, order="F")
    )
    observed, repaired = gxeldcore.protected_matmul_nn(
        left, right, _configured_native_threads()
    )
    np.testing.assert_array_equal(observed, left @ right)
    assert repaired == 0
    assert observed.flags.f_contiguous
    assert observed.flags.writeable

    evidence = [
        dict(item)
        for item in gxeldcore.get_native_gemm_output_numa_evidence()
    ]
    assert evidence == [
        {
            "schema": _SCHEMA,
            "schema_version": 2,
            "applicable": True,
            "operand_role": "protected_gemm_output",
            "contract_required": False,
            "complete": False,
            "call_id": first_call_id,
            "logical_rows": 5,
            "logical_columns": 7,
            "storage_layout": "column_major",
            "logical_byte_count": 5 * 7 * np.dtype(np.float64).itemsize,
            # One-shot outputs allocate exactly their logical shape, so the
            # v2 capacity equals the logical byte count.
            "capacity_byte_count": 5 * 7 * np.dtype(np.float64).itemsize,
            "allocation_mode": "legacy_posix_memalign",
        }
    ]
    status = dict(gxeldcore.native_gemm_output_numa_evidence_status())
    assert status == {
        "schema_version": 1,
        "capacity": status["capacity"],
        "buffered_records": 1,
        "captured_records": 1,
        "dropped_records": 0,
        "next_call_id": first_call_id + 1,
        "attempted_calls": 1,
        "verified_calls": 0,
        "legacy_calls": 1,
        "failed_calls": 0,
        "query_chunk_page_limit": 65536,
    }
    assert status["capacity"] > 0

    vendor_records = [dict(item) for item in gxeldcore.get_gemm_telemetry()]
    assert len(vendor_records) in (0, 1)
    if vendor_records:
        assert dict(vendor_records[0]["native_gemm_output_numa"]) == evidence[0]

    retained_view = observed[:, 1:]
    expected_sum = float(retained_view.sum())
    del observed
    gc.collect()
    assert float(retained_view.sum()) == expected_sum

    consumed = [
        dict(item)
        for item in gxeldcore.consume_native_gemm_output_numa_evidence()
    ]
    assert consumed == evidence
    assert gxeldcore.native_gemm_output_numa_evidence_status()[
        "buffered_records"
    ] == 0


def test_native_gemm_output_rejects_partial_contract_without_evidence(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED", raising=False)
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_PROVENANCE", "pre_numeric_import")
    gxeldcore.reset_gemm_telemetry()
    left = np.asfortranarray(np.ones((3, 2), dtype=np.float64))
    right = np.asfortranarray(np.ones((2, 4), dtype=np.float64))

    with pytest.raises(RuntimeError, match="Partial .* markers are forbidden"):
        gxeldcore.protected_matmul_nn(
            left, right, _configured_native_threads()
        )

    status = dict(gxeldcore.native_gemm_output_numa_evidence_status())
    assert status["attempted_calls"] == 1
    assert status["verified_calls"] == 0
    assert status["legacy_calls"] == 0
    assert status["failed_calls"] == 1
    assert status["captured_records"] == 0
    assert status["buffered_records"] == 0
    assert list(gxeldcore.get_gemm_telemetry()) == []


def test_native_gemm_tt_row_view_retains_distinct_vendor_layout(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED", raising=False)
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_PROVENANCE", raising=False)
    gxeldcore.reset_gemm_telemetry()
    rng = np.random.default_rng(93811)
    weights = np.asfortranarray(rng.normal(size=(13, 5)))
    genotype = np.asfortranarray(rng.normal(size=(17, 13)))

    observed, repaired = gxeldcore.protected_matmul_tt_row_major_output(
        weights, genotype, _configured_native_threads()
    )
    np.testing.assert_allclose(
        observed, genotype @ weights, rtol=2e-14, atol=2e-14
    )
    assert repaired == 0
    assert observed.flags.c_contiguous
    evidence = dict(gxeldcore.consume_native_gemm_output_numa_evidence()[0])
    assert evidence["storage_layout"] == "row_major"
    assert evidence["logical_rows"] == genotype.shape[0]
    assert evidence["logical_columns"] == weights.shape[1]

    vendor_records = [dict(item) for item in gxeldcore.consume_gemm_telemetry()]
    assert len(vendor_records) in (0, 1)
    if vendor_records:
        record = vendor_records[0]
        assert record["operation"] == "dgemm_tt"
        assert record["layout"] == "column_major"
        assert dict(record["native_gemm_output_numa"])["call_id"] == evidence[
            "call_id"
        ]


def test_native_gemm_output_contract_queries_every_page_in_fresh_process(
    tmp_path: Path,
) -> None:
    from summit import _early_numa

    if sys.platform != "linux" or ctypes.util.find_library("numa") is None:
        pytest.skip("native GEMM output NUMA contract requires Linux libnuma")

    selected_node = _early_numa._read_mems_allowed()[0]
    package_root = Path(_early_numa.__file__).resolve().parent.parent
    native_directory = Path(gxeldcore.__file__).resolve().parent
    dependency_root = Path(np.__file__).resolve().parent.parent
    code = textwrap.dedent(
        """
        import gc
        import hashlib
        import os
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
        import numpy as np

        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        rows, columns = page_size // 8 + 3, 5
        left = np.asfortranarray(
            np.arange(rows * 7, dtype=np.float64).reshape(rows, 7, order="F")
        )
        right = np.asfortranarray(
            np.arange(7 * columns, dtype=np.float64).reshape(7, columns, order="F")
        )
        gxeldcore.reset_gemm_telemetry()
        observed, repaired = gxeldcore.protected_matmul_nn(left, right, 1)
        np.testing.assert_array_equal(observed, left @ right)
        assert repaired == 0
        evidence = dict(
            gxeldcore.consume_native_gemm_output_numa_evidence()[0]
        )
        logical_bytes = rows * columns * np.dtype(np.float64).itemsize
        page_count = (logical_bytes - 1) // page_size + 1
        assert evidence["schema"] == "summit.native_gemm_output_numa.v1"
        assert evidence["applicable"] is True
        assert evidence["contract_required"] is True
        assert evidence["complete"] is True
        assert evidence["logical_rows"] == rows
        assert evidence["logical_columns"] == columns
        assert evidence["storage_layout"] == "column_major"
        assert evidence["logical_byte_count"] == logical_bytes
        assert evidence["mapping_bytes"] == page_count * page_size
        assert evidence["mapping_page_count"] == page_count
        assert list(evidence["selected_nodes"]) == [node]
        assert evidence["allocation_mode"] == "mmap_private_anonymous"
        assert evidence["anonymous_private_mapping"] is True
        assert evidence["page_aligned_mapping"] is True
        assert evidence["writable_output"] is True
        assert evidence["bound_before_first_touch"] is True
        assert evidence["pre_touch_live_owner_policy_verified"] is True
        assert evidence["pre_touch_range_policy_verified"] is True
        assert evidence["post_repair_live_owner_policy_verified"] is True
        assert evidence["post_repair_range_policy_verified"] is True
        assert evidence["post_repair_complete_page_query"] is True
        assert evidence["queried_pages"] == page_count
        assert evidence["resolved_pages"] == page_count
        assert dict(evidence["node_histogram"]) == {str(node): page_count}
        assert evidence["ordered_status_sha256"] == hashlib.sha256(
            b"".join(struct.pack("<i", node) for _ in range(page_count))
        ).hexdigest()
        assert evidence["post_repair_strict_policy_verified"] is True
        assert evidence["sealed_read_only"] is False
        status = dict(gxeldcore.native_gemm_output_numa_evidence_status())
        assert status["attempted_calls"] == status["verified_calls"] == 1
        assert status["legacy_calls"] == status["failed_calls"] == 0

        # Exercise the target path independently: one immutable packed pair,
        # one combined protected output allocation, and two retained views.
        gxeldcore.reset_gemm_telemetry()
        target_k = 17
        target_rows = page_size // 16 + 3
        target_columns = 5
        target_left = np.asfortranarray(
            (
                np.arange(target_k * target_rows, dtype=np.float64)
                .reshape(target_k, target_rows, order="F")
                % 7
            ) - 3.0
        )
        target_right = np.asfortranarray(
            (
                np.arange(target_k * target_columns, dtype=np.float64)
                .reshape(target_k, target_columns, order="F")
                % 5
            ) - 2.0
        )
        row_weights = np.asfortranarray(
            np.full((target_k, 1), 2.0, dtype=np.float64)
        )
        right_pair = gxeldcore.prepare_protected_row_weighted_pair(
            target_right, row_weights, 1
        )
        combined, target_repaired = gxeldcore.protected_matmul_tn_pair(
            target_left, right_pair, 1
        )
        expected_first = target_left.T @ target_right
        expected_combined = np.column_stack(
            (expected_first, 2.0 * expected_first)
        )
        assert combined.shape == (target_rows, 2 * target_columns)
        assert combined.flags.f_contiguous
        assert combined.flags.writeable
        assert target_repaired == 0
        np.testing.assert_array_equal(combined, expected_combined)

        target_evidence_records = list(
            gxeldcore.consume_native_gemm_output_numa_evidence()
        )
        assert len(target_evidence_records) == 1
        target_evidence = dict(target_evidence_records[0])
        target_logical_bytes = (
            target_rows * 2 * target_columns * np.dtype(np.float64).itemsize
        )
        target_page_count = (
            (target_logical_bytes - 1) // page_size + 1
        )
        assert target_evidence["schema"] == (
            "summit.native_gemm_output_numa.v1"
        )
        assert target_evidence["applicable"] is True
        assert target_evidence["contract_required"] is True
        assert target_evidence["complete"] is True
        assert target_evidence["logical_rows"] == target_rows
        assert target_evidence["logical_columns"] == 2 * target_columns
        assert target_evidence["storage_layout"] == "column_major"
        assert target_evidence["logical_byte_count"] == target_logical_bytes
        assert target_evidence["mapping_bytes"] == target_page_count * page_size
        assert target_evidence["mapping_page_count"] == target_page_count
        assert list(target_evidence["selected_nodes"]) == [node]
        assert target_evidence["writable_output"] is True
        assert target_evidence["post_repair_complete_page_query"] is True
        assert target_evidence["queried_pages"] == target_page_count
        assert target_evidence["resolved_pages"] == target_page_count
        assert dict(target_evidence["node_histogram"]) == {
            str(node): target_page_count
        }
        assert target_evidence["ordered_status_sha256"] == hashlib.sha256(
            b"".join(
                struct.pack("<i", node) for _ in range(target_page_count)
            )
        ).hexdigest()
        assert target_evidence["post_repair_strict_policy_verified"] is True
        assert target_evidence["sealed_read_only"] is False

        target_vendor_records = [
            dict(item) for item in gxeldcore.consume_gemm_telemetry()
        ]
        assert len(target_vendor_records) in (0, 1)
        if target_vendor_records:
            target_vendor = target_vendor_records[0]
            assert target_vendor["operation"] == "dgemm_tn"
            assert target_vendor["layout"] == "column_major"
            assert target_vendor["m"] == target_rows
            assert target_vendor["n"] == 2 * target_columns
            assert target_vendor["k"] == target_k
            assert dict(target_vendor["native_gemm_output_numa"]) == (
                target_evidence
            )

        first_view = combined[:, :target_columns]
        second_view = combined[:, target_columns:]
        assert first_view.flags.writeable
        assert second_view.flags.writeable
        del combined
        gc.collect()
        np.testing.assert_array_equal(first_view, expected_first)
        np.testing.assert_array_equal(second_view, 2.0 * expected_first)
        first_view[0, 0] += 1.0
        second_view[0, 0] -= 1.0
        assert first_view[0, 0] == expected_first[0, 0] + 1.0
        assert second_view[0, 0] == 2.0 * expected_first[0, 0] - 1.0

        target_status = dict(
            gxeldcore.native_gemm_output_numa_evidence_status()
        )
        assert target_status["attempted_calls"] == 1
        assert target_status["verified_calls"] == 1
        assert target_status["legacy_calls"] == 0
        assert target_status["failed_calls"] == 0
        assert target_status["captured_records"] == 1
        assert target_status["buffered_records"] == 0
        assert target_status["dropped_records"] == 0
        print("NATIVE_GEMM_OUTPUT_NUMA_PASS")
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
                "BLIS_NUM_THREADS": "1",
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
    assert completed.stdout.strip() == "NATIVE_GEMM_OUTPUT_NUMA_PASS"
