from __future__ import annotations

import os

import numpy as np
import pytest

from summit import gxeldcore


def _configured_native_threads() -> int:
    desired = min(2, len(os.sched_getaffinity(0)))
    try:
        return int(gxeldcore.configure_blas_threads(desired))
    except RuntimeError as exc:
        assert "different thread count" in str(exc)
        configured = int(gxeldcore.build_info()["blas_runtime_threads"])
        assert gxeldcore.configure_blas_threads(configured) == configured
        return configured


def test_nn_integrity_diagnostic_below_threshold_is_inert() -> None:
    info = dict(gxeldcore.build_info())
    if not (
        info["gemm_integrity_enabled"] and info["gemm_checksum_enabled"]
    ):
        pytest.skip(
            "private NN diagnostic requires integrity and checksum guards"
        )

    diagnostic_api = getattr(
        gxeldcore, "_test_protected_matmul_nn_integrity_diagnostic"
    )
    threads = _configured_native_threads()
    rng = np.random.default_rng(981731)
    left = np.asfortranarray(rng.normal(size=(17, 23)))
    right = np.asfortranarray(rng.normal(size=(23, 11)))
    left_before = left.copy(order="F")
    right_before = right.copy(order="F")

    observed, repaired, raw_diagnostic = diagnostic_api(left, right, threads)
    diagnostic = dict(raw_diagnostic)

    np.testing.assert_allclose(
        observed, left @ right, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)
    assert np.asarray(observed).flags.f_contiguous
    assert repaired == 0
    assert diagnostic["schema_version"] == 1
    assert diagnostic["integrity_check_eligible"] is False
    assert diagnostic["diagnostic_executed"] is False
    assert diagnostic["classification"] == "not_executed"
    capabilities = dict(diagnostic["floating_point_capabilities"])
    long_double = dict(capabilities["long_double"])
    binary64 = dict(capabilities["double"])
    assert long_double["sizeof_bytes"] >= binary64["sizeof_bytes"]
    assert long_double["digits"] > binary64["digits"]
    assert 0.0 < long_double["epsilon"] < binary64["epsilon"]
    assert diagnostic["flagged_column_count"] == 0
    assert diagnostic["captured_flagged_column_count"] == 0
    assert diagnostic["dropped_flagged_column_count"] == 0
    assert diagnostic["flagged_check_count"] == 0
    assert diagnostic["captured_flagged_check_count"] == 0
    assert diagnostic["dropped_flagged_check_count"] == 0
    assert diagnostic["first_flagged_column"] is None
    assert diagnostic["first_flagged_check"] is None
    assert list(diagnostic["flagged_columns"]) == []
    assert list(diagnostic["captured_flagged_columns"]) == []
    assert list(diagnostic["flagged_check_ids"]) == []
    assert list(diagnostic["flagged_checks"]) == []
    assert list(diagnostic["column_comparisons"]) == []
    assert np.asarray(diagnostic["raw_vendor_columns"]).shape == (17, 0)
    assert np.asarray(diagnostic["deterministic_tiled_columns"]).shape == (17, 0)
    assert np.asarray(diagnostic["long_double_reference_columns"]).shape == (
        17,
        0,
    )
    assert list(diagnostic["long_double_reference_decimal_columns"]) == []
    assert dict(diagnostic["fingerprints"]) == {
        "original_b": {},
        "protected_b_snapshot": {},
    }
    assert dict(diagnostic["fingerprint_equalities"]) == {}
    assert dict(diagnostic["fault_injection"]) == {
        "enabled": False,
        "row": None,
        "column": None,
        "delta": None,
        "timing": None,
    }
    assert diagnostic["classification_changes_production_decision"] is False


def test_nn_integrity_diagnostic_retains_injected_vendor_failure() -> None:
    info = dict(gxeldcore.build_info())
    if not (
        info["gemm_integrity_enabled"] and info["gemm_checksum_enabled"]
    ):
        pytest.skip(
            "private NN diagnostic requires integrity and checksum guards"
        )

    diagnostic_api = getattr(
        gxeldcore, "_test_protected_matmul_nn_integrity_diagnostic"
    )
    threads = _configured_native_threads()
    rng = np.random.default_rng(70131)
    m, n, k = 512, 512, 2048
    injected_row = 17
    injected_column = 29
    injected_delta = 1.0
    left = np.asfortranarray(rng.normal(size=(m, k)))
    right = np.asfortranarray(rng.normal(size=(k, n)))
    left_before = left.copy(order="F")
    right_before = right.copy(order="F")

    observed, repaired, raw_diagnostic = diagnostic_api(
        left,
        right,
        threads,
        fault_injection_row=injected_row,
        fault_injection_column=injected_column,
        fault_injection_delta=injected_delta,
    )
    diagnostic = dict(raw_diagnostic)

    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)
    assert repaired == 1
    assert diagnostic["integrity_check_eligible"] is True
    assert diagnostic["diagnostic_executed"] is True
    assert diagnostic["classification"] == (
        "vendor_result_outside_forward_error_bound"
    )
    assert diagnostic["flagged_column_count"] == 1
    assert diagnostic["captured_flagged_column_count"] == 1
    assert diagnostic["dropped_flagged_column_count"] == 0
    assert list(diagnostic["flagged_columns"]) == [injected_column]
    assert list(diagnostic["captured_flagged_columns"]) == [injected_column]
    assert diagnostic["first_flagged_column"] == injected_column
    assert diagnostic["flagged_check_count"] >= 1
    assert {
        int(identifier["column"])
        for identifier in diagnostic["flagged_check_ids"]
    } == {injected_column}

    raw_vendor = np.asarray(diagnostic["raw_vendor_columns"])
    tiled = np.asarray(diagnostic["deterministic_tiled_columns"])
    long_double_reference = np.asarray(
        diagnostic["long_double_reference_columns"]
    )
    assert raw_vendor.shape == tiled.shape == long_double_reference.shape == (
        m,
        1,
    )
    assert abs(raw_vendor[injected_row, 0] - tiled[injected_row, 0]) > 0.9
    np.testing.assert_array_equal(
        np.asarray(observed)[:, injected_column], tiled[:, 0]
    )
    comparison = dict(diagnostic["column_comparisons"][0])
    assert comparison["column"] == injected_column
    assert comparison["classification"] == (
        "vendor_result_outside_forward_error_bound"
    )
    assert comparison["vendor_rows_outside_forward_error_bound"] >= 1
    assert comparison["tiled_rows_outside_forward_error_bound"] == 0
    assert comparison["max_abs_vendor_minus_tiled"] > 0.9
    assert len(diagnostic["long_double_reference_decimal_columns"][0]) == m
    assert dict(diagnostic["fault_injection"]) == {
        "enabled": True,
        "row": injected_row,
        "column": injected_column,
        "delta": injected_delta,
        "timing": "after_vendor_call_before_observed_checksum",
    }
    assert all(dict(diagnostic["fingerprint_equalities"]).values())
