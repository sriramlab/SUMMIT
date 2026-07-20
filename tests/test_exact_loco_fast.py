from __future__ import annotations

import unittest

import numpy as np

from summit import utils


def _scalar_symmetrize(trace, delete_matrix, unit_sizes, *, center="mean"):
    trace = np.asarray(trace, dtype=np.float64)
    B = trace.shape[0] - 1
    K = trace.shape[1]
    out = trace.copy()
    for k in range(K):
        for l in range(k + 1, K):
            v1, v2, c12 = utils._delete_sets_pair_moments(
                x_rep=trace[:B, k, l],
                y_rep=trace[:B, l, k],
                x_full=float(trace[B, k, l]),
                y_full=float(trace[B, l, k]),
                D=delete_matrix,
                unit_sizes=unit_sizes,
                center=center,
                nan_policy="omit",
            )
            weight = (
                utils._sym_clip_weight(v1, v2, c12)
                if np.isfinite(v1) and np.isfinite(v2) and np.isfinite(c12)
                else 0.5
            )
            value = weight * trace[:, k, l] + (1.0 - weight) * trace[:, l, k]
            out[:, k, l] = value
            out[:, l, k] = value
    return out


class ExactLocoFastTest(unittest.TestCase):
    def test_fast_path_is_bitwise_equal_to_scalar_path(self):
        rng = np.random.default_rng(82731)
        for K in (2, 7, 60):
            U = 10
            delete_matrix = np.eye(U)[rng.permutation(U)]
            unit_sizes = rng.integers(100, 1000, U).astype(np.float64)
            trace = rng.normal(size=(U + 1, K, K))
            for center in ("mean", "full", "median"):
                expected = _scalar_symmetrize(
                    trace, delete_matrix, unit_sizes, center=center
                )
                observed = utils.symmetrize_trace_with_jackknife(
                    trace,
                    jk_delete_matrix=delete_matrix,
                    jk_unit_sizes=unit_sizes,
                    center=center,
                    exact_loco_fast=True,
                )
                np.testing.assert_array_equal(observed, expected)

    def test_nonfinite_input_uses_exact_scalar_fallback(self):
        rng = np.random.default_rng(6612)
        trace = rng.normal(size=(6, 5, 5))
        trace[1, 0, 2] = np.nan
        delete_matrix = np.eye(5)
        unit_sizes = np.arange(1.0, 6.0)

        expected = _scalar_symmetrize(trace, delete_matrix, unit_sizes)
        observed = utils.symmetrize_trace_with_jackknife(
            trace,
            jk_delete_matrix=delete_matrix,
            jk_unit_sizes=unit_sizes,
            exact_loco_fast=True,
        )
        np.testing.assert_array_equal(observed, expected)

    def test_default_path_remains_scalar(self):
        rng = np.random.default_rng(1024)
        trace = rng.normal(size=(5, 4, 4))
        delete_matrix = np.eye(4)
        unit_sizes = np.arange(10.0, 50.0, 10.0)

        expected = _scalar_symmetrize(trace, delete_matrix, unit_sizes)
        observed = utils.symmetrize_trace_with_jackknife(
            trace,
            jk_delete_matrix=delete_matrix,
            jk_unit_sizes=unit_sizes,
        )
        np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
