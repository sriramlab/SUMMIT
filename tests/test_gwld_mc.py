from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from summit import gwldcore
from summit.cli import _dispatch_ldscore, build_parser
from summit.ldscore.gw_ldscore import (
    _column_rms_chunked,
    _integrated_mc_variance_from_probe_sums,
    _mc_variance_from_probe_sums,
)


def test_mc_sufficient_statistics_match_explicit_probe_sample_variance():
    y = np.asarray(
        [
            [[1.0, 2.0], [2.0, 0.5], [4.0, 1.5], [3.0, 2.5]],
            [[0.5, 3.0], [1.5, 1.0], [2.5, 2.0], [4.5, 4.0]],
            [[2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [2.0, 4.0]],
        ],
        dtype=np.float64,
    )
    # Axes are SNP, probe, annotation.
    first = y.sum(axis=1)
    second = np.square(y).sum(axis=1)
    expected = y.var(axis=1, ddof=1) / y.shape[1]
    observed = _mc_variance_from_probe_sums(first, second, y.shape[1])
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=2e-15)

    aggregate_second = second.sum(axis=0)
    integrated = _integrated_mc_variance_from_probe_sums(
        first, aggregate_second, y.shape[1]
    )
    np.testing.assert_allclose(
        integrated, expected.sum(axis=0), rtol=2e-15, atol=2e-15
    )


def test_dense_native_phase2_mc_accumulators_match_exact_matrix_oracle():
    geno = np.asfortranarray(
        np.asarray(
            [
                [0.2, -0.4, 0.8, 0.1],
                [1.1, 0.3, -0.2, 0.5],
                [-0.7, 0.9, 0.4, -0.6],
                [0.6, -1.2, 0.7, 0.3],
                [0.1, 0.8, -0.5, 1.0],
            ],
            dtype=np.float64,
        )
    )
    bins = 2
    probes = 3
    xz = np.asfortranarray(
        np.asarray(
            [
                [0.3, -0.1, 0.7, 0.2, 0.5, -0.4],
                [1.0, 0.6, -0.2, 0.8, -0.3, 0.9],
                [-0.5, 0.4, 0.1, -0.7, 0.2, 0.3],
                [0.9, -0.8, 0.3, 0.4, 0.7, -0.1],
                [0.2, 0.5, -0.6, 1.1, -0.9, 0.6],
            ],
            dtype=np.float64,
        )
    )
    inv = np.asarray([1.0, 0.75, 1.25, 0.6], dtype=np.float64)
    n_denom = 6
    denom = float(n_denom - 1)

    work = geno.T @ xz
    y = np.empty((geno.shape[1], probes, bins), dtype=np.float64)
    for k in range(bins):
        wk = work[:, k * probes:(k + 1) * probes]
        y[:, :, k] = np.square(wk) * np.square(inv[:, None]) / (denom * denom)

    point = np.zeros((geno.shape[1], bins), dtype=np.float64)
    aggregate_second = np.zeros(bins, dtype=np.float64)
    per_snp_second = np.zeros_like(point)
    gwldcore.phase2_accum_XtXz_geno(
        Geno=geno.copy(order="F"),
        blk_start=0,
        inv_left=inv,
        tile_nvecs=probes,
        Xz2d=xz,
        meansq_accum=point,
        mc_aggregate_sumsq=aggregate_second,
        mc_per_snp_sumsq=per_snp_second,
        N_denom=n_denom,
    )

    np.testing.assert_allclose(point, y.sum(axis=1), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        per_snp_second, np.square(y).sum(axis=1), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        aggregate_second, np.square(y).sum(axis=(0, 1)), rtol=2e-14, atol=2e-14
    )

    expected_var = y.var(axis=1, ddof=1) / probes
    np.testing.assert_allclose(
        _mc_variance_from_probe_sums(point, per_snp_second, probes),
        expected_var,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        _integrated_mc_variance_from_probe_sums(
            point, aggregate_second, probes
        ),
        expected_var.sum(axis=0),
        rtol=2e-14,
        atol=2e-14,
    )

    point_without_mc = np.zeros_like(point)
    gwldcore.phase2_accum_XtXz_geno(
        Geno=geno.copy(order="F"),
        blk_start=0,
        inv_left=inv,
        tile_nvecs=probes,
        Xz2d=xz,
        meansq_accum=point_without_mc,
        N_denom=n_denom,
    )
    np.testing.assert_array_equal(point_without_mc, point)


def test_chunked_mc_reductions_and_rms_match_explicit_float64_oracle():
    rng = np.random.default_rng(82)
    y = rng.lognormal(mean=-0.3, sigma=0.7, size=(23, 17, 4))
    first = y.sum(axis=1).astype(np.float32)
    second = np.square(y).sum(axis=1, dtype=np.float64)
    per_snp = _mc_variance_from_probe_sums(
        first, second, y.shape[1], chunk_rows=5
    )
    integrated = _integrated_mc_variance_from_probe_sums(
        first, second.sum(axis=0), y.shape[1], chunk_rows=6
    )
    expected = (
        second - np.square(first.astype(np.float64)) / float(y.shape[1])
    ) / float(y.shape[1] * (y.shape[1] - 1))
    np.testing.assert_allclose(per_snp, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(integrated, expected.sum(axis=0), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        _column_rms_chunked(first, chunk_rows=4),
        np.sqrt(np.mean(np.square(first.astype(np.float64)), axis=0)),
        rtol=2e-15,
        atol=2e-15,
    )


def test_mc_cancellation_is_reported_instead_of_material_negative_clipping():
    first = np.asarray([[10.0], [10.0]], dtype=np.float64)
    # The first row is within the roundoff tolerance; the second is materially
    # inconsistent with a sum of squares and must become NaN, not zero.
    second = np.asarray([[50.0 - 1e-14], [40.0]], dtype=np.float64)
    observed, info = _mc_variance_from_probe_sums(
        first, second, 2, return_info=True
    )
    assert observed[0, 0] == 0.0
    assert np.isnan(observed[1, 0])
    assert info["n_roundoff_clipped"][0] == 1
    assert info["n_numerical_failures"][0] == 1


def test_mc_per_snp_output_reuses_second_moment_storage_and_counts_nonfinite():
    first = np.asarray([[3.0], [np.nan]], dtype=np.float64)
    second = np.asarray([[5.0], [7.0]], dtype=np.float64)
    observed, info = _mc_variance_from_probe_sums(
        first, second, 2, return_info=True, out=second
    )
    assert observed is second
    np.testing.assert_allclose(observed[0, 0], 0.25)
    assert np.isnan(observed[1, 0])
    assert info["n_numerical_failures"][0] == 1

    integrated, aggregate_info = _integrated_mc_variance_from_probe_sums(
        first, np.asarray([12.0]), 2, return_info=True
    )
    assert np.isnan(integrated[0])
    assert aggregate_info["n_numerical_failures"][0] == 1


def test_native_phase2_legacy_positional_argument_order_is_preserved():
    geno = np.asfortranarray(np.asarray([[1.0], [-1.0], [0.5]], dtype=np.float64))
    xz = np.asfortranarray(np.asarray([[0.2], [0.3], [-0.4]], dtype=np.float64))
    inv = np.asarray([1.0], dtype=np.float64)
    point = np.zeros((1, 1), dtype=np.float64)
    # This is the pre-MC public positional signature: C, R, N_denom follow the
    # point accumulator. New MC outputs must remain append-only.
    gwldcore.phase2_accum_XtXz_geno(
        geno, 0, inv, 1, xz, point, None, None, 4
    )
    assert np.isfinite(point[0, 0])


def test_mc_cli_flags_are_mutually_exclusive_and_rejected_outside_standard_gwld():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--write-ld-mc-var", "--skip-ld-mc"])

    class Log:
        def _log(self, _message):
            pass

    with pytest.raises(ValueError, match="GxE LD scores do not yet support"):
        _dispatch_ldscore(
            SimpleNamespace(
                env="env.tsv",
                ld_wind_kb=None,
                write_ld_mc_var=True,
                skip_ld_mc=False,
            ),
            Log(),
            False,
            {},
        )
    with pytest.raises(ValueError, match="Windowed LD scores are deterministic"):
        _dispatch_ldscore(
            SimpleNamespace(
                env=None,
                ld_wind_kb=1.0,
                write_ld_mc_var=False,
                skip_ld_mc=True,
            ),
            Log(),
            False,
            {},
        )
