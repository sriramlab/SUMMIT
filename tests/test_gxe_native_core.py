from __future__ import annotations

import os
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed
from scripts.gxe import benchmark_native

from summit import gwldcore, gxeldcore
from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _loaded_native_binary_record,
    _make_seed,
    _native_gemm_integrity_workspace_elements,
    _native_strict_feature_moment_verification_policy,
    _orthonormalize_columns,
    _validate_native_blas_runtime,
)
from summit.ldscore.gxe_score import score_phenotype_from_reference
from summit.logger import Logger


def _write_plink(tmp_path: Path, raw: np.ndarray, name: str) -> Path:
    prefix = tmp_path / name
    to_bed(str(prefix) + ".bed", np.asarray(raw, dtype=np.float64))
    return prefix


def _variant_count(prefix: Path) -> int:
    with open(str(prefix) + ".bim", "rb") as handle:
        return sum(1 for line in handle if line.strip())


def _design(n: int, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8931)
    env = rng.normal(size=n)
    env = (env - env.mean()) / env.std(ddof=ddof)
    cov = rng.normal(size=(n, 2))
    cov = cov - cov.mean(axis=0)
    q = _orthonormalize_columns(np.column_stack([np.ones(n), env, cov]))
    return np.asarray(env, dtype=np.float64), np.asfortranarray(q)


def _open_descriptors(prefix: Path) -> list[int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    return [os.open(str(prefix) + extension, flags) for extension in (".bed", ".bim", ".fam")]


@contextmanager
def _native_context(
    prefix: Path,
    env: np.ndarray,
    q: np.ndarray,
    *,
    row_sel: np.ndarray | None = None,
    ddof: int = 1,
    decode_threads: int = 2,
    max_workspace_bytes: int = 1 << 30,
    target_panel_columns: int = 7,
    strict_feature_moment_verification: bool = True,
):
    descriptors = _open_descriptors(prefix)
    context = None
    try:
        context = gxeldcore.DirectContext(
            bed_descriptor=descriptors[0],
            bim_descriptor=descriptors[1],
            fam_descriptor=descriptors[2],
            row_sel=row_sel,
            ddof=ddof,
            env=np.asarray(env, dtype=np.float64),
            q_basis=np.asfortranarray(q, dtype=np.float64),
            decode_threads=decode_threads,
            max_workspace_bytes=max_workspace_bytes,
            target_panel_columns=target_panel_columns,
            strict_feature_moment_verification=strict_feature_moment_verification,
        )
        yield context
    finally:
        if context is not None:
            context.close()
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _python_features(
    raw: np.ndarray, env: np.ndarray, q: np.ndarray, ddof: int = 1
) -> dict[str, np.ndarray]:
    filled = np.array(raw, copy=True, dtype=np.float64)
    means = np.nanmean(filled, axis=0)
    rows, cols = np.where(np.isnan(filled))
    filled[rows, cols] = means[cols]
    filled -= means
    filled /= filled.std(axis=0, ddof=ddof)
    p = np.eye(filled.shape[0]) - q @ q.T
    rank = filled.shape[0] - q.shape[1]
    x = p @ filled
    w = p @ (env[:, None] * filled)
    scale_x = np.sqrt(rank / np.sum(x * x, axis=0))
    scale_w = np.sqrt(rank / np.sum(w * w, axis=0))
    x *= scale_x
    w *= scale_w
    return {
        "g": filled,
        "x": x,
        "w": w,
        "scale_x": scale_x,
        "scale_w": scale_w,
        "norm_x": np.sum(x * x, axis=0) / rank,
        "norm_w": np.sum(w * w, axis=0) / rank,
        "diag_nxe_x": np.sum((env[:, None] * x) ** 2, axis=0) / rank,
        "diag_nxe_w": np.sum((env[:, None] * w) ** 2, axis=0) / rank,
        "corr_xw": np.sum(x * w, axis=0) / rank,
    }


def test_native_blas_safety_mode_and_fixed_configuration():
    info = dict(gxeldcore.build_info())
    assert info["multi_environment_native_kernel_supported"] is True
    assert (
        info["multi_environment_native_kernel_schema"]
        == "summit.multi_environment_native_kernel.v1"
    )
    assert (
        info["multi_environment_native_kernel_execution"]
        == "feature_source_projection_target_reduction_normalization"
    )
    isolation = info.get("blas_runtime_isolation")
    integrity = info.get("gemm_integrity_enabled")
    checksum = info.get("gemm_checksum_enabled")
    assert type(checksum) is bool
    assert isolation == "private_static" or integrity is True
    if isolation == "private_static":
        assert info["api_version"] >= 4
        private_backend = info.get("private_blas_backend", "openblas")
        archive_sha256 = info.get(
            "private_blas_archive_sha256",
            info.get("private_openblas_archive_sha256"),
        )
        assert re.fullmatch(r"[0-9a-f]{64}", str(archive_sha256))
        assert info["blas_runtime_threading_layer"] in {"pthreads", "openmp"}
        assert info["gemm_execution_mode"] == {
            "openblas": "serialized_fixed_private_openblas",
            "upstream_blis": "serialized_fixed_private_blis",
        }[private_backend]
        if private_backend == "upstream_blis":
            assert integrity is True
            # Production private BLIS defaults to the guarded, no-redundant-
            # checksum path. An explicit diagnostic build may opt back in.
            assert checksum is False
            assert info["blas_runtime_threading_layer"] == "pthreads"
            assert (
                info["blas_runtime_worker_affinity_policy"]
                == "inherit_authenticated_selected_cpu_set_per_call"
            )
            assert info["blas_runtime_tls_enabled"] is True
        else:
            assert checksum is True
    else:
        # The process-shared OpenBLAS compatibility build retains checksum
        # detection because it does not provide private immutable ownership.
        assert checksum is True
        assert gxeldcore.configure_blas_threads(2) == 2
        with pytest.raises(RuntimeError, match="differ"):
            gxeldcore.configure_blas_threads(1)


@pytest.mark.parametrize(
    ("root_seed", "block_start", "probe_start", "rows", "probes"),
    (
        (0, 0, 0, 1, 1),
        (20260808, 0, 0, 17, 5),
        (2718, 13, 7, 4099, 3),
    ),
)
def test_native_philox_rademacher_matches_numpy_exactly(
    root_seed, block_start, probe_start, rows, probes
):
    keys = np.empty((probes, 2), dtype=np.uint64)
    expected = np.empty((rows, probes), dtype=np.float64, order="F")
    for local_probe in range(probes):
        seed = _make_seed(root_seed, block_start, probe_start + local_probe)
        bit_generator = np.random.Philox(seed)
        keys[local_probe] = np.asarray(
            bit_generator.state["state"]["key"], dtype=np.uint64
        )
        generator = np.random.Generator(np.random.Philox(seed))
        values = generator.integers(0, 2, size=rows, dtype=np.int8)
        expected[:, local_probe] = 2.0 * values.astype(np.float64) - 1.0

    observed = np.asarray(
        gxeldcore.numpy_philox_rademacher_block(keys, rows, 2),
        dtype=np.float64,
    )
    assert observed.flags.f_contiguous
    np.testing.assert_array_equal(observed, expected)


def test_protected_shared_gemm_entry_points_match_numpy():
    rng = np.random.default_rng(12817)
    left_nn = np.asfortranarray(rng.normal(size=(37, 11)))
    right_nn = np.asfortranarray(rng.normal(size=(11, 7)))
    observed_nn, repaired_nn = gxeldcore.protected_matmul_nn(
        left_nn, right_nn, 2
    )
    np.testing.assert_allclose(observed_nn, left_nn @ right_nn, rtol=2e-14, atol=2e-14)
    assert np.asarray(observed_nn).flags.f_contiguous
    assert repaired_nn == 0

    left_tn = np.asfortranarray(rng.normal(size=(41, 13)))
    right_tn = np.asfortranarray(rng.normal(size=(41, 5)))
    observed_tn, repaired_tn = gxeldcore.protected_matmul_tn(
        left_tn, right_tn, 2
    )
    np.testing.assert_allclose(observed_tn, left_tn.T @ right_tn, rtol=2e-14, atol=2e-14)
    assert np.asarray(observed_tn).flags.f_contiguous
    assert repaired_tn == 0

    basis = np.asfortranarray(rng.normal(size=(37, 3)))
    coefficients = np.asfortranarray(rng.normal(size=(3, 11)))
    target = np.asfortranarray(rng.normal(size=(37, 11)))
    expected_update = target - basis @ coefficients
    gxeldcore.protected_rank_update_nn(basis, coefficients, target, 2)
    np.testing.assert_allclose(target, expected_update, rtol=2e-14, atol=2e-14)


def test_fused_feature_scalar_moments_match_dense_reductions():
    rng = np.random.default_rng(71821)
    genotype = np.asfortranarray(rng.normal(size=(503, 37)))
    environments = np.asfortranarray(rng.normal(size=(503, 5)))
    observed = np.asarray(
        gxeldcore.fused_feature_scalar_moments(genotype, environments, 2)
    )
    expected = np.empty_like(observed)
    genotype_squared = genotype * genotype
    expected[0] = np.sum(genotype_squared, axis=0, dtype=np.float64)
    for index in range(environments.shape[1]):
        environment = environments[:, index]
        environment_squared = environment * environment
        expected[1 + 3 * index] = np.sum(
            environment[:, None] * genotype_squared, axis=0, dtype=np.float64
        )
        expected[2 + 3 * index] = np.sum(
            environment_squared[:, None] * genotype_squared,
            axis=0,
            dtype=np.float64,
        )
        expected[3 + 3 * index] = np.sum(
            (environment_squared * environment_squared)[:, None]
            * genotype_squared,
            axis=0,
            dtype=np.float64,
        )
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=4e-12)
    assert observed.flags.f_contiguous


@pytest.mark.parametrize("hwe_scale", [False, True])
def test_parallel_genotype_standardization_matches_python(hwe_scale):
    rng = np.random.default_rng(18271)
    raw = np.asfortranarray(rng.integers(0, 3, size=(503, 37)).astype(np.float64))
    raw[3, 2] = np.nan
    raw[17:22, 11] = np.nan
    raw[:, 19] = 1.0
    raw[:, 31] = np.nan
    targets = rng.normal(size=(503, 2))
    targets -= targets.mean(axis=0)
    targets = np.asfortranarray(targets)

    expected = raw.copy(order="F")
    mask = np.isnan(expected)
    counts = mask.sum(axis=0, dtype=np.int64)
    nobs = expected.shape[0] - counts
    means = np.divide(
        np.nansum(expected, axis=0, dtype=np.float64),
        nobs,
        out=np.zeros(expected.shape[1], dtype=np.float64),
        where=nobs > 0,
    )
    rows, columns = np.where(mask)
    expected[rows, columns] = means[columns]
    expected -= means
    if hwe_scale:
        scales = np.sqrt(np.maximum(means * (1.0 - 0.5 * means), 0.0))
    else:
        scales = np.sqrt(np.einsum("ij,ij->j", expected, expected) / 502.0)
    good = np.isfinite(scales) & (scales > 1.0e-10)
    expected[:, good] /= scales[good]
    expected[:, ~good] = 0.0
    expected_correlations = np.zeros((2, expected.shape[1]), dtype=np.float64)
    valid = (counts > 0) & (counts < expected.shape[0])
    for target_index in range(targets.shape[1]):
        expected_correlations[target_index, valid] = (
            np.asarray(mask[:, valid], dtype=np.float64).T @ targets[:, target_index]
        ) / (
            np.sqrt(counts[valid] * (1.0 - counts[valid] / expected.shape[0]))
            * np.linalg.norm(targets[:, target_index])
        )

    observed = raw.copy(order="F")
    observed_counts, observed_correlations = gxeldcore.standardize_genotype_block(
        observed, targets, 1, hwe_scale, 1.0e-10, 2
    )
    np.testing.assert_array_equal(observed_counts, counts)
    np.testing.assert_allclose(observed_correlations, expected_correlations, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)
    assert observed.flags.f_contiguous


def test_protected_row_weighted_pair_is_sealed_and_matches_numpy():
    rng = np.random.default_rng(98127)
    left = np.asfortranarray(rng.normal(size=(43, 9)))
    right = np.asfortranarray(rng.normal(size=(43, 12)))
    row_weights = np.asfortranarray(rng.normal(size=(43, 3)))
    expected_right = right.copy(order="F")
    expected_weights = row_weights.copy(order="F")
    pair = gxeldcore.prepare_protected_row_weighted_pair(
        right, row_weights, 2
    )
    assert pair.rows == 43
    assert pair.columns == 12

    # The opaque native pair owns the only operand used by subsequent calls.
    # Mutating the Python construction arrays cannot alter the sealed panel.
    right.fill(0.0)
    row_weights.fill(0.0)
    observed, repaired = gxeldcore.protected_matmul_tn_pair(left, pair, 2)
    expected_weighted = np.empty_like(expected_right)
    for group in range(3):
        segment = slice(group * 4, (group + 1) * 4)
        expected_weighted[:, segment] = (
            expected_weights[:, group, None] * expected_right[:, segment]
        )
    expected = left.T @ np.column_stack([expected_right, expected_weighted])
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)
    assert np.asarray(observed).flags.f_contiguous
    assert repaired == 0


def test_protected_shared_gemm_rejects_incompatible_shapes():
    with pytest.raises(RuntimeError, match="incompatible dimensions"):
        gxeldcore.protected_matmul_nn(
            np.ones((5, 3), dtype=np.float64, order="F"),
            np.ones((4, 2), dtype=np.float64, order="F"),
            1,
        )
    with pytest.raises(RuntimeError, match="incompatible dimensions"):
        gxeldcore.protected_matmul_tn(
            np.ones((5, 3), dtype=np.float64, order="F"),
            np.ones((4, 2), dtype=np.float64, order="F"),
            1,
        )


def _feature(context, m: int, *, require_missing_free: bool = True):
    return context.feature_block(
        blk_start=0,
        blk_end=m,
        eps_var=1e-10,
        require_missing_free=require_missing_free,
    )


def _ordinary_residual_scales(
    prefix: Path,
    m: int,
    row_sel: np.ndarray | None,
    covariate: np.ndarray,
) -> np.ndarray:
    covariate = np.asfortranarray(covariate.reshape(-1, 1), dtype=np.float64)
    residualizer = np.asfortranarray(np.linalg.pinv(covariate), dtype=np.float64)
    scales, _ = gwldcore.precompute_residual_variances_bed(
        bed_prefix=str(prefix),
        fam_path=str(prefix) + ".fam",
        nsnps=m,
        step_size=m,
        row_sel=row_sel,
        ddof=1,
        eps=1e-10,
        compute_mu22=False,
        annot_all=None,
        C=covariate,
        R=residualizer,
        impute_mode="mean",
        impute_seed=0,
    )
    return np.asarray(scales)


def test_native_feature_moment_verification_defaults_to_protected_gemm(monkeypatch):
    monkeypatch.delenv("SUMMIT_GXE_VERIFY_FEATURE_MOMENTS", raising=False)
    enabled, reason = _native_strict_feature_moment_verification_policy(
        {
            "blas_vendor": "OpenBLAS",
            "blas_runtime_config": "OpenBLAS 0.3.30 DYNAMIC_ARCH Zen",
        }
    )
    assert enabled is False
    assert "deterministic tiled feature GEMMs" in reason
    assert "serialized-entry internally threaded OpenBLAS" in reason

    enabled, reason = _native_strict_feature_moment_verification_policy(
        {
            "blas_vendor": "OpenBLAS",
            "blas_runtime_config": "OpenBLAS 0.3.33 DYNAMIC_ARCH USE_OPENMP Zen",
        }
    )
    assert enabled is False
    assert "deterministic tiled feature GEMMs" in reason

    enabled, reason = _native_strict_feature_moment_verification_policy(
        {
            "blas_vendor": "OpenBLAS",
            "blas_runtime_config": "OpenBLAS 0.3.33 DYNAMIC_ARCH Zen",
        }
    )
    assert enabled is False
    assert "deterministic tiled feature GEMMs" in reason

    enabled, reason = _native_strict_feature_moment_verification_policy(
        {
            "blas_vendor": "Intel10_64_dyn",
            "blas_runtime_config": None,
        }
    )
    assert enabled is False
    assert "deterministic disjoint-output tiled GEMMs" in reason
    assert "Intel10_64_dyn" in reason

    monkeypatch.setenv("SUMMIT_GXE_VERIFY_FEATURE_MOMENTS", "always")
    enabled, reason = _native_strict_feature_moment_verification_policy(
        {
            "blas_vendor": "OpenBLAS",
            "blas_runtime_config": "OpenBLAS 0.3.33 DYNAMIC_ARCH USE_OPENMP Zen",
        }
    )
    assert enabled is True
    assert "forced" in reason


def test_native_integrity_workspace_is_vendor_independent():
    dimensions = (10_000, 64, 1_000)
    openblas = _native_gemm_integrity_workspace_elements(
        {"blas_vendor": "OpenBLAS"}, *dimensions
    )
    mkl = _native_gemm_integrity_workspace_elements(
        {"blas_vendor": "Intel10_64_dyn"}, *dimensions
    )
    assert openblas == mkl
    assert openblas == 8 * (
        dimensions[0] + 2 * dimensions[2] + 2 * dimensions[1]
    ) + dimensions[2] * dimensions[1]
    assert _native_gemm_integrity_workspace_elements(None, *dimensions) == 0


def test_native_blas_runtime_requires_one_matching_library(monkeypatch):
    single = {
        "user_api": "blas",
        "internal_api": "openblas",
        "version": "0.3.34",
        "filepath": "/opt/lib/libopenblas.so",
        "num_threads": 16,
    }
    monkeypatch.setattr(
        "summit.ldscore.gwe_ldscore.threadpool_info", lambda: [single]
    )
    observed = _validate_native_blas_runtime(
        {
            "blas_vendor": "OpenBLAS",
            "blas_runtime_config": "OpenBLAS 0.3.34 DYNAMIC_ARCH Zen",
        }
    )
    assert observed["internal_api"] == "openblas"
    assert observed["version"] == "0.3.34"
    assert observed["num_threads"] == 16

    unsupported = {
        **single,
        "version": "0.3.30",
        "filepath": "/opt/lib/libopenblas-0.3.30.so",
    }
    monkeypatch.setattr(
        "summit.ldscore.gwe_ldscore.threadpool_info", lambda: [unsupported]
    )
    with pytest.raises(RuntimeError, match="0.3.31 or newer"):
        _validate_native_blas_runtime(
            {
                "blas_vendor": "OpenBLAS",
                "blas_runtime_config": "OpenBLAS 0.3.30 DYNAMIC_ARCH Zen",
            }
        )

    second = {
        **single,
        "version": "0.3.30",
        "filepath": "/wheel/numpy.libs/libopenblas.so",
    }
    monkeypatch.setattr(
        "summit.ldscore.gwe_ldscore.threadpool_info", lambda: [single, second]
    )
    with pytest.raises(RuntimeError, match="exactly one process-wide BLAS"):
        _validate_native_blas_runtime({"blas_vendor": "OpenBLAS"})

    monkeypatch.setattr(
        "summit.ldscore.gwe_ldscore.threadpool_info", lambda: [single]
    )
    with pytest.raises(RuntimeError, match="disagree"):
        _validate_native_blas_runtime({"blas_vendor": "Intel10_64_dyn"})

    with pytest.raises(RuntimeError, match="OpenBLAS version disagree"):
        _validate_native_blas_runtime(
            {
                "blas_vendor": "OpenBLAS",
                "blas_runtime_config": "OpenBLAS 0.3.30 DYNAMIC_ARCH Haswell",
            }
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("blas_vendor", "OpenBLAS"),
        ("gemm_integrity_enabled", False),
        ("gemm_vendor_entry_outer_openmp_guard", False),
        ("blas_runtime_owner_thread_configured", False),
        ("blas_runtime_threading_layer", "openmp"),
        ("blas_runtime_worker_affinity_policy", "unrestricted"),
    ],
)
def test_private_blis_runtime_validation_requires_guarded_contract(
    monkeypatch, field, invalid
):
    monkeypatch.setattr(
        "summit.ldscore.gwe_ldscore.threadpool_info", lambda: []
    )
    build_info = {
        "blas_vendor": "BLIS",
        "blas_runtime_isolation": "private_static",
        "gemm_execution_mode": "serialized_fixed_private_blis",
        "private_blas_backend": "upstream_blis",
        "private_blas_archive_sha256": "a" * 64,
        "private_blas_source_commit": "b" * 40,
        "private_blas_source_tree_sha256": "c" * 64,
        "private_blas_config_family": "zen",
        "private_blas_header_sha256": "d" * 64,
        "private_blas_cblas_header_sha256": "e" * 64,
        "blas_runtime_config": "BLIS 2.0 config=zen",
        "blas_runtime_corename": "zen",
        "blas_runtime_threads": 2,
        "blas_runtime_threading_layer": "pthreads",
        "blas_runtime_worker_affinity_policy": (
            "inherit_authenticated_selected_cpu_set_per_call"
        ),
        "blas_runtime_thread_strategy": "automatic",
        "blas_runtime_thread_ways": {
            "jc": 1, "pc": 1, "ic": 1, "jr": 1, "ir": 1,
        },
        "blas_runtime_tls_enabled": True,
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_owner_thread_configured": True,
        "blas_runtime_environment_immutable": True,
        "blas_runtime_environment_contract": "blis_process_start_v1",
        "gemm_integrity_enabled": True,
        "gemm_vendor_entry_outer_openmp_guard": True,
    }
    assert _validate_native_blas_runtime(build_info)["internal_api"] == "blis"
    build_info[field] = invalid
    with pytest.raises(RuntimeError, match="contract|threading layer"):
        _validate_native_blas_runtime(build_info)


def test_native_compute_block_coalescing_preserves_probe_streams():
    blocks = [(0, 5), (5, 10), (10, 15), (15, 18)]
    groups = GenomewideEnvLDScore._coalesce_contiguous_blocks(blocks, 10)
    assert groups == [((0, 5), (5, 10)), ((10, 15), (15, 18))]

    estimator = object.__new__(GenomewideEnvLDScore)
    estimator.dtype = np.float32
    estimator.root_seed = 4129
    estimator.probe_offset = 0
    estimator.rand_dist = "rademacher"
    combined = estimator._generate_random_group(groups[0], v_count=7, v_start=3)
    separate = np.asfortranarray(np.concatenate([
        estimator._generate_random_block(5, 7, 0, 3),
        estimator._generate_random_block(5, 7, 5, 3),
    ]))
    assert combined.flags.f_contiguous
    np.testing.assert_array_equal(combined, separate)


def test_native_feature_blocks_use_available_workspace_beyond_legacy_width():
    estimator = object.__new__(GenomewideEnvLDScore)
    estimator.native_strict_feature_moment_verification = False
    estimator.p_eff = 2
    estimator.nsamp = 1_000
    estimator.nsnps = 10_000
    estimator.step_size = 500
    estimator.native_workspace_gib = 0.1
    estimator._native_build_info = {"blas_vendor": "OpenBLAS"}
    estimator.log = Logger(suppress=True)

    blocks = estimator._make_native_feature_compute_blocks()

    assert estimator.native_feature_step_size > 1_000
    assert blocks[0] == (0, estimator.native_feature_step_size)
    assert blocks[-1][1] == estimator.nsnps


@pytest.mark.parametrize("strict_feature_moment_verification", [False, True])
def test_native_feature_source_target_match_dense_oracle(
    tmp_path, strict_feature_moment_verification
):
    rng = np.random.default_rng(1123)
    n, m, probes = 43, 17, 13
    raw = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "complete")
    env, q = _design(n)
    expected = _python_features(raw, env, q)

    with _native_context(
        prefix,
        env,
        q,
        target_panel_columns=7,
        strict_feature_moment_verification=strict_feature_moment_verification,
    ) as context:
        info = context.info()
        assert (info["n_total"], info["m_total"], info["n_selected"]) == (n, m, n)
        assert info["decode_threads"] == 2
        assert info["target_panel_columns"] == 7
        assert info["projected_target_full_width"] is True
        assert info["strict_feature_moment_verification"] is strict_feature_moment_verification
        assert info["feature_moment_integrity_mode"] == (
            "strict_duplicate"
            if strict_feature_moment_verification
            else "deterministic_disjoint_output_tiled_gemm"
        )
        assert info["repaired_gemm_output_columns"] >= 0
        assert info["retried_gemm_input_mutations"] >= 0
        feature = _feature(context, m)
        assert feature["strict_feature_moment_verification"] is strict_feature_moment_verification
        assert feature["repaired_additive_moment_columns"] >= 0
        for name in (
            "scale_x", "scale_w", "norm_x", "norm_w",
            "diag_nxe_x", "diag_nxe_w", "corr_xw",
        ):
            np.testing.assert_allclose(feature[name], expected[name], rtol=2e-12, atol=2e-12)
        assert feature["missing_genotype_calls"] == 0
        assert feature["max_projection_leakage_additive"] < 2e-14
        assert feature["max_projection_leakage_interaction"] < 2e-14

        z = np.empty((m, probes), dtype=np.float64, order="F")
        for local_probe in range(probes):
            generator = np.random.Generator(np.random.Philox(_make_seed(91, 0, local_probe)))
            z[:, local_probe] = 2.0 * generator.integers(0, 2, size=m, dtype=np.int8) - 1.0
        annotation = np.sqrt(0.2 + np.arange(m, dtype=np.float64) / (m + 2.0))
        group_ids = (np.arange(m) * 3 // m).astype(np.int32)
        source_x, source_w, missing = context.source_block(
            0, m,
            np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
            annotation, z, group_ids, 3, True,
        )
        assert missing == 0
        assert source_w.ctypes.data - source_x.ctypes.data == source_x.nbytes
        for group in range(3):
            keep = group_ids == group
            expected_x = expected["x"][:, keep] @ (annotation[keep, None] * z[keep])
            expected_w = expected["w"][:, keep] @ (annotation[keep, None] * z[keep])
            segment = slice(group * probes, (group + 1) * probes)
            np.testing.assert_allclose(source_x[:, segment], expected_x, rtol=3e-12, atol=3e-12)
            np.testing.assert_allclose(source_w[:, segment], expected_w, rtol=3e-12, atol=3e-12)

        sources = np.asfortranarray(np.column_stack([source_x, source_w]))
        assert context.validate_projected_sources(sources, 1e-10) < 1e-12
        sealed_sources = sources.copy(order="F")
        projected_panel = context.prepare_projected_sources(sources, 1e-10)
        sources[:] = rng.normal(size=sources.shape)
        work_x, work_w, missing, leakage = context.target_projected_block(
            0, m,
            np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
            projected_panel, True,
        )
        assert missing == 0
        assert leakage < 1e-12
        np.testing.assert_allclose(work_x, expected["x"].T @ sealed_sources, rtol=4e-12, atol=4e-12)
        np.testing.assert_allclose(work_w, expected["w"].T @ sealed_sources, rtol=4e-12, atol=4e-12)

        # The generic path explicitly projects an arbitrary source panel.
        arbitrary = np.asfortranarray(rng.normal(size=(n, 5)))
        projected = arbitrary - q @ (q.T @ arbitrary)
        generic_x, generic_w, _, generic_leakage = context.target_block(
            0, m,
            np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
            arbitrary, True,
        )
        assert generic_leakage > 0.0
        np.testing.assert_allclose(generic_x, expected["x"].T @ projected, rtol=4e-12, atol=4e-12)
        np.testing.assert_allclose(generic_w, expected["w"].T @ projected, rtol=4e-12, atol=4e-12)


def test_native_source_forms_interaction_weights_without_scale_ratio_overflow(tmp_path):
    rng = np.random.default_rng(731)
    n, m = 37, 9
    raw = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "source_scale_ratio")
    env, q = _design(n)
    probes = np.asfortranarray(rng.choice([-1.0, 1.0], size=(m, 5)))

    with _native_context(prefix, env, q) as context:
        source_x, source_w, missing = context.source_block(
            0,
            m,
            np.full(m, 1.0e-320, dtype=np.float64),
            np.ones(m, dtype=np.float64),
            np.ones(m, dtype=np.float64),
            probes,
            np.zeros(m, dtype=np.int32),
            1,
            True,
        )

    assert missing == 0
    assert np.isfinite(source_x).all()
    assert np.isfinite(source_w).all()


def test_projected_panel_is_opaque_snapshot_bound_to_one_context(tmp_path):
    rng = np.random.default_rng(919)
    n, m = 31, 8
    raw = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "opaque")
    env, q = _design(n)
    sources = np.asfortranarray(rng.normal(size=(n, 6)))
    sources -= q @ (q.T @ sources)
    with pytest.raises(TypeError):
        gxeldcore.ProjectedPanel()

    with _native_context(prefix, env, q) as first, _native_context(prefix, env, q) as second:
        feature = _feature(first, m)
        panel = first.prepare_projected_sources(sources, 1e-10)
        sources[:] = np.nan
        first.target_projected_block(
            0, m, np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
            panel, True,
        )
        with pytest.raises(RuntimeError, match="does not belong to this context"):
            second.target_projected_block(
                0, m,
                np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
                panel, True,
            )


def test_native_binary_record_binds_the_loaded_inode(tmp_path):
    descriptor, record = _loaded_native_binary_record(gxeldcore)
    try:
        assert len(record["sha256"]) == 64
        assert record["bytes"] > 0
        assert os.fstat(descriptor).st_ino == record["identity"][1]
    finally:
        os.close(descriptor)

    copied = tmp_path / Path(gxeldcore.__file__).name
    copied.write_bytes(Path(gxeldcore.__file__).read_bytes())
    fake = type("FakeNative", (), {"__file__": str(copied)})()
    with pytest.raises(RuntimeError, match="does not identify the inode loaded"):
        _loaded_native_binary_record(fake)


def test_native_mean_imputation_matches_oracle_but_production_gate_rejects(tmp_path):
    rng = np.random.default_rng(741)
    n, m = 37, 11
    raw = rng.binomial(2, rng.uniform(0.2, 0.4, size=m), size=(n, m)).astype(float)
    raw[2, 1] = np.nan
    raw[9, 1] = np.nan
    raw[5, 7] = np.nan
    prefix = _write_plink(tmp_path, raw, "missing")
    env, q = _design(n)
    with _native_context(prefix, env, q) as context:
        with pytest.raises(RuntimeError, match="missing-free"):
            _feature(context, m, require_missing_free=True)
        feature = _feature(context, m, require_missing_free=False)
        expected = _python_features(raw, env, q)
        assert feature["missing_genotype_calls"] == 3
        for name in (
            "scale_x", "scale_w", "norm_x", "norm_w",
            "diag_nxe_x", "diag_nxe_w", "corr_xw",
        ):
            np.testing.assert_allclose(feature[name], expected[name], rtol=3e-12, atol=3e-12)

        probes = np.asfortranarray(rng.choice([-1.0, 1.0], size=(m, 6)))
        source_x, source_w, source_missing = context.source_block(
            0, m,
            np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
            np.ones(m), probes, np.zeros(m, dtype=np.int32), 1, False,
        )
        assert source_missing == 3
        np.testing.assert_allclose(source_x, expected["x"] @ probes, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(source_w, expected["w"] @ probes, rtol=3e-12, atol=3e-12)
        sources = np.asfortranarray(np.column_stack([source_x, source_w]))
        panel = context.prepare_projected_sources(sources, 1e-10)
        work_x, work_w, target_missing, _ = context.target_projected_block(
            0, m,
            np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
            panel, False,
        )
        assert target_missing == 3
        np.testing.assert_allclose(work_x, expected["x"].T @ sources, rtol=4e-12, atol=4e-12)
        np.testing.assert_allclose(work_w, expected["w"].T @ sources, rtol=4e-12, atol=4e-12)

    env0, q0 = _design(n, ddof=0)
    with _native_context(prefix, env0, q0, ddof=0) as context:
        feature_ddof0 = _feature(context, m, require_missing_free=False)
    expected_ddof0 = _python_features(raw, env0, q0, ddof=0)
    for name in (
        "scale_x", "scale_w", "norm_x", "norm_w",
        "diag_nxe_x", "diag_nxe_w", "corr_xw",
    ):
        np.testing.assert_allclose(
            feature_ddof0[name], expected_ddof0[name], rtol=3e-12, atol=3e-12
        )


def test_native_fused_phenotype_score_matches_dense_oracle(tmp_path):
    rng = np.random.default_rng(77291)
    n, m, traits = 113, 47, 3
    raw = rng.binomial(
        2, rng.uniform(0.08, 0.48, size=m), size=(n, m)
    ).astype(float)
    raw[3, 4] = np.nan
    raw[77, 21] = np.nan
    prefix = _write_plink(tmp_path, raw, "phenotype_score")
    env, q = _design(n)
    expected = _python_features(raw, env, q)
    phenotype = rng.normal(size=(n, traits))
    phenotype -= q @ (q.T @ phenotype)
    phenotype = np.asfortranarray(phenotype)
    residual_rank = n - q.shape[1]

    with _native_context(
        prefix,
        env,
        q,
        target_panel_columns=traits,
        strict_feature_moment_verification=False,
    ) as context:
        panel = context.prepare_projected_sources(phenotype, 1.0e-10)
        observed = dict(
            context.phenotype_score_block(0, m, panel, 1.0e-10, False)
        )
        context_info = dict(context.info())

    np.testing.assert_allclose(
        observed["score_x"], expected["x"].T @ phenotype,
        rtol=3e-12, atol=3e-12,
    )
    np.testing.assert_allclose(
        observed["score_w"], expected["w"].T @ phenotype,
        rtol=3e-12, atol=3e-12,
    )
    for name in (
        "scale_x", "scale_w", "norm_x", "norm_w",
        "diag_nxe_x", "diag_nxe_w", "corr_xw",
    ):
        np.testing.assert_allclose(
            observed[name], expected[name], rtol=3e-12, atol=3e-12
        )
    np.testing.assert_allclose(
        observed["diag_nxe_x"],
        (env * env) @ (expected["x"] * expected["x"]) / residual_rank,
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        observed["diag_nxe_w"],
        (env * env) @ (expected["w"] * expected["w"]) / residual_rank,
        rtol=3e-12,
        atol=3e-12,
    )
    assert observed["missing_genotype_calls"] == 2
    assert observed["repaired_feature_moment_columns"] == 0
    assert context_info["repaired_gemm_output_columns"] == 0
    assert context_info["retried_gemm_input_mutations"] == 0


def test_native_context_owns_descriptors_and_isolated_same_path_replacement(tmp_path):
    rng = np.random.default_rng(334)
    n, m = 41, 13
    raw_a = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    raw_b = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw_a, "stable")
    replacement = _write_plink(tmp_path, raw_b, "replacement")
    env, q = _design(n)

    descriptors = _open_descriptors(prefix)
    context = gxeldcore.DirectContext(
        descriptors[0], descriptors[1], descriptors[2], None, 1, env, q, 2, 1 << 30, 7
    )
    for descriptor in descriptors:
        os.close(descriptor)
    try:
        before = _feature(context, m)
        for extension in (".bed", ".bim", ".fam"):
            os.replace(str(replacement) + extension, str(prefix) + extension)
        # The descriptor remains pinned, but unlinking the construction inode
        # changes its ctime; the context must reject rather than silently use a
        # stale same-path mapping.
        with pytest.raises(RuntimeError, match="changed after context construction"):
            _feature(context, m)
        expected_a = _python_features(raw_a, env, q)
        np.testing.assert_allclose(before["scale_w"], expected_a["scale_w"], rtol=3e-12, atol=3e-12)
    finally:
        context.close()

    with _native_context(prefix, env, q) as replacement_context:
        observed_b = _feature(replacement_context, m)
    expected_b = _python_features(raw_b, env, q)
    np.testing.assert_allclose(observed_b["scale_x"], expected_b["scale_x"], rtol=3e-12, atol=3e-12)
    assert not np.array_equal(before["scale_x"], observed_b["scale_x"])


def test_native_context_snapshots_rows_and_rejects_invalid_indices(tmp_path):
    rng = np.random.default_rng(812)
    n_total, m = 62, 12
    raw = rng.binomial(2, rng.uniform(0.12, 0.46, size=m), size=(n_total, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "row_selection")
    row_sel = np.arange(0, n_total, 2, dtype=np.int64)
    env, q = _design(row_sel.size)

    with _native_context(prefix, env, q, row_sel=row_sel) as context:
        observed_even = _feature(context, m)
        expected_even = _python_features(raw[row_sel], env, q)
        row_sel[:] = np.arange(1, n_total, 2, dtype=np.int64)
        observed_after_mutation = _feature(context, m)
        assert np.array_equal(observed_even["scale_x"], observed_after_mutation["scale_x"])
    with _native_context(prefix, env, q, row_sel=row_sel) as context:
        observed_odd = _feature(context, m)
    expected_odd = _python_features(raw[row_sel], env, q)
    assert not np.array_equal(observed_even["scale_x"], observed_odd["scale_x"])
    np.testing.assert_allclose(observed_even["scale_w"], expected_even["scale_w"], rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(observed_odd["scale_w"], expected_odd["scale_w"], rtol=3e-12, atol=3e-12)

    bad_row_sel = np.asarray([0, 1, 2**32], dtype=np.int64)
    bad_env, bad_q = _design(bad_row_sel.size)
    with pytest.raises(RuntimeError, match="out-of-range"):
        with _native_context(prefix, bad_env, bad_q, row_sel=bad_row_sel):
            pass


def test_existing_native_caches_version_paths_and_explicit_row_contents(tmp_path):
    rng = np.random.default_rng(930)
    n, m = 54, 10
    raw_a = rng.binomial(2, rng.uniform(0.12, 0.45, size=m), size=(n, m)).astype(float)
    raw_b = rng.binomial(2, rng.uniform(0.12, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw_a, "ordinary_cache")
    replacement = _write_plink(tmp_path, raw_b, "ordinary_replacement")

    row_sel = np.arange(0, n, 2, dtype=np.int64)
    covariate = rng.normal(size=row_sel.size)
    covariate -= covariate.mean()
    even = _ordinary_residual_scales(prefix, m, row_sel, covariate)
    row_sel[:] = np.arange(1, n, 2, dtype=np.int64)
    mutated_odd = _ordinary_residual_scales(prefix, m, row_sel, covariate)
    fresh_odd = _ordinary_residual_scales(prefix, m, row_sel.copy(), covariate)
    np.testing.assert_array_equal(mutated_odd, fresh_odd)
    assert not np.array_equal(even, mutated_odd)

    full_covariate = rng.normal(size=n)
    full_covariate -= full_covariate.mean()
    gwldcore._clear_bed_mapping_cache()
    before_replacement = _ordinary_residual_scales(
        prefix, m, None, full_covariate
    )
    after_first = dict(gwldcore._bed_mapping_cache_info())
    repeated = _ordinary_residual_scales(prefix, m, None, full_covariate)
    after_second = dict(gwldcore._bed_mapping_cache_info())
    np.testing.assert_array_equal(repeated, before_replacement)
    assert after_first["entries"] == 1
    assert after_first["misses"] == 1
    assert after_second["entries"] == 1
    assert after_second["misses"] == after_first["misses"]
    assert after_second["hits"] > after_first["hits"]
    expected_replacement = _ordinary_residual_scales(
        replacement, m, None, full_covariate
    )
    after_oracle = dict(gwldcore._bed_mapping_cache_info())
    for extension in (".bed", ".bim", ".fam"):
        os.replace(str(replacement) + extension, str(prefix) + extension)
    after_replacement = _ordinary_residual_scales(
        prefix, m, None, full_covariate
    )
    after_third = dict(gwldcore._bed_mapping_cache_info())
    assert after_third["entries"] == after_oracle["entries"] + 1
    assert after_third["misses"] == after_oracle["misses"] + 1
    np.testing.assert_array_equal(after_replacement, expected_replacement)
    assert not np.array_equal(before_replacement, after_replacement)


def test_existing_native_bed_cache_is_bounded_lru(tmp_path):
    rng = np.random.default_rng(211)
    n, m = 24, 4
    covariate = rng.normal(size=n)
    covariate -= covariate.mean()
    gwldcore._clear_bed_mapping_cache()
    capacity = int(gwldcore._bed_mapping_cache_info()["capacity"])
    assert capacity > 1
    first_prefix = None
    for index in range(capacity + 2):
        raw = rng.binomial(2, 0.3, size=(n, m)).astype(float)
        raw[0, :] = 0.0
        raw[1, :] = 2.0
        prefix = _write_plink(tmp_path, raw, f"lru-{index}")
        if first_prefix is None:
            first_prefix = prefix
        _ordinary_residual_scales(prefix, m, None, covariate)
        assert int(gwldcore._bed_mapping_cache_info()["entries"]) <= capacity
    info = dict(gwldcore._bed_mapping_cache_info())
    assert info["entries"] == capacity
    assert info["evictions"] == 2
    misses = info["misses"]
    _ordinary_residual_scales(first_prefix, m, None, covariate)
    assert int(gwldcore._bed_mapping_cache_info()["misses"]) == misses + 1


def test_native_context_rejects_mutation_bad_design_nonfinite_and_workspace(tmp_path):
    rng = np.random.default_rng(118)
    n, m = 35, 9
    raw = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "validation")
    env, q = _design(n)

    with _native_context(prefix, env, q) as context:
        with open(str(prefix) + ".bed", "r+b") as handle:
            handle.seek(3)
            value = handle.read(1)
            handle.seek(3)
            handle.write(bytes([value[0] ^ 0x03]))
            handle.flush()
            os.fsync(handle.fileno())
        observed_state = os.stat(str(prefix) + ".bed")
        os.utime(
            str(prefix) + ".bed",
            ns=(observed_state.st_atime_ns, observed_state.st_mtime_ns + 1_000_000_000),
        )
        with pytest.raises(RuntimeError, match="changed after context construction"):
            _feature(context, m)

    clean_prefix = _write_plink(tmp_path, raw, "validation_clean")
    constant = np.zeros(n, dtype=np.float64)
    with pytest.raises(RuntimeError, match="nonconstant, centered, and standardized"):
        with _native_context(clean_prefix, constant, q):
            pass
    with pytest.raises(RuntimeError, match="nonconstant, centered, and standardized"):
        with _native_context(clean_prefix, 2.0 * env, q):
            pass
    if hasattr(os, "sched_getaffinity"):
        with pytest.raises(RuntimeError, match="CPU-affinity limit"):
            with _native_context(
                clean_prefix,
                env,
                q,
                decode_threads=len(os.sched_getaffinity(0)) + 1,
            ):
                pass
    bad_q = q.copy(order="F")
    bad_q[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        with _native_context(clean_prefix, env, bad_q):
            pass
    with _native_context(clean_prefix, env, q, max_workspace_bytes=64) as context:
        with pytest.raises(RuntimeError, match="workspace bytes"):
            _feature(context, m)

    with _native_context(clean_prefix, env, q) as context:
        feature = _feature(context, m)
        probes = np.ones((m, 3), dtype=np.float64, order="F")
        probes[0, 0] = np.nan
        with pytest.raises(RuntimeError, match="probe contains a non-finite"):
            context.source_block(
                0, m,
                np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
                np.ones(m), probes, np.zeros(m, dtype=np.int32), 1, True,
            )
        sources = np.ones((n, 3), dtype=np.float64, order="F")
        sources[0, 0] = np.inf
        with pytest.raises(RuntimeError, match="non-finite"):
            context.prepare_projected_sources(sources, 1e-10)
    with pytest.raises(RuntimeError, match="closed"):
        context.info()


def test_native_pass_probe_tile_and_fixed_thread_determinism(tmp_path):
    parallel_threads = min(2, len(os.sched_getaffinity(0)))
    if parallel_threads < 2:
        pytest.skip("thread-determinism test requires at least two affinity-visible CPUs")
    rng = np.random.default_rng(662)
    n, m, probes = 53, 19, 16
    raw = rng.binomial(2, rng.uniform(0.12, 0.43, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "determinism")
    env, q = _design(n)
    z = np.asfortranarray(rng.choice([-1.0, 1.0], size=(m, probes)))
    annotation = np.ones(m, dtype=np.float64)
    groups = np.zeros(m, dtype=np.int32)

    def source(start: int, stop: int, probe_slice: slice, threads: int):
        with _native_context(prefix, env, q, decode_threads=threads, target_panel_columns=5) as context:
            feature = _feature(context, m)
            result = context.source_block(
                start, stop,
                np.asarray(feature["scale_x"])[start:stop],
                np.asarray(feature["scale_w"])[start:stop],
                annotation[start:stop],
                np.asfortranarray(z[start:stop, probe_slice]),
                groups[start:stop], 1, True,
            )[:2]
            assert context.info()["decode_threads"] == threads
            return result, feature

    (one_x, one_w), feature = source(0, m, slice(None), parallel_threads)
    (four_x, four_w), _ = source(0, m, slice(None), parallel_threads)
    assert np.array_equal(one_x, four_x)
    assert np.array_equal(one_w, four_w)

    (left_x, left_w), _ = source(0, 8, slice(None), parallel_threads)
    (right_x, right_w), _ = source(8, m, slice(None), parallel_threads)
    np.testing.assert_allclose(left_x + right_x, one_x, rtol=3e-15, atol=3e-13)
    np.testing.assert_allclose(left_w + right_w, one_w, rtol=3e-15, atol=3e-13)

    tile_x, tile_w = [], []
    for start, stop in ((0, 5), (5, 11), (11, probes)):
        (x, w), _ = source(0, m, slice(start, stop), parallel_threads)
        tile_x.append(x)
        tile_w.append(w)
    np.testing.assert_allclose(np.column_stack(tile_x), one_x, rtol=2e-15, atol=2e-14)
    np.testing.assert_allclose(np.column_stack(tile_w), one_w, rtol=2e-15, atol=2e-14)

    sources = np.asfortranarray(np.column_stack([one_x, one_w]))
    target_runs = []
    for _ in range(2):
        with _native_context(
            prefix, env, q, decode_threads=parallel_threads,
            target_panel_columns=5,
        ) as context:
            panel = context.prepare_projected_sources(sources, 1e-10)
            target_runs.append(context.target_projected_block(
                0, m,
                np.asarray(feature["scale_x"]), np.asarray(feature["scale_w"]),
                panel, True,
            )[:2])
    assert np.array_equal(target_runs[0][0], target_runs[1][0])
    assert np.array_equal(target_runs[0][1], target_runs[1][1])


def test_opt_in_native_reference_matches_python_artifacts(tmp_path):
    rng = np.random.default_rng(991)
    n, m = 47, 23
    raw = rng.binomial(2, rng.uniform(0.14, 0.46, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "reference")
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env_path = tmp_path / "reference.env.tsv"
    cov_path = tmp_path / "reference.cov.tsv"
    ids.assign(E=rng.normal(size=n)).to_csv(env_path, sep="\t", index=False)
    ids.assign(C=rng.normal(size=n)).to_csv(cov_path, sep="\t", index=False)

    def estimator(name: str, backend: str) -> GenomewideEnvLDScore:
        return GenomewideEnvLDScore(
            bed_path=str(prefix), env_path=str(env_path), covar_path=str(cov_path),
            annot_path=None, out_path=str(tmp_path / name), log=Logger(suppress=True),
            rand_dist="rademacher", low_level=None, num_vecs=20, step_size=6,
            seed=20260809, dtype="float64", num_threads=2,
            kernel_mode="standardized", genotype_scale="sample", impute_method="mean",
            target_xz_mem=0.01, native_backend=backend,
            native_workspace_gib=0.25,
        )

    python_estimator = estimator("python", "python")
    native_estimator = estimator("native", "direct")
    try:
        def dense_feature_path_is_forbidden(*args, **kwargs):
            raise AssertionError("native backend materialized a dense X/W feature block")

        native_estimator._prepare_additive_block = dense_feature_path_is_forbidden
        native_estimator._prepare_interaction_block = dense_feature_path_is_forbidden
        python_estimator._compute_ldscore()
        native_estimator._compute_ldscore()
        for name in (
            "inv_sqrt_resvar_x_all", "inv_sqrt_resvar_w_all", "norm_x_all",
            "norm_w_all", "diag_nxe_x_all", "diag_nxe_w_all", "corr_xw_all",
            "gxx_ldscore", "gxe_ldscore", "exg_ldscore", "gee_ldscore",
        ):
            np.testing.assert_allclose(
                getattr(native_estimator, name), getattr(python_estimator, name),
                rtol=2e-11, atol=2e-11,
            )
        assert native_estimator.resource_estimates["target_source_columns"] == 40
        assert "jackknife_scratch_total_gib" not in native_estimator.resource_estimates
        manifest = json.loads(
            (tmp_path / "native.gxe.ref.json").read_text(encoding="utf-8")
        )
        backend = manifest["backend_provenance"]
        assert backend["schema_version"] == 3
        assert backend["backend_name"] == "gxeldcore_direct"
        assert backend["source_commit"] != "unknown"
        assert len(backend["source_tree_sha256"]) == 64
        assert len(backend["native_binary_sha256"]) == 64
        assert (
            backend["compile_options"]["blas_vendor"]
            == gxeldcore.build_info()["blas_vendor"]
        )
        assert backend["native_workspace_cap_bytes"] == int(0.25 * 1024**3)
        assert backend["actual_global_2b_source_columns"] == 40
        assert backend["actual_jackknife_2b_source_columns"] == 0
        assert backend["actual_target_source_columns"] == 40
        assert (
            native_estimator.resource_estimates[
                "native_opaque_projected_panel_prepare_peak_gib"
            ]
            > native_estimator.resource_estimates[
                "native_opaque_projected_panel_resident_gib"
            ]
        )
        feature_backend = manifest["feature_backend_provenance"]
        assert feature_backend["artifact_stage"] == "feature_construction"
        assert feature_backend["backend_name"] == "gxeldcore_direct"
        assert "jackknife" not in manifest
        assert not (tmp_path / "python.gxe.jackknife.npz").exists()
        assert not (tmp_path / "native.gxe.jackknife.npz").exists()

        phenotype_values = rng.normal(size=n)
        phenotype_path = tmp_path / "score.pheno.tsv"
        ids.assign(Y=phenotype_values).to_csv(
            phenotype_path, sep="\t", index=False
        )
        score_artifacts = score_phenotype_from_reference(
            reference_manifest=tmp_path / "native.gxe.ref.json",
            bed_path=prefix,
            env_path=env_path,
            covar_path=cov_path,
            pheno_path=phenotype_path,
            pheno_col="Y",
            output_prefix=tmp_path / "native-score",
            step_size=6,
            num_threads=2,
        )
        fixed_basis = np.asarray(native_estimator.C_int, dtype=np.float64)
        projected_phenotype = phenotype_values.astype(np.float64)
        projected_phenotype -= fixed_basis @ (
            fixed_basis.T @ projected_phenotype
        )
        projected_phenotype -= projected_phenotype.mean()
        residual_rank = n - fixed_basis.shape[1] - 1
        projected_phenotype *= np.sqrt(
            residual_rank / np.dot(projected_phenotype, projected_phenotype)
        )
        q_basis = np.asfortranarray(
            np.column_stack(
                [np.full(n, 1.0 / np.sqrt(n), dtype=np.float64), fixed_basis]
            )
        )
        expected_score_features = _python_features(
            raw, np.asarray(native_estimator.env), q_basis
        )
        observed_gwas = pd.read_csv(score_artifacts.gwas, sep=r"\s+")
        observed_gwis = pd.read_csv(score_artifacts.gwis, sep=r"\s+")
        np.testing.assert_allclose(
            observed_gwas["SCORE"],
            expected_score_features["x"].T @ projected_phenotype
            / np.sqrt(residual_rank),
            rtol=2e-11,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            observed_gwis["SCORE"],
            expected_score_features["w"].T @ projected_phenotype
            / np.sqrt(residual_rank),
            rtol=2e-11,
            atol=2e-11,
        )
        score_moments = json.loads(score_artifacts.moments.read_text())
        score_backend = score_moments["score_backend_provenance"]
        native_build = dict(gxeldcore.build_info())
        native_score_eligible = (
            native_build["blas_runtime_isolation"] == "private_static"
            and native_build["gemm_integrity_enabled"] is False
            and native_build["gemm_execution_mode"]
            == "serialized_fixed_private_openblas"
        )
        expected_score_mode = (
            "native_fused_sealed_scale_one_pass"
            if native_score_eligible
            else "python_numpy_one_pass"
        )
        assert score_backend["execution_mode"] == expected_score_mode
        if expected_score_mode == "native_fused_sealed_scale_one_pass":
            assert score_backend["api_version"] >= 6
            assert score_backend["gemm_integrity_enabled"] is False
            assert score_backend["strict_feature_moment_verification"] is False
            assert score_backend["repaired_feature_moment_columns"] == 0
            assert score_backend["repaired_gemm_output_columns"] == 0
            assert score_backend["retried_gemm_input_mutations"] == 0
        else:
            assert "api_version" not in score_backend
    finally:
        python_estimator.close()
        native_estimator.close()


def test_opt_in_native_backend_supports_float32_storage(tmp_path):
    rng = np.random.default_rng(177)
    n, m = 29, 9
    raw = rng.binomial(2, rng.uniform(0.15, 0.45, size=m), size=(n, m)).astype(float)
    prefix = _write_plink(tmp_path, raw, "gate")
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env_path = tmp_path / "gate.env.tsv"
    ids.assign(E=rng.normal(size=n)).to_csv(env_path, sep="\t", index=False)

    def estimator(name: str, backend: str) -> GenomewideEnvLDScore:
        return GenomewideEnvLDScore(
            bed_path=str(prefix), env_path=str(env_path), covar_path=None,
            annot_path=None, out_path=str(tmp_path / name), log=Logger(suppress=True),
                rand_dist="rademacher", low_level=None, num_vecs=10, step_size=5,
                seed=4, dtype="float32", kernel_mode="standardized",
                genotype_scale="sample", impute_method="mean", native_backend=backend,
                native_workspace_gib=0.25, num_threads=2,
            )

    python_estimator = estimator("gate-python", "python")
    native_estimator = estimator("gate-native", "direct")
    try:
        python_estimator._compute_ldscore()
        native_estimator._compute_ldscore()
        assert native_estimator.dtype is np.float32
        assert native_estimator.inv_sqrt_resvar_x_all.dtype == np.float64
        for name in (
            "gxx_ldscore", "gxe_ldscore", "exg_ldscore", "gee_ldscore"
        ):
            np.testing.assert_allclose(
                getattr(native_estimator, name),
                getattr(python_estimator, name),
                rtol=3e-6,
                atol=3e-6,
            )
    finally:
        python_estimator.close()
        native_estimator.close()


def test_float32_panel_gate_allows_bounded_accumulation_roundoff():
    class FakePanel:
        leakage = 1.1e-5

    class FakeContext:
        def __init__(self):
            self.tolerance = None
            self.sources = None

        def prepare_projected_sources(self, *, sources, tolerance):
            self.sources = sources
            self.tolerance = tolerance
            return FakePanel()

    estimator = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    estimator._native_context = FakeContext()
    estimator.nsnps = 454_207
    estimator.step_size = 500
    sources = np.zeros((3, 2), dtype=np.float32, order="F")

    panel, leakage = estimator._native_prepare_projected_sources(sources)

    expected = (
        32.0
        * np.finfo(np.float32).eps
        * np.sqrt(np.ceil(estimator.nsnps / estimator.step_size))
    )
    assert panel is not None
    assert leakage == 1.1e-5
    assert estimator._native_context.tolerance == pytest.approx(expected)
    assert 1.1e-5 < estimator._native_context.tolerance < 1.0e-3
    assert estimator._native_context.sources.dtype == np.float64


def test_checked_in_benchmark_records_4b_raw_timings_and_b128_scratch():
    args = benchmark_native._parser().parse_args(
        [
            "--n", "32", "--m", "12", "--probe-counts", "1024",
            "--repeats", "1", "--warmups", "0", "--decode-threads", "2",
            "--blas-threads", "2", "--skip-legacy",
        ]
    )
    payload = benchmark_native.run(args)
    assert payload["schema"] == "summit-native-gxe-benchmark-v2"
    assert len(payload["provenance"]["native_binary_sha256"]) == 64
    case = payload["cases"][0]
    assert case["actual_target_columns_2b"] == 2_048
    assert case["actual_target_columns_4b"] == 4_096
    assert case["production_scratch_model"]["probe_tiles"] == [
        [start, 128] for start in range(0, 1_024, 128)
    ]
    assert case["production_scratch_model"]["peak_tile_gib"] == pytest.approx(
        57.220458984375
    )
    assert len(case["timings_seconds"]["native_target_4b"]["raw"]) == 1
    assert case["correctness"]["max_abs_target_4b_error"] <= 1e-8
