from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generalized_gxe"
    / "simulate_generalized_gxe.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("generalized_gxe_simulator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
WORKFLOW = sys.modules["workflow"]


class _QualifiedNative:
    @staticmethod
    def build_info() -> dict[str, object]:
        return {
            "blas_vendor": "BLIS",
            "private_blas_backend": "upstream_blis",
            "blas_runtime_isolation": "private_static",
            "blas_runtime_owner_thread_enforced": True,
            "blas_runtime_environment_immutable": True,
            "blas_runtime_threads": 2,
        }

    @staticmethod
    def configured_blas_threads() -> int:
        return 2

    @staticmethod
    def configure_openmp_placement(
        cpu_ids: list[int], threads: int
    ) -> dict[str, object]:
        return {
            "schema": "summit.openmp_placement_attestation.v1",
            "requested_threads": threads,
            "expected_cpu_ids": cpu_ids,
            "team_size": threads,
            "exact_singleton_places": True,
            "exact_team_coverage": True,
            "immutable": True,
            "verified": True,
        }


def test_validate_omega_accepts_psd_and_reconstructs() -> None:
    omega = np.asarray(
        [[0.2, 0.03, -0.02], [0.03, 0.15, 0.08], [-0.02, 0.08, 0.18]]
    )
    observed, root = MODULE.validate_omega(omega, 3)
    np.testing.assert_allclose(root @ root.T, observed, rtol=2e-14, atol=2e-14)


def test_validate_omega_rejects_asymmetry_and_indefiniteness() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        MODULE.validate_omega([[1.0, 0.2], [0.1, 1.0]], 2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        MODULE.validate_omega([[1.0, 2.0], [2.0, 1.0]], 2)


def test_block_scaling_mean_imputes_and_uses_sample_sd() -> None:
    raw = np.asarray(
        [[0.0, 2.0], [1.0, np.nan], [2.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
        order="F",
    )
    scaled, means, inverse, missing = MODULE.standardize_genotype_block(raw)
    np.testing.assert_allclose(means, [1.0, 1.0])
    np.testing.assert_array_equal(missing, [0, 1])
    np.testing.assert_allclose(np.mean(scaled, axis=0), 0.0, atol=2e-16)
    np.testing.assert_allclose(np.sum(scaled * scaled, axis=0), 3.0)
    assert np.all(inverse > 0.0)


def test_two_environments_are_centered_scaled_and_orthogonal() -> None:
    environment = MODULE.generate_environment(200, np.random.SeedSequence(91))
    np.testing.assert_allclose(np.mean(environment, axis=0), 0.0, atol=2e-16)
    np.testing.assert_allclose(environment.T @ environment, 199.0 * np.eye(2), atol=2e-12)


def test_symmetric_residual_basis_matches_context_pair_order() -> None:
    basis = np.asarray(
        [[1.0, -1.0, 2.0], [1.0, 0.5, -3.0], [1.0, 2.0, 4.0]]
    )
    residual, names, pairs = WORKFLOW.symmetric_context_residual_basis(basis)
    assert pairs == ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    assert names == (
        "residual_0_0",
        "residual_1_1",
        "residual_2_2",
        "residual_0_1",
        "residual_0_2",
        "residual_1_2",
    )
    expected = np.column_stack(
        [
            np.ones(3),
            basis[:, 1] ** 2,
            basis[:, 2] ** 2,
            2.0 * basis[:, 1],
            2.0 * basis[:, 2],
            2.0 * basis[:, 1] * basis[:, 2],
        ]
    )
    np.testing.assert_array_equal(residual, expected)


def test_private_blis_workflow_seals_owner_placement(monkeypatch) -> None:
    monkeypatch.setattr(WORKFLOW, "_PRE_NUMERICAL_CPU_AFFINITY", (3, 7))
    environment = {
        "OMP_NUM_THREADS": "2",
        "OMP_THREAD_LIMIT": "2",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": "{3},{7}",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "BLIS_NUM_THREADS": "2",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for name in (
        *WORKFLOW._OPENMP_AFFINITY_CONFLICTS,
        *WORKFLOW._BLIS_AUTOMATIC_CONFLICTS,
    ):
        monkeypatch.delenv(name, raising=False)
    _, threads, placement = WORKFLOW.require_private_blis(_QualifiedNative())
    assert threads == 2
    assert placement["expected_cpu_ids"] == [3, 7]


def test_private_blis_workflow_rejects_implicit_owner_binding(monkeypatch) -> None:
    monkeypatch.setattr(WORKFLOW, "_PRE_NUMERICAL_CPU_AFFINITY", (3, 7))
    environment = {
        "OMP_NUM_THREADS": "2",
        "OMP_THREAD_LIMIT": "2",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "BLIS_NUM_THREADS": "2",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="noncanonical generalized GxE threading"):
        WORKFLOW.require_private_blis(_QualifiedNative())
