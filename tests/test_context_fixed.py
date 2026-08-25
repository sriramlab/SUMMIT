from __future__ import annotations

import numpy as np

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis


def test_thin_rank_revealing_basis_handles_dependent_columns_without_n_squared() -> None:
    rng = np.random.default_rng(20260824)
    design = rng.normal(size=(257, 5))
    design = np.column_stack(
        [np.ones(design.shape[0]), design, design[:, 1] - 3.0 * design[:, 3]]
    )
    basis = thin_rank_revealing_fixed_effect_basis(design)
    assert basis.shape == (design.shape[0], np.linalg.matrix_rank(design))
    assert basis.flags.f_contiguous
    np.testing.assert_allclose(
        basis.T @ basis,
        np.eye(basis.shape[1]),
        rtol=0.0,
        atol=8.0e-15,
    )
    np.testing.assert_allclose(
        basis @ (basis.T @ design),
        design,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_thin_qr_projection_matches_dense_oracle_on_small_fixture() -> None:
    rng = np.random.default_rng(17)
    design = np.column_stack([np.ones(41), rng.normal(size=(41, 4))])
    panel = rng.normal(size=(41, 7))
    basis = thin_rank_revealing_fixed_effect_basis(design)
    thin = panel - basis @ (basis.T @ panel)
    dense = (np.eye(41) - basis @ basis.T) @ panel
    np.testing.assert_allclose(thin, dense, rtol=0.0, atol=3.0e-15)
