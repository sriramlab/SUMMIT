import numpy as np
import pytest
from summit.epistasis.split_reference import ridge_transfer, split_scalar_reference


@pytest.mark.parametrize("m", [13, 80])
def test_transfer_against_augmented_fit_and_noise(m):
    rng = np.random.default_rng(520671)
    n, k = 60, 50
    x, z = rng.normal(size=(n, m)), rng.normal(size=(k, m))
    c = np.column_stack([np.ones(n), x[:, :2]])
    d = np.column_stack([np.ones(k), z[:, :2]])
    f0, f1 = x[:, 0] * x[:, 2], z[:, 0] * z[:, 2]
    y = rng.normal(size=n)
    ly = ridge_transfer(x, z, c, d, 17.0)
    design = np.column_stack([c, x])
    penalty = np.column_stack([np.zeros((m, 3)), np.sqrt(17) * np.eye(m)])
    coef = np.linalg.lstsq(
        np.vstack([design, penalty]), np.r_[y, np.zeros(m)], rcond=None
    )[0]
    np.testing.assert_allclose(ly @ y, np.column_stack([d, z]) @ coef, atol=2e-13)
    v0, v1 = 0.3 + x[:, 0] ** 2, 0.3 + z[:, 0] ** 2
    mu0, mu1 = x[:, 3], z[:, 3]
    diagnostic = split_scalar_reference(
        f0,
        f1,
        d,
        ly,
        ly,
        train_mean=mu0,
        test_mean=mu1,
        train_variance=v0,
        test_variance=v1,
    )
    assert diagnostic["interaction_response"] == pytest.approx(1.0)
    draws = 20000
    estimate = diagnostic["test_contrast"] @ (
        mu1[:, None] + np.sqrt(v1[:, None]) * rng.normal(size=(k, draws))
    ) - diagnostic["training_contrast"] @ (
        mu0[:, None] + np.sqrt(v0[:, None]) * rng.normal(size=(n, draws))
    )
    variance = diagnostic["total_noise_variance"]
    assert abs(estimate.mean() - diagnostic["expectation"]) < 5 * np.sqrt(
        variance / draws
    )
    assert abs(estimate.var() / variance - 1) < 0.035
    lf = ridge_transfer(x, z, c, d, 900.0)
    different = split_scalar_reference(
        f0,
        f1,
        d,
        ly,
        lf,
        train_mean=mu0,
        test_mean=mu1,
        train_variance=v0,
        test_variance=v1,
    )
    assert abs(different["interaction_response"] - 1) > 1e-3


def test_split_reference_rejects_unidentified_fixed_effects():
    with pytest.raises(ValueError, match="not identified"):
        ridge_transfer(np.eye(5), np.eye(5), np.zeros((5, 1)), np.ones((5, 1)), 1)


def test_crossfit_combines_overlapping_training_influence():
    from summit.epistasis.split_reference import crossfit_scalar_reference

    rng = np.random.default_rng(74091)
    n, m = 120, 80
    x = rng.normal(size=(n, m))
    c = np.ones((n, 1))
    f = x[:, 0] * x[:, 1]
    folds = np.array_split(np.arange(n), 3)
    transfers = []
    for ids in folds:
        train = np.setdiff1d(np.arange(n), ids)
        transfers.append(ridge_transfer(x[train], x[ids], c[train], c[ids], m))
    ref = crossfit_scalar_reference(f, c, folds, transfers, np.ones(n))
    assert ref["interaction_response"] == pytest.approx(1.0)
    assert (
        abs(ref["variance"] - ref["variance_if_folds_incorrectly_independent"]) > 1e-4
    )
    y = rng.normal(size=(n, 12000))
    estimates = ref["influence"] @ y
    assert abs(estimates.var() / ref["variance"] - 1) < 0.035


def test_estimated_noise_surface_against_explicit_residual_operator():
    from summit.epistasis.split_reference import estimated_contrast_noise

    rng = np.random.default_rng(79251)
    n = 180
    design = np.column_stack([np.ones(n), rng.normal(size=(n, 12))])
    d = design @ rng.normal(size=(13, 2)) / n
    y = design @ rng.normal(size=(13, 3)) + rng.normal(size=(n, 3))
    projection = np.eye(n) - design @ np.linalg.pinv(design)
    basis = np.column_stack([np.ones(n), design[:, 1] ** 2])
    coefficients = np.linalg.lstsq(
        (projection * projection) @ basis, (projection @ y) ** 2, rcond=None
    )[0]
    result = estimated_contrast_noise(d, y, design, basis)
    np.testing.assert_allclose(
        result["variance_coefficients"], coefficients, atol=1e-13
    )
    for j in range(3):
        np.testing.assert_allclose(
            result["covariance"][j],
            d.T @ np.diag(basis @ coefficients[:, j]) @ d,
            atol=1e-13,
        )
    result = estimated_contrast_noise(d, y, design)
    np.testing.assert_allclose(
        result["variance_coefficients"][0],
        np.sum((projection @ y) ** 2, axis=0) / (n - 13),
    )
    with pytest.raises(ValueError, match="mean span"):
        estimated_contrast_noise(rng.normal(size=n), y, design)
    from scipy.optimize import nnls

    constrained = estimated_contrast_noise(
        d, y, design, basis, nonnegative_components=True
    )
    expected = np.column_stack(
        [nnls((projection * projection) @ basis, r * r)[0] for r in (projection @ y).T]
    )
    np.testing.assert_allclose(
        constrained["variance_coefficients"], expected, atol=1e-13
    )
