from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from summit.context.oracle import (
    coefficients_to_omegas,
    combine_genetic_kernels,
    common_scale_features,
    context_covariance_surface,
    contextual_scores,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    exact_same_person_matrix,
    genetic_residual_cross_traces,
    genetic_rhs_from_scores,
    genetic_trace_from_features,
    kernel_gram,
    kernel_rhs,
    kernel_traces,
    omegas_to_coefficients,
    project_normalize_phenotype,
    rank_revealing_projector,
    residual_moments_low_rank,
    scale_aware_max_discrepancy,
    symmetric_rank_diagnostics,
    transfer_reference_gram,
    transform_omega,
)
from summit.context.spec import ContextComponentIndex, ContextPairIndex
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore


def _fixture(*, seed: int, n: int, m: int, q: int, k: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    genotype = rng.normal(size=(n, m)) + 0.15 * latent[:, :1] * rng.normal(size=(1, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    phi = np.ones((n, q), dtype=np.float64)
    for column in range(1, q):
        phi[:, column] = rng.normal(size=n) + (0.2 * column) * latent[:, 0]
    fixed = np.column_stack([np.ones(n), phi[:, 1:], latent[:, 1]])
    projector = rank_revealing_projector(fixed)
    y = project_normalize_phenotype(rng.normal(size=n), projector)
    if k == 1:
        annotations = np.ones((m, 1), dtype=np.float64)
        names = ("all",)
    else:
        annotations = np.zeros((m, 2), dtype=np.float64)
        annotations[: m // 2, 0] = 1.0
        annotations[m // 2 :, 1] = 1.0
        names = ("first", "second")
    residual = np.column_stack([np.ones(n), phi[:, min(1, q - 1)] ** 2 + 0.25])
    components = ContextComponentIndex(names, ContextPairIndex(q))
    return genotype, phi, fixed, projector, y, annotations, residual, components


@pytest.mark.parametrize(
    "n,m,q,k",
    [
        (20, 15, 1, 1),
        (20, 40, 2, 2),
        (50, 15, 3, 1),
        (50, 40, 4, 2),
        (100, 100, 4, 1),
    ],
)
def test_dense_summary_reconstructs_every_normal_equation_block(
    n: int, m: int, q: int, k: int
) -> None:
    (
        genotype,
        phi,
        _,
        projector,
        y,
        annotations,
        residual,
        components,
    ) = _fixture(seed=1000 + n + m + q + k, n=n, m=m, q=q, k=k)
    features = common_scale_features(genotype, phi, projector.projector)
    genetic_kernels = dense_genetic_kernels(features, annotations, components)
    residual_kernels = dense_residual_kernels(projector.projector, residual)
    dense = dense_normal_equations(
        genetic_kernels,
        residual_kernels,
        y,
        components.names,
        ("residual:constant", "residual:context"),
    )

    scores = contextual_scores(features, y, projector.residual_rank)
    genetic_rhs = genetic_rhs_from_scores(
        scores, annotations, components, projector.residual_rank
    )
    genetic_traces = genetic_trace_from_features(features, annotations, components)
    genetic_residual = genetic_residual_cross_traces(
        features, residual, annotations, components
    )
    residual_moments = residual_moments_low_rank(projector.fixed_basis, residual, y)
    p = len(components)

    assert scale_aware_max_discrepancy(genetic_rhs, dense.rhs[:p]) < 1e-12
    assert scale_aware_max_discrepancy(genetic_traces, dense.traces[:p]) < 1e-12
    assert scale_aware_max_discrepancy(genetic_residual, dense.matrix[:p, p:]) < 1e-12
    assert scale_aware_max_discrepancy(residual_moments.rhs, dense.rhs[p:]) < 1e-12
    assert (
        scale_aware_max_discrepancy(residual_moments.traces, dense.traces[p:]) < 1e-12
    )
    assert (
        scale_aware_max_discrepancy(residual_moments.gram, dense.matrix[p:, p:]) < 1e-12
    )
    np.testing.assert_allclose(
        kernel_gram(genetic_kernels), dense.matrix[:p, :p], rtol=2e-15, atol=2e-13
    )
    np.testing.assert_allclose(
        kernel_rhs(genetic_kernels, y), dense.rhs[:p], rtol=2e-14, atol=2e-13
    )
    np.testing.assert_allclose(
        kernel_traces(genetic_kernels), dense.traces[:p], rtol=2e-14, atol=2e-13
    )
    same = exact_same_person_matrix(genetic_kernels)
    diagonal = np.diagonal(genetic_kernels, axis1=1, axis2=2)
    np.testing.assert_allclose(same, diagonal @ diagonal.T, rtol=0.0, atol=0.0)
    assert np.linalg.eigvalsh(same)[0] > -1e-11


def test_off_diagonal_factor_two_is_in_kernel_and_score_once() -> None:
    features = np.array(
        [
            [[1.0, -2.0], [0.5, 1.0], [-1.5, 0.25]],
            [[-0.5, 1.0], [2.0, -0.25], [1.0, 0.5]],
        ]
    )
    y = np.array([1.0, -2.0, 0.5])
    annotations = np.ones((2, 1))
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    kernels = dense_genetic_kernels(features, annotations, components)
    expected_cross = (features[0] @ features[1].T + features[1] @ features[0].T) / 2.0
    np.testing.assert_array_equal(kernels[2], expected_cross)
    scores = contextual_scores(features, y, residual_rank=2)
    rhs = genetic_rhs_from_scores(scores, annotations, components, residual_rank=2)
    expected_rhs = float(y @ expected_cross @ y)
    assert expected_rhs < 0.0
    assert rhs[2] == pytest.approx(expected_rhs, rel=2e-15, abs=2e-15)
    one_orientation = float(y @ (features[0] @ features[1].T / 2.0) @ y)
    assert rhs[2] == pytest.approx(2.0 * one_orientation)
    assert not np.isclose(rhs[2], one_orientation)


def test_off_diagonal_factor_propagates_twice_into_gram_and_same_person() -> None:
    rng = np.random.default_rng(71)
    features = rng.normal(size=(2, 8, 6))
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    kernels = dense_genetic_kernels(features, np.ones((6, 1)), components)
    half_cross = 0.5 * kernels[2]
    gram = kernel_gram(kernels)
    same = exact_same_person_matrix(kernels)
    assert gram[0, 2] == pytest.approx(2.0 * np.sum(kernels[0] * half_cross), rel=3e-15)
    assert gram[2, 2] == pytest.approx(4.0 * np.sum(half_cross * half_cross), rel=3e-15)
    diag0 = np.diag(kernels[0])
    diag_half = np.diag(half_cross)
    assert same[0, 2] == pytest.approx(2.0 * np.dot(diag0, diag_half), rel=3e-15)
    assert same[2, 2] == pytest.approx(4.0 * np.dot(diag_half, diag_half), rel=3e-15)


def test_projection_and_context_multiplication_do_not_commute() -> None:
    rng = np.random.default_rng(81)
    n, m = 20, 15
    context = rng.normal(size=n)
    genotype = rng.normal(size=(n, m))
    projector = rank_revealing_projector(
        np.column_stack([np.ones(n), context, rng.normal(size=n)])
    ).projector
    target = projector @ (context[:, None] * genotype)
    wrong = projector @ (context[:, None] * (projector @ genotype))
    assert np.linalg.norm(target - wrong) > 0.1
    features = common_scale_features(
        genotype, np.column_stack([np.ones(n), context]), projector
    )
    np.testing.assert_allclose(features[1], target, rtol=0.0, atol=0.0)


def test_existing_raw_projected_option_matches_declared_common_scale_features() -> None:
    rng = np.random.default_rng(83)
    n, m = 30, 12
    genotype = rng.normal(size=(n, m))
    environment = rng.normal(size=n)
    covariate = rng.normal(size=n)
    covariate -= covariate.mean()
    covariate /= np.linalg.norm(covariate)
    legacy = SimpleNamespace(
        p_eff=1,
        C_int=covariate[:, None],
        cov_R_int=covariate[None, :],
        kernel_mode="raw_projected",
        env=environment,
        dtype=np.float64,
        inv_sqrt_resvar_x_all=None,
        inv_sqrt_resvar_w_all=None,
    )
    legacy._project_and_center_inplace = MethodType(
        GenomewideEnvLDScore._project_and_center_inplace, legacy
    )
    additive = GenomewideEnvLDScore._prepare_additive_block(
        legacy, 0, m, G=genotype, apply_scale=True, out_dtype=np.float64
    )
    interaction = GenomewideEnvLDScore._prepare_interaction_block(
        legacy, 0, m, G=genotype, apply_scale=True, out_dtype=np.float64
    )
    projector = rank_revealing_projector(
        np.column_stack([np.ones(n), covariate])
    ).projector
    contextual = common_scale_features(
        genotype, np.column_stack([np.ones(n), environment]), projector
    )
    np.testing.assert_allclose(contextual[0], additive, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(contextual[1], interaction, rtol=0.0, atol=3e-15)


def test_common_scale_basis_equivariance_for_q3_nonorthogonal_transform() -> None:
    (
        genotype,
        phi,
        _,
        projector,
        _,
        annotations,
        _,
        components,
    ) = _fixture(seed=221, n=50, m=40, q=3, k=1)
    transform = np.array([[1.0, 0.3, -0.2], [0.2, 1.2, 0.4], [-0.1, 0.25, 0.9]])
    assert abs(np.linalg.det(transform)) > 0.5
    transformed_phi = phi @ transform.T
    features = common_scale_features(genotype, phi, projector.projector)
    transformed_features = common_scale_features(
        genotype, transformed_phi, projector.projector
    )
    np.testing.assert_allclose(
        transformed_features,
        np.einsum("aq,qnm->anm", transform, features),
        rtol=3e-15,
        atol=3e-14,
    )
    kernels = dense_genetic_kernels(features, annotations, components)
    transformed_kernels = dense_genetic_kernels(
        transformed_features, annotations, components
    )
    omega = np.array([[0.8, -0.15, 0.12], [-0.15, 0.5, 0.08], [0.12, 0.08, 0.35]])
    transformed_omega = transform_omega(omega, transform)
    covariance = combine_genetic_kernels(
        kernels, omegas_to_coefficients(omega[None, :, :], components)
    )
    transformed_covariance = combine_genetic_kernels(
        transformed_kernels,
        omegas_to_coefficients(transformed_omega[None, :, :], components),
    )
    assert scale_aware_max_discrepancy(covariance, transformed_covariance) < 1e-12
    context_grid = np.array([[1.0, -1.0, 0.5], [1.0, 0.0, -0.25], [1.0, 1.0, 1.5]])
    surface = context_covariance_surface(omega, context_grid)
    transformed_surface = context_covariance_surface(
        transformed_omega, context_grid @ transform.T
    )
    assert scale_aware_max_discrepancy(surface, transformed_surface) < 1e-12


def test_binary_one_hot_and_intercept_binary_covariance_are_equivalent() -> None:
    e = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    one_hot = np.column_stack([1.0 - e, e])
    intercept = np.column_stack([np.ones(e.size), e])
    transform = np.array([[1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(intercept, one_hot @ transform.T)
    omega_one_hot = np.array([[0.4, 0.2], [0.2, 0.7]])
    omega_intercept = transform_omega(omega_one_hot, transform)
    np.testing.assert_allclose(
        context_covariance_surface(omega_one_hot, one_hot),
        context_covariance_surface(omega_intercept, intercept),
        rtol=2e-15,
        atol=2e-15,
    )


def test_three_category_one_hot_is_a_valid_full_genetic_basis() -> None:
    rng = np.random.default_rng(91)
    categories = np.repeat(np.arange(3), 8)
    one_hot = np.eye(3)[categories]
    genotype = rng.normal(size=(categories.size, 12))
    # The nuisance design may use reference coding while the genetic basis keeps
    # all three categories.
    projector = rank_revealing_projector(
        np.column_stack([np.ones(categories.size), one_hot[:, 1:]])
    ).projector
    components = ContextComponentIndex(("all",), ContextPairIndex(3))
    kernels = dense_genetic_kernels(
        common_scale_features(genotype, one_hot, projector),
        np.ones((genotype.shape[1], 1)),
        components,
    )
    assert kernels.shape == (6, categories.size, categories.size)
    diagnostics = symmetric_rank_diagnostics(kernel_gram(kernels))
    assert diagnostics.rank == 6


def test_rank_diagnostics_and_transfer_fail_closed() -> None:
    n = 20
    duplicated = np.column_stack([np.ones(n), np.arange(n), np.arange(n)])
    projector = rank_revealing_projector(duplicated)
    assert projector.rank == 2
    assert projector.residual_rank == 18
    np.testing.assert_allclose(
        projector.projector @ duplicated, 0.0, rtol=0.0, atol=2e-13
    )
    np.testing.assert_allclose(
        projector.projector @ projector.projector,
        projector.projector,
        rtol=0.0,
        atol=2e-14,
    )
    gram = np.array([[4.0, -1.0], [-1.0, 3.0]])
    same = np.array([[1.5, -0.25], [-0.25, 1.0]])
    np.testing.assert_array_equal(
        transfer_reference_gram(gram, same, reference_n=50, study_n=50), gram
    )
    expected = (30 / 50) * same + (30 * 29 / (50 * 49)) * (gram - same)
    np.testing.assert_allclose(
        transfer_reference_gram(gram, same, reference_n=50, study_n=30),
        expected,
    )
    with pytest.raises(ValueError, match="reference_n"):
        transfer_reference_gram(gram, same, reference_n=1, study_n=20)

    rng = np.random.default_rng(15)
    genotype = rng.normal(size=(n, 15))
    phi = np.column_stack([np.ones(n), np.ones(n)])
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    kernels = dense_genetic_kernels(
        common_scale_features(genotype, phi, projector.projector),
        np.ones((15, 1)),
        components,
    )
    rank = symmetric_rank_diagnostics(kernel_gram(kernels))
    assert rank.rank == 1
    assert rank.null_space.shape == (3, 2)
    assert np.isinf(rank.condition_number)


def test_coefficient_pack_round_trip_has_no_hidden_off_diagonal_factor() -> None:
    components = ContextComponentIndex(("a", "b"), ContextPairIndex(3))
    omega = np.array(
        [
            [[1.0, -0.2, 0.3], [-0.2, 2.0, 0.4], [0.3, 0.4, 3.0]],
            [[0.5, 0.1, -0.1], [0.1, 0.75, 0.2], [-0.1, 0.2, 1.25]],
        ]
    )
    packed = omegas_to_coefficients(omega, components)
    assert packed[:6].tolist() == [1.0, 2.0, 3.0, -0.2, 0.3, 0.4]
    np.testing.assert_array_equal(coefficients_to_omegas(packed, components), omega)
