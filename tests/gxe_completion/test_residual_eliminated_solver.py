from __future__ import annotations

import numpy as np
import pytest

from summit.inference.gxe import (
    GxENormalEquations,
    solve_normal_equations,
    transfer_reference_normal_equations,
)


def _equations(reduced: np.ndarray, truth: np.ndarray) -> GxENormalEquations:
    residual_trace = 10.0
    traces = np.asarray([2.0, 3.0, residual_trace])
    matrix = np.empty((3, 3), dtype=np.float64)
    matrix[:-1, :-1] = reduced + np.outer(traces[:-1], traces[:-1]) / residual_trace
    matrix[:-1, -1] = traces[:-1]
    matrix[-1, :-1] = traces[:-1]
    matrix[-1, -1] = residual_trace
    rhs = matrix @ truth
    assert rhs[-1] == pytest.approx(residual_trace)
    return GxENormalEquations(
        matrix=matrix,
        rhs=rhs,
        traces=traces,
        component_names=("G:a", "GxE:a", "residual"),
    )


def test_residual_eliminated_svd_matches_dense_unconstrained_solution() -> None:
    nonresidual = np.asarray([0.2, -0.1])
    residual = 1.0 - np.dot([2.0, 3.0], nonresidual) / 10.0
    truth = np.concatenate([nonresidual, [residual]])
    equations = _equations(np.asarray([[5.0, 1.0], [1.0, 4.0]]), truth)
    fit = solve_normal_equations(equations)
    np.testing.assert_allclose(fit.coefficients, truth, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        fit.coefficients,
        np.linalg.solve(equations.matrix, equations.rhs),
        rtol=2e-15,
        atol=2e-15,
    )
    assert fit.coefficients[1] < 0.0  # no silent non-negativity constraint
    assert fit.solve_method == "residual_eliminated_svd"
    assert fit.rank == 3
    assert fit.identifiable is True
    assert fit.normal_symmetry_error == 0.0
    assert fit.cauchy_schwarz_max_violation == 0.0
    assert fit.component_influence.shape == (3,)
    assert fit.relative_residual < 2e-16


def test_solver_rejects_material_asymmetry_before_symmetrizing() -> None:
    truth = np.asarray([0.1, 0.2, 0.92])
    equations = _equations(np.asarray([[5.0, 1.0], [1.0, 4.0]]), truth)
    equations.matrix[0, 1] += 1.0e-4
    with pytest.raises(ValueError, match="material asymmetry"):
        solve_normal_equations(equations)


def test_solver_rejects_cauchy_schwarz_violation() -> None:
    truth = np.asarray([0.1, 0.2, 0.92])
    equations = _equations(np.asarray([[5.0, 1.0], [1.0, 4.0]]), truth)
    equations.matrix[0, 1] = equations.matrix[1, 0] = 9.0
    with pytest.raises(ValueError, match="Cauchy"):
        solve_normal_equations(equations)


def test_rank_deficiency_fails_closed_or_is_explicit_when_overridden() -> None:
    nonresidual = np.asarray([0.2, 0.2])
    residual = 1.0 - np.dot([2.0, 3.0], nonresidual) / 10.0
    equations = _equations(
        np.asarray([[2.0, 2.0], [2.0, 2.0]]),
        np.concatenate([nonresidual, [residual]]),
    )
    with pytest.raises(ValueError, match="not identifiable"):
        solve_normal_equations(equations)
    fit = solve_normal_equations(equations, allow_ill_conditioned=True)
    assert fit.rank == 2
    assert fit.identifiable is False
    assert np.isinf(fit.condition_number) or fit.condition_number > 1e15


@pytest.mark.parametrize("study_n", [30, 50, 80])
def test_population_transfer_records_interpolation_identity_and_extrapolation(
    study_n: int,
) -> None:
    reference_n = 50
    reference_rank = 47
    study_rank = study_n - 3
    traces = np.asarray([reference_rank, reference_rank, 7.0, reference_rank])
    genetic = np.asarray([[140.0, 35.0], [35.0, 110.0]])
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[:2, :2] = genetic
    matrix[:2, 2] = matrix[2, :2] = [4.0, 5.0]
    matrix[:2, 3] = matrix[3, :2] = reference_rank
    matrix[2:, 2:] = [[12.0, 7.0], [7.0, reference_rank]]
    reference = GxENormalEquations(
        matrix, np.zeros(4), traces, ("G:a", "GxE:a", "NxE", "residual")
    )
    same = np.asarray([[40.0, 10.0], [10.0, 30.0]])
    transferred = transfer_reference_normal_equations(
        reference,
        reference_n_samples=reference_n,
        study_n_samples=study_n,
        reference_residual_rank=reference_rank,
        study_residual_rank=study_rank,
        same_individual_products=same,
        genetic_nxe_traces=np.asarray([3.0, 4.0]),
        q_nxe=6.0,
        q_residual=float(study_rank),
        trace_nxe=8.0,
        trace_nxe_sq=13.0,
    )
    expected = (
        (study_n / reference_n) * same
        + (study_n * (study_n - 1) / (reference_n * (reference_n - 1)))
        * (genetic - same)
    )
    np.testing.assert_allclose(transferred.matrix[:2, :2], expected)
    diagnostics = transferred.diagnostics
    assert diagnostics is not None
    assert diagnostics["extrapolation"] is (study_n > reference_n)
    assert diagnostics["same_individual_minimum_eigenvalue"] > 0.0
    if study_n == reference_n:
        np.testing.assert_array_equal(transferred.matrix[:2, :2], genetic)
