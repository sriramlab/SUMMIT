"""Shared panel means retain dense GLS results without duplicate large QRs."""
from types import SimpleNamespace

import numpy as np
import pytest

import summit.prediction.solver as solver
from summit.prediction.spec import SolverSpec


@pytest.mark.parametrize("shared", [True, False])
def test_shared_fixed_panel_matches_dense_gls(monkeypatch, shared):
    rng = np.random.default_rng(739512)
    n = 96
    base = np.column_stack([np.ones(n), rng.normal(size=(n, 7))])
    fixed = np.column_stack([base, 2 * base[:, 2]])
    other = fixed.copy()
    other[:, 1] = rng.normal(size=n)
    designs = [fixed if shared else fixed.copy() for _ in range(3)] + [other]
    traits, covariances = [], {}
    for j, design in enumerate(designs):
        z = rng.normal(size=(n, 12))
        covariance = z @ z.T / 12 + np.diag(rng.uniform(.8, 1.2, n))
        covariances[(str(j), "null")] = covariance
        traits.append(SimpleNamespace(id=str(j), rows=np.arange(n), fixed=design,
            y=rng.normal(size=n), phi=np.ones((n, 1)), candidates=[SimpleNamespace(
                id="null", covariance=np.zeros((1, 1)), residual=np.diag(covariance))]))

    class Operator:
        row_diagonal = {0: np.zeros(n)}
        trait_group = {t.id: 0 for t in traits}

        def apply(self, vectors, *, phase):
            return {key: covariances[key] @ value for key, value in vectors.items()}

    operator = Operator()
    operator.traits = traits
    original_qr, original_lstsq = solver.thin_rank_revealing_fixed_effect_basis, solver.linalg.lstsq
    qr_inputs, reduced_designs = [], []

    def qr(value, **kwargs):
        qr_inputs.append(value)
        return original_qr(value, **kwargs)

    def least_squares(a, b, **kwargs):
        reduced_designs.append(a)
        return original_lstsq(a, b, **kwargs)

    monkeypatch.setattr(solver, "thin_rank_revealing_fixed_effect_basis", qr)
    monkeypatch.setattr(solver.linalg, "lstsq", least_squares)
    result = solver.solve(operator, SolverSpec(rtol=1e-11))
    assert len(qr_inputs) == (2 if shared else 4)
    assert len({id(v) for v in reduced_designs}) == (2 if shared else 4)
    for trait in traits:
        key = (trait.id, "null")
        # An independent dense GLS on a nonredundant design verifies both
        # projected solutions and the nuisance fitted values, including the
        # separate trait whose fixed-effect span differs from the others.
        c = trait.fixed[:, :8]
        inverse_c = np.linalg.solve(covariances[key], c)
        inverse_y = np.linalg.solve(covariances[key], trait.y)
        nuisance = np.linalg.solve(c.T @ inverse_c, c.T @ inverse_y)
        expected = inverse_y - inverse_c @ nuisance
        np.testing.assert_allclose(result.solutions[key], expected, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(trait.fixed @ result.fixed_coefficients[key], c @ nuisance,
            rtol=2e-9, atol=2e-10)
        assert result.reports[key]["relative_true_residual"] <= 1e-11

