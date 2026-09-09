from __future__ import annotations

import numpy as np
import pytest

from summit.prediction._validation import digest
from summit.prediction.spec import VariantAxis, TraitTraining, CandidatePrior, SolverSpec
from summit.prediction.genotype import ArrayGenotypeSource, estimate_scale, standardize
from summit.prediction.priors import ResponseGeometry, separate_scales, recode_response
from summit.prediction.batch import plan_prediction
from summit.prediction.operator import GenotypeOperator
from summit.prediction.solver import solve, ConvergenceError


def fixture(seed=911, n=61, m=43):
    rng = np.random.default_rng(seed)
    calls = rng.binomial(2, .35, size=(n, m)).astype(float)
    calls[rng.random(calls.shape) < .07] = np.nan
    calls[:, 0], calls[:, 1] = 1, np.nan
    axis = VariantAxis(tuple(f"rs{i}" for i in range(m)), ("1",)*m, tuple(range(1, m+1)), ("A",)*m, ("G",)*m, "GRCh37")
    source = ArrayGenotypeSource(calls, [("f", str(i)) for i in range(n)], axis, hard_calls=True)
    traits = []
    omega = np.array([[.3, .055, -.02], [.055, .09, .01], [-.02, .01, .07]])
    for ti in range(2):
        rows = np.flatnonzero(np.arange(n) % (5+ti) != ti)
        variants = np.arange(m) if ti == 0 else np.flatnonzero(np.arange(m) % 7 != 3)
        e = rng.normal(size=(len(rows), 2))
        phi = np.column_stack([np.ones(len(rows)), e])
        fixed = np.column_stack([phi, e[:, 0]**2, 2*e[:, 0]])
        residual = .3*np.exp(.6*e[:, 0])
        y = rng.normal(size=len(rows)) + phi @ np.array([2., .4, -.3])
        matrices = [omega, separate_scales(omega, .3, .7), separate_scales(omega, .7, 0),
                    np.diag([.2, 0, 0]), np.zeros((3, 3))]
        scale = estimate_scale(source, rows, variants, block_size=7)
        candidates = tuple(CandidatePrior(f"m{j}", mat, residual, {"operation": "supplied", "residual": "test"}) for j, mat in enumerate(matrices))
        traits.append(TraitTraining(f"t{ti}", rows, variants, y, phi, fixed, scale, candidates,
            {"names": ["intercept", "e1", "e2"]}, {"names": ["intercept", "e1", "e2", "e1_squared", "duplicate"]},
            {"units": "test"}, ResponseGeometry(omega, np.eye(2), "test")))
    return source, traits


def dense(source, trait, covariance, residual):
    raw = source.values[np.ix_(trait.rows, trait.variants)]
    g = standardize(raw, trait.scale.mean, trait.scale.inverse_scale)
    v = (g @ g.T / g.shape[1]) * (trait.phi @ covariance @ trait.phi.T) + np.diag(residual)
    # Independent direct GLS using an identified subset of the fixed design.
    z = trait.fixed[:, :4]
    vi_y, vi_z = np.linalg.solve(v, trait.y), np.linalg.solve(v, z)
    alpha = np.linalg.solve(z.T @ vi_z, z.T @ vi_y)
    u = vi_y - vi_z @ alpha
    weights = g.T @ (trait.phi * u[:, None]) @ covariance / g.shape[1]
    return u, weights, z @ alpha, g


@pytest.mark.parametrize("storage", ["stream", "compact", "standardized"])
@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_masked_joint_solve_dense_gls_and_passes(storage, backend):
    source, traits = fixture()
    plan = plan_prediction(traits, source, storage=storage, block_size=7, rhs_columns=6)
    operator = GenotypeOperator(source, traits, plan, backend=backend)
    operator.setup()
    result = solve(operator, SolverSpec(rtol=1e-10, max_iterations=160))
    weights = {(t.id, c.id): np.zeros((len(t.variants), t.phi.shape[1])) for t in traits for c in t.candidates}
    def sink(key, lo, hi, block):
        weights[key][lo:hi] = block
    operator.extract(result.solutions, sink)
    for t in traits:
        for c in t.candidates:
            key = (t.id, c.id)
            u, w, fixed_prediction, g = dense(source, t, c.covariance, c.residual)
            np.testing.assert_allclose(result.solutions[key], u, rtol=2e-8, atol=2e-8)
            np.testing.assert_allclose(weights[key], w, rtol=2e-8, atol=2e-9)
            np.testing.assert_allclose(t.fixed @ result.fixed_coefficients[key], fixed_prediction, rtol=2e-9, atol=2e-9)
            assert result.reports[key]["relative_true_residual"] <= 1e-10
            np.testing.assert_array_equal(weights[key][:2], 0)
            amplification, responses = t.geometry.decompose(g @ weights[key], t.phi[:, 1:])
            np.testing.assert_allclose(amplification+responses.sum(axis=1), np.sum((g @ weights[key])*t.phi, axis=1), atol=1e-13)
    ledger = operator.ledger
    traversals = sum(ledger.traversals.values())
    m = plan.union_variants
    assert ledger.source_variants + ledger.cache_variants == m*traversals
    assert ledger.source_blocks + ledger.cache_blocks == ((m+6)//7)*traversals
    if storage != "stream":
        assert ledger.source_variants == m


def test_prior_identities_and_basis_recoding():
    source, traits = fixture()
    t = traits[0]
    omega = t.geometry.omega
    np.testing.assert_allclose(separate_scales(omega, .7, .7), .7*omega, atol=1e-15)
    assert np.linalg.matrix_rank(separate_scales(omega, 1, 0), tol=1e-12) == 1
    transform = np.array([[2., .3], [.5, 1.1]])
    om, h = recode_response(omega, t.geometry.metric, transform)
    phi = t.phi.copy()
    phi[:, 1:] = phi[:, 1:] @ transform.T
    np.testing.assert_allclose(phi @ om @ phi.T, t.phi @ omega @ t.phi.T, atol=2e-14)
    geo = ResponseGeometry(om, h, "test")
    np.testing.assert_allclose(geo.eigenvalues, t.geometry.eigenvalues, atol=1e-14)
    shrunk = t.geometry.prior(tau=.6)
    shrunk_recoded, _ = recode_response(shrunk, t.geometry.metric, transform)
    np.testing.assert_allclose(geo.prior(tau=.6), shrunk_recoded, atol=1e-14)


def test_invalid_inputs_and_resource_admission():
    source, traits = fixture()
    with pytest.raises(MemoryError):
        plan_prediction(traits, source, memory_bytes=1024)
    with pytest.raises(ValueError, match="complete response"):
        plan_prediction(traits, source, rhs_columns=2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        CandidatePrior("bad", np.diag([1, -1]), np.ones(2), {"source": "test"})
    with pytest.raises(ValueError, match="strictly positive"):
        CandidatePrior("bad", np.eye(2), np.array([0., 1]), {"source": "test"})
    bad = np.full(source.values.shape, .1)
    with pytest.raises(ValueError, match="fractional"):
        ArrayGenotypeSource(bad, source.samples, source.variants, hard_calls=True)
    operator = GenotypeOperator(source, traits, plan_prediction(traits, source), backend="numpy")
    operator.setup()
    with pytest.raises(ConvergenceError) as exc:
        solve(operator, SolverSpec(rtol=1e-14, max_iterations=1))
    assert any(not r["converged"] for r in exc.value.reports.values())
