from dataclasses import replace
import numpy as np
import pytest

from prediction_helpers import prediction_threads
from test_prediction_core import fixture
from summit.prediction import AnnotationDesign, AnnotationPrior, CandidatePrior, SolverSpec
from summit.prediction.annotations import write_annotation_design, load_annotation_design
from summit.prediction.api import fit_prediction
from summit.prediction.batch import plan_prediction
from summit.prediction.genotype import standardize
from summit.prediction.operator import GenotypeOperator
from summit.prediction.score import ScoreInput, score_prediction
from summit.prediction.solver import solve


def annotated_fixture(overlap=True):
    source, original = fixture()
    t = original[0]; m = len(t.variants)
    weights = np.column_stack([np.ones(m), np.arange(m) % 3 == 0]) if overlap else np.eye(2)[np.arange(m) % 2]
    design = AnnotationDesign(weights, ('background', 'coding'), t.scale.variant_identity)
    covariance = np.stack([t.candidates[0].covariance*.7, np.diag([.1, .025, 0])])
    prior = AnnotationPrior(design, covariance)
    zero = AnnotationPrior(design, np.zeros_like(covariance))
    candidates = (t.candidates[0], prior.candidate('annotated', t.candidates[0].residual, {'source': 'test'}),
                  zero.candidate('zero_annotated', t.candidates[0].residual, {'source': 'test'}))
    return source, replace(t, candidates=candidates)


def dense_annotated(source, trait, candidate):
    g = standardize(source.values[np.ix_(trait.rows, trait.variants)], trait.scale.mean, trait.scale.inverse_scale)
    p = candidate.annotation_prior
    covariances = (np.broadcast_to(candidate.covariance/len(trait.variants),
        (len(trait.variants), *candidate.covariance.shape)) if p is None else
        np.einsum('mk,kqr->mqr', p.design.weights/p.design.masses, p.covariances))
    # Explicit variant-wise covariance construction, independent of tiled
    # operator products and annotation aggregation code.
    v = np.diag(candidate.residual)
    for j in range(len(trait.variants)):
        feature = g[:, j, None]*trait.phi
        v += feature@covariances[j]@feature.T
    z = trait.fixed[:, :4]
    vy, vz = np.linalg.solve(v, trait.y), np.linalg.solve(v, z)
    fixed = np.linalg.solve(z.T@vz, z.T@vy)
    u = vy-vz@fixed
    weights = np.einsum('mq,mqr->mr', g.T@(trait.phi*u[:, None]), covariances)
    return u, weights, z@fixed, g


@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('storage', ['stream', 'compact', 'standardized'])
@pytest.mark.parametrize('backend', ['numpy', 'native'])
def test_annotated_operator_dense_gls_and_export(tmp_path, overlap, storage, backend):
    source, trait = annotated_fixture(overlap)
    plan = plan_prediction([trait], source, storage=storage, block_size=7, rhs_columns=6, threads=prediction_threads())
    operator = GenotypeOperator(source, [trait], plan, backend=backend)
    operator.setup()
    result = solve(operator, SolverSpec(rtol=1e-10, max_iterations=160))
    weights = {(trait.id, c.id): np.zeros((len(trait.variants), trait.phi.shape[1])) for c in trait.candidates}
    def sink(key, lo, hi, value):
        weights[key][lo:hi] = value
    operator.extract(result.solutions, sink)
    for candidate in trait.candidates:
        key = (trait.id, candidate.id)
        u, w, mean, _ = dense_annotated(source, trait, candidate)
        np.testing.assert_allclose(result.solutions[key], u, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(weights[key], w, rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(trait.fixed@result.fixed_coefficients[key], mean, rtol=2e-9, atol=2e-9)
        assert result.reports[key]['relative_true_residual'] <= 1e-10
    traversals = sum(operator.ledger.traversals.values())
    assert operator.ledger.source_variants+operator.ledger.cache_variants == len(trait.variants)*traversals
    if storage != 'stream':
        assert operator.ledger.source_variants == len(trait.variants)
    operator.release()


def test_annotation_normalization_identity_and_validation(tmp_path):
    source, trait = annotated_fixture()
    p = trait.candidates[1].annotation_prior
    np.testing.assert_allclose(p.block(0, len(trait.variants)).sum(axis=0)/len(trait.variants), p.aggregate.ravel(), atol=1e-15)
    with pytest.raises(ValueError):
        p.design.weights.setflags(write=True)
    with pytest.raises(ValueError):
        p.covariances.setflags(write=True)
    for weights in (np.zeros((5, 2)), -np.ones((5, 2)), np.full((5, 2), np.nan)):
        with pytest.raises(ValueError):
            AnnotationDesign(weights, ('a', 'b'), 'variants')
    with pytest.raises(ValueError, match='positive semidefinite'):
        AnnotationPrior(p.design, -np.ones((2, 3, 3)))
    wrong = AnnotationPrior(AnnotationDesign(p.design.weights, p.design.names, 'wrong-axis'), p.covariances)
    wrong_trait = replace(trait, candidates=(wrong.candidate('wrong', trait.candidates[0].residual, {'source': 'test'}),))
    with pytest.raises(ValueError, match='annotation variant/order'):
        plan_prediction([wrong_trait], source)
    changed = AnnotationPrior(p.design, p.covariances*2)
    changed_trait = replace(trait, candidates=(trait.candidates[0], changed.candidate('annotated', trait.candidates[1].residual, {'source': 'test'})))
    assert plan_prediction([trait], source).fit_identity != plan_prediction([changed_trait], source).fit_identity
    path = tmp_path/'design.npz'
    write_annotation_design(path, p.design)
    reloaded = load_annotation_design(path)
    assert reloaded.identity == p.design.identity
    with pytest.raises(FileExistsError):
        write_annotation_design(path, p.design)
    # Scaling one annotation and its mass equally changes no per-SNP prior.
    a = p.design.weights.copy(); a[:, 1] *= 17
    equivalent = AnnotationPrior(AnnotationDesign(a, p.design.names, p.design.variant_identity), p.covariances)
    np.testing.assert_allclose(equivalent.block(0, len(a)), p.block(0, len(a)), atol=1e-15)


@pytest.mark.parametrize('integrity,checksum', [(False, False), (True, False), (False, True)])
def test_unqualified_annotation_blis_rejected_before_genotype_preparation(monkeypatch, integrity, checksum):
    from types import SimpleNamespace
    source, trait = annotated_fixture()
    plan = plan_prediction([trait], source, block_size=7, rhs_columns=6, threads=prediction_threads())
    native = SimpleNamespace(build_info=lambda: dict(blas_vendor='BLIS',
        gemm_integrity_enabled=integrity, gemm_checksum_enabled=checksum))
    monkeypatch.setattr('summit.prediction.operator.native_module', lambda: native)
    monkeypatch.setattr(source, 'prepare', lambda *args: pytest.fail('must reject before genotype preparation'))
    with pytest.raises(ValueError, match='GXELDCORE_GEMM_CHECKSUM=ON'):
        GenotypeOperator(source, [trait], plan, backend='native')


@pytest.mark.parametrize('backend', ['numpy', 'native'])
def test_annotated_checkpoint_reload_and_score(tmp_path, monkeypatch, backend):
    source, trait = annotated_fixture()
    plan = plan_prediction([trait], source, storage='compact', block_size=7, rhs_columns=6, threads=prediction_threads())
    spec = SolverSpec(rtol=1e-10)
    expected = fit_prediction([trait], source, output=tmp_path/'expected', plan=plan, solver=spec, backend=backend)
    checkpoint = tmp_path/'checkpoint.npz'
    original = GenotypeOperator.apply
    calls = 0
    def interrupt(self, vectors, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise InterruptedError('termination during annotation fit')
        return original(self, vectors, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(GenotypeOperator, 'apply', interrupt)
        with pytest.raises(InterruptedError):
            fit_prediction([trait], source, output=tmp_path/'actual', plan=plan, solver=spec,
                backend=backend, checkpoint=checkpoint)
    prior = trait.candidates[1].annotation_prior
    changed_prior = AnnotationPrior(prior.design, prior.covariances*1.1)
    changed = replace(trait, candidates=(trait.candidates[0],
        changed_prior.candidate('annotated', trait.candidates[1].residual, {'source': 'test'}), trait.candidates[2]))
    with monkeypatch.context() as patch:
        patch.setattr(GenotypeOperator, 'setup', lambda self: pytest.fail('identity must fail before genotype reading'))
        with pytest.raises(ValueError, match='identity mismatch'):
            fit_prediction([changed], source, output=tmp_path/'bad', solver=spec,
                backend=backend, checkpoint=checkpoint, resume=True,
                plan=plan_prediction([changed], source, storage='compact', block_size=7, rhs_columns=6, threads=prediction_threads()))
    actual = fit_prediction([trait], source, output=tmp_path/'actual', plan=plan, solver=spec,
        backend=backend, checkpoint=checkpoint, resume=True)
    inputs = {trait.id: ScoreInput(trait.rows, trait.phi, trait.fixed, trait.context_spec, trait.fixed_spec)}
    score = score_prediction(actual, source, inputs, backend=backend, block_size=11, threads=prediction_threads())
    for a, b, candidate in zip(actual, expected, trait.candidates):
        np.testing.assert_array_equal(a.weights, b.weights)
        _, weights, _, g = dense_annotated(source, trait, candidate)
        np.testing.assert_allclose(score.components[a.key], g@weights, atol=1e-9)
        if candidate.annotation_prior is not None:
            assert a.prior_spec['annotation_prior']['identity'] == candidate.annotation_prior.identity
