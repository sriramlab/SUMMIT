"""Independent arithmetic and continuation checks for exact mixture work reduction."""
from dataclasses import replace
import json
import numpy as np
import pytest

from prediction_helpers import prediction_threads
from test_prediction_core import fixture
from summit.prediction.mixture import MixtureSolverSpec, MixtureSpec, fit_mixture_prediction


@pytest.mark.parametrize('mixture', [MixtureSpec(.5,.5), MixtureSpec(.03,.1)])
def test_padded_posterior_reuse_is_bitwise_exact(mixture, monkeypatch):
    from summit.prediction.mixture import _posterior
    rng = np.random.default_rng(744)
    x = rng.normal(size=(11,5,5))
    covariance = x @ x.transpose(0,2,1)
    covariance[0] = 0.
    gram = np.repeat(np.eye(5)[None],11,axis=0)
    probabilities, components = mixture.components(covariance)
    probabilities = np.repeat(probabilities/2,2)
    components = np.repeat(components,2,axis=0)
    expected_covariance, expected_normalizers = [], []
    for probability, component in zip(probabilities,components):
        values,vectors = np.linalg.eigh(component)
        scale = np.maximum(np.max(abs(values),axis=1,keepdims=True),np.finfo(float).tiny)
        values = np.where(values>scale*1e-12,values,0)
        factor = vectors*np.sqrt(values[:,None,:])
        precision = np.eye(5)+factor.transpose(0,2,1)@gram@factor
        sign,logdet = np.linalg.slogdet(precision)
        assert np.all(sign>0)
        expected_covariance.append(factor@np.linalg.solve(precision,factor.transpose(0,2,1)))
        expected_normalizers.append(np.log(probability)-.5*logdet)
    count = []
    original = np.linalg.eigh
    def counted(value):
        count.append(1)
        return original(value)
    monkeypatch.setattr(np.linalg,'eigh',counted)
    actual,normalizers = _posterior(gram,covariance,mixture,4)
    np.testing.assert_array_equal(actual,np.stack(expected_covariance,axis=1))
    np.testing.assert_array_equal(normalizers,np.stack(expected_normalizers,axis=1))
    assert len(count)==(1 if mixture.probability==.5 else 2)


@pytest.mark.parametrize('rank', [0, 7])
@pytest.mark.parametrize('basis_error', [0., 1e-7])
def test_deferred_projection_matches_explicit_updates(rank, basis_error):
    from summit import gxeldcore as native
    from summit.prediction.runtime import configure_prediction_threads
    threads = prediction_threads()
    configure_prediction_threads(native, threads)
    rng = np.random.default_rng(883)
    n, d, k = 229, 33, 5
    basis = np.asfortranarray(np.linalg.qr(rng.normal(size=(n, rank)))[0] * (1+basis_error))
    yw = rng.normal(size=n)
    old = native.PredictionMixtureResidual(yw, basis, k, threads)
    new = native.PredictionMixtureResidual(yw, basis, k, threads, True)
    residual = np.repeat((yw-basis@(basis.T@yw))[:, None], k, axis=1)
    for step in range(17):
        w = np.asfortranarray(rng.normal(size=(n, d)))
        projection = np.asfortranarray(basis.T @ w)
        a, b = (np.empty((d, k), order='F') for _ in range(2))
        old.score(w, projection, a)
        new.score(w, projection, b)
        expected = w.T @ (residual-basis@(basis.T@residual))
        np.testing.assert_allclose(b, expected, rtol=2e-12, atol=2e-11)
        np.testing.assert_allclose(b, a, rtol=2e-12, atol=2e-11)
        delta = np.asfortranarray(rng.normal(scale=.02, size=(d,k)))
        old.update(w, delta, projection)
        new.update(w, delta, projection)
        residual -= (w-basis@projection) @ delta
        out = np.empty((n,k), order='F')
        new.copy_residual(out)
        np.testing.assert_allclose(out, residual, rtol=2e-12, atol=2e-12)
        if step % 5 == 0:
            new.synchronize()
            new.copy_residual(out)
            restarted = native.PredictionMixtureResidual(yw, basis, k, threads, True)
            restarted.restore(out)
            new.score(w, projection, a)
            restarted.score(w, projection, b)
            np.testing.assert_array_equal(a,b)


def solve(root, source, traits, *, freeze=True, deferred=True, **kwargs):
    return fit_mixture_prediction(traits, source, output=root, storage='compact', block_size=7,
        threads=prediction_threads(), solver=MixtureSolverSpec(rtol=1e-10, max_sweeps=150,
            freeze_converged=freeze, deferred_projection=deferred),
        mixtures={(t.id,c.id): MixtureSpec(.1,.1) for t in traits for c in t.candidates}, **kwargs)


@pytest.mark.parametrize('backend', ['numpy','native'])
def test_certified_compaction_preserves_fits_and_reduces_work(tmp_path, backend):
    source, traits = fixture(n=73,m=31)
    # One trait also has distinct candidate-specific heterogeneous noise surfaces.
    t = traits[0]
    candidates = list(t.candidates)
    candidates[1] = replace(candidates[1], residual=2*candidates[1].residual)
    traits[0] = replace(t,candidates=tuple(candidates))
    progress = []
    optimized = solve(tmp_path/'optimized',source,traits,backend=backend,progress=progress.append)
    baseline = solve(tmp_path/'baseline',source,traits,freeze=False,deferred=False,backend=backend)
    assert progress[0]['active_candidates'] < sum(len(t.candidates) for t in traits)
    assert progress[-1]['active_candidates'] == 0
    for a,b in zip(optimized,baseline):
        assert a.key == b.key
        np.testing.assert_allclose(a.weights,b.weights,rtol=2e-7,atol=2e-9)
        np.testing.assert_allclose(a.fixed_coefficients,b.fixed_coefficients,rtol=2e-7,atol=2e-9)
        assert a.convergence['true_residual_norm'] <= a.convergence['threshold']
    reports = [json.loads((tmp_path/mode/'manifest.json').read_text())['run_report']
               for mode in ('optimized','baseline')]
    assert reports[0]['candidate_work']['candidate_block_updates'] < reports[1]['candidate_work']['candidate_block_updates']


@pytest.mark.parametrize('backend', ['numpy','native'])
def test_resume_after_compaction_is_bitwise_identical(tmp_path, backend):
    source, traits = fixture(n=53,m=23)
    checkpoint = tmp_path/'checkpoint.npz'
    class Interrupted(Exception): pass
    def interrupt(record):
        assert 0 < record['active_candidates'] < 10
        raise Interrupted
    with pytest.raises(Interrupted):
        solve(tmp_path/'interrupted',source,traits,backend=backend,checkpoint=checkpoint,progress=interrupt)
    resumed = solve(tmp_path/'resumed',source,traits,backend=backend,checkpoint=checkpoint,resume=True)
    direct = solve(tmp_path/'direct',source,traits,backend=backend)
    for a,b in zip(resumed,direct):
        np.testing.assert_array_equal(a.weights,b.weights)
        np.testing.assert_array_equal(a.fixed_coefficients,b.fixed_coefficients)
        assert a.convergence['iterations'] == b.convergence['iterations']


def test_no_stale_certificate_can_export(tmp_path):
    from summit.prediction._validation import canonical
    from summit.prediction.solver import ConvergenceError
    source, traits = fixture(n=53,m=23)
    checkpoint = tmp_path/'checkpoint.npz'
    class Interrupted(Exception): pass
    def interrupt(record): raise Interrupted
    with pytest.raises(Interrupted):
        solve(tmp_path/'interrupted',source,traits,checkpoint=checkpoint,progress=interrupt)
    with np.load(checkpoint) as archive:
        data = {k:archive[k] for k in archive.files}
    meta = json.loads(data['metadata'].tobytes())
    # Forge structurally valid certificates, preserving array hashes. The final
    # fresh residual/fixed-point check must reject the unconverged weights.
    meta['active'] = [[] for _ in traits]
    meta['certified_sweeps'] = [[1]*len(t.candidates) for t in traits]
    data['metadata'] = np.frombuffer(canonical(meta).encode(),dtype=np.uint8)
    with checkpoint.open('wb') as stream:
        np.savez(stream,**data)
    with pytest.raises(ConvergenceError):
        solve(tmp_path/'bad',source,traits,checkpoint=checkpoint,resume=True)
    assert not (tmp_path/'bad').exists()


def test_whole_inactive_trait_and_easy_batch_finish(tmp_path):
    source, traits = fixture(n=53,m=23)
    traits[0] = replace(traits[0],candidates=traits[0].candidates[-1:])
    models = solve(tmp_path/'mixed',source,traits)
    null = next(m for m in models if m.trait_id==traits[0].id)
    assert null.convergence['iterations']==1
    np.testing.assert_array_equal(null.weights,0.)
    source,traits = fixture(n=31,m=3)
    traits = [replace(traits[0],candidates=traits[0].candidates[:1])]
    models = solve(tmp_path/'easy',source,traits)
    assert models[0].convergence['iterations'] < 5
    run = json.loads((tmp_path/'easy/manifest.json').read_text())['run_report']
    assert run['ledger']['traversals']['mixture_verification']==1


def test_resume_completed_certificate_still_rechecks(tmp_path):
    source, traits = fixture(n=31,m=3)
    traits = [replace(traits[0],candidates=traits[0].candidates[:1])]
    checkpoint = tmp_path/'checkpoint.npz'
    class Interrupted(Exception): pass
    def stop_after_certificate(record):
        if record['active_candidates']==0:
            raise Interrupted
    with pytest.raises(Interrupted):
        solve(tmp_path/'interrupted',source,traits,checkpoint=checkpoint,progress=stop_after_certificate)
    resumed = solve(tmp_path/'resumed',source,traits,checkpoint=checkpoint,resume=True)[0]
    direct = solve(tmp_path/'direct',source,traits)[0]
    np.testing.assert_array_equal(resumed.weights,direct.weights)
    np.testing.assert_array_equal(resumed.fixed_coefficients,direct.fixed_coefficients)
    run = json.loads((tmp_path/'resumed/manifest.json').read_text())['run_report']
    assert run['ledger']['traversals']['mixture_verification']==1
