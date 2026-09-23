from dataclasses import replace
import json
import numpy as np
import pytest

from prediction_helpers import prediction_threads
from test_prediction_core import fixture, dense
from summit.prediction.mixture import MixtureSpec, MixtureSolverSpec, fit_mixture_prediction
from summit.prediction.genotype import standardize


@pytest.mark.parametrize('q,independent', [(1,False),(5,False),(32,False),(5,True)])
def test_native_coordinate_columns_match_numpy_with_rounding_asymmetry(q,independent):
    from summit import gxeldcore as native
    from summit.prediction.runtime import configure_prediction_threads
    from summit.prediction.mixture import _numpy_update, _posterior
    threads=prediction_threads();configure_prediction_threads(native,threads)
    rng=np.random.default_rng(9922+q);b,k=7,3;d=b*q
    design=rng.normal(size=(d+20,d));gram=np.ascontiguousarray(design.T@design/(d+20))
    # Cross-block asymmetry must not be silently replaced by the other triangle.
    gram[0,-1]+=.01
    diagonal=np.array([gram[j*q:(j+1)*q,j*q:(j+1)*q] for j in range(b)])
    cov=np.repeat((np.eye(q)*.1)[None],b,axis=0)
    pc,ln=_posterior(diagonal,cov,MixtureSpec(.1,.2))
    pc=np.tile(pc.reshape(1,-1),(k,1));ln=np.tile(ln.reshape(1,-1),(k,1))
    score=rng.normal(size=(k,d));weights=rng.normal(scale=.01,size=(k,d));expected=weights.copy()
    penalty=np.empty((k,b));reference_penalty=np.empty_like(penalty);metrics=np.empty((k,2))
    reference_metrics=_numpy_update(gram,score,pc,ln,expected,reference_penalty,q,8,0.,independent)
    native.prediction_mixture_block(gram,score,pc,ln,weights,penalty,metrics,q,8,0.,independent,threads)
    np.testing.assert_allclose(weights,expected,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(penalty,reference_penalty,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(metrics,reference_metrics,rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize('rank', [0, 3])
def test_native_residual_owned_updates_reconstruction_and_restore(rank):
    from summit import gxeldcore as native
    from summit.prediction.runtime import configure_prediction_threads
    threads=prediction_threads(); configure_prediction_threads(native,threads)
    rng=np.random.default_rng(975)
    n,b,q,k=212581,5,3,2
    basis=np.asfortranarray(np.linalg.qr(rng.normal(size=(n,rank)))[0])
    yw=rng.normal(size=n); genotype=np.asfortranarray(rng.normal(size=(n,b)))
    phi=np.asfortranarray(rng.normal(size=(n,q)))
    w=np.empty((n,b*q),order='F')
    native.prediction_interaction_design(genotype,phi,w,threads)
    np.testing.assert_array_equal(w.reshape(n,b,q),genotype[:,:,None]*phi[:,None,:])
    projection=np.asfortranarray(basis.T@w)
    state=native.PredictionMixtureResidual(yw,basis,k,threads)
    yp=yw-basis@(basis.T@yw)
    expected=np.repeat(yp[:,None],k,axis=1)
    total=np.zeros((b*q,k),order='F')
    residual=np.empty((n,k),order='F')
    for _ in range(4):
        delta=np.asfortranarray(rng.normal(scale=.01,size=total.shape));total+=delta
        state.update(w,delta,projection)
        expected-=w@delta-basis@(projection@delta)
        state.copy_residual(residual)
        np.testing.assert_allclose(residual,expected,atol=1e-13,rtol=1e-12)
        score=np.empty((b*q,k),order='F');state.score(w,projection,score)
        np.testing.assert_allclose(score,w.T@expected-projection.T@(basis.T@expected),atol=2e-10,rtol=1e-10)
    snapshot=residual.copy(order='F');residual.fill(123.)
    state.copy_residual(residual)
    np.testing.assert_array_equal(residual,snapshot)
    other=native.PredictionMixtureResidual(yw,basis,k,threads);other.restore(residual)
    other.copy_residual(residual);np.testing.assert_array_equal(residual,snapshot)
    state.begin_reconstruction();state.add_prediction(w,total)
    assert state.finish_reconstruction()<1e-10
    state.copy_residual(residual);np.testing.assert_allclose(residual,expected,atol=1e-13,rtol=1e-12)
    with pytest.raises(RuntimeError,match='not started'):state.finish_reconstruction()
    with pytest.raises(RuntimeError,match='shape'):state.restore(np.empty((n,1),order='F'))


def test_failed_residual_guard_preserves_nonresumable_evidence(tmp_path,monkeypatch):
    from summit import gxeldcore as native
    from summit.prediction._validation import array_digest
    actual=native.PredictionMixtureResidual
    class CorruptResidual:
        def __init__(self,yw,basis,models,threads):
            self.state=actual(yw,basis,models,threads)
            self.shape=(len(yw),models);self.corrupted=False
        def __getattr__(self,name):return getattr(self.state,name)
        def update(self,*args):
            self.state.update(*args)
            if not self.corrupted:
                residual=np.empty(self.shape,order='F');self.state.copy_residual(residual)
                residual[0,0]+=.1;self.state.restore(residual);self.corrupted=True
    monkeypatch.setattr(native,'PredictionMixtureResidual',CorruptResidual)
    source,traits=fixture(n=43,m=17)
    t=replace(traits[0],candidates=traits[0].candidates[:1],geometry=None)
    with pytest.raises(FloatingPointError,match='residual drift'):
        fit_mixture_prediction([t],source,output=tmp_path/'models',checkpoint=tmp_path/'checkpoint.npz',
            mixtures={(t.id,t.candidates[0].id):MixtureSpec(.1,.2)},threads=prediction_threads(),block_size=8)
    assert not (tmp_path/'checkpoint.npz').exists()
    assert not (tmp_path/'models/COMPLETE.json').exists()
    with np.load(tmp_path/'checkpoint.arithmetic_failure.npz') as z:
        metadata=json.loads(z['metadata'].tobytes())
        assert metadata['passed'] is False and metadata['resumable'] is False
        assert metadata['drifts'][0]>.09
        for name,record in metadata['arrays'].items():assert array_digest(z[name])==record['sha256']
        np.testing.assert_allclose(np.linalg.norm(z['residual_0']-z['incremental_residual_0']),metadata['drifts'][0],atol=1e-10)


@pytest.mark.parametrize('coupling', [-.7,0.,.8])
def test_coupled_sparsity_preserves_marginals_and_covariance(coupling):
    from summit.prediction.mixture import SeparateSparsitySpec,mixture_from_dict
    a=MixtureSpec(.07,.2);u=MixtureSpec(.31,.1)
    spec=SeparateSparsitySpec(a,u,coupling)
    covariance=np.array([[[2.,.4,.1],[.4,1.,.2],[.1,.2,.5]]])
    p,c=spec.components(covariance)
    assert np.all(p>0)
    np.testing.assert_allclose(p.reshape(2,2).sum(1),a.probabilities,atol=1e-15)
    np.testing.assert_allclose(p.reshape(2,2).sum(0),u.probabilities,atol=1e-15)
    np.testing.assert_allclose(np.einsum('c,cbij->bij',p,c),covariance,atol=1e-15)
    from dataclasses import asdict
    assert mixture_from_dict(dict(kind='separate_sparsity',**asdict(spec)))==spec


def test_directional_shrinkage_preserves_amplification_and_basis_invariance():
    from summit.prediction import shrink_orthogonal_covariance
    covariance=np.array([[2.,.4,.1],[.4,1.,.2],[.1,.2,.5]])
    metric=np.array([[2.,.3],[.3,.7]])
    shrunk=shrink_orthogonal_covariance(covariance,[1.,.2],environment_metric=metric)
    np.testing.assert_array_equal(shrunk[0],covariance[0])
    assert np.linalg.eigvalsh(shrunk).min()>0
    assert np.linalg.eigvalsh(covariance-shrunk).min()>-1e-14
    transform=np.eye(3);transform[1:,1:]=[[2.,.3],[-.5,1.]]
    inverse=np.linalg.inv(transform[1:,1:])
    changed=shrink_orthogonal_covariance(transform@covariance@transform.T,[1.,.2],
        environment_metric=inverse.T@metric@inverse)
    np.testing.assert_allclose(changed,transform@shrunk@transform.T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(shrink_orthogonal_covariance(covariance,[1.,1.]),covariance)
    with pytest.raises(ValueError,match='tied'):
        shrink_orthogonal_covariance(np.eye(3),[1.,.2])


def fit(tmp_path, source, traits, mixture, backend='native', **kwargs):
    return fit_mixture_prediction(traits, source, output=tmp_path,
        mixtures={(t.id, c.id): mixture for t in traits for c in t.candidates},
        backend=backend, block_size=7, threads=prediction_threads(),
        solver=MixtureSolverSpec(rtol=1e-10, max_sweeps=150), **kwargs)


@pytest.mark.parametrize('backend', ['numpy', 'native'])
def test_gaussian_limit_dense_gls_singular_and_zero(tmp_path, backend):
    source, traits = fixture(n=41, m=23)
    models = fit(tmp_path/'models', source, traits, MixtureSpec(.5, .5), backend)
    for model in models:
        trait = next(t for t in traits if t.id == model.trait_id)
        candidate = next(c for c in trait.candidates if c.id == model.model_id)
        _, weights, mean, _ = dense(source, trait, candidate.covariance, candidate.residual)
        np.testing.assert_allclose(model.weights, weights, rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(trait.fixed@model.fixed_coefficients, mean, rtol=2e-8, atol=2e-8)
        assert model.convergence['method'] == 'mixture_vb_fixed_point'
        assert model.convergence['true_residual_norm'] <= model.convergence['threshold']


def test_native_numpy_mixture_rotation_and_storage(tmp_path):
    source, traits = fixture(n=47, m=19)
    trait = replace(traits[0], candidates=traits[0].candidates[:1], geometry=None)
    native = fit(tmp_path/'native', source, [trait], MixtureSpec(.07, .2), storage='compact')[0]
    reference = fit(tmp_path/'numpy', source, [trait], MixtureSpec(.07, .2), 'numpy')[0]
    np.testing.assert_allclose(native.weights, reference.weights, rtol=1e-8, atol=1e-10)
    rotation = np.eye(3)
    rotation[1:, 1:] = [[.8, -.6], [.6, .8]]
    candidate = replace(trait.candidates[0], covariance=rotation.T@trait.candidates[0].covariance@rotation)
    changed = replace(trait, phi=trait.phi@rotation, candidates=(candidate,))
    rotated = fit(tmp_path/'rotated', source, [changed], MixtureSpec(.07, .2))[0]
    np.testing.assert_allclose(rotated.weights@rotation.T, native.weights, rtol=2e-8, atol=1e-9)
    report = json.loads((tmp_path/'native/manifest.json').read_text())['run_report']
    assert report['ledger']['source_variants'] == len(trait.variants)
    history = np.array(report['objective_history'])
    assert np.min(np.diff(history[:, 0])) >= -1e-8


def test_mixture_checkpoint_resume_and_identity(tmp_path):
    source, traits = fixture(n=43, m=17)
    traits = [replace(traits[0], candidates=traits[0].candidates[:1])]
    class Interrupted(Exception):
        pass
    def interrupt(record):
        raise Interrupted()
    checkpoint = tmp_path/'state.npz'
    with pytest.raises(Interrupted):
        fit(tmp_path/'interrupted', source, traits, MixtureSpec(.1, .1),
            checkpoint=checkpoint, progress=interrupt)
    assert not (tmp_path/'interrupted').exists()
    resumed = fit(tmp_path/'resumed', source, traits, MixtureSpec(.1, .1), checkpoint=checkpoint, resume=True)
    complete = fit(tmp_path/'complete', source, traits, MixtureSpec(.1, .1))
    np.testing.assert_array_equal(resumed[0].weights, complete[0].weights)
    with pytest.raises(ValueError, match='identity mismatch'):
        fit(tmp_path/'bad', source, traits, MixtureSpec(.2, .1), checkpoint=checkpoint, resume=True)


def test_warm_mixture_checkpoint_can_resume(tmp_path):
    source,traits=fixture(n=43,m=17)
    traits=[replace(traits[0],candidates=traits[0].candidates[:1])]
    t=traits[0];warm={(t.id,t.candidates[0].id):np.full((len(t.variants),t.phi.shape[1]),.001)}
    class Interrupted(Exception): pass
    def interrupt(record): raise Interrupted()
    checkpoint=tmp_path/'warm.npz'
    with pytest.raises(Interrupted):
        fit(tmp_path/'interrupted',source,traits,MixtureSpec(.1,.1),initial_weights=warm,
            checkpoint=checkpoint,progress=interrupt)
    with np.load(checkpoint) as z:initial=json.loads(z['metadata'].tobytes())['initialization']
    assert len(initial)==64
    resumed=fit(tmp_path/'resumed',source,traits,MixtureSpec(.1,.1),checkpoint=checkpoint,resume=True)
    direct=fit(tmp_path/'direct',source,traits,MixtureSpec(.1,.1),initial_weights=warm)
    np.testing.assert_array_equal(resumed[0].weights,direct[0].weights)
    with np.load(checkpoint) as z:assert json.loads(z['metadata'].tobytes())['initialization']==initial


@pytest.mark.parametrize('backend', ['numpy', 'native'])
def test_one_snp_exact_quadrature(tmp_path, backend):
    from numpy.polynomial.hermite import hermgauss
    source, traits = fixture(n=41, m=5)
    trait = traits[0]
    # A single polymorphic SNP and two contexts. A one-site posterior is exact.
    j = 2
    from summit.prediction.genotype import estimate_scale
    scale = estimate_scale(source, trait.rows, np.array([j]), ddof=1)
    candidate = replace(trait.candidates[0], covariance=np.array([[.05, .01], [.01, .025]]))
    trait = replace(trait, variants=np.array([j]), scale=scale, phi=trait.phi[:, :2],
        candidates=(candidate,), context_spec={'names': ['intercept', 'e1']}, geometry=None)
    mixture = MixtureSpec(.2, .3)
    model = fit(tmp_path/'model', source, [trait], mixture, backend)[0]
    genotype = standardize(source.values[np.ix_(trait.rows, trait.variants)], scale.mean, scale.inverse_scale)
    noise = np.sqrt(candidate.residual)
    c = trait.fixed[:, :4]/noise[:, None]
    basis = np.linalg.qr(c)[0]
    w = genotype*trait.phi/noise[:, None]
    w -= basis@(basis.T@w)
    y = trait.y/noise
    y -= basis@(basis.T@y)
    nodes, weights = hermgauss(180)
    points = np.array(np.meshgrid(nodes, nodes, indexing='ij')).reshape(2, -1).T*np.sqrt(2)
    quadrature = np.outer(weights, weights).ravel()/np.pi
    all_beta, all_weights = [], []
    for p, s in zip(mixture.probabilities, mixture.scales):
        beta = points@np.linalg.cholesky(candidate.covariance).T*np.sqrt(s)
        likelihood = np.exp(-.5*np.sum((y[:, None]-w@beta.T)**2, axis=0))
        all_beta.append(beta); all_weights.append(p*quadrature*likelihood)
    weights = np.concatenate(all_weights)
    expected = weights@np.vstack(all_beta)/weights.sum()
    np.testing.assert_allclose(model.weights[0], expected, rtol=2e-6, atol=2e-8)


def test_annotation_mixture_gaussian_limit(tmp_path):
    from test_prediction_annotations import annotated_fixture, dense_annotated
    source, trait = annotated_fixture()
    trait = replace(trait, candidates=trait.candidates[1:])
    models = fit(tmp_path/'models', source, [trait], MixtureSpec(.5, .5))
    for model, candidate in zip(models, trait.candidates):
        _, expected, _, _ = dense_annotated(source, trait, candidate)
        np.testing.assert_allclose(model.weights, expected, rtol=2e-7, atol=2e-9)


def test_reject_invalid_mixture_parameters():
    for p, f in [(0, .1), (1, .1), (.1, -1), (.1, np.nan)]:
        with pytest.raises(ValueError):
            MixtureSpec(p, f)


def test_no_fixed_covariates(tmp_path):
    source, traits = fixture(n=43,m=17)
    t = traits[0]
    t = replace(t, fixed=np.empty((len(t.rows),0)), fixed_spec={'names': []},
                candidates=t.candidates[:1], geometry=None)
    actual = fit(tmp_path/'native',source,[t],MixtureSpec(.1,.2))[0]
    expected = fit(tmp_path/'reference',source,[t],MixtureSpec(.1,.2),'numpy')[0]
    np.testing.assert_allclose(actual.weights,expected.weights,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('ddof',[0,1])
def test_large_hardcall_affines_missing_rare_and_monomorphic(ddof):
    from summit.prediction.genotype import native_module
    native = native_module()
    rng = np.random.default_rng(1701); n=291273
    raw = np.asfortranarray(rng.integers(0,3,size=(n,9)),dtype=np.int8)
    raw[:,0]=0;raw[:,1]=1;raw[:,2]=2;raw[:,3]=-127
    raw[:,4]=2;raw[:3,4]=1
    raw[:,5]=0;raw[:3,5]=1
    raw[::7,6:]=-127
    means=np.empty(9);inverse=np.empty(9)
    native.prediction_hardcall_scale(raw,means,inverse,ddof,prediction_threads())
    observed=raw!=-127;calls=np.where(observed,raw,0).astype(float)
    counts=observed.sum(axis=0)
    mu=np.divide(calls.sum(axis=0),counts,out=np.zeros(9),where=counts>0)
    centered=np.where(observed,calls-mu,0)
    ss=np.einsum('ij,ij->j',centered,centered)
    inv=np.sqrt(np.divide(n-ddof,ss,out=np.ones(9),where=ss>0))
    np.testing.assert_array_equal(means,mu)
    np.testing.assert_allclose(inverse,inv,rtol=1e-10,atol=0)
    bad=raw.copy(order='F');bad[0,0]=3
    with pytest.raises(RuntimeError,match='Invalid genotype code'):
        native.prediction_hardcall_scale(bad,means,inverse,ddof,prediction_threads())


def test_scale_array_fallback_without_native(monkeypatch):
    from summit.prediction import genotype
    source,traits=fixture(n=71,m=19)
    def unavailable():
        raise ImportError('optional native module unavailable')
    monkeypatch.setattr(genotype,'native_module',unavailable)
    t=traits[0]
    scale=genotype.estimate_scale(source,t.rows,t.variants,ddof=t.scale.ddof)
    np.testing.assert_array_equal(scale.mean,t.scale.mean)
    np.testing.assert_allclose(scale.inverse_scale,t.scale.inverse_scale,rtol=1e-12,atol=0)


def test_separate_sparsity_preserves_covariance_and_zero_baseline():
    from summit.prediction.mixture import SeparateSparsitySpec,mixture_from_dict
    spec=SeparateSparsitySpec(MixtureSpec(.02,.1),MixtureSpec(.3,.2))
    covariance=np.array([[[.3,.04],[.04,.02]],[[0,0],[0,.1]],[[0,0],[0,0]]])
    probability,components=spec.components(covariance)
    np.testing.assert_allclose(np.einsum('c,cbij->bij',probability,components),covariance,atol=1e-15)
    assert np.linalg.eigvalsh(components).min()>-1e-14
    assert mixture_from_dict(dict(kind='separate_sparsity',baseline=dict(probability=.02,small_variance_fraction=.1),
        response=dict(probability=.3,small_variance_fraction=.2)))==spec


def test_separate_sparsity_pure_amplification_spikes():
    from summit.prediction.mixture import SeparateSparsitySpec, _posterior
    anchor=np.array([1.,.173,-.289])
    covariance=(.37*np.outer(anchor,anchor))[None]
    spec=SeparateSparsitySpec(MixtureSpec(.07,0),MixtureSpec(.2,0))
    probabilities,components=spec.components(covariance)
    np.testing.assert_allclose(np.einsum('c,cbij->bij',probabilities,components),covariance,atol=1e-15)
    posterior,normalizer=_posterior(np.eye(3)[None],covariance,spec)
    assert np.isfinite(posterior).all() and np.isfinite(normalizer).all()


@pytest.mark.parametrize('backend',['numpy','native'])
def test_separate_sparsity_single_snp_exact_posterior(tmp_path,backend):
    from summit.prediction.mixture import SeparateSparsitySpec
    from summit.prediction.genotype import estimate_scale
    from scipy.special import logsumexp
    source,traits=fixture(n=43,m=17)
    t=traits[0];variants=np.array([2]);scale=estimate_scale(source,t.rows,variants)
    covariance=np.array([[.3,.04],[.04,.02]])
    c=replace(t.candidates[0],covariance=covariance)
    t=replace(t,variants=variants,scale=scale,phi=t.phi[:,:2],candidates=(c,),
        context_spec={'names':['intercept','e1']},geometry=None)
    spec=SeparateSparsitySpec(MixtureSpec(.1,.2),MixtureSpec(.3,.1))
    model=fit(tmp_path/'fit',source,[t],spec,backend)[0]
    w=standardize(source.values[np.ix_(t.rows,variants)],scale.mean,scale.inverse_scale)*t.phi/np.sqrt(c.residual)[:,None]
    fixed=t.fixed/np.sqrt(c.residual)[:,None];u=np.linalg.svd(fixed,full_matrices=False)[0][:,:np.linalg.matrix_rank(fixed)]
    w-=u@(u.T@w);y=t.y/np.sqrt(c.residual);y-=u@(u.T@y)
    probability,components=spec.components(covariance[None]);g=w.T@w;s=w.T@y
    posterior=[];logweights=[]
    for p,v in zip(probability,components[:,0]):
        variance=np.linalg.inv(np.linalg.inv(v)+g);posterior.append(variance@s)
        logweights.append(np.log(p)-.5*np.linalg.slogdet(np.eye(2)+v@g)[1]+.5*s@variance@s)
    expected=np.exp(logweights-logsumexp(logweights))@np.array(posterior)
    np.testing.assert_allclose(model.weights[0],expected,rtol=1e-8,atol=1e-10)


def test_separate_sparsity_rotation_and_mixed_menu(tmp_path):
    from summit.prediction.mixture import SeparateSparsitySpec
    source,traits=fixture(n=47,m=19);t=traits[0]
    t=replace(t,candidates=t.candidates[:1],geometry=None)
    spec=SeparateSparsitySpec(MixtureSpec(.07,.2),MixtureSpec(.25,.1))
    original=fit(tmp_path/'original',source,[t],spec)[0]
    rotation=np.eye(3);rotation[1:,1:]=[[.8,-.6],[.6,.8]]
    c=replace(t.candidates[0],covariance=rotation.T@t.candidates[0].covariance@rotation)
    rotated=fit(tmp_path/'rotated',source,[replace(t,phi=t.phi@rotation,candidates=(c,))],spec)[0]
    np.testing.assert_allclose(rotated.weights@rotation.T,original.weights,rtol=2e-8,atol=1e-9)
    c=replace(t.candidates[0],id='radial');both=replace(t,candidates=(t.candidates[0],c))
    models=fit_mixture_prediction([both],source,output=tmp_path/'menu',
        mixtures={(t.id,t.candidates[0].id):spec,(t.id,'radial'):MixtureSpec(.5,.5)},
        threads=prediction_threads(),block_size=7,solver=MixtureSolverSpec(rtol=1e-10))
    np.testing.assert_allclose(models[0].weights,original.weights,rtol=1e-8,atol=1e-10)
    warm=fit(tmp_path/'warm',source,[t],spec,initial_weights={original.key:original.weights})[0]
    np.testing.assert_allclose(warm.weights,original.weights,rtol=1e-7,atol=1e-9)


def test_full_sample_projection_and_reconstruction_normal_equations(tmp_path):
    # Exercise biobank-length vectors without a dense participant covariance.
    # The independent seven-parameter Gaussian normal equations include a
    # nonzero fixed mean, heteroscedasticity and rank-deficient covariates.
    from scipy.linalg import qr
    source,traits=fixture(n=212581,m=7)
    t=traits[0];c=replace(t.candidates[0],covariance=np.array([[.3]]))
    t=replace(t,phi=t.phi[:,:1],candidates=(c,),context_spec={'names':['intercept']},geometry=None)
    model=fit_mixture_prediction([t],source,output=tmp_path/'large',
        mixtures={(t.id,c.id):MixtureSpec(.5,.5)},storage='compact',block_size=7,
        threads=prediction_threads(),solver=MixtureSolverSpec(rtol=1e-9,residual_refresh=1))[0]
    scale=np.sqrt(c.residual);fixed=t.fixed[:,:4]/scale[:,None]
    basis=qr(fixed,mode='economic')[0]
    w=standardize(source.values[np.ix_(t.rows,t.variants)],t.scale.mean,t.scale.inverse_scale)/scale[:,None]
    y=t.y/scale
    w-=np.einsum('nr,rj->nj',basis,np.einsum('nr,nj->rj',basis,w,optimize=False),optimize=False)
    y-=np.einsum('nr,r->n',basis,np.einsum('nr,n->r',basis,y,optimize=False),optimize=False)
    gram=np.einsum('ni,nj->ij',w,w,optimize=False)
    rhs=np.einsum('ni,n->i',w,y,optimize=False)
    expected=np.linalg.solve(gram+np.eye(len(t.variants))*len(t.variants)/.3,rhs)
    np.testing.assert_allclose(model.weights[:,0],expected,rtol=1e-7,atol=1e-10)
    assert model.convergence['residual_reconstruction_error']<1e-9
