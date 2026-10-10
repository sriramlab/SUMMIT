"""Joint pair/group conditional checks against independent dense linear algebra."""
import numpy as np
import pytest

from prediction_helpers import prediction_threads
from test_epistasis_polygenic_operator import fixture


def panel_reference(source, rows, mode):
    """Use the public feature builder; genotype-only cohort scaling is fixed."""
    from summit.epistasis.prepare import SelectedStudy, fit_scale
    from summit.epistasis.features import prepare_feature_reference
    m = len(source.variants.ids)
    annotation = {
        'all': np.ones(m),
        'a': np.array([1., .5, 1., 0., 0., 0.]),
        'b': np.array([0., 0., .8, 1., .7, 0.]),
    }
    scale = fit_scale(source, rows, threads=prediction_threads())
    study = SelectedStudy(source, rows, scale, fixed_effects=np.ones((len(rows),1)),
        modifiers=np.ones((len(rows),1)), weights=np.ones((m,1)),
        component_names=['additive'], definitions={'additive_annotations':['all'],'interactions':[]},
        threads=prediction_threads(), memory_bytes=2**30)
    if mode == 'pairs':
        job = dict(pairs=[['v0','v1'],['v0','v2']], additive_annotations=['all'])
    elif mode == 'overlap':
        job = dict(groups=[dict(name='within_a',mode='within',left='a'),
            dict(name='cross_ab',mode='cross',left='a',right='b')],additive_annotations=['all'])
    else:
        group=dict(name=mode,mode=mode,left='a')
        if mode == 'cross':
            group['right']='b'
        job=dict(groups=[group],additive_annotations=['all'])
    reference=prepare_feature_reference(study,job,annotation,
        main_effects='tested_variants',dominance='tested_variants')
    return reference


@pytest.mark.parametrize('mode',['pairs','within','cross','overlap','remainder'])
def test_native_conditional_panels_match_dense_and_portable_joint_fit(mode, tmp_path):
    from summit.prediction.genotype import ArrayGenotypeSource
    from summit.epistasis.polygenic import conditional_score, he_geometry, estimate_components
    from summit.epistasis.conditional import conditional_mean_summary
    from summit.epistasis.robust import write_robust_scores,load_robust_scores,robust_score_tests
    from scripts.epistasis.conditional_polygenic_reference import (
        conditional_null, conditional_mean_tangents)
    rng,make,kernels,i0,i1,contexts=fixture()
    training=make(i0,'packed')
    complete=make(np.arange(len(contexts)))
    # Six supplied panel SNPs, with a distinct 73-SNP covariance background.
    owner=make(np.arange(len(contexts))).stream.source
    owner.prepare(np.arange(len(contexts)),8,prediction_threads())
    raw=owner.read(np.arange(6))
    source=ArrayGenotypeSource(raw,owner.samples,owner.variants.subset(np.arange(6)),hard_calls=True)
    ref=panel_reference(source,np.arange(len(contexts)),mode)
    c=np.column_stack([ref.fixed_effects,contexts[:,1],contexts[:,1]**2])
    f=ref.features
    generating=np.array([.8,.3,.4,.7,.2])
    v=np.einsum('k,kij->ij',generating,kernels)
    y=np.linalg.cholesky(v)@rng.normal(size=len(c))
    # Outcome-dependent null fitting is rerun, not replaced by generating truth.
    theta=estimate_components(training,y[i0],he_geometry(training,c[i0],exact=True))[:,0]
    fit=conditional_score(training,complete,i0,i1,y[i0],y[i1],f[i0],f[i1],
        c[i0],c[i1],theta,mean_tangents=True)
    order=np.r_[i0,i1];kk=kernels[:,order][:,:,order]
    fitted_v=np.einsum('k,kij->ij',theta,kk)
    transfer,q=conditional_null(fitted_v,c[order],len(i0))
    tangent=conditional_mean_tangents(fitted_v,kk,c[order],len(i0),y[i0])
    nuisance=np.column_stack([c[i1],tangent])
    # Independent dense projection and generalized inverse, no native solver.
    residualizer=np.eye(len(i1))-nuisance@np.linalg.pinv(nuisance,rcond=1e-11)
    d=residualizer@(f[i1]-transfer@f[i0])
    scale=np.linalg.norm(d,axis=0)
    scale[scale==0]=1.
    a=np.linalg.pinv(d/scale,rcond=1e-11)/scale[:,None]
    expected_beta=a@(y[i1]-transfer@y[i0])
    expected_covariance=a@q@a.T
    np.testing.assert_allclose(fit['beta'],expected_beta,atol=2e-7,rtol=2e-6)
    np.testing.assert_allclose(fit['covariance'],expected_covariance,atol=2e-7,rtol=2e-6)
    np.testing.assert_allclose(fit['contrast']@tangent,0,atol=2e-7)
    np.testing.assert_allclose(fit['information'],d.T@d,atol=2e-6,rtol=2e-6)
    assert fit['diagnostics']['outside_confirmation_design']  # Bounded algebra check.
    if mode == 'overlap':
        assert fit['diagnostics']['feature_rank'] < f.shape[1]
    identity={key: str(j)*64 for j,key in enumerate(
        ['genotype_reference','direction','training_outcomes','confirmation_outcomes','null_fit'])}
    summary=conditional_mean_summary(fit['beta'],fit['covariance'],fit['information'],
        feature_names=ref.metadata['feature_names'],trait_name='y',trait_unit='cm',
        identities=identity,diagnostics=fit['diagnostics'])
    write_robust_scores(summary,tmp_path/'panel.npz')
    test=robust_score_tests(load_robust_scores(tmp_path/'panel.npz'))
    expected=expected_beta@np.linalg.pinv(expected_covariance,rcond=1e-11)@expected_beta
    np.testing.assert_allclose(test['joint_wald'],expected,atol=2e-6,rtol=2e-6)
    assert test['joint_df']==fit['diagnostics']['feature_rank']


def test_joint_single_column_matches_scalar_tangent_workflow():
    from summit.epistasis.polygenic import conditional_score, conditional_scores_batch
    rng,make,kernels,i0,i1,c=fixture()
    theta=np.array([.8,.3,.4,.7,.2])
    y=rng.normal(size=len(c)); f=rng.normal(size=len(c))
    extra=rng.normal(size=(len(i1),2))
    joint=conditional_score(make(i0),make(np.arange(len(c))),i0,i1,
        y[i0],y[i1],f[i0],f[i1],c[i0],np.column_stack([c[i1],extra]),theta,mean_tangents=True)
    scalar=conditional_scores_batch(make(i0),make(np.arange(len(c))),i0,i1,
        y[i0,None],y[i1,None],f[i0,None],f[i1,None],c[i0],c[i1],[extra],
        theta[:,None],mean_tangents=True)
    np.testing.assert_allclose(joint['beta'],scalar['beta'],rtol=2e-6,atol=3e-8)
    np.testing.assert_allclose(joint['covariance'][0,0],scalar['variance'][0],rtol=2e-6,atol=3e-8)
    np.testing.assert_allclose(joint['contrast'][0],scalar['contrasts'][:,0],rtol=2e-6,atol=3e-8)
    with pytest.raises(ValueError,match='separation'):
        conditional_score(make(i0),make(np.arange(len(c))),i0,np.r_[i0[0],i1[1:]],
            y[i0],y[i1],f[i0],f[i1],c[i0],c[i1],theta)


def test_training_absorbed_products_keep_confirmation_information():
    from summit.epistasis.polygenic import conditional_score
    from scripts.epistasis.conditional_polygenic_reference import conditional_null, innovation_score
    rng,make,kernels,i0,i1,c=fixture()
    theta=np.array([.8,.3,.4,.7,.2])
    y=rng.normal(size=len(c))
    # Perfect training LD can put products in the local A/H mean span.
    # Confirmation can still identify their deviation from that training span.
    f0=c[i0].copy()
    f1=rng.normal(size=(len(i1),2))
    fit=conditional_score(make(i0),make(np.arange(len(c))),i0,i1,
        y[i0],y[i1],f0,f1,c[i0],c[i1],theta)
    order=np.r_[i0,i1]
    v=np.einsum('k,kij->ij',theta,kernels[:,order][:,:,order])
    transfer,q=conditional_null(v,c[order],len(i0))
    expected=innovation_score(y[i0],y[i1],f0,f1,c[i1],transfer,q)
    np.testing.assert_allclose(fit['beta'],expected['beta'],atol=3e-8,rtol=2e-6)
    np.testing.assert_allclose(fit['covariance'],expected['covariance'],atol=3e-8,rtol=2e-6)
    assert fit['rhs_spans'][0]['solved_rank']==1
    assert fit['rhs_spans'][0]['rhs_columns']==3
    assert fit['diagnostics']['feature_rank']==2


def test_joint_memory_budget_with_partitioned_operator_caps():
    from summit.epistasis.polygenic import conditional_score, conditional_score_memory_plan
    from scripts.epistasis.conditional_polygenic_reference import conditional_null, innovation_score
    rng, make, kernels, i0, i1, c = fixture()
    training, complete = make(i0, 'packed'), make(np.arange(len(c)))
    y = rng.normal(size=len(c)); f = rng.normal(size=(len(c), 2))
    theta = np.array([.8, .3, .4, .7, .2])
    plan = conditional_score_memory_plan(training, complete, feature_count=2,
        training_fixed_count=c.shape[1], confirmation_fixed_count=c.shape[1])
    # The caller reserves the other operator before assigning each local cap.
    # Both operators fit in the shared allocation, but neither local cap is an
    # aggregate allowance. This reproduces the full-cohort driver's failure.
    budget = plan['total_bytes']+2**20
    training.memory_bytes = budget-complete.base_bytes
    complete.memory_bytes = budget-training.base_bytes
    caps = training.memory_bytes, complete.memory_bytes
    args = (training, complete, i0, i1, y[i0], y[i1], f[i0], f[i1], c[i0], c[i1], theta)
    calls = training.stream.ledger.operator_calls, complete.stream.ledger.operator_calls
    with pytest.raises(MemoryError, match='aggregate budget'):
        conditional_score(*args)
    with pytest.raises(MemoryError, match='aggregate budget'):
        conditional_score(*args, memory_bytes=plan['total_bytes']-1)
    assert calls == (training.stream.ledger.operator_calls, complete.stream.ledger.operator_calls)
    actual = conditional_score(*args, memory_bytes=budget)
    assert caps == (training.memory_bytes, complete.memory_bytes)
    assert actual['memory_plan'] == plan
    assert actual['memory_budget_bytes'] == budget
    order = np.r_[i0, i1]
    v = np.einsum('k,kij->ij', theta, kernels[:, order][:, :, order])
    transfer, q = conditional_null(v, c[order], len(i0))
    expected = innovation_score(y[i0], y[i1], f[i0], f[i1], c[i1], transfer, q)
    np.testing.assert_allclose(actual['beta'], expected['beta'], atol=3e-8, rtol=2e-6)
    np.testing.assert_allclose(actual['covariance'], expected['covariance'], atol=3e-8, rtol=2e-6)
    # An explicit aggregate allowance must not disable local product admission.
    training.memory_bytes = training.base_bytes
    with pytest.raises(MemoryError, match='memory'):
        conditional_score(*args, memory_bytes=budget)
    for invalid in (0, -1, True, float(budget)):
        with pytest.raises(ValueError, match='integer aggregate'):
            conditional_score(*args, memory_bytes=invalid)


def test_joint_memory_preflight_includes_tangent_nuisance_and_product_workspace():
    from types import SimpleNamespace
    from summit.epistasis.polygenic import conditional_score_memory_plan
    # Shape-only preflight runs before genotype loading or expensive HE fits.
    def operator(n, base):
        return SimpleNamespace(rows=range(n), count=5, base_bytes=base,
            contexts=SimpleNamespace(shape=(n, 2)))
    training, complete = operator(63784, 9*2**30), operator(319132, 3*2**30)
    plain = conditional_score_memory_plan(training, complete, feature_count=4,
        training_fixed_count=1590, confirmation_fixed_count=1590)
    tangent = conditional_score_memory_plan(training, complete, feature_count=4,
        training_fixed_count=1590, confirmation_fixed_count=1590, mean_tangents=True)
    assert tangent['total_bytes'] > plain['total_bytes']
    assert tangent['total_bytes'] < 48*2**30
    assert tangent['product_workspace_bytes'] > 0
    for key in ('feature_count', 'training_fixed_count', 'confirmation_fixed_count'):
        for invalid in (0, -1, True, 2.5):
            counts = dict(feature_count=4, training_fixed_count=1590, confirmation_fixed_count=1590)
            counts[key] = invalid
            with pytest.raises(ValueError, match='counts'):
                conditional_score_memory_plan(training, complete, **counts)

def test_redundant_panel_never_reports_nonidentifiable_standard_errors():
    from summit.epistasis.robust import RobustScoreSummary, robust_score_tests
    basis=np.array([[1.,0.],[0.,1.],[1.,1.]])
    information=basis@basis.T
    direction=information@np.array([0.,1.,1.])
    summary=RobustScoreSummary(direction[:,None],information,
        np.outer(direction,direction)[None],('a','b','a_plus_b'),('y',),dict(method='fixture',inference='Gaussian'))
    with np.errstate(invalid='raise'):
        fit=robust_score_tests(summary,contrasts={'identified':[0.,1.,1.]})
    assert not np.any(fit['estimable_coefficients'])
    assert np.all(np.isnan(fit['standard_errors']))
    assert fit['coefficient_contrasts'][0]['p'] is not None

def test_joint_panel_rescaling_preserves_test_and_original_effect_units():
    from summit.epistasis.polygenic import conditional_score
    rng,make,kernels,i0,i1,c=fixture()
    theta=np.array([.8,.3,.4,.7,.2])
    y=rng.normal(size=len(c));f=rng.normal(size=(len(c),2))
    def fit(features):
        return conditional_score(make(i0),make(np.arange(len(c))),i0,i1,
            y[i0],y[i1],features[i0],features[i1],c[i0],c[i1],theta,mean_tangents=True)
    base=fit(f);units=np.array([1e-7,-1e5]);other=fit(f*units)
    np.testing.assert_allclose(other['beta']*units,base['beta'],atol=3e-8,rtol=2e-6)
    np.testing.assert_allclose(other['covariance']*units[:,None]*units[None,:],
        base['covariance'],atol=3e-8,rtol=2e-6)
    np.testing.assert_allclose(other['diagnostics']['information_condition'],
        base['diagnostics']['information_condition'],atol=3e-8,rtol=2e-6)

def test_distinct_nuisance_spans_restricted_gaussian_identity_and_invariance():
    from scipy.linalg import null_space
    from summit.epistasis.polygenic import conditional_score
    rng,make,kernels,i0,i1,c=fixture()
    theta=np.array([.8,.3,.4,.7,.2])
    y=rng.normal(size=len(c));f=rng.normal(size=(len(c),2))
    c0=np.column_stack([c[i0],np.zeros(len(i0))])
    c1=np.column_stack([c[i1],rng.normal(size=len(i1))])
    def fit(yy0,yy1):
        return conditional_score(make(i0),make(np.arange(len(c))),i0,i1,
            yy0,yy1,f[i0],f[i1],c0,c1,theta)
    actual=fit(y[i0],y[i1])
    order=np.r_[i0,i1];v=np.einsum('k,kij->ij',theta,kernels[:,order][:,:,order]);n0=len(i0)
    q0=null_space(c0.T);q1=null_space(c1.T)
    reduced=q0.T@v[:n0,:n0]@q0
    transfer=q1.T@v[n0:,:n0]@q0@np.linalg.inv(reduced)
    d=q1.T@f[i1]-transfer@q0.T@f[i0]
    aa=np.linalg.pinv(d)
    residual=q1.T@y[i1]-transfer@q0.T@y[i0]
    joint=np.column_stack([-transfer@q0.T,q1.T])
    # This is the independence identity in restricted coordinates, including
    # a genuinely new confirmation nuisance direction.
    np.testing.assert_allclose(joint@v[:,:n0]@q0,0,atol=1e-12)
    np.testing.assert_allclose(actual['beta'],aa@residual,rtol=2e-6,atol=3e-8)
    np.testing.assert_allclose(actual['covariance'],aa@joint@v@joint.T@aa.T,rtol=2e-6,atol=3e-8)
    shifted=fit(y[i0]+c0@np.array([3.,-2.,9.]),y[i1]+c1@np.array([-4.,2.,7.]))
    np.testing.assert_allclose(shifted['beta'],actual['beta'],rtol=2e-6,atol=3e-8)
