"""Independent bounded checks before any conditional-polygenic integration."""
import numpy as np


def test_training_conditioning_covariance_and_actual_response():
    from scripts.epistasis.conditional_polygenic_reference import conditional_null, innovation_score
    rng=np.random.default_rng(69173)
    n0,n1,m=32,48,70
    g=rng.normal(size=(n0+n1,m))
    x=g[:,0]
    c=np.column_stack([np.ones(len(g)),x])
    v=.7*g@g.T/m+np.diag(.4+.6*x*x)
    transfer,covariance=conditional_null(v,c,n0)
    joint_contrast=np.column_stack([-transfer,np.eye(n1)])
    np.testing.assert_allclose(joint_contrast@c,0,atol=1e-12)
    np.testing.assert_allclose(covariance,joint_contrast@v@joint_contrast.T,atol=1e-12)
    # Independent block-regression identity, including nuisance estimation.
    whitened=np.linalg.solve(v[:n0,:n0],c[:n0])
    b=np.linalg.inv(c[:n0].T@whitened)
    p=np.linalg.inv(v[:n0,:n0])-whitened@b@whitened.T
    expected=v[n0:,:n0]@p+c[n0:]@b@whitened.T
    np.testing.assert_allclose(transfer,expected,atol=1e-12)
    p0=np.eye(n0)-c[:n0]@np.linalg.pinv(c[:n0])
    np.testing.assert_allclose(joint_contrast@v[:,:n0]@p0,0,atol=1e-12)
    f=x*g[:,1]
    fit=innovation_score(.3*f[:n0],.3*f[n0:],f[:n0],f[n0:],c[n0:],transfer,covariance)
    np.testing.assert_allclose(fit['beta'],.3,atol=1e-12)
    np.testing.assert_allclose(fit['contrast']@(f[n0:]-transfer@f[:n0]),1,atol=1e-12)
    naive=np.sum((f[n0:]-c[n0:]@np.linalg.lstsq(c[n0:],f[n0:],rcond=None)[0])**2)
    assert not np.isclose(naive,np.sum(fit['response']**2),rtol=.001)
    # The intended scalable implementation needs products and a projected
    # training solve, not a stored conditional covariance matrix.
    features=np.column_stack([f,x*g[:,2]])
    beta=np.array([.3,-.2])
    joint=innovation_score(features[:n0]@beta,features[n0:]@beta,
        features[:n0],features[n0:],c[n0:],transfer,covariance)
    np.testing.assert_allclose(joint['beta'],beta,atol=1e-12)
    a=joint['contrast'].T
    rhs=v[:n0,n0:]@a
    from_products=a.T@v[n0:,n0:]@a-rhs.T@p@rhs
    np.testing.assert_allclose(joint['covariance'],from_products,atol=1e-12)
    np.testing.assert_allclose(a.T@(features[n0:]-v[n0:,:n0]@p@features[:n0]),np.eye(2),atol=1e-12)


def test_estimated_covariance_uses_outcomes_and_kernel_scale():
    from scripts.epistasis.conditional_polygenic_reference import fit_covariance
    rng=np.random.default_rng(71957)
    n,m=96,48
    g=rng.normal(size=(n,m)); x=g[:,0]
    c=np.column_stack([np.ones(n),x])
    kernels=np.stack([g@g.T/m,np.eye(n),np.diag(x*x)])
    y=g@rng.normal(size=m)/np.sqrt(m)+rng.normal(size=n)*np.sqrt(.4+.6*x*x)
    theta,record=fit_covariance(y,c,kernels)
    scaled,_=fit_covariance(3*y+2*x,c,kernels)
    np.testing.assert_allclose(scaled,9*theta,atol=1e-10,rtol=1e-8)
    recoded,_=fit_covariance(y,c,kernels*np.array([7,.5,3])[:,None,None])
    np.testing.assert_allclose(recoded*np.array([7,.5,3]),theta,atol=1e-10,rtol=1e-8)
    reml,_=fit_covariance(y,c,kernels,method='reml')
    assert np.all(reml>0) and record['moment_condition']<1e8


def test_adaptive_direction_known_covariance_null():
    from scipy.stats import chi2
    from scripts.epistasis.conditional_polygenic_reference import conditional_null, innovation_score
    rng=np.random.default_rng(81397)
    n0,n1,m=16,32,48
    g=rng.normal(size=(n0+n1,m)); x=g[:,0]
    c=np.column_stack([np.ones(n0+n1),x])
    v=.8*g@g.T/m+np.eye(n0+n1)
    transfer,covariance=conditional_null(v,c,n0)
    p=np.eye(n0)-c[:n0]@np.linalg.pinv(c[:n0])
    f=x[:,None]*g[:,1:]
    kernel=f@f[:n0].T/(m-1)
    draws=np.linalg.cholesky(v)@rng.normal(size=(n0+n1,1000))
    hits=0
    for y in draws.T:
        # Deliberately adaptive, using only fixed-mean-free training contrasts.
        direction=kernel@(p@y[:n0])
        fit=innovation_score(y[:n0],y[n0:],direction[:n0],direction[n0:],c[n0:],transfer,covariance)
        statistic=fit['beta']@np.linalg.solve(fit['covariance'],fit['beta'])
        hits+=statistic>chi2.ppf(.95,1)
    assert 25<=hits<=80


def test_genotype_weighted_covariance_moments_and_units():
    from scripts.epistasis.conditional_polygenic_reference import covariance_geometry, fit_prepared_covariance, fit_covariance
    rng=np.random.default_rng(194753)
    n,m=96,80
    g=rng.normal(size=(n,m));x=rng.normal(size=n)
    x[0]=9.
    c=np.column_stack([np.ones(n),x])
    k=np.stack([g@g.T/m,(x[:,None]*g)@(x[:,None]*g).T/m,np.eye(n),np.diag(x*x)])
    y=rng.normal(size=n)*np.sqrt(.4+.6*x*x)
    geometry=covariance_geometry(c,k,diagonal_preconditioning=True)
    theta,_=fit_prepared_covariance(y,geometry)
    diagonal=np.stack([np.diag(v) for v in k])
    w=1/np.sqrt((diagonal/diagonal.mean(1)[:,None]).mean(0))
    expected,_=fit_covariance(w*y,w[:,None]*c,k*w[None,:,None]*w[None,None,:])
    np.testing.assert_allclose(theta,expected,atol=1e-12)
    changed=covariance_geometry(c,k*np.array([.1,7,2,4])[:,None,None],diagonal_preconditioning=True)
    rescaled,_=fit_prepared_covariance(3*y+2*x,changed)
    np.testing.assert_allclose(rescaled*np.array([.1,7,2,4]),9*theta,atol=1e-10,rtol=1e-8)


def test_conditional_mean_tangents_independent_derivative_and_joint_response():
    from scipy.linalg import null_space
    from scripts.epistasis.conditional_polygenic_reference import (
        conditional_null, conditional_mean_tangents, innovation_score,
    )
    rng=np.random.default_rng(978251)
    n0,n1,m=28,54,47
    g=rng.normal(size=(n0+n1,m)); x=g[:,0]
    c=np.column_stack([np.ones(len(g)),x,x])  # Same span, deficient basis.
    kernels=np.stack([g@g.T/m,(g*x[:,None])@(g*x[:,None]).T/m,
        np.eye(len(g)),np.diag(.3+x*x)])
    theta=np.array([.8,.4,.6,.3])
    v=np.einsum('k,kij->ij',theta,kernels)
    y0=rng.normal(size=n0)
    tangent=conditional_mean_tangents(v,kernels,c,n0,y0)
    # Independent restricted-coordinate inverse, not the implementation's
    # subtraction of weighted fixed-effect projection from the inverse.
    q=null_space(c[:n0].T)
    p=q@np.linalg.solve(q.T@v[:n0,:n0]@q,q.T)
    expected=np.column_stack([(k[n0:,:n0]-v[n0:,:n0]@p@k[:n0,:n0])@p@y0
        for k in kernels])
    np.testing.assert_allclose(tangent,expected,atol=3e-13)
    q1=null_space(c[n0:].T)
    for j,k in enumerate(kernels):
        step=1e-5
        lp,_=conditional_null(v+step*k,c,n0)
        lm,_=conditional_null(v-step*k,c,n0)
        derivative=(lp-lm)@y0/(2*step)
        np.testing.assert_allclose(q1.T@tangent[:,j],q1.T@derivative,atol=2e-9,rtol=2e-8)
    # Global covariance rescaling cannot change the conditional predictor.
    np.testing.assert_allclose(tangent@theta,0,atol=2e-13)
    transfer,covariance=conditional_null(v,c,n0)
    f=x[:,None]*g[:,1:3]
    beta=np.array([.2,-.3])
    fitted=innovation_score(f[:n0]@beta,f[n0:]@beta,f[:n0],f[n0:],
        np.column_stack([c[n0:],tangent]),transfer,covariance)
    np.testing.assert_allclose(fitted['beta'],beta,atol=2e-13)
    np.testing.assert_allclose(fitted['contrast']@tangent,0,atol=2e-13)
    a=fitted['contrast']; b=-a@transfer
    complete=np.column_stack([b,a])
    np.testing.assert_allclose(fitted['covariance'],complete@v@complete.T,atol=2e-13)
    rescaled=conditional_mean_tangents(9*v,kernels,c,n0,3*y0+2*x[:n0])
    np.testing.assert_allclose(rescaled,tangent/3,atol=3e-13)


def test_conditional_portable_summary_public_fit_preserves_joint_covariance(tmp_path,monkeypatch):
    import json
    import pytest
    from scipy.stats import chi2
    from summit.epistasis.conditional import conditional_mean_summary
    from summit.epistasis.robust import write_robust_scores,load_robust_scores
    from summit.epistasis.cli import main
    from summit.prediction.genotype import FileGenotypeSource
    from scripts.epistasis.conditional_polygenic_reference import conditional_null,innovation_score
    rng=np.random.default_rng(598241)
    n0,n1,m=37,61,53
    g=rng.normal(size=(n0+n1,m));x=g[:,0]
    c=np.column_stack([np.ones(len(g)),x])
    v=.7*g@g.T/m+np.diag(.6+.4*x*x)
    transfer,variance=conditional_null(v,c,n0)
    f=x[:,None]*g[:,1:3]
    y=rng.normal(size=len(g))
    fitted=innovation_score(y[:n0],y[n0:],f[:n0],f[n0:],c[n0:],transfer,variance)
    h=fitted['response'].T@fitted['response']
    identity=dict(zip(['genotype_reference','direction','training_outcomes','confirmation_outcomes','null_fit'],
        [str(i)*64 for i in range(5)]))
    summary=conditional_mean_summary(fitted['beta'],fitted['covariance'],h,feature_names=['a','b'],
        trait_name='y',trait_unit='cm',identities=identity)
    path=tmp_path/'conditional.npz';write_robust_scores(summary,path)
    def unavailable(*args,**kwargs):
        raise AssertionError('genotype access during portable conditional fitting')
    monkeypatch.setattr(FileGenotypeSource,'__init__',unavailable)
    main(['fit',str(path),'--out',str(tmp_path/'fit.json')])
    out=json.loads((tmp_path/'fit.json').read_text())['fits'][0]
    np.testing.assert_allclose(out['beta'],fitted['beta'],atol=2e-13)
    np.testing.assert_allclose(out['coefficient_covariance'],fitted['covariance'],atol=2e-13)
    expected=fitted['beta']@np.linalg.solve(fitted['covariance'],fitted['beta'])
    np.testing.assert_allclose(out['joint_wald'],expected,atol=2e-13)
    assert out['method']=='conditional_polygenic_mean_tangent_v1'
    assert out['joint_df']==2
    assert out['diagnostics']['status']==summary.metadata['status']
    assert abs(fitted['covariance'][0,1])>1e-5
    # Generic saved-summary follow-up must retain the covariance model rather
    # than relabeling a conditional polygenic covariance as HC3.
    from summit.epistasis.robust import robust_followup
    followup=robust_followup(summary,{'both':[0,1]})
    assert followup['inference']==summary.metadata['inference']
    assert followup['method']==summary.metadata['method']
    assert followup['status']==summary.metadata['status']
    assert 'HC3' not in followup['correction']
    np.testing.assert_allclose(followup['groups'][0]['group_adjusted_p'],out['joint_p'],atol=2e-14)
    loaded=load_robust_scores(path)
    np.testing.assert_array_equal(loaded.score_covariance,summary.score_covariance)
    # Re-expressing a direction rescales coefficients/covariance, preserving
    # the joint test. Preserve negative unit changes and covariance signs.
    scale=np.array([-3.,.2]); inv=1/scale
    converted=conditional_mean_summary(fitted['beta']*inv,
        fitted['covariance']*inv[:,None]*inv[None,:],h*scale[:,None]*scale[None,:],
        feature_names=['a','b'],trait_name='y',trait_unit='cm',identities=identity)
    from summit.epistasis.robust import robust_score_tests
    np.testing.assert_allclose(robust_score_tests(converted)['joint_wald'],expected,atol=2e-13)
    changed=conditional_mean_summary(fitted['beta'],fitted['covariance'],h,feature_names=['a','b'],
        trait_name='y',trait_unit='cm',identities=dict(identity,confirmation_outcomes='f'*64))
    assert changed.metadata['preparation_identity']!=summary.metadata['preparation_identity']
    assert converted.metadata['preparation_identity']!=summary.metadata['preparation_identity']
    # Duplicated directions identify their sum, not two separate coefficients.
    span=conditional_mean_summary([.1,.1],np.ones((2,2))*.02,np.ones((2,2))*3,
        feature_names=['a','a_copy'],trait_name='y',trait_unit='cm',identities=identity)
    rank_fit=robust_score_tests(span,contrasts={'sum':[1,1]})
    assert rank_fit['joint_df']==1 and not np.any(rank_fit['estimable_coefficients'])
    np.testing.assert_allclose(rank_fit['coefficient_contrasts'][0]['beta'],.2,atol=1e-14)
    with pytest.raises(ValueError,match='authenticate'):
        conditional_mean_summary(fitted['beta'],fitted['covariance'],h,feature_names=['a','b'],
            trait_name='y',trait_unit='cm',identities={})
    with pytest.raises(ValueError,match='indefinite'):
        conditional_mean_summary(fitted['beta'],[[1.,2.],[2.,1.]],h,feature_names=['a','b'],
            trait_name='y',trait_unit='cm',identities=identity)


def test_random_effect_interaction_truth_and_second_order_mean_remainder():
    from scripts.epistasis.conditional_polygenic_reference import (
        conditional_null,conditional_mean_tangents,innovation_score,
    )
    rng=np.random.default_rng(357173)
    n0,n1,m=31,59,43
    g=rng.normal(size=(n0+n1,m));x=g[:,0]
    c=np.column_stack([np.ones(len(g)),x])
    kernels=np.stack([g@g.T/m,np.eye(len(g)),np.diag(.2+x*x)])
    theta=np.array([.8,.7,.3]);v=np.einsum('k,kij->ij',theta,kernels)
    true_transfer,_=conditional_null(v,c,n0)
    y0=rng.normal(size=n0)
    # A prespecified generating architecture differs from the learned feature.
    injected=.3*x*g[:,1]
    f=x*g[:,2]
    conditional_outcome=injected[n0:]+true_transfer@(y0-injected[:n0])
    errors=[]
    for step in (1e-2,5e-3,2.5e-3):
        working=v+step*kernels[0]
        transfer,covariance=conditional_null(working,c,n0)
        nuisance=np.column_stack([c[n0:],conditional_mean_tangents(working,kernels,c,n0,y0)])
        fitted=innovation_score(y0,conditional_outcome,f[:n0],f[n0:],nuisance,transfer,covariance)
        a=fitted['contrast'][0]
        target=a@(injected[n0:]-true_transfer@injected[:n0])
        bias=a@(true_transfer-transfer)@y0
        np.testing.assert_allclose(fitted['beta'][0]-target,bias,atol=1e-13)
        old_target=a@(injected[n0:]-transfer@injected[:n0])
        assert abs(old_target-target)>abs(bias)
        errors.append(abs(bias))
    # Removing derivatives of L Y0 suppresses the first-order mean error.
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4.,rtol=.04)


def test_unknown_common_scale_needs_studentization_even_with_known_shape():
    """A controlled counterexample to exact finite-N plug-in normal tails."""
    from scipy.stats import norm, t
    from scripts.epistasis.conditional_polygenic_reference import conditional_null, innovation_score
    rng=np.random.default_rng(817391)
    n0,n1,m,draws=18,36,30,4000
    g=rng.normal(size=(n0+n1,m));x=g[:,0]
    c=np.column_stack([np.ones(len(g)),x])
    shape=.8*g@g.T/m+np.diag(.5+.5*x*x)
    transfer,q=conditional_null(shape,c,n0)
    from scipy.linalg import null_space
    residual_basis=null_space(c[:n0].T)
    p=residual_basis@np.linalg.solve(residual_basis.T@shape[:n0,:n0]@residual_basis,residual_basis.T)
    df=residual_basis.shape[1]
    f=x[:,None]*g[:,1:]
    learning=f@f[:n0].T/(m-1)
    phenotypes=np.linalg.cholesky(shape)@rng.normal(size=(len(g),draws))
    statistics=[]
    for y in phenotypes.T:
        # Every draw relearns its direction from finite-mean-free contrasts.
        direction=learning@(p@y[:n0])
        fitted=innovation_score(y[:n0],y[n0:],direction[:n0],direction[n0:],
            c[n0:],transfer,q)
        estimated_scale=float(y[:n0]@p@y[:n0])/df
        statistics.append(float(fitted['beta'][0]/np.sqrt(fitted['covariance'][0,0]*estimated_scale)))
    statistics=np.asarray(statistics)
    for alpha in (.05,.005):
        observed=float(np.mean(abs(statistics)>t.isf(alpha/2,df)))
        assert abs(observed-alpha)<6*np.sqrt(alpha*(1-alpha)/draws)
        plug_in=2*t.sf(norm.isf(alpha/2),df)
        observed_normal=float(np.mean(abs(statistics)>norm.isf(alpha/2)))
        assert abs(observed_normal-plug_in)<6*np.sqrt(plug_in*(1-plug_in)/draws)
    assert 2*t.sf(norm.isf(.005/2),df)>.01
