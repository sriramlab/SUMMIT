"""Honest projection, independent OLS contrasts, and outer-set coverage."""
import numpy as np
import pytest
from scipy.stats import chi2, norm

from summit.epistasis.robust import prepare_robust_geometry
from summit.epistasis.scale import boxcox_scale_test
from summit.epistasis.scale_honest import boxcox_honest_scale_test


def design(seed, n):
    rng = np.random.default_rng(seed)
    g = rng.binomial(2,.35,size=(n,3)).astype(float)
    c = np.column_stack([np.ones(n),g,g==1])
    f = np.column_stack([g[:,0]*g[:,1],g[:,0]*g[:,2],g[:,1]*g[:,2]])
    latent = 2+.9*g[:,0]-.6*g[:,2]+.3*(g[:,1]==1)+.3*rng.uniform(-1,1,n)
    y = (1+.371913*latent)**(1/.371913)
    return c,f,y


def test_outer_confidence_set_contains_every_accepted_dense_oracle_power():
    c,f,y = design(456,1000)
    geometry = prepare_robust_geometry(f,c)
    result = boxcox_scale_test(geometry,y,alpha=.005,confidence_width=.03,max_evaluations=65)
    x = np.column_stack([c,f]);inv = np.linalg.inv(x.T@x)
    weights = x@inv;h = np.sum(weights*x,axis=1)
    t = np.log(y)-np.log(y).mean()
    accepted = 0
    for power in np.linspace(-2,2,101):
        z = t if power == 0 else np.expm1(power*t)/power
        b = inv@x.T@z;e = z-x@b
        influence = weights[:,-3:]*(e/(1-h))[:,None]
        v = influence.T@influence
        p = chi2.sf(b[-3:]@np.linalg.solve(v,b[-3:]),3)
        if p>=.005:
            accepted += 1
            assert any(a<=power<=b for a,b in result['confidence_sets']['joint'])
    assert accepted
    unresolved = boxcox_scale_test(geometry,y,alpha=.005,confidence_width=.001,max_evaluations=3)
    assert unresolved['confidence_sets']['joint']==[[-2.,2.]]


def test_disconnected_search_preserved_or_conservatively_coarsened():
    c,f,y = design(11,600);geometry=prepare_robust_geometry(f,c)
    domains=[[-2.,-1.],[.2,.6]]
    result=boxcox_scale_test(geometry,y,search_intervals=domains,max_evaluations=17)
    assert not result['domain_coarsened']
    assert all(any(a<=v['power']<=b for a,b in domains) for v in result['points'])
    small=boxcox_scale_test(geometry,y,search_intervals=domains,max_evaluations=3)
    assert small['domain_coarsened'] and small['search_intervals']==[(-2.,.6)]
    with pytest.raises(ValueError,match='within bounds'):
        boxcox_scale_test(geometry,y,search_intervals=[[-3,0]])


def test_pilot_projection_matches_full_conditional_ols_at_confirmation_powers():
    pc,pf,py=design(8,1200);cc,cf,cy=design(9,1600)
    pg=prepare_robust_geometry(pf,pc);cg=prepare_robust_geometry(cf,cc)
    result=boxcox_honest_scale_test(pg,py,cg,cy,pilot_ids=np.arange(1200),
        confirmation_ids=np.arange(1200,2800),groups={'pair':[0,1]},
        max_pilot_evaluations=65,max_evaluations=33)
    test=result['tests']['pair']
    assert test['projection']=='pilot_direction_removed' and test['df']==1
    contrast=np.asarray(test['contrast'])
    x=np.column_stack([cc,cf]);inv=np.linalg.inv(x.T@x)
    weights=x@inv;h=np.sum(weights*x,axis=1)
    t=np.log(cy)-np.log(cy).mean()
    for point in test['confirmation']['points']:
        power=point['power'];z=t if power==0 else np.expm1(power*t)/power
        b=inv@x.T@z;e=z-x@b
        influence=weights[:,-3:]*(e/(1-h))[:,None]
        v=contrast@(influence.T@influence)@contrast.T
        eta=contrast@b[-3:]
        expected=chi2.sf(eta@np.linalg.solve(v,eta),1)
        np.testing.assert_allclose(point['p']['pair'],expected,rtol=1e-7,atol=1e-10)
    assert test['p_upper']==min(1.,result['gamma']+test['confirmation']['tests']['pair']['p_upper'])
    # Shared geometry must not be mutated by the orthogonal reparameterization.
    np.testing.assert_allclose(cg.h,cg.r.T@cg.r,rtol=1e-12,atol=1e-12)


def test_full_fallback_scalar_and_independence_validation():
    pc,pf,py=design(18,700);cc,cf,cy=design(19,900)
    pg=prepare_robust_geometry(pf,pc);cg=prepare_robust_geometry(cf,cc)
    args=dict(pilot_ids=np.arange(700),confirmation_ids=np.arange(700,1600),
              groups={'scalar':[0]},max_pilot_evaluations=17,max_evaluations=9)
    result=boxcox_honest_scale_test(pg,py,cg,cy,**args)
    assert result['tests']['scalar']['projection']=='full_block_fallback'
    assert result['tests']['scalar']['df']==1
    with pytest.raises(ValueError,match='overlap'):
        boxcox_honest_scale_test(pg,py,cg,cy,**(args|{'confirmation_ids':np.arange(900)}))
    with pytest.raises(ValueError,match='gamma'):
        boxcox_honest_scale_test(pg,py,cg,cy,gamma=.05,**args)
    with pytest.raises(ValueError,match='unique'):
        boxcox_honest_scale_test(pg,py,cg,cy,**(args|{'pilot_ids':np.zeros(700,dtype=int)}))
    with pytest.raises(ValueError,match='overlap'):
        boxcox_honest_scale_test(pg,py,cg,cy,**(args|{
            'pilot_ids':np.arange(700).astype('S'),
            'confirmation_ids':np.arange(900).astype('U')}))
    bad=np.arange(700).astype('U20');bad[0]='first\nsecond'
    with pytest.raises(ValueError,match='single-line'):
        boxcox_honest_scale_test(pg,py,cg,cy,**(args|{'pilot_ids':bad}))


def test_empty_pilot_set_requires_no_confirmation_fit():
    rng=np.random.default_rng(32)
    a,b=rng.binomial(1,.5,(2,1400))
    c=np.column_stack([np.ones(len(a)),a,b]);f=(a*b)[:,None].astype(float)
    y=np.exp(.6*a+.5*b-1.3*a*b+.02*rng.normal(size=len(a)))
    pg=prepare_robust_geometry(f[:700],c[:700]);cg=prepare_robust_geometry(f[700:],c[700:])
    result=boxcox_honest_scale_test(pg,y[:700],cg,y[700:],pilot_ids=np.arange(700),
        confirmation_ids=np.arange(700,1400),max_pilot_evaluations=65)
    test=result['tests']['joint']
    assert test['pilot_confidence_set']==[]
    assert test['p_upper']==result['gamma'] and test['confirmation_evaluations']==0


def test_honest_cli_native_path_schema_memory_and_no_overwrite(tmp_path):
    import json
    from summit.epistasis.cli import main
    from epistasis_helpers import epistasis_threads
    c,f,y=design(321,1800)
    path=tmp_path/'honest.npz';out=tmp_path/'result.json'
    np.savez(path,features=f,fixed_effects=c,phenotype=y,sample_ids=np.arange(len(y)),
             pilot_mask=np.arange(len(y))<700)
    args=['scale-test-honest',str(path),'--out',str(out),'--num-threads',str(epistasis_threads()),
          '--max-pilot-evaluations','17','--max-evaluations','9','--contrast-mode','hybrid']
    with pytest.raises(MemoryError):main(args+['--memory-gib','.00001'])
    assert main(args)==0
    result=json.loads(out.read_text())
    assert result['pilot_n']==700 and result['confirmation_n']==1100
    assert result['contrast_mode']=='hybrid'
    assert result['native_execution']['output_numa_status']
    with pytest.raises(FileExistsError):main(args)


def test_hybrid_score_dense_oracle_combination_and_pilot_only_selection():
    pc,pf,py=design(8,1800);cc,cf,cy=design(9,2200)
    pg=prepare_robust_geometry(pf,pc);cg=prepare_robust_geometry(cf,cc)
    args=dict(pilot_ids=np.arange(len(py)),confirmation_ids=np.arange(len(py),len(py)+len(cy)),
        max_pilot_evaluations=33,max_evaluations=17,contrast_mode='hybrid')
    result=boxcox_honest_scale_test(pg,py,cg,cy,**args)
    test=result['tests']['joint'];assert test['contrast_mode']=='hybrid'
    contrast=np.asarray(test['score_contrast'])
    x=np.column_stack([cc,cf]);inv=np.linalg.inv(x.T@x)
    weights=x@inv;h=np.sum(weights*x,axis=1)
    t=np.log(cy)-np.log(cy).mean()
    for point in test['confirmation']['points']:
        power=point['power'];z=t if power==0 else np.expm1(power*t)/power
        b=inv@x.T@z;e=z-x@b
        influence=weights[:,-3:]*(e/(1-h))[:,None]
        v=contrast@(influence.T@influence)@contrast.T;eta=contrast@b[-3:]
        expected=norm.sf(float(eta[0]/np.sqrt(v[0,0])))
        components=point['p_components']['joint']
        np.testing.assert_allclose(components['score'],expected,rtol=1e-7,atol=1e-10)
        full_v=influence.T@influence
        full_p=chi2.sf(b[-3:]@np.linalg.solve(full_v,b[-3:]),3)
        np.testing.assert_allclose(components['omnibus'],full_p,rtol=1e-7,atol=1e-10)
        assert point['p']['joint']==min(1.,2*min(components.values()))
        si=(influence@contrast.T)[:,0]**2
        diagnostic=point['score_influence_support']['joint']
        np.testing.assert_allclose(diagnostic['minimum_coordinate_ess'],si.sum()**2/(si@si),rtol=1e-7)
        np.testing.assert_allclose(diagnostic['maximum_coordinate_variance_share'],si.max()/si.sum(),rtol=1e-7)
    assert test['p_upper']==min(1.,result['gamma']+test['confirmation']['tests']['joint']['p_upper'])
    for leaf in test['confirmation']['intervals']:
        for power in np.linspace(leaf['lower'],leaf['upper'],7):
            z=t if power==0 else np.expm1(power*t)/power
            b=inv@x.T@z;e=z-x@b;influence=weights[:,-3:]*(e/(1-h))[:,None]
            full_v=influence.T@influence;eta=contrast@b[-3:]
            full_p=chi2.sf(b[-3:]@np.linalg.solve(full_v,b[-3:]),3)
            score_p=norm.sf(eta[0]/np.sqrt((contrast@full_v@contrast.T)[0,0]))
            assert min(1.,2*min(full_p,score_p))<=leaf['p_upper']['joint']+1e-8
    changed=boxcox_honest_scale_test(pg,py,cg,cy[::-1],**args)['tests']['joint']
    np.testing.assert_array_equal(changed['score_contrast'],test['score_contrast'])
    assert changed['pilot_confidence_set']==test['pilot_confidence_set']
    # The hybrid keeps the ORIGINAL block: an alternative parallel to the
    # scale derivative must remain visible in its omnibus safeguard.
    small=boxcox_honest_scale_test(pg,py,cg,cy,groups={'pair':[0,1]},**args)['tests']['pair']
    assert small['df']==2 and small['contrast_mode']=='hybrid'
    np.testing.assert_array_equal(small['contrast'],np.eye(3)[:2])
    scalar=boxcox_honest_scale_test(pg,py,cg,cy,groups={'one':[0]},**args)['tests']['one']
    assert scalar['df']==1 and scalar['contrast_mode']=='omnibus'
    with pytest.raises(ValueError,match='contrast_mode'):
        boxcox_honest_scale_test(pg,py,cg,cy,**(args|{'contrast_mode':'best_p'}))


def test_influence_screen_flags_concentrated_transformed_outcomes():
    c,f,y=design(903,2000);g=prepare_robust_geometry(f,c)
    y[0]*=1000
    result=boxcox_scale_test(g,y,max_evaluations=3)
    extreme=next(v for v in result['points'] if v['power']==2.)
    support=extreme['influence_support']['joint']
    assert support['flagged'] and support['minimum_coordinate_ess']<10
    assert not result['tests']['joint']['influence_screen_passed_at_evaluated_powers']


@pytest.mark.parametrize('sign',[-1.,1.])
def test_signed_continuous_envelope_covers_dense_ols_for_both_orientations(sign):
    c,f,y=design(916,700);f=f.copy();f[:,0]*=sign
    geometry=prepare_robust_geometry(f,c)
    result=boxcox_scale_test(geometry,y,groups={'signed':[0]},alternative='greater',
        max_evaluations=17,confidence_width=.1)
    x=np.column_stack([c,f]);inv=np.linalg.inv(x.T@x)
    w=x@inv;h=np.sum(w*x,axis=1);t=np.log(y)-np.log(y).mean()
    for leaf in result['intervals']:
        for power in np.linspace(leaf['lower'],leaf['upper'],7):
            z=t if power==0 else np.expm1(power*t)/power
            b=inv@x.T@z;e=z-x@b;influence=w[:,-3]*(e/(1-h))
            expected=norm.sf(b[-3]/np.linalg.norm(influence))
            assert expected<=leaf['p_upper']['signed']+1e-8
    with pytest.raises(ValueError,match='scalar'):
        boxcox_scale_test(geometry,y,alternative='greater')
