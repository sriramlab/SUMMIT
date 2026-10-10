"""Honest projection, independent OLS contrasts, and outer-set coverage."""
import numpy as np
import pytest
from scipy.stats import chi2

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
          '--max-pilot-evaluations','17','--max-evaluations','9']
    with pytest.raises(MemoryError):main(args+['--memory-gib','.00001'])
    assert main(args)==0
    result=json.loads(out.read_text())
    assert result['pilot_n']==700 and result['confirmation_n']==1100
    assert result['native_execution']['output_numa_status']
    with pytest.raises(FileExistsError):main(args)
