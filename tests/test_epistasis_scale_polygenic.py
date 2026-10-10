"""Independent dense checks of genomic scale inference and its continuous bounds."""
import numpy as np
import pytest
from scipy.stats import chi2, boxcox

from test_epistasis_polygenic_operator import fixture
from summit.epistasis.scale_polygenic import (
    prepare_polygenic_scale, PolygenicScaleEvaluator, boxcox_polygenic_scale_test,
)


def setup():
    rng,make,kernels,a,b,contexts = fixture()
    features = rng.normal(size=(len(contexts),3))
    fixed = np.column_stack([contexts,contexts[:,1]**2])
    ga,gb = make(a),make(b)
    geometry = prepare_polygenic_scale(ga,gb,features[a],features[b],fixed[a],fixed[b],
        nn=lambda x,y:x@y,tn=lambda x,y:x.T@y,exact=True)
    latent = .25*rng.normal(size=len(contexts))+.3*features[:,0]+.07*contexts[:,1]
    y = np.exp(latent)
    return rng,geometry,kernels,a,b,features,fixed,y,ga,gb


def test_component_grams_match_dense_and_one_pass():
    rng,make,kernels,a,b,_ = fixture()
    for rows in (a,b):
        operator = make(rows)
        v = rng.normal(size=(len(rows),7))
        actual = operator.component_grams(v)
        k = kernels[:,rows][:,:,rows]
        np.testing.assert_allclose(actual['gram'],v.T@k@v,rtol=2e-12,atol=2e-11)
        np.testing.assert_allclose(actual['trace'],np.trace(k,axis1=1,axis2=2),atol=1e-12)
        assert operator.stream.ledger.traversals == {'polygenic_component_grams':1}
        operator.memory_bytes = operator.base_bytes
        with pytest.raises(MemoryError,match='Gram contraction'):
            operator.component_grams(v)
        assert operator.stream.ledger.traversals == {'polygenic_component_grams':1}


@pytest.mark.parametrize('mode',['omnibus','omnibus_sparse'])
def test_coefficients_covariance_and_scale_bounds_against_dense(mode):
    from scripts.epistasis.conditional_polygenic_reference import fit_covariance
    _,g,k,a,b,f,c,y,ga,gb = setup()
    evaluator = PolygenicScaleEvaluator(g,y[a],y[b],groups={'block':[0,1],'last':[2]},contrast_mode=mode)
    powers = [-1.,-.73,0.,.24,1.]
    evaluator.evaluate(powers)
    assert ga.stream.ledger.traversals['scale_training_moments']==1
    assert gb.stream.ledger.traversals['scale_confirmation_grams']==1
    design = np.column_stack([c[b],f[b]])
    weights = np.linalg.pinv(design)[-3:].T
    np.testing.assert_allclose(g.weights,weights,atol=1e-13)
    for power in powers:
        ya,yb = boxcox(y[a]/np.exp(evaluator.log_reference),power),boxcox(y[b]/np.exp(evaluator.log_reference),power)
        theta,_ = fit_covariance(ya,np.column_stack([c[a],f[a]]),k[:,a][:,:,a])
        covariance = weights.T@np.einsum('k,kij->ij',theta,k[:,b][:,:,b])@weights
        actual = evaluator.cache[power]
        np.testing.assert_allclose(actual['theta'],theta,atol=1e-12)
        np.testing.assert_allclose(actual['beta'][:,0],weights.T@yb,atol=2e-14)
        np.testing.assert_allclose(actual['covariance'],covariance,atol=1e-14)
        idx = [0,1]; beta = weights.T@yb
        statistic = beta[idx]@np.linalg.solve(covariance[np.ix_(idx,idx)],beta[idx])
        np.testing.assert_allclose(actual['records']['block']['omnibus_p'],chi2.sf(statistic,2),atol=2e-14)
    # Independently evaluated powers were not used to construct these bounds.
    nontrivial = False
    for center,radius in [(0.,1.),(-.73,.13),(.24,.007)]:
        bound = evaluator.interval(center-radius,center+radius)
        nontrivial |= any(v < .99 for v in bound['p_upper'].values())
        unseen = center+radius*np.linspace(-.991,.983,19)
        evaluator.evaluate(unseen)
        for name in evaluator.groups:
            assert max(evaluator.cache[v]['records'][name]['p'] for v in unseen) <= bound['p_upper'][name]+1e-12
    # Deliberately small sample exercises zero-valued NNLS components.
    assert any(np.any(v['theta']==0) for v in evaluator.cache.values())
    assert len({tuple(v['theta']==0) for v in evaluator.cache.values()}) > 1
    assert nontrivial
    # Confirmation contractions are reused regardless of the number of scales.
    assert gb.stream.ledger.traversals['scale_confirmation_grams']==1


def test_units_search_and_validation():
    _,g,_,a,b,_,_,y,_,_ = setup()
    kwargs = dict(bounds=(-.5,.5),alpha=.05,max_evaluations=9,groups={'block':[0,1,2]})
    first = boxcox_polygenic_scale_test(g,y[a],y[b],**kwargs)
    second = boxcox_polygenic_scale_test(g,y[a]*1e4,y[b]*1e4,**kwargs)
    for key in ('p_sup_lower','p_upper'):
        np.testing.assert_allclose(first['tests']['block'][key],second['tests']['block'][key],atol=1e-12)
    assert first['evaluations'] <= 9
    assert first['tests']['block']['p_upper'] >= first['tests']['block']['p_sup_lower']
    for extra in ({'groups':{'bad':[0,0]}},{'contrast_mode':'learned'},{'alpha':0.},
                  {'batch_size':1},{'order':0},{'max_evaluations':2}):
        with pytest.raises(ValueError):
            boxcox_polygenic_scale_test(g,y[a],y[b],**dict(kwargs,**extra))
    with pytest.raises(ValueError,match='positive finite'):
        boxcox_polygenic_scale_test(g,-y[a],y[b])


def test_he_mean_removes_all_features_including_untested_terms():
    from summit.epistasis.polygenic import estimate_components
    rng,g,_,a,_,f,c,y,ga,_ = setup()
    original = estimate_components(ga,np.log(y[a]),g.he)
    changed = estimate_components(ga,np.log(y[a])+np.column_stack([c[a],f[a]])@rng.normal(size=c.shape[1]+f.shape[1]),g.he)
    np.testing.assert_allclose(original,changed,atol=1e-12)
