"""Independent calculus/OLS checks and continuous, off-grid scale nulls."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

from summit.epistasis.robust import prepare_robust_geometry, prepare_robust_nuisance
from summit.epistasis.scale import boxcox_derivatives, boxcox_scale_test


def panel(seed=951, n=1200):
    rng = np.random.default_rng(seed)
    a, b = rng.binomial(1, .5, (2, n))
    c = np.column_stack([np.ones(n), a, b])
    f = (a*b)[:, None].astype(float)
    return rng, a, b, c, f


def test_boxcox_derivatives_against_independent_integrals():
    t = np.array([-5., -1., -.01, 0., .01, 1., 5.])
    for power in (-2., -.1, -1e-8, 0., 1e-8, .1, 2.):
        got = boxcox_derivatives(t, power, order=4)
        expected = np.array([[quad(lambda s: s**k*np.exp(power*s), 0, v,
            epsabs=1e-11,epsrel=1e-12)[0] for k in range(5)] for v in t])
        np.testing.assert_allclose(got,expected,rtol=2e-11,atol=1e-11)


def test_reused_hc3_geometry_against_full_design_ols():
    rng, _, _, c, f = panel(n=257)
    y = rng.normal(size=(len(c), 5))
    nuisance = prepare_robust_nuisance(c)
    geometry = prepare_robust_geometry(f, nuisance=nuisance)
    scores, beta, e, meat, _ = geometry.fit(y)
    x = np.column_stack([c, f])
    inv = np.linalg.inv(x.T@x)
    expected = inv@x.T@y
    residual = y-x@expected
    h = np.sum((x@inv)*x,axis=1)
    np.testing.assert_allclose(beta,expected[-1:],atol=1e-13)
    np.testing.assert_allclose(e,residual,atol=1e-13)
    for j in range(y.shape[1]):
        influence = (x@inv)*(residual[:,j]/(1-h))[:,None]
        variance = influence.T@influence
        np.testing.assert_allclose(geometry.inverse@meat[j]@geometry.inverse,
            variance[-1:,-1:],atol=1e-13)
    second = prepare_robust_geometry(np.column_stack([f, rng.normal(size=len(f))]),nuisance=nuisance)
    assert second.u is geometry.u


def test_continuous_envelope_dominates_independent_dense_sweep_and_units():
    rng,a,b,c,f = panel(n=1200)
    y = np.exp(.6*a+.4*b-1.2*a*b+.15*rng.normal(size=len(a)))
    geometry = prepare_robust_geometry(f,c)
    result = boxcox_scale_test(geometry,y,max_evaluations=65)
    assert result['tests']['joint']['status']=='rejected_specified_scale_family'
    x = np.column_stack([c,f]); inverse = np.linalg.inv(x.T@x)
    influence = (x@inverse)[:,-1]
    h = np.sum((x@inverse)*x,axis=1)
    t = np.log(y)-np.log(y).mean()
    # This OLS oracle does not call the reusable geometry or scale derivative
    # code. Compare each individual interval, not only the global envelope.
    for leaf in result['intervals']:
        powers = np.linspace(leaf['lower'],leaf['upper'],11)
        for power in powers:
            z = t if power==0 else np.expm1(power*t)/power
            beta = inverse@x.T@z
            e = z-x@beta
            variance = np.sum((influence*e/(1-h))**2)
            p = chi2.sf(beta[-1]**2/variance,1)
            assert p <= leaf['p_upper']['joint']+1e-12
    scaled = boxcox_scale_test(geometry,1000*y,max_evaluations=65)
    np.testing.assert_allclose(result['tests']['joint']['p_upper'],
        scaled['tests']['joint']['p_upper'],rtol=1e-6,atol=1e-12)


def test_narrow_off_grid_additive_scale_cannot_be_falsely_certified():
    rng,a,b,c,f = panel(seed=4421,n=1600)
    power = .371913
    latent = .5*a+.7*b+.0005*rng.normal(size=len(a))
    y = (1+power*latent)**(1/power)
    geometry = prepare_robust_geometry(f,c)
    transformed = np.expm1(power*np.log(y))/power
    _, beta, _, meat, _ = geometry.fit(transformed)
    variance = float((geometry.inverse@meat[0]@geometry.inverse)[0,0])
    oracle_p = float(chi2.sf(beta[0,0]**2/variance,1))
    assert oracle_p > .05
    result = boxcox_scale_test(geometry,y,max_evaluations=129)
    assert result['tests']['joint']['p_upper'] >= oracle_p-1e-10
    assert result['tests']['joint']['status']!='rejected_specified_scale_family'
    bounded = boxcox_scale_test(geometry,y,max_evaluations=3)
    assert bounded['tests']['joint']['status']=='unresolved_search_bound'
    assert bounded['tests']['joint']['p_upper'] >= oracle_p


def test_scale_validation_and_conditional_groups():
    rng,a,b,c,f = panel(n=500)
    extra = rng.normal(size=(len(c),2))
    geometry = prepare_robust_geometry(np.column_stack([f,extra]),c)
    y = np.exp(.2*a+.2*b+.1*rng.normal(size=len(c)))
    result = boxcox_scale_test(geometry,y,groups={'all':[0,1,2],'pair':[0]},max_evaluations=9)
    assert set(result['tests'])=={'all','pair'}
    assert all(0 <= v['p_sup_lower'] <= v['p_upper'] <= 1 for v in result['tests'].values())
    with pytest.raises(ValueError,match='positive'):
        boxcox_scale_test(geometry,-y)
    with pytest.raises(ValueError,match='intercept'):
        boxcox_scale_test(prepare_robust_geometry(f,c[:,1:]),y)
    with pytest.raises(ValueError,match='identifiable'):
        boxcox_scale_test(prepare_robust_geometry(np.column_stack([f,f]),c),y)
    degenerate=boxcox_scale_test(geometry,np.ones(len(y)))
    assert degenerate['tests']['joint']['status']=='unresolved_covariance'
    assert degenerate['tests']['joint']['p_upper']==1.


def test_public_scale_cli_native_products_and_no_overwrite(tmp_path):
    import json
    from summit.epistasis.cli import main
    from epistasis_helpers import epistasis_threads
    rng,a,b,c,f=panel(n=1100)
    y=np.exp(.2*a+.2*b+.2*rng.normal(size=len(a)))
    path=tmp_path/'input.npz';out=tmp_path/'scale.json'
    np.savez(path,features=f,fixed_effects=c,phenotype=y)
    args=['scale-test',str(path),'--out',str(out),'--num-threads',str(epistasis_threads())]
    assert main(args)==0
    result=json.loads(out.read_text())
    direct=boxcox_scale_test(prepare_robust_geometry(f,c),y)
    np.testing.assert_allclose(result['tests']['joint']['p_upper'],direct['tests']['joint']['p_upper'],atol=1e-10)
    assert len(result['input_sha256'])==64
    with pytest.raises(FileExistsError):
        main(args)
    from summit.epistasis.scale_workflow import run_scale_arrays
    wide=tmp_path/'wide.npz'
    np.savez(wide,features=np.zeros((150,120)),fixed_effects=np.ones((150,1)),phenotype=np.ones(150))
    with pytest.raises(MemoryError,match='covariance cache'):
        run_scale_arrays(wide,tmp_path/'wide.json',memory_bytes=300*2**20)
