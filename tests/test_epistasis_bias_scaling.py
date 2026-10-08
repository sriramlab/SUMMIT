"""Independent identities underlying the matched bias diagnostic."""
import numpy as np

from scripts.epistasis.conditional_bias_scaling import (
    conditional_bias_components, exact_tangent_remainder, fit_contrast,
    restricted_geometry, transfer,
)


def fixture():
    rng=np.random.default_rng(62180)
    g=rng.normal(size=(48,19))
    h=rng.normal(size=(48,13))
    kernels=np.stack([g@g.T/19,h@h.T/13,np.eye(48)])
    fixed=np.column_stack([np.ones(48),rng.normal(size=48)])
    geometry=restricted_geometry(kernels,fixed,24,128,918)
    return rng,kernels,fixed,geometry


def test_fixed_bias_decomposition_is_exact_for_unchanged_contrast():
    rng,_,_,geom=fixture()
    mt=transfer(geom,np.array([.8,.4,.6]))[0]
    mh=transfer(geom,np.array([.3,.7,.9]))[0]
    a=rng.normal(size=24)
    g0=rng.normal(size=geom['q'].shape[1]);g1=rng.normal(size=24)
    noise=rng.normal(size=len(g0))
    parts=conditional_bias_components(a,g0,g1,noise,g0+noise,mt,mh)
    np.testing.assert_allclose(parts['total'],parts['oracle_remaining']+parts['covariance'],atol=1e-12)
    assert abs(parts['identity_error'])<1e-12
    # An oracle covariance does not imply zero realized-architecture bias.
    oracle=conditional_bias_components(a,g0,g1,noise,g0+noise,mt,mt)
    assert oracle['covariance']==0
    assert abs(oracle['total'])>.1


def test_tangent_resolvent_remainder_and_second_order_rate():
    rng,_,fixed,geom=fixture()
    fitted=np.array([.8,.4,.6])
    y0=rng.normal(size=geom['q'].shape[1])
    f0=rng.normal(size=len(y0));f1=rng.normal(size=24)
    a,m,fac,v00,v10,_,_=fit_contrast(geom,fitted,y0,f0,f1,fixed[24:],tangent=True)
    errors=[]
    for step in (.02,.01,.005):
        theta=fitted+step*np.array([.4,-.2,.3])
        mt,_,v00t,v10t=transfer(geom,theta)
        direct=float(a@(mt-m)@y0)
        first,second=exact_tangent_remainder(a,y0,m,fac,v00,v10,v00t,v10t)
        np.testing.assert_allclose(first,0,atol=1e-13)
        np.testing.assert_allclose(direct,first+second,atol=1e-13)
        errors.append(abs(direct))
    np.testing.assert_allclose(np.array(errors[1:])/errors[:-1],.25,rtol=.025)


def test_known_covariance_innovation_matches_full_joint_covariance():
    _,kernels,_,geom=fixture()
    theta=np.array([.8,.4,.6])
    m,_,_,v10=transfer(geom,theta)
    q=geom['q'];v=np.einsum('k,kij->ij',theta,kernels)
    contrast=np.column_stack([-m@q.T,np.eye(24)])
    expected=v[24:,24:]-m@v10.T
    np.testing.assert_allclose(contrast@v@contrast.T,expected,atol=1e-12)
    np.testing.assert_allclose(contrast@v[:,:24]@q,0,atol=1e-12)


def test_exact_trace_moment_covariance_matches_gaussian_monte_carlo():
    rng,_,_,geom=fixture()
    theta=np.array([.8,.4,.6])
    k=geom['k00'];v=np.einsum('k,kij->ij',theta,k)
    kv=k@v
    expected=2*np.einsum('aij,bji->ab',kv,kv)
    draws=np.linalg.cholesky(v)@rng.normal(size=(len(v),30000))
    moments=np.einsum('ir,aij,jr->ar',draws,k,draws)
    np.testing.assert_allclose(np.cov(moments),expected,rtol=.04)


def test_trace_and_component_estimates_match_production_equations():
    from summit.epistasis.polygenic import he_geometry, estimate_components
    from scripts.epistasis.conditional_bias_scaling import estimate
    rng,kernels,fixed,geom=fixture()
    class DenseOperator:
        rows=np.arange(24)
        count=3
        base_bytes=0
        memory_bytes=2**30
        identity='bounded dense test'
        def apply(self,values,**kwargs):
            return kernels[:,:24,:24]@values
    operator=DenseOperator()
    for exact in (True,False):
        production=he_geometry(operator,fixed[:24],probes=128,seed=918,exact=exact,keep_products=False)
        gram=geom['exact' if exact else 'sketch']
        np.testing.assert_allclose(production['gram'],gram,atol=1e-11,rtol=1e-11)
        y=rng.normal(size=(24,2))
        u=geom['q'].T@y
        moments=np.einsum('ir,aij,jr->ar',u,geom['k00'],u)
        expected=np.column_stack([estimate(moments[:,i],gram) for i in range(2)])
        np.testing.assert_allclose(estimate_components(operator,y,production),expected,atol=1e-11,rtol=1e-10)


def test_oracle_fixed_mean_displacement_is_accounted_genetic_uncertainty():
    from scipy.linalg import cho_factor, cho_solve
    rng,kernels,_,geom=fixture()
    theta=np.array([.8,.4,.6])
    m,_,_,v10=transfer(geom,theta)
    genetic=np.einsum('k,kij->ij',theta[:2],kernels[:2])
    q=geom['q'];v00=q.T@(genetic[:24,:24]+theta[2]*np.eye(24))@q
    cross=genetic[24:,:24]@q
    y0=np.linalg.cholesky(v00)@rng.normal(size=len(v00))
    # Adaptive direction chosen using only the observed training contrasts.
    a=np.sin(cross@y0);a/=np.linalg.norm(a)
    posterior_mean=cross@cho_solve(cho_factor(v00,lower=True),y0)
    posterior_cov=genetic[24:,24:]-cross@cho_solve(cho_factor(v00,lower=True),cross.T)
    full_v=np.einsum('k,kij->ij',theta,kernels)
    innovation=full_v[24:,24:]-m@v10.T
    expected=float(a@(innovation-theta[2]*np.eye(24))@a)
    np.testing.assert_allclose(a@(posterior_mean-m@y0),0,atol=1e-12)
    np.testing.assert_allclose(a@posterior_cov@a,expected,atol=1e-12)
    draws=posterior_mean[:,None]+np.linalg.cholesky(posterior_cov)@rng.normal(size=(24,30000))
    displacement=a@(draws-m@y0[:,None])
    assert expected>.1
    np.testing.assert_allclose(np.mean(displacement**2),expected,rtol=.03)
