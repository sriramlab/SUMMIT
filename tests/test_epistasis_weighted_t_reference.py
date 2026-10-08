"""Independent density/convolution checks for heavy-error diagnostic tails."""
import numpy as np
from scipy.integrate import quad
from scipy.stats import t,norm


def test_weighted_t5_inversion_against_single_and_convolution():
    from scripts.epistasis.weighted_t_reference import weighted_t5_sf,weighted_t5_rejection
    scale=np.sqrt(3/5)
    for x in (-4.,-.2,0.,1.,3.,8.):
        actual=weighted_t5_sf(x,np.array([0.,-1.]))
        np.testing.assert_allclose(actual['probability'],t.sf(x/scale,5),atol=2e-10)
    weights=np.array([.4,-.7]);value=2.1
    expected,error=quad(lambda x:t.pdf(x,5)*t.sf((value-scale*weights[0]*x)/(scale*abs(weights[1])),5),
        -np.inf,np.inf,epsabs=1e-10)
    actual=weighted_t5_sf(value,weights)
    np.testing.assert_allclose(actual['probability'],expected,atol=2e-9)
    result=weighted_t5_rejection(2.4,.3,weights)
    np.testing.assert_allclose(result['probability'],
        weighted_t5_sf(2.7,weights)['probability']+weighted_t5_sf(2.1,weights)['probability'],atol=1e-13)


def test_weighted_t5_large_diffuse_array_and_scale():
    from scripts.epistasis.weighted_t_reference import weighted_t5_sf
    n=131072;weights=np.full(n,1/np.sqrt(n));value=2.807033768343811
    actual=weighted_t5_sf(value,weights)
    # Finite excess kurtosis gives an O(1/N) correction, not an assumed exact
    # Gaussian result. This test also exercises the intended full cohort size.
    leading=(6/(24*n))*(value**3-3*value)*norm.pdf(value)
    assert abs(actual['probability']-norm.sf(value)-leading)<2e-8
    scaled=weighted_t5_sf(-3*value,-3*weights)
    np.testing.assert_allclose(scaled['probability'],1-actual['probability'],atol=1e-12)


def test_fixed_architecture_conditioning_retains_realized_training_noise():
    from scripts.epistasis.public_conditional_validation import outcome_reference
    rng=np.random.default_rng(712309);n0,n1=16,32;i0=np.arange(n0);i1=np.arange(n0,n0+n1)
    mean=rng.normal(size=n0+n1);signal=rng.normal(size=n0+n1)
    y0=mean[i0]+rng.normal(size=n0);a=rng.normal(size=n1)/n1;b=rng.normal(size=n0)/n0
    variance=rng.uniform(.2,2.,n0+n1);v=float(np.sum(a*a*variance[i1]));se=1.2*np.sqrt(v)
    result=outcome_reference(a,b,-b,v,signal=signal,mean=mean,variance=variance,y0=y0,
        i0=i0,i1=i1,definition=dict(sampling_law='fixed_architecture',error_law='gaussian'),se=se)
    expected_mean=float(a@mean[i1]+b@y0)
    np.testing.assert_allclose(result['conditional_mean'],expected_mean,atol=1e-15)
    np.testing.assert_allclose(result['conditional_interaction_truth'],a@signal[i1]+b@signal[i0],atol=1e-15)
    threshold=norm.isf(.05/2)*se
    expected=norm.cdf((-threshold-expected_mean)/np.sqrt(v))+norm.sf((threshold-expected_mean)/np.sqrt(v))
    np.testing.assert_allclose(result['conditional_rejection_probability']['0.05'],expected,atol=1e-15)
