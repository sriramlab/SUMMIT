import numpy as np
from summit.pcgc.sampling import hoeffding_working_covariance


def test_hoeffding_projection_is_psd_and_preserves_positive_components():
    rng = np.random.default_rng(29817)
    A = rng.normal(size=(7,5)); first = A.T@A
    pair = np.diag([.2,.5,.8])
    covariance = first.copy(); covariance[:3,:3] += pair
    delta,diag = hoeffding_working_covariance(covariance,pair)
    np.testing.assert_allclose(delta,0.,atol=1e-13)
    covariance[0,0] -= 12.
    delta,diag = hoeffding_working_covariance(covariance,pair)
    assert np.linalg.eigvalsh(covariance+delta).min()>0
    assert np.linalg.eigvalsh(delta).min()>-1e-12
    assert diag['adjusted_directions']>0
    # The metric and both covariance blocks transform with the parameters.
    T = np.zeros((5,5)); T[:3,:3] = rng.normal(size=(3,3)); T[3:,3:] = rng.normal(size=(2,2))
    changed,_ = hoeffding_working_covariance(T@covariance@T.T,T[:3,:3]@pair@T[:3,:3].T)
    np.testing.assert_allclose(changed,T@delta@T.T,rtol=3e-10,atol=2e-12)


def test_redundant_population_moments_keep_their_null_space():
    covariance = np.diag([-.2,.4,.1,0.])
    pair = np.diag([.1,.1])
    delta,_ = hoeffding_working_covariance(covariance,pair)
    np.testing.assert_allclose(np.diag(covariance+delta),[.1,.4,.1,0.],atol=1e-15)


def test_equation_and_population_units_do_not_determine_numerical_rank():
    covariance = np.array([[-.2,.03,.04],[.03,.4,-.02],[.04,-.02,.3]])
    pair = np.diag([.1,.15])
    delta,_ = hoeffding_working_covariance(covariance,pair)
    # Study equations can be many orders smaller than population summaries.
    scales = np.array([1e-12,1e-10,1e3])
    scaled_covariance = covariance*np.outer(scales,scales)
    scaled_pair = pair*np.outer(scales[:2],scales[:2])
    changed,_ = hoeffding_working_covariance(scaled_covariance,scaled_pair)
    np.testing.assert_allclose(changed/np.outer(scales,scales),delta,rtol=1e-11,atol=1e-13)
