import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import ndtr

spec = importlib.util.spec_from_file_location("pcgc_discrete", Path(__file__).parents[1]/"scripts/pcgc/discrete.py")
discrete = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discrete)


def test_characteristic_function_inversion_against_gaussian_and_exact_single_snp_mixture():
    cut = np.linspace(-4, 4, 17)
    distribution = discrete.LiabilityDistribution(np.zeros(1), np.zeros(1), 1.)
    np.testing.assert_allclose(distribution.survival(cut), ndtr(-cut), atol=2e-13)
    beta = .7
    residual = .6
    distribution = discrete.LiabilityDistribution(np.array([beta]), np.zeros(1), residual)
    shifts = (np.arange(3)-.6)/np.sqrt(.42)*beta
    exact = ndtr((shifts-cut[:, None])/np.sqrt(residual)) @ np.array([.49, .42, .09])
    np.testing.assert_allclose(distribution.survival(cut), exact, atol=2e-13)


def test_markov_haplotype_ld_and_liability_cdf_match_population_sampling():
    rng = np.random.default_rng(174)
    rho = np.r_[0., np.full(9, .7), 0., np.full(9, .2)]
    calls = discrete.marker_calls(rng, 100000, rho)
    assert set(np.unique(calls)) == {0, 1, 2}
    x = (calls-.6)/np.sqrt(.42)
    assert np.max(np.abs(x.mean(axis=0))) < .015
    empirical = x.T @ x / len(x)
    expected = np.zeros((20, 20))
    for start, r in ((0, .7), (10, .2)):
        expected[start:start+10, start:start+10] = r**np.abs(np.subtract.outer(np.arange(10), np.arange(10)))
    np.testing.assert_allclose(empirical, expected, atol=.02)
    beta = rng.normal(size=20)*.08
    distribution = discrete.LiabilityDistribution(beta, rho, .6)
    l = x @ beta + rng.normal(size=len(x))*np.sqrt(.6)
    cut = np.array([-.5, 0., 1., 2.])
    expected_risk = distribution.survival(cut)
    observed = np.mean(l[:, None] > cut, axis=0)
    assert np.all(np.abs(observed-expected_risk) < 5*np.sqrt(expected_risk*(1-expected_risk)/len(x)))
