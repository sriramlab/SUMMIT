import numpy as np
import pytest
from scipy.stats import norm

from pcgc_oracle import pair_moment, risks
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk, fit_binary_risk_logistic, risk_objective


def test_risk_weights_match_original_pcgc_and_liability_derivative():
    k = np.array([.001, .01, .1, .3])
    y = np.array([0., 1., 0., 1.])
    risk = prepare_binary_risk(y, .05, population_risk=k)
    z, d = risks(y, .05, k)
    np.testing.assert_allclose(risk.z, z, rtol=2e-14)
    np.testing.assert_allclose(risk.sensitivity, d, rtol=2e-14)
    a = .05/.95
    for i in range(len(k)):
        for j in range(len(k)):
            derivative = (pair_moment(k[i], k[j], a, 1e-5)-pair_moment(k[i], k[j], a, -1e-5))/2e-5
            assert derivative == pytest.approx(d[i]*d[j], rel=2e-8)


def test_constant_risk_factor_and_case_control_recoding():
    y = np.r_[np.ones(13), np.zeros(27)]
    risk = fit_binary_risk(y, .02)
    expected = norm.pdf(norm.isf(.02))**2*y.mean()*(1-y.mean())/(.02*.98)**2
    np.testing.assert_allclose(risk.sensitivity**2, expected)
    opposite = fit_binary_risk(1-y, .98)
    np.testing.assert_allclose(opposite.z, -risk.z)
    np.testing.assert_allclose(opposite.sensitivity, risk.sensitivity)
    assert risk.covariate_variance == 0


def test_ascertainment_likelihood_gradient():
    rng = np.random.default_rng(18)
    X = np.column_stack([np.ones(80), rng.normal(size=(80, 2))])
    y = rng.binomial(1, .4, 80)
    coef = np.array([-2., .4, -.2])
    f, g = risk_objective(coef, X, y, -2.)
    step = np.eye(coef.size) * 1e-5
    numerical = np.array([(risk_objective(coef + h, X, y, -2.)[0] -
                           risk_objective(coef - h, X, y, -2.)[0]) / 2e-5 for h in step])
    np.testing.assert_allclose(g, numerical, atol=1e-10)


def test_binary_strata_risk_fit_recovers_saturated_sample_proportions():
    cov = np.repeat([[-1.], [1.]], 200, axis=0)
    y = np.r_[np.ones(30), np.zeros(170), np.ones(130), np.zeros(70)]
    risk = fit_binary_risk(y, .05, cov)
    np.testing.assert_allclose(risk.sample_risk, np.repeat([.15, .65], 200), atol=2e-7)
    assert risk.covariate_variance > 0
    assert not risk.z.flags.writeable
    changed_units = fit_binary_risk(y, .05, cov*1e-20)
    np.testing.assert_allclose(changed_units.sample_risk, risk.sample_risk, atol=1e-12)
    logistic = fit_binary_risk_logistic(y, .05, cov)
    np.testing.assert_allclose(logistic.population_risk, risk.population_risk, atol=2e-8)
    np.testing.assert_allclose(logistic.sensitivity, risk.sensitivity, atol=2e-8)
    assert logistic.source == "sample_logistic_backtransform"


@pytest.mark.parametrize('y', [[1,1,1], [0,0,0], [0,1,2], [0,1,np.nan]])
def test_invalid_phenotypes_rejected(y):
    with pytest.raises(ValueError):
        fit_binary_risk(y, .1)


def test_invalid_risk_designs_rejected():
    y = np.tile([0., 1.], 20)
    with pytest.raises(ValueError, match='constant'):
        fit_binary_risk(y, .1, np.ones((40, 1)))
    with pytest.raises(ValueError, match='rank deficient'):
        fit_binary_risk(y, .1, np.column_stack([y, y]))
    with pytest.raises(ValueError):
        prepare_binary_risk(y, .1, population_risk=np.zeros(40))
    with pytest.raises((ValueError, RuntimeError)):
        fit_binary_risk(y, .1, y[:, None])
    with pytest.raises(ValueError, match="separated"):
        fit_binary_risk([0, 0, 1, 0, 1, 1], .1, np.array([-1, -1, 0, 0, 1, 1])[:, None])
