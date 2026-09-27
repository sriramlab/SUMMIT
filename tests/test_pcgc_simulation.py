"""Independent rejection-sampling check of the efficient qualification sampler."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("pcgc_validation", Path(__file__).parents[1]/"scripts/pcgc/validate_pcgc.py")
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


@pytest.mark.parametrize("scenario", ["S2", "S3", "S4"])
def test_conditional_sampler_matches_independent_population_rejection(scenario):
    d = validation.generate(611, scenario, 24000, 40, 100, 40)
    rng = np.random.default_rng(9145)
    # Independent ordinary population simulation, without any conditional update.
    n = 200000
    x = rng.multivariate_normal(np.zeros(40), d["R"], size=n)
    c = rng.choice([-1., 1.], n) if scenario == "S2" else rng.normal(size=n)
    g = x @ d["beta"]
    liability = g + d["gamma"]*c + rng.normal(size=n)*np.sqrt(1-sum(d["truth"]))
    y = liability > d["threshold"]
    K = validation.SCENARIOS[scenario]["K"]
    assert abs(y.mean()-K) < 5*np.sqrt(K*(1-K)/n)
    observed = np.column_stack([d["x"] @ d["beta"], d["cov"]])
    rejected = np.column_stack([g, c])
    for status in (0, 1):
        a, b = observed[d["y"] == status], rejected[y == status]
        for transform in (lambda z: z, lambda z: z*z, lambda z: (z[:, :1]*z[:, 1:])):
            aa, bb = transform(a), transform(b)
            uncertainty = np.sqrt(aa.var(axis=0)/len(aa) + bb.var(axis=0)/len(bb))
            assert np.all(np.abs(aa.mean(axis=0)-bb.mean(axis=0)) < 5*uncertainty + 1e-12)


def test_he_comparator_matches_independent_ols_beta_se_path():
    from types import SimpleNamespace
    from summit.inference.h2core import prepare_h2, fit_h2
    from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
    from summit.pcgc.research import exact_external_ld
    from summit.sumstats.binary import fit_binary_risk
    rng=np.random.default_rng(838)
    n,m=80,12
    x=rng.normal(size=(n,m))
    y=(np.arange(n) % 3 == 0).astype(float)
    a=np.ones((m,1))
    ld=exact_external_ld(rng.normal(size=(240,m)),a)
    beta,se=[],[]
    for j in range(m):
        design=np.column_stack([np.ones(n),x[:,j]])
        coef=np.linalg.lstsq(design,y,rcond=None)[0]
        residual=y-design @ coef
        beta.append(coef[1])
        se.append(np.sqrt((residual @ residual)/(n-2)*np.linalg.inv(design.T @ design)[1,1]))
    trace=SimpleNamespace(nsnps=m,nbins=1,snps=np.arange(m),annot=a,ldscores=ld,delta=None,annot_header=['all'])
    matched=SimpleNamespace(nsnps=m,snps=trace.snps,nsamp=n,n=np.full(m,n),beta=np.asarray(beta),se=np.asarray(se))
    jack=JackknifeDesign.from_trace_view(trace,JackknifeSpec.parse(4))
    fit=fit_h2(prepare_h2(trace,matched,jack),report_tau=False)
    correction=fit_binary_risk(y,.1).sensitivity[0]**2
    actual=validation.summit_he_baseline(x,a,y,.1,ld,4)
    np.testing.assert_allclose(actual['marginal_total'],fit.h2_reps[-1,0]/correction,atol=1e-12)
    np.testing.assert_allclose(actual['conditional_total_standard_error'],fit.h2[-1,1]/correction,atol=1e-12)
