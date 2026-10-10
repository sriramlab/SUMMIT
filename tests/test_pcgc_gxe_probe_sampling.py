"""Per-probe equations and independent-reference U-statistic oracles."""
import numpy as np
import pytest
from summit.pcgc.gxe import context_pairs,prepare_gxe_moments
from summit.pcgc.reference import ProbeGramCollector,population_ld_reference
from summit.pcgc.reference_sampling import reference_pair_covariance,ReferenceSamplingOperator
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from summit.ldscore.generalized_gxe_variant import generate_global_variant_probes
from summit.sumstats.binary import prepare_binary_risk
from prediction_helpers import prediction_threads


@pytest.mark.parametrize('native',[False,True])
def test_saved_probe_equations_match_dense_variant_products(native):
    rng = np.random.default_rng(7193)
    n,m,B = 48,29,73
    x = rng.normal(size=(n,m))
    phi = np.column_stack((np.ones(n),rng.uniform(-1,1,n)))
    a = np.column_stack((np.ones(m),rng.uniform(.1,1,m)))
    risk = prepare_binary_risk(np.arange(n)%2,.1,population_risk=np.full(n,.1))
    moments,_ = prepare_gxe_moments(Operator(x),a,risk,phi,liability_sd=1.,sampling_partners=8,
        native=native,probes=B,seed=612,block_size=11,threads=prediction_threads())
    ps = context_pairs(2)
    F = [x*feature[:,None] for feature in (phi*risk.sensitivity[:,None]).T]
    probes = generate_global_variant_probes(np.arange(m),np.arange(B),root_seed=612)
    direct = np.zeros((B,6,6))
    for b in range(2):
        cross = {(u,v):F[u].T@F[v]@(np.sqrt(a[:,b,None])*probes) for u in range(2) for v in range(2)}
        for p,(u,v) in enumerate(ps):
            for s,(r,t) in enumerate(ps):
                product = sum(cross[i,h]*cross[j,l] for i,j in ({(u,v),(v,u)}) for h,l in ({(r,t),(t,r)}))
                direct[:,p::3,b*3+s] = (a.T@product).T
    mass = np.repeat(a.sum(0),3)
    direct /= np.outer(mass,mass)
    np.testing.assert_allclose(moments.reference_probe_deviations,direct-direct.mean(0),rtol=5e-12,atol=1e-11)


@pytest.mark.parametrize('native',[False,True])
def test_rank_one_probe_equations_and_reference_sampling_share_two_passes(native):
    rng = np.random.default_rng(9122)
    n,m,B = 47,31,71
    x = rng.normal(size=(n,m))
    a = np.column_stack((np.ones(m),rng.uniform(.2,1,m)))
    collector = ProbeGramCollector(a,1,B,n,native=native,threads=prediction_threads())
    sampler = ReferenceSamplingOperator(Operator(x),a,partners=12,seed=411,threads=prediction_threads(),native=native)
    pop,_ = population_ld_reference(sampler,a,probes=B,seed=777,block_size=7,native=native,
        threads=prediction_threads(),probe_square_sink=collector.add_squared)
    signs = generate_global_variant_probes(np.arange(m),np.arange(B),root_seed=777)
    cross = x.T@x
    direct = np.stack([(a.T@(cross@(np.sqrt(a[:,k,None])*signs))**2).T for k in range(2)],axis=-1)
    direct /= np.outer(a.sum(0),a.sum(0))
    np.testing.assert_allclose(collector.deviations(),direct-direct.mean(0),rtol=3e-12,atol=2e-11)
    base = np.stack([(x*ann)@x.T/mass for ann,mass in zip(a.T,a.sum(0))],axis=-1)
    np.testing.assert_allclose(sampler.relatedness,base[np.arange(n)[:,None],sampler.partners],atol=2e-16)
    assert sampler.observed_passes == 2
    np.testing.assert_allclose(sampler.covariance(),reference_pair_covariance(sampler.relatedness),atol=0)


def test_reference_pair_covariance_and_partner_noise_correction():
    rng = np.random.default_rng(6192)
    n,m = 18,13
    x = rng.normal(size=(n,m))
    kernels = np.stack((x@x.T/m,(x[:,:6]@x[:,:6].T)/6),axis=-1)
    other = np.array([np.delete(np.arange(n),i) for i in range(n)])
    pairs = kernels[np.arange(n)[:,None],other]
    target = reference_pair_covariance(pairs,sampled=False)
    h = (pairs[:,:,:,None]*pairs[:,:,None,:]).reshape(n,n-1,4)
    rows = h.sum(1); total = rows.sum(0)
    expected = (n*(n-1)*(4*rows.T@rows-2*h.reshape(-1,4).T@h.reshape(-1,4))
                -2*(2*n-3)*np.outer(total,total))/((n-2)*(n-3)*(n*(n-1))**2)
    np.testing.assert_allclose(target,expected,rtol=2e-13,atol=1e-17)
    draws = []
    for _ in range(600):
        indices = rng.integers(0,n-1,(n,7))
        indices += indices >= np.arange(n)[:,None]
        draws.append(reference_pair_covariance(kernels[np.arange(n)[:,None],indices]))
    draws = np.asarray(draws)
    np.testing.assert_array_less(np.abs(draws.mean(0)-target),4.5*draws.std(0,ddof=1)/np.sqrt(len(draws))+1e-16)


@pytest.mark.parametrize('native',[False,True])
def test_large_reference_pair_normalization_matches_row_moments(native):
    n = 337534
    values = (1+np.arange(n)%3).astype(float)
    # Identical partners within each row make the partner-noise term zero.
    pairs = np.repeat(values[:,None,None],2,axis=1)
    expected = (4*n-6)*np.var(values**2)/((n-2)*(n-3))
    np.testing.assert_allclose(reference_pair_covariance(pairs,native=native,threads=prediction_threads()),
        [[expected]],rtol=1e-11,atol=0)
