"""Independent liability-integral and failure-contract checks."""
import numpy as np
import pytest
from scipy.special import ndtr
from pcgc_oracle import pair_moment
from summit.pcgc.gxe import liability_inputs, prepare_gxe_moments, prepare_gxe_external, EXTERNAL_CONTRACT
from summit.sumstats.binary import prepare_binary_risk
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from test_pcgc_gxe import fixture
from scripts.pcgc.benchmark_gxe import exact_variant_moments
from pcgc_gxe_oracle import exact_rows


def test_ascertained_bivariate_liability_derivative_with_heterogeneous_scale():
    rng=np.random.default_rng(71803)
    n,m,q=9,17,3
    x=rng.normal(size=(n,m));phi=np.column_stack((np.ones(n),rng.uniform(-1,1,(n,q-1))))
    sd=1+.3*phi[:,1]**2
    k=ndtr(-1.5+.4*phi[:,1]-.3*phi[:,2])
    risk=prepare_binary_risk(np.arange(n)%2,.1,population_risk=k)
    omega=np.array([[.12,.01,-.008],[.01,.03,.004],[-.008,.004,.02]])
    covariance=(x@x.T/m)*(phi@omega@phi.T)
    K,P=risk.population_prevalence,risk.sample_prevalence
    sampling=K*(1-P)/(P*(1-K))
    phi0,sd0,psi,_,_=liability_inputs(phi,risk,sd,"pcgc")
    eps=1e-4;errors=[]
    for i in range(n):
        for j in range(i):
            rho=covariance[i,j]/(sd[i]*sd[j])
            derivative=(pair_moment(k[i],k[j],sampling,eps*rho)-pair_moment(k[i],k[j],sampling,-eps*rho))/(2*eps)
            expected=risk.sensitivity[i]*risk.sensitivity[j]*rho
            errors.append(abs(derivative-expected))
    assert max(errors)<2e-8
    np.testing.assert_allclose(psi,phi*(risk.sensitivity/sd)[:,None],rtol=1e-14)


def test_benchmark_variant_oracle_matches_independent_sample_matrices():
    x,a,risk,phi,sd=fixture(q=3,k=2,n=23,m=17)
    for method in ("pcgc","pcgc-inverse"):
        got=exact_variant_moments(x,a,risk,phi,sd,method)
        ld,rhs=exact_rows(x,a,phi,risk.z,risk.sensitivity/sd,method=="pcgc-inverse")
        np.testing.assert_allclose(got.ldscores,ld,rtol=2e-12,atol=1e-12)
        np.testing.assert_allclose(got.rhs_rows,rhs,rtol=2e-12,atol=1e-11)


def test_tiny_inverse_sensitivity_fails_before_any_pass():
    risk=prepare_binary_risk([1,0,1,0,1,0],.1,population_risk=[1e-200,.1,.1,.1,.1,.1])
    x=np.arange(24,dtype=float).reshape(6,4)
    phi=np.column_stack((np.ones(6),np.arange(6)))
    op=Operator(x)
    with pytest.raises(ValueError,match="finite"):
        prepare_gxe_moments(op,np.ones((4,1)),risk,phi,"pcgc-inverse",liability_sd=1.,native=False)
    assert op.observed_passes==0


def test_external_requires_contract_and_scale_before_traversal():
    x,a,risk,phi,sd=fixture(q=2)
    for contract,scale,match in [("unverified","test_common_scale_v1","factorization"),
                                (EXTERNAL_CONTRACT,"wrong","scales")]:
        op,ref=Operator(x),Operator(x,genotype_scale_id=scale)
        with pytest.raises(ValueError,match=match):
            prepare_gxe_external(op,ref,a,risk,phi,liability_sd=sd,factorization_contract=contract,native=False)
        assert op.observed_passes==ref.observed_passes==0


def test_resident_output_is_admitted_before_reference_passes():
    x,a,risk,phi,sd=fixture(q=3,n=800,m=400)
    op=Operator(x)
    with pytest.raises(MemoryError):
        prepare_gxe_moments(op,a,risk,phi,liability_sd=sd,probes=4000,memory_bytes=75*1024**2,native=False)
    assert op.observed_passes==0


def test_tiling_and_seed_reproducibility_under_bounded_workspace():
    x,a,risk,phi,sd=fixture(q=2,n=80,m=47)
    outputs=[]
    for width in (5,17):
        op=Operator(x)
        out,_=prepare_gxe_moments(op,a,risk,phi,liability_sd=sd,native=False,
            probes=113,seed=293,block_size=width,memory_bytes=192*1024**2)
        outputs.append(out)
        assert op.observed_passes==2
    np.testing.assert_allclose(outputs[0].ldscores,outputs[1].ldscores,rtol=2e-12,atol=1e-12)
    np.testing.assert_allclose(outputs[0].rhs_rows,outputs[1].rhs_rows,rtol=2e-12,atol=1e-11)

def test_dimension_only_plan_matches_execution_and_bounds_large_axes():
    from summit.pcgc.gxe import plan_gxe_reference
    x,a,risk,phi,sd=fixture(q=3,n=51,m=37,k=2)
    options=dict(probes=41,block_size=11,memory_bytes=192*1024**2)
    planned=plan_gxe_reference(num_samples=len(x),num_variants=x.shape[1],
        num_contexts=3,num_annotations=2,genotype_format="bed",**options)
    _,actual=prepare_gxe_moments(Operator(x),a,risk,phi,liability_sd=sd,native=False,**options)
    assert planned["peak_planned_workspace_bytes"]==actual["peak_planned_workspace_bytes"]
    assert actual["peak_planned_workspace_bytes"]<=options["memory_bytes"]
    assert actual["reference_execution"]["requested_memory_bytes"]==options["memory_bytes"]
    large=plan_gxe_reference(num_samples=100000,num_variants=1000000,
        num_contexts=3,num_annotations=2,memory_bytes=16*2**30)
    assert large["peak_planned_workspace_bytes"]<=16*2**30
    assert large["resident_directional_output_bytes"]==8*1000000*6*12

@pytest.mark.parametrize("scenario",["binary_exposure","heterogeneous_liability","ld_partitioned"])
def test_gxe_sampler_matches_independent_population_rejection(scenario):
    from scripts.pcgc.benchmark_gxe import generate
    d,p=generate(928132,scenario,16000,40,return_population=True)
    x,a,risk,phi,sd,omega=d
    rng=np.random.default_rng(49285)
    n=180000
    states=rng.choice(len(p["states"]),size=n,p=p["weights"])
    pop_phi=p["states"][states]
    pop_x=rng.normal(size=(n,40))@np.linalg.cholesky(p["R"]).T
    genetic=np.einsum("ij,ij->i",pop_x@p["beta"],pop_phi)
    effect=p["states"]@p["beta"].T
    variance=np.einsum("ij,ij->i",effect@p["R"],effect)
    liability=p["mu"][states]+genetic+rng.normal(size=n)*np.sqrt(p["scale"][states]**2-variance[states])
    y=liability>p["cut"]
    observed=np.column_stack((np.einsum("ij,ij->i",x@p["beta"],phi),phi[:,1:]))
    rejected=np.column_stack((genetic,pop_phi[:,1:]))
    for case in (False,True):
        left=observed[(risk.z>0)==case];right=rejected[y==case]
        for transform in (lambda z:z,lambda z:z*z,lambda z:z[:,:1]*z[:,1:]):
            ll,rr=transform(left),transform(right)
            mcse=np.sqrt(ll.var(axis=0)/len(ll)+rr.var(axis=0)/len(rr))
            assert np.all(np.abs(ll.mean(axis=0)-rr.mean(axis=0))<5*mcse+1e-12)
