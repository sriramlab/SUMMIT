import dataclasses
import numpy as np
import pytest

from summit.pcgc.gxe import (
    GxEMoments, prepare_gxe_moments, prepare_gxe_external, fit_gxe,
    liability_inputs, context_pair_gram, evaluate_contexts, EXTERNAL_CONTRACT,
)
from summit.sumstats.binary import prepare_binary_risk
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec
from pcgc_gxe_oracle import exact_rows, pair_equations, pairs
from prediction_helpers import prediction_threads


def fixture(q=3,k=2,n=39,m=29):
    rng=np.random.default_rng(8162)
    x=rng.normal(size=(n,m))
    phi=np.column_stack((np.ones(n),rng.uniform(-1,1,(n,q-1))))
    a=rng.uniform(.1,1,(m,k))
    risk=prepare_binary_risk(np.arange(n)%2,.1,population_risk=rng.uniform(.03,.3,n))
    sd=1+.2*phi[:,-1]**2
    return x,a,risk,phi,sd


def exact_moments(method="pcgc",q=3,k=2):
    x,a,risk,phi,sd=fixture(q=q,k=k)
    ld,rhs=exact_rows(x,a,phi,risk.z,risk.sensitivity/sd,method=="pcgc-inverse")
    moments=GxEMoments(a,ld,rhs,q,len(x),method,phi.T@phi/len(phi),1.3)
    return moments,(x,a,risk,phi,sd)


@pytest.mark.parametrize("inverse",[False,True])
def test_equations_and_target_only_deletions_against_explicit_people(inverse):
    m,(x,a,risk,phi,sd)=exact_moments("pcgc-inverse" if inverse else "pcgc")
    blocks=np.arange(len(a))%4
    for deleted in (None,0,1):
        retained=None if deleted is None else blocks!=deleted
        H,b=pair_equations(x,a,phi,risk.z,risk.sensitivity/sd,inverse,retained)
        got_H,got_b=m.equations(retained)
        np.testing.assert_allclose(got_H,H,rtol=2e-13,atol=2e-12)
        np.testing.assert_allclose(got_b,b,rtol=2e-13,atol=2e-12)
    result=fit_gxe(m,block_ids=blocks)
    loo=np.array([np.linalg.solve(*pair_equations(x,a,phi,risk.z,risk.sensitivity/sd,inverse,blocks!=j)) for j in range(4)])
    np.testing.assert_allclose(result["jackknife_replicates"],loo,rtol=2e-10,atol=2e-11)
    centered=loo-loo.mean(axis=0)
    np.testing.assert_allclose(result["covariance"],.75*centered.T@centered,rtol=2e-10,atol=2e-11)
    states=np.array([[1,.2,-.1],[1,-.6,.4]])
    evaluated=evaluate_contexts(result,states)
    values=np.array([evaluate_contexts(dict(result,omega=np.array([np.array([[t[0],t[3],t[4]],[t[3],t[1],t[5]],[t[4],t[5],t[2]]]) for t in v.reshape(2,6)]).tolist()),states)["genetic_covariance"] for v in loo])
    np.testing.assert_allclose(evaluated["standard_errors"],np.sqrt(.75*((values-values.mean(axis=0))**2).sum(axis=0)),rtol=2e-10)


@pytest.mark.parametrize("method",["pcgc","pcgc-inverse","pcgc-basis"])
@pytest.mark.parametrize("native",[False,True])
def test_streamed_generalized_moments_fixed_probe_oracle(method,native):
    x,a,risk,phi,sd=fixture()
    extra=dict(basis=np.column_stack((np.ones(len(x)),risk.sensitivity)),coefficients=[0,1]) if method=="pcgc-basis" else {}
    op=Operator(x)
    threads=prediction_threads() if native else 1
    got,diagnostics=prepare_gxe_moments(op,a,risk,phi,method,liability_sd=sd,
        probes=43,seed=172,block_size=7,native=native,threads=threads,**extra)
    w=risk.sensitivity/sd
    psi=phi if method=="pcgc-inverse" else phi*w[:,None]
    F=[psi[:,i,None]*x for i in range(phi.shape[1])]
    probe=GlobalVariantProbeSpec(root_seed=172,probe_offset=0,probe_count=43).generate(np.arange(len(a)))
    ps=pairs(phi.shape[1])
    expected=np.zeros_like(got.ldscores)
    for k,annot in enumerate(a.T):
        for p,(u,v) in enumerate(ps):
            for s,(c,d) in enumerate(ps):
                for i,j in ([(u,v)] if u==v else [(u,v),(v,u)]):
                    for h,l in ([(c,d)] if c==d else [(c,d),(d,c)]):
                        left=(F[j].T@F[h])@(np.sqrt(annot)[:,None]*probe)
                        right=(F[i].T@F[l])@(np.sqrt(annot)[:,None]*probe)
                        expected[:,p,k*len(ps)+s]+=np.mean(left*right,axis=1)/len(x)**2
                target_diag=(1 if u==v else 2)*F[u]*F[v]
                source_diag=(1 if c==d else 2)*((F[c]*F[d])@annot)
                expected[:,p,k*len(ps)+s]-=target_diag.T@source_diag/len(x)**2
    np.testing.assert_allclose(got.ldscores,expected,rtol=2e-11,atol=1e-12)
    _,rhs=exact_rows(x,a,phi,risk.z,w,method=="pcgc-inverse")
    np.testing.assert_allclose(got.rhs_rows,rhs,rtol=2e-12,atol=2e-11)
    assert op.observed_passes==2
    assert diagnostics["exact_same_person_per_target_snp"]
    assert diagnostics["peak_planned_workspace_bytes"]<=2**30


def test_basis_agreement_and_additive_reduction():
    from summit.pcgc.reference import prepare_moments
    x,a,risk,_,_=fixture(q=1)
    phi=np.ones((len(x),1))
    common=dict(probes=51,seed=827,block_size=9,native=False)
    old,_=prepare_moments(Operator(x),a,risk,**common)
    new,_=prepare_gxe_moments(Operator(x),a,risk,phi,liability_sd=1.,**common)
    np.testing.assert_allclose(new.ldscores[:,0,:],old.ldscores,rtol=2e-13,atol=1e-13)
    np.testing.assert_allclose(new.rhs_rows[:,0],old.rhs_rows,rtol=2e-13,atol=1e-12)
    basis,_=prepare_gxe_moments(Operator(x),a,risk,phi,"pcgc-basis",liability_sd=1.,
        basis=risk.sensitivity[:,None],coefficients=[1.],**common)
    np.testing.assert_array_equal(basis.ldscores,new.ldscores)


def test_external_pair_factor_and_fixed_probe_transfer():
    from summit.pcgc.reference import population_ld_reference
    x,a,risk,phi,sd=fixture(q=2)
    psi=phi*(risk.sensitivity/sd)[:,None]
    contexts=np.array([np.outer(psi[:,u],psi[:,v]) if u==v else np.outer(psi[:,u],psi[:,v])+np.outer(psi[:,v],psi[:,u]) for u,v in pairs(2)])
    for c in contexts: np.fill_diagonal(c,0)
    expected_factor=np.einsum("pij,sij->ps",contexts,contexts)
    np.testing.assert_allclose(context_pair_gram(psi),expected_factor,rtol=3e-14)
    ref=np.random.default_rng(912).normal(size=(53,x.shape[1]))
    options=dict(probes=79,seed=132,block_size=8,native=False)
    study_op,ref_op=Operator(x),Operator(ref)
    # Array operators use a data-derived identity; declare a shared known scale.
    ref_op.genotype_scale_id=study_op.genotype_scale_id
    got,diagnostics=prepare_gxe_external(study_op,ref_op,a,risk,phi,
        liability_sd=sd,factorization_contract=EXTERNAL_CONTRACT,**options)
    external,_=population_ld_reference(Operator(ref),a,**options)
    want=(external[:,None,:,None]*expected_factor[None,:,None,:]/len(x)**2).reshape(got.ldscores.shape)
    np.testing.assert_allclose(got.ldscores,want,rtol=3e-14,atol=1e-14)
    assert study_op.observed_passes==1 and ref_op.observed_passes==2
    assert "approximation" in diagnostics["reference_kind"]


def test_basis_reparameterization_exact_and_population_variance():
    m,(x,a,risk,phi,sd)=exact_moments(q=2,k=1)
    T=np.array([[1,.3],[0,1.7]])
    transformed=phi@T
    ld,rhs=exact_rows(x,a,transformed,risk.z,risk.sensitivity/sd)
    alt=GxEMoments(a,ld,rhs,2,len(x),"pcgc",T.T@m.population_second_moment@T,1.3)
    f,g=fit_gxe(m),fit_gxe(alt)
    np.testing.assert_allclose(T@np.array(g["omega"])[0]@T.T,np.array(f["omega"])[0],atol=1e-12)
    assert f["population_heritability"]==pytest.approx(g["population_heritability"])


def test_invalid_inputs_fail_before_traversal():
    x,a,risk,phi,sd=fixture()
    for options,match in [
        (dict(liability_sd=None),"declare"),
        (dict(liability_sd=-1),"positive"),
        (dict(liability_sd=sd,memory_bytes=1),"memory"),
        (dict(liability_sd=sd,method="liability"),"four|requires"),
        (dict(liability_sd=sd,method="pcgc-basis",basis=np.ones((len(x),1)),coefficients=[1]),"span"),
    ]:
        op=Operator(x)
        with pytest.raises((ValueError,MemoryError),match=match):
            prepare_gxe_moments(op,a,risk,phi,native=False,**options)
        assert op.observed_passes==0
    with pytest.raises(ValueError,match="rank deficient"):
        liability_inputs(np.ones((len(x),2)),risk,1.,"pcgc")
    m,_=exact_moments()
    with pytest.raises(ValueError,match="axes"):
        dataclasses.replace(m,rhs_rows=m.rhs_rows[:,0:1])
    with pytest.raises(ValueError,match="semidefinite"):
        dataclasses.replace(m,population_second_moment=-np.eye(3))
