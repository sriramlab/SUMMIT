import numpy as np
import pytest

from summit.context.cross_trait_fit import CrossTraitMomentPlan, fit_cross_trait, cross_trait_derived
from summit.context.cross_trait_gram import ChromosomeGram
from summit.context.cross_trait_uncertainty import derived_jacobians, derived_uncertainty


@pytest.mark.parametrize('nblocks',[4,200])
@pytest.mark.parametrize('chromosomes',[1,2])
def test_proportional_target_deletions_preserve_full_estimate(nblocks,chromosomes):
    rng=np.random.default_rng(14);q=2;p=q*q;mass=1000.
    b=rng.normal(size=(p,2));a=np.diag([3.,4.,5.,6.]);d=np.eye(p)*12
    gram=a+b@b.T;theta=rng.normal(size=p);psi=rng.normal(size=2)
    rhs=gram@theta+b@psi;rrhs=b.T@theta+psi
    weights=np.arange(1,nblocks+1,dtype=float);weights/=weights.sum()
    records=[]
    for _ in range(chromosomes):
        records.append(dict(block_ids=np.arange(nblocks),block_masses=weights[:,None]*mass/chromosomes,
            block_rhs=(weights[:,None]*rhs*mass/chromosomes).reshape(nblocks,1,q,q),
            block_genetic_residual=(weights[:,None,None]*b*mass/chromosomes)[:,None],
            gram=ChromosomeGram(weights[:,None,None]*(gram-d)/chromosomes,d,{})))
    plan=CrossTraitMomentPlan(records,residual_gram=np.eye(2),residual_rhs=rrhs,num_basis=q,
        annotation_names=('all',),full_same_person=d)
    result=fit_cross_trait(plan)
    np.testing.assert_allclose(result['omega_xy'].ravel(),theta,atol=1e-13)
    np.testing.assert_allclose(result['loo_omega_xy'].reshape(nblocks,p),
        np.broadcast_to(theta,(nblocks,p)),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(plan.target_deletions.full_profile,a,atol=1e-13)
    assert not result['loo_genetic_mass_restored']
    with pytest.raises(ValueError,match='already use full-mass'):
        fit_cross_trait(plan,restore_mass=True)


def test_multianotation_target_block_sum_and_direct_subtraction():
    rng=np.random.default_rng(952);bcount=6;q=2;p=8;h=3
    masses=rng.uniform(10,20,(bcount,2));full=masses.sum(0);inverse=np.repeat(1/full,4)
    off=rng.normal(size=(bcount,p,p));d=np.eye(p)*10
    gr=rng.normal(size=(bcount,2,4,h));rhs=rng.normal(size=(bcount,2,q,q))
    record=dict(block_ids=np.arange(bcount),block_masses=masses,block_rhs=rhs,
        block_genetic_residual=gr,gram=ChromosomeGram(off,d,{}))
    plan=CrossTraitMomentPlan([record],residual_gram=np.eye(h),residual_rhs=np.ones(h),
        num_basis=q,annotation_names=('a','b'))
    full_eq=plan.equations().equations;target=plan.target_deletions
    b=gr.sum(0).reshape(p,h)*inverse[:,None]
    expected=full_eq.matrix[:p,:p]-b@b.T
    np.testing.assert_allclose(target.full_profile,expected,atol=1e-13)
    deleted=plan.equations((1,3)).equations
    np.testing.assert_allclose(deleted.matrix[:p,:p]-b@b.T,
        expected-target.profile_blocks[[1,3]].sum(0),atol=1e-13)
    np.testing.assert_allclose(deleted.matrix[:p,p:]@deleted.matrix[p:,p:]@deleted.matrix[p:,:p],
        b@b.T,atol=1e-13)


@pytest.mark.parametrize('method',['target_moments','legacy'])
def test_batched_rhs_matches_independent_rank_revealing_solves(method):
    from summit.context.cross_trait_fit import fit_cross_trait_rhs_batch
    rng=np.random.default_rng(917);q=2;p=4;nb=5;nr=7
    masses=np.arange(5,10.)[:,None];b=rng.normal(size=(nb,1,p,2))
    rhs=rng.normal(size=(nb,1,q,q,nr));rrhs=rng.normal(size=(2,nr))
    gram=ChromosomeGram(np.broadcast_to(np.eye(p)*3,(nb,p,p)),np.eye(p),{})
    def plan(i):
        record=dict(block_ids=np.arange(nb),block_masses=masses,block_rhs=rhs[...,i],
            block_genetic_residual=b,gram=gram)
        return CrossTraitMomentPlan([record],residual_gram=np.eye(2)*2,residual_rhs=rrhs[:,i],
            num_basis=q,annotation_names=('all',),deletion_method=method)
    batch=fit_cross_trait_rhs_batch(plan(0),[rhs],rrhs)
    for i in range(nr):
        direct=fit_cross_trait(plan(i))
        for key in ('omega_xy','loo_omega_xy','raw_loo_coefficients'):
            np.testing.assert_allclose(batch[key][i],direct[key],rtol=1e-12,atol=1e-13)


def test_heterogeneous_directed_blocks_preserve_population_moment_truth():
    # Target kernels vary by block; their inner products with frozen source
    # kernels are directional. Symmetrizing each block changes E[q_b].
    blocks=np.array([[[3.,2.],[.4,4.]],[[7.,-2.],[-.4,6.]]])
    theta=np.array([.2,.7]);masses=np.full((2,2),10.)
    rhs=(blocks@theta)*20
    record=dict(block_ids=np.arange(2),block_masses=masses,block_rhs=rhs.reshape(2,2,1,1),
        block_genetic_residual=np.empty((2,2,1,0)),
        gram=ChromosomeGram(blocks,np.zeros((2,2)),{}))
    plan=CrossTraitMomentPlan([record],residual_gram=np.empty((0,0)),residual_rhs=np.empty(0),
        num_basis=1,annotation_names=('a','b'))
    result=fit_cross_trait(plan)
    np.testing.assert_allclose(result['omega_xy'].ravel(),theta,rtol=1e-13)
    np.testing.assert_allclose(result['loo_omega_xy'].reshape(2,2),np.broadcast_to(theta,(2,2)),rtol=1e-12)


def test_unequal_annotation_mass_fractions_preserve_population_truth():
    fractions=np.array([[.1,.6],[.3,.1],[.6,.3]])
    a=np.array([[4.,1.],[1.,3.]]);d=np.array([[.4,.1],[.1,.2]])
    theta=np.array([.2,.7]);blocks=fractions[:,:,None]*a
    record=dict(block_ids=np.arange(3),block_masses=fractions*[100,200],
        block_rhs=((blocks@theta)*[100,200]).reshape(3,2,1,1),
        block_genetic_residual=np.empty((3,2,1,0)),
        gram=ChromosomeGram(blocks-fractions[:,:,None]*d,d,{}))
    plan=CrossTraitMomentPlan([record],residual_gram=np.empty((0,0)),residual_rhs=np.empty(0),
        num_basis=1,annotation_names=('a','b'))
    result=fit_cross_trait(plan)
    np.testing.assert_allclose(result['loo_omega_xy'].reshape(3,2),np.broadcast_to(theta,(3,2)),rtol=1e-12)


def fixture():
    rng=np.random.default_rng(157);q=3
    a=rng.normal(size=(2*q,2*q));joint=a@a.T+np.eye(2*q)
    xx,yy,xy=joint[:q,:q],joint[q:,q:],joint[:q,q:]
    options=dict(mean_x=[.2,-.1],mean_y=[-.4,.1],context_covariance=[[1.,.2],[.2,1.]])
    return rng,xx,yy,xy,options


def test_complex_step_jacobian_against_independent_central_difference():
    rng,xx,yy,xy,options=fixture();q=3
    jac=derived_jacobians(xy,xx,yy,**options)
    direction=rng.normal(size=27);flat=np.r_[xx.ravel(),yy.ravel(),xy.ravel()]
    step=1e-5
    def evaluate(vector):
        x,y,z=vector.reshape(3,q,q)
        return cross_trait_derived(z,x,y,**options)
    plus=evaluate(flat+step*direction);minus=evaluate(flat-step*direction)
    for name,j in jac.items():
        np.testing.assert_allclose(j@direction,(plus[name]-minus[name]).ravel()/(2*step),rtol=2e-8,atol=3e-9)


def test_paired_delta_contrast_includes_shared_covariance_and_domains():
    rng,xx,yy,xy,options=fixture();b=80
    noise=rng.normal(size=(b,3,3,3))*1e-3
    loox,looy,looz=xx+noise[:,0],yy+noise[:,1],xy+noise[:,2]
    result=derived_uncertainty(xy,xx,yy,looz,loox,looy,**options)
    j=derived_jacobians(xy,xx,yy,**options)
    contrast=j['orthogonal_rg']-j['baseline_rg']
    np.testing.assert_allclose(result['orthogonal_minus_baseline_rg_covariance'],
        contrast@result['joint_coefficient_covariance']@contrast.T,rtol=1e-13)
    for name in j:
        np.testing.assert_allclose(result[name+'_delta_covariance'],result[name+'_jackknife_covariance'],
            atol=1e-8,rtol=.005)
    invalid=xx.copy();invalid[0,0]=-100
    bad=derived_uncertainty(xy,invalid,yy,looz,loox-xx+invalid,looy,**options)
    assert np.isnan(bad['baseline_rg_covariance']).all()
    assert np.isnan(bad['orthogonal_rg_covariance']).all()


def test_general_delta_function_includes_covariance_between_numerator_and_denominator():
    from summit.context.cross_trait_uncertainty import paired_delta_covariances
    from summit.context.annotations import _jackknife_covariance
    rng=np.random.default_rng(50);point=np.array([2.,3.]);deleted=point+rng.normal(size=(20,2))*.01
    result=paired_delta_covariances(lambda a:dict(ratio=a[...,0]/a[...,1]),point,deleted)
    jac=np.array([1/3,-2/9])
    expected=jac@_jackknife_covariance(deleted)@jac
    np.testing.assert_allclose(result['ratio'],[[expected]],rtol=1e-14)
