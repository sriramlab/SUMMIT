import numpy as np
import pytest

from summit.context.cross_trait_fit import (assemble_cross_trait_normal_equations,
    solve_cross_trait_normal_equations,cross_trait_derived,CrossTraitMomentPlan,fit_cross_trait)
from summit.context.cross_trait_gram import ChromosomeGram
from summit.context.cross_trait_oracle import dense_cross_trait_moments


def test_dense_oracle_and_redundant_residual_basis():
    rng=np.random.default_rng(901);n,m,q=71,97,3
    phi=np.c_[np.ones(n),rng.normal(size=n),rng.integers(2,size=n)]
    # Different ranks and genuinely different fixed-effect spaces are allowed
    # by the fitter/oracle, independently of the shared-span batch accelerator.
    x=np.arange(55);y=np.arange(14,n)
    ux=np.linalg.qr(np.c_[np.ones(len(x)),rng.normal(size=(len(x),3))])[0]
    uy=np.linalg.qr(np.c_[np.ones(len(y)),phi[y,1]])[0]
    d=np.c_[phi,phi[:,1]**2,phi[:,1]*phi[:,2],phi[:,2]**2]
    result=dense_cross_trait_moments(rng.normal(size=(n,m)),phi,rows_x=x,rows_y=y,
        fixed_x=ux,fixed_y=uy,phenotype_x=rng.normal(size=len(x)),phenotype_y=rng.normal(size=len(y)),
        residual_basis=d)
    moments={key:result[key] for key in ('genetic_gram','genetic_rhs','genetic_residual','residual_gram','residual_rhs')}
    equations=assemble_cross_trait_normal_equations(**moments,num_basis=q,annotation_masses=[m])
    assert equations.residual_rank==5
    solve=solve_cross_trait_normal_equations(equations)
    raw=np.block([[result['genetic_gram'],result['genetic_residual']],
                  [result['genetic_residual'].T,result['residual_gram']]])
    expected=np.linalg.lstsq(raw,np.r_[result['genetic_rhs'],result['residual_rhs']],rcond=1e-12)[0]
    np.testing.assert_allclose(solve.coefficients[:q*q],expected[:q*q],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(result['genetic_rhs'],(result['scores_x'].T@result['scores_y']/m).ravel(),rtol=1e-12,atol=1e-12)
    assert solve.rank==q*q+5
    assert solve.relative_residual<1e-12


def test_centered_nonsymmetric_orthogonal_covariance_and_zero_program():
    rng=np.random.default_rng(197);q=3
    a=rng.normal(size=(2*q,2*q));joint=a@a.T
    xx,yy,xy=joint[:q,:q],joint[q:,q:],joint[:q,q:]
    mx=np.array([.4,-.3]);my=np.array([-.7,.2]);s=np.array([[1.,.2],[.2,.8]])
    result=cross_trait_derived(xy,xx,yy,mean_x=mx,mean_y=my,context_covariance=s)
    cx=np.eye(q);cy=np.eye(q);cx[0,1:]=mx;cy[0,1:]=my
    xx0=cx@xx@cx.T;yy0=cy@yy@cy.T;xy0=cx@xy@cy.T
    tx=np.c_[-xx0[0,1:]/xx0[0,0],np.eye(q-1)]
    ty=np.c_[-yy0[0,1:]/yy0[0,0],np.eye(q-1)]
    np.testing.assert_allclose(result['h_xy'],tx@xy0@ty.T,atol=1e-13)
    reverse=cross_trait_derived(xy.T,yy,xx,mean_x=my,mean_y=mx,context_covariance=s)
    np.testing.assert_allclose(reverse['h_xy'],result['h_xy'].T,atol=1e-13)
    np.testing.assert_allclose(reverse['orthogonal_rg'],result['orthogonal_rg'])
    bx=np.array([1.,.2,.4]);by=np.array([1.,-.3,.1]);xy=.2*np.outer(bx,by)
    xx=np.outer(bx,bx)+np.diag([0,.5,.3]);yy=np.outer(by,by)+np.diag([0,.2,.7])
    zero=cross_trait_derived(xy,xx,yy,mean_x=np.zeros(2),mean_y=np.zeros(2),context_covariance=s)
    np.testing.assert_allclose(zero['h_xy'],0,atol=1e-16)
    assert zero['baseline_rg']>0


def test_mass_restored_deletions_keep_same_person_frozen():
    q=1;blocks=np.arange(4);mass=np.array([[10.],[20.],[30.],[40.]])
    off=np.array([2.,3.,4.,5.])[:,None,None];d=np.array([[7.]])
    record=dict(block_ids=blocks,block_masses=mass,block_rhs=np.arange(4.)[:,None,None,None]+1,
        block_genetic_residual=np.zeros((4,1,1,0)),gram=ChromosomeGram(off,d,{}))
    plan=CrossTraitMomentPlan([record],residual_gram=np.empty((0,0)),residual_rhs=np.empty(0),
        num_basis=q,annotation_names=('all',))
    full=plan.equations();deleted=plan.equations((2,))
    assert full.equations.matrix[0,0]==21
    np.testing.assert_allclose(deleted.equations.matrix[0,0],7+10*(100/70)**2,rtol=1e-15)
    np.testing.assert_allclose(deleted.equations.rhs,[7/70])
    fit=fit_cross_trait(plan)
    assert fit['loo_omega_xy'].shape==(4,1,1,1)
    assert fit['covariance'][0,0]>0
