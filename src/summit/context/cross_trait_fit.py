"""Ordered cross-trait MoM using the existing rank-revealing SUMMIT solver.

Residual kernels span only symmetric exposure products on shared people.
They are restricted to their identifiable span before the genetic/residual
system is solved. No symmetry constraint or PSD clipping is imposed on XY.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fit import ContextNormalEquations, solve_context_normal_equations
from .annotations import _jackknife_covariance
from .cross_trait_zpass import write_array_artifact


@dataclass(frozen=True)
class CrossTraitEquations:
    equations: ContextNormalEquations
    residual_transform: np.ndarray
    num_basis: int
    residual_rank: int


def assemble_cross_trait_normal_equations(*, genetic_gram, genetic_rhs,
        genetic_residual, residual_gram, residual_rhs, num_basis, annotation_masses,
        annotation_names=None, deleted_blocks=(), reference_n=0, n_x=0, n_y=0,
        residual_rtol=1e-12):
    """Assemble from moments already on ordinary kernel (1/M) normalization."""
    q=int(num_basis);masses=np.asarray(annotation_masses,dtype=float)
    k=len(masses);p=k*q*q
    gg=np.asarray(genetic_gram,dtype=float);gr=np.asarray(genetic_residual,dtype=float)
    rr=np.asarray(residual_gram,dtype=float);rhs=np.asarray(genetic_rhs,dtype=float).reshape(-1)
    r_rhs=np.asarray(residual_rhs,dtype=float).reshape(-1)
    if (q<1 or masses.ndim!=1 or np.any(masses<=0) or gg.shape!=(p,p)
            or rr.ndim!=2 or rr.shape!=(len(r_rhs),len(r_rhs))
            or gr.shape!=(p,len(r_rhs)) or rhs.shape!=(p,)
            or not all(np.isfinite(v).all() for v in (masses,gg,gr,rr,rhs,r_rhs))):
        raise ValueError('invalid cross-trait moment dimensions/values')
    if (not np.allclose(gg,gg.T,rtol=1e-12,atol=1e-10)
            or not np.allclose(rr,rr.T,rtol=1e-12,atol=1e-10)):
        raise ValueError('moment Grams must be symmetric')
    ev,u=np.linalg.eigh((rr+rr.T)/2)
    tol=max(np.max(np.abs(ev),initial=0)*residual_rtol,np.finfo(float).eps*100)
    if np.min(ev,initial=0)<-tol:
        raise ValueError('residual Gram is not positive semidefinite')
    # Keep residual units comparable to the genetic Gram. Whitening only the
    # residual block would artificially inflate full-cohort condition numbers
    # by O(N), potentially changing the genetic rank decision.
    keep=ev>tol;transform=u[:,keep]
    # Null residual directions cannot carry a material RHS or genetic moment.
    null=u[:,~keep]
    if (np.linalg.norm(gr@null)>1e-8*max(1,np.linalg.norm(gr))
            or np.linalg.norm(null.T@r_rhs)>1e-8*max(1,np.linalg.norm(r_rhs))):
        raise ValueError('moments have nonzero components in the residual null space')
    b=gr@transform;rank=int(keep.sum())
    matrix=np.block([[gg,b],[b.T,transform.T@rr@transform]])
    names=tuple(annotation_names) if annotation_names is not None else tuple(str(i) for i in range(k))
    if len(names)!=k:
        raise ValueError('annotation names disagree')
    components=tuple(f'{name}:({a},{b})' for name in names for a in range(q) for b in range(q))
    equations=ContextNormalEquations(matrix=matrix,rhs=np.r_[rhs,transform.T@r_rhs],
        traces=np.zeros(p+rank),component_names=components+tuple(f'residual:{j}' for j in range(rank)),
        genetic_count=p,annotation_masses=masses,deleted_groups=tuple(map(str,deleted_blocks)),
        reference_genetic_gram=gg,transferred_genetic_gram=gg,reference_n=reference_n,study_n=n_x)
    return CrossTraitEquations(equations,transform,q,rank)


def solve_cross_trait_normal_equations(equations, *, rtol=None, require_full_rank=True):
    """Return the in-house solver result; genetic coefficients are row-major XY."""
    if not isinstance(equations,CrossTraitEquations):
        raise TypeError('cross-trait normal equations required')
    return solve_context_normal_equations(equations.equations,rtol=rtol,require_full_rank=require_full_rank)


class CrossTraitMomentPlan:
    """Cached chromosome Grams with paired, frozen-source deletion assembly.

    Each record contains block_ids, block_masses, block_rhs[B,K,Q,Q],
    block_genetic_residual[B,K,Q²,H], and a ChromosomeGram.
    The full same-person Gram is formed AFTER summing chromosome diagonals,
    then frozen on actual overlap rows. The full-data different-person Gram
    sums chromosome contributions (no cross-chromosome LD). Deletions retain
    the existing frozen-source residual-profile change, subtracting its
    full-data value so it cannot replace the explicit full same-person term.
    """
    def __init__(self, chromosomes, *, residual_gram, residual_rhs, num_basis,
                 annotation_names, reference_n=0, n_x=0, n_y=0,full_same_person=None):
        self.chromosomes=tuple(chromosomes)
        if not self.chromosomes:
            raise ValueError('at least one chromosome required')
        self.rr=np.asarray(residual_gram,dtype=float);self.rrhs=np.asarray(residual_rhs,dtype=float)
        self.q=num_basis;self.names=tuple(annotation_names)
        self.meta=dict(reference_n=reference_n,n_x=n_x,n_y=n_y)
        self.masses=sum(c['block_masses'].sum(axis=0) for c in self.chromosomes)
        self.block_ids=np.unique(np.concatenate([c['block_ids'] for c in self.chromosomes]))
        self.rr_inverse=np.linalg.pinv(self.rr,rcond=1e-12,hermitian=True)
        p=len(self.names)*self.q*self.q
        if full_same_person is None:
            if len(self.chromosomes)!=1:
                raise ValueError('multiple chromosomes require the Gram of the summed per-person diagonals')
            full_same_person=self.chromosomes[0]['gram'].same_person
        self.same_person=np.asarray(full_same_person,dtype=float)
        if (self.same_person.shape!=(p,p) or not np.isfinite(self.same_person).all()
                or not np.allclose(self.same_person,self.same_person.T,rtol=1e-12,atol=1e-10)):
            raise ValueError('invalid full same-person Gram')
        inv=np.repeat(1/self.masses,self.q*self.q)
        bs=[c['block_genetic_residual'].sum(0).reshape(p,-1)*inv[:,None] for c in self.chromosomes]
        bt=sum(bs)
        self.full_cross_profile=(bt@self.rr_inverse@bt.T
            -sum(b@self.rr_inverse@b.T for b in bs)) if len(bs)>1 else np.zeros((p,p))

    def equations(self,deleted_blocks=()):
        if not np.isin(deleted_blocks,self.block_ids).all():
            raise ValueError('unknown deleted block')
        masks=[~np.isin(c['block_ids'],deleted_blocks) for c in self.chromosomes]
        masses=sum(c['block_masses'][take].sum(axis=0) for c,take in zip(self.chromosomes,masks))
        if np.any(masses<=0):
            raise ValueError('deletion exhausts an annotation')
        inv=np.repeat(1/masses,self.q*self.q);restore=np.repeat(self.masses/masses,self.q*self.q)
        p=len(inv);gram=self.same_person.copy();rhs=np.zeros(p);btotal=np.zeros((p,len(self.rr)))
        for c,take in zip(self.chromosomes,masks):
            g=c['gram']
            gram+=g.different_person_blocks[take].sum(axis=0)*restore[:,None]*restore[None]
            b=c['block_genetic_residual'][take].sum(axis=0).reshape(p,-1)*inv[:,None]
            if len(deleted_blocks):
                source=c['block_genetic_residual'].sum(axis=0).reshape(p,-1)*inv[:,None]
                profile=b@self.rr_inverse@source.T
                gram-=(profile+profile.T)/2
            btotal+=b
            rhs+=c['block_rhs'][take].sum(axis=0).reshape(p)*inv
        if len(deleted_blocks):
            gram+=btotal@self.rr_inverse@btotal.T-self.full_cross_profile
        return assemble_cross_trait_normal_equations(genetic_gram=(gram+gram.T)/2,genetic_rhs=rhs,
            genetic_residual=btotal,residual_gram=self.rr,residual_rhs=self.rrhs,num_basis=self.q,
            annotation_masses=masses,annotation_names=self.names,deleted_blocks=deleted_blocks,**self.meta)


def cross_trait_derived(omega_xy,omega_xx,omega_yy,*,mean_x,mean_y,context_covariance):
    """Named-exposure quantities centered separately at each trait's mean.

    Correlations outside their positive-variance domain are NaN; estimates
    outside [-1,1] are retained as raw estimates and flagged, never clipped.
    The caller supplies one common exposure covariance metric S for both
    denominators and records its cohort in provenance.
    """
    xy=np.asarray(omega_xy,dtype=float);xx=np.asarray(omega_xx,dtype=float);yy=np.asarray(omega_yy,dtype=float)
    raw_xy,raw_xx,raw_yy=xy,xx,yy
    q=xy.shape[-1];cx=np.eye(q);cy=np.eye(q)
    cx[0,1:]=mean_x;cy[0,1:]=mean_y
    xy=cx@xy@cy.T;xx=cx@xx@cx.T;yy=cy@yy@cy.T
    s=np.asarray(context_covariance,dtype=float)
    if xy.shape!=xx.shape or xy.shape!=yy.shape or xy.shape[-2:]!=(q,q) or s.shape!=(q-1,q-1):
        raise ValueError('incompatible cross/within covariance dimensions')
    with np.errstate(invalid='ignore',divide='ignore'):
        ax=xx[...,0,1:]/xx[...,0,0,None];ay=yy[...,0,1:]/yy[...,0,0,None]
        h=xy[...,1:,1:]-ax[..., :,None]*xy[...,None,0,1:]
        h-=xy[...,1:,0,None]*ay[...,None,:]
        h+=ax[..., :,None]*xy[...,0,0,None,None]*ay[...,None,:]
        hx=xx[...,1:,1:]-xx[...,1:,0,None]*xx[...,None,0,1:]/xx[...,0,0,None,None]
        hy=yy[...,1:,1:]-yy[...,1:,0,None]*yy[...,None,0,1:]/yy[...,0,0,None,None]
        trace=np.einsum('ij,...ji->...',s,h)
        tx=np.einsum('ij,...ji->...',s,hx);ty=np.einsum('ij,...ji->...',s,hy)
        def ratio(c,vx,vy):
            return np.where((vx>0)&(vy>0),c/np.sqrt(vx*vy),np.nan)
        baseline=ratio(raw_xy[...,0,0],raw_xx[...,0,0],raw_yy[...,0,0])
        centered_baseline=ratio(xy[...,0,0],xx[...,0,0],yy[...,0,0])
        response=ratio(np.diagonal(xy,axis1=-2,axis2=-1)[...,1:],
                       np.diagonal(xx,axis1=-2,axis2=-1)[...,1:],np.diagonal(yy,axis1=-2,axis2=-1)[...,1:])
        baseline_valid=(xx[...,0,0]>0)&(yy[...,0,0]>0)
        orthogonal=np.where(baseline_valid,ratio(trace,tx,ty),np.nan)
    return dict(omega_centered=xy,baseline_covariance=raw_xy[...,0,0],baseline_rg=baseline,
        centered_baseline_covariance=xy[...,0,0],centered_baseline_rg=centered_baseline,
        response_block=xy[...,1:,1:],response_rg=response,h_xy=h,h_xx=hx,h_yy=hy,
        orthogonal_trace=trace,orthogonal_rg=orthogonal,
        response_minus_baseline_rg=response-baseline[...,None],
        orthogonal_minus_baseline_rg=orthogonal-baseline,
        baseline_rg_admissible=np.isfinite(baseline)&(np.abs(baseline)<=1),
        response_rg_admissible=np.isfinite(response)&(np.abs(response)<=1),
        orthogonal_baseline_variances_positive=baseline_valid,
        orthogonal_rg_admissible=np.isfinite(orthogonal)&(np.abs(orthogonal)<=1))


def fit_cross_trait(plan, *, rtol=None):
    full=plan.equations();point=solve_cross_trait_normal_equations(full,rtol=rtol)
    p=full.equations.genetic_count
    loo=[];loo_full=[];loo_rank=[];loo_condition=[]
    for block in plan.block_ids:
        result=solve_cross_trait_normal_equations(plan.equations((block,)),rtol=rtol)
        loo_full.append(result.coefficients)
        loo.append(result.coefficients[:p]);loo_rank.append(result.rank);loo_condition.append(result.condition_number)
    loo=np.asarray(loo)
    loo_full=np.asarray(loo_full)
    return dict(omega_xy=point.coefficients[:p].reshape(-1,plan.q,plan.q),
        loo_omega_xy=loo.reshape(len(loo),-1,plan.q,plan.q),
        covariance=_jackknife_covariance(loo),block_ids=plan.block_ids,
        coefficients=point.coefficients,residual_coefficients=full.residual_transform@point.coefficients[p:],
        loo_coefficients=loo_full,coefficient_covariance=_jackknife_covariance(loo_full),
        residual_transform=full.residual_transform,
        loo_residual_coefficients=loo_full[:,p:]@full.residual_transform.T,
        same_person_gram=plan.same_person,
        frozen_full_cross_chromosome_profile=plan.full_cross_profile,
        condition_number=np.array(point.condition_number),rank=np.array(point.rank),
        residual_rank=np.array(full.residual_rank),relative_residual=np.array(point.relative_residual),
        minimum_gram_eigenvalue=np.array(point.minimum_gram_eigenvalue),
        loo_rank=np.asarray(loo_rank),loo_condition=np.asarray(loo_condition))


def write_cross_trait_fit(path,fit,*,provenance,within_x=None,within_y=None,
                          mean_x=None,mean_y=None,context_covariance=None):
    arrays=dict(fit)
    if within_x is not None and within_y is not None:
        for within in (within_x,within_y):
            if 'block_ids' not in within or not np.array_equal(within['block_ids'],fit['block_ids']):
                raise ValueError('within/cross deletion blocks must be explicitly paired')
        point=cross_trait_derived(fit['omega_xy'],within_x['omega'],within_y['omega'],
            mean_x=mean_x,mean_y=mean_y,context_covariance=context_covariance)
        deleted=cross_trait_derived(fit['loo_omega_xy'],within_x['loo'],within_y['loo'],
            mean_x=mean_x,mean_y=mean_y,context_covariance=context_covariance)
        for name,value in point.items():
            arrays[name]=value;arrays['loo_'+name]=deleted[name]
            if np.asarray(value).dtype.kind=='f':
                rows=deleted[name].reshape(len(fit['block_ids']),-1)
                arrays[name+'_covariance']=_jackknife_covariance(rows)
        arrays['context_covariance']=context_covariance
        arrays['mean_x']=mean_x;arrays['mean_y']=mean_y
    write_array_artifact(path,kind='summit.cross_trait.fit',arrays=arrays,provenance=provenance)
