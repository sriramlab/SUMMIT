"""Paired coefficient covariance and full-point delta or nonlinear uncertainty.

Complex-step differentiation evaluates the analytic covariance formulas in
one vectorized call. No subtraction or finite-difference step selection is
needed. Variance-domain checks use the real full estimate; inadmissible
denominators remain undefined. This does not make a weak ratio Gaussian.
"""
import numpy as np

from .annotations import _jackknife_covariance
from .cross_trait_fit import cross_trait_derived


def paired_delta_covariances(function, point, deleted):
    """Delta covariance for a dictionary-valued analytic array function.

    The function must preserve complex dtype and use real-part domain checks.
    Deletions must already have the same coefficient units as the full fit.
    """
    point=np.asarray(point,dtype=float);deleted=np.asarray(deleted,dtype=float)
    if (deleted.ndim!=point.ndim+1 or deleted.shape[1:]!=point.shape or len(deleted)<2
            or not np.isfinite(point).all() or not np.isfinite(deleted).all()):
        raise ValueError('finite point and paired deletion arrays required')
    p=point.size;step=1e-30
    perturbed=(point.ravel()[None]+1j*step*np.eye(p)).reshape((p,)+point.shape)
    values=function(point);derivatives=function(perturbed)
    covariance=_jackknife_covariance(deleted.reshape(len(deleted),-1))
    result={}
    for name,value in derivatives.items():
        if np.asarray(value).dtype.kind!='c':continue
        jacobian=np.asarray(value).imag.reshape(p,-1).T/step
        jacobian=np.where(np.isfinite(values[name]).ravel()[:,None],jacobian,np.nan)
        cov=jacobian@covariance@jacobian.T
        result[name]=(cov+cov.T)/2
    return result


def derived_jacobians(omega_xy, omega_xx, omega_yy, **kwargs):
    arrays=[np.asarray(a,dtype=float) for a in (omega_xx,omega_yy,omega_xy)]
    if any(a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('finite, matching primitive covariance arrays required')
    shape=arrays[0].shape;p=arrays[0].size
    flat=np.concatenate([a.ravel() for a in arrays])
    step=1e-30
    perturbed=flat[None].astype(complex)+1j*step*np.eye(3*p)
    xx,yy,xy=[perturbed[:,i*p:(i+1)*p].reshape((3*p,)+shape) for i in range(3)]
    values=cross_trait_derived(xy,xx,yy,**kwargs)
    point=cross_trait_derived(arrays[2],arrays[0],arrays[1],**kwargs)
    return {key:np.where(np.isfinite(point[key]).ravel()[:,None],
                value.imag.reshape(3*p,-1).T/step,np.nan)
            for key,value in values.items() if value.dtype.kind=='c'}


def derived_uncertainty(omega_xy,omega_xx,omega_yy,loo_xy,loo_xx,loo_yy,*,
                        mean_x,mean_y,context_covariance,method='delta'):
    if method not in ('delta','jackknife'):
        raise ValueError('uncertainty method must be delta or jackknife')
    point_arrays=[np.asarray(a,dtype=float) for a in (omega_xx,omega_yy,omega_xy)]
    loo_arrays=[np.asarray(a,dtype=float) for a in (loo_xx,loo_yy,loo_xy)]
    b=len(loo_arrays[0])
    if b<2 or any(a.shape!=(b,)+v.shape or not np.isfinite(a).all()
                  for a,v in zip(loo_arrays,point_arrays)):
        raise ValueError('at least two finite paired coefficient deletions required')
    s=np.asarray(context_covariance,dtype=float)
    if (s.ndim!=2 or s.shape[0]!=s.shape[1] or not np.isfinite(s).all()
            or not np.allclose(s,s.T,rtol=1e-12,atol=1e-14)
            or np.linalg.eigvalsh(s).min(initial=0)<-1e-12*max(1,np.linalg.norm(s,2))):
        raise ValueError('context covariance must be finite symmetric positive semidefinite')
    kwargs=dict(mean_x=mean_x,mean_y=mean_y,context_covariance=s)
    covariance=_jackknife_covariance(np.concatenate([a.reshape(b,-1) for a in loo_arrays],axis=1))
    jacobians=derived_jacobians(omega_xy,omega_xx,omega_yy,**kwargs)
    deleted=cross_trait_derived(loo_xy,loo_xx,loo_yy,**kwargs)
    result=dict(joint_coefficient_covariance=covariance,
        joint_coefficient_order=np.array(['xx','yy','xy']),uncertainty_method=np.array(method))
    for name,jacobian in jacobians.items():
        delta=jacobian@covariance@jacobian.T
        nonlinear=_jackknife_covariance(deleted[name].reshape(b,-1))
        result[name+'_delta_covariance']=(delta+delta.T)/2
        result[name+'_jackknife_covariance']=nonlinear
        result[name+'_covariance']=result[name+'_'+method+'_covariance']
        result[name+'_finite_deletions']=np.isfinite(deleted[name]).sum(0)
    return result
