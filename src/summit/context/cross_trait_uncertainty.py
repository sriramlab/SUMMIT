"""Paired coefficient covariance and full-point delta or nonlinear uncertainty.

Complex-step differentiation evaluates the analytic covariance formulas in
one vectorized call. No subtraction or finite-difference step selection is
needed. Variance-domain checks use the real full estimate; inadmissible
denominators remain undefined. This does not make a weak ratio Gaussian.
"""
import numpy as np

from .annotations import _jackknife_covariance
from .cross_trait_fit import cross_trait_derived


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
