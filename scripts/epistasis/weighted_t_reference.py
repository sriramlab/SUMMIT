"""Diagnostic tails of weighted independent, unit-variance Student t5 errors.

For U=sqrt(3/5) T5, phi_U(t)=exp(-sqrt(3)|t|)(1+sqrt(3)|t|+t²).
This follows from the normal/inverse-chi-square mixture and the half-integer
Bessel recurrence (NIST DLMF 10.39.2 and 10.29.1). Independent products and
Fourier inversion give tails; no phenotype or fitted P value is rescaled.
"""
import numpy as np
from scipy.integrate import quad


def weighted_t5_sf(value,weights,*,absolute_tolerance=2e-9):
    """Return P(sum weights_i U_i > value), with numerical integration error.

    log(phi(sqrt(s))) is convex in s>=0 with value zero at zero. Thus for
    sum w_i²=1, product phi(t*w_i)<=phi(t). The explicit single-t tail bound
    controls truncation of the inversion integral, including concentrated w.
    """
    weights=np.asarray(weights,float)
    if (weights.ndim!=1 or not weights.size or not np.all(np.isfinite(weights))
            or not np.isfinite(value) or not 1e-12<=absolute_tolerance<=1e-4):
        raise ValueError('finite weighted t reference and numerical tolerance required')
    scale=np.linalg.norm(weights)
    if not scale>0:raise ValueError('positive weighted t variance required')
    w=np.abs(weights[weights!=0])/scale;x=float(value/scale);c=np.sqrt(3.)
    end=24.
    tail=np.exp(-c*end)*(1/(c*end)+1+end/c+1/c**2)/np.pi
    def integrand(t):
        z=c*t*w
        log_characteristic=np.sum(np.log1p(z+z*z/3)-z)
        return float(x*np.sinc(t*x/np.pi)*np.exp(log_characteristic))
    integral,error,info=quad(integrand,0,end,epsabs=absolute_tolerance*np.pi/2,
        epsrel=0.,limit=1000,full_output=1)[:3]
    error=float(error/np.pi+tail)
    probability=float(.5-integral/np.pi)
    if error>absolute_tolerance or probability < -error or probability > 1+error:
        raise ArithmeticError('weighted t inversion did not reach its numerical tolerance')
    return dict(probability=float(np.clip(probability,0,1)),estimated_absolute_error=error,
        truncation_error_bound=float(tail),evaluations=info['neval'])


def weighted_t5_rejection(threshold,mean,weights):
    left=weighted_t5_sf(threshold+mean,weights)
    right=weighted_t5_sf(threshold-mean,weights)
    return dict(probability=left['probability']+right['probability'],
        estimated_absolute_error=left['estimated_absolute_error']+right['estimated_absolute_error'])
