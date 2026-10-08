"""Bounded development reference for a conditional polygenic null.

Dense covariance matrices here are limited to 2,048 people. This is not a public
estimator or a scalable implementation. It checks estimated covariance, training
conditioning and interaction response before native-operator integration.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space
from scipy.optimize import nnls, minimize


def covariance_geometry(fixed, kernels, *, diagonal_preconditioning=False):
    """Cache only genotype-dependent restricted covariance moments."""
    c, k = np.asarray(fixed, float), np.asarray(kernels, float)
    if len(c) > 2048 or k.ndim != 3 or k.shape[1:] != (len(c), len(c)):
        raise ValueError("bounded covariance reference requires N<=2048 and aligned kernels")
    if not all(np.all(np.isfinite(v)) for v in (c,k)):
        raise ValueError("finite aligned training inputs required")
    multiplier = np.ones(len(c))
    if diagonal_preconditioning:
        diagonal = np.diagonal(k,axis1=1,axis2=2)
        mass = diagonal.mean(1)
        if np.any(mass<=0):
            raise ValueError('covariance components need positive average diagonal')
        working = (diagonal/mass[:,None]).mean(0)
        if np.any(working<=0):
            raise ValueError('working covariance has nonpositive diagonal')
        multiplier = 1/np.sqrt(working)
        c = c*multiplier[:,None]
        k = k*multiplier[None,:,None]*multiplier[None,None,:]
    q = null_space(c.T, rcond=1e-11)
    if q.shape[1] < 10:
        raise ValueError("insufficient covariance estimation degrees of freedom")
    projected = np.stack([q.T @ v @ q for v in k])
    norms = np.sqrt(np.einsum("aij,aij->a", projected, projected))
    if np.any(norms <= 0):
        raise ValueError("covariance basis contains an absorbed component")
    scaled = projected/norms[:,None,None]
    h = np.einsum("aij,bij->ab",scaled,scaled)
    if np.linalg.cond(h) > 1e8:
        raise ValueError("covariance components are not separately identifiable")
    return dict(q=q, projected=projected, norms=norms, h=h, scaled=scaled,multiplier=multiplier)


def fit_covariance(y, fixed, kernels, *, method="he"):
    """Training-only estimated covariance; no generating parameters enter."""
    return fit_prepared_covariance(y, covariance_geometry(fixed,kernels), method=method)


def fit_prepared_covariance(y, geometry, *, method="he"):
    q,projected,norms,h,scaled=(geometry[k] for k in ("q","projected","norms","h","scaled"))
    y=np.asarray(y,float)
    if y.shape != (len(q),) or not np.all(np.isfinite(y)):
        raise ValueError("finite aligned training outcomes required")
    residual=q.T@(y*geometry.get('multiplier',1.))
    moments=np.einsum("i,aij,j->a",residual,scaled,residual)
    values, vectors = np.linalg.eigh(h)
    root = np.sqrt(values)[:,None] * vectors.T
    rhs = (vectors.T @ moments)/np.sqrt(values)
    initial = nnls(root,rhs)[0]/norms
    if method == "he":
        return initial, dict(method="nonnegative_HE",moment_condition=float(np.linalg.cond(h)))
    if method != "reml":
        raise ValueError("method must be he or reml")
    typical = float(np.mean(residual**2))
    initial = np.maximum(initial,typical*1e-4)
    def objective(logtheta):
        theta = np.exp(logtheta)
        covariance = np.einsum("a,aij->ij",theta,projected)
        factor = cho_factor(covariance,lower=True)
        alpha = cho_solve(factor,residual)
        inverse = cho_solve(factor,np.eye(len(residual)))
        value = .5*(2*np.log(np.diag(factor[0])).sum()+residual@alpha)
        gradient = .5*theta*(np.einsum("ij,aji->a",inverse,projected)
            -np.einsum("i,aij,j->a",alpha,projected,alpha))
        return value, gradient
    result = minimize(objective,np.log(initial),jac=True,method="L-BFGS-B",
        bounds=[(np.log(typical)-18,np.log(typical)+5)]*len(initial),
        options=dict(ftol=1e-10,gtol=1e-7,maxiter=150))
    if not result.success:
        raise ArithmeticError("bounded REML did not converge: "+result.message)
    return np.exp(result.x), dict(method="REML",iterations=result.nit,
        moment_condition=float(np.linalg.cond(h)))


def conditional_null(covariance, fixed, n_training):
    """Universal-kriging L and covariance of Y_test-L Y_train.

    The innovation is independent of every training contrast annihilating C0
    under the Gaussian random-effects null. Thus a direction learned only from
    those contrasts may be conditioned on. Estimated covariance parameters need
    separate full-pipeline validation; this identity alone does not justify them.
    """
    v,c = np.asarray(covariance,float),np.asarray(fixed,float)
    n0=n_training
    if v.shape != (len(c),len(c)) or len(c)>2048 or not 1<n0<len(c)-1:
        raise ValueError("bounded aligned training/confirmation covariance required")
    c0,c1=c[:n0],c[n0:]
    w=cho_solve(cho_factor(v[:n0,:n0],lower=True),np.eye(n0))
    wc=w@c0
    b=np.linalg.pinv(c0.T@wc,rcond=1e-11)
    if np.linalg.norm(c1-c1@b@(c0.T@wc)) > 1e-8*max(np.linalg.norm(c1),1.):
        raise ValueError("confirmation fixed effects exceed the training fixed span")
    delta=c1-v[n0:,:n0]@wc
    transfer=v[n0:,:n0]@w+delta@b@wc.T
    covariance=v[n0:,n0:]-v[n0:,:n0]@w@v[:n0,n0:]+delta@b@delta.T
    return transfer,(covariance+covariance.T)/2


def conditional_mean_tangents(covariance, kernels, fixed, n_training, y0):
    """Training-only sensitivity of the predicted mean, modulo C1.

    For restricted training precision P, dP/dtheta_k = -P K_k,00 P.
    Hence d(V10 P y0)/dtheta_k = (K_k,10 - V10 P K_k,00) P y0.
    The omitted fixed-effect derivative is in the declared C1 span. Adding
    these columns to confirmation adjustment annihilates first-order changes
    of the fitted conditional mean at the estimated covariance. It is not an
    exact correction for covariance estimation and needs pipeline validation.
    """
    v,k,c,y0 = (np.asarray(value,float) for value in
        (covariance,kernels,fixed,y0))
    n0=n_training
    if (len(c)>2048 or v.shape!=(len(c),len(c)) or k.ndim!=3
            or k.shape[1:]!=v.shape or y0.shape!=(n0,)
            or not 1<n0<len(c)-1):
        raise ValueError('bounded aligned covariance tangent inputs required')
    if not all(np.all(np.isfinite(value)) for value in (v,k,c,y0)):
        raise ValueError('finite covariance tangent inputs required')
    w=cho_solve(cho_factor(v[:n0,:n0],lower=True),np.eye(n0))
    wc=w@c[:n0]
    p=w-wc@np.linalg.pinv(c[:n0].T@wc,rcond=1e-11)@wc.T
    alpha=p@y0
    training=np.einsum('kij,j->ik',k[:,:n0,:n0],alpha)
    confirmation=np.einsum('kij,j->ik',k[:,n0:,:n0],alpha)
    return confirmation-v[n0:,:n0]@(p@training)


def innovation_score(y0,y1,f0,f1,c1,transfer,covariance):
    """Normalize by the actual response F1-L F0, with full score covariance.

    The parameter is a coefficient in the interaction-response span after
    conditional additive-null prediction. It is not the previous confirmation
    population projection coefficient or an interaction variance component.
    """
    d=np.asarray(f1)-transfer@np.asarray(f0)
    u=np.linalg.svd(c1,full_matrices=False)
    q=u[0][:,u[1]>u[1][0]*1e-11]
    d=d-q@(q.T@d)
    if d.ndim==1:
        d=d[:,None]
    h=d.T@d
    if np.linalg.cond(h)>1e8:
        raise ValueError("unidentified interaction response")
    contrast=np.linalg.solve(h,d.T)
    beta=contrast@(np.asarray(y1)-transfer@np.asarray(y0))
    cov=contrast@covariance@contrast.T
    return dict(beta=beta,covariance=cov,contrast=contrast,response=d)
