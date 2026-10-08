"""Tiny independent dense oracle; never imported by production code."""
import numpy as np


def pairs(q):
    return [(u,u) for u in range(q)]+[(u,v) for u in range(q) for v in range(u+1,q)]


def exact_rows(x, annotations, phi, z, weight, inverse=False):
    # Explicit feature-kernel products on both sample and variant axes.
    n,m = x.shape
    psi = phi if inverse else phi*weight[:,None]
    y = z/weight if inverse else z
    F = [psi[:,j,None]*x for j in range(phi.shape[1])]
    ps = pairs(phi.shape[1])
    kernels = []
    for a in annotations.T:
        for u,v in ps:
            B = (F[u]*a)@F[v].T
            if u != v:
                B += (F[v]*a)@F[u].T
            np.fill_diagonal(B,0)
            kernels.append(B)
    ld = np.empty((m,len(ps),len(kernels)))
    rhs = np.empty((m,len(ps)))
    for j in range(m):
        for p,(u,v) in enumerate(ps):
            B = np.outer(F[u][:,j],F[v][:,j])
            if u != v:
                B += B.T
            np.fill_diagonal(B,0)
            rhs[j,p] = y@B@y
            ld[j,p] = [np.sum(B*C)/n**2 for C in kernels]
    return ld,rhs


def pair_equations(x,a,phi,z,w,inverse=False,retained=None):
    n,m = x.shape
    if retained is None:
        retained = np.ones(m,dtype=bool)
    ps = pairs(phi.shape[1]); p=len(ps); c=a.shape[1]*p
    H,b = np.zeros((c,c)),np.zeros(c)
    for i in range(n):
        for j in range(i):
            contexts=np.array([(1 if u==v else 0)*phi[i,u]*phi[j,v] if u==v else
                phi[i,u]*phi[j,v]+phi[i,v]*phi[j,u] for u,v in ps])
            source=np.outer((x[i]*x[j])@a/a.sum(axis=0),contexts).ravel()
            target=np.outer((x[i,retained]*x[j,retained])@a[retained]/a[retained].sum(axis=0),contexts).ravel()
            y=z[i]*z[j]
            if inverse:
                y/=w[i]*w[j]
            else:
                source*=w[i]*w[j];target*=w[i]*w[j]
            H+=2*np.outer(target,source); b+=2*target*y
    return H,b
