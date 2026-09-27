"""Dense cross-trait oracle for validation, never the production estimator."""
from __future__ import annotations
import numpy as np


def dense_cross_trait_moments(genotype,basis,*,rows_x,rows_y,fixed_x,fixed_y,
        phenotype_x,phenotype_y,residual_basis,annotations=None,normalize=True):
    g=np.asarray(genotype,dtype=float);phi=np.asarray(basis,dtype=float)
    x,y=np.asarray(rows_x),np.asarray(rows_y);ux=np.asarray(fixed_x);uy=np.asarray(fixed_y)
    n,m=g.shape;q=phi.shape[1]
    annotations=np.ones((m,1)) if annotations is None else np.asarray(annotations,dtype=float)
    if phi.shape[0]!=n or annotations.shape[0]!=m or np.any(annotations.sum(0)<=0):
        raise ValueError('invalid oracle dimensions/annotations')
    features=[];projectors=[];phenotypes=[]
    for idx,u,z in ((x,ux,phenotype_x),(y,uy,phenotype_y)):
        if not np.allclose(u.T@u,np.eye(u.shape[1]),atol=1e-11,rtol=0):
            raise ValueError('fixed bases must be orthonormal')
        p=np.eye(len(idx))-u@u.T
        f=np.stack([p@(phi[idx,a,None]*g[idx]) for a in range(q)])
        z=p@np.asarray(z,dtype=float)
        if normalize:
            z*=np.sqrt((len(idx)-u.shape[1])/(z@z))
        features.append(f);projectors.append(p);phenotypes.append(z)
    kernels=np.stack([features[0][a]@(features[1][b]*w).T/w.sum()
                     for w in annotations.T for a in range(q) for b in range(q)])
    overlap,lx,ly=np.intersect1d(x,y,return_indices=True)
    d=np.asarray(residual_basis);raw=np.zeros((d.shape[1],len(x),len(y)))
    raw[:,lx,ly]=d[overlap].T
    residual=projectors[0]@raw@projectors[1]
    flat=kernels.reshape(len(kernels),-1);rflat=residual.reshape(len(residual),-1)
    diagonal=kernels[:,lx,ly]
    return dict(genetic_gram=flat@flat.T,genetic_residual=flat@rflat.T,
        residual_gram=rflat@rflat.T,
        genetic_rhs=np.einsum('i,kij,j->k',phenotypes[0],kernels,phenotypes[1]),
        residual_rhs=np.einsum('i,hij,j->h',phenotypes[0],residual,phenotypes[1]),
        same_person=diagonal@diagonal.T,scores_x=np.einsum('aij,i->ja',features[0],phenotypes[0]),
        scores_y=np.einsum('aij,i->ja',features[1],phenotypes[1]),
        kernels=kernels,residual_kernels=residual)
