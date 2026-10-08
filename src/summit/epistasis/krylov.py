"""Bounded reuse of exact covariance products for subsequent linear solves.

This changes a computational inverse, never the fitted covariance. Every solve
must still verify its residual with the complete genotype operator.
"""
import json
from zipfile import ZipFile

import numpy as np

from summit.prediction._validation import array_digest, digest


class KrylovSpace:
    def __init__(self, operator, theta, basis, *, capacity=64):
        if type(capacity) is not int or not 1<=capacity<=256:
            raise ValueError('recycled covariance space capacity must be 1..256')
        self.operator_identity=operator.identity
        self.theta=np.array(theta,float,copy=True)
        self.basis=np.asarray(basis,float)
        self.capacity=capacity
        n=len(operator.rows)
        if (self.theta.shape!=(operator.count,) or self.basis.ndim!=2 or len(self.basis)!=n
                or not np.all(np.isfinite(self.theta)) or np.any(self.theta<0)):
            raise ValueError('aligned fixed covariance and finite mean basis required')
        self.vectors=[];self.products=[]
        self.reserved_bytes=16*n*capacity

    def add(self,vector,product):
        if len(self.vectors)>=self.capacity:
            return
        d=np.asarray(vector,float);ad=np.asarray(product,float)
        if d.shape!=(len(self.basis),) or ad.shape!=d.shape:
            raise ValueError('recycled covariance product axes differ')
        d=d-self.basis@(self.basis.T@d)
        ad=ad-self.basis@(self.basis.T@ad)
        scale=np.linalg.norm(d)
        if scale>0:
            self.vectors.append(d/scale);self.products.append(ad/scale)

    def freeze(self,diagonal):
        n=len(self.basis)
        d=np.column_stack(self.vectors) if self.vectors else np.empty((n,0))
        ad=np.column_stack(self.products) if self.products else np.empty((n,0))
        return RecycledInverse(self.operator_identity,self.theta,d,ad,diagonal)


class RecycledInverse:
    """SPD two-level inverse using exact pairs D and A D.

    With E=D' A D, normalize D' A D=I on its positive identifiable span. Then
    B = D D' + (I-D (A D)') diag(A)^-1 (I-(A D) D').
    This is positive definite, and B A D=D. Neither the low-rank span nor its
    training-outcome dependence changes the final solution of A x=b.
    """
    def __init__(self,operator_identity,theta,vectors,products,diagonal):
        self.operator_identity=operator_identity
        self.theta=np.array(theta,copy=True)
        d,ad=np.asarray(vectors,float),np.asarray(products,float)
        self.diagonal=np.array(diagonal,float,copy=True)
        if (d.ndim!=2 or ad.shape!=d.shape or self.diagonal.shape!=(len(d),)
                or np.any(self.diagonal<=0)
                or not all(np.all(np.isfinite(v)) for v in (d,ad,self.diagonal))):
            raise ValueError('finite aligned positive recycled-inverse inputs required')
        if d.shape[1]:
            e=d.T@ad;e=(e+e.T)/2
            value,vector=np.linalg.eigh(e)
            tolerance=128*len(e)*np.finfo(float).eps*max(value[-1],np.finfo(float).tiny)
            if value[0]<-tolerance:
                raise ValueError('recycled covariance products are not positive semidefinite')
            keep=value>tolerance
            transform=vector[:,keep]/np.sqrt(value[keep])
            self.d=d@transform;self.ad=ad@transform
        else:
            self.d=d.copy();self.ad=ad.copy()
        self._finish()

    def _finish(self):
        if self.theta.ndim!=1 or not np.all(np.isfinite(self.theta)) or np.any(self.theta<0):
            raise ValueError('finite nonnegative recycled covariance required')
        self.identity=digest([self.operator_identity,array_digest(self.theta),array_digest(self.d),
            array_digest(self.ad),array_digest(self.diagonal),'two_level_recycled_inverse_v1'])
        self.nbytes=sum(v.nbytes for v in (self.theta,self.diagonal,self.d,self.ad))
        for v in (self.theta,self.diagonal,self.d,self.ad):v.setflags(write=False)

    def apply(self,value):
        v=np.asarray(value,float)
        if v.ndim not in (1,2) or len(v)!=len(self.diagonal) or not np.all(np.isfinite(v)):
            raise ValueError('finite aligned recycled-inverse right-hand sides required')
        coefficient=self.d.T@v
        remainder=v-self.ad@coefficient
        z=remainder/self.diagonal if v.ndim==1 else remainder/self.diagonal[:,None]
        return z-self.d@(self.ad.T@z)+self.d@coefficient


def write_recycled_inverses(path, inverses, *, identity):
    """Publish once before any dependent solve checkpoint can be written."""
    from .summary import _publish_bundle
    arrays={f'{name}_{j}':getattr(p,name) for j,p in enumerate(inverses)
        for name in ('theta','d','ad','diagonal')}
    manifest=dict(kind='summit.epistasis.recycled_inverse',schema_version=1,
        identity=identity,operators=[p.operator_identity for p in inverses],
        inverse_identities=[p.identity for p in inverses],
        digests={k:array_digest(v) for k,v in arrays.items()})
    _publish_bundle(path,manifest,arrays)


def load_recycled_inverses(path, *, identity, operator, theta, memory_bytes):
    """Load exact normalized pairs: re-diagonalizing would change checkpoint IDs."""
    with ZipFile(path) as archive:
        stored=sum(v.file_size for v in archive.infolist())
    if 3*stored+256*2**20>memory_bytes:
        raise MemoryError('recycled inverse storage exceeds preparation memory')
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['manifest']))
        fields={f'{name}_{j}' for j in range(theta.shape[1]) for name in ('theta','d','ad','diagonal')}
        if (meta.get('kind')!='summit.epistasis.recycled_inverse' or meta.get('schema_version')!=1
                or meta['identity']!=identity or set(archive.files)!={'manifest',*fields}
                or meta['operators']!=[operator.identity]*theta.shape[1]):
            raise ValueError('recycled covariance definition changed')
        arrays={k:archive[k] for k in fields}
    if meta['digests']!={k:array_digest(v) for k,v in arrays.items()}:
        raise ValueError('recycled covariance array digest mismatch')
    result=[]
    for j in range(theta.shape[1]):
        p=object.__new__(RecycledInverse)
        p.operator_identity=operator.identity
        for name in ('theta','d','ad','diagonal'):setattr(p,name,arrays[f'{name}_{j}'])
        if (not np.array_equal(p.theta,theta[:,j]) or p.d.ndim!=2
                or p.d.shape[0]!=len(operator.rows) or p.d.shape[1]>256 or p.ad.shape!=p.d.shape
                or p.diagonal.shape!=(len(operator.rows),) or np.any(p.diagonal<=0)
                or not all(np.all(np.isfinite(v)) for v in (p.d,p.ad,p.diagonal))):
            raise ValueError('invalid recycled covariance arrays')
        p._finish()
        if p.identity!=meta['inverse_identities'][j]:
            raise ValueError('recycled inverse identity mismatch')
        result.append(p)
    return result
