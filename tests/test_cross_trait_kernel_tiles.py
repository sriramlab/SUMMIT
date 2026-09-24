import importlib.util
from pathlib import Path
import numpy as np


def test_sample_tiled_dense_oracle_has_two_distinct_projectors():
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_real_gram.py'
    spec=importlib.util.spec_from_file_location('cross_trait_real_gram',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(400);n,m,q=37,59,3
    g=rng.normal(size=(n,m));phi=np.c_[np.ones(n),rng.normal(size=(n,q-1))]
    x=np.arange(31);y=np.arange(5,n);ux=np.linalg.qr(rng.normal(size=(len(x),4)))[0]
    uy=np.linalg.qr(rng.normal(size=(len(y),6)))[0];px=np.eye(len(x))-ux@ux.T;py=np.eye(len(y))-uy@uy.T
    yx=rng.normal(size=len(x));yy=rng.normal(size=len(y));_,lx,ly=np.intersect1d(x,y,return_indices=True)
    kernel=g[x]@g[y].T/m
    gram,diagonal,rhs=module.tiled_moments(kernel,phi[x],phi[y],ux,uy,shared_x=lx,shared_y=ly,yx=yx,yy=yy,tile=7)
    dense=np.stack([px@(phi[x,a,None]*kernel*phi[y,b])@py for a in range(q) for b in range(q)])
    flat=dense.reshape(q*q,-1)
    np.testing.assert_allclose(gram,flat@flat.T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(diagonal,dense[:,lx,ly],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(rhs,np.einsum('i,aij,j->a',yx,dense,yy),rtol=1e-12,atol=1e-12)
