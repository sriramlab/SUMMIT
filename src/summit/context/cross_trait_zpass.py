"""Small projection moments collected in one traversal or the shared study pass.

Z[j,a,c] = sum_i G[i,j] phi[i,a] U[i,c]. No Z rows are persisted. Annotation
weights must be retained on the target as well as the source axis: unweighted
target-block V matrices do not suffice for annotation-specific corrections.

The quadratic antisymmetric correction repairs the symmetrized target-block
Gram for a single annotation when supplied exact (not finite-probe) saved
moments. Cross-annotation mixed symmetric/antisymmetric terms are not repaired
by this quadratic correction. The repair helper requires a single annotation;
callers must establish target/source annotation identity from provenance.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from .spec import array_sha256, canonical_json
from .cross_trait_gram import saved, reconstruct_ordered


def write_array_artifact(path, *, kind, arrays, provenance):
    """Exclusive, checksummed NumPy publication without pickled payloads."""
    if not provenance or not kind or 'manifest' in arrays:
        raise ValueError('artifact kind, provenance and nonreserved arrays are required')
    values={name:np.asarray(value) for name,value in arrays.items()}
    if any(v.dtype.hasobject for v in values.values()):
        raise ValueError('object arrays cannot be published')
    metadata=dict(kind=kind,schema_version=1,provenance=provenance,
                  arrays={k:array_sha256(v) for k,v in values.items()})
    with Path(path).open('xb') as f:
        np.savez(f,manifest=np.array(canonical_json(metadata)),**values)


def load_array_artifact(path, *, kind):
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(str(z['manifest']))
        if m['kind']!=kind or m['schema_version']!=1 or not m['provenance']:
            raise ValueError('artifact kind/version/provenance mismatch')
        if set(z.files)!={'manifest',*m['arrays']}:
            raise ValueError('artifact array set mismatch')
        arrays={name:z[name] for name in m['arrays']}
    if any(array_sha256(v)!=m['arrays'][name] for name,v in arrays.items()):
        raise ValueError('artifact array checksum mismatch')
    return arrays,m['provenance']


class ZMomentAccumulator:
    """Accumulate annotation-weighted block V, with global W as its sum."""
    def __init__(self, block_ids, num_annotations, num_basis, fixed_rank):
        self.block_ids=np.asarray(block_ids)
        if (self.block_ids.ndim!=1 or self.block_ids.dtype.kind not in 'iu'
                or not len(self.block_ids) or np.any(np.diff(self.block_ids)<=0)):
            raise ValueError('block IDs must be sorted unique integers')
        if min(num_annotations,num_basis)<1 or fixed_rank<0:
            raise ValueError('invalid Z dimensions')
        self.k,self.q,self.c=num_annotations,num_basis,fixed_rank
        self.products=np.zeros((len(self.block_ids),self.k,self.q,self.q,self.c,self.c))
        self.variants=0

    def add(self,z,annotations,groups):
        z=np.asarray(z,dtype=float);a=np.asarray(annotations,dtype=float);g=np.asarray(groups)
        if (z.shape!=(len(g),self.q,self.c) or a.shape!=(len(g),self.k)
                or not np.isfinite(z).all() or not np.isfinite(a).all() or np.any(a<0)):
            raise ValueError('Z tile dimensions/values disagree')
        labels=np.unique(g)
        if not np.isin(labels,self.block_ids).all():
            raise ValueError('unknown target block')
        flat=z.reshape(len(z),self.q*self.c)
        for label in labels:
            take=g==label;f=flat[take]
            value=f.T[None] @ (a[take].T[:,:,None]*f[None])
            value=value.reshape(self.k,self.q,self.c,self.q,self.c).transpose(0,1,3,2,4)
            self.products[np.searchsorted(self.block_ids,label)]+=value
        self.variants+=len(z)

    @property
    def global_products(self):
        return self.products.sum(axis=0)

    def write(self,path,*,provenance):
        write_array_artifact(path,kind='summit.cross_trait.z_moments',
            arrays=dict(block_ids=self.block_ids,block_products=self.products,
                        global_products=self.global_products),
            provenance=dict(provenance,variants=self.variants,
                exactness_scope='single_annotation_symmetrized_target_gram',
                target_annotation_weighted=True))


def antisymmetric_gram(target_products, source_products):
    """Return T^A[(ab),(cd)] = <A_ac,A_bd>, on unnormalized G scale.

    Supports arbitrary leading batch axes; target/source products have last
    axes Q,Q,C,C. No large SNP-by-SNP directional LD matrices are formed.
    """
    v=np.asarray(target_products,dtype=float);w=np.asarray(source_products,dtype=float)
    if (v.ndim<4 or w.ndim<4 or v.shape[-4:]!=w.shape[-4:]
            or v.shape[-4]!=v.shape[-3] or v.shape[-2]!=v.shape[-1]
            or not np.isfinite(v).all() or not np.isfinite(w).all()):
        raise ValueError('invalid target/source Z moments')
    q=v.shape[-4]
    # C^A[ab,cd]; trace(A B) contracts the two small covariate axes.
    ca=.25*(np.einsum('...caij,...bdji->...abcd',v,w,optimize=True)
        -np.einsum('...daij,...bcji->...abcd',v,w,optimize=True)
        -np.einsum('...cbij,...adji->...abcd',v,w,optimize=True)
        +np.einsum('...dbij,...acji->...abcd',v,w,optimize=True))
    axes=list(range(ca.ndim));axes[-3],axes[-2]=axes[-2],axes[-3]
    return ca.transpose(axes).reshape(*ca.shape[:-4],q*q,q*q)


def repair_single_annotation(saved_blocks, target_products, source_products, *, mass):
    """Repair the symmetric part of one annotation's frozen-target Gram.

    Inputs use ordinary kernel normalization (not residual-rank normalized
    directional scores). The output still has any noise in saved_blocks.
    It is not a claim that a finite-probe reference equals a dense Gram.
    """
    v=np.asarray(target_products);w=np.asarray(source_products)
    if not np.isfinite(mass) or mass<=0:
        raise ValueError('positive annotation mass required')
    q=w.shape[-4]
    ta=antisymmetric_gram(v,w)/mass**2
    ta=(ta+ta.swapaxes(-1,-2))/2
    s=np.asarray(saved_blocks,dtype=float)
    s=(s+s.swapaxes(-1,-2))/2
    return reconstruct_ordered(s-saved(ta,q),q)+ta
