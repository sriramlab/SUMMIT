import numpy as np
import pytest

from summit.context.cross_trait_gram import saved
from summit.context.cross_trait_zpass import (ZMomentAccumulator, antisymmetric_gram,
    repair_single_annotation, load_array_artifact)
from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch


def dense(seed=71):
    rng=np.random.default_rng(seed);n,m,q,c=41,37,3,5
    phi=np.c_[np.ones(n),rng.normal(size=(n,q-1))]
    u=np.linalg.qr(np.c_[phi,rng.normal(size=(n,c-q))])[0]
    g=rng.normal(size=(n,m))
    f=np.einsum('na,nj->anj',phi,g)
    f-=np.einsum('ni,mi,amj->anj',u,u,f)
    r=np.einsum('anj,bnk->abjk',f,f)
    z=np.einsum('nj,na,nc->jac',g,phi,u)
    return phi,u,g,f,r,z


def test_weighted_z_correction_and_symmetric_target_repair(tmp_path):
    phi,u,g,f,r,z=dense();m=len(z);q=phi.shape[1]
    weights=np.linspace(.1,1.7,m)[:,None]
    groups=np.arange(m)//13
    moments=ZMomentAccumulator(np.unique(groups),1,q,u.shape[1])
    for start in range(0,m,7):
        moments.add(z[start:start+7],weights[start:start+7],groups[start:start+7])
    a=(r-r.swapaxes(0,1))/2
    for b in np.unique(groups):
        target=weights[:,0]*(groups==b);source=weights[:,0]
        ta=np.einsum('j,k,acjk,bdjk->abcd',target,source,a,a).reshape(q*q,q*q)
        got=antisymmetric_gram(moments.products[b,0],moments.global_products[0])
        np.testing.assert_allclose(got,ta,rtol=1e-12,atol=1e-10)
        t=np.einsum('j,k,acjk,bdjk->abcd',target,source,r,r).reshape(q*q,q*q)/weights.sum()**2
        repaired=repair_single_annotation(saved(t,q),moments.products[b,0],
            moments.global_products[0],mass=weights.sum())
        np.testing.assert_allclose(repaired,(t+t.T)/2,rtol=1e-12,atol=1e-12)
    path=tmp_path/'z.npz'
    moments.write(path,provenance={'test':'dense'})
    values,provenance=load_array_artifact(path,kind='summit.cross_trait.z_moments')
    np.testing.assert_array_equal(values['block_products'],moments.products)
    assert provenance['variants']==m
    with pytest.raises(FileExistsError):
        moments.write(path,provenance={'test':'dense'})
    values['global_products'][0,0,0,0,0]+=1
    with np.load(path) as original:
        manifest=original['manifest']
    np.savez(path,manifest=manifest,**values)
    with pytest.raises(ValueError,match='checksum'):
        load_array_artifact(path,kind='summit.cross_trait.z_moments')


def test_masked_callbacks_expose_existing_products_without_changing_outputs():
    phi,u,g,f,r,z=dense();rng=np.random.default_rng(1)
    traits=[]
    for i,idx in enumerate((np.arange(len(phi)),np.arange(7,len(phi)))):
        traits.append(dict(name=str(i),indices=idx,fixed_basis=np.linalg.qr(u[idx])[0],
                           phenotype=rng.normal(size=len(idx))))
    batch=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=phi,traits=traits)
    expected=list(batch.block(g.T));scores=[];zs=[]
    got=list(batch.block(g.T,score_callback=scores.append,z_callback=zs.append))
    for left,right in zip(got,expected):
        assert left[0]==right[0]
        for a,b in zip(left[1:],right[1:]):
            np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(zs[0],z,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(scores[0],np.stack([item[1] for item in got],axis=1))


def test_different_annotations_have_a_mixed_sector_outside_single_annotation_repair():
    phi,u,g,f,r,z=dense();m=len(z);q=phi.shape[1]
    weights=np.c_[np.linspace(.1,1.7,m),np.linspace(1.9,.2,m)]
    moments=ZMomentAccumulator([0],2,q,u.shape[1])
    moments.add(z,weights,np.zeros(m,dtype=int))
    mass=weights.sum(0)
    true=np.einsum('j,k,acjk,bdjk->abcd',weights[:,0],weights[:,1],r,r).reshape(q*q,q*q)/mass.prod()
    # Applying the SAME-annotation formula here deliberately violates its
    # precondition. It cannot recover the cross-annotation antisymmetric part.
    candidate=repair_single_annotation(saved(true,q),moments.global_products[0],
        moments.global_products[1],mass=np.sqrt(mass.prod()))
    assert np.linalg.norm(true-true.T)/np.linalg.norm(true)>1e-3
    assert np.linalg.norm(candidate-true)/np.linalg.norm(true)>1e-3


def test_guarded_shared_product_supplies_z_without_an_extra_product():
    from summit import gxeldcore
    from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
    phi,u,g,_,_,z=dense()
    batch=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=phi,traits=[dict(name='x',
        indices=np.arange(len(phi)),fixed_basis=u,phenotype=np.random.default_rng(61).normal(size=len(phi)))])
    expected=list(batch.block(g.T));values=[]
    operator=ProtectedTNOperator(threads=int(gxeldcore.build_info()['blas_runtime_threads']))
    operator.begin_execution()
    actual=list(batch.block(g.T,z_callback=values.append,shared_tn_operator=operator))
    assert operator.calls==1
    np.testing.assert_allclose(values[0],z,rtol=1e-12,atol=1e-12)
    for left,right in zip(actual[0][1:],expected[0][1:]):
        np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-12)
    assert operator.finish_execution()['available']
