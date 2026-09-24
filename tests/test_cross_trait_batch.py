import numpy as np
import pytest

from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
from summit.ldscore.generalized_gxe_cross_trait_batch import CrossTraitBatch, CrossTraitScoreAccumulator


def test_all_pairs_batched_scores_and_requested_order():
    rng=np.random.default_rng(617);m,t,q,k=39,6,3,2
    scores=rng.normal(size=(m,t,q));a=rng.uniform(size=(m,k));groups=np.arange(m)//11
    pairs=[(5,2),(1,4),(2,5)]
    acc=CrossTraitScoreAccumulator(trait_names=[str(x) for x in range(t)],num_basis=q,
        num_annotations=k,block_ids=np.unique(groups),pairs=pairs)
    for start in range(0,m,7):
        acc.add(scores[start:start+7],a[start:start+7],groups[start:start+7])
    for b in np.unique(groups):
        take=groups==b
        for j,(x,y) in enumerate(pairs):
            expected=np.einsum('jk,jq,jr->kqr',a[take],scores[take,x],scores[take,y])
            np.testing.assert_allclose(acc.rhs[b,j],expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(acc.rhs[:,0],acc.rhs[:,2].swapaxes(-1,-2),atol=1e-13)


@pytest.mark.parametrize('disjoint',[False,True])
def test_cross_projected_residual_moments_dense(disjoint):
    rng=np.random.default_rng(411);n,m,q,c=67,31,3,5
    phi=np.c_[np.ones(n),rng.normal(size=(n,q-1))]
    fixed=np.linalg.qr(np.c_[phi,rng.normal(size=(n,c-q))])[0]
    residual=np.c_[phi,phi[:,1:]**2,phi[:,1]*phi[:,2]]
    gen=rng.normal(size=(n,m));a=rng.uniform(size=(m,2));groups=np.arange(m)//11
    rows=(np.arange(29),np.arange(29,n)) if disjoint else (np.arange(53),np.arange(13,n))
    traits=[dict(name=str(i),indices=idx,fixed_basis=np.linalg.qr(fixed[idx])[0],
                 phenotype=rng.normal(size=len(idx))) for i,idx in enumerate(rows)]
    masked=MaskedTraitBatch(basis=phi,fixed_basis=fixed,residual_basis=residual,traits=traits)
    batch=CrossTraitBatch(masked,block_ids=np.unique(groups),annotation_names=('a','b'),pairs=[(0,1),(1,0)])
    for start in range(0,m,7):
        list(batch.block(gen[:,start:start+7].T,a[start:start+7],groups[start:start+7]))
    features=[];projectors=[]
    for trait in traits:
        idx=trait['indices'];u=trait['fixed_basis'];p=np.eye(len(idx))-u@u.T
        features.append(np.einsum('il,la,lj->aij',p,phi[idx],gen[idx]))
        projectors.append(p)
    overlap,lx,ly=np.intersect1d(*rows,return_indices=True)
    raw=np.zeros((residual.shape[1],len(rows[0]),len(rows[1])))
    raw[:,lx,ly]=residual[overlap].T
    r=projectors[0]@raw@projectors[1]
    rr=np.einsum('hij,lij->hl',r,r)
    np.testing.assert_allclose(batch.residual_gram[0],rr,rtol=1e-11,atol=1e-11)
    ys=[t['common'].normalized_phenotypes[:,0] for t in masked.traits]
    np.testing.assert_allclose(batch.residual_rhs[0],np.einsum('i,hij,j->h',ys[0],r,ys[1]),atol=1e-11)
    for b in np.unique(groups):
        for k in range(2):
            w=a[:,k]*(groups==b)
            kernels=np.einsum('j,aij,blj->abil',w,features[0],features[1])
            gr=np.einsum('abil,hil->abh',kernels,r).reshape(q*q,-1)
            rhs=np.einsum('i,abil,l->ab',ys[0],kernels,ys[1])
            np.testing.assert_allclose(batch.genetic_residual[b,0,k],gr,atol=1e-9,rtol=1e-11)
            np.testing.assert_allclose(batch.scores.rhs[b,0,k],rhs,atol=1e-10,rtol=1e-11)
    np.testing.assert_allclose(batch.genetic_residual[:,0].reshape(3,2,q,q,-1),
        batch.genetic_residual[:,1].reshape(3,2,q,q,-1).swapaxes(2,3),atol=1e-10)


def test_shared_missing_pattern_cache_preserves_exact_residual_moments():
    rng=np.random.default_rng(889);n,m,q=321,29,3
    phi=np.c_[np.ones(n),rng.normal(size=(n,2))];u=np.linalg.qr(phi)[0]
    traits=[]
    for t in range(4):
        idx=np.setdiff1d(np.arange(100,n),np.arange(100+10*t,110+10*t))
        traits.append(dict(name=str(t),indices=idx,fixed_basis=np.linalg.qr(u[idx])[0],phenotype=rng.normal(size=len(idx))))
    masked=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=phi,traits=traits)
    args=dict(block_ids=np.arange(3),annotation_names=('all',))
    uncached=CrossTraitBatch(masked,**args,max_cached_missing_patterns=0)
    cached=CrossTraitBatch(masked,**args)
    assert len(cached.missing_pattern_rows)==1 and len(cached.missing_pattern_rows[0])==100
    g=rng.normal(size=(m,n));a=np.ones((m,1));groups=np.arange(m)//10
    list(uncached.block(g,a,groups));list(cached.block(g,a,groups))
    np.testing.assert_allclose(cached.genetic_residual,uncached.genetic_residual,rtol=1e-12,atol=1e-10)
    np.testing.assert_array_equal(cached.scores.rhs,uncached.scores.rhs)
