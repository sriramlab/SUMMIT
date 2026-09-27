import os
import sys
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
@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('rank_y',[0,3,5])
def test_cross_projected_residual_moments_dense(disjoint,workers,rank_y):
    rng=np.random.default_rng(411);n,m,q,c=67,31,3,5
    phi=np.c_[np.ones(n),rng.normal(size=(n,q-1))]
    fixed=np.linalg.qr(np.c_[phi,rng.normal(size=(n,c-q))])[0]
    residual=np.c_[phi,phi[:,1:]**2,phi[:,1]*phi[:,2]]
    gen=rng.normal(size=(n,m));a=rng.uniform(size=(m,2));groups=np.arange(m)//11
    rows=(np.arange(29),np.arange(29,n)) if disjoint else (np.arange(53),np.arange(13,n))
    traits=[dict(name=str(i),indices=idx,fixed_basis=np.linalg.qr(fixed[idx,:c if i==0 else rank_y])[0],
                 phenotype=rng.normal(size=len(idx))) for i,idx in enumerate(rows)]
    masked=MaskedTraitBatch(basis=phi,fixed_basis=fixed,residual_basis=residual,traits=traits)
    cpus=sorted(getattr(sys.modules.get('workflow'),'_PRE_NUMERICAL_CPU_AFFINITY',os.sched_getaffinity(0)))
    if len(cpus)<workers:pytest.skip('not enough reserved CPUs for concurrent residual qualification')
    batch=CrossTraitBatch(masked,block_ids=np.unique(groups),annotation_names=('a','b'),pairs=[(0,1),(1,0)],
        residual_workers=workers,residual_cpus=cpus)
    with batch:
        for start in range(0,m,7):
            list(batch.block(gen[:,start:start+7].T,a[start:start+7],groups[start:start+7]))
    if workers>1:
        assert batch.residual_worker_affinity
        for cpu,affinity in batch.residual_worker_affinity.items():
            assert affinity==[cpu] and cpu in cpus[:workers]
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


@pytest.mark.parametrize('minimum_rows',[16,64])
@pytest.mark.parametrize('pair_sums',[0,8])
def test_shared_missing_pattern_cache_preserves_exact_residual_moments(minimum_rows,pair_sums):
    rng=np.random.default_rng(889);n,m,q=321,29,3
    phi=np.c_[np.ones(n),rng.normal(size=(n,2))];u=np.linalg.qr(phi)[0]
    traits=[]
    for t in range(4):
        missing=np.r_[np.arange(100+10*t,110+10*t),np.arange(200,220) if t<3 else np.array([],int)]
        idx=np.setdiff1d(np.arange(100,n),missing)
        traits.append(dict(name=str(t),indices=idx,fixed_basis=np.linalg.qr(u[idx])[0],phenotype=rng.normal(size=len(idx))))
    masked=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=phi,traits=traits)
    args=dict(block_ids=np.arange(3),annotation_names=('all',))
    uncached=CrossTraitBatch(masked,**args,max_cached_missing_patterns=0)
    cached=CrossTraitBatch(masked,**args,minimum_cached_pattern_rows=minimum_rows,
        max_cached_pair_sums=pair_sums)
    assert sorted(map(len,cached.missing_pattern_rows))==([20,100] if minimum_rows==16 else [100])
    assert bool(cached.cached_pair_sum_groups)==(bool(pair_sums) and minimum_rows==16)
    g=rng.normal(size=(m,n));a=np.ones((m,1));groups=np.arange(m)//10
    list(uncached.block(g,a,groups));list(cached.block(g,a,groups))
    np.testing.assert_allclose(cached.genetic_residual,uncached.genetic_residual,rtol=1e-12,atol=1e-10)
    np.testing.assert_array_equal(cached.scores.rhs,uncached.scores.rhs)


@pytest.mark.parametrize('budget',[1,4,32])
@pytest.mark.parametrize('parallel',[False,True])
def test_pooled_mask_cache_reuses_disjoint_partial_patterns_exactly(budget,parallel):
    rng=np.random.default_rng(728);n,m,q=283,23,3
    phi=np.c_[np.ones(n),rng.normal(size=(n,2))];u=np.linalg.qr(phi)[0]
    present=rng.random((n,6))>.16
    # Two nonidentical patterns have a useful common missing-trait subset.
    present[:80,:3]=False;present[:40,3]=False;present[40:80,4]=False
    traits=[]
    for t in range(6):
        idx=np.flatnonzero(present[:,t])
        traits.append(dict(name=str(t),indices=idx,fixed_basis=np.linalg.qr(u[idx])[0],
                           phenotype=rng.normal(size=len(idx))))
    masked=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=phi,traits=traits)
    args=dict(block_ids=np.arange(3),annotation_names=('all',),pairs=[(x,y) for x in range(6) for y in range(6)])
    uncached=CrossTraitBatch(masked,**args,max_cached_missing_patterns=0)
    cpus=sorted(getattr(sys.modules.get('workflow'),'_PRE_NUMERICAL_CPU_AFFINITY',os.sched_getaffinity(0)))
    if parallel and len(cpus)<2:pytest.skip('two reserved CPUs required for parallel cache qualification')
    pooled=CrossTraitBatch(masked,**args,max_cached_missing_patterns=budget,
        minimum_cached_pattern_rows=1,missing_cache_strategy='pooled',cache_addition_penalty_rows=0,
        max_cached_pair_sums=8,
        parallel_cache_products=parallel,residual_workers=2 if parallel else 1,residual_cpus=cpus)
    assert 0<len(pooled.missing_pattern_rows)<=budget
    if budget==32:assert pooled.cached_pair_sum_groups
    used=np.concatenate(pooled.missing_pattern_rows)
    assert len(used)==len(np.unique(used))
    if budget<32:
        assert any(len(np.unique(present[rows],axis=0))>1 for rows in pooled.missing_pattern_rows)
    for geom,(x,y) in zip(pooled.geometry,pooled.scores.pairs):
        for i in geom.get('cached_patterns',()):
            rows=pooled.missing_pattern_rows[i]
            assert not present[rows,x].any() and not present[rows,y].any()
    g=rng.normal(size=(m,n));a=np.ones((m,1));groups=np.arange(m)//8
    list(uncached.block(g,a,groups))
    with pooled:list(pooled.block(g,a,groups))
    np.testing.assert_allclose(pooled.genetic_residual,uncached.genetic_residual,rtol=1e-12,atol=1e-10)
    np.testing.assert_array_equal(pooled.scores.rhs,uncached.scores.rhs)
