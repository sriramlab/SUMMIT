"""Dense independent references for the streamed research covariance path."""
import numpy as np
import pytest
import os

from prediction_helpers import prediction_threads


def fixture():
    from summit.prediction.genotype import ArrayGenotypeSource
    from summit.prediction.spec import VariantAxis
    from summit.epistasis.polygenic import fit_kernel_scales, PolygenicKernels
    rng = np.random.default_rng(94713)
    n, m = 144, 73
    raw = rng.binomial(2, rng.uniform(.15, .45, m), (n, m)).astype(np.int8)
    raw[rng.random(raw.shape)<.03] = -127
    axis = VariantAxis(tuple(f"v{j}" for j in range(m)), ("2",)*m, tuple(range(1,m+1)), ("A",)*m, ("C",)*m)
    samples = [(str(i), str(i)) for i in range(n)]
    def source():
        return ArrayGenotypeSource(raw, samples, axis, hard_calls=True)
    i0 = np.arange(0, n, 2)
    i1 = np.arange(1, n, 2)
    scales = fit_kernel_scales(source(), i0, np.arange(m), block_size=19, threads=prediction_threads())
    contexts = np.column_stack([np.ones(n), rng.normal(size=n)])
    noise = np.column_stack([np.ones(n), .2+contexts[:, 1]**2])
    kernels = []
    designs = []
    for k in range(2):
        value = raw.astype(float) if k == 0 else (raw == 1).astype(float)
        observed = raw != -127
        mean = (value[i0]*observed[i0]).sum(0)/observed[i0].sum(0)
        ss = np.sum(np.where(observed[i0], value[i0]-mean, 0.)**2, axis=0)
        inverse = np.sqrt((len(i0)-1)/ss)
        np.testing.assert_allclose(scales['mean'][k], mean, atol=1e-14)
        np.testing.assert_allclose(scales['inverse_scale'][k], inverse, atol=1e-14)
        designs.append(np.where(observed, value-mean, 0.)*inverse)
    for context in contexts.T:
        d = designs[0]*context[:, None]
        kernels.append(d@d.T/m)
    kernels.append(designs[1]@designs[1].T/m)
    kernels.extend(np.diag(v) for v in noise.T)
    def operator(rows, storage='stream', block_size=19):
        return PolygenicKernels(source(), rows, np.arange(m), scales, contexts[rows], noise[rows],
            block_size=block_size, storage=storage, threads=prediction_threads())
    return rng, operator, np.stack(kernels), i0, i1, contexts


def test_streamed_covariance_he_and_native_solver(tmp_path):
    from summit.epistasis.polygenic import he_geometry, estimate_components, projected_solve
    from scripts.epistasis.conditional_polygenic_reference import fit_covariance
    rng, make, kernels, i0, _, contexts = fixture()
    operator = make(i0, 'packed')
    k = kernels[:, i0][:, :, i0]
    vectors = rng.normal(size=(len(i0), 4))
    np.testing.assert_allclose(operator.apply(vectors), k@vectors, atol=1e-12)
    theta = rng.uniform(.1, 1., (len(k), 4))
    expected = np.einsum('kij,jb,kb->ib', k, vectors, theta)
    np.testing.assert_allclose(operator.apply(vectors, theta), expected, atol=1e-12)
    operator.rhs_columns=2
    np.testing.assert_allclose(operator.apply(vectors),k@vectors,atol=1e-12)
    np.testing.assert_allclose(operator.apply(vectors,theta),expected,atol=1e-12)
    for active in ([0,3],[1,2,4],[3,4]):
        selected=np.zeros_like(theta);selected[active]=theta[active]
        np.testing.assert_allclose(operator.apply(vectors,selected),
            np.einsum('kij,jb,kb->ib',k,vectors,selected),atol=1e-12)
    operator.rhs_columns=128
    c = np.column_stack([contexts[i0], contexts[i0, 1]**2])
    y = rng.normal(size=(len(i0), 2))
    reference = he_geometry(operator, c, exact=True)
    estimate = estimate_components(operator, y, reference)
    for j in range(2):
        independent, _ = fit_covariance(y[:, j], c, k)
        np.testing.assert_allclose(estimate[:, j], independent, atol=1e-10, rtol=1e-8)
    solutions, report = projected_solve(operator, y, c, theta[:, :2], checkpoint=tmp_path/'cg.npz')
    for j in range(2):
        covariance = np.einsum('k,kij->ij', theta[:, j], k)
        w = np.linalg.inv(covariance)
        p = w-w@c@np.linalg.solve(c.T@w@c, c.T@w)
        np.testing.assert_allclose(solutions[:, j], p@y[:, j], atol=2e-8, rtol=1e-7)
        assert report.reports[(str(j), 'null')]['converged']
    resumed, _ = projected_solve(operator, y, c, theta[:, :2], checkpoint=tmp_path/'cg.npz', resume=True)
    np.testing.assert_array_equal(resumed, solutions)
    with pytest.raises(ValueError, match='identity'):
        projected_solve(operator, y+1, c, theta[:, :2], checkpoint=tmp_path/'cg.npz', resume=True)
    changed = dict(reference, operator_identity='changed')
    with pytest.raises(ValueError, match='inputs changed'):
        estimate_components(operator, y, changed)
    from summit.epistasis.polygenic import prepare_preconditioners
    preconditioners=prepare_preconditioners(operator,reference,theta[:,:2])
    faster,fast_report=projected_solve(operator,y,c,theta[:,:2],preconditioners=preconditioners,
        checkpoint=tmp_path/'preconditioned.npz')
    np.testing.assert_allclose(faster,solutions,atol=2e-8,rtol=1e-7)
    assert all(r['relative_true_residual']<=1e-8 for r in fast_report.reports.values())
    repeated=prepare_preconditioners(operator,reference,theta[:,[0,0]])
    assert repeated[0] is repeated[1]
    with pytest.raises(ValueError,match='match each fitted covariance'):
        projected_solve(operator,y,c,theta[:,:2],preconditioners=preconditioners[::-1])
    with pytest.raises(ValueError,match='identity'):
        projected_solve(operator,y,c,theta[:,:2],checkpoint=tmp_path/'preconditioned.npz',resume=True)


@pytest.mark.parametrize('exact',[True,False])
def test_genotype_weighted_he_matches_dense_moments_and_preserves_native_covariance(exact):
    from scipy.linalg import null_space
    from summit.epistasis.polygenic import he_geometry,estimate_components,NystromPreconditioner
    from scripts.epistasis.conditional_bias_scaling import estimate
    rng,make,kernels,i0,_,contexts=fixture()
    operator=make(i0,'packed');k=kernels[:,i0][:,:,i0]
    c=np.column_stack([contexts[i0],contexts[i0,1]**2])
    reference=he_geometry(operator,c,exact=exact,probes=32,seed=818,
        moment_weighting='genotype_diagonal')
    diagonal=np.diagonal(k,axis1=1,axis2=2)
    weights=1/np.sqrt((diagonal/diagonal.mean(1)[:,None]).mean(0))
    np.testing.assert_allclose(reference['moment_weights'],weights,atol=1e-13)
    q=null_space((c*weights[:,None]).T,rcond=1e-11)
    projected=np.stack([q.T@(weights[:,None]*v*weights[None,:])@q for v in k])
    z=np.eye(len(i0)) if exact else np.random.default_rng(818).choice([-1.,1.],(len(i0),32))
    products=projected@(q.T@z)
    gram=np.einsum('anb,cnb->ac',products,products)/(1 if exact else 32)
    np.testing.assert_allclose(reference['gram'],gram,atol=2e-11,rtol=2e-12)
    y=rng.normal(size=(len(i0),3))
    u=q.T@(weights[:,None]*y)
    moments=np.einsum('ir,aij,jr->ar',u,projected,u)
    expected=np.column_stack([estimate(moments[:,j],gram) for j in range(3)])
    actual=estimate_components(operator,y,reference)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-9)
    # An arbitrary finite nuisance mean must not change the fitted components.
    np.testing.assert_allclose(estimate_components(operator,y+c@rng.normal(size=(c.shape[1],3)),reference),
        actual,atol=2e-10,rtol=2e-9)
    np.testing.assert_allclose(operator.apply(y),k@y,atol=2e-12)
    theta=np.ones(operator.count)
    preconditioner=NystromPreconditioner(operator,reference,theta)
    assert np.all(np.isfinite(preconditioner.apply(y)))
    bad=dict(reference,moment_weights=-weights)
    with pytest.raises(ValueError,match='moment weights'):
        estimate_components(operator,y,bad)


def test_cross_cohort_products_and_input_quadratic_forms():
    rng,make,kernels,i0,i1,contexts=fixture()
    operator=make(np.arange(len(contexts)))
    assert not operator.stream.ledger.traversals
    for take,give in [(i0,i1),(i1,i0),(i0,np.arange(len(contexts)))]:
        v=rng.normal(size=(len(take),3))
        for theta in (None,rng.uniform(.1,1.,(5,3)),np.vstack([np.zeros((3,3)),np.ones((2,3))])):
            actual,quadratic=operator.cross_products(v,take,give,theta)
            expected=kernels[:,give][:,:,take]@v
            q=np.einsum('ib,kij,jb->kb',v,kernels[:,take][:,:,take],v)
            if theta is not None:
                expected=np.einsum('kib,kb->ib',expected,theta)
                q=np.sum(q*theta,axis=0)
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            np.testing.assert_allclose(quadratic,q,atol=1e-10)
    with pytest.raises(ValueError,match='ordered unique'):
        operator.cross_products(v,i0[::-1],i1)
    assert 'polygenic_setup' not in operator.stream.ledger.traversals
    # Requesting a solve later must still get the exact preconditioner,
    # calculated once and kept immutable. Cross-products never need this pass.
    np.testing.assert_allclose(operator.diagonal,np.diagonal(kernels,axis1=1,axis2=2),atol=1e-12)
    assert operator.stream.ledger.traversals['polygenic_setup']==1
    assert not operator.diagonal.flags.writeable
    assert operator.stream.ledger.traversals['polygenic_setup']==1


def test_weighted_cross_products_fit_a_single_output_budget():
    rng,make,kernels,i0,i1,contexts=fixture()
    operator=make(np.arange(len(contexts)))
    vectors=rng.normal(size=(len(i0),13))
    theta=rng.uniform(.1,1.,(len(kernels),13))
    # Enough for weighted output and one RHS tile, but not the component
    # stack. This models the wide covariance-derivative transfer in a pilot.
    operator.memory_bytes=operator.base_bytes+112000
    actual,quadratic=operator.cross_products(vectors,i0,i1,theta)
    expected=np.einsum('kib,kb->ib',kernels[:,i1][:,:,i0]@vectors,theta)
    expected_quadratic=np.einsum('ib,kij,jb,kb->b',vectors,
        kernels[:,i0][:,:,i0],vectors,theta)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(quadratic,expected_quadratic,rtol=2e-12,atol=2e-12)
    traversals=dict(operator.stream.ledger.traversals)
    with pytest.raises(MemoryError,match='cross-cohort'):
        operator.cross_products(vectors,i0,i1)
    operator.memory_bytes=operator.base_bytes+60000
    with pytest.raises(MemoryError,match='cross-cohort'):
        operator.cross_products(vectors,i0,i1,theta)
    assert operator.stream.ledger.traversals==traversals


def test_random_effect_generation_matches_frozen_kernel_law():
    from scripts.epistasis.full_polygenic_simulation import genetic_draws
    from summit.epistasis.polygenic import fit_kernel_scales
    _,make,kernels,i0,_,contexts=fixture()
    source=make(np.arange(len(contexts))).stream.source
    rows=np.arange(len(contexts)); variants=np.arange(len(source.variants.ids))
    scales=fit_kernel_scales(source,i0,variants,threads=prediction_threads())
    seed=681239
    actual,expected_variances,ledger=genetic_draws(source,rows,variants,scales,contexts[:,1],
        seed=seed,replicates=4,block_size=13,threads=prediction_threads())
    assert ledger['traversals']=={'simulation_genetic_means':1}
    raw=source.values
    observed=raw!=-127
    designs=[np.where(observed,(raw if k==0 else raw==1)-scales['mean'][k],0.)*scales['inverse_scale'][k]
        for k in range(2)]
    designs.append(designs[0]*contexts[:,1,None])
    for k,g in enumerate(designs):
        weights=np.column_stack([np.random.default_rng(np.random.SeedSequence([seed,k,r])).normal(size=len(variants))
            for r in range(4)])/np.sqrt(len(variants))
        np.testing.assert_allclose(actual[:,:,k],g@weights,atol=2e-14,rtol=1e-13)
    ordered=kernels[[0,2,1]]
    expected=np.array([(np.trace(k)-k.sum()/len(rows))/len(rows) for k in ordered])
    np.testing.assert_allclose(expected_variances,expected,atol=2e-14)
    other,_,_=genetic_draws(source,rows,variants,scales,contexts[:,1],
        seed=seed,replicates=2,block_size=19,threads=prediction_threads())
    np.testing.assert_allclose(actual[:,:2],other,atol=2e-14,rtol=1e-13)
    later,_,_=genetic_draws(source,rows,variants,scales,contexts[:,1],
        seed=seed,replicates=2,replicate_start=2,block_size=17,threads=prediction_threads())
    np.testing.assert_allclose(actual[:,2:],later,atol=2e-14,rtol=1e-13)
    with pytest.raises(ValueError,match='identity'):
        genetic_draws(source,rows,variants,dict(scales,source='changed'),contexts[:,1],seed=seed,replicates=2)


def test_frozen_local_dominance_imputation_and_allele_reversal():
    from summit.prediction.genotype import ArrayGenotypeSource,estimate_scale,native_module
    from summit.prediction.spec import VariantAxis
    from summit.epistasis.models import target_design
    raw=np.array([[0,0],[1,1],[2,1],[0,2],[1,0],[2,1],[0,-127],[1,-127],[2,0]],dtype=np.int8)
    axis=VariantAxis(('target','local'),('2','2'),(1,2),('A','C'),('C','T'))
    samples=[(str(i),str(i)) for i in range(len(raw))]
    source=ArrayGenotypeSource(raw,samples,axis,hard_calls=True)
    rows=np.arange(6,len(raw));train=np.arange(6)
    scale=estimate_scale(source,train,np.arange(2),threads=prediction_threads())
    options=dict(components=[dict(name='interaction',target='target',background='all')],
        annotations={'all':np.ones(2)},additive_annotations=['all'],local_variants=['target','local'],
        dominance_variants=['target','local'],threads=prediction_threads(),native=native_module(),
        dominance_imputation={'target':1/3,'local':.5})
    fixed=target_design(source,rows,scale,**options)
    # Columns: intercept, target/local dosage, target/local heterozygote, target.
    np.testing.assert_array_equal(fixed['fixed_effects'][:,4],[.5,.5,0])
    assert fixed['definitions']['dominance_imputation']['local']==.5
    reversed_raw=np.where(raw==-127,-127,2-raw).astype(np.int8)
    reversed_axis=VariantAxis(axis.ids,axis.chromosome,axis.position,axis.other,axis.counted)
    flipped=ArrayGenotypeSource(reversed_raw,samples,reversed_axis,hard_calls=True)
    flipped_scale=estimate_scale(flipped,train,np.arange(2),threads=prediction_threads())
    other=target_design(flipped,rows,flipped_scale,**options)
    np.testing.assert_allclose(other['modifiers'][:,1],-fixed['modifiers'][:,1],atol=1e-14)
    np.testing.assert_array_equal(other['fixed_effects'][:,3:5],fixed['fixed_effects'][:,3:5])
    with pytest.raises(ValueError,match='probability'):
        target_design(source,rows,scale,**dict(options,dominance_imputation={'local':1.2}))


def test_genotype_sketch_preconditioner_independent_inverse():
    from types import SimpleNamespace
    from summit.epistasis.polygenic import he_geometry,NystromPreconditioner
    rng=np.random.default_rng(472937)
    n,m=128,63
    g=rng.normal(size=(n,m))+15*rng.normal(size=(n,2))@rng.normal(size=(2,m))
    kernels=np.stack([g@g.T/m,np.eye(n)])
    operator=SimpleNamespace(rows=np.arange(n),count=2,identity='bounded_sketch_reference',
        memory_bytes=2**30,base_bytes=0,diagonal=np.diagonal(kernels,axis1=1,axis2=2))
    operator.apply=lambda z,**kwargs:kernels@z
    c=np.ones((n,1))
    reference=he_geometry(operator,c,probes=16,seed=938731)
    theta=np.array([.8,.6])
    preconditioner=NystromPreconditioner(operator,reference,theta)
    matrix=np.diag(preconditioner.remainder)+preconditioner.low_rank@preconditioner.low_rank.T
    v=rng.normal(size=(n,3))
    np.testing.assert_allclose(preconditioner.apply(v),np.linalg.solve(matrix,v),rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(preconditioner.apply(v[:,0]),np.linalg.solve(matrix,v[:,0]),rtol=2e-10,atol=2e-10)
    covariance=np.einsum('k,kij->ij',theta,kernels)
    from scipy.linalg import eigvalsh
    before=eigvalsh(covariance,np.diag(np.diag(covariance)))
    after=eigvalsh(covariance,matrix)
    assert after[-1]/after[0]<.01*(before[-1]/before[0])
    with pytest.raises(ValueError,match='changed'):
        NystromPreconditioner(operator,dict(reference,operator_identity='changed'),theta)


def test_streamed_conditional_response_and_joint_covariance():
    from summit.epistasis.polygenic import conditional_score
    from scripts.epistasis.conditional_polygenic_reference import conditional_null, innovation_score
    rng, make, kernels, i0, i1, contexts = fixture()
    theta = np.array([.8, .3, .4, .7, .2])
    c = np.column_stack([contexts, contexts[:, 1]**2])
    y = rng.normal(size=len(c))
    # Fixed-mean dependent columns may be redundant but span must be retained.
    c = np.column_stack([c, c[:, 1]])
    f = rng.normal(size=(len(c), 2))
    c1 = np.column_stack([c[i1], rng.normal(size=(len(i1), 2))])
    fit = conditional_score(make(i0, 'packed'), make(np.arange(len(c))), i0, i1,
        y[i0], y[i1], f[i0], f[i1], c[i0], c1, theta)
    order = np.r_[i0, i1]
    v = np.einsum('k,kij->ij', theta, kernels)[order][:, order]
    transfer, variance = conditional_null(v, c[order], len(i0))
    expected = innovation_score(y[i0], y[i1], f[i0], f[i1], c1, transfer, variance)
    for field in ('beta', 'covariance', 'response', 'contrast'):
        np.testing.assert_allclose(fit[field], expected[field], atol=2e-8, rtol=1e-7)
    # Confirmation projection can remove a nuisance absent in training.
    # Here it duplicates the intercept, so the numerical answer is unchanged.
    changed0 = np.column_stack([c[i0], np.zeros(len(i0))])
    changed1 = np.column_stack([c1, np.ones(len(i1))])
    redundant = conditional_score(make(i0), make(np.arange(len(c))), i0, i1,
        y[i0], y[i1], f[i0], f[i1], changed0, changed1, theta)
    np.testing.assert_allclose(redundant['beta'],fit['beta'],atol=2e-8,rtol=1e-7)
    np.testing.assert_allclose(redundant['covariance'],fit['covariance'],atol=2e-8,rtol=1e-7)


def test_genotype_only_reference_driver(tmp_path):
    import json
    from types import SimpleNamespace
    import pandas as pd
    from bed_reader import to_bed
    from scripts.epistasis.full_polygenic_reference import run
    _, make, _, i0, i1, contexts = fixture()
    source = make(np.arange(len(contexts))).stream.source
    raw = source.values.astype(float)
    raw[raw == -127] = np.nan
    n, m = raw.shape
    ids = list(map(str, range(n)))
    to_bed(tmp_path/'input.bed', raw, properties=dict(fid=ids,iid=ids,
        sid=list(source.variants.ids),chromosome=['2']*m,bp_position=list(range(1,m+1)),
        allele_1=['A']*m,allele_2=['C']*m))
    observed = np.isfinite(raw[:,0])
    for name, rows in [('training',i0[observed[i0]]),('confirmation',i1[observed[i1]])]:
        pd.DataFrame(dict(FID=[ids[i] for i in rows],IID=[ids[i] for i in rows])).to_csv(
            tmp_path/f'{name}.tsv',sep='\t',index=False)
    pd.DataFrame(dict(FID=ids,IID=ids,PC1=contexts[:,1])).to_csv(tmp_path/'cov.tsv',sep='\t',index=False)
    (tmp_path/'variants.txt').write_text('\n'.join(source.variants.ids[1:])+'\n')
    spec=dict(genotypes=dict(geno='input.bed'),samples='training.tsv',variants='variants.txt',
        target='v0',local_variants=['v0','v1'],dominance_variants=['v0','v1'],
        covariates=dict(file='cov.tsv',columns=['PC1'],varying_effects=['PC1']))
    (tmp_path/'train.json').write_text(json.dumps(spec))
    run(SimpleNamespace(training=tmp_path/'train.json',confirmation=tmp_path/'confirmation.tsv',
        out=tmp_path/'reference',num_threads=prediction_threads(),memory_gib=2,probes=128,seed=71939))
    record=json.loads((tmp_path/'reference/reference.json').read_text())
    assert record['markers']==m-1 and record['n_training']==sum(observed[i0])
    assert record['covariance_condition']<1e8
    with np.load(tmp_path/'reference/reference.npz') as archive:
        assert archive['probe_gram'].shape==(128,5,5)
        np.testing.assert_allclose(archive['gram'],archive['probe_gram'].mean(0),atol=1e-12)


def test_batched_conditional_tests_preserve_independent_operators(tmp_path):
    from summit.epistasis.polygenic import conditional_scores_batch, conditional_score
    rng,make,_,i0,i1,contexts=fixture()
    theta=rng.uniform(.1,1.,(5,2))
    c=np.column_stack([contexts,contexts[:,1]**2])
    y=rng.normal(size=(len(c),2)); f=rng.normal(size=y.shape)
    extras=[np.column_stack([c[i1,0],rng.normal(size=(len(i1),3))]) for j in range(2)]
    batch=conditional_scores_batch(make(i0,'packed'),make(np.arange(len(c))),i0,i1,
        y[i0],y[i1],f[i0],f[i1],c[i0],c[i1],extras,theta,checkpoint_dir=tmp_path)
    for j in range(2):
        single=conditional_score(make(i0),make(np.arange(len(c))),i0,i1,y[i0,j],y[i1,j],
            f[i0,j],f[i1,j],c[i0],np.column_stack([c[i1],extras[j]]),theta[:,j])
        np.testing.assert_allclose(batch['beta'][j],single['beta'][0],atol=2e-8)
        np.testing.assert_allclose(batch['variance'][j],single['covariance'][0,0],atol=2e-8)
        np.testing.assert_allclose(batch['response'][:,j],single['response'][:,0],atol=2e-8)
        # Common participant-axis influence reconstructs the actual statistic.
        expected=batch['contrasts'][:,j]@y[i1,j]+batch['training_contrasts'][:,j]@y[i0,j]
        np.testing.assert_allclose(batch['beta'][j],expected,atol=2e-8)


@pytest.mark.parametrize('recycle,stage,blocks',[(False,'mean_derivative',(19,19)),
    (True,'mean_derivative',(19,19)),(True,'outcome_feature',(19,19)),
    (True,'mean_derivative',(7,31)),(True,'outcome_feature',(7,31))])
def test_streamed_mean_tangents_and_checkpoint_recovery(tmp_path,monkeypatch,recycle,stage,blocks):
    from summit.epistasis.polygenic import conditional_scores_batch
    from scripts.epistasis.conditional_polygenic_reference import (
        conditional_null,conditional_mean_tangents,innovation_score,
    )
    from summit.prediction.checkpoint import SolverCheckpoint
    rng,make,kernels,i0,i1,contexts=fixture()
    theta=rng.uniform(.1,1.,(5,2))
    c=np.column_stack([contexts,contexts[:,1]**2])
    y=rng.normal(size=(len(c),2)); f=rng.normal(size=y.shape)
    extras=[rng.normal(size=(len(i1),2)) for _ in range(2)]
    def run(block):
        return conditional_scores_batch(make(i0,'packed',block),make(np.arange(len(c)),block_size=block),i0,i1,
            y[i0],y[i1],f[i0],f[i1],c[i0],c[i1],extras,theta,
            checkpoint_dir=tmp_path,mean_tangents=True,recycle=recycle)
    save=SolverCheckpoint.save
    def stop(self,state):
        save(self,state)
        if self.path.name==stage+'.npz' and state['iteration']>=2:
            raise RuntimeError('interrupt derivative solve')
    with monkeypatch.context() as patch:
        patch.setattr(SolverCheckpoint,'save',stop)
        with pytest.raises(RuntimeError,match='interrupt derivative'):
            run(blocks[0])
    resumed_block=blocks[1]
    if stage=='mean_derivative' and blocks[0]!=blocks[1]:
        # A derivative RHS is recomputed by streamed sums. Changing their
        # grouping changes its exact bytes, so the strict checkpoint must
        # refuse it, even when the covariance products agree numerically.
        # The failed attempt must leave the original restart usable.
        with pytest.raises(ValueError,match='identity mismatch'):
            run(resumed_block)
        resumed_block=blocks[0]
    actual=run(resumed_block)
    order=np.r_[i0,i1]
    reordered=kernels[:,order][:,:,order]
    for j in range(2):
        v=np.einsum('k,kij->ij',theta[:,j],reordered)
        transfer,covariance=conditional_null(v,c[order],len(i0))
        t=conditional_mean_tangents(v,reordered,c[order],len(i0),y[i0,j])
        expected=innovation_score(y[i0,j],y[i1,j],f[i0,j],f[i1,j],
            np.column_stack([c[i1],extras[j],t]),transfer,covariance)
        np.testing.assert_allclose(actual['beta'][j],expected['beta'][0],atol=2e-7)
        np.testing.assert_allclose(actual['variance'][j],expected['covariance'][0,0],atol=2e-7)
        np.testing.assert_allclose(actual['response'][:,j],expected['response'][:,0],atol=2e-6)
        np.testing.assert_allclose(actual['contrasts'][:,j]@t,0,atol=2e-7)
    assert len(actual['solver_reports'])==3
    if recycle:
        assert (tmp_path/'recycled_inverse.npz').exists()
        assert all(0<=rank<=64 for rank in actual['recycled_ranks'])
        again=run(resumed_block)
        np.testing.assert_allclose(again['beta'],actual['beta'],atol=1e-12)
        np.testing.assert_allclose(again['variance'],actual['variance'],atol=1e-12)


def test_redundant_tangent_reconstruction_at_covariance_boundaries():
    from scipy.linalg import null_space
    from summit.epistasis.polygenic import projected_solve,conditional_tangents_batch
    rng,make,kernels,i0,i1,contexts=fixture()
    theta=np.array([[.7,0.],[0.,0.],[0.,0.],[.2,1.],[.8,0.]])
    c=np.column_stack([contexts[i0],contexts[i0,1]**2]);q=null_space(c.T)
    y=rng.normal(size=(len(i0),2));training=make(i0,'packed');complete=make(np.arange(len(contexts)))
    alpha,_=projected_solve(training,y,c,theta)
    actual,report=conditional_tangents_batch(training,complete,i0,i1,alpha,c,theta)
    assert len(report.reports)==(training.count-1)*2
    for j in range(2):
        v=np.einsum('k,kij->ij',theta[:,j],kernels)
        p=q@np.linalg.solve(q.T@v[i0][:,i0]@q,q.T)
        expected=np.column_stack([(kernel[i1][:,i0]-v[i1][:,i0]@p@kernel[i0][:,i0])@p@y[:,j]
            for kernel in kernels])
        np.testing.assert_allclose(actual[:,j],expected,atol=6e-8,rtol=2e-6)
        np.testing.assert_allclose(actual[:,j]@theta[:,j],0,atol=5e-15)
    # A single iid component leaves its own mean derivative identically zero,
    # while derivatives into currently zero genetic components are retained.
    np.testing.assert_array_equal(actual[:,1,3],0.)
    assert np.linalg.norm(actual[:,1,:3])>1.


def test_postfit_comparators_keep_the_same_conditional_estimator(tmp_path):
    from summit.epistasis.polygenic import conditional_scores_batch,conditional_tangents_batch,projected_solve
    from scripts.epistasis.conditional_comparators import compare_directions
    rng,make,_,i0,i1,contexts=fixture()
    theta=rng.uniform(.1,1.,(5,2));c=np.column_stack([contexts,contexts[:,1]**2])
    y=rng.normal(size=(len(c),2));f=rng.normal(size=y.shape)
    extras=[rng.normal(size=(len(i1),2)) for _ in range(2)]
    training,complete=make(i0,'packed'),make(np.arange(len(c)))
    public=conditional_scores_batch(training,complete,i0,i1,y[i0],y[i1],f[i0],f[i1],
        c[i0],c[i1],extras,theta,mean_tangents=True)
    alpha,_=projected_solve(training,y[i0],c[i0],theta)
    tangents,_=conditional_tangents_batch(training,complete,i0,i1,alpha,c[i0],theta)
    other=compare_directions(training,complete,i0,i1,f0=f[i0],f1=f[i1],y1=y[i1],
        prediction=public['prediction'],fixed0=c[i0],fixed1=c[i1],extras=extras,
        tangents=tangents,theta=theta,checkpoint_dir=tmp_path/'comparators')
    for field in ('beta','variance','response','contrasts','training_contrasts'):
        np.testing.assert_allclose(other[field],public[field],rtol=2e-6,atol=2e-8)
    # Equal comparator geometry is reused without inspecting confirmation
    # outcomes. Different Y columns still receive their own coefficients.
    mapping=np.array([0,0,1,0]);changed_y=y[i1][:,mapping]+rng.normal(size=(len(i1),4))
    duplicate=compare_directions(training,complete,i0,i1,f0=f[i0][:,mapping],f1=f[i1][:,mapping],
        y1=changed_y,prediction=public['prediction'][:,mapping],fixed0=c[i0],fixed1=c[i1],
        extras=[extras[j] for j in mapping],tangents=tangents[:,mapping],theta=theta[:,mapping],
        checkpoint_dir=tmp_path/'duplicates')
    assert duplicate['unique_columns']==[0,2] and duplicate['column_mapping']==mapping.tolist()
    for field in ('response','contrasts','training_contrasts'):
        np.testing.assert_array_equal(duplicate[field],other[field][:,mapping])
    np.testing.assert_array_equal(duplicate['variance'],other['variance'][mapping])
    np.testing.assert_allclose(duplicate['beta'],np.sum(other['contrasts'][:,mapping]*
        (changed_y-public['prediction'][:,mapping]),axis=0),atol=1e-14)


@pytest.mark.parametrize('weighting',['none','genotype_diagonal'])
@pytest.mark.parametrize('exact',[False,True])
def test_component_precision_matches_dense_moments_and_probe_linearization(weighting, exact):
    from summit.epistasis.polygenic import he_geometry,estimate_components
    rng,make,kernels,i0,_,contexts=fixture()
    operator=make(i0,'packed')
    k=kernels[:,i0][:,:,i0];n=len(i0)
    c=np.column_stack([contexts[i0],contexts[i0,1]**2])
    geometry=he_geometry(operator,c,probes=32,exact=exact,moment_weighting=weighting)
    y=rng.normal(size=(n,2))
    theta,uncertainty=estimate_components(operator,y,geometry,return_uncertainty=True)
    np.testing.assert_array_equal(theta,estimate_components(operator,y,geometry))
    before=operator.stream.ledger.operator_calls
    deferred,influence=estimate_components(operator,y,geometry,return_uncertainty='directional')
    assert operator.stream.ledger.operator_calls-before==1
    np.testing.assert_array_equal(deferred,theta)
    np.testing.assert_array_equal(influence['trace'],uncertainty['trace'])
    for j in range(2):
        v=np.einsum('k,kij->ij',theta[:,j],k)
        scores=influence['influence'][:,:,j]
        np.testing.assert_allclose(2*scores@v@scores.T,uncertainty['sampling'][j],rtol=2e-10,atol=2e-12)
    u=geometry['basis'];d=np.diag(geometry.get('moment_weights',np.ones(n)))
    t=d@(np.eye(n)-u@u.T)@d
    quadratic=t@k@t
    inverse=np.linalg.inv(geometry['gram'])
    for j in range(2):
        v=np.einsum('k,kij->ij',theta[:,j],k)
        score=(quadratic@y[:,j]).T
        expected=2*inverse@(score.T@v@score)@inverse.T
        np.testing.assert_allclose(uncertainty['sampling'][j],expected,rtol=2e-10,atol=2e-12)
        if exact:
            np.testing.assert_array_equal(uncertainty['trace'][j],0)
        else:
            shifts=np.stack([inverse@(h@theta[:,j]) for h in geometry['probe_gram']])
            np.testing.assert_allclose(uncertainty['trace'][j],np.cov(shifts.T)/len(shifts),
                rtol=2e-10,atol=2e-12)
        assert np.linalg.eigvalsh(uncertainty['sampling'][j])[0]>-1e-12
    # Scientific-unit and finite-mean invariance includes the saved precision.
    converted,scaled=estimate_components(operator,10*y+c@rng.normal(size=(c.shape[1],2)),
        geometry,return_uncertainty=True)
    np.testing.assert_allclose(converted,100*theta,rtol=1e-10,atol=1e-10)
    for name in uncertainty:
        np.testing.assert_allclose(scaled[name],10000*uncertainty[name],rtol=2e-9,atol=1e-9)


def test_streamed_quadratics_and_fixed_contrast_variance_gradient():
    from summit.epistasis.polygenic import projected_solve,conditional_variance_precision
    rng,make,kernels,i0,i1,contexts=fixture()
    complete=make(np.arange(len(contexts)))
    training=make(i0,'packed')
    k=training.count
    theta=rng.uniform(.2,.8,(k,2))
    a=rng.normal(size=(len(i1),2))
    rhs,_=complete.cross_products(a,i1,i0,theta)
    inverse,_=projected_solve(training,rhs,contexts[i0],theta)
    vectors=np.empty((len(contexts),2));vectors[i0]=-inverse;vectors[i1]=a
    expected=np.einsum('ib,kij,jb->kb',vectors,kernels,vectors)
    np.testing.assert_allclose(complete.quadratic_forms(vectors),expected,rtol=1e-12,atol=1e-10)
    variance=np.sum(expected*theta,axis=0)
    fit=dict(training_contrasts=-inverse,contrasts=a,variance=variance)
    covariance=np.stack([np.eye(k)*.02]*2)
    precision=conditional_variance_precision(complete,i0,i1,fit,theta,
        dict(sampling=covariance,trace=covariance*.1))
    for j in range(2):
        np.testing.assert_allclose(precision[j]['fixed_contrast_variance_relative_sd'],
            np.sqrt(expected[:,j]@(covariance[j]*1.1)@expected[:,j])/variance[j])
        # Differentiate the Schur complement independently, fixing a but
        # allowing the conditional predictor to change with theta.
        def schur(coefficients):
            v=np.einsum('k,kij->ij',coefficients,kernels)
            w=np.linalg.inv(v[i0][:,i0]);c=contexts[i0]
            p=w-w@c@np.linalg.solve(c.T@w@c,c.T@w)
            return a[:,j]@(v[i1][:,i1]-v[i1][:,i0]@p@v[i0][:,i1])@a[:,j]
        delta=np.eye(k)*1e-5
        derivative=np.array([(schur(theta[:,j]+step)-schur(theta[:,j]-step))/2e-5 for step in delta])
        np.testing.assert_allclose(expected[:,j],derivative,rtol=1e-7,atol=1e-6)
    with pytest.raises(ValueError,match='quadratic forms disagree'):
        conditional_variance_precision(complete,i0,i1,dict(fit,variance=variance*2),theta,
            dict(sampling=covariance,trace=covariance))


@pytest.mark.parametrize('weighting',['none','genotype_diagonal'])
def test_directional_precision_equals_full_matrix_with_one_batched_product(weighting):
    from summit.epistasis.polygenic import (he_geometry,estimate_components,
        projected_solve,conditional_variance_precision)
    rng,make,_,i0,i1,contexts=fixture()
    training=make(i0,'packed');complete=make(np.arange(len(contexts)))
    geometry=he_geometry(training,contexts[i0],probes=32,moment_weighting=weighting)
    y=rng.normal(size=(len(i0),2))
    theta,full=estimate_components(training,y,geometry,return_uncertainty=True)
    other,deferred=estimate_components(training,y,geometry,return_uncertainty='directional')
    np.testing.assert_array_equal(theta,other)
    a=rng.normal(size=(len(i1),2))
    rhs,quadratic=complete.cross_products(a,i1,i0,theta)
    inverse,_=projected_solve(training,rhs,contexts[i0],theta)
    fit=dict(training_contrasts=-inverse,contrasts=a,variance=quadratic-np.sum(rhs*inverse,axis=0))
    expected=conditional_variance_precision(complete,i0,i1,fit,theta,full)
    before=training.stream.ledger.operator_calls
    actual=conditional_variance_precision(complete,i0,i1,fit,theta,deferred,training=training)
    assert training.stream.ledger.operator_calls-before==1
    for j in range(2):
        for key in ('sampling_relative_sd','trace_relative_sd','fixed_contrast_variance_relative_sd',
                    'fixed_contrast_effective_df'):
            np.testing.assert_allclose(actual[j][key],expected[j][key],rtol=2e-10,atol=1e-12)
    with pytest.raises(ValueError,match='aligned training moment influences'):
        conditional_variance_precision(complete,i0,i1,fit,theta,deferred)


def test_precision_recovers_classical_common_scale_degrees_of_freedom():
    from types import SimpleNamespace
    from summit.epistasis.polygenic import he_geometry,estimate_components
    rng=np.random.default_rng(120873)
    n=128;c=np.column_stack([np.ones(n),rng.normal(size=(n,2))])
    op=SimpleNamespace(count=1,rows=np.arange(n),contexts=np.ones((n,1)),base_bytes=0,
        memory_bytes=2**30,identity='iid common scale')
    op.apply=lambda v,coefficients=None,**kw: v[None,:,:] if coefficients is None else v*coefficients
    geometry=he_geometry(op,c,exact=True)
    theta,uncertainty=estimate_components(op,rng.normal(size=(n,4)),geometry,return_uncertainty=True)
    # In this special case the diagnostic reduces exactly to N-rank(C).
    np.testing.assert_allclose(2*theta[0]**2/uncertainty['sampling'][:,0,0],n-c.shape[1],rtol=1e-12)
