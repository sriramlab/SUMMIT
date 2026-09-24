import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from summit.context.cross_trait_gram import orientation_matrix


def test_common_within_score_expansion_preserves_ordered_oracle():
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_pilot_fit.py'
    spec=importlib.util.spec_from_file_location('pilot_fit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(911);q,b,h,m=5,7,13,32
    scores=rng.normal(size=(b,m,q));weights=rng.uniform(size=(b,m))
    rhs=np.einsum('bm,bmq,bmr->bqr',weights,scores,scores).reshape(b,q*q)
    gr=rng.normal(size=(b,q,q,h));gr=(gr+gr.swapaxes(1,2))/2;gr=gr.reshape(b,q*q,h)
    saved=orientation_matrix(q)
    study=SimpleNamespace(num_basis=q,block_ids=np.arange(b),block_masses=weights.sum(1)[:,None],
        block_genetic_rhs=(rhs@saved.T)[...,None],block_genetic_residual=saved@gr)
    result=module.ordered_within_record(study)
    np.testing.assert_allclose(result['block_rhs'][:,0].reshape(b,q*q),rhs,atol=1e-13)
    np.testing.assert_array_equal(result['block_genetic_residual'][:,0],gr)


def test_authenticated_pilot_all_modes_end_to_end(tmp_path):
    import json
    from summit.context.reference_zpass_cli import file_sha256
    from summit.context.cross_trait_zpass import ZMomentAccumulator,load_array_artifact
    from summit.context.spec import ContextPairIndex
    from summit.ldscore.generalized_gxe_chromosome import ChromosomeMoments,write_chromosome_moments
    from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
    from summit.ldscore.generalized_gxe_cross_trait_batch import CrossTraitBatch
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_pilot_fit.py'
    spec=importlib.util.spec_from_file_location('pilot_pipeline',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(1013);n,m,q=91,179,3
    phi=np.c_[np.ones(n),rng.normal(size=(n,2))];u=np.linalg.qr(phi)[0]
    pairs=ContextPairIndex(q).entries;p=len(pairs);h=p
    residual=np.column_stack([phi[:,x.q]*phi[:,x.r] for x in pairs])
    g=rng.normal(size=(n,m));groups=np.arange(m)//30;labels=np.unique(groups)
    a=np.eye(3)[np.arange(m)%3];mass=a.sum(0);bm=np.array([a[groups==b].sum(0) for b in labels])
    names=('rare','low','common');inputs=tmp_path/'full_cohort_inputs_20260916';inputs.mkdir()
    root=tmp_path/'shared_reference_full_20260916';root.mkdir();chrroot=root/'chr22';chrroot.mkdir()
    expanded=tmp_path/'imputed_expansion_20260917';expanded.mkdir()
    (expanded/'MANIFEST.json').write_text(json.dumps(dict(traits=[])))
    traits=[];hashes={}
    for t,name in enumerate(module.TRAITS):
        idx=np.arange(n) if name=='height_raw' else np.setdiff1d(np.arange(n),np.arange(t,t+11))
        ut=np.linalg.qr(u[idx])[0];y=rng.normal(size=len(idx))
        values=dict(rows=idx,y=y,fixed=ut)
        if name=='height_raw':values.update(phi=phi,basis_names=np.array(['intercept','age','bmi_raw']))
        ip=inputs/f'{name}.npz';np.savez(ip,**values);hashes[str(ip)]=file_sha256(ip)
        traits.append(dict(name=name,indices=idx,fixed_basis=ut,phenotype=y))
    manifest=dict(traits=list(module.TRAITS),trait_sources={Path(k).stem:v for k,v in hashes.items()},
        panels=[dict(chromosome=22,m=m)],global_masses=mass.tolist())
    (root/'MANIFEST.json').write_text(json.dumps(manifest));mh=file_sha256(root/'MANIFEST.json')
    f=np.stack([phi[:,j,None]*g for j in range(q)]);f-=u@np.einsum('nc,anj->acj',u,f)
    def kernels(weights):
        result=[]
        for k in range(3):
            for pair in pairs:
                matrix=(f[pair.q]*weights[:,k])@f[pair.r].T/mass[k]
                if pair.q!=pair.r:matrix=matrix+matrix.T
                result.append(matrix)
        return np.array(result)
    full=kernels(a);flat=full.reshape(3*p,-1)
    directed=np.array([kernels(a*(groups==b)[:,None]).reshape(3*p,-1)@flat.T for b in labels])
    masses=np.repeat(mass,p);directed*=masses[None,:,None]*masses[None,None,:]/(n-u.shape[1])**2
    def publish(name,nrows,rank,rhs,cross):
        moments=ChromosomeMoments('22',name,names,(name,),q,nrows,rank,labels,bm,directed,rhs,cross)
        write_chromosome_moments(chrroot/f'{name}.npz',moments,provenance={'fixture':'dense'})
    publish('reference',n,n-u.shape[1],np.zeros((len(labels),3*p,1)),np.zeros((len(labels),3*p,h)))
    np.savez(chrroot/'reference_aux.npz',global_kernel_diagonal=np.diagonal(full,axis1=1,axis2=2))
    reference_files={name:file_sha256(chrroot/name) for name in ('reference.npz','reference_aux.npz')}
    (chrroot/'REFERENCE_COMPLETE.json').write_text(json.dumps(dict(passed=True,manifest_sha256=mh,files=reference_files)))
    masked=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=residual,traits=traits)
    stats=list(masked.block(g.T));study_files={}
    for t,(name,scores,_,information) in enumerate(stats):
        rhs=np.zeros((len(labels),3*p,1));cross=np.zeros((len(labels),3*p,h))
        for block in labels:
            weights=a*(groups==block)[:,None]
            for pair in pairs:
                factor=1 if pair.q==pair.r else 2
                rhs[block,pair.index::p,0]=factor*(weights.T@(scores[:,pair.q]*scores[:,pair.r]))
                cross[block,pair.index::p]=factor*np.einsum('mk,mh->kh',weights,information[:,pair.index])
        publish(name,len(traits[t]['indices']),masked.traits[t]['common'].residual_rank,rhs,cross)
        common=masked.traits[t]['common']
        np.savez(chrroot/f'{name}_common.npz',residual_gram=common.residual_gram,residual_rhs=common.residual_rhs)
        for filename in (f'{name}.npz',f'{name}_common.npz'):study_files[filename]=file_sha256(chrroot/filename)
    (chrroot/'STUDY_COMPLETE.json').write_text(json.dumps(dict(passed=True,manifest_sha256=mh,files=study_files)))
    crossroot=tmp_path/'cross';crossroot.mkdir();(crossroot/'chr22').mkdir()
    batch=CrossTraitBatch(masked,block_ids=labels,annotation_names=('common',))
    list(batch.block(g.T,a[:,2:3],groups))
    batch.write(crossroot/'chr22/cross_trait_summary.npz',provenance=dict(reference_files=reference_files,
        common_only=True,genotype_traversals=1,variant_visits=m,input_sha256=hashes))
    zroot=tmp_path/'z';zroot.mkdir();acc=ZMomentAccumulator(labels,3,q,u.shape[1])
    acc.add(np.einsum('nj,na,nc->jac',g,phi,u),a,groups)
    acc.write(zroot/'zpass_chr22.npz',provenance=dict(reference_files=reference_files,
        master_input_sha256=hashes[str(inputs/'height_raw.npz')],execution_ledger=dict(retained_variant_visits=m)))
    output=tmp_path/'fits';output.mkdir()
    module.fit_pilot(SimpleNamespace(base=tmp_path,study_root=crossroot,z_root=zroot,output=output,
        modes=module.MODES,chromosomes=[22]))
    assert len(list(output.glob('*.npz')))==28*4
    result,meta=load_array_artifact(output/f'{module.TRAITS[0]}__{module.TRAITS[1]}__factorized.npz',kind='summit.cross_trait.fit')
    assert result['omega_xy'].shape==(1,q,q)
    assert result['loo_h_xy'].shape==(len(labels),1,q-1,q-1)
    assert meta['genotype_traversals']==0
