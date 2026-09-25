"""Real chromosome qualification using the frozen 2026-09-19 matched menu.

Derived from benchmark_native_context_mixture_matched_20260919.py. Scientific
preparation, priors and held-out selection are retained. This driver selects a
complete chromosome and records resource use. It never changes existing fits.
"""
from pathlib import Path
from dataclasses import replace
import argparse,ctypes,importlib.util,json,os,sys,time,resource
CPUS=tuple(sorted(os.sched_getaffinity(0)))
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trait',default='c_reactive_prot_log')
    p.add_argument('--old',type=Path,default=Path('/data1/bronsonj/general_gxe_expanded_20260911'))
    p.add_argument('--prefix',type=Path,default=Path('/home/bronsonj/UKBB/geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc_imp.bed'))
    p.add_argument('--matched-config',type=Path)
    p.add_argument('--stop-after-sweep',type=int,default=0,help='Qualification: terminate only after an atomic checkpoint')
    p.add_argument('--menu',choices=['core','extensions','best'],default='core')
    p.add_argument('--chromosome',default='22');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--block-size',type=int,default=32)
    p.add_argument('--support-native',type=Path)
    p.add_argument('--storage',choices=['compact','packed'],default='compact')
    p.add_argument('--migration-module',type=Path)
    p.add_argument('--export-checkpoint',type=Path)
    p.add_argument('--legacy-protocol',type=Path)
    p.add_argument('--export-destination',type=Path)
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--receipt-sha256')
    p.add_argument('--portable-source',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--held-rows',type=int)
    a=p.parse_args();os.umask(0o077)
    a.qualify_variants=1  # bounded panel: do not compare full-panel external scores
    sys.meta_path[:]=[v for v in sys.meta_path if type(v).__module__!='_gwldcore_editable']
    sys.path.insert(0,str(a.runtime/'src'))
    import summit
    for name in ('gxeldcore','gwldcore','winldcore'):
        directory=a.native if list(a.native.glob(name+'*.so')) else a.support_native
        spec=importlib.util.spec_from_file_location('summit.'+name,next(directory.glob(name+'*.so')))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    import numpy as np
    import pandas as pd
    from summit import gxeldcore as native
    from summit.prediction import (FileGenotypeSource,TraitTraining,AnnotationDesign,AnnotationPrior,
        MixtureSpec,SeparateSparsitySpec,MixtureSolverSpec,fit_mixture_prediction,plan_mixture_prediction,
        load_prediction_models,estimate_scale,ScoreInput,score_prediction,shrink_orthogonal_covariance)
    from summit.prediction.artifacts import file_digest
    from summit.prediction.runtime import configure_prediction_threads
    configure_prediction_threads(native,len(CPUS))
    def save(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2)
    def product(left,right):
        out=np.empty((len(left),right.shape[1]),order='F')
        native.prediction_product(np.asfortranarray(left),np.asfortranarray(right),out,False,len(CPUS),True)
        return out
    from threadpoolctl import threadpool_limits
    preparation_threads=threadpool_limits(limits=1, user_api="blas")
    old=a.old;secure=old/'prediction_refits/secure'
    train_path=secure/(a.trait+'.npz');prior_path=old/'prediction_refits'/a.trait/'priors.npz'
    with np.load(train_path) as z:
        rows=z['rows'];y=z['y'];phi=z['phi'];fixed=z['fixed'];transform=z['fixed_transform'];names=z['basis_names'].tolist()
    center=phi[:,1:].mean(0);phi=phi.copy();phi[:,1:]-=center
    metric=np.cov(phi[:,1:],rowvar=False,bias=True);change=np.eye(5);change[0,1:]=center
    with np.load(prior_path) as z:
        j=list(z['models']).index('psd_k1_floor0.05');omega=change@z['omega'][j]@change.T
    assert omega[0,0]>0
    matched_config=json.loads(a.matched_config.read_text()) if a.matched_config else None
    if matched_config:
        assert matched_config['trait']==a.trait and matched_config['passed']
        h,power,f,probability=(matched_config[k] for k in ('h','power','f','p'))
    else:
        assert a.trait in ('c_reactive_prot_log','lipo_a_log')
        h,power,f=(.3184,0.,.3) if a.trait=='c_reactive_prot_log' else (.5,-1.,.1)
        probability=.01
    variance=float(np.mean((y-product(fixed,np.linalg.lstsq(fixed,y,rcond=None)[0][:,None])[:,0])**2))
    omega*=h*variance/omega[0,0]
    amplified=np.outer(omega[:,0],omega[0])/omega[0,0]
    additive=np.zeros((5,5));additive[0,0]=omega[0,0]
    orthogonal=omega-amplified
    independent=np.diag(np.diag(omega))
    base_mix=MixtureSpec(probability,f);separate=SeparateSparsitySpec(base_mix,MixtureSpec(.1,.3))
    held={}
    with np.load(a.root/'prediction_comparisons/discovery_coefficients.npz') as z:mean_coef=z[a.trait]
    for split in ('pilot','replication'):
        with np.load(secure/(split+'.npz')) as z:
            assert {'rows','basis','fixed_raw','phenotypes','trait_ids','eligible'}<=set(z.files)
            index=list(z['trait_ids']).index(a.trait);take=np.flatnonzero(z['eligible']&np.isfinite(z['phenotypes'][index]))
            if a.held_rows is not None:take=take[:a.held_rows]
            ep=z['basis'][take].copy();ep[:,1:]-=center
            held[split]=dict(rows=z['rows'][take],y=z['phenotypes'][index,take],phi=ep,
                fixed=product(z['fixed_raw'][take],transform),mean=product(z['fixed_raw'][take],mean_coef[:,None])[:,0])
        assert not np.intersect1d(rows,held[split]['rows']).size
    assert not np.intersect1d(held['pilot']['rows'],held['replication']['rows']).size
    # Read and authenticate the held-out external baseline schema before a fit.
    external_scores={};external_files={}
    if a.qualify_variants is None:
        paths={'ldak_bolt_tuned':a.root/'pgs_shrinkage_20260916'/a.trait/f'bolt_h{h:g}_power{power:g}',
               'ldak_gaussian_tuned':a.root/'pgs_gaussian_grid_20260916'/a.trait/'projected_scores'}
        if matched_config:paths={label:Path(path) for label,path in matched_config['external_scores'].items()}
        for label,path in paths.items():
            receipt=json.loads((path/'COMPLETE.json').read_text());assert receipt['passed']
            external_files[str(path/'COMPLETE.json')]=file_digest(path/'COMPLETE.json')
            values={};model_names=None
            for split,part in held.items():
                file=path/(split+'_scores.npz');digest=file_digest(file)
                if 'files' in receipt:assert digest==receipt['files'][file.name]
                external_files[str(file)]=digest
                with np.load(file) as z:
                    ix=pd.Index(z['rows']).get_indexer(part['rows']);assert np.all(ix>=0)
                    values[split]=z['predictions'][ix];names_here=z['models'].tolist()
                    assert values[split].shape==(len(ix),len(names_here)) and np.isfinite(values[split]).all()
                    if model_names is not None:assert model_names==names_here
                    model_names=names_here
            external_scores[label]=(model_names,values)
    matched_path=a.root/'pgs_diagnosis_20260915/full'/a.trait/'models'
    scale=load_prediction_models(matched_path)[0].scale if matched_path.exists() else None
    assert scale is not None or (matched_config and not a.plan_only and a.qualify_variants is None)
    all_weights=None if scale is None else (1/scale.inverse_scale**2)**(1+power)
    full_mass=None if all_weights is None else float(np.sum(all_weights.astype(np.longdouble)))
    prefix=str(a.prefix)
    started=time.monotonic()
    with FileGenotypeSource(prefix,genome_build='GRCh37') as source:
        if a.portable_source:source.authenticate_content()
        variants=np.arange(len(source.variants.ids))
        if a.qualify_variants is not None:
            variants=np.flatnonzero(np.array(source.variants.chromosome)==a.chromosome)
            assert len(variants)>32, 'chromosome has too few variants'
            scale=estimate_scale(source,rows,variants,ddof=0,block_size=128,threads=len(CPUS))
        elif a.plan_only:
            scale=replace(scale,provenance={**scale.provenance,'source':source.identity,
                'scope':'planning only; fitting independently verifies the full affine scale'})
        else:
            checked=estimate_scale(source,rows,variants,ddof=0,block_size=128,threads=len(CPUS))
            if scale is not None:
                np.testing.assert_allclose(checked.mean,scale.mean,rtol=0,atol=1e-12)
                np.testing.assert_allclose(checked.inverse_scale,scale.inverse_scale,rtol=1e-11,atol=0)
            scale=checked
        if all_weights is None:
            all_weights=(1/scale.inverse_scale**2)**(1+power)
            full_mass=float(np.sum(all_weights.astype(np.longdouble)))
        weights=all_weights[variants];fraction=float(weights.sum())/full_mass
        design=AnnotationDesign(weights[:,None],('matched_empirical_power',),scale.variant_identity)
        candidates=[];mixtures={};metadata=[]
        residual=np.full(len(rows),(1-h)*variance)
        core=[('additive_gaussian',additive,MixtureSpec(.5,.5)),('additive_mixture',additive,base_mix),
              ('amplification_mixture',amplified,base_mix),('full_gaussian',omega,MixtureSpec(.5,.5)),
              ('full_radial_mixture',omega,base_mix),('full_separate_mixture',omega,separate),
              ('independent_gaussian',independent,MixtureSpec(.5,.5)),
              ('independent_radial_mixture',independent,base_mix),
              ('independent_separate_mixture',independent,separate)]
        extra=[('coupled_negative',omega,SeparateSparsitySpec(base_mix,separate.response,-.5)),
               ('coupled_positive',omega,SeparateSparsitySpec(base_mix,separate.response,.5)),
               ('directional_shrinkage',shrink_orthogonal_covariance(omega,[1.,.5,.5,.5],environment_metric=metric),separate)]
        for strength in (.5,1.,2.):
            specs=list(core) if a.menu in ('core','best') else [('full_separate_mixture',omega,separate),*extra]
            if a.menu=='best':
                for family,cov,mix in extra:
                    specs.extend([('full_'+family,cov,mix),('independent_'+family,np.diag(np.diag(cov)),mix)])
            for family,cov,mix in specs:
                model_id=f'{family}_s{strength:g}'
                prior=AnnotationPrior(design,(cov*strength*fraction)[None])
                candidates.append(prior.candidate(model_id,residual,dict(family=family,strength=strength,
                    anchor='discovery-centered',residual='common homoscedastic matched additive residual')))
                mixtures[a.trait,model_id]=mix;metadata.append(dict(id=model_id,family=family,strength=strength))
            if a.menu in ('extensions','best'):
                # Keep the aggregate covariance fixed and redistribute only
                # orthogonal-response prior variance across MAF bins.
                maf=np.minimum(scale.mean/2,1-scale.mean/2);is_low=maf<.05
                annotations=np.column_stack([weights*is_low,weights*(~is_low)])
                bin_design=AnnotationDesign(annotations,('MAF_below_5_percent','MAF_at_least_5_percent'),scale.variant_identity)
                shares=bin_design.masses/bin_design.masses.sum()
                for factor in (.5,2.):
                    redistribution=np.array([factor,1.]);redistribution/=shares@redistribution
                    cov=np.array([shares[b]*(amplified+redistribution[b]*orthogonal) for b in range(2)])
                    np.testing.assert_allclose(cov.sum(0),omega,rtol=1e-12,atol=1e-12)
                    family=f'maf_response_{factor:g}'
                    versions=[(family,cov)] if a.menu=='extensions' else [
                        ('full_'+family,cov),('independent_'+family,np.array([np.diag(np.diag(v)) for v in cov]))]
                    for name,prior_covariance in versions:
                        model_id=f'{name}_s{strength:g}'
                        prior=AnnotationPrior(bin_design,prior_covariance*strength*fraction)
                        candidates.append(prior.candidate(model_id,residual,dict(family=name,strength=strength,
                            scope='tuned redistribution with common amplification; not estimated MAF-specific amplification directions')))
                        mixtures[a.trait,model_id]=separate;metadata.append(dict(id=model_id,family=name,strength=strength))
        if a.menu in ('core','best'):
            by_id={c.id:c for c in candidates}
            # Every unrestricted candidate has an exact diagonal counterpart,
            # including each annotation's marginal variances and mixture law.
            for j,item in enumerate(metadata):
                if not item['family'].startswith('full_'):continue
                other=item['id'].replace('full_','independent_',1)
                assert other in by_id and mixtures[a.trait,item['id']]==mixtures[a.trait,other]
                full_prior=by_id[item['id']].annotation_prior
                diagonal_prior=by_id[other].annotation_prior
                assert full_prior.design.identity==diagonal_prior.design.identity
                np.testing.assert_allclose(diagonal_prior.covariances,
                    np.array([np.diag(np.diag(v)) for v in full_prior.covariances]),rtol=1e-11,atol=1e-14)
        context=dict(names=names,discovery_center=center.tolist());fixed_spec=dict(names=[f'fixed{i}' for i in range(fixed.shape[1])])
        trait=TraitTraining(a.trait,rows,variants,y,phi,fixed,scale,tuple(candidates),context,fixed_spec,{'units':'discovery_standardized_residual'})
        if a.export_checkpoint or a.warm_start:
            spec=importlib.util.spec_from_file_location('summit.prediction.migration',a.migration_module)
            migration=importlib.util.module_from_spec(spec);spec.loader.exec_module(migration)
        if a.export_checkpoint:
            from dataclasses import asdict
            legacy_protocol=json.loads(a.legacy_protocol.read_text())
            migration.export_legacy_checkpoint(a.export_checkpoint,a.export_destination,[trait],mixtures,source,
                legacy_python=a.runtime/'src/summit/prediction',legacy_native=native.__file__,
                legacy_solver=asdict(MixtureSolverSpec(rtol=1e-7,max_sweeps=250)),
                protocol_fit_identity=legacy_protocol['plan']['fit_identity'],block_size=a.block_size,
                threads=len(CPUS))
            return
        initial_weights=None
        if a.warm_start:
            initial_weights,migration_receipt=migration.load_legacy_checkpoint_weights(
                a.warm_start,[trait],mixtures,source,receipt_sha256=a.receipt_sha256)
        options=dict(storage=a.storage,block_size=a.block_size,threads=len(CPUS),memory_bytes=192*2**30)
        plan=plan_mixture_prediction([trait],source,**options).to_dict()
        a.output.mkdir(parents=True,exist_ok=a.resume)
        protocol=dict(chromosome=a.chromosome,native_sha256=file_digest(native.__file__),
            portable_source=a.portable_source,
            migration_receipt_sha256=a.receipt_sha256,trait=a.trait,menu=a.menu,candidates=metadata,plan=plan,discovery_n=len(rows),variants=len(variants),
            held_n={k:len(v['rows']) for k,v in held.items()},bounded_qualification=a.qualify_variants is not None,
            source_sha256=file_digest(__file__),inputs={str(p):file_digest(p) for p in [train_path,prior_path]+([a.matched_config] if a.matched_config else [])},
            calibration='Five-fold pilot cross-validation chooses candidate; full pilot calibrates replication prediction. Discovery fixed-effect mean common to all methods.',
            external_baselines=(['tuned LDAK Bolt','LDAK Gaussian with matched h/power (no independent Gaussian grid)']
                if matched_config else ['tuned LDAK Bolt','tuned LDAK Gaussian (qualified six-candidate grid)']),
            deferred_external_baselines=['LDAK Elastic: completion and scoring are not yet qualified'],
            external_score_files=external_files,
            internal_baselines=['additive Gaussian','equally specified additive mixture','additive mixture PGS × environments','amplification-only mixture','full Gaussian'],
            primary_arms=['adaptive_pipeline','full_covariance_only','independent_environment_covariance'],
            primary_definition='Full-only fixes the full covariance structure and tunes shrinkage within that structure. Independent uses diag(Omega) in the same discovery-centered named-environment basis and joint likelihood. Adaptive selects covariance structures and shrinkage using the same pilot folds.',
            independent_definition='Zero prior covariance between all baseline/environment coefficients, with matched marginal variances. Non-Gaussian mixture states may remain shared; this is a covariance ablation, not factorized joint prior independence or separate univariate regressions.',
            primary_scope=('All implemented core and extension candidates with matched diagonal counterparts; experimental menu, not a completed performance result.' if a.menu=='best' else
                'Core covariance-family comparison; use the best menu to include matched extensions.'),
            matching='same participants/SNPs/alleles/affine scale/fixed span/residual model and three covariance-strength values; additive p/f comes from completed LDAK tuning',
            limitations=['Stage-one menu does not exhaust response sparsity tuning.','Annotation extension redistributes response variance; training-only MAF covariance estimation remains additional work.'])
        if not a.resume:save(a.output/'PROTOCOL.json',protocol)
        else:
            previous=json.loads((a.output/'PROTOCOL.json').read_text());assert previous==protocol
        preparation_threads.restore_original_limits()
        print(json.dumps(dict(stage='plan',models=len(candidates),n=len(rows),m=len(variants),peak_bytes=plan['estimated_peak_bytes'])),flush=True)
        if a.plan_only:return
        checkpoint=a.output/'checkpoint.npz';model_path=a.output/'models'
        def progress(d):
            print(json.dumps(d),flush=True)
            if initial_weights is not None and 'legacy_objective' in migration_receipt:
                previous=np.array(migration_receipt['legacy_objective'])
                current=np.array(d['elbo'])
                if current.shape!=previous.shape or np.any(current<previous-1e-8*np.maximum(abs(previous),1.)):
                    raise FloatingPointError('warm start lost legacy variational objective')
            if a.stop_after_sweep and d.get('sweep',0)>=a.stop_after_sweep:
                assert checkpoint.exists()
                os._exit(75)
        if model_path.exists():models=load_prediction_models(model_path)
        else:
            models=fit_mixture_prediction([trait],source,output=model_path,mixtures=mixtures,
                solver=MixtureSolverSpec(rtol=1e-7,max_sweeps=250),checkpoint=checkpoint,
                resume=a.resume and checkpoint.exists(),progress=progress,
                **({'initial_weights':initial_weights} if initial_weights is not None and not (a.resume and checkpoint.exists()) else {}),**options)
        fit_seconds=time.monotonic()-started
        for split,part in held.items():
            path=a.output/(split+'_scores.npz')
            if path.exists():
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['rows'],part['rows']);assert list(z['models'])==[m.model_id for m in models]
                    part['scores']=z['predictions']
            else:
                score=score_prediction(models,source,{a.trait:ScoreInput(part['rows'],part['phi'],part['fixed'],context,fixed_spec)},
                    block_size=128,rhs_columns=40,threads=len(CPUS),memory_bytes=32*2**30)
                part['scores']=np.column_stack([score.prediction[m.key] for m in models])
                with path.open('xb') as f:np.savez(f,rows=part['rows'],predictions=part['scores'],models=[m.model_id for m in models])
    scoring_seconds=time.monotonic()-started-fit_seconds
    assert [m.model_id for m in models]==[m['id'] for m in metadata], 'Candidate order changed before selection'
    pilot,rep=held['pilot'],held['replication'];yp=pilot['y']-pilot['mean']
    folds=np.random.default_rng(20260917).permutation(len(yp))%5
    families={}
    for j,model in enumerate(models):families.setdefault(metadata[j]['family'],[]).append(j)
    if a.menu in ('core','best'):
        families['adaptive_pipeline']=list(range(len(models)))
        families['full_covariance_only']=[j for j,m in enumerate(metadata) if m['family'].startswith('full_')]
        families['independent_environment_covariance']=[j for j,m in enumerate(metadata) if m['family'].startswith('independent_')]
        assert len(families['full_covariance_only'])==len(families['independent_environment_covariance'])==(24 if a.menu=='best' else 9)
    if 'additive_mixture' in families:families['additive_mixture_pgs_x_e']=families['additive_mixture']
    predictions={};selection={};records=[]
    for family,columns in families.items():
        losses=[];designs=[]
        for j in columns:
            xp=pilot['scores'][:,j,None];xr=rep['scores'][:,j,None]
            if family.endswith('pgs_x_e'):xp=xp*pilot['phi'];xr=xr*rep['phi']
            xp=np.column_stack([np.ones(len(xp)),xp]);xr=np.column_stack([np.ones(len(xr)),xr])
            predicted=np.empty_like(yp)
            for fold in range(5):
                train=folds!=fold;test=~train
                beta=np.linalg.lstsq(xp[train],yp[train],rcond=None)[0];predicted[test]=xp[test]@beta
            losses.append(float(np.mean((yp-predicted)**2)));designs.append((xp,xr))
        best=int(np.argmin(losses));xp,xr=designs[best]
        beta=np.linalg.lstsq(xp,yp,rcond=None)[0];prediction=rep['mean']+xr@beta
        predictions[family]=prediction;selection[family]=dict(model=models[columns[best]].model_id,pilot_cv_mse=losses[best],calibration=beta.tolist())
        mse=float(np.mean((rep['y']-prediction)**2));records.append(dict(procedure=family,n=len(prediction),mse=mse,r2=1-mse/np.var(rep['y'])))
    # Full-panel external scores are not compared to a bounded-SNP qualification.
    for label,(external_names,external) in external_scores.items():
        losses=[];designs=[]
        for j in range(len(external_names)):
            xp=np.column_stack([np.ones(len(yp)),external['pilot'][:,j]])
            xr=np.column_stack([np.ones(len(rep['y'])),external['replication'][:,j]])
            predicted=np.empty_like(yp)
            for fold in range(5):
                train=folds!=fold;test=~train
                beta=np.linalg.lstsq(xp[train],yp[train],rcond=None)[0];predicted[test]=xp[test]@beta
            losses.append(float(np.mean((yp-predicted)**2)));designs.append((xp,xr))
        best=int(np.argmin(losses));xp,xr=designs[best]
        beta=np.linalg.lstsq(xp,yp,rcond=None)[0];prediction=rep['mean']+xr@beta;predictions[label]=prediction
        selection[label]=dict(model=external_names[best],pilot_cv_mse=losses[best],calibration=beta.tolist())
        mse=float(np.mean((rep['y']-prediction)**2));records.append(dict(procedure=label,n=len(prediction),mse=mse,r2=1-mse/np.var(rep['y'])))
    contrasts=[]
    for reference in ('independent_environment_covariance','full_covariance_only','additive_mixture','additive_mixture_pgs_x_e','ldak_bolt_tuned'):
        if reference not in predictions:continue
        for name,prediction in predictions.items():
            if name==reference:continue
            d=(rep['y']-predictions[reference])**2-(rep['y']-prediction)**2
            variance=np.var(rep['y']);delta=float(d.mean()/variance)
            se=float(np.std((d-delta*(rep['y']-rep['y'].mean())**2)/variance,ddof=1)/np.sqrt(len(d)))
            contrasts.append(dict(procedure=name,versus=reference,delta_r2=delta,se=se,lo=delta-1.96*se,hi=delta+1.96*se))
    pd.DataFrame(records).to_csv(a.output/'metrics.tsv',sep='\t',index=False,mode='x')
    pd.DataFrame(contrasts).to_csv(a.output/'contrasts.tsv',sep='\t',index=False,mode='x')
    save(a.output/'SELECTION.json',selection)
    save(a.output/'COMPLETE.json',dict(passed=True,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        fit_seconds=fit_seconds,scoring_seconds=scoring_seconds,
        seconds=time.monotonic()-started,models=len(models),variants=len(variants),n=len(rows),
        bounded_qualification=a.qualify_variants is not None,protocol_sha256=file_digest(a.output/'PROTOCOL.json'),
        model_manifest_sha256=file_digest(model_path/'manifest.json'),
        files={p.name:file_digest(p) for p in a.output.iterdir() if p.is_file() and p.suffix in ('.tsv','.npz','.json')},
        interpretation='Conditional paired replication intervals; the bounded panel is an arithmetic/runtime check, not a performance result.'))
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':main()
