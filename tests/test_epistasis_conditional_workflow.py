"""Public matched conditional preparation versus independent bounded matrices."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed


def test_public_conditional_batch_reference_fit_and_new_trait(tmp_path, monkeypatch):
    from summit.epistasis.cli import main
    from summit.epistasis.robust import load_robust_scores, robust_score_tests
    from summit.prediction.artifacts import load_prediction_models
    from summit.prediction.genotype import FileGenotypeSource
    from scripts.epistasis.conditional_polygenic_reference import (
        fit_covariance, conditional_null, conditional_mean_tangents, innovation_score)
    from scipy.linalg import orth
    from prediction_helpers import prediction_threads
    threads=prediction_threads()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):
        monkeypatch.setenv(key,str(threads))
    rng=np.random.default_rng(991732)
    n0,n1,m=128,192,120; n=n0+n1
    raw=rng.binomial(2,rng.uniform(.2,.45,m),(n,m)).astype(float)
    raw[rng.random(raw.shape)<.01]=np.nan
    raw[:,0]=rng.binomial(2,.35,n)
    samples=[str(i) for i in range(n)]; names=[f'v{i}' for i in range(m)]
    to_bed(tmp_path/'g.bed',raw,properties=dict(fid=samples,iid=samples,sid=names,
        chromosome=['12']*3+['5']*(m-3),bp_position=list(range(1,m+1)),
        allele_1=['A']*m,allele_2=['C']*m))
    z=rng.normal(size=n)
    pd.DataFrame(dict(FID=samples,IID=samples,PC1=z)).to_csv(tmp_path/'cov.tsv',sep='\t',index=False)
    x=(raw[:,0]-raw[:n0,0].mean())/np.sqrt(raw[:n0,0].mean()*(1-raw[:n0,0].mean()/2))
    y=rng.normal(size=(n,2))*np.sqrt(.4+.6*x*x)[:,None]
    y[:,1] += .4*x*np.nan_to_num(raw[:,3]-np.nanmean(raw[:n0,3]))+.3*z*z
    pd.DataFrame(dict(FID=samples,IID=samples,y=y[:,0],new_trait=y[:,1])).to_csv(tmp_path/'phenotypes.tsv',sep='\t',index=False)
    for label,selected in [('training',samples[:n0]),('confirmation',samples[n0:])]:
        pd.DataFrame(dict(FID=selected,IID=selected)).to_csv(tmp_path/(label+'.tsv'),sep='\t',index=False)
    (tmp_path/'variants.txt').write_text('\n'.join(names[1:])+'\n')
    (tmp_path/'background.txt').write_text('\n'.join(names[3:])+'\n')
    train=dict(kind='summit.epistasis.train_direction',schema_version=1,
        genotypes=dict(geno='g.bed',content_identity=True),samples='training.tsv',target='v0',variants='variants.txt',
        interaction_variants='background.txt',trans_only=True,
        phenotypes=dict(file='phenotypes.tsv',columns=['y','new_trait'],unit='cm'),
        covariates=dict(file='cov.tsv',columns=['PC1'],varying_effects=['PC1']),
        local_variants=names[:3],dominance_variants=names[:3],
        prior=dict(additive=.5,interaction=.05,residual=1.),storage='packed',solver=dict(rtol=1e-10))
    (tmp_path/'train.json').write_text(json.dumps(train))
    controls=['--num-threads',str(threads),'--memory-gib','2']
    recipe=dict(kind='summit.epistasis.trans_inputs',schema_version=1,
        genotypes=dict(geno='g.bed',genome_build='GRCh37',content_identity=True),target='v0',
        background_chromosome='5',training_samples='training.tsv',confirmation_samples='confirmation.tsv',
        phenotype=dict(file='phenotypes.tsv',column='y',unit='cm'),
        covariates=dict(file='cov.tsv',columns=['PC1'],varying_effects=['PC1']),
        inference=dict(method='conditional_polygenic_mean'))
    (tmp_path/'recipe.json').write_text(json.dumps(recipe))
    main(['make-inputs',str(tmp_path/'recipe.json'),'--out',str(tmp_path/'constructed'),'--num-threads',str(threads)])
    weighted_recipe=json.loads(json.dumps(recipe))
    weighted_recipe['inference']['moment_weighting']='genotype_diagonal'
    (tmp_path/'weighted-recipe.json').write_text(json.dumps(weighted_recipe))
    main(['make-inputs',str(tmp_path/'weighted-recipe.json'),'--out',str(tmp_path/'weighted-constructed'),
        '--num-threads',str(threads)])
    weighted_preparation=json.loads((tmp_path/'weighted-constructed/prepare.json').read_text())
    assert weighted_preparation['inference']['moment_weighting']=='genotype_diagonal'
    for command,name,destination in [('train-direction','train.json','trained'),('prepare','prepare.json','prepared')]:
        main([command,str(tmp_path/'constructed'/name),'--out',str(tmp_path/'constructed'/destination),*controls])
    built=json.loads((tmp_path/'constructed/prepared/preparation.json').read_text())
    assert len(built['summaries'])==1
    assert built['scoring']['adaptive_blocks']
    assert (built['scoring']['allocated_output_bytes']+built['scoring']['estimated_scratch_bytes']
        +built['scoring']['model_and_axis_bytes']+built['scoring']['retained_reference_bytes'])<=2*2**30
    assert len(built['covariance_precision'])==1
    assert built['covariance_precision'][0]['fixed_contrast_variance_relative_sd']>0
    # Automatic input construction must retain the supplied random-slope
    # covariance family. Scaling z rescales its variance component; centering
    # z would also require an additive/slope cross-covariance component.
    constructed=tmp_path/'constructed'
    unscaled=json.loads((constructed/'train.json').read_text())
    unscaled['covariates']['file']='../cov.tsv'
    (constructed/'unscaled-train.json').write_text(json.dumps(unscaled))
    unscaled_prepare=json.loads((constructed/'prepare.json').read_text())
    unscaled_prepare['training']='unscaled-train.json'
    unscaled_prepare['directions'][0]['direction']='unscaled-trained/direction.json'
    (constructed/'unscaled-prepare.json').write_text(json.dumps(unscaled_prepare))
    main(['train-direction',str(constructed/'unscaled-train.json'),
        '--out',str(constructed/'unscaled-trained'),*controls])
    main(['prepare',str(constructed/'unscaled-prepare.json'),
        '--out',str(constructed/'unscaled-prepared'),*controls])
    built_fit=robust_score_tests(load_robust_scores(constructed/'prepared/y.robust-score.npz'))
    # The learner is linear in the outcome after removing the declared mean.
    # Changing cm to mm and adding a finite covariate mean multiplies its raw
    # interaction score by ten and the fitted covariance by 100. Because the
    # final feature uses that score, its response-normalized coefficient and
    # coefficient SE remain invariant (they are in raw learner-score units).
    converted=pd.read_csv(tmp_path/'phenotypes.tsv',sep='\t',dtype={'FID':str,'IID':str})
    converted['y']=10*y[:,0]+3*z
    converted.to_csv(constructed/'units.tsv',sep='\t',index=False)
    units_train=json.loads((constructed/'train.json').read_text())
    units_train['phenotype'].update(file='units.tsv',unit='mm')
    (constructed/'units-train.json').write_text(json.dumps(units_train))
    units_prepare=json.loads((constructed/'prepare.json').read_text())
    units_prepare.update(kind='summit.epistasis.prepare_traits',training='units-train.json')
    units_prepare['phenotypes'].update(file='units.tsv',unit='mm')
    units_prepare['directions'][0]['direction']='units-trained/direction.json'
    units_prepare['inference']['reference']='prepared/cohort-reference.npz'
    (constructed/'units-prepare.json').write_text(json.dumps(units_prepare))
    main(['train-direction',str(constructed/'units-train.json'),
        '--out',str(constructed/'units-trained'),*controls])
    main(['prepare-traits',str(constructed/'units-prepare.json'),
        '--out',str(constructed/'units-prepared'),*controls])
    units_fit=robust_score_tests(load_robust_scores(constructed/'units-prepared/y.robust-score.npz'))
    for field in ('beta','coefficient_covariance','kernel_p'):
        np.testing.assert_allclose(units_fit[field],built_fit[field],rtol=3e-6,atol=3e-8)
    units_record=json.loads((constructed/'units-prepared/preparation.json').read_text())
    assert units_record['reference_reused'] and not units_record['training_covariance_reused']
    np.testing.assert_allclose(units_record['covariance_components'],
        100*np.asarray(built['covariance_components']),rtol=5e-8,atol=1e-8)
    assert units_fit['diagnostics']['trait_unit']=='mm'
    for field in ('sampling_relative_sd','trace_relative_sd','fixed_contrast_variance_relative_sd'):
        np.testing.assert_allclose(units_fit['diagnostics']['covariance_precision'][field],
            built_fit['diagnostics']['covariance_precision'][field],rtol=1e-6)
    unscaled_fit=robust_score_tests(load_robust_scores(constructed/'unscaled-prepared/y.robust-score.npz'))
    for field in ('beta','coefficient_covariance','kernel_p'):
        np.testing.assert_allclose(built_fit[field],unscaled_fit[field],rtol=3e-6,atol=3e-8)
    unscaled_record=json.loads((constructed/'unscaled-prepared/preparation.json').read_text())
    expected_theta=np.asarray(unscaled_record['covariance_components']).copy()
    expected_theta[unscaled_record['kernel_names'].index('additive_by_PC1')]*=z[:n0].std()**2
    np.testing.assert_allclose(built['covariance_components'],expected_theta,rtol=5e-8,atol=1e-9)
    with np.load(constructed/'prepared/cohort-reference.npz') as left, \
         np.load(constructed/'unscaled-prepared/cohort-reference.npz') as right:
        qleft,qright=orth(left['fixed']),orth(right['fixed'])
        assert qleft.shape==qright.shape
        np.testing.assert_allclose(qleft-qright@(qright.T@qleft),0.,atol=2e-12)
    built_cov=pd.read_csv(constructed/'covariates.tsv',sep='\t')
    np.testing.assert_allclose(built_cov['PC1'].to_numpy()*z[:n0].std(),z,atol=2e-14)
    scaling=json.loads((constructed/'inputs.json').read_text())['covariate_scaling']
    assert scaling['preserved_origins']==['PC1'] and scaling['center']==[0.]
    main(['train-direction',str(tmp_path/'train.json'),'--out',str(tmp_path/'trained'),*controls])
    spec=dict(kind='summit.epistasis.prepare',schema_version=1,training='train.json',samples='confirmation.tsv',
        phenotypes=dict(file='phenotypes.tsv',columns=['y','new_trait'],unit='cm'),
        directions=[dict(phenotype=name,direction=f'trained/direction.{i}.json') for i,name in enumerate(['y','new_trait'])],
        inference=dict(method='conditional_polygenic_mean'))
    (tmp_path/'prepare.json').write_text(json.dumps(spec))
    legacy=json.loads((tmp_path/'trained/direction.0.json').read_text())
    legacy.pop('additive_model_identity')
    (tmp_path/'legacy-direction.json').write_text(json.dumps(legacy))
    legacy_spec=json.loads(json.dumps(spec))
    legacy_spec['directions'][0]['direction']='legacy-direction.json'
    (tmp_path/'legacy-prepare.json').write_text(json.dumps(legacy_spec))
    from summit.epistasis import conditional_workflow
    with monkeypatch.context() as preflight:
        preflight.setattr(conditional_workflow,'source_from_spec',
            lambda *a,**k:pytest.fail('genotypes opened before rejecting legacy learner'))
        with pytest.raises(ValueError,match='requires matched interaction and additive-null models'):
            main(['prepare',str(tmp_path/'legacy-prepare.json'),'--out',str(tmp_path/'legacy-prepared'),*controls])
    assert not (tmp_path/'legacy-prepared').exists()
    main(['prepare',str(tmp_path/'prepare.json'),'--out',str(tmp_path/'prepared'),'--exact',*controls])
    result=json.loads((tmp_path/'prepared/preparation.json').read_text())
    assert len(result['summaries'])==2 and not result['reference_reused']
    assert not result['training_covariance_reused']
    assert 'polygenic_setup' not in result['genotype_traversals']['confirmation']['traversals']
    weighted_spec=json.loads(json.dumps(spec))
    weighted_spec['inference']['moment_weighting']='genotype_diagonal'
    (tmp_path/'weighted-prepare.json').write_text(json.dumps(weighted_spec))
    main(['prepare',str(tmp_path/'weighted-prepare.json'),'--out',str(tmp_path/'weighted-prepared'),'--exact',*controls])
    weighted_result=json.loads((tmp_path/'weighted-prepared/preparation.json').read_text())
    assert weighted_result['covariance_moment_weighting']=='genotype_diagonal'
    assert len(weighted_result['summaries'])==2
    with np.load(tmp_path/'weighted-prepared/cohort-reference.npz') as archive:
        weighted_manifest=json.loads(str(archive['manifest']))
        assert weighted_manifest['schema_version']==3
        assert np.all(archive['moment_weights']>0)
        weighted_identity=weighted_manifest['metadata']['identity']
    with np.load(tmp_path/'prepared/cohort-reference.npz') as archive:
        assert json.loads(str(archive['manifest']))['metadata']['identity']!=weighted_identity
    main(['prepare',str(tmp_path/'weighted-prepare.json'),'--out',str(tmp_path/'weighted-prepared'),
        '--exact','--resume',*controls])
    weighted_spec['inference']['reference']='prepared/cohort-reference.npz'
    (tmp_path/'wrong-weighted-reference.json').write_text(json.dumps(weighted_spec))
    with pytest.raises(ValueError,match='reference sample, variant or mean inputs changed'):
        main(['prepare',str(tmp_path/'wrong-weighted-reference.json'),'--out',str(tmp_path/'wrong-weighted-prepared'),
            '--exact',*controls])
    # Interrupt after the outcome-dependent covariance has been published.
    # Recovery must omit its full-marker moment product and retain the same
    # public summaries that are checked against dense matrices below.
    from summit.epistasis import conditional_workflow
    def stop_after_covariance(*args, **kwargs):
        raise RuntimeError('interruption after training covariance')
    with monkeypatch.context() as interrupted:
        interrupted.setattr(conditional_workflow,'conditional_scores_batch',stop_after_covariance)
        with pytest.raises(RuntimeError,match='interruption after training covariance'):
            main(['prepare',str(tmp_path/'prepare.json'),'--out',str(tmp_path/'covariance-recovery'),
                '--exact',*controls])
    covariance_checkpoint=tmp_path/'covariance-recovery/training-covariance.npz'
    saved_covariance=covariance_checkpoint.read_bytes()
    with np.load(covariance_checkpoint,allow_pickle=False) as archive:
        altered={k:archive[k] for k in archive.files}
    altered['coefficients'][0,0]+=.01
    np.savez(covariance_checkpoint,**altered)
    with pytest.raises(ValueError,match='covariance checkpoint inputs or values changed'):
        main(['prepare',str(tmp_path/'prepare.json'),'--out',str(tmp_path/'covariance-recovery'),
            '--exact','--resume',*controls])
    covariance_checkpoint.write_bytes(saved_covariance)
    with monkeypatch.context() as recovered:
        recovered.setattr(conditional_workflow,'estimate_components',
            lambda *a,**k:pytest.fail('training covariance moment product repeated'))
        main(['prepare',str(tmp_path/'prepare.json'),'--out',str(tmp_path/'covariance-recovery'),
            '--exact','--resume',*controls])
    recovery=json.loads((tmp_path/'covariance-recovery/preparation.json').read_text())
    assert recovery['training_covariance_reused']
    assert 'polygenic_covariance_precision' not in recovery['genotype_traversals']['training']['traversals']
    assert recovery['covariance_precision_implementation']=='directional_exact_contraction_v1'
    assert recovery['genotype_traversals']['training']['traversals']['polygenic_directional_precision']==1
    assert 'polygenic_trait_moments' not in recovery['genotype_traversals']['training']['traversals']
    for column in ['y','new_trait']:
        for field in ('scores','information','score_covariance'):
            np.testing.assert_allclose(
                getattr(load_robust_scores(tmp_path/f'covariance-recovery/{column}.robust-score.npz'),field),
                getattr(load_robust_scores(tmp_path/f'prepared/{column}.robust-score.npz'),field),rtol=2e-12,atol=2e-12)
    with np.load(tmp_path/'prepared/cohort-reference.npz') as archive:
        ref={k:archive[k] for k in archive.files if k!='manifest'}
    # Independently construct the finite mean and covariance from observed calls.
    observed=np.isfinite(raw)
    means=np.nanmean(raw[:n0],0)
    hmean=np.sum((raw[:n0]==1),0)/observed[:n0].sum(0)
    a=np.where(observed,raw,means)
    h=np.where(observed,raw==1,hmean)
    c=np.column_stack([np.ones(n),z,a[:,:3],h[:,:3],z*z,z*x])
    q=orth(c); qref=orth(ref['fixed'])
    assert q.shape[1]==qref.shape[1]
    np.testing.assert_allclose(q-qref@(qref.T@q),0,atol=2e-13)
    kernels=[]
    for k,values in enumerate((raw,raw==1)):
        g=np.where(observed[:,1:],values[:,1:]-ref['mean'][k],0.)*ref['inverse_scale'][k]
        if k==0:
            kernels.extend([g@g.T/(m-1),(g*z[:,None])@(g*z[:,None]).T/(m-1)])
        else:
            kernels.append(g@g.T/(m-1))
    kernels.extend([np.eye(n),np.diag(x*x)])
    kernels=np.stack(kernels)
    models={model.key:model for model in load_prediction_models(tmp_path/'trained/models')}
    for j,column in enumerate(['y','new_trait']):
        model=models[(f'direction.{j}','prespecified')]
        additive=models[(f'direction.{j}','additive_null')]
        def score(model,k):
            g=np.where(observed[:,1:],raw[:,1:]-model.scale.mean,0.)*model.scale.inverse_scale
            return g@model.weights[:,k]
        e,pgs=score(model,1),score(additive,0)
        theta,_=fit_covariance(y[:n0,j],c[:n0],kernels[:,:n0,:n0])
        np.testing.assert_allclose(theta,np.asarray(result['covariance_components'])[:,j],atol=2e-10)
        v=np.einsum('k,kij->ij',theta,kernels)
        transfer,cov=conditional_null(v,c,n0)
        tangent=conditional_mean_tangents(v,kernels,c,n0,y[:n0,j])
        extras=np.column_stack([e,pgs,z*e,z*pgs])[n0:]
        expected=innovation_score(y[:n0,j],y[n0:,j],(x*e)[:n0],(x*e)[n0:],
            np.column_stack([c[n0:],extras,tangent]),transfer,cov)
        summary=load_robust_scores(tmp_path/f'prepared/{column}.robust-score.npz')
        assert summary.feature_names==('v0_by_frozen_trans_score',)
        fitted=robust_score_tests(summary)
        np.testing.assert_allclose(fitted['beta'],expected['beta'],atol=3e-7,rtol=2e-6)
        np.testing.assert_allclose(fitted['coefficient_covariance'],expected['covariance'],atol=3e-7,rtol=2e-6)
        with monkeypatch.context() as guard:
            guard.setattr(FileGenotypeSource,'__init__',lambda *a,**k:pytest.fail('genotypes read during fit'))
            main(['fit',str(tmp_path/f'prepared/{column}.robust-score.npz'),'--out',str(tmp_path/f'{column}.json')])
    # Reuse genotype moments for a different outcome and its own matched learner.
    # This is cohort-side preparation, not fixed old uncertainty or genotype-free.
    reused=json.loads(json.dumps(spec));reused['kind']='summit.epistasis.prepare_traits'
    reused['phenotypes']['columns']=['new_trait'];reused['directions']=reused['directions'][1:]
    # prepare-traits uses random probes, so first form a corresponding ordinary
    # reference. Its new-trait batch and separate calculations must agree.
    main(['prepare',str(tmp_path/'prepare.json'),'--out',str(tmp_path/'random-prepared'),*controls])
    reused['inference']['reference']='random-prepared/cohort-reference.npz'
    (tmp_path/'reuse.json').write_text(json.dumps(reused))
    main(['prepare-traits',str(tmp_path/'reuse.json'),'--out',str(tmp_path/'reused'),
        '--block-size','64',*controls])
    for field in ('scores','information','score_covariance'):
        np.testing.assert_allclose(getattr(load_robust_scores(tmp_path/'reused/new_trait.robust-score.npz'),field),
            getattr(load_robust_scores(tmp_path/'random-prepared/new_trait.robust-score.npz'),field),rtol=2e-7,atol=1e-9)
    from scripts.epistasis.verify_conditional_batch import compare
    compared=compare(tmp_path/'random-prepared',tmp_path/'reused')
    assert compared['verified'] and compared['overlapping_learners']==1
    assert compared['additional_learning_replicates']==0
    # A differently scaled covariance-family recipe has equivalent numerical
    # answers above, but is not the same authenticated learning replicate.
    with pytest.raises(ValueError,match='genotype_reference differs'):
        compare(constructed/'prepared',constructed/'unscaled-prepared')
    rerun=json.loads((tmp_path/'reused/preparation.json').read_text())
    assert rerun['reference_reused']
    assert not rerun['training_covariance_reused']
    assert rerun['genotype_traversals']['training']['traversals']['polygenic_trait_moments']==1
    # Reuse must expose the same block-size and recovery controls as prepare;
    # a completed resume preserves the published summary byte for byte.
    saved=(tmp_path/'reused/new_trait.robust-score.npz').read_bytes()
    main(['prepare-traits',str(tmp_path/'reuse.json'),'--out',str(tmp_path/'reused'),
        '--block-size','64','--resume',*controls])
    assert (tmp_path/'reused/new_trait.robust-score.npz').read_bytes()==saved
    main(['prepare',str(tmp_path/'prepare.json'),'--out',str(tmp_path/'prepared'),'--exact','--resume',*controls])
    # Reverse target and background allele coding, preserving missing calls and
    # participant outcomes. Independent refitting must preserve x*learned_score,
    # the nuisance span and inference; an old genotype reference must not load.
    flipped=raw.copy(); changed_alleles=[0,4,17]
    flipped[:,changed_alleles]=2-flipped[:,changed_alleles]
    to_bed(tmp_path/'flipped.bed',flipped,properties=dict(fid=samples,iid=samples,sid=names,
        chromosome=['12']*3+['5']*(m-3),bp_position=list(range(1,m+1)),
        allele_1=['C' if k in changed_alleles else 'A' for k in range(m)],
        allele_2=['A' if k in changed_alleles else 'C' for k in range(m)]))
    flip_train=json.loads(json.dumps(train));flip_train['genotypes']['geno']='flipped.bed'
    (tmp_path/'flip-train.json').write_text(json.dumps(flip_train))
    main(['train-direction',str(tmp_path/'flip-train.json'),'--out',str(tmp_path/'flip-trained'),*controls])
    flip_spec=json.loads(json.dumps(spec));flip_spec['training']='flip-train.json'
    for i,d in enumerate(flip_spec['directions']):d['direction']=f'flip-trained/direction.{i}.json'
    (tmp_path/'flip-prepare.json').write_text(json.dumps(flip_spec))
    main(['prepare',str(tmp_path/'flip-prepare.json'),'--out',str(tmp_path/'flip-prepared'),'--exact',*controls])
    for column in ['y','new_trait']:
        expected=robust_score_tests(load_robust_scores(tmp_path/f'prepared/{column}.robust-score.npz'))
        actual=robust_score_tests(load_robust_scores(tmp_path/f'flip-prepared/{column}.robust-score.npz'))
        for field in ('beta','coefficient_covariance','kernel_p'):
            np.testing.assert_allclose(actual[field],expected[field],rtol=2e-6,atol=3e-8)
    flip_spec['inference']['reference']='prepared/cohort-reference.npz'
    (tmp_path/'flip-invalid-reference.json').write_text(json.dumps(flip_spec))
    with pytest.raises(ValueError,match='reference sample, variant or mean inputs changed'):
        main(['prepare',str(tmp_path/'flip-invalid-reference.json'),'--out',str(tmp_path/'flip-invalid'),
            '--exact',*controls])
    bad=pd.read_csv(tmp_path/'phenotypes.tsv',sep='\t',dtype={'FID':str,'IID':str})
    bad.loc[0,'new_trait']+=.2;bad.to_csv(tmp_path/'phenotypes.tsv',sep='\t',index=False)
    with pytest.raises(ValueError,match='phenotype, nuisance or target changed'):
        main(['prepare-traits',str(tmp_path/'reuse.json'),'--out',str(tmp_path/'changed'),*controls])
    cv=pd.read_csv(tmp_path/'cov.tsv',sep='\t',dtype={'FID':str,'IID':str});cv.loc[n0,'PC1']+=.2
    cv.to_csv(tmp_path/'cov.tsv',sep='\t',index=False)
    with pytest.raises(ValueError,match='reference sample, variant or mean inputs changed'):
        main(['prepare-traits',str(tmp_path/'reuse.json'),'--out',str(tmp_path/'changed-covariate'),*controls])
