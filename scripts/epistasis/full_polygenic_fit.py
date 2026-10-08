"""Full-marker development of estimated conditional polygenic mean inference.

Reuse independently fitted matched learners, never their old uncertainty.
Every covariance and conditional outcome prediction is fitted to that learner's
training phenotype. Simulation means are read only for post-fit diagnostics.
This research driver is not yet the public preparation/portable-summary path.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from scipy.stats import chi2, norm

from summit.prediction.artifacts import load_prediction_models
from summit.prediction.cli import _aligned_table, _rows, _variants
from summit.prediction._validation import array_digest
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.score import score_prediction, ScoreInput
from scripts.epistasis.polygenic_operator import PolygenicKernels, estimate_components, conditional_scores_batch
from scripts.epistasis.full_matched import write_json, load_reference, load_experiment_inputs
from scripts.epistasis.benchmark_robust_workflow import io


def verify_training_mean(source,spec,work,rows,model,fixed,basis,*,threads,memory_bytes):
    """Authenticate the actual public learner mean using bounded local calls."""
    from summit.prediction.genotype import ArrayGenotypeSource,estimate_scale,native_module
    from summit.prediction.spec import GenotypeScale
    from summit.epistasis.models import target_design
    from summit.epistasis.nuisance import select_structure_covariates,varying_main_effects
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
    names=set(spec.get('local_variants',[]))|set(spec.get('dominance_variants',[]))|{spec['target']}
    if len(names)>4096:
        raise ValueError('bounded public training-mean check requires at most 4096 local markers')
    lookup={v:j for j,v in enumerate(source.variants.ids)}
    selected=np.array(sorted(lookup[v] for v in names))
    source.prepare(rows,max(128,len(selected)),threads)
    raw=source.read(selected)
    samples=[source.samples[i] for i in rows]
    local=ArrayGenotypeSource(raw.copy(),samples,source.variants.subset(selected),hard_calls=True)
    empirical=estimate_scale(local,np.arange(len(rows)),np.arange(len(selected)),threads=threads)
    scale=GenotypeScale(empirical.mean,1/np.sqrt(empirical.mean*(1-empirical.mean/2)),
        empirical.variant_identity,empirical.sample_identity,dict(empirical.provenance),ddof=0)
    cv=spec['covariates']
    cov=_aligned_table(work/cv['file'],samples)[cv['columns']].to_numpy(float)
    prepared=target_design(local,np.arange(len(rows)),scale,
        components=[dict(name='interaction',target=spec['target'],background='all')],
        annotations={'all':np.ones(len(selected))},additive_annotations=['all'],covariates=cov,
        local_variants=spec.get('local_variants',[]),dominance_variants=spec.get('dominance_variants',[]),
        threads=threads,native=native_module(),memory_bytes=memory_bytes)
    structure,z=select_structure_covariates(cv,cov)
    public=prepared['fixed_effects']
    if structure:
        public,_=varying_main_effects(public,z,prepared['modifiers'][:,1:],
            covariate_names=structure,main_names=[spec['target']],memory_bytes=memory_bytes)
    provenance=model.provenance['training'][model.trait_id]
    if array_digest(public)!=provenance['fixed'] or array_digest(prepared['modifiers'])!=provenance['phi']:
        raise ValueError('public training mean or target differs from the frozen learner')
    reference=thin_rank_revealing_fixed_effect_basis(fixed,rtol=1e-11)
    if (reference.shape!=basis.shape or not np.all(np.isfinite(basis))
            or np.linalg.norm(basis.T@basis-np.eye(basis.shape[1]))>1e-8
            or np.linalg.norm(reference-basis@(basis.T@reference))>1e-8):
        raise ValueError('covariance reference mean changed after its moments were prepared')
    other=thin_rank_revealing_fixed_effect_basis(public,rtol=1e-11)
    first=float(np.linalg.norm(basis-other@(other.T@basis))/np.sqrt(basis.shape[1]))
    second=float(np.linalg.norm(other-basis@(basis.T@other))/np.sqrt(other.shape[1]))
    if max(first,second)>1e-8:
        raise ValueError('learner and conditional covariance fixed-mean spans differ')
    return dict(frozen_fixed_hash=provenance['fixed'],frozen_context_hash=provenance['phi'],
        public_rank=other.shape[1],covariance_rank=basis.shape[1],relative_span_errors=[first,second])


def _run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    a.out.chmod(0o700)
    original=json.loads((a.input/'design.json').read_text())
    count=getattr(a,'replicates',None)
    count=original['arguments']['replicates'] if count is None else count
    if not 1<=count<=original['arguments']['replicates']:
        raise ValueError('replicates must select a nonempty initial subset of the scheduled learners')
    settings=a.settings.split(',')
    if set(settings)-set(original['arguments']['settings'].split(',')):
        raise ValueError('settings require existing complete matched learning experiments')
    rm=json.loads((a.reference/'reference.json').read_text())
    with np.load(a.reference/'reference.npz') as archive:
        data={k:archive[k] for k in archive.files}
    rows,i0,i1,variants=(data[k] for k in ('rows','training_index','confirmation_index','variants'))
    scales=dict(rm['scales'],mean=data['mean'],inverse_scale=data['inverse_scale'])
    geometry=dict(rm['geometry'],**{k:data[k] for k in ('basis','gram','norms','h','probe_gram')})
    training_spec=json.loads(Path(rm['training_manifest']).read_text())
    geno=Path(rm['training_manifest']).parent/rm['genotypes']['geno']
    start,cpu,before=time.perf_counter(),time.process_time(),io()
    measurements=[]
    def measure(name,call):
        begin=time.perf_counter()
        result=call()
        measurements.append(dict(stage=name,seconds=time.perf_counter()-begin))
        print(name,round(measurements[-1]['seconds'],2),flush=True)
        return result
    records=[]
    recovery=getattr(a,'recover_from',None)
    if recovery is not None:
        previous=json.loads((recovery/'design.json').read_text())
        if (previous['parent']!=str(a.input.resolve()) or previous['reference']!=str(a.reference.resolve())
                or previous['settings']!=settings or not previous.get('covariance_mean_tangents')
                or previous['scheduled_per_setting']!=count
                or (recovery/'resources.json').exists()):
            raise ValueError('recovery requires an incomplete identical research fit')
    write_json(a.out/'design.json',dict(parent=str(a.input.resolve()),reference=str(a.reference.resolve()),
        recovery_from=None if recovery is None else str(recovery.resolve()),
        independent_new_learning_models=0,
        phase='development',primary='learned scalar conditional polygenic response with covariance-mean tangent adjustment',
        covariance_mean_tangents=True,
        kernel_block_sizes=dict(training=4096,confirmation=1024),
        parameter='response-normalized coefficient after additive-null conditional prediction; Gaussian random genetic effects with prespecified covariance basis',
        fixed_architecture_interpretation='stress test of biological-null rejection; the injected response is the error/coverage target; conditional fixed-architecture mean and noise are separate diagnostics',
        status='research path; no public portable-summary integration or qualification',settings=settings,
        thresholds=[.05,.005],material_inflation_tolerances=[.075,.01],
        scheduled_per_setting=count,learner_selection='first scheduled learners, without outcome selection'))
    with FileGenotypeSource(geno,genome_build=rm['genotypes'].get('genome_build')) as source0, \
         FileGenotypeSource(geno,genome_build=rm['genotypes'].get('genome_build')) as source1:
        if source0.identity!=rm['source']:
            raise ValueError('reference genotype inputs changed')
        participants=[source0.samples[i] for i in rows]
        controls=dict(threads=a.num_threads,block_size=512,memory_bytes=int(a.memory_gib*2**30))
        # Distinct source owners prevent a new scorer from changing a live stream.
        training=measure('training_kernel_setup',lambda:PolygenicKernels(source0,rows[i0],variants,scales,
            data['contexts'][i0],data['noise'][i0],storage='packed',**dict(controls,block_size=4096)))
        complete=measure('complete_kernel_setup',lambda:PolygenicKernels(source1,rows,variants,scales,
            data['contexts'],data['noise'],**dict(controls,block_size=1024)))
        shared_reserve=(sum(v.nbytes for v in data.values())+256*2**20
            +8*count*(8*len(rows)+(2+2*(data['contexts'].shape[1]-1))*len(i1)
                +max(2,training.count)*(24+2*data['fixed'].shape[1])*len(i0)
                +training.count*(3*len(i1)+5*len(i0))))
        total_memory=int(a.memory_gib*2**30)
        if training.base_bytes+complete.base_bytes+shared_reserve>=total_memory:
            raise MemoryError('joint training/confirmation covariance workspaces exceed memory budget')
        training.memory_bytes=total_memory-complete.base_bytes-shared_reserve
        complete.memory_bytes=total_memory-training.base_bytes-shared_reserve
        for setting in settings:
            work=a.input/f'{setting}_000'
            spec=json.loads((work/'train.json').read_text())
            for key in ('target','covariates','local_variants','dominance_variants'):
                if spec[key]!=training_spec[key]:
                    raise ValueError('matched learner and covariance reference definitions differ: '+key)
            if (not np.array_equal(np.sort(_rows(source0,work/spec['samples'])),rows[i0])
                    or not np.array_equal(_variants(source0,work/spec['variants']),variants)):
                raise ValueError('matched training participant or variant axes differ')
            columns=spec['phenotypes']['columns']
            if len(columns)!=original['arguments']['replicates']:
                raise ValueError('this development driver requires one complete shared-mask batch')
            columns=columns[:count]
            y=_aligned_table(work/spec['phenotypes']['file'],participants)[columns].to_numpy(float)
            directory=a.out/setting; directory.mkdir()
            if recovery is not None:
                for name in ('outcome_feature.npz','mean_derivative.npz','covariance_contrast.npz'):
                    old=recovery/setting/name
                    if old.exists():
                        # Preserve the interrupted generation. SolverCheckpoint
                        # authenticates phenotype, operators, solver and native
                        # identity before the copied vectors may be resumed.
                        with old.open('rb') as src,(directory/name).open('xb') as dst:
                            shutil.copyfileobj(src,dst)
            theta=measure(setting+'/covariance',lambda:estimate_components(training,y[i0],geometry))
            write_json(directory/'components.json',dict(names=rm['kernel_names'],traits=columns,estimates=theta,
                covariance_reference=str(a.reference.resolve())))
            models=load_prediction_models(work/'trained/models')
            lookup={m.identity:m for m in models}
            directions=[json.loads((work/f'trained/direction.{j}.json').read_text()) for j in range(len(columns))]
            for j,d in enumerate(directions):
                model=lookup[d['model_identity']]
                if (model.scale.sample_identity!=scales['samples']
                        or model.provenance['training'][model.trait_id]['phenotype']!=array_digest(y[i0,j])):
                    raise ValueError('training outcomes changed after direction learning')
            with FileGenotypeSource(geno,genome_build=rm['genotypes'].get('genome_build')) as checking_source:
                verified=verify_training_mean(checking_source,spec,work,rows[i0],lookup[directions[0]['model_identity']],
                    data['fixed'][i0],data['basis'],threads=a.num_threads,memory_bytes=int(a.memory_gib*2**30))
            for d in directions[1:]:
                model=lookup[d['model_identity']]
                declared=model.provenance['training'][model.trait_id]
                if declared['fixed']!=verified['frozen_fixed_hash'] or declared['phi']!=verified['frozen_context_hash']:
                    raise ValueError('batched learners used different training nuisance designs')
            write_json(directory/'training_mean.json',verified)
            selected={d['model_identity']:lookup[d['model_identity']] for d in directions}
            selected.update({d.get('additive_model_identity',d['model_identity']):lookup[d.get('additive_model_identity',d['model_identity'])] for d in directions})
            inputs={m.trait_id:ScoreInput(rows,np.ones((len(rows),m.weights.shape[1])),np.empty((len(rows),0)),
                m.context_spec,m.fixed_spec) for m in selected.values()}
            with FileGenotypeSource(geno,genome_build=rm['genotypes'].get('genome_build')) as scoring_source:
                scored=measure(setting+'/frozen_scores',lambda:score_prediction(selected.values(),scoring_source,
                    inputs,components_only=True,**controls))
            e=np.column_stack([scored.components[lookup[d['model_identity']].key][:,1] for d in directions])
            pgs=np.column_stack([scored.components[lookup[d.get('additive_model_identity',d['model_identity'])].key][:,0] for d in directions])
            f=data['target'][:,None]*e
            z=data['contexts'][i1,1:]
            extras=[np.column_stack([e[i1,j],pgs[i1,j],z*e[i1,j,None],z*pgs[i1,j,None]]) for j in range(len(columns))]
            fit=measure(setting+'/conditional_fit',lambda:conditional_scores_batch(training,complete,i0,i1,
                y[i0],y[i1],f[i0],f[i1],data['fixed'][i0],data['fixed'][i1],extras,theta,
                checkpoint_dir=directory,mean_tangents=True))
            # Truth is never supplied to estimated covariance or either solve.
            meta,truth,_=load_experiment_inputs(Path(original['reference']),threads=a.num_threads,
                local_stress=setting.startswith('local_withheld'))
            take=np.searchsorted(truth['rows'],rows)
            if not np.array_equal(truth['rows'][take],rows):
                raise ValueError('diagnostic generating distribution/sample mismatch')
            index=meta['settings'].index(setting)
            signal=truth['signals'][take,index]*original['arguments']['signal_multiplier']
            mean=truth['means'][take,index]-truth['signals'][take,index]+signal
            noise=truth['variance'][take]
            for j,column in enumerate(columns):
                a1,b0=fit['contrasts'][:,j],fit['training_contrasts'][:,j]
                conditional_truth=float(a1@mean[i1]+b0@y[i0,j])
                generating_projection=float(a1@mean[i1]+b0@mean[i0])
                signal_response=float(a1@signal[i1]+b0@signal[i0])
                beta,se=float(fit['beta'][j]),float(np.sqrt(fit['variance'][j]))
                public=load_reference(work/f'prepared/{column}_learned.cohort-reference.npz')
                conversion=np.column_stack([f[i1,j],e[i1,j]])
                coef=np.linalg.lstsq(conversion,public.features[:,0],rcond=1e-11)[0]
                conversion_error=float(np.linalg.norm(public.features[:,0]-conversion@coef)/np.linalg.norm(public.features[:,0]))
                if conversion_error>1e-8:
                    raise ArithmeticError('frozen full-cohort scoring differs from public interaction feature span')
                baseline=json.loads((work/f'{column}_learned.fit.json').read_text())['fits'][0]
                known_confirmation_se=float(np.sqrt(np.dot(a1*a1,noise[i1])))
                conditional_rates={}
                if setting!='heavy':
                    for alpha in (.05,.005):
                        threshold=norm.isf(alpha/2)*se
                        conditional_rates[str(alpha)]=float(norm.cdf((-threshold-conditional_truth)/known_confirmation_se)
                            +norm.sf((threshold-conditional_truth)/known_confirmation_se))
                records.append(dict(setting=setting,replicate=j,method='estimated_conditional_polygenic',
                    failed=False,biological_null=meta['definitions'][setting]['biological_null'],beta=beta,se=se,
                    p=float(chi2.sf(beta*beta/se**2,1)),conditional_fixed_architecture_mean=conditional_truth,
                    error=beta-signal_response,coverage=bool(abs(beta-signal_response)<=1.959963984540054*se),
                    conditional_mean_error=beta-conditional_truth,
                    conditional_mean_coverage=bool(abs(beta-conditional_truth)<=1.959963984540054*se),
                    known_confirmation_noise_se=known_confirmation_se,
                    fixed_architecture_conditional_rejection_probability=conditional_rates,
                    fixed_contrast_noise_se=float(np.sqrt(np.dot(a1*a1,noise[i1])+np.dot(b0*b0,noise[i0]))),
                    generating_mean_projection=generating_projection,interaction_response=signal_response,
                    residual_noninteraction_projection=generating_projection-signal_response,
                    remaining_signal_variance=float(np.var(signal[i1])),
                    public_feature_conversion=coef,public_feature_relative_error=conversion_error,
                    old_public_baseline_p=baseline['kernel_p'],**fit['diagnostics'][j]))
            with (a.out/'replicates.jsonl').open('a') as handle:
                from summit.epistasis.cli import _jsonable
                for record in records[-len(columns):]: handle.write(json.dumps(_jsonable(record))+'\n')
            write_json(directory/'solver.json',dict(reports=[{'/'.join(k):v for k,v in report.items()} for report in fit['solver_reports']]))
        after=io()
        write_json(a.out/'resources.json',dict(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            io={k:after[k]-v for k,v in before.items()},stages=measurements,
            training_ledger=asdict(training.stream.ledger),complete_ledger=asdict(complete.stream.ledger)))


def run(a):
    """Keep every scheduled fit in the failure ledger if a batch cannot finish."""
    existed=a.out.exists()
    start,cpu=time.perf_counter(),time.process_time()
    try:
        return _run(a)
    except Exception as error:
        design_path=a.out/'design.json'
        if not existed and design_path.exists() and not (a.out/'resources.json').exists():
            design=json.loads(design_path.read_text())
            path=a.out/'replicates.jsonl'
            records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            completed={(r['setting'],r['replicate']) for r in records}
            missing=[dict(setting=setting,replicate=rep,method='estimated_conditional_polygenic',
                failed=True,reason=f'{type(error).__name__}: {error}')
                for setting in design['settings'] for rep in range(design['scheduled_per_setting'])
                if (setting,rep) not in completed]
            with path.open('a') as handle:
                for record in missing: handle.write(json.dumps(record)+'\n')
            write_json(a.out/'failure.json',dict(error_type=type(error).__name__,reason=str(error),
                completed=len(completed),failed_or_unexecuted=len(missing),
                scheduled=len(design['settings'])*design['scheduled_per_setting'],
                seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        raise


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--settings',default='structure,structure_mixed')
    p.add_argument('--recover-from',type=Path)
    p.add_argument('--replicates',type=int)
    p.add_argument('--num-threads',type=int,default=1);p.add_argument('--memory-gib',type=float,default=32)
    run(p.parse_args())


if __name__=='__main__':
    main()
