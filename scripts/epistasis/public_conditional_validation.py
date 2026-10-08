"""Diagnose matched simulations after the actual public conditional workflow.

The public pipeline sees only its declared cohort inputs. Generating quantities
are opened after preparation and portable fitting complete. Instrumentation
retains its already-computed contrasts; it never replaces a fitting calculation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from scipy.stats import norm, beta as beta_dist

from summit.epistasis.cli import main as cli
from summit.epistasis import conditional_workflow
from summit.epistasis import polygenic
from summit.epistasis.polygenic import PolygenicKernels, projected_solve
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.prediction.genotype import source_from_spec
from scripts.epistasis.full_matched import write_json
from scripts.epistasis.replicate_schedule import simulation_replicates, replicate_columns


def diagnostic_identity(a,spec):
    """Bind recovery to the completed public fit and its actual small inputs."""
    from summit.context.spec import canonical_sha256
    from summit.prediction.artifacts import file_digest
    root=a.manifest.parent;training=(root/spec['training']).resolve()
    train=json.loads(training.read_text());phenotype=train.get('phenotypes',train.get('phenotype'))
    preparation=a.out/'prepared/preparation.json'
    record=json.loads(preparation.read_text())
    paths=[a.manifest,training,root/spec['phenotypes']['file'],training.parent/phenotype['file'],
        preparation,Path(record['reference']),Path(__file__),
        Path(__file__).with_name('conditional_comparators.py'),Path(__file__).with_name('weighted_t_reference.py')]
    paths += [root/d['direction'] for d in spec['directions']]
    paths += [a.out/(column+'.fit.json') for column in spec['phenotypes']['columns']]
    paths += [a.out/'prepared'/r['file'] for r in record['summaries']]
    # This function runs only after public preparation/fitting, so binding the
    # generating reference here cannot influence the fitted procedure.
    simulation=json.loads(a.simulation.read_text())
    paths += [a.simulation,a.simulation.parent/a.setting/'diagnostic_truth.npz',
        Path(simulation['signal_reference'])/'reference.json',Path(simulation['signal_reference'])/'reference.npz']
    return canonical_sha256({str(p.resolve()):file_digest(p) for p in paths})


def save_diagnostic_state(a,spec,values,public_seconds):
    """Cohort-only restart state; do not duplicate the large finite mean arrays."""
    from summit.context.spec import array_sha256
    from summit.epistasis.summary import _publish_bundle
    arrays={k:values[k] for k in ('y0','y1','f0','f1','theta','tangents')}
    arrays['score_main']=np.stack([v[:,:2] for v in values['extras']],axis=1)
    for k in ('beta','variance','contrasts','training_contrasts','response','prediction'):
        arrays['fit_'+k]=values['fit'][k]
    _publish_bundle(a.out/'diagnostic-inputs.npz',dict(kind='cohort_conditional_validation_state',schema=1,
        identity=diagnostic_identity(a,spec),public_seconds=public_seconds,
        diagnostics=values['fit']['diagnostics'],digests={k:array_sha256(v) for k,v in arrays.items()}),arrays)


def load_diagnostic_state(a,spec):
    from summit.context.spec import array_sha256
    path=a.out/'diagnostic-inputs.npz'
    with ZipFile(path) as archive:
        if 3*sum(v.file_size for v in archive.infolist())+256*2**20>a.memory_gib*2**30:
            raise MemoryError('cohort diagnostic restart exceeds memory budget')
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['manifest']))
        if (metadata.get('kind')!='cohort_conditional_validation_state' or metadata.get('schema')!=1
                or metadata['identity']!=diagnostic_identity(a,spec)):
            raise ValueError('diagnostic restart inputs or implementation changed')
        arrays={k:archive[k] for k in archive.files if k!='manifest'}
    if metadata['digests']!={k:array_sha256(v) for k,v in arrays.items()}:
        raise ValueError('diagnostic restart array digest mismatch')
    record=json.loads((a.out/'prepared/preparation.json').read_text())
    with np.load(record['reference'],allow_pickle=False) as ref:
        i0,i1=ref['training_index'],ref['confirmation_index'];fixed=ref['fixed'];z=ref['contexts'][i1,1:]
    main=arrays.pop('score_main');fit={k[4:]:arrays.pop(k) for k in list(arrays) if k.startswith('fit_')}
    fit['diagnostics']=metadata['diagnostics']
    arrays.update(i0=i0,i1=i1,c0=fixed[i0],c1=fixed[i1],fit=fit,
        extras=[np.column_stack([main[:,j],z*main[:,j,0,None],z*main[:,j,1,None]]) for j in range(main.shape[1])])
    return arrays,metadata['public_seconds']


def summarize(records,scheduled):
    """Keep learner denominators distinct from analytic conditional tail rates."""
    good=[r for r in records if not r['failed']]
    if len(records)>scheduled or len({r['replicate'] for r in records})!=len(records):
        raise ValueError('scheduled learners must have distinct replicate IDs')
    failures=scheduled-len(good)
    result=dict(scheduled=scheduled,completed=len(good),failures=failures,
        unsupported=sum(bool(r.get('outside_confirmation_design')) for r in good),
        conditional_rates_are='generating-law tail probabilities for each distinct fitted learner; not extra learning replicates')
    if not good:return result
    def interval(hits,n):
        return [0. if not hits else float(beta_dist.ppf(.025,hits,n-hits+1)),
                1. if hits==n else float(beta_dist.ppf(.975,hits+1,n-hits))]
    error=np.array([r['error'] for r in good]);se=np.array([r['se'] for r in good])
    displacement=np.array([r['conditional_mean_bias'] for r in good])
    noise_variance=np.array([r['known_conditional_se']**2 for r in good])
    coverage=sum(r['coverage'] for r in good)
    result.update(bias=float(error.mean()),error_sd=float(error.std(ddof=1)) if len(good)>1 else None,
        mean_se=float(se.mean()),rms_se=float(np.sqrt(np.mean(se*se))),
        coverage=coverage/len(good),coverage_mc95=interval(coverage,len(good)),
        rms_conditional_bias_in_se=float(np.sqrt(np.mean([(r['conditional_mean_bias']/r['se'])**2 for r in good]))),
        rms_reported_to_known_se=float(np.sqrt(np.mean(se*se)/np.mean([r['known_conditional_se']**2 for r in good]))))
    # Conditional on the distinct fitted learners, errors have independent
    # confirmation innovations with means delta_j and variances sigma_j^2.
    # This expected sample variance is not the SD of changing true coefficients,
    # and sqrt(E[s^2]) is not E[s]. Only finite noise variance is required.
    expected_variance=(float(noise_variance.mean()+displacement.var(ddof=1))
        if len(good)>1 else None)
    result['conditional_error_moments']=dict(
        expected_mean=float(displacement.mean()),
        variance_of_mean=float(noise_variance.sum()/len(good)**2),
        expected_sample_variance=expected_variance,
        sqrt_expected_sample_variance=(np.sqrt(expected_variance) if expected_variance is not None else None),
        expected_mean_square=float(np.mean(noise_variance+displacement**2)),
        interpretation='Generating-law diagnostic conditional on these learners; '
            'not extra learning replicates or an estimated production uncertainty.')
    for alpha,tolerance in [(.05,.075),(.005,.01)]:
        hits=sum(r['p']<alpha for r in good)
        rates=np.array([r['conditional_rejection_probability'][str(alpha)] for r in good])
        # Resample entire training realizations. No confirmation draws are
        # counted as extra learners; the interval is an approximate learner MC CI.
        selected=np.random.default_rng(918273).integers(len(good),size=(10000,len(good)))
        mean_rates=rates[selected].mean(1)
        upper=float(np.quantile(mean_rates,.95))
        result[str(alpha)]=dict(rejected=hits,successful_denominator=len(good),scheduled_denominator=scheduled,
            observed_rate=hits/len(good),observed_mc95=interval(hits,len(good)),
            analytic_conditional_rate=float(rates.mean()),
            learner_bootstrap_mc95=np.quantile(mean_rates,[.025,.975]).tolist(),
            one_sided_mc_upper=upper,material_tolerance=tolerance,
            material_size_screen_excludes_tolerance=bool(len(good)>=10 and not failures
                and not result['unsupported'] and all(r['biological_null'] for r in good) and upper<tolerance))
        if all('conditional_coverage_probability' in r for r in good):
            probabilities=np.array([r['conditional_coverage_probability'][str(1-alpha)] for r in good])
            result[str(alpha)].update(analytic_conditional_coverage=float(probabilities.mean()),
                coverage_learner_bootstrap_mc95=np.quantile(probabilities[selected].mean(1),[.025,.975]).tolist())
    if any('comparisons' in r for r in good):
        result['comparisons']={}
        for name in ('burden','oracle'):
            rows=[]
            for r in good:
                v=r.get('comparisons',{}).get(name)
                if v is None:continue
                rows.append(dict(v,replicate=r['replicate'],failed=False,biological_null=r['biological_null'],
                    error=v['beta']-v['conditional_interaction_truth']))
            result['comparisons'][name]=summarize(rows,scheduled)
    if any('baselines' in r for r in good):
        result['baselines']={}
        for name in ('local','finite_varying_mean'):
            rows=[r['baselines'][name] for r in good if name in r.get('baselines',{})]
            result['baselines'][name]=dict(scheduled=scheduled,completed=len(rows),failures=scheduled-len(rows),
                unsupported=sum(bool(r['outside_confirmation_design']) for r in rows),
                interpretation='observed rejection using the same frozen learner; no conditional tail approximation for outcome-dependent HC3 SE')
            for level in (.05,.005):
                if not rows:continue
                hits=sum(r['p']<level for r in rows)
                result['baselines'][name][str(level)]=dict(rejected=hits,successful_denominator=len(rows),
                    scheduled_denominator=scheduled,rate=hits/len(rows),mc95=interval(hits,len(rows)))
    return result


def run(a):
    resume=getattr(a,'resume',False)
    phase=getattr(a,'phase','development')
    if phase not in ('development','confirmation'):
        raise ValueError('declare development or fresh confirmation')
    a.out.mkdir(parents=True,exist_ok=resume);a.out.chmod(0o700)
    spec=json.loads(a.manifest.read_text());root=a.manifest.parent.resolve()
    columns=spec['phenotypes']['columns']
    preparation_command=('prepare-traits' if spec['kind']=='summit.epistasis.prepare_traits' else 'prepare')
    design=dict(manifest=str(a.manifest.resolve()),setting=a.setting,
        scheduled=len(columns),phase=phase,thresholds=[.05,.005],material_inflation_tolerances=[.075,.01],
        primary='conditional_polygenic_mean_tangent_v1',
        restriction='same public preparation and independent learned direction under null and alternative',
        diagnostics='generating means/covariances are read only after public preparation and portable fit')
    if (a.out/'design.json').exists():
        if not resume or json.loads((a.out/'design.json').read_text())!=design:
            raise ValueError('validation restart definition changed')
        if (a.out/'prepared/preparation.json').exists() and not (a.out/'diagnostic-inputs.npz').exists():
            raise ValueError('completed preparation lacks cohort diagnostic state; retain public summaries and checkpoints')
    else:
        write_json(a.out/'design.json',design)
    captured={};method=conditional_workflow.conditional_scores_batch
    tangent_method=polygenic.conditional_tangents_batch
    def capture(*args,**kwargs):
        result=method(*args,**kwargs)
        for name,value in zip(('training','complete','i0','i1','y0','y1','f0','f1','c0','c1','extras','theta'),args):
            if name not in ('training','complete'):captured[name]=value
        captured['fit']=result
        return result
    def capture_tangents(*args,**kwargs):
        result=tangent_method(*args,**kwargs)
        captured['tangents']=result[0]
        return result
    start,cpu=time.perf_counter(),time.process_time()
    controls=['--num-threads',str(a.num_threads),'--memory-gib',str(a.memory_gib)]
    try:
        restored=resume and (a.out/'diagnostic-inputs.npz').exists()
        if restored:
            captured,public_seconds=load_diagnostic_state(a,spec)
            if (a.out/'results.json').exists():return json.loads((a.out/'results.json').read_text())['records']
        else:
            with patch.object(conditional_workflow,'conditional_scores_batch',capture), \
                 patch.object(polygenic,'conditional_tangents_batch',capture_tangents):
                cli([preparation_command,str(a.manifest),'--out',str(a.out/'prepared'),
                    '--block-size',str(a.block_size),*(['--resume'] if resume else []),*controls])
        fits=[]
        for column in columns:
            if not restored:
                cli(['fit',str(a.out/f'prepared/{column}.robust-score.npz'),'--out',str(a.out/(column+'.fit.json'))])
            fits.append(json.loads((a.out/(column+'.fit.json')).read_text())['fits'][0])
        if not restored:
            public_seconds=time.perf_counter()-start
            save_diagnostic_state(a,spec,captured,public_seconds)
        records=diagnose(a,spec,captured,fits)
        write_json(a.out/'results.json',dict(setting=a.setting,scheduled=len(columns),records=records,
            summary=summarize(records,len(columns)),
            public_seconds=public_seconds,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        print(a.setting,'public matched fits',len(records),round(public_seconds,2),flush=True)
        return records
    except Exception as error:
        failure=a.out/'failure.json'
        if failure.exists():failure=a.out/f'failure.{len(list(a.out.glob("failure*.json")))}.json'
        write_json(failure,dict(setting=a.setting,scheduled=len(columns),
            failed_or_unexecuted=len(columns),reason=type(error).__name__+': '+str(error),
            seconds=time.perf_counter()-start))
        raise


def outcome_reference(a1,b0,known_inverse,known_variance,*,signal,mean,variance,y0,i0,i1,definition,se):
    """Conditional diagnostic given this learner, with the stated generating law."""
    fixed=definition.get('sampling_law')=='fixed_architecture'
    target=float(a1@signal[i1]-known_inverse@signal[i0])
    displacement=(float(a1@(mean[i1]-signal[i1])+b0@(y0-signal[i0])) if fixed
        else float((known_inverse+b0)@y0))
    expected=target+displacement;known_se=float(np.sqrt(known_variance))
    rates={};errors={};coverage={};coverage_errors={}
    for level in (.05,.005):
        threshold=norm.isf(level/2)*se
        def tail(location):
            if definition.get('error_law')=='t5_unit_variance':
                if not fixed:raise ValueError('exact t5 diagnostic requires fixed genetic architecture')
                from scripts.epistasis.weighted_t_reference import weighted_t5_rejection
                result=weighted_t5_rejection(threshold,location,a1*np.sqrt(variance[i1]))
                return result['probability'],result['estimated_absolute_error']
            return float(norm.cdf((-threshold-location)/known_se)+norm.sf((threshold-location)/known_se)),0.
        rates[str(level)],errors[str(level)]=tail(expected)
        # Coverage centers on the learner-specific interaction target, not
        # the potentially displaced conditional mean. Under the null this
        # reuses the same tail; under an alternative it is distinct from power.
        missed,error=tail(displacement) if target!=0 else (rates[str(level)],errors[str(level)])
        coverage[str(1-level)]=1-missed;coverage_errors[str(1-level)]=error
    return dict(conditional_interaction_truth=target,conditional_mean_bias=displacement,
        conditional_mean=expected,known_conditional_se=known_se,conditional_rejection_probability=rates,
        tail_numerical_error=errors,conditional_coverage_probability=coverage,
        coverage_numerical_error=coverage_errors,
        diagnostic_target=('fixed-operator interaction response; fixed-architecture stress, not a Gaussian random-effect coverage guarantee'
            if fixed else 'interaction contribution conditional on training under the generating Gaussian covariance'))


def nuisance_prediction_quality(mean, predictions, basis):
    """Post-fit prediction of the noninteraction generating mean, beyond C.

    Squared alignment permits the fitted confirmation coefficient of a PGS;
    unit-scale R2 separately exposes shrinkage/miscalibration. Neither metric
    selects a predictor or enters inference. A finite mean already spanned by
    C has no remaining prediction target and is reported as undefined.
    """
    target=polygenic.project(basis,np.asarray(mean,float))
    energy=float(target@target)
    output=dict(residual_mean_variance=energy/len(target),predictors={})
    identified=energy>1e-20*max(1.,float(np.asarray(mean)@np.asarray(mean)))
    for name,prediction in predictions.items():
        value=polygenic.project(basis,np.asarray(prediction,float))
        norm2=float(value@value)
        output['predictors'][name]=dict(
            squared_alignment=(float((target@value)**2/(energy*norm2))
                if identified and norm2>1e-20*max(1.,energy) else None),
            unit_scale_r2=(float(1.-np.sum((target-value)**2)/energy) if identified else None))
    return output


def fixed_primary_records(values, public_fits, truth, definition, columns, scheduled, setting):
    """Cheap post-fit diagnostics before additional genotype-based comparisons.

    Conditioning fixes the fitted contrasts and training outcome. Only new
    confirmation errors remain random here. This does not replace the random-
    genetic-effect reference, whose conditional law needs a separate solve.
    """
    if definition.get('sampling_law') != 'fixed_architecture':
        raise ValueError('fixed primary diagnostics require a fixed generating architecture')
    i0,i1=values['i0'],values['i1'];fit=values['fit']
    mean,signal,variance=(truth[k] for k in ('mean','signal','variance'))
    records=[]
    for j,public in enumerate(public_fits):
        a,b=fit['contrasts'][:,j],fit['training_contrasts'][:,j]
        y0,y1=values['y0'][:,j],values['y1'][:,j]
        estimate=float(public['beta'][0]);se=float(np.sqrt(public['coefficient_covariance'][0][0]))
        np.testing.assert_allclose(a@y1+b@y0,estimate,rtol=2e-7,atol=2e-9)
        np.testing.assert_allclose(se*se,fit['variance'][j],rtol=2e-12,atol=2e-12)
        response=float(a@values['f1'][:,j]+b@values['f0'][:,j])
        np.testing.assert_allclose(response,1.,rtol=2e-7,atol=2e-9)
        diagnostic=outcome_reference(a,b,-b,float(np.sum(a*a*variance[i1])),
            signal=signal,mean=mean,variance=variance,y0=y0,i0=i0,i1=i1,
            definition=definition,se=se)
        genetic=float(a@(mean[i1]-signal[i1])+b@(mean[i0]-signal[i0]))
        training_noise=float(b@(y0-mean[i0]))
        np.testing.assert_allclose(genetic+training_noise,diagnostic['conditional_mean_bias'],
            rtol=2e-12,atol=2e-12)
        target=diagnostic['conditional_interaction_truth']
        records.append(dict(setting=setting,replicate=int(columns[j][3:]),failed=False,
            biological_null=definition['biological_null'],beta=estimate,se=se,p=float(public['kernel_p']),
            **diagnostic,error=estimate-target,coverage=bool(abs(estimate-target)<=norm.isf(.025)*se),
            noninteraction_mean_contribution=genetic,realized_training_noise_contribution=training_noise,
            feature_response=response,**fit['diagnostics'][j]))
    return records


def diagnose(a,spec,values,public_fits):
    """Known generating quantities enter only this post-fit diagnostic."""
    simulation=json.loads(a.simulation.read_text())
    definition=simulation['definitions'][a.setting]
    fixed_architecture=definition.get('sampling_law')=='fixed_architecture'
    with np.load(a.simulation.parent/a.setting/'diagnostic_truth.npz',allow_pickle=False) as archive:
        truth={k:archive[k] for k in archive.files}
    record=json.loads((a.out/'prepared/preparation.json').read_text())
    with np.load(record['reference'],allow_pickle=False) as archive:
        manifest=json.loads(str(archive['manifest']))['metadata']
        ref={k:archive[k] for k in ('rows','variants','mean','inverse_scale','contexts','noise','target')}
    if not np.array_equal(ref['rows'],truth['rows']):
        raise ValueError('diagnostic generating rows differ from the public cohort')
    for name in ('i0','i1','y0','y1','f0','f1','c0','c1','extras','theta'):
        if name not in values:raise ValueError('public calculation was not captured')
    i0,i1=values['i0'],values['i1'];fit=values['fit'];b=len(public_fits)
    scales=dict(manifest['scales'],mean=ref['mean'],inverse_scale=ref['inverse_scale'])
    training_path=(a.manifest.parent/spec['training']).resolve()
    training_spec=json.loads(training_path.read_text())
    signal=truth['signal'];variance=truth['variance']
    scheduled=replicate_columns(simulation_replicates(simulation))
    columns=spec['phenotypes']['columns']
    mean_shape=(len(ref['rows']),) if fixed_architecture else (len(ref['rows']),len(scheduled))
    if (truth['mean'].shape!=mean_shape or set(columns)-set(scheduled)
            or signal.shape!=(len(ref['rows']),)):
        raise ValueError('diagnostic truth and scheduled learner axes differ')
    theta=np.zeros(len(manifest['kernel_names']))
    if not fixed_architecture:
        genetic=definition['genetic_variances']
        for name,value in [('additive',genetic[0]),('dominance',genetic[1]),('additive_by_PC1',genetic[2]),
                           ('noise_iid',definition['noise_variances'][0]),('noise_target_squared',definition['noise_variances'][1])]:
            theta[manifest['kernel_names'].index(name)]=value
    np.testing.assert_allclose(ref['noise']@np.asarray(definition['noise_variances']),variance,rtol=2e-13,atol=2e-13)
    if fixed_architecture:
        primary=fixed_primary_records(values,public_fits,truth,definition,columns,scheduled,a.setting)
        early=dict(identity=diagnostic_identity(a,spec),preparation_identity=record['identity'],
            setting=a.setting,scheduled=len(columns),records=primary,summary=summarize(primary,len(columns)),
            scope='Completed public primary fits and fixed-architecture conditional diagnostics; '
                'comparators remain separate. These are the same learners as the final results.')
        early_path=a.out/'primary_diagnostics.json'
        if early_path.exists():
            if json.loads(early_path.read_text())['identity']!=early['identity']:
                raise ValueError('saved primary diagnostic inputs changed')
        else:
            write_json(early_path,early)
    # Complementary comparisons are fixed before inspecting results. The burden
    # has uniform weights on the reference genotype scale. The mixed oracle is
    # diagnostic and never replaces the learned primary direction.
    teacher_path=Path(simulation['signal_reference'])
    teacher=json.loads((teacher_path/'reference.json').read_text())
    with np.load(teacher_path/'reference.npz',allow_pickle=False) as archive:
        take=np.searchsorted(archive['rows'],ref['rows'])
        if not np.array_equal(archive['rows'][take],ref['rows']):
            raise ValueError('direction comparator reference does not cover fitted rows')
        xref=archive['target'][take]
        if np.any(xref==0):
            raise ValueError('stored reference interactions do not identify comparator scores at zero target')
        scores=np.column_stack([archive['signals'][take,teacher['settings'].index(name)]/xref
            for name in ('aligned',definition.get('direction','mixed'))])
    comparator_features=np.tile(ref['target'][:,None]*scores,(1,b))
    z=ref['contexts'][i1,1:]
    comparator_extras=[]
    for j in range(b):
        pgs=values['extras'][j][:,1]
        for k in range(2):
            score=scores[i1,k]
            comparator_extras.append(np.column_stack([score,pgs,z*score[:,None],z*pgs[:,None]]))
    # A separate known-covariance solve supplies conditional truth and variance.
    # It is never reused as an estimated coefficient, SE, P value or selector.
    with source_from_spec(training_spec['genotypes'],training_path.parent) as s0, \
         source_from_spec(training_spec['genotypes'],training_path.parent) as s1:
        training=PolygenicKernels(s0,ref['rows'][i0],ref['variants'],scales,ref['contexts'][i0],ref['noise'][i0],
            threads=a.num_threads,block_size=a.block_size,memory_bytes=int(a.memory_gib*2**30),storage='packed')
        complete=PolygenicKernels(s1,ref['rows'],ref['variants'],scales,ref['contexts'],ref['noise'],
            threads=a.num_threads,block_size=min(a.block_size,1024),memory_bytes=int(a.memory_gib*2**30))
        reserve=sum(v.nbytes for v in ref.values())+sum(v.nbytes for v in values.values() if isinstance(v,np.ndarray))
        reserve+=sum(v.nbytes for v in fit.values() if isinstance(v,np.ndarray))+256*2**20
        reserve+=comparator_features.nbytes+sum(v.nbytes for v in comparator_extras)
        reserve+=3*values['tangents'].nbytes+8*len(ref['rows'])*b*24
        budget=int(a.memory_gib*2**30)
        training.memory_bytes=budget-complete.base_bytes-reserve
        complete.memory_bytes=budget-training.base_bytes-reserve
        if min(training.memory_bytes-training.base_bytes,complete.memory_bytes-complete.base_bytes)<=0:
            raise MemoryError('known-covariance diagnostic exceeds memory budget')
        from scripts.epistasis.conditional_comparators import compare_directions
        comparison=compare_directions(training,complete,i0,i1,
            f0=comparator_features[i0],f1=comparator_features[i1],
            y1=np.repeat(values['y1'],2,axis=1),prediction=np.repeat(fit['prediction'],2,axis=1),
            fixed0=values['c0'],fixed1=values['c1'],extras=comparator_extras,
            tangents=np.repeat(values['tangents'],2,axis=1),theta=np.repeat(values['theta'],2,axis=1),
            checkpoint_dir=a.out/'comparator_solvers',resume=getattr(a,'resume',False))
        contrasts=np.column_stack([fit['contrasts'],comparison['contrasts']])
        if fixed_architecture:
            inverse=-np.column_stack([fit['training_contrasts'],comparison['training_contrasts']])
            true_variance=np.sum(contrasts*contrasts*variance[i1,None],axis=0)
            known_reports=None
        else:
            repeated=np.repeat(theta[:,None],3*b,axis=1)
            rhs,quadratic=complete.cross_products(contrasts,i1,i0,repeated)
            known_checkpoint=a.out/'known_covariance.npz'
            inverse,solved=projected_solve(training,rhs,values['c0'],repeated,checkpoint=known_checkpoint,
                resume=getattr(a,'resume',False) and known_checkpoint.exists())
            true_variance=quadratic-np.sum(rhs*inverse,axis=0)
            known_reports={'/'.join(k):v for k,v in solved.reports.items()}
        if np.any(true_variance<=0):raise ValueError('nonpositive diagnostic conditional variance')
        resource_path=a.out/'diagnostic_resources.json'
        if resource_path.exists():
            resource_path=a.out/f'diagnostic_resources.{len(list(a.out.glob("diagnostic_resources*.json")))}.json'
        write_json(resource_path,dict(
            training_ledger=asdict(training.stream.ledger),confirmation_ledger=asdict(complete.stream.ledger),
            solver=known_reports,
            scope='post-fit comparators and known-law diagnostics; excluded from production resource timing'))
    records=[]
    mean_basis=polygenic.thin_rank_revealing_fixed_effect_basis(values['c1'],rtol=1e-11)
    for j,public in enumerate(public_fits):
        a1,b0=fit['contrasts'][:,j],fit['training_contrasts'][:,j]
        estimate=float(public['beta'][0]);se=float(np.sqrt(public['coefficient_covariance'][0][0]))
        np.testing.assert_allclose(estimate,fit['beta'][j],rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(se*se,fit['variance'][j],rtol=2e-12,atol=2e-12)
        generating_mean=truth['mean'] if fixed_architecture else truth['mean'][:,scheduled.index(columns[j])]
        diagnostic=outcome_reference(a1,b0,inverse[:,j],true_variance[j],signal=signal,mean=generating_mean,
            variance=variance,y0=values['y0'][:,j],i0=i0,i1=i1,definition=definition,se=se)
        target=diagnostic['conditional_interaction_truth']
        fitted_response=float(a1@signal[i1]+b0@signal[i0])
        # Simple finite-mean comparison uses the public cohort feature/mean
        # arrays, with no conditional covariance or generating parameters.
        added=manifest['mean']['fixed_definition'].get('added_columns',0)
        local_columns=values['c1'].shape[1]-added
        baselines={}
        for label,nuisance in (
                ('local',np.column_stack([values['c1'][:,:local_columns],values['extras'][j][:,:2]])),
                ('finite_varying_mean',np.column_stack([values['c1'],values['extras'][j]]))):
            summary=prepare_robust_scores(values['f1'][:,j,None],values['y1'][:,j],nuisance,
                feature_names=['learned'],trait_names=['y'],metadata={})
            test=robust_score_tests(summary)
            baselines[label]=dict(beta=float(test['beta'][0]),
                se=float(np.sqrt(test['coefficient_covariance'][0][0])),p=float(test['kernel_p']),
                **{k:summary.metadata[k] for k in ('fixed_rank','max_leverage','outside_confirmation_design')})
        alignment=None if not np.var(signal[i1]) else float(np.corrcoef(values['f1'][:,j],signal[i1])[0,1]**2)
        comparisons={}
        for k,label in enumerate(('burden','oracle')):
            jj=2*j+k;aa=comparison['contrasts'][:,jj];bb=comparison['training_contrasts'][:,jj]
            value=float(comparison['beta'][jj]);sd=float(np.sqrt(comparison['variance'][jj]))
            other=outcome_reference(aa,bb,inverse[:,b+jj],true_variance[b+jj],signal=signal,mean=generating_mean,
                variance=variance,y0=values['y0'][:,j],i0=i0,i1=i1,definition=definition,se=sd)
            scientific=other['conditional_interaction_truth']
            comparisons[label]=dict(beta=value,se=sd,p=float(2*norm.sf(abs(value/sd))),
                **other,coverage=bool(abs(value-scientific)<=norm.isf(.025)*sd),
                response_information=float(comparison['response'][:,jj]@comparison['response'][:,jj]),
                response_energy_fraction=float(np.sum(comparison['response'][:,jj]**2)/np.sum(comparator_features[i1,jj]**2)),
                **comparison['diagnostics'][jj])
        records.append(dict(setting=a.setting,replicate=int(columns[j][3:]),failed=False,biological_null=definition['biological_null'],
            beta=estimate,se=se,p=float(public['kernel_p']),**diagnostic,
            error=estimate-target,coverage=bool(abs(estimate-target)<=norm.isf(.025)*se),
            fitted_operator_interaction_response=fitted_response,
            feature_response=float(a1@values['f1'][:,j]+b0@values['f0'][:,j]),
            response_information=float(fit['response'][:,j]@fit['response'][:,j]),
            response_energy_fraction=float(np.sum(fit['response'][:,j]**2)/np.sum(values['f1'][:,j]**2)),
            nuisance_prediction=nuisance_prediction_quality(generating_mean[i1]-signal[i1],
                dict(additive_pgs=values['extras'][j][:,1],conditional_prediction=fit['prediction'][:,j]),mean_basis),
            comparisons=comparisons,
            direction_alignment=alignment,baselines=baselines,
            baseline_p=baselines['finite_varying_mean']['p'],
            baseline_definition='legacy baseline_p is finite varying mean; local and finite baselines use the same primary learner and confirmation HC3',
            realized_training_signal_variance=float(np.var(signal[i0])),
            realized_confirmation_signal_variance=float(np.var(signal[i1])),**fit['diagnostics'][j]))
    return records


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--simulation',type=Path,required=True)
    p.add_argument('--setting',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-threads',type=int,default=1)
    p.add_argument('--block-size',type=int,default=4096)
    p.add_argument('--memory-gib',type=float,default=32)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--phase',choices=['development','confirmation'],default='development',
        help='confirmation requires a separately frozen procedure and fresh prespecified simulation seeds')
    run(p.parse_args())


if __name__=='__main__':main()
