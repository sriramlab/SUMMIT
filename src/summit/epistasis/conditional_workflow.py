"""Public preparation of the conditional polygenic research procedure.

All genotype-dependent preparation stays cohort-side. Saved score fitting is
genotype-free. A new phenotype must provide its own matched training direction
and outcomes; only the genotype reference is reused across traits.
"""
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
from zipfile import ZipFile

import numpy as np

from summit.context.spec import array_sha256, canonical_sha256
from summit.prediction._validation import array_digest, closed
from summit.prediction.artifacts import load_prediction_models, file_digest
from summit.prediction.cli import _rows, _variants, _aligned_table
from summit.prediction.genotype import source_from_spec
from summit.prediction.score import ScoreInput, score_prediction
from .conditional import conditional_mean_summary
from .conditional_reference import prepare_conditional_reference
from .directions import sample_tokens
from .polygenic import (PolygenicKernels, estimate_components, conditional_scores_batch,
    conditional_variance_precision)
from .robust import write_robust_scores, load_robust_scores
from .summary import _publish_bundle


def _write_json(path, value):
    from .cli import _jsonable
    with Path(path).open('x') as handle:
        json.dump(_jsonable(value), handle, indent=2, allow_nan=False)
        handle.write('\n')
    Path(path).chmod(0o600)


def _training_covariance(path, identity, training, y0, geometry):
    """Recover this preparation's fitted null, never another trait's fit.

    The enclosing identity binds both outcomes, all feature definitions and
    the implementation. Publish before solver checkpoints so an interruption
    does not repeat the full-marker phenotype moment product.
    """
    shape = (training.count, y0.shape[1])
    fields = ('coefficients','influence','trace')
    if path.exists():
        with ZipFile(path) as archive:
            if sum(v.file_size for v in archive.infolist()) > 16*np.prod(shape)*(1+len(training.rows)+shape[0])+65536:
                raise ValueError('training covariance checkpoint exceeds its declared axes')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'manifest', *fields}:
                raise ValueError('unsupported training covariance checkpoint; use a new preparation path')
            record = json.loads(str(archive['manifest']))
            values = {name:archive[name] for name in fields}
        if (record.get('kind') != 'summit.epistasis.training_covariance'
                or record.get('schema_version') != 3 or record.get('identity') != identity
                or record.get('digests') != {k:array_sha256(v) for k,v in values.items()}):
            raise ValueError('training covariance checkpoint inputs or values changed')
        theta = values.pop('coefficients')
        uncertainty = values
    else:
        theta, uncertainty = estimate_components(training, y0, geometry, return_uncertainty='directional')
    if theta.shape != shape or not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError('invalid training covariance coefficients')
    influence,trace=uncertainty['influence'],uncertainty['trace']
    if (influence.shape != (shape[0],len(training.rows),shape[1])
            or not np.all(np.isfinite(influence))
            or trace.shape != (shape[1],shape[0],shape[0]) or not np.all(np.isfinite(trace))
            or not np.allclose(trace,trace.transpose(0,2,1),rtol=1e-10,atol=1e-12)):
        raise ValueError('invalid training covariance uncertainty')
    if not path.exists():
        values = dict(coefficients=theta,**uncertainty)
        _publish_bundle(path, dict(kind='summit.epistasis.training_covariance', schema_version=3,
            identity=identity, digests={k:array_sha256(v) for k,v in values.items()}), values)
    return theta, uncertainty


def _direction_record(path):
    record=json.loads(Path(path).read_text())
    if (record.get('kind')!='summit.epistasis.frozen_direction'
            or record.get('schema_version')!=1
            or not record.get('model_identity') or not record.get('additive_model_identity')):
        raise ValueError('conditional preparation requires matched interaction and additive-null models; '
            'rerun train-direction with this version')
    return record


def _score_directions(definitions, root, source, rows, training_rows, training_spec,
                      y0, mean, *, threads, block_size, memory_bytes):
    """Authenticate matched learners before scoring both disjoint cohorts."""
    loaded, selected, directions = {}, {}, []
    columns = [d['phenotype'] for d in definitions]
    for j,d in enumerate(definitions):
        closed(d, ('phenotype','direction'), name='conditional direction')
        path = (root/d['direction']).resolve()
        record = _direction_record(path)
        if (record.get('kind') != 'summit.epistasis.frozen_direction'
                or record.get('schema_version') != 1
                or record['training_source'] != source.identity
                or record['training_manifest'] != training_spec
                or record['training_samples'] != sample_tokens([source.samples[i] for i in training_rows])):
            raise ValueError('conditional direction training definition or sample separation changed')
        model_path = path.parent/record['models']
        if model_path not in loaded:
            loaded[model_path] = {m.identity:m for m in load_prediction_models(model_path)}
        models = loaded[model_path]
        interaction = models[record['model_identity']]
        additive = models[record['additive_model_identity']]
        for m in (interaction, additive):
            declared = m.provenance['training'][m.trait_id]
            if (m.scale.sample_identity != mean['metadata']['training_samples']
                    or declared['phenotype'] != array_digest(y0[:,j])
                    or declared['fixed'] != mean['metadata']['training_fixed_hash']
                    or declared['phi'] != mean['metadata']['training_context_hash']
                    or m.weights.shape[1] != 2):
                raise ValueError('conditional training phenotype, nuisance or target changed after learning')
            if m.key in selected and selected[m.key].identity != m.identity:
                raise ValueError('conditional batch model keys conflict; use a shared-mask training batch')
            selected[m.key] = m
        expected = record.get('phenotype_column', training_spec.get('phenotype',{}).get('column'))
        if expected != columns[j]:
            raise ValueError('conditional direction phenotype does not match requested trait')
        directions.append((record,interaction,additive))
    inputs = {m.trait_id:ScoreInput(rows,np.ones((len(rows),2)),np.empty((len(rows),0)),
        m.context_spec,m.fixed_spec) for m in selected.values()}
    scored = score_prediction(selected.values(),source,inputs,components_only=True,
        threads=threads,block_size=block_size,memory_bytes=memory_bytes,adaptive_blocks=True)
    e = np.column_stack([scored.components[m.key][:,1] for _,m,_ in directions])
    pgs = np.column_stack([scored.components[m.key][:,0] for _,_,m in directions])
    identities = [canonical_sha256(dict(interaction=m.identity,additive=a.identity,
        training_target=r['training_target'])) for r,m,a in directions]
    return e, pgs, identities, scored.report


def prepare_conditional(path, output, *, threads=1, block_size=512,
                        memory_bytes=4*2**30, resume=False, exact=False):
    """One training-matched scalar direction per trait; no confirmation tuning."""
    path = Path(path).resolve(); root = path.parent
    spec = json.loads(path.read_text())
    closed(spec, ('kind','schema_version','training','samples','phenotypes','directions','inference'),
        name='conditional preparation')
    if spec['kind'] not in ('summit.epistasis.prepare','summit.epistasis.prepare_traits') or spec['schema_version'] != 1:
        raise ValueError('unsupported conditional preparation manifest')
    settings = spec['inference']
    closed(settings, ('method',), ('reference','probes','seed','moment_weighting'), name='conditional inference')
    if settings['method'] != 'conditional_polygenic_mean':
        raise ValueError('unsupported conditional estimator')
    p = spec['phenotypes']
    closed(p, ('file','columns','unit'), name='conditional phenotypes')
    columns = p['columns']; definitions = spec['directions']
    if (not isinstance(p['unit'],str) or not p['unit'] or not isinstance(columns,list) or not columns
            or len(set(columns)) != len(columns)
            or [d['phenotype'] for d in definitions] != columns
            or any(not isinstance(c,str) or not c or Path(c).name != c or c in ('.','..') for c in columns)):
        raise ValueError('distinct file-safe trait names, aligned directions and scientific units required')
    training_path = (root/spec['training']).resolve()
    training_spec = json.loads(training_path.read_text()); training_root = training_path.parent
    if (training_spec.get('kind') != 'summit.epistasis.train_direction'
            or training_spec.get('schema_version') != 1 or not training_spec.get('trans_only')
            or not training_spec.get('covariates')):
        raise ValueError('conditional preparation requires the matched supplied-trans training recipe')
    tp = training_spec.get('phenotypes',training_spec.get('phenotype'))
    if not tp or tp['unit'] != p['unit']:
        raise ValueError('training and confirmation phenotype units differ')
    # Reject older joint-only learners before constructing a full-marker
    # reference. Their additive component is not the declared additive-null fit.
    for definition in definitions:
        closed(definition, ('phenotype','direction'), name='conditional direction')
        _direction_record(root/definition['direction'])
    output = Path(output)
    output.mkdir(parents=True,exist_ok=resume); output.chmod(0o700)
    start,cpu = time.perf_counter(),time.process_time()
    with ExitStack() as stack:
        source = stack.enter_context(source_from_spec(training_spec['genotypes'],training_root))
        source1 = stack.enter_context(source_from_spec(training_spec['genotypes'],training_root))
        training_rows = np.sort(_rows(source,training_root/training_spec['samples']))
        confirmation_rows = np.sort(_rows(source,root/spec['samples']))
        variants = _variants(source,training_root/training_spec['variants'])
        reference_path = root/settings['reference'] if settings.get('reference') else output/'cohort-reference.npz'
        reference = prepare_conditional_reference(source,training_rows,confirmation_rows,variants,
            training_spec,training_root,path=reference_path,threads=threads,block_size=block_size,
            memory_bytes=memory_bytes,probes=settings.get('probes',128),seed=settings.get('seed',871631),exact=exact,
            moment_weighting=settings.get('moment_weighting','none'))
        mean = reference['mean']; rows=mean['rows']; i0=mean['training_index']; i1=mean['confirmation_index']
        y0 = _aligned_table(training_root/tp['file'],[source.samples[i] for i in training_rows])[columns].to_numpy(float)
        y1 = _aligned_table(root/p['file'],[source.samples[i] for i in confirmation_rows])[columns].to_numpy(float)
        if not np.all(np.isfinite(y0)) or not np.all(np.isfinite(y1)):
            raise ValueError('conditional preparation needs the complete frozen phenotype mask')
        # Scoring finishes before the confirmation operator is constructed.
        # Reuse its authenticated source instead of hashing and opening the
        # same BED/BIM/FAM trio for a third time. The packed training operator
        # keeps its separate source and every source retains live-file checks.
        training = reference['operator']
        reference_bytes = sum(v.nbytes for v in mean.values() if isinstance(v,np.ndarray))
        reference_bytes += sum(v.nbytes for v in reference['geometry'].values() if isinstance(v,np.ndarray))
        scoring_memory = memory_bytes-training.base_bytes-reference_bytes-y0.nbytes-y1.nbytes-256*2**20
        if scoring_memory <= 0:
            raise MemoryError('retained conditional reference leaves no scoring workspace')
        e,pgs,identities,score_report = _score_directions(definitions,root,source1,rows,training_rows,
            training_spec,y0,mean,threads=threads,block_size=block_size,memory_bytes=scoring_memory)
        score_report['memory_budget_bytes'] = scoring_memory
        score_report['retained_reference_bytes'] = memory_bytes-scoring_memory
        # Keep the public learner's raw score units. This normalization is
        # explicit and identical in training and confirmation.
        f = mean['target'][:,None]*e
        z = mean['contexts'][i1,1:]
        extras = [np.column_stack([e[i1,j],pgs[i1,j],z*e[i1,j,None],z*pgs[i1,j,None]])
            for j in range(len(columns))]
        complete = PolygenicKernels(source1,rows,variants,reference['scales'],mean['contexts'],mean['noise'],
            threads=threads,block_size=min(block_size,1024),memory_bytes=memory_bytes)
        arrays = [v for v in mean.values() if isinstance(v,np.ndarray)]
        arrays += [v for v in reference['geometry'].values() if isinstance(v,np.ndarray)]
        b,k=len(columns),training.count
        reserve = sum(v.nbytes for v in arrays)+256*2**20+3*16*len(i0)*64*b+8*b*(8*len(rows)
            +(2+2*z.shape[1])*len(i1)+k*(24+2*mean['fixed'].shape[1])*len(i0)
            +k*(3*len(i1)+5*len(i0)))
        if training.base_bytes+complete.base_bytes+reserve >= memory_bytes:
            raise MemoryError('joint conditional workspaces exceed memory budget; reduce trait batch size')
        training.memory_bytes=memory_bytes-complete.base_bytes-reserve
        complete.memory_bytes=memory_bytes-training.base_bytes-reserve
        identity = canonical_sha256(dict(reference=reference['identity'],directions=identities,
            training=array_sha256(y0),confirmation=array_sha256(y1),spec=spec,
            scoring_block_size=score_report['block_size'],
            algorithm={name:file_digest(Path(__file__).with_name(name)) for name in
                ('conditional_workflow.py','conditional_reference.py','conditional.py','polygenic.py','krylov.py')},
            scoring_implementation=file_digest(Path(__file__).parents[1]/'prediction/score.py')))
        input_record = output/'inputs.json'
        if input_record.exists():
            if not resume or json.loads(input_record.read_text())['identity'] != identity:
                raise ValueError('conditional preparation inputs changed; use a new output path')
        else:
            _write_json(input_record,dict(identity=identity,manifest=spec))
        finished = output/'preparation.json'
        if finished.exists():
            previous=json.loads(finished.read_text())
            if previous['identity'] != identity:
                raise ValueError('completed conditional preparation changed')
            for record in previous['summaries']:
                summary=load_robust_scores(output/record['file'])
                if summary.metadata['preparation_identity'] != record['identity']:
                    raise ValueError('completed conditional summary changed')
            return previous
        covariance_path = output/'training-covariance.npz'
        covariance_reused = covariance_path.exists()
        theta, uncertainty = _training_covariance(covariance_path,identity,
            training,y0,reference['geometry'])
        (output/'solver').mkdir(exist_ok=resume)
        fitted = conditional_scores_batch(training,complete,i0,i1,y0,y1,f[i0],f[i1],
            mean['fixed'][i0],mean['fixed'][i1],extras,theta,checkpoint_dir=output/'solver',mean_tangents=True,recycle=True)
        precision = conditional_variance_precision(complete,i0,i1,fitted,theta,uncertainty,training=training)
        records=[]
        for j,column in enumerate(columns):
            d=fitted['diagnostics'][j]
            h=np.array([[fitted['response'][:,j]@fitted['response'][:,j]]])
            summary=conditional_mean_summary(fitted['beta'][j:j+1],fitted['variance'][j:j+1,None],h,
                feature_names=[training_spec['target']+'_by_frozen_trans_score'],trait_name=column,trait_unit=p['unit'],
                identities=dict(genotype_reference=reference['identity'],direction=identities[j],
                    training_outcomes=array_sha256(y0[:,j]),confirmation_outcomes=array_sha256(y1[:,j]),
                    null_fit=canonical_sha256(dict(theta=array_sha256(theta[:,j]),procedure=identity))),
                diagnostics=dict(nuisance_training_n=len(i0),confirmation_n=len(i1),fixed_rank=d['rank']-1,
                    feature_rank=1,max_leverage=d['max_leverage'],
                    minimum_feature_effective_support=d['feature_effective_support'],
                    outside_confirmation_design=d['outside_confirmation_design'],information_condition=1.,
                    covariance_precision=precision[j]))
            filename=column+'.robust-score.npz'; target=output/filename
            if target.exists():
                if not resume or load_robust_scores(target).metadata['preparation_identity'] != summary.metadata['preparation_identity']:
                    raise ValueError('partial conditional summary changed')
            else:
                write_robust_scores(summary,target)
            records.append(dict(file=filename,identity=summary.metadata['preparation_identity']))
        record=dict(identity=identity,method='conditional_polygenic_mean_tangent_v1',
            status='research; full-marker statistical qualification incomplete',summaries=records,
            reference=str(reference_path.resolve()),reference_reused=reference['reused'],
            training_covariance_reused=covariance_reused,
            covariance_components=theta,kernel_names=reference['metadata']['kernel_names'],
            covariance_precision=precision,covariance_trace_uncertainty=uncertainty['trace'],
            covariance_precision_implementation='directional_exact_contraction_v1',
            covariance_moment_weighting=settings.get('moment_weighting','none'),
            solver_reports=[{'/'.join(k):v for k,v in report.items()} for report in fitted['solver_reports']],
            recycled_ranks=fitted['recycled_ranks'],
            genotype_traversals=dict(training=asdict(training.stream.ledger),confirmation=asdict(complete.stream.ledger)),
            scoring=score_report,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        _write_json(finished,record)
        return record
