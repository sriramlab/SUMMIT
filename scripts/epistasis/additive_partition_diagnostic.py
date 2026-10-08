"""Development diagnosis of background-partitioned additive prediction.

Partition weights using the externally prespecified interaction background,
without refitting or selecting on confirmation outcomes. Known means only
diagnose leakage; this driver does not qualify a production correction.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2,ncx2

from summit.epistasis.directions import sample_tokens
from summit.prediction._validation import digest
from summit.prediction.artifacts import load_prediction_models
from summit.prediction.cli import _rows,_aligned_table
from summit.prediction.genotype import FileGenotypeSource,RawBlockStream,StandardizedBlock,native_module
from summit.prediction.operator import GenotypeOperator
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.score import align_variants
from scripts.epistasis.full_matched import write_json,load_reference,load_experiment_inputs
from scripts.epistasis.full_matched_population import PopulationRegression


def score_partition(source,rows,models,variants,*,threads=1,memory_bytes=4*2**30):
    if np.any(np.diff(rows)<=0):raise ValueError('ordered unique scoring rows required')
    if 64*len(rows)*min(512,len(variants))+256*2**20>memory_bytes:
        raise MemoryError('partition score workspace exceeds memory budget')
    first=models[0]
    if any(m.scale.identity!=first.scale.identity or m.variants.identity!=first.variants.identity for m in models):
        raise ValueError('partition scoring requires the same frozen genotype scale and axis')
    mr,sr,flips=align_variants(first.variants,source.variants)
    selected=np.array([first.variants.ids[j] in variants for j in mr])
    mr,sr,flips=mr[selected],sr[selected],flips[selected]
    if not len(mr) or len(mr)!=len(variants):
        raise ValueError('background must be a nonempty subset of the frozen variant axis')
    order=np.argsort(sr);mr,sr,flips=mr[order],sr[order],flips[order]
    native=native_module();configure_prediction_threads(native,threads)
    stream=RawBlockStream(source,rows,sr,block_size=512,threads=threads)
    affine=StandardizedBlock(native,threads)
    operator=SimpleNamespace(native=native,plan=SimpleNamespace(threads=threads))
    out=np.zeros((len(rows),len(models)))
    for begin,selected,raw in stream.blocks('additive_partition'):
        take=mr[begin:begin+len(selected)]
        g=affine.prepare(raw,np.arange(len(rows)),np.arange(len(selected)),
            first.scale.mean[take],first.scale.inverse_scale[take],flips[begin:begin+len(selected)])
        weights=np.column_stack([m.weights[take,0] for m in models])
        out+=GenotypeOperator.product(operator,g,weights)
    return out,asdict(stream.ledger)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();records=[]
    design=json.loads((a.input/'design.json').read_text());args=design['arguments']
    meta,truth,_=load_experiment_inputs(Path(design['reference']))
    work=a.input/f'{a.setting}_000'
    spec=json.loads((work/'prepare.json').read_text())
    if not 1<=a.models<=min(args['replicates'],args['batch_size']):
        raise ValueError('select the first prespecified models in one complete batch')
    definitions={v['name']:v for v in spec['frozen_scores']}
    jobs=[j for j in spec['jobs'] if j['id'].endswith('_learned')][:a.models]
    models=[];frozen=[];cache={}
    for job in jobs:
        definition=definitions[job['adjust_scores'][0]]
        path=(work/definition['direction']).resolve();d=json.loads(path.read_text())
        root=path.parent/d['models']
        if root not in cache:cache[root]=load_prediction_models(root)
        identity=d.get('additive_model_identity',d['model_identity'])
        models.append(next(m for m in cache[root] if m.identity==identity));frozen.append(d)
    write_json(a.out/'design.json',dict(parent=str(a.input.resolve()),setting=a.setting,
        scheduled_models=a.models,selection='first models in scheduled order; no phenotype selection',
        change='add background-only additive prediction and its declared structure-covariate slopes to the existing finite confirmation mean',
        diagnostic_only=True,known_mean_role='post-fit leakage and information diagnostics only'))
    with FileGenotypeSource(work/spec['genotypes']['geno'],genome_build='GRCh37') as source:
        rows=_rows(source,work/spec['samples']);samples=[source.samples[i] for i in rows]
        if any(set(sample_tokens(samples))&set(d['training_samples']) for d in frozen):
            raise ValueError('partition scoring overlaps direction training')
        if source.identity!=meta['source_identity']:
            raise ValueError('reference genotype source changed')
        values,ledger=score_partition(source,rows,models,set(meta['background_variants']),threads=a.num_threads,
            memory_bytes=int(a.memory_gib*2**30))
        cv=spec['covariates'];z=_aligned_table(work/cv['file'],samples)[cv['varying_effects']].to_numpy(float)
        take=np.searchsorted(truth['rows'],rows)
        if not np.array_equal(truth['rows'][take],rows):raise ValueError('donor rows differ from reference')
        idx=meta['settings'].index(a.setting)
        mean=truth['means'][take,idx]+(args['signal_multiplier']-1)*truth['signals'][take,idx]
        variance=truth['variance'][take]
        n=design.get('intended_confirmation_n',args['confirmation_samples'])
        for j,job in enumerate(jobs):
            ref=load_reference(work/f"prepared/{job['id']}.cohort-reference.npz")
            if ref.metadata['sample_hash']!=digest(samples):raise ValueError('cohort reference sample order differs')
            for label,c in [('pooled',ref.fixed_effects),('partitioned',np.column_stack([
                    ref.fixed_effects,values[:,j],z*values[:,j,None]]))]:
                regression=PopulationRegression(c,ref.features)
                target,covariance=regression.truth(mean,variance,n)
                nc=float(target@np.linalg.solve(covariance,target))
                records.append(dict(model=j,adjustment=label,projection=float(target[0]),
                    known_population_se=float(np.sqrt(covariance[0,0])),noncentrality=nc,
                    diagnostic_asymptotic_null_rejection={str(alpha):float(ncx2.sf(chi2.isf(alpha,1),1,nc)) for alpha in (.05,.005)},
                    interaction_information=float(1/regression.transform[0,0]**2),
                    finite_rank=regression.design.shape[1]))
            print(job['id'],records[-2:],flush=True)
    write_json(a.out/'results.json',dict(records=records,seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,genotype_ledger=ledger))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--setting',required=True);p.add_argument('--models',type=int,default=3)
    p.add_argument('--num-threads',type=int,default=1);p.add_argument('--memory-gib',type=float,default=16)
    run(p.parse_args())


if __name__=='__main__':main()
