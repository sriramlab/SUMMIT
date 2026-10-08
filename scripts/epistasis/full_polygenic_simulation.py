"""Stream matched full-marker outcomes for conditional covariance validation.

Genetic coefficients are redrawn per replicate, shared between its training
and confirmation individuals, and never normalized after drawing. Known means
are saved separately for diagnostics. Fitting manifests contain only observed
phenotypes, covariates and the externally supplied target/background.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from summit.prediction.genotype import FileGenotypeSource, RawBlockStream, StandardizedBlock, native_module
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction._validation import array_digest
from scripts.epistasis.full_matched import write_json
from scripts.epistasis.benchmark_robust_workflow import io
from scripts.epistasis.replicate_schedule import replicate_ids, replicate_columns


def genetic_draws(source, rows, variants, scales, context, *, seed, replicates,
                  replicate_start=0, block_size=512, threads=1, memory_bytes=8*2**30):
    """Independent A, H and covariate-dependent A draws with exact kernel law.

    Coefficients have variance 1/M on the saved training scale. Random streams
    are indexed by component and replicate, preserving batch/block equivalence.
    Returns N by replicate by component, never a complete genotype matrix.
    """
    rows, variants = np.asarray(rows), np.asarray(variants)
    ids = replicate_ids(replicates, replicate_start)
    n, m = len(rows), len(variants)
    context = np.asarray(context, float)
    if (not 1 <= replicates <= 100 or not m or context.shape != (n,)
            or not np.all(np.isfinite(context)) or np.any(np.diff(rows) <= 0)
            or np.any(np.diff(variants) <= 0)):
        raise ValueError('ordered nonempty axes, finite context and 1..100 replicates required')
    if scales['source'] != source.identity or scales['variants'] != source.variants.subset(variants).identity:
        raise ValueError('simulation source/variant scale identity changed')
    mean, inverse = (np.asarray(scales[k]) for k in ('mean', 'inverse_scale'))
    if (mean.shape != (2,m) or inverse.shape != (2,m)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(inverse))
            or np.any(inverse <= 0)):
        raise ValueError('invalid saved additive/dominance scales')
    # Raw, heterozygote calls, standardization and native multiplication copies.
    if 80*n*min(block_size,m)+8*n*replicates*8+mean.nbytes+inverse.nbytes+256*2**20 > memory_bytes:
        raise MemoryError('streamed genetic draws exceed memory budget')
    native=native_module(); configure_prediction_threads(native,threads)
    stream=RawBlockStream(source,rows,variants,block_size=block_size,threads=threads,native=native)
    standard=StandardizedBlock(native,threads)
    rng=[[np.random.default_rng(np.random.SeedSequence([seed,k,r])) for r in ids] for k in range(3)]
    output=np.zeros((n,replicates,3))
    expected=np.zeros(3)
    for begin,selected,raw in stream.blocks('simulation_genetic_means'):
        width=len(selected); end=begin+width
        for kind,components in [(0,[0,2]),(1,[1])]:
            value=raw if kind==0 else np.asfortranarray(np.where(raw==-127,-127,raw==1),dtype=np.int8)
            g=standard.prepare(value,np.arange(n),np.arange(width),mean[kind,begin:end],inverse[kind,begin:end])
            weights=np.asfortranarray(np.column_stack([r.normal(size=width)/np.sqrt(m) for k in components for r in rng[k]]))
            product=np.empty((n,len(components)*replicates),order='F')
            native.prediction_product(np.asfortranarray(g),weights,product,False,threads)
            for j,k in enumerate(components):
                v=product[:,j*replicates:(j+1)*replicates]
                if k==2:
                    v*=context[:,None]
                    centered_energy=np.sum(g*g*context[:,None]**2)-np.sum((context@g)**2)/n
                else:
                    centered_energy=np.sum(g*g)-np.sum(np.sum(g,axis=0)**2)/n
                output[:,:,k]+=v
                expected[k]+=centered_energy/(n*m)
    return output,expected,asdict(stream.ledger)


def run(a):
    ids = replicate_ids(a.replicates, getattr(a, 'replicate_start', 0))
    a.out.mkdir(parents=True,exist_ok=False); a.out.chmod(0o700)
    start,cpu,before=time.perf_counter(),time.process_time(),io()
    rm=json.loads((a.reference/'reference.json').read_text())
    teacher=json.loads((a.signal_reference/'reference.json').read_text())
    with np.load(a.reference/'reference.npz') as archive:
        data={k:archive[k] for k in ('rows','training_index','confirmation_index','variants','mean','inverse_scale','contexts','target')}
    with np.load(a.signal_reference/'reference.npz') as archive:
        positions=np.searchsorted(archive['rows'],data['rows'])
        if not np.array_equal(archive['rows'][positions],data['rows']):
            raise ValueError('interaction teacher does not cover the actual participant axis')
        signal=archive['signals'][positions,teacher['settings'].index('mixed')]*a.signal_multiplier
    if not np.isfinite(a.signal_multiplier) or a.signal_multiplier<=0:
        raise ValueError('a fixed positive interaction strength is required')
    scales=dict(rm['scales'],mean=data['mean'],inverse_scale=data['inverse_scale'])
    spec=json.loads(Path(rm['training_manifest']).read_text())
    oldroot=Path(rm['training_manifest']).parent
    if 'PC1' not in spec['covariates'].get('varying_effects',[]):
        raise ValueError('this prespecified structure experiment requires PC1')
    z=data['contexts'][:,1+spec['covariates']['varying_effects'].index('PC1')]
    settings=['random_dense','random_structure','random_structure_mixed']
    definitions={s:dict(biological_null=not s.endswith('mixed'),
        genetic_variances=[.8,.5,.5] if 'structure' in s else [.8,0.,0.],
        noise_variances=[.4,.6]) for s in settings}
    write_json(a.out/'simulation.json',dict(reference=str(a.reference.resolve()),
        signal_reference=str(a.signal_reference.resolve()),settings=settings,definitions=definitions,
        seed=a.seed,replicates=a.replicates,replicate_ids=ids,signal_multiplier=a.signal_multiplier,
        signal_reference_variance=teacher['definitions']['mixed']['reference_signal_variance']*a.signal_multiplier**2,
        source_identity=rm['source'],rows_hash=array_digest(data['rows']),
        conditioning='actual intact distinct genotype rows; independent marker effects per replicate and independent errors per individual',
        paired_settings='random_dense and random_structure share the additive draw; structure and structure_mixed share all genetic draws but have independent errors',
        strength='fixed marker-coefficient variance theta/M on the frozen training scale; no realized-phenotype normalization',
        generating_mean='0.5 target + 0.2 PC1 plus independent A, observed-heterozygote H, and PC1-dependent A random effects',
        status='simulation inputs only; no method qualification'))
    with FileGenotypeSource(oldroot/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as source:
        if source.identity!=rm['source'] or source.identity!=teacher['source_identity']:
            raise ValueError('genotype source changed')
        samples=[source.samples[i] for i in data['rows']]
        genetic,expected,ledger=genetic_draws(source,data['rows'],data['variants'],scales,z,
            seed=a.seed,replicates=a.replicates,replicate_start=ids[0],threads=a.num_threads,block_size=a.block_size,
            memory_bytes=int(a.memory_gib*2**30))
    columns=replicate_columns(ids)
    samples=pd.DataFrame(samples,columns=['FID','IID'])
    fixed=.5*data['target']+.2*z
    noise=.4+.6*data['target']**2
    moments=[]
    for si,setting in enumerate(settings):
        root=a.out/setting; root.mkdir()
        mean=fixed[:,None]+genetic@np.sqrt(definitions[setting]['genetic_variances'])
        interaction=signal if setting.endswith('mixed') else np.zeros(len(signal))
        mean+=interaction[:,None]
        outcomes=samples.copy()
        for r,name in enumerate(columns):
            rng=np.random.default_rng(np.random.SeedSequence([a.seed,731,si,ids[r]]))
            outcomes[name]=mean[:,r]+np.sqrt(noise)*rng.normal(size=len(mean))
        outcomes.to_csv(root/'phenotypes.tsv',sep='\t',index=False)
        np.savez(root/'diagnostic_truth.npz',rows=data['rows'],mean=mean,signal=interaction,
                 variance=noise,genetic_coefficients_variance=np.array(definitions[setting]['genetic_variances'])/len(data['variants']))
        train=json.loads(json.dumps(spec))
        train.pop('phenotype',None)
        train['phenotypes']=dict(file='phenotypes.tsv',columns=columns,unit='fixed reference units')
        for name in ('samples','variants','interaction_variants'):
            train[name]=str((oldroot/train[name]).resolve())
        train['genotypes']['geno']=str((oldroot/train['genotypes']['geno']).resolve())
        train['covariates']['file']=str((oldroot/train['covariates']['file']).resolve())
        write_json(root/'train.json',train)
        write_json(root/'prepare.json',dict(kind='summit.epistasis.prepare',schema_version=1,
            training='train.json',samples=str(Path(rm['confirmation_samples']).resolve()),
            phenotypes=train['phenotypes'],
            directions=[dict(phenotype=column,direction=f'trained/direction.{r}.json')
                for r,column in enumerate(columns)],
            inference=dict(method='conditional_polygenic_mean')))
        moments.append(dict(setting=setting,expected_random_genetic_variance=float(expected@definitions[setting]['genetic_variances']),
            realized_genetic_variance=np.var(mean-fixed[:,None]-interaction[:,None],axis=0),
            realized_training_signal_variance=float(np.var(interaction[data['training_index']])),
            realized_confirmation_signal_variance=float(np.var(interaction[data['confirmation_index']]))))
    after=io()
    write_json(a.out/'resources.json',dict(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        io={k:after[k]-v for k,v in before.items()},source_traversals=ledger,
        expected_unit_component_variances=expected,moments=moments))
    print('random full-marker outcomes',round(time.perf_counter()-start),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--signal-reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--replicates',type=int,default=12)
    p.add_argument('--replicate-start',type=int,default=0,
        help='First stable learner ID; split batches share the same seed and reference')
    p.add_argument('--signal-multiplier',type=float,default=np.sqrt(.2))
    p.add_argument('--block-size',type=int,default=1024)
    p.add_argument('--num-threads',type=int,default=1)
    p.add_argument('--memory-gib',type=float,default=16)
    run(p.parse_args())


if __name__=='__main__': main()
