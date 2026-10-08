"""Time actual paired-prior prediction products over bounded real-marker tiles.

Random RHS are shared across block sizes. This measures arithmetic scheduling,
not learning quality, convergence or full-marker throughput.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.prediction.annotations import AnnotationDesign,AnnotationPrior
from summit.prediction.batch import plan_prediction
from summit.prediction.cli import _rows,_variants
from summit.prediction.genotype import FileGenotypeSource,estimate_scale
from summit.prediction.operator import GenotypeOperator
from summit.prediction.spec import CandidatePrior,TraitTraining
from scripts.epistasis.full_matched import write_json


def run(a):
    root=a.training.parent;spec=json.loads(a.training.read_text())
    rng=np.random.default_rng(712873);records=[]
    with FileGenotypeSource(root/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as source:
        rows=np.sort(_rows(source,root/spec['samples']))
        available=_variants(source,root/spec['variants'])
        variants=np.sort(rng.choice(available,a.markers,replace=False))
        background=_variants(source,root/spec['interaction_variants'])
        target=source.variants.ids.index(spec['target'])
        source.prepare(rows,1,a.num_threads);x=source.read(np.array([target]))[:,0].astype(float)
        if np.any(x==-127):raise ValueError('benchmark requires the retained complete target mask')
        x=(x-x.mean())/x.std(ddof=1);phi=np.column_stack([np.ones(len(rows)),x])
        scale=estimate_scale(source,rows,variants,threads=a.num_threads,block_size=128)
        annotation=AnnotationDesign(np.column_stack([np.ones(len(variants)),np.isin(variants,background)]),
            ('additive','trans'),scale.variant_identity)
        prior=AnnotationPrior(annotation,np.array([[[.5,0],[0,0]],[[0,0],[0,.05]]]))
        residual=np.ones(len(rows))
        candidates=(prior.candidate('interaction',residual,dict(method='prespecified')),
            CandidatePrior('additive_null',np.diag([.5,0]),residual,dict(method='prespecified')))
        for count in map(int,a.traits.split(',')):
            traits=[TraitTraining(f't{j}',rows,variants,np.zeros(len(rows)),phi,np.ones((len(rows),1)),scale,
                candidates,dict(names=['intercept','target']),dict(names=['intercept']),dict(units='benchmark')) for j in range(count)]
            vectors={(t.id,c.id):rng.normal(size=len(rows)) for t in traits for c in candidates}
            expected=None
            for block in map(int,a.block_sizes.split(',')):
                plan=plan_prediction(traits,source,storage='packed',block_size=block,
                    rhs_columns=max(8,min(64,4*count)),threads=a.num_threads,memory_bytes=int(a.memory_gib*2**30))
                operator=GenotypeOperator(source,traits,plan);operator.setup()
                begin,cpu=time.perf_counter(),time.process_time()
                result=operator.apply(vectors)
                seconds,cpu_seconds=time.perf_counter()-begin,time.process_time()-cpu
                difference=0.
                if expected is not None:
                    for key in result:
                        np.testing.assert_allclose(result[key],expected[key],atol=1e-9,rtol=1e-10)
                        difference=max(difference,float(np.max(abs(result[key]-expected[key]))))
                else:expected=result
                record=dict(traits=count,candidates=2,block_size=block,seconds=seconds,cpu_seconds=cpu_seconds,
                    maximum_difference=difference,plan=plan.to_dict(),ledger=asdict(operator.ledger))
                records.append(record);print(json.dumps({k:v for k,v in record.items() if k not in ('plan','ledger')}),flush=True)
                del operator
        write_json(a.out,dict(rows=len(rows),markers=len(variants),records=records,
            source=source.identity,native_plan='same two contexts and paired priors as direction training; trans annotation retained',
            scope='bounded real-marker products, random RHS, warm/contended timings; not full fitting or full-marker throughput',
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=4096);p.add_argument('--traits',default='1,12')
    p.add_argument('--block-sizes',default='128,512,4096')
    p.add_argument('--num-threads',type=int,default=2);p.add_argument('--memory-gib',type=float,default=16)
    run(p.parse_args())


if __name__=='__main__':main()
