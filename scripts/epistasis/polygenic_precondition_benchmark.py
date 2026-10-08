"""Measured computational inverse comparison on a fixed real-marker subset."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.prediction.genotype import FileGenotypeSource
from summit.epistasis.polygenic import (PolygenicKernels,he_geometry,
    projected_solve,prepare_preconditioners)
from scripts.epistasis.full_matched import write_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=4096)
    p.add_argument('--probes',type=int,default=128)
    p.add_argument('--method',choices=['nystrom','recycle'],default='nystrom')
    p.add_argument('--num-threads',type=int,default=1)
    p.add_argument('--memory-gib',type=float,default=16)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((a.reference/'reference.json').read_text())
    spec_path=Path(metadata['training_manifest'])
    spec=json.loads(spec_path.read_text())
    with np.load(a.reference/'reference.npz') as archive:
        data={k:archive[k] for k in ('rows','training_index','variants','mean','inverse_scale','fixed','contexts','noise')}
    rng=np.random.default_rng(927413)
    selected=np.sort(rng.choice(len(data['variants']),a.markers,replace=False))
    variants=data['variants'][selected]
    i0=data['training_index'];rows=data['rows'][i0];fixed=data['fixed'][i0]
    theta=np.asarray(json.loads(a.components.read_text())['estimates'])[:,:1]
    theta=np.repeat(theta,2,axis=1)
    y=rng.normal(size=(len(rows),2))
    records=[];start=time.perf_counter()
    with FileGenotypeSource(spec_path.parent/spec['genotypes']['geno'],
            genome_build=spec['genotypes'].get('genome_build')) as source:
        scales=dict(metadata['scales'],mean=data['mean'][:,selected],inverse_scale=data['inverse_scale'][:,selected],
            variants=source.variants.subset(variants).identity)
        operator=PolygenicKernels(source,rows,variants,scales,data['contexts'][i0],data['noise'][i0],
            threads=a.num_threads,block_size=4096,memory_bytes=int(a.memory_gib*2**30),storage='packed')
        begin=time.perf_counter()
        if a.method=='nystrom':
            geometry=he_geometry(operator,fixed,probes=a.probes,seed=871631)
            preconditioners=prepare_preconditioners(operator,geometry,theta)
        else:
            from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
            from summit.epistasis.krylov import KrylovSpace
            basis=thin_rank_revealing_fixed_effect_basis(fixed,rtol=1e-11)
            space=KrylovSpace(operator,theta[:,0],basis,capacity=64)
            initial,first=projected_solve(operator,y,fixed,theta,recycle_spaces=[space,space])
            # Subsequent actual covariance-mean derivative right-hand sides,
            # the computation that motivates recycling in conditional inference.
            y=operator.apply(initial[:,:1])[:,:,0].T
            theta=np.repeat(theta[:,:1],operator.count,axis=1)
            inverse=space.freeze(operator.diagonal.T@theta[:,0])
            preconditioners=[inverse]*operator.count
            write_json(a.out/'initial_solve.json',dict(reports={'/'.join(k):v for k,v in first.reports.items()},
                recycle_rank=inverse.d.shape[1],seconds=first.elapsed_seconds))
            print('initial recycle solve',round(first.elapsed_seconds,2),flush=True)
        sketch_seconds=time.perf_counter()-begin
        for label,inverse in [(a.method,preconditioners),('jacobi',None)]:
            begin,cpu=time.perf_counter(),time.process_time()
            solution,fit=projected_solve(operator,y,fixed,theta,preconditioners=inverse,
                checkpoint=a.out/(label+'.npz'))
            if label==a.method:
                expected=solution
                difference=0.
            else:
                difference=float(np.linalg.norm(solution-expected)/np.linalg.norm(expected))
                np.testing.assert_allclose(solution,expected,rtol=2e-6,atol=2e-7)
            record=dict(method=label,seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,
                reports={'/'.join(k):v for k,v in fit.reports.items()},relative_solution_difference=difference)
            records.append(record);write_json(a.out/(label+'.json'),record)
            print(label,round(record['seconds'],2),flush=True)
        write_json(a.out/'comparison.json',dict(rows=len(rows),markers=len(variants),rhs=y.shape[1],
            covariance_components=theta[:,0],genotype_only_probe_count=a.probes,sketch_seconds=sketch_seconds,
            preconditioner_bytes=preconditioners[0].nbytes,records=records,
            seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            ledger=asdict(operator.stream.ledger),
            scope='warm fixed-marker-subset computational comparison; no statistical evidence or full-marker throughput claim'))


if __name__=='__main__':
    main()
