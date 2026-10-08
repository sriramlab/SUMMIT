"""Compare the complete derivative span with its exact redundant-column reduction."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.prediction.genotype import FileGenotypeSource
from summit.epistasis.krylov import KrylovSpace
from summit.epistasis.polygenic import PolygenicKernels,projected_solve,conditional_tangents_batch
from scripts.epistasis.full_matched import write_json


def full_reference(training,complete,i0,i1,alpha,fixed,theta,inverses):
    """Original complete-component calculation, retained only for comparison."""
    n,b,k=len(complete.rows),alpha.shape[1],training.count
    component,_=complete.cross_products(alpha,i0,np.arange(n))
    rhs=component[:,i0,:].transpose(1,2,0).reshape(len(i0),b*k)
    repeated=np.repeat(theta,k,axis=1)
    derivative,report=projected_solve(training,rhs,fixed,repeated,
        preconditioners=[p for p in inverses for _ in range(k)])
    tangent=component[:,i1,:].transpose(1,2,0).copy()
    correction,_=complete.cross_products(derivative,i0,i1,repeated)
    tangent-=correction.reshape(len(i1),b,k)
    scale=theta/np.linalg.norm(theta,axis=0)
    tangent-=np.einsum('nbk,kb->nb',tangent,scale)[:,:,None]*scale.T[None,:,:]
    return tangent,report


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=4096)
    p.add_argument('--num-threads',type=int,default=1)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((a.reference/'reference.json').read_text())
    path=Path(metadata['training_manifest']);spec=json.loads(path.read_text())
    with np.load(a.reference/'reference.npz') as archive:
        data={k:archive[k] for k in ('rows','training_index','confirmation_index','variants','mean','inverse_scale','fixed','contexts','noise')}
    rng=np.random.default_rng(927413)
    selected=np.sort(rng.choice(len(data['variants']),a.markers,replace=False))
    variants=data['variants'][selected];i0=data['training_index'];i1=data['confirmation_index']
    theta=np.asarray(json.loads(a.components.read_text())['estimates'])[:,:2]
    y=rng.normal(size=(len(i0),2));basis=thin_rank_revealing_fixed_effect_basis(data['fixed'][i0],rtol=1e-11)
    start=time.perf_counter()
    with FileGenotypeSource(path.parent/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as s0, \
         FileGenotypeSource(path.parent/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as s1:
        scales=dict(metadata['scales'],mean=data['mean'][:,selected],inverse_scale=data['inverse_scale'][:,selected],
            variants=s0.variants.subset(variants).identity)
        training=PolygenicKernels(s0,data['rows'][i0],variants,scales,data['contexts'][i0],data['noise'][i0],
            threads=a.num_threads,block_size=4096,memory_bytes=16*2**30,storage='packed')
        complete=PolygenicKernels(s1,data['rows'],variants,scales,data['contexts'],data['noise'],
            threads=a.num_threads,block_size=512,memory_bytes=16*2**30)
        spaces=[KrylovSpace(training,theta[:,j],basis,capacity=64) for j in range(2)]
        alpha,initial=projected_solve(training,y,data['fixed'][i0],theta,recycle_spaces=spaces)
        inverses=[s.freeze(training.diagonal.T@theta[:,j]) for j,s in enumerate(spaces)]
        print('initial solve',initial.elapsed_seconds,flush=True)
        records=[];results=[]
        for label in ('complete','reduced'):
            begin,cpu=time.perf_counter(),time.process_time()
            if label=='complete':
                value,fit=full_reference(training,complete,i0,i1,alpha,data['fixed'][i0],theta,inverses)
            else:
                value,fit=conditional_tangents_batch(training,complete,i0,i1,alpha,data['fixed'][i0],theta,
                    preconditioners=inverses)
            record=dict(method=label,seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,
                reports={'/'.join(k):v for k,v in fit.reports.items()})
            write_json(a.out/(label+'.json'),record);records.append(record);results.append(value)
            print(label,record['seconds'],flush=True)
        expected,actual=results;comparisons=[]
        # Projectors are checked through their small cross-Gram, never N x N.
        for j in range(2):
            q0=thin_rank_revealing_fixed_effect_basis(expected[:,j],rtol=1e-11)
            q1=thin_rank_revealing_fixed_effect_basis(actual[:,j],rtol=1e-11)
            projection_error=np.linalg.norm(q0-q1@(q1.T@q0))
            comparisons.append(dict(rank_complete=q0.shape[1],rank_reduced=q1.shape[1],
                relative_tangent_difference=float(np.linalg.norm(actual[:,j]-expected[:,j])/np.linalg.norm(expected[:,j])),
                projection_error=float(projection_error)))
            assert q0.shape[1]==q1.shape[1]
            assert projection_error<2e-6
        write_json(a.out/'comparison.json',dict(rows=len(data['rows']),training=len(i0),markers=len(variants),
            traits=2,initial_seconds=initial.elapsed_seconds,records=records,comparisons=comparisons,
            wall_reduction_fraction=1-records[1]['seconds']/records[0]['seconds'],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.perf_counter()-start,
            training_ledger=asdict(training.stream.ledger),confirmation_ledger=asdict(complete.stream.ledger),
            scope='warm real genotype subset computational comparison, no statistical qualification'))


if __name__=='__main__':main()
