"""Measure reuse of an unchanged covariance RHS across genotype blocks.

Research-only comparison: identical native products, marker order and priors.
The cached path is restricted to a single admitted RHS tile. No solver or
statistical tolerance changes, and no complete genotype array is constructed.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.epistasis.polygenic import PolygenicKernels
from summit.prediction.genotype import FileGenotypeSource
from scripts.epistasis.full_matched import write_json


def cached_product(operator, vectors, theta):
    n,b=vectors.shape;q=operator.contexts.shape[1]
    # Same conservative workspace bound as the existing single-tile path.
    needed=operator.base_bytes+8*n*(b*(operator.count+4)+b*(8*q+8))
    if needed>operator.memory_bytes:
        raise MemoryError('single covariance RHS tile exceeds memory budget')
    active=np.flatnonzero(np.any(theta[:q]!=0,axis=1))
    prepared=[]
    for kind in range(2):
        if (kind==0 and not len(active)) or (kind==1 and not np.any(theta[q])):
            continue
        context=np.asfortranarray(operator.contexts[:,active]) if kind==0 else np.ones((n,1),order='F')
        qk=context.shape[1]
        packed=np.asfortranarray((vectors[:,:,None]*context[:,None,:]).reshape(n,-1))
        prior=np.zeros((b,qk,qk))
        prior[:,np.arange(qk),np.arange(qk)]=theta[active].T if kind==0 else theta[q,:,None]
        prepared.append((kind,context,packed,prior.reshape(b,-1)))
    out=np.zeros((n,b),order='F')
    for start,selected,raw in operator.stream.blocks('cached_packing_comparison'):
        for kind,context,packed,prior in prepared:
            g=operator._block(raw,start,len(selected),kind)
            operator.native.prediction_covariance_block(g,packed,prior,context,out,
                float(len(operator.variants)),operator.plan.threads,operator.workspace)
    for k in range(operator.noise.shape[1]):
        out+=operator.noise[:,k,None]*vectors*theta[q+1+k]
    return out


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=4096)
    p.add_argument('--rhs',default='28,156')
    p.add_argument('--block-size',type=int,default=512)
    p.add_argument('--num-threads',type=int,default=1)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    metadata=json.loads((a.reference/'reference.json').read_text())
    path=Path(metadata['training_manifest']);spec=json.loads(path.read_text())
    with np.load(a.reference/'reference.npz',allow_pickle=False) as archive:
        data={k:archive[k] for k in ('rows','training_index','variants','mean','inverse_scale','contexts','noise')}
    rng=np.random.default_rng(827417)
    selected=np.sort(rng.choice(len(data['variants']),a.markers,replace=False))
    variants=data['variants'][selected];i0=data['training_index'];rows=data['rows'][i0]
    coefficients=np.asarray(json.loads(a.components.read_text())['estimates'])
    records=[]
    with FileGenotypeSource(path.parent/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as source:
        scales=dict(metadata['scales'],mean=data['mean'][:,selected],inverse_scale=data['inverse_scale'][:,selected],
            variants=source.variants.subset(variants).identity)
        operator=PolygenicKernels(source,rows,variants,scales,data['contexts'][i0],data['noise'][i0],
            threads=a.num_threads,block_size=a.block_size,memory_bytes=16*2**30,storage='packed')
        for b in map(int,a.rhs.split(',')):
            theta=coefficients[:,np.arange(b)%coefficients.shape[1]]
            vectors=rng.normal(size=(len(rows),b));expected=None
            for label in ('current','cached','cached','current'):
                operator.native.reset_gemm_telemetry()
                start,cpu=time.perf_counter(),time.process_time()
                value=operator.apply(vectors,theta) if label=='current' else cached_product(operator,vectors,theta)
                seconds,cpus=time.perf_counter()-start,time.process_time()-cpu
                if expected is None:expected=value.copy()
                np.testing.assert_allclose(value,expected,rtol=2e-12,atol=2e-12)
                calls=operator.native.consume_gemm_telemetry()
                row=dict(method=label,rhs=b,seconds=seconds,cpu_seconds=cpus,
                    maximum_difference=float(np.max(abs(value-expected))),
                    vendor_calls=len(calls),vendor_seconds=sum(v['wall_seconds'] for v in calls))
                records.append(row);print(json.dumps(row),flush=True)
    write_json(a.out,dict(rows=len(rows),markers=len(variants),block_size=a.block_size,threads=a.num_threads,
        records=records,seed=827417,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gate='at least 10 percent paired median wall gain at both RHS widths, allclose rtol/atol 2e-12',
        scope='fixed real-genotype marker subset, warm-cache products only; not solver or full-marker throughput'))


if __name__=='__main__':main()
