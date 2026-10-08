"""Measure exact zero-pattern grouping using the existing covariance kernel.

Research comparison only. No positive component is removed. The two measured
covariance fits and a genotype-only marker sample precede this timing experiment.
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


def grouped_product(operator, vectors, theta):
    """Prototype: decode once, batch RHSs with identical nonzero component sets."""
    n,b=vectors.shape;q=operator.contexts.shape[1]
    groups={}
    for j in range(b):
        groups.setdefault(tuple(np.flatnonzero(theta[:q,j])),[]).append(j)
    prepared=[]
    for active,columns in groups.items():
        if not active:continue
        columns=np.asarray(columns);active=np.asarray(active)
        phi=np.asfortranarray(operator.contexts[:,active]);qk=len(active)
        packed=np.asfortranarray((vectors[:,columns,None]*phi[:,None,:]).reshape(n,-1))
        prior=np.zeros((len(columns),qk,qk))
        prior[:,np.arange(qk),np.arange(qk)]=theta[active[:,None],columns].T
        prepared.append((columns,phi,packed,prior.reshape(len(columns),-1)))
    dominance=np.flatnonzero(theta[q]!=0)
    one=np.ones((n,1),order='F');out=np.zeros((n,b),order='F')
    for start,selected,raw in operator.stream.blocks('grouped_comparison'):
        if prepared:
            g=operator._block(raw,start,len(selected),0)
            for columns,phi,packed,prior in prepared:
                subtotal=np.zeros((n,len(columns)),order='F')
                operator.native.prediction_covariance_block(g,packed,prior,phi,subtotal,
                    float(len(operator.variants)),operator.plan.threads,operator.workspace)
                out[:,columns]+=subtotal
        if len(dominance):
            g=operator._block(raw,start,len(selected),1)
            subtotal=np.zeros((n,len(dominance)),order='F')
            operator.native.prediction_covariance_block(g,np.asfortranarray(vectors[:,dominance]),
                np.ascontiguousarray(theta[q,dominance,None]),one,subtotal,
                float(len(operator.variants)),operator.plan.threads,operator.workspace)
            out[:,dominance]+=subtotal
    for k in range(operator.noise.shape[1]):
        out+=operator.noise[:,k,None]*vectors*theta[q+1+k]
    return out


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=4096)
    p.add_argument('--num-threads',type=int,default=8)
    p.add_argument('--memory-gib',type=float,default=16)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    metadata=json.loads((a.reference/'reference.json').read_text())
    path=Path(metadata['training_manifest']);spec=json.loads(path.read_text())
    with np.load(a.reference/'reference.npz',allow_pickle=False) as data:
        arrays={k:data[k] for k in ('rows','training_index','variants','mean','inverse_scale','contexts','noise')}
    rng=np.random.default_rng(491831)
    selected=np.sort(rng.choice(len(arrays['variants']),a.markers,replace=False))
    variants=arrays['variants'][selected];i0=arrays['training_index'];rows=arrays['rows'][i0]
    coefficients=np.asarray(json.loads(a.components.read_text())['estimates'])
    records=[]
    with FileGenotypeSource(path.parent/spec['genotypes']['geno'],
            genome_build=spec['genotypes'].get('genome_build')) as source:
        scales=dict(metadata['scales'],mean=arrays['mean'][:,selected],
            inverse_scale=arrays['inverse_scale'][:,selected],variants=source.variants.subset(variants).identity)
        operator=PolygenicKernels(source,rows,variants,scales,arrays['contexts'][i0],arrays['noise'][i0],
            threads=a.num_threads,block_size=1024,memory_bytes=int(a.memory_gib*2**30),storage='packed')
        for repeats in (2,operator.count):
            theta=np.repeat(coefficients,repeats,axis=1);v=rng.normal(size=(len(rows),theta.shape[1]))
            expected=None
            for label in ('current','grouped','grouped','current'):
                operator.native.reset_gemm_telemetry()
                begin,cpu=time.perf_counter(),time.process_time()
                value=operator.apply(v,theta) if label=='current' else grouped_product(operator,v,theta)
                seconds,cpus=time.perf_counter()-begin,time.process_time()-cpu
                if expected is None:expected=value.copy()
                difference=float(np.linalg.norm(value-expected)/np.linalg.norm(expected))
                np.testing.assert_allclose(value,expected,rtol=2e-10,atol=2e-10)
                vendor=operator.native.consume_gemm_telemetry()
                row=dict(method=label,rhs=v.shape[1],seconds=seconds,cpu_seconds=cpus,
                    relative_difference=difference,vendor_calls=len(vendor),
                    vendor_seconds=sum(x['wall_seconds'] for x in vendor),
                    vendor_flops=sum(x['flop_count'] for x in vendor))
                records.append(row);print(json.dumps(row),flush=True)
    write_json(a.out,dict(records=records,training_n=len(rows),markers=len(variants),
        coefficients=coefficients,seed=491831,order='current/grouped/grouped/current for each RHS width',
        gate='at least 10 percent median speed gain at both RHS widths, unchanged numerical result',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='warm fixed-marker subset, same native kernel; not a full-marker speed or statistical claim'))


if __name__=='__main__':main()
