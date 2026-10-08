"""Bounded performance/equivalence check of covariance precision contraction.

Synthetic hard calls exercise the production native operators with 14 kernels.
This benchmark measures computation, not calibration or full-data throughput.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.prediction.genotype import ArrayGenotypeSource
from summit.prediction.spec import VariantAxis
from summit.prediction.artifacts import file_digest
from summit.epistasis.polygenic import PolygenicKernels,fit_kernel_scales,he_geometry,estimate_components
from scripts.epistasis.full_matched import write_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-threads',type=int,default=1)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('benchmark output must be new')
    rng=np.random.default_rng(9263801)
    n,m,b=4096,4096,3
    raw=rng.binomial(2,rng.uniform(.15,.45,m),(n,m)).astype(np.int8)
    axis=VariantAxis(tuple('v'+str(j) for j in range(m)),('2',)*m,
        tuple(range(1,m+1)),('A',)*m,('C',)*m)
    source=ArrayGenotypeSource(raw,[(str(i),str(i)) for i in range(n)],axis,hard_calls=True)
    rows=np.arange(n);variants=np.arange(m)
    contexts=np.column_stack([np.ones(n),rng.normal(size=(n,10))])
    target=(raw[:,0]-raw[:,0].mean())/raw[:,0].std()
    noise=np.column_stack([np.ones(n),target*target])
    fixed=np.column_stack([contexts,contexts[:,1:]**2])
    scales=fit_kernel_scales(source,rows,variants,threads=args.num_threads,block_size=256,memory_bytes=2**30)
    operator=PolygenicKernels(source,rows,variants,scales,contexts,noise,threads=args.num_threads,
        block_size=256,memory_bytes=2**30,storage='packed')
    geometry=he_geometry(operator,fixed,probes=32,keep_products=False)
    y=rng.normal(size=(n,b))
    for j in range(b):
        y[:,j]+=(raw[:,1:11]-.6)@rng.normal(size=10)/np.sqrt(10)
        y[:,j]+=np.sum(contexts[:,1:]*(raw[:,11:21]-.6),axis=1)/np.sqrt(10)
    gradients=rng.uniform(.1,1.,(operator.count,b))
    records=[]
    # Reverse execution order in the second comparison to expose warm-cache
    # effects without adding a grid of statistical settings.
    for order in [('full','directional'),('directional','full')]:
        results={}
        for mode in order:
            calls=operator.stream.ledger.operator_calls
            begin=time.perf_counter()
            theta,uncertainty=estimate_components(operator,y,geometry,
                return_uncertainty=True if mode=='full' else 'directional')
            if mode=='full':
                variance=np.einsum('kb,bkl,lb->b',gradients,uncertainty['sampling'],gradients)
            else:
                direction=np.einsum('knb,kb->nb',uncertainty['influence'],gradients)
                variance=2*np.sum(direction*operator.apply(direction,theta,phase='benchmark_directional_precision'),axis=0)
            results[mode]=(theta,variance)
            records.append(dict(order=list(order),mode=mode,seconds=time.perf_counter()-begin,
                operator_calls=operator.stream.ledger.operator_calls-calls))
        np.testing.assert_array_equal(results['full'][0],results['directional'][0])
        np.testing.assert_allclose(results['full'][1],results['directional'][1],rtol=2e-11,atol=1e-12)
    ratios=[records[0]['seconds']/records[1]['seconds'],records[3]['seconds']/records[2]['seconds']]
    record=dict(n=n,m=m,traits=b,components=operator.count,threads=args.num_threads,records=records,
        observed_speedups=ratios,equivalent=True,maximum_relative_difference=float(np.max(
            np.abs(results['full'][1]-results['directional'][1])/results['full'][1])),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_sha256=file_digest(Path(__file__)),scope='bounded computational comparison; excludes shared genotype preparation and does not predict full-cohort wall time')
    write_json(args.out,record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
