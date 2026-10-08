"""Measure covariance RHS tiling on real blocks without storing genotypes."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.cli import _rows,_variants,_aligned_table
from scripts.epistasis.polygenic_operator import fit_kernel_scales,PolygenicKernels
from scripts.epistasis.full_matched import write_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=256);p.add_argument('--probes',type=int,default=128)
    p.add_argument('--block-sizes',default='128')
    p.add_argument('--rhs-columns',default='1408,128')
    p.add_argument('--weighted',action='store_true')
    p.add_argument('--num-threads',type=int,default=1);p.add_argument('--memory-gib',type=float,default=16)
    a=p.parse_args();root=a.training.parent
    spec=json.loads(a.training.read_text())
    rng=np.random.default_rng(720397)
    with FileGenotypeSource(root/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as source:
        rows=np.sort(_rows(source,root/spec['samples']))
        full=_variants(source,root/spec['variants'])
        variants=np.sort(rng.choice(full,a.markers,replace=False))
        scale=fit_kernel_scales(source,rows,variants,threads=a.num_threads)
        cv=spec['covariates']
        z=_aligned_table(root/cv['file'],[source.samples[i] for i in rows])[cv['varying_effects']].to_numpy(float)
        contexts=np.column_stack([np.ones(len(rows)),z])
        noise=np.ones((len(rows),1))
        vectors=rng.choice([-1.,1.],(len(rows),a.probes))
        records=[]
        expected=None
        coefficients=None
        for block in map(int,a.block_sizes.split(',')):
            operator=PolygenicKernels(source,rows,variants,scale,contexts,noise,threads=a.num_threads,
                block_size=block,storage='packed',memory_bytes=int(a.memory_gib*2**30))
            product=operator.product
            elapsed={}
            def measured(left,right,*,transpose=False):
                begin=time.perf_counter()
                result=product(left,right,transpose=transpose)
                key='transpose' if transpose else 'forward'
                elapsed[key]=elapsed.get(key,0.)+time.perf_counter()-begin
                return result
            operator.product=measured
            if a.weighted and expected is None:
                coefficients=rng.uniform(.1,1.,(operator.count,a.probes))
                components=operator.apply(vectors)
                expected=np.einsum('knb,kb->nb',components,coefficients)
                del components
            for tile in map(int,a.rhs_columns.split(',')):
                elapsed.clear()
                operator.rhs_columns=tile
                operator.native.reset_gemm_telemetry()
                begin,cpu=time.perf_counter(),time.process_time()
                value=operator.apply(vectors,coefficients)
                seconds,cpu_seconds=time.perf_counter()-begin,time.process_time()-cpu
                error=0. if expected is None else float(np.max(abs(value-expected)))
                if expected is not None:
                    np.testing.assert_allclose(value,expected,atol=1e-8,rtol=1e-10)
                vendor=operator.native.consume_gemm_telemetry()
                record=dict(block_size=block,rhs_columns=tile,weighted=a.weighted,seconds=seconds,cpu_seconds=cpu_seconds,
                    product_temporary_bytes=8*len(rows)*contexts.shape[1]*min(a.probes,tile//contexts.shape[1]),
                    maximum_difference=error,product_seconds=dict(elapsed),
                    vendor_calls=len(vendor),vendor_seconds=sum(v['wall_seconds'] for v in vendor),
                    vendor_cpu_seconds=sum(v['process_cpu_seconds'] for v in vendor),
                    vendor_flops=sum(v['flop_count'] for v in vendor),
                    vendor_shapes=sorted({(v['m'],v['n'],v['k']) for v in vendor}),
                    vendor_threads=sorted({v['backend_threads'] for v in vendor}))
                records.append(record)
                print(json.dumps(record),flush=True)
                if expected is None: expected=value
        write_json(a.out,dict(rows=len(rows),markers=len(variants),probes=a.probes,records=records,
            context_count=contexts.shape[1],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            scope='small real-genotype block throughput; warm-cache comparison; not full-marker readiness',
            peak_memory_scope='both outputs retained for direct equivalence; not separate-process peak comparison'))
        print(json.dumps(records),flush=True)


if __name__=='__main__': main()
