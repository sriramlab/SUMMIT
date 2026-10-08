"""Compare selected-row covariance products with complete padded products."""
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
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=1024)
    p.add_argument('--columns',type=int,default=4)
    p.add_argument('--block-size',type=int,default=1024)
    p.add_argument('--num-threads',type=int,default=1)
    p.add_argument('--memory-gib',type=float,default=32)
    a=p.parse_args();root=a.training.parent
    if a.out.exists():
        raise FileExistsError(a.out)
    spec=json.loads(a.training.read_text());rng=np.random.default_rng(720411)
    records=[]
    with FileGenotypeSource(root/spec['genotypes']['geno'],genome_build=spec['genotypes'].get('genome_build')) as source:
        training=np.sort(_rows(source,root/spec['samples']))
        confirmation=np.sort(_rows(source,a.confirmation.resolve()))
        if np.intersect1d(training,confirmation).size:
            raise ValueError('benchmark cohorts must be disjoint')
        rows=np.union1d(training,confirmation)
        i0,i1=np.searchsorted(rows,training),np.searchsorted(rows,confirmation)
        full=_variants(source,root/spec['variants'])
        variants=np.sort(rng.choice(full,a.markers,replace=False))
        scales=fit_kernel_scales(source,training,variants,threads=a.num_threads)
        cv=spec['covariates']
        z=_aligned_table(root/cv['file'],[source.samples[i] for i in rows])[cv['varying_effects']].to_numpy(float)
        contexts=np.column_stack([np.ones(len(rows)),z])
        operator=PolygenicKernels(source,rows,variants,scales,contexts,np.ones((len(rows),1)),
            threads=a.num_threads,block_size=a.block_size,memory_bytes=int(a.memory_gib*2**30))
        theta=rng.uniform(.1,1.,(operator.count,a.columns))
        for label,take,give in [('training_to_confirmation',i0,i1),('confirmation_to_training',i1,i0)]:
            v=rng.normal(size=(len(take),a.columns))
            padded=np.zeros((len(rows),a.columns));padded[take]=v
            begin,cpu=time.perf_counter(),time.process_time()
            expected=operator.apply(padded,theta)
            baseline_seconds=time.perf_counter()-begin;baseline_cpu=time.process_time()-cpu
            expected_quadratic=np.sum(v*expected[take],axis=0)
            operator.native.reset_gemm_telemetry()
            begin,cpu=time.perf_counter(),time.process_time()
            actual,quadratic=operator.cross_products(v,take,give,theta)
            seconds,cpu_seconds=time.perf_counter()-begin,time.process_time()-cpu
            vendor=operator.native.consume_gemm_telemetry()
            np.testing.assert_allclose(actual,expected[give],atol=1e-8,rtol=1e-9)
            np.testing.assert_allclose(quadratic,expected_quadratic,atol=1e-7,rtol=1e-10)
            record=dict(direction=label,seconds=seconds,cpu_seconds=cpu_seconds,
                padded_seconds=baseline_seconds,padded_cpu_seconds=baseline_cpu,
                maximum_product_difference=float(np.max(abs(actual-expected[give]))),
                maximum_quadratic_relative_difference=float(np.max(abs(quadratic-expected_quadratic)/expected_quadratic)),
                vendor_calls=len(vendor),vendor_seconds=sum(r['wall_seconds'] for r in vendor),
                vendor_cpu_seconds=sum(r['process_cpu_seconds'] for r in vendor))
            records.append(record);print(json.dumps(record),flush=True)
    write_json(a.out,dict(training_n=len(training),confirmation_n=len(confirmation),
        markers=a.markers,columns=a.columns,block_size=a.block_size,records=records,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='real-row warm-block comparison with both outputs held; not full-marker throughput or separate-run peak memory'))


if __name__=='__main__':main()
