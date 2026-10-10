"""Matched PCGC reducer timings; run through scripts/pcgc/gxe_python.py.

The baseline is an existing source snapshot. Both versions use the same native
runtime, inputs, seeds and worker placement. No genotype or result file is
overwritten. This measures the reducers, not a full-trait analysis.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
from scipy.special import ndtr
from summit.pcgc import architecture,sampling
from summit.ldscore.matrix_products import MatrixProducts
from summit.sumstats.binary import prepare_binary_risk


def previous(root,name,package='pcgc'):
    path=root/'src/summit'/package/f'{name}.py'
    spec=importlib.util.spec_from_file_location(f'summit.{package}._baseline_{name}',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--n',type=int,default=1024)
    parser.add_argument('--m',type=int,default=256)
    parser.add_argument('--partners',type=int,default=32)
    parser.add_argument('--architecture-probes',type=int,default=8)
    parser.add_argument('--reference-probes',type=int,default=64)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    old_sampling,sh=previous(args.baseline,'sampling')
    old_architecture,ah=previous(args.baseline,'architecture')
    old_pass2,ph=previous(args.baseline,'generalized_gxe_pass2','ldscore')
    backend=MatrixProducts(native=True)
    threads=backend.threads
    rng=np.random.default_rng(20261009)
    n,m,k,q=args.n,args.m,8,3
    c=k*q*(q+1)//2
    context=np.column_stack((np.ones(n),rng.uniform(-1,1,(n,q-1))))
    risk=prepare_binary_risk(np.arange(n)%2,.1,population_risk=ndtr(-1.3+.3*context[:,1]))
    features=context*risk.sensitivity[:,None]
    probabilities=sampling.partner_proposal(features,risk.z)
    partner=sampling.sample_partners(probabilities,args.partners,7712)
    options=dict(pair_kernels=rng.normal(0,.03,(n,args.partners,k)),partners=partner,
        partner_probabilities=probabilities,kernel_actions=rng.normal(size=(n,c)),
        genotype_diagonal=np.ones((n,k)),contexts=context,features=features,response=risk.z,
        risk=risk,sd=np.ones(n),method='pcgc',native=True,threads=threads)
    records=[]
    for repeat in range(args.repeats):
        results={}
        order=('baseline','optimized') if repeat%2==0 else ('optimized','baseline')
        for name in order:
            module=old_sampling if name=='baseline' else sampling
            evidence={}
            gc.collect()
            started=perf_counter()
            results[name]=module.build_sampling_moments(**options,execution=evidence)
            record=dict(stage='sampling',repeat=repeat,version=name,seconds=perf_counter()-started,
                execution=evidence)
            records.append(record); print(json.dumps(record),flush=True)
        differences={}
        for name in ('constant','linear','quadratic','pair_constant','pair_linear','pair_quadratic'):
            left,right=getattr(results['baseline'],name),getattr(results['optimized'],name)
            np.testing.assert_allclose(left,right,rtol=3e-10,atol=3e-14)
            differences[name]=float(np.max(np.abs(left-right)))
        records.append(dict(stage='sampling_agreement',repeat=repeat,max_absolute_difference=differences))
        del results,left,right
    del options
    x=rng.normal(size=(n,m))
    annotation=np.eye(k)[np.arange(m)%k]
    blocks=[(start,min(start+args.block_size,m),np.asfortranarray(x[:,start:start+args.block_size]))
            for start in range(0,m,args.block_size)]
    del x
    for repeat in range(args.repeats):
        results={}
        order=('baseline','optimized') if repeat%2==0 else ('optimized','baseline')
        for name in order:
            module=old_architecture if name=='baseline' else architecture
            backend=MatrixProducts(native=True,threads=threads)
            sketch=module.ArchitectureSketch(features,annotation,risk.z>0,probes=args.architecture_probes,
                seed=11371,nn=backend.left,tn=backend.right,native=True,threads=threads)
            gc.collect()
            started=perf_counter()
            for traversal in (1,2):
                for start,stop,block in blocks:
                    sketch.read_block(traversal,start,stop,block)
                    backend.drain()
            record=dict(stage='architecture',repeat=repeat,version=name,seconds=perf_counter()-started,
                execution=backend.drain())
            records.append(record); print(json.dumps(record),flush=True)
            results[name]=sketch
        differences={}
        for name in ('source','left','right'):
            left,right=getattr(results['baseline'],name),getattr(results['optimized'],name)
            np.testing.assert_allclose(left,right,rtol=3e-10,atol=3e-14)
            differences[name]=float(np.max(np.abs(left-right)))
        records.append(dict(stage='architecture_agreement',repeat=repeat,max_absolute_difference=differences))
        del results,sketch,left,right
    from summit.pcgc import reference
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
    current_pass2=reference.GeneralizedGxEPass2Executor
    x=np.asfortranarray(np.column_stack([block for _,_,block in blocks]))
    try:
        for repeat in range(args.repeats):
            results={}
            order=('baseline','optimized') if repeat%2==0 else ('optimized','baseline')
            for name in order:
                reference.GeneralizedGxEPass2Executor=(old_pass2.GeneralizedGxEPass2Executor
                    if name=='baseline' else current_pass2)
                collector=reference.ProbeGramCollector(annotation,q,args.reference_probes,n,native=True,threads=threads)
                gc.collect()
                started=perf_counter()
                result,operator,_=reference.generalized_reference(ArraySequentialGenotypeOperator(x),annotation,features,
                    probes=args.reference_probes,seed=76413,native=True,threads=threads,block_size=args.block_size,
                    memory_bytes=4*2**30,probe_product_sink=collector)
                record=dict(stage='reference',repeat=repeat,version=name,seconds=perf_counter()-started,
                    pass2=dict(result.telemetry['phase_seconds']),target_tn=dict(result.telemetry['target_tn']))
                assert operator.observed_passes==2
                results[name]=(result,collector.deviations())
                records.append(record); print(json.dumps(record),flush=True)
            differences={}
            for name in ('directional_ldscores','genetic_gram','same_person'):
                left,right=getattr(results['baseline'][0],name),getattr(results['optimized'][0],name)
                np.testing.assert_allclose(left,right,rtol=3e-11,atol=2e-12)
                differences[name]=float(np.max(np.abs(left-right)))
            left,right=results['baseline'][1],results['optimized'][1]
            np.testing.assert_allclose(left,right,rtol=3e-10,atol=3e-11)
            differences['probe_deviations']=float(np.max(np.abs(left-right)))
            records.append(dict(stage='reference_agreement',repeat=repeat,max_absolute_difference=differences))
            del results,result,collector,left,right
    finally:
        reference.GeneralizedGxEPass2Executor=current_pass2
    import summit
    root=Path(summit.__file__).resolve().parents[2]
    result=dict(passed=True,n=n,m=m,k=k,q=q,partners=args.partners,architecture_probes=args.architecture_probes,
        block_size=args.block_size,threads=threads,numpy=np.__version__,records=records,
        baseline_sha256=dict(sampling=sh,architecture=ah,pass2=ph),
        current_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('src/summit/pcgc/sampling.py','src/summit/pcgc/architecture.py','src/native/pcgc_moments.inc',
             'src/summit/ldscore/generalized_gxe_pass2.py')},
        combined_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        native_build=backend.module.build_info())
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')


if __name__=='__main__': main()
