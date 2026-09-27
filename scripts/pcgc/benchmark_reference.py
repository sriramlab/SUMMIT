#!/usr/bin/env python3
"""Bounded reference throughput/parity check and full-shape memory plans.

Use --real-genotypes for existing EUR BED calls with the population scale.
Responses/risks then remain synthetic: this is a throughput check, not a trait
analysis. Only aggregate timing, numerical errors and axis hashes are saved.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--extension-dir',type=Path)
    parser.add_argument('--auxiliary-extension-dir',type=Path)
    parser.add_argument('--samples',type=int,default=4000)
    parser.add_argument('--variants',type=int,default=4000)
    parser.add_argument('--annotations',type=int,default=8)
    parser.add_argument('--probes',type=int,default=128)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--layout',choices=['partition','overlap'],default='partition')
    parser.add_argument('--engines',nargs='+',choices=['generalized','rank_one'],default=['generalized','rank_one'])
    parser.add_argument('--real-genotypes',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    sys.meta_path=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
    sys.path.insert(0,str(root/'src'))
    import summit
    if args.extension_dir:
        summit.__path__.append(str(args.extension_dir.resolve(strict=True)))
    if args.auxiliary_extension_dir:
        summit.__path__.append(str(args.auxiliary_extension_dir.resolve(strict=True)))
    import numpy as np
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
    from summit.pcgc.reference import generalized_reference,contract_reference
    from summit.pcgc.rank_one import rank_one_reference,plan_pcgc_reference
    from summit.ldscore.generalized_gxe_variant import GeneralizedGxEPlanInputs,plan_generalized_gxe_variant_work
    from summit.context.spec import array_sha256

    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(41,1,0,0,0) or libc.prctl(42,0,0,0,0)!=1:
        raise RuntimeError('THP guard failed')
    affinity=sorted(os.sched_getaffinity(0))
    if args.extension_dir and len(affinity)!=args.threads:
        raise ValueError('bind process to one physical CPU per requested thread')
    files=[Path(__file__),root/'src/summit/pcgc/rank_one.py',root/'src/summit/pcgc/reference.py',root/'src/summit/pcgc/genotype.py']
    result=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                affinity=affinity,thp_disabled=True,repeats=[],plans=[])
    for k in (1,8,24,100):
        p=plan_pcgc_reference(num_samples=300000,num_variants=454000,num_annotations=k,probes=256,
                              threads=8,memory_bytes=128*2**30)
        old=plan_generalized_gxe_variant_work(GeneralizedGxEPlanInputs(
            num_samples=300000,num_variants=454000,num_basis=1,num_annotations=k,num_probes=256,
            threads=8,memory_limit_bytes=256*2**30,genotype_format='bed',
            preferred_variant_block_width=256,preferred_rhs_tile_columns=64,rhs_policy='tiled'))
        result['plans'].append(dict(annotations=k,rank_one=p.to_dict(),generalized_reference_only_bytes=old.peak_resident_bytes))
    n,m,k=args.samples,args.variants,args.annotations
    rng=np.random.default_rng(488163)
    source=None
    if args.real_genotypes:
        if not args.extension_dir:
            parser.error('real genotypes require native extensions')
        import pandas as pd
        from summit.prediction.genotype import FileGenotypeSource
        from summit.prediction.spec import GenotypeScale
        from summit.pcgc.genotype import scaled_file_operator
        source=FileGenotypeSource('/home/bronsonj/UKBB/geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc_imp.bed',genome_build='GRCh37')
        if n>len(source.samples):
            raise ValueError('requested more people than existing panel')
        path=Path('/home/bronsonj/UKBB/02_genetics_of_recognition/gwas/pcgc/freq_all.afreq')
        af=pd.read_csv(path,sep='\t').set_index('ID').loc[list(source.variants.ids)]
        if tuple(af.ALT)!=source.variants.counted or tuple(af.REF)!=source.variants.other:
            raise ValueError('frequency allele mismatch')
        f=af.ALT_FREQS.to_numpy(float)
        variants=np.sort(rng.choice(np.flatnonzero((f>.05)&(f<.95)),m,replace=False))
        rows=np.arange(n,dtype=np.int64)
        scale=GenotypeScale(2*f,1/np.sqrt(2*f*(1-f)),source.variants.identity,'population_frequency',{'population_scale':True},ddof=0)
        result.update(genotype_identity=source.identity,variant_indices_sha256=array_sha256(variants),
                      frequency_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        def operator():
            return scaled_file_operator(source,scale,rows,threads=args.threads,variant_indices=variants)
    else:
        x=rng.normal(size=(n,m))
        def operator():
            return ArraySequentialGenotypeOperator(x)
    if args.layout=='partition':
        a=np.eye(k)[np.arange(m)%k]
    else:
        a=rng.uniform(.1,1.,(m,k)); a[:,0]=1.
    w=rng.uniform(.2,1.5,n)
    responses=rng.normal(size=(n,1))
    opts=dict(probes=args.probes,seed=19247,threads=args.threads,block_size=256,
              memory_bytes=16*2**30,native=bool(args.extension_dir))
    previous=None
    try:
        for repeat in range(args.repeats):
            # Alternate order to avoid assigning all warm-cache runs to one engine.
            engines=args.engines if repeat%2==0 else args.engines[::-1]
            for engine in engines:
                op=operator(); start=time.perf_counter()
                if engine=='generalized':
                    ref,scored,plan=generalized_reference(op,a,w[:,None],responses=responses,collect_diagonal_rows=True,**opts)
                    ld,sp=contract_reference(ref,np.ones(1))
                    ld-=scored.reference_diagonal_rows[:,0]/n**2
                    score,diag=scored.scores,scored.diagonals
                    planned=plan.peak_resident_bytes+scored.pcgc_score_buffer_bytes
                    extra={}
                else:
                    ref=rank_one_reference(op,a,w,responses=responses,**opts)
                    ld,sp,score,diag=ref.ldscores,ref.same_person,ref.scores,ref.diagonals
                    planned=ref.plan.peak_resident_bytes
                    extra=ref.diagnostics
                seconds=time.perf_counter()-start
                arrays=(ld.copy(),sp.copy(),(score[:,0]**2-diag[:,0]).copy())
                if previous is None:
                    previous=arrays
                errors=[float(np.linalg.norm(cur-old)/max(np.linalg.norm(old),1e-300)) for cur,old in zip(arrays,previous)]
                if max(errors)>2e-10:
                    raise RuntimeError('reference parity failed: '+str(errors))
                # Small designs need not yield an identifiable finite-probe fit;
                # reference parity is tested independently of a solver gate.
                entry=dict(engine=engine,repeat=repeat,seconds=seconds,planned_bytes=planned,
                           relative_errors=errors,passes=op.observed_passes,details=extra)
                result['repeats'].append(entry)
                print(json.dumps({key:entry[key] for key in ('engine','repeat','seconds','planned_bytes','relative_errors')}),flush=True)
                del ref,ld,sp,score,diag,arrays,op
                if engine=='generalized':
                    del scored
    finally:
        if source is not None:
            source.close()
    result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if args.extension_dir:
        from summit.prediction.genotype import native_module
        module=native_module()
        result['openmp_placement']=module.configure_openmp_placement(affinity,args.threads)
        info=module.build_info()
        result['native_build']={key:info.get(key) for key in ('blas_vendor','source_commit','source_tree_sha256',
            'blas_runtime_threads','blas_runtime_worker_affinity_policy','blas_runtime_thread_strategy')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False); handle.write('\n')


if __name__=='__main__':
    main()
