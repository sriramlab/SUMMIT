"""Disk-backed native workloads by model family; warm-cache I/O is labeled."""
import argparse
import json
import os
from pathlib import Path
import resource
import tempfile
import time
import numpy as np
from bed_reader import to_bed
from summit.prediction.genotype import FileGenotypeSource,native_module
from summit.prediction.runtime import configure_prediction_threads
from summit.epistasis.models import annotation_weights,target_design
from summit.epistasis.prepare import SelectedStudy,fit_scale
from summit.epistasis.features import prepare_feature_reference,prepare_shared_target_sources
from summit.epistasis.score import prepare_linear_scores,linear_score_tests
from summit.epistasis.cli import _jsonable


def io():
    return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=4096);parser.add_argument('--variants',type=int,default=8192)
    parser.add_argument('--num-threads',type=int,default=2);args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    native=native_module();configure_prediction_threads(native,args.num_threads)
    rng=np.random.default_rng(39214);n,m=args.samples,args.variants;records=[]
    def measured(name,function):
        started=time.perf_counter();cpu=time.process_time();before=io();result=function();after=io()
        records.append(dict(workload=name,seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            io_delta={k:after[k]-before[k] for k in before}))
        print(name,round(records[-1]['seconds'],3),flush=True);return result
    with tempfile.TemporaryDirectory(prefix='summit-epistasis-disk-') as temporary:
        root=Path(temporary);raw=rng.binomial(2,.3,(n,m)).astype(np.int8)
        to_bed(root/'input.bed',raw,properties=dict(sid=[f'v{i}' for i in range(m)],chromosome=['1']*m,
            bp_position=np.arange(1,m+1),allele_1=['A']*m,allele_2=['G']*m));del raw
        with FileGenotypeSource(root/'input.bed') as source:
            scale=measured('common_genotype_scale',lambda:fit_scale(source,np.arange(n),threads=args.num_threads,block_size=128))
            annotations=annotation_weights(source.variants.ids,{'a':{f'v{i}':1. for i in range(16)},
                'b':{f'v{i}':1. for i in range(8,40)}})
            y=rng.normal(size=(n,4))
            def study(job):
                design=target_design(source,np.arange(n),scale,components=job.get('components',[]),annotations=annotations,
                    additive_annotations=job['additive_annotations'],allow_additive_only='components' not in job,
                    threads=args.num_threads,block_size=128,native=native)
                return SelectedStudy(source,np.arange(n),scale,**design,threads=args.num_threads,block_size=128,memory_bytes=2**30)
            one=dict(id='single',additive_annotations=['all'],components=[dict(name='epi',target='v0',background='all')])
            single=study(one)
            reference=measured('single_target_generalized_reference_two_pass',lambda:single.reference(nvecs=32,seed=18))
            summary=measured('single_target_four_traits_analytic_two_pass',lambda:single.summarize(reference,y,trait_names=('a','b','c','d'))[0])
            records[-1]['passes']=dict(single.stream.ledger.traversals)
            jobs=[dict(id=f'target{i}',additive_annotations=['all'],components=[dict(name='epi',target=f'v{i}',background='all')]) for i in range(4)]
            targets=[study(j) for j in jobs]
            shared=measured('four_independent_target_source_sketches_one_pass',lambda:prepare_shared_target_sources(targets[0],
                [s.weights[:,1] for s in targets],dimensions=64,seed=18))
            references=measured('four_independent_target_features_separate_projections',lambda:[prepare_feature_reference(s,j,annotations,
                sketch_dimensions=64,seed=18,shared_sources=shared) for s,j in zip(targets,jobs)])
            for s,ref in zip(targets,references):
                s.nn.begin_execution();s.tn.begin_execution()
                scores=measured('independent_target_four_traits_scores',lambda:prepare_linear_scores(ref.features,y,ref.fixed_effects,
                    feature_names=ref.metadata['feature_names'],trait_names=('a','b','c','d'),metadata={},nn=s._nn,tn=s._tn))
                s.nn.finish_execution()
                records[-1]['feature_count']=ref.features.shape[1]
            del references,shared,targets
            for name,job,r in [
                ('annotations',dict(id='annotations',additive_annotations=['all','a','b'],components=[
                    dict(name='one',target='v0',background='a'),dict(name='two',target='v0',background='b')]),64),
                ('overlapping_groups',dict(id='groups',additive_annotations=['all','a','b'],groups=[dict(name='cross',mode='cross',left='a',right='b')]),64),
                ('set_remainder',dict(id='remainder',additive_annotations=['all','a'],groups=[dict(name='remainder',mode='remainder',left='a')]),64),
                ('within_group',dict(id='within',additive_annotations=['all','b'],groups=[dict(name='within',mode='within',left='b')]),64),
                ('supplied_pairs',dict(id='pairs',additive_annotations=['all'],pairs=[['v0',f'v{i}'] for i in range(1,13)]),None)]:
                s=study(job)
                ref=measured(name+'_features',lambda:prepare_feature_reference(s,job,annotations,sketch_dimensions=r,seed=18))
                s.nn.begin_execution();s.tn.begin_execution()
                scores=measured(name+'_four_traits_scores',lambda:prepare_linear_scores(ref.features,y,ref.fixed_effects,
                    feature_names=ref.metadata['feature_names'],trait_names=('a','b','c','d'),metadata={},nn=s._nn,tn=s._tn))
                s.nn.finish_execution()
                records[-1].update(feature_count=ref.features.shape[1],fixed_columns=ref.fixed_effects.shape[1])
                measured(name+'_four_traits_inference',lambda:[linear_score_tests(scores,trait=i) for i in range(4)])
            record=dict(n=n,m=m,seed=39214,traits=4,threads=args.num_threads,records=records,
                input='temporary synthetic disk-backed PLINK BED; all genotype files removed after measurement',
                cache='warm operating-system cache after writing; read_bytes and rchar reported separately; no cold-storage claim',
                interpretation='family workload qualification only; iid score timing does not validate omission of a polygenic null',
                native_path=native.__file__,native_build=native.build_info(),worker_affinity={p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:json.dump(_jsonable(record),handle,indent=2,allow_nan=False)


if __name__=='__main__':main()
