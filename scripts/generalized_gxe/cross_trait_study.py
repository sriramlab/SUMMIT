"""Authenticated one-traversal masked cross-trait study and chr22 benchmark.

New outputs only. The protected live shared_research_batch.py is not modified.
Benchmark timings separate score accumulation from exact residual correction.
"""
from pathlib import Path
import argparse
import ctypes
import json
import os
import resource
import sys
import time

CPUS=tuple(sorted(os.sched_getaffinity(0)))
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(41,1,0,0,0)==0 and libc.prctl(42,0,0,0,0)==1
ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
import pgenlib
from threadpoolctl import threadpool_limits,threadpool_info
import workflow
CPUS=tuple(workflow._PRE_NUMERICAL_CPU_AFFINITY)
from summit.context.reference_zpass_cli import authenticated_reference,file_sha256
from summit.context.spec import array_sha256,canonical_sha256
from summit.context.cross_trait_zpass import ZMomentAccumulator
from summit.context.cross_trait_checkpoint import save_study_checkpoint,restore_study_checkpoint
from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
from summit.ldscore.generalized_gxe_cross_trait_batch import CrossTraitBatch

PILOT=('ldl_raw','apo_b_raw','cholesterol_raw','non_hdl_cholesterol_raw','hba1c_raw',
       'diastolic_blood_pressure_raw','height_raw','platelet_count_raw')


def select_trait_pairs(available, names, pairs_path=None):
    """Resolve requested names once; preserve orientation and reject aliases."""
    names=tuple(names)
    if not names or len(set(names))!=len(names) or not set(names)<=set(available):
        raise ValueError('trait names must be unique and present in the sealed inputs')
    if pairs_path is None:
        return names,None
    requested=json.loads(pairs_path.read_text())
    if not isinstance(requested,list) or not requested:
        raise ValueError('pairs file must be a nonempty list of trait-name pairs')
    indices=[];seen=set()
    for pair in requested:
        if not isinstance(pair,list) or len(pair)!=2 or any(t not in names for t in pair):
            raise ValueError('each requested pair must contain two selected trait names')
        i,j=map(names.index,pair)
        if i==j or tuple(sorted((i,j))) in seen:
            raise ValueError('requested cross-trait pairs must be distinct and nonduplicated')
        seen.add(tuple(sorted((i,j))));indices.append((i,j))
    return names,indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['benchmark','study'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bed-prefix',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--chromosome',type=int,default=22);p.add_argument('--width',type=int,default=128)
    p.add_argument('--variants',type=int,default=256);p.add_argument('--traits',type=int,choices=[6,8,42],default=8)
    p.add_argument('--trait-names',nargs='+',help='explicit subset of authenticated sealed trait inputs')
    p.add_argument('--pairs-file',type=Path,help='JSON list of ordered trait-name pairs; one shared traversal')
    p.add_argument('--threads',type=int,default=8);p.add_argument('--common-only',action='store_true')
    p.add_argument('--z-output',type=Path,help='collect guarded reference Z moments in this same traversal')
    p.add_argument('--max-cached-missing-patterns',type=int,default=64)
    p.add_argument('--minimum-cached-pattern-rows',type=int,default=64)
    p.add_argument('--missing-cache-strategy',choices=('patterns','pooled'),default='patterns')
    p.add_argument('--cache-addition-penalty-rows',type=float,default=8.)
    p.add_argument('--parallel-cache-products',action='store_true')
    p.add_argument('--max-cached-pair-sums',type=int,default=0,
        help='bound reusable sums of identical missing-group combinations across pairs')
    p.add_argument('--checkpoint-every-blocks',type=int,default=0,
        help='publish immutable running sums every this many decoded tiles (study only)')
    p.add_argument('--resume-from',type=Path,help='continue after this authenticated completed-tile checkpoint')
    p.add_argument('--max-run-seconds',type=float,default=0,
        help='gracefully checkpoint at a tile boundary after this process time; exit 75')
    p.add_argument('--stop-after-blocks',type=int,default=0,
        help='qualification: checkpoint and exit 75 after this many new tiles')
    p.add_argument('--residual-workers',type=int,default=1,
        help='independent residual-pair workers, one BLAS thread each on reserved CPUs')
    a=p.parse_args()
    if a.z_output is not None and a.mode!='study':p.error('Z publication requires a complete study chromosome')
    if a.max_cached_missing_patterns<0:p.error('missing-pattern cache limit must be nonnegative')
    if a.max_cached_pair_sums<0:p.error('cached pair-sum limit must be nonnegative')
    if a.minimum_cached_pattern_rows<1:p.error('cached patterns require a positive minimum row count')
    if not np.isfinite(a.cache_addition_penalty_rows) or a.cache_addition_penalty_rows<0:
        p.error('cache addition penalty must be finite and nonnegative')
    if (min(a.checkpoint_every_blocks,a.stop_after_blocks)<0 or not np.isfinite(a.max_run_seconds)
            or a.max_run_seconds<0):p.error('checkpoint limits must be finite and nonnegative')
    checkpointing=bool(a.checkpoint_every_blocks or a.resume_from or a.max_run_seconds or a.stop_after_blocks)
    if checkpointing and a.mode!='study':p.error('checkpoint/resume requires study mode')
    a.output.mkdir(exist_ok=a.resume_from is not None);start=time.monotonic()
    if (a.output/'COMPLETE.json').exists() or (a.output/'cross_trait_summary.npz').exists():
        raise FileExistsError('study publication already exists; use a new output path for recovery')
    source_hashes={str(path.relative_to(ROOT)):file_sha256(path) for path in (
        Path(__file__),ROOT/'src/summit/ldscore/generalized_gxe_masked_batch.py',
        ROOT/'src/summit/ldscore/generalized_gxe_cross_trait_batch.py',
        ROOT/'src/summit/context/cross_trait_checkpoint.py')}
    refroot=a.base/'shared_reference_full_20260916';master=a.base/'full_cohort_inputs_20260916/height_raw.npz'
    manifest,panel,reference,record=authenticated_reference(refroot/'MANIFEST.json',refroot,a.chromosome,master)
    expansion=a.base/'imputed_expansion_20260917';expanded=json.loads((expansion/'MANIFEST.json').read_text())
    inputroots={t:(a.base/'full_cohort_inputs_20260916',manifest) for t in manifest['traits']}
    inputroots.update({t:(expansion/'inputs',expanded) for t in expanded['traits'] if t!='height_raw'})
    names=a.trait_names if a.trait_names else sorted(inputroots) if a.traits==42 else PILOT[:a.traits]
    names,requested_pairs=select_trait_pairs(inputroots,names,a.pairs_file)
    if not a.trait_names and len(names)!=a.traits:
        raise ValueError(f'expected {a.traits} input traits, found {len(names)}')
    with np.load(master) as z:rows,phi,fixed,basis_names=(z[k] for k in ('rows','phi','fixed','basis_names'))
    input_hashes={}
    def load_traits():
        # The batch keeps transforms and sufficient moments. Supply one
        # full fixed basis at a time instead of retaining all T input bases.
        for name in names:
            inputroot,source_manifest=inputroots[name];path=inputroot/f'{name}.npz'
            digest=file_sha256(path)
            if digest!=source_manifest['trait_sources'][name]:raise ValueError(f'input checksum differs: {path}')
            input_hashes[str(path)]=digest
            with np.load(path) as z:
                idx=np.searchsorted(rows,z['rows']);np.testing.assert_array_equal(rows[idx],z['rows'])
                u=z['fixed'] if 'fixed' in z else fixed[idx]@z['fixed_from_master']
                yield dict(name=name,indices=idx,fixed_basis=u,phenotype=z['y'])
    annotations=np.load(a.annotations,allow_pickle=False)
    if array_sha256(annotations)!=panel['annotation_sha256']:raise ValueError('annotation checksum differs')
    prefix=str(a.bed_prefix)
    for suffix,key in (('.bim','bim_sha256'),('.fam','fam_sha256')):
        if file_sha256(prefix+suffix)!=panel[key]:raise ValueError(f'panel checksum differs: {suffix}')
    if Path(prefix+'.bed').stat().st_size!=panel['bed_size']:raise ValueError('BED size differs')
    with np.load(reference/'reference_aux.npz') as z:mean,inverse=z['affine_mean'],z['affine_inverse_scale']
    full_annotations=annotations;annotation_names=manifest['annotation_names']
    if a.common_only:
        # The sealed annotations are ordered rare, low-frequency, common.
        annotation_names=[annotation_names[2]];annotations=annotations[:,2:3]
    m=panel['m'];limit=min(a.variants,m) if a.mode=='benchmark' else m
    groups=(np.arange(m)+panel['global_start'])*manifest['njack']//manifest['variants']
    with threadpool_limits(limits=1):
        residual,residual_names,_=workflow.rank_reduced_symmetric_context_residual_basis(phi,tuple(basis_names.astype(str)))
        masked=MaskedTraitBatch(basis=phi,fixed_basis=fixed,residual_basis=residual,traits=load_traits())
        batch=CrossTraitBatch(masked,block_ids=np.unique(groups[:limit]),annotation_names=annotation_names,
            pairs=requested_pairs,
            max_cached_missing_patterns=a.max_cached_missing_patterns,
            minimum_cached_pattern_rows=a.minimum_cached_pattern_rows,
            missing_cache_strategy=a.missing_cache_strategy,
            cache_addition_penalty_rows=a.cache_addition_penalty_rows,
            parallel_cache_products=a.parallel_cache_products,
            max_cached_pair_sums=a.max_cached_pair_sums,
            residual_workers=a.residual_workers,residual_cpus=CPUS)
    z_accumulator=None;shared_tn=None;z_telemetry=[]
    if a.z_output is not None:
        if a.z_output.exists():raise FileExistsError(a.z_output)
        z_accumulator=ZMomentAccumulator(np.unique(groups[:limit]),full_annotations.shape[1],masked.q,masked.fixed_rank)
        shared_tn=ProtectedTNOperator(threads=a.threads)
        native_build=dict(shared_tn._module.build_info())
        if not native_build['gemm_integrity_enabled'] or not native_build['gemm_checksum_enabled']:
            raise RuntimeError('fused Z collection requires both native GEMM guards')
        masked.fixed_weights=np.asfortranarray(masked.fixed_weights)
        shared_tn.begin_execution()
    print(json.dumps(dict(masked.report,phase='prepared',pairs=len(batch.scores.pairs),seconds=time.monotonic()-start,
                          cached_missing_patterns=len(batch.missing_pattern_rows),
                          cached_pair_sum_groups=len(batch.cached_pair_sum_groups),
                          cached_missing_people=sum(map(len,batch.missing_pattern_rows)),
                          cpus=CPUS,blas=threadpool_info())),flush=True)
    identity=dict(chromosome=a.chromosome,variants=m,width=a.width,traits=list(names),
        pairs=batch.scores.pairs.tolist(),
        source_sha256=source_hashes,input_sha256=input_hashes,reference_files=record['files'],
        manifest_sha256=file_sha256(refroot/'MANIFEST.json'),master_sha256=file_sha256(master),
        annotation_sha256=panel['annotation_sha256'],residual_basis_sha256=array_sha256(residual),
        common_only=a.common_only,z_enabled=z_accumulator is not None,threads=a.threads,
        max_cached_missing_patterns=a.max_cached_missing_patterns,
        minimum_cached_pattern_rows=a.minimum_cached_pattern_rows,missing_cache_strategy=a.missing_cache_strategy,
        cache_addition_penalty_rows=a.cache_addition_penalty_rows,parallel_cache_products=a.parallel_cache_products,
        max_cached_pair_sums=a.max_cached_pair_sums,
        residual_workers=a.residual_workers,
        native_sha256=file_sha256(shared_tn._module.__file__) if shared_tn is not None else None)
    previous=(restore_study_checkpoint(a.resume_from,batch=batch,z=z_accumulator,identity=identity)
              if a.resume_from is not None else {})
    raw_n=sum(1 for _ in open(prefix+'.fam'));raw=np.empty((a.width,len(rows)),dtype=np.int8)
    timings=list(previous.get('timings',[]));z_telemetry=list(previous.get('z_telemetry',[]))
    traversal=time.monotonic();visits=previous.get('next_variant',0);segment_begin=visits;new_tiles=0
    prior_calls=previous.get('protected_tn_calls',0);prior_repairs=previous.get('repaired_columns',0)
    def progress():
        segments=list(previous.get('segments',[]))
        if visits>segment_begin:
            segments.append(dict(begin=segment_begin,end=visits,hostname=os.uname().nodename,
                job_id=os.environ.get('JOB_ID'),cpu_ids=list(CPUS),seconds=time.monotonic()-start,
                max_run_seconds=a.max_run_seconds,checkpoint_every_blocks=a.checkpoint_every_blocks))
        return dict(next_variant=visits,score_variants=batch.scores.variants,
            z_variants=z_accumulator.variants if z_accumulator is not None else 0,
            score_seconds=batch.scores.seconds,residual_seconds=batch.residual_seconds,
            residual_phase_seconds=batch.residual_phase_seconds,timings=timings,z_telemetry=z_telemetry,
            protected_tn_calls=prior_calls+(shared_tn.calls if shared_tn is not None else 0),
            repaired_columns=prior_repairs+(shared_tn.repaired_columns if shared_tn is not None else 0),
            seconds=previous.get('seconds',0)+time.monotonic()-start,
            traversal_seconds=previous.get('traversal_seconds',0)+time.monotonic()-traversal,segments=segments,
            peak_rss_kib=max(previous.get('peak_rss_kib',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    def checkpoint():
        path=a.output/f'checkpoint_{visits:012d}.npz'
        save_study_checkpoint(path,batch=batch,z=z_accumulator,identity=identity,progress=progress())
        print(json.dumps(dict(phase='checkpoint',next_variant=visits,path=str(path),sha256=file_sha256(path))),flush=True)
        return path
    with batch,threadpool_limits(limits=a.threads),pgenlib.PgenReader(os.fsencode(prefix+'.bed'),raw_sample_ct=raw_n,
            variant_ct=panel.get('bed_variant_count',m),sample_subset=rows.astype(np.uint32)) as reader:
        for begin in range(visits,limit,a.width):
            end=min(begin+a.width,limit);width=end-begin;tick=time.monotonic()
            offset=panel.get('physical_start',0);reader.read_range(begin+offset,end+offset,raw[:width],allele_idx=1)
            x=raw[:width].astype(float);missing=x==-9;x-=mean[begin:end,None];x*=inverse[begin:end,None];x[missing]=0
            # Study mode does not run a separate within-only timing pass.
            # JSON null records that absence without invalidating canonical
            # provenance after a complete chromosome traversal.
            decode=time.monotonic()-tick;baseline=None
            tile_annotations=annotations[begin:end];tile_groups=groups[begin:end]
            if a.common_only and z_accumulator is None:
                active=np.any(tile_annotations>0,axis=1)
                x=x[active];tile_annotations=tile_annotations[active];tile_groups=tile_groups[active]
            if a.mode=='benchmark':
                tick=time.monotonic()
                for _ in masked.block(x):pass
                baseline=time.monotonic()-tick
            score_before=batch.scores.seconds;residual_before=batch.residual_seconds;tick=time.monotonic()
            residual_phases_before=dict(batch.residual_phase_seconds)
            if len(x):
                callback=(None if z_accumulator is None else
                    lambda z:z_accumulator.add(z,full_annotations[begin:end],groups[begin:end]))
                for _ in batch.block(x,tile_annotations,tile_groups,z_callback=callback,shared_tn_operator=shared_tn):pass
                if shared_tn is not None:z_telemetry.append(canonical_sha256(shared_tn.finish_execution()))
            total=time.monotonic()-tick;visits+=width
            row=dict(begin=begin,end=end,decode_seconds=decode,within_seconds=baseline,cross_total_seconds=total,
                score_seconds=batch.scores.seconds-score_before,residual_seconds=batch.residual_seconds-residual_before,
                residual_phase_seconds={key:value-residual_phases_before[key] for key,value in batch.residual_phase_seconds.items()})
            timings.append(row)
            new_tiles+=1
            if a.mode=='benchmark' or len(timings)%32==0:print(json.dumps(row),flush=True)
            stop=(a.stop_after_blocks and new_tiles>=a.stop_after_blocks
                  or a.max_run_seconds and time.monotonic()-start>=a.max_run_seconds)
            if checkpointing and (visits==limit or stop or
                    a.checkpoint_every_blocks and len(timings)%a.checkpoint_every_blocks==0):
                path=checkpoint()
                if stop and visits<limit:
                    print(json.dumps(dict(phase='paused',next_variant=visits,resume_from=str(path))),flush=True)
                    raise SystemExit(75)
    final_progress=progress()
    provenance=dict(chromosome=a.chromosome,traits=list(names),input_sha256=input_hashes,
        reference_files=record['files'],reference_manifest_sha256=file_sha256(refroot/'MANIFEST.json'),
        source_sha256=source_hashes,variant_visits=visits,genotype_traversals=1,
        genotype_scale='sealed_affine_mean_imputed',common_only=a.common_only,threads=a.threads,cpus=CPUS,
        max_cached_missing_patterns=a.max_cached_missing_patterns,
        minimum_cached_pattern_rows=a.minimum_cached_pattern_rows,
        missing_cache_strategy=a.missing_cache_strategy,
        cache_addition_penalty_rows=a.cache_addition_penalty_rows,
        parallel_cache_products=a.parallel_cache_products,
        max_cached_pair_sums=a.max_cached_pair_sums,
        residual_workers=a.residual_workers,residual_worker_affinity=batch.residual_worker_affinity,
        residual_phase_timing='sum of worker elapsed times; residual_seconds measures wall time',
        timings=timings,seconds=final_progress['seconds'],traversal_seconds=final_progress['traversal_seconds'],
        process_segments=final_progress['segments'],
        resumed_checkpoint_sha256=file_sha256(a.resume_from) if a.resume_from is not None else None,
        resumed_checkpoint=str(a.resume_from.resolve()) if a.resume_from is not None else None,
        peak_rss_kib=final_progress['peak_rss_kib'],
        residual_basis_sha256=array_sha256(residual),residual_names=list(residual_names))
    if a.mode=='benchmark':
        measured=timings[1:] or timings
        within=sum(t['within_seconds'] for t in measured);full=sum(t['cross_total_seconds'] for t in measured)
        score=sum(t['score_seconds'] for t in measured)
        provenance.update(seconds_per_128_snp_block=full/len(measured)*128/a.width,
            within_seconds_per_128_snp_block=within/len(measured)*128/a.width,
            score_fraction_of_cross=score/full,score_overhead_fraction_of_within=score/within,
            total_increment_fraction=(full-within)/within,warmup_blocks=int(len(timings)>1))
    else:
        if visits!=m:raise RuntimeError('incomplete chromosome traversal')
        batch.write(a.output/'cross_trait_summary.npz',provenance=provenance)
    if z_accumulator is not None:
        if z_accumulator.variants!=limit:raise RuntimeError('fused Z visit ledger differs')
        z_accumulator.write(a.z_output,provenance=dict(provenance,master_input_sha256=file_sha256(master),
            manifest_sha256=file_sha256(refroot/'MANIFEST.json'),annotation_sha256=array_sha256(full_annotations),
            native_build=native_build,telemetry_sha256=z_telemetry,
            execution_ledger=dict(observed_genotype_passes=1,retained_variant_visits=limit,
                protected_tn_calls=final_progress['protected_tn_calls'],repaired_columns=final_progress['repaired_columns']),
            z_source='guarded shared fixed_weights contraction; no additional genotype product or pass'))
    with (a.output/'COMPLETE.json').open('x') as f:json.dump(provenance,f,indent=2)


if __name__=='__main__':main()
