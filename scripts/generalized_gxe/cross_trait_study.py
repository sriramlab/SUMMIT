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
from summit.context.reference_zpass_cli import authenticated_reference,file_sha256
from summit.context.spec import array_sha256
from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
from summit.ldscore.generalized_gxe_cross_trait_batch import CrossTraitBatch

PILOT=('ldl_raw','apo_b_raw','cholesterol_raw','non_hdl_cholesterol_raw','hba1c_raw',
       'diastolic_blood_pressure_raw','height_raw','platelet_count_raw')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['benchmark','study'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bed-prefix',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--chromosome',type=int,default=22);p.add_argument('--width',type=int,default=128)
    p.add_argument('--variants',type=int,default=256);p.add_argument('--traits',type=int,choices=[6,8,42],default=8)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--common-only',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=False);start=time.monotonic()
    refroot=a.base/'shared_reference_full_20260916';master=a.base/'full_cohort_inputs_20260916/height_raw.npz'
    manifest,panel,reference,record=authenticated_reference(refroot/'MANIFEST.json',refroot,a.chromosome,master)
    expansion=a.base/'imputed_expansion_20260917';expanded=json.loads((expansion/'MANIFEST.json').read_text())
    inputroots={t:(a.base/'full_cohort_inputs_20260916',manifest) for t in manifest['traits']}
    inputroots.update({t:(expansion/'inputs',expanded) for t in expanded['traits'] if t!='height_raw'})
    names=sorted(inputroots) if a.traits==42 else PILOT[:a.traits]
    if len(names)!=a.traits:raise ValueError(f'expected {a.traits} input traits, found {len(names)}')
    with np.load(master) as z:rows,phi,fixed,basis_names=(z[k] for k in ('rows','phi','fixed','basis_names'))
    traits=[];input_hashes={}
    for name in names:
        inputroot,source_manifest=inputroots[name];path=inputroot/f'{name}.npz'
        digest=file_sha256(path)
        if digest!=source_manifest['trait_sources'][name]:raise ValueError(f'input checksum differs: {path}')
        input_hashes[str(path)]=digest
        with np.load(path) as z:
            idx=np.searchsorted(rows,z['rows']);np.testing.assert_array_equal(rows[idx],z['rows'])
            u=z['fixed'] if 'fixed' in z else fixed[idx]@z['fixed_from_master']
            traits.append(dict(name=name,indices=idx,fixed_basis=u,phenotype=z['y']))
    annotations=np.load(a.annotations,allow_pickle=False)
    if array_sha256(annotations)!=panel['annotation_sha256']:raise ValueError('annotation checksum differs')
    prefix=str(a.bed_prefix)
    for suffix,key in (('.bim','bim_sha256'),('.fam','fam_sha256')):
        if file_sha256(prefix+suffix)!=panel[key]:raise ValueError(f'panel checksum differs: {suffix}')
    if Path(prefix+'.bed').stat().st_size!=panel['bed_size']:raise ValueError('BED size differs')
    with np.load(reference/'reference_aux.npz') as z:mean,inverse=z['affine_mean'],z['affine_inverse_scale']
    annotation_names=manifest['annotation_names']
    if a.common_only:
        # The sealed annotations are ordered rare, low-frequency, common.
        annotation_names=[annotation_names[2]];annotations=annotations[:,2:3]
    m=panel['m'];limit=min(a.variants,m) if a.mode=='benchmark' else m
    groups=(np.arange(m)+panel['global_start'])*manifest['njack']//manifest['variants']
    with threadpool_limits(limits=1):
        residual,residual_names,_=workflow.rank_reduced_symmetric_context_residual_basis(phi,tuple(basis_names.astype(str)))
        masked=MaskedTraitBatch(basis=phi,fixed_basis=fixed,residual_basis=residual,traits=traits)
        batch=CrossTraitBatch(masked,block_ids=np.unique(groups[:limit]),annotation_names=annotation_names)
    print(json.dumps(dict(masked.report,phase='prepared',pairs=len(batch.scores.pairs),seconds=time.monotonic()-start,
                          cpus=CPUS,blas=threadpool_info())),flush=True)
    raw_n=sum(1 for _ in open(prefix+'.fam'));raw=np.empty((a.width,len(rows)),dtype=np.int8)
    timings=[];traversal=time.monotonic();visits=0
    with threadpool_limits(limits=a.threads),pgenlib.PgenReader(os.fsencode(prefix+'.bed'),raw_sample_ct=raw_n,
            variant_ct=panel.get('bed_variant_count',m),sample_subset=rows.astype(np.uint32)) as reader:
        for begin in range(0,limit,a.width):
            end=min(begin+a.width,limit);width=end-begin;tick=time.monotonic()
            offset=panel.get('physical_start',0);reader.read_range(begin+offset,end+offset,raw[:width],allele_idx=1)
            x=raw[:width].astype(float);missing=x==-9;x-=mean[begin:end,None];x*=inverse[begin:end,None];x[missing]=0
            decode=time.monotonic()-tick;baseline=np.nan
            if a.mode=='benchmark':
                tick=time.monotonic()
                for _ in masked.block(x):pass
                baseline=time.monotonic()-tick
            score_before=batch.scores.seconds;residual_before=batch.residual_seconds;tick=time.monotonic()
            for _ in batch.block(x,annotations[begin:end],groups[begin:end]):pass
            total=time.monotonic()-tick;visits+=width
            row=dict(begin=begin,end=end,decode_seconds=decode,within_seconds=baseline,cross_total_seconds=total,
                score_seconds=batch.scores.seconds-score_before,residual_seconds=batch.residual_seconds-residual_before)
            timings.append(row)
            if a.mode=='benchmark' or len(timings)%32==0:print(json.dumps(row),flush=True)
    provenance=dict(chromosome=a.chromosome,traits=list(names),input_sha256=input_hashes,
        reference_files=record['files'],reference_manifest_sha256=file_sha256(refroot/'MANIFEST.json'),
        script_sha256=file_sha256(__file__),variant_visits=visits,genotype_traversals=1,
        genotype_scale='sealed_affine_mean_imputed',common_only=a.common_only,threads=a.threads,cpus=CPUS,
        timings=timings,seconds=time.monotonic()-start,traversal_seconds=time.monotonic()-traversal,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
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
    with (a.output/'COMPLETE.json').open('x') as f:json.dump(provenance,f,indent=2)


if __name__=='__main__':main()
