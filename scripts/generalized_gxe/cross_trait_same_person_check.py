"""Measure block-mass apportionment error in the LD scalar on sealed chr22.

This validation traversal computes only the exact baseline target diagonals.
It neither changes the reference nor participates in the shared trait pass.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
import pgenlib
from summit.context.reference_zpass_cli import authenticated_reference,file_sha256
from summit.context.spec import array_sha256,canonical_sha256
from summit.ldscore.generalized_gxe_chromosome import load_chromosome_moments
from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
from cross_trait_refit import write_table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','bed-prefix','annotations','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--width',type=int,default=128)
    args=p.parse_args();args.output.mkdir(exist_ok=False);start=time.monotonic()
    master=args.base/'full_cohort_inputs_20260916/height_raw.npz';root=args.base/'shared_reference_full_20260916'
    manifest,panel,r,record=authenticated_reference(root/'MANIFEST.json',root,22,master)
    ref,_=load_chromosome_moments(r/'reference.npz');q=ref.num_basis;pairs=q*(q+1)//2
    with np.load(master) as z:rows,u=z['rows'],z['fixed']
    with np.load(r/'reference_aux.npz') as z:
        mean,inv,diag=z['affine_mean'],z['affine_inverse_scale'],z['global_kernel_diagonal'][::pairs]
    a=np.load(args.annotations);assert array_sha256(a)==panel['annotation_sha256']
    prefix=str(args.bed_prefix)
    for suffix,key in (('.bim','bim_sha256'),('.fam','fam_sha256')):
        assert file_sha256(prefix+suffix)==panel[key]
    assert Path(prefix+'.bed').stat().st_size==panel['bed_size']
    masses=np.asarray(manifest['global_masses']);groups=(np.arange(panel['m'])+panel['global_start'])*manifest['njack']//manifest['variants']
    exact=np.zeros((len(ref.block_ids),len(masses),len(masses)))
    tn=ProtectedTNOperator(threads=args.threads);nn=ProtectedNNOperator(threads=args.threads)
    tn.begin_execution();nn.begin_execution();telemetry=[]
    u=np.asfortranarray(u);ut=np.asfortranarray(u.T)
    info=dict(tn._module.build_info());assert info['gemm_integrity_enabled'] and info['gemm_checksum_enabled']
    raw=np.empty((args.width,len(rows)),dtype=np.int8);raw_n=sum(1 for _ in open(prefix+'.fam'))
    with pgenlib.PgenReader(os.fsencode(prefix+'.bed'),raw_sample_ct=raw_n,variant_ct=panel['m'],
            sample_subset=rows.astype(np.uint32)) as reader:
        for begin in range(0,panel['m'],args.width):
            end=min(begin+args.width,panel['m']);w=end-begin
            reader.read_range(begin,end,raw[:w],allele_idx=1)
            g=raw[:w].astype(float);missing=g==-9;g-=mean[begin:end,None];g*=inv[begin:end,None];g[missing]=0
            z=tn.matmul_tn(np.asfortranarray(g.T),u)
            g-=nn.matmul(np.asfortranarray(z),ut)
            g*=g;products=g@diag.T
            labels=groups[begin:end]
            for label in np.unique(labels):
                take=labels==label;exact[np.searchsorted(ref.block_ids,label)]+=(a[begin:end][take].T@products[take])/masses[:,None]
            telemetry.append(canonical_sha256(dict(tn=tn.finish_execution(),nn=nn.finish_execution())))
            if begin//args.width%100==0 or end==panel['m']:
                print(json.dumps(dict(variants=end,seconds=time.monotonic()-start)),flush=True)
    same=diag@diag.T
    np.testing.assert_allclose(exact.sum(0),same,rtol=1e-9,atol=1e-10)
    fractions=ref.block_masses/ref.block_masses.sum(0)
    apportioned=fractions[:,:,None]*same
    total=ref.block_directed[:,::pairs,::pairs]*ref.residual_rank**2/masses[None,:,None]/masses[None,None,:]
    error=np.abs(exact-apportioned)/np.abs(total-exact)
    table=[dict(block=int(ref.block_ids[b]),target_annotation=ref.annotation_names[k],source_annotation=ref.annotation_names[l],
        exact_same_person=exact[b,k,l],apportioned_same_person=apportioned[b,k,l],
        relative_ld_scalar_error=error[b,k,l],same_person_share=exact[b,k,l]/total[b,k,l])
        for b,k,l in np.ndindex(exact.shape)]
    write_table(args.output/'same_person_apportionment.tsv',table)
    with (args.output/'COMPLETE.json').open('x') as f:
        json.dump(dict(maximum_relative_ld_scalar_error=float(error.max()),passes_1e_minus_3=bool(error.max()<1e-3),
            seconds=time.monotonic()-start,reference_files=record['files'],script_sha256=file_sha256(__file__),
            telemetry_sha256=telemetry,genotype_traversals=1,variant_visits=panel['m'],
            table_sha256=file_sha256(args.output/'same_person_apportionment.tsv')),f,indent=2)


if __name__=='__main__':main()
