"""Read-only sealed-reference Z pass; publish a separate checksummed artifact."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .spec import array_sha256,canonical_sha256
from .cross_trait_zpass import ZMomentAccumulator
from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*2**20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def authenticated_reference(manifest_path,reference_root,chromosome,master_input):
    """Validate sealed files before any genotype traversal."""
    manifest_path=Path(manifest_path);root=Path(reference_root)/f'chr{chromosome}'
    manifest=json.loads(manifest_path.read_text())
    record=json.loads((root/'REFERENCE_COMPLETE.json').read_text())
    if not record['passed'] or record['manifest_sha256']!=file_sha256(manifest_path):
        raise ValueError('reference completion does not authenticate this manifest')
    for name,checksum in record['files'].items():
        if file_sha256(root/name)!=checksum:
            raise ValueError(f'reference checksum differs: {name}')
    if file_sha256(master_input)!=manifest['trait_sources']['height_raw']:
        raise ValueError('master cohort checksum differs')
    panel=next(p for p in manifest['panels'] if int(p['chromosome'])==chromosome)
    return manifest,panel,root,record


def run_zpass(*,manifest_path,reference_root,chromosome,master_input,bed_prefix,
              annotations_path,output,threads=1,width=128,native_module=None):
    import pgenlib
    if Path(output).exists():
        raise FileExistsError(output)
    if min(threads,width)<1:
        raise ValueError('positive thread count and tile width required')
    # Same process-local allocation guard as the research launchers.
    if os.name=='posix' and hasattr(os,'sched_getaffinity'):
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(41,1,0,0,0)!=0 or libc.prctl(42,0,0,0,0)!=1:
            raise RuntimeError('process-local THP guard failed')
    start=time.monotonic()
    manifest,panel,root,record=authenticated_reference(manifest_path,reference_root,chromosome,master_input)
    prefix=str(bed_prefix)
    for suffix,key in (('.bim','bim_sha256'),('.fam','fam_sha256')):
        if file_sha256(prefix+suffix)!=panel[key]:
            raise ValueError(f'panel checksum differs: {suffix}')
    if Path(prefix+'.bed').stat().st_size!=panel['bed_size']:
        raise ValueError('BED size differs')
    a=np.load(annotations_path,allow_pickle=False)
    if array_sha256(a)!=panel['annotation_sha256']:
        raise ValueError('annotation checksum differs')
    with np.load(master_input,allow_pickle=False) as z:
        rows,phi,u=(z[name] for name in ('rows','phi','fixed'))
    with np.load(root/'reference_aux.npz',allow_pickle=False) as z:
        mean,inverse=z['affine_mean'],z['affine_inverse_scale']
    if (a.shape!=(panel['m'],len(manifest['annotation_names'])) or len(mean)!=len(a)
            or len(inverse)!=len(a) or not np.allclose(u.T@u,np.eye(u.shape[1]),rtol=0,atol=1e-10)):
        raise ValueError('sealed dimensions or fixed basis disagree')
    n,q=phi.shape;c=u.shape[1];m=len(a)
    rhs=np.asfortranarray((phi[:,:,None]*u[:,None,:]).reshape(n,q*c))
    groups=(np.arange(m)+panel['global_start'])*manifest['njack']//manifest['variants']
    accumulator=ZMomentAccumulator(np.unique(groups),a.shape[1],q,c)
    operator=ProtectedTNOperator(threads=threads,native_module=native_module)
    build=dict(operator._module.build_info())
    if not build.get('gemm_integrity_enabled') or not build.get('gemm_checksum_enabled'):
        raise RuntimeError('Z pass requires both native GEMM integrity and checksum guards')
    raw_n=sum(1 for _ in open(prefix+'.fam'))
    raw=np.empty((width,n),dtype=np.int8)
    operator.begin_execution();telemetry_hashes=[];tn_seconds=0.;reduce_seconds=0.
    traversal=time.monotonic();blocks=0;offset=panel.get('physical_start',0)
    with pgenlib.PgenReader(os.fsencode(prefix+'.bed'),raw_sample_ct=raw_n,
            variant_ct=panel.get('bed_variant_count',m),sample_subset=rows.astype(np.uint32)) as reader:
        for begin in range(0,m,width):
            end=min(begin+width,m);w=end-begin
            reader.read_range(begin+offset,end+offset,raw[:w],allele_idx=1)
            x=raw[:w].astype(float);missing=x==-9
            x-=mean[begin:end,None];x*=inverse[begin:end,None];x[missing]=0
            tick=time.monotonic()
            z=operator.matmul_tn(np.asfortranarray(x.T),rhs).reshape(w,q,c)
            tn_seconds+=time.monotonic()-tick;tick=time.monotonic()
            accumulator.add(z,a[begin:end],groups[begin:end])
            reduce_seconds+=time.monotonic()-tick;blocks+=1
            # Consume the in-house telemetry before its bounded ring can fill.
            evidence=operator.finish_execution()
            telemetry_hashes.append(canonical_sha256(evidence))
            if blocks%100==0 or end==m:
                print(json.dumps(dict(phase='zpass',chromosome=chromosome,variants=end,total=m,
                    seconds=time.monotonic()-traversal)),flush=True)
    if accumulator.variants!=m or operator.calls!=blocks:
        raise RuntimeError('one-pass Z traversal ledger is incomplete')
    provenance=dict(chromosome=chromosome,manifest_sha256=file_sha256(manifest_path),
        reference_files=record['files'],master_input_sha256=file_sha256(master_input),
        annotation_sha256=array_sha256(a),native_build=build,threads=threads,width=width,
        execution_ledger=dict(observed_genotype_passes=1,retained_variant_visits=m,
                              protected_tn_calls=operator.calls,repaired_columns=operator.repaired_columns),
        telemetry_sha256=telemetry_hashes,seconds=time.monotonic()-start,
        traversal_seconds=time.monotonic()-traversal,tn_seconds=tn_seconds,reduction_seconds=reduce_seconds,
        rows_sha256=array_sha256(rows),basis_sha256=array_sha256(phi),fixed_sha256=array_sha256(u))
    accumulator.write(output,provenance=provenance)
    return provenance


def main(argv=None):
    parser=argparse.ArgumentParser(prog='summit reference zpass',description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--chromosome',type=int,required=True)
    parser.add_argument('--master-input',type=Path,required=True)
    parser.add_argument('--bed-prefix',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-threads',type=int,default=1)
    parser.add_argument('--width',type=int,default=128)
    args=parser.parse_args(argv)
    result=run_zpass(manifest_path=args.manifest,reference_root=args.reference_root,
        chromosome=args.chromosome,master_input=args.master_input,bed_prefix=args.bed_prefix,
        annotations_path=args.annotations,output=args.output,threads=args.num_threads,width=args.width)
    print(json.dumps(dict(output=str(args.output),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    main()
