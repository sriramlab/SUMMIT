import json
import numpy as np
from bed_reader import to_bed

from summit.context.reference_zpass_cli import run_zpass,file_sha256
from summit.context.cross_trait_zpass import load_array_artifact,ZMomentAccumulator
from summit.context.spec import array_sha256


def test_native_zpass_bed_scaling_ledger_and_hashes(tmp_path):
    from summit import gxeldcore
    rng=np.random.default_rng(29);n,m,q,c=27,23,3,4
    raw=rng.integers(0,3,size=(n,m)).astype(float);raw[2,1]=np.nan
    prefix=tmp_path/'geno';to_bed(str(prefix)+'.bed',raw)
    rows=np.arange(n,dtype=np.uint32);phi=np.c_[np.ones(n),rng.normal(size=(n,q-1))]
    u=np.linalg.qr(np.c_[phi,rng.normal(size=n)])[0]
    master=tmp_path/'master.npz';np.savez(master,rows=rows,phi=phi,fixed=u)
    a=rng.uniform(size=(m,2));annotations=tmp_path/'annotations.npy';np.save(annotations,a)
    mean=np.nanmean(raw,axis=0);x=np.nan_to_num(raw-mean);inverse=1/np.sqrt(np.sum(x*x,axis=0)/(n-1))
    root=tmp_path/'ref';root.mkdir();chrom=root/'chr22';chrom.mkdir()
    np.savez(chrom/'reference_aux.npz',affine_mean=mean,affine_inverse_scale=inverse)
    (chrom/'reference.npz').write_bytes(b'authenticated payload for Z-only test')
    panel=dict(chromosome=22,m=m,global_start=0,bed_size=(tmp_path/'geno.bed').stat().st_size,
        bim_sha256=file_sha256(str(prefix)+'.bim'),fam_sha256=file_sha256(str(prefix)+'.fam'),annotation_sha256=array_sha256(a))
    manifest=root/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(panels=[panel],trait_sources={'height_raw':file_sha256(master)},
        annotation_names=['a','b'],njack=4,variants=m)))
    (chrom/'REFERENCE_COMPLETE.json').write_text(json.dumps(dict(passed=True,manifest_sha256=file_sha256(manifest),
        files={name:file_sha256(chrom/name) for name in ('reference.npz','reference_aux.npz')})))
    output=tmp_path/'z.npz'
    threads=int(gxeldcore.build_info()['blas_runtime_threads'])
    run_zpass(manifest_path=manifest,reference_root=root,chromosome=22,master_input=master,
        bed_prefix=prefix,annotations_path=annotations,output=output,threads=threads,width=7)
    arrays,provenance=load_array_artifact(output,kind='summit.cross_trait.z_moments')
    groups=np.arange(m)*4//m
    expected=ZMomentAccumulator(np.unique(groups),2,q,c)
    expected.add(np.einsum('nj,na,nc->jac',x*inverse,phi,u),a,groups)
    np.testing.assert_allclose(arrays['block_products'],expected.products,atol=1e-10,rtol=1e-11)
    assert provenance['execution_ledger']['observed_genotype_passes']==1
    assert provenance['execution_ledger']['retained_variant_visits']==m
    assert provenance['execution_ledger']['protected_tn_calls']==4
