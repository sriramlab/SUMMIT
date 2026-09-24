"""Run the actual fused study driver, including canonical artifact publication."""
import importlib.util
import json
from pathlib import Path
import sys
import os
import subprocess
import numpy as np
import pytest
from bed_reader import to_bed

from summit.context.cross_trait_zpass import load_array_artifact,write_array_artifact,ZMomentAccumulator
from summit.context.reference_zpass_cli import file_sha256
from summit.context.spec import array_sha256


def test_invalid_provenance_does_not_create_an_empty_artifact(tmp_path):
    path=tmp_path/'invalid.npz'
    with pytest.raises(ValueError,match='canonical JSON'):
        write_array_artifact(path,kind='test',arrays={'x':np.arange(3)},provenance={'timing':np.nan})
    assert not path.exists()


@pytest.mark.parametrize('resume',[False,True])
def test_fused_study_driver_publishes_complete_summary_and_z(tmp_path,monkeypatch,resume):
    from summit import gxeldcore
    script=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_study.py'
    spec=importlib.util.spec_from_file_location('study_publication',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(14886593);n,m,q,c=73,53,3,4
    phi=np.c_[np.ones(n),rng.normal(size=(n,q-1))]
    u=np.linalg.qr(np.c_[phi,rng.normal(size=n)])[0]
    raw=rng.integers(0,3,size=(n,m)).astype(float);raw[2,1]=np.nan
    bed=tmp_path/'geno';to_bed(str(bed)+'.bed',raw)
    mean=np.nanmean(raw,axis=0);g=np.nan_to_num(raw-mean)
    inverse=1/np.sqrt(np.sum(g*g,axis=0)/(n-1));g*=inverse
    annotations=np.eye(3)[np.arange(m)%3];annotation_path=tmp_path/'annotations.npy';np.save(annotation_path,annotations)
    inputs=tmp_path/'full_cohort_inputs_20260916';inputs.mkdir()
    root=tmp_path/'shared_reference_full_20260916';root.mkdir();chrom=root/'chr22';chrom.mkdir()
    expanded=tmp_path/'imputed_expansion_20260917';expanded.mkdir()
    (expanded/'MANIFEST.json').write_text(json.dumps(dict(traits=[])))
    sources={};traits=[]
    for t,name in enumerate(module.PILOT):
        rows=np.arange(n) if name=='height_raw' else np.setdiff1d(np.arange(n),np.arange(t,t+5))
        fixed=np.linalg.qr(u[rows])[0];y=rng.normal(size=len(rows))
        values=dict(rows=rows.astype(np.uint32),fixed=fixed,y=y)
        if name=='height_raw':values.update(phi=phi,fixed=u,basis_names=np.array(['intercept','age','bmi_raw']))
        path=inputs/f'{name}.npz';np.savez(path,**values);sources[name]=file_sha256(path)
        traits.append((rows,values['fixed'],y))
    panel=dict(chromosome=22,m=m,global_start=0,bed_size=Path(str(bed)+'.bed').stat().st_size,
        bim_sha256=file_sha256(str(bed)+'.bim'),fam_sha256=file_sha256(str(bed)+'.fam'),
        annotation_sha256=array_sha256(annotations))
    manifest=root/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(panels=[panel],traits=list(module.PILOT),trait_sources=sources,
        annotation_names=['rare','low','common'],njack=4,variants=m)))
    np.savez(chrom/'reference_aux.npz',affine_mean=mean,affine_inverse_scale=inverse)
    (chrom/'reference.npz').write_bytes(b'authenticated payload; study reads only affine auxiliary arrays')
    (chrom/'REFERENCE_COMPLETE.json').write_text(json.dumps(dict(passed=True,manifest_sha256=file_sha256(manifest),
        files={name:file_sha256(chrom/name) for name in ('reference.npz','reference_aux.npz')})))
    output=tmp_path/'study';zpath=tmp_path/'z.npz';threads=int(gxeldcore.build_info()['blas_runtime_threads'])
    command=[str(script),'study','--base',str(tmp_path),'--bed-prefix',str(bed),
        '--annotations',str(annotation_path),'--output',str(output),'--z-output',str(zpath),
        '--traits','8','--common-only','--threads',str(threads),'--width','7']
    if resume:
        # Separate OS processes: no in-memory accumulator/native state can
        # survive the exit-75 boundary. Restore only the authenticated NPZ.
        native=gxeldcore.build_info()
        launch=([sys.executable,str(script.with_name('private_python.py')),'cross_trait_study']
                if native.get('private_blas_backend')=='upstream_blis' else [sys.executable,str(script)])
        launch=['taskset','-c',','.join(map(str,module.CPUS)),*launch]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        first=subprocess.run([*launch,*command[1:],'--stop-after-blocks','2'],env=env,
            capture_output=True,text=True)
        assert first.returncode==75,first.stdout+first.stderr
        checkpoint=output/'checkpoint_000000000014.npz'
        assert checkpoint.exists() and not (output/'COMPLETE.json').exists() and not zpath.exists()
        digest=file_sha256(checkpoint)
        wrong=command.copy();wrong[wrong.index('--output')+1]=str(tmp_path/'wrong_width')
        rejected=subprocess.run([*launch,*wrong[1:],'--resume-from',str(checkpoint),'--width','8'],
            env=env,capture_output=True,text=True)
        assert rejected.returncode!=0 and 'checkpoint study identity differs' in rejected.stderr
        second=subprocess.run([*launch,*command[1:],'--resume-from',str(checkpoint)],env=env,
            capture_output=True,text=True)
        assert second.returncode==0,second.stdout+second.stderr
        assert file_sha256(checkpoint)==digest
        # Independent uninterrupted run, with the identical tile grouping.
        full=command.copy();full[full.index('--output')+1]=str(tmp_path/'uninterrupted')
        full[full.index('--z-output')+1]=str(tmp_path/'uninterrupted_z.npz')
        monkeypatch.setattr(sys,'argv',full);module.main()
        resumed,_=load_array_artifact(output/'cross_trait_summary.npz',kind='summit.cross_trait.summary')
        whole,_=load_array_artifact(tmp_path/'uninterrupted/cross_trait_summary.npz',kind='summit.cross_trait.summary')
        for key in whole:np.testing.assert_array_equal(resumed[key],whole[key])
        rz,_=load_array_artifact(zpath,kind='summit.cross_trait.z_moments')
        wz,_=load_array_artifact(tmp_path/'uninterrupted_z.npz',kind='summit.cross_trait.z_moments')
        for key in wz:np.testing.assert_array_equal(rz[key],wz[key])
    else:
        monkeypatch.setattr(sys,'argv',command);module.main()
    data,provenance=load_array_artifact(output/'cross_trait_summary.npz',kind='summit.cross_trait.summary')
    z,zmeta=load_array_artifact(zpath,kind='summit.cross_trait.z_moments')
    assert provenance['variant_visits']==m and provenance['genotype_traversals']==1
    assert all(t['within_seconds'] is None for t in provenance['timings'])
    assert json.loads((output/'COMPLETE.json').read_text())['variant_visits']==m
    assert zmeta['execution_ledger']['retained_variant_visits']==m
    assert zmeta['execution_ledger']['protected_tn_calls']==(m+6)//7
    assert [(s['begin'],s['end']) for s in provenance['process_segments']]==([(0,14),(14,m)] if resume else [(0,m)])
    groups=np.arange(m)*4//m;expected=ZMomentAccumulator(np.unique(groups),3,q,c)
    expected.add(np.einsum('nj,na,nc->jac',g,phi,u),annotations,groups)
    np.testing.assert_allclose(z['block_products'],expected.products,rtol=1e-11,atol=1e-10)
    scores=[]
    for rows,fixed,y in traits[:2]:
        y=y-fixed@(fixed.T@y);y*=np.sqrt((len(rows)-fixed.shape[1])/(y@y))
        scores.append(g[rows].T@(phi[rows]*y[:,None]))
    pair=np.flatnonzero(np.all(data['pairs']==[0,1],axis=1)).item()
    np.testing.assert_allclose(data['block_rhs'][:,pair,0].sum(0),
        scores[0].T@(annotations[:,2,None]*scores[1]),rtol=1e-11,atol=1e-10)
    qualification=script.with_name('cross_trait_qualification.py')
    spec=importlib.util.spec_from_file_location('study_qualification',qualification)
    check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
    assert check.validate(output,zpath,manifest)['qualified']
    complete=json.loads((output/'COMPLETE.json').read_text());complete['variant_visits']-=1
    (output/'COMPLETE.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError,match='provenance differs'):
        check.validate(output,zpath,manifest)
