"""Exercise the installed public conditional CLI from outside the checkout.

Run with the new environment's Python. No editable loader or native override is
used. This is a numerical/package check, not a calibration experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import runpy

runpy.run_path(str(Path(__file__).with_name("cpu_policy.py")))["require_numerical_startup"]()

if sys.platform.startswith('linux'):
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(41,1,0,0,0)!=0 or libc.prctl(42,0,0,0,0)!=1:
        raise RuntimeError('process-local THP guard failed')

# Capture the reservation before importing libgomp, which may bind the main
# thread to one OpenMP place. Each ordinary CLI child receives the full set.
LAUNCH_CPUS=tuple(sorted(os.sched_getaffinity(0))) if hasattr(os,'sched_getaffinity') else ()

import numpy as np
import pandas as pd
from bed_reader import to_bed
import summit
from summit import gxeldcore
from summit.epistasis import conditional_workflow,conditional_reference,krylov


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-threads',type=int,default=1)
    parser.add_argument('--moment-weighting',choices=('none','genotype_diagonal'),default='none')
    args=parser.parse_args();root=args.out.resolve();root.mkdir();root.chmod(0o700)
    if args.num_threads<1:raise ValueError('positive thread count required')
    prefix=Path(sys.prefix).resolve()
    for module in (summit,gxeldcore,conditional_workflow,conditional_reference,krylov):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError('check requires a clean installed package and native extensions')
    cli=prefix/'bin/summit'
    rng=np.random.default_rng(892714);n,m=2400,96
    raw=rng.binomial(2,.35,(n,m)).astype(float)
    raw[rng.random(raw.shape)<.002]=np.nan;raw[:,0]=rng.binomial(2,.35,n)
    ids=list(map(str,range(n)));names=['12:66358347']+[f'v{i}' for i in range(1,m)]
    to_bed(root/'g.bed',raw,properties=dict(fid=ids,iid=ids,sid=names,
        chromosome=['12']*8+['5']*(m-8),bp_position=[66358347+i for i in range(m)],
        allele_1=['A']*m,allele_2=['C']*m))
    g=np.nan_to_num(raw-.7);z=rng.normal(size=n)
    y=.2*g[:,2]+.4*g[:,0]*g[:,8]+.3*z+rng.normal(size=n)
    pd.DataFrame(dict(FID=ids,IID=ids,PC1=z)).to_csv(root/'cov.tsv',sep='\t',index=False)
    pd.DataFrame(dict(FID=ids,IID=ids,y=y,twice_y=2*y)).to_csv(root/'phen.tsv',sep='\t',index=False)
    for label,rows in [('train',range(1200)),('confirm',range(1200,n))]:
        pd.DataFrame(dict(FID=[ids[i] for i in rows],IID=[ids[i] for i in rows])).to_csv(root/(label+'.tsv'),sep='\t',index=False)
    recipe=dict(kind='summit.epistasis.trans_inputs',schema_version=1,
        genotypes=dict(geno='g.bed',genome_build='GRCh37',content_identity=True),
        target=names[0],background_chromosome='5',training_samples='train.tsv',confirmation_samples='confirm.tsv',
        phenotype=dict(file='phen.tsv',column='y',unit='simulation units'),
        covariates=dict(file='cov.tsv',columns=['PC1'],varying_effects=['PC1']),
        inference=dict(method='conditional_polygenic_mean',moment_weighting=args.moment_weighting))
    private=gxeldcore.build_info().get('private_blas_backend')=='upstream_blis'
    physical=[]
    if private:
        if len(LAUNCH_CPUS)!=args.num_threads:
            raise ValueError('private installed check requires exactly the reserved worker CPUs')
        for cpu in LAUNCH_CPUS:
            topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            physical.append(tuple((topology/name).read_text().strip() for name in ('physical_package_id','core_id')))
        if len(set(physical))!=len(physical):raise ValueError('private workers require distinct physical cores')
    threads=str(args.num_threads)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS=threads,OMP_NUM_THREADS=threads,BLIS_NUM_THREADS=threads)
    for key in ('PYTHONPATH','SUMMIT_NATIVE_DIR','SUMMIT_PRIVATE_NATIVE_DIR'):environment.pop(key,None)
    records=[]
    def run(*args,expected_error=None):
        command=[str(cli),'epistasis',*map(str,args)]
        if LAUNCH_CPUS:command=['taskset','-c',','.join(map(str,LAUNCH_CPUS)),*command]
        begin=time.perf_counter();result=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
        records.append(dict(command=command,seconds=time.perf_counter()-begin,
            returncode=result.returncode,expected_error=expected_error))
        if expected_error is not None:
            if not result.returncode or expected_error not in result.stdout+result.stderr:
                raise RuntimeError('expected early input rejection: '+result.stdout+result.stderr)
        elif result.returncode:raise RuntimeError(result.stdout+result.stderr)
    for label,column in [('first','y'),('second','twice_y')]:
        recipe['phenotype']['column']=column
        (root/(label+'.json')).write_text(json.dumps(recipe))
        run('make-inputs',label+'.json','--out',label,'--num-threads',threads)
        inputs=json.loads((root/label/'inputs.json').read_text())
        scaling=inputs['covariate_scaling']
        assert scaling['preserved_origins']==['PC1'] and scaling['center']==[0.]
        made=pd.read_csv(root/label/'covariates.tsv',sep='\t')
        np.testing.assert_allclose(made.PC1.to_numpy()*scaling['scale'][0],z,rtol=2e-13,atol=2e-13)
        run('train-direction',label+'/train.json','--out',label+'/trained','--num-threads',threads,'--memory-gib','2')
        if label=='first':
            old=json.loads((root/label/'trained/direction.json').read_text())
            old.pop('additive_model_identity')
            (root/label/'legacy-direction.json').write_text(json.dumps(old))
            invalid_train=json.loads((root/label/'train.json').read_text())
            invalid_train['genotypes']['geno']='deliberately-unavailable.bed'
            (root/label/'legacy-train.json').write_text(json.dumps(invalid_train))
            invalid=json.loads((root/label/'prepare.json').read_text())
            invalid['training']='legacy-train.json'
            invalid['directions'][0]['direction']='legacy-direction.json'
            (root/label/'legacy-prepare.json').write_text(json.dumps(invalid))
            run('prepare',label+'/legacy-prepare.json','--out',label+'/legacy-prepared',
                '--num-threads',threads,expected_error='requires matched interaction and additive-null models')
            assert not (root/label/'legacy-prepared').exists()
            run('prepare',label+'/prepare.json','--out',label+'/prepared','--num-threads',threads,'--memory-gib','2')
        else:
            reuse=json.loads((root/label/'prepare.json').read_text())
            reuse['kind']='summit.epistasis.prepare_traits'
            reuse['inference']['reference']='../first/prepared/cohort-reference.npz'
            (root/label/'reuse.json').write_text(json.dumps(reuse))
            run('prepare-traits',label+'/reuse.json','--out',label+'/prepared','--num-threads',threads,
                '--block-size','1024','--memory-gib','2')
            run('prepare-traits',label+'/reuse.json','--out',label+'/prepared','--num-threads',threads,
                '--block-size','1024','--memory-gib','2','--resume')
        run('fit',f'{label}/prepared/{column}.robust-score.npz','--out',label+'/fit.json')
    first=json.loads((root/'first/fit.json').read_text())['fits'][0]
    second=json.loads((root/'second/fit.json').read_text())['fits'][0]
    assert first['feature_names']==[names[0]+'_by_frozen_trans_score']
    # A refitted linear direction doubles with the outcome: F and Y both
    # double, so its response-normalized coefficient and SE stay invariant.
    np.testing.assert_allclose(second['beta'],first['beta'],rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(second['standard_errors'],first['standard_errors'],rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(second['joint_p'],first['joint_p'],rtol=2e-6,atol=1e-8)
    second_record=json.loads((root/'second/prepared/preparation.json').read_text())
    first_record=json.loads((root/'first/prepared/preparation.json').read_text())
    assert second_record['reference_reused']
    assert first_record['covariance_moment_weighting']==args.moment_weighting
    for field in ('sampling_relative_sd','trace_relative_sd','fixed_contrast_variance_relative_sd'):
        np.testing.assert_allclose(first['diagnostics']['covariance_precision'][field],
            second['diagnostics']['covariance_precision'][field],rtol=2e-6,atol=1e-10)
    np.testing.assert_allclose(second_record['covariance_components'],4*np.asarray(first_record['covariance_components']),rtol=2e-8,atol=1e-9)
    portable=root/'portable';portable.mkdir()
    shutil.copyfile(root/'first/prepared/y.robust-score.npz',portable/'scores.npz')
    run('fit','portable/scores.npz','--out','portable/fit.json')
    assert json.loads((portable/'fit.json').read_text())==json.loads((root/'first/fit.json').read_text())
    training_builds=[json.loads((root/label/'trained/models/manifest.json').read_text())['provenance']['native_build']
        for label in ('first','second')]
    if private:
        for build in training_builds:
            placement=build['openmp_placement_contract_evidence']
            assert build['blas_runtime_isolation']=='private_static'
            assert build['blas_runtime_threads']==args.num_threads
            assert build['blas_runtime_worker_affinity_policy']=='inherit_authenticated_selected_cpu_set_per_call'
            assert build['gemm_integrity_enabled'] and build['gemm_checksum_enabled']
            assert placement['verified'] and placement['immutable'] and placement['exact_team_coverage']
            assert placement['expected_cpu_ids']==list(LAUNCH_CPUS) and placement['team_size']==args.num_threads
    record=dict(package=summit.__file__,native=gxeldcore.__file__,native_build=gxeldcore.build_info(),
        training_native_builds=training_builds,launch_cpus=LAUNCH_CPUS,physical_cores=physical,
        threads=args.num_threads,private_placement_verified=private,
        commands=records,n=n,m=m,conditional_modules_in_package=True,portable_fit_identical=True,
        reference_reused=True,new_trait_covariance_reestimated=True,
        covariance_precision_portable_and_unit_invariant=True,moment_weighting=args.moment_weighting,
        supplied_random_slope_origin_preserved=True,
        legacy_learner_rejected_before_genotypes=True,
        portable_feature_identifies_target=True,
        estimator_units='learned raw-score units; refitting a doubled outcome doubles the score',
        scope='clean installed CLI and numerical reuse; no statistical qualification')
    (root/'VERIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(stages=len(records),seconds=sum(v['seconds'] for v in records),passed=True)))


if __name__=='__main__':main()
