"""Bounded subprocess checks for durable monitoring and scoped budget stops."""
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import zipfile

import pytest

from scripts.epistasis.run_measured import command_digest, execute, local_lease, local_plan


def test_local_budget_cannot_be_multiply_reserved(tmp_path):
    path=tmp_path/'local.lock'
    with local_lease(path):
        with pytest.raises(ValueError,match='aggregate budget'):
            with local_lease(path):
                pytest.fail('second reservation admitted')
    with local_lease(path):
        pass


def test_failed_reserve_preflight_does_not_launch(tmp_path):
    marker=tmp_path/'child_was_started'
    plan=dict(max_seconds=1,max_rss_bytes=2**20,max_output_growth_bytes=2**20,
        min_available_memory_bytes=2**80,min_free_disk_bytes=0)
    command=[sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).touch()',str(marker)]
    with pytest.raises(ValueError,match='preflight refused'):
        execute(command,tmp_path/'accounting.json',plan=plan)
    assert not marker.exists()


def test_local_launch_requires_exact_approved_bounded_plan(tmp_path):
    command=[sys.executable,'-c','pass']
    with pytest.raises(ValueError,match='disabled'):
        local_plan(None,command)
    plan=dict(approved=False,approval_reference='bounded test only',command_sha256=command_digest(command),
        output_root=str(tmp_path),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cpu_ids=[min(os.sched_getaffinity(0))],max_seconds=1,max_rss_bytes=128*2**20,
        max_output_growth_bytes=2**20,min_available_memory_bytes=256*2**30,min_free_disk_bytes=100*2**30)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='explicit approved'):
        local_plan(path,command)
    plan['approved']=True;path.write_text(json.dumps(plan))
    assert local_plan(path,command)==plan
    plan['min_available_memory_bytes']=128*2**30;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='256 GiB'):
        local_plan(path,command)
    plan['min_available_memory_bytes']=256*2**30;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='command differs'):
        local_plan(path,[sys.executable,'-c','print(1)'])
    plan['boot_id']='previous boot';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='previous boot'):
        local_plan(path,command)
    plan['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    plan['max_seconds']=7*3600;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='ceiling'):
        local_plan(path,command)
    plan['max_seconds']=1
    plan['work_roots']=[str(tmp_path)]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='disjoint'):
        local_plan(path,command)


def test_supervisor_stops_only_owned_group_and_preserves_durable_output(tmp_path):
    # This lightweight subprocess writes a complete archive, then waits. It
    # exercises process ownership/termination, not any numerical estimator.
    checkpoint=tmp_path/'solver.npz'
    script="""import os,sys,time,zipfile
from pathlib import Path
p=Path(sys.argv[1]);temporary=p.with_suffix('.tmp')
with temporary.open('wb') as f:
 with zipfile.ZipFile(f,'w') as z:z.writestr('payload','preserved checkpoint fixture')
 f.flush();os.fsync(f.fileno())
os.replace(temporary,p)
time.sleep(30)
"""
    # OpenMP can narrow only the main thread; retain the complete already
    # admitted test-worker allocation without changing any existing mask.
    from scripts.epistasis.cpu_policy import require_affinity
    allocated=sorted({cpu for mask in require_affinity().values() for cpu in mask})
    plan=dict(cpu_ids=allocated,max_seconds=.5,
        max_rss_bytes=128*2**20,max_output_growth_bytes=2**20,
        min_available_memory_bytes=0,min_free_disk_bytes=0)
    output=tmp_path/'accounting.json'
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        result=execute([sys.executable,'-c',script,str(checkpoint)],output,interval=.05,plan=plan)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate();unrelated.wait(timeout=5)
    assert result<0
    final=json.loads(output.read_text());assert final['stop_reason']=='wall budget'
    rows=[json.loads(v) for v in output.with_suffix('.progress.jsonl').read_text().splitlines()]
    assert len(rows)>=2 and rows[-1]['stop_reason']=='wall budget'
    assert all(r['boot_id'] and r['elapsed_seconds']>=0 for r in rows)
    assert any(r['rss_sum_bytes']>0 and r['checkpoints'] for r in rows)
    assert final['peak_child_rss_bytes']>0
    with zipfile.ZipFile(checkpoint) as z:
        assert z.testzip() is None and z.read('payload')==b'preserved checkpoint fixture'
    for process in rows[-1]['processes']:
        assert not Path('/proc',str(process['pid'])).exists()
    with pytest.raises(ValueError,match='new accounting'):
        execute([sys.executable,'-c','pass'],output)


def test_tabla_cli_refuses_unapproved_launch(tmp_path,monkeypatch):
    from scripts.epistasis import run_measured
    monkeypatch.setattr(run_measured.socket,'gethostname',lambda:'tabla')
    monkeypatch.setattr(sys,'argv',['run_measured.py','--out',str(tmp_path/'resources.json'),
        '--',sys.executable,'-c','raise RuntimeError("must not run")'])
    with pytest.raises(ValueError,match='disabled'):
        run_measured.main()
    assert not list(tmp_path.iterdir())


def test_local_shell_refuses_cohort_work_before_cpu_setup(tmp_path):
    # Stub only host identification, never invoke conda or load genotypes.
    hostname=tmp_path/'hostname';hostname.write_text('#!/bin/sh\nprintf "tabla\\n"\n');hostname.chmod(0o700)
    env=dict(os.environ,HOSTNAME='Tabla',PATH=str(tmp_path)+os.pathsep+os.environ['PATH'])
    script=Path(__file__).resolve().parents[1]/'scripts/epistasis/run_local.sh'
    for command in ('train-direction','make-inputs','prepare','prepare-traits'):
        result=subprocess.run(['bash',str(script),command],env=env,capture_output=True,text=True)
        assert result.returncode==2 and 'Tabla cohort execution is disabled' in result.stderr


def test_explicit_supervisor_stop_reaches_child(tmp_path):
    marker=tmp_path/'child.json'
    child_code='import os,json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid()})); time.sleep(30)'
    code='import sys; from scripts.epistasis.run_measured import execute; execute([sys.executable,"-c",sys.argv[1],sys.argv[2]],sys.argv[3],interval=30)'
    repo=Path(__file__).resolve().parents[1]
    process=subprocess.Popen([sys.executable,'-c',code,child_code,str(marker),str(tmp_path/'resources.json')],
        cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        assert marker.exists()
        child=json.loads(marker.read_text())['pid']
        process.terminate();process.wait(timeout=5)
        assert not Path('/proc',str(child)).exists()
        assert (tmp_path/'resources.progress.jsonl').is_file()
    finally:
        if process.poll() is None:
            process.terminate();process.wait(timeout=5)


def test_two_local_jobs_share_one_monitored_budget(tmp_path):
    import hashlib
    from scripts.epistasis.local_batch import validate_batch
    physical={}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=tuple((topology/f).read_text() for f in ('physical_package_id','core_id'))
        physical.setdefault(key,cpu)
    if len(physical)<2:pytest.skip('two physical cores required for disjoint placement')
    cpus=list(physical.values())[:2]
    topology={cpu:key for key,cpu in physical.items()}
    code="""import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);(root/'child.json').write_text(json.dumps(dict(pid=os.getpid(),cpus=sorted(os.sched_getaffinity(0)))))
(root/'solver.npz').write_bytes(b'completed child output')
time.sleep(30)
"""
    repo=Path(__file__).resolve().parents[1]
    members=[];roots=[]
    for j,cpu in enumerate(cpus):
        root=tmp_path/f'job{j}';root.mkdir();roots.append(root)
        members.append(dict(name=f'job{j}',command=[sys.executable,'-c',code,str(root)],
            cwd=str(repo),cpu_ids=[cpu],environment={},log=str(root/'run.log')))
    spec=dict(schema_version=1,members=members)
    assert validate_batch(spec,cpus,topology)==members
    members[1]['cpu_ids']=[cpus[0]]
    with pytest.raises(ValueError,match='disjoint physical cores'):
        validate_batch(spec,cpus,topology)
    members[1]['cpu_ids']=[cpus[1]]
    manifest=tmp_path/'batch.json';manifest.write_text(json.dumps(spec))
    output=tmp_path/'monitor';output.mkdir()
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    command=[sys.executable,'-m','scripts.epistasis.local_batch',str(manifest),'--sha256',digest]
    plan=dict(cpu_ids=cpus,max_seconds=1.,max_rss_bytes=128*2**20,
        max_output_growth_bytes=2**20,min_available_memory_bytes=0,min_free_disk_bytes=0,
        output_root=str(output),work_roots=list(map(str,roots)))
    result=execute(command,output/'resources.json',interval=.1,plan=plan)
    assert result!=0
    rows=[json.loads(line) for line in (output/'resources.progress.jsonl').read_text().splitlines()]
    assert any(len(row['processes'])>=3 for row in rows)
    assert {p['root'] for p in rows[-1]['checkpoints']}==set(map(str,roots))
    assert rows[-1]['output_growth_bytes']>=2*len(b'completed child output')
    for root,cpu in zip(roots,cpus):
        member=json.loads((root/'child.json').read_text())
        assert member['cpus']==[cpu] and not Path('/proc',str(member['pid'])).exists()
        assert (root/'solver.npz').read_bytes()==b'completed child output'


def test_genotype_progress_preserves_values_and_pass_counts(tmp_path,monkeypatch):
    import numpy as np
    from dataclasses import asdict
    from summit.prediction.genotype import ArrayGenotypeSource, RawBlockStream
    from summit.prediction.spec import VariantAxis
    from scripts.epistasis.stream_progress import record_progress
    raw=np.array([[0,1,-127,2,0],[1,0,2,1,2],[2,1,1,0,1]],dtype=np.int8)
    axis=VariantAxis(tuple(f'v{i}' for i in range(5)),('2',)*5,tuple(range(1,6)),('A',)*5,('C',)*5)
    def stream():
        source=ArrayGenotypeSource(raw,[(str(i),str(i)) for i in range(3)],axis,hard_calls=True)
        return RawBlockStream(source,np.arange(3),np.arange(5),block_size=2,storage='packed')
    baseline=stream();expected=np.column_stack([b.copy() for _,_,b in baseline.blocks('build',build_cache=True)])
    actual=stream();path=tmp_path/'genotype.progress.jsonl'
    with record_progress(path,interval=3600):
        observed=np.column_stack([b.copy() for _,_,b in actual.blocks('build',build_cache=True)])
        cached=np.column_stack([b.copy() for _,_,b in actual.blocks('cached')])
    expected_cached=np.column_stack([b.copy() for _,_,b in baseline.blocks('cached')])
    np.testing.assert_array_equal(observed,expected)
    np.testing.assert_array_equal(cached,expected_cached)
    assert asdict(actual.ledger)==asdict(baseline.ledger)
    records=[json.loads(v) for v in path.read_text().splitlines()]
    assert [r['event'] for r in records]==['started','finished','started','finished']
    assert [r['completed_variants'] for r in records]==[0,5,0,5]
    assert all(r['rows']==3 and r['boot_id'] for r in records)
    with record_progress(tmp_path/'stopped.jsonl'):
        iterator=actual.blocks('partial');next(iterator);iterator.close()
    final=json.loads((tmp_path/'stopped.jsonl').read_text().splitlines()[-1])
    assert final['event']=='closed_before_completion' and final['completed_variants']==0
    with record_progress(tmp_path/'retained.jsonl'):
        retained=actual.blocks('partial');next(retained)
    retained.close()
    final=json.loads((tmp_path/'retained.jsonl').read_text().splitlines()[-1])
    assert final['event']=='closed_before_completion'
    from itertools import count
    from types import SimpleNamespace
    from scripts.epistasis import stream_progress
    ticks=count(0,10)
    with monkeypatch.context() as patcher:
        patcher.setattr(stream_progress,'time',SimpleNamespace(monotonic=lambda:next(ticks),time=lambda:123.))
        with record_progress(tmp_path/'periodic.jsonl',interval=25):
            list(actual.blocks('periodic'))
    records=[json.loads(v) for v in (tmp_path/'periodic.jsonl').read_text().splitlines()]
    assert [r['event'] for r in records]==['started','progress','finished']
    assert [r['completed_variants'] for r in records]==[0,4,5]


def test_command_exit_cannot_leave_unmonitored_descendants(tmp_path):
    from scripts.epistasis.run_measured import live_session_members
    marker=tmp_path/'child.json'
    child_code='import os,json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),session=os.getsid(0)))); time.sleep(30)'
    parent_code='''import subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]],process_group=0)
deadline=time.monotonic()+5
while not Path(sys.argv[2]).exists() and time.monotonic()<deadline:time.sleep(.01)
assert Path(sys.argv[2]).exists()
'''
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        result=execute([sys.executable,'-c',parent_code,child_code,str(marker)],tmp_path/'accounting.json',interval=.1)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate();unrelated.wait(timeout=5)
    record=json.loads((tmp_path/'accounting.json').read_text())
    assert result==125 and record['exit_code']==0 and record['supervisor_exit_code']==125
    assert record['stop_reason']=='owned descendants outlived command'
    assert not live_session_members(json.loads(marker.read_text())['session'])


def test_tabla_exclusion_is_host_specific_and_checks_every_thread(monkeypatch):
    from scripts.epistasis import cpu_policy
    # Pure mask checks: never schedule work on excluded cores to test refusal.
    for cpus in ([0], [7, 8], [64], [71], list(range(8, 17)), [72]):
        with pytest.raises(ValueError, match='Tabla'):
            cpu_policy.validate_cpus(cpus, hostname='Tabla')
    cpu_policy.validate_cpus([8, 16, 32, 48], hostname='Tabla')
    cpu_policy.validate_cpus([0, 1, 64], hostname='n6430')
    monkeypatch.setattr(cpu_policy.socket, 'gethostname', lambda:'Tabla')
    monkeypatch.setattr(cpu_policy, 'thread_masks', lambda pid=None:{1:[8], 2:[64]})
    with pytest.raises(ValueError, match='excludes'):
        cpu_policy.require_affinity([8, 9])
    monkeypatch.setattr(cpu_policy, 'thread_masks', lambda pid=None:{1:[8], 2:[9]})
    assert cpu_policy.require_affinity([8, 9]) == {1:[8], 2:[9]}
    with pytest.raises(ValueError, match='escaped'):
        cpu_policy.require_affinity([8])
    monkeypatch.setattr(cpu_policy, 'descendants', lambda pid:{1, 2})
    monkeypatch.setattr(cpu_policy, 'thread_masks', lambda pid:{pid:[8 if pid==1 else 10]})
    with pytest.raises(ValueError, match='descendant thread'):
        cpu_policy.require_tree_affinity(1, [8, 9])


def test_local_shell_rejects_suspect_cpu_before_interpreter(tmp_path):
    hostname=tmp_path/'hostname';hostname.write_text('#!/bin/sh\nprintf "Tabla\\n"\n');hostname.chmod(0o700)
    script=Path(__file__).resolve().parents[1]/'scripts/epistasis/run_local.sh'
    for cpus in ('0', '64', '8,9,10,11,12,13,14,15,16'):
        result=subprocess.run(['bash',str(script),'fit','unused'],capture_output=True,text=True,
            env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],SUMMIT_CPUS=cpus,SUMMIT_CONDA_ROOT='/must/not/start'))
        assert result.returncode==2 and 'Tabla' in result.stderr
        assert '/must/not/start' not in result.stderr


def test_recovery_wrapper_recomputes_full_operator_and_rejects_bad_solution(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from scripts.epistasis.stream_progress import verify_recovered_solves
    from summit.epistasis import polygenic
    rng=np.random.default_rng(34911)
    fixed=np.ones((12,1));y=rng.normal(size=(12,2));y-=y.mean(axis=0)
    diagonal=np.arange(1,13,dtype=float)
    # Direct augmented system supplies an independent constrained solution.
    a=np.block([[np.diag(diagonal),fixed],[fixed.T,np.zeros((1,1))]])
    sol=np.linalg.solve(a,np.vstack([y,np.zeros((1,2))]))[:12]
    calls=[]
    def apply(value,theta,phase):
        calls.append(phase);return diagonal[:,None]*value
    operator=SimpleNamespace(apply=apply)
    reports={(str(j),'null'):dict(threshold=1e-8) for j in range(2)}
    monkeypatch.setattr(polygenic,'projected_solve',lambda *a,**k:(sol.copy(),SimpleNamespace(reports=reports)))
    with verify_recovered_solves(tmp_path/'valid.jsonl'):
        polygenic.projected_solve(operator,y,fixed,np.ones((1,2)),resume=True,checkpoint='frozen.npz')
    assert calls==['recovery_full_residual']
    assert json.loads((tmp_path/'valid.jsonl').read_text())['passed']
    bad=sol.copy();bad[0,0]+=.01
    monkeypatch.setattr(polygenic,'projected_solve',lambda *a,**k:(bad.copy(),SimpleNamespace(reports=reports)))
    with verify_recovered_solves(tmp_path/'invalid.jsonl'):
        with pytest.raises(RuntimeError,match='fresh full-operator'):
            polygenic.projected_solve(operator,y,fixed,np.ones((1,2)),resume=True)
    assert not json.loads((tmp_path/'invalid.jsonl').read_text())['passed']


def test_snapshot_exit_race_does_not_hide_live_accounting_failure(tmp_path,monkeypatch):
    from scripts.epistasis.run_measured import snapshot
    import time
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    read_text=Path.read_text
    inaccessible=False
    dead=False
    def racing_read(path,*args,**kwargs):
        nonlocal inaccessible
        if path==Path(f'/proc/{child.pid}/io'):
            inaccessible=True
            raise PermissionError('injected exit-time accounting race')
        text=read_text(path,*args,**kwargs)
        if path==Path(f'/proc/{child.pid}/stat') and inaccessible and dead:
            end=text.rfind(')')+2
            text=text[:end]+'Z'+text[end+1:]
        return text
    try:
        monkeypatch.setattr(Path,'read_text',racing_read)
        # Unreadable accounting while alive cannot silently remove a worker.
        with pytest.raises(PermissionError,match='accounting race'):
            snapshot(child.pid,tmp_path,time.monotonic(),0)
        inaccessible=False;dead=True
        result=snapshot(child.pid,tmp_path,time.monotonic(),0)
        assert result['processes']==[] and result['rss_sum_bytes']==0
    finally:
        child.terminate();child.wait(timeout=5)


def test_supervisor_thread_escape_is_a_budget_stop():
    from scripts.epistasis.run_measured import budget_reason
    plan=dict(cpu_ids=[8,9],max_seconds=60,max_rss_bytes=100,max_output_growth_bytes=100,
        min_available_memory_bytes=100,min_free_disk_bytes=100)
    row=dict(processes=[dict(cpu_ids=[8])],supervisor_thread_affinities={1:[8],2:[9]},
        elapsed_seconds=1,rss_sum_bytes=10,output_growth_bytes=10,
        available_memory_bytes=1000,free_disk_bytes=1000)
    assert budget_reason(row,plan) is None
    # Pure mask fixture: no thread is moved to an unallocated or suspect CPU.
    row['supervisor_thread_affinities'][2]=[10]
    assert budget_reason(row,plan)=='supervisor affinity escaped assigned CPUs'


def test_transient_proc_io_denial_keeps_live_worker_accounting(tmp_path,monkeypatch):
    from scripts.epistasis.run_measured import snapshot
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    read_text=Path.read_text;denials=0
    def transient_read(path,*args,**kwargs):
        nonlocal denials
        if path==Path(f'/proc/{child.pid}/io') and denials<1:
            denials+=1
            raise PermissionError('transient accounting denial')
        return read_text(path,*args,**kwargs)
    try:
        monkeypatch.setattr(Path,'read_text',transient_read)
        row=snapshot(child.pid,tmp_path,time.monotonic(),0)
        assert denials==1 and [p['pid'] for p in row['processes']]==[child.pid]
        assert row['processes'][0]['io']
    finally:
        child.terminate();child.wait(timeout=5)


def test_numerical_startup_rejects_inherited_placement_before_imports(monkeypatch):
    from scripts.epistasis import cpu_policy
    monkeypatch.setattr(cpu_policy.socket, 'gethostname', lambda:'Tabla')
    monkeypatch.setattr(cpu_policy, 'thread_masks', lambda pid=None:{1:[8,9]})
    monkeypatch.setattr(cpu_policy.os, 'sched_getaffinity', lambda pid:{8,9})
    valid=dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', BLIS_NUM_THREADS='2',
        OMP_PLACES='{8},{9}')
    assert cpu_policy.require_numerical_startup(environment=valid)=={1:[8,9]}
    for change in (dict(OMP_PLACES='{0},{8}'),dict(OMP_PLACES='{8},{64}'),
            dict(OMP_PLACES='{8},{10}'),dict(OMP_PLACES='cores'),dict(OMP_PLACES='{8},{8}'),
            dict(OMP_NUM_THREADS='3'),dict(OPENBLAS_NUM_THREADS='0'),dict(BLIS_NUM_THREADS='1,2'),
            dict(GOMP_CPU_AFFINITY='8 64'),dict(GOMP_CPU_AFFINITY='0-127'),
            dict(KMP_AFFINITY='explicit,proclist=[0]'),dict(KMP_HW_SUBSET='1s')):
        with pytest.raises(ValueError):
            cpu_policy.require_numerical_startup(environment=dict(valid,**change))
    # Host-specific exclusion must not be imposed on scheduler CPU numbering.
    monkeypatch.setattr(cpu_policy.socket, 'gethostname', lambda:'n6430')
    monkeypatch.setattr(cpu_policy, 'thread_masks', lambda pid=None:{1:[0,1]})
    assert cpu_policy.require_numerical_startup(environment=dict(OMP_PLACES='{0},{1}'))=={1:[0,1]}


def test_checkout_launcher_rejects_unsafe_places_without_loading_numerics():
    import os, socket, subprocess, sys
    from pathlib import Path
    if socket.gethostname().split('.')[0].lower()!='tabla':
        pytest.skip('Tabla launch guard')
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,OMP_PLACES='{0}',OMP_PROC_BIND='SPREAD',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BLIS_NUM_THREADS='1')
    # This exits in standard-library checks; it must never initialize libgomp
    # on the requested excluded CPU. No diagnostic work runs on that CPU.
    result=subprocess.run([sys.executable,'-I',str(root/'scripts/epistasis/checkout_python.py'),
        'a_module_that_does_not_exist'],env=env,capture_output=True,text=True)
    assert result.returncode and 'OMP_PLACES escaped' in result.stderr
    assert 'a_module_that_does_not_exist' not in result.stderr
