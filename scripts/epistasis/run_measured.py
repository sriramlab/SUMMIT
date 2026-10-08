"""Execute one command with durable progress and final child accounting.

No restart is automatic. Tabla launches require a command-specific approved
bounded plan. Atomic solver checkpoints remain the recovery mechanism; this
supervisor never changes a checkpoint or injects a signal handler into a fit.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time
import runpy

# File loading also works when invoked as a standalone frozen script.
_cpu_policy=runpy.run_path(str(Path(__file__).with_name("cpu_policy.py")))


def command_digest(command):
    return hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest()


@contextmanager
def local_lease(path):
    """One aggregate supervisor per user; never unlink a held lock."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        if os.fstat(fd).st_uid!=os.getuid():
            raise ValueError('local execution lock has a different owner')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another local epistasis pilot owns the aggregate budget') from None
        yield
    finally:
        os.close(fd)


def local_plan(path, command):
    if path is None:
        raise ValueError('Tabla batch execution is disabled: a command-specific approved local plan is required')
    plan=json.loads(Path(path).read_text())
    required={'approved','approval_reference','command_sha256','cpu_ids','max_seconds',
        'max_rss_bytes','max_output_growth_bytes','min_available_memory_bytes','min_free_disk_bytes',
        'output_root','boot_id'}
    if (set(plan)-{'work_roots'}!=required or plan['approved'] is not True
            or not plan['approval_reference']):
        raise ValueError('explicit approved local plan required; a budget alone is not approval')
    if plan['command_sha256']!=command_digest(command):
        raise ValueError('approved command differs')
    if plan['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('local plan belongs to a previous boot; automatic recovery is forbidden')
    root=Path(plan['output_root'])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('approved output root must be an existing absolute directory')
    cpus=plan['cpu_ids']
    if (not isinstance(cpus,list) or not 1<=len(cpus)<=8 or len(set(cpus))!=len(cpus)
            or not set(cpus)<=os.sched_getaffinity(0)):
        raise ValueError('approve at most eight available CPU IDs')
    _cpu_policy['validate_cpus'](cpus)
    physical=[]
    for cpu in cpus:
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append(tuple((topology/name).read_text().strip()
            for name in ('physical_package_id','core_id')))
    if len(set(physical))!=len(cpus):
        raise ValueError('use one hardware thread per distinct physical core')
    bounds={'max_seconds':6*3600,'max_rss_bytes':48*2**30,'max_output_growth_bytes':8*2**30}
    for key,upper in bounds.items():
        if not isinstance(plan[key],(int,float)) or not 0<plan[key]<=upper:
            raise ValueError('local pilot exceeds the conservative '+key+' ceiling')
    if plan['min_available_memory_bytes']<256*2**30 or plan['min_free_disk_bytes']<100*2**30:
        raise ValueError('retain at least 256 GiB available RAM and 100 GiB free disk')
    if 'work_roots' in plan:
        roots=plan['work_roots']
        if not isinstance(roots,list) or not 1<=len(roots)<=2:
            raise ValueError('declare at most two cohort output roots')
        resolved=[root.resolve()]
        for entry in roots:
            p=Path(entry)
            if not p.is_absolute() or not p.is_dir() or p.is_symlink():
                raise ValueError('cohort output roots must be existing absolute directories')
            resolved.append(p.resolve())
        if any(a.is_relative_to(b) or b.is_relative_to(a)
                for j,a in enumerate(resolved) for b in resolved[j+1:]):
            raise ValueError('aggregate output roots must be disjoint')
    return plan


def output_bytes(root):
    total=0
    for directory,_,files in os.walk(root,followlinks=False):
        for name in files:
            path=Path(directory)/name
            try:
                if not path.is_symlink():total+=path.stat().st_size
            except FileNotFoundError:
                pass  # An atomic checkpoint may replace its temporary file.
    return total


def process_io(entry, start_ticks):
    """Read accounting through Linux's brief non-dumpable exit transition.

    A live process with persistently inaccessible accounting is an error. Only
    observed death, disappearance or replacement permits omitting that sample.
    """
    deadline=time.monotonic()+.1
    while True:
        try:
            return {k:int(value) for k,value in
                (line.split(':',1) for line in (entry/'io').read_text().splitlines())}
        except PermissionError:
            raw=(entry/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            if fields[0] in ('Z','X') or int(fields[19])!=start_ticks:
                raise ProcessLookupError('accounted process exited or was replaced') from None
            if time.monotonic()>=deadline:raise
            time.sleep(.005)


def snapshot(pid, root, started, initial_bytes, work_roots=()):
    processes=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            raw=(entry/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
            # Popen creates one new session; include its non-detached descendants.
            if int(v[3])!=pid or v[0] in ('Z','X'):continue
            io=process_io(entry,int(v[19]))
            masks=_cpu_policy['thread_masks'](int(entry.name))
            cpus={cpu for mask in masks.values() for cpu in mask}
            processes.append(dict(pid=int(entry.name),start_ticks=int(v[19]),
                rss_bytes=int(v[21])*os.sysconf('SC_PAGE_SIZE'),
                cpu_seconds=(int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK'),
                io=io,cpu_ids=sorted(cpus),thread_affinities=masks))
        except (FileNotFoundError,ProcessLookupError):
            continue
    memory={k:int(v.split()[0])*1024 for k,v in
        (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    roots=[Path(root),*map(Path,work_roots)]
    disks=[os.statvfs(p) for p in roots]
    checkpoints=[]
    for directory in roots:
        for p in directory.rglob('*.npz'):
            if p.name not in ('solver.npz','outcome_feature.npz','mean_derivative.npz',
                    'covariance_contrast.npz','feature.npz','variance.npz','known_covariance.npz'):continue
            try:s=p.stat()
            except FileNotFoundError:continue
            checkpoints.append(dict(root=str(directory),path=str(p.relative_to(directory)),
                bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    return dict(unix_time=time.time(),elapsed_seconds=time.monotonic()-started,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        processes=processes,supervisor_thread_affinities=_cpu_policy['thread_masks'](),rss_sum_bytes=sum(p['rss_bytes'] for p in processes),
        cpu_seconds=sum(p['cpu_seconds'] for p in processes),host_load_average=list(os.getloadavg()),
        available_memory_bytes=memory['MemAvailable'],
        free_disk_bytes=min(d.f_bavail*d.f_frsize for d in disks),
        output_growth_bytes=max(0,sum(output_bytes(p) for p in roots)-initial_bytes),checkpoints=checkpoints)


def budget_reason(row, plan):
    affinity_violation=any(set(p.get('cpu_ids',[]))-set(plan.get('cpu_ids',[]))
        for p in row.get('processes',[]))
    supervisor_violation=('cpu_ids' in plan and any(
        set(cpus)-set(plan['cpu_ids'])
        for cpus in row.get('supervisor_thread_affinities',{}).values()))
    checks=[(affinity_violation,'worker affinity escaped assigned CPUs'),
        (supervisor_violation,'supervisor affinity escaped assigned CPUs'),
        (row['elapsed_seconds']>=plan['max_seconds'],'wall budget'),
        (row['rss_sum_bytes']>plan['max_rss_bytes'],'aggregate RSS budget'),
        (row['output_growth_bytes']>plan['max_output_growth_bytes'],'output growth budget'),
        (row['available_memory_bytes']<plan['min_available_memory_bytes'],'host available RAM'),
        (row['free_disk_bytes']<plan['min_free_disk_bytes'],'free disk reserve')]
    return next((reason for failed,reason in checks if failed),None)


def signal_group(pid, sig):
    try:os.killpg(pid,sig)
    except ProcessLookupError:pass  # Child may finish between measurement and stop.


def live_session_members(session):
    """Identify remaining owned descendants, excluding already-dead zombies."""
    result=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            raw=(entry/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
            if int(v[3])==session and v[0] not in ('Z','X'):
                result.append((int(entry.name),int(v[19])))
        except (FileNotFoundError,ProcessLookupError):pass
    return result


def stop_session_members(session, members, sig):
    # Authenticate start ticks again immediately before each signal. Session
    # membership covers a child that created another process group, too.
    current=set(live_session_members(session))
    for pid,start in members:
        if (pid,start) in current:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass


def finish_owned_session(session, log):
    members=live_session_members(session)
    if not members:return False
    log.write(json.dumps(dict(event='owned_descendants_outlived_command',
        unix_time=time.time(),session=session,processes=members))+'\n')
    log.flush();os.fsync(log.fileno())
    stop_session_members(session,members,signal.SIGTERM)
    deadline=time.monotonic()+10
    while members and time.monotonic()<deadline:
        time.sleep(.1);members=live_session_members(session)
    if members:
        stop_session_members(session,members,signal.SIGKILL)
        deadline=time.monotonic()+5
        while members and time.monotonic()<deadline:
            time.sleep(.1);members=live_session_members(session)
    if members:raise RuntimeError('owned descendants remain after forced stop')
    return True


@contextmanager
def stop_on_term():
    """A supervisor stop must reach its child, which has a separate session."""
    previous=signal.getsignal(signal.SIGTERM)
    def requested(signum,frame):
        raise InterruptedError('supervisor stop requested by SIGTERM')
    signal.signal(signal.SIGTERM,requested)
    try:yield
    finally:signal.signal(signal.SIGTERM,previous)


def execute(command, output, *, interval=300., plan=None, work_root=None):
    """Own only the launched process group; save a flushed snapshot each interval."""
    _cpu_policy['require_affinity']()
    output=Path(output);progress=output.with_suffix('.progress.jsonl')
    root=Path(work_root or output.parent)
    if plan and 'output_root' in plan:
        root=Path(plan['output_root'])
        if not output.resolve().is_relative_to(root.resolve()):
            raise ValueError('local accounting must be inside the approved output root')
    if output.exists() or progress.exists() or interval<=0:
        raise ValueError('new accounting/progress paths and a positive interval required')
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    work_roots=plan.get('work_roots',[]) if plan else []
    initial=sum(output_bytes(p) for p in [root,*work_roots])
    if plan:
        reason=budget_reason(snapshot(-1,root,start,initial,work_roots),plan)
        if reason:raise ValueError('local preflight refused: '+reason)
    def affinity():
        os.sched_setaffinity(0,plan['cpu_ids'])
    stopped=None
    with stop_on_term(), progress.open('x') as log:
        os.chmod(progress,0o600)
        directory_fd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        process=subprocess.Popen(command,start_new_session=True,
            preexec_fn=affinity if plan is not None else None)
        try:
            while True:
                row=snapshot(process.pid,root,start,initial,work_roots)
                if plan:
                    stopped=budget_reason(row,plan)
                try:
                    _cpu_policy['require_affinity'](plan.get('cpu_ids') if plan else None)
                except ValueError as error:
                    stopped=str(error)
                row['stop_reason']=stopped
                log.write(json.dumps(row,separators=(',',':'))+'\n');log.flush();os.fsync(log.fileno())
                if stopped:
                    signal_group(process.pid,signal.SIGTERM)
                    break
                try:
                    remaining=plan['max_seconds']-row['elapsed_seconds'] if plan else interval
                    process.wait(timeout=min(interval,max(.001,remaining)))
                    break
                except subprocess.TimeoutExpired:
                    pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_group(process.pid,signal.SIGKILL);process.wait()
        except BaseException:
            signal_group(process.pid,signal.SIGTERM)
            if process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_group(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            if finish_owned_session(process.pid,log) and stopped is None:
                stopped='owned descendants outlived command'
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    supervisor_exit=process.returncode or (125 if stopped else 0)
    with output.open('x') as handle:
        json.dump(dict(command=command,exit_code=process.returncode,supervisor_exit_code=supervisor_exit,stop_reason=stopped,
            seconds=time.monotonic()-start,user_seconds=after.ru_utime-before.ru_utime,
            system_seconds=after.ru_stime-before.ru_stime,peak_child_rss_bytes=after.ru_maxrss*1024,
            input_blocks=after.ru_inblock-before.ru_inblock,output_blocks=after.ru_oublock-before.ru_oublock,
            progress=str(progress),work_root=str(root),local_plan=plan,
            accounting='Linux wait child accounting including waited descendants; block counts are not application read/write bytes; progress RSS sums may count shared pages more than once'),handle,indent=2)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    return supervisor_exit


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--local-plan',type=Path)
    parser.add_argument('--work-root',type=Path,help='Cohort output/checkpoint root; defaults to accounting parent')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.out.exists():
        raise ValueError('a command and new resource-output path are required')
    local=socket.gethostname().split('.')[0].lower()=='tabla'
    plan=local_plan(args.local_plan,command) if local else None
    if local:
        _cpu_policy['require_affinity'](plan['cpu_ids'])
        with local_lease(Path.home()/'.local/state/summit/epistasis-local.lock'):
            return execute(command,args.out,interval=60.,plan=plan)
    return execute(command,args.out,interval=300.,work_root=args.work_root)


if __name__=='__main__':
    raise SystemExit(main())
