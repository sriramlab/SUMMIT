"""Run at most two disjoint local jobs inside one aggregate supervised session.

Launch only through run_measured.py with a boot-specific aggregate plan. This
coordinator neither detaches children nor resumes or retries a failed command.
The enclosing supervisor owns resource limits and durable progress evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.epistasis.hoffman_launch import environment
from scripts.epistasis.cpu_policy import validate_cpus, require_affinity, require_tree_affinity


def validate_batch(spec, available, topology):
    if set(spec)!={'schema_version','members'} or spec['schema_version']!=1:
        raise ValueError('unsupported local batch definition')
    members=spec['members']
    if not isinstance(members,list) or not 1<=len(members)<=2:
        raise ValueError('a local batch contains one or two jobs')
    names,cores,logs=set(),set(),set()
    for member in members:
        if set(member)!={'name','command','cwd','cpu_ids','environment','log'}:
            raise ValueError('incomplete local batch member')
        cpus=member['cpu_ids'];command=member['command']
        if (not isinstance(cpus,list) or not 1<=len(cpus)<=4
                or len(set(cpus))!=len(cpus) or not set(cpus)<=set(available)):
            raise ValueError('each job needs at most four inherited CPUs')
        validate_cpus(cpus)
        selected={topology[c] for c in cpus}
        if len(selected)!=len(cpus) or selected&cores:
            raise ValueError('local jobs must use disjoint physical cores')
        cores.update(selected)
        if (not isinstance(command,list) or not command
                or any(not isinstance(v,str) or not v for v in command)
                or not Path(command[0]).is_absolute()):
            raise ValueError('use an absolute executable and nonempty command arguments')
        if (not member['name'] or member['name'] in names
                or member['log'] in logs):
            raise ValueError('local job names and logs must be distinct')
        names.add(member['name']);logs.add(member['log'])
        if (not Path(member['cwd']).is_absolute() or not Path(member['cwd']).is_dir()
                or not Path(member['log']).is_absolute() or Path(member['log']).exists()):
            raise ValueError('existing absolute code directories and new absolute logs required')
        if (not isinstance(member['environment'],dict)
                or any(not isinstance(k,str) or not isinstance(v,str)
                    for k,v in member['environment'].items())):
            raise ValueError('environment entries must be strings')
    return members


def run(spec):
    require_affinity()
    available=sorted(os.sched_getaffinity(0));topology={}
    for cpu in available:
        directory=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology[cpu]=tuple((directory/name).read_text().strip()
            for name in ('physical_package_id','core_id'))
    members=validate_batch(spec,available,topology)
    children=[];handles=[];finished=set();allocations={m['name']:m['cpu_ids'] for m in members}
    previous=signal.getsignal(signal.SIGTERM)
    def requested(signum,frame):
        raise InterruptedError('local batch stop requested')
    signal.signal(signal.SIGTERM,requested)
    try:
        for member in members:
            cpus=member['cpu_ids']
            env=environment(len(cpus),cpus,topology,dict(os.environ,**member['environment']))
            handle=Path(member['log']).open('x');handles.append(handle)
            os.chmod(member['log'],0o600)
            process=subprocess.Popen(member['command'],cwd=member['cwd'],env=env,
                stdout=handle,stderr=subprocess.STDOUT,
                preexec_fn=lambda assigned=cpus:os.sched_setaffinity(0,assigned))
            children.append((member['name'],process))
            print(json.dumps(dict(event='started',name=member['name'],pid=process.pid,
                cpu_ids=cpus,session=os.getsid(process.pid))),flush=True)
        while len(finished)<len(children):
            for name,process in children:
                require_tree_affinity(process.pid,allocations[name])
                result=process.poll()
                if result is None or name in finished:continue
                finished.add(name)
                print(json.dumps(dict(event='finished',name=name,pid=process.pid,exit_code=result)),flush=True)
                if result:return result
            if len(finished)<len(children):time.sleep(1)
        return 0
    finally:
        for _,process in children:
            if process.poll() is None:process.terminate()
        for _,process in children:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for handle in handles:handle.close()
        signal.signal(signal.SIGTERM,previous)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    data=args.manifest.read_bytes()
    if hashlib.sha256(data).hexdigest()!=args.sha256:
        raise ValueError('local batch definition changed after admission')
    return run(json.loads(data))


if __name__=='__main__':raise SystemExit(main())
