"""Launch private SUMMIT on exactly the physical CPUs assigned by Hoffman.

Use with ``-pe shared N -binding set linear:1``. This program requests no
resources and never expands the inherited scheduler CPU set.
"""
import json
import os
from pathlib import Path
import socket
import sys


def physical_cpu_subset(slots, cpus, topology):
    """Select one inherited hardware thread per allocated physical core.

    Some execution hosts expose both SMT siblings for each bound core. Accept
    that case only when the inherited mask contains exactly NSLOTS physical
    cores; a broad unbound host mask or an undersized allocation still fails.
    """
    if slots<1 or not cpus or len(set(cpus))!=len(cpus) or any(c not in topology for c in cpus):
        raise RuntimeError('scheduler affinity must identify exactly NSLOTS physical cores')
    selected={}
    for cpu in sorted(cpus):selected.setdefault(topology[cpu],cpu)
    if len(selected)!=slots:
        raise RuntimeError('scheduler affinity must identify exactly NSLOTS physical cores')
    return sorted(selected.values())


def environment(slots, cpus, topology, inherited):
    if slots < 1 or len(cpus) != slots or len(set(cpus)) != slots:
        raise RuntimeError('scheduler affinity must contain exactly NSLOTS CPUs')
    if len({topology[cpu] for cpu in cpus}) != slots:
        raise RuntimeError('scheduler CPUs must represent distinct physical cores')
    cpus=sorted(cpus)
    result=dict(inherited)
    for key in ('GOMP_CPU_AFFINITY','KMP_AFFINITY','KMP_HW_SUBSET','KMP_PLACE_THREADS',
            'OMP_NESTED','BLIS_NT','BLIS_TI','BLIS_THREAD_IMPL','BLIS_JC_NT','BLIS_PC_NT',
            'BLIS_IC_NT','BLIS_JR_NT','BLIS_IR_NT','BLIS_ARCH_TYPE','BLIS_ARCH_DEBUG',
            'BLIS_PACK_A','BLIS_PACK_B'):
        result.pop(key,None)
    result.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(slots),
        OMP_THREAD_LIMIT=str(slots),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='SPREAD',
        OMP_PLACES=','.join(f'{{{cpu}}}' for cpu in cpus),OMP_MAX_ACTIVE_LEVELS='1',
        OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',BLIS_NUM_THREADS=str(slots),
        OPENBLAS_NUM_THREADS=str(slots),MKL_NUM_THREADS=str(slots))
    return result


def main():
    if len(sys.argv)<2:
        raise ValueError('supply a module and its arguments')
    slots=int(os.environ['NSLOTS'])
    cpus=sorted(os.sched_getaffinity(0))
    topology={}
    for cpu in cpus:
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology[cpu]=tuple(int((root/name).read_text()) for name in ('physical_package_id','core_id'))
    print(json.dumps(dict(phase='scheduler_affinity',host=socket.gethostname(),
        slots=slots,initial_affinity=cpus,physical_cores=[topology[c] for c in cpus])),flush=True)
    selected=physical_cpu_subset(slots,cpus,topology)
    env=environment(slots,selected,topology,os.environ)
    if selected!=cpus:
        os.sched_setaffinity(0,selected)
        if sorted(os.sched_getaffinity(0))!=selected:
            raise RuntimeError('scheduler CPU subset was not applied')
    print(json.dumps(dict(phase='scheduler_placement',host=socket.gethostname(),
        job_id=os.environ.get('JOB_ID'),task_id=os.environ.get('SGE_TASK_ID'),
        slots=slots,initial_affinity=cpus,selected_affinity=selected,
        physical_cores=[topology[c] for c in selected])),flush=True)
    launcher=Path(__file__).resolve().parents[1]/'generalized_gxe/private_python.py'
    os.execve(sys.executable,[sys.executable,str(launcher),*sys.argv[1:]],env)


if __name__=='__main__': main()
