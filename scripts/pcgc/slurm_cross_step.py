"""Enter the existing PCGC guard on physical cores assigned to one Slurm step."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

root=Path(__file__).resolve().parents[2]
if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
    raise RuntimeError("run inside an allocated srun step")
slots=int(os.environ["SLURM_CPUS_PER_TASK"])
if int(os.environ.get("SLURM_NTASKS","1"))!=1:
    raise RuntimeError("one shared-memory task is required")
spec=importlib.util.spec_from_file_location("placement",root/"scripts/epistasis/hoffman_launch.py")
placement=importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)
cpus=sorted(os.sched_getaffinity(0))
topology={}
for cpu in cpus:
    path=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    topology[cpu]=tuple(int((path/key).read_text()) for key in ("physical_package_id","core_id"))
selected=placement.physical_cpu_subset(slots,cpus,topology)
env=placement.environment(slots,selected,topology,os.environ)
mode,native,*args=sys.argv[1:]
if mode=="portable":
    env["OPENBLAS_NUM_THREADS"]=env["MKL_NUM_THREADS"]=str(slots)
os.sched_setaffinity(0,selected)
print(json.dumps(dict(phase="cross_pcgc_slurm_placement",job=os.environ["SLURM_JOB_ID"],
    step=os.environ["SLURM_STEP_ID"],host=socket.gethostname(),initial_affinity=cpus,
    selected_affinity=selected,physical_cores=[topology[c] for c in selected])),flush=True)
os.execve(sys.executable,[sys.executable,str(root/"scripts/pcgc/gxe_python.py"),mode,native,*args],env)
