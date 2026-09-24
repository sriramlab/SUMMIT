"""Run a module under this checkout's verified private-BLIS placement contract.

Launch with taskset (local) or the verified scheduler CPU set, and the same
thread environment used by workflow.require_private_blis. Native modules
default to build/private_blis; SUMMIT_PRIVATE_NATIVE_DIR can select a staged
build. Usage: python private_python.py module.name [module arguments ...]
"""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

cpus=tuple(sorted(os.sched_getaffinity(0)))
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(41,1,0,0,0)==0 and libc.prctl(42,0,0,0,0)==1
root=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(root/'src'),str(root/'scripts/generalized_gxe')]
import summit
assert Path(summit.__file__).resolve()==root/'src/summit/__init__.py'
# Importing libgomp can bind the main thread to its first singleton place.
# Warm NumPy's separate pthread BLAS pool beforehand, while workers inherit
# the full reserved CPU set. Reducing its active count later preserves the
# pool; increasing it during a study must not create single-CPU workers.
import numpy as np
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=len(cpus),user_api='blas'):
    warm=np.ones((512,512));assert (warm@warm)[0,0]==512
del warm
numpy_worker_affinity={int(p.name):sorted(os.sched_getaffinity(int(p.name)))
    for p in Path('/proc/self/task').iterdir() if int(p.name)!=os.getpid()}
if any(mask!=list(cpus) for mask in numpy_worker_affinity.values()):
    raise RuntimeError('NumPy workers did not inherit the reserved CPU set before OpenMP initialization')
native=Path(os.environ.get('SUMMIT_PRIVATE_NATIVE_DIR',str(root/'build/private_blis')))
# Load only extensions from the qualified runtime. A runtime directory may
# also contain old Python modules, which must never shadow this checkout.
for leaf in ('gxeldcore','gwldcore','winldcore'):
    candidates=list(native.glob(leaf+'*.so'))
    if len(candidates)!=1:
        raise RuntimeError(f'exactly one {leaf} extension required in {native}')
    spec=importlib.util.spec_from_file_location('summit.'+leaf,candidates[0])
    extension=importlib.util.module_from_spec(spec);sys.modules[spec.name]=extension
    spec.loader.exec_module(extension)
from summit import gxeldcore
import workflow
workflow._PRE_NUMERICAL_CPU_AFFINITY=cpus
workflow._NUMPY_POOL_WORKER_AFFINITY=numpy_worker_affinity
info,threads,placement=workflow.require_private_blis(gxeldcore)
assert info['gemm_integrity_enabled'] and info['gemm_checksum_enabled']
print(json.dumps(dict(phase='runtime_placement',launch_cpu_ids=cpus,threads=threads,
    numpy_worker_affinity=numpy_worker_affinity,
    placement=placement,native_path=gxeldcore.__file__,native_build=info)),flush=True)
module=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_module(module,run_name='__main__')
