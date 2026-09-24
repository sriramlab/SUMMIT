"""Run a module under this checkout's verified private-BLIS placement contract.

Launch with taskset (local) or the verified scheduler CPU set, and the same
thread environment used by workflow.require_private_blis. Native modules
default to build/private_blis; SUMMIT_PRIVATE_NATIVE_DIR can select a staged
build. Usage: python private_python.py module.name [module arguments ...]
"""
import ctypes
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
summit.__path__.insert(0,os.environ.get('SUMMIT_PRIVATE_NATIVE_DIR',str(root/'build/private_blis')))
from summit import gxeldcore
import workflow
workflow._PRE_NUMERICAL_CPU_AFFINITY=cpus
info,threads,placement=workflow.require_private_blis(gxeldcore)
assert info['gemm_integrity_enabled'] and info['gemm_checksum_enabled']
module=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_module(module,run_name='__main__')
