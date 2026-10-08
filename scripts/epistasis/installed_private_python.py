"""Run an installed SUMMIT entry point with the research runtime safeguards.

Use the installed environment's Python, explicit singleton OMP_PLACES and an
outer taskset reservation. This launcher never adds checkout packages to the
import path. ``--stop-checkpoint`` interrupts after the second atomic training
checkpoint; otherwise arguments are passed to the ordinary installed CLI.
"""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    runpy.run_path(str(Path(__file__).with_name('cpu_policy.py')))['require_numerical_startup']()
    cpus = tuple(sorted(os.sched_getaffinity(0)))
    prefix = Path(sys.prefix).resolve()
    if any(name in os.environ for name in
           ('PYTHONPATH', 'SUMMIT_NATIVE_DIR', 'SUMMIT_PRIVATE_NATIVE_DIR')):
        raise RuntimeError('installed verification requires no source/native overrides')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError('process-local transparent hugepage guard failed')
    sys.dont_write_bytecode = True
    import numpy as np
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=len(cpus), user_api='blas'):
        warm = np.ones((512, 512))
        assert (warm @ warm)[0, 0] == 512
    del warm
    workers = {int(p.name): sorted(os.sched_getaffinity(int(p.name)))
               for p in Path('/proc/self/task').iterdir() if int(p.name) != os.getpid()}
    if any(mask != list(cpus) for mask in workers.values()):
        raise RuntimeError('NumPy workers did not inherit the complete selected CPU set')
    import summit
    from summit import gxeldcore, gwldcore, winldcore
    for module in (summit, gxeldcore, gwldcore, winldcore):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError('SUMMIT must be imported from the installed environment')
    # Reuse the launcher's placement checks without importing checkout SUMMIT.
    helper = Path(__file__).resolve().parents[1] / 'generalized_gxe/workflow.py'
    spec = importlib.util.spec_from_file_location('_installed_runtime_checks', helper)
    workflow = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = workflow
    spec.loader.exec_module(workflow)
    workflow._PRE_NUMERICAL_CPU_AFFINITY = cpus
    workflow._NUMPY_POOL_WORKER_AFFINITY = workers
    info, threads, placement = workflow.require_private_blis(gxeldcore)
    if not info['gemm_integrity_enabled'] or not info['gemm_checksum_enabled']:
        raise RuntimeError('native integrity checks are required')
    physical = []
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append(tuple((topology / name).read_text().strip()
                              for name in ('physical_package_id', 'core_id')))
    if len(set(physical)) != len(cpus):
        raise RuntimeError('distinct physical cores required')
    print(json.dumps(dict(phase='installed_runtime_placement', launch_cpu_ids=cpus,
        physical_cores=physical, numpy_worker_affinity=workers,
        threads=threads, placement=placement, thp_disabled=True,
        package_path=summit.__file__, native_path=gxeldcore.__file__,
        native_build=info)), flush=True)
    args = sys.argv[1:]
    if args[:1] == ['--stop-checkpoint']:
        from summit.prediction.checkpoint import SolverCheckpoint
        original = SolverCheckpoint.save

        def stop(self, state):
            original(self, state)
            if state['iteration'] >= 2:
                os._exit(75)

        SolverCheckpoint.save = stop
        args = args[1:]
    cli = prefix / 'bin/summit'
    sys.argv = [str(cli), *args]
    runpy.run_path(str(cli), run_name='__main__')


if __name__ == '__main__':
    main()
