#!/usr/bin/env python3
"""Run bounded GxE PCGC verification with pre-import placement and THP checks.

Invoke under taskset and process-start BLAS/OpenMP settings. The first argument
is portable/private, the second is an extension directory; remaining arguments
are a Python module and its arguments (for example pytest -q ...). This wrapper
does not alter an existing environment, extension, or another process.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import tempfile
import threading

def main():
    mode, extension_dir, module, *args = sys.argv[1:]
    if mode not in ("portable", "private"):
        raise ValueError("choose portable or private")
    cpus = set(os.sched_getaffinity(0))
    if os.uname().nodename.lower() == "tabla":
        if not cpus or not cpus <= set(range(8,64)):
            raise RuntimeError("Tabla verification requires allocated physical CPUs 8-63")
    limit = 2 if os.uname().nodename.lower() == 'tabla' else 8
    if not 1 <= len(cpus) <= limit:
        raise RuntimeError(f"this launcher admits at most {limit} physical cores on this host")
    threads = len(cpus)
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","OMP_THREAD_LIMIT","BLIS_NUM_THREADS","MKL_NUM_THREADS"):
        allowed = {str(threads)}
        if mode == 'private' and name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            allowed.add('1')
        if os.environ.get(name) not in allowed:
            raise RuntimeError(f"{name} must equal allocated CPU count")
    physical = {(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text().strip(),
                 Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text().strip()) for c in cpus}
    if len(physical) != threads:
        raise RuntimeError('allocation must contain distinct physical cores')
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "1")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41,1,0,0,0) != 0 or libc.prctl(42,0,0,0,0) != 1:
        raise RuntimeError("process-local THP guard failed")
    root = Path(__file__).resolve().parents[2]
    extensions = Path(extension_dir).resolve(strict=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stop = threading.Event()
    def check_tree():
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != boot:
            raise RuntimeError("boot identity changed")
        pending, seen = [os.getpid()], set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                tasks = list(Path(f"/proc/{pid}/task").iterdir())
                for task in tasks:
                    try:
                        if not set(os.sched_getaffinity(int(task.name))) <= cpus:
                            raise RuntimeError(f"thread {task.name} escaped allocated CPUs {sorted(cpus)}")
                        pending.extend(map(int,(task/"children").read_text().split()))
                    except (FileNotFoundError,ProcessLookupError):
                        pass
            except (FileNotFoundError,ProcessLookupError):
                pass
        return len(seen)
    def monitor():
        while not stop.wait(1.):
            try:
                check_tree()
            except Exception as exc:
                print(f"runtime placement failure: {exc}", file=sys.stderr, flush=True)
                os._exit(70)
    check_tree()
    threading.Thread(target=monitor, daemon=True).start()
    # A temporary bootstrap reaches test subprocesses without changing installed
    # packages. Keep native modules explicitly selected and prevent editable hooks.
    bootstrap = ("import sys\n"
        "sys.dont_write_bytecode=True\n"
        "sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']\n"
        "import summit\n"
        f"summit.__path__.append({str(extensions)!r})\n")
    with tempfile.TemporaryDirectory(prefix="summit-gxe-pcgc-bootstrap-") as tmp:
        Path(tmp,"sitecustomize.py").write_text(bootstrap)
        os.environ["PYTHONPATH"] = os.pathsep.join([tmp,str(root/"src"),str(root/"tests")])
        sys.path[:0] = [str(root/"src"),str(root/"tests"),str(root)]
        exec(bootstrap)
        import summit
        if Path(summit.__file__).resolve() != root/"src/summit/__init__.py":
            raise RuntimeError("wrong Python checkout")
        import numpy as np
        warm = np.ones((512,512)); assert (warm@warm)[0,0] == 512
        del warm
        numpy_workers = {int(p.name):sorted(os.sched_getaffinity(int(p.name)))
            for p in Path('/proc/self/task').iterdir() if int(p.name) != os.getpid()}
        if any(mask != sorted(cpus) for mask in numpy_workers.values()):
            raise RuntimeError('NumPy workers did not inherit the allocated CPU set before OpenMP initialization')
        from summit import gxeldcore
        from summit.prediction.runtime import configure_prediction_threads
        configure_prediction_threads(gxeldcore,threads)
        info = gxeldcore.build_info()
        if mode == "private":
            if info.get("private_blas_backend") != "upstream_blis":
                raise RuntimeError("wrong private backend")
            if not (info.get("gemm_integrity_enabled") and info.get("gemm_checksum_enabled")):
                raise RuntimeError("private verification requires GEMM integrity and checksum checks")
            sys.path.insert(0,str(root/"scripts/generalized_gxe"))
            import workflow
            workflow._PRE_NUMERICAL_CPU_AFFINITY = tuple(sorted(cpus))
            workflow._NUMPY_POOL_WORKER_AFFINITY = numpy_workers
            workflow.require_private_blis(gxeldcore)
        elif info.get("blas_vendor") != "OpenBLAS":
            raise RuntimeError("wrong portable backend")
        check_tree()
        print(f"verification: boot={boot} cpus={sorted(cpus)} backend={info['blas_vendor']} native={gxeldcore.__file__}", flush=True)
        sys.argv = [module,*args]
        try:
            if module == "-c":
                if len(args) != 1:
                    raise ValueError("-c requires exactly one Python program")
                exec(compile(args[0], "<verification-command>", "exec"), {"__name__": "__main__"})
            else:
                runpy.run_module(module,run_name="__main__")
        finally:
            check_tree()
            stop.set()


if __name__ == "__main__":
    main()
