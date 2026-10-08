"""Run a module from this checkout, bypassing editable-install redirection.

For private BLIS use scripts/generalized_gxe/private_python.py instead.
SUMMIT_NATIVE_DIR optionally selects a qualified portable native build.
"""
import importlib.util
import os
from pathlib import Path
import runpy
import sys

# Reject unsafe local placement before imports or BLAS warmup. Do not narrow
# here: callers must apply the assigned mask at the outer command boundary.
runpy.run_path(str(Path(__file__).resolve().parents[2] /
    'scripts/epistasis/cpu_policy.py'))['require_numerical_startup']()

if sys.platform.startswith("linux"):
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError("process-local THP guard failed")

root = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.meta_path[:] = [
    f for f in sys.meta_path if type(f).__module__ != "_gwldcore_editable"
]
sys.path[:0] = [str(root / "src"), str(root)]
import summit

assert Path(summit.__file__).resolve() == root / "src/summit/__init__.py"
if os.environ.get("SUMMIT_NATIVE_DIR"):
    native = Path(os.environ["SUMMIT_NATIVE_DIR"])
    for leaf in ("gxeldcore", "gwldcore", "winldcore"):
        candidates = list(native.glob(leaf + "*.so"))
        if len(candidates) != 1:
            raise RuntimeError(f"exactly one {leaf} extension required in {native}")
        spec = importlib.util.spec_from_file_location("summit." + leaf, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
if __name__ == "__main__":
    module = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_module(module, run_name="__main__")
