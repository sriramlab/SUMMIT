"""Run a module from this checkout, bypassing editable-install redirection.

For private BLIS use scripts/generalized_gxe/private_python.py instead.
SUMMIT_NATIVE_DIR optionally selects a qualified portable native build.
"""
import importlib.machinery
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
installed = importlib.util.find_spec("summit")


class CheckoutFinder:
    """Resolve checkout modules before any editable-install finder."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname != "summit" and not fullname.startswith("summit."):
            return None
        parent = root / "src"
        for part in fullname.split(".")[:-1]:
            parent /= part
        return importlib.machinery.PathFinder.find_spec(fullname, [str(parent)])


sys.meta_path.insert(0, CheckoutFinder)
sys.path[:0] = [str(root / "src"), str(root)]
spec = CheckoutFinder.find_spec("summit")
summit = importlib.util.module_from_spec(spec)
# Keep installed extensions available when no separate native build is given.
# Python modules are always resolved from the checkout by CheckoutFinder.
if installed is not None:
    summit.__path__.extend(installed.submodule_search_locations or ())
sys.modules["summit"] = summit
spec.loader.exec_module(summit)

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
