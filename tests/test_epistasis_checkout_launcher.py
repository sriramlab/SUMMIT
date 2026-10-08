"""The research launcher must honor its checkout across editable installs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("preloaded", [False, True])
def test_checkout_precedes_installed_package_and_editable_finder(tmp_path, preloaded):
    installed = tmp_path / "summit"
    installed.mkdir()
    (installed / "__init__.py").write_text('origin = "installed"\n')
    stale = installed / "epistasis.py"
    stale.write_text('raise AssertionError("stale installed module")\n')
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.util, runpy, sys
from pathlib import Path
root, installed = map(Path, sys.argv[1:3])
preloaded = sys.argv[3] == 'True'
class EditableFinder:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == 'summit':
            return importlib.util.spec_from_file_location(name, installed/'__init__.py')
        if name == 'summit.epistasis':
            return importlib.util.spec_from_file_location(name, installed/'epistasis.py')
sys.meta_path.insert(0, EditableFinder)
if preloaded:
    import summit
    assert summit.origin == 'installed'
runpy.run_path(str(root/'scripts/epistasis/checkout_python.py'))
import summit
assert Path(summit.__file__).resolve() == root/'src/summit/__init__.py'
assert str(installed) in summit.__path__
spec = importlib.util.find_spec('summit.epistasis')
assert Path(spec.origin).resolve() == root/'src/summit/epistasis/__init__.py'
"""
    env = dict(os.environ)
    env.pop("SUMMIT_NATIVE_DIR", None)
    prefix = []
    if hasattr(os, "sched_getaffinity"):
        cpus = getattr(sys.modules.get("workflow"), "_PRE_NUMERICAL_CPU_AFFINITY",
                       tuple(sorted(os.sched_getaffinity(0))))
        prefix = ["taskset", "-c", ",".join(map(str, cpus))]
    # A fresh isolated interpreter bypasses the local verification bootstrap.
    subprocess.run(
        prefix + [sys.executable, "-I", "-c", code, str(root), str(installed), str(preloaded)],
        env=env, check=True, capture_output=True, text=True,
    )
