"""Qualification-only native preload for an unchanged cross-trait subprocess.

Activate explicitly through PYTHONPATH when using staged portable extensions;
the production epistasis launcher itself already authenticates its imports.
"""
from pathlib import Path
import runpy
import sys

if sys.argv[0].endswith("cross_trait_study.py"):
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "checkout_python.py"),
        run_name="qualified_native_subprocess",
    )
