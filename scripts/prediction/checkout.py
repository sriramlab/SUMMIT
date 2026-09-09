#!/usr/bin/env python3
"""Run this checkout without altering a shared environment's editable install.

Usage: python scripts/prediction/checkout.py test [pytest arguments]
       python scripts/prediction/checkout.py benchmark [benchmark arguments]
       python scripts/prediction/checkout.py {plan,fit,score,scale,inspect} ...
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
# The existing scikit-build editable redirector precedes PYTHONPATH. Removing
# just this distribution's finder is process-local and preserves other installs.
sys.meta_path[:] = [finder for finder in sys.meta_path if type(finder).__module__ != "_gwldcore_editable"]
import summit
if Path(summit.__file__).resolve() != ROOT/"src/summit/__init__.py":
    raise RuntimeError("checkout bootstrap resolved the wrong SUMMIT package")

if len(sys.argv) > 1 and sys.argv[1] == "test":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import pytest
    args = sys.argv[2:]
    explicit_paths = any((ROOT/arg.split("::", 1)[0]).exists() for arg in args if not arg.startswith("-"))
    if not explicit_paths:
        args = [*[str(p) for p in sorted((ROOT/"tests").glob("test_prediction_*.py"))], *args]
    raise SystemExit(pytest.main([*args, "-p", "no:cacheprovider"]))
elif len(sys.argv) > 1 and sys.argv[1] in ("benchmark", "demo"):
    from importlib import import_module
    main = import_module(sys.argv[1]).main
    raise SystemExit(main(sys.argv[2:]))
else:
    from summit.prediction.cli import main
    raise SystemExit(main())
