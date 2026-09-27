#!/usr/bin/env python3
"""Test this checkout with explicitly selected, ABI-matching SUMMIT extensions.

The temporary bootstrap also reaches test subprocesses. It avoids an installed
editable package redirecting tests to another checkout; no environment files
or compiled extensions are modified.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", action="append", required=True, type=Path)
    parser.add_argument("tests", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    paths = [str(p.resolve(strict=True)) for p in args.extension_dir]
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="summit-pcgc-native-tests-") as tmp:
        bootstrap = ("import sys\n"
            "sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != '_gwldcore_editable']\n"
            "import summit\n"
            f"summit.__path__.extend({paths!r})\n")
        Path(tmp, "sitecustomize.py").write_text(bootstrap)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join([str(root/"src"), tmp, os.environ.get("PYTHONPATH", "")]))
        return subprocess.call([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *args.tests],
                               cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
