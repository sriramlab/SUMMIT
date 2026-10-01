#!/usr/bin/env python3
"""Reproduce manuscript figures and associated summaries from aggregate results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a package-relative path: {value}")
    return path


def validate(entries):
    for identifier, entry in entries.items():
        required = [entry["script"], *entry["inputs"]]
        outputs = [entry["output"], *entry.get("extra_outputs", [])]
        if "prepare" in entry:
            required.append(entry["prepare"]["script"])
            outputs.append(entry["prepare"]["output"])
        for name in required:
            if not (ROOT / relative_path(name)).is_file():
                raise FileNotFoundError(f"Figure {identifier}: {name}")
        for name in outputs:
            relative_path(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", nargs="+", help="Figure IDs, e.g. 4 S12 S29; default: all."
    )
    parser.add_argument(
        "--list", action="store_true", help="List figure IDs and output files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the manifest and inputs without plotting.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output",
        help="New or empty output directory (default: manuscript/output).",
    )
    args = parser.parse_args()
    entries = json.loads((ROOT / "figures.json").read_text())
    selected = args.figure or sorted(
        entries, key=lambda key: (key.startswith("S"), int(key.lstrip("S")))
    )
    unknown = set(selected) - set(entries)
    if unknown:
        parser.error(f"Unknown figure IDs: {', '.join(sorted(unknown))}")
    chosen = {key: entries[key] for key in selected}
    validate(chosen)
    if args.list or args.check:
        for identifier, entry in chosen.items():
            print(f"{identifier:>3}  {entry['output']}")
        if args.check:
            print(f"Checked inputs for {len(chosen)} figures.")
        return

    output = args.out_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(
            "Use a new or empty --out-dir; existing results are not overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "figs" / "main").mkdir(parents=True)
    (output / "figs" / "supplementary").mkdir()
    derived = {
        entry["prepare"]["output"] for entry in chosen.values() if "prepare" in entry
    }
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    completed = set()

    # The plotting code keeps its original relative paths. An isolated working
    # directory directs outputs to --out-dir and protects the supplied tables.
    with tempfile.TemporaryDirectory(prefix=".work-", dir=output) as name:
        work = Path(name)
        for folder in ("scripts", "metadata"):
            (work / folder).symlink_to(ROOT / folder, target_is_directory=True)
        (work / "figs").symlink_to(output / "figs", target_is_directory=True)
        for source in (ROOT / "data").rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(ROOT)
            destination = work / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if str(relative) in derived:
                shutil.copyfile(source, destination)
            else:
                destination.symlink_to(source)

        def execute(entry):
            command = (sys.executable, "-B", entry["script"], *entry.get("args", []))
            if command in completed:
                return
            subprocess.run(command, cwd=work, env=environment, check=True)
            completed.add(command)

        for identifier, entry in chosen.items():
            print(f"Reproducing Figure {identifier}", flush=True)
            if "prepare" in entry:
                execute(entry["prepare"])
            execute(entry)
            for expected in [entry["output"], *entry.get("extra_outputs", [])]:
                if not (work / expected).is_file():
                    raise RuntimeError(
                        f"Figure {identifier} did not produce {expected}"
                    )
        for relative in sorted(derived):
            destination = output / "derived" / Path(relative).name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(work / relative, destination)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
