#!/usr/bin/env python3
"""Flag participant inputs and credentials in tracked files or Git history.

This check supplements manual review of data provenance. It reports filenames
and rule names without printing matched values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


DATA_SUFFIXES = (
    ".bed", ".bim", ".fam", ".pgen", ".pvar", ".pvar.zst", ".psam",
    ".bgen", ".sample", ".cov", ".covar", ".pheno", ".keep",
    ".sumstat", ".sumstats", ".sumstats.gz",
)
RULES = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}"),
    "numeric participant row": re.compile(rb"^\s*([0-9]{6,9})[ \t]+\1(?:[ \t]|$)", re.M),
}


def findings(path: str, content: bytes) -> list[str]:
    result = []
    if path.endswith(DATA_SUFFIXES):
        result.append("dataset file; generate synthetic examples locally")
    if path.endswith(".npz") and not path.startswith("tests/fixtures/"):
        result.append("array bundle outside the reviewed synthetic fixtures")
    for label, pattern in RULES.items():
        if pattern.search(content):
            result.append(label)
    return result


def check_worktree(root: Path) -> tuple[int, set[tuple[str, str]]]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")[:-1]
    found = set()
    checked = 0
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        checked += 1
        found.update((name, issue) for issue in findings(name, path.read_bytes()))
    return checked, found


def check_history(root: Path) -> tuple[int, set[tuple[str, str]]]:
    objects = {}
    for line in subprocess.check_output(["git", "rev-list", "--objects", "--all"], cwd=root).decode().splitlines():
        obj, _, name = line.partition(" ")
        objects[obj] = name
    checked, found = 0, set()
    # Stream objects to avoid holding the repository's full contents in memory.
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        for obj, name in objects.items():
            proc.stdin.write((obj + "\n").encode())
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            size = int(header[2])
            content = proc.stdout.read(size)
            proc.stdout.read(1)
            if header[1] == b"blob":
                checked += 1
                found.update((name, issue) for issue in findings(name, content))
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Git object reader failed")
    return checked, found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="inspect blobs reachable from every local ref")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    checked, found = check_history(root) if args.history else check_worktree(root)
    print(f"Checked {checked} {'history blobs' if args.history else 'tracked files'}; {len(found)} findings.")
    for name, issue in sorted(found):
        print(f"{name}: {issue}")
    return int(bool(found))


if __name__ == "__main__":
    raise SystemExit(main())
