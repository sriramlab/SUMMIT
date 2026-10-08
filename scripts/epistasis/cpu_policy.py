"""Tabla exclusion and thread checks; standard library only, before numerics.

This rejects unsafe placement. It never expands or changes a process mask.
The hostname rule is local to Tabla, not a Hoffman allocation rule.
"""
import os
import re
from pathlib import Path
import socket

TABLA_EXCLUDED = frozenset(range(8)) | frozenset(range(64, 72))
TABLA_PHYSICAL = frozenset(range(8, 64))


def validate_cpus(cpus, *, hostname=None, compute=True):
    host = (hostname or socket.gethostname()).split('.')[0].lower()
    selected = set(cpus)
    if host != 'tabla':
        return
    if not selected or selected & TABLA_EXCLUDED:
        raise ValueError('Tabla excludes CPUs 0-7 and 64-71; apply outer taskset before launch')
    if compute and (not selected <= TABLA_PHYSICAL or len(selected) > 8):
        raise ValueError('Tabla compute requires at most eight assigned physical CPUs in 8-63')


def thread_masks(pid=None):
    pid = os.getpid() if pid is None else pid
    result = {}
    try:
        tasks = list(Path(f'/proc/{pid}/task').iterdir())
    except FileNotFoundError:
        return result
    for task in tasks:
        try:
            result[int(task.name)] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    return result


def require_affinity(assigned=None):
    masks = thread_masks()
    for cpus in masks.values():
        validate_cpus(cpus)
        if assigned is not None and not set(cpus) <= set(assigned):
            raise ValueError('thread affinity escaped assigned CPUs')
    return masks


def descendants(pid):
    """Owned process tree, including descendants with a different process group."""
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            raw = (entry/'stat').read_text()
            fields = raw[raw.rfind(')')+2:].split()
            if fields[0] not in ('Z', 'X'):
                parents[int(entry.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            pass
    found = {pid}
    while True:
        added = {child for child, parent in parents.items() if parent in found} - found
        if not added:
            return found & parents.keys()
        found.update(added)


def require_tree_affinity(pid, assigned):
    for child in descendants(pid):
        for cpus in thread_masks(child).values():
            validate_cpus(cpus)
            if not set(cpus) <= set(assigned):
                raise ValueError('descendant thread affinity escaped its job allocation')


def require_numerical_startup(*, hostname=None, environment=None):
    """Reject placement overrides before a numerical library can act on them.

    Explicit singleton OpenMP places are the supported bound configuration.
    No environment or affinity is changed here. Hoffman retains its independent
    scheduler-placement rules; Tabla additionally checks inherited overrides.
    """
    masks = require_affinity()
    host = (hostname or socket.gethostname()).split('.')[0].lower()
    if host != 'tabla':
        return masks
    selected = set(os.sched_getaffinity(0))
    env = os.environ if environment is None else environment
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
        value = env.get(name)
        if value is not None and (not value.isdecimal() or not 1 <= int(value) <= len(selected)):
            raise ValueError(name+' exceeds the Tabla CPU allocation or is not a positive integer')
    places = env.get('OMP_PLACES', '').strip()
    if places:
        if not re.fullmatch(r'\{[0-9]+\}(?:\s*,\s*\{[0-9]+\})*', places):
            raise ValueError('Tabla OMP_PLACES must list explicit singleton assigned CPUs')
        cpus = [int(v) for v in re.findall(r'[0-9]+', places)]
        if len(cpus) != len(set(cpus)) or not set(cpus) <= selected:
            raise ValueError('OMP_PLACES escaped the Tabla CPU allocation')
    affinity = env.get('GOMP_CPU_AFFINITY', '').strip()
    if affinity:
        # Reject complex inherited placement rather than expanding any mask.
        # Our launchers use OMP_PLACES and remove this competing override.
        tokens = affinity.split()
        if any(not v.isdecimal() for v in tokens) or not {int(v) for v in tokens} <= selected:
            raise ValueError('GOMP_CPU_AFFINITY must list only assigned Tabla CPUs')
    if env.get('KMP_AFFINITY', '').strip() or env.get('KMP_HW_SUBSET', '').strip():
        raise ValueError('remove unverified Intel OpenMP placement overrides on Tabla')
    return masks
