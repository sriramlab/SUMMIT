"""Terminate an actual training process after its second atomic checkpoint.

Validation launcher only. No solver arithmetic or saved state is modified.
Exit 75 denotes the requested hard process interruption, not convergence.
"""
import os
import runpy
import sys
from summit.prediction.checkpoint import SolverCheckpoint
from summit.epistasis.cli import main

original = SolverCheckpoint.save


def stop(self, state):
    original(self, state)
    if state["iteration"] >= 2:
        os._exit(75)


if __name__ == "__main__":
    SolverCheckpoint.save = stop
    if len(sys.argv)>2 and sys.argv[1]=='--module':
        module=sys.argv[2]
        sys.argv=sys.argv[2:]
        runpy.run_module(module,run_name='__main__')
    else:
        main(sys.argv[1:])
    raise RuntimeError("fit completed before the requested interruption")
