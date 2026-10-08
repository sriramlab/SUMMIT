"""Attest Python/native paths and process worker placement after spectral work."""
import json
import os
from pathlib import Path
import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info
from summit.prediction.genotype import native_module
import summit

x=np.eye(512)
scipy.linalg.eigh(x,check_finite=False)
native=native_module()
record=dict(python_package=summit.__file__,native_path=native.__file__,native_build=native.build_info(),
    pools=threadpool_info(),worker_affinities={p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()})
print(json.dumps(record),flush=True)
