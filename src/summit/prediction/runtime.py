"""Validate prediction worker placement before allocating genotype caches."""
import os
import re


def configure_prediction_threads(native, threads):
    # OMP can bind the caller to one CPU before BLIS creates pthread workers.
    # Register the explicit launcher contract so the existing protected BLIS
    # entry exposes the whole reserved team, then restores the calling thread.
    places = os.environ.get("OMP_PLACES", "")
    binding = os.environ.get("OMP_PROC_BIND", "false").strip().lower()
    if binding not in ("false", "0", ""):
        if not re.fullmatch(r"\{\d+\}(,\{\d+\})*", places):
            raise RuntimeError("Bound prediction workers require explicit singleton OMP_PLACES, "
                               "set before importing numerical libraries")
        cpus = [int(x) for x in re.findall(r"\d+", places)]
        if len(cpus) != threads or len(set(cpus)) != threads:
            raise RuntimeError("Prediction OMP_PLACES must match the requested thread count")
        native.configure_openmp_placement(cpus, threads)
    elif hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < threads:
        raise RuntimeError("Prediction thread request exceeds the calling-thread CPU affinity; "
                           "use an explicit process-start placement contract")
    native.configure_blas_threads(threads)
