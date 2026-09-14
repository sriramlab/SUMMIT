"""Respect the native process-wide thread contract in a mixed test session."""
import os


def prediction_threads():
    from summit.prediction.genotype import native_module
    native = native_module()
    info = native.build_info()
    if info.get("blas_runtime_environment_immutable", False):
        desired = int(info["blas_runtime_threads"])
    else:
        desired = int(native.configured_blas_threads()) or min(2, len(os.sched_getaffinity(0)))
    return int(native.configure_blas_threads(desired))
