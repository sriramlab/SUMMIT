"""Native-backed array entry point for the continuous finite-mean scale test."""
import json
from pathlib import Path
import zipfile

import numpy as np

from .robust import prepare_robust_geometry
from .scale import boxcox_scale_test


def protected_scale_products(threads):
    """Use the existing shared NN/TN implementation; no new native backend."""
    from summit.prediction.genotype import native_module
    from summit.prediction.runtime import configure_prediction_threads
    from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
    from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
    native = native_module()
    configure_prediction_threads(native, threads)
    left = ProtectedNNOperator(threads=threads, native_module=native)
    right = ProtectedTNOperator(threads=threads, native_module=native)
    left.begin_execution(); right.begin_execution()
    return (lambda a,b:left.matmul(np.asfortranarray(a),np.asfortranarray(b)),
            lambda a,b:right.matmul_tn(np.asfortranarray(a),np.asfortranarray(b)),left)


def run_scale_arrays(path, output, *, groups=None, threads=1, memory_bytes=4*2**30,
                     bounds=(-2.,2.),alpha=.05,max_evaluations=129):
    """NPZ requires aligned features, fixed_effects, and one positive phenotype.

    The caller supplies scientifically frozen rows/features. This entry point
    performs no variant or phenotype-dependent feature selection.
    """
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    from summit.prediction.artifacts import file_digest
    input_digest = file_digest(path)
    with zipfile.ZipFile(path) as archive:
        expanded = sum(item.file_size for item in archive.infolist())
    if 6*expanded+256*2**20 > memory_bytes:
        raise MemoryError('scale input and reusable geometry exceed the declared memory budget')
    with np.load(path,allow_pickle=False) as archive:
        f,c,y = (archive[k] for k in ('features','fixed_effects','phenotype'))
    nn,tn,products=protected_scale_products(threads)
    geometry=prepare_robust_geometry(f,c,nn=nn,tn=tn)
    result=boxcox_scale_test(geometry,y,groups=groups,bounds=bounds,
        alpha=alpha,max_evaluations=max_evaluations)
    if file_digest(path) != input_digest:
        raise ValueError('scale input changed during analysis')
    result.update(input_sha256=input_digest,groups=groups,
        feature_shape=f.shape,fixed_shape=c.shape,native_execution=products.finish_execution())
    from .cli import _jsonable
    with output.open('x') as handle:
        json.dump(_jsonable(result),handle,indent=2,allow_nan=False);handle.write('\n')
    return result
