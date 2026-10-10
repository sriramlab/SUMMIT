"""Native-backed array entry point for the continuous finite-mean scale test."""
import json
from pathlib import Path
import zipfile

import numpy as np

from .robust import prepare_robust_geometry
from .scale import boxcox_scale_test


def protected_scale_products(threads):
    """Use the existing shared NN/TN implementation; no new native backend."""
    from summit.ldscore.matrix_products import MatrixProducts
    products = MatrixProducts(native=True, threads=threads)
    products.left.begin_execution()
    return products.nn, products.tn, products.left


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
    if f.ndim != 2 or c.ndim != 2 or y.shape != (len(f),) or len(c) != len(f):
        raise ValueError('sample-aligned feature/fixed matrices and one phenotype required')
    if type(max_evaluations) is not int or max_evaluations < 3:
        raise ValueError('at least three integer scale evaluations required')
    # Four covariance matrices per cached power (outcome + three derivatives),
    # each requested block inverse, simultaneous batch meat/covariance, and
    # transformed-outcome work. Include overlapping user-supplied groups.
    # ZIP expansion alone misses the quadratic feature-space search cache.
    n,p = f.shape
    group_sizes = [p] if groups is None else [np.asarray(v).size for v in groups.values()]
    planned = 6*expanded+256*2**20+8*((4*max_evaluations+72)*p*p
        +max_evaluations*sum(q*q+q for q in group_sizes)
        +192*n+8*max_evaluations*p)
    if planned > memory_bytes:
        raise MemoryError('scale geometry and continuous-search covariance cache exceed the declared memory budget')
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


def run_honest_scale_arrays(path, output, *, groups=None, threads=1,
                           memory_bytes=4*2**30, bounds=(-2.,2.), alpha=.05,
                           gamma=None, max_pilot_evaluations=257,max_evaluations=129):
    """Aligned NPZ: features, fixed_effects, phenotype, sample_ids, pilot_mask.

    One common feature/nuisance encoding is split by a frozen Boolean mask.
    Outcome-dependent feature learning must use an independent third sample.
    """
    from .scale_honest import boxcox_honest_scale_test
    from summit.prediction.artifacts import file_digest
    path,output=Path(path),Path(output)
    if output.exists():raise FileExistsError(output)
    input_digest=file_digest(path)
    for count in (max_pilot_evaluations,max_evaluations):
        if type(count) is not int or count<3:raise ValueError('at least three evaluations required')
    with zipfile.ZipFile(path) as archive:
        expanded=sum(item.file_size for item in archive.infolist())
    if 8*expanded+256*2**20>memory_bytes:
        raise MemoryError('honest scale inputs and split geometry exceed memory budget')
    with np.load(path,allow_pickle=False) as archive:
        f,c,y,ids,mask=(archive[k] for k in ('features','fixed_effects','phenotype','sample_ids','pilot_mask'))
    if (f.ndim!=2 or c.ndim!=2 or len(f)!=len(c) or y.shape!=(len(f),)
            or ids.shape!=(len(f),) or mask.shape!=(len(f),) or mask.dtype!=np.bool_
            or not mask.any() or mask.all()):
        raise ValueError('aligned arrays and a nontrivial Boolean pilot mask required')
    n,p=f.shape
    sizes=[p] if groups is None else [np.asarray(v).size for v in groups.values()]
    cache_count=max(max_pilot_evaluations,max_evaluations)
    planned=8*expanded+256*2**20+8*((4*cache_count+72)*p*p
        +cache_count*sum(q*q+q for q in sizes)+192*n+8*cache_count*p+2*n*p)
    if planned>memory_bytes:raise MemoryError('honest scale search cache exceeds memory budget')
    nn,tn,products=protected_scale_products(threads)
    pilot=prepare_robust_geometry(f[mask],c[mask],nn=nn,tn=tn)
    confirmation=prepare_robust_geometry(f[~mask],c[~mask],nn=nn,tn=tn)
    result=boxcox_honest_scale_test(pilot,y[mask],confirmation,y[~mask],
        pilot_ids=ids[mask],confirmation_ids=ids[~mask],groups=groups,bounds=bounds,
        alpha=alpha,gamma=gamma,max_pilot_evaluations=max_pilot_evaluations,max_evaluations=max_evaluations)
    if file_digest(path)!=input_digest:raise ValueError('honest scale input changed during analysis')
    native=products.finish_execution()
    result.update(input_sha256=input_digest,feature_shape=f.shape,fixed_shape=c.shape,
        planned_memory_bytes=planned,native_execution={k:native[k] for k in
            ('gemm_status','output_numa_status','gemm_record_count')})
    from .cli import _jsonable
    with output.open('x') as handle:
        json.dump(_jsonable(result),handle,indent=2,allow_nan=False);handle.write('\n')
    return result
