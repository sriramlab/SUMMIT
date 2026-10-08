"""Cohort-private, outcome-independent conditional covariance references."""
from pathlib import Path
import json
from zipfile import ZipFile

import numpy as np

from summit.context.spec import array_sha256, canonical_sha256
from .conditional import frozen_local_mean
from .polygenic import PolygenicKernels, fit_kernel_scales, he_geometry
from .summary import _publish_bundle


GEOMETRY_ARRAYS = ('basis', 'gram', 'norms', 'h', 'probe_gram')
LEGACY_SKETCH_ARRAYS = ('preconditioner_products', 'preconditioner_probes')
MEAN_ARRAYS = ('rows', 'training_index', 'confirmation_index', 'fixed',
               'target', 'contexts', 'noise')
REFERENCE_ARRAYS = (*MEAN_ARRAYS, 'variants', 'mean', 'inverse_scale', *GEOMETRY_ARRAYS)


def load_conditional_reference(path, *, identity, memory_bytes):
    """Admit decompressed arrays before loading; verify all array definitions."""
    with ZipFile(path) as archive:
        stored = sum(v.file_size for v in archive.infolist())
    if 3*stored + 256*2**20 > memory_bytes:
        raise MemoryError('conditional genotype reference exceeds preparation memory')
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest']))
        version=manifest.get('schema_version')
        fields=((*REFERENCE_ARRAYS, *LEGACY_SKETCH_ARRAYS) if version==1 else
                (*REFERENCE_ARRAYS, 'moment_weights') if version==3 else REFERENCE_ARRAYS)
        if (manifest.get('kind') != 'summit.epistasis.conditional_reference'
                or version not in (1,2,3) or set(archive.files) != {'manifest', *fields}):
            raise ValueError('unsupported conditional genotype reference')
        weighting=manifest['metadata']['geometry'].get('moment_weighting','none')
        if (weighting not in ('none','genotype_diagonal')
                or (version==3)!=(weighting=='genotype_diagonal')):
            raise ValueError('conditional covariance moment weighting differs from its schema')
        if manifest['metadata']['identity'] != identity:
            raise ValueError('conditional reference sample, variant or mean inputs changed')
        arrays = {k: archive[k] for k in fields}
    if manifest['digests'] != {k: array_sha256(v) for k,v in arrays.items()}:
        raise ValueError('conditional reference array digest mismatch')
    for name in LEGACY_SKETCH_ARRAYS:arrays.pop(name,None)
    return arrays, manifest['metadata']


def prepare_conditional_reference(source, training_rows, confirmation_rows, variants,
                                  specification, root, *, path=None, threads=1,
                                  block_size=512, memory_bytes=4*2**30,
                                  probes=128, seed=871631, exact=False,
                                  moment_weighting='none'):
    """Return one frozen mean, training operator and covariance geometry.

    Phenotypes and learned directions do not enter this reference. Reuse checks
    actual aligned covariates, local genotypes, sample masks and source identity,
    not just their filenames. Scale fitting uses training participants only.
    """
    if moment_weighting not in ('none','genotype_diagonal'):
        raise ValueError('unknown covariance moment weighting')
    geometry_fields=(*GEOMETRY_ARRAYS,'moment_weights') if moment_weighting!='none' else GEOMETRY_ARRAYS
    mean = frozen_local_mean(source, training_rows, confirmation_rows, specification,
        root, threads=threads, memory_bytes=memory_bytes)
    variants = np.asarray(variants)
    definition = dict(method='conditional_reference_v1',
        source=source.identity, variants=array_sha256(variants),
        mean={k:array_sha256(mean[k]) for k in MEAN_ARRAYS},
        definition=mean['metadata'], probes=probes, seed=seed, exact=exact)
    if moment_weighting!='none':
        definition['moment_weighting']='genotype_diagonal_v1'
    identity = canonical_sha256(definition)
    path = None if path is None else Path(path)
    reuse = path is not None and path.exists()
    i0 = mean['training_index']
    if reuse:
        arrays, metadata = load_conditional_reference(path, identity=identity, memory_bytes=memory_bytes)
        scales = dict(metadata['scales'], mean=arrays['mean'], inverse_scale=arrays['inverse_scale'])
        geometry = dict(metadata['geometry'], **{k:arrays[k] for k in geometry_fields})
    else:
        scales = fit_kernel_scales(source, training_rows, variants, threads=threads,
            block_size=block_size, memory_bytes=memory_bytes)
    operator = PolygenicKernels(source, training_rows, variants, scales,
        mean['contexts'][i0], mean['noise'][i0], threads=threads,
        block_size=block_size, memory_bytes=memory_bytes, storage='packed')
    if reuse:
        if geometry['operator_identity'] != operator.identity:
            raise ValueError('conditional reference covariance operator changed')
        # The manifest identity binds these arrays to the freshly checked mean.
        for k in MEAN_ARRAYS:
            if not np.array_equal(arrays[k], mean[k]):
                raise ValueError('conditional reference finite mean changed')
    else:
        geometry = he_geometry(operator, mean['fixed'][i0], probes=probes, seed=seed,
            exact=exact, keep_products=False,moment_weighting=moment_weighting)
        # The sketch inverse was slower than Jacobi in the measured comparison.
        # Do not retain unused sketch products in public cohort references.
        for name in LEGACY_SKETCH_ARRAYS:geometry.pop(name,None)
        arrays = {k:mean[k] for k in MEAN_ARRAYS}
        arrays.update(variants=variants, mean=scales['mean'], inverse_scale=scales['inverse_scale'],
                      **{k:geometry[k] for k in geometry_fields})
        metadata = dict(identity=identity, mean=mean['metadata'],
            scales={k:v for k,v in scales.items() if k not in ('mean','inverse_scale')},
            geometry={k:v for k,v in geometry.items() if k not in geometry_fields},
            kernel_names=['additive', *['additive_by_'+v for v in
                specification['covariates'].get('varying_effects',[])],
                'dominance', 'noise_iid', 'noise_target_squared'],
            scope='cohort-only genotype arrays; no outcomes or fitted covariance')
        if path is not None:
            _publish_bundle(path, dict(kind='summit.epistasis.conditional_reference',
                schema_version=3 if moment_weighting!='none' else 2, metadata=metadata,
                digests={k:array_sha256(v) for k,v in arrays.items()}), arrays)
    return dict(mean=mean, scales=scales, geometry=geometry, operator=operator,
                identity=identity, reused=reuse, metadata=metadata)
