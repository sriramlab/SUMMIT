"""Authenticated study accumulators at a completed SNP-tile boundary.

Checkpoints preserve the running FP64 sums, rather than regrouping fragment
sums. Resumption starts at the next unread variant. Published checkpoints
are immutable; an incomplete temporary publication is never a resume input.
"""
from pathlib import Path
import os
import uuid
import numpy as np

from .cross_trait_zpass import load_array_artifact, write_array_artifact
from .spec import canonical_sha256


def _arrays(batch, z):
    result=dict(block_ids=batch.scores.block_ids, pairs=batch.scores.pairs,
        block_rhs=batch.scores.rhs, block_masses=batch.scores.masses,
        block_genetic_residual=batch.genetic_residual,
        residual_gram=batch.residual_gram, residual_rhs=batch.residual_rhs)
    if z is not None:
        result['z_products']=z.products
    return result


def save_study_checkpoint(path, *, batch, z, identity, progress):
    """Publish only after closing/fsyncing a complete checksummed NPZ."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    write_array_artifact(temporary,kind='summit.cross_trait.study_checkpoint',
        arrays=_arrays(batch,z),provenance=dict(identity=identity,
            identity_sha256=canonical_sha256(identity),progress=progress))
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    # link is atomic and refuses an existing destination (unlike rename).
    os.link(temporary,path)
    temporary.unlink()


def restore_study_checkpoint(path, *, batch, z, identity):
    arrays,meta=load_array_artifact(path,kind='summit.cross_trait.study_checkpoint')
    if (meta['identity_sha256']!=canonical_sha256(identity)
            or canonical_sha256(meta['identity'])!=meta['identity_sha256']):
        raise ValueError('checkpoint study identity differs')
    targets=_arrays(batch,z)
    if set(arrays)!=set(targets):raise ValueError('checkpoint accumulator set differs')
    for name,target in targets.items():
        value=arrays[name]
        if value.shape!=target.shape or value.dtype!=target.dtype or not np.isfinite(value).all():
            raise ValueError(f'checkpoint accumulator dimensions/values differ: {name}')
        if name in ('block_ids','pairs'):
            np.testing.assert_array_equal(value,target)
        elif name in ('residual_gram','residual_rhs'):
            np.testing.assert_allclose(value,target,rtol=1e-12,atol=1e-10)
    progress=meta['progress'];position=0
    for row in progress['timings']:
        if row['begin']!=position or row['end']<=position:
            raise ValueError('checkpoint variant ranges overlap or have a gap')
        position=row['end']
    if (position!=progress['next_variant'] or not 0<position<=identity['variants']
            or (position<identity['variants'] and position%identity['width'])):
        raise ValueError('checkpoint variant cursor differs')
    if z is not None and (progress['z_variants']!=position
                         or progress['protected_tn_calls']!=len(progress['timings'])):
        raise ValueError('checkpoint guarded traversal ledger differs')
    # All validation precedes mutation of a live accumulator.
    for name,target in targets.items():
        if name not in ('block_ids','pairs'):target[...] = arrays[name]
    batch.scores.variants=progress['score_variants']
    batch.scores.seconds=progress['score_seconds']
    batch.residual_seconds=progress['residual_seconds']
    batch.residual_phase_seconds=dict(progress['residual_phase_seconds'])
    if z is not None:z.variants=progress['z_variants']
    return progress
