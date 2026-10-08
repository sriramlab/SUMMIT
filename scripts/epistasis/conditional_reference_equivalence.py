"""Verify cohort-reference equivalence for cross-host simulation collection.

This reads cohort-private references and a pinned, original-host genotype-content
receipt. It does not migrate models, solver state, covariance estimates or scores.
Only execution-location identities may differ. Scientific arrays must be exact;
small reduction differences are allowed only in outcome-independent HE geometry.
"""
import json
from pathlib import Path

import numpy as np

from summit.context.spec import array_sha256, canonical_sha256
from summit.prediction._validation import array_digest, digest, closed
from summit.prediction.artifacts import file_digest
from summit.epistasis.conditional_reference import (
    load_conditional_reference, MEAN_ARRAYS, GEOMETRY_ARRAYS)


GEOMETRY_RTOL = 2e-8


def _pinned(spec, root):
    closed(spec, ('file', 'sha256'), name='reference evidence')
    path = Path(root)/spec['file']
    if file_digest(path) != spec['sha256']:
        raise ValueError('reference evidence checksum changed')
    return path


def _reference(path):
    # Check the archive limit before NumPy reads any array, including manifest.
    from zipfile import ZipFile
    with ZipFile(path) as archive:
        if sum(v.file_size for v in archive.infolist()) > 384*2**20:
            raise MemoryError('reference comparison exceeds its bounded memory plan')
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest']))
    meta = manifest['metadata']
    arrays, checked = load_conditional_reference(path, identity=meta['identity'], memory_bytes=2**31)
    definition = dict(method='conditional_reference_v1',
        source=meta['mean']['source'], variants=array_sha256(arrays['variants']),
        mean={k:array_sha256(arrays[k]) for k in MEAN_ARRAYS}, definition=meta['mean'],
        probes=meta['geometry']['probes'], seed=meta['geometry']['seed'],
        exact=meta['geometry']['exact'])
    weighting=meta['geometry'].get('moment_weighting','none')
    if weighting not in ('none','genotype_diagonal'):
        raise ValueError('unknown covariance moment weighting')
    if weighting!='none':
        definition['moment_weighting']='genotype_diagonal_v1'
    identity = canonical_sha256(definition)
    # Exact (bounded) references store effective probe count rather than the
    # originally requested probe argument; they are not a cross-host use case.
    if meta['geometry']['exact']:
        raise ValueError('cross-host equivalence requires the declared stochastic reference')
    if identity != meta['identity']:
        raise ValueError('reference definition does not reconstruct its identity')
    i0 = arrays['training_index']
    operator = digest([meta['mean']['source'], array_digest(arrays['rows'][i0]),
        meta['scales']['samples'], array_digest(arrays['variants']),
        array_digest(arrays['mean']), array_digest(arrays['inverse_scale']),
        array_digest(arrays['contexts'][i0]), array_digest(arrays['noise'][i0]),
        meta['scales']['definition']])
    if (operator != meta['geometry']['operator_identity']
            or meta['scales']['source'] != meta['mean']['source']
            or meta['scales']['samples'] != meta['mean']['training_samples']):
        raise ValueError('reference covariance operator definition differs')
    if any(not np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError('nonfinite reference array')
    return arrays, checked


def verify_equivalence(spec, root):
    """Return evidence only after validating two explicitly pinned references."""
    closed(spec, ('references', 'source_transfer'), name='reference equivalence')
    if len(spec['references']) != 2:
        raise ValueError('declare exactly two references for an equivalence check')
    receipt = _pinned(spec['source_transfer'], root)
    transfer = json.loads(receipt.read_text())
    if (transfer.get('kind') != 'summit.epistasis.matched_simulation_reference_transfer'
            or transfer.get('schema_version') != 1):
        raise ValueError('original-host genotype content receipt required')
    content = transfer['source_content']
    if content['format'] != 'bed' or len(content['files']) != 3:
        raise ValueError('BED, BIM and FAM content evidence required')
    source_ids = {transfer['source_identity'], digest(['prediction_file_content_v1', content])}
    paths = [_pinned(item, root) for item in spec['references']]
    left, lm = _reference(paths[0]); right, rm = _reference(paths[1])
    if any(m['mean']['source'] not in source_ids for m in (lm, rm)):
        raise ValueError('reference source not authenticated by genotype content receipt')
    # The only permitted metadata differences are the three derived identities.
    def normalized(meta):
        value = json.loads(json.dumps(meta))
        value.pop('identity')
        value['mean'].pop('source'); value['scales'].pop('source')
        value['geometry'].pop('operator_identity')
        return value
    if normalized(lm) != normalized(rm):
        raise ValueError('reference scientific metadata differs')
    if set(left) != set(right):
        raise ValueError('reference array axes differ')
    scientific = sorted(set(left)-set(GEOMETRY_ARRAYS))
    for name in scientific:
        if left[name].dtype != right[name].dtype or not np.array_equal(left[name], right[name]):
            raise ValueError('reference scientific array differs: '+name)
    differences = {}
    for name in GEOMETRY_ARRAYS:
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('reference geometry axis differs: '+name)
        # Compare the finite-mean projection, allowing a different orthonormal
        # basis orientation. This forms only N by rank products, never N by N.
        if name == 'basis':
            eye = np.eye(a.shape[1])
            errors = [np.linalg.norm(v.T@v-eye, ord=2) for v in (a,b)]
            errors += [np.linalg.norm(a-b@(b.T@a), ord='fro')/max(1., np.sqrt(a.shape[1]))]
            error = max(errors)
        else:
            error = np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-300)
        if not np.isfinite(error) or error > GEOMETRY_RTOL:
            raise ValueError('reference numerical geometry differs: '+name)
        differences[name] = float(error)
    for item in (*spec['references'], spec['source_transfer']):
        _pinned(item, root)
    return dict(reference_ids=[lm['identity'],rm['identity']],
        files={str(p):file_digest(p) for p in (*paths,receipt)},
        exact_scientific_arrays=scientific, geometry_relative_differences=differences,
        geometry_tolerance=GEOMETRY_RTOL, source_content_identity=digest(content),
        scope='Collection equivalence only; no fitted quantity or artifact identity rewritten')
