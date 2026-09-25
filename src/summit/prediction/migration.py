"""Authenticated warm starts across mixture implementations, never identity rewrites.

Export on the original source host, or supply independently hashed descriptor
metadata attested there, so its strict legacy identity can be reconstructed.
Snapshot one atomic checkpoint generation by hard link, or copy
one open descriptor across filesystems. The writer keeps replacing its own path.
Import validates file content
and every scientific input, then returns weights for ordinary ``initial_weights``.
Residuals and convergence certificates are not transplanted into a new solver.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import os
import errno
import shutil
from types import SimpleNamespace

import numpy as np

from ._validation import array_digest, canonical, digest
from .artifacts import file_digest, _sync_directory
from .batch import plan_prediction
from .mixture import SeparateSparsitySpec


def _source_content(source):
    """Hash the already-open input descriptors without changing their offsets."""
    source.check()
    if getattr(source,'_content_json',None) is not None:
        return json.loads(source._content_json)
    if hasattr(source, '_fds'):
        records=[]
        for fd in source._fds:
            before=os.fstat(fd)
            h=hashlib.sha256()
            offset=0
            while offset<before.st_size:
                chunk=os.pread(fd,min(8*2**20,before.st_size-offset),offset)
                if not chunk:
                    raise ValueError('genotype descriptor shortened while authenticating')
                h.update(chunk)
                offset+=len(chunk)
            if source._state(os.fstat(fd)) != source._state(before):
                raise ValueError('genotype descriptor changed while authenticating')
            records.append(dict(bytes=before.st_size,sha256=h.hexdigest()))
        result=dict(format=source.input.format,files=records)
    elif hasattr(source, 'values'):
        result=dict(format='array',calls=array_digest(source.values),hard_calls=source.hard_calls)
    else:
        raise TypeError('migration requires a file or array genotype source')
    source.check()
    return dict(result,samples=digest(source.samples),variants=source.variants.identity)


def scientific_contract(traits, mixtures, source):
    """Exact content identity, excluding paths, device IDs and execution options."""
    records=[]
    for t in traits:
        records.append(dict(id=t.id,
            arrays={name:array_digest(getattr(t,name)) for name in ('rows','variants','y','phi','fixed')},
            scale=dict(mean=array_digest(t.scale.mean),inverse_scale=array_digest(t.scale.inverse_scale),
                ddof=t.scale.ddof,arithmetic=t.scale.arithmetic,
                provenance={k:v for k,v in t.scale.provenance.items() if k!='source'},
                variant_identity=t.scale.variant_identity,sample_identity=t.scale.sample_identity),
            context=t.context_spec,fixed=t.fixed_spec,phenotype=t.phenotype_spec,
            geometry=None if t.geometry is None else dict(omega=array_digest(t.geometry.omega),
                metric=array_digest(t.geometry.metric),reference=t.geometry.reference,anchor=t.geometry.anchor),
            candidates=[dict(id=c.id,covariance=array_digest(c.covariance),residual=array_digest(c.residual),
                specification=c.specification,annotation=None if c.annotation_prior is None else c.annotation_prior.identity,
                mixture=dict(kind=type(mixtures[t.id,c.id]).__name__,**asdict(mixtures[t.id,c.id])))
                for c in t.candidates]))
    return dict(source=_source_content(source),traits=records)


def _groups(traits):
    groups=[]
    for t in traits:
        surfaces={}
        for c in t.candidates:
            surfaces.setdefault(array_digest(c.residual),[]).append(c)
        for candidates in surfaces.values():
            groups.append((t,candidates))
    return groups


def _read_weights(path, expected_identity, traits):
    result={}
    with np.load(path,allow_pickle=False) as data:
        raw=data['metadata']
        if raw.dtype!=np.uint8 or raw.ndim!=1 or raw.nbytes>16*2**20:
            raise ValueError('invalid legacy checkpoint metadata')
        meta=json.loads(raw.tobytes())
        if meta.get('schema')!=1 or meta.get('kind')!='mixture' or meta.get('identity')!=expected_identity:
            raise ValueError('legacy checkpoint identity mismatch')
        if type(meta.get('sweep')) is not int or meta['sweep']<1:
            raise ValueError('legacy checkpoint has no completed sweep')
        if not isinstance(meta.get('elapsed'),(int,float)) or not np.isfinite(meta['elapsed']) or meta['elapsed']<0:
            raise ValueError('invalid legacy checkpoint elapsed time')
        expected=set()
        for j,(t,candidates) in enumerate(_groups(traits)):
            k,m,q,n=len(candidates),len(t.variants),t.phi.shape[1],len(t.rows)
            for field,shape in (('weights',(k,m,q)),('residual',(n,k)),('penalty',(k,m))):
                name=f'{field}_{j}'
                expected.add(name)
                value=data[name]
                record=meta['arrays'][name]
                if (value.dtype!=np.float64 or value.shape!=shape or record['shape']!=list(shape)
                        or not np.isfinite(value).all() or array_digest(value)!=record['sha256']):
                    raise ValueError('legacy checkpoint shape/content mismatch: '+name)
                if field=='weights':
                    for i,c in enumerate(candidates):
                        result[t.id,c.id]=value[i]
        if set(meta['arrays'])!=expected or set(data.files)!=expected|{'metadata'}:
            raise ValueError('legacy checkpoint has unexpected arrays')
    return result,meta


def _snapshot_checkpoint(checkpoint, snapshot):
    """Retain one atomic generation, including when its pathname is replaced."""
    try:
        os.link(checkpoint,snapshot)
    except OSError as error:
        if error.errno!=errno.EXDEV:
            raise
        # Open once: following the pathname again could switch generations.
        with open(checkpoint,'rb') as source, open(snapshot,'xb') as target:
            before=os.fstat(source.fileno())
            shutil.copyfileobj(source,target,length=8*2**20)
            target.flush()
            os.fsync(target.fileno())
            after=os.fstat(source.fileno())
            if ((before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)
                    or target.tell()!=before.st_size):
                raise ValueError('legacy checkpoint changed during snapshot copy')


def _legacy_source_view(traits, source, attestation, expected_sha256, expected_host, expected_job):
    """Reconstruct a shared-file identity from original-host descriptor evidence.

    This view is used only for legacy metadata authentication, never for reads or
    fitting. The live source keeps its original identity and mutation guards.
    """
    payload=Path(attestation).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('source attestation hash differs')
    record=json.loads(payload)
    if (record.get('kind')!='legacy_genotype_descriptor_attestation' or record.get('schema')!=1
            or record.get('host')!=expected_host or record.get('job')!=expected_job):
        raise ValueError('source attestation host/job differs')
    source.check()
    if not hasattr(source,'_fds') or record.get('paths')!=list(source._paths):
        raise ValueError('source attestation paths differ')
    states=record.get('states')
    if not isinstance(states,list) or len(states)!=len(source._fds):
        raise ValueError('invalid source attestation descriptor states')
    for state,fd in zip(states,source._fds):
        if (not isinstance(state,list) or len(state)!=5
                or any(type(v) is not int or v<0 for v in state)
                or state[1:]!=source._state(os.fstat(fd))[1:]):
            raise ValueError('shared source inode/size/timestamps differ')
    identity=digest([source.input.format,list(source._paths),states,digest(source.samples),source.variants.identity])
    view=SimpleNamespace(identity=identity,hard_calls=source.hard_calls,
        samples=source.samples,variants=source.variants)
    legacy_traits=tuple(replace(t,scale=replace(t.scale,
        provenance={**t.scale.provenance,'source':identity})) for t in traits)
    source.check()
    return legacy_traits,view,record


def export_legacy_checkpoint(checkpoint, destination, traits, mixtures, source, *,
                             legacy_python, legacy_native, legacy_solver,
                             protocol_fit_identity, block_size, threads, storage='compact',
                             source_attestation=None, attestation_sha256=None,
                             original_host=None, original_job=None):
    """Create a new immutable generation and receipt; fail closed on any mismatch.

    ``protocol_fit_identity`` must come from the original run's PROTOCOL.json.
    ``legacy_python`` is its frozen prediction package directory. No checkpoint
    metadata is edited, and no lock held by the running solver is acquired.
    """
    traits=tuple(traits)
    identity_traits,identity_source,attestation_record=traits,source,None
    if source_attestation is not None:
        if not attestation_sha256 or not original_host or type(original_job) is not int:
            raise ValueError('source attestation requires independently recorded hash, host and job')
        identity_traits,identity_source,attestation_record=_legacy_source_view(
            traits,source,source_attestation,attestation_sha256,original_host,original_job)
    expected_controls={'rtol','atol','max_sweeps','block_sweeps','qr_rtol','residual_refresh'}
    if set(legacy_solver)!=expected_controls:
        raise ValueError('unsupported legacy solver controls')
    options=dict(storage=storage,block_size=block_size,threads=threads,memory_bytes=1024*2**30)
    if plan_prediction(identity_traits,identity_source,**options).fit_identity!=protocol_fit_identity:
        raise ValueError('legacy preparation does not match original protocol')
    enriched=[]
    for t in identity_traits:
        candidates=[]
        for c in t.candidates:
            if 'mixture' in c.specification:
                raise ValueError('migration requires unenriched candidate specifications')
            mix=mixtures[t.id,c.id]
            specification=dict(c.specification,mixture=dict(
                kind='separate_sparsity' if isinstance(mix,SeparateSparsitySpec) else 'covariance_preserving_two_normal',
                **asdict(mix),fixed_effects='integrated'))
            candidates.append(replace(c,specification=specification))
        enriched.append(replace(t,candidates=tuple(candidates)))
    python_files={p.name:file_digest(p) for p in sorted(Path(legacy_python).glob('*.py'))}
    if not {'mixture.py','batch.py','spec.py','genotype.py'}<=python_files.keys():
        raise ValueError('legacy prediction package is incomplete')
    expected_identity=digest(dict(kind='mixture_v3_native_state',
        fit=plan_prediction(enriched,identity_source,**options).fit_identity,solver=legacy_solver,
        python=digest(python_files),native=file_digest(legacy_native),
        block_size=block_size,threads=threads,storage=storage))
    destination=Path(destination)
    destination.mkdir(mode=0o700,parents=False,exist_ok=False)
    snapshot=destination/'checkpoint.npz'
    # Atomic source replacement leaves this inode intact, including across a save
    # racing with link(). Never copy from a pathname that may change mid-read.
    _snapshot_checkpoint(checkpoint,snapshot)
    _sync_directory(destination)
    weights,meta=_read_weights(snapshot,expected_identity,traits)
    del weights
    history=np.asarray(meta.get('history'),dtype=float)
    models=sum(len(t.candidates) for t in traits)
    if history.shape!=(meta['sweep'],models) or not np.isfinite(history).all():
        raise ValueError('legacy checkpoint objective history is invalid')
    receipt=dict(kind='mixture_checkpoint_warm_start',schema=1,
        legacy_identity=expected_identity,legacy_python=python_files,
        checkpoint_sha256=file_digest(snapshot),sweep=meta['sweep'],elapsed_seconds=meta['elapsed'],
        legacy_objective=history[-1].tolist(),
        protocol_fit_identity=protocol_fit_identity,scientific_contract=scientific_contract(traits,mixtures,source),
        group_keys=[[[t.id,c.id] for c in candidates] for t,candidates in _groups(traits)])
    if attestation_record is not None:
        receipt['source_attestation']=dict(sha256=attestation_sha256,evidence=attestation_record)
    with (destination/'MIGRATION.json').open('x') as stream:
        stream.write(canonical(receipt)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(destination)
    return receipt


def load_legacy_checkpoint_weights(bundle, traits, mixtures, source, *, receipt_sha256):
    """Return authenticated starting weights, with the old generation untouched."""
    bundle=Path(bundle)
    receipt_path=bundle/'MIGRATION.json'
    receipt_bytes=receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest()!=receipt_sha256:
        raise ValueError('migration receipt identity mismatch')
    receipt=json.loads(receipt_bytes)
    if receipt.get('kind')!='mixture_checkpoint_warm_start' or receipt.get('schema')!=1:
        raise ValueError('unsupported migration receipt')
    if receipt['scientific_contract']!=scientific_contract(traits,mixtures,source):
        raise ValueError('migration scientific inputs differ')
    keys=[[[t.id,c.id] for c in candidates] for t,candidates in _groups(traits)]
    if receipt['group_keys']!=keys:
        raise ValueError('migration candidate ordering differs')
    snapshot=bundle/'checkpoint.npz'
    # Hash and decode the same open generation, even if a path is replaced.
    with snapshot.open('rb') as stream:
        before=os.fstat(stream.fileno())
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(8*2**20),b''):
            h.update(block)
        if h.hexdigest()!=receipt['checkpoint_sha256']:
            raise ValueError('migration checkpoint content differs')
        stream.seek(0)
        weights,meta=_read_weights(stream,receipt['legacy_identity'],traits)
        after=os.fstat(stream.fileno())
        # Link count/ctime can legitimately change when the old writer replaces
        # its path. Contents, size and mtime of this generation must stay fixed.
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('migration checkpoint changed during import')
    if meta['sweep']!=receipt['sweep']:
        raise ValueError('migration checkpoint sweep differs')
    return weights,receipt
