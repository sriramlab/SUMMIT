"""Fail-closed legacy checkpoint authentication and exact scientific contracts."""
from dataclasses import replace
import json
import numpy as np
import pytest

from test_prediction_core import fixture
from summit.prediction._validation import array_digest, canonical
from summit.prediction.artifacts import file_digest
from summit.prediction.mixture import MixtureSpec
from summit.prediction.migration import _read_weights, _groups, scientific_contract, load_legacy_checkpoint_weights


def bundle(tmp_path):
    source,traits=fixture(n=31,m=11)
    mixtures={(t.id,c.id):MixtureSpec(.03,.1) for t in traits for c in t.candidates}
    arrays={}
    expected={}
    rng=np.random.default_rng(803)
    for j,(t,candidates) in enumerate(_groups(traits)):
        k,m,q,n=len(candidates),len(t.variants),t.phi.shape[1],len(t.rows)
        for field,shape in [('weights',(k,m,q)),('residual',(n,k)),('penalty',(k,m))]:
            arrays[f'{field}_{j}']=rng.normal(size=shape)
        for i,c in enumerate(candidates):expected[t.id,c.id]=arrays[f'weights_{j}'][i]
    meta=dict(schema=1,kind='mixture',identity='authenticated_legacy_test',sweep=7,elapsed=123.,
        arrays={k:dict(shape=list(v.shape),sha256=array_digest(v)) for k,v in arrays.items()})
    arrays['metadata']=np.frombuffer(canonical(meta).encode(),dtype=np.uint8)
    np.savez(tmp_path/'checkpoint.npz',**arrays)
    receipt=dict(kind='mixture_checkpoint_warm_start',schema=1,legacy_identity=meta['identity'],sweep=7,
        checkpoint_sha256=file_digest(tmp_path/'checkpoint.npz'),
        scientific_contract=scientific_contract(traits,mixtures,source),
        group_keys=[[[t.id,c.id] for c in candidates] for t,candidates in _groups(traits)])
    (tmp_path/'MIGRATION.json').write_text(canonical(receipt))
    return source,traits,mixtures,arrays,expected


def test_import_preserves_all_saved_weights_and_file_bytes(tmp_path):
    source,traits,mixtures,arrays,expected=bundle(tmp_path)
    original=file_digest(tmp_path/'checkpoint.npz')
    actual,receipt=load_legacy_checkpoint_weights(tmp_path,traits,mixtures,source,
        receipt_sha256=file_digest(tmp_path/'MIGRATION.json'))
    assert actual.keys()==expected.keys()
    for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
    assert file_digest(tmp_path/'checkpoint.npz')==original
    assert receipt['sweep']==7


@pytest.mark.parametrize('field', ['weights_0','residual_0','penalty_1'])
def test_corruption_is_rejected_even_if_outer_file_hash_is_refreshed(tmp_path,field):
    source,traits,mixtures,arrays,_=bundle(tmp_path)
    arrays[field].flat[0]+=1.
    np.savez(tmp_path/'checkpoint.npz',**arrays)
    receipt=json.loads((tmp_path/'MIGRATION.json').read_text())
    receipt['checkpoint_sha256']=file_digest(tmp_path/'checkpoint.npz')
    (tmp_path/'MIGRATION.json').write_text(canonical(receipt))
    with pytest.raises(ValueError,match='shape/content'):
        load_legacy_checkpoint_weights(tmp_path,traits,mixtures,source,
            receipt_sha256=file_digest(tmp_path/'MIGRATION.json'))


@pytest.mark.parametrize('change', ['phenotype','context','scale','prior','mixture','order','genotypes'])
def test_changed_scientific_inputs_are_rejected(tmp_path,change):
    source,traits,mixtures,arrays,_=bundle(tmp_path)
    t=traits[0]
    if change=='phenotype':traits[0]=replace(t,y=t.y+.01)
    elif change=='context':
        phi=t.phi.copy();phi[:,1]+=.01
        traits[0]=replace(t,phi=phi)
    elif change=='scale':traits[0]=replace(t,scale=replace(t.scale,mean=t.scale.mean+.01))
    elif change=='prior':
        candidates=list(t.candidates);candidates[0]=replace(candidates[0],covariance=candidates[0].covariance*1.01)
        traits[0]=replace(t,candidates=tuple(candidates))
    elif change=='mixture':mixtures[t.id,t.candidates[0].id]=MixtureSpec(.04,.1)
    elif change=='order':traits[0]=replace(t,candidates=t.candidates[::-1])
    elif change=='genotypes':
        source.values=source.values.copy();source.values[-1,-1]=1-source.values[-1,-1]
    with pytest.raises(ValueError,match='scientific inputs'):
        load_legacy_checkpoint_weights(tmp_path,traits,mixtures,source,
            receipt_sha256=file_digest(tmp_path/'MIGRATION.json'))


def test_wrong_receipt_and_legacy_identity_are_rejected(tmp_path):
    source,traits,mixtures,arrays,_=bundle(tmp_path)
    with pytest.raises(ValueError,match='receipt identity'):
        load_legacy_checkpoint_weights(tmp_path,traits,mixtures,source,receipt_sha256='wrong')
    with pytest.raises(ValueError,match='legacy checkpoint identity'):
        _read_weights(tmp_path/'checkpoint.npz','wrong',traits)


@pytest.mark.parametrize('cross_filesystem',[False,True])
def test_snapshot_keeps_one_generation_during_atomic_replacement(tmp_path,monkeypatch,cross_filesystem):
    import errno,os
    from summit.prediction.migration import _snapshot_checkpoint
    source=tmp_path/'live.npz';snapshot=tmp_path/'snapshot.npz';new=tmp_path/'new.npz'
    original=b'completed generation\n'*100
    source.write_bytes(original);new.write_bytes(b'next completed generation')
    if cross_filesystem:
        def cross_device(*args):raise OSError(errno.EXDEV,'different filesystem')
        monkeypatch.setattr(os,'link',cross_device)
        actual_fstat=os.fstat
        replaced=False
        def replace_after_open(fd):
            nonlocal replaced
            result=actual_fstat(fd)
            if not replaced:
                os.replace(new,source);replaced=True
            return result
        monkeypatch.setattr(os,'fstat',replace_after_open)
    _snapshot_checkpoint(source,snapshot)
    if not cross_filesystem:os.replace(new,source)
    assert snapshot.read_bytes()==original
    assert source.read_bytes()==b'next completed generation'


@pytest.mark.parametrize('mismatch',[None,'inode','size','mtime','ctime','path','host','job','hash'])
def test_original_host_attestation_only_substitutes_device_id(tmp_path,mismatch):
    from test_prediction_io import write_bed
    from summit.prediction.genotype import FileGenotypeSource,estimate_scale
    from summit.prediction.migration import _legacy_source_view
    from summit.prediction._validation import digest
    from prediction_helpers import prediction_threads
    _,traits,path=write_bed(tmp_path)
    with FileGenotypeSource(path,genome_build='GRCh37') as source:
        traits=[replace(t,scale=estimate_scale(source,t.rows,t.variants,threads=prediction_threads())) for t in traits]
        original_identity=source.identity
        states=[list(x) for x in source._states]
        for state in states:state[0]+=12345  # A different node's mount device number.
        record=dict(kind='legacy_genotype_descriptor_attestation',schema=1,host='original-node',job=123,
            paths=list(source._paths),states=states)
        if mismatch in ('inode','size','mtime','ctime'):
            states[0][{'inode':1,'size':2,'mtime':3,'ctime':4}[mismatch]]+=1
        elif mismatch=='path':record['paths'][0]+='.different'
        elif mismatch=='host':record['host']='wrong-node'
        elif mismatch=='job':record['job']=124
        attestation=tmp_path/'attestation.json';attestation.write_text(canonical(record))
        sha='wrong' if mismatch=='hash' else file_digest(attestation)
        if mismatch:
            with pytest.raises(ValueError):
                _legacy_source_view(traits,source,attestation,sha,'original-node',123)
        else:
            adapted,view,_=_legacy_source_view(traits,source,attestation,sha,'original-node',123)
            expected=digest([source.input.format,list(source._paths),states,digest(source.samples),source.variants.identity])
            assert view.identity==expected!=source.identity
            assert all(t.scale.provenance['source']==expected for t in adapted)
            assert all(t.scale.provenance['source']==original_identity for t in traits)
        assert source.identity==original_identity
        source.check()


def test_content_identity_survives_relocation_but_keeps_mutation_guard(tmp_path):
    import shutil
    from test_prediction_io import write_bed
    from summit.prediction.genotype import FileGenotypeSource
    _,_,path=write_bed(tmp_path)
    second=tmp_path/'relocated';second.mkdir()
    for suffix in ('.bed','.bim','.fam'):
        shutil.copyfile(path.with_suffix(suffix),second/('calls'+suffix))
    with FileGenotypeSource(path,genome_build='GRCh37') as a, FileGenotypeSource(second/'calls.bed',genome_build='GRCh37') as b:
        assert a.identity!=b.identity
        assert a.authenticate_content()==b.authenticate_content()
        assert a.identity==b.identity
        with (second/'calls.bed').open('ab') as stream:stream.write(b'\0')
        with pytest.raises(RuntimeError,match='modified or replaced'):b.check()


def test_checkpoint_resume_after_real_file_relocation_is_bitwise_identical(tmp_path):
    import shutil
    from test_prediction_io import write_bed
    from prediction_helpers import prediction_threads
    from summit.prediction.genotype import FileGenotypeSource,estimate_scale
    from summit.prediction.mixture import MixtureSolverSpec,fit_mixture_prediction
    _,traits,path=write_bed(tmp_path)
    second=tmp_path/'relocated';second.mkdir()
    for suffix in ('.bed','.bim','.fam'):
        shutil.copyfile(path.with_suffix(suffix),second/('calls'+suffix))
    threads=prediction_threads()
    options=dict(storage='packed',block_size=7,threads=threads,
        solver=MixtureSolverSpec(rtol=1e-9,max_sweeps=200))
    checkpoint=tmp_path/'resume.npz'
    class Interrupted(Exception):pass
    def interrupt(record):raise Interrupted
    with FileGenotypeSource(path,genome_build='GRCh37') as source:
        source.authenticate_content()
        traits=[replace(t,scale=estimate_scale(source,t.rows,t.variants,threads=threads)) for t in traits]
        mixtures={(t.id,c.id):MixtureSpec(.1,.3) for t in traits for c in t.candidates}
        with pytest.raises(Interrupted):
            fit_mixture_prediction(traits,source,output=tmp_path/'partial',mixtures=mixtures,
                checkpoint=checkpoint,progress=interrupt,**options)
        with pytest.raises(RuntimeError,match='before source preparation'):source.authenticate_content()
    with FileGenotypeSource(second/'calls.bed',genome_build='GRCh37') as source:
        source.authenticate_content()
        resumed=fit_mixture_prediction(traits,source,output=tmp_path/'resumed',mixtures=mixtures,
            checkpoint=checkpoint,resume=True,**options)
        direct=fit_mixture_prediction(traits,source,output=tmp_path/'direct',mixtures=mixtures,**options)
    for a,b in zip(resumed,direct):
        np.testing.assert_array_equal(a.weights,b.weights)
        np.testing.assert_array_equal(a.fixed_coefficients,b.fixed_coefficients)
