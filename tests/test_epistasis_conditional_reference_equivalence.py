"""Cross-host collection cannot weaken reference or learner authentication."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.prediction.artifacts import file_digest
from scripts.epistasis.conditional_reference_equivalence import verify_equivalence


@pytest.fixture
def references(tmp_path,request):
    from summit.prediction.genotype import FileGenotypeSource
    from summit.epistasis.conditional_reference import prepare_conditional_reference
    from prediction_helpers import prediction_threads
    rng = np.random.default_rng(973621)
    weighting=getattr(request,'param','none')
    n0, n1, m = 80, 100, 60
    samples = list(map(str, range(n0+n1))); variants = [f'v{i}' for i in range(m)]
    raw = rng.binomial(2, .35, (n0+n1,m)).astype(float)
    raw[0,3] = np.nan
    to_bed(tmp_path/'g.bed', raw, properties=dict(fid=samples,iid=samples,sid=variants,
        chromosome=['1']*3+['2']*(m-3), bp_position=list(range(1,m+1)),
        allele_1=['A']*m,allele_2=['G']*m))
    pd.DataFrame(dict(FID=samples,IID=samples,PC1=rng.normal(size=n0+n1))).to_csv(
        tmp_path/'cov.tsv',sep='\t',index=False)
    spec=dict(target='v0',local_variants=variants[:3],dominance_variants=variants[:3],
        covariates=dict(file='cov.tsv',columns=['PC1'],varying_effects=['PC1']))
    with FileGenotypeSource(tmp_path/'g.bed') as source:
        legacy = source.identity
        content = source.authenticate_content()
        prepare_conditional_reference(source,np.arange(n0),np.arange(n0,n0+n1),np.arange(1,m),
            spec,tmp_path,path=tmp_path/'content.npz',probes=32,seed=871631,
            threads=prediction_threads(),memory_bytes=2**30,moment_weighting=weighting)
    with FileGenotypeSource(tmp_path/'g.bed') as source:
        prepare_conditional_reference(source,np.arange(n0),np.arange(n0,n0+n1),np.arange(1,m),
            spec,tmp_path,path=tmp_path/'legacy.npz',probes=32,seed=871631,
            threads=prediction_threads(),memory_bytes=2**30,moment_weighting=weighting)
    (tmp_path/'TRANSFER.json').write_text(json.dumps(dict(
        kind='summit.epistasis.matched_simulation_reference_transfer',schema_version=1,
        source_identity=legacy,source_content=content)))
    pinned=lambda name:dict(file=name,sha256=file_digest(tmp_path/name))
    return dict(references=[pinned('legacy.npz'),pinned('content.npz')],
        source_transfer=pinned('TRANSFER.json'))


@pytest.mark.parametrize('references',['none','genotype_diagonal'],indirect=True)
def test_reference_equivalence_uses_content_and_preserves_archives(tmp_path,references):
    before={p:p.read_bytes() for p in tmp_path.glob('*.npz')}
    actual=verify_equivalence(references,tmp_path)
    assert len(set(actual['reference_ids']))==2
    assert all(v<1e-12 for v in actual['geometry_relative_differences'].values())
    assert 'fixed' in actual['exact_scientific_arrays']
    assert before=={p:p.read_bytes() for p in before}
    wrong=copy.deepcopy(references);wrong['source_transfer']['sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'):
        verify_equivalence(wrong,tmp_path)
    receipt=tmp_path/'TRANSFER.json';record=json.loads(receipt.read_text())
    record['source_content']['files'][0]['sha256']='0'*64
    receipt.write_text(json.dumps(record));wrong=copy.deepcopy(references)
    wrong['source_transfer']['sha256']=file_digest(receipt)
    with pytest.raises(ValueError,match='not authenticated'):
        verify_equivalence(wrong,tmp_path)


@pytest.mark.parametrize('changed', ['stale_digest','geometry','scientific','operator'])
def test_reference_equivalence_rejects_changed_inputs(tmp_path,references,changed):
    from summit.context.spec import array_sha256,canonical_sha256
    from summit.epistasis.conditional_reference import MEAN_ARRAYS
    path=tmp_path/'content.npz'
    with np.load(path,allow_pickle=False) as archive:
        values={k:archive[k] for k in archive.files}
    manifest=json.loads(str(values['manifest']));meta=manifest['metadata']
    if changed=='operator':
        meta['geometry']['operator_identity']='0'*64
    else:
        name='h' if changed in ('stale_digest','geometry') else 'fixed'
        values[name]=values[name].copy();values[name].flat[0]+=.1
        if changed!='stale_digest':
            manifest['digests'][name]=array_sha256(values[name])
        if changed=='scientific':
            meta['identity']=canonical_sha256(dict(method='conditional_reference_v1',
                source=meta['mean']['source'],variants=array_sha256(values['variants']),
                mean={k:array_sha256(values[k]) for k in MEAN_ARRAYS},definition=meta['mean'],
                probes=meta['geometry']['probes'],seed=meta['geometry']['seed'],exact=False))
    values['manifest']=np.array(json.dumps(manifest));np.savez(path,**values)
    spec=copy.deepcopy(references);spec['references'][1]['sha256']=file_digest(path)
    message={'stale_digest':'array digest','geometry':'numerical geometry',
        'scientific':'scientific array','operator':'operator definition'}[changed]
    with pytest.raises(ValueError,match=message):
        verify_equivalence(spec,tmp_path)


def test_assessment_requires_verified_reference_membership(tmp_path,references):
    from test_epistasis_conditional_assessment import _bundle
    from scripts.epistasis.conditional_assessment import assess,METHOD
    from summit.epistasis.robust import load_robust_scores,write_robust_scores
    from summit.epistasis.conditional import conditional_mean_summary
    evidence=verify_equivalence(references,tmp_path)
    for i,name in enumerate(('left','right')):
        _bundle(tmp_path/name,[i])
        path=tmp_path/name/'prepared'/f'rep{i:03}.robust-score.npz'
        old=load_robust_scores(path)
        # New test fixture with unchanged statistics but authentic reference IDs.
        ids=dict(old.metadata['identities'],genotype_reference=evidence['reference_ids'][i])
        new=conditional_mean_summary([.01*(i+1)],[[.04]],[[10.]],
            feature_names=old.feature_names,trait_name=old.trait_names[0],identities=ids,
            trait_unit='test units',diagnostics=dict(nuisance_training_n=100,
                confirmation_n=200,fixed_rank=4,feature_rank=1,outside_confirmation_design=[]))
        path.unlink();write_robust_scores(new,path)
        prep=path.parent/'preparation.json';value=json.loads(prep.read_text())
        value['summaries'][0]['identity']=new.metadata['preparation_identity']
        prep.write_text(json.dumps(value))
    group=dict(id='null',setting='null',phase='development',scheduled_replicates=[0,1],
        completed_results=['left','right'])
    design=dict(schema_version=1,method=METHOD,groups=[group])
    with pytest.raises(ValueError,match='reference differs'):
        assess(design,tmp_path)
    group['reference_equivalence']=references
    result=assess(design,tmp_path)['groups'][0]
    assert result['completed_replicates']==[0,1] and result['summary']['scheduled']==2
    assert result['reference_equivalence']==evidence
    # A third, unverified reference cannot enter even when an equivalence exists.
    _bundle(tmp_path/'third',[2]);group['scheduled_replicates'].append(2)
    group['completed_results'].append('third')
    with pytest.raises(ValueError,match='outside verified equivalence'):
        assess(design,tmp_path)
