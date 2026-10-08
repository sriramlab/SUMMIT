"""Byte-verified relocation retains the scientific reference, rejecting changes."""
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed


def test_matched_reference_transfer(tmp_path):
    from summit.prediction.genotype import FileGenotypeSource
    from summit.prediction.artifacts import file_digest
    from scripts.epistasis.transfer_matched_reference import export_reference,import_reference,MEMBERS
    original=tmp_path/'original';original.mkdir()
    raw=np.random.default_rng(8813).binomial(2,.3,(32,9)).astype(float)
    raw[2,3]=np.nan
    ids=list(map(str,range(len(raw))))
    properties=dict(fid=ids,iid=ids,chromosome=['1']*9,
        sid=[f'v{i}' for i in range(9)],bp_position=list(range(1,10)),
        allele_1=['A']*9,allele_2=['C']*9)
    to_bed(original/'g.bed',raw,properties=properties)
    with FileGenotypeSource(original/'g.bed',genome_build='GRCh37') as source:
        identity=source.identity
    np.savez(original/'reference.npz',rows=np.arange(32),means=np.arange(64).reshape(32,2))
    pd.DataFrame(dict(FID=ids,IID=ids)).to_csv(original/'samples.tsv',sep='\t',index=False)
    for name in ('covariates.tsv','variants.txt','interaction_variants.txt'):(original/name).write_text('retained input\n')
    meta=dict(genotypes=str(original/'g.bed'),source_identity=identity,source_n=32,n=32,m=9)
    (original/'reference.json').write_text(json.dumps(meta))
    bundle=tmp_path/'bundle';receipt=export_reference(original,bundle)
    destination=tmp_path/'destination';destination.mkdir()
    for suffix in ('bed','bim','fam'):shutil.copyfile(original/f'g.{suffix}',destination/f'g.{suffix}')
    imported=tmp_path/'imported'
    result=import_reference(bundle,file_digest(receipt),destination/'g.bed',imported)
    changed=json.loads(result.read_text())
    assert changed['source_identity']!=identity and changed['genotypes']==str(destination/'g.bed')
    assert changed['reference_transfer']['original_source_identity']==identity
    for name in MEMBERS:
        target='original.reference.json' if name=='reference.json' else name
        assert file_digest(original/name)==file_digest(imported/target)
    with pytest.raises(ValueError,match='receipt checksum'):
        import_reference(bundle,'0'*64,destination/'g.bed',tmp_path/'bad_receipt')
    # A syntactically valid BED with the same IDs and one changed call fails.
    different=tmp_path/'different';different.mkdir();raw[0,0]=(raw[0,0]+1)%3
    to_bed(different/'g.bed',raw,properties=properties)
    with pytest.raises(ValueError,match='genotype bytes'):
        import_reference(bundle,file_digest(receipt),different/'g.bed',tmp_path/'bad_genotype')
    (bundle/'covariates.tsv').write_text('modified input\n')
    with pytest.raises(ValueError,match='reference member changed'):
        import_reference(bundle,file_digest(receipt),destination/'g.bed',tmp_path/'bad_member')
    assert not any((tmp_path/name).exists() for name in ('bad_receipt','bad_genotype','bad_member'))
