"""Transfer a fixed simulation reference between byte-identical genotype sources.

This transports cohort-side simulation inputs, not portable inference summaries.
It never migrates fitted models, uncertainty, solver state or convergence claims.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from summit.prediction.artifacts import file_digest
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.migration import _source_content
from scripts.epistasis.full_matched import write_json


MEMBERS=('reference.json','reference.npz','covariates.tsv','samples.tsv',
         'variants.txt','interaction_variants.txt')
KIND='summit.epistasis.matched_simulation_reference_transfer'


def copy_checked(source,destination,expected):
    if source.is_symlink() or not source.is_file() or file_digest(source)!=expected:
        raise ValueError('reference member changed: '+source.name)
    with source.open('rb') as src,destination.open('xb') as dst:
        shutil.copyfileobj(src,dst,length=8*2**20)
    if file_digest(destination)!=expected or file_digest(source)!=expected:
        raise ValueError('reference member changed during copy: '+source.name)


def export_reference(reference,out):
    start=time.perf_counter()
    meta=json.loads((reference/'reference.json').read_text())
    with FileGenotypeSource(meta['genotypes'],genome_build='GRCh37') as source:
        if (source.identity!=meta['source_identity'] or len(source.samples)!=meta['source_n']
                or len(source.variants.ids)!=meta['m']):
            raise ValueError('original reference genotype inputs changed')
        with np.load(reference/'reference.npz',allow_pickle=False) as data:
            rows=data['rows']
        if (rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(rows)!=meta['n']
                or not len(rows) or rows[0]<0 or rows[-1]>=len(source.samples)
                or np.any(np.diff(rows)<=0)):
            raise ValueError('invalid original reference row axis')
        samples=pd.read_csv(reference/'samples.tsv',sep='\t',dtype=str)
        if list(samples[['FID','IID']].itertuples(index=False,name=None))!=[source.samples[i] for i in rows]:
            raise ValueError('original reference sample alignment differs')
        content=_source_content(source)
        files={name:file_digest(reference/name) for name in MEMBERS}
        out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
        for name,expected in files.items():copy_checked(reference/name,out/name,expected)
        source.check()
        write_json(out/'TRANSFER.json',dict(kind=KIND,schema_version=1,source_content=content,
            source_identity=source.identity,files=files,genome_build='GRCh37',
            scientific_scope='identical fixed generating means and intact donor rows; no fitted quantities',
            source=str(reference.resolve()),seconds=time.perf_counter()-start))
    return out/'TRANSFER.json'


def import_reference(bundle,receipt_sha256,genotypes,out):
    start=time.perf_counter()
    receipt=bundle/'TRANSFER.json'
    if file_digest(receipt)!=receipt_sha256:
        raise ValueError('reference transfer receipt checksum differs')
    record=json.loads(receipt.read_text())
    if (record['kind']!=KIND or record['schema_version']!=1 or set(record['files'])!=set(MEMBERS)):
        raise ValueError('unsupported simulation reference transfer')
    for name,expected in record['files'].items():
        if (bundle/name).is_symlink() or file_digest(bundle/name)!=expected:
            raise ValueError('reference member changed: '+name)
    meta=json.loads((bundle/'reference.json').read_text())
    if meta['source_identity']!=record['source_identity']:
        raise ValueError('original reference identity differs')
    with FileGenotypeSource(genotypes,genome_build=record['genome_build']) as source:
        if _source_content(source)!=record['source_content']:
            raise ValueError('destination genotype bytes or ordered scientific axes differ')
        out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
        for name,expected in record['files'].items():
            destination='original.reference.json' if name=='reference.json' else name
            copy_checked(bundle/name,out/destination,expected)
        meta.update(genotypes=str(genotypes.resolve()),source_identity=source.identity,
            reference_transfer=dict(receipt_sha256=receipt_sha256,
                original_source_identity=record['source_identity'],
                scope='byte-identical genotype source; simulation arrays and scales copied unchanged'))
        source.check()
        write_json(out/'reference.json',meta)
        write_json(out/'TRANSFERRED.json',dict(receipt_sha256=receipt_sha256,
            source_identity=source.identity,source_content=record['source_content'],
            unchanged_members={name:digest for name,digest in record['files'].items() if name!='reference.json'},
            reference_sha256=file_digest(out/'reference.json'),seconds=time.perf_counter()-start))
    return out/'reference.json'


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--reference',type=Path,required=True)
    i=sub.add_parser('import');i.add_argument('--bundle',type=Path,required=True)
    i.add_argument('--receipt-sha256',required=True);i.add_argument('--genotypes',type=Path,required=True)
    for parser in (e,i):parser.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=(export_reference(a.reference,a.out) if a.command=='export'
        else import_reference(a.bundle,a.receipt_sha256,a.genotypes,a.out))
    print(result,flush=True)


if __name__=='__main__':main()
