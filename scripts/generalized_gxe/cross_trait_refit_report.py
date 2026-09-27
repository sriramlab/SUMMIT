"""Authenticate completed refits and link shared diagnostics without changing fits."""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
from summit.context.cross_trait_zpass import load_array_artifact
from summit.context.reference_zpass_cli import file_sha256
from cross_trait_refit import write_table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('refits','diagnostics','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    completed=json.loads((a.refits/'COMPLETE.json').read_text())
    if completed['failures']:raise ValueError('refits contain failures')
    for path,digest in completed['tables'].items():
        if file_sha256(path)!=digest:raise ValueError('refit table checksum differs')
    diagnostic,meta=load_array_artifact(a.diagnostics/'reference_diagnostics.npz',kind='summit.cross_trait.reference_diagnostics')
    for key in ('manifest_sha256','source_completion_sha256'):
        if meta[key]!=completed[key]:raise ValueError('refit and diagnostic input identities differ')
    artifacts={};ranks=[]
    for path in sorted(a.refits.glob('*.npz')):
        if path.name=='reference_diagnostics.npz':continue
        data,provenance=load_array_artifact(path,kind='summit.cross_trait.within_refit')
        np.testing.assert_array_equal(data['block_ids'],np.arange(200))
        artifacts[path.name]=file_sha256(path)
    if len(artifacts)!=len(completed['traits'])*6:raise ValueError('incomplete six-arm artifact set')
    with (a.refits/'within_trait_mode_comparison.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    flags=[r for r in rows if r['flag_shift_gt_one_se']=='True']
    summary=[]
    for trait in completed['traits']:
        selected=[r for r in rows if r['trait']==trait and np.isfinite(float(r['shift_from_default_se']))]
        largest=max(selected,key=lambda r:abs(float(r['shift_from_default_se'])))
        summary.append(dict(trait=trait,flagged_entries=sum(r['trait']==trait for r in flags),
            maximum_absolute_shift_se=abs(float(largest['shift_from_default_se'])),
            annotation=largest['annotation'],quantity=largest['quantity'],arm=largest['arm']))
    write_table(a.output/'trait_mode_shift_summary.tsv',summary)
    if flags:write_table(a.output/'entries_above_one_se.tsv',flags)
    result=dict(refit_completion_sha256=file_sha256(a.refits/'COMPLETE.json'),
        shared_reference_diagnostics=dict(path=str(a.diagnostics/'reference_diagnostics.npz'),
            sha256=file_sha256(a.diagnostics/'reference_diagnostics.npz')),
        fit_sha256=artifacts,rows=len(rows),flagged_entries=len(flags),
        flagged_traits=[r['trait'] for r in summary if r['flagged_entries']],
        maximum_absolute_shift_se=max(r['maximum_absolute_shift_se'] for r in summary),
        tables={path.name:file_sha256(path) for path in a.output.glob('*.tsv')},
        script_sha256=file_sha256(__file__))
    with (a.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='fit_sha256'},indent=2))


if __name__=='__main__':main()
