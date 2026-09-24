"""Authenticate the completed pilot traversal and export its execution ledger."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import numpy as np
from cross_trait_qualification import validate
from summit.context.cross_trait_zpass import load_array_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study-root','z-root','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--accounting',type=Path,nargs='+',required=True)
    a=p.parse_args();sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    accounting=[]
    for path in a.accounting:
        for block in re.split(r'^=+\s*$',path.read_text(),flags=re.M):
            fields={m[1]:m[2].strip() for m in re.finditer(r'^(\w+)\s+([^\n]+)$',block,re.M)}
            if 'jobnumber' in fields and fields.get('pe_taskid')=='NONE':accounting.append(fields)
    manifest=json.loads(a.manifest.read_text());panels=manifest['panels']
    assert sorted(int(p['chromosome']) for p in panels)==list(range(1,23))
    rows=[];blocks=[];common_mass=0;identities=None
    for ch in range(1,23):
        root=a.study_root/f'chr{ch}';zp=a.z_root/f'zpass_chr{ch}.npz'
        record=validate(root,zp,a.manifest,chromosome=ch)
        complete=json.loads((root/'COMPLETE.json').read_text())
        score,sp=load_array_artifact(root/'cross_trait_summary.npz',kind='summit.cross_trait.summary')
        _,z=load_array_artifact(zp,kind='summit.cross_trait.z_moments')
        current={k:sp[k] for k in ('traits','input_sha256','genotype_scale','residual_basis_sha256','residual_names')}
        if identities is None:identities=current
        assert current==identities,('cohort identity differs',ch)
        segments=complete['process_segments'];assert len(segments)==1
        segment=segments[0];ledger=z['execution_ledger']
        assert ledger['repaired_columns']==0
        assert ledger['protected_tn_calls']==(record['variants']+127)//128
        assert complete['resumed_checkpoint'] is None
        matches=[f for f in accounting if f['jobnumber']==segment['job_id'] and f['taskid'] in (str(ch),'undefined')]
        assert len(matches)==1,(ch,'missing or duplicated accounting')
        fields=matches[0]
        assert fields['failed']=='0' and fields['exit_status']=='0' and int(fields['slots'])==8
        assert fields['hostname']==segment['hostname']
        mass=float(score['block_masses'].sum());common_mass+=mass;blocks.extend(score['block_ids'].tolist())
        rows.append(dict(chromosome=ch,variants=record['variants'],common_annotation_mass=mass,
            chromosome_block_fragments=len(score['block_ids']),
            guarded_tn_calls=ledger['protected_tn_calls'],repaired_columns=ledger['repaired_columns'],
            process_segments=len(segments),job_id=segment['job_id'],hostname=segment['hostname'],
            accounted_seconds=float(fields['ru_wallclock']),peak_rss_kib=int(fields['ru_maxrss']),
            program_seconds=complete['seconds'],traversal_seconds=complete['traversal_seconds'],
            score_sha256=record['summary_sha256'],z_sha256=record['z_sha256'],
            completion_sha256=record['completion_sha256']))
    assert len(set(blocks))==200
    assert common_mass==manifest['global_masses'][2]
    assert sum(row['variants'] for row in rows)==sum(panel['m'] for panel in panels)
    a.output.mkdir(exist_ok=False)
    table=a.output/'pilot_chromosome_execution.tsv'
    with table.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    result=dict(chromosomes=22,variants=sum(row['variants'] for row in rows),
        common_annotation_mass=common_mass,paired_blocks=len(set(blocks)),
        chromosome_block_fragments=len(blocks),guarded_tn_calls=sum(row['guarded_tn_calls'] for row in rows),
        all_single_contiguous_traversals=True,repaired_columns=0,
        aggregate_accounted_seconds=sum(row['accounted_seconds'] for row in rows),
        maximum_peak_rss_kib=max(row['peak_rss_kib'] for row in rows),
        script_sha256=sha(Path(__file__)),reference_manifest_sha256=sha(a.manifest),
        accounting_sha256={str(p):sha(p) for p in a.accounting},
        tables={table.name:sha(table)},genotype_traversals_in_audit=0)
    with (a.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
