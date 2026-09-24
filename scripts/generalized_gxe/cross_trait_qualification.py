"""Authenticate a completed fused study before releasing dependent work."""
import argparse
import json
from pathlib import Path

import numpy as np
from summit.context.cross_trait_zpass import load_array_artifact
from summit.context.reference_zpass_cli import file_sha256


def validate(study_output, z_output, manifest, *, chromosome=22):
    complete=json.loads((study_output/'COMPLETE.json').read_text())
    design=json.loads(manifest.read_text())
    panel=next(p for p in design['panels'] if int(p['chromosome'])==chromosome)
    summary_path=study_output/'cross_trait_summary.npz'
    summary,sprov=load_array_artifact(summary_path,kind='summit.cross_trait.summary')
    z,zprov=load_array_artifact(z_output,kind='summit.cross_trait.z_moments')
    for label,provenance in (('score',sprov),('Z',zprov)):
        for key,value in complete.items():
            if provenance.get(key)!=value:
                raise ValueError(f'{label} provenance differs from completion receipt: {key}')
    expected=dict(chromosome=chromosome,variant_visits=panel['m'],genotype_traversals=1,
        common_only=True,reference_manifest_sha256=file_sha256(manifest))
    for key,value in expected.items():
        if complete.get(key)!=value:raise ValueError(f'incomplete or different qualification: {key}')
    ledger=zprov['execution_ledger']
    if (ledger['observed_genotype_passes']!=1 or ledger['retained_variant_visits']!=panel['m']
            or ledger['protected_tn_calls']<1 or zprov['variants']!=panel['m']):
        raise ValueError('qualification traversal ledger differs')
    if not all(zprov['native_build'][key] for key in ('gemm_integrity_enabled','gemm_checksum_enabled')):
        raise ValueError('qualification did not use both native GEMM guards')
    if 'process_segments' in complete:
        position=0
        for segment in complete['process_segments']:
            if segment['begin']!=position or segment['end']<=position:
                raise ValueError('qualification process segments overlap or have a gap')
            position=segment['end']
        if position!=panel['m']:raise ValueError('qualification process segments are incomplete')
    np.testing.assert_array_equal(summary['block_ids'],z['block_ids'])
    np.testing.assert_array_equal(summary['trait_names'],complete['traits'])
    return dict(qualified=True,chromosome=chromosome,variants=panel['m'],
        completion_sha256=file_sha256(study_output/'COMPLETE.json'),
        summary_sha256=file_sha256(summary_path),z_sha256=file_sha256(z_output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study-output','z-output','manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--chromosome',type=int,choices=range(1,23),default=22)
    args=parser.parse_args()
    print(json.dumps(validate(args.study_output,args.z_output,args.manifest,chromosome=args.chromosome)),flush=True)


if __name__=='__main__':main()
