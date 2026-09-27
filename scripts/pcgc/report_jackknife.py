#!/usr/bin/env python3
"""Audit summary with paired reproduction checks and Monte Carlo intervals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from report_pcgc import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--components', type=Path, required=True)
    p.add_argument('--old-root', type=Path, required=True)
    p.add_argument('--real', type=Path, nargs=2, required=True)
    p.add_argument('--real-final', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    rows = json.loads((args.audit/'replicates.json').read_text())
    report, inputs = [], {}
    for scenario in sorted({r['scenario'] for r in rows}):
        path = args.old_root/f'confirmation_{scenario.lower()}'/'replicates.json'
        old = json.loads(path.read_text())
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for risk in ('supplied','fitted'):
            prior = {r['seed']:r for r in old if r['risk']==risk and r['method']=='pcgc'}
            for blocks in sorted({r['blocks'] for r in rows}):
                group = [r for r in rows if (r['scenario'],r['risk'],r['blocks'])==(scenario,risk,blocks)]
                estimate = np.array([r['marginal_total'] for r in group])
                truth = np.array([r['truth'] for r in group])
                old_estimate = np.array([prior[r['seed']]['marginal_total'] for r in group])
                # Historical threaded runs differed by up to 1.85e-8 in one
                # fitted-risk estimate; preserve and report the discrepancy.
                np.testing.assert_allclose(estimate, old_estimate, atol=1e-7, rtol=0)
                newse = np.array([r['marginal_total_standard_error'] for r in group])
                oldse = np.array([r['old_marginal_se'] for r in group])
                if blocks==50:
                    savedse = [prior[r['seed']]['conditional_total_standard_error']/(1+r['covariate_variance']) for r in group]
                    np.testing.assert_allclose(oldse, savedse, atol=1e-7, rtol=0)
                report.append(dict(scenario=scenario, risk=risk, blocks=blocks,
                                   maximum_point_reproduction_error=float(np.max(abs(estimate-old_estimate))),
                                   mean_se_multiplier=float(np.mean(newse/oldse)),
                                   old=metrics(estimate,truth,oldse,len(group)),
                                   corrected=metrics(estimate,truth,newse,len(group)),
                                   known_risk_independent_Bernoulli_null_rms_se=float(np.sqrt(np.mean([r['null_pair_variance_se']**2 for r in group])))))
    components = []
    comp = json.loads((args.components/'replicates.json').read_text())
    for risk in ('supplied','fitted'):
        group = [r for r in comp if r['risk']==risk]
        for j, truth in enumerate((.04,.16)):
            components.append(dict(risk=risk, component=j,
                corrected=metrics([r['marginal_components'][j] for r in group],np.full(len(group),truth),
                                  [r['marginal_standard_errors'][j] for r in group],len(group),component=True)))
    real = [json.loads((d/'results.json').read_text()) for d in args.real]
    manifests = [json.loads((d/'manifest.json').read_text()) for d in args.real]
    for key in ('variant_axis_identity','scale_identity','reference_sample_identity'):
        assert manifests[0][key] == manifests[1][key]
    comparisons = []
    for low, high in zip(*real):
        assert low['sample_identity']==high['sample_identity']
        methods = []
        for l,h in zip(low['methods'],high['methods']):
            assert l['method']==h['method']
            methods.append(dict(method=l['method'],lower_probe_estimate=l['marginal_total'],
                                higher_probe_estimate=h['marginal_total'],higher_probe_se=h['marginal_total_standard_error'],
                                absolute_shift=float(abs(l['marginal_total']-h['marginal_total'])),
                                shift_over_se=float(abs(l['marginal_total']-h['marginal_total'])/h['marginal_total_standard_error'])))
        comparisons.append(dict(case_fraction=low['case_fraction'],methods=methods,baselines=high['baselines']))
    for d in [args.audit,args.components,*args.real]:
        for f in d.glob('*.json'):
            inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    final = None
    if args.real_final:
        final = json.loads((args.real_final/'results.json').read_text())
        meta = json.loads((args.real_final/'manifest.json').read_text())
        assert meta['population_reference_estimator']=='exact_small_data_oracle'
        for key in ('variant_axis_identity','scale_identity','reference_sample_identity'):
            assert meta[key]==manifests[1][key]
        for old,row in zip(real[1],final):
            assert old['sample_identity']==row['sample_identity']
            assert row['baseline_score_contract']=='SUMMIT_beta_se_exact_h2'
            a=next(m for m in old['methods'] if m['method']=='pcgc')
            b=next(m for m in row['methods'] if m['method']=='pcgc')
            assert a['marginal_total']==b['marginal_total']
        for f in args.real_final.iterdir():
            if f.is_file():
                inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=False)
    output = dict(inputs=inputs, paired_simulations=report, component_pilot=components, real_probe_comparison=comparisons,
                  real_final=final, report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for r in report:
        m=r['corrected']
        print(r['scenario'],r['risk'],r['blocks'],f"SE/SD {m['rms_se_over_sd']:.4f}",
              f"coverage {m['coverage']:.4f}",f"screen {m['calibration_screen']}")


if __name__=='__main__':
    main()
