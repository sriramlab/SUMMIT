"""Authenticate and compare complete, same-input chromosome qualifications."""
from pathlib import Path
import argparse, json, sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    import numpy as np
    from summit.prediction.artifacts import file_digest,load_prediction_models
    def load(folder):
        receipt=json.loads((folder/'COMPLETE.json').read_text())
        if not receipt['passed']:raise ValueError('unsuccessful fit: '+str(folder))
        for name,expected in receipt['files'].items():
            if Path(name).name!=name or file_digest(folder/name)!=expected:
                raise ValueError('completion file hash differs: '+str(folder/name))
        if file_digest(folder/'models/manifest.json')!=receipt['model_manifest_sha256']:
            raise ValueError('model manifest differs')
        models=load_prediction_models(folder/'models')
        if not all(m.convergence['converged'] for m in models):raise ValueError('uncertified model')
        return dict(receipt=receipt,models=models,
            manifest=json.loads((folder/'models/manifest.json').read_text()),
            protocol=json.loads((folder/'PROTOCOL.json').read_text()),
            selection=json.loads((folder/'SELECTION.json').read_text()))
    reference=load(a.baseline)
    results=[]
    for folder in a.candidate:
        candidate=load(folder)
        for name in ('models','n','variants'):
            if candidate['receipt'][name]!=reference['receipt'][name]:raise ValueError('panel differs')
        if candidate['protocol']['plan']['fit_identity']!=reference['protocol']['plan']['fit_identity']:
            raise ValueError('scientific preparation identity differs')
        weight_errors=[]
        for left,right in zip(reference['models'],candidate['models']):
            if left.key!=right.key or left.prior_spec!=right.prior_spec:raise ValueError('candidate differs')
            if left.variants!=right.variants or left.scale.identity!=right.scale.identity:
                raise ValueError('genotype axes/scale differ')
            if left.context_spec!=right.context_spec or left.fixed_spec!=right.fixed_spec:
                raise ValueError('context/fixed definitions differ')
            np.testing.assert_array_equal(left.covariance,right.covariance)
            if left.convergence['threshold']!=right.convergence['threshold']:
                raise ValueError('convergence tolerance differs')
            difference=np.asarray(right.weights)-left.weights
            weight_errors.append(dict(model=left.model_id,max_abs=float(np.max(abs(difference))),
                relative_norm=float(np.linalg.norm(difference)/max(1e-30,np.linalg.norm(left.weights)))))
        scores={}
        for split in ('pilot','replication'):
            with np.load(a.baseline/(split+'_scores.npz')) as left, np.load(folder/(split+'_scores.npz')) as right:
                np.testing.assert_array_equal(left['rows'],right['rows'])
                np.testing.assert_array_equal(left['models'],right['models'])
                x,y=left['predictions'],right['predictions']
                if not np.isfinite(y).all():raise ValueError('nonfinite scores')
                difference=y-x
                relative=np.linalg.norm(difference,axis=0)/np.maximum(1.,np.linalg.norm(x,axis=0))
                scores[split]=dict(max_abs=float(np.max(abs(difference))),
                    max_candidate_rmse=float(np.max(np.sqrt(np.mean(difference**2,axis=0)))),
                    max_relative_norm=float(np.max(relative)))
        selected={key:dict(baseline=value['model'],candidate=candidate['selection'][key]['model'])
            for key,value in reference['selection'].items()}
        selection_agrees=all(v['baseline']==v['candidate'] for v in selected.values())
        old_bound=np.asarray(reference['manifest']['run_report']['objective_history'][-1])
        new_bound=np.asarray(candidate['manifest']['run_report']['objective_history'][-1])
        deficit=old_bound-new_bound
        objective_ok=bool(np.all(deficit<=1e-8*np.maximum(abs(old_bound),1.)))
        score_ok=all(v['max_relative_norm']<1e-5 and v['max_candidate_rmse']<1e-5 for v in scores.values())
        results.append(dict(path=str(folder.resolve()),complete_sha256=file_digest(folder/'COMPLETE.json'),
            passed=selection_agrees and objective_ok and score_ok,
            fit_speedup=reference['receipt']['fit_seconds']/candidate['receipt']['fit_seconds'],
            total_speedup=reference['receipt']['seconds']/candidate['receipt']['seconds'],
            peak_rss_ratio=candidate['receipt']['peak_rss_kib']/reference['receipt']['peak_rss_kib'],
            timings={k:candidate['receipt'][k] for k in ('fit_seconds','scoring_seconds','seconds','peak_rss_kib')},
            scores=scores,weights=weight_errors,selection_agrees=selection_agrees,selections=selected,
            maximum_objective_deficit=float(np.max(deficit)),objective_guard_passed=objective_ok,
            guard_counts=candidate['manifest']['run_report'].get('guarded_gemm'),
            run_report={k:v for k,v in candidate['manifest']['run_report'].items() if k!='objective_history'}))
    record=dict(passed=all(x['passed'] for x in results),baseline=str(a.baseline.resolve()),
        baseline_complete_sha256=file_digest(a.baseline/'COMPLETE.json'),comparisons=results,
        scope='Single-run complete chr22 fit/scoring comparison; all original convergence checks pass. '
              'Score comparison additionally requires relative norm and phenotype-unit RMSE below 1e-5, '
              'identical selected candidates, and the original 1e-8 relative objective guard. '
              'This is not a genome-wide runtime estimate.')
    with a.output.open('x') as stream:json.dump(record,stream,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='comparisons'},indent=2))
    for item in results:print(json.dumps({k:item[k] for k in ('path','passed','total_speedup','peak_rss_ratio','scores','selection_agrees')}))
    if not record['passed']:raise SystemExit(1)

if __name__=='__main__':main()
