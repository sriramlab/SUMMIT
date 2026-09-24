"""Authenticate simulation artifacts and audit each recorded coverage result."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--fit',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(a.repo/'src'))
import numpy as np
from summit.context.cross_trait_zpass import load_array_artifact
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
complete=json.loads((a.fit/'COMPLETE.json').read_text())
for name,digest in complete['files'].items():
    if sha(a.fit/name)!=digest:raise ValueError(f'table checksum differs: {name}')
generated=a.root/'simulation_generate/generated.npz'
score=a.root/'simulation_score/scores.npz'
gen,gp=load_array_artifact(generated,kind='summit.cross_trait.simulation')
scores,sp=load_array_artifact(score,kind='summit.cross_trait.simulation_scores')
fit,fp=load_array_artifact(a.fit/'fits.npz',kind='summit.cross_trait.simulation_fit')
assert sp['generated_sha256']==sha(generated)==fp['generated_sha256']
assert sha(score)==fp['scores_sha256']==complete['scores_sha256']
assert gp['genotype_traversals']==sp['genotype_traversals']==1
assert gp['variant_visits']==sp['variant_visits']==454207
assert gp['replicates']==100 and len(scores['block_ids'])==200
assert fit['omega'].shape==(2,100,3,3,3)
assert fit['loo_omega'].shape==(2,100,3,200,3,3)
assert np.isfinite(fit['omega']).all() and np.isfinite(fit['loo_omega']).all()
read=lambda name:list(csv.DictReader((a.fit/name).open(),delimiter='\t'))
rows=read('simulation_summary.tsv');replicates=read('simulation_replicates.tsv')
failures=[];invalid=[]
for row in rows:
    rr=[r for r in replicates if all(r[k]==row[k] for k in ('scenario','quantity','entry'))]
    assert len(rr)==100 and sorted(int(r['replicate']) for r in rr)==list(range(100))
    points=np.array([float(r['estimate']) for r in rr]);ses=np.array([float(r['jackknife_se']) for r in rr])
    truth=float(row['truth']);valid=np.isfinite(points)&np.isfinite(ses);errors=points[valid]-truth
    coverage=float(np.mean(abs(errors)<=1.96*ses[valid]));sd=np.std(points[valid],ddof=1)
    expected=dict(bias=np.mean(errors),rmse=np.sqrt(np.mean(errors**2)),empirical_sd=sd,
        mean_jackknife_se=np.mean(ses[valid]),se_calibration=np.mean(ses[valid])/sd,coverage_95=coverage)
    for name,value in expected.items():np.testing.assert_allclose(value,float(row[name]),rtol=1e-12,atol=1e-14)
    assert int(valid.sum())==int(row['valid_replicates'])
    accepted=bool(valid.all() and .90<=coverage<=.98)
    assert accepted==(row['coverage_acceptance']=='True')
    if not accepted:failures.append(row)
    if row['quantity']=='response_rg':
        case=gp['scenario_names'].index(row['scenario']);k=1+int(row['entry'])
        for ix in np.flatnonzero(~valid):
            rep=int(rr[ix]['replicate'])
            point=fit['omega'][case,rep,:2,k,k]
            deleted=fit['loo_omega'][case,rep,:2,:,k,k]
            bad=np.any(deleted<=0,axis=0)
            invalid.append(dict(scenario=row['scenario'],replicate=rep,context_index=k,
                point_variances=point.tolist(),minimum_deleted_variances=deleted.min(axis=1).tolist(),
                nonpositive_variance_deletion_blocks=scores['block_ids'][bad].tolist()))
assert complete['all_coverage_gates_pass']==(len(failures)==0)
a.output.mkdir(exist_ok=False)
with (a.output/'coverage_failures.tsv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(failures)
result=dict(mode=complete['gram_mode'],summary_rows=len(rows),passed_rows=len(rows)-len(failures),
    failed_rows=len(failures),all_coverage_gates_pass=not failures,
    omega_passed=sum(r['quantity']=='omega_xy' and r['coverage_acceptance']=='True' for r in rows),
    omega_rows=sum(r['quantity']=='omega_xy' for r in rows),invalid_correlations=invalid,
    source_sha256={str(p):sha(p) for p in (generated,score,a.fit/'fits.npz',a.fit/'COMPLETE.json')},
    input_tables=complete['files'],genotype_traversals_in_audit=0,script_sha256=sha(Path(__file__)),
    files={'coverage_failures.tsv':sha(a.output/'coverage_failures.tsv')})
with (a.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','input_tables')},indent=2))
