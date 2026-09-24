"""Authenticate pilot tables against paired deletion arrays without refitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--fits',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--expected-blocks',type=int,default=200)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(a.repo/'src'))
import numpy as np
from summit.context.cross_trait_zpass import load_array_artifact
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:list(csv.DictReader(path.open(),delimiter='\t'))
close=lambda x,y:np.testing.assert_allclose(x,y,rtol=2e-11,atol=1e-13,equal_nan=True)
fc=json.loads((a.fits/'COMPLETE.json').read_text())
rc=json.loads((a.report/'COMPLETE.json').read_text())
assert fc['deleted_genetic_mass_restored'] is True
assert rc['pilot_completion_sha256']==sha(a.fits/'COMPLETE.json')
for root,complete in ((a.fits,fc),(a.report,rc)):
    for name,digest in complete['tables'].items():
        assert sha(root/name)==digest,(root,name)
rows=read(a.fits/'pilot_estimates.tsv')
key=lambda r:(r['trait_x'],r['trait_y'],r['mode'],r['quantity'],r['entry'])
table={key(r):r for r in rows};assert len(table)==len(rows)
fit_sha={};seen=set();ids=None;conditions=[];ranks=set()
for path in sorted(a.fits.glob('*.npz')):
    f,pr=load_array_artifact(path,kind='summit.cross_trait.fit')
    x,y,mode=pr['trait_x'],pr['trait_y'],pr['gram_mode']
    for name,value in fc.items():
        if name not in ('seconds','tables'):assert pr[name]==value,(path,name)
    assert bool(f['loo_genetic_mass_restored'])
    assert len(f['block_ids'])==a.expected_blocks
    if ids is None:ids=f['block_ids']
    np.testing.assert_array_equal(ids,f['block_ids'])
    assert len(np.unique(ids))==len(ids)
    q=f['omega_xy'].shape[-1];width=f['omega_xy'].size
    expected=f['raw_loo_coefficients'].copy()
    expected[:,:width]*=np.repeat(f['loo_mass_restoration'],q*q,axis=1)
    np.testing.assert_array_equal(expected,f['loo_coefficients'])
    np.testing.assert_array_equal(expected[:,:width].reshape(f['loo_omega_xy'].shape),f['loo_omega_xy'])
    for quantity in ('omega_xy','h_xy','response_rg','baseline_rg','orthogonal_rg','orthogonal_trace',
                     'orthogonal_minus_baseline_rg','response_minus_baseline_rg'):
        point=f[quantity].ravel();loo=f['loo_'+quantity].reshape(len(ids),-1)
        se=np.sqrt((len(ids)-1)*np.var(loo,axis=0))
        centered=loo-loo.mean(axis=0)
        covariance=(len(ids)-1)/len(ids)*(centered.T@centered)
        saved=f['covariance'] if quantity=='omega_xy' else f[quantity+'_covariance']
        close(covariance,saved)
        for j,value in enumerate(point):
            k=(x,y,mode,quantity,str(j));row=table[k];seen.add(k)
            for name,expected_value in (('estimate',value),('jackknife_se',se[j]),
                    ('lower_95',value-1.96*se[j]),('upper_95',value+1.96*se[j])):
                close(expected_value,float(row[name]))
            for name in ('n_x','n_y','n_overlap'):assert int(row[name])==pr[name]
    for prefix in ('','loo_'):
        h,hx,hy=(f[prefix+k] for k in ('h_xy','h_xx','h_yy'))
        metric=f['context_covariance']
        trace=np.einsum('ij,...ji->...',metric,h)
        tx=np.einsum('ij,...ji->...',metric,hx);ty=np.einsum('ij,...ji->...',metric,hy)
        with np.errstate(invalid='ignore',divide='ignore'):
            corr=np.where((tx>0)&(ty>0)&f[prefix+'orthogonal_baseline_variances_positive'],trace/np.sqrt(tx*ty),np.nan)
        close(trace,f[prefix+'orthogonal_trace']);close(corr,f[prefix+'orthogonal_rg'])
        close(f[prefix+'orthogonal_rg']-f[prefix+'baseline_rg'],f[prefix+'orthogonal_minus_baseline_rg'])
    fit_sha[path.name]=sha(path);conditions.append(float(f['condition_number']));ranks.add(int(f['rank']))
assert seen==set(table) and len(fit_sha)==112
shifts=read(a.fits/'pilot_maximum_mode_shifts.tsv')
for row in shifts:
    x,y,quantity,entry=(row[n] for n in ('trait_x','trait_y','quantity','entry'))
    default=table[x,y,'factorized',quantity,entry]
    valid=[];invalid=[]
    for mode in ('factorized','factorized_plus_residual','legacy_transport','legacy_transport_exact'):
        other=table[x,y,mode,quantity,entry]
        se=float(default['jackknife_se'])
        delta=(float(other['estimate'])-float(default['estimate']))/se if se>0 else np.nan
        close(delta,float(other['shift_from_default_se']))
        if np.isfinite(delta):valid.append(abs(delta))
        else:invalid.append(mode)
    close(max(valid) if valid else np.nan,float(row['maximum_mode_shift_se']))
    assert int(row['valid_modes'])==len(valid)
    assert row['undefined_shift_modes']==','.join(invalid)
comparisons=read(a.report/'baseline_vs_orthogonal_response.tsv')
blocks=read(a.report/'age_bmi_cross_exposure_covariance.tsv')
for row in comparisons:
    for quantity in ('baseline_rg','orthogonal_rg','orthogonal_minus_baseline_rg'):
        source=table[row['trait_x'],row['trait_y'],'factorized',quantity,'0']
        for name in ('estimate','jackknife_se','lower_95','upper_95'):close(float(row[quantity+'_'+name]),float(source[name]))
for row in blocks:
    matches=[v for k,v in table.items() if k[:3]==(row['trait_x'],row['trait_y'],'factorized')
        and k[3]=='h_xy' and all(v[n]==row[n] for n in ('exposure_x','exposure_y'))]
    assert len(matches)==1
    for name in ('estimate','jackknife_se','lower_95','upper_95'):close(float(row[name]),float(matches[0][name]))
assert len(comparisons)==28 and len(blocks)==112
result=dict(verified_pair_mode_artifacts=len(fit_sha),verified_estimate_rows=len(rows),
    paired_blocks=len(ids),block_ids=ids.tolist(),chromosomes=fc['chromosomes'],
    verified_mode_shift_rows=len(shifts),verified_paired_contrasts=len(comparisons),
    verified_age_bmi_entries=len(blocks),ranks=sorted(ranks),
    condition_number_range=[min(conditions),max(conditions)],
    fit_sha256=fit_sha,source_completion_sha256={str(root/'COMPLETE.json'):sha(root/'COMPLETE.json') for root in (a.fits,a.report)},
    script_sha256=sha(Path(__file__)),genotype_traversals=0,normal_equation_solves=0)
a.output.mkdir(exist_ok=False)
with (a.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ('fit_sha256','block_ids')},indent=2))
