"""Diagnose the saved two-scenario, Q=3 overlapping-cohort simulations.

This is a diagnostic, not a change to the production interval convention.
It uses the same covariance estimates and all paired deletion coefficients;
no genotypes, reference redraws or normal-equation solves are needed.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_gwldcore_editable']
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from summit.context.cross_trait_fit import cross_trait_derived
from summit.context.cross_trait_zpass import load_array_artifact
from summit.context.reference_zpass_cli import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--axes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    fit, provenance = load_array_artifact(a.fit,kind='summit.cross_trait.simulation_fit')
    with np.load(a.axes,allow_pickle=False) as z:
        phi = z['basis']
    if (phi.ndim!=2 or phi.shape[1]!=3 or fit['omega'].ndim!=5
            or fit['omega'].shape[0]!=2 or fit['omega'].shape[-3:]!=(3,3,3)
            or fit['loo_omega'].shape[:3]!=fit['omega'].shape[:3]
            or fit['truth'].shape!=(2,6,6)):
        raise ValueError('expected the two-scenario Q=3 simulation study schema')
    n = int(len(phi)/1.4)
    metric = np.cov(phi[:,1:].T,bias=True)
    kw = dict(mean_x=phi[:n,1:].mean(0),mean_y=phi[-n:,1:].mean(0),context_covariance=metric)
    def evaluate(w):
        result = cross_trait_derived(w[...,2,:,:],w[...,0,:,:],w[...,1,:,:],**kw)
        result['omega_xy'] = w[...,2,:,:]
        result['orthogonal_variance_x'] = np.einsum('ij,...ji->...',metric,result['h_xx'])
        result['orthogonal_variance_y'] = np.einsum('ij,...ji->...',metric,result['h_yy'])
        names = ('omega_xy','h_xy','baseline_rg','centered_baseline_rg','response_rg',
                 'orthogonal_trace','orthogonal_variance_x','orthogonal_variance_y','orthogonal_rg')
        shape = w.shape[:-3]
        return np.concatenate([result[k].reshape(*shape,-1) for k in names],axis=-1)
    names = ([('omega_xy',i) for i in range(9)] + [('h_xy',i) for i in range(4)]
        + [('baseline_rg',0),('centered_baseline_rg',0),('response_rg',0),('response_rg',1)]
        + [('orthogonal_trace',0),('orthogonal_variance_x',0),('orthogonal_variance_y',0),('orthogonal_rg',0)])
    point = fit['omega']
    deleted = fit['loo_omega'].transpose(0,1,3,2,4,5)
    truth = fit['truth']
    truth = np.stack((truth[:,:3,:3],truth[:,3:,3:],truth[:,:3,3:]),axis=1)
    nblocks = deleted.shape[2]
    delta = deleted-deleted.mean(2,keepdims=True)
    recentered = point[:,:,None]+delta
    values = evaluate(point)
    target = evaluate(truth)
    ordinary = evaluate(deleted)
    centered = evaluate(recentered)
    # Central differences at two resolutions check numerical differentiation.
    def jacobian(w,step):
        q = w.shape[-1]
        perturb = np.eye(3*q*q).reshape(3*q*q,3,q,q)*step
        return ((evaluate(w[...,None,:,:,:]+perturb)-evaluate(w[...,None,:,:,:]-perturb))/(2*step))
    step = 1e-6
    for refinement in range(6):
        jac = jacobian(point,step)
        finer = jacobian(point,step/2)
        good = np.isfinite(jac)&np.isfinite(finer)
        gradient_error = np.linalg.norm((jac-finer)[good])/np.linalg.norm(finer[good])
        if np.isfinite(gradient_error) and gradient_error<=1e-6:
            break
        step /= 10
    else:
        raise ValueError(f'finite-difference check failed: {gradient_error}')
    truth_jac = jacobian(truth,5e-7)
    linear = np.einsum('srbd,srdv->srbv',delta.reshape(*delta.shape[:3],-1),finer)
    oracle_linear = np.einsum('srbd,sdv->srbv',delta.reshape(*delta.shape[:3],-1),truth_jac)
    # Compare a truth-linearized SE with the SAME linearized estimator, not
    # with the nonlinear ratio's SD. This separates covariance calibration
    # from curvature/weak-denominator effects in the full estimator.
    point_error = (point-truth[:,None]).reshape(*point.shape[:2],-1)
    truth_linearized = target[:,None] + np.einsum('srd,sdv->srv',point_error,truth_jac)
    # A separate diagnostic holds the three trace quantities at their full-fit
    # center before forming the ratio, isolating denominator drift from H drift.
    z = ordinary[...,-4:-1]
    z_centered = values[:,:,None,-4:-1] + z-z.mean(2,keepdims=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        trace_ratio = z_centered[...,0]/np.sqrt(z_centered[...,1]*z_centered[...,2])
    ses = {'legacy_nonlinear':np.sqrt((nblocks-1)*np.var(ordinary,axis=2)),
           'recentered_nonlinear':np.sqrt((nblocks-1)*np.var(centered,axis=2)),
           'delta_at_full_fit':np.sqrt((nblocks-1)*np.var(linear,axis=2)),
           'delta_at_truth_diagnostic':np.sqrt((nblocks-1)*np.var(oracle_linear,axis=2))}
    rows=[];replicates=[];drift=[];linearized_rows=[]
    for sc in range(2):
        for j,(name,entry) in enumerate(names):
            pp = values[sc,:,j]
            for method,all_se in ses.items():
                se = all_se[sc,:,j];valid = np.isfinite(pp)&np.isfinite(se)
                errors = pp[valid]-target[sc,j];sd = np.std(pp[valid],ddof=1)
                rows.append(dict(scenario=sc,quantity=name,entry=entry,method=method,
                    valid=int(valid.sum()),truth=target[sc,j],bias=np.mean(errors),
                    empirical_sd=sd,mean_se=np.mean(se[valid]),rms_se=np.sqrt(np.mean(se[valid]**2)),
                    mean_se_over_sd=np.mean(se[valid])/sd,rms_se_over_sd=np.sqrt(np.mean(se[valid]**2))/sd,
                    coverage_valid=np.mean(abs(errors)<=1.96*se[valid]),
                    coverage_all=np.sum(abs(errors)<=1.96*se[valid])/len(pp)))
                for rep in range(len(pp)):
                    replicates.append(dict(scenario=sc,replicate=rep,quantity=name,entry=entry,
                        method=method,estimate=pp[rep],truth=target[sc,j],se=se[rep]))
            drift.append(dict(scenario=sc,quantity=name,entry=entry,
                point_mean=np.nanmean(pp),deletion_mean=np.nanmean(ordinary[sc,:,:,j]),
                mean_deletion_shift=np.nanmean(ordinary[sc,:,:,j].mean(1)-pp),
                median_deletion_shift=np.nanmedian(ordinary[sc,:,:,j].mean(1)-pp)))
            lp = truth_linearized[sc,:,j]
            ls = ses['delta_at_truth_diagnostic'][sc,:,j]
            valid = np.isfinite(lp)&np.isfinite(ls)
            linearized_rows.append(dict(scenario=sc,quantity=name,entry=entry,
                valid=int(valid.sum()),empirical_linearized_sd=np.std(lp[valid],ddof=1),
                rms_linearized_se=np.sqrt(np.mean(ls[valid]**2)),
                rms_se_over_linearized_sd=np.sqrt(np.mean(ls[valid]**2))/np.std(lp[valid],ddof=1),
                nonlinear_sd_over_linearized_sd=np.std(pp[np.isfinite(pp)],ddof=1)/np.std(lp[valid],ddof=1)))
    a.output.mkdir(exist_ok=False)
    for name,data in (('calibration_methods.tsv',rows),('calibration_replicates.tsv',replicates),
                      ('deletion_drift.tsv',drift),('truth_linearized_calibration.tsv',linearized_rows)):
        with (a.output/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t');writer.writeheader();writer.writerows(data)
    summary=dict(fit_sha256=file_sha256(a.fit),axes_sha256=file_sha256(a.axes),
        script_sha256=file_sha256(__file__),source_provenance=provenance,
        finite_difference_relative_error=gradient_error,genotype_traversals=0,normal_equation_solves=0,
        finite_difference_step=step,finite_difference_refinements=refinement,
        trace_centered_ratio_mean_se=[float(np.nanmean(np.sqrt((nblocks-1)*np.var(trace_ratio[sc],axis=1)))) for sc in range(2)],
        tables={p.name:file_sha256(p) for p in a.output.glob('*.tsv')},
        interpretation='diagnostic on existing replicates; not independent validation of a revised interval')
    (a.output/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps([r for r in rows if r['quantity']=='orthogonal_rg'],indent=2))


if __name__=='__main__':main()
