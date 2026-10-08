"""Reduce persisted replicate summaries; never rerun or overwrite simulations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.epistasis.validate import interval


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    folders=['confirmed_estimated_discrete_v2_20261001','confirmed_estimated_real_v2_20261001',
             'confirmed_linear_real_identifiable_v2_20261001']
    rows=[]
    for folder in folders:
        frame=pd.read_csv(args.root/folder/'replicates.csv')
        for (setting,method),part in frame.groupby(['setting','method'],sort=False):
            hits=int(np.count_nonzero(part.p.to_numpy(float)<=.05));lo,hi=interval(hits,len(part))
            valid=~part.failed.to_numpy(bool)
            row=dict(panel=folder,setting=setting,method=method,fits=len(part),failures=int(np.count_nonzero(~valid)),
                rejection=hits/len(part),lower=lo,upper=hi,rejection_valid=float(np.mean(part.p.to_numpy(float)[valid]<=.05)) if np.any(valid) else np.nan)
            if method=='FAME_Wald':
                coefficient=part.coefficient.to_numpy(float);expected=part.expected.to_numpy(float);se=part.se.to_numpy(float)
                covered=(np.abs(coefficient-expected)<=1.95996398454*se)&valid
                row.update(mean_coefficient=np.mean(coefficient),expected=expected[0],bias=np.mean(coefficient-expected),
                    empirical_sd=np.std(coefficient,ddof=1),mean_se=np.nanmean(se),
                    coverage=float(np.mean(covered)),coverage_valid=float(np.mean(covered[valid])),
                    mean_estimate_mc_se=float(np.std(coefficient,ddof=1)/np.sqrt(len(coefficient))))
            if 'mean_beta_coverage' in part:row['mean_beta_coverage']=part.mean_beta_coverage.mean()
            rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(args.out/'summary.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    part=table[(table.panel=='confirmed_estimated_real_v2_20261001')&(table.method=='FAME_Wald')]
    axes[0].errorbar(range(len(part)),part.mean_coefficient-part.expected,yerr=1.96*part.mean_estimate_mc_se,fmt='o',ms=3)
    axes[0].axhline(0,color='grey');axes[0].set_ylabel('FAME bias relative to moment projection; MC 95% interval')
    axes[1].plot(range(len(part)),part.empirical_sd,'o',label='Empirical SD');axes[1].plot(range(len(part)),part.mean_se,'x',label='Mean analytic SE')
    axes[1].legend();axes[1].set_ylabel('Coefficient uncertainty')
    axes[2].plot(range(len(part)),part.coverage,'o');axes[2].axhline(.95,color='grey');axes[2].set_ylim(0,1);axes[2].set_ylabel('95% coverage, all scheduled fits')
    for ax in axes:ax.set_xticks(range(len(part)),part.setting,rotation=70,ha='right',fontsize=6)
    fig.savefig(args.out/'estimation_uncertainty.png',dpi=170)
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    part=table[table.panel=='confirmed_estimated_real_v2_20261001'];settings=list(part.setting.unique())
    for j,method in enumerate(part.method.unique()):
        values=part[part.method==method]
        ax.errorbar(np.arange(len(settings))+(j-2)*.13,values.rejection,
            yerr=np.array([values.rejection-values.lower,values.upper-values.rejection]),fmt='o',ms=3,label=method)
    ax.axhline(.05,color='grey');ax.set_xticks(range(len(settings)),settings,rotation=65,ha='right',fontsize=7)
    ax.set_ylabel('Rejection / power at 0.05 with exact binomial 95% interval');ax.legend(fontsize=8)
    fig.savefig(args.out/'estimated_null_power.png',dpi=170)
    (args.out/'reduction.json').write_text(json.dumps(dict(inputs=folders,
        correction='Recomputed FAME coverage from per-replicate coefficients/SEs; initial pandas object reduction of numpy.bool_ mixed with missing values returned 1/n. Original replicate rows and all simulations retained unchanged.',
        inference='Coverage of pseudo-true moment coefficients does not establish absence of biological false positives.'),indent=2)+'\n')
    print(table[(table.method=='REML_kernel')&table.setting.str.endswith('_null')].to_string(index=False))


if __name__=='__main__':main()
