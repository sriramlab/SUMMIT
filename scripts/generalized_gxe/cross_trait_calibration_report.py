"""Compare saved inference arms without fitting or revisiting genotypes.

Report finite-interval counts and coverage over all simulation draws as well
as coverage conditional on a finite interval. Wilson intervals quantify only
Monte Carlo uncertainty in the coverage estimate, not estimator uncertainty.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as stream:
        rows=list(csv.DictReader(stream,delimiter='\t'))
    groups={}
    for row in rows:
        key=(row['scenario'],row['quantity'],int(row['entry']))
        groups.setdefault(key,[]).append(row)
    for values in groups.values():
        values.sort(key=lambda row:int(row['replicate']))
        if len({row['replicate'] for row in values})!=len(values):
            raise ValueError('duplicate simulation replicate')
    return groups


def wilson(successes,total):
    if not total:return np.nan,np.nan
    z=1.959963984540054;rate=successes/total
    center=(rate+z*z/(2*total))/(1+z*z/total)
    half=z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/(1+z*z/total)
    return center-half,center+half


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy',type=Path)
    parser.add_argument('--corrected',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    corrected=read(args.corrected);arms=[('corrected nonlinear jackknife',corrected,'jackknife_se'),
        ('corrected full-point delta',corrected,'standard_error')]
    if args.legacy:
        legacy=read(args.legacy)
        if legacy.keys()!=corrected.keys():raise ValueError('simulation quantities differ')
        for key,values in corrected.items():
            previous=legacy[key]
            if [r['replicate'] for r in values]!=[r['replicate'] for r in previous]:
                raise ValueError('replicate identity differs')
            for column in ('truth','estimate'):
                np.testing.assert_allclose([float(r[column]) for r in values],
                    [float(r[column]) for r in previous],rtol=1e-11,atol=1e-12,equal_nan=True)
        arms.insert(0,('legacy nonlinear jackknife',legacy,'jackknife_se'))
    rows=[]
    for arm,groups,se_column in arms:
        for (scenario,quantity,entry),values in groups.items():
            truth=np.array([float(r['truth']) for r in values])
            estimate=np.array([float(r['estimate']) for r in values])
            se=np.array([float(r[se_column]) for r in values]);n=len(values)
            valid=np.isfinite(truth)&np.isfinite(estimate)&np.isfinite(se)&(se>=0)
            point_valid=np.isfinite(truth)&np.isfinite(estimate)
            covered=valid&(abs(estimate-truth)<=1.959963984540054*se)
            sd=np.std(estimate[point_valid],ddof=1)
            lo,hi=wilson(int(covered.sum()),n)
            rows.append(dict(scenario=scenario,quantity=quantity,entry=entry,method=arm,
                replicates=n,finite_estimates=int(point_valid.sum()),finite_intervals=int(valid.sum()),
                bias=float(np.mean((estimate-truth)[point_valid])),empirical_sd=float(sd),
                mean_se=float(np.mean(se[valid])),rms_se=float(np.sqrt(np.mean(se[valid]**2))),
                mean_se_over_sd=float(np.mean(se[valid])/sd),
                rms_se_over_sd=float(np.sqrt(np.mean(se[valid]**2))/sd),
                coverage_all=float(covered.sum()/n),
                coverage_finite=float(covered.sum()/valid.sum()) if valid.any() else np.nan,
                coverage_mc_lower_95=float(lo),coverage_mc_upper_95=float(hi)))
    table=args.output/'calibration_comparison.tsv'
    with table.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t')
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenarios=list(dict.fromkeys(r['scenario'] for r in rows));fig,axes=plt.subplots(1,len(scenarios),figsize=(11,4),squeeze=False)
    for ax,scenario in zip(axes[0],scenarios):
        selected=[r for r in rows if r['scenario']==scenario and r['quantity']=='orthogonal_rg']
        labels=[r['method'].replace(' ','\n',1) for r in selected]
        ax.bar(range(len(selected)),[r['mean_se_over_sd'] for r in selected],color=['#999999','#E69F00','#0072B2'][-len(selected):])
        for i,row in enumerate(selected):
            ax.text(i,row['mean_se_over_sd']+.03,f"{row['mean_se_over_sd']:.3f}\ncoverage {row['coverage_all']:.1%}",ha='center',fontsize=8)
        ax.axhline(1,color='black',lw=.8,ls='--');ax.set_ylim(0,1.35)
        ax.set_xticks(range(len(selected)),labels,fontsize=8)
        ax.set_title(scenario.replace('_',' '),fontsize=10)
        ax.set_ylabel('Mean SE / empirical sampling SD');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Orthogonal-response correlation: simulation calibration',fontsize=12)
    fig.text(.02,.015,'Same fixed reference and genotype panel; new uncertainty calculations on saved draws.\n'
        'Coverage uses every draw; non-finite intervals count as failures. Monte Carlo intervals and RMS SE ratios are in the table.',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.95))
    for suffix in ('png','pdf'):fig.savefig(args.output/f'calibration_comparison.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    sources={str(path):sha(path) for path in (args.legacy,args.corrected) if path}
    files={path.name:sha(path) for path in args.output.iterdir()}
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(sources=sources,script_sha256=sha(Path(__file__)),files=files),indent=2)+'\n')


if __name__=='__main__':main()
