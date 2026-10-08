"""Summarize matched covariance-bias experiments without pooling conditioning laws."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from summit.prediction.artifacts import file_digest
from scripts.epistasis.full_matched import write_json


def clustered_interval(records, function, seed=317917):
    ids=sorted({r['architecture_id'] for r in records})
    values=np.array([np.mean([function(r) for r in records if r['architecture_id']==i]) for i in ids])
    sampled=np.random.default_rng(seed).integers(len(ids),size=(5000,len(ids)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[sampled].mean(1),[.025,.975]).tolist())


def report(root, output, figure):
    if output.exists() or figure.exists():
        raise FileExistsError('new report and figure paths required')
    roots={name:root/name for name in ('matched','weighted_matched','dense_markers_fresh')}
    data={name:json.loads((path/'results.json').read_text()) for name,path in roots.items()}
    designs={name:json.loads((path/'design.json').read_text()) for name,path in roots.items()}
    for field in ('sample_rows','markers','kernel_sha256','fixed_sha256'):
        assert designs['matched'][field]==designs['weighted_matched'][field]
    for field in ('seed','architectures','noise_repeats','training_sizes'):
        assert designs['matched']['arguments'][field]==designs['weighted_matched']['arguments'][field]
    panels={512:data['matched']['records']+data['weighted_matched']['records'],
            8192:data['dense_markers_fresh']['records']}
    grouped=[];paired=[]
    for markers,records in panels.items():
        for n0 in sorted({r['n0'] for r in records}):
            for method in sorted({r['method'] for r in records}):
                rs=[r for r in records if r['n0']==n0 and r['method']==method]
                if not rs:continue
                mean=lambda key:float(np.mean([r[key] for r in rs]))
                rms=lambda key:float(np.sqrt(np.mean([r[key]**2 for r in rs])))
                entry=dict(markers=markers,n0=n0,method=method,learners=len(rs),
                    architecture_clusters=len({r['architecture_id'] for r in rs}),
                    covariance_bias_rms_se=rms('covariance_bias_z'),
                    fixed_displacement_rms_se=rms('fixed_bias_z'),
                    accounted_genetic_uncertainty_rms_se=float(np.sqrt(np.mean([
                        r['random_sd_ratio']**2-r['fixed_sd_ratio']**2 for r in rs]))),
                    zero_mean_tail=mean('random_zero_bias_tail_005'),
                    mean_error_tail_increment=mean('random_tail_005')-mean('random_zero_bias_tail_005'),
                    trace_transfer_error_rms_se=rms('trace_bias_z'),
                    random_tail=clustered_interval(rs,lambda r:r['random_tail_005']),
                    fixed_tail=clustered_interval(rs,lambda r:r['fixed_tail_005']),
                    posterior_second_moment_residual=clustered_interval(rs,lambda r:
                        r['fixed_bias_z']**2-r['covariance_bias_z']**2-r['random_sd_ratio']**2+r['fixed_sd_ratio']**2),
                    oracle_transfer_mse_reduction=1-np.mean([r['oracle_remaining_z']**2 for r in rs])/np.mean([r['fixed_bias_z']**2 for r in rs]),
                    exact_variance_tail=clustered_interval(rs,lambda r:
                        norm.cdf(-norm.isf(.0025)-r['covariance_bias_z']/r['random_sd_ratio'])+
                        norm.sf(norm.isf(.0025)-r['covariance_bias_z']/r['random_sd_ratio'])))
                grouped.append(entry)
            methods=('he_tangent','weighted_he_tangent') if markers==512 else ('probe_he_tangent','weighted_probe_he_tangent')
            left={(r['architecture_id'],r['noise_id']):r for r in records if r['n0']==n0 and r['method']==methods[0]}
            right={(r['architecture_id'],r['noise_id']):r for r in records if r['n0']==n0 and r['method']==methods[1]}
            assert left.keys()==right.keys()
            differences=[dict(architecture_id=k[0],tail=right[k]['random_tail_005']-left[k]['random_tail_005'],
                              bias_squared=right[k]['covariance_bias_z']**2-left[k]['covariance_bias_z']**2) for k in left]
            paired.append(dict(markers=markers,n0=n0,methods=methods,
                tail_difference=clustered_interval(differences,lambda r:r['tail']),
                covariance_bias_squared_difference=clustered_interval(differences,lambda r:r['bias_squared'])))
    result=dict(kind='summit.epistasis.matched_bias_assessment',schema_version=1,
        phase='development; denser-marker comparison uses fresh phenotype seeds',
        total_fits=sum(len(v['records']) for v in data.values()),
        independent_architecture_clusters=64,full_outcome_vectors=256,
        interpretation='Four environmental draws within each architecture; nested N and methods are paired, not independent replications. Two marker panels are distinct experiments.',
        evidence={str(p/name):file_digest(p/name) for p in roots.values() for name in ('design.json','results.json')},
        summaries=grouped,paired_comparisons=paired,
        limitations=['Five covariance components and a simplified adaptive reference learner; not the full-marker 14-component public workflow.',
            'No alternative-model coverage or genome-wide tail qualification in this null-only experiment.',
            'Posterior accounting identities require the declared Gaussian effect law; fixed-architecture sensitivity is a separate target.',
            'Weighting reduces covariance mean-error outliers but does not uniformly improve all tail estimates or make finite-N normal tails exact.'])
    repo=Path(__file__).resolve().parents[2]
    source_files=['scripts/epistasis/conditional_bias_scaling.py','scripts/epistasis/conditional_bias_report.py',
        'scripts/epistasis/conditional_polygenic_reference.py','scripts/epistasis/conditional_reference_equivalence.py',
        'src/summit/epistasis/polygenic.py','src/summit/epistasis/conditional_reference.py',
        'src/summit/epistasis/conditional_workflow.py','src/summit/epistasis/inputs.py',
        'tests/test_epistasis_bias_scaling.py','tests/test_epistasis_polygenic_operator.py',
        'tests/test_epistasis_conditional_reference_equivalence.py','tests/test_epistasis_conditional_workflow.py']
    result['source_sha256']={name:file_digest(repo/name) for name in source_files}
    result['implementation_validation']={}
    for name,passed,skipped in [('portable_focused',31,0),('portable_full',167,1),('private_focused',41,0)]:
        receipt=json.loads((root/(name+'.runtime.json')).read_text())
        assert receipt['exit_code']==0 and receipt['failure'] is None
        result['implementation_validation'][name]=dict(passed=passed,skipped=skipped,**receipt)
    result['native_sha256']={str(p):file_digest(p) for mode in ('portable','private')
        for p in Path('/data1/bronsonj/epistasis_native_compat_20261003/build',mode).glob('*.so')}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.4,6.9),sharex=True)
    for column,markers in enumerate((512,8192)):
        methods=('he_tangent','weighted_he_tangent') if markers==512 else ('probe_he_tangent','weighted_probe_he_tangent')
        for method,label,color in zip(methods,('HE + tangents','Weighted HE + tangents'),('#59677d','#087f8c')):
            rows=sorted((r for r in grouped if r['markers']==markers and r['method']==method),key=lambda r:r['n0'])
            x=[r['n0'] for r in rows]
            axes[0,column].plot(x,[r['covariance_bias_rms_se'] for r in rows],'o-',label=label,color=color)
            means=np.array([r['random_tail']['mean']*100 for r in rows])
            intervals=np.array([r['random_tail']['ci95'] for r in rows])*100
            axes[1,column].errorbar(x,means,yerr=np.array([means-intervals[:,0],intervals[:,1]-means]),
                fmt='o-',capsize=3,label=label,color=color)
        axes[0,column].set_title(f'{markers:,} markers — '+('exact traces' if markers==512 else '128 probes'))
        axes[1,column].axhline(.5,color='#333333',ls='--',lw=1,label='Nominal 0.5%')
        axes[1,column].set_xlabel('Training participants; confirmation fixed at 512')
        for row in (0,1):
            axes[row,column].set_xscale('log',base=2)
            axes[row,column].set_xticks([128,256,512,1024],labels=['128','256','512','1,024'])
            axes[row,column].grid(alpha=.2)
            axes[row,column].spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Covariance-induced mean error\nRMS in reported SE units')
    axes[1,0].set_ylabel('Conditional null rejection probability (%)')
    axes[0,0].legend(frameon=False,fontsize=9)
    axes[1,1].legend(frameon=False,fontsize=9)
    fig.suptitle('Matched sample-size diagnosis: mean error and variance calibration',fontsize=13)
    fig.text(.5,.012,'128 learners per point; 32 architecture clusters × 4 noise draws. Error bars: cluster bootstrap 95% intervals.\nDevelopment evidence; curves do not qualify full-marker or genome-wide inference.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.96))
    figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure,dpi=160)
    plt.close(fig)
    print(json.dumps(dict(fits=result['total_fits'],summaries=len(grouped),output=str(output),figure=str(figure))))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True)
    a=p.parse_args();report(a.root,a.out,a.figure)
