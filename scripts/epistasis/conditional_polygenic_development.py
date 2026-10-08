"""Bounded intact-genotype development of estimated conditional covariance.

This experiment separates a Gaussian random-effect null from fixed-architecture
stress tests. Its matrices are diagnostic references, not scalable execution.
"""
import argparse
import json
from pathlib import Path
import time
import resource

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, norm

from summit.prediction.genotype import FileGenotypeSource, estimate_scale, StandardizedBlock, native_module
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.cli import _aligned_table
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from scripts.epistasis.full_matched import write_json
from scripts.epistasis.full_matched_population import binomial_interval
from scripts.epistasis.conditional_polygenic_reference import (
    covariance_geometry, fit_prepared_covariance, conditional_null, innovation_score,
    conditional_mean_tangents,
)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter()
    meta=json.loads((a.reference/'reference.json').read_text())
    with np.load(a.reference/'reference.npz') as archive:
        available=archive['rows']
    if a.training_samples+a.confirmation_samples>2048:
        raise ValueError('bounded covariance experiment requires total N<=2048')
    n0,n1=a.training_samples,a.confirmation_samples
    n=n0+n1
    rng=np.random.default_rng(917431)
    native=native_module(); configure_prediction_threads(native,a.num_threads)
    with FileGenotypeSource(meta['genotypes'],genome_build='GRCh37') as source:
        if source.identity!=meta['source_identity']:
            raise ValueError('genotype source changed')
        rows=np.sort(rng.choice(available,n,replace=False))
        j=source.variants.ids.index(meta['target'])
        local=np.array([source.variants.ids.index(v) for v in meta['local_variants']])
        marker=np.sort(rng.choice(np.delete(np.arange(len(source.variants.ids)),j),a.markers,replace=False))
        selected=np.unique(np.r_[marker,local,j])
        scale=estimate_scale(source,rows,selected,threads=a.num_threads,block_size=128)
        source.prepare(rows,len(selected),a.num_threads)
        raw=source.read(selected)
        g=StandardizedBlock(native,a.num_threads).prepare(raw,np.arange(n),np.arange(len(selected)),
            scale.mean,scale.inverse_scale).copy()
        p=scale.mean/2
        h=np.where(raw==-127,2*p*(1-p),raw==1).astype(float)
        h-=h.mean(0); spread=h.std(0); spread[spread==0]=1; h/=spread
        x=g[:,np.searchsorted(selected,j)]
        samples=[source.samples[i] for i in rows]
        cov=_aligned_table(a.reference/'covariates.tsv',samples)[meta['covariates']].to_numpy(float)
        z=cov[:,meta['covariates'].index('PC1')]
        c=np.column_stack([np.ones(n),cov,g[:,np.searchsorted(selected,local)],h[:,np.searchsorted(selected,local)]])
        gm=g[:,np.searchsorted(selected,marker)]
        hm=h[:,np.searchsorted(selected,marker)]
        background=np.array([i for i,jj in enumerate(marker) if source.variants.chromosome[jj]!=source.variants.chromosome[j]])
        gb=gm[:,background]
        structure=gm*z[:,None]
    designs=[gm,hm,structure]
    genetic=[d@d.T/a.markers for d in designs]
    kernels=np.stack([*genetic,np.eye(n),np.diag(x*x)])
    geometry=covariance_geometry(c[:n0],kernels[:,:n0,:n0])
    methods=a.methods.split(',')
    if not methods or set(methods)-{'known','he','weighted_he','he_tangent'}:
        raise ValueError('unknown bounded comparison method')
    weighted_geometry=(covariance_geometry(c[:n0],kernels[:,:n0,:n0],diagonal_preconditioning=True)
        if 'weighted_he' in methods else None)
    fixed=.5*x+.2*z
    weights=[rng.normal(size=a.markers)/np.sqrt(a.markers) for _ in designs]
    interaction_weights=rng.normal(size=len(background))
    signal=x*(gb@interaction_weights)
    signal*=np.sqrt(a.signal_variance/signal.var())
    kbg=gb@gb[:n0].T/len(background)
    kinteraction=x[:,None]*kbg*x[None,:n0]
    learner=.5*genetic[0][:n0,:n0]+.05*kinteraction[:n0]+np.eye(n0)
    q=geometry['q']; factor=cho_factor(q.T@learner@q,lower=True)
    settings=a.settings.split(',')
    allowed={'random_dense','fixed_dense','random_structure','fixed_structure','random_structure_mixed','fixed_structure_mixed','fixed_heavy'}
    if set(settings)-allowed:
        raise ValueError('unknown prespecified setting')
    write_json(a.out/'design.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},source_identity=meta['source_identity'],
        sample_rows=rows.tolist(),covariance_basis=['additive','dominance','PC1_dependent_additive','iid_noise','target_squared_noise'],
        scientific_null='no generating cross-locus interaction; Gaussian random marker effects for random_*; fixed marker coefficients for fixed_* stress tests',
        primary=methods[-1],comparisons=methods[:-1],
        tangent_adjustment='derivatives of the training-only predicted conditional mean with respect to every covariance component, modulo finite confirmation fixed effects',
        covariance_weighting='weighted_he only: inverse square root of average normalized component diagonal; genotype only',
        estimand='random effects: conditional interaction contribution a*(i1-L_true*i0); fixed-architecture stress: fitted-operator interaction response a*(i1-L_hat*i0)',
        coverage_revision='The original driver used the fitted-operator response for both laws. Shared random genetic effects require the true transfer in the conditional scientific target. Both quantities are retained; this changes no estimator or rejection decision.',
        conditional_mean_diagnostic='the conditional expectation of the estimator is reported separately and never used as the scientific coverage target',
        strength_fixed_before_replicates=a.signal_variance,thresholds=[.05,.005],
        material_inflation_tolerances=[.075,.01],bounded_reference=True))
    records=[]
    numerical_checks=[]
    for setting in settings:
        structured='structure' in setting
        theta=np.array([.8,.5 if structured else 0.,.5 if structured else 0.,.4,.6])
        true_cov=np.einsum('a,aij->ij',theta,kernels)
        true_transfer,true_conditional=conditional_null(true_cov,c,n0)
        contrast=np.column_stack([-true_transfer,np.eye(n1)])
        direct=contrast@true_cov@contrast.T
        cross=contrast@true_cov[:,:n0]@q
        np.testing.assert_allclose(direct,true_conditional,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(cross,0,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(contrast@c,0,atol=1e-8,rtol=1e-8)
        numerical_checks.append(dict(setting=setting,covariance_error=float(np.max(abs(direct-true_conditional))),
            independence_error=float(np.max(abs(cross))),finite_mean_error=float(np.max(abs(contrast@c)))))
        true_signal=signal if setting.endswith('mixed') else np.zeros(n)
        for rep in range(a.replicates):
            rng=np.random.default_rng(np.random.SeedSequence([a.seed,sorted(allowed).index(setting),rep]))
            chosen=[rng.normal(size=a.markers)/np.sqrt(a.markers) for _ in designs] if setting.startswith('random') else weights
            mean=fixed+sum(np.sqrt(v)*(d@w) for v,d,w in zip(theta,designs,chosen))+true_signal
            error=rng.standard_t(5,n)/np.sqrt(5/3) if setting=='fixed_heavy' else rng.normal(size=n)
            y=mean+np.sqrt(.4+.6*x*x)*error
            dual=q@cho_solve(factor,q.T@y[:n0])
            e=.05*kbg@(x[:n0]*dual)
            f=x*e
            pgs=.5*genetic[0][:,:n0]@dual
            c1=np.column_stack([c[n0:],e[n0:],pgs[n0:]])
            baseline=robust_score_tests(prepare_robust_scores(f[n0:,None],y[n0:],c1,
                feature_names=['learned'],trait_names=['y'],metadata={},sampling_model='fixed_design_correct_mean'))
            he_estimate=fit_prepared_covariance(y[:n0],geometry)[0]
            for method in methods:
                try:
                    estimate=(theta if method=='known' else fit_prepared_covariance(y[:n0],weighted_geometry)[0]
                        if method=='weighted_he' else he_estimate)
                    estimated_covariance=np.einsum('a,aij->ij',estimate,kernels)
                    transfer,covariance=(true_transfer,true_conditional) if method=='known' else conditional_null(estimated_covariance,c,n0)
                    nuisance=c1
                    if method=='he_tangent':
                        nuisance=np.column_stack([c1,conditional_mean_tangents(
                            estimated_covariance,kernels,c,n0,y[:n0])])
                    fit=innovation_score(y[:n0],y[n0:],f[:n0],f[n0:],nuisance,transfer,covariance)
                    oracle=innovation_score(y[:n0],y[n0:],signal[:n0],signal[n0:],nuisance,transfer,covariance)
                    arow=fit['contrast'][0]
                    if setting.startswith('random'):
                        conditional_mean=arow@((true_transfer-transfer)@y[:n0]+true_signal[n0:]-true_transfer@true_signal[:n0])
                        actual_variance=arow@true_conditional@arow
                    else:
                        conditional_mean=arow@(mean[n0:]-transfer@y[:n0])
                        actual_variance=np.dot(arow**2,.4+.6*x[n0:]**2)
                    beta=float(fit['beta'][0]); variance=float(fit['covariance'][0,0])
                    fitted_response=float(arow@(true_signal[n0:]-transfer@true_signal[:n0]))
                    target=float(arow@(true_signal[n0:]-true_transfer@true_signal[:n0])) if setting.startswith('random') else fitted_response
                    conditional_rates={}
                    if setting!='fixed_heavy':
                        for alpha in (.05,.005):
                            threshold=norm.isf(alpha/2)*np.sqrt(variance)
                            conditional_rates[str(alpha)]=float(norm.cdf((-threshold-conditional_mean)/np.sqrt(actual_variance))
                                +norm.sf((threshold-conditional_mean)/np.sqrt(actual_variance)))
                    records.append(dict(setting=setting,replicate=rep,method=method,failed=False,
                        beta=beta,se=np.sqrt(variance),interaction_response=target,error=beta-target,
                        fitted_operator_interaction_response=fitted_response,
                        fitted_operator_response_error=beta-fitted_response,
                        conditional_mean=float(conditional_mean),conditional_mean_error=beta-conditional_mean,
                        conditional_mean_coverage=bool((beta-conditional_mean)**2/variance<=chi2.ppf(.95,1)),
                        conditional_mean_bias=float(conditional_mean-target),
                        known_conditional_se=np.sqrt(actual_variance),p=float(chi2.sf(beta*beta/variance,1)),
                        coverage=bool((beta-target)**2/variance<=chi2.ppf(.95,1)),
                        estimated_components=estimate,baseline_p=baseline['joint_p'],
                        oracle_p=float(chi2.sf(oracle['beta'][0]**2/oracle['covariance'][0,0],1)),
                        oracle_response_information=float(np.sum(oracle['response']**2)),
                        conditional_rejection_probability=conditional_rates,
                        direction_alignment=float(np.corrcoef(f,signal)[0,1]**2),
                        nuisance_rank=int(np.linalg.matrix_rank(nuisance)),
                        response_information=float(np.sum(fit['response']**2))))
                except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
                    records.append(dict(setting=setting,replicate=rep,method=method,failed=True,reason=str(error)))
        print(setting,'completed',a.replicates,'seconds',round(time.perf_counter()-begin),flush=True)
    summaries=[]
    for setting in settings:
        for method in methods:
            values=[r for r in records if r['setting']==setting and r['method']==method and not r['failed']]
            summary=dict(setting=setting,method=method,scheduled=a.replicates,failures=a.replicates-len(values))
            if values:
                for label,alpha in [('05',.05),('005',.005)]:
                    hits=sum(r['p']<alpha for r in values)
                    summary['rate_'+label]=hits/a.replicates
                    summary['mc95_'+label]=binomial_interval(hits,a.replicates)
                    summary['oracle_rate_'+label]=sum(r['oracle_p']<alpha for r in values)/a.replicates
                    if setting!='fixed_heavy':
                        rates=np.array([r['conditional_rejection_probability'][str(alpha)] for r in values])
                        resampling=np.random.default_rng(591713).integers(len(rates),size=(10000,len(rates)))
                        summary['conditional_rate_'+label]=float(rates.mean())
                        summary['conditional_rate_mc95_'+label]=np.quantile(rates[resampling].mean(1),[.025,.975]).tolist()
                summary.update(bias=float(np.mean([r['error'] for r in values])),
                    error_sd=float(np.std([r['error'] for r in values],ddof=1)),
                    mean_se=float(np.mean([r['se'] for r in values])),
                    coverage=float(np.mean([r['coverage'] for r in values])),
                    rms_mean_bias_in_se=float(np.sqrt(np.mean([(r['conditional_mean_bias']/r['se'])**2 for r in values]))),
                    rms_se_ratio=float(np.sqrt(np.mean([r['se']**2 for r in values])/np.mean([r['known_conditional_se']**2 for r in values]))),
                    mean_known_conditional_se=float(np.mean([r['known_conditional_se'] for r in values])),
                    mean_oracle_response_information=float(np.mean([r['oracle_response_information'] for r in values])),
                    baseline_rate_05=float(np.mean([r['baseline_p']<.05 for r in values])))
            summaries.append(summary)
    write_json(a.out/'results.json',dict(records=records,summaries=summaries,numerical_checks=numerical_checks,
        seconds=time.perf_counter()-begin,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--training-samples',type=int,default=512); p.add_argument('--confirmation-samples',type=int,default=1024)
    p.add_argument('--markers',type=int,default=512); p.add_argument('--replicates',type=int,default=20)
    p.add_argument('--settings',default='random_dense,fixed_dense,random_structure,fixed_structure,random_structure_mixed,fixed_structure_mixed,fixed_heavy')
    p.add_argument('--methods',default='known,he,he_tangent')
    p.add_argument('--signal-variance',type=float,default=.1); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--num-threads',type=int,default=2); p.add_argument('--memory-gib',type=float,default=16)
    run(p.parse_args())


if __name__=='__main__':
    main()
