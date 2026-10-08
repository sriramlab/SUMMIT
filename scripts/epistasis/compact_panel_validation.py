"""Bounded real-EUR joint-panel development; no production qualification claim.

Uses existing genotype decoding, feature construction and HE geometry. Refit
the covariance for every phenotype. Dense restricted-coordinate solves allow
many small repetitions; first-replicate results are checked against the native
joint path and the public portable-score inference.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space, orth
from scipy.optimize import nnls
from scipy.stats import chi2, beta as beta_dist

from summit.epistasis.polygenic import PolygenicKernels, fit_kernel_scales, he_geometry, conditional_score, estimate_components
from summit.epistasis.prepare import SelectedStudy, fit_scale
from summit.epistasis.features import prepare_feature_reference
from summit.epistasis.conditional import conditional_mean_summary
from summit.epistasis.robust import robust_score_tests
from summit.prediction.genotype import FileGenotypeSource, ArrayGenotypeSource
from summit.prediction.cli import _aligned_table
from summit.prediction.artifacts import file_digest
from summit.context.spec import array_sha256, canonical_sha256


def interval(hits, total):
    return [0. if hits == 0 else float(beta_dist.ppf(.025,hits,total-hits+1)),
        1. if hits == total else float(beta_dist.ppf(.975,hits+1,total-hits))]


def design(reference, n0, n1, markers, seed):
    rng=np.random.default_rng(seed)
    meta=json.loads((reference/'reference.json').read_text())
    with np.load(reference/'reference.npz',allow_pickle=False) as saved:
        available=saved['rows']
    n=n0+n1
    with FileGenotypeSource(meta['genotypes'],genome_build='GRCh37') as original:
        if original.identity != meta['source_identity']:
            raise ValueError('frozen genotype source changed')
        rows=np.sort(rng.choice(available,n,replace=False))
        i0=np.sort(rng.choice(n,n0,replace=False));i1=np.setdiff1d(np.arange(n),i0)
        # A contiguous real LD window, selected without any phenotype.
        candidates=np.array([j for j,(chrom,pos) in enumerate(zip(
            original.variants.chromosome,original.variants.position))
            if chrom == '5' and 80000000 <= pos <= 81000000])[:64]
        if len(candidates)<6:
            raise ValueError('insufficient supplied window markers')
        background=np.sort(rng.choice(len(original.variants.ids),markers,replace=False))
        selected=np.union1d(background,candidates)
        original.prepare(rows,len(selected),1)
        raw=original.read(selected).copy()
        observed=raw!=-127
        def supported(value):
            mu=np.sum(np.where(observed[i0],value[i0],0),axis=0)/observed[i0].sum(0)
            ss=np.sum(np.where(observed[i0],value[i0]-mu,0)**2,axis=0)
            return ss>1e-8
        keep=supported(raw.astype(float))&supported((raw==1).astype(float))
        raw=raw[:,keep];selected=selected[keep]
        lookup={int(v):j for j,v in enumerate(selected)}
        panel=np.array([lookup[int(v)] for v in candidates if int(v) in lookup][:6])
        if len(panel)!=6:
            raise ValueError('six supported panel loci required')
        axis=original.variants.subset(selected)
        samples=[original.samples[i] for i in rows]
        cv=_aligned_table(reference/'covariates.tsv',samples)[meta['covariates']].to_numpy(float)
        source_identity=original.identity
    def source():
        return ArrayGenotypeSource(raw,samples,axis,hard_calls=True)
    scales=fit_kernel_scales(source(),i0,np.arange(len(selected)),threads=1)
    values=[]
    for j,raw_value in enumerate((raw.astype(float),(raw==1).astype(float))):
        values.append(np.where(raw!=-127,raw_value-scales['mean'][j],0)*scales['inverse_scale'][j])
    g,h=values
    z=cv[:,meta['covariates'].index('PC1')]
    z=z/np.std(z[i0])  # Preserve the declared PC origin.
    contexts=np.column_stack([np.ones(n),z])
    noise=np.column_stack([np.ones(n),z*z])
    def operator(take):
        return PolygenicKernels(source(),take,np.arange(len(selected)),scales,contexts[take],
            noise[take],threads=1,block_size=128,memory_bytes=2**30,storage='packed' if len(take)==n0 else 'stream')
    kernels=np.stack([g@g.T/len(selected),(g*z[:,None])@(g*z[:,None]).T/len(selected),
        h@h.T/len(selected),np.eye(n),np.diag(z*z)])
    small=ArrayGenotypeSource(raw[:,panel],samples,axis.subset(panel),hard_calls=True)
    # The finite panel uses phenotype-independent cohort genotype scaling.
    scale=fit_scale(small,np.arange(n),threads=1)
    study=SelectedStudy(small,np.arange(n),scale,fixed_effects=np.ones((n,1)),
        modifiers=np.ones((n,1)),weights=np.ones((6,1)),component_names=['additive'],
        definitions={'interactions':[],'additive_annotations':['all']},threads=1,memory_bytes=2**30)
    names=small.variants.ids
    annotations={'all':np.ones(6),'a':np.array([1.,.5,1.,0,0,0]),
        'b':np.array([0.,.8,1.,1.,.7,0])}
    groups=[dict(name='within',mode='within',left='a'),
        dict(name='cross',mode='cross',left='a',right='b'),
        dict(name='remainder',mode='remainder',left='a')]
    jobs={'pairs':dict(pairs=[[names[0],names[1]],[names[0],names[2]]])}
    jobs.update({v['name']:dict(groups=[v]) for v in groups})
    jobs['overlap']=dict(groups=groups[:2])
    features={}
    for name,job in jobs.items():
        features[name]=prepare_feature_reference(study,dict(job,additive_annotations=['all']),annotations,
            main_effects='tested_variants',dominance='tested_variants')
    fixed=np.column_stack([features['remainder'].fixed_effects,cv,z*z])
    # Include all tested-locus A/H effects in every model; do not change the
    # null simply because a panel's retained interaction span is different.
    theta=np.array([.8,.3,.4,.6,.2])
    mean=.2*z+.3*g[:,panel[0]]+.2*h[:,panel[1]]
    signal=features['pairs'].features[:,0].copy()
    signal*=np.sqrt(.05/np.var(signal))
    fixed_polygenic=np.sqrt(theta[0])*g@rng.normal(size=len(selected))/np.sqrt(len(selected))
    fixed_polygenic+=np.sqrt(theta[1])*(g*z[:,None])@rng.normal(size=len(selected))/np.sqrt(len(selected))
    fixed_polygenic+=np.sqrt(theta[2])*h@rng.normal(size=len(selected))/np.sqrt(len(selected))
    receipt=dict(source_identity=source_identity,rows_sha256=array_sha256(rows),
        training_rows_sha256=array_sha256(rows[i0]),variants_sha256=array_sha256(selected),
        panel_variants=list(names),n_training=n0,n_confirmation=n1,markers=len(selected),
        panel_scale='cohort genotype-only HWE scale; covariance A/H scales fitted in training',
        contexts='intercept and PC1; original PC1 divided by training SD, no recentering',
        covariance_components=['additive','PC1-dependent additive','dominance','iid noise','PC1-squared noise'],
        units='raw phenotype; common injected pair has cohort variance .05',
        phase='development',held_out_confirmation=False,
        status='bounded mechanism check; N below the production confirmation support floor')
    return operator,kernels,features,fixed,mean,signal,fixed_polygenic,theta,i0,i1,receipt


def restricted_fit(kernels,c0,theta,q0):
    v=np.einsum('k,kij->ij',theta,kernels)
    n0=len(c0)
    p=q0@cho_solve(cho_factor(q0.T@v[:n0,:n0]@q0,lower=True),q0.T)
    return v,p,v[n0:,:n0]@p


def dense_panel(y,f,c1,kernels,v,p,transfer,*,tangent):
    n0=len(p)
    nuisance=c1
    if tangent:
        alpha=p@y[:n0]
        t=np.column_stack([(k[n0:,:n0]-transfer@k[:n0,:n0])@alpha for k in kernels])
        nuisance=np.column_stack([c1,t])
    norms=np.linalg.norm(nuisance,axis=0)
    u=orth(nuisance[:,norms>0]/norms[norms>0],rcond=1e-11)
    raw=f[n0:]-transfer@f[:n0]
    d=raw-u@(u.T@raw)
    scale=np.linalg.norm(d,axis=0)
    absorbed=scale<=1e-11*np.linalg.norm(raw,axis=0)
    d[:,absorbed]=0.;scale[absorbed]=1.
    left,s,right=np.linalg.svd(d/scale,full_matrices=False)
    keep=s>s[0]*1e-11
    if not np.any(keep) or (s[0]/s[keep][-1])**2>1e8:
        raise ValueError('unidentified panel span')
    a=np.linalg.pinv(d/scale,rcond=1e-11).T/scale
    a[:,absorbed]=0.
    beta=a.T@(y[n0:]-transfer@y[:n0])
    cross=v[:n0,n0:]@a
    covariance=a.T@v[n0:,n0:]@a-cross.T@p@cross
    covariance=(covariance+covariance.T)/2
    vv=covariance*scale[:,None]*scale[None,:]
    precision=np.linalg.pinv(vv,rcond=1e-11)*scale[:,None]*scale[None,:]
    return dict(beta=beta,covariance=covariance,information=d.T@d,a=a,
        statistic=float(beta@precision@beta),rank=int(keep.sum()),precision=precision)


def run(a):
    if (min(a.training_samples,a.confirmation_samples)<64
            or a.training_samples+a.confirmation_samples>2048
            or not 16<=a.markers<=4096 or not 1<=a.replicates<=2048):
        raise ValueError('bounded 64<=split N, total N<=2048, 16..4096 markers and 1..2048 replicates required')
    a.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    make,k,refs,c,mean,signal,fixed_polygenic,truth,i0,i1,receipt=design(
        a.reference,a.training_samples,a.confirmation_samples,a.markers,a.seed)
    train=make(i0);complete=make(np.arange(len(c)))
    geometry=he_geometry(train,c[i0],probes=128,seed=a.seed+1)
    order=np.r_[i0,i1];n0=len(i0)
    k=k[:,order][:,:,order];c=c[order];mean=mean[order];signal=signal[order];fixed_polygenic=fixed_polygenic[order]
    q0=null_space(c[:n0].T,rcond=1e-11)
    val,vec=np.linalg.eigh(geometry['h']);root=np.sqrt(val)[:,None]*vec.T
    def estimated(y):
        residual=y[:n0]-geometry['basis']@(geometry['basis'].T@y[:n0])
        moments=np.einsum('i,kij,j->k',residual,k[:,:n0,:n0],residual)/geometry['norms']
        return nnls(root,(vec.T@moments)/np.sqrt(val))[0]/geometry['norms']
    true_v,true_p,true_transfer=restricted_fit(k,c[:n0],truth,q0)
    factor=np.linalg.cholesky(true_v)
    sigma=np.sqrt(truth[-2]+truth[-1]*np.diag(k[-1]))
    settings=['random_combined_null','random_combined_pair','fixed_combined_null']
    methods=['known_random_covariance','estimated_tangent']
    records=[];checks=[]
    receipt.update(arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()},
        source_files={str(path.resolve()):file_digest(path) for path in
            [Path(__file__),*[Path(__file__).resolve().parents[2]/name for name in (
                'src/summit/epistasis/polygenic.py','src/summit/epistasis/features.py',
                'src/summit/epistasis/conditional.py','src/summit/epistasis/robust.py',
                'src/summit/prediction/genotype.py','src/summit/prediction/solver.py')]]},
        methods=methods,settings=settings,
        distinction='known covariance is a diagnostic; estimated_tangent fits each phenotype anew',
        exact_group_features=True,gaussian_errors=True,adaptively_learned_directions=False,
        geometry='production 128-probe nonnegative HE; genotype-only geometry reused',
        thresholds=[.05,.005],parameters_fixed_before_repetitions=True)
    (a.out/'design.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for setting_index,setting in enumerate(settings):
        for rep in range(a.replicates):
            rng=np.random.default_rng(np.random.SeedSequence([a.seed,setting_index,rep]))
            injected=signal if setting.endswith('pair') else np.zeros(len(c))
            if setting.startswith('random'):
                y=mean+injected+factor@rng.normal(size=len(c))
            else:
                y=mean+fixed_polygenic+sigma*rng.normal(size=len(c))
            estimate=estimated(y)
            if rep==0:
                np.testing.assert_allclose(estimate,estimate_components(train,y[:n0],geometry)[:,0],rtol=2e-6,atol=3e-8)
            for method in methods:
                try:
                    theta=truth if method=='known_random_covariance' else estimate
                    v,p,transfer=restricted_fit(k,c[:n0],theta,q0)
                except (ValueError,np.linalg.LinAlgError) as error:
                    for name in refs:
                        records.append(dict(setting=setting,replicate=rep,method=method,panel=name,failed=True,error=str(error)))
                    continue
                for name,reference in refs.items():
                    try:
                        f=reference.features[order]
                        fit=dense_panel(y,f,c[n0:],k,v,p,transfer,tangent=method=='estimated_tangent')
                        probability=float(chi2.sf(fit['statistic'],fit['rank']))
                        # Conditional generating-law diagnostics; no truth enters fitting.
                        aa=fit['a']
                        if setting.startswith('random'):
                            conditional_mean=mean[n0:]+injected[n0:]+true_transfer@(y[:n0]-mean[:n0]-injected[:n0])
                            true_response=aa.T@(injected[n0:]-true_transfer@injected[:n0])
                            actual=aa.T@(true_v[n0:,n0:]-true_v[n0:,:n0]@true_p@true_v[:n0,n0:])@aa
                        else:
                            conditional_mean=mean[n0:]+fixed_polygenic[n0:]
                            true_response=aa.T@(injected[n0:]-transfer@injected[:n0])
                            actual=aa.T@(sigma[n0:,None]**2*aa)
                        displacement=aa.T@(conditional_mean-transfer@y[:n0])-true_response
                        centered=fit['beta']-true_response
                        record=dict(setting=setting,replicate=rep,method=method,panel=name,failed=False,
                            p=probability,rank=fit['rank'],estimated_components=theta.tolist(),
                            joint_95_coverage=bool(centered@fit['precision']@centered<=chi2.ppf(.95,fit['rank'])),
                            conditional_bias_mahalanobis=float(displacement@fit['precision']@displacement),
                            true_to_reported_variance_trace=float(np.trace(fit['precision']@actual)/fit['rank']))
                        if rep==0:
                            native=conditional_score(train,complete,i0,i1,y[:n0],y[n0:],f[:n0],f[n0:],
                                c[:n0],c[n0:],theta,mean_tangents=method=='estimated_tangent')
                            for key in ('beta','covariance','information'):
                                np.testing.assert_allclose(native[key],fit[key],rtol=2e-6,atol=3e-7)
                            # Only the tangent method is labeled by this adapter.
                            if method=='estimated_tangent':
                                identity={key:canonical_sha256([setting,name,key]) for key in
                                    ('genotype_reference','direction','training_outcomes','confirmation_outcomes','null_fit')}
                                summary=conditional_mean_summary(native['beta'],native['covariance'],native['information'],
                                    feature_names=reference.metadata['feature_names'],trait_name='simulated',trait_unit='simulation units',
                                    identities=identity,diagnostics=native['diagnostics'])
                                portable=robust_score_tests(summary)
                                np.testing.assert_allclose(portable['joint_p'],probability,rtol=2e-6,atol=3e-7)
                            checks.append(dict(setting=setting,method=method,panel=name,
                                feature_columns=f.shape[1],rank=fit['rank'],
                                outside_confirmation_design=native['diagnostics']['outside_confirmation_design']))
                        records.append(record)
                    except (ValueError,np.linalg.LinAlgError) as error:
                        records.append(dict(setting=setting,replicate=rep,method=method,panel=name,failed=True,error=str(error)))
        print(setting,'completed',a.replicates,'seconds',round(time.monotonic()-started),flush=True)
    summaries=[]
    for setting in settings:
        for method in methods:
            for name in refs:
                rows=[r for r in records if r['setting']==setting and r['method']==method and r['panel']==name]
                good=[r for r in rows if not r['failed']]
                item=dict(setting=setting,method=method,panel=name,scheduled=a.replicates,completed=len(good),
                    failures=sum(r['failed'] for r in rows))
                if good:
                    item['coverage95']=float(np.mean([r['joint_95_coverage'] for r in good]))
                    for alpha in (.05,.005):
                        hits=sum(r['p']<alpha for r in good)
                        item[str(alpha)]=dict(hits=hits,successful_rate=hits/len(good),
                            binomial_mc95=interval(hits,len(good)),scheduled_denominator=a.replicates)
                    item['mean_bias_mahalanobis']=float(np.mean([r['conditional_bias_mahalanobis'] for r in good]))
                    item['mean_true_to_reported_variance_trace']=float(np.mean([r['true_to_reported_variance_trace'] for r in good]))
                summaries.append(item)
    result=dict(design=receipt,records=records,summaries=summaries,numerical_checks=checks,seconds=time.monotonic()-started)
    with (a.out/'results.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(fits=len(records),failures=sum(r['failed'] for r in records),native_checks=len(checks),
        seconds=result['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--training-samples',type=int,default=256);p.add_argument('--confirmation-samples',type=int,default=512)
    p.add_argument('--markers',type=int,default=512);p.add_argument('--replicates',type=int,default=128)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    existed=args.out.exists()
    try:
        run(args)
    except Exception as error:
        if not existed and args.out.is_dir():
            report=dict(error_type=type(error).__name__,error=str(error),
                status='incomplete; not a completed statistical replicate batch',
                source_sha256=file_digest(Path(__file__)))
            if hasattr(error,'reports'):
                report['solver_reports']={'/'.join(key):value for key,value in error.reports.items()}
            with (args.out/'FAILED.json').open('x') as handle:
                json.dump(report,handle,indent=2,allow_nan=False)
        raise


if __name__=='__main__':
    main()
