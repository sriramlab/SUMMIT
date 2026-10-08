"""Frozen bounded confirmation: separate valid Gaussian models and misspecification."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.epistasis.validate import interval
from summit.epistasis.cli import _jsonable
from summit.epistasis.score import prepare_linear_scores,linear_score_tests,refit_bootstrap
from summit.epistasis.null import GaussianNullReference,GeneralGaussianNullReference
from summit.epistasis.oracle import explicit_pair_features,dense_summary
from summit.epistasis.summary import fit_epistasis
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.prediction.genotype import FileGenotypeSource


def panel(path,seed,n=192,m=96):
    rng=np.random.default_rng(seed)
    if path:
        with FileGenotypeSource(path) as source:
            rows=np.sort(rng.choice(len(source.samples),n,replace=False))
            start=int(rng.integers(5000,len(source.variants.ids)-3000))
            chosen=np.r_[np.arange(start,start+768),np.linspace(0,len(source.variants.ids)-1,512).astype(int)]
            chosen=np.unique(chosen)
            source.prepare(rows,len(chosen),1)
            raw=source.read(chosen).astype(float);raw[raw==-127]=np.nan
            af=np.nanmean(raw,axis=0)/2
            keep=np.flatnonzero((af>.15)&(af<.85)&(np.mean(np.isnan(raw),axis=0)<.02))
            # Local LD plus distal background, without phenotype selection.
            local=keep[(chosen[keep]>=start)&(chosen[keep]<start+768)]
            distal=np.setdiff1d(keep,local)
            selected=np.r_[local[:m//2],distal[:m//2]]
            if len(selected)!=m: raise ValueError("insufficient real panel support")
            meta=dict(source_identity=source.identity,seed=seed,n=n,m=m,
                      variant_indices=chosen[selected].tolist(),window_start=start)
            raw=raw[:,selected]
    else:
        # Discrete diploid haplotypes with local LD.
        raw=np.zeros((n,m))
        for hap in range(2):
            h=rng.binomial(1,.3,(n,m))
            for j in range(1,m):
                if j%8: h[:,j]=np.where(rng.random(n)<.65,h[:,j-1],h[:,j])
            raw+=h
        meta=dict(seed=seed,n=n,m=m,kind="synthetic_discrete_LD")
    mean=np.nanmean(raw,axis=0);x=(raw-mean)/np.sqrt(mean*(1-mean/2));x=np.nan_to_num(x)
    h=(raw==1).astype(float)
    correlation=np.corrcoef(x.T);np.fill_diagonal(correlation,0)
    meta.update(max_abs_ld=float(abs(correlation).max()),missing_fraction=float(np.mean(np.isnan(raw))),
                min_genotype_cell=int(min(np.sum(raw==j,axis=0).min() for j in (0,1,2))))
    return x,h,meta


def add_record(records,setting,method,rep,p,error=None,**kwargs):
    records.append(dict(setting=setting,method=method,replicate=rep,p=p,failed=error is not None,error=error,**kwargs))


def linear_panel(x,h,seed,reps,label,records,design):
    n,m=x.shape;rng=np.random.default_rng(seed)
    a=np.zeros(m);a[0:4]=1;b=np.zeros(m);b[4:8]=1;over=b.copy();over[2:4]=1
    cases={"cross":(a,b,False),"overlap":(a,over,False),"within":(a,a,True),"remainder":(a,(a==0).astype(float),False)}
    # Same fixed effect policy for every family, includes dominance of tested
    # group loci. No causal-variant oracle: all observed additive loci included.
    fixed=np.column_stack([np.ones(n),x,h[:,:8]])
    u=thin_rank_revealing_fixed_effect_basis(fixed)
    r=n-u.shape[1]
    fixed_mean=x@rng.normal(size=m)/np.sqrt(m)+h[:,:8]@rng.normal(size=8)/3
    for name,(left,right,within) in cases.items():
        f,_,_=explicit_pair_features(x,left,right,within=within)
        # A large independent-effects group uses a frozen feature projection.
        if f.shape[1]>=r:
            f=f@rng.normal(size=(f.shape[1],32))/np.sqrt(32)
            name+="_fixed_sketch32"
        pf=f-u@(u.T@f)
        coefficient=rng.normal(size=f.shape[1]);coefficient*=np.sqrt(.03*r/(np.sum((pf@coefficient)**2)))
        for alt in (False,True):
            y=fixed_mean[:,None]+(f@coefficient)[:,None]*alt+rng.normal(size=(n,reps))
            setting=f"{label}_{name}_{'signal03' if alt else 'null'}"
            summary=prepare_linear_scores(f,y,fixed,feature_names=tuple(f"p{i}" for i in range(f.shape[1])),
                trait_names=tuple(map(str,range(reps))),metadata={})
            design.append(dict(setting=setting,n=n,m=m,features=f.shape[1],residual_rank=r,
                effects="fixed across residual replicates",realized_signal_variance=.03 if alt else 0))
            for i in range(reps):
                try:
                    result=linear_score_tests(summary,trait=i,burden=np.ones(f.shape[1]))
                    for method,key in (("kernel","kernel_p"),("burden","burden_p"),("sparse","sparse_bonferroni_p"),("adaptive","adaptive_bonferroni_p")):
                        add_record(records,setting,method,i,result[key])
                    coverage=np.mean((result["joint_beta_interval_95"][:,0]<=coefficient*alt)&(result["joint_beta_interval_95"][:,1]>=coefficient*alt)) if "joint_beta_interval_95" in result else np.nan
                    records[-4].update(mean_beta_coverage=coverage)
                except (ValueError,ArithmeticError,np.linalg.LinAlgError) as e:
                    for method in ("kernel","burden","sparse","adaptive"): add_record(records,setting,method,i,np.nan,str(e))


def estimated_panel(x,h,seed,reps,draws,label,records,design):
    rng=np.random.default_rng(seed);n,m=x.shape
    f=x[:,0,None]*x[:,8:14]/np.sqrt(6)
    fixed=np.column_stack([np.ones(n),x[:,0],x[:,8:14]])
    g=x@x.T/m
    ref=GaussianNullReference(g,fixed,identity=label)
    rotated=ref.rotation.T@f
    k=rotated@rotated.T
    # Same supplied six-pair hypotheses and estimated nuisance information.
    score_kernels=[k,rotated@np.ones((6,6))@rotated.T]+[np.outer(rotated[:,i],rotated[:,i]) for i in range(6)]
    p=ref.rotation@ref.rotation.T
    fame_k=np.stack([p@g@p,p@f@f.T@p,p])
    for case in ("null","mixed03","sparse03","burden03","dominance_misspecified","heteroskedastic_misspecified","heavy_tail","omitted_additive"):
        noise=np.full(n,.7)
        if case=="heteroskedastic_misspecified":noise=.7*(.3+.7*x[:,0]**2)
        y=np.sqrt(.3/m)*x@rng.normal(size=(m,reps))
        y+=np.sqrt(noise)[:,None]*(rng.standard_t(4,(n,reps))/np.sqrt(2) if case=="heavy_tail" else rng.normal(size=(n,reps)))
        truth_g=.3*g+np.diag(noise)
        signal=np.zeros((n,reps));expected_signal=0.
        if case in ("mixed03","sparse03","burden03"):
            sf=f if case=="mixed03" else f[:,:1] if case=="sparse03" else f.sum(axis=1,keepdims=True)
            sf=sf*np.sqrt(.03*ref.rank/np.sum((p@sf)**2))
            signal=sf@rng.normal(size=(sf.shape[1],reps));y+=signal
            truth_g+=sf@sf.T;expected_signal=.03
        if case=="dominance_misspecified":
            nuisance=h[:,0]-h[:,0].mean();nuisance/=np.std(nuisance)
            y+=np.sqrt(.4)*nuisance[:,None]*rng.normal(size=(1,reps));truth_g+=.4*np.outer(nuisance,nuisance)
        actual_ref=ref;actual_k=score_kernels;actual_fame=fame_k
        if case=="omitted_additive":
            # Broader causal panel is already in generating g; fitted excludes
            # every other distal marker and all local variants except target.
            fit_indices=np.r_[0,np.arange(8,m,2)]
            wrong_g=x[:,fit_indices]@x[:,fit_indices].T/len(fit_indices)
            actual_ref=GaussianNullReference(wrong_g,fixed,identity=label+case)
            z=actual_ref.rotation.T@f
            actual_k=[z@z.T,z@np.ones((6,6))@z.T]+[np.outer(z[:,i],z[:,i]) for i in range(6)]
            actual_fame=np.stack([p@wrong_g@p,fame_k[1],p])
        setting=f"{label}_{case}"
        summary=dense_summary(actual_fame,y,component_names=("additive","epistasis","residual"),
            trait_names=tuple(map(str,range(reps))),residual_rank=ref.rank,metadata={"setting":setting})
        expected=np.linalg.solve(summary.matrix,np.einsum("aij,ji->a",actual_fame,truth_g))[1]
        design.append(dict(setting=setting,n=n,m=m,expected_interaction_variance=expected_signal,
            mean_realized_interaction_variance=float(np.mean(np.sum((p@signal)**2,axis=0)/ref.rank)),
            pseudo_true_fame_coefficient=float(expected),biological_epistasis=expected_signal>0,
            effects="random effects and residuals regenerated; fixed causal set and covariance coefficients"))
        for i in range(reps):
            fit=fit_epistasis(summary,i)
            add_record(records,setting,"FAME_Wald",i,fit["wald_p_one_sided"][1],None if fit["covariance_valid"] else "invalid covariance",
                coefficient=fit["coefficients"][1],se=fit["standard_errors"][1],expected=expected,
                covered=bool(abs(fit["coefficients"][1]-expected)<=1.95996398454*fit["standard_errors"][1]))
            try:
                boot=refit_bootstrap(actual_ref,actual_k,y[:,i],draws=draws,seed=seed+10000+i)
                sparse=(1+np.sum(boot["bootstrap_statistics"][:,2:].max(axis=1)>=boot["statistics"][2:].max()))/(draws+1)
                for method,value in (("REML_kernel",boot["p"][0]),("REML_burden",boot["p"][1]),("REML_sparse",sparse),
                                     ("REML_adaptive",min(1.,3*min(boot["p"][0],boot["p"][1],sparse)))):
                    add_record(records,setting,method,i,value)
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as e:
                for method in ("REML_kernel","REML_burden","REML_sparse","REML_adaptive"):
                    add_record(records,setting,method,i,np.nan,str(e))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--real-genotypes")
    parser.add_argument("--family",choices=("linear","estimated"),required=True)
    parser.add_argument("--replicates",type=int,default=100);parser.add_argument("--bootstrap-draws",type=int,default=99)
    parser.add_argument("--seed-offset",type=int,default=0)
    args=parser.parse_args()
    if not 1<=args.replicates<=100:raise ValueError("bounded full-fit panel")
    args.out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((Path(__file__).resolve().parents[2]/"benchmarks/epistasis/confirmation_design_v2.json").read_text())
    records=[];design=[];panels=[];started=time.perf_counter()
    for i,seed in enumerate(frozen["genotype_seeds"]):
        seed+=args.seed_offset
        x,h,meta=panel(args.real_genotypes,seed,n=384 if args.family=="linear" else 192,
                       m=96 if args.family=="linear" else 256);panels.append(meta)
        label=("real" if args.real_genotypes else "discrete")+str(i)
        if args.family=="linear":linear_panel(x,h,frozen["phenotype_seed"]+i+args.seed_offset,args.replicates,label,records,design)
        else:estimated_panel(x,h,frozen["phenotype_seed"]+i+args.seed_offset,args.replicates,args.bootstrap_draws,label,records,design)
        print(label,"finished",round(time.perf_counter()-started,2),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(args.out/"replicates.csv",index=False)
    report=[]
    for (setting,method),part in frame.groupby(["setting","method"],sort=False):
        hits=int((part.p<=.05).sum());lo,hi=interval(hits,len(part))
        row=dict(setting=setting,method=method,fits=len(part),failures=int(part.failed.sum()),rejection=hits/len(part),lower=lo,upper=hi,
                 rejection_valid=float((part.loc[~part.failed,"p"]<=.05).mean()))
        if "coefficient" in part and part.coefficient.notna().any():
            row.update(bias=float((part.coefficient-part.expected).mean()),sd=part.coefficient.std(),mean_se=part.se.mean(),coverage=float(np.nanmean(part.covered.to_numpy(dtype=float))))
        if "mean_beta_coverage" in part:row["mean_beta_coverage"]=part.mean_beta_coverage.mean()
        report.append(row)
    table=pd.DataFrame(report);table.to_csv(args.out/"summary.csv",index=False)
    (args.out/"design.json").write_text(json.dumps(_jsonable(dict(frozen=frozen,settings=design,panels=panels,
        actual_replicates=args.replicates,bootstrap_draws=args.bootstrap_draws,seed_offset=args.seed_offset,seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),indent=2,allow_nan=False)+"\n")
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
    settings=list(table.setting.unique());methods=list(table.method.unique())
    for j,method in enumerate(methods):
        part=table[table.method==method]
        ax.errorbar([settings.index(s)+(j-(len(methods)-1)/2)*.12 for s in part.setting],part.rejection,
            yerr=np.array([part.rejection-part.lower,part.upper-part.rejection]),fmt="o",ms=3,label=method)
    ax.axhline(.05,color="grey");ax.set_xticks(range(len(settings)),settings,rotation=60,ha="right",fontsize=7)
    ax.set_ylabel("Rejection / power at 0.05, exact binomial 95% interval");ax.legend(fontsize=8)
    fig.savefig(args.out/"confirmation.png",dpi=170)
    print(table.to_string(index=False))


if __name__=="__main__":main()
