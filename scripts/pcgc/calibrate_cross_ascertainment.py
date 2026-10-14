"""Retrospective own-case-status simulations with disjoint cohorts.

Small exact variant-Gram oracle only. Population Gaussian genotypes are sampled
conditionally on each cohort's case status, using shared SNP effects. This tests
point estimation with supplied or fitted risks; it does not certify intervals.

Risk modes: "supplied" (true population risks; needs the explicit
allow_uncalibrated_supplied_risks override), "fitted" (probit design with the
context column passed explicitly), "fitted_default" (no risk arguments; the
model adds every context column, identical to "fitted" here), and the
diagnostic "supplied_recentred" (true risks, then z projected off the
risk-weighted context columns; not a production option).
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr,ndtri
from summit.pcgc.cross_gxe_model import binary_trait
from summit.pcgc.cross_gxe import pair_layout


RISK_MODES=("supplied","fitted","fitted_default","supplied_recentred")


def risk_trait(name,ids,y,phi,risk,prevalence,mode):
    common=dict(context_names=("intercept","E"),liability_sd=1.,
                selection="own_case_status",prevalence=prevalence)
    if mode=="fitted":
        return binary_trait(name,ids,y,phi,risk_covariates=phi[:,1:],**common)
    if mode=="fitted_default":
        return binary_trait(name,ids,y,phi,**common)
    if mode not in ("supplied","supplied_recentred"):
        raise ValueError("unknown risk mode")
    trait=binary_trait(name,ids,y,phi,population_risk=risk,allow_uncalibrated_supplied_risks=True,**common)
    if mode=="supplied_recentred":
        from dataclasses import replace
        weighted=trait.sensitivity[:,None]*trait.contexts
        z=trait.response-weighted@np.linalg.lstsq(weighted,trait.response,rcond=None)[0]
        trait=replace(trait,response=z,metadata={**trait.metadata,"risk_calibration":"diagnostic_recentred"})
    return trait


def one(seed,n,m,*,fitted,rare,blocks=None,products=None,risk_mode=None):
    mode=risk_mode or ("fitted" if fitted else "supplied")
    rng=np.random.default_rng(seed)
    aa=np.diag([.12,.035]);bb=np.diag([.1,.03]);ab=np.array([[.035,.009],[-.004,.012]])
    covariance=np.block([[aa,ab],[ab.T,bb]])
    effects=rng.normal(size=(m,4))@np.linalg.cholesky(covariance).T/np.sqrt(m)
    basis=np.array([[1.,-1.],[1.,1.]])
    xs=[];traits=[]
    for t in range(2):
        beta=effects[:,2*t:2*t+2]
        b=basis@beta.T
        genetic=np.sum(b*b,axis=1)
        if genetic.max()>=1:
            raise RuntimeError("nonpositive environmental variance in retrospective generator")
        k=ndtr((-2.25 if rare else -1.1)+(.45+.15*t)*basis[:,1])
        prevalence=k.mean()
        y=(np.arange(n)<n//2).astype(float);rng.shuffle(y)
        probabilities=np.where(y[:,None]==1,k[None]/k.sum(),(1-k[None])/(1-k).sum())
        context=(rng.uniform(size=n)>probabilities[:,0]).astype(int)
        phi=basis[context];risk=k[context]
        u=rng.uniform(np.finfo(float).eps,1-np.finfo(float).eps,n)
        liability=np.where(y==1,-ndtri(u*risk),ndtri(u*(1-risk)))
        x=rng.normal(size=(n,m))
        noise=rng.normal(size=n)*np.sqrt(1-genetic[context])
        x+=(liability-np.sum(x*b[context],axis=1)-noise)[:,None]*b[context]
        ids=tuple((str(t),str(i)) for i in range(n))
        trait=risk_trait(str(t),ids,y,phi,risk,prevalence,mode)
        xs.append(x);traits.append(trait)
    features=[[x*(t.sensitivity*t.contexts[:,q])[:,None] for q in range(2)] for x,t in zip(xs,traits)]
    if blocks is not None:
        from scripts.pcgc.calibrate_cross_scaling import exact_equations, fit_responses
        from summit.ldscore.matrix_products import MatrixProducts
        design=dict(features=features,traits=traits,ann=np.ones((m,1)),
            products=MatrixProducts(native=False,threads=1) if products is None else products,
            rows=[np.arange(n),np.arange(n,2*n)],truth_matrices=(aa,ab,bb))
        models,truth=exact_equations(design,blocks)
        return fit_responses(design,models,[t.response[:,None] for t in traits]),truth
    point=[];true=[]
    for left,right,matrix in [(0,0,aa),(0,1,ab),(1,1,bb)]:
        layout=pair_layout(2,2,left==right)
        source=[sum(features[left][q]@features[right][r].T for q,r in orient)/m for orient in layout]
        # Independent rectangular person kernels, including exact diagonal exclusion.
        if left==right:
            for kernel in source:
                np.fill_diagonal(kernel,0.)
        H=np.array([[np.sum(x*y) for y in source] for x in source])
        b=np.array([traits[left].response@kernel@traits[right].response for kernel in source])
        point.extend(np.linalg.solve(H,b))
        true.extend([matrix[q,r] for orient in layout for q,r in orient[:1]])
    return np.asarray(point),np.asarray(true)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates",type=int,default=500)
    parser.add_argument("--n",type=int,default=192)
    parser.add_argument("--m",type=int,default=64)
    parser.add_argument("--variant",action="store_true",help="bounded variant-Gram oracle with block uncertainty")
    parser.add_argument("--blocks",type=int,nargs="+",default=[8,32,128])
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--risk-modes",nargs="+",choices=RISK_MODES,default=["supplied","fitted"],
                        help="variant oracle risk modes")
    parser.add_argument("--settings",nargs="+",choices=["common","rare"],default=["common","rare"],
                        help="variant oracle disease settings")
    parser.add_argument("--seed-offset",type=int,default=0,help="first replicate index (variant oracle)")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.replicates<500:
        raise ValueError("at least 500 replicates required")
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.variant:
        return variant_main(args)
    results={};start=time.perf_counter()
    os.umask(0o077)
    for fitted in (False,True):
        for rare in (False,True):
            key=f"{'fitted' if fitted else 'supplied'}_{'rare' if rare else 'common'}"
            receipt=args.out.with_name(args.out.stem+"_"+key+".json")
            if receipt.exists():
                raise FileExistsError(receipt)
            values=[];rejected=[]
            for rep in range(args.replicates):
                try:
                    estimate,truth=one(81838+rep,args.n,args.m,fitted=fitted,rare=rare)
                    values.append(estimate)
                except ValueError as exc:
                    if "risk fit is separated" not in str(exc):
                        raise
                    rejected.append(dict(seed=81838+rep,reason=str(exc)))
            values=np.asarray(values)
            bias=values.mean(0)-truth;sd=values.std(0,ddof=1);mcse=sd/np.sqrt(len(values))
            key=f"{'fitted' if fitted else 'supplied'}_{'rare' if rare else 'common'}"
            result=dict(replicates=len(values),requested_replicates=args.replicates,
                rejected_replicates=rejected,truth=truth.tolist(),bias=bias.tolist(),
                bias_mcse=mcse.tolist(),empirical_sd=sd.tolist(),
                bias_criterion_pass=(abs(bias)<=np.maximum(3*mcse,.05*abs(truth))).tolist(),
                selection="disjoint_by_design_own_case_status",reference="exact_dense_person_kernels",
                intervals="not_validated_for_fitted_nuisances",
                scope="conditional_on_finite_risk_fit_when_replicates_rejected")
            with receipt.open("x") as stream:
                json.dump(result,stream,indent=2)
            results[key]=str(receipt)
            print(json.dumps(dict(scenario=key,seconds=time.perf_counter()-start,
                                  bias=bias.tolist())),flush=True)
    os.umask(0o077)
    with args.out.open("x") as stream:
        json.dump(dict(elapsed_seconds=time.perf_counter()-start,results=results),stream,indent=2)


def variant_main(args):
    import hashlib
    import resource
    from summit.ldscore.matrix_products import MatrixProducts
    from summit.ldscore.generalized_gxe_pass1 import _peak_rss_bytes
    from scripts.pcgc.calibrate_cross_scaling import report
    if args.n<16 or args.m>2048 or min(args.blocks)<2 or max(args.blocks)>args.m:
        raise ValueError("invalid bounded ascertainment dimensions")
    products=MatrixProducts(native=True,threads=args.threads)
    start=time.perf_counter();cpu_start=resource.getrusage(resource.RUSAGE_SELF)
    results={}
    for mode in args.risk_modes:
        for rare in [setting=="rare" for setting in args.settings]:
            fitted=mode.startswith("fitted")
            key=f"{mode}_{'rare' if rare else 'common'}"
            receipt=args.out.with_name(args.out.stem+"_"+key+".json")
            if receipt.exists(): raise FileExistsError(receipt)
            points={j:[] for j in args.blocks};variances={j:[] for j in args.blocks}
            ratios={j:[] for j in args.blocks};covariances={j:[] for j in args.blocks}
            rejected=[]
            for rep in range(args.seed_offset,args.seed_offset+args.replicates):
                try:
                    fitted_values,truth=one(81838+rep,args.n,args.m,fitted=fitted,rare=rare,
                                           blocks=args.blocks,products=products,risk_mode=mode)
                except ValueError as exc:
                    if "risk fit is separated" not in str(exc): raise
                    rejected.append(dict(seed=81838+rep,reason=str(exc)))
                    continue
                for count,(point,loo) in fitted_values.items():
                    point=point[0];loo=loo[:,0];centered=loo-loo.mean(0)
                    covariance=(count-1)/count*centered.T@centered
                    contrast=np.eye(10)[[0,7,3]]
                    v=contrast@point;vcov=contrast@covariance@contrast.T
                    ratio=np.full(3,np.nan)
                    if min(v[:2])>0:
                        ratio[0]=v[2]/np.sqrt(v[0]*v[1])
                        gradient=np.array([-ratio[0]/(2*v[0]),-ratio[0]/(2*v[1]),1/np.sqrt(v[0]*v[1])])
                        ratio[1]=np.sqrt(max(0,gradient@vcov@gradient))
                        ratio[2]=np.any(v[:2]-1.96*np.sqrt(np.diag(vcov)[:2])<=0)
                    points[count].append(point);variances[count].append(np.diag(covariance))
                    ratios[count].append(ratio);covariances[count].append(covariance)
            value={str(j):report(np.asarray(points[j]),np.asarray(variances[j]),np.asarray(ratios[j]),
                    truth,j,np.ones(len(points[j]),dtype=bool),np.mean(covariances[j],axis=0)) for j in args.blocks}
            result=dict(configuration={**vars(args),"out":str(args.out)},risk_mode=mode,fitted=fitted,rare=rare,
                selection="disjoint_by_design_own_case_status",reference="exact_bounded_variant_Gram_oracle",
                rejected_risk_fits=rejected,results=value,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            os.umask(0o077)
            with receipt.open("x") as stream: json.dump(result,stream,indent=2)
            results[key]=str(receipt)
            print(json.dumps(dict(scenario=key,elapsed_seconds=time.perf_counter()-start,
                coverage={j:[min(v["normal"]["coverage"]),max(v["normal"]["coverage"])] for j,v in value.items()})),flush=True)
    cpu_end=resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds=cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime
    with args.out.open("x") as stream:
        json.dump(dict(elapsed_seconds=time.perf_counter()-start,cpu_seconds=cpu_seconds,
            peak_rss_bytes=_peak_rss_bytes(),results=results),stream,indent=2)


if __name__=="__main__":
    main()
