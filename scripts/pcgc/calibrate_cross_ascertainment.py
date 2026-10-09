"""Retrospective own-case-status simulations with disjoint cohorts.

Small exact variant-Gram oracle only. Population Gaussian genotypes are sampled
conditionally on each cohort's case status, using shared SNP effects. This tests
point estimation with supplied or fitted risks; it does not certify intervals.
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


def one(seed,n,m,*,fitted,rare):
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
        trait=binary_trait(str(t),ids,y,phi,context_names=("intercept","E"),liability_sd=1.,
            selection="own_case_status",prevalence=prevalence,
            population_risk=None if fitted else risk,
            risk_covariates=phi[:,1:] if fitted else None)
        xs.append(x);traits.append(trait)
    features=[[x*(t.sensitivity*t.contexts[:,q])[:,None] for q in range(2)] for x,t in zip(xs,traits)]
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
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.replicates<500:
        raise ValueError("at least 500 replicates required")
    if args.out.exists():
        raise FileExistsError(args.out)
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
                    estimate,truth=one(81838+rep,192,64,fitted=fitted,rare=rare)
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


if __name__=="__main__":
    main()
