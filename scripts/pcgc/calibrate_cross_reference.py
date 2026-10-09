"""Joint trait/reference calibration using a bounded exact variant-Gram oracle.

Each replicate redraws effects, residuals and the shared global reference
probes. No individual-level results are saved. This is a research calculation,
not a scalable replacement for production's two-pass genotype reference.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec
from scripts.pcgc.calibrate_cross_scaling import (
    make_design,exact_equations,responses,fit_responses,report)
from scripts.pcgc.calibrate_cross_gxe import wilson


def random_equations(design, models, probes, seed):
    products=design["products"];a=design["ann"];mass=a.sum(0);k=len(mass);m=len(a)
    z=GlobalVariantProbeSpec(seed,0,probes).generate(np.arange(m))
    rhs=(np.sqrt(a)[:,:,None]*z[:,None,:]).reshape(m,-1)
    panels=[]
    for gram in design["grams"]:
        panels.append({key:products.nn(value,rhs).reshape(m,k,probes) for key,value in gram.items()})
    random_models=[]
    for model in models:
        l,r=model["left"],model["right"];layout=model["layout"];p=len(layout);c=k*p
        ld=np.empty((m,p,c));draw=np.zeros((probes,c,c))
        for target,orientation in enumerate(layout):
            for source,orientation2 in enumerate(layout):
                value=sum(panels[l][q,u]*panels[r][s,v]
                          for q,s in orientation for u,v in orientation2)
                ld[:,target,source::p]=value.mean(2)/mass
                for annotation in range(k):
                    draw[:,target::p,annotation*p+source]=(a.T@value[:,annotation]).T/mass[annotation]
        work={}
        for count,exact in model["blocks"].items():
            bounds=exact["boundaries"]
            hs=np.stack([a[first:last].T@ld[first:last].reshape(last-first,-1)
                         for first,last in zip(bounds[:-1],bounds[1:])]).reshape(count,c,c)
            hs-=exact["correction_blocks"];h=hs.sum(0)
            if np.linalg.cond(h)>1e8:
                raise ValueError("weak randomized normal equations")
            work[count]={**exact,"H":h,"H_blocks":hs,"inverse":np.linalg.inv(h),
                         "deleted_inverse":np.linalg.inv(h[None]-hs)}
        random_models.append({**model,"blocks":work,"draw":draw})
    products.drain()
    return random_models


def joint_covariance(models, point, loo, blocks):
    centered=loo-loo.mean(0)
    snp=(blocks-1)/blocks*centered.T@centered
    actions=[];offset=0
    for model in models:
        work=model["blocks"][blocks];count=len(work["H"])
        dev=model["draw"]-model["draw"].mean(0)
        action=-np.linalg.solve(work["H"],np.einsum("bij,j->bi",dev,point[offset:offset+count]).T).T
        actions.append(action);offset+=count
    actions=np.concatenate(actions,axis=1)
    reference=actions.T@actions/(len(actions)*(len(actions)-1))
    return snp,reference


def ratio(point,cov,k):
    contrast=np.zeros((3,len(point)))
    contrast[0,np.arange(k)*3]=1
    contrast[1,7*k+np.arange(k)*3]=1
    contrast[2,3*k+np.arange(k)*4]=1
    v=contrast@point;vcov=contrast@cov@contrast.T
    out=np.full(3,np.nan)
    if min(v[:2])>0:
        out[0]=v[2]/np.sqrt(v[0]*v[1])
        gradient=np.array([-out[0]/(2*v[0]),-out[0]/(2*v[1]),1/np.sqrt(v[0]*v[1])])
        out[1]=np.sqrt(max(0,gradient@vcov@gradient))
        out[2]=np.any(v[:2]-1.96*np.sqrt(np.diag(vcov)[:2])<=0)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",type=int,default=1024);parser.add_argument("--m",type=int,default=512)
    parser.add_argument("--replicates",type=int,default=1000)
    parser.add_argument("--blocks",type=int,default=128)
    parser.add_argument("--probes",type=int,nargs="+",default=[32,128,512])
    parser.add_argument("--kind",choices=["qq","bq","bb"],default="bb")
    parser.add_argument("--overlap",type=float,default=.5)
    parser.add_argument("--annotations",type=int,default=1)
    parser.add_argument("--prevalence",type=float,default=.2)
    parser.add_argument("--seed",type=int,default=617340)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    if args.replicates<500 or not 2<=args.blocks<=args.m or args.m>2048 or min(args.probes)<2:
        raise ValueError("invalid bounded reference calibration dimensions")
    start=perf_counter()
    design=make_design(args.n,args.m,kind=args.kind,overlap=args.overlap,annotations=args.annotations,
        seed=args.seed,prevalence=args.prevalence,native=True,threads=args.threads)
    models,truth=exact_equations(design,[args.blocks],retain_grams=True)
    rng=np.random.default_rng(args.seed+92481)
    data={b:dict(point=[],variance=[],ratio=[],cov=[],reference=[],exact=[],eligible=[],student_hits=[]) for b in args.probes}
    for rep in range(args.replicates):
        z,eligible=responses(design,rng,1)
        exact=fit_responses(design,models,z)[args.blocks][0][0]
        for probes in args.probes:
            random=random_equations(design,models,probes,args.seed+rep+1)
            point,loo=fit_responses(design,random,z)[args.blocks]
            point=point[0];loo=loo[:,0]
            snp,mc=joint_covariance(random,point,loo,args.blocks);cov=snp+mc
            v=np.diag(cov);vs=np.diag(snp);vr=np.diag(mc)
            df=v*v/(vs*vs/(args.blocks-1)+vr*vr/(probes-1))
            hit=abs(point-truth)<=student_t.ppf(.975,df)*np.sqrt(v)
            d=data[probes]
            d["point"].append(point);d["variance"].append(v);d["cov"].append(cov)
            d["ratio"].append(ratio(point,cov,args.annotations));d["reference"].append(mc)
            d["exact"].append(exact);d["eligible"].append(bool(eligible[0]));d["student_hits"].append(hit)
    result={}
    for probes,d in data.items():
        point=np.asarray(d["point"]);delta=point-np.asarray(d["exact"]);mc=np.mean(d["reference"],axis=0)
        value=report(point,np.asarray(d["variance"]),np.asarray(d["ratio"]),truth,args.blocks,
                     np.asarray(d["eligible"]),np.mean(d["cov"],axis=0))
        hit=np.asarray(d["student_hits"]).sum(0)
        value.update(reference_probes=probes,mean_reference_covariance=mc.tolist(),
            paired_reference_bias=delta.mean(0).tolist(),
            paired_reference_bias_mcse=(delta.std(0,ddof=1)/np.sqrt(len(delta))).tolist(),
            paired_reference_mse=(delta*delta).mean(0).tolist(),
            satterthwaite_coverage=(hit/len(point)).tolist(),
            satterthwaite_coverage_wilson95=[wilson(h,len(point)) for h in hit])
        result[str(probes)]=value
    os.umask(0o077);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f:
        json.dump(dict(configuration={**vars(args),"out":str(args.out)},results=result,
            elapsed_seconds=perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
    print(json.dumps({b:dict(normal=[min(v["normal"]["coverage"]),max(v["normal"]["coverage"])],
        student=[min(v["satterthwaite_coverage"]),max(v["satterthwaite_coverage"])]) for b,v in result.items()}),flush=True)


if __name__=="__main__":
    main()
