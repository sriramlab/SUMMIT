"""Known-truth calibration of conditional SNP-block cross-PCGC intervals.

Uses independent dense person kernels on small fixed designs. The reference
is exact to isolate sampling coverage from randomized-reference noise. Gaussian
SNP effects, individual residuals and (where requested) liabilities are redrawn.
This is not a study of unknown prevalence or population-moment estimation.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import t as student_t
from summit.pcgc.cross_gxe_model import binary_trait,quantitative_trait
from summit.pcgc.cross_gxe import pair_layout
from summit.context.annotations import _jackknife_covariance


def wilson(count,n):
    p=count/n;z=1.959963984540054;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(center-half),float(center+half)]


def scenario(seed,kind,n,m,q,k,overlap,signal,risk_strength,prevalence,heterogeneous):
    rng=np.random.default_rng(seed)
    shared=int(n*overlap);total=2*n-shared
    rows=[np.arange(n),np.r_[np.arange(shared),np.arange(n,total)]]
    ids=tuple((str(i),str(i)) for i in range(total))
    x=rng.normal(size=(total,m))
    annotations=np.eye(k)[np.arange(m)%k]
    phi=[np.column_stack((np.ones(n),rng.uniform(-1,1,(n,q-1)))) for _ in range(2)]
    sds=[np.sqrt(1+.35*p[:,1]) if heterogeneous and q>1 else np.ones(n) for p in phi]
    risks=[ndtr(ndtri(prevalence)+risk_strength*(p[:,1] if q>1 else np.zeros(n))) for p in phi]
    traits=[]
    for t in range(2):
        common=dict(name=str(t),samples=tuple(ids[i] for i in rows[t]),contexts=phi[t],
                    context_names=tuple(str(i) for i in range(q)),selection="outcome_independent")
        if kind[t]=="b":
            traits.append(binary_trait(y=np.arange(n)%2,population_risk=risks[t],liability_sd=sds[t],**common))
        else:
            design=np.column_stack((np.ones(n),np.cos(np.arange(n)*2*np.pi/n),
                                    np.sin(np.arange(n)*2*np.pi/n))) if t else np.ones((n,1))
            # Recorded experiments used exactly these designs (no added context main effects).
            traits.append(quantitative_trait(y=np.zeros(n),population_sd=1.,fixed_effects=design,
                                             context_main_effects=False,**common))
    aa=np.diag([.12]+[.035]*(q-1));bb=np.diag([.10]+[.03]*(q-1))
    ab=np.diag([signal]+[signal/3]*(q-1))
    if q>1 and signal:
        ab[0,1]=.01;ab[1,0]=-.004
    if np.linalg.eigvalsh(np.block([[aa,ab],[ab.T,bb]])).min()<=0:
        raise ValueError("invalid generating joint effect covariance")
    generating={(0,0):aa,(1,1):bb,(0,1):ab}
    raw=[np.stack([x[r]*p[:,j,None] for j in range(q)]) for r,p in zip(rows,phi)]
    g={}
    for left,right in generating:
        g[left,right]=np.einsum("qim,qr,rjm->ij",raw[left],generating[left,right],raw[right])/m
    ga,gb=g[0,0],g[1,1]
    residual=[sds[t]**2-np.diag(g[t,t]) for t in range(2)]
    if any(np.any(r<=0) for r in residual):
        raise ValueError("nonpositive generating environmental variance")
    cross=g[0,1].copy()
    _,ia,ib=np.intersect1d(*rows,return_indices=True)
    cross[ia,ib]+=.2*np.sqrt(residual[0][ia]*residual[1][ib])
    latent=np.block([[ga+np.diag(residual[0]),cross],[cross.T,gb+np.diag(residual[1])]])
    root=np.linalg.cholesky(latent)
    projections=[np.eye(n)-t.fixed_basis@t.fixed_basis.T for t in traits]
    f=[[projections[t]@(raw[t][q_]*traits[t].sensitivity[:,None]) for q_ in range(q)] for t in range(2)]
    blocks=np.arange(m)*8//m
    # Aggregate dense nuisance-orthogonal instruments for each SNP block.
    models=[]
    truth=[]
    for left,right in [(0,0),(0,1),(1,1)]:
        layout=pair_layout(q,q,left==right);p=len(layout)
        _,ia,ib=np.intersect1d(rows[left],rows[right],return_indices=True)
        pa,pb=projections[left],projections[right]
        C=pa[np.ix_(ia,ia)]*pb[np.ix_(ib,ib)]
        source=[]
        targets=[]
        for ann in annotations.T:
            for orientations in layout:
                source.append(sum((f[left][u]*ann)@f[right][v].T for u,v in orientations)/ann.sum())
        for block in range(8):
            part=[]
            for ann in annotations.T:
                weight=ann*(blocks==block)/ann.sum()
                for orientations in layout:
                    T=sum((f[left][u]*weight)@f[right][v].T for u,v in orientations)
                    if len(ia):
                        coeff=np.linalg.solve(C,T[ia,ib])
                        T-= (pa[:,ia]*coeff)@pb[ib,:]
                    part.append(T)
            targets.append(part)
        targets=np.asarray(targets);source=np.asarray(source)
        H=np.einsum("bcij,dij->bcd",targets,source)
        full=H.sum(0)
        if np.linalg.cond(full)>1e8:
            raise ValueError("weak generating normal equations")
        inverse=np.linalg.inv(full)
        deleted=np.linalg.inv(full[None]-H)
        matrix=generating[left,right]/k
        theta=np.tile(np.array([matrix[u,v] for orientations in layout for u,v in orientations[:1]]),k)
        truth.extend(theta)
        models.append((left,right,targets,inverse,deleted))
    oracle_se=None
    if kind=="qq":
        instruments=[]
        for left,right,targets,inverse,_ in models:
            fitted=np.einsum("cd,dij->cij",inverse,targets.sum(0))
            for instrument in fitted:
                joint=np.zeros((2*n,2*n))
                if left==right:
                    joint[left*n:(left+1)*n,left*n:(left+1)*n]=instrument
                else:
                    joint[:n,n:]=instrument/2
                    joint[n:,:n]=instrument.T/2
                instruments.append(joint)
        products=np.asarray(instruments)@latent
        oracle_se=np.sqrt(2*np.einsum("cij,cji->c",products,products))
    return dict(models=models,truth=np.asarray(truth),root=root,traits=traits,
                risks=risks,sds=sds,n=n,kind=kind,seed=seed,oracle_se=oracle_se,q=q,k=k)


def run(spec,replicates):
    rng=np.random.default_rng(spec["seed"]+1098)
    estimates=[];ses=[];cover=[];rejections=[];eligible=[]
    ratios=[];ratio_ses=[];weak=0
    q,k=spec["q"],spec["k"]
    within=q*(q+1)//2
    contrast=np.zeros((3,len(spec["truth"])))
    contrast[0,np.arange(k)*within]=1
    contrast[1,k*(within+q*q)+np.arange(k)*within]=1
    contrast[2,k*within+np.arange(k)*q*q]=1
    true_variances=contrast@spec["truth"]
    true_rg=true_variances[2]/np.sqrt(true_variances[0]*true_variances[1])
    n=spec["n"]
    for rep in range(replicates):
        latent=spec["root"]@rng.normal(size=2*n)
        z=[]
        nondegenerate=True
        for t in range(2):
            value=latent[t*n:(t+1)*n]
            if spec["kind"][t]=="b":
                y=(value+spec["sds"][t]*ndtri(spec["risks"][t])>0).astype(float)
                nondegenerate=nondegenerate and bool(0<y.sum()<n)
                risk=spec["risks"][t]
                value=(y-risk)/np.sqrt(risk*(1-risk))
            z.append(value)
        point=[];deleted=[]
        for left,right,targets,inverse,loo_inv in spec["models"]:
            b=np.einsum("i,bcij,j->bc",z[left],targets,z[right],optimize=True)
            point.append(inverse@b.sum(0))
            deleted.append(np.einsum("bij,bj->bi",loo_inv,b.sum(0)[None]-b))
        point=np.concatenate(point)
        loo=np.concatenate(deleted,axis=1)
        covariance=_jackknife_covariance(loo)
        se=np.sqrt(np.maximum(0,np.diag(covariance)))
        v=contrast@point;vcov=contrast@covariance@contrast.T
        if v[0]>0 and v[1]>0:
            r=v[2]/np.sqrt(v[0]*v[1])
            jac=np.array([-r/(2*v[0]),-r/(2*v[1]),1/np.sqrt(v[0]*v[1])])
            ratios.append(r)
            ratio_ses.append(np.sqrt(max(0,jac@vcov@jac)))
            weak+=int(np.any(v[:2]-1.96*np.sqrt(np.diag(vcov)[:2])<=0))
        eligible.append(nondegenerate)
        estimates.append(point);ses.append(se)
        cover.append(abs(point-spec["truth"])<=1.959963984540054*se)
        rejections.append(abs(point)>1.959963984540054*se)
    estimates=np.asarray(estimates);ses=np.asarray(ses)
    bias=estimates.mean(0)-spec["truth"]
    sd=estimates.std(0,ddof=1);mcse=sd/np.sqrt(replicates)
    cover=np.asarray(cover).sum(0);reject=np.asarray(rejections).sum(0)
    intervals=np.array([wilson(c,replicates) for c in cover])
    out=dict(replicates=replicates,truth=spec["truth"].tolist(),bias=bias.tolist(),bias_mcse=mcse.tolist(),
        empirical_sd=sd.tolist(),mean_reported_se=ses.mean(0).tolist(),se_to_empirical_sd=(ses.mean(0)/sd).tolist(),
        coverage=(cover/replicates).tolist(),coverage_wilson95=intervals.tolist(),
        null_type1=[float(reject[i]/replicates) if v==0 else None for i,v in enumerate(spec["truth"])],
        null_type1_wilson95=[wilson(int(reject[i]),replicates) if v==0 else None for i,v in enumerate(spec["truth"])],
        bias_criterion_pass=(abs(bias)<=np.maximum(3*mcse,.05*abs(spec["truth"]))).tolist(),
        coverage_criterion_pass=((intervals[:,0]<=.95)&(intervals[:,1]>=.95)).tolist(),
        reference="independent_exact_dense_kernels",
        uncertainty="paired_frozen_target_SNP_block_only",
        scope="fixed_genotype_context_design_supplied_risks_and_scales",
        omitted="ascertainment_fitted_risks_population_moments_randomized_reference")
    eligible=np.asarray(eligible)
    out["production_rejected_constant_binary_replicates"]=int(np.sum(~eligible))
    out["coverage_among_nondegenerate_outcomes"]=np.mean(
        abs(estimates[eligible]-spec["truth"])<=1.959963984540054*ses[eligible],axis=0).tolist()
    out["unconditional_experiment"]=True
    ratios=np.asarray(ratios);ratio_ses=np.asarray(ratio_ses)
    covered=int(np.sum(abs(ratios-true_rg)<=1.959963984540054*ratio_ses))
    out["baseline_rg"]=dict(truth=float(true_rg),defined_replicates=len(ratios),
        undefined_nonpositive_denominator=replicates-len(ratios),unreliable_weak_denominator=weak,
        outside_unit_interval=int(np.sum(abs(ratios)>1)),
        empirical_sd=float(ratios.std(ddof=1)),mean_reported_se=float(ratio_ses.mean()),
        mean=float(ratios.mean()),coverage_among_defined=covered/len(ratios),
        coverage_among_defined_wilson95=wilson(covered,len(ratios)),
        conditioning="positive_estimated_denominators; weak cases retained and counted")
    out["student_t7_coverage_diagnostic"]=(np.mean(abs(estimates-spec["truth"])<=student_t.ppf(.975,7)*ses,axis=0)).tolist()
    if spec["oracle_se"] is not None:
        out["known_gaussian_quadratic_se"]=spec["oracle_se"].tolist()
        out["known_gaussian_se_to_empirical_sd"]=(spec["oracle_se"]/sd).tolist()
        out["known_gaussian_normal_interval_coverage"]=np.mean(
            abs(estimates-spec["truth"])<=1.959963984540054*spec["oracle_se"],axis=0).tolist()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates",type=int,default=500)
    parser.add_argument("--n",type=int,default=128)
    parser.add_argument("--m",type=int,default=64)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.replicates<500:
        raise ValueError("calibration requires at least 500 replicates")
    if args.out.exists():
        raise FileExistsError(args.out)
    cases=[]
    for kind in ("qq","bq","bb"):
        for overlap in (0.,.5,1.):
            cases.append((f"{kind}_overlap{overlap}",dict(kind=kind,overlap=overlap,q=2,k=1,signal=.035,
                risk_strength=.6,prevalence=.2,heterogeneous=False)))
    cases.extend([
        ("bb_null",dict(kind="bb",overlap=1.,q=2,k=1,signal=0.,risk_strength=.6,prevalence=.2,heterogeneous=False)),
        ("bq_rare_heterogeneous",dict(kind="bq",overlap=.5,q=2,k=2,signal=.035,risk_strength=1.,prevalence=.02,heterogeneous=True)),
        ("qq_annotations",dict(kind="qq",overlap=1.,q=2,k=2,signal=.035,risk_strength=0.,prevalence=.2,heterogeneous=False))])
    report={}
    start=time.perf_counter()
    for i,(name,options) in enumerate(cases):
        spec=scenario(91640+i,n=args.n,m=args.m,**options)
        report[name]=dict(configuration=options,**run(spec,args.replicates))
        print(json.dumps(dict(scenario=name,seconds=time.perf_counter()-start,
            maximum_absolute_bias=max(abs(x) for x in report[name]["bias"]),
            coverage_range=[min(report[name]["coverage"]),max(report[name]["coverage"])])),flush=True)
    os.umask(0o077)
    with args.out.open("x") as stream:
        json.dump(dict(elapsed_seconds=time.perf_counter()-start,results=report),stream,indent=2)


if __name__=="__main__":
    main()
