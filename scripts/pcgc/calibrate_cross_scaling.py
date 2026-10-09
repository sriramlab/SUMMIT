"""Calibration oracle varying N, M and target-block count independently.

This research driver forms bounded M by M variant Grams, never N by N kernels.
Those exact Grams are an oracle, not a production reference implementation.
Simulation responses and SNP scores are batched; no participant arrays are saved.
"""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import t as student_t
from summit.pcgc.cross_gxe import Products, pair_layout, project
from summit.pcgc.cross_gxe_model import binary_trait, quantitative_trait, DiagonalRemoval
from summit.ldscore.generalized_gxe_pass1 import _peak_rss_bytes
from scripts.pcgc.calibrate_cross_gxe import wilson


def make_design(n, m, *, kind, overlap, annotations=1, seed=729103,
                prevalence=.2, risk_strength=.6, heterogeneous=False,
                genotype="gaussian", signal=.035, native=False, threads=1, annotation_mixing=0., genetic_scale=1.):
    rng=np.random.default_rng(seed)
    shared=int(n*overlap);total=2*n-shared
    rows=[np.arange(n),np.r_[np.arange(shared),np.arange(n,total)]]
    if genotype=="gaussian":
        x=rng.normal(size=(total,m))
    else:
        # Four-SNP haplotype blocks with common MAF and copy-induced LD.
        blocks=(m+3)//4
        p=rng.uniform(.05,.5,blocks)
        shared_allele=rng.uniform(size=(total,blocks,2,1))<p[None,:,None,None]
        own=rng.uniform(size=(total,blocks,2,4))<p[None,:,None,None]
        copy=rng.uniform(size=(total,blocks,2,4))<.65
        raw=np.where(copy,shared_allele,own).sum(2).reshape(total,-1)[:,:m]
        p=np.repeat(p,4)[:m]
        x=(raw-2*p)/np.sqrt(2*p*(1-p))
    if not 0<=annotation_mixing<1:
        raise ValueError("annotation mixing must lie in [0,1)")
    ann=(1-annotation_mixing)*np.eye(annotations)[np.arange(m)%annotations]+annotation_mixing/annotations
    phi=[np.column_stack((np.ones(n),rng.uniform(-1,1,n))) for _ in range(2)]
    sd=[np.sqrt(1+.35*f[:,1]) if heterogeneous else np.ones(n) for f in phi]
    risk=[ndtr(ndtri(prevalence)+risk_strength*f[:,1]) for f in phi]
    aa=np.diag([.12,.035]);bb=np.diag([.1,.03])
    ab=np.array([[signal,.01 if signal else 0.],[-.004 if signal else 0.,signal/3]])
    if not np.isfinite(genetic_scale) or genetic_scale<=0:
        raise ValueError("genetic scale must be positive")
    aa*=genetic_scale;bb*=genetic_scale;ab*=genetic_scale
    omega=np.block([[aa,ab],[ab.T,bb]])
    root=np.linalg.cholesky(omega)
    traits=[]
    for t in range(2):
        ids=tuple(("s",str(i)) for i in rows[t])
        common=dict(name=str(t),samples=ids,contexts=phi[t],context_names=("i","E"),
                    selection="outcome_independent")
        if kind[t]=="b":
            traits.append(binary_trait(y=np.arange(n)%2,population_risk=risk[t],liability_sd=sd[t],**common))
        else:
            fixed=np.ones((n,1)) if t==0 else np.column_stack((np.ones(n),
                np.cos(np.arange(n)*2*np.pi/n),np.sin(np.arange(n)*2*np.pi/n)))
            traits.append(quantitative_trait(y=np.zeros(n),population_sd=1.,fixed_effects=fixed,**common))
    products=Products(native,threads)
    bases=[np.asfortranarray(x[r]) for r in rows]
    features=[[project(base*(t.sensitivity*t.contexts[:,q])[:,None],t.fixed_basis,products)
               for q in range(2)] for base,t in zip(bases,traits)]
    variance=[]
    for t,mat in enumerate((aa,bb)):
        genetic=np.sum(bases[t]**2,axis=1)/m*np.einsum("iq,qr,ir->i",phi[t],mat,phi[t])
        variance.append(sd[t]**2-genetic)
    if min(v.min() for v in variance)<=0:
        raise ValueError("generating residual variance is nonpositive")
    del x
    return dict(n=n,m=m,rows=rows,ann=ann,traits=traits,phi=phi,sd=sd,risk=risk,
                omega=root,truth_matrices=(aa,ab,bb),residual=variance,bases=bases,
                features=features,products=products,kind=kind,seed=seed,shared=shared)


def exact_equations(design, block_counts, *, retain_grams=False):
    """Variant-Gram oracle checked against explicit person kernels in tests."""
    f=design["features"];traits=design["traits"];a=design["ann"];m=len(a);k=a.shape[1]
    products=design["products"];mass=a.sum(0)
    gram=[]
    for features in f:
        g={}
        for q in range(2):
            for u in range(q,2):
                g[q,u]=products.tn(features[q],features[u])
                g[u,q]=g[q,u].T
        gram.append(g)
        products.drain()
    if retain_grams: design["grams"]=gram
    models=[];truth=[]
    for (left,right),matrix in zip(((0,0),(0,1),(1,1)),design["truth_matrices"]):
        layout=pair_layout(2,2,left==right);p=len(layout);c=k*p
        _,ia,ib=np.intersect1d(design["rows"][left],design["rows"][right],return_indices=True)
        remover=DiagonalRemoval(traits[left].fixed_basis[ia],traits[right].fixed_basis[ib],products)
        diagonal=[]
        for orientation in layout:
            diagonal.append(sum(f[left][q][ia]*f[right][r][ib] for q,r in orientation))
        ds=np.concatenate([d@(a/mass) for d in diagonal],axis=1).reshape(len(ia),p,k).transpose(0,2,1).reshape(len(ia),c)
        corrected=remover.apply(ds)
        ld=np.empty((m,p,c))
        for target,orient in enumerate(layout):
            for source,orient2 in enumerate(layout):
                cross=sum(gram[left][q,u]*gram[right][r,v] for q,r in orient for u,v in orient2)
                ld[:,target,source::p]=cross@(a/mass)
            ld[:,target]-=diagonal[target].T@corrected
        models_by_blocks={}
        for count in block_counts:
            block_ids=np.arange(m)*count//m
            boundaries=np.r_[0,np.flatnonzero(np.diff(block_ids))+1,m]
            hs=np.stack([a[first:last].T@ld[first:last].reshape(last-first,-1)
                         for first,last in zip(boundaries[:-1],boundaries[1:])]).reshape(count,c,c)
            h=hs.sum(0)
            if np.linalg.cond(h)>1e8:
                raise ValueError("weak oracle normal equations")
            db=np.empty((len(ia),count,c))
            for annotation in range(k):
                for target in range(p):
                    db[:,:,annotation*p+target]=np.add.reduceat(diagonal[target]*a[:,annotation],boundaries[:-1],axis=1)
            models_by_blocks[count]=dict(boundaries=boundaries,inverse=np.linalg.inv(h),
                deleted_inverse=np.linalg.inv(h[None]-hs),diagonal=db,H=h,H_blocks=hs,
                correction_blocks=(db.reshape(len(ia),count*c).T@corrected).reshape(count,c,c))
        models.append(dict(left=left,right=right,layout=layout,overlap=(ia,ib),remover=remover,blocks=models_by_blocks))
        truth.extend(np.outer(mass/m,[matrix[q,r] for orientation in layout for q,r in orientation[:1]]).ravel())
        products.drain()
    return models,np.asarray(truth)


def responses(design, rng, batch):
    n,m=design["n"],design["m"]
    products=design["products"]
    beta=(rng.normal(size=(m*batch,4))@design["omega"].T/np.sqrt(m)).reshape(m,batch,4)
    residual=[rng.normal(size=(n,batch)),rng.normal(size=(n,batch))]
    shared=design["shared"]
    residual[1][:shared]=.2*residual[0][:shared]+np.sqrt(.96)*residual[1][:shared]
    z=[];eligible=np.ones(batch,dtype=bool);case_counts=[]
    for t in range(2):
        effect=beta[:,:,2*t:2*t+2].transpose(0,2,1).reshape(m,2*batch)
        genetic=products.nn(design["bases"][t],effect).reshape(n,2,batch)
        latent=np.sum(genetic*design["phi"][t][:,:,None],axis=1)+residual[t]*np.sqrt(design["residual"][t])[:,None]
        if design["kind"][t]=="b":
            risk=design["risk"][t]
            y=latent+design["sd"][t][:,None]*ndtri(risk)[:,None]>0
            case_counts.append(y.sum(0))
            eligible &= (y.sum(0)>0)&(y.sum(0)<n)
            latent=(y-risk[:,None])/np.sqrt(risk*(1-risk))[:,None]
        if design["kind"][t]!="b": case_counts.append(None)
        z.append(project(latent,design["traits"][t].fixed_basis,products))
    design["last_case_counts"]=case_counts
    return z,eligible


def fit_responses(design,models,z):
    products=design["products"];a=design["ann"];k=a.shape[1]
    score=[[products.tn(f,y) for f in ff] for ff,y in zip(design["features"],z)]
    full={};deleted={}
    for model in models:
        l,r=model["left"],model["right"];layout=model["layout"];p=len(layout)
        ia,ib=model["overlap"]
        correction=model["remover"].apply(z[l][ia]*z[r][ib])
        values=np.stack([sum(score[l][q]*score[r][s] for q,s in orientation) for orientation in layout],axis=1)
        for count,work in model["blocks"].items():
            br=np.empty((count,k*p,z[0].shape[1]))
            for annotation in range(k):
                br[:,annotation*p:(annotation+1)*p]=np.add.reduceat(values*a[:,annotation,None,None],work["boundaries"][:-1],axis=0)
            if len(ia):
                br-=products.tn(work["diagonal"].reshape(len(ia),count*k*p),correction).reshape(br.shape)
            rhs=br.sum(0)
            full.setdefault(count,[]).append((work["inverse"]@rhs).T)
            deleted.setdefault(count,[]).append(np.einsum("bij,bjr->bri",work["deleted_inverse"],rhs[None]-br))
        products.drain()
    return {j:(np.concatenate(full[j],axis=1),np.concatenate(deleted[j],axis=2)) for j in full}


def report(estimates,variances,rg,truth,blocks,eligible,mean_covariance):
    n=len(estimates);sd=estimates.std(0,ddof=1);bias=estimates.mean(0)-truth
    se=np.sqrt(variances);z=1.959963984540054;t=student_t.ppf(.975,blocks-1)
    discrepancy=(estimates-truth)**2-variances
    out=dict(replicates=n,truth=truth.tolist(),bias=bias.tolist(),bias_mcse=(sd/np.sqrt(n)).tolist(),
        empirical_covariance=np.cov(estimates,rowvar=False).tolist(),
        mean_reported_covariance=mean_covariance.tolist(),
        mse_minus_mean_variance=discrepancy.mean(0).tolist(),
        mse_minus_mean_variance_mcse=(discrepancy.std(0,ddof=1)/np.sqrt(n)).tolist(),
        empirical_sd=sd.tolist(),mean_se=se.mean(0).tolist(),se_to_sd=(se.mean(0)/sd).tolist(),
        mean_variance_to_empirical_variance=(variances.mean(0)/sd**2).tolist(),
        rejected_constant_binary_outcomes=int((~eligible).sum()),blocks=blocks,
        gaussian_student_reference_normal_coverage=float(2*student_t.cdf(z,blocks-1)-1))
    for label,critical in (("normal",z),("student",t)):
        hit=(abs(estimates-truth)<=critical*se).sum(0)
        reject=(abs(estimates)>critical*se).sum(0)
        out[label]=dict(coverage=(hit/n).tolist(),coverage_wilson95=[wilson(h,n) for h in hit],
            null_type1=[float(reject[i]/n) if v==0 else None for i,v in enumerate(truth)],
            null_type1_wilson95=[wilson(reject[i],n) if v==0 else None for i,v in enumerate(truth)])
    valid=np.isfinite(rg[:,0]);r=rg[valid];k=len(truth)//10;target=truth[3*k:7*k:4].sum()/np.sqrt(truth[:3*k:3].sum()*truth[7*k::3].sum())
    out["rg"]=dict(defined=int(valid.sum()),undefined=int((~valid).sum()),
        weak=int(r[:,2].sum()),outside_unit_interval=int((abs(r[:,0])>1).sum()),truth=float(target),
        normal_coverage_among_defined=float(np.mean(abs(r[:,0]-target)<=z*r[:,1])),
        student_coverage_among_defined=float(np.mean(abs(r[:,0]-target)<=t*r[:,1])))
    for label,critical in (("normal",z),("student",t)):
        hit=int(np.sum(abs(r[:,0]-target)<=critical*r[:,1]))
        out["rg"][label+"_coverage_wilson95"]=wilson(hit,len(r)) if len(r) else None
    return out


def run(design,models,truth,*,replicates,batch):
    counts=list(models[0]["blocks"]);estimates={j:[] for j in counts};variances={j:[] for j in counts};ratios={j:[] for j in counts}
    covariance_sums={j:np.zeros((len(truth),len(truth))) for j in counts}
    case_counts=[[],[]]
    eligible=[];rng=np.random.default_rng(design["seed"]+8392)
    k=design["ann"].shape[1];contrast=np.zeros((3,len(truth)))
    contrast[0,np.arange(k)*3]=1;contrast[1,7*k+np.arange(k)*3]=1;contrast[2,3*k+np.arange(k)*4]=1
    for first in range(0,replicates,batch):
        z,valid=responses(design,rng,min(batch,replicates-first));eligible.append(valid)
        for t,count in enumerate(design["last_case_counts"]):
            if count is not None: case_counts[t].extend(count.tolist())
        fitted=fit_responses(design,models,z)
        baseline=fitted[counts[0]][0]
        for count,(point,loo) in fitted.items():
            np.testing.assert_allclose(point,baseline,rtol=2e-10,atol=2e-10)
            centered=loo-loo.mean(0)
            cov=(count-1)/count*np.einsum("jrc,jrd->rcd",centered,centered)
            covariance_sums[count]+=cov.sum(0)
            estimates[count].append(point);variances[count].append(np.diagonal(cov,axis1=1,axis2=2))
            v=point@contrast.T;vcov=np.einsum("ac,rcd,bd->rab",contrast,cov,contrast)
            good=(v[:,0]>0)&(v[:,1]>0)
            ratio=np.full((len(v),3),np.nan)
            vg=v[good];r=vg[:,2]/np.sqrt(vg[:,0]*vg[:,1])
            gradient=np.column_stack((-r/(2*vg[:,0]),-r/(2*vg[:,1]),1/np.sqrt(vg[:,0]*vg[:,1])))
            ratio[good,0]=r;ratio[good,1]=np.sqrt(np.maximum(0,np.einsum("ri,rij,rj->r",gradient,vcov[good],gradient)))
            ratio[good,2]=np.any(vg[:,:2]-1.96*np.sqrt(np.diagonal(vcov[good],axis1=1,axis2=2)[:,:2])<=0,axis=1)
            ratios[count].append(ratio)
    results={str(j):report(np.concatenate(estimates[j]),np.concatenate(variances[j]),np.concatenate(ratios[j]),truth,j,np.concatenate(eligible),covariance_sums[j]/replicates) for j in counts}
    for value in results.values():
        value["binary_case_count_quantiles"]=[None if not count else np.quantile(count,[0,.025,.5,.975,1]).tolist() for count in case_counts]
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",type=int,default=512);parser.add_argument("--m",type=int,default=512)
    parser.add_argument("--replicates",type=int,default=1000);parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--blocks",type=int,nargs="+",default=[8,32,128]);parser.add_argument("--kind",choices=["qq","bq","bb"],default="qq")
    parser.add_argument("--overlap",type=float,default=1.);parser.add_argument("--annotations",type=int,default=1)
    parser.add_argument("--prevalence",type=float,default=.2);parser.add_argument("--risk-strength",type=float,default=.6)
    parser.add_argument("--heterogeneous",action="store_true");parser.add_argument("--genotype",choices=["gaussian","hardcall_ld"],default="gaussian")
    parser.add_argument("--genetic-scale",type=float,default=1.)
    parser.add_argument("--annotation-mixing",type=float,default=0.)
    parser.add_argument("--signal",type=float,default=.035);parser.add_argument("--seed",type=int,default=729103)
    parser.add_argument("--threads",type=int,default=2);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    if min(args.n,args.m,args.batch,args.annotations)<1 or args.annotations>args.m or len(set(args.blocks))!=len(args.blocks) or not 0<=args.overlap<=1 or args.replicates<500 or args.m>8192 or min(args.blocks)<2 or max(args.blocks)>args.m:
        raise ValueError("invalid bounded calibration dimensions")
    start=perf_counter();cpu_start=resource.getrusage(resource.RUSAGE_SELF)
    design=make_design(args.n,args.m,kind=args.kind,overlap=args.overlap,annotations=args.annotations,
        seed=args.seed,prevalence=args.prevalence,risk_strength=args.risk_strength,heterogeneous=args.heterogeneous,
        genotype=args.genotype,signal=args.signal,native=True,threads=args.threads,annotation_mixing=args.annotation_mixing,genetic_scale=args.genetic_scale)
    prepared=perf_counter();models,truth=exact_equations(design,args.blocks);reference=perf_counter()
    results=run(design,models,truth,replicates=args.replicates,batch=args.batch)
    cpu_end=resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds=cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime
    out=dict(cpu_seconds=cpu_seconds,average_active_cores=cpu_seconds/(perf_counter()-start),configuration={**vars(args),"out":str(args.out)},results=results,
        design_seconds=prepared-start,exact_reference_seconds=reference-prepared,simulation_seconds=perf_counter()-reference,
        peak_rss_bytes=_peak_rss_bytes(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference="exact_bounded_variant_Gram_oracle",fitted_nuisances=False,reference_randomization=False,
        normal_conditions=[float(np.linalg.cond(model["blocks"][args.blocks[0]]["H"])) for model in models],
        gemm_repaired_columns=design["products"].left.repaired_columns+design["products"].right.repaired_columns)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    old=os.umask(0o077)
    try:
        with args.out.open("x") as f: json.dump(out,f,indent=2)
    finally: os.umask(old)
    print(json.dumps({j:{"coverage_normal":[min(r["normal"]["coverage"]),max(r["normal"]["coverage"])],
                       "coverage_student":[min(r["student"]["coverage"]),max(r["student"]["coverage"])]} for j,r in results.items()}),flush=True)


if __name__=="__main__":
    main()
