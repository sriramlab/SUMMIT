"""Reproducible bounded binary GxE calibration and streamed scaling benchmark.

Dense *variant-axis* equations are used only in the small simulation oracle.
Production numerical/scaling checks call the unchanged streamed entry points.
No participant data or genotype pools are saved.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from summit.pcgc.gxe import (
    GxEMoments, context_pairs, context_pair_gram, population_metric,
    prepare_gxe_moments, prepare_gxe_external, fit_gxe, EXTERNAL_CONTRACT,
)
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from summit.prediction.genotype import native_module

SCENARIOS = {
    "null": dict(q=2,K=.1,signal="null",sd=False),
    "additive_only": dict(q=2,K=.1,signal="additive",sd=False),
    "binary_exposure": dict(q=2,K=.1,signal="joint",sd=False),
    "continuous_exposure": dict(q=2,K=.1,signal="joint",sd=False,continuous=True),
    "multiple_exposures": dict(q=3,K=.1,signal="joint",sd=False),
    "heterogeneous_liability": dict(q=2,K=.01,signal="joint",sd=True,continuous=True),
    "ld_partitioned": dict(q=2,K=.1,signal="joint",sd=False,ld=True),
}


def exact_variant_moments(x,a,risk,phi,sd,method):
    n,m=x.shape;q=phi.shape[1];ps=context_pairs(q);p=len(ps);k=a.shape[1]
    w=risk.sensitivity/sd
    psi=phi if method=="pcgc-inverse" else phi*w[:,None]
    y=risk.z/w if method=="pcgc-inverse" else risk.z
    F=[psi[:,u,None]*x for u in range(q)]
    cross={(u,v):F[u].T@F[v] for u in range(q) for v in range(u,q)}
    cross.update({(v,u):value.T for (u,v),value in list(cross.items()) if u!=v})
    ld=np.empty((m,p,k*p))
    squares=x*x; diagonal_base=squares@a
    rhs=np.empty((m,p))
    scores=x.T@(psi*y[:,None])
    for t,(u,v) in enumerate(ps):
        orientations=[(u,v)] if u==v else [(u,v),(v,u)]
        h=(1 if u==v else 2)*psi[:,u]*psi[:,v]
        rhs[:,t]=(1 if u==v else 2)*scores[:,u]*scores[:,v]-squares.T@(h*y*y)
        for s,(c,d) in enumerate(ps):
            source=[(c,d)] if c==d else [(c,d),(d,c)]
            product=sum(cross[j,l]*cross[i,h_] for i,j in orientations for l,h_ in source)
            full=product@a
            other=(1 if c==d else 2)*psi[:,c]*psi[:,d]
            full-=squares.T@((h*other)[:,None]*diagonal_base)
            ld[:,t,s::p]=full/n**2
    metric,var=population_metric(phi,risk,sd)
    return GxEMoments(a,ld,rhs,q,n,method,metric,var)


def generate(seed,scenario,n,m,*,return_population=False,case_fraction=.5):
    rng=np.random.default_rng(seed); spec=SCENARIOS[scenario];q=spec["q"]
    if spec.get("continuous"):
        e=np.linspace(-1,1,21)
        states=np.column_stack((np.ones(len(e)),e))
    elif q==2:
        states=np.array([[1.,-1.],[1.,1.]])
    else:
        states=np.array([[1.,a,b] for a in (-1.,1.) for b in (-1.,1.)])
    weights=np.full(len(states),1/len(states))
    omega=np.diag([.15]+[.035]*(q-1))
    if spec["signal"]=="null":omega[:]=0
    elif spec["signal"]=="additive":omega[1:,1:]=0
    else:
        omega[0,1]=omega[1,0]=.018
        if q==3:omega[0,2]=omega[2,0]=-.012;omega[1,2]=omega[2,1]=.009
    a=np.ones((m,1))
    if spec.get("ld"):
        a=np.column_stack((np.ones(m),(np.arange(m)<m//2).astype(float)))
        omega=np.stack((.6*omega,.4*omega))
    components=omega[None] if omega.ndim==2 else omega
    beta=np.zeros((m,q))
    for annotation,matrix in zip(a.T,components):
        eig,vec=np.linalg.eigh(matrix)
        beta+=rng.normal(size=(m,q))@(vec*np.sqrt(np.maximum(eig,0))).T*np.sqrt(annotation/annotation.sum())[:,None]
    # Gaussian markers allow an exact retrospective conditional sampler.
    # Include block LD and overlapping annotations in the dedicated scenario.
    R=population_correlation(spec,m)
    effects=states@beta.T
    rb=effects@R
    genetic=np.sum(effects*rb,axis=1)
    scale=np.sqrt(1+.35*states[:,1]+.25*states[:,1]**2) if spec["sd"] else np.ones(len(states))
    if np.any(genetic>=scale**2):raise RuntimeError("simulation draw has nonpositive environmental variance")
    mu=.45*states[:,1]+(-.2*states[:,2] if q==3 else 0)
    K=spec["K"]
    cut=brentq(lambda t:weights@ndtr((mu-t)/scale)-K,-12,12)
    k=ndtr((mu-cut)/scale)
    cases = int(n*case_fraction)
    if not 0<cases<n: raise ValueError("simulation needs cases and controls")
    y=np.r_[np.ones(cases),np.zeros(n-cases)];rng.shuffle(y)
    index=np.empty(n,dtype=int)
    for case in (0,1):
        selected=y==case
        probabilities=weights*(k/K if case else (1-k)/(1-K))
        index[selected]=rng.choice(len(states),size=selected.sum(),p=probabilities/probabilities.sum())
    phi=states[index];sd=scale[index];ki=k[index]
    t=(cut-mu[index])/sd
    u=rng.uniform(np.finfo(float).eps,1-np.finfo(float).eps,n)
    liab=sd*np.where(y==1,-ndtri(u*ndtr(-t)),ndtri(u*ndtr(t)))
    x=rng.normal(size=(n,m))
    if spec.get("ld"):
        for start in range(0,m,8):
            stop=min(m,start+8)
            x[:,start:stop]=x[:,start:stop]@np.linalg.cholesky(R[start:stop,start:stop]).T
    b=effects[index]
    env=rng.normal(size=n)*np.sqrt(sd**2-genetic[index])
    x+=(liab-np.sum(x*b,axis=1)-env)[:,None]*rb[index]/sd[:,None]**2
    risk=prepare_binary_risk(y,K,population_risk=ki)
    result=(x,a,risk,phi,sd,omega)
    if return_population:
        return result,dict(states=states,weights=weights,beta=beta,R=R,scale=scale,mu=mu,cut=cut)
    return result


def population_correlation(spec,m):
    if not spec.get("ld"): return np.eye(m)
    index=np.arange(m)
    return .4**np.abs(index[:,None]-index[None,:])*(index[:,None]//8==index[None,:]//8)


def simulation(args):
    rows=[];start=time.perf_counter()
    for scenario in args.scenarios:
        for rep in range(args.replicates):
            seed=args.seed+100000*list(SCENARIOS).index(scenario)+rep
            x,a,risk,phi,sd,omega=generate(seed,scenario,args.samples,args.variants)
            components=omega[None] if omega.ndim==2 else omega
            truth=np.array([matrix[u,v] for matrix in components for u,v in context_pairs(phi.shape[1])])
            pop_ld=population_correlation(SCENARIOS[scenario],args.variants)**2@a
            blocks=np.arange(args.variants)*args.blocks//args.variants
            risk_models={"supplied":risk}
            if scenario in ("null","additive_only","binary_exposure"):
                risk_models["fitted"]=fit_binary_risk((risk.z>0).astype(float),risk.population_prevalence,phi[:,1:])
            for risk_name,selected_risk in risk_models.items():
                standard=exact_variant_moments(x,a,selected_risk,phi,sd,"pcgc")
                inverse=exact_variant_moments(x,a,selected_risk,phi,sd,"pcgc-inverse")
                # Analytic infinite population LD isolates factorization error
                # from finite-reference/probe noise.
                factor=context_pair_gram(phi*(selected_risk.sensitivity/sd)[:,None])
                external=replace(standard,method="pcgc-ld",
                    ldscores=(pop_ld[:,None,:,None]*factor[None,:,None,:]/len(x)**2).reshape(standard.ldscores.shape))
                for method,moments in (("pcgc",standard),("pcgc-basis",replace(standard,method="pcgc-basis")),
                                        ("pcgc-inverse",inverse),("pcgc-ld",external)):
                    row=dict(scenario=scenario,replicate=rep,seed=seed,risk=risk_name,method=method,truth=truth.tolist())
                    try:
                        fit=fit_gxe(moments,block_ids=blocks)
                        row.update(estimate=fit["components"],se=fit["standard_errors"],
                                   condition=fit["normal_condition"])
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        row["failure"]=str(exc)
                    rows.append(row)
    summary=[]
    for key in sorted({(r["scenario"],r["risk"],r["method"]) for r in rows}):
        group=[r for r in rows if (r["scenario"],r["risk"],r["method"])==key]
        good=[r for r in group if "failure" not in r]
        item=dict(zip(("scenario","risk","method"),key),replicates=len(group),failures=len(group)-len(good))
        if len(good)>=2:
            est=np.array([r["estimate"] for r in good]);truth=np.array([r["truth"] for r in good]);se=np.array([r["se"] for r in good])
            err=est-truth
            item.update(mean=est.mean(axis=0).tolist(),bias=err.mean(axis=0).tolist(),
                bias_mcse=(err.std(axis=0,ddof=1)/np.sqrt(len(good))).tolist(),
                empirical_sd=est.std(axis=0,ddof=1).tolist(),mean_se=se.mean(axis=0).tolist(),
                coverage_95=(np.abs(err)<=1.96*se).mean(axis=0).tolist(),
                rejection_at_zero=(np.abs(est)>1.96*se).mean(axis=0).tolist())
        summary.append(item)
    return dict(design=vars(args),rows=rows,summary=summary,seconds=time.perf_counter()-start)


def numerical(args):
    """All production paths, paired fixed seeds, independent exact oracle."""
    output=[]
    rng=np.random.default_rng(args.seed)
    for n,m,q in ((128,96,2),(256,128,3)):
        x=rng.normal(size=(n,m));phi=np.column_stack((np.ones(n),rng.uniform(-1,1,(n,q-1))))
        a=rng.uniform(.1,1,(m,2));sd=1+.2*phi[:,1]**2
        risk=prepare_binary_risk(np.arange(n)%2,.1,population_risk=ndtr(-1.5+.3*phi[:,1]))
        for method in ("pcgc","pcgc-basis","pcgc-inverse"):
            exact=exact_variant_moments(x,a,risk,phi,sd,method)
            extra=dict(basis=risk.sensitivity[:,None],coefficients=[1]) if method=="pcgc-basis" else {}
            for probes in args.probes:
                start=time.perf_counter()
                options=dict(probes=probes,seed=args.seed,block_size=37,liability_sd=sd,**extra)
                got,diagnostics=prepare_gxe_moments(Operator(x),a,risk,phi,method,native=True,threads=args.threads,**options)
                numpy,_=prepare_gxe_moments(Operator(x),a,risk,phi,method,native=False,**options)
                H,b=got.equations();H0,b0=exact.equations()
                assert np.max(np.abs(got.ldscores-numpy.ldscores))<1e-9
                assert np.max(np.abs(b-b0))<1e-7
                output.append(dict(samples=n,variants=m,contexts=q,annotations=2,method=method,probes=probes,
                    native_numpy_max_error=float(np.max(np.abs(got.ldscores-numpy.ldscores))),
                    rhs_exact_max_error=float(np.max(np.abs(b-b0))),
                    relative_gram_probe_error=float(np.linalg.norm(H-H0)/np.linalg.norm(H0)),
                    genotype_passes=diagnostics["genotype_passes"],seconds=time.perf_counter()-start,
                    planned_bytes=diagnostics["peak_planned_workspace_bytes"]))
    return dict(design=vars(args),rows=output,native_build=native_module().build_info())


def scaling(args):
    rng=np.random.default_rng(args.seed);output=[]
    for n,m,q in ((1000,1000,2),(2000,2000,2),(2000,2000,3)):
        x=rng.normal(size=(n,m));phi=np.column_stack((np.ones(n),rng.uniform(-1,1,(n,q-1))))
        risk=prepare_binary_risk(np.arange(n)%2,.05,population_risk=ndtr(-1.7+.3*phi[:,1]))
        a=np.ones((m,1))
        for method in ("pcgc","pcgc-inverse","pcgc-basis","pcgc-ld"):
            start=time.perf_counter()
            if method=="pcgc-ld":
                ref=Operator(rng.normal(size=(n,m)))
                moments,d=prepare_gxe_external(Operator(x),ref,a,risk,phi,liability_sd=1.,
                    factorization_contract=EXTERNAL_CONTRACT,probes=256,seed=args.seed,block_size=256,
                    threads=args.threads,memory_bytes=2*2**30)
            else:
                extra=dict(basis=risk.sensitivity[:,None],coefficients=[1]) if method=="pcgc-basis" else {}
                moments,d=prepare_gxe_moments(Operator(x),a,risk,phi,method,liability_sd=1.,
                    probes=256,seed=args.seed,block_size=256,threads=args.threads,memory_bytes=2*2**30,**extra)
            fitted=fit_gxe(moments,block_ids=np.arange(m)*20//m)
            output.append(dict(samples=n,variants=m,contexts=q,method=method,seconds=time.perf_counter()-start,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                planned_workspace_bytes=d["peak_planned_workspace_bytes"],
                moment_bytes=sum(getattr(moments,s).nbytes for s in ("annotations","ldscores","rhs_rows")),
                diagnostics=d,condition=fitted["normal_condition"]))
            print(json.dumps({k:v for k,v in output[-1].items() if k!="diagnostics"}),flush=True)
    return dict(design=vars(args),rows=output,native_build=native_module().build_info())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("simulation","numerical","scaling"));p.add_argument("--out",required=True)
    p.add_argument("--replicates",type=int,default=100);p.add_argument("--samples",type=int,default=1200)
    p.add_argument("--variants",type=int,default=256);p.add_argument("--blocks",type=int,default=16)
    p.add_argument("--seed",type=int,default=72914);p.add_argument("--threads",type=int,default=1)
    p.add_argument("--probes",nargs="+",type=int,default=[64,256,1024])
    p.add_argument("--scenarios",nargs="+",choices=tuple(SCENARIOS),default=list(SCENARIOS))
    args=p.parse_args()
    path=Path(args.out)
    if path.exists():raise FileExistsError(path)
    if args.samples<8 or args.variants<4 or not 2<=args.blocks<=args.variants or args.replicates<1:
        raise ValueError("invalid bounded simulation dimensions")
    result=globals()[args.mode](args)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(path),rows=len(result["rows"]),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)

if __name__=="__main__":
    main()
