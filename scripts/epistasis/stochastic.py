"""Separate trace-probe, pair-sketch and phenotype variation on a fixed panel."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from summit.epistasis.features import prepare_feature_reference
from summit.epistasis.sketch import independent_bank_moments
from summit.epistasis.score import prepare_linear_scores,linear_score_tests
from summit.epistasis.models import annotation_weights,target_design
from summit.epistasis.prepare import SelectedStudy,fit_scale
from summit.epistasis.summary import fit_epistasis
from summit.epistasis.cli import _jsonable
from summit.prediction.spec import VariantAxis
from summit.prediction.genotype import ArrayGenotypeSource
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.quadratic import quadratic_sf
from scipy.stats import chi2


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(57031);n,m=160,32;started=time.perf_counter()
    raw=rng.binomial(2,.3,(n,m));axis=VariantAxis(tuple(f"v{i}" for i in range(m)),("1",)*m,tuple(range(1,m+1)),("A",)*m,("G",)*m)
    source=ArrayGenotypeSource(raw,[(str(i),str(i)) for i in range(n)],axis,hard_calls=True)
    scale=fit_scale(source,np.arange(n));x=(raw-scale.mean)*scale.inverse_scale
    annotations=annotation_weights(axis.ids,{"a":{f"v{i}":1. for i in range(6)},"b":{f"v{i}":1. for i in range(3,9)}})
    design=target_design(source,np.arange(n),scale,components=[dict(name="epi",target="v0",background="all")],annotations=annotations,additive_annotations=["all"])
    study=SelectedStudy(source,np.arange(n),scale,**design,backend="numpy",block_size=8)
    exact=study.reference(exact=True)
    y=np.sqrt(.3/m)*x@rng.normal(size=(m,32))+np.sqrt(.7)*rng.normal(size=(n,32))
    summary,_=study.summarize(exact,y,trait_names=tuple(map(str,range(32))))
    truth=np.stack([fit_epistasis(summary,i)["coefficients"] for i in range(32)])
    q=summary.rhs;t=exact.matrix
    l=np.linalg.cholesky(t);probe_records=[]
    for probes in (16,64,256):
        coefficients=[];errors=[]
        for seed in range(12):
            reference=study.reference(nvecs=probes,seed=31000+seed)
            fitted=np.linalg.solve(reference.matrix,q).T;coefficients.append(fitted)
            contrast=np.linalg.solve(l,reference.matrix-t);contrast=np.linalg.solve(l,contrast.T).T
            eta=float(np.linalg.norm(contrast,2));errors.append(eta)
            probe_records.append(dict(probes=probes,seed=seed,relative_normal_energy_error=eta,
                inverse_energy_error_bound=eta/(1-eta) if eta<1 else None,
                fixed_phenotype_coefficient=float(fitted[0,1]),exact_coefficient=float(truth[0,1])))
        coefficients=np.stack(coefficients)
        print("probes",probes,"complete",flush=True)
        probe_records.append(dict(probes=probes,seed="aggregate",max_phenotype_bias=float(np.max(abs(coefficients.mean(axis=0)[:,1]-truth[:,1]))),
            probe_sd_fixed_y=float(coefficients[:,0,1].std(ddof=1)),phenotype_sd_fixed_probes=float(coefficients[0,:,1].std(ddof=1)),
            phenotype_sd_exact=float(truth[:,1].std(ddof=1)),max_relative_normal_energy_error=max(errors)))
    pd.DataFrame(probe_records).to_csv(args.out/"trace_probes.csv",index=False)
    design=target_design(source,np.arange(n),scale,components=[],annotations=annotations,additive_annotations=["all","a","b"],allow_additive_only=True)
    study=SelectedStudy(source,np.arange(n),scale,**design,backend="numpy",block_size=8)
    job=dict(id="overlap",additive_annotations=["all","a","b"],groups=[dict(name="overlap",mode="cross",left="a",right="b")])
    exact=prepare_feature_reference(study,job,annotations,main_effects="all_genotypes")
    u=thin_rank_revealing_fixed_effect_basis(exact.fixed_effects);pf=exact.features-u@(u.T@exact.features)
    k=pf@pf.T;ty=float(np.sum(k*k));phenotypes=rng.normal(size=(n,100))
    signal=pf@rng.normal(size=(pf.shape[1],100));signal*=np.sqrt(.04*(n-u.shape[1])/np.sum(pf*pf))
    phenotypes=np.column_stack([phenotypes,phenotypes+signal])
    benchmark=prepare_linear_scores(exact.features,phenotypes,exact.fixed_effects,feature_names=exact.metadata["feature_names"],
        trait_names=tuple(map(str,range(200))),metadata={})
    exact_p=np.array([linear_score_tests(benchmark,trait=i)["kernel_p"] for i in range(200)])
    rows=[];moments=[]
    for r in (16,64,256):
        banks=[]
        for bank in range(12):
            ref=prepare_feature_reference(study,job,annotations,sketch_dimensions=r,seed=48,bank=bank,main_effects="all_genotypes")
            f=ref.features-u@(u.T@ref.features);kh=f@f.T
            test=prepare_linear_scores(ref.features,phenotypes,ref.fixed_effects,feature_names=ref.metadata["feature_names"],
                trait_names=benchmark.trait_names,metadata={})
            pp=np.array([linear_score_tests(test,trait=i)["kernel_p"] for i in range(200)])
            rows.append(dict(dimensions=r,bank=bank,kernel_relative_error=float(np.linalg.norm(kh-k)/np.linalg.norm(k)),
                naive_squared_trace_bias=float(np.sum(kh*kh)-ty),fixed_y_p=float(pp[0]),exact_fixed_y_p=float(exact_p[0]),
                null_rejection=float(np.mean(pp[:100]<=.05)),power=float(np.mean(pp[100:]<=.05)),
                exact_null_rejection=float(np.mean(exact_p[:100]<=.05)),exact_power=float(np.mean(exact_p[100:]<=.05))))
            banks.append(ref)
            if len(banks)==3:
                mm=independent_bank_moments(banks,phenotypes[:,0]);banks=[]
                moments.append(dict(dimensions=r,bank_triple=bank//3,squared_trace_error=float(mm["matrix"][0,0]-ty),
                    cubic_error=float(mm["cubic"][0,0,0,0]-phenotypes[:,0]@k@k@k@phenotypes[:,0])))
        print("pair sketches",r,"complete",flush=True)
    pd.DataFrame(rows).to_csv(args.out/"pair_sketches.csv",index=False)
    pd.DataFrame(moments).to_csv(args.out/"independent_moments.csv",index=False)
    tails=[]
    for alpha in (.05,.001,5e-8):
        q=chi2.isf(alpha,12)
        # Unequal signed weights: convolution integral independent check in tests;
        # this nearly equal spectrum checks inversion at extreme positive tails.
        lam=np.ones(12);lam[-1]+=1e-8
        result=quadratic_sf(q,lam,atol=1e-11)
        tails.append(dict(alpha=alpha,result=result,scaled_chi_square_reference=float(chi2.sf(q,12))))
    (args.out/"design.json").write_text(json.dumps(_jsonable(dict(seed=57031,n=n,m=m,seconds=time.perf_counter()-started,
        tails=tails,contract="Fixed genotype panel and phenotypes across banks; same probes across traits. Different trace probes and pair banks are separate experiments. No SE correction inferred. Independent-bank products are unbiased before inverse fitting; their finite averages are noisy.")),indent=2)+"\n")


if __name__=="__main__":main()
