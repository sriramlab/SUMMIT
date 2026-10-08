"""Frozen fixed-mean Gaussian check of overlapping annotations and score modifiers.

This validates global-null score inference, not conditional variance-component
attribution. Genotypes and realized effects are fixed; only residuals change.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scripts.epistasis.confirm_inference import panel
from scripts.epistasis.validate import interval
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.score import prepare_linear_scores, linear_score_tests


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--real-genotypes',required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    records=[];design=[];started=time.perf_counter()
    for seed in (3618201,3718291):
        x,h,meta=panel(args.real_genotypes,seed,n=384,m=96)
        n,m=x.shape;rng=np.random.default_rng(seed+910000)
        fixed=np.column_stack([np.ones(n),x,h[:,:16]])
        u=thin_rank_revealing_fixed_effect_basis(fixed);rank=n-u.shape[1]
        mean=x@rng.normal(size=m)/np.sqrt(m)+h[:,:16]@rng.normal(size=16)/4
        # Weights, loci and signals fixed before looking at phenotype outcomes.
        left=x[:,0,None]*x[:,8:12]*np.sqrt(np.array([1,2,1,.5])/4.5)
        right=x[:,0,None]*x[:,10:16]*np.sqrt(np.array([.5,1,2,1,1,.5])/6)
        modifier=.7*x[:,0]-.4*x[:,1]
        score=modifier[:,None]*x[:,8:16]/np.sqrt(8)
        for name,f in (('overlapping_annotations_global',np.column_stack([left,right])),
                       ('weighted_score_modifier',score)):
            pf=f-u@(u.T@f)
            beta=rng.normal(size=f.shape[1]);beta*=np.sqrt(.03*rank/np.sum((pf@beta)**2))
            for signal in (False,True):
                y=mean[:,None]+signal*(f@beta)[:,None]+rng.normal(size=(n,100))
                summary=prepare_linear_scores(f,y,fixed,feature_names=tuple(map(str,range(f.shape[1]))),
                    trait_names=tuple(map(str,range(100))),metadata={})
                setting=f'{seed}_{name}_{"signal03" if signal else "null"}'
                for i in range(100):
                    try:
                        test=linear_score_tests(summary,trait=i)
                        record=dict(setting=setting,replicate=i,p=test['kernel_p'],failed=False,error=None)
                    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
                        record=dict(setting=setting,replicate=i,p=np.nan,failed=True,error=str(error))
                    records.append(record)
                design.append(dict(setting=setting,panel=meta,residual_rank=rank,
                    features=f.shape[1],realized_projected_signal_variance=.03 if signal else 0.,
                    interpretation='global-null combined kernel, no conditional component attribution'))
    frame=pd.DataFrame(records);frame.to_csv(args.out/'replicates.csv',index=False)
    rows=[]
    for name,part in frame.groupby('setting',sort=False):
        hits=int((part.p<=.05).sum());lo,hi=interval(hits,len(part))
        rows.append(dict(setting=name,fits=len(part),failures=int(part.failed.sum()),
            rejection=hits/len(part),lower=lo,upper=hi))
    pd.DataFrame(rows).to_csv(args.out/'summary.csv',index=False)
    (args.out/'design.json').write_text(json.dumps(dict(settings=design,
        fixed='genotypes, causal features, main and interaction coefficients, covariance model',
        regenerated='iid Gaussian residuals only',reference_probes='none',pair_sketches='none',
        acceptance='Same bounded screen as confirmation_design_v2.json; no tuning or extreme-tail claim',
        seconds=time.perf_counter()-started),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
