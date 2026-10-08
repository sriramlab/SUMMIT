"""Cheap conditional eigenvalue checks; not additional full nuisance-refit simulations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scripts.epistasis.confirm_inference import panel
from scripts.epistasis.validate import interval
from summit.epistasis.oracle import explicit_pair_features
from summit.epistasis.score import prepare_linear_scores
from summit.epistasis.quadratic import quadratic_sf


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--real-genotypes',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);records=[];rng=np.random.default_rng(982515)
    for seed in (1763011,1773019):
        x,h,meta=panel(a.real_genotypes,seed,n=384,m=96);fixed=np.column_stack([np.ones(384),x,h[:,:8]])
        left=np.zeros(96);left[:4]=1;right=np.zeros(96);right[2:8]=1
        for name,b,within in (('overlap',right,False),('within',left,True)):
            f,_,_=explicit_pair_features(x,left,b,within=within)
            summary=prepare_linear_scores(f,rng.normal(size=384),fixed,feature_names=tuple(map(str,range(f.shape[1]))),trait_names=('y',),metadata={})
            lam=np.linalg.eigvalsh(summary.information);lam=lam[lam>lam[-1]*1e-10]
            r=summary.residual_rank;d=len(lam)
            chi=rng.chisquare(1,(50000,d));other=rng.chisquare(r-d,50000)
            ratios=(chi@lam)/(chi.sum(axis=1)+other)
            for alpha in (.05,.001):
                def sf(q):return quadratic_sf(0,np.r_[lam-q,-q],multiplicities=np.r_[np.ones(d),r-d],atol=1e-10)['p']
                cutoff=brentq(lambda q:sf(q)-alpha,0,float(lam[-1]),xtol=1e-10)
                hits=int(np.sum(ratios>=cutoff));lo,hi=interval(hits,len(ratios))
                records.append(dict(seed=seed,group=name,alpha=alpha,rank=d,residual_rank=r,cutoff=cutoff,
                    numeric_tail=sf(cutoff),draws=len(ratios),rejection=hits/len(ratios),lower=lo,upper=hi))
    (a.out/'tails.json').write_text(json.dumps(dict(records=records,seed=982515,
        contract='Genotypes and resulting spectra fixed. Independent chi-square draws check the exact ratio distribution, including unknown scale. This is not evidence for Gaussian covariance nuisance refitting or misspecification robustness.'),indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
