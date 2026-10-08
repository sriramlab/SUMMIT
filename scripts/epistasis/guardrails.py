"""Bounded independent check of estimated residual surfaces and group nuisance fits."""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
import pandas as pd
from scripts.epistasis.confirm_inference import panel
from scripts.epistasis.validate import interval
from summit.epistasis.null import GaussianNullReference,GeneralGaussianNullReference
from summit.epistasis.score import refit_bootstrap
from summit.epistasis.oracle import explicit_pair_features
from summit.epistasis.cli import _jsonable


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--real-genotypes');parser.add_argument('--replicates',type=int,default=40)
    parser.add_argument('--bootstrap-draws',type=int,default=49)
    parser.add_argument('--seed-offset',type=int,default=0);args=parser.parse_args()
    if not 1<=args.replicates<=100:raise ValueError('replicate limit')
    args.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    x,h,meta=panel(args.real_genotypes,2583141+args.seed_offset,n=96,m=64)
    n,m=x.shape;rng=np.random.default_rng(7811643+args.seed_offset)
    g=x@x.T/m;records=[];details=[]
    # Frozen definitions; neither loci nor nuisance policy depend on phenotype.
    for case in ('dominance_guarded','heterogeneous_guarded','heterogeneous_unmodelled','group_annotation_null','group_annotation_signal'):
        target=case.startswith('dominance') or case.startswith('heterogeneous')
        fixed=np.column_stack([np.ones(n),x[:,0],h[:,0]]) if target else np.ones((n,1))
        if target:
            f=x[:,0,None]*x[:,8:16]/np.sqrt(8)
            if case=='heterogeneous_guarded':
                nuisance=[g,np.diag(x[:,0]**2),np.eye(n)];coefficients=[.3,.49,.21]
                null=GeneralGaussianNullReference(nuisance,fixed,names=('additive','modifier_square','residual'),identity=case)
            else:
                nuisance=[g,np.eye(n)];coefficients=[.3,.7]
                null=GaussianNullReference(g,fixed,identity=case)
            y=np.sqrt(.3/m)*x@rng.normal(size=(m,args.replicates))
            variance=.7*(.3+.7*x[:,0]**2) if case.startswith('heterogeneous') else np.full(n,.7)
            y+=np.sqrt(variance)[:,None]*rng.normal(size=(n,args.replicates))
            if case.startswith('dominance'):y+=2*h[:,0,None]*rng.normal(size=(1,args.replicates))
        else:
            a=np.zeros(m);a[:4]=1;b=np.zeros(m);b[2:6]=1
            f,_,_=explicit_pair_features(x,a,b)
            ga=x[:,:4]@x[:,:4].T/4;gb=x[:,2:6]@x[:,2:6].T/4
            nuisance=[g,ga,gb,np.eye(n)];coefficients=[.2,.3,.3,.7]
            null=GeneralGaussianNullReference(nuisance,fixed,names=('additive','group_a','group_b','residual'),identity=case)
            y=np.linalg.cholesky(sum(c*k for c,k in zip(coefficients,nuisance)))@rng.normal(size=(n,args.replicates))
            if case.endswith('signal'):y+=np.sqrt(.08)*f@rng.normal(size=(f.shape[1],args.replicates))
        z=null.rotation.T@f;k=z@z.T
        details.append(dict(case=case,genotypes='fixed',causal_sets='fixed',effects='regenerated random effects',
            residuals='regenerated Gaussian',nuisance='all fitted, bootstrap refits',sketches='none',
            coefficients=coefficients,expected_interaction_contribution=.08*np.sum(z*z)/null.rank if case.endswith('signal') else 0.))
        for i in range(args.replicates):
            try:
                result=refit_bootstrap(null,[k],y[:,i],draws=args.bootstrap_draws,seed=17110+i+args.seed_offset)
                record=dict(setting=case,replicate=i,p=float(result['p'][0]),failed=False,error=None,
                    information_fraction=result['efficient_information_fraction'][0],boundary=result['null_fit']['genetic_boundary'])
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as e:
                record=dict(setting=case,replicate=i,p=np.nan,failed=True,error=str(e))
            records.append(record)
        print(case,round(time.perf_counter()-start,2),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(args.out/'replicates.csv',index=False)
    table=[]
    for name,part in frame.groupby('setting',sort=False):
        hits=int((part.p<=.05).sum());lo,hi=interval(hits,len(part))
        table.append(dict(setting=name,fits=len(part),failures=int(part.failed.sum()),rejection=hits/len(part),lower=lo,upper=hi))
    pd.DataFrame(table).to_csv(args.out/'summary.csv',index=False)
    (args.out/'design.json').write_text(json.dumps(_jsonable(dict(panel=meta,settings=details,draws=args.bootstrap_draws,
        phenotype_seed=7811643+args.seed_offset,seed_offset=args.seed_offset,
        seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        purpose='bounded estimated-nuisance safeguard screen; 40 replicates cannot establish production calibration')),indent=2)+'\n')
    print(pd.DataFrame(table).to_string(index=False))


if __name__=='__main__':main()
