"""Focused known-truth qualification of fixed-contrast genomic scale inference.

Genetic effects are redrawn each repetition and shared across both cohorts.
Population structure, dominance and cross-split genotype relatives coexist.
Report pointwise calibration separately from conservative family decisions.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm, beta as beta_dist

from summit.prediction.genotype import ArrayGenotypeSource
from summit.prediction.spec import VariantAxis
from summit.epistasis.polygenic import PolygenicKernels, fit_kernel_scales, components_from_normalized_moments
from summit.epistasis.scale_polygenic import prepare_polygenic_scale, boxcox_polygenic_scale_test
from summit.epistasis.scale_workflow import protected_scale_products


def study(*, n, replicates, continuous_replicates, case, seed, threads,
          max_evaluations=65, continuous_indices=None):
    start=time.monotonic(); rng=np.random.default_rng(seed); m=256; p=6
    nn,tn,products=protected_scale_products(threads)
    z=rng.normal(size=n)
    frequency=expit(rng.uniform(-1.8,-.3,m)[None,:]+.22*z[:,None]*rng.normal(size=m)[None,:])
    raw=rng.binomial(2,frequency).astype(np.int8)
    a=np.arange(n//5); b=np.arange(n//5,n)
    # Identical genotypes across the split induce genuine cross-cohort genomic
    # covariance; iid residual draws remain separate. No phenotype selection.
    related=min(len(a)//3,len(b));raw[b[:related]]=raw[a[:related]];z[b[:related]]=z[a[:related]]
    axis=VariantAxis(tuple('v'+str(j) for j in range(m)),('1',)*m,tuple(range(1,m+1)),('A',)*m,('C',)*m)
    source=ArrayGenotypeSource(raw,[(str(j),str(j)) for j in range(n)],axis,hard_calls=True)
    scales=fit_kernel_scales(source,a,np.arange(m),threads=threads)
    additive=(raw-scales['mean'][0])*scales['inverse_scale'][0]
    dominance=((raw==1)-scales['mean'][1])*scales['inverse_scale'][1]
    contexts=np.column_stack([np.ones(n),z]);noise=np.column_stack([np.ones(n),z*z])
    f=np.column_stack([additive[:,j]*additive[:,j+8] for j in range(p)])
    c=np.column_stack([contexts,z*z,additive[:,:16],dominance[:,:16]])
    def op(rows):
        owned=ArrayGenotypeSource(raw,[(str(j),str(j)) for j in range(n)],axis,hard_calls=True)
        return PolygenicKernels(owned,rows,np.arange(m),scales,contexts[rows],noise[rows],threads=threads)
    ga,gb=op(a),op(b)
    g=prepare_polygenic_scale(ga,gb,f[a],f[b],c[a],c[b],nn=nn,tn=tn,probes=128,seed=seed+1)
    theta=np.array([.2,.1,.1,.55,.05])
    if case=='boundary':theta[1:3]=0
    effect=np.zeros(p)
    if case=='sparse':effect[0]=.09
    if case=='diffuse':effect[:]=.045
    # Gaussian latent outcome, strictly positive observed phenotype; lambda=0
    # is the exact null scale. Other scales need not satisfy the null model.
    covariance=np.einsum('k,kij->ij',theta,g.grams)
    inverse=np.linalg.inv(covariance)
    counts={name:{'0.05':0,'0.01':0} for name in ('oracle','plugin_omnibus','plugin_sparse_hybrid')}
    selected=set(range(continuous_replicates)) if continuous_indices is None else set(continuous_indices)
    if not selected<=set(range(replicates)):raise ValueError('continuous repetition outside generated range')
    continuous=[]; boundaries=np.zeros(5,int); discrepancy=0.
    for begin in range(0,replicates,32):
        r=min(32,replicates-begin)
        genetic=(nn(additive,rng.normal(size=(m,r)))*np.sqrt(theta[0]/m)
                 +nn(additive,rng.normal(size=(m,r)))*z[:,None]*np.sqrt(theta[1]/m)
                 +nn(dominance,rng.normal(size=(m,r)))*np.sqrt(theta[2]/m))
        errors=rng.normal(size=(n,r)) if case!='heavy_noise' else rng.standard_t(8,size=(n,r))*np.sqrt(6/8)
        latent=genetic+errors*np.sqrt(theta[3]+theta[4]*z*z)[:,None]
        latent+=(.15*additive[:,0]+.08*dominance[:,3]+.1*z+f@effect)[:,None]
        residual=latent[a].copy();residual-=nn(g.he['basis'],tn(g.he['basis'],residual))
        moments=ga.component_grams(residual,phase='benchmark_training_moments')['gram']
        fitted=components_from_normalized_moments(np.diagonal(moments,axis1=1,axis2=2)/g.he['norms'][:,None],g.he)
        coefficients=tn(g.weights,latent[b])
        boundaries+=(fitted==0).sum(1)
        for j in range(r):
            estimate=coefficients[:,j];v=np.einsum('k,kij->ij',fitted[:,j],g.grams)
            full=float(chi2.sf(estimate@np.linalg.solve(v,estimate),p))
            sparse=min(1.,p*float(np.min(2*norm.sf(abs(estimate)/np.sqrt(np.diag(v))))))
            values=dict(oracle=float(chi2.sf(estimate@inverse@estimate,p)),plugin_omnibus=full,
                        plugin_sparse_hybrid=min(1.,2*min(full,sparse)))
            for name,value in values.items():
                for threshold in counts[name]:counts[name][threshold]+=int(value<float(threshold))
            if begin+j in selected:
                y=np.exp(.3*latent[:,j])
                result=boxcox_polygenic_scale_test(g,y[a],y[b],bounds=(-2.,2.),alpha=.05,max_evaluations=max_evaluations)
                point=next(x for x in result['points'] if x['power']==0.)
                discrepancy=max(discrepancy,abs(point['p']['joint']-full))
                test=result['tests']['joint']
                if test['p_upper']+1e-10<full:raise ArithmeticError('continuous envelope below true-scale p')
                continuous.append(dict(**test,replicate=begin+j,evaluations=result['evaluations']))
        print(json.dumps(dict(phase='benchmark',case=case,n=n,completed=begin+r,seconds=time.monotonic()-start)),flush=True)
    rates={name:{threshold:dict(count=count,rate=count/replicates,
            interval95=[float(beta_dist.ppf(.025,count,replicates-count+1)) if count else 0.,
                        float(beta_dist.ppf(.975,count+1,replicates-count)) if count<replicates else 1.])
            for threshold,count in values.items()} for name,values in counts.items()}
    native=products.finish_execution()
    return dict(n=n,training_n=len(a),markers=m,features=p,case=case,replicates=replicates,seed=seed,
        related_cross_split_pairs=related,true_components=theta.tolist(),pointwise=rates,
        boundary_counts=boundaries.tolist(),continuous=continuous,true_point_max_discrepancy=discrepancy,
        max_evaluations=max_evaluations,continuous_indices=sorted(selected),
        diagnostics=g.diagnostics,seconds=time.monotonic()-start,
        native_execution={k:native[k] for k in ('gemm_status','output_numa_status','gemm_record_count')},
        scope='Focused nominal-tail model calibration; not extreme-tail certification or real-model adequacy.')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--n',type=int,default=8192);parser.add_argument('--replicates',type=int,default=1024)
    parser.add_argument('--continuous-replicates',type=int,default=32)
    parser.add_argument('--continuous-indices',type=int,nargs='+')
    parser.add_argument('--max-evaluations',type=int,default=65)
    parser.add_argument('--case',choices=('null','boundary','sparse','diffuse','heavy_noise'),default='null')
    parser.add_argument('--seed',type=int,default=731401);parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=study(n=args.n,replicates=args.replicates,continuous_replicates=args.continuous_replicates,
                 case=args.case,seed=args.seed,threads=args.threads,max_evaluations=args.max_evaluations,
                 continuous_indices=args.continuous_indices)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
