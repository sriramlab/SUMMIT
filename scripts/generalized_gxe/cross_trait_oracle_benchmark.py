"""Independent dense N=2000, M=3000 cross-trait oracle; exclusive JSON output."""
from pathlib import Path
import argparse
import json
import os
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from summit.context.cross_trait_oracle import dense_cross_trait_moments
from summit.context.cross_trait_fit import assemble_cross_trait_normal_equations,solve_cross_trait_normal_equations
from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
from summit.ldscore.generalized_gxe_cross_trait_batch import CrossTraitBatch
from summit.context.reference_zpass_cli import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=2000);p.add_argument('--m',type=int,default=3000)
    p.add_argument('--seed',type=int,default=230923);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();rng=np.random.default_rng(args.seed)
    n,m,q=args.n,args.m,3
    phi=np.c_[np.ones(n),rng.normal(size=(n,2))]
    fixed=np.linalg.qr(np.c_[phi,rng.normal(size=(n,2))])[0]
    d=np.c_[phi,phi[:,1]**2,phi[:,1]*phi[:,2],phi[:,2]**2]
    nx=int(n/1.4);offset=n-nx;rows=(np.arange(nx),np.arange(offset,n))
    g=rng.normal(size=(n,m));traits=[]
    for i,idx in enumerate(rows):
        u=np.linalg.qr(fixed[idx])[0]@np.linalg.qr(rng.normal(size=(5,5)))[0]
        traits.append(dict(name=str(i),indices=idx,fixed_basis=u,phenotype=rng.normal(size=len(idx))))
    oracle=dense_cross_trait_moments(g,phi,rows_x=rows[0],rows_y=rows[1],fixed_x=traits[0]['fixed_basis'],
        fixed_y=traits[1]['fixed_basis'],phenotype_x=traits[0]['phenotype'],phenotype_y=traits[1]['phenotype'],residual_basis=d)
    masked=MaskedTraitBatch(basis=phi,fixed_basis=fixed,residual_basis=d,traits=traits)
    groups=np.arange(m)*20//m
    batch=CrossTraitBatch(masked,block_ids=np.unique(groups),annotation_names=('all',))
    for begin in range(0,m,128):
        end=min(begin+128,m)
        list(batch.block(g[:,begin:end].T,np.ones((end-begin,1)),groups[begin:end]))
    features=[]
    for idx,trait in zip(rows,traits):
        u=trait['fixed_basis'];f=np.einsum('ia,ij->aij',phi[idx],g[idx])
        f-=np.einsum('ic,lc,alj->aij',u,u,f,optimize=True);features.append(f)
    # Independent SNP-axis contractions, not an oracle kernel reshaping.
    rx=np.einsum('aij,cik->acjk',features[0],features[0],optimize=True)
    ry=np.einsum('bij,dik->bdjk',features[1],features[1],optimize=True)
    gram=np.einsum('acjm,bdjm->abcd',rx,ry,optimize=True).reshape(q*q,q*q)/m**2
    overlap,lx,ly=np.intersect1d(*rows,return_indices=True)
    diag=np.einsum('aij,bij->abi',features[0][:,lx],features[1][:,ly],optimize=True).reshape(q*q,-1)/m
    observed=dict(genetic_gram=gram,genetic_rhs=batch.scores.rhs.sum(axis=0).reshape(-1)/m,
        genetic_residual=batch.genetic_residual.sum(axis=0).reshape(q*q,-1)/m,
        residual_gram=batch.residual_gram[0],residual_rhs=batch.residual_rhs[0],same_person=diag@diag.T)
    errors={}
    for name,value in observed.items():
        errors[name]=float(np.linalg.norm(value-oracle[name])/max(1,np.linalg.norm(oracle[name])))
        if errors[name]>1e-9:
            raise AssertionError((name,errors[name]))
    keys=('genetic_gram','genetic_rhs','genetic_residual','residual_gram','residual_rhs')
    eq=assemble_cross_trait_normal_equations(**{k:observed[k] for k in keys},num_basis=q,annotation_masses=[m])
    solve=solve_cross_trait_normal_equations(eq)
    dense_eq=np.block([[oracle['genetic_gram'],oracle['genetic_residual']],
                       [oracle['genetic_residual'].T,oracle['residual_gram']]])
    exact=np.linalg.solve(dense_eq,np.r_[oracle['genetic_rhs'],oracle['residual_rhs']])
    actual=np.r_[solve.coefficients[:q*q],eq.residual_transform@solve.coefficients[q*q:]]
    errors['coefficients']=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
    assert errors['coefficients']<1e-9
    result=dict(n=n,m=m,q=q,n_x=nx,n_y=nx,n_overlap=len(overlap),seed=args.seed,
        relative_errors=errors,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpus=sorted(os.sched_getaffinity(0)),script_sha256=file_sha256(__file__),passed=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
