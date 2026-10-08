"""Bounded computational experiment; not a public covariance solver.

Each block expands an A-orthonormal Galerkin space with preconditioned
residuals. Two reorthogonalization passes retain the full covariance product
of every vector. Rank revelation removes dependent right-hand sides, not
estimable directions: convergence requires a fresh full-operator residual
for every original right-hand side. Restart bounds the stored space.

The prespecified promotion gate is >=20% derivative-stage wall-time reduction
against the existing recycled solver, at the same 1e-8 true-residual tolerance,
with relative solution difference <=1e-7. Passing a subset benchmark alone
does not establish a full-marker gain or qualify statistical inference.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np


def block_inverse(apply, rhs, precondition, project=lambda x: x, *,
                  rtol=1e-8, max_steps=100, capacity=256):
    """Solve on the supplied projected span, with bounded exact product pairs."""
    b=project(np.asarray(rhs,float))
    if b.ndim!=2 or not np.all(np.isfinite(b)) or not b.shape[1]:
        raise ValueError('finite nonempty matrix of right-hand sides required')
    if not 0<rtol<1 or max_steps<1 or capacity<b.shape[1]:
        raise ValueError('positive tolerance, steps and sufficient block capacity required')
    scale=np.linalg.norm(b,axis=0)
    scale=np.where(scale>0,scale,1.)
    b=b/scale
    x=np.zeros_like(b);r=b.copy()
    d=np.empty((len(b),0));ad=d.copy()
    history=[];calls=0;restarts=0;start=time.perf_counter()
    for step in range(max_steps+1):
        residual=np.linalg.norm(r,axis=0)
        if residual.max()<=rtol or step==max_steps:
            r=b-project(apply(x));calls+=1
            residual=np.linalg.norm(r,axis=0)
            if residual.max()<=rtol:
                return x*scale,dict(steps=step,operator_calls=calls,restarts=restarts,
                    relative_true_residual=residual.tolist(),history=history,
                    seconds=time.perf_counter()-start)
            # True-residual replacement also discards the accumulated basis;
            # subsequent directions solve the remaining error, not a new RHS.
            d=np.empty((len(b),0));ad=d.copy();restarts+=1
        if step==max_steps:
            break
        if d.shape[1]+b.shape[1]>capacity:
            r=b-project(apply(x));calls+=1
            d=np.empty((len(b),0));ad=d.copy();restarts+=1
        z=project(precondition(r));az=project(apply(z));calls+=1
        for _ in range(2):
            coefficient=d.T@az
            z-=d@coefficient;az-=ad@coefficient
        gram=z.T@az;gram=(gram+gram.T)/2
        values,vectors=np.linalg.eigh(gram)
        threshold=128*len(values)*np.finfo(float).eps*max(values[-1],np.finfo(float).tiny)
        if values[0]<-threshold:
            raise ValueError('nonpositive block covariance span')
        keep=values>threshold
        if not np.any(keep):
            raise RuntimeError('block covariance space stalled before verified convergence')
        transform=vectors[:,keep]/np.sqrt(values[keep])
        q=z@transform;aq=az@transform
        coefficient=q.T@r
        x+=q@coefficient;r-=aq@coefficient
        d=np.column_stack([d,q]);ad=np.column_stack([ad,aq])
        history.append(dict(step=step+1,block_rank=int(keep.sum()),
            space_rank=d.shape[1],max_recursive_residual=float(np.linalg.norm(r,axis=0).max())))
    raise RuntimeError(f'block inverse did not converge: residual {residual.max():.6g}')


def main():
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
    from summit.prediction.genotype import FileGenotypeSource
    from summit.epistasis.krylov import KrylovSpace
    from summit.epistasis.polygenic import PolygenicKernels,projected_solve
    from scripts.epistasis.full_matched import write_json
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--markers',type=int,default=4096)
    p.add_argument('--num-threads',type=int,default=1)
    p.add_argument('--memory-gib',type=float,default=16)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((a.reference/'reference.json').read_text())
    spec_path=Path(metadata['training_manifest']);spec=json.loads(spec_path.read_text())
    with np.load(a.reference/'reference.npz') as archive:
        data={k:archive[k] for k in ('rows','training_index','variants','mean','inverse_scale','fixed','contexts','noise')}
    rng=np.random.default_rng(927413)
    selected=np.sort(rng.choice(len(data['variants']),a.markers,replace=False))
    variants=data['variants'][selected];i0=data['training_index']
    rows=data['rows'][i0];fixed=data['fixed'][i0]
    theta=np.asarray(json.loads(a.components.read_text())['estimates'])[:,0]
    y=rng.normal(size=(len(rows),2))
    write_json(a.out/'prespecified.json',dict(seed=927413,minimum_speedup_fraction=.20,
        rtol=1e-8,relative_solution_difference_limit=1e-7,capacity=256,
        interpretation='bounded warm-marker computational experiment, not statistical evidence'))
    start=time.perf_counter()
    with FileGenotypeSource(spec_path.parent/spec['genotypes']['geno'],
            genome_build=spec['genotypes'].get('genome_build')) as source:
        scales=dict(metadata['scales'],mean=data['mean'][:,selected],inverse_scale=data['inverse_scale'][:,selected],
            variants=source.variants.subset(variants).identity)
        operator=PolygenicKernels(source,rows,variants,scales,data['contexts'][i0],data['noise'][i0],
            threads=a.num_threads,block_size=4096,memory_bytes=int(a.memory_gib*2**30),storage='packed')
        basis=thin_rank_revealing_fixed_effect_basis(fixed,rtol=1e-11)
        project=lambda x:x-basis@(basis.T@x)
        space=KrylovSpace(operator,theta,basis,capacity=64)
        initial,first=projected_solve(operator,y,fixed,theta,recycle_spaces=[space,space])
        y=operator.apply(initial[:,:1])[:,:,0].T
        inverse=space.freeze(operator.diagonal.T@theta)
        write_json(a.out/'initial_solve.json',dict(reports={'/'.join(k):v for k,v in first.reports.items()},
            recycle_rank=inverse.d.shape[1],seconds=first.elapsed_seconds))
        print('initial recycle solve',round(first.elapsed_seconds,2),flush=True)
        theta_batch=np.repeat(theta[:,None],y.shape[1],axis=1)
        begin,cpu=time.perf_counter(),time.process_time()
        expected,fit=projected_solve(operator,y,fixed,theta_batch,preconditioners=[inverse]*y.shape[1])
        serial=dict(seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,
            reports={'/'.join(k):v for k,v in fit.reports.items()})
        write_json(a.out/'existing.json',serial)
        print('existing recycled',round(serial['seconds'],2),flush=True)
        begin,cpu=time.perf_counter(),time.process_time()
        actual,report=block_inverse(lambda x:operator.apply(x,np.repeat(theta[:,None],x.shape[1],axis=1)),
            y,inverse.apply,project)
        blocked=dict(seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,report=report)
        difference=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        gain=1-blocked['seconds']/serial['seconds']
        write_json(a.out/'comparison.json',dict(rows=len(rows),markers=len(variants),rhs=y.shape[1],
            covariance_components=theta,existing=serial,blocked=blocked,relative_solution_difference=difference,
            wall_reduction_fraction=gain,passes_computational_gate=bool(difference<=1e-7 and gain>=.20),
            seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            ledger=asdict(operator.stream.ledger)))
        print('blocked',round(blocked['seconds'],2),'relative difference',difference,'wall reduction',gain,flush=True)


if __name__=='__main__':
    main()
