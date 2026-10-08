"""Diagnostic direction comparisons with the same fitted conditional null.

Only post-fit experiments call this module. It reuses the public estimator's
outcome prediction and covariance-mean tangents, never generating means as
nuisance inputs. Each comparator has its own declared score main effects.
"""
from pathlib import Path

import numpy as np

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.polygenic import projected_solve, project
from summit.prediction._validation import array_digest


def compare_directions(training, complete, i0, i1, *, f0, f1, y1, prediction,
                       fixed0, fixed1, extras, tangents, theta, checkpoint_dir, resume=False):
    """One scalar contrast per column, with full training covariance correction."""
    b=f0.shape[1]
    if (f1.shape[1]!=b or y1.shape!=f1.shape or prediction.shape!=f1.shape
            or theta.shape!=(training.count,b) or len(extras)!=b or tangents.shape[:2]!=f1.shape):
        raise ValueError('comparator traits, features, means and covariance axes differ')
    # Equal geometry has the same contrast, regardless of confirmation Y.
    # In null comparisons the prespecified burden and oracle often coincide.
    # Reuse only exact arrays, retaining both reported tests and their family.
    representatives=[];indices=[];seen={}
    for j in range(b):
        arrays=(f0[:,j],f1[:,j],theta[:,j],extras[j],tangents[:,j])
        key=tuple(array_digest(v) for v in arrays)
        if key not in seen:
            seen[key]=len(representatives);representatives.append(j)
        else:
            k=representatives[seen[key]]
            if not all(np.array_equal(v,w) for v,w in zip(arrays,
                    (f0[:,k],f1[:,k],theta[:,k],extras[k],tangents[:,k]))):
                raise ValueError('comparator digest collision')
        indices.append(seen[key])
    keep=np.asarray(representatives);mapping=np.asarray(indices)
    result=_compare_unique(training,complete,i0,i1,f0=f0[:,keep],f1=f1[:,keep],
        fixed0=fixed0,fixed1=fixed1,extras=[extras[j] for j in keep],
        tangents=tangents[:,keep],theta=theta[:,keep],checkpoint_dir=checkpoint_dir,resume=resume)
    for name in ('contrasts','training_contrasts','response'):
        result[name]=result[name][:,mapping]
    result['variance']=result['variance'][mapping]
    result['diagnostics']=[result['diagnostics'][j] for j in mapping]
    result['beta']=np.sum(result['contrasts']*(y1-prediction),axis=0)
    result['unique_columns']=representatives
    result['column_mapping']=indices
    return result


def _compare_unique(training,complete,i0,i1,*,f0,f1,fixed0,fixed1,extras,
                    tangents,theta,checkpoint_dir,resume):
    b=f0.shape[1]
    root=Path(checkpoint_dir);root.mkdir(exist_ok=resume)
    first_path=root/'feature.npz'
    inverse,first=projected_solve(training,f0,fixed0,theta,checkpoint=first_path,
        resume=resume and first_path.exists())
    transfer,_=complete.cross_products(inverse,i0,i1,theta)
    response=f1-transfer
    a=np.empty_like(response);diagnostics=[]
    for j in range(b):
        basis=thin_rank_revealing_fixed_effect_basis(
            np.column_stack([fixed1,extras[j],tangents[:,j]]),rtol=1e-11)
        residual=project(basis,project(basis,response[:,j]))
        energy=residual@residual
        if energy<=1e-20*max(float(f1[:,j]@f1[:,j]),np.finfo(float).tiny):
            raise ValueError('comparator interaction is absorbed by its nuisance mean')
        response[:,j]=residual;a[:,j]=residual/energy
        rank=basis.shape[1]+1
        leverage=np.sum(basis*basis,axis=1)+residual*residual/energy
        effective=float(energy**2/np.sum(residual**4))
        outside=[reason for failed,reason in ((len(i1)<1000,'N below 1000'),
            (rank/len(i1)>.05,'fitted rank exceeds 5% of N'),
            (leverage.max()>.1,'maximum leverage exceeds .1'),
            (effective<100,'feature effective support below 100')) if failed]
        diagnostics.append(dict(rank=rank,max_leverage=float(leverage.max()),
            feature_effective_support=effective,outside_confirmation_design=outside))
    rhs,quadratic=complete.cross_products(a,i1,i0,theta)
    second_path=root/'variance.npz'
    inverse,second=projected_solve(training,rhs,fixed0,theta,checkpoint=second_path,
        resume=resume and second_path.exists())
    variance=quadratic-np.sum(rhs*inverse,axis=0)
    if np.any(variance<=0):raise ValueError('nonpositive comparator covariance')
    return dict(variance=variance,
        contrasts=a,training_contrasts=-inverse,response=response,
        solver_reports=[first.reports,second.reports],diagnostics=diagnostics)
