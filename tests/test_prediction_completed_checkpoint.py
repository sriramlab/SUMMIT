"""Completed restart states require fresh numerical, not only byte, checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from summit.prediction.checkpoint import SolverCheckpoint
from summit.prediction.solver import solve
from summit.prediction.spec import SolverSpec


def test_completed_checkpoint_rechecks_operator_and_reconstructs_fixed_mean(tmp_path, monkeypatch):
    rng=np.random.default_rng(417349)
    n=24;z=rng.normal(size=(n,6));a=z@z.T+np.diag(rng.uniform(.5,1.5,n))
    fixed=np.column_stack([np.ones(n),rng.normal(size=n)])
    y=rng.normal(size=n)+fixed@np.array([3.,-2.])
    key=('trait','candidate');calls=[]
    def product(vectors, *, phase):
        calls.append((phase,tuple(vectors)))
        return {k:a@v for k,v in vectors.items()}
    candidate=SimpleNamespace(id=key[1],residual=np.diag(a),covariance=np.zeros((1,1)))
    trait=SimpleNamespace(id=key[0],rows=np.arange(n),y=y,fixed=fixed,phi=np.ones((n,1)),candidates=[candidate])
    operator=SimpleNamespace(traits=[trait],row_diagonal={0:np.zeros(n)},trait_group={key[0]:0},apply=product)
    spec=SolverSpec(rtol=1e-10,max_iterations=100)
    path=tmp_path/'completed.npz'
    with SolverCheckpoint(path,'bounded_operator',resume=False) as checkpoint:
        initial=solve(operator,spec,checkpoint=checkpoint)
    original=path.read_bytes();calls.clear()
    # Independently solve the constrained full system, including its mean.
    reference=np.linalg.solve(np.block([[a,fixed],[fixed.T,np.zeros((2,2))]]),np.r_[y,[0.,0.]])
    with SolverCheckpoint(path,'bounded_operator',resume=True) as checkpoint:
        resumed=solve(operator,spec,checkpoint=checkpoint)
    assert calls==[('verification',(key,))]
    np.testing.assert_allclose(resumed.solutions[key],reference[:n],rtol=2e-8,atol=2e-10)
    np.testing.assert_allclose(resumed.fixed_coefficients[key],reference[n:],rtol=2e-8,atol=2e-10)
    np.testing.assert_array_equal(resumed.solutions[key],initial.solutions[key])
    assert path.read_bytes()==original

    load=SolverCheckpoint.load
    def stale_mean(self,*args):
        state=load(self,*args);state['fixed_coefficients'][key][:]=1e6
        return state
    with monkeypatch.context() as patch:
        patch.setattr(SolverCheckpoint,'load',stale_mean)
        with SolverCheckpoint(path,'bounded_operator',resume=True) as checkpoint:
            recomputed=solve(operator,spec,checkpoint=checkpoint)
    np.testing.assert_allclose(recomputed.fixed_coefficients[key],reference[n:],atol=2e-10)

    def bad_solution(self,*args):
        state=load(self,*args);state['x'][key][0]+=.01
        # A stale or corrupt reported tolerance must not authorize the error.
        state['reports'][key]['threshold']=1e6
        return state
    with monkeypatch.context() as patch:
        patch.setattr(SolverCheckpoint,'load',bad_solution)
        with SolverCheckpoint(path,'bounded_operator',resume=True) as checkpoint:
            with pytest.raises(ValueError,match='Completed checkpoint failed fresh covariance'):
                solve(operator,spec,checkpoint=checkpoint)
    assert path.read_bytes()==original
