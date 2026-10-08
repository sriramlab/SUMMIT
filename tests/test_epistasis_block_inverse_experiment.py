"""Independent checks of the bounded computational prototype."""
import numpy as np
import pytest

from scripts.epistasis.block_inverse_experiment import block_inverse


@pytest.mark.parametrize('capacity',[16,256])
def test_block_galerkin_matches_dense_with_dependent_rhs_and_fixed_mean(capacity):
    from scipy.linalg import null_space
    rng=np.random.default_rng(327173)
    n=80
    c=np.column_stack([np.ones(n),rng.normal(size=(n,3))])
    u=np.linalg.qr(c)[0];z=null_space(c.T)
    g=rng.normal(size=(n,15))
    covariance=g@g.T/15+np.diag(rng.uniform(.7,1.1,n))
    y=rng.normal(size=(n,4))
    y=np.column_stack([y,y[:,0]+3*y[:,1],np.zeros(n)])
    expected=z@np.linalg.solve(z.T@covariance@z,z.T@y)
    actual,report=block_inverse(lambda x:covariance@x,y,
        lambda x:x/np.diag(covariance)[:,None],lambda x:x-u@(u.T@x),
        rtol=1e-10,capacity=capacity)
    np.testing.assert_allclose(actual,expected,atol=5e-10,rtol=5e-9)
    np.testing.assert_allclose(c.T@actual,0,atol=2e-13)
    assert max(report['relative_true_residual'])<=1e-10
    assert report['history'][0]['block_rank']<=4
    # Restarted true residuals can have an additional roundoff direction;
    # check the original dependence in the returned solution instead.
    np.testing.assert_allclose(actual[:,4],actual[:,0]+3*actual[:,1],atol=8e-10)
    if capacity==16:assert report['restarts']>0


def test_recycled_block_solution_uses_complete_operator():
    from test_epistasis_polygenic_operator import fixture
    from summit.epistasis.polygenic import projected_solve
    from summit.epistasis.krylov import KrylovSpace
    rng,make,_,i0,_,contexts=fixture()
    operator=make(i0,'packed');theta=rng.uniform(.1,1.,operator.count)
    c=np.column_stack([contexts[i0],contexts[i0,1]**2]);u=np.linalg.qr(c)[0]
    project=lambda x:x-u@(u.T@x)
    space=KrylovSpace(operator,theta,u,capacity=48)
    initial,_=projected_solve(operator,rng.normal(size=(len(i0),2)),c,theta,recycle_spaces=[space,space])
    inverse=space.freeze(operator.diagonal.T@theta)
    y=operator.apply(initial[:,:1])[:,:,0].T
    expected,_=projected_solve(operator,y,c,theta,preconditioners=[inverse]*y.shape[1])
    actual,report=block_inverse(lambda x:operator.apply(x,np.repeat(theta[:,None],x.shape[1],axis=1)),
        y,inverse.apply,project)
    np.testing.assert_allclose(actual,expected,atol=3e-8,rtol=3e-7)
    assert max(report['relative_true_residual'])<=1e-8
