"""Independent inverse and full-operator checks for computational recycling."""
import numpy as np
import pytest


def test_recycled_inverse_is_spd_and_exact_on_supplied_span():
    from summit.epistasis.krylov import RecycledInverse
    rng=np.random.default_rng(591727)
    n=64
    q=rng.normal(size=(n,n));a=q@q.T+np.diag(rng.uniform(.2,.8,n))
    d=rng.normal(size=(n,12));d=np.column_stack([d,d[:,0]])
    inverse=RecycledInverse('fixture',np.array([1.]),d,a@d,np.diag(a))
    e=np.linalg.pinv(d.T@a@d,rcond=1e-12)
    left=np.eye(n)-d@e@d.T@a
    expected=d@e@d.T+left@np.diag(1/np.diag(a))@left.T
    actual=inverse.apply(np.eye(n))
    np.testing.assert_allclose(actual,expected,atol=3e-15)
    assert np.linalg.eigvalsh(actual)[0]>0
    np.testing.assert_allclose(actual@a@d,d,atol=2e-13)
    assert not inverse.d.flags.writeable
    with pytest.raises(ValueError,match='positive semidefinite'):
        RecycledInverse('bad',np.array([1.]),d,-a@d,np.diag(a))


def test_recycling_native_solver_preserves_fixed_mean_and_true_residual(tmp_path):
    from test_epistasis_polygenic_operator import fixture
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
    from summit.epistasis.polygenic import projected_solve
    from summit.epistasis.krylov import KrylovSpace
    rng,make,_,i0,_,contexts=fixture()
    operator=make(i0,'packed');theta=rng.uniform(.1,1.,operator.count)
    c=np.column_stack([contexts[i0],contexts[i0,1]**2]);u=thin_rank_revealing_fixed_effect_basis(c,rtol=1e-11)
    space=KrylovSpace(operator,theta,u,capacity=48)
    first,_=projected_solve(operator,rng.normal(size=(len(i0),2)),c,theta,recycle_spaces=[space,space])
    inverse=space.freeze(operator.diagonal.T@theta)
    assert inverse.d.shape[1]>2 and inverse.d.shape[1]<=48
    y=rng.normal(size=(len(i0),3))+c@rng.normal(size=(c.shape[1],3))
    expected,_=projected_solve(operator,y,c,theta)
    actual,fit=projected_solve(operator,y,c,theta,preconditioners=[inverse]*3,checkpoint=tmp_path/'recycled.npz')
    np.testing.assert_allclose(actual,expected,atol=3e-8,rtol=3e-7)
    assert all(v['relative_true_residual']<=1e-8 for v in fit.reports.values())
    np.testing.assert_allclose(u.T@actual,0,atol=1e-12)
    resumed,_=projected_solve(operator,y,c,theta,preconditioners=[inverse]*3,checkpoint=tmp_path/'recycled.npz',resume=True)
    np.testing.assert_array_equal(resumed,actual)
    from summit.epistasis.krylov import write_recycled_inverses,load_recycled_inverses
    write_recycled_inverses(tmp_path/'inverse.npz',[inverse],identity='matched')
    saved=load_recycled_inverses(tmp_path/'inverse.npz',identity='matched',operator=operator,
        theta=theta[:,None],memory_bytes=2**30)[0]
    assert saved.identity==inverse.identity
    np.testing.assert_array_equal(saved.apply(y),inverse.apply(y))
    with pytest.raises(ValueError,match='definition changed'):
        load_recycled_inverses(tmp_path/'inverse.npz',identity='changed',operator=operator,
            theta=theta[:,None],memory_bytes=2**30)
    with pytest.raises(ValueError,match='invalid recycled'):
        load_recycled_inverses(tmp_path/'inverse.npz',identity='matched',operator=operator,
            theta=theta[:,None]+.1,memory_bytes=2**30)
