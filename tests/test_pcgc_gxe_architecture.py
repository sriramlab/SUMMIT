"""Fourth-order LD traces checked independently in variant/context space."""
import numpy as np
import pytest
from summit.pcgc.architecture import ArchitectureSketch,gaussian_architecture_covariance
from summit.pcgc.gxe import context_pairs
from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator,ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator,ProtectedTNOperator
from prediction_helpers import prediction_threads


@pytest.mark.parametrize('native',[False,True])
def test_four_group_architecture_matches_exact_variant_trace(native):
    rng = np.random.default_rng(3271)
    n,m,q = 32,9,2
    x = rng.normal(size=(n,m))
    psi = np.column_stack((np.ones(n),rng.normal(size=n)))
    a = np.column_stack((np.ones(m),rng.uniform(.1,1,m)))
    threads = prediction_threads()
    nn = ProtectedNNOperator(threads=threads) if native else NumpyNNOperator(threads=threads)
    tn = ProtectedTNOperator(threads=threads) if native else NumpyTNOperator(threads=threads)
    sketch = ArchitectureSketch(psi,a,np.arange(n)%2==0,probes=m,seed=221,nn=nn,tn=tn,native=native,threads=threads)
    # A complete orthogonal probe basis removes random trace error and tests
    # every source/target orientation, annotation mass and participant group.
    sketch.probe_block = lambda start,stop,family: np.asfortranarray(np.sqrt(m)*np.eye(m)[start:stop])
    for traversal in (1,2):
        for start in range(0,m,4):
            sketch.read_block(traversal,start,min(m,start+4),np.asfortranarray(x[:,start:start+4]))
    F = (x[:,:,None]*psi[:,None,:]).reshape(n,m*q)
    L = [F.T@((sketch.weights*(sketch.group == g))[:,None]*F) for g in range(4)]
    matrices = []
    omega = np.array([[[.2,.015],[.015,.04]],[[.07,-.01],[-.01,.03]]])
    V = np.zeros((m*q,m*q))
    theta = []
    for ann,mass,om in zip(a.T,a.sum(0),omega):
        V += np.kron(np.diag(ann/mass),om)
        for u,v in context_pairs(q):
            context = np.zeros((q,q))
            context[u,v] = context[v,u] = 1
            matrices.append(np.kron(np.diag(ann/mass),context))
            theta.append(om[u,v])
    expected = np.array([[2*np.trace(L[0]@C@L[1]@V@L[2]@D@L[3]@V) for D in matrices] for C in matrices])
    actual,diag = gaussian_architecture_covariance(theta,q,psi.T@psi/n,sketch.left,sketch.right,m)
    np.testing.assert_allclose(actual,(expected+expected.T)/2,rtol=5e-13,atol=1e-15)
    np.testing.assert_allclose(diag['working_omega'],omega,rtol=3e-14,atol=2e-16)
    # A nonorthogonal basis change also exercises the working PSD projection
    # when one unrestricted annotation estimate is indefinite.
    from summit.pcgc.gxe import _omega
    basis = np.array([[1.,.3],[-.2,1.4]])
    inverse = np.linalg.inv(basis)
    indefinite = omega.copy(); indefinite[0,1,1] = -.01
    coefficients = np.array([item[u,v] for item in indefinite for u,v in context_pairs(q)])
    cov0,d0 = gaussian_architecture_covariance(coefficients,q,psi.T@psi/n,sketch.left,sketch.right,m)
    transformed = ArchitectureSketch(psi@basis,a,np.arange(n)%2==0,probes=m,seed=221,nn=nn,tn=tn,native=native,threads=threads)
    transformed.probe_block = sketch.probe_block
    for traversal in (1,2):
        for start in range(0,m,4):
            transformed.read_block(traversal,start,min(m,start+4),np.asfortranarray(x[:,start:start+4]))
    changed = inverse@indefinite@inverse.T
    coefficient_new = np.array([item[u,v] for item in changed for u,v in context_pairs(q)])
    cov1,d1 = gaussian_architecture_covariance(coefficient_new,q,(psi@basis).T@(psi@basis)/n,
        transformed.left,transformed.right,m)
    expected_working = inverse@np.asarray(d0['working_omega'])@inverse.T
    np.testing.assert_allclose(d1['working_omega'],expected_working,rtol=2e-12,atol=2e-15)
    J = np.empty((6,6))
    for column in range(6):
        changed = basis@_omega(np.eye(6)[column],q)@basis.T
        J[:,column] = [item[u,v] for item in changed for u,v in context_pairs(q)]
    np.testing.assert_allclose(cov1,J.T@cov0@J,rtol=2e-11,atol=2e-15)
