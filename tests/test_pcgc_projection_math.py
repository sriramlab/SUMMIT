"""Independent checks of the proposed projected PCGC equations.

These tests qualify the algebra. They do not add a production projected mode
or establish finite-sample calibration when risks/projections are fitted.
"""
import itertools

import numpy as np
import pytest
from scipy.special import ndtr


def offdiag(a):
    return a - np.diag(np.diag(a))


def diagonal_projected_diagonal(v, u):
    leverage = np.sum(u*u, axis=1)
    middle = u.T @ (v[:, None]*u)
    return v*(1-2*leverage) + np.einsum('ir,rs,is->i', u, middle, u)


def example(n=7, m=13):
    rng = np.random.default_rng(83512)
    x = rng.normal(size=(n, m))
    phi = np.column_stack([np.ones(n), np.linspace(-1, 1, n), rng.normal(size=n)])
    features = [phi[:, k, None]*x for k in range(3)]
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    targets = []
    for j in range(m):
        for a, b in pairs:
            matrix = np.outer(features[a][:, j], features[b][:, j])
            if a != b:
                matrix += matrix.T
            targets.append(matrix)
    targets = np.asarray(targets).reshape(m, 6, n, n)
    u = np.linalg.qr(phi[:, :2], mode='reduced')[0]
    return targets, u


def test_low_rank_corrections_match_every_dense_target_row():
    targets, u = example(n=11)
    P = np.eye(len(u))-u@u.T
    sources = targets.mean(axis=0)
    z = np.linspace(-1.2, 1.5, len(u))
    r = P@z
    source_projected = [P@b@P for b in sources]
    source_offdiag_projected = [P@offdiag(b)@P for b in sources]
    response = P@offdiag(np.outer(z, z))@P
    for target in targets.reshape(-1, len(u), len(u)):
        projected = P@target@P
        exact = P@offdiag(target)@P
        v, a = np.diag(target), np.diag(projected)
        rhs = r@target@r - a@(z*z) + v@(diagonal_projected_diagonal(z*z, u)-r*r)
        np.testing.assert_allclose(rhs, np.sum(exact*response), atol=2e-12)
        for source, projected_source, exact_source in zip(sources, source_projected, source_offdiag_projected):
            w, b = np.diag(source), np.diag(projected_source)
            gram = (np.sum(projected*projected_source)-a@w-v@b
                    + v@diagonal_projected_diagonal(w, u))
            np.testing.assert_allclose(gram, np.sum(exact*exact_source), atol=3e-12)
    v = np.linspace(.5, 1.8, len(u))
    np.testing.assert_allclose(diagonal_projected_diagonal(v, u), np.diag(P@np.diag(v)@P), atol=1e-14)


@pytest.mark.parametrize('inverse_scale', [False, True])
def test_projected_equations_recover_exact_binary_pair_moments(inverse_scale):
    targets, u = example()
    kernels = targets.mean(axis=0)
    n = len(u)
    p = np.linspace(.12, .83, n)
    states = np.asarray(list(itertools.product([0., 1.], repeat=n)))
    prob0 = np.prod(np.where(states == 1, p, 1-p), axis=1)
    z = (states-p)/np.sqrt(p*(1-p))
    # A valid non-Gaussian joint distribution with exactly the specified pair
    # moments. This tests the linear PCGC moment model independently of a
    # liability approximation or a Gaussian fourth-moment approximation.
    theta = np.array([.02, .008, .009, .004, -.003, .001])
    genetic = np.einsum('c,cij->ij', theta, kernels)
    i, j = np.triu_indices(n, 1)
    multiplier = 1 + (z[:, i]*z[:, j])@genetic[i, j]
    assert multiplier.min() > 0
    prob = prob0*multiplier
    np.testing.assert_allclose(prob.sum(), 1, atol=1e-14)
    scale = np.linspace(.7, 1.6, n) if inverse_scale else np.ones(n)
    response = z*scale
    kernels = kernels*scale[None, :, None]*scale[None, None, :]
    expected = (response.T*prob)@response
    V = np.diag(scale*scale)
    np.testing.assert_allclose(expected, V+offdiag(np.einsum('c,cij->ij', theta, kernels)), atol=2e-14)
    P = np.eye(n)-u@u.T
    designs = np.asarray([P@offdiag(b)@P for b in kernels])
    H = np.einsum('cij,dij->cd', designs, designs)
    # Known marginal second moments give an analytic projected baseline.
    centered = P@(expected-V)@P
    rhs = np.einsum('cij,ij->c', designs, centered)
    np.testing.assert_allclose(np.linalg.solve(H, rhs), theta, atol=2e-13)
    # Observed diagonal removal before projection gives the same expectation
    # without specifying the diagonal second moments.
    observed = sum(weight*(P@offdiag(np.outer(y, y))@P) for weight, y in zip(prob, response))
    np.testing.assert_allclose(observed, centered, atol=2e-14)
    np.testing.assert_allclose(np.linalg.solve(H, np.einsum('cij,ij->c', designs, observed)), theta, atol=2e-13)
    assert np.linalg.norm(offdiag(P@V@P)) > .1
    naive = np.asarray([P@b@P for b in kernels])
    naive_H = np.einsum('cij,dij->cd', naive, naive)
    naive_rhs = np.einsum('cij,ij->c', naive, centered)
    assert np.linalg.norm(np.linalg.solve(naive_H, naive_rhs)-theta) > 1e-3


def test_null_pair_covariance_and_linear_estimator_efficiency():
    targets, u = example()
    kernels = targets.mean(axis=0)
    n = len(u)
    p = np.linspace(.12, .83, n)
    states = np.asarray(list(itertools.product([0., 1.], repeat=n)))
    prob = np.prod(np.where(states == 1, p, 1-p), axis=1)
    z = (states-p)/np.sqrt(p*(1-p))
    i, j = np.triu_indices(n, 1)
    products = z[:, i]*z[:, j]
    np.testing.assert_allclose(prob@products, 0, atol=1e-14)
    np.testing.assert_allclose((products.T*prob)@products, np.eye(len(i)), atol=2e-14)
    X = kernels[:, i, j].T
    ols = np.linalg.solve(X.T@X, X.T)
    sensitivity = np.linspace(.6, 1.3, n)
    weight = 1/(sensitivity[i]*sensitivity[j])**2
    wls = np.linalg.solve(X.T@(weight[:, None]*X), X.T*weight)
    np.testing.assert_allclose(ols@X, np.eye(X.shape[1]), atol=3e-14)
    np.testing.assert_allclose(wls@X, np.eye(X.shape[1]), atol=3e-14)
    difference = wls@wls.T-ols@ols.T
    np.testing.assert_allclose(difference, (wls-ols)@(wls-ols).T, atol=5e-12)
    assert np.linalg.eigvalsh(difference).min() >= -1e-10
    assert np.trace(difference) > .01
    # The exact null covariance of ordinary PCGC is 2 H^-1 when H sums
    # ordered person pairs. Projection changes the instrument covariance.
    H = 2*X.T@X
    np.testing.assert_allclose(ols@ols.T, 2*np.linalg.inv(H), atol=5e-12)
    P = np.eye(n)-u@u.T
    designs = np.asarray([P@offdiag(b)@P for b in kernels])
    instruments = np.asarray([offdiag(d) for d in designs])
    Hp = np.einsum('cij,dij->cd', designs, designs)
    Lp = 2*np.linalg.solve(Hp, instruments[:, i, j])
    np.testing.assert_allclose(Lp@X, np.eye(X.shape[1]), atol=2e-12)
    Gp = np.einsum('cij,dij->cd', instruments, instruments)
    inverse = np.linalg.inv(Hp)
    np.testing.assert_allclose(Lp@Lp.T, 2*inverse@Gp@inverse.T, rtol=1e-12, atol=1e-10)
    assert np.linalg.eigvalsh(Lp@Lp.T-ols@ols.T).min() >= -1e-9


def test_probit_risk_nuisance_score_and_residual_derivative():
    n = 11
    C = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    beta = np.array([-1.3, .5])
    sampling_ratio = .1*(1-.4)/(.4*(1-.1))
    def probability(b):
        k = ndtr(C@b)
        return k/(k+sampling_ratio*(1-k))
    p = probability(beta)
    k = ndtr(C@beta)
    d = np.exp(-.5*(C@beta)**2)/np.sqrt(2*np.pi)*np.sqrt(p*(1-p))/(k*(1-k))
    J = C*d[:, None]
    y = (np.arange(n)%3 == 0).astype(float)
    z = (y-p)/np.sqrt(p*(1-p))
    step = 1e-5
    def expected_residual(b):
        other = probability(b)
        return (p-other)/np.sqrt(other*(1-other))
    def likelihood(b):
        other = probability(b)
        return np.sum(y*np.log(other)+(1-y)*np.log1p(-other))
    eye = np.eye(len(beta))
    derivative = np.column_stack([(expected_residual(beta+step*v)-expected_residual(beta-step*v))/(2*step) for v in eye])
    score = np.array([(likelihood(beta+step*v)-likelihood(beta-step*v))/(2*step) for v in eye])
    np.testing.assert_allclose(derivative, -J, atol=2e-10)
    np.testing.assert_allclose(score, J.T@z, atol=2e-9)
    def realized_residual(b):
        other = probability(b)
        return (y-other)/np.sqrt(other*(1-other))
    realized_derivative = np.column_stack([(realized_residual(beta+step*v)-realized_residual(beta-step*v))/(2*step) for v in eye])
    T = J*((1-2*p)/(2*np.sqrt(p*(1-p))))[:, None]
    np.testing.assert_allclose(realized_derivative, -J-z[:, None]*T, atol=2e-9)


def test_total_genetic_variance_uses_adjusted_kernel_diagonals():
    rng = np.random.default_rng(21608)
    n, m = 31, 19
    G = rng.normal(size=(n, m))
    G = (G-G.mean(axis=0))/G.std(axis=0)
    C = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    u = np.linalg.qr(C, mode='reduced')[0][:, 1:]
    adjusted = G-u@(u.T@G)
    phi = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    omega = np.array([[.14, .01, -.006], [.01, .02, .004], [-.006, .004, .03]])
    weights = np.linspace(.5, 2.5, n)
    weights /= weights.sum()
    kernel = (adjusted@adjusted.T/m)*(phi@omega@phi.T)
    diagonal = np.sum(adjusted*adjusted, axis=1)/m
    metric = phi.T@((weights*diagonal)[:, None]*phi)
    np.testing.assert_allclose(np.trace(omega@metric), weights@np.diag(kernel), atol=1e-14)
    context_only = phi.T@(weights[:, None]*phi)
    assert abs(np.trace(omega@context_only)-weights@np.diag(kernel)) > 1e-3
