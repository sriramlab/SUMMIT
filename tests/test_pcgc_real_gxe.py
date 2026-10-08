"""Research adapter differentials; no real participant data required."""
import numpy as np
from scripts.pcgc.real_gxe import AncestryOperator, HEScorer, basis, fit_he
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator, NumpyNNOperator
from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator
from summit.ldscore.generalized_gxe_trait_summary import generalized_gxe_per_variant_trait_statistics
from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec
from prediction_helpers import prediction_threads


def fixture():
    rng = np.random.default_rng(6218)
    n, m = 47, 31
    x = rng.normal(size=(n, m))
    phi = np.column_stack([np.ones(n), rng.normal(size=n), (np.arange(n)%2)*2-1.])
    u = basis(np.column_stack([phi, rng.normal(size=(n, 2))]))
    y = rng.binomial(1, .35, n).astype(float)
    return x, phi, u, y


def test_ancestry_before_context_and_streamed_he_statistics():
    x, phi, u, y = fixture()
    nn, tn = NumpyNNOperator(), NumpyTNOperator()
    upc = u[:, [3, 4]]
    raw = ArraySequentialGenotypeOperator(x)
    op = AncestryOperator(raw, upc, nn, tn)
    d = np.column_stack([np.ones(len(x)), phi[:, 1], phi[:, 2], phi[:, 1]**2, phi[:, 1]*phi[:, 2]])
    scorer = HEScorer(op, phi, u, y, d, tn)
    for iteration in (1, 2):
        scorer.begin_pass(iteration)
        for start in range(0, x.shape[1], 7):
            stop = min(start+7, x.shape[1])
            block = scorer.read_block(start, stop)
            np.testing.assert_allclose(block.values, (x-upc@(upc.T@x))[:, start:stop], atol=1e-13)
        scorer.finish_pass()
    adjusted = x-upc@(upc.T@x)
    # Removing centered PC directions must preserve population-scale SNP
    # mean shifts; an added intercept would silently recenter ascertained G.
    np.testing.assert_allclose(adjusted.mean(axis=0), x.mean(axis=0), atol=1e-13)
    expected = generalized_gxe_per_variant_trait_statistics(genotype=adjusted, basis=phi,
        fixed_basis=u, phenotypes=y, residual_basis=d)
    for field in ('scores', 'information', 'residual_information', 'residual_rhs', 'residual_gram'):
        np.testing.assert_allclose(getattr(scorer.statistics(), field), getattr(expected, field), atol=2e-12)
    assert np.linalg.norm(phi[:, 1, None]*adjusted-(phi[:, 1, None]*x-upc@(upc.T@(phi[:, 1, None]*x)))) > 1


def test_he_reference_and_fit_against_dense_fixed_probe_person_moments():
    from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
    from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
    from summit.prediction.genotype import native_module
    from summit.pcgc.gxe import context_pairs
    x, phi, u, y = fixture()
    n, m = x.shape
    t = prediction_threads()
    nn, tn = ProtectedNNOperator(threads=t, native_module=native_module()), ProtectedTNOperator(threads=t, native_module=native_module())
    op = ArraySequentialGenotypeOperator(x)
    op.configure_block_width = lambda width: None
    B, seed = 1536, 767
    got = fit_he(op, phi, u, y, np.arange(m)*4//m,
        dict(threads=t, probes=B, seed=seed, memory_bytes=3*2**30, block_size=7), nn, tn)
    P = np.eye(n)-u@u.T
    features = [P@(phi[:, q, None]*x) for q in range(3)]
    probes = GlobalVariantProbeSpec(root_seed=seed, probe_offset=0, probe_count=B).generate(np.arange(m))
    ps = context_pairs(3)
    orientation = lambda a, b: [(a, b)] if a == b else [(a, b), (b, a)]
    directed = np.zeros((6, 6))
    kernels = []
    for j, (a, b) in enumerate(ps):
        kernels.append(sum(features[v]@features[w].T for v, w in orientation(a, b))/m)
        for k, (c, d) in enumerate(ps):
            for v, w in orientation(a, b):
                for h, ell in orientation(c, d):
                    directed[j, k] += np.sum(((features[w].T@features[h])@probes)*((features[v].T@features[ell])@probes))/B/m**2
    d = np.column_stack([np.ones(n), phi[:, 1], phi[:, 2], phi[:, 1]**2, phi[:, 1]*phi[:, 2]])
    residuals = [P@np.diag(col)@P for col in d.T]
    matrices = kernels+residuals
    H = np.array([[np.sum(a*b) for a in matrices] for b in matrices])
    H[:6, :6] = (directed+directed.T)/2
    yr = P@y
    yr *= np.sqrt((n-u.shape[1])/(yr@yr))
    rhs = np.array([yr@a@yr for a in matrices])
    np.testing.assert_allclose(got['normal_matrix'], H, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(got['normal_rhs'], rhs, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(got['components'], np.linalg.solve(H, rhs)[:6], rtol=2e-9, atol=2e-9)
