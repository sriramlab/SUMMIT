import numpy as np
import pytest

from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec
from summit.pcgc.cross import prepare_pair, QuantitativeResponse, response_vectors, SELECTION_CONTRACT
from summit.sumstats.binary import prepare_binary_risk


@pytest.mark.parametrize("overlap", [0, 5, 12])
@pytest.mark.parametrize("inverse", [False, True])
def test_rectangular_cross_moments_and_overlap_diagonal_with_fixed_probes(overlap, inverse):
    rng = np.random.default_rng(538)
    n1, n2, m = 12, 15, 18
    n = n1+n2-overlap
    i = np.arange(n1)
    j = np.r_[np.arange(overlap), np.arange(n1, n)]
    x = rng.normal(size=(n, m))
    a = rng.uniform(.1, 1., (m, 2))
    left = prepare_binary_risk(np.arange(n1) % 2, .1, population_risk=rng.uniform(.02, .3, n1))
    right = prepare_binary_risk(np.arange(n2) % 2, .2, population_risk=rng.uniform(.02, .3, n2))
    method = "pcgc-inverse" if inverse else "pcgc"
    pair = prepare_pair(ArraySequentialGenotypeOperator(x), a, left, right, i, j,
                        selection_contract=SELECTION_CONTRACT, method=method, probes=71, seed=88, native=False)
    wl, vl = response_vectors(left, method)
    wr, vr = response_vectors(right, method)
    f1, f2 = wl[:, None]*x[i], wr[:, None]*x[j]
    c1, c2 = f1.T @ f1, f2.T @ f2
    probes = GlobalVariantProbeSpec(root_seed=88, probe_offset=0, probe_count=71).generate(np.arange(m))
    ld = np.column_stack([np.mean((c1 @ (np.sqrt(col)[:, None]*probes))*(c2 @ (np.sqrt(col)[:, None]*probes)), axis=1)
                          for col in a.T])/n**2
    overlap_rows, il, jr = np.intersect1d(i, j, return_indices=True)
    sq = x[overlap_rows]**2*(wl[il]*wr[jr])[:, None]
    ld -= sq.T @ (sq @ a)/n**2
    np.testing.assert_allclose(pair.cross.ldscores, ld, rtol=2e-12, atol=2e-12)
    kernels = np.array([(f1*col) @ f2.T/col.sum() for col in a.T])
    sp = np.zeros((len(a.T), len(a.T)))
    rhs = np.zeros(len(a.T))
    r1, r2 = vl/wl, vr/wr
    for l in range(n1):
        for r in range(n2):
            if i[l] == j[r]:
                sp += np.outer(kernels[:, l, r], kernels[:, l, r])
            else:
                rhs += kernels[:, l, r]*r1[l]*r2[r]
    np.testing.assert_allclose(pair.cross.same_person, sp, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(pair.cross.equations()[1], rhs, rtol=2e-12, atol=2e-12)
    assert pair.overlap_count == overlap


def test_binary_quantitative_contract_and_selection_rejection():
    rng = np.random.default_rng(482)
    x = rng.normal(size=(30, 20))
    left = prepare_binary_risk(np.arange(15) % 2, .1)
    right = QuantitativeResponse(rng.normal(size=15))
    with pytest.raises(ValueError, match="selection"):
        prepare_pair(ArraySequentialGenotypeOperator(x), np.ones((20, 1)), left, right,
                      np.arange(15), np.arange(15, 30), selection_contract="joint_eligible_controls")
    pair = prepare_pair(ArraySequentialGenotypeOperator(x), np.ones((20, 1)), left, right,
                         np.arange(15), np.arange(15, 30), selection_contract=SELECTION_CONTRACT, probes=100, native=False)
    expected = (x[:15].T @ (left.sensitivity*left.z))*(x[15:].T @ right.z)
    np.testing.assert_allclose(pair.cross.rhs_rows, expected, atol=1e-12)


def test_native_cross_moments_match_python_and_exact_rectangular_gram():
    pytest.importorskip("summit.gxeldcore")
    from prediction_helpers import prediction_threads
    from summit.pcgc.research import exact_pair
    rng = np.random.default_rng(5282)
    x = rng.normal(size=(65, 60))
    a = rng.uniform(.2, 1., (60, 2))
    i, j = np.arange(40), np.r_[np.arange(15), np.arange(40, 65)]
    left = prepare_binary_risk(np.arange(len(i)) % 2, .1)
    right = prepare_binary_risk(np.arange(len(j)) % 2, .2)
    threads = prediction_threads()
    pairs = [prepare_pair(ArraySequentialGenotypeOperator(x), a, left, right, i, j,
                          selection_contract=SELECTION_CONTRACT, probes=101, seed=928, native=native, threads=threads)
             for native in (False, True)]
    for part in ("left", "right", "cross"):
        for name in ("ldscores", "rhs_rows", "same_person"):
            np.testing.assert_allclose(getattr(getattr(pairs[0], part), name), getattr(getattr(pairs[1], part), name),
                                       rtol=3e-12, atol=1e-10)
    exact = exact_pair(x, a, left, right, i, j)
    H, b = exact.cross.equations()
    kernels = np.array([((x[i]*left.sensitivity[:, None])*col) @
                        (x[j]*right.sensitivity[:, None]).T/col.sum() for col in a.T])
    expected_H, expected_b = np.zeros_like(H), np.zeros_like(b)
    for row1, person1 in enumerate(i):
        for row2, person2 in enumerate(j):
            if person1 != person2:
                v = kernels[:, row1, row2]
                expected_H += np.outer(v, v)
                expected_b += v*left.z[row1]*right.z[row2]
    np.testing.assert_allclose(H, expected_H, atol=2e-11)
    np.testing.assert_allclose(b, expected_b, atol=2e-11)


def test_trait_exchange_and_case_recoding_preserve_cross_contract():
    from summit.pcgc.research import exact_pair
    rng = np.random.default_rng(971)
    x, a = rng.normal(size=(34, 40)), np.ones((40, 1))
    i, j = np.arange(20), np.arange(14, 34)
    left = prepare_binary_risk(np.arange(20) % 2, .1, population_risk=rng.uniform(.02, .2, 20))
    right = prepare_binary_risk(np.arange(20) % 2, .2, population_risk=rng.uniform(.02, .3, 20))
    reverse = prepare_binary_risk(left.z < 0, .9, population_risk=1-left.population_risk)
    for method in ("pcgc", "pcgc-inverse"):
        H, b = exact_pair(x, a, left, right, i, j, method=method).cross.equations()
        h2, b2 = exact_pair(x, a, right, left, j, i, method=method).cross.equations()
        h3, b3 = exact_pair(x, a, reverse, right, i, j, method=method).cross.equations()
        np.testing.assert_allclose(h2, H, rtol=2e-13)
        np.testing.assert_allclose(b2, b, rtol=2e-13)
        np.testing.assert_allclose(h3, H, rtol=2e-13)
        np.testing.assert_allclose(b3, -b, rtol=2e-13)


def test_cross_deletion_against_rectangular_person_pairs_and_marginal_se():
    from summit.pcgc.research import exact_pair
    from summit.pcgc.cross import fit_pair
    rng = np.random.default_rng(3901)
    x, a = rng.normal(size=(17, 20)), rng.uniform(.2, 1., (20, 2))
    i, j = np.arange(10), np.arange(5, 17)
    left = prepare_binary_risk(np.arange(len(i)) % 2, .1, population_risk=np.linspace(.02, .3, len(i)))
    right = prepare_binary_risk(np.arange(len(j)) % 2, .2, population_risk=np.linspace(.04, .4, len(j)))
    pair = exact_pair(x,a,left,right,i,j)
    keep = np.arange(len(a)) % 4 != 0
    f1, f2 = x[i]*left.sensitivity[:,None], x[j]*right.sensitivity[:,None]
    full = np.array([(f1*col) @ f2.T/col.sum() for col in a.T])
    target = np.array([(f1[:,keep]*col) @ f2[:,keep].T/col.sum() for col in a[keep].T])
    H, b = np.zeros((2,2)), np.zeros(2)
    for l, person1 in enumerate(i):
        for r, person2 in enumerate(j):
            if person1 != person2:
                H += np.outer(target[:,l,r],full[:,l,r])
                b += target[:,l,r]*left.z[l]*right.z[r]
    actualH, actualb = pair.cross.equations(keep)
    np.testing.assert_allclose(actualH,H,atol=1e-11)
    np.testing.assert_allclose(actualb,b,atol=1e-11)
    fit = fit_pair(pair,block_ids=np.arange(len(a)) % 4)
    np.testing.assert_allclose(fit['marginal_covariance_standard_error'],
                               fit['conditional_covariance_standard_error']/pair.covariance_scale)
