import numpy as np
import pytest

from pcgc_oracle import pair_equations
from summit.sumstats.binary import prepare_binary_risk
from summit.pcgc.moments import BinaryMoments, fit_moments, solve, LEGACY_DIAGONAL
from summit.pcgc.reference import prepare_moments, generalized_reference, contract_reference, population_ld_reference
from summit.pcgc.research import exact_moments, exact_external_ld, external_ld_moments
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec


def fixture():
    rng = np.random.default_rng(7361)
    x = rng.normal(size=(23, 17))
    a = rng.uniform(.1, 1., (17, 2))
    risk = prepare_binary_risk(np.arange(23) % 2, .05, population_risk=rng.uniform(.02, .3, 23))
    return x, a, risk


@pytest.mark.parametrize("method", ["pcgc", "pcgc-inverse"])
def test_exact_equations_against_explicit_pairs(method):
    x, a, risk = fixture()
    moments = exact_moments(x, a, risk, method)
    H, b = pair_equations(x, a, risk.z, risk.sensitivity, inverse=method == "pcgc-inverse")
    mh, mb = moments.equations()
    np.testing.assert_allclose(mh, H, rtol=3e-14, atol=1e-12)
    np.testing.assert_allclose(mb, b, rtol=3e-14, atol=1e-12)
    np.testing.assert_allclose(fit_moments(moments)["conditional_components"], np.linalg.solve(H, b), rtol=5e-12)


def test_constant_risk_methods_coincide():
    x, a, old_risk = fixture()
    risk = prepare_binary_risk(old_risk.z > 0, .05)
    fits = [fit_moments(exact_moments(x, a, risk, method))["conditional_components"]
            for method in ("liability", "pcgc", "pcgc-inverse")]
    np.testing.assert_allclose(fits, np.tile(fits[0], (3, 1)), atol=1e-13)
    errors = [fit_moments(exact_moments(x,a,risk,method),block_ids=np.arange(len(a)) % 4)['conditional_standard_errors']
              for method in ('liability','pcgc','pcgc-inverse')]
    np.testing.assert_allclose(errors,np.tile(errors[0],(3,1)),atol=1e-12)
    with pytest.raises(ValueError, match="constant"):
        exact_moments(x, a, old_risk, "liability")


@pytest.mark.parametrize("method", ["pcgc", "pcgc-inverse"])
def test_streamed_fixed_probe_moments_and_two_passes(method):
    x, a, risk = fixture()
    moments, diagnostics = prepare_moments(ArraySequentialGenotypeOperator(x), a, risk, method,
        probes=31, seed=572, block_size=6, native=False)
    d = risk.sensitivity if method == "pcgc" else np.ones(len(x))
    f = d[:, None]*x
    z = GlobalVariantProbeSpec(root_seed=572, probe_offset=0, probe_count=31).generate(np.arange(x.shape[1]))
    cross = f.T @ f
    expected = np.column_stack([np.mean((cross @ (np.sqrt(w)[:, None]*z))**2, axis=1) for w in a.T]) / len(x)**2
    expected -= (f*f).T @ ((f*f) @ a)/len(x)**2
    np.testing.assert_allclose(moments.ldscores, expected, rtol=2e-14)
    exact = exact_moments(x, a, risk, method)
    np.testing.assert_allclose(moments.rhs_rows, exact.rhs_rows, atol=2e-12)
    np.testing.assert_allclose(moments.same_person, exact.same_person, rtol=2e-14)
    assert diagnostics["genotype_passes"] == 2
    assert diagnostics["reference_execution"]["probes"]["probe_count"] == 31
    assert diagnostics["reference_execution"]["backend"] == "numpy_differential"


def test_shared_basis_contracts_exactly_with_common_probes():
    x, a, risk = fixture()
    phi = np.column_stack([np.ones(len(x)), risk.sensitivity])
    c = np.array([.2, .8])
    options = dict(probes=41, seed=827, block_size=5, native=False)
    reference, _, _ = generalized_reference(ArraySequentialGenotypeOperator(x), a, phi, **options)
    direct, _, _ = generalized_reference(ArraySequentialGenotypeOperator(x), a, (phi @ c)[:, None], **options)
    ld, sp = contract_reference(reference, c)
    dl, ds = contract_reference(direct, np.ones(1))
    np.testing.assert_allclose(ld, dl, rtol=2e-14)
    np.testing.assert_allclose(sp, ds, rtol=2e-14)
    with pytest.raises(ValueError, match="span"):
        prepare_moments(ArraySequentialGenotypeOperator(x), a, risk, "pcgc-basis", basis=phi, coefficients=c, **options)
    approx, diag = prepare_moments(ArraySequentialGenotypeOperator(x), a, risk, "pcgc-basis",
                                  basis=phi, coefficients=c, allow_basis_approximation=True, **options)
    assert diag["basis_relative_error"] > 0
    f = (phi @ c)[:, None]*x
    np.testing.assert_allclose(approx.ldscores, dl-(f*f).T @ ((f*f) @ a)/len(x)**2, rtol=2e-14)


def test_exact_basis_checks_tiny_individual_sensitivities_not_just_global_norm():
    risk = prepare_binary_risk([1, 0, 1, 0, 1, 0], .1,
                               population_risk=[1e-200, .1, .1, .1, .1, .1])
    phi = risk.sensitivity[:, None].copy()
    phi[0] += 1e-14
    assert np.linalg.norm(phi[:, 0]-risk.sensitivity)/np.linalg.norm(risk.sensitivity) < 1e-12
    with pytest.raises(ValueError, match="span"):
        prepare_moments(ArraySequentialGenotypeOperator(np.ones((6, 4))), np.ones((4, 1)), risk,
                        "pcgc-basis", basis=phi, coefficients=[1.], native=False)


def test_score_memory_is_admitted_before_large_buffers_are_allocated(monkeypatch):
    import summit.pcgc.rank_one as reference
    x,a,risk=fixture()
    def unexpected_allocation(*args,**kwargs):
        raise AssertionError('score buffers allocated before admission')
    monkeypatch.setattr(reference.np,'zeros',unexpected_allocation)
    with pytest.raises(MemoryError,match='score buffers'):
        prepare_moments(ArraySequentialGenotypeOperator(x),a,risk,native=False,memory_bytes=1)


def test_nonfinite_inverse_score_squares_fail_before_reference_passes():
    risk=prepare_binary_risk([1,0,1,0,1,0],.1,population_risk=[1e-200,.1,.1,.1,.1,.1])
    operator=ArraySequentialGenotypeOperator(np.ones((6,4)))
    with pytest.raises(ValueError,match='diagonal responses'):
        prepare_moments(operator,np.ones((4,1)),risk,'pcgc-inverse',native=False)
    assert operator.observed_passes==0


def test_jackknife_matches_independent_rectangular_sample_pair_equations():
    x, a, risk = fixture()
    moments = exact_moments(x, a, risk)
    ids = np.arange(x.shape[1]) % 4
    fitted = fit_moments(moments, block_ids=ids)
    independent = []
    for block in range(4):
        keep = ids != block
        mass = a[keep].sum(axis=0)
        # Target kernel is deleted, source kernel is the FULL genetic model.
        # Enumerating people pairs independently catches wrong mass scalings,
        # diagonal removal, and erroneous symmetrization of overlapping bins.
        kernels = np.array([(x*col) @ x.T/col.sum() for col in a.T])
        targets = np.array([(x[:, keep]*col) @ x[:, keep].T/col.sum() for col in a[keep].T])
        H, b = np.zeros((2, 2)), np.zeros(2)
        for i in range(len(x)):
            for j in range(i):
                d = risk.sensitivity[i]*risk.sensitivity[j]
                H += 2*d*d*np.outer(targets[:, i, j], kernels[:, i, j])
                b += 2*d*targets[:, i, j]*risk.z[i]*risk.z[j]
        assert np.linalg.norm(H-H.T) > 1e-5
        independent.append(np.linalg.solve(H, b))
    np.testing.assert_allclose(fitted["jackknife_replicates"], independent, atol=1e-12)
    assert fitted["uncertainty_status"] == "estimated"


def test_external_ld_removes_finite_reference_diagonal_before_transfer():
    x, a, risk = fixture()
    ld = exact_external_ld(x, a)
    expected = np.zeros_like(ld)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                pair = x[i]*x[j]
                expected += pair[:, None] * (pair @ a)[None, :]
    np.testing.assert_allclose(ld, expected/(len(x)*(len(x)-1)), atol=2e-14)
    base = exact_moments(x, a, risk)
    transferred = external_ld_moments(base.rhs_rows, a, risk, ld)
    H, _ = transferred.equations()
    population = a.T @ ld
    pair_weight = sum(risk.sensitivity[i]**2*risk.sensitivity[j]**2
                      for i in range(len(x)) for j in range(len(x)) if i != j)
    expected_H = pair_weight * (population+population.T)/2 / np.outer(a.sum(axis=0), a.sum(axis=0))
    np.testing.assert_allclose(H, expected_H, rtol=2e-14)


def test_deletions_preserve_full_genome_parameter_and_arbitrary_block_labels():
    x, a, risk = fixture()
    exact = exact_moments(x, a, risk)
    truth = np.array([.12, .28])
    # Noise-free population score expectations from the FULL source model.
    rows = len(x)**2*exact.ldscores @ (truth/a.sum(axis=0))
    m = BinaryMoments(a, exact.ldscores, exact.same_person, rows, len(x), 'pcgc')
    ids = np.arange(len(a)) % 4
    expected = np.broadcast_to(truth, (4, 2))
    for labels in (ids, 10*ids+3, -ids-1):
        fit = fit_moments(m, block_ids=labels)
        np.testing.assert_allclose(fit['jackknife_replicates'], expected, atol=1e-12)
        assert fit['conditional_total_standard_error'] < 1e-12
    with pytest.raises(ValueError, match='two nonempty'):
        fit_moments(m, block_ids=np.zeros(len(a), dtype=int))


def test_legacy_diagonal_moments_are_point_only():
    x, a, risk = fixture()
    m = exact_moments(x, a, risk)
    f2 = (x*risk.sensitivity[:, None])**2
    old = BinaryMoments(a, m.ldscores+f2.T @ (f2 @ a)/len(x)**2, m.same_person,
                        m.rhs_rows, len(x), 'pcgc', ldscore_contract=LEGACY_DIAGONAL)
    np.testing.assert_allclose(old.equations()[0], m.equations()[0], atol=1e-12)
    with pytest.raises(ValueError, match='regenerate'):
        fit_moments(old, block_ids=np.arange(len(a)) % 4)


def test_exact_null_covariance_by_enumerating_all_binary_phenotypes():
    # Independent non-identical Bernoulli responses, conditional on genotypes
    # and risks: Cov(b)=2H and Cov(theta)=2 H^{-1}. This tests heteroskedastic
    # diagonal removal without a simulation tolerance or fitted nuisance model.
    import itertools
    rng = np.random.default_rng(341)
    x, a = rng.normal(size=(6, 9)), rng.uniform(.1, 1., (9, 2))
    p, d = np.linspace(.08, .83, 6), np.linspace(.2, .9, 6)
    f = d[:,None]*x
    f2 = f*f
    raw = ((f.T @ f)**2) @ a-f2.T @ (f2 @ a)
    sp = (f2 @ a/a.sum(axis=0)).T @ (f2 @ a/a.sum(axis=0))
    values, probability = [], []
    for bits in itertools.product((0, 1), repeat=6):
        y = np.array(bits)
        v = d*(y-p)/np.sqrt(p*(1-p))
        m = BinaryMoments(a, raw/36, sp, (x.T @ v)**2-(x*x).T @ (v*v), 6, 'pcgc')
        H, b = m.equations()
        values.append(np.linalg.solve(H, b))
        probability.append(np.prod(np.where(y, p, 1-p)))
    values, probability = np.asarray(values), np.asarray(probability)
    np.testing.assert_allclose(probability @ values, 0, atol=1e-12)
    np.testing.assert_allclose(values.T @ (probability[:,None]*values), 2*np.linalg.inv(H), rtol=2e-13)


def test_streamed_external_diagonal_correction_and_fixed_probes():
    x, a, _ = fixture()
    operator = ArraySequentialGenotypeOperator(x)
    ld, _ = population_ld_reference(operator, a, probes=59, seed=153, block_size=5, native=False)
    z = GlobalVariantProbeSpec(root_seed=153, probe_offset=0, probe_count=59).generate(np.arange(x.shape[1]))
    cross = x.T @ x
    full = np.column_stack([np.mean((cross @ (np.sqrt(w)[:, None]*z))**2, axis=1) for w in a.T])
    diagonal = (x*x).T @ ((x*x) @ a)
    np.testing.assert_allclose(ld, (full-diagonal)/(len(x)*(len(x)-1)), atol=2e-14)
    assert operator.observed_passes == 2


def test_exact_external_reference_smaller_axis_matches_variant_gram():
    rng = np.random.default_rng(911)
    x, a = rng.normal(size=(17, 53)), rng.uniform(.1, 1., (53, 3))
    a[:,0] = 1
    expected = (((x.T @ x)**2) @ a-(x*x).T @ ((x*x) @ a))/(17*16)
    np.testing.assert_allclose(exact_external_ld(x,a,block_size=7), expected, atol=2e-14)


def test_nonidentifiability_and_indefinite_probe_estimates_fail_explicitly():
    with pytest.raises(ValueError, match="rank"):
        solve(np.ones((2, 2)), np.ones(2))
    with pytest.raises(ValueError, match="nonpositive"):
        solve(-np.eye(2), np.ones(2))


def test_frozen_deletion_is_distinct_from_recomputing_both_kernel_sides():
    x, a, risk = fixture()
    full = exact_moments(x, a, risk)
    retained = np.arange(len(a)) % 4 != 0
    frozen_H, frozen_b = full.equations(retained)
    recomputed_H, recomputed_b = exact_moments(x[:, retained], a[retained], risk).equations()
    np.testing.assert_allclose(frozen_b, recomputed_b, atol=1e-12)
    assert np.linalg.norm(frozen_H-recomputed_H)/np.linalg.norm(recomputed_H) > .01


def test_exact_moments_preserve_allele_orientation_and_case_control_recoding():
    x, a, risk = fixture()
    flipped = x*np.where(np.arange(x.shape[1]) % 2, -1, 1)
    reversed_risk = prepare_binary_risk(risk.z < 0, 1-risk.population_prevalence,
                                        population_risk=1-risk.population_risk)
    for method in ("pcgc", "pcgc-inverse"):
        H, b = exact_moments(x, a, risk, method).equations()
        h2, b2 = exact_moments(flipped, a, reversed_risk, method).equations()
        np.testing.assert_allclose(h2, H, rtol=2e-13, atol=1e-11)
        np.testing.assert_allclose(b2, b, rtol=2e-13, atol=1e-11)


def test_binary_projection_does_not_commute_and_creates_offdiagonal_noise():
    x, _, risk = fixture()
    cov = np.column_stack([np.ones(len(x)), np.linspace(-1, 1, len(x))])
    projection = np.eye(len(x))-cov @ np.linalg.solve(cov.T @ cov, cov.T)
    left = projection @ (risk.sensitivity[:, None]*x)
    right = risk.sensitivity[:, None]*(projection @ x)
    assert np.linalg.norm(left-right)/np.linalg.norm(left) > .01
    noise = projection @ np.diag(1/risk.sensitivity**2) @ projection
    offdiagonal = noise-np.diag(noise.diagonal())
    assert np.linalg.norm(offdiagonal)/np.linalg.norm(noise) > .01
