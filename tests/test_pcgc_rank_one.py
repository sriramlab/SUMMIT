import itertools
import numpy as np
import pytest

from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.pcgc.moments import BinaryMoments, fit_moments
from summit.pcgc.rank_one import rank_one_reference, plan_pcgc_reference
from summit.pcgc.reference import generalized_reference, contract_reference


@pytest.mark.parametrize('layout', ['partition', 'overlap', 'continuous'])
@pytest.mark.parametrize('probes', [31, 257])
def test_specialization_matches_generalized_with_identical_probes(layout, probes):
    rng = np.random.default_rng(8225)
    x = rng.normal(size=(43, 29))
    if layout == 'partition':
        a = np.eye(3)[np.arange(29) % 3]*rng.uniform(.5, 1., (29, 1))
        a[2] = 0  # Unannotated variants remain valid targets.
    elif layout == 'overlap':
        a = np.column_stack([np.ones(29), np.arange(29) % 2, np.arange(29) > 10])
    else:
        a = rng.uniform(.1, 1., (29, 3))
    w = rng.uniform(.1, 2., 43)
    responses = rng.normal(size=(43, 2))
    options = dict(probes=probes, seed=510, block_size=7, native=False)
    expected, scored, _ = generalized_reference(ArraySequentialGenotypeOperator(x), a, w[:,None],
                           responses=responses, collect_diagonal_rows=True, **options)
    ld, sp = contract_reference(expected, np.ones(1))
    expected_ld = ld-scored.reference_diagonal_rows[:,0]/len(x)**2
    actual = rank_one_reference(ArraySequentialGenotypeOperator(x), a, w, responses=responses, **options)
    np.testing.assert_allclose(actual.ldscores, expected_ld, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(actual.same_person, sp, rtol=2e-13)
    np.testing.assert_allclose(actual.scores, scored.scores, atol=1e-12)
    np.testing.assert_allclose(actual.diagonals, scored.diagonals, atol=1e-12)
    assert actual.diagnostics['two_pass_ledger']['observed_retained_variant_visits'] == 58
    if layout == 'partition':
        assert actual.diagnostics['source_gemm_flops'] == 2*43*28*probes
    for values in (expected_ld, actual.ldscores):
        moments = BinaryMoments(a, values, sp, actual.scores[:,0]**2-actual.diagonals[:,0],43,'pcgc')
        fit = fit_moments(moments, block_ids=np.arange(29) % 4)
        if values is expected_ld:
            old = fit
        else:
            np.testing.assert_allclose(fit['jackknife_replicates'],old['jackknife_replicates'],rtol=2e-11,atol=1e-12)


def test_full_panel_plan_has_one_linear_source_arena_and_bounded_admission():
    for k in (1, 8, 24, 100):
        plan = plan_pcgc_reference(num_samples=300000, num_variants=454000, num_annotations=k,
                                   probes=256, threads=8, memory_bytes=128*2**30)
        assert plan.memory['source_arena'] == 8*300000*k*256
        assert plan.peak_resident_bytes < 128*2**30
        assert plan.descriptor['planned_complete_passes'] == 2
    with pytest.raises(MemoryError, match='source arena'):
        plan_pcgc_reference(num_samples=300000, num_variants=454000, num_annotations=100,
                            probes=256, memory_bytes=2**30)


def test_tighter_budget_balances_tiles_without_changing_probe_count():
    options = dict(num_samples=10000,num_variants=10000,num_annotations=2,probes=64)
    full = plan_pcgc_reference(**options)
    smaller = plan_pcgc_reference(**options, memory_bytes=full.peak_resident_bytes-1)
    assert smaller.tiling != full.tiling
    assert smaller.dimensions['B'] == full.dimensions['B']
    limited = plan_pcgc_reference(num_samples=300000,num_variants=454000,num_annotations=1)
    assert limited.tiling['variant_block_width'] >= 8
    assert limited.peak_resident_bytes <= 2**30


def test_budget_and_overflow_fail_before_genotype_traversal():
    operator = ArraySequentialGenotypeOperator(np.ones((6, 9)))
    with pytest.raises(MemoryError):
        rank_one_reference(operator,np.ones((9,1)),np.ones(6),memory_bytes=1,native=False)
    assert operator.observed_passes == 0
    with pytest.raises(ValueError, match='fourth feature'):
        rank_one_reference(operator,np.ones((9,1)),np.full(6,1e100),native=False)
    assert operator.observed_passes == 0


def test_exact_expected_jackknife_covariance_by_exhaustive_null_outcomes():
    # This verifies the expectation of the SE estimator itself, beyond the
    # earlier test of the point estimator's conditional covariance.
    rng = np.random.default_rng(713)
    n, m, blocks = 6, 12, 3
    x = rng.normal(size=(n,m))
    a = rng.uniform(.1,1.,(m,2))
    p, d = np.linspace(.1,.8,n),np.linspace(.3,1.,n)
    f=x*d[:,None]; f2=f*f
    U=(f.T@f)**2-f2.T@f2
    mass=a.sum(axis=0)
    diagonal=f2@a/mass
    sp=diagonal.T@diagonal
    ids=np.arange(m)*blocks//m
    selectors=[]
    for label in range(blocks):
        target=a.copy(); target[ids==label]=0
        selectors.append(np.linalg.solve((target.T@U@a)/mass[None,:],target.T))
    selectors=np.asarray(selectors); selectors-=selectors.mean(axis=0)
    expected=sum(2*c@U@c.T for c in selectors)*(blocks-1)/blocks
    observed=np.zeros((2,2))
    for bits in itertools.product((0,1),repeat=n):
        y=np.array(bits); v=d*(y-p)/np.sqrt(p*(1-p))
        moments=BinaryMoments(a,U@a/n**2,sp,(x.T@v)**2-(x*x).T@(v*v),n,'pcgc')
        fit=fit_moments(moments,block_ids=ids)
        observed+=np.prod(np.where(y,p,1-p))*np.asarray(fit['conditional_jackknife_covariance'])
    np.testing.assert_allclose(observed,expected,rtol=2e-12,atol=1e-12)
