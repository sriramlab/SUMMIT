import numpy as np
import pytest

from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
from summit.ldscore.generalized_gxe_trait_summary import generalized_gxe_per_variant_trait_statistics


def inputs():
    rng = np.random.default_rng(16092026)
    n, m = 83, 31
    phi = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    fixed = np.linalg.qr(np.column_stack([phi, rng.normal(size=(n, 2))]))[0]
    residual = np.column_stack([np.ones(n), phi[:, 1:], phi[:, 1:]**2])
    x = rng.normal(size=(m, n))
    x[rng.uniform(size=x.shape) < .1] = 0
    traits = []
    for name, size in [('complete', n), ('few_missing', 71), ('mostly_missing', 24)]:
        idx = np.sort(rng.choice(n, size, replace=False))
        # Rotate the orthonormal basis: covariance spans, not basis columns,
        # must agree between traits.
        q = np.linalg.qr(fixed[idx])[0] @ np.linalg.qr(rng.normal(size=(5, 5)))[0]
        traits.append(dict(name=name, indices=idx, fixed_basis=q, phenotype=rng.normal(size=size)))
    return phi, fixed, residual, x, traits


def test_masked_batch_matches_separate_projected_traits():
    phi, fixed, residual, x, traits = inputs()
    batch = MaskedTraitBatch(basis=phi, fixed_basis=fixed, residual_basis=residual, traits=traits)
    for begin, end in [(0, 7), (7, len(x))]:
        for spec, (name, scores, info, rinfo) in zip(traits, batch.block(x[begin:end])):
            idx = spec['indices']
            oracle = generalized_gxe_per_variant_trait_statistics(genotype=x[begin:end, idx].T,
                basis=phi[idx], fixed_basis=spec['fixed_basis'], residual_basis=residual[idx],
                phenotypes=spec['phenotype'])
            assert name == spec['name']
            for actual, expected in [(scores, oracle.scores[:, :, 0]),
                    (info, oracle.information), (rinfo, oracle.residual_information)]:
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
            common = next(t['common'] for t in batch.traits if t['name'] == name)
            for key in ('residual_rhs', 'residual_traces', 'residual_gram', 'normalized_phenotypes'):
                np.testing.assert_allclose(getattr(common, key), getattr(oracle, key), rtol=2e-11, atol=2e-11)
    assert batch.report['equivalent_cohort_products'] < 2


def test_masked_batch_rejects_wrong_covariate_span_and_duplicate_rows():
    phi, fixed, residual, x, traits = inputs()
    traits[1]['fixed_basis'] = np.linalg.qr(np.random.default_rng(1).normal(size=(71, 5)))[0]
    with pytest.raises(ValueError, match='outside'):
        MaskedTraitBatch(basis=phi, fixed_basis=fixed, residual_basis=residual, traits=traits)
    traits[0]['indices'] = np.zeros(len(phi), dtype=np.uint32)
    with pytest.raises(ValueError, match='indices'):
        MaskedTraitBatch(basis=phi, fixed_basis=fixed, residual_basis=residual, traits=traits[:1])
