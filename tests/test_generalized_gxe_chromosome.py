from dataclasses import replace
import os
import numpy as np
import pytest

from test_generalized_gxe_native import _fixture, _reference
from summit.ldscore.generalized_gxe_native import GeneralizedGxENativeBEDExecutor
from summit.ldscore.generalized_gxe_variant import GeneralizedGxEPlanInputs, plan_generalized_gxe_variant_work
from summit.ldscore.generalized_gxe_trait_summary import generalized_gxe_per_variant_trait_statistics
from summit.ldscore.generalized_gxe_chromosome import reduce_chromosome_result, joint_chromosome_equations


@pytest.mark.parametrize('q,k,backend', [(2, 1, 'dense'), (3, 2, 'dense'), (3, 2, 'packed')])
def test_interval_fused_statistics_and_joint_profile(tmp_path, q, k, backend):
    prefix, genotype, phi, fixed, annotations, _, probe = _fixture(
        tmp_path, q_count=q, annotation_count=k, seed=1413)
    rng = np.random.default_rng(17)
    phenotype = rng.normal(size=(len(phi), 2))
    residual = np.column_stack([np.ones(len(phi)), phi[:, 1], phi[:, 1]**2])
    full = generalized_gxe_per_variant_trait_statistics(genotype=genotype, basis=phi,
        fixed_basis=fixed, phenotypes=phenotype, residual_basis=residual)
    chunks, expected_grams, expected_cross, expected_rhs = [], [], [], []
    p = q*(q+1)//2
    masses = annotations.sum(axis=0)
    for chromosome, (start, stop) in enumerate(((0, 5), (5, 12)), 1):
        a = annotations[start:stop]
        inputs = GeneralizedGxEPlanInputs(len(phi), stop-start, q, k, probe.probe_count,
            512*2**20, 'bed', fixed_effect_rank=fixed.shape[1],
            preferred_variant_block_width=3, preferred_rhs_tile_columns=q*q*4,
            component_diagonal_sample_tile_width=4, num_traits=2, num_residual_components=3)
        plan = plan_generalized_gxe_variant_work(inputs)
        descriptors = {suffix: os.open(str(prefix)+suffix, os.O_RDONLY) for suffix in ('.bed', '.bim', '.fam')}
        try:
            result = GeneralizedGxENativeBEDExecutor(stable_descriptors=descriptors,
                row_selection=None, ddof=1, basis=phi, fixed_effect_basis=fixed,
                annotations=a, annotation_names=tuple(f'a{j}' for j in range(k)),
                annotation_masses=np.asarray([np.cumsum(a[:, j], dtype=np.longdouble)[-1] for j in range(k)], float),
                probe_spec=probe, work_plan=plan, threads=1, backend=backend,
                variant_start=start, phenotypes=full.normalized_phenotypes, residual_basis=residual,
                same_person_sample_tile_width=4, publish_component_kernel_diagonal=True).execute()
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)
        np.testing.assert_allclose(result.trait_scores, full.scores[start:stop], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(result.trait_residual_information, full.residual_information[start:stop], rtol=1e-11, atol=1e-11)
        assert result.ledger['observed_retained_variant_visits'] == 2*(stop-start)
        assert result.ledger['physical_variant_start'] == start
        assert result.ledger['physical_variant_stop'] == stop
        assert not result.trait_scores.flags.writeable
        # Independent Python stochastic oracle on exactly the chromosome
        # columns catches accidental full-panel source/target traversal.
        oracle_plan = plan_generalized_gxe_variant_work(replace(inputs, num_traits=0, num_residual_components=0))
        _, oracle = _reference(genotype=genotype[:, start:stop], basis=phi, fixed=fixed,
            annotations=a, probe_spec=probe, plan=oracle_plan, probe_width=4)
        np.testing.assert_allclose(result.directional_ldscores, oracle.directional_ldscores, rtol=1e-10, atol=1e-10)
        chunk = reduce_chromosome_result(result, annotations=a, annotation_names=tuple(f'a{j}' for j in range(k)),
            active_annotations=np.arange(k), block_ids=np.arange(start, stop)//3,
            chromosome=str(chromosome), cohort_identity='same-cohort-and-design', trait_names=('t1', 't2'))
        chunks.append(chunk)
        scale = np.repeat(a.sum(axis=0)/masses, p)
        expected_grams.append(result.genetic_gram*scale[:, None]*scale[None, :])
        expected_cross.append(result.component_kernel_diagonal@residual*scale[:, None])
        rhs = []
        for j in range(k):
            for left, right in result.pair_table:
                rhs.append((1 if left == right else 2)*
                    (a[:, j]@(full.scores[start:stop, left]*full.scores[start:stop, right]))/masses[j])
        expected_rhs.append(np.array(rhs))
    kwargs = dict(residual_gram=full.residual_gram, residual_rhs=full.residual_rhs,
        residual_traces=full.residual_traces, residual_names=('one', 'e', 'e2'))
    equations = joint_chromosome_equations(chunks, **kwargs, expected_chromosomes=(1, 2))
    total_cross = sum(expected_cross); rg = full.residual_gram
    expected_profile = sum(g-b@np.linalg.solve(rg, b.T) for g, b in zip(expected_grams, expected_cross))
    c = k*p
    actual_profile = equations.matrix[:c, :c]-equations.matrix[:c, c:]@np.linalg.solve(rg, equations.matrix[c:, :c])
    np.testing.assert_allclose(actual_profile, expected_profile, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(equations.matrix[:c, c:], total_cross, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(equations.rhs[:c], sum(expected_rhs)[:, 0], rtol=1e-11, atol=1e-11)
    # Target-only deletion is a linear reduction on frozen rows. Validate the
    # global retained mass and asymmetric source/target nuisance correction.
    deleted = joint_chromosome_equations(chunks, **kwargs, deleted_blocks=(1,))
    retained_masses = sum(x.block_masses[x.block_ids != 1].sum(axis=0) for x in chunks)
    inverse = np.repeat(1/retained_masses, p)
    profiled = np.zeros((c, c))
    for x in chunks:
        take = x.block_ids != 1
        d = x.block_directed[take].sum(axis=0)*(x.residual_rank**2)*np.outer(inverse, inverse)
        bt = x.block_genetic_residual[take].sum(axis=0)*inverse[:, None]
        bs = x.block_genetic_residual.sum(axis=0)*inverse[:, None]
        value = d-bt@np.linalg.solve(rg, bs.T)
        profiled += (value+value.T)/2
    actual = deleted.matrix[:c, :c]-deleted.matrix[:c, c:]@np.linalg.solve(rg, deleted.matrix[c:, :c])
    np.testing.assert_allclose(actual, profiled, rtol=1e-11, atol=1e-11)
    with pytest.raises(ValueError, match='duplicate chromosome'):
        joint_chromosome_equations(chunks+chunks[:1], **kwargs)
    with pytest.raises(ValueError, match='incomplete'):
        joint_chromosome_equations(chunks, **kwargs, expected_chromosomes=(1, 2, 3))
    with pytest.raises(ValueError, match='identities'):
        joint_chromosome_equations([chunks[0], replace(chunks[1], cohort_identity='different')], **kwargs)
