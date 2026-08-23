"""Independent dense oracle for generalized per-variant GxE LD scores.

This module is test-only.  It intentionally materializes dense projectors,
kernels, all-pairs correlations, and probe panels.  It has no dependency on the
native extension or on the production executor that will implement the same
contract at reference-panel scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray
Pair = tuple[int, int]


@dataclass(frozen=True)
class GeneralizedLDScoreResult:
    pairs: tuple[Pair, ...]
    component_annotation: Array
    component_pair: Array
    directional_ldscores: Array
    directed_numerator: Array
    symmetric_numerator: Array
    gram: Array
    annotation_masses: Array
    residual_rank: int


def pair_order(num_basis: int) -> tuple[Pair, ...]:
    """Return diagonals first, then lexicographic off-diagonals."""
    if num_basis < 1:
        raise ValueError("num_basis must be positive")
    return tuple((q, q) for q in range(num_basis)) + tuple(
        (q, r) for q in range(num_basis) for r in range(q + 1, num_basis)
    )


def orientations(pair: Pair) -> tuple[Pair, ...]:
    q, r = pair
    return ((q, q),) if q == r else ((q, r), (r, q))


def orthonormalize(columns: Array, *, tolerance: float = 1.0e-12) -> Array:
    design = np.asarray(columns, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError("columns must be a matrix")
    if design.shape[1] == 0:
        return np.empty((design.shape[0], 0), dtype=np.float64)
    left, singular, _right = np.linalg.svd(design, full_matrices=False)
    threshold = tolerance * max(1.0, float(singular[0]))
    rank = int(np.sum(singular > threshold))
    if rank != design.shape[1]:
        raise ValueError("fixed-effect design is rank deficient")
    return np.asarray(left[:, :rank], dtype=np.float64)


def projector_matrix(fixed_basis: Array) -> Array:
    basis = np.asarray(fixed_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("fixed_basis must be a matrix")
    return np.eye(basis.shape[0], dtype=np.float64) - basis @ basis.T


def apply_projector(value: Array, fixed_basis: Array) -> Array:
    array = np.asarray(value, dtype=np.float64)
    basis = np.asarray(fixed_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != array.shape[0]:
        raise ValueError("fixed_basis is incompatible with value")
    return array - basis @ (basis.T @ array)


def contextual_features(genotype: Array, basis: Array, fixed_basis: Array) -> Array:
    """Return ``F[q] = P diag(phi_q) G`` with shape ``(Q,N,M)``."""
    genotype_array = np.asarray(genotype, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    if (
        genotype_array.ndim != 2
        or basis_array.ndim != 2
        or basis_array.shape[0] != genotype_array.shape[0]
    ):
        raise ValueError("genotype and basis dimensions disagree")
    return np.stack(
        [
            apply_projector(
                basis_array[:, q, None] * genotype_array,
                fixed_basis,
            )
            for q in range(basis_array.shape[1])
        ],
        axis=0,
    )


def validate_annotations(annotations: Array, n_variants: int) -> tuple[Array, Array]:
    weights = np.asarray(annotations, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != n_variants:
        raise ValueError("annotations must have shape (M,K)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("annotations must be finite and nonnegative")
    masses = np.sum(weights, axis=0, dtype=np.float64)
    if np.any(masses <= 0.0):
        raise ValueError("all annotation masses must be positive")
    return weights, masses


def component_map(num_annotations: int, pairs: Sequence[Pair]) -> tuple[Array, Array]:
    annotations: list[int] = []
    pair_indices: list[int] = []
    for annotation in range(num_annotations):
        for pair_index in range(len(pairs)):
            annotations.append(annotation)
            pair_indices.append(pair_index)
    return (
        np.asarray(annotations, dtype=np.int64),
        np.asarray(pair_indices, dtype=np.int64),
    )


def snp_atoms(features: Array, pairs: Sequence[Pair]) -> Array:
    """Return unnormalized ``B[j,p]`` SNP atoms with shape ``(M,P,N,N)``."""
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.ndim != 3:
        raise ValueError("features must have shape (Q,N,M)")
    pair_tuple = tuple(pairs)
    n_variants = feature_array.shape[2]
    n_samples = feature_array.shape[1]
    atoms = np.empty(
        (n_variants, len(pair_tuple), n_samples, n_samples),
        dtype=np.float64,
    )
    for variant in range(n_variants):
        for pair_index, (q, r) in enumerate(pair_tuple):
            left = feature_array[q, :, variant]
            right = feature_array[r, :, variant]
            atom = np.outer(left, right)
            if q != r:
                atom += np.outer(right, left)
            atoms[variant, pair_index] = atom
    return atoms


def normalized_kernels(
    features: Array,
    annotations: Array,
    pairs: Sequence[Pair],
) -> tuple[Array, Array, Array, Array]:
    """Assemble annotation-major normalized kernels from explicit SNP atoms."""
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.ndim != 3:
        raise ValueError("features must have shape (Q,N,M)")
    weights, masses = validate_annotations(annotations, feature_array.shape[2])
    pair_tuple = tuple(pairs)
    component_annotation, component_pair = component_map(
        weights.shape[1], pair_tuple
    )
    atoms = snp_atoms(feature_array, pair_tuple)
    kernels = np.empty(
        (component_annotation.size, feature_array.shape[1], feature_array.shape[1]),
        dtype=np.float64,
    )
    for component, (annotation, pair_index) in enumerate(
        zip(component_annotation, component_pair, strict=True)
    ):
        kernels[component] = np.einsum(
            "m,mij->ij",
            weights[:, annotation],
            atoms[:, pair_index],
            optimize=True,
        ) / masses[annotation]
    return kernels, masses, component_annotation, component_pair


def dense_kernel_gram(
    genotype: Array,
    basis: Array,
    fixed_basis: Array,
    annotations: Array,
) -> Array:
    features = contextual_features(genotype, basis, fixed_basis)
    kernels, _masses, _component_annotation, _component_pair = normalized_kernels(
        features,
        annotations,
        pair_order(np.asarray(basis).shape[1]),
    )
    return np.einsum("cij,dji->cd", kernels, kernels, optimize=True)


def trait_rhs_and_traces(kernels: Array, phenotype: Array) -> tuple[Array, Array]:
    """Return exact ``y'K_c y`` and ``tr(K_c)`` for every kernel."""
    kernel_array = np.asarray(kernels, dtype=np.float64)
    phenotype_array = np.asarray(phenotype, dtype=np.float64)
    if kernel_array.ndim != 3 or phenotype_array.ndim != 1:
        raise ValueError("kernels must be 3-D and phenotype must be 1-D")
    if kernel_array.shape[1:] != (phenotype_array.size, phenotype_array.size):
        raise ValueError("kernel and phenotype dimensions disagree")
    rhs = np.einsum(
        "i,cij,j->c",
        phenotype_array,
        kernel_array,
        phenotype_array,
        optimize=True,
    )
    traces = np.trace(kernel_array, axis1=1, axis2=2)
    return rhs, traces


def assemble_from_directional(
    directional_ldscores: Array,
    annotations: Array,
    residual_rank: int,
    pairs: Sequence[Pair],
) -> GeneralizedLDScoreResult:
    scores = np.asarray(directional_ldscores, dtype=np.float64)
    pair_tuple = tuple(pairs)
    if scores.ndim != 3 or scores.shape[1] != len(pair_tuple):
        raise ValueError("directional_ldscores has the wrong shape")
    weights, masses = validate_annotations(annotations, scores.shape[0])
    component_annotation, component_pair = component_map(
        weights.shape[1], pair_tuple
    )
    if scores.shape[2] != component_annotation.size:
        raise ValueError("source component axis is inconsistent")
    directed = np.empty(
        (component_annotation.size, component_annotation.size),
        dtype=np.float64,
    )
    for component, (annotation, pair_index) in enumerate(
        zip(component_annotation, component_pair, strict=True)
    ):
        directed[component] = (
            weights[:, annotation] @ scores[:, pair_index, :]
        )
    symmetric = 0.5 * (directed + directed.T)
    denominator = (
        masses[component_annotation, None] * masses[component_annotation][None, :]
    )
    gram = float(residual_rank**2) * symmetric / denominator
    return GeneralizedLDScoreResult(
        pairs=pair_tuple,
        component_annotation=component_annotation,
        component_pair=component_pair,
        directional_ldscores=scores,
        directed_numerator=directed,
        symmetric_numerator=symmetric,
        gram=gram,
        annotation_masses=masses,
        residual_rank=int(residual_rank),
    )


def exact_directional_ldscores(
    features: Array,
    annotations: Array,
    residual_rank: int,
    pairs: Sequence[Pair] | None = None,
) -> GeneralizedLDScoreResult:
    """Compute all target-pair to source-component SNP scores exactly."""
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.ndim != 3:
        raise ValueError("features must have shape (Q,N,M)")
    if residual_rank < 1:
        raise ValueError("residual_rank must be positive")
    q_count, _n_samples, n_variants = feature_array.shape
    pair_tuple = pair_order(q_count) if pairs is None else tuple(pairs)
    weights, _masses = validate_annotations(annotations, n_variants)
    component_annotation, component_pair = component_map(
        weights.shape[1], pair_tuple
    )

    # R[a,b,j,m] = f_aj' f_bm / residual_rank.
    cross = np.empty(
        (q_count, q_count, n_variants, n_variants),
        dtype=np.float64,
    )
    for left in range(q_count):
        for right in range(q_count):
            cross[left, right] = (
                feature_array[left].T @ feature_array[right]
            ) / float(residual_rank)

    scores = np.zeros(
        (n_variants, len(pair_tuple), component_annotation.size),
        dtype=np.float64,
    )
    for target_pair_index, target_pair in enumerate(pair_tuple):
        for source_component, (source_annotation, source_pair_index) in enumerate(
            zip(component_annotation, component_pair, strict=True)
        ):
            source_pair = pair_tuple[int(source_pair_index)]
            value = np.zeros(n_variants, dtype=np.float64)
            for target_left, target_right in orientations(target_pair):
                for source_left, source_right in orientations(source_pair):
                    value += (
                        cross[target_right, source_left]
                        * cross[target_left, source_right]
                    ) @ weights[:, source_annotation]
            scores[:, target_pair_index, source_component] = value
    return assemble_from_directional(
        scores,
        weights,
        residual_rank,
        pair_tuple,
    )


def pass1_sources(
    genotype: Array,
    basis: Array,
    fixed_basis: Array,
    annotations: Array,
    probes: Array,
) -> tuple[Array, Array]:
    """Complete all base and contextual global sources for fixed probes."""
    genotype_array = np.asarray(genotype, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    probe_array = np.asarray(probes, dtype=np.float64)
    if genotype_array.ndim != 2 or basis_array.ndim != 2 or probe_array.ndim != 2:
        raise ValueError("genotype, basis, and probes must be matrices")
    n_samples, n_variants = genotype_array.shape
    if basis_array.shape[0] != n_samples or probe_array.shape[0] != n_variants:
        raise ValueError("input dimensions disagree")
    weights, _masses = validate_annotations(annotations, n_variants)
    base = np.empty(
        (weights.shape[1], n_samples, probe_array.shape[1]),
        dtype=np.float64,
    )
    contextual = np.empty(
        (
            weights.shape[1],
            basis_array.shape[1],
            n_samples,
            probe_array.shape[1],
        ),
        dtype=np.float64,
    )
    for annotation in range(weights.shape[1]):
        base[annotation] = genotype_array @ (
            np.sqrt(weights[:, annotation])[:, None] * probe_array
        )
        for coordinate in range(basis_array.shape[1]):
            contextual[annotation, coordinate] = apply_projector(
                basis_array[:, coordinate, None] * base[annotation],
                fixed_basis,
            )
    return base, contextual


def pass2_cross_sketches(
    genotype: Array,
    basis: Array,
    contextual_sources: Array,
    residual_rank: int,
) -> Array:
    """Return ``U[a,b,k] = G' D_a Y[k,b] / residual_rank``."""
    genotype_array = np.asarray(genotype, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    sources = np.asarray(contextual_sources, dtype=np.float64)
    if residual_rank < 1:
        raise ValueError("residual_rank must be positive")
    if (
        sources.ndim != 4
        or sources.shape[1] != basis_array.shape[1]
        or sources.shape[2] != genotype_array.shape[0]
    ):
        raise ValueError("contextual source dimensions disagree")
    cross = np.empty(
        (
            basis_array.shape[1],
            basis_array.shape[1],
            sources.shape[0],
            genotype_array.shape[1],
            sources.shape[3],
        ),
        dtype=np.float64,
    )
    for target_coordinate in range(basis_array.shape[1]):
        for source_coordinate in range(basis_array.shape[1]):
            for annotation in range(sources.shape[0]):
                cross[target_coordinate, source_coordinate, annotation] = (
                    genotype_array.T
                    @ (
                        basis_array[:, target_coordinate, None]
                        * sources[annotation, source_coordinate]
                    )
                    / float(residual_rank)
                )
    return cross


def randomized_directional_products(
    cross_sketches: Array,
    pairs: Sequence[Pair],
) -> Array:
    """Form fixed-probe row products with shape ``(M,P,K*P)``."""
    cross = np.asarray(cross_sketches, dtype=np.float64)
    pair_tuple = tuple(pairs)
    if cross.ndim != 5 or cross.shape[0] != cross.shape[1]:
        raise ValueError("cross_sketches must have shape (Q,Q,K,M,B)")
    component_annotation, component_pair = component_map(cross.shape[2], pair_tuple)
    scores = np.zeros(
        (cross.shape[3], len(pair_tuple), component_annotation.size),
        dtype=np.float64,
    )
    for target_pair_index, target_pair in enumerate(pair_tuple):
        for source_component, (annotation, source_pair_index) in enumerate(
            zip(component_annotation, component_pair, strict=True)
        ):
            source_pair = pair_tuple[int(source_pair_index)]
            value = np.zeros(cross.shape[3], dtype=np.float64)
            for target_left, target_right in orientations(target_pair):
                for source_left, source_right in orientations(source_pair):
                    value += np.mean(
                        cross[target_right, source_left, annotation]
                        * cross[target_left, source_right, annotation],
                        axis=1,
                        dtype=np.float64,
                    )
            scores[:, target_pair_index, source_component] = value
    return scores


def randomized_two_pass_ldscores(
    genotype: Array,
    basis: Array,
    fixed_basis: Array,
    annotations: Array,
    probes: Array,
) -> tuple[GeneralizedLDScoreResult, Array, Array]:
    """Run the dense fixed-probe two-pass construction."""
    basis_array = np.asarray(basis, dtype=np.float64)
    residual_rank = np.asarray(genotype).shape[0] - np.asarray(fixed_basis).shape[1]
    _base, sources = pass1_sources(
        genotype,
        basis_array,
        fixed_basis,
        annotations,
        probes,
    )
    cross = pass2_cross_sketches(genotype, basis_array, sources, residual_rank)
    pairs = pair_order(basis_array.shape[1])
    scores = randomized_directional_products(cross, pairs)
    return (
        assemble_from_directional(scores, annotations, residual_rank, pairs),
        sources,
        cross,
    )


def block_directed_numerators(
    directional_ldscores: Array,
    annotations: Array,
    block_ids: Array,
    pairs: Sequence[Pair],
) -> tuple[Array, Array]:
    """Accumulate target-row directed numerators and masses by SNP block."""
    scores = np.asarray(directional_ldscores, dtype=np.float64)
    weights, _masses = validate_annotations(annotations, scores.shape[0])
    blocks = np.asarray(block_ids, dtype=np.int64)
    if blocks.shape != (scores.shape[0],) or np.any(blocks < 0):
        raise ValueError("block_ids must be nonnegative and variant-aligned")
    block_count = int(blocks.max(initial=-1)) + 1
    if block_count < 2 or set(np.unique(blocks)) != set(range(block_count)):
        raise ValueError("block IDs must be contiguous and contain at least two blocks")
    component_annotation, component_pair = component_map(weights.shape[1], pairs)
    block_numerators = np.zeros(
        (block_count, component_annotation.size, component_annotation.size),
        dtype=np.float64,
    )
    block_masses = np.zeros((block_count, weights.shape[1]), dtype=np.float64)
    for block in range(block_count):
        selected = blocks == block
        block_masses[block] = np.sum(weights[selected], axis=0, dtype=np.float64)
        for component, (annotation, pair_index) in enumerate(
            zip(component_annotation, component_pair, strict=True)
        ):
            block_numerators[block, component] = (
                weights[selected, annotation]
                @ scores[selected, pair_index, :]
            )
    return block_numerators, block_masses


def frozen_ldscore_delete_block(
    result: GeneralizedLDScoreResult,
    annotations: Array,
    block_ids: Array,
) -> tuple[Array, Array, Array]:
    """Delete target rows while retaining their fixed full-genome LD scores."""
    weights, masses = validate_annotations(
        annotations,
        result.directional_ldscores.shape[0],
    )
    block_numerators, block_masses = block_directed_numerators(
        result.directional_ldscores,
        weights,
        block_ids,
        result.pairs,
    )
    if not np.allclose(
        np.sum(block_numerators, axis=0),
        result.directed_numerator,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "block directed numerators do not reconstruct the full numerator"
        )
    component_annotation = result.component_annotation
    deleted = np.empty_like(block_numerators)
    for block in range(block_numerators.shape[0]):
        retained_mass = masses - block_masses[block]
        if np.any(retained_mass <= 0.0):
            raise ValueError(f"deleting block {block} empties an annotation")
        directed = result.directed_numerator - block_numerators[block]
        symmetric = 0.5 * (directed + directed.T)
        denominator = (
            retained_mass[component_annotation, None]
            * retained_mass[component_annotation][None, :]
        )
        deleted[block] = float(result.residual_rank**2) * symmetric / denominator
    return deleted, block_numerators, block_masses


def exact_recomputed_delete_block_grams(
    genotype: Array,
    basis: Array,
    fixed_basis: Array,
    annotations: Array,
    block_ids: Array,
) -> Array:
    """Literal two-sided deletion oracle; deliberately not the production JK."""
    genotype_array = np.asarray(genotype, dtype=np.float64)
    weights, _masses = validate_annotations(annotations, genotype_array.shape[1])
    blocks = np.asarray(block_ids, dtype=np.int64)
    output = []
    for block in range(int(blocks.max()) + 1):
        retained = blocks != block
        output.append(
            dense_kernel_gram(
                genotype_array[:, retained],
                basis,
                fixed_basis,
                weights[retained],
            )
        )
    return np.asarray(output, dtype=np.float64)


def exact_same_person_matrix(
    genotype: Array,
    basis: Array,
    fixed_basis: Array,
    annotations: Array,
) -> Array:
    features = contextual_features(genotype, basis, fixed_basis)
    kernels, _masses, _component_annotation, _component_pair = normalized_kernels(
        features,
        annotations,
        pair_order(np.asarray(basis).shape[1]),
    )
    diagonals = np.diagonal(kernels, axis1=1, axis2=2)
    return diagonals @ diagonals.T


def same_person_ustatistic(
    contextual_sources: Array,
    annotations: Array,
    pairs: Sequence[Pair],
) -> Array:
    """Signed global variant-probe U-statistic from completed pass-1 sources."""
    sources = np.asarray(contextual_sources, dtype=np.float64)
    if sources.ndim != 4:
        raise ValueError("contextual_sources must have shape (K,Q,N,B)")
    weights, masses = validate_annotations(annotations, np.asarray(annotations).shape[0])
    if (
        sources.shape[0] != weights.shape[1]
        or sources.shape[3] < 2
        or sources.shape[1] < 1
    ):
        raise ValueError("source dimensions or probe count are invalid")
    component_annotation, component_pair = component_map(weights.shape[1], pairs)
    values = np.empty(
        (component_annotation.size, sources.shape[2], sources.shape[3]),
        dtype=np.float64,
    )
    for component, (annotation, pair_index) in enumerate(
        zip(component_annotation, component_pair, strict=True)
    ):
        q, r = pairs[int(pair_index)]
        factor = 1.0 if q == r else 2.0
        values[component] = (
            factor
            * sources[annotation, q]
            * sources[annotation, r]
            / masses[annotation]
        )
    probe_sums = np.sum(values, axis=2, dtype=np.float64)
    same_probe = np.einsum("civ,div->cd", values, values, optimize=True)
    probe_count = sources.shape[3]
    return (probe_sums @ probe_sums.T - same_probe) / float(
        probe_count * (probe_count - 1)
    )


def contiguous_slices(length: int, width: int) -> tuple[slice, ...]:
    if length < 1 or width < 1:
        raise ValueError("length and width must be positive")
    return tuple(
        slice(start, min(length, start + width))
        for start in range(0, length, width)
    )


def randomized_two_pass_ldscores_streaming(
    genotype: Array,
    basis: Array,
    fixed_basis: Array,
    annotations: Array,
    probes: Array,
    block_ids: Array,
    *,
    variant_block_width: int,
    probe_chunk_width: int,
) -> tuple[GeneralizedLDScoreResult, Array, Array, Array, dict[str, int]]:
    """Tiny execution-plan oracle with variant blocks outermost in both passes."""
    genotype_array = np.asarray(genotype, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    probe_array = np.asarray(probes, dtype=np.float64)
    n_samples, n_variants = genotype_array.shape
    weights, _masses = validate_annotations(annotations, n_variants)
    residual_rank = n_samples - np.asarray(fixed_basis).shape[1]
    pairs = pair_order(basis_array.shape[1])
    variant_slices = contiguous_slices(n_variants, variant_block_width)
    probe_slices = contiguous_slices(probe_array.shape[1], probe_chunk_width)

    base_sources = np.zeros(
        (weights.shape[1], n_samples, probe_array.shape[1]),
        dtype=np.float64,
    )
    pass1_visits = 0
    for variants in variant_slices:
        genotype_block = genotype_array[:, variants]
        pass1_visits += variants.stop - variants.start
        for probes_slice in probe_slices:
            for annotation in range(weights.shape[1]):
                base_sources[annotation, :, probes_slice] += genotype_block @ (
                    np.sqrt(weights[variants, annotation])[:, None]
                    * probe_array[variants, probes_slice]
                )

    sources = np.empty(
        (
            weights.shape[1],
            basis_array.shape[1],
            n_samples,
            probe_array.shape[1],
        ),
        dtype=np.float64,
    )
    for annotation in range(weights.shape[1]):
        for coordinate in range(basis_array.shape[1]):
            sources[annotation, coordinate] = apply_projector(
                basis_array[:, coordinate, None] * base_sources[annotation],
                fixed_basis,
            )

    # Hard barrier: every source is complete before the first target block.
    component_annotation, _component_pair = component_map(weights.shape[1], pairs)
    scores = np.zeros(
        (n_variants, len(pairs), component_annotation.size),
        dtype=np.float64,
    )
    pass2_visits = 0
    for variants in variant_slices:
        genotype_block = genotype_array[:, variants]
        block_size = variants.stop - variants.start
        pass2_visits += block_size
        for source_annotation in range(weights.shape[1]):
            accumulated = np.zeros(
                (block_size, len(pairs), len(pairs)),
                dtype=np.float64,
            )
            for probes_slice in probe_slices:
                cross = np.empty(
                    (
                        basis_array.shape[1],
                        basis_array.shape[1],
                        block_size,
                        probes_slice.stop - probes_slice.start,
                    ),
                    dtype=np.float64,
                )
                for target_coordinate in range(basis_array.shape[1]):
                    for source_coordinate in range(basis_array.shape[1]):
                        cross[target_coordinate, source_coordinate] = (
                            genotype_block.T
                            @ (
                                basis_array[:, target_coordinate, None]
                                * sources[
                                    source_annotation,
                                    source_coordinate,
                                    :,
                                    probes_slice,
                                ]
                            )
                            / float(residual_rank)
                        )
                for target_pair_index, target_pair in enumerate(pairs):
                    for source_pair_index, source_pair in enumerate(pairs):
                        for target_left, target_right in orientations(target_pair):
                            for source_left, source_right in orientations(source_pair):
                                accumulated[
                                    :, target_pair_index, source_pair_index
                                ] += np.sum(
                                    cross[target_right, source_left]
                                    * cross[target_left, source_right],
                                    axis=1,
                                    dtype=np.float64,
                                )
            accumulated /= float(probe_array.shape[1])
            components = slice(
                source_annotation * len(pairs),
                (source_annotation + 1) * len(pairs),
            )
            scores[variants, :, components] = accumulated

    result = assemble_from_directional(
        scores,
        weights,
        residual_rank,
        pairs,
    )
    block_numerators, block_masses = block_directed_numerators(
        scores,
        weights,
        block_ids,
        pairs,
    )
    ledger = {
        "planned_reference_genotype_passes": 2,
        "observed_reference_genotype_passes": 2,
        "pass1_variant_visits": pass1_visits,
        "pass2_variant_visits": pass2_visits,
        "observed_retained_variant_visits": pass1_visits + pass2_visits,
        "duplicate_retained_variant_visits": 0,
    }
    return result, sources, block_numerators, block_masses, ledger
