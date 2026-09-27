"""Reuse SUMMIT's descriptor-owned BED/PGEN readers and affine scaling."""
import numpy as np

from summit.context.spec import canonical_sha256
from summit.ldscore.generalized_gxe_pass1 import MatureSequentialGenotypeOperator
from summit.prediction.genotype import StandardizedBlock, native_module
from summit.prediction._validation import indices
from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator, NumpyTNOperator
from .artifacts import make_artifact
from .reference import prepare_moments, population_ld_reference, reference_provenance
from .moments import method_vectors, external_ld_moments


def scaled_file_operator(source, scale, rows, *, block_size=256, threads=1, native=True, variant_indices=None):
    rows = indices(rows, name="binary sample rows", size=len(source.samples))
    if np.any(np.diff(rows) <= 0):
        raise ValueError("binary sample rows must be sorted")
    if scale.variant_identity != source.variants.identity:
        raise ValueError("population genotype scale and source variant/allele axes disagree")
    if scale.provenance.get("population_scale") is not True:
        raise ValueError("PCGC requires a scale explicitly declared to describe the population")
    if len(scale.mean) != len(source.variants.ids):
        raise ValueError("genotype scale and variant counts disagree")
    variants = (np.arange(len(source.variants.ids), dtype=np.int64) if variant_indices is None else
                indices(variant_indices, name="binary variant rows", size=len(source.variants.ids)))
    if not len(variants) or np.any(np.diff(variants) <= 0):
        raise ValueError("binary variant rows must be nonempty, unique and sorted")
    module = native_module() if native else None
    if native:
        from summit.prediction.runtime import configure_prediction_threads
        configure_prediction_threads(module, threads)
    generation = source.generation
    prepared = False
    capacity = block_size
    standardizer = StandardizedBlock(module, threads)
    local_rows = np.arange(len(rows), dtype=np.int64)

    def read_block(start, stop):
        nonlocal generation, prepared
        if source.generation != generation:
            raise RuntimeError("binary genotype source was prepared by another operation")
        if not prepared:
            # The reference planner admits its complete workspace before the
            # decoder allocates any sample-by-block buffers.
            source.prepare(rows, capacity, threads)
            generation, prepared = source.generation, True
        selected = variants[start:stop]
        raw = source.read(selected)
        return np.asfortranarray(standardizer.prepare(raw, local_rows, np.arange(stop-start, dtype=np.int64),
                                                     scale.mean[selected], scale.inverse_scale[selected]))

    def configure_block_width(width):
        nonlocal capacity
        if prepared or type(width) is not int or not 1 <= width <= block_size:
            raise ValueError("configure binary block width before traversal, within requested capacity")
        capacity = width

    operator = MatureSequentialGenotypeOperator(
        num_samples=len(rows), num_variants=len(variants), genotype_format=source.input.format,
        genotype_scale_id=scale.identity, stable_descriptors={f"genotype_input_{i}": fd for i, fd in enumerate(source._fds)},
        read_block=read_block, backend_name="FileGenotypeSource.population_affine64")
    operator.configure_block_width = configure_block_width
    return operator


def prepare_from_source(source, scale, rows, risk, annotations, *, annotation_names,
                        method="pcgc", block_size=256, threads=1, native=True,
                        reference_source=None, variant_indices=None, **options):
    """Prepare aligned scores/reference on one sealed population genotype scale.

    Study-specific PCGC reads the study twice. External LD uses one study-score
    traversal plus exactly two external-reference traversals. Ordinary GWAS
    beta/SE cannot substitute for these raw score moments.
    """
    if len(rows) != risk.n_samples:
        raise ValueError("binary sample rows must align to risks")
    if (method == "pcgc-ld") != (reference_source is not None):
        raise ValueError("pcgc-ld requires an explicit external reference; other methods use the study")
    common = dict(block_size=block_size, threads=threads, native=native)
    operator = scaled_file_operator(source, scale, rows, variant_indices=variant_indices, **common)
    if method == "pcgc-ld":
        if set(source.samples[i] for i in rows) & set(reference_source.samples):
            raise ValueError("external population reference must not overlap the ascertained study")
        if any(options.get(key) is not None for key in ("basis", "coefficients")):
            raise ValueError("basis contraction is unavailable for the unweighted external reference")
        options = {k: v for k, v in options.items() if k not in ("basis", "coefficients")}
        tn = ProtectedTNOperator(threads=threads) if native else NumpyTNOperator(threads=threads)
        _, response = method_vectors(risk, method)
        response = np.asfortranarray(response[:, None])
        response2 = np.asfortranarray(response**2)
        rhs_rows = np.empty(operator.num_variants)
        operator.begin_pass(1)
        for start in range(0, operator.num_variants, block_size):
            stop = min(start+block_size, operator.num_variants)
            x = operator.read_block(start, stop).values
            scores = tn.matmul_tn(x, response)[:, 0]
            diagonal = tn.matmul_tn(np.asfortranarray(x*x), response2)[:, 0]
            rhs_rows[start:stop] = scores**2-diagonal
        operator.finish_pass()
        ref_operator = scaled_file_operator(reference_source, scale, np.arange(len(reference_source.samples)),
                                            variant_indices=variant_indices, **common)
        ld, plan = population_ld_reference(ref_operator, annotations, **common, **options)
        moments = external_ld_moments(rhs_rows, annotations, risk, ld)
        diagnostics = dict(reference_kind="external_population_risk_independent_approximation",
                           study_genotype_passes=1, reference_genotype_passes=ref_operator.observed_passes,
                           reference_samples=ref_operator.num_samples,
                           reference_sample_identity=canonical_sha256({"samples": reference_source.samples}),
                           peak_planned_reference_bytes=plan.peak_resident_bytes,
                           genotype_scale_id=scale.identity,
                           reference_execution=reference_provenance(dict(common, **options)))
    else:
        moments, diagnostics = prepare_moments(operator, annotations, risk, method, **common, **options)
    axis = source.variants if variant_indices is None else source.variants.subset(variant_indices)
    return make_artifact(moments, variant_axis=axis, annotation_names=annotation_names,
                         sample_identity=canonical_sha256({"samples": [source.samples[i] for i in rows]}),
                         genotype_scale_identity=scale.identity, risk=risk, diagnostics=diagnostics)
