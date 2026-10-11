"""Write small artifacts in the format used before study same-person terms.

Run once with the pre-fix library (SUMMIT f5d607a) on the import path:

    python tests/fixtures/same_person_transport/generate_legacy_fixtures.py OUTDIR

It writes a summary-mode generalized variant reference, a generalized trait
summary for a context-selected study subset, a development (v0) context
reference and trait summary for the same design, and ``legacy_expected.json``
with the fits the pre-fix code produced.  These artifacts have no context
fourth moments.  tests/test_same_person_transport_artifacts.py loads them
with the current code and checks the fits are unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))

from generalized_gxe_variant_ldscore_oracle import orthonormalize  # noqa: E402
from summit.context import (  # noqa: E402
    ContextComponentIndex,
    ContextPairIndex,
    build_context_reference,
    build_context_trait_summary,
    fit_context_model,
    rank_revealing_projector,
    write_context_reference,
    write_context_trait_summary,
)
from summit.context.spec import canonical_sha256  # noqa: E402
from summit.ldscore.generalized_gxe_fit_v1 import (  # noqa: E402
    fit_generalized_gxe_variant_model_v1,
)
from summit.ldscore.generalized_gxe_pass1 import (  # noqa: E402
    ArraySequentialGenotypeOperator,
    GeneralizedGxEPass1Executor,
    NumpyNNOperator,
)
from summit.ldscore.generalized_gxe_pass2 import (  # noqa: E402
    GeneralizedGxEPass2Executor,
    NumpyTNOperator,
)
from summit.ldscore.generalized_gxe_reference_v1 import (  # noqa: E402
    build_generalized_gxe_variant_reference_v1,
    reduce_generalized_gxe_reference_for_inference,
    serialize_generalized_gxe_inference_axes,
    write_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_trait_summary import (  # noqa: E402
    aggregate_generalized_gxe_trait_statistics,
    generalized_gxe_per_variant_trait_statistics,
    write_generalized_gxe_trait_summary,
)
from summit.ldscore.generalized_gxe_variant import (  # noqa: E402
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


def legacy_design(seed: int = 20261011):
    """Deterministic inputs shared by the generator and the test."""
    rng = np.random.default_rng(seed)
    n, m = 24, 18
    sex = (np.arange(n) % 2).astype(float)
    age = rng.normal(size=n)
    basis = np.column_stack([np.ones(n), sex - 0.5, age])
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    fixed_design = np.column_stack([np.ones(n), sex, age])
    rows = np.flatnonzero((sex > 0) | (age > 0.3))
    phenotype = rng.normal(size=len(rows))
    annotations = np.ones((m, 1))
    block_ids = np.arange(m, dtype=np.int64) % 3
    block_labels = ("b0", "b1", "b2")
    return dict(
        n=n,
        m=m,
        basis=basis,
        genotype=genotype,
        fixed_design=fixed_design,
        rows=rows,
        phenotype=phenotype,
        annotations=annotations,
        block_ids=block_ids,
        block_labels=block_labels,
    )


def _generalized_reference(design):
    genotype = np.asfortranarray(design["genotype"])
    basis = np.asfortranarray(design["basis"])
    fixed = orthonormalize(design["fixed_design"])
    annotations = design["annotations"]
    probe_spec = GlobalVariantProbeSpec(4417, 0, 8)
    plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=basis.shape[1],
            num_annotations=1,
            num_probes=probe_spec.probe_count,
            memory_limit_bytes=256 * 1024**2,
            genotype_format="bed",
            fixed_effect_rank=fixed.shape[1],
            preferred_variant_block_width=5,
            preferred_rhs_tile_columns=3 * basis.shape[1] ** 2,
            rhs_policy="tiled",
        )
    )
    operator = ArraySequentialGenotypeOperator(genotype)
    masses = annotations.sum(axis=0)
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=("all",),
        annotation_masses=masses,
        probe_spec=probe_spec,
        work_plan=plan,
        nn_operator=NumpyNNOperator(),
        annotation_tile_width=1,
        probe_tile_width=4,
        native_probe_module=False,
    ).execute()
    result = GeneralizedGxEPass2Executor(
        pass1_result=pass1,
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=("all",),
        work_plan=plan,
        tn_operator=NumpyTNOperator(),
        probe_tile_width=3,
        component_diagonal_sample_tile_width=6,
    ).execute()
    block_directed, block_masses, error = reduce_generalized_gxe_reference_for_inference(
        directional_ldscores=result.directional_ldscores,
        annotations=annotations,
        variant_block_ids=design["block_ids"],
        block_labels=design["block_labels"],
    )
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=genotype.shape[1],
        num_samples=genotype.shape[0],
        basis_names=("intercept", "sex", "age"),
        fixed_effect_rank=fixed.shape[1],
        annotation_names=("all",),
        annotation_masses=masses,
        variant_block_ids=design["block_ids"],
        block_labels=design["block_labels"],
        residual_component_names=("identity",),
    )
    phases = {name: 0.0 for name in ("pass1", "barrier", "pass2", "finalize")}
    return build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe_spec,
        genotype_scale_plan={
            "genotype_scale_policy": "fixture_standardized_v1",
            "allele_orientation": "fixture_a1_v1",
        },
        arrays={
            "directed_numerator": result.directed_numerator,
            "symmetric_numerator": result.symmetric_numerator,
            "genetic_gram": result.genetic_gram,
            "block_directed_numerator": block_directed,
            "block_annotation_mass": block_masses,
            "same_person": result.same_person,
        },
        pass_ledger=result.ledger.to_dict(),
        performance_ledger={
            "backend": "numpy_fixture",
            "threads": 1,
            "affinity": {},
            "numa_evidence": {},
            "phase_wall_seconds": phases,
            "phase_cpu_seconds": phases,
            "bytes_read": 0,
            "gemm_dimensions": [],
            "peak_rss_bytes": 0,
            "output_bytes": 0,
        },
        provenance={"fixture": "legacy_same_person_transport"},
        diagnostics={"block_reconstruction_error": error},
        mode="summary",
    )


def _generalized_trait(design):
    rows = design["rows"]
    statistics = generalized_gxe_per_variant_trait_statistics(
        genotype=design["genotype"][rows],
        basis=design["basis"][rows],
        fixed_basis=orthonormalize(design["fixed_design"][rows]),
        phenotypes=design["phenotype"][:, None],
        residual_basis=np.ones((len(rows), 1)),
    )
    return aggregate_generalized_gxe_trait_statistics(
        statistics,
        annotations=design["annotations"],
        annotation_names=("all",),
        variant_group_ids=design["block_ids"],
        group_labels=design["block_labels"],
        trait_ids=("trait",),
        residual_names=("identity",),
        n_samples=len(rows),
        retain_per_variant=False,
    )


def context_objects(design):
    """Development (v0) reference on everyone and summary on the study rows."""
    rows = design["rows"]
    components = ContextComponentIndex(("all",), ContextPairIndex(3))
    groups = tuple(f"group:{index % 3}" for index in range(design["m"]))
    common = dict(
        annotations=design["annotations"],
        component_index=components,
        loo_groups=groups,
        basis_hash=canonical_sha256({"basis": "intercept_sex_age"}),
        variant_hash=canonical_sha256({"variants": list(range(design["m"]))}),
        genotype_scaling="pre_scaled_input",
    )
    reference = build_context_reference(
        genotype=design["genotype"],
        basis=design["basis"],
        projector=rank_revealing_projector(design["fixed_design"]),
        fixed_effect_hash=canonical_sha256({"fixed": "reference"}),
        gram_method="exact",
        same_person_method="exact",
        **common,
    )
    summary = build_context_trait_summary(
        genotype=design["genotype"][rows],
        basis=design["basis"][rows],
        projector=rank_revealing_projector(design["fixed_design"][rows]),
        phenotype=design["phenotype"],
        residual_basis=np.ones((len(rows), 1)),
        residual_names=("residual:constant",),
        fixed_effect_hash=canonical_sha256({"fixed": "study"}),
        **common,
    )
    return reference, summary


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    design = legacy_design()
    reference = _generalized_reference(design)
    trait = _generalized_trait(design)
    write_generalized_gxe_variant_reference_v1(reference, output / "legacy_reference")
    write_generalized_gxe_trait_summary(trait, output / "legacy_trait")
    fit = fit_generalized_gxe_variant_model_v1(reference, trait)
    context_reference, context_summary = context_objects(design)
    write_context_reference(context_reference, output / "legacy_context")
    write_context_trait_summary(context_summary, output / "legacy_context")
    context_fit = fit_context_model(context_reference, context_summary)
    expected = {
        "generator_source": "SUMMIT f5d607a (pre study same-person transport)",
        "generalized_raw_coefficients": [float(x) for x in fit.raw_coefficients],
        "generalized_raw_standard_errors": [float(x) for x in fit.raw_standard_errors],
        "context_raw_coefficients": [float(x) for x in context_fit.raw_coefficients],
        "context_standard_errors": [float(x) for x in context_fit.standard_errors],
    }
    (output / "legacy_expected.json").write_text(json.dumps(expected, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
