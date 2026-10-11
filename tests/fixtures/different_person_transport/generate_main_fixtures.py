"""Write artifacts in the format of SUMMIT 4757960 (fourth moments, no second moments).

Run once with that library on the import path:

    python tests/fixtures/different_person_transport/generate_main_fixtures.py OUTDIR

It reuses the design of ../same_person_transport/generate_legacy_fixtures.py
and writes a summary-mode generalized reference and a generalized trait summary
that carry context fourth moments, a development (v0) reference and trait
summary built by that code (also with fourth moments), and
``main_expected.json`` with the fits it produced: own-study same-person term
from fourth moments, legacy scaled different-person term.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "legacy_generator", HERE.parent / "same_person_transport" / "generate_legacy_fixtures.py"
)
legacy = importlib.util.module_from_spec(spec)
sys.dont_write_bytecode = True
spec.loader.exec_module(legacy)


from summit.context import (  # noqa: E402
    context_fourth_moments,
    fit_context_model,
    write_context_reference,
    write_context_trait_summary,
)
from summit.ldscore.generalized_gxe_fit_v1 import (  # noqa: E402
    fit_generalized_gxe_variant_model_v1,
)
from summit.ldscore.generalized_gxe_reference_v1 import (  # noqa: E402
    build_generalized_gxe_variant_reference_v1,
    write_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_trait_summary import (  # noqa: E402
    write_generalized_gxe_trait_summary,
)


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    design = legacy.legacy_design()
    plain = legacy._generalized_reference(design)
    arrays = {
        name: getattr(plain, name)
        for name in (
            "directed_numerator", "symmetric_numerator", "genetic_gram",
            "block_directed_numerator", "block_annotation_mass", "same_person",
        )
    }
    arrays["context_fourth_moments"] = context_fourth_moments(design["basis"])
    manifest = plain.manifest
    reference = build_generalized_gxe_variant_reference_v1(
        axes=manifest["axes"],
        probe_spec=legacy.GlobalVariantProbeSpec(4417, 0, 8),
        genotype_scale_plan=dict(plain.genotype_scale),
        arrays=arrays,
        pass_ledger=dict(manifest["pass_ledger"]),
        performance_ledger=dict(manifest["performance_ledger"]),
        provenance={"fixture": "main_4757960_same_person_transport"},
        diagnostics=dict(manifest["diagnostics"]),
        mode="summary",
    )
    trait = legacy._generalized_trait(design)
    assert trait.context_fourth_moments is not None
    assert not hasattr(trait, "context_second_moments")
    write_generalized_gxe_variant_reference_v1(reference, output / "main_reference")
    write_generalized_gxe_trait_summary(trait, output / "main_trait")
    fit = fit_generalized_gxe_variant_model_v1(reference, trait)
    context_reference, context_summary = legacy.context_objects(design)
    assert context_reference.context_fourth_moments is not None
    write_context_reference(context_reference, output / "main_context")
    write_context_trait_summary(context_summary, output / "main_context")
    context_fit = fit_context_model(context_reference, context_summary)
    expected = {
        "generator_source": "SUMMIT 4757960 (own-study same-person term, scaled different-person term)",
        "generalized_same_person_transport": fit.manifest["same_person_transport"],
        "generalized_raw_coefficients": [float(x) for x in fit.raw_coefficients],
        "generalized_raw_standard_errors": [float(x) for x in fit.raw_standard_errors],
        "context_same_person_transport": context_fit.manifest["assembly"]["same_person_transport"],
        "context_raw_coefficients": [float(x) for x in context_fit.raw_coefficients],
        "context_standard_errors": [float(x) for x in context_fit.standard_errors],
    }
    (output / "main_expected.json").write_text(json.dumps(expected, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
