from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from summit.context.fit import ContextRankError
from summit.context.oracle import transfer_reference_gram
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    canonical_sha256,
)
from summit.context.trait_v1 import (
    load_contextual_trait_v1,
    trait_moments_after_deleting_groups_v1,
    write_contextual_trait_v1,
)
from summit.ldscore.generalized_gxe_fit_v1 import (
    assemble_generalized_gxe_normal_equations_v1,
    fit_generalized_gxe_variant_model_v1,
    validate_generalized_gxe_trait_compatibility_v1,
)
from summit.ldscore.generalized_gxe_reference_v1 import (
    build_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_variant import (
    GlobalVariantProbeSpec,
    TwoPassLedger,
    serialize_generalized_gxe_axes,
)
from test_context_stage4_fit_v1 import _artifacts, _sha
from test_context_stage4_trait_v1 import _artifact as stable_trait_artifact


def _completed_ledger(block_ids: np.ndarray) -> dict[str, int]:
    ledger = TwoPassLedger(len(block_ids))
    boundaries = np.flatnonzero(np.diff(block_ids)) + 1
    stops = [*boundaries.tolist(), len(block_ids)]
    for pass_number in (1, 2):
        ledger.begin_pass(pass_number)
        start = 0
        for stop in stops:
            ledger.record_block(start, stop)
            start = stop
        ledger.finish_pass()
    ledger.validate_clean_completion()
    return ledger.to_dict()


def _matching_reference(trait, *, include_panel: bool = False):
    identity = trait.manifest["identity"]
    maps = trait.manifest["maps"]
    counts = np.asarray(trait.group_variant_counts, dtype=np.int64)
    block_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    n_reference = max(10, trait.n_samples + 2)
    fixed_rank = 2
    residual_rank = n_reference - fixed_rank
    axes = serialize_generalized_gxe_axes(
        num_variants=trait.n_variants,
        variant_digest=identity["variant_order_allele_sha256"],
        retained_variant_digest=identity["retained_variant_order_sha256"],
        num_samples=n_reference,
        sample_digest=canonical_sha256({"reference_samples": n_reference}),
        basis_names=tuple(
            f"basis_{index}"
            for index in range(trait.component_index.pair_index.num_basis)
        ),
        basis_digest=identity["basis_specification_sha256"],
        basis_calibration_digest=identity["basis_calibration_sha256"],
        fixed_effect_digest=identity["fixed_effect_spec_sha256"],
        fixed_effect_rank=fixed_rank,
        annotation_names=trait.component_index.annotation_names,
        annotation_digest=maps["annotation_map_sha256"],
        annotation_masses=trait.annotation_masses,
        variant_block_ids=block_ids,
        block_labels=trait.group_ids,
        jackknife_block_digest=maps["group_map_sha256"],
        residual_component_names=trait.residual_names,
    )
    c = len(trait.component_index)
    masses = np.asarray(trait.annotation_masses, dtype=np.float64)
    component_annotations = np.fromiter(
        (entry.annotation_index for entry in trait.component_index.entries),
        dtype=np.int64,
        count=c,
    )
    component_masses = masses[component_annotations]
    gram = 2.0 * np.eye(c) + 0.05 * np.ones((c, c))
    directed = (
        gram
        * np.outer(component_masses, component_masses)
        / float(residual_rank**2)
    )
    fractions = counts.astype(np.float64) / float(np.sum(counts))
    block_directed = fractions[:, None, None] * directed[None, :, :]
    block_masses = np.asarray(trait.group_annotation_masses, dtype=np.float64)
    deleted = np.empty_like(block_directed)
    for block in range(len(counts)):
        retained_masses = masses - block_masses[block]
        retained_components = retained_masses[component_annotations]
        retained_directed = directed - block_directed[block]
        deleted[block] = (
            float(residual_rank**2)
            * 0.5 * (retained_directed + retained_directed.T)
            / np.outer(retained_components, retained_components)
        )
    arrays = {
        "directed_numerator": directed,
        "symmetric_numerator": directed.copy(),
        "genetic_gram": gram,
        "block_directed_numerator": block_directed,
        "block_annotation_mass": block_masses,
        "deleted_genetic_gram": deleted,
        "same_person": 0.2 * np.eye(c),
    }
    if include_panel:
        p = len(trait.component_index.pair_index)
        arrays["directional_ldscores"] = np.zeros(
            (trait.n_variants, p, c), dtype=np.float64
        )
    phases = {name: 0.0 for name in ("pass1", "barrier", "pass2", "finalize")}
    probe = GlobalVariantProbeSpec(7007, 0, 7)
    diagnostics = {
        "maximum_source_projection_leakage": 0.0,
        "maximum_presymmetry_absolute_error": 0.0,
        "maximum_presymmetry_relative_error": 0.0,
        "block_reconstruction_error": 0.0,
        "same_person_probe_count": probe.probe_count,
        "same_person_cross_tile_finalized": True,
        "minimum_annotation_mass": float(np.min(masses)),
        "minimum_deleted_annotation_mass": float(
            np.min(masses[None, :] - block_masses)
        ),
        "all_values_finite": True,
        "normal_matrix_rank": c,
        "normal_matrix_condition": 1.0,
        "dense_oracle_fixture_version": "stage07_fit_v1",
        "backend_fixed_probe_maximum_error": 0.0,
    }
    return build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe,
        genotype_scale_plan=trait.scale_plan,
        arrays=arrays,
        pass_ledger=_completed_ledger(block_ids),
        performance_ledger={
            "backend": "fixture",
            "threads": 1,
            "affinity": {},
            "numa_evidence": {},
            "phase_wall_seconds": dict(phases),
            "phase_cpu_seconds": dict(phases),
            "bytes_read": 0,
            "gemm_dimensions": [],
            "peak_rss_bytes": 0,
            "output_bytes": sum(value.nbytes for value in arrays.values()),
        },
        provenance={
            "source_commit": "d" * 40,
            "source_tree_sha256": "e" * 64,
            "native_binary_sha256": "f" * 64,
        },
        diagnostics=diagnostics,
    )


def _synthetic_pair(monkeypatch: pytest.MonkeyPatch, *, singular: bool = False):
    _, trait = _artifacts(monkeypatch, trait_count=1, singular=singular)
    trait.manifest["identity"]["fixed_effect_spec_sha256"] = _sha(
        "fixed-effect-spec"
    )
    return _matching_reference(trait), trait


def test_loaded_existing_trait_artifact_binds_to_new_reference(
    tmp_path: Path,
) -> None:
    trait = stable_trait_artifact()
    path = write_contextual_trait_v1(trait, tmp_path / "trait")
    loaded = load_contextual_trait_v1(path)
    reference = _matching_reference(loaded)
    compatibility = validate_generalized_gxe_trait_compatibility_v1(
        reference, loaded
    )
    assert compatibility["reference_kind"] == (
        "summit.generalized_gxe.variant_ldscore_reference"
    )
    assert compatibility["reference_manifest_sha256"] == (
        reference.manifest_sha256
    )


def test_full_fit_matches_direct_dense_symmetric_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    result = fit_generalized_gxe_variant_model_v1(reference, trait)
    expected = np.linalg.solve(
        0.5 * (result.normal_matrix + result.normal_matrix.T),
        result.normal_rhs,
    )
    np.testing.assert_allclose(
        result.raw_coefficients, expected, rtol=2.0e-14, atol=2.0e-14
    )
    assert result.raw_rank == result.normal_matrix.shape[0]
    assert result.manifest["solve"]["ridge"] is False
    assert result.manifest["solve"]["pseudoinverse"] is False


def test_every_deleted_fit_matches_direct_frozen_row_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    result = fit_generalized_gxe_variant_model_v1(reference, trait)
    axes = reference.manifest["axes"]
    component_annotations = np.asarray(
        [entry[0] for entry in axes["components"]["table"]], dtype=np.int64
    )
    full_masses = reference.annotation_masses
    for index, block in enumerate(reference.block_labels):
        retained_masses = full_masses - reference.block_annotation_mass[index]
        directed = (
            reference.directed_numerator
            - reference.block_directed_numerator[index]
        )
        component_masses = retained_masses[component_annotations]
        direct_gram = (
            float(reference.residual_rank**2)
            * 0.5 * (directed + directed.T)
            / np.outer(component_masses, component_masses)
        )
        trait_moments = trait_moments_after_deleting_groups_v1(trait, (block,))
        transferred = transfer_reference_gram(
            direct_gram,
            reference.same_person,
            reference_n=reference.n_samples,
            study_n=trait.n_samples,
        )
        c = len(reference.component_index)
        matrix = np.block(
            [
                [transferred, trait_moments.genetic_residual],
                [
                    trait_moments.genetic_residual.T,
                    trait_moments.residual_gram,
                ],
            ]
        )
        rhs = np.concatenate(
            (trait_moments.genetic_rhs[:, 0], trait_moments.residual_rhs[:, 0])
        )
        expected = np.linalg.solve(0.5 * (matrix + matrix.T), rhs)
        np.testing.assert_allclose(
            result.raw_loo_coefficients[index],
            expected,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        equations = assemble_generalized_gxe_normal_equations_v1(
            reference, trait, deleted_blocks=(block,)
        )
        np.testing.assert_array_equal(
            equations.reference_genetic_gram, direct_gram
        )
    assert result.manifest["same_person_jackknife"] == (
        "reuse_full_same_person_v1"
    )


def test_fit_uses_only_compact_summaries_when_panel_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    assert reference.directional_ldscores is None
    result = fit_generalized_gxe_variant_model_v1(reference, trait)
    assert result.manifest["summary_only_fit"] is True
    assert result.manifest["per_variant_panel_accessed"] is False


def test_fit_amortizes_artifact_verification_across_all_deleted_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    calls = {"reference": 0, "trait": 0}
    reference_verify = type(reference).verify
    trait_verify = type(trait).verify

    def counted_reference_verify(self) -> None:
        calls["reference"] += 1
        reference_verify(self)

    def counted_trait_verify(self) -> None:
        calls["trait"] += 1
        trait_verify(self)

    monkeypatch.setattr(type(reference), "verify", counted_reference_verify)
    monkeypatch.setattr(type(trait), "verify", counted_trait_verify)
    fit_generalized_gxe_variant_model_v1(reference, trait)
    assert calls == {"reference": 2, "trait": 2}


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("identity", "variant_order_allele_sha256"),
        ("identity", "retained_variant_order_sha256"),
        ("identity", "basis_specification_sha256"),
        ("identity", "basis_calibration_sha256"),
        ("identity", "fixed_effect_spec_sha256"),
        ("maps", "annotation_map_sha256"),
        ("maps", "group_map_sha256"),
    ),
)
def test_incompatible_trait_identities_fail_closed(
    monkeypatch: pytest.MonkeyPatch, section: str, field: str
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    trait.manifest[section][field] = canonical_sha256({"mismatch": field})
    with pytest.raises(ValueError, match=field):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)


def test_incompatible_pair_annotation_and_group_axes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(
        trait,
        "group_ids",
        tuple(reversed(trait.group_ids)),
    )
    with pytest.raises(ValueError, match="group_ids"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)

    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(
        trait,
        "component_index",
        ContextComponentIndex(("different",), ContextPairIndex(1)),
    )
    with pytest.raises(ValueError, match="component_map_sha256"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)

    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(trait, "annotation_masses", np.asarray([7.0]))
    with pytest.raises(ValueError, match="annotation_masses"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)


def test_incompatible_common_scale_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(
        trait,
        "scale_plan",
        replace(
            trait.scale_plan,
            affine_inverse_scale_sha256=canonical_sha256(
                {"different": "scale"}
            ),
        ),
    )
    with pytest.raises(ValueError, match="genotype_scale_plan_sha256"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)


def test_raw_rank_deficiency_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch, singular=True)
    with pytest.raises(ContextRankError, match="not identifiable"):
        fit_generalized_gxe_variant_model_v1(reference, trait)
