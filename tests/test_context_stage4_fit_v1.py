from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from summit.context.fit import ContextRankError, load_context_fit
from summit.context.fit_v1 import (
    CONTEXTUAL_FIT_V1_MAGIC,
    ContextualFitArtifactV1,
    assemble_contextual_normal_equations_v1,
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    validate_contextual_fit_compatibility_v1,
    write_contextual_fit_v1,
)
from summit.context.reference_v1 import ContextualReferenceArtifactV1
from summit.context.schema import GenotypeScalePlanV1, GenotypeScalePolicy
from summit.context.spec import (
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    ContextPairIndex,
    canonical_json,
    canonical_sha256,
)
from summit.context.trait_v1 import ContextualTraitArtifactV1
from summit.context.summary import load_context_trait_summary


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _set_fields(value: Any, **fields: Any) -> Any:
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    return value


def _artifacts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    annotation_mode: str = "strict_disjoint_binary_v1",
    trait_count: int = 2,
    singular: bool = False,
    negative_rhs: bool = False,
) -> tuple[ContextualReferenceArtifactV1, ContextualTraitArtifactV1]:
    # These are deliberately compact scientific fixtures.  Stage 3 and the
    # trait-V1 tests validate the two source artifact constructors themselves;
    # this file isolates the fit boundary by bypassing those large manifests.
    monkeypatch.setattr(ContextualReferenceArtifactV1, "verify", lambda self: None)
    monkeypatch.setattr(ContextualTraitArtifactV1, "verify", lambda self: None)
    from test_context_stage3_reference_v1 import _native_build_provenance

    scale = GenotypeScalePlanV1(
        policy=GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1,
        retained_variant_order_sha256=_sha("retained-order"),
        allele_orientation="bim_a1_counted_v1",
        allele_coding="plink_bed_snp_major_diploid_hardcall_v1",
        centering_source="provided_v1",
        centering_formula="provided_variant_affine_mean_v1",
        scaling_formula="dosage_minus_mean_times_inverse_scale_v1",
        missing_imputation="sealed_mean_v1",
        ploidy_policy="diploid_v1",
        affine_mean_sha256=_sha("affine-mean"),
        affine_inverse_scale_sha256=_sha("affine-inverse"),
    )
    components = ContextComponentIndex(("annotation",), ContextPairIndex(1))
    groups = ("g0", "g1", "g2")
    identity = {
        "variant_order_allele_sha256": _sha("variant-allele"),
        "retained_variant_order_sha256": scale.retained_variant_order_sha256,
        "basis_specification_sha256": _sha("basis-spec"),
        "basis_calibration_sha256": _sha("basis-calibration"),
        "evaluated_phi_sha256": _sha("evaluated-phi"),
        "missingness_sha256": _sha("missingness"),
        "source_tree_sha256": _sha("source-tree"),
    }
    maps = {
        "annotation_map_sha256": _sha("annotation-map"),
        "group_map_sha256": _sha("group-map"),
    }
    common = {
        "native_api_version": 1,
        "native_backend_version": "plink_bed_descriptor_stream_stage2_v1:2",
        "build_id": "3" * 40,
        "grouped_encoding_version": "grouped_unnormalized_dense_v1",
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "genotype_scale_plan_sha256": scale.digest,
        "execution": {
            "annotation_mode": annotation_mode,
            "numeric_policy": "fp64_v1",
            "build_provenance": _native_build_provenance("3" * 40, _sha("source-tree")),
        },
        "maps": maps,
        "identity": identity,
    }
    reference = _set_fields(
        object.__new__(ContextualReferenceArtifactV1),
        manifest={
            **common,
            "artifact_family": "contextual_reference",
            "logical_schema_version": "contextual_reference_v1",
            "dimensions": {
                "N_reference": 10,
                "M": 6,
                "Q": 1,
                "K": 1,
                "C": 1,
                "J": 3,
            },
        },
        component_index=components,
        scale_plan=scale,
        group_ids=groups,
        gram=np.asarray([[2.0]]),
        same_person=np.asarray([[0.5]]),
        group_gram_unnormalized_num=np.asarray([[[20.0]], [[24.0]], [[28.0]]]),
        annotation_masses=np.asarray([6.0]),
        group_annotation_masses=np.asarray([[2.0], [2.0], [2.0]]),
        group_variant_counts=np.asarray([2, 2, 2], dtype=np.int64),
    )
    trait_ids = tuple(f"trait-{index}" for index in range(trait_count))
    genetic_rhs = np.asarray(
        [
            [
                (-1.0 if negative_rhs else 1.0) * (index + 1)
                for index in range(trait_count)
            ]
        ]
    )
    raw_group_rhs = np.asarray(
        [
            [
                [
                    (-1.0 if negative_rhs else 1.0) * value * (index + 1)
                    for index in range(trait_count)
                ]
            ]
            for value in (1.0, 2.0, 3.0)
        ]
    )
    residual_gram = np.asarray([[0.0 if singular else 1.5]])
    genetic_residual = np.asarray([[0.0 if singular or negative_rhs else 0.1]])
    residual_rhs = (
        np.zeros((1, trait_count))
        if negative_rhs
        else np.asarray([[0.5 + 0.25 * index for index in range(trait_count)]])
    )
    trait_manifest = {
        **common,
        "native_backend_version": "plink_bed_descriptor_stream_trait_v1:2",
        "artifact_family": "contextual_trait",
        "logical_schema_version": "contextual_trait_v1",
        "dimensions": {
            "N_study": 8,
            "M": 6,
            "Q": 1,
            "K": 1,
            "C": 1,
            "J": 3,
        },
        "compatible_reference_identity_sha256": reference.manifest_sha256,
    }
    trait = _set_fields(
        object.__new__(ContextualTraitArtifactV1),
        manifest=trait_manifest,
        component_index=components,
        scale_plan=scale,
        group_ids=groups,
        trait_ids=trait_ids,
        residual_names=("residual",),
        genetic_rhs=genetic_rhs,
        genetic_traces=np.asarray([2.0]),
        genetic_residual=genetic_residual,
        residual_rhs=residual_rhs,
        residual_traces=np.asarray([1.0]),
        residual_gram=residual_gram,
        group_rhs_unnormalized_num=raw_group_rhs,
        group_trace_unnormalized_num=np.asarray([[3.0], [4.0], [5.0]]),
        group_genetic_residual_num=(
            np.zeros((3, 1, 1))
            if singular or negative_rhs
            else np.asarray([[[0.1]], [[0.2]], [[0.3]]])
        ),
        annotation_masses=np.asarray([6.0]),
        group_annotation_masses=np.asarray([[2.0], [2.0], [2.0]]),
        group_variant_counts=np.asarray([2, 2, 2], dtype=np.int64),
    )
    return reference, trait


def _rewrite_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


def _write_strict_compatible_summary_inputs(output_dir: Path, row_source: Path) -> None:
    """Publish a real strict reference/trait pair for the subprocess gate."""
    from test_context_stage3_reference_v1 import _artifact as reference_artifact
    from test_context_stage4_trait_v1 import (
        _execution_evidence,
        _native_result,
        _refresh_trait_scientific_digests,
    )
    from test_context_stage3_reference_v1 import (
        _file_content_identity,
        _native_build_provenance,
    )

    from summit.context.reference_v1 import write_contextual_reference_v1
    from summit.context.trait_v1 import (
        ContextualTraitPublicationIdentityV1,
        adapt_native_contextual_trait_v1,
        write_contextual_trait_v1,
    )

    reference = reference_artifact()
    native, _ = _native_result()
    with np.load(row_source, allow_pickle=False) as archive:
        genotype = np.asarray(archive["genotype"], dtype=np.float64)
        phenotypes = np.asarray(archive["phenotypes"], dtype=np.float64)
    if genotype.shape != (7, reference.n_variants) or phenotypes.shape != (7, 2):
        raise AssertionError("Fresh-process fixture row-level source is malformed.")

    components = reference.component_index
    c = len(components)
    j = len(reference.group_ids)
    l = phenotypes.shape[1]
    h = 1
    # The compact trait numerators depend deterministically on the temporary
    # row-level source.  Their exact values are immaterial to the I/O gate, but
    # the stable adapter verifies that every grouped field reconstructs its
    # corresponding full moment.
    source_seed = int(
        abs(
            float(
                np.dot(genotype.ravel(), np.resize(phenotypes, genotype.shape).ravel())
            )
        )
        * 1.0e6
    ) % (2**32)
    rng = np.random.default_rng(source_seed)
    group_rhs = rng.normal(size=(j, c, l))
    group_traces = np.abs(rng.normal(size=(j, c))) + 1.0
    group_genetic_residual = np.zeros((j, c, h), dtype=np.float64)
    component_annotation = np.asarray(
        [entry.annotation_index for entry in components.entries], dtype=np.int64
    )
    component_masses = reference.annotation_masses[component_annotation]
    native.update(
        {
            "genetic_rhs": np.sum(group_rhs, axis=0) / component_masses[:, None],
            "genetic_traces": np.sum(group_traces, axis=0) / component_masses,
            "genetic_residual": np.zeros((c, h), dtype=np.float64),
            "residual_rhs": np.asarray([[1.0, 2.0]]),
            "residual_traces": np.asarray([1.0]),
            "residual_gram": np.asarray([[10.0]]),
            "group_rhs_unnormalized_num": group_rhs,
            "group_trace_unnormalized_num": group_traces,
            "group_genetic_residual_num": group_genetic_residual,
            "annotation_masses": np.array(reference.annotation_masses, copy=True),
            "group_annotation_masses": np.array(
                reference.group_annotation_masses, copy=True
            ),
            "group_variant_counts": np.array(reference.group_variant_counts, copy=True),
            "study_n": genotype.shape[0],
            "n_variants": reference.n_variants,
            "residual_rank": genotype.shape[0] - 2,
            "q": components.pair_index.num_basis,
            "trait_count": l,
            "residual_component_count": h,
            "pair_q": np.asarray(
                [entry.q for entry in components.pair_index.entries], dtype=np.int64
            ),
            "pair_r": np.asarray(
                [entry.r for entry in components.pair_index.entries], dtype=np.int64
            ),
            "pair_eta": np.asarray(
                [entry.kernel_factor for entry in components.pair_index.entries],
                dtype=np.int64,
            ),
            "component_annotation": component_annotation,
            "component_pair": np.asarray(
                [entry.pair_index for entry in components.entries], dtype=np.int64
            ),
            "annotation_names": list(components.annotation_names),
            "group_names": list(reference.group_ids),
            "trait_names": ["trait-0", "trait-1"],
            "residual_names": ["residual"],
            "annotation_mode": reference.manifest["execution"]["annotation_mode"],
            "annotation_map_sha256": reference.manifest["maps"][
                "annotation_map_sha256"
            ],
            "group_map_sha256": reference.manifest["maps"]["group_map_sha256"],
            "genotype_scale_policy": reference.scale_plan.policy.value,
            "allele_orientation": reference.scale_plan.allele_orientation,
            "allele_coding": reference.scale_plan.allele_coding,
            "centering_source": reference.scale_plan.centering_source,
            "centering_formula": reference.scale_plan.centering_formula,
            "scaling_formula": reference.scale_plan.scaling_formula,
            "missing_imputation": reference.scale_plan.missing_imputation,
            "ploidy_policy": reference.scale_plan.ploidy_policy,
            "retained_variant_order_sha256": (
                reference.scale_plan.retained_variant_order_sha256
            ),
            "affine_mean_sha256": reference.scale_plan.affine_mean_sha256,
            "affine_inverse_scale_sha256": (
                reference.scale_plan.affine_inverse_scale_sha256
            ),
            "scale_plan_sha256": reference.scale_plan.digest,
            "variant_order_allele_sha256": reference.manifest["identity"][
                "variant_order_allele_sha256"
            ],
            "contextual_backend": "plink_bed_descriptor_stream_trait_v1",
            "contextual_backend_version": 2,
            "contextual_build_id": reference.manifest["build_id"],
        }
    )
    (
        native["admission"],
        native["diagnostics"],
        native["telemetry"],
    ) = _execution_evidence(m=reference.n_variants, l=l)
    native["file_content_identity"] = _file_content_identity(reference.n_variants)
    native["build_provenance"] = _native_build_provenance(
        native["contextual_build_id"], native["source_tree_sha256"]
    )
    _refresh_trait_scientific_digests(native)
    publication = ContextualTraitPublicationIdentityV1(
        sample_order_sha256=_sha("fresh-process-study-samples"),
        variant_order_allele_sha256=reference.manifest["identity"][
            "variant_order_allele_sha256"
        ],
        fixed_effect_spec_sha256=_sha("fresh-process-study-fixed-effects"),
        basis_specification_sha256=reference.manifest["identity"][
            "basis_specification_sha256"
        ],
        basis_calibration_sha256=reference.manifest["identity"][
            "basis_calibration_sha256"
        ],
        compatible_reference_identity_sha256=reference.manifest_sha256,
        retained_sample_map_sha256=native["retained_sample_map_sha256"],
        retained_variant_order_sha256=native["retained_variant_order_sha256"],
        fixed_basis_sha256=native["fixed_basis_sha256"],
        evaluated_phi_sha256=native["evaluated_phi_sha256"],
        genotype_scale_plan_sha256=native["scale_plan_sha256"],
        missingness_sha256=native["missingness_sha256"],
        annotation_map_sha256=native["annotation_map_sha256"],
        annotation_names=tuple(native["annotation_names"]),
        group_map_sha256=native["group_map_sha256"],
        group_ids=tuple(native["group_names"]),
        phenotype_batch_sha256=native["phenotype_batch_sha256"],
        residual_basis_sha256=native["residual_basis_sha256"],
        trait_ids=tuple(native["trait_names"]),
        residual_names=tuple(native["residual_names"]),
    )
    trait = adapt_native_contextual_trait_v1(native, publication)
    write_contextual_reference_v1(reference, output_dir / "reference")
    write_contextual_trait_v1(trait, output_dir / "trait")


def test_trait_selector_is_exact_and_required_for_multiple_traits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _artifacts(monkeypatch)
    with pytest.raises(ValueError, match="trait_selector is required"):
        fit_contextual_model_v1(reference, trait)
    with pytest.raises(ValueError, match="exact trait ID"):
        fit_contextual_model_v1(reference, trait, trait_selector=True)
    with pytest.raises(ValueError, match="Unknown"):
        fit_contextual_model_v1(reference, trait, trait_selector="Trait-0")
    by_name = fit_contextual_model_v1(reference, trait, trait_selector="trait-1")
    by_index = fit_contextual_model_v1(reference, trait, trait_selector=1)
    np.testing.assert_array_equal(by_name.raw_coefficients, by_index.raw_coefficients)
    assert by_name.selected_trait_id == "trait-1"
    single_reference, single_trait = _artifacts(monkeypatch, trait_count=1)
    single = fit_contextual_model_v1(single_reference, single_trait)
    assert single.selected_trait_id == "trait-0"


def test_assembly_uses_sample_count_transfer_and_closed_deletions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _artifacts(monkeypatch)
    equations = assemble_contextual_normal_equations_v1(
        reference, trait, trait_selector="trait-0"
    )
    expected_transfer = 8.0 / 10.0 * 0.5 + 8.0 * 7.0 / (10.0 * 9.0) * 1.5
    assert equations.transferred_genetic_gram[0, 0] == pytest.approx(expected_transfer)
    assert equations.matrix.shape == (2, 2)
    assert equations.rhs.shape == (2,)
    deleted = assemble_contextual_normal_equations_v1(
        reference,
        trait,
        trait_selector=0,
        deleted_groups=("g0", "g2"),
    )
    assert deleted.deleted_groups == ("g0", "g2")
    with pytest.raises(ValueError, match="unique"):
        assemble_contextual_normal_equations_v1(
            reference,
            trait,
            trait_selector=0,
            deleted_groups=("g0", "g0"),
        )
    with pytest.raises(ValueError, match="Unknown"):
        assemble_contextual_normal_equations_v1(
            reference,
            trait,
            trait_selector=0,
            deleted_groups=("not-a-group",),
        )


@pytest.mark.parametrize(
    ("mode", "label", "combined_default"),
    [
        ("strict_disjoint_binary_v1", "standalone_covariance", False),
        ("generic_nonnegative_weights_v1", "conditional_contribution", True),
    ],
)
def test_every_group_jackknife_and_all_raw_surfaces_have_interpretation_labels(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    label: str,
    combined_default: bool,
) -> None:
    reference, trait = _artifacts(monkeypatch, annotation_mode=mode)
    fit = fit_contextual_model_v1(
        reference,
        trait,
        trait_selector="trait-0",
        context_grid=np.asarray([[1.0], [2.0], [-1.0]]),
        basis_metric=np.asarray([[1.0]]),
        evaluation_grid_role="non_row_evaluation_grid_v1",
        evaluation_grid_provenance_sha256=_sha("stage4-evaluation-grid"),
        evaluation_grid_trusted_non_row=True,
    )
    assert fit.group_ids == ("g0", "g1", "g2")
    assert fit.raw_loo_coefficients.shape == (3, 2)
    assert fit.raw_covariance_surfaces.shape == (4, 1, 3, 3)
    assert fit.raw_combined_covariance_surfaces.shape == (4, 3, 3)
    np.testing.assert_allclose(
        fit.raw_covariance_surfaces[:, 0],
        fit.raw_combined_covariance_surfaces,
    )
    assert fit.manifest["surfaces"]["replicates"] == [
        "full",
        "delete:g0",
        "delete:g1",
        "delete:g2",
    ]
    assert fit.manifest["interpretation"]["per_annotation_surface_label"] == label
    assert (
        fit.manifest["interpretation"]["combined_total_surface_default"]
        is combined_default
    )
    fit.verify()
    with pytest.raises(ValueError, match="supplied together"):
        fit_contextual_model_v1(
            reference,
            trait,
            trait_selector=0,
            context_grid=np.ones((1, 1)),
        )


def test_raw_rank_failure_and_separately_named_strict_only_psd_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    singular_reference, singular_trait = _artifacts(monkeypatch, singular=True)
    with pytest.raises(ContextRankError):
        fit_contextual_model_v1(
            singular_reference, singular_trait, trait_selector="trait-0"
        )
    overlap_reference, overlap_trait = _artifacts(
        monkeypatch,
        annotation_mode="generic_nonnegative_weights_v1",
        negative_rhs=True,
    )
    with pytest.raises(ValueError, match="overlapping annotations"):
        fit_contextual_model_v1(
            overlap_reference,
            overlap_trait,
            trait_selector=0,
            project_psd=True,
        )
    reference, trait = _artifacts(monkeypatch, negative_rhs=True)
    fit = fit_contextual_model_v1(
        reference,
        trait,
        trait_selector=0,
        project_psd=True,
        context_grid=np.asarray([[1.0], [2.0]]),
        basis_metric=np.asarray([[1.0]]),
        evaluation_grid_role="non_row_evaluation_grid_v1",
        evaluation_grid_provenance_sha256=_sha("stage4-psd-evaluation-grid"),
        evaluation_grid_trusted_non_row=True,
    )
    assert fit.raw_omegas[0, 0, 0] < 0.0
    assert fit.psd_omegas[0, 0, 0] >= -1.0e-12
    assert fit.manifest["optional_interpretation"]["raw_output_replaced"] is False
    raw_before = fit.raw_coefficients.copy()
    path = write_contextual_fit_v1(fit, tmp_path / "negative")
    loaded = load_contextual_fit_v1(path)
    np.testing.assert_array_equal(loaded.raw_coefficients, raw_before)
    np.testing.assert_allclose(loaded.psd_coefficients, fit.psd_coefficients)
    np.testing.assert_allclose(
        loaded.psd_covariance_surfaces, fit.psd_covariance_surfaces
    )


def test_single_file_schema_is_immutable_compact_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference, trait = _artifacts(monkeypatch)
    fit = fit_contextual_model_v1(reference, trait, trait_selector=0)
    path = write_contextual_fit_v1(fit, tmp_path / "fit")
    assert path.name.endswith(".contextual-fit-v1.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive.files) == len(set(archive.files))
        assert set(archive.files) == {
            *fit.manifest["arrays"],
            "manifest_json",
            "manifest_sha256",
        }
        assert not any("sample" in name or "variant" in name for name in archive.files)
    loaded = load_contextual_fit_v1(path)
    assert isinstance(loaded, ContextualFitArtifactV1)
    assert loaded.manifest_sha256 == fit.manifest_sha256
    for name in loaded.manifest["arrays"]:
        value = getattr(loaded, name)
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert all(
        "sample" not in axes and "variant" not in axes
        for axes in loaded.manifest["layouts"].values()
    )
    with pytest.raises(ValueError, match="exact V1 suffix"):
        load_contextual_fit_v1(tmp_path / "fit.context-fit.npz")

    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["raw_coefficients"][0] += 1.0
    corrupt = tmp_path / "corrupt.contextual-fit-v1.npz"
    _rewrite_npz(corrupt, values)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_contextual_fit_v1(corrupt)

    manifest = json.loads(str(values["manifest_json"].item()))
    manifest["magic"] = "SUMMIT_CONTEXTUAL_REFERENCE_V1"
    values["manifest_json"] = np.asarray(canonical_json(manifest))
    values["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    wrong_magic = tmp_path / "wrong-magic.contextual-fit-v1.npz"
    _rewrite_npz(wrong_magic, values)
    with pytest.raises(ValueError, match="magic mismatch"):
        load_contextual_fit_v1(wrong_magic)
    assert CONTEXTUAL_FIT_V1_MAGIC not in wrong_magic.name


def test_fresh_process_fits_from_only_stable_summary_artifacts(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(repository / "tests"))
    )
    row_source = tmp_path / "individual-row-source.npz"
    rng = np.random.default_rng(4_040_404)
    with row_source.open("wb") as handle:
        np.savez_compressed(
            handle,
            genotype=rng.normal(size=(7, 6)),
            phenotypes=rng.normal(size=(7, 2)),
        )
    builder = """
from pathlib import Path
import sys
from test_context_stage4_fit_v1 import _write_strict_compatible_summary_inputs

_write_strict_compatible_summary_inputs(Path(sys.argv[1]), Path(sys.argv[2]))
"""
    built = subprocess.run(
        [sys.executable, "-c", builder, str(tmp_path), str(row_source)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    reference_path = tmp_path / "reference.contextual-reference-v1.npz"
    trait_path = tmp_path / "trait.contextual-trait-v1.npz"
    assert reference_path.is_file() and trait_path.is_file()

    # The process that observed row-level arrays has terminated.  Remove its
    # only individual-level file before starting the summary-only fitter.
    row_source.unlink()
    assert not row_source.exists()
    fit_prefix = tmp_path / "fresh-fit"
    fitter = """
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from summit.context.fit_v1 import (
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.reference_v1 import load_contextual_reference_v1
from summit.context.trait_v1 import load_contextual_trait_v1

reference_path = Path(sys.argv[1]).resolve()
trait_path = Path(sys.argv[2]).resolve()
workspace = Path(sys.argv[4]).resolve()
assert {path.resolve() for path in workspace.glob("*.npz")} == {
    reference_path,
    trait_path,
}
reference = load_contextual_reference_v1(reference_path)
trait = load_contextual_trait_v1(trait_path)
fit = fit_contextual_model_v1(
    reference,
    trait,
    trait_selector="trait-0",
    context_grid=np.asarray([[1.0, 0.0], [1.0, 1.0]]),
    basis_metric=np.eye(2),
    evaluation_grid_role="non_row_evaluation_grid_v1",
    evaluation_grid_provenance_sha256=hashlib.sha256(
        b"stage4-fresh-process-fixed-grid-v1"
    ).hexdigest(),
    evaluation_grid_trusted_non_row=True,
)
fit_path = write_contextual_fit_v1(fit, Path(sys.argv[3]))
loaded = load_contextual_fit_v1(fit_path)
assert loaded.raw_rank == loaded.raw_coefficients.size
assert loaded.raw_loo_coefficients.shape[0] == len(reference.group_ids)
assert loaded.raw_covariance_surfaces.shape[0] == len(reference.group_ids) + 1
assert all(
    "sample" not in axes and "variant" not in axes
    for axes in loaded.manifest["layouts"].values()
)
print(json.dumps({"status": "ok", "fit": str(fit_path)}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            fitter,
            str(reference_path),
            str(trait_path),
            str(fit_prefix),
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip())
    assert result["status"] == "ok"
    fit_path = Path(result["fit"])
    assert fit_path.is_file()

    # Development-family loaders reject stable containers from their suffixes
    # before opening or returning any scientific array.
    with pytest.raises(ValueError, match="Stable contextual trait V1"):
        load_context_trait_summary(trait_path)
    with pytest.raises(ValueError, match="Stable contextual fit V1"):
        load_context_fit(fit_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "build",
        "backend",
        "basis",
        "annotation",
        "variant",
        "group",
        "scale",
        "build_provenance",
        "reference_binding",
    ],
)
def test_cross_artifact_compatibility_rejects_every_stable_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    reference, trait = _artifacts(monkeypatch)
    if mutation == "build":
        trait.manifest["build_id"] = "4" * 40
    elif mutation == "backend":
        trait.manifest[
            "native_backend_version"
        ] = "plink_bed_descriptor_stream_trait_v1:1"
    elif mutation == "basis":
        trait.manifest["identity"] = {
            **trait.manifest["identity"],
            "basis_specification_sha256": _sha("other-basis"),
        }
    elif mutation == "annotation":
        trait.manifest["execution"] = {
            **trait.manifest["execution"],
            "annotation_mode": "generic_nonnegative_weights_v1",
        }
    elif mutation == "variant":
        trait.manifest["identity"] = {
            **trait.manifest["identity"],
            "variant_order_allele_sha256": _sha("other-variant-alleles"),
        }
    elif mutation == "group":
        object.__setattr__(trait, "group_ids", ("g0", "g1", "other"))
    elif mutation == "scale":
        object.__setattr__(
            trait,
            "scale_plan",
            GenotypeScalePlanV1(
                **{
                    **reference.scale_plan.__dict__,
                    "affine_mean_sha256": _sha("other-mean"),
                }
            ),
        )
    elif mutation == "build_provenance":
        provenance = trait.manifest["execution"]["build_provenance"]
        trait.manifest["execution"] = {
            **trait.manifest["execution"],
            "build_provenance": {
                **provenance,
                "effective_optimization": "-O1",
            },
        }
    else:
        trait.manifest["compatible_reference_identity_sha256"] = _sha("other-ref")
    with pytest.raises(ValueError, match="compatibility mismatch"):
        validate_contextual_fit_compatibility_v1(reference, trait)
