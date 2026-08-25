from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from summit.context.reference_v1 import load_contextual_reference_v1
from summit.context.spec import canonical_json
from summit.ldscore import generalized_gxe_reference_v1 as reference_module
from summit.ldscore.generalized_gxe_reference_v1 import (
    GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX,
    GeneralizedGxEVariantReferenceArtifactV1,
    load_generalized_gxe_variant_reference_v1,
    reference_moments_after_deleting_variant_blocks_v1,
    write_generalized_gxe_variant_reference_v1,
)
from test_generalized_gxe_variant_contracts import _valid_artifact


def _artifact(
    *, omit_panel: bool = False, omit_deleted: bool = False
) -> GeneralizedGxEVariantReferenceArtifactV1:
    manifest, source_arrays = _valid_artifact()
    manifest = copy.deepcopy(manifest)
    arrays = {name: value.copy() for name, value in source_arrays.items()}
    if omit_panel:
        arrays.pop("directional_ldscores")
        manifest["numeric_arrays"].pop("directional_ldscores")
        manifest["per_variant_panel"] = {
            "storage": "omitted",
            "logical_layout": "variant_target_pair_source_component_c",
            "logical_compute_dtype": "float64",
        }
    if omit_deleted:
        arrays.pop("deleted_genetic_gram")
        manifest["numeric_arrays"].pop("deleted_genetic_gram")
    return GeneralizedGxEVariantReferenceArtifactV1(
        manifest=manifest,
        **{
            name: arrays.get(name)
            for name in (
                "directed_numerator",
                "symmetric_numerator",
                "genetic_gram",
                "block_directed_numerator",
                "block_annotation_mass",
                "same_person",
                "deleted_genetic_gram",
                "directional_ldscores",
            )
        },
    )


@pytest.mark.parametrize(
    ("omit_panel", "omit_deleted"),
    ((False, False), (True, False), (True, True)),
)
def test_atomic_reference_roundtrip_owns_immutable_arrays(
    tmp_path: Path, omit_panel: bool, omit_deleted: bool
) -> None:
    artifact = _artifact(omit_panel=omit_panel, omit_deleted=omit_deleted)
    target = write_generalized_gxe_variant_reference_v1(
        artifact, tmp_path / "reference"
    )
    assert target.name.endswith(GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX)
    loaded = load_generalized_gxe_variant_reference_v1(target)
    assert loaded.manifest == artifact.manifest
    assert loaded.manifest["kind"] == (
        "summit.generalized_gxe.variant_ldscore_reference"
    )
    for name in loaded.manifest["numeric_arrays"]:
        observed = getattr(loaded, name)
        np.testing.assert_array_equal(observed, getattr(artifact, name))
        assert observed.flags.c_contiguous
        assert not observed.flags.writeable
        assert observed.flags.owndata
    assert (loaded.directional_ldscores is None) is omit_panel
    assert (loaded.deleted_genetic_gram is None) is omit_deleted


def test_frozen_row_deletions_match_cached_grams_and_reuse_same_person() -> None:
    artifact = _artifact()
    np.testing.assert_allclose(
        np.sum(artifact.block_directed_numerator, axis=0),
        artifact.directed_numerator,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(artifact.block_annotation_mass, axis=0),
        artifact.annotation_masses,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    for index, block in enumerate(artifact.block_labels):
        deleted = reference_moments_after_deleting_variant_blocks_v1(
            artifact, (block,)
        )
        np.testing.assert_allclose(
            deleted.gram,
            artifact.deleted_genetic_gram[index],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(deleted.same_person, artifact.same_person)
    assert artifact.manifest["jackknife"]["source_scores_recomputed"] is False
    assert artifact.manifest["jackknife"]["retained_ldscores_frozen"] is True


def test_optional_panel_does_not_change_compact_deletions() -> None:
    complete = _artifact()
    compact = _artifact(omit_panel=True, omit_deleted=True)
    np.testing.assert_array_equal(complete.genetic_gram, compact.genetic_gram)
    np.testing.assert_array_equal(complete.same_person, compact.same_person)
    for block in complete.block_labels:
        left = reference_moments_after_deleting_variant_blocks_v1(
            complete, (block,)
        )
        right = reference_moments_after_deleting_variant_blocks_v1(
            compact, (block,)
        )
        np.testing.assert_array_equal(left.gram, right.gram)
        np.testing.assert_array_equal(left.same_person, right.same_person)


def test_contextual_and_generalized_loaders_cannot_cross_load(
    tmp_path: Path,
) -> None:
    source = write_generalized_gxe_variant_reference_v1(
        _artifact(), tmp_path / "reference"
    )
    with pytest.raises(ValueError, match="suffix"):
        load_contextual_reference_v1(source)
    contextual_name = tmp_path / "copied.contextual-reference-v1.npz"
    shutil.copyfile(source, contextual_name)
    with pytest.raises(ValueError, match="Contextual reference V1"):
        load_contextual_reference_v1(contextual_name)
    with pytest.raises(ValueError, match="suffix"):
        load_generalized_gxe_variant_reference_v1(contextual_name)


def test_loader_rejects_sample_probe_kind_before_arrays(tmp_path: Path) -> None:
    artifact = _artifact()
    wrong_manifest = json.loads(canonical_json(artifact.manifest))
    wrong_manifest["kind"] = "summit.context.reference.v1"
    target = tmp_path / (
        "wrong" + GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX
    )
    np.savez_compressed(
        target,
        manifest_json=np.asarray(canonical_json(wrong_manifest)),
        **{
            name: getattr(artifact, name)
            for name in artifact.manifest["numeric_arrays"]
        },
    )
    with pytest.raises(ValueError, match="other artifact families"):
        load_generalized_gxe_variant_reference_v1(target)


def test_numeric_values_are_checked_structurally_and_missing_member_is_rejected(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    arrays = {
        name: np.asarray(getattr(artifact, name)).copy()
        for name in artifact.manifest["numeric_arrays"]
    }
    arrays["same_person"][0, 0] += 1.0
    corrupt = tmp_path / (
        "corrupt" + GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX
    )
    np.savez_compressed(
        corrupt,
        manifest_json=np.asarray(canonical_json(artifact.manifest)),
        **arrays,
    )
    loaded = load_generalized_gxe_variant_reference_v1(corrupt)
    np.testing.assert_array_equal(loaded.same_person, arrays["same_person"])

    unexpected = tmp_path / (
        "unexpected" + GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX
    )
    np.savez_compressed(
        unexpected,
        manifest_json=np.asarray(canonical_json(artifact.manifest)),
        obsolete_member=np.asarray(1),
        **arrays,
    )
    with pytest.raises(ValueError, match="container key mismatch"):
        load_generalized_gxe_variant_reference_v1(unexpected)

    arrays.pop("same_person")
    missing = tmp_path / (
        "missing" + GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX
    )
    np.savez_compressed(
        missing,
        manifest_json=np.asarray(canonical_json(artifact.manifest)),
        **arrays,
    )
    with pytest.raises(ValueError, match="key mismatch|missing required member"):
        load_generalized_gxe_variant_reference_v1(missing)


def test_atomic_writer_is_no_replace(tmp_path: Path) -> None:
    artifact = _artifact()
    target = write_generalized_gxe_variant_reference_v1(
        artifact, tmp_path / "reference"
    )
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_generalized_gxe_variant_reference_v1(artifact, target)
    assert target.read_bytes() == original
    load_generalized_gxe_variant_reference_v1(target).verify()
