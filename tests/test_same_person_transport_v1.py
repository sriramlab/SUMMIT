"""Context fourth moments in the strict contextual V1 artifacts and fits."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from summit.context.fit_v1 import (
    assemble_contextual_normal_equations_v1,
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.oracle import (
    SamePersonTransportWarning,
    approximate_study_same_person,
    context_fourth_moments,
)
from summit.context.reference_v1 import (
    adapt_native_contextual_reference_v1,
    load_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.spec import array_sha256, canonical_json, canonical_sha256
from summit.context.trait_v1 import (
    adapt_native_contextual_trait_v1,
    load_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import _native_result as reference_native
from test_context_stage4_fit_v1 import _artifacts, _rewrite_npz, _set_fields
from test_context_stage4_trait_v1 import _native_result as trait_native


def _with_basis(native, publication, basis):
    digest = array_sha256(basis)
    native = dict(native, evaluated_phi_sha256=digest)
    return native, dataclasses.replace(publication, evaluated_phi_sha256=digest)


def test_reference_v1_publishes_basis_checked_fourth_moments(tmp_path: Path) -> None:
    native, publication = reference_native()
    rng = np.random.default_rng(31)
    basis = np.column_stack([np.ones(native["reference_n"]), rng.normal(size=native["reference_n"])])
    native, publication = _with_basis(native, publication, basis)
    artifact = adapt_native_contextual_reference_v1(native, publication, basis=basis)
    np.testing.assert_array_equal(artifact.context_fourth_moments, context_fourth_moments(basis))
    assert not artifact.context_fourth_moments.flags.writeable
    assert artifact.manifest["layouts"]["context_fourth_moments"] == [
        "context_pair",
        "context_pair",
    ]
    artifact.verify()
    path = write_contextual_reference_v1(artifact, tmp_path / "reference")
    loaded = load_contextual_reference_v1(path)
    assert loaded.manifest_sha256 == artifact.manifest_sha256
    np.testing.assert_array_equal(loaded.context_fourth_moments, artifact.context_fourth_moments)
    bridged = loaded.to_development_grouped_reference(
        basis_hash=canonical_sha256({"b": 1}),
        fixed_effect_hash=canonical_sha256({"f": 1}),
        variant_hash=canonical_sha256({"v": 1}),
        annotation_hash=canonical_sha256({"a": 1}),
        loo_grouping_hash=canonical_sha256({"g": 1}),
    )
    np.testing.assert_array_equal(bridged.context_fourth_moments, artifact.context_fourth_moments)
    # Without a basis the artifact is as before: no optional array or layout.
    plain = adapt_native_contextual_reference_v1(native, publication)
    assert plain.context_fourth_moments is None
    assert "context_fourth_moments" not in plain.manifest["arrays"]
    assert "context_fourth_moments" not in plain.manifest["layouts"]
    with pytest.raises(ValueError, match="evaluated_phi_sha256"):
        adapt_native_contextual_reference_v1(native, publication, basis=basis + 1.0)
    # The array, its envelope and its layout must travel together.
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    manifest = json.loads(str(values["manifest_json"].item()))
    del manifest["layouts"]["context_fourth_moments"]
    values["manifest_json"] = np.asarray(canonical_json(manifest))
    values["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    broken = tmp_path / "broken.contextual-reference-v1.npz"
    _rewrite_npz(broken, values)
    with pytest.raises(ValueError, match="layouts mismatch"):
        load_contextual_reference_v1(broken)


def test_trait_v1_publishes_basis_checked_fourth_moments(tmp_path: Path) -> None:
    native, publication = trait_native()
    basis = np.ones((native["study_n"], 1))
    native, publication = _with_basis(native, publication, basis)
    artifact = adapt_native_contextual_trait_v1(native, publication, basis=basis)
    np.testing.assert_array_equal(artifact.context_fourth_moments, [[float(native["study_n"])]])
    path = write_contextual_trait_v1(artifact, tmp_path / "trait")
    loaded = load_contextual_trait_v1(path)
    assert loaded.manifest_sha256 == artifact.manifest_sha256
    np.testing.assert_array_equal(loaded.context_fourth_moments, artifact.context_fourth_moments)
    assert adapt_native_contextual_trait_v1(native, publication).context_fourth_moments is None
    with pytest.raises(ValueError, match="evaluated_phi_sha256"):
        adapt_native_contextual_trait_v1(native, publication, basis=2.0 * basis)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    del values["context_fourth_moments"]
    missing = tmp_path / "missing.contextual-trait-v1.npz"
    _rewrite_npz(missing, values)
    with pytest.raises(ValueError):
        load_contextual_trait_v1(missing)


def test_fit_v1_uses_fourth_moment_difference_and_records_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference, trait = _artifacts(monkeypatch)
    with pytest.warns(SamePersonTransportWarning):
        legacy = assemble_contextual_normal_equations_v1(reference, trait, trait_selector=0)
    assert legacy.same_person_transport == "scaled_legacy"
    # Intercept-only basis: G counts people.  The study value is deliberately
    # off its sample count so the composition term is nonzero.
    _set_fields(reference, context_fourth_moments=np.asarray([[10.0]]))
    _set_fields(trait, context_fourth_moments=np.asarray([[9.0]]))
    equations = assemble_contextual_normal_equations_v1(reference, trait, trait_selector=0)
    assert equations.same_person_transport == "fourth_moment_difference"
    study_same = approximate_study_same_person(
        reference.same_person,
        reference.context_fourth_moments,
        trait.context_fourth_moments,
        reference_n=10,
        study_n=8,
        annotation_count=1,
    )
    np.testing.assert_allclose(study_same, [[0.8 * 0.5 + 0.05 * (9.0 - 8.0)]], rtol=1e-15)
    expected = study_same[0, 0] + 8.0 * 7.0 / (10.0 * 9.0) * (2.0 - 0.5)
    assert equations.transferred_genetic_gram[0, 0] == pytest.approx(expected, rel=1e-14)
    deleted = assemble_contextual_normal_equations_v1(
        reference, trait, trait_selector=0, deleted_groups=("g1",)
    )
    assert deleted.same_person_transport == "fourth_moment_difference"
    fit = fit_contextual_model_v1(reference, trait, trait_selector=0)
    assert fit.same_person_transport == "fourth_moment_difference"
    assert fit.manifest["same_person_transport"] == "fourth_moment_difference"
    path = write_contextual_fit_v1(fit, tmp_path / "fit")
    assert load_contextual_fit_v1(path).same_person_transport == "fourth_moment_difference"
    # A fit written before the field existed has no key and loads as legacy.
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    manifest = json.loads(str(values["manifest_json"].item()))
    del manifest["same_person_transport"]
    values["manifest_json"] = np.asarray(canonical_json(manifest))
    values["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    older = tmp_path / "older.contextual-fit-v1.npz"
    _rewrite_npz(older, values)
    assert load_contextual_fit_v1(older).same_person_transport == "scaled_legacy"
    manifest["same_person_transport"] = "guessed"
    values["manifest_json"] = np.asarray(canonical_json(manifest))
    values["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    _rewrite_npz(older, values)
    with pytest.raises(ValueError, match="same-person transport"):
        load_contextual_fit_v1(older)


def test_fit_v1_composition_correction_and_its_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from summit.context.fit import development_reference_ld_scalars
    from summit.context.oracle import (
        DifferentPersonTransportWarning,
        different_person_correction,
        resolve_different_person_exposure,
        transfer_reference_gram,
    )

    reference, trait = _artifacts(monkeypatch)
    with pytest.warns(DifferentPersonTransportWarning):
        legacy = assemble_contextual_normal_equations_v1(reference, trait, trait_selector=0)
    assert legacy.different_person_transport == "scaled_legacy"
    # Intercept-only basis: G and M count people. A study with exact counts
    # has F_S = s F_R, so the correction vanishes (composition is irrelevant).
    _set_fields(reference, context_fourth_moments=np.asarray([[10.0]]),
                context_second_moments=np.asarray([[10.0]]))
    _set_fields(trait, context_fourth_moments=np.asarray([[8.0]]),
                context_second_moments=np.asarray([[8.0]]))
    exact_counts = assemble_contextual_normal_equations_v1(reference, trait, trait_selector=0)
    assert exact_counts.different_person_transport == "composition_corrected"
    np.testing.assert_array_equal(
        exact_counts.transferred_genetic_gram, legacy.transferred_genetic_gram
    )
    # Off-count moments give a nonzero correction with the documented form.
    _set_fields(trait, context_second_moments=np.asarray([[8.5]]))
    equations = assemble_contextual_normal_equations_v1(
        reference, trait, trait_selector=0, deleted_groups=("g1",)
    )
    exposure, _ = resolve_different_person_exposure(
        reference_n=10, study_n=8,
        reference_second_moments=[[10.0]], reference_fourth_moments=[[10.0]],
        study_second_moments=[[8.5]], study_fourth_moments=[[8.0]],
    )
    scalars, labels = development_reference_ld_scalars(reference)
    retained_masses = np.asarray([4.0])
    correction = different_person_correction(
        scalars, exposure, masses=retained_masses,
        retained=np.asarray([label != "g1" for label in labels]),
    )
    deleted_gram = (20.0 + 28.0) / 16.0
    np.testing.assert_allclose(
        equations.transferred_genetic_gram,
        transfer_reference_gram(
            np.asarray([[deleted_gram]]), np.asarray([[0.5]]), reference_n=10, study_n=8,
            study_same_person=np.asarray([[0.8 * 0.5]]),
            different_person_correction=correction,
        ),
        rtol=1e-14,
    )
    assert np.any(correction)
    fit = fit_contextual_model_v1(reference, trait, trait_selector=0)
    assert fit.different_person_transport == "composition_corrected"
    path = write_contextual_fit_v1(fit, tmp_path / "fit")
    assert load_contextual_fit_v1(path).different_person_transport == "composition_corrected"


def test_reference_and_trait_v1_publish_second_moments(tmp_path: Path) -> None:
    native, publication = reference_native()
    rng = np.random.default_rng(32)
    basis = np.column_stack([np.ones(native["reference_n"]), rng.normal(size=native["reference_n"])])
    native, publication = _with_basis(native, publication, basis)
    artifact = adapt_native_contextual_reference_v1(native, publication, basis=basis)
    np.testing.assert_array_equal(artifact.context_second_moments, basis.T @ basis)
    assert artifact.manifest["layouts"]["context_second_moments"] == ["basis", "basis"]
    loaded = load_contextual_reference_v1(write_contextual_reference_v1(artifact, tmp_path / "r"))
    np.testing.assert_array_equal(loaded.context_second_moments, artifact.context_second_moments)
    trait, trait_publication = trait_native()
    trait_basis = np.ones((trait["study_n"], 1))
    trait, trait_publication = _with_basis(trait, trait_publication, trait_basis)
    trait_artifact = adapt_native_contextual_trait_v1(trait, trait_publication, basis=trait_basis)
    np.testing.assert_array_equal(trait_artifact.context_second_moments, [[float(trait["study_n"])]])
    loaded_trait = load_contextual_trait_v1(write_contextual_trait_v1(trait_artifact, tmp_path / "t"))
    np.testing.assert_array_equal(loaded_trait.context_second_moments, [[float(trait["study_n"])]])
