from __future__ import annotations

from copy import deepcopy
import errno
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
from typing import Any
import warnings
import zipfile

import numpy as np
import pytest

import summit.context.fit_v1 as fit_module
import summit.context._artifact_io as artifact_io
import summit.context.reference_v1 as reference_module
import summit.context.trait_v1 as trait_module
from summit.context.fit import load_context_fit
from summit.context.fit_v1 import (
    CONTEXTUAL_FIT_V1_MAGIC,
    CONTEXTUAL_FIT_V1_SUFFIX,
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.reference import load_context_reference
from summit.context.reference_v1 import (
    CONTEXTUAL_REFERENCE_V1_MAGIC,
    CONTEXTUAL_REFERENCE_V1_SUFFIX,
    adapt_native_contextual_reference_v1,
    load_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.spec import array_sha256, canonical_json, canonical_sha256
from summit.context.summary import load_context_trait_summary
from summit.context.trait_v1 import (
    CONTEXTUAL_TRAIT_V1_MAGIC,
    CONTEXTUAL_TRAIT_V1_SUFFIX,
    ContextualTraitPublicationIdentityV1,
    adapt_native_contextual_trait_v1,
    load_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import _artifact as _reference_artifact
from test_context_stage4_fit_v1 import _artifacts as _fit_source_artifacts
from test_context_stage4_trait_v1 import _artifact as _trait_artifact


def _npy_bytes(value: Any, *, allow_pickle: bool = False) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=allow_pickle)
    return output.getvalue()


def _zip_entries(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path, mode="r") as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _rewrite_zip(
    source: Path,
    target: Path,
    *,
    replacements: dict[str, bytes | None] | None = None,
    extras: tuple[tuple[str, bytes], ...] = (),
    duplicate: str | None = None,
) -> None:
    replacements = {} if replacements is None else replacements
    entries = _zip_entries(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(
            target, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            duplicate_payload = None
            for name, payload in entries:
                payload = replacements.get(name, payload)
                if payload is not None:
                    archive.writestr(name, payload)
                    if name == duplicate:
                        duplicate_payload = payload
            for name, payload in extras:
                archive.writestr(name, payload)
            if duplicate is not None:
                assert duplicate_payload is not None
                archive.writestr(duplicate, duplicate_payload)


def _rewrite_canonical_zip(
    source: Path,
    target: Path,
    *,
    replacements: dict[str, bytes] | None = None,
    first_member_extra: bytes = b"",
) -> None:
    replacements = {} if replacements is None else replacements
    with zipfile.ZipFile(
        target, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for index, (name, original) in enumerate(_zip_entries(source)):
            payload = replacements.get(name, original)
            member: str | zipfile.ZipInfo = name
            if index == 0 and first_member_extra:
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.extra = first_member_extra
                member = info
            with archive.open(member, mode="w", force_zip64=True) as stream:
                stream.write(payload)


def _npy_with_hidden_header_comment(payload: bytes) -> bytes:
    value = bytearray(payload)
    assert value[:8] == b"\x93NUMPY\x01\x00"
    header_length = struct.unpack_from("<H", value, 8)[0]
    begin = 10
    end = begin + header_length
    header = bytes(value[begin:end])
    closing = header.index(b"}") + 1
    marker = b" #hidden-header-payload"
    assert header[-1:] == b"\n" and set(header[closing:-1]) <= {0x20}
    assert closing + len(marker) < len(header)
    value[begin:end] = (
        header[:closing]
        + marker
        + b" " * (len(header) - closing - len(marker) - 1)
        + b"\n"
    )
    return bytes(value)


def _insert_gap_before_central_directory(payload: bytes) -> bytes:
    value = bytearray(payload)
    eocd = len(value) - 22
    assert value[eocd : eocd + 4] == b"PK\x05\x06"
    directory_offset = struct.unpack_from("<I", value, eocd + 16)[0]
    hidden = b"hidden-inter-member-gap"
    value[directory_offset:directory_offset] = hidden
    struct.pack_into(
        "<I",
        value,
        eocd + len(hidden) + 16,
        directory_offset + len(hidden),
    )
    return bytes(value)


def _corrupt_stored_member_crc(path: Path, member: str) -> None:
    with zipfile.ZipFile(path, mode="r") as archive:
        info = archive.getinfo(member)
    data = bytearray(path.read_bytes())
    offset = info.header_offset
    assert data[offset : offset + 4] == b"PK\x03\x04"
    name_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
    payload_begin = offset + 30 + name_length + extra_length
    assert info.compress_type == zipfile.ZIP_STORED and info.compress_size > 0
    data[payload_begin + info.compress_size - 1] ^= 0x01
    path.write_bytes(data)


@pytest.fixture
def stable_family_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], ...]:
    reference = _reference_artifact()
    trait = _trait_artifact()
    fit_reference, fit_trait = _fit_source_artifacts(monkeypatch)
    fit = fit_contextual_model_v1(
        fit_reference,
        fit_trait,
        trait_selector="trait-0",
    )
    cases = (
        {
            "name": "reference",
            "label": "Contextual reference V1",
            "artifact": reference,
            "writer": write_contextual_reference_v1,
            "loader": load_contextual_reference_v1,
            "module": reference_module,
            "suffix": CONTEXTUAL_REFERENCE_V1_SUFFIX,
            "magic": CONTEXTUAL_REFERENCE_V1_MAGIC,
            "logical": "contextual_reference_v1",
            "backend": "plink_bed_descriptor_stream_stage2_v1:2",
            "array": "gram",
        },
        {
            "name": "trait",
            "label": "Contextual trait V1",
            "artifact": trait,
            "writer": write_contextual_trait_v1,
            "loader": load_contextual_trait_v1,
            "module": trait_module,
            "suffix": CONTEXTUAL_TRAIT_V1_SUFFIX,
            "magic": CONTEXTUAL_TRAIT_V1_MAGIC,
            "logical": "contextual_trait_v1",
            "backend": "plink_bed_descriptor_stream_trait_v1:2",
            "array": "genetic_rhs",
        },
        {
            "name": "fit",
            "label": "Contextual fit V1",
            "artifact": fit,
            "writer": write_contextual_fit_v1,
            "loader": load_contextual_fit_v1,
            "module": fit_module,
            "suffix": CONTEXTUAL_FIT_V1_SUFFIX,
            "magic": CONTEXTUAL_FIT_V1_MAGIC,
            "logical": "contextual_fit_v1",
            "backend": "python_numpy_scipy_summary_fit_v1",
            "array": "normal_matrix",
        },
    )
    for case in cases:
        case["path"] = case["writer"](case["artifact"], tmp_path / case["name"])
    return cases


def _subprocess_environment(repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(repository / "tests"))
    )
    return environment


def test_actual_native_two_process_teardown_fits_only_stable_artifacts(
    tmp_path: Path,
) -> None:
    """A second process fits after every row-level/native source is gone."""
    from summit import gxeldcore

    repository = Path(__file__).resolve().parents[1]
    environment = _subprocess_environment(repository)
    native_module_dir = Path(gxeldcore.__file__).resolve().parent
    source_dir = tmp_path / "native-sources"
    stable_dir = tmp_path / "stable-only"
    source_dir.mkdir()
    stable_dir.mkdir()

    builder = r"""
from pathlib import Path
import json
import sys

import numpy as np
import summit

native_module_dir = str(Path(sys.argv[1]).resolve())
if native_module_dir not in summit.__path__:
    summit.__path__.append(native_module_dir)

from summit.context import array_sha256
from summit.context.reference_v1 import (
    run_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.trait_v1 import (
    ContextualTraitPublicationIdentityV1,
    run_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage2_streamed_reference import _make_case
from test_context_stage3_complete_reference_native import (
    _publication_identity,
    _stage3_executor,
    _variant_probes,
)
from test_context_stage4_trait_native import _trait_executor, _trait_inputs

source_dir = Path(sys.argv[2]).resolve()
stable_dir = Path(sys.argv[3]).resolve()
case = _make_case(
    source_dir,
    q_count=2,
    annotation_mode="strict_disjoint_binary_v1",
    name="native-input",
)
variant_probes = _variant_probes(case)
reference_publication = _publication_identity(case)
reference_executor = _stage3_executor(case, variant_probes)
reference = run_contextual_reference_v1(
    reference_executor, reference_publication
)
del reference_executor

phenotypes, residual_basis = _trait_inputs(case)
phenotype_path = source_dir / "phenotype-inputs.npz"
with phenotype_path.open("wb") as handle:
    np.savez_compressed(
        handle,
        phenotypes=phenotypes,
        residual_basis=residual_basis,
    )
with np.load(phenotype_path, allow_pickle=False) as archive:
    phenotypes = np.array(archive["phenotypes"], dtype=np.float64, order="F")
    residual_basis = np.array(
        archive["residual_basis"], dtype=np.float64, order="F"
    )

trait_publication = ContextualTraitPublicationIdentityV1(
    sample_order_sha256=reference_publication.sample_order_sha256,
    variant_order_allele_sha256=(
        reference_publication.variant_order_allele_sha256
    ),
    fixed_effect_spec_sha256=reference_publication.fixed_effect_spec_sha256,
    basis_specification_sha256=(
        reference_publication.basis_specification_sha256
    ),
    basis_calibration_sha256=(
        reference_publication.basis_calibration_sha256
    ),
    compatible_reference_identity_sha256=reference.manifest_sha256,
    retained_sample_map_sha256=array_sha256(case.retained_sample_rows),
    retained_variant_order_sha256=case.retained_variant_order_sha256,
    fixed_basis_sha256=array_sha256(case.fixed_basis),
    evaluated_phi_sha256=array_sha256(case.phi),
    genotype_scale_plan_sha256=reference.scale_plan.digest,
    missingness_sha256=case.missingness_sha256,
    annotation_map_sha256=array_sha256(case.annotations),
    annotation_names=case.annotation_names,
    group_map_sha256=array_sha256(case.group_index),
    group_ids=case.group_names,
    phenotype_batch_sha256=array_sha256(phenotypes),
    residual_basis_sha256=array_sha256(residual_basis),
    trait_ids=tuple(
        f"trait-{index}" for index in range(phenotypes.shape[1])
    ),
    residual_names=tuple(
        f"residual-{index}" for index in range(residual_basis.shape[1])
    ),
)
trait_executor = _trait_executor(
    case,
    phenotypes,
    residual_basis,
    variant_block=4,
    trait_feature_tile=5,
)
trait = run_contextual_trait_v1(trait_executor, trait_publication)
del trait_executor

reference_path = write_contextual_reference_v1(
    reference, stable_dir / "reference"
)
trait_path = write_contextual_trait_v1(trait, stable_dir / "trait")
print(json.dumps({
    "reference": str(reference_path),
    "trait": str(trait_path),
}))
"""
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            builder,
            str(native_module_dir),
            str(source_dir),
            str(stable_dir),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    published = json.loads(built.stdout.strip().splitlines()[-1])
    reference_path = Path(published["reference"])
    trait_path = Path(published["trait"])
    assert reference_path.is_file() and trait_path.is_file()

    source_paths = [
        source_dir / "native-input.bed",
        source_dir / "native-input.bim",
        source_dir / "native-input.fam",
        source_dir / "phenotype-inputs.npz",
    ]
    assert all(path.is_file() for path in source_paths)
    for path in source_paths:
        path.unlink()
    assert not any(source_dir.iterdir())
    source_dir.rmdir()
    assert not source_dir.exists()

    fitter = r"""
from pathlib import Path
import json
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
fit_prefix = Path(sys.argv[3]).resolve()
stable_dir = reference_path.parent
assert {path.resolve() for path in stable_dir.iterdir()} == {
    reference_path,
    trait_path,
}

reference = load_contextual_reference_v1(reference_path)
trait = load_contextual_trait_v1(trait_path)
fit = fit_contextual_model_v1(
    reference,
    trait,
    trait_selector="trait-0",
)
fit_path = write_contextual_fit_v1(fit, fit_prefix)
loaded = load_contextual_fit_v1(fit_path)
assert loaded.raw_rank == loaded.raw_coefficients.size
assert loaded.raw_loo_coefficients.shape[0] == len(reference.group_ids)

archive_keys = {}
for family, path in (
    ("reference", reference_path),
    ("trait", trait_path),
    ("fit", fit_path),
):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        expected = {"manifest_json", "manifest_sha256", *manifest["arrays"]}
        assert set(archive.files) == expected
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        archive_keys[family] = sorted(archive.files)
    assert all(
        "sample" not in axes and "variant" not in axes and "row" not in axes
        for axes in manifest["layouts"].values()
    )

print(json.dumps({"fit": str(fit_path), "archive_keys": archive_keys}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            fitter,
            str(reference_path),
            str(trait_path),
            str(stable_dir / "fit"),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    fitted = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(fitted["fit"]).is_file()
    assert set(fitted["archive_keys"]) == {"reference", "trait", "fit"}

    source_tokens = (
        str(source_dir),
        "native-input.bed",
        "native-input.bim",
        "native-input.fam",
        "phenotype-inputs.npz",
    )
    for path in (reference_path, trait_path, Path(fitted["fit"])):
        with np.load(path, allow_pickle=False) as archive:
            manifest_text = str(archive["manifest_json"].item())
        assert not any(token in manifest_text for token in source_tokens)


def test_recovered_native_faults_publish_roundtrip_and_reject_telemetry_grafts(
    tmp_path: Path,
) -> None:
    from test_context_stage2_streamed_reference import _make_case
    from test_context_stage3_complete_reference_native import (
        _fault_executor,
        _publication_identity,
        _stable_scale_plan,
        _variant_probes,
    )
    from test_context_stage4_trait_native import _trait_executor, _trait_inputs

    case = _make_case(
        tmp_path,
        q_count=2,
        annotation_mode="strict_disjoint_binary_v1",
        name="recovered-publication",
    )
    variant_probes = _variant_probes(case)
    reference_publication = _publication_identity(case)
    reference_native = dict(
        _fault_executor(case, variant_probes, "source_tn", "one_shot").run()
    )
    reference = adapt_native_contextual_reference_v1(
        reference_native,
        reference_publication,
    )
    reference_path = write_contextual_reference_v1(
        reference,
        tmp_path / "recovered-reference",
    )
    assert (
        load_contextual_reference_v1(reference_path).manifest_sha256
        == reference.manifest_sha256
    )

    phenotypes, residual_basis = _trait_inputs(case)
    anchor_executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=4,
        trait_feature_tile=5,
    )
    trait_anchor = next(
        item["semantic_anchor"]
        for item in anchor_executor.semantic_anchors()
        if item["operation"] == "trait_score_tn"
    )
    trait_native = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=4,
            trait_feature_tile=5,
            fault_operation="trait_score_tn",
            fault_mode="one_shot",
            fault_semantic_anchor=trait_anchor,
        ).run()
    )
    trait_publication = ContextualTraitPublicationIdentityV1(
        sample_order_sha256=reference_publication.sample_order_sha256,
        variant_order_allele_sha256=(reference_publication.variant_order_allele_sha256),
        fixed_effect_spec_sha256=reference_publication.fixed_effect_spec_sha256,
        basis_specification_sha256=(reference_publication.basis_specification_sha256),
        basis_calibration_sha256=reference_publication.basis_calibration_sha256,
        compatible_reference_identity_sha256=reference.manifest_sha256,
        retained_sample_map_sha256=array_sha256(case.retained_sample_rows),
        retained_variant_order_sha256=case.retained_variant_order_sha256,
        fixed_basis_sha256=array_sha256(case.fixed_basis),
        evaluated_phi_sha256=array_sha256(case.phi),
        genotype_scale_plan_sha256=_stable_scale_plan(case).digest,
        missingness_sha256=case.missingness_sha256,
        annotation_map_sha256=array_sha256(case.annotations),
        annotation_names=case.annotation_names,
        group_map_sha256=array_sha256(case.group_index),
        group_ids=case.group_names,
        phenotype_batch_sha256=array_sha256(phenotypes),
        residual_basis_sha256=array_sha256(residual_basis),
        trait_ids=tuple(f"trait-{index}" for index in range(phenotypes.shape[1])),
        residual_names=tuple(
            f"residual-{index}" for index in range(residual_basis.shape[1])
        ),
    )
    trait = adapt_native_contextual_trait_v1(trait_native, trait_publication)
    trait_path = write_contextual_trait_v1(
        trait,
        tmp_path / "recovered-trait",
    )
    assert load_contextual_trait_v1(trait_path).manifest_sha256 == trait.manifest_sha256

    for adapter, native_result, publication in (
        (
            adapt_native_contextual_reference_v1,
            reference_native,
            reference_publication,
        ),
        (adapt_native_contextual_trait_v1, trait_native, trait_publication),
    ):
        for field, replacement in (
            ("fault_operation", "action_gram_tn"),
            ("fault_mode", "nan"),
            ("fault_semantic_anchor", "grafted-anchor"),
        ):
            tampered = deepcopy(native_result)
            tampered["telemetry"][field] = replacement
            with pytest.raises(ValueError, match="fault|telemetry"):
                adapter(tampered, publication)

        tampered = deepcopy(native_result)
        events = [
            event
            for event in tampered["telemetry"]["events"]
            if event["event_class"] != "fault_detection"
        ]
        for sequence, event in enumerate(events, start=1):
            event["sequence"] = sequence
        tampered["telemetry"]["events"] = events
        tampered["telemetry"]["observed_events"] = len(events)
        with pytest.raises(ValueError, match="fault|telemetry"):
            adapter(tampered, publication)

        tampered = deepcopy(native_result)
        detection = next(
            event
            for event in tampered["telemetry"]["events"]
            if event["event_class"] == "fault_detection"
        )
        detection["attempt"] = 3
        with pytest.raises(ValueError, match="fault|attempt"):
            adapter(tampered, publication)

        tampered = deepcopy(native_result)
        retry_index, retry_event = next(
            (index, event)
            for index, event in enumerate(tampered["telemetry"]["events"])
            if event["event_class"] == "retry"
        )
        target = next(
            event
            for event in tampered["telemetry"]["events"]
            if event["event_class"] == "protected_call"
            and event["operation"] != tampered["telemetry"]["fault_operation"]
            and event["transpose_left"] == retry_event["transpose_left"]
        )
        grafted = deepcopy(target)
        grafted.update(
            {
                "event_class": "retry",
                "resolution": "deterministic_tiled_retry",
                "attempt": 2,
                "sequence": retry_event["sequence"],
            }
        )
        tampered["telemetry"]["events"][retry_index] = grafted
        with pytest.raises(ValueError, match="recovered-fault telemetry"):
            adapter(tampered, publication)

        tampered = deepcopy(native_result)
        for event in tampered["telemetry"]["events"]:
            if event["event_class"] in {
                "fault_injection",
                "fault_detection",
                "repair",
                "retry",
                "trusted_fallback",
            }:
                old_coordinate = (
                    f"|canonical={event['canonical_begin']}:"
                    f"{event['canonical_end']}"
                )
                event["canonical_begin"] += 1000
                event["canonical_end"] += 1000
                new_coordinate = (
                    f"|canonical={event['canonical_begin']}:"
                    f"{event['canonical_end']}"
                )
                event["semantic_anchor"] = event["semantic_anchor"].replace(
                    old_coordinate,
                    new_coordinate,
                )
        with pytest.raises(ValueError, match="recovered-fault telemetry"):
            adapter(tampered, publication)

        tampered = deepcopy(native_result)
        phase_event = next(
            event
            for event in tampered["telemetry"]["events"]
            if event["event_class"] == "protected_call"
        )
        phase_event["phase"] = "publication"
        with pytest.raises(ValueError, match="phase"):
            adapter(tampered, publication)

        tampered = deepcopy(native_result)
        tampered["telemetry"]["events"][1]["process_id"] += 1
        with pytest.raises(ValueError, match="sequence/ownership"):
            adapter(tampered, publication)

        for field, replacement in (
            ("variant_coordinate_mode", "exact_membership_sha256_v1"),
            ("variant_membership_count", 1),
            ("variant_membership_sha256", "a" * 64),
            ("variant_range_is_exact", False),
        ):
            tampered = deepcopy(native_result)
            protected = next(
                event
                for event in tampered["telemetry"]["events"]
                if event["event_class"] == "protected_call"
            )
            protected[field] = replacement
            with pytest.raises(ValueError, match="variant|anchor"):
                adapter(tampered, publication)

    for operation in ("full_target_nn", "group_target_nn", "group_cross_gram_tn"):
        tampered = deepcopy(reference_native)
        protected = next(
            event
            for event in tampered["telemetry"]["events"]
            if event["event_class"] == "protected_call"
            and event["operation"] == operation
        )
        assert protected["variant_coordinate_mode"] == ("exact_membership_sha256_v1")
        protected["variant_coordinate_mode"] = "logical_half_open_range_v1"
        protected["variant_range_is_exact"] = True
        protected["variant_membership_count"] = 0
        protected["variant_membership_sha256"] = ""
        protected["semantic_anchor"] = protected["semantic_anchor"].split(
            "|variant_mode=", 1
        )[0]
        with pytest.raises(ValueError, match="operation/variant mode"):
            adapt_native_contextual_reference_v1(tampered, reference_publication)

    tampered = deepcopy(reference_native)
    for event in tampered["telemetry"]["events"]:
        if (
            event["operation"] == "same_person_gram_tn"
            and event["semantic_role"] == "global_merge"
        ):
            event["semantic_role"] = "tile"
            event["semantic_anchor"] = event["semantic_anchor"].replace(
                "|role=global_merge|", "|role=tile|"
            )
    with pytest.raises(ValueError, match="tile/global-merge"):
        adapt_native_contextual_reference_v1(tampered, reference_publication)

    direct_native = dict(
        _fault_executor(
            case,
            variant_probes,
            "direct_grouped_tn",
            "one_shot",
        ).run()
    )
    adapt_native_contextual_reference_v1(direct_native, reference_publication)
    tampered = deepcopy(direct_native)
    for event in tampered["telemetry"]["events"]:
        if (
            event["operation"] == "direct_grouped_tn"
            and event["semantic_placement"] == "genotype_scaled"
        ):
            event["semantic_placement"] = "action_scaled"
            event["semantic_anchor"] = event["semantic_anchor"].replace(
                "|placement=genotype_scaled|", "|placement=action_scaled|"
            )
    with pytest.raises(ValueError, match="placement event ledger"):
        adapt_native_contextual_reference_v1(tampered, reference_publication)


def test_stable_schema_compatibility_and_migration_matrix_is_closed(
    stable_family_cases: tuple[dict[str, Any], ...], tmp_path: Path
) -> None:
    for case in stable_family_cases:
        manifest = case["artifact"].manifest
        assert manifest["magic"] == case["magic"]
        assert manifest["logical_schema_version"] == case["logical"]
        assert manifest["native_backend_version"] == case["backend"]
        assert Path(case["path"]).name.endswith(case["suffix"])
        assert case["loader"](case["path"]).manifest_sha256 == canonical_sha256(
            manifest
        )
        with zipfile.ZipFile(case["path"], mode="r") as archive:
            names = archive.namelist()
            assert len(names) == len(set(names))
            assert set(names) == {
                "manifest_json.npy",
                "manifest_sha256.npy",
                *(f"{name}.npy" for name in manifest["arrays"]),
            }

    development_loaders = (
        load_context_reference,
        load_context_trait_summary,
        load_context_fit,
    )
    for case, development_loader in zip(
        stable_family_cases, development_loaders, strict=True
    ):
        with pytest.raises(ValueError):
            development_loader(case["path"])

    for source, destination in zip(
        stable_family_cases,
        (*stable_family_cases[1:], stable_family_cases[0]),
        strict=True,
    ):
        renamed = tmp_path / f"renamed-{source['name']}{destination['suffix']}"
        shutil.copyfile(source["path"], renamed)
        with pytest.raises(ValueError, match=destination["label"]):
            destination["loader"](renamed)


def test_all_stable_loaders_preflight_duplicate_corrupt_and_noncanonical_members(
    stable_family_cases: tuple[dict[str, Any], ...], tmp_path: Path
) -> None:
    for case in stable_family_cases:
        source = Path(case["path"])
        raw_member = f"{case['array']}.npy"
        with np.load(source, allow_pickle=False) as archive:
            original = np.array(archive[case["array"]], copy=True)
        original_member = dict(_zip_entries(source))[raw_member]
        wrong_shape = original.reshape((original.size,))
        mutations: dict[str, dict[str, Any]] = {
            "duplicate": {"duplicate": raw_member},
            "member-truncated": {"replacements": {raw_member: original_member[:-8]}},
            "object": {
                "replacements": {
                    raw_member: _npy_bytes(
                        np.full(original.shape, "unsafe", dtype=object),
                        allow_pickle=True,
                    )
                }
            },
            "dtype": {"replacements": {raw_member: _npy_bytes(original.astype("<f4"))}},
            "endian": {
                "replacements": {raw_member: _npy_bytes(original.astype(">f8"))}
            },
            "shape": {"replacements": {raw_member: _npy_bytes(wrong_shape)}},
            "extra": {"extras": (("unexpected.npy", _npy_bytes(np.asarray([1.0]))),)},
            "missing": {"replacements": {raw_member: None}},
        }
        for mutation, options in mutations.items():
            damaged = tmp_path / f"{case['name']}-{mutation}{case['suffix']}"
            _rewrite_zip(source, damaged, **options)
            with pytest.raises(ValueError, match=case["label"]):
                case["loader"](damaged)

        crc = tmp_path / f"{case['name']}-crc{case['suffix']}"
        _rewrite_zip(source, crc)
        _corrupt_stored_member_crc(crc, raw_member)
        with pytest.raises(ValueError, match=case["label"]):
            case["loader"](crc)

        truncated = tmp_path / f"{case['name']}-archive-truncated{case['suffix']}"
        data = source.read_bytes()
        truncated.write_bytes(data[: -min(64, len(data) // 4)])
        with pytest.raises(ValueError, match=case["label"]):
            case["loader"](truncated)

        prefixed = tmp_path / f"{case['name']}-prefixed{case['suffix']}"
        prefixed.write_bytes(b"hidden-prefix" + data)
        with pytest.raises(ValueError, match=case["label"]):
            case["loader"](prefixed)

        gapped = tmp_path / f"{case['name']}-gapped{case['suffix']}"
        gapped.write_bytes(_insert_gap_before_central_directory(data))
        with pytest.raises(ValueError, match=case["label"]):
            case["loader"](gapped)

        extra = tmp_path / f"{case['name']}-unknown-extra{case['suffix']}"
        _rewrite_canonical_zip(
            source,
            extra,
            first_member_extra=struct.pack("<HH4s", 0xCAFE, 4, b"hide"),
        )
        with pytest.raises(ValueError, match=case["label"]):
            case["loader"](extra)

        hidden_header = tmp_path / f"{case['name']}-header-comment{case['suffix']}"
        _rewrite_canonical_zip(
            source,
            hidden_header,
            replacements={raw_member: _npy_with_hidden_header_comment(original_member)},
        )
        with pytest.raises(ValueError, match=case["label"]):
            case["loader"](hidden_header)


def test_stable_reader_bounds_the_zip_directory_before_zipfile_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = tmp_path / "forged-member-count.npz"
    forged.write_bytes(
        struct.pack(
            "<4s4H2IH",
            b"PK\x05\x06",
            0,
            0,
            9,
            9,
            0,
            0,
            0,
        )
    )
    constructed = False

    def forbidden_zipfile(*_args: Any, **_kwargs: Any) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("ZipFile must not see an unbounded directory")

    monkeypatch.setattr(artifact_io.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="member count"):
        with artifact_io.StableNpzReader(
            forged,
            family="Stage5 forged archive",
            maximum_members=4,
        ):
            raise AssertionError("forged archive unexpectedly admitted")
    assert constructed is False


def test_stable_python_source_and_runtime_provenance_is_closed_and_tamper_evident(
    stable_family_cases: tuple[dict[str, Any], ...], tmp_path: Path
) -> None:
    reference, trait, fit = stable_family_cases
    for case in (reference, trait):
        provenance = case["artifact"].manifest["execution"]["python_adapter_provenance"]
        assert provenance["ordered_modules"]
        assert set(provenance["module_sha256"]) == set(provenance["ordered_modules"])
        assert set(provenance["runtime_versions"]) == {
            "python_implementation",
            "python_version",
            "numpy_version",
        }
        assert provenance["claim"] == (
            "provenance_only_no_binary_reproducibility_claim_v1"
        )
    fit_provenance = fit["artifact"].manifest["compatibility"][
        "python_fit_implementation_provenance"
    ]
    assert set(fit_provenance["runtime_versions"]) == {
        "python_implementation",
        "python_version",
        "numpy_version",
        "scipy_version",
    }
    assert {
        "fit.py",
        "fit_v1.py",
        "reference_v1.py",
        "trait_v1.py",
        "schema.py",
        "spec.py",
    }.issubset(fit_provenance["ordered_modules"])

    manifest = json.loads(canonical_json(fit["artifact"].manifest))
    provenance = manifest["compatibility"]["python_fit_implementation_provenance"]
    provenance["module_sha256"]["fit.py"] = "0" * 64
    provenance_sha256 = canonical_sha256(provenance)
    manifest["compatibility"][
        "python_fit_implementation_provenance_sha256"
    ] = provenance_sha256
    manifest["identity"][
        "python_fit_implementation_provenance_sha256"
    ] = provenance_sha256
    replacements = {
        "manifest_json.npy": _npy_bytes(np.asarray(canonical_json(manifest))),
        "manifest_sha256.npy": _npy_bytes(np.asarray(canonical_sha256(manifest))),
    }
    damaged = tmp_path / f"fit-provenance-tamper{CONTEXTUAL_FIT_V1_SUFFIX}"
    _rewrite_canonical_zip(Path(fit["path"]), damaged, replacements=replacements)
    with pytest.raises(ValueError, match="aggregate digest mismatch"):
        load_contextual_fit_v1(damaged)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_commit", "not-a-commit", "source commit"),
        ("compiler_id", "", "build string"),
        ("asan_enabled", True, "sanitizer provenance"),
        ("native_arch_optimization_enabled", True, "architecture provenance"),
        ("private_openblas_enabled", True, "private-BLAS provenance"),
    ],
)
def test_native_build_provenance_rejects_contradictory_claims(
    stable_family_cases: tuple[dict[str, Any], ...],
    field: str,
    value: Any,
    message: str,
) -> None:
    for case, module in (
        (stable_family_cases[0], reference_module),
        (stable_family_cases[1], trait_module),
    ):
        artifact = case["artifact"]
        provenance = dict(artifact.manifest["execution"]["build_provenance"])
        provenance[field] = value
        with pytest.raises(ValueError, match=message):
            module._validate_build_provenance(
                provenance,
                build_id=artifact.manifest["build_id"],
                source_tree_sha256=artifact.manifest["identity"]["source_tree_sha256"],
            )


def test_all_stable_writers_fsync_parent_after_publish_and_preserve_old_target(
    stable_family_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for case in stable_family_cases:
        durable_calls: list[Path] = []
        durable_target = tmp_path / f"durable-{case['name']}{case['suffix']}"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                case["module"],
                "fsync_parent_directory",
                lambda path: durable_calls.append(Path(path)),
            )
            written = case["writer"](case["artifact"], durable_target)
        assert durable_calls == [written]

        target = Path(case["path"])
        original = target.read_bytes()
        temporary_before = set(target.parent.glob(f".{target.name}.*"))

        with pytest.raises(FileExistsError) as error:
            case["writer"](case["artifact"], target)
        assert error.value.errno == errno.EEXIST
        assert target.read_bytes() == original
        assert set(target.parent.glob(f".{target.name}.*")) == temporary_before
