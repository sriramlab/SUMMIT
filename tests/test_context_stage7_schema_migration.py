from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pytest

import summit.context._artifact_io as artifact_io
from summit.context.fit_v1 import (
    CONTEXTUAL_FIT_V1_SUFFIX,
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.reference_v1 import (
    CONTEXTUAL_REFERENCE_V1_SUFFIX,
    load_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.spec import canonical_json, canonical_sha256
from summit.context.trait_v1 import (
    CONTEXTUAL_TRAIT_V1_SUFFIX,
    load_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import _artifact as _reference_artifact
from test_context_stage4_fit_v1 import _artifacts as _fit_source_artifacts
from test_context_stage4_trait_v1 import _artifact as _trait_artifact


def _npy_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(value), allow_pickle=False)
    return output.getvalue()


def _zip_entries(path: Path) -> tuple[tuple[str, bytes], ...]:
    with zipfile.ZipFile(path, mode="r") as archive:
        return tuple(
            (member.filename, archive.read(member)) for member in archive.infolist()
        )


def _rewrite_manifest_canonically(
    source: Path,
    target: Path,
    manifest: dict[str, Any],
) -> None:
    replacements = {
        "manifest_json.npy": _npy_bytes(canonical_json(manifest)),
        "manifest_sha256.npy": _npy_bytes(canonical_sha256(manifest)),
    }
    with zipfile.ZipFile(
        target,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for name, original in _zip_entries(source):
            with archive.open(name, mode="w", force_zip64=True) as stream:
                stream.write(replacements.get(name, original))


@pytest.fixture
def stable_family_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], ...]:
    reference = _reference_artifact()
    trait = _trait_artifact()
    with monkeypatch.context() as scoped:
        fit_reference, fit_trait = _fit_source_artifacts(scoped)
        fit = fit_contextual_model_v1(
            fit_reference,
            fit_trait,
            trait_selector="trait-0",
        )
    cases: tuple[dict[str, Any], ...] = (
        {
            "name": "reference",
            "artifact": reference,
            "writer": write_contextual_reference_v1,
            "loader": load_contextual_reference_v1,
            "suffix": CONTEXTUAL_REFERENCE_V1_SUFFIX,
            "other_family": "contextual_trait",
            "old_backend": "plink_bed_descriptor_stream_stage2_v1:1",
            "future_backend": "plink_bed_descriptor_stream_stage2_v1:3",
        },
        {
            "name": "trait",
            "artifact": trait,
            "writer": write_contextual_trait_v1,
            "loader": load_contextual_trait_v1,
            "suffix": CONTEXTUAL_TRAIT_V1_SUFFIX,
            "other_family": "contextual_fit",
            "old_backend": "plink_bed_descriptor_stream_trait_v1:1",
            "future_backend": "plink_bed_descriptor_stream_trait_v1:3",
        },
        {
            "name": "fit",
            "artifact": fit,
            "writer": write_contextual_fit_v1,
            "loader": load_contextual_fit_v1,
            "suffix": CONTEXTUAL_FIT_V1_SUFFIX,
            "other_family": "contextual_reference",
            "old_backend": "python_numpy_scipy_summary_fit_v0",
            "future_backend": "python_numpy_scipy_summary_fit_v2",
        },
    )
    for case in cases:
        case["path"] = case["writer"](
            case["artifact"],
            tmp_path / case["name"],
        )
    return cases


def _migration_mutations(
    case: dict[str, Any]
) -> tuple[tuple[str, dict[str, Any]], ...]:
    return (
        (
            "wrong-family",
            {"artifact_family": case["other_family"]},
        ),
        ("future-logical", {"logical_schema_version": f"contextual_{case['name']}_v2"}),
        (
            "future-grouped-encoding",
            {"grouped_encoding_version": "grouped_unnormalized_dense_v2"},
        ),
        ("future-api", {"native_api_version": 2}),
        ("boolean-api", {"native_api_version": True}),
        ("old-backend", {"native_backend_version": case["old_backend"]}),
        ("future-backend", {"native_backend_version": case["future_backend"]}),
        ("nonpublished", {"terminal_status": "complete"}),
    )


def test_canonical_manifest_rewrite_is_an_admitted_control(
    stable_family_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
) -> None:
    for case in stable_family_cases:
        rewritten = tmp_path / f"control-{case['name']}{case['suffix']}"
        _rewrite_manifest_canonically(
            Path(case["path"]),
            rewritten,
            dict(case["artifact"].manifest),
        )
        with zipfile.ZipFile(rewritten, mode="r") as archive:
            assert all(
                member.compress_type == zipfile.ZIP_DEFLATED
                for member in archive.infolist()
            )
        assert (
            case["loader"](rewritten).manifest_sha256
            == case["artifact"].manifest_sha256
        )


def test_all_stable_families_reject_migration_before_scientific_array_preflight(
    stable_family_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_array_preflight(
        _reader: artifact_io.StableNpzReader,
        _expected: object,
    ) -> None:
        raise AssertionError("schema migration reached scientific array preflight")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            artifact_io.StableNpzReader,
            "preflight_arrays",
            forbidden_array_preflight,
        )
        for case in stable_family_cases:
            source = Path(case["path"])
            for mutation, changes in _migration_mutations(case):
                manifest = json.loads(canonical_json(case["artifact"].manifest))
                manifest.update(changes)
                damaged = tmp_path / f"{case['name']}-{mutation}{case['suffix']}"
                _rewrite_manifest_canonically(source, damaged, manifest)
                with pytest.raises(ValueError):
                    case["loader"](damaged)
