from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import summit.context as context
import summit.context.fit_v1 as fit_module
import summit.context.reference_v1 as reference_module
import summit.context.trait_v1 as trait_module
from summit.context.fit_v1 import (
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.reference_v1 import (
    CONTEXTUAL_REFERENCE_V1_SUFFIX,
    ContextualReferenceArtifactV1,
    adapt_native_contextual_reference_v1,
    load_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.spec import canonical_sha256
from summit.context.trait_v1 import (
    CONTEXTUAL_TRAIT_V1_SUFFIX,
    ContextualTraitArtifactV1,
    adapt_native_contextual_trait_v1,
    load_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import _native_result as _reference_native
from test_context_stage4_fit_v1 import _write_strict_compatible_summary_inputs
from test_context_stage4_trait_v1 import _native_result as _trait_native


_SAMPLE_SENTINEL = "STAGE7_SAMPLE_ROW_SENTINEL_7d013"
_VARIANT_SENTINEL = "STAGE7_VARIANT_SENTINEL_1b2f9"
_SOURCE_PATH_SENTINEL = "/stage7/raw/source/SOURCE_PATH_SENTINEL_a4c31"
_PHENOTYPE_SENTINEL = 1_234_567.8901234567
_RAW_TEXT_SENTINELS = (
    _SAMPLE_SENTINEL,
    _VARIANT_SENTINEL,
    _SOURCE_PATH_SENTINEL,
)
_FORBIDDEN_STABLE_AXES = frozenset(
    {"N", "M", "sample", "samples", "variant", "variants", "row", "rows"}
)

_PRIVATE_STABLE_ARTIFACT_V1_ALLOWLIST = frozenset(
    {
        "CONTEXTUAL_REFERENCE_V1_MAGIC",
        "CONTEXTUAL_REFERENCE_V1_SUFFIX",
        "ContextualReferenceArtifactV1",
        "ContextualReferencePublicationIdentityV1",
        "adapt_native_contextual_reference_v1",
        "contextual_variant_order_allele_sha256_v1",
        "load_contextual_reference_v1",
        "reference_moments_after_deleting_groups_v1",
        "run_contextual_reference_v1",
        "write_contextual_reference_v1",
        "CONTEXTUAL_TRAIT_V1_MAGIC",
        "CONTEXTUAL_TRAIT_V1_SUFFIX",
        "ContextualTraitArtifactV1",
        "ContextualTraitMomentsV1",
        "ContextualTraitPublicationIdentityV1",
        "adapt_native_contextual_trait_v1",
        "load_contextual_trait_v1",
        "run_contextual_trait_v1",
        "trait_moments_after_deleting_groups_v1",
        "write_contextual_trait_v1",
        "CONTEXTUAL_FIT_V1_MAGIC",
        "CONTEXTUAL_FIT_V1_SUFFIX",
        "ContextualFitArtifactV1",
        "assemble_contextual_normal_equations_v1",
        "fit_contextual_model_v1",
        "load_contextual_fit_v1",
        "validate_contextual_fit_compatibility_v1",
        "write_contextual_fit_v1",
    }
)

_LEGACY_SOURCE_SHA256 = {
    "src/summit/cli.py": (
        "57cfafc0c854094befa2c52de4ceb07a529034017c62faff2be7f13e2839605c"
    ),
    "src/summit/__init__.py": (
        "448f5770e516c134d875bdd95627aca81eb6520c9026681f959eea903e3e0153"
    ),
    "src/summit/__main__.py": (
        "d7815f9da17bcd8e79d1d47170e5795e29cc5837f97d1049664813068a9e563f"
    ),
    "pyproject.toml": (
        "e077aa42e0cb53e401b4139fdd7264c3b416397be2a8933572ef6f58c2d521b0"
    ),
}
_LEGACY_HELP_SHA256 = "85f402d01c85ca7da9194405fa2ca22bd7c043da3f6f11de19a1508fc641c07b"
_LEGACY_DEFAULTS_SHA256 = (
    "29b7813d8dc5e8be8778405addd0a10cb0111eac7f79426846687bec06229210"
)


def _assert_no_raw_sentinel(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_raw_sentinel(key, path=f"{path}.<key>")
            _assert_no_raw_sentinel(item, path=f"{path}.{key}")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "USO":
            _assert_no_raw_sentinel(value.tolist(), path=path)
        elif value.dtype.kind in "iufc" and np.any(value == _PHENOTYPE_SENTINEL):
            raise AssertionError(f"raw phenotype sentinel leaked at {path}")
        return
    if isinstance(value, np.generic):
        _assert_no_raw_sentinel(value.item(), path=path)
        return
    if isinstance(value, str):
        for sentinel in _RAW_TEXT_SENTINELS:
            assert sentinel not in value, f"raw text sentinel leaked at {path}"
        return
    if isinstance(value, (bytes, bytearray)):
        _assert_no_raw_sentinel(
            bytes(value).decode("utf-8", errors="ignore"), path=path
        )
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _assert_no_raw_sentinel(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (int, float, complex)) and value == _PHENOTYPE_SENTINEL:
        raise AssertionError(f"raw phenotype sentinel leaked at {path}")


def _assert_stable_layouts_have_no_row_axis(manifest: Mapping[str, Any]) -> None:
    layouts = manifest["layouts"]
    for name, axes in layouts.items():
        leaked = _FORBIDDEN_STABLE_AXES.intersection(axes)
        assert not leaked, f"stable member {name!r} has row axes {sorted(leaked)}"


def _decoded_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        result: dict[str, object] = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    result["manifest_json"] = json.loads(str(result["manifest_json"].item()))
    return result


@pytest.mark.parametrize(
    ("family", "location", "field", "raw_value", "expected_error"),
    (
        (
            "reference",
            "top",
            "sample_ids",
            [_SAMPLE_SENTINEL],
            "result schema mismatch",
        ),
        (
            "reference",
            "top",
            "variant_ids",
            [_VARIANT_SENTINEL],
            "result schema mismatch",
        ),
        (
            "reference",
            "file",
            "bed_path",
            _SOURCE_PATH_SENTINEL,
            "file-content identity is invalid",
        ),
        (
            "trait",
            "top",
            "sample_ids",
            [_SAMPLE_SENTINEL],
            "result schema mismatch",
        ),
        (
            "trait",
            "top",
            "variant_ids",
            [_VARIANT_SENTINEL],
            "result schema mismatch",
        ),
        (
            "trait",
            "top",
            "phenotype_values",
            [_PHENOTYPE_SENTINEL],
            "result schema mismatch",
        ),
        (
            "trait",
            "file",
            "fam_path",
            _SOURCE_PATH_SENTINEL,
            "file-content identity is invalid",
        ),
    ),
)
def test_native_adapters_reject_undeclared_raw_row_fields(
    family: str,
    location: str,
    field: str,
    raw_value: object,
    expected_error: str,
) -> None:
    if family == "reference":
        native, publication = _reference_native()
        adapter = adapt_native_contextual_reference_v1
    else:
        native, publication = _trait_native()
        adapter = adapt_native_contextual_trait_v1
    native = deepcopy(native)
    if location == "file":
        native["file_content_identity"] = deepcopy(native["file_content_identity"])
        native["file_content_identity"][field] = raw_value
    else:
        native[field] = raw_value
    with pytest.raises(ValueError, match=expected_error):
        adapter(native, publication)


def test_stable_outputs_and_native_controller_evidence_do_not_leak_raw_rows(
    tmp_path: Path,
) -> None:
    sample_digest = canonical_sha256({"sample_ids": [_SAMPLE_SENTINEL]})
    variant_digest = canonical_sha256({"variant_ids": [_VARIANT_SENTINEL]})
    source_digest = canonical_sha256({"source_path": _SOURCE_PATH_SENTINEL})
    phenotype_digest = canonical_sha256({"phenotype_values": [_PHENOTYPE_SENTINEL]})

    reference_native, reference_publication = _reference_native()
    reference_native = deepcopy(reference_native)
    reference_publication = replace(
        reference_publication,
        sample_order_sha256=sample_digest,
        retained_sample_map_sha256=sample_digest,
        variant_order_allele_sha256=variant_digest,
    )
    reference_native["retained_sample_map_sha256"] = sample_digest
    reference_native["variant_order_allele_sha256"] = variant_digest
    reference_native["file_content_identity"] = deepcopy(
        reference_native["file_content_identity"]
    )
    reference_native["file_content_identity"]["bim_full_sha256"] = source_digest
    reference_evidence = adapt_native_contextual_reference_v1(
        reference_native,
        reference_publication,
    )

    trait_native, trait_publication = _trait_native()
    trait_native = deepcopy(trait_native)
    trait_publication = replace(
        trait_publication,
        sample_order_sha256=sample_digest,
        retained_sample_map_sha256=sample_digest,
        variant_order_allele_sha256=variant_digest,
        phenotype_batch_sha256=phenotype_digest,
    )
    trait_native["retained_sample_map_sha256"] = sample_digest
    trait_native["variant_order_allele_sha256"] = variant_digest
    trait_native["phenotype_batch_sha256"] = phenotype_digest
    trait_native["file_content_identity"] = deepcopy(
        trait_native["file_content_identity"]
    )
    trait_native["file_content_identity"]["fam_full_sha256"] = source_digest
    trait_evidence = adapt_native_contextual_trait_v1(
        trait_native,
        trait_publication,
    )

    raw_source = (
        tmp_path / "stage7" / "raw" / "source" / "SOURCE_PATH_SENTINEL_a4c31.npz"
    )
    raw_source.parent.mkdir(parents=True)
    assert _SOURCE_PATH_SENTINEL in str(raw_source)
    genotype = (np.arange(42, dtype=np.float64).reshape(7, 6) - 20.0) / 7.0
    phenotypes = np.arange(14, dtype=np.float64).reshape(7, 2) / 5.0
    phenotypes[0, 0] = _PHENOTYPE_SENTINEL
    sample_ids = np.asarray(
        [_SAMPLE_SENTINEL, *(f"sample-{index}" for index in range(1, 7))]
    )
    variant_ids = np.asarray(
        [_VARIANT_SENTINEL, *(f"variant-{index}" for index in range(1, 6))]
    )
    with raw_source.open("xb") as handle:
        np.savez_compressed(
            handle,
            genotype=genotype,
            phenotypes=phenotypes,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
        )
    stable_sources = tmp_path / "compatible-stable-sources"
    stable_sources.mkdir()
    _write_strict_compatible_summary_inputs(stable_sources, raw_source)
    raw_source.unlink()
    assert not raw_source.exists()

    baseline_reference = load_contextual_reference_v1(
        stable_sources / f"reference{CONTEXTUAL_REFERENCE_V1_SUFFIX}"
    )
    baseline_trait = load_contextual_trait_v1(
        stable_sources / f"trait{CONTEXTUAL_TRAIT_V1_SUFFIX}"
    )
    baseline_reference_sha256 = baseline_reference.manifest_sha256
    baseline_trait_sha256 = baseline_trait.manifest_sha256

    reference_manifest = deepcopy(dict(baseline_reference.manifest))
    reference_manifest["identity"].update(
        {
            "sample_order_sha256": sample_digest,
            "retained_sample_map_sha256": sample_digest,
            "variant_order_allele_sha256": variant_digest,
        }
    )
    reference_manifest["execution"]["file_content_identity"][
        "bim_full_sha256"
    ] = source_digest
    reference = replace(baseline_reference, manifest=reference_manifest)
    assert isinstance(reference, ContextualReferenceArtifactV1)

    trait_manifest = deepcopy(dict(baseline_trait.manifest))
    trait_manifest["identity"].update(
        {
            "sample_order_sha256": sample_digest,
            "retained_sample_map_sha256": sample_digest,
            "variant_order_allele_sha256": variant_digest,
            "phenotype_batch_sha256": phenotype_digest,
        }
    )
    trait_manifest["execution"]["file_content_identity"][
        "fam_full_sha256"
    ] = source_digest
    trait_manifest["compatible_reference_identity_sha256"] = reference.manifest_sha256
    trait = replace(baseline_trait, manifest=trait_manifest)
    assert isinstance(trait, ContextualTraitArtifactV1)

    assert reference.manifest_sha256 != baseline_reference_sha256
    assert trait.manifest_sha256 != baseline_trait_sha256
    grid_provenance = canonical_sha256({"grid": "independently-generated-stage7"})
    context_grid = np.asarray([[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    basis_metric = np.eye(reference.component_index.pair_index.num_basis)
    with pytest.raises(ValueError, match="trusted-caller non-row assertion"):
        fit_contextual_model_v1(
            reference,
            trait,
            trait_selector="trait-0",
            context_grid=context_grid,
            basis_metric=basis_metric,
            evaluation_grid_role="non_row_evaluation_grid_v1",
            evaluation_grid_provenance_sha256=grid_provenance,
        )
    fit = fit_contextual_model_v1(
        reference,
        trait,
        trait_selector="trait-0",
        context_grid=context_grid,
        basis_metric=basis_metric,
        evaluation_grid_role="non_row_evaluation_grid_v1",
        evaluation_grid_provenance_sha256=grid_provenance,
        evaluation_grid_trusted_non_row=True,
    )

    assert reference.manifest["identity"]["sample_order_sha256"] == sample_digest
    assert (
        reference.manifest["identity"]["variant_order_allele_sha256"] == variant_digest
    )
    assert (
        reference.manifest["execution"]["file_content_identity"]["bim_full_sha256"]
        == source_digest
    )
    assert trait.manifest["identity"]["sample_order_sha256"] == sample_digest
    assert trait.manifest["identity"]["variant_order_allele_sha256"] == variant_digest
    assert trait.manifest["identity"]["phenotype_batch_sha256"] == phenotype_digest
    assert (
        trait.manifest["execution"]["file_content_identity"]["fam_full_sha256"]
        == source_digest
    )
    assert fit.manifest["identity"]["reference_manifest_sha256"] == (
        reference.manifest_sha256
    )
    assert fit.manifest["identity"]["trait_manifest_sha256"] == (trait.manifest_sha256)
    assert fit.manifest["compatibility"]["matched_variant_order_allele_sha256"] == (
        variant_digest
    )
    assert fit.manifest["surfaces"]["trusted_caller_non_row_assertion"] is True

    admitted_native_controller_evidence = {
        "reference_native_terminal_result": reference_native,
        "reference_publication_identity": reference_publication.to_dict(),
        "reference_adapter_artifact_manifest": reference_evidence.manifest,
        "trait_native_terminal_result": trait_native,
        "trait_publication_identity": trait_publication.to_dict(),
        "trait_adapter_artifact_manifest": trait_evidence.manifest,
        "compatible_fit_reference_manifest": reference.manifest,
        "compatible_fit_trait_manifest": trait.manifest,
    }
    _assert_no_raw_sentinel(
        admitted_native_controller_evidence,
        path="admitted_native_controller",
    )

    cases = (
        (
            "reference",
            reference,
            write_contextual_reference_v1,
            load_contextual_reference_v1,
        ),
        ("trait", trait, write_contextual_trait_v1, load_contextual_trait_v1),
        ("fit", fit, write_contextual_fit_v1, load_contextual_fit_v1),
    )
    for name, artifact, writer, loader in cases:
        _assert_stable_layouts_have_no_row_axis(artifact.manifest)
        in_memory = {
            "manifest": artifact.manifest,
            "arrays": {
                member: getattr(artifact, member)
                for member in artifact.manifest["arrays"]
            },
        }
        _assert_no_raw_sentinel(in_memory, path=f"{name}.memory")
        output = writer(artifact, tmp_path / name)
        decoded = _decoded_npz(output)
        _assert_no_raw_sentinel(decoded, path=f"{name}.npz")
        assert loader(output).manifest_sha256 == artifact.manifest_sha256


def test_private_stable_v1_allowlist_and_legacy_cli_surface_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_module_exports = frozenset(
        {
            *reference_module.__all__,
            *trait_module.__all__,
            *fit_module.__all__,
        }
    )
    assert stable_module_exports == _PRIVATE_STABLE_ARTIFACT_V1_ALLOWLIST
    assert stable_module_exports <= frozenset(context.__all__)
    assert len(context.__all__) == len(set(context.__all__))

    repository = Path(__file__).resolve().parents[1]
    observed_source_hashes = {
        relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for relative in _LEGACY_SOURCE_SHA256
    }
    assert observed_source_hashes == _LEGACY_SOURCE_SHA256

    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    from summit import cli

    parser = cli.build_parser()
    parser.prog = "summit"
    help_text = parser.format_help()
    defaults = json.dumps(
        vars(parser.parse_args([])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    long_options = {
        option for option in parser._option_string_actions if option.startswith("--")
    }
    assert len(parser._option_string_actions) == 134
    assert len(long_options) == 133
    assert not any("context" in option.lower() for option in long_options)
    assert "context" not in help_text.lower()
    assert hashlib.sha256(help_text.encode("utf-8")).hexdigest() == (
        _LEGACY_HELP_SHA256
    )
    assert hashlib.sha256(defaults).hexdigest() == _LEGACY_DEFAULTS_SHA256
