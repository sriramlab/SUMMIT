"""Closed V1 artifact for generalized per-variant GxE LD-score references."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from summit.context._artifact_io import (
    StableNpzReader,
    _preflight_stable_npz_members,
    _publish_stable_npz_no_replace,
    _validate_stable_npz_writer_temp,
    fsync_parent_directory,
)
from summit.context.reference import ReferenceMoments
from summit.context.schema import GenotypeScalePlanV1
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    canonical_json,
    canonical_sha256,
    freeze_context_mapping,
)
from summit.ldscore.generalized_gxe_variant import (
    GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
    GlobalVariantProbeSpec,
    _scale_plan_from_manifest,
    build_generalized_gxe_variant_manifest,
    validate_generalized_gxe_variant_manifest,
)


GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX = (
    ".generalized-gxe-variant-ldscore-v1.npz"
)
_FAMILY = "Generalized GxE variant-LD-score reference V1"
_REQUIRED_ARRAY_NAMES = (
    "directed_numerator",
    "symmetric_numerator",
    "genetic_gram",
    "block_directed_numerator",
    "block_annotation_mass",
    "same_person",
)
_OPTIONAL_ARRAY_NAMES = (
    "deleted_genetic_gram",
    "directional_ldscores",
)


def _owned_readonly(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=invalid_constant,
    )


def _artifact_arrays(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
) -> dict[str, np.ndarray]:
    result = {
        name: getattr(artifact, name) for name in _REQUIRED_ARRAY_NAMES
    }
    for name in _OPTIONAL_ARRAY_NAMES:
        value = getattr(artifact, name)
        if value is not None:
            result[name] = value
    return result


@dataclass(frozen=True)
class GeneralizedGxEVariantReferenceArtifactV1:
    """Owned immutable aggregate reference plus optional inline SNP panel."""

    manifest: Mapping[str, Any]
    directed_numerator: np.ndarray
    symmetric_numerator: np.ndarray
    genetic_gram: np.ndarray
    block_directed_numerator: np.ndarray
    block_annotation_mass: np.ndarray
    same_person: np.ndarray
    deleted_genetic_gram: np.ndarray | None = None
    directional_ldscores: np.ndarray | None = None
    component_index: ContextComponentIndex = field(init=False)
    scale_plan: GenotypeScalePlanV1 = field(init=False)
    block_labels: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        for name in _REQUIRED_ARRAY_NAMES:
            object.__setattr__(self, name, _owned_readonly(getattr(self, name)))
        for name in _OPTIONAL_ARRAY_NAMES:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _owned_readonly(value))
        manifest = _strict_json_loads(canonical_json(self.manifest))
        validate_generalized_gxe_variant_manifest(
            manifest,
            arrays=_artifact_arrays(self),
        )
        axes = manifest["axes"]
        pair_index = ContextPairIndex(len(axes["basis"]["names"]))
        component_index = ContextComponentIndex(
            tuple(axes["annotations"]["names"]), pair_index
        )
        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(
            self,
            "scale_plan",
            _scale_plan_from_manifest(manifest["genotype_scale_plan"]),
        )
        object.__setattr__(
            self,
            "block_labels",
            tuple(axes["jackknife_blocks"]["block_labels"]),
        )
        object.__setattr__(self, "manifest", freeze_context_mapping(manifest))

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.manifest)

    @property
    def n_samples(self) -> int:
        return int(self.manifest["axes"]["samples"]["count"])

    @property
    def reference_n(self) -> int:
        return self.n_samples

    @property
    def n_variants(self) -> int:
        return int(self.manifest["axes"]["variants"]["count"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["axes"]["fixed_effects"]["residual_rank"])

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self.block_labels

    @property
    def annotation_masses(self) -> np.ndarray:
        result = np.asarray(
            self.manifest["axes"]["annotations"]["masses"],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @property
    def group_variant_counts(self) -> np.ndarray:
        block_ids = np.asarray(
            self.manifest["axes"]["jackknife_blocks"]["variant_block_ids"],
            dtype=np.int64,
        )
        result = np.bincount(block_ids, minlength=len(self.block_labels))
        result.setflags(write=False)
        return result

    @property
    def full_moments(self) -> ReferenceMoments:
        return ReferenceMoments(
            annotation_masses=self.annotation_masses,
            gram=self.genetic_gram,
            same_person=self.same_person,
        )

    def verify(self) -> None:
        validate_generalized_gxe_variant_manifest(
            self.manifest,
            arrays=_artifact_arrays(self),
        )


def build_generalized_gxe_variant_reference_v1(
    *,
    axes: Mapping[str, Any],
    probe_spec: GlobalVariantProbeSpec,
    genotype_scale_plan: GenotypeScalePlanV1,
    arrays: Mapping[str, np.ndarray],
    pass_ledger: Mapping[str, Any],
    performance_ledger: Mapping[str, Any],
    provenance: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> GeneralizedGxEVariantReferenceArtifactV1:
    """Validate complete native outputs and construct an unpublished artifact."""
    owned = {name: _owned_readonly(value) for name, value in arrays.items()}
    manifest = build_generalized_gxe_variant_manifest(
        axes=axes,
        probe_spec=probe_spec,
        arrays=owned,
        pass_ledger=pass_ledger,
        genotype_scale_plan=genotype_scale_plan,
        performance_ledger=performance_ledger,
        provenance=provenance,
        diagnostics=diagnostics,
    )
    return GeneralizedGxEVariantReferenceArtifactV1(
        manifest=manifest,
        **{name: owned.get(name) for name in (*_REQUIRED_ARRAY_NAMES, *_OPTIONAL_ARRAY_NAMES)},
    )


def _canonical_native_pass_ledger(
    value: Mapping[str, Any], *, num_variants: int
) -> dict[str, int]:
    ledger = dict(value)
    canonical_names = {
        "planned_reference_genotype_passes",
        "observed_reference_genotype_passes",
        "planned_retained_variant_visits",
        "observed_retained_variant_visits",
        "duplicate_retained_variant_visits",
        "pass1_decoded_blocks",
        "pass2_decoded_blocks",
        "retry_count",
        "repair_count",
        "fallback_count",
        "integrity_failure_count",
    }
    if canonical_names <= set(ledger):
        return {name: int(ledger[name]) for name in canonical_names}
    block_reads = ledger.get("observed_block_reads")
    if (
        isinstance(block_reads, bool)
        or not isinstance(block_reads, int)
        or block_reads < 2
        or block_reads % 2 != 0
    ):
        raise ValueError("native block-read ledger cannot be split across two passes")
    result = {
        "planned_reference_genotype_passes": int(
            ledger.get("planned_reference_genotype_passes", -1)
        ),
        "observed_reference_genotype_passes": int(
            ledger.get("observed_reference_genotype_passes", -1)
        ),
        "planned_retained_variant_visits": int(
            ledger.get("planned_retained_variant_visits", -1)
        ),
        "observed_retained_variant_visits": int(
            ledger.get("observed_retained_variant_visits", -1)
        ),
        "duplicate_retained_variant_visits": int(
            ledger.get("duplicate_variant_visits", -1)
        ),
        "pass1_decoded_blocks": block_reads // 2,
        "pass2_decoded_blocks": block_reads // 2,
        "retry_count": int(ledger.get("retry_count", -1)),
        "repair_count": int(ledger.get("repair_count", -1)),
        "fallback_count": int(ledger.get("fallback_count", -1)),
        "integrity_failure_count": int(ledger.get("integrity_failures", -1)),
    }
    if (
        result["observed_reference_genotype_passes"] != 2
        or result["observed_retained_variant_visits"] != 2 * num_variants
    ):
        raise ValueError("native result is not a complete two-pass execution")
    return result


def build_generalized_gxe_variant_reference_from_native_v1(
    native_result: Any,
    *,
    axes: Mapping[str, Any],
    probe_spec: GlobalVariantProbeSpec,
    genotype_scale_plan: GenotypeScalePlanV1,
    performance_ledger: Mapping[str, Any],
    provenance: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    include_directional_panel: bool = False,
) -> GeneralizedGxEVariantReferenceArtifactV1:
    """Publishable adapter from the Stage 06 native result, without genotypes."""
    if not isinstance(include_directional_panel, bool):
        raise ValueError("native artifact inclusion policy must be boolean")
    native_scale_plan = getattr(native_result, "genotype_scale_plan", None)
    if not isinstance(native_scale_plan, GenotypeScalePlanV1):
        raise ValueError("native result lacks a sealed genotype scale plan")
    if native_scale_plan.digest != genotype_scale_plan.digest:
        raise ValueError("native and requested genotype scale plans differ")
    retained_digest = str(axes["variants"]["retained_order_digest"])
    if retained_digest != native_scale_plan.retained_variant_order_sha256:
        raise ValueError(
            "native genotype scale plan and retained variant axis differ"
        )
    required = {
        "directed_numerator": "directed_numerator",
        "symmetric_numerator": "symmetric_numerator",
        "genetic_gram": "genetic_gram",
        "block_directed_numerator": "block_directed_numerator",
        "block_annotation_mass": "block_annotation_mass",
        "same_person": "same_person",
    }
    arrays: dict[str, np.ndarray] = {}
    for output_name, attribute in required.items():
        if not hasattr(native_result, attribute):
            raise ValueError(f"native result lacks {attribute!r}")
        arrays[output_name] = np.asarray(getattr(native_result, attribute))
    if include_directional_panel:
        arrays["directional_ldscores"] = np.asarray(
            getattr(native_result, "directional_ldscores")
        )
    raw_ledger = getattr(native_result, "ledger", None)
    if not isinstance(raw_ledger, Mapping):
        raise ValueError("native result lacks a pass ledger")
    pass_ledger = _canonical_native_pass_ledger(
        raw_ledger,
        num_variants=int(axes["variants"]["count"]),
    )
    return build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe_spec,
        genotype_scale_plan=genotype_scale_plan,
        arrays=arrays,
        pass_ledger=pass_ledger,
        performance_ledger=performance_ledger,
        provenance=provenance,
        diagnostics=diagnostics,
    )


def reference_moments_after_deleting_variant_blocks_v1(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
    blocks: Sequence[str],
) -> ReferenceMoments:
    """Subtract fixed full-genome target rows without any genotype access."""
    if not isinstance(artifact, GeneralizedGxEVariantReferenceArtifactV1):
        raise ValueError("artifact must be a generalized variant-LD-score reference")
    artifact.verify()
    return _reference_moments_after_deleting_variant_blocks_prevalidated_v1(
        artifact, blocks
    )


def _reference_moments_after_deleting_variant_blocks_prevalidated_v1(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
    blocks: Sequence[str],
) -> ReferenceMoments:
    """Subtract target rows after an enclosing artifact verification."""
    if isinstance(blocks, (str, bytes)):
        raise ValueError("deleted variant blocks must be a sequence")
    requested = tuple(blocks)
    if any(not isinstance(value, str) or not value for value in requested):
        raise ValueError("deleted variant block labels must be nonempty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("deleted variant block labels must be unique")
    unknown = set(requested) - set(artifact.block_labels)
    if unknown:
        raise ValueError(f"unknown variant blocks: {sorted(unknown)}")
    if not requested:
        return artifact.full_moments
    indices = np.fromiter(
        (artifact.block_labels.index(value) for value in requested),
        dtype=np.int64,
        count=len(requested),
    )
    masses = artifact.annotation_masses
    retained_masses = masses - np.sum(
        artifact.block_annotation_mass[indices], axis=0, dtype=np.float64
    )
    if np.any(retained_masses <= 0.0):
        raise ValueError("variant-block deletion empties an annotation")
    directed = artifact.directed_numerator - np.sum(
        artifact.block_directed_numerator[indices], axis=0, dtype=np.float64
    )
    symmetric = 0.5 * (directed + directed.T)
    component_annotations = np.fromiter(
        (entry.annotation_index for entry in artifact.component_index.entries),
        dtype=np.int64,
        count=len(artifact.component_index),
    )
    component_masses = retained_masses[component_annotations]
    gram = (
        float(artifact.residual_rank**2)
        * symmetric
        / np.outer(component_masses, component_masses)
    )
    return ReferenceMoments(
        annotation_masses=retained_masses,
        gram=gram,
        same_person=artifact.same_person,
    )


def write_generalized_gxe_variant_reference_v1(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
    output: str | Path,
) -> Path:
    """Atomically publish one complete strict V1 NPZ container."""
    if not isinstance(artifact, GeneralizedGxEVariantReferenceArtifactV1):
        raise ValueError("only the generalized variant-LD-score V1 artifact is accepted")
    artifact.verify()
    arrays = _artifact_arrays(artifact)
    manifest_json, manifest_sha256, admitted = _preflight_stable_npz_members(
        manifest_json=canonical_json(artifact.manifest),
        manifest_sha256=artifact.manifest_sha256,
        arrays=arrays,
        family=_FAMILY,
    )
    path = Path(output)
    if not path.name.endswith(GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX):
        path = Path(str(path) + GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                manifest_json=manifest_json,
                manifest_sha256=manifest_sha256,
                **admitted,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _validate_stable_npz_writer_temp(
            temporary_name,
            family=_FAMILY,
            manifest_json=manifest_json,
            manifest_sha256=manifest_sha256,
            arrays=admitted,
        )
        _publish_stable_npz_no_replace(temporary_name, path)
        fsync_parent_directory(path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_generalized_gxe_variant_reference_v1(
    path: str | Path,
) -> GeneralizedGxEVariantReferenceArtifactV1:
    """Load only the distinct generalized variant-LD-score artifact family."""
    source = Path(path)
    if not source.name.endswith(GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX):
        raise ValueError("generalized variant-LD-score loader requires its V1 suffix")
    try:
        with StableNpzReader(
            source,
            family=_FAMILY,
            maximum_members=len(_REQUIRED_ARRAY_NAMES) + len(_OPTIONAL_ARRAY_NAMES) + 2,
        ) as archive:
            manifest = _strict_json_loads(
                archive.read_text_scalar(
                    "manifest_json", maximum_bytes=16 * 1024 * 1024
                )
            )
            if manifest.get("kind") != GENERALIZED_GXE_VARIANT_REFERENCE_KIND:
                raise ValueError(
                    "sample-probe or legacy artifacts are not generalized variant LD scores"
                )
            digest = archive.read_text_scalar(
                "manifest_sha256", maximum_bytes=1024
            )
            if digest != canonical_sha256(manifest):
                raise ValueError("generalized reference manifest SHA-256 mismatch")
            metadata = manifest.get("numeric_arrays")
            if not isinstance(metadata, Mapping):
                raise ValueError("generalized reference numeric metadata is missing")
            names = set(metadata)
            required = set(_REQUIRED_ARRAY_NAMES)
            optional = set(_OPTIONAL_ARRAY_NAMES)
            if not required <= names or not names <= required | optional:
                raise ValueError("generalized reference numeric member set is invalid")
            specs: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
            for name, record in metadata.items():
                if (
                    not isinstance(record, Mapping)
                    or record.get("dtype") != "float64"
                    or record.get("order") != "C"
                    or not isinstance(record.get("shape"), list)
                ):
                    raise ValueError(f"generalized array metadata {name!r} is invalid")
                shape = tuple(record["shape"])
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                ):
                    raise ValueError(f"generalized array shape {name!r} is invalid")
                specs[name] = (np.dtype(np.float64).newbyteorder("<"), shape)
            archive.preflight_arrays(specs)
            arrays = {name: archive.load_array(name) for name in metadata}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and (
            str(exc).startswith(_FAMILY)
            or "generalized" in str(exc)
            or "sample-probe" in str(exc)
        ):
            raise
        raise ValueError("not a valid generalized variant-LD-score artifact") from exc
    return GeneralizedGxEVariantReferenceArtifactV1(
        manifest=manifest,
        **{
            name: arrays.get(name)
            for name in (*_REQUIRED_ARRAY_NAMES, *_OPTIONAL_ARRAY_NAMES)
        },
    )
