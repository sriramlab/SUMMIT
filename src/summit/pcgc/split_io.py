"""Separate PCGC reference and trait files with content-based matching."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile

import numpy as np

from summit.context.spec import canonical_json, canonical_sha256
from summit.ldscore.generalized_gxe_reference_v1 import _strict_json_loads
from . import artifacts as additive
from . import gxe_io as contextual

SUMSTATS_SUFFIX = ".binary.sumstats.npz"
LDSCORES_SUFFIX = ".binary.ldscores.npz"
SUMSTATS_KIND = "summit.pcgc.sumstats"
REFERENCE_KIND = "summit.pcgc.reference"
_REFERENCE_ARRAYS = frozenset((
    "annotations", "ldscores", "same_person", "reference_probe_deviations",
    "external_reference_covariance", "external_context_gram",
))
_SHARED_FIELDS = (
    "kind", "schema_version", "contract", "method", "n_samples", "variant_axis",
    "annotation_names", "sample_identity", "genotype_scale_identity", "risk_identity",
)
_CONTEXT_FIELDS = (
    "num_contexts", "context_names", "context_identity", "liability_sd_identity",
    "liability_scale_contract",
)


def output_paths(prefix):
    """Return summary and reference paths for a preparation prefix."""
    return Path(str(prefix) + SUMSTATS_SUFFIX), Path(str(prefix) + LDSCORES_SUFFIX)


def _seal(header):
    return {**header, "manifest_hash": canonical_sha256(header)}


def _verify_hash(header):
    if not isinstance(header, Mapping) or header.get("manifest_hash") != canonical_sha256(
        {k: v for k, v in header.items() if k != "manifest_hash"}
    ):
        raise ValueError("PCGC metadata checksum mismatch")


def _read_manifest(archive):
    if "manifest_json" not in archive.files:
        raise ValueError("PCGC input has no manifest; prepare PCGC summary statistics and LD scores")
    value = archive["manifest_json"]
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError("PCGC manifest must be a scalar JSON string")
    header = _strict_json_loads(str(value.item()))
    if not isinstance(header, dict):
        raise ValueError("PCGC manifest must be a JSON object")
    _verify_hash(header)
    return header


def _layout(header):
    """Check the original moment metadata and partition its numeric arrays."""
    _verify_hash(header)
    if header.get("kind") == additive.KIND:
        arrays = additive.ARRAYS
        required = (*_SHARED_FIELDS, "covariate_variance")
    elif header.get("kind") == contextual.KIND:
        features = contextual._manifest_features(header)
        arrays = contextual.ARRAYS + tuple(
            name for feature in features for name in contextual.FEATURE_ARRAYS[feature]
        )
        required = (*_SHARED_FIELDS, *_CONTEXT_FIELDS, "population_liability_variance")
    else:
        raise ValueError("unsupported PCGC moment format")
    if not set(required) <= set(header):
        raise ValueError("PCGC preparation metadata are incomplete")
    hashes = header.get("array_hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != set(arrays):
        raise ValueError("PCGC array metadata disagree with the moment format")
    reference = tuple(name for name in arrays if name in _REFERENCE_ARRAYS)
    summary = tuple(name for name in arrays if name not in _REFERENCE_ARRAYS)
    return summary, reference


def _reference_manifest(header, reference_arrays):
    fields = (*_SHARED_FIELDS, *_CONTEXT_FIELDS) if header["kind"] == contextual.KIND else _SHARED_FIELDS
    definition = {name: header[name] for name in fields}
    if header["kind"] == additive.KIND:
        definition["ldscore_contract"] = (
            additive.LEGACY_DIAGONAL if header["schema_version"] == 1 else header.get("ldscore_contract")
        )
    # PC adjustment precedes context/risk weighting; retain its definition too.
    diagnostics = header.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise ValueError("PCGC diagnostics must be a mapping")
    for name in ("genotype_ancestry_adjustment", "reference_genotype_ancestry_adjustment"):
        if name in diagnostics:
            definition[name] = diagnostics[name]
    return _seal(dict(
        kind=REFERENCE_KIND, schema_version=1, definition=definition,
        array_hashes={name: header["array_hashes"][name] for name in reference_arrays},
    ))


def _check_members(archive, arrays):
    names = (*arrays, "manifest_json")
    if len(archive.files) != len(names) or set(archive.files) != set(names):
        raise ValueError("PCGC archive arrays do not match its declared file role")


def write_split_artifact(artifact, sumstats_path, ldscores_path):
    """Write two checked files without copying arrays or overwriting outputs.

    Both archives finish writing before either is published. A handled failure
    removes only files created by this call. After an unhandled interruption,
    an incomplete pair cannot be used for fitting.
    """
    summary_path, reference_path = Path(sumstats_path), Path(ldscores_path)
    if summary_path.resolve() == reference_path.resolve():
        raise ValueError("PCGC summary and reference paths must differ")
    for path in (summary_path, reference_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    artifact.verify()
    original = artifact.manifest
    summary_arrays, reference_arrays = _layout(original)
    reference = _reference_manifest(original, reference_arrays)
    summary = _seal(dict(
        kind=SUMSTATS_KIND, schema_version=1,
        reference_hash=reference["manifest_hash"], preparation=original,
    ))
    value = (lambda name: contextual._array_value(artifact.moments, name)) if original["kind"] == contextual.KIND else (
        lambda name: getattr(artifact.moments, name)
    )
    temporary = []
    published = []
    try:
        # Publish the summary last; it identifies the already completed reference.
        for path, header, names in (
            (reference_path, reference, reference_arrays),
            (summary_path, summary, summary_arrays),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            temporary.append((path, temp))
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, manifest_json=np.asarray(canonical_json(header)),
                                    **{name: value(name) for name in names})
                handle.flush()
                os.fsync(handle.fileno())
        for path, temp in temporary:
            os.link(temp, path)
            published.append((path, temp))
    except BaseException:
        for path, temp in reversed(published):
            try:
                owned = os.stat(temp)
                current = os.stat(path, follow_symlinks=False)
                if (owned.st_dev, owned.st_ino) == (current.st_dev, current.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temp in temporary:
            os.unlink(temp)
    return summary_path, reference_path


def load_split_artifact(sumstats_path, ldscores_path):
    """Reject incompatible metadata before loading the large numeric arrays."""
    with ExitStack() as stack:
        summary = stack.enter_context(np.load(sumstats_path, allow_pickle=False))
        reference = stack.enter_context(np.load(ldscores_path, allow_pickle=False))
        sh, rh = _read_manifest(summary), _read_manifest(reference)
        if sh.get("kind") != SUMSTATS_KIND or rh.get("kind") != REFERENCE_KIND:
            raise ValueError("use PCGC summary statistics with --h2 and PCGC reference LD scores with --ldscores")
        if type(sh.get("schema_version")) is not int or sh["schema_version"] != 1 or type(rh.get("schema_version")) is not int or rh["schema_version"] != 1:
            raise ValueError("unsupported separate PCGC file version")
        if set(sh) != {"kind", "schema_version", "reference_hash", "preparation", "manifest_hash"}:
            raise ValueError("unsupported PCGC summary metadata")
        original = sh["preparation"]
        summary_arrays, reference_arrays = _layout(original)
        expected = _reference_manifest(original, reference_arrays)
        if sh["reference_hash"] != expected["manifest_hash"]:
            raise ValueError("PCGC summary reference identity disagrees with its preparation metadata")
        if rh != expected:
            observed = rh.get("definition", {})
            changed = [name for name, value in expected["definition"].items()
                       if not isinstance(observed, Mapping) or observed.get(name) != value]
            detail = ", ".join(changed) if changed else "LD scores or reference uncertainty arrays"
            raise ValueError(f"incompatible PCGC reference ({detail}); use the reference saved with these summary statistics")
        _check_members(summary, summary_arrays)
        _check_members(reference, reference_arrays)
        arrays = {name: summary[name] for name in summary_arrays}
        arrays.update({name: reference[name] for name in reference_arrays})
    module = contextual if original["kind"] == contextual.KIND else additive
    return module._artifact_from_arrays(original, arrays)


def load_fit_artifact(sumstats_path, ldscores_path=None):
    """Read separate inputs or an existing combined archive."""
    if ldscores_path is not None:
        return load_split_artifact(sumstats_path, ldscores_path)
    with np.load(sumstats_path, allow_pickle=False) as archive:
        header = _read_manifest(archive)
    if header.get("kind") == SUMSTATS_KIND:
        raise ValueError("separate PCGC summary statistics require --ldscores with their matching reference")
    if header.get("kind") == contextual.KIND:
        return contextual.load_gxe_artifact(sumstats_path)
    if header.get("kind") == additive.KIND:
        return additive.load_artifact(sumstats_path)
    raise ValueError("--h2 requires PCGC summary statistics or a combined .binary.npz file")
