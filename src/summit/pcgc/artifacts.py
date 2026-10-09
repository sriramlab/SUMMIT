"""Typed joint PCGC moments; no logistic/OLS summary-statistic coercion."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import numpy as np

from summit.context.spec import array_sha256, canonical_json, canonical_sha256, freeze_context_mapping
from summit.prediction.spec import VariantAxis
from summit.ldscore.generalized_gxe_reference_v1 import _strict_json_loads
from .moments import BinaryMoments, OFFDIAGONAL, LEGACY_DIAGONAL

KIND = "summit.pcgc.joint_moments"
CONTRACT = "unprojected_population_scale_offdiagonal_pcgc_v1"
ARRAYS = ("annotations", "ldscores", "same_person", "rhs_rows")


@dataclass(frozen=True)
class BinaryArtifact:
    moments: BinaryMoments
    manifest: dict

    def __post_init__(self):
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        self.verify()

    def verify(self):
        m, header = self.moments, self.manifest
        if header.get("kind") != KIND or header.get("schema_version") not in (1, 2) or header.get("contract") != CONTRACT:
            raise ValueError("not a supported typed PCGC artifact; prepare raw binary scores first")
        row_contract = LEGACY_DIAGONAL if header["schema_version"] == 1 else header.get("ldscore_contract")
        if row_contract != m.ldscore_contract:
            raise ValueError("PCGC LD row contract disagrees with its manifest")
        if header.get("method") != m.method or header.get("n_samples") != m.n_samples:
            raise ValueError("PCGC method or sample count disagrees with its manifest")
        if header.get("covariate_variance") != m.covariate_variance:
            raise ValueError("PCGC liability scale metadata disagree")
        required = {"variant_axis", "annotation_names", "array_hashes", "risk", "diagnostics"}
        if not required <= set(header):
            raise ValueError("PCGC manifest is missing required metadata")
        if any(not isinstance(header[key], Mapping) for key in ("variant_axis", "array_hashes", "risk", "diagnostics")):
            raise ValueError("PCGC manifest metadata must be mappings")
        if set(header["variant_axis"]) != {"ids", "chromosome", "position", "counted", "other", "genome_build"}:
            raise ValueError("PCGC variant axis metadata are incomplete or unsupported")
        axis = VariantAxis(**header["variant_axis"])
        if len(axis.ids) != len(m.rhs_rows):
            raise ValueError("PCGC variant axis has the wrong size")
        names = header["annotation_names"]
        if len(names) != m.annotations.shape[1] or len(set(names)) != len(names) or any(not isinstance(s, str) or not s for s in names):
            raise ValueError("invalid annotation names")
        for key in ("sample_identity", "genotype_scale_identity", "risk_identity"):
            value = header.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"missing or malformed {key}")
        for name in ARRAYS:
            if header["array_hashes"].get(name) != array_sha256(getattr(m, name)):
                raise ValueError(f"PCGC {name} checksum mismatch")
        sealed = {k: v for k, v in header.items() if k != "manifest_hash"}
        if header.get("manifest_hash") != canonical_sha256(sealed):
            raise ValueError("PCGC metadata checksum mismatch")
        if header.get("score_contract") != "raw_cross_product_squared_minus_exact_same_person_v1":
            raise ValueError("unsupported binary score contract")
        risk = header["risk"]
        for key in ("population_prevalence", "sample_prevalence"):
            value = risk.get(key)
            if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("invalid binary prevalence metadata")
        if risk.get("covariate_variance") != m.covariate_variance or risk.get("liability_residual_variance") != 1.:
            raise ValueError("binary risk scale metadata disagree")


def make_artifact(moments, *, variant_axis, annotation_names, sample_identity,
                  genotype_scale_identity, risk, diagnostics):
    risk_identity = canonical_sha256({
        "diagnostics": risk.diagnostics(),
        **{k: array_sha256(getattr(risk, k)) for k in ("population_risk", "sample_risk", "z", "sensitivity", "coefficients")}})
    if moments.ldscore_contract != OFFDIAGONAL:
        raise ValueError("new PCGC artifacts require offdiagonal SNP rows")
    header = dict(kind=KIND, schema_version=2, contract=CONTRACT, ldscore_contract=OFFDIAGONAL, method=moments.method,
                  n_samples=moments.n_samples, covariate_variance=moments.covariate_variance,
                  variant_axis=variant_axis.to_dict(), annotation_names=list(annotation_names),
                  sample_identity=sample_identity, genotype_scale_identity=genotype_scale_identity,
                  risk_identity=risk_identity, risk=risk.diagnostics(), diagnostics=diagnostics,
                  score_contract="raw_cross_product_squared_minus_exact_same_person_v1",
                  array_hashes={k: array_sha256(getattr(moments, k)) for k in ARRAYS})
    header["manifest_hash"] = canonical_sha256(header)
    return BinaryArtifact(moments, header)


def write_artifact(artifact, path):
    artifact.verify()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, manifest_json=np.asarray(canonical_json(artifact.manifest)),
                                **{k: getattr(artifact.moments, k) for k in ARRAYS})
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def load_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != len(ARRAYS)+1 or set(data.files) != set(ARRAYS) | {"manifest_json"}:
            raise ValueError("typed PCGC archive members do not match the contract")
        header = _strict_json_loads(str(data["manifest_json"].item()))
        if not isinstance(header, dict) or header.get("kind") != KIND:
            raise ValueError("not a typed PCGC artifact; OLS/logistic statistics cannot be converted implicitly")
        if not {"n_samples", "method", "covariate_variance"} <= set(header):
            raise ValueError("typed PCGC manifest is missing moment metadata")
        return _artifact_from_arrays(header, data)


def _artifact_from_arrays(header, data):
    """Apply the same moment and metadata checks to combined or separate files."""
    moments = BinaryMoments(**{k: data[k] for k in ARRAYS}, n_samples=header["n_samples"],
                            method=header["method"], covariate_variance=header["covariate_variance"],
                            ldscore_contract=LEGACY_DIAGONAL if header.get("schema_version") == 1 else header.get("ldscore_contract"))
    return BinaryArtifact(moments, header)
