"""Sealed, non-interchangeable binary contextual moments and file preparation."""
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import os
import tempfile
import numpy as np

from summit.context.spec import array_sha256, canonical_sha256, canonical_json, freeze_context_mapping
from summit.prediction.spec import VariantAxis
from summit.ldscore.generalized_gxe_reference_v1 import _strict_json_loads
from .gxe import GxEMoments, prepare_gxe_moments, prepare_gxe_external, SCALE_CONTRACT
from .genotype import scaled_file_operator

KIND = "summit.pcgc.context_moments"
CONTRACT = "unprojected_population_scaled_context_offdiagonal_pcgc_v1"
ARRAYS = ("annotations", "ldscores", "rhs_rows", "population_second_moment")
KERNEL_ARRAY = "population_kernel_second_moment"
SAMPLING_ARRAYS = ("sampling_constant","sampling_linear","sampling_quadratic")
ARCHITECTURE_ARRAYS = ("architecture_left","architecture_right")
PAIR_ARRAYS = ("pair_constant","pair_linear","pair_quadratic")
FEATURE_ARRAYS = {
    "population_kernel_metric": (KERNEL_ARRAY,),
    "individual_sampling_covariance": SAMPLING_ARRAYS,
    "gaussian_architecture_covariance": ARCHITECTURE_ARRAYS,
    "reference_probe_covariance": ("reference_probe_deviations",),
    "external_reference_covariance": ("external_reference_covariance","external_context_gram"),
    "hoeffding_pair_covariance": PAIR_ARRAYS,
}


def _features(moments):
    result = []
    if moments.population_kernel_second_moment is not None:
        result.append("population_kernel_metric")
    if moments.sampling_moments is not None:
        result.append("individual_sampling_covariance")
        if moments.sampling_moments.architecture_probes:
            result.append("gaussian_architecture_covariance")
        if moments.sampling_moments.pair_constant is not None:
            result.append("hoeffding_pair_covariance")
    if moments.reference_probe_deviations is not None:
        result.append("reference_probe_covariance")
    if moments.external_reference_covariance is not None:
        result.append("external_reference_covariance")
    return tuple(sorted(result))


def _manifest_features(header):
    version = header.get("schema_version")
    if version == 1:
        if header.get("features"):
            raise ValueError("legacy contextual artifact cannot declare extension features")
        return ()
    features = header.get("features")
    if version != 2 or not isinstance(features,(list,tuple)) or any(not isinstance(f,str) for f in features):
        raise ValueError("unsupported contextual artifact feature metadata")
    if tuple(features) != tuple(sorted(set(features))) or set(features)-set(FEATURE_ARRAYS):
        raise ValueError("unsupported or duplicated contextual artifact features")
    if set(features) & {"gaussian_architecture_covariance", "reference_probe_covariance", "external_reference_covariance", "hoeffding_pair_covariance"} and "individual_sampling_covariance" not in features:
        raise ValueError("extended covariance requires individual-sampling moments")
    return tuple(features)


def _arrays(moments):
    return ARRAYS+tuple(name for feature in _features(moments) for name in FEATURE_ARRAYS[feature])


def _array_value(moments,name):
    if name in SAMPLING_ARRAYS:
        return getattr(moments.sampling_moments,name.removeprefix("sampling_"))
    if name in ARCHITECTURE_ARRAYS+PAIR_ARRAYS:
        return getattr(moments.sampling_moments,name)
    return getattr(moments,name)


def _names(values, count, kind):
    result = tuple(values)
    if len(result) != count or len(set(result)) != count or any(not isinstance(v, str) or not v for v in result):
        raise ValueError(f"invalid {kind} names")
    return result


@dataclass(frozen=True)
class GxEArtifact:
    moments: GxEMoments
    manifest: dict

    def __post_init__(self):
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        self.verify()

    def verify(self):
        h, m = self.manifest, self.moments
        if h.get("kind") != KIND or h.get("schema_version") not in (1,2) or h.get("contract") != CONTRACT:
            raise ValueError("not a supported contextual PCGC artifact")
        if _manifest_features(h) != _features(m):
            raise ValueError("contextual artifact features and arrays disagree")
        has_architecture = m.sampling_moments is not None and bool(m.sampling_moments.architecture_probes)
        if has_architecture and h.get("architecture_probes") != m.sampling_moments.architecture_probes:
            raise ValueError("architecture probe counts disagree")
        for name in ("method", "n_samples", "num_contexts", "population_liability_variance"):
            if h.get(name) != getattr(m, name):
                raise ValueError(f"contextual PCGC {name} metadata disagree")
        if h.get("liability_scale_contract") != SCALE_CONTRACT:
            raise ValueError("unsupported liability scale contract")
        _names(h.get("annotation_names", ()), m.annotations.shape[1], "annotation")
        _names(h.get("context_names", ()), m.num_contexts, "context")
        for name in ("variant_axis", "array_hashes", "risk", "diagnostics"):
            if not isinstance(h.get(name), Mapping):
                raise ValueError(f"contextual PCGC {name} must be a mapping")
        if set(h["variant_axis"]) != {"ids", "chromosome", "position", "counted", "other", "genome_build"}:
            raise ValueError("contextual variant metadata are incomplete")
        if len(VariantAxis(**h["variant_axis"]).ids) != len(m.annotations):
            raise ValueError("contextual variant axis has the wrong length")
        for name in ("sample_identity", "genotype_scale_identity", "risk_identity", "context_identity", "liability_sd_identity"):
            value = h.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid {name}")
        if set(h["array_hashes"]) != set(_arrays(m)):
            raise ValueError("contextual PCGC array hash names disagree")
        for name in _arrays(m):
            if h["array_hashes"].get(name) != array_sha256(_array_value(m,name)):
                raise ValueError(f"contextual {name} checksum mismatch")
        for name in ("population_prevalence", "sample_prevalence"):
            value = h["risk"].get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("invalid contextual risk metadata")
        if h.get("manifest_hash") != canonical_sha256({k:v for k,v in h.items() if k != "manifest_hash"}):
            raise ValueError("contextual PCGC metadata checksum mismatch")


def make_gxe_artifact(moments, *, variant_axis, annotation_names, context_names,
                      sample_identity, genotype_scale_identity, risk, contexts, liability_sd, diagnostics):
    risk_identity = canonical_sha256({"diagnostics": risk.diagnostics(),
        **{k: array_sha256(getattr(risk,k)) for k in ("population_risk", "sample_risk", "z", "sensitivity", "coefficients")}})
    sd = np.broadcast_to(np.asarray(liability_sd, dtype=float), (risk.n_samples,))
    # BinaryRisk's unit-scale diagnostic describes its normalized thresholds;
    # the GxE scale is separately sealed and must not be confused with it.
    features = _features(moments)
    version = 2 if features else 1
    h = dict(kind=KIND, schema_version=version, contract=CONTRACT,
        method=moments.method, n_samples=moments.n_samples, num_contexts=moments.num_contexts,
        population_liability_variance=moments.population_liability_variance,
        liability_scale_contract=SCALE_CONTRACT, liability_sd_identity=array_sha256(sd),
        context_identity=array_sha256(np.asarray(contexts, dtype=float)),
        variant_axis=variant_axis.to_dict(), annotation_names=list(annotation_names),
        context_names=list(context_names), sample_identity=sample_identity,
        genotype_scale_identity=genotype_scale_identity, risk_identity=risk_identity,
        risk=risk.diagnostics(), diagnostics=diagnostics,
        array_hashes={k:array_sha256(_array_value(moments,k)) for k in _arrays(moments)})
    if features:
        h["features"] = list(features)
    if "gaussian_architecture_covariance" in features:
        h["architecture_probes"] = moments.sampling_moments.architecture_probes
    h["manifest_hash"] = canonical_sha256(h)
    return GxEArtifact(moments, h)


def write_gxe_artifact(artifact, path):
    artifact.verify()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, manifest_json=np.asarray(canonical_json(artifact.manifest)),
                                **{k:_array_value(artifact.moments,k) for k in _arrays(artifact.moments)})
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def load_gxe_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        if "manifest_json" not in data:
            raise ValueError("contextual PCGC archive members do not match the contract")
        h = _strict_json_loads(str(data["manifest_json"].item()))
        if not isinstance(h, dict) or h.get("kind") != KIND:
            raise ValueError("not a contextual PCGC artifact")
        features = _manifest_features(h)
        arrays = ARRAYS+tuple(name for feature in features for name in FEATURE_ARRAYS[feature])
        if len(data.files) != len(arrays)+1 or set(data.files) != set(arrays)|{"manifest_json"}:
            raise ValueError("contextual PCGC archive members do not match the contract")
        return _artifact_from_arrays(h, data)


def _artifact_from_arrays(h, data):
    """Reconstruct contextual moments after checking archive members."""
    features = _manifest_features(h)
    arrays = ARRAYS+tuple(name for feature in features for name in FEATURE_ARRAYS[feature])
    names = ("n_samples", "num_contexts", "method", "population_liability_variance")
    if not set(names) <= set(h):
        raise ValueError("contextual PCGC moment metadata are incomplete")
    values = {k:data[k] for k in arrays if k not in SAMPLING_ARRAYS+ARCHITECTURE_ARRAYS+PAIR_ARRAYS}
    if "individual_sampling_covariance" in features:
        from .sampling import SamplingMoments
        architecture = ({**{k:data[k] for k in ARCHITECTURE_ARRAYS},"architecture_probes":h.get("architecture_probes")} if "gaussian_architecture_covariance" in features else {})
        if "hoeffding_pair_covariance" in features: architecture.update({k:data[k] for k in PAIR_ARRAYS})
        values["sampling_moments"] = SamplingMoments(**{k.removeprefix("sampling_"):data[k] for k in SAMPLING_ARRAYS},n_samples=h["n_samples"],**architecture)
    m = GxEMoments(**values, **{k:h[k] for k in names})
    return GxEArtifact(m, h)


def is_gxe_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        if "manifest_json" not in data:
            return False
        h = _strict_json_loads(str(data["manifest_json"].item()))
        return isinstance(h, dict) and h.get("kind") == KIND


def prepare_gxe_from_source(source, scale, rows, risk, annotations, contexts, *,
                            annotation_names, context_names, liability_sd, method="pcgc",
                            reference_source=None, variant_indices=None,
                            genotype_covariates=None, reference_genotype_covariates=None,
                            block_size=256, threads=1, native=True, **options):
    _names(annotation_names, np.asarray(annotations).shape[1], "annotation")
    _names(context_names, np.asarray(contexts).shape[1], "context")
    if len(rows) != risk.n_samples:
        raise ValueError("contextual sample rows must align to risks")
    if (method == "pcgc-ld") != (reference_source is not None):
        raise ValueError("pcgc-ld needs an independent reference; study methods do not")
    common = dict(block_size=block_size, threads=threads, native=native)
    if reference_genotype_covariates is not None and (reference_source is None or genotype_covariates is None):
        raise ValueError("reference genotype covariates require an external reference and study genotype covariates")
    if reference_source is not None and genotype_covariates is not None and reference_genotype_covariates is None:
        raise ValueError("external PC-adjusted analysis requires aligned reference genotype covariates")
    op = scaled_file_operator(source, scale, rows, variant_indices=variant_indices, **common)
    adjustment_bytes = 0
    if genotype_covariates is not None:
        from .ancestry import AncestryAdjustedOperator
        op = AncestryAdjustedOperator(op,genotype_covariates,threads=threads,native=native)
        adjustment_bytes = op.workspace_bytes(block_size)
    ref = None
    if method == "pcgc-ld":
        if set(source.samples[i] for i in rows) & set(reference_source.samples):
            raise ValueError("external reference must not overlap the ascertained study")
        ref = scaled_file_operator(reference_source, scale, np.arange(len(reference_source.samples)),
                                   variant_indices=variant_indices, **common)
        if reference_genotype_covariates is not None:
            ref = AncestryAdjustedOperator(ref,reference_genotype_covariates,threads=threads,native=native)
            if ref.u.shape[1] != op.u.shape[1]:
                raise ValueError("study and reference ancestry designs must have the same column count")
            adjustment_bytes += ref.workspace_bytes(block_size)
    if adjustment_bytes:
        memory = options.get("memory_bytes",2**30)
        if type(memory) is not int or memory <= adjustment_bytes:
            raise MemoryError("PCGC memory budget cannot accommodate ancestry adjustment")
        options["memory_bytes"] = memory-adjustment_bytes
    if method == "pcgc-ld":
        moments, diagnostics = prepare_gxe_external(op, ref, annotations, risk, contexts,
                                                    liability_sd=liability_sd, **common, **options)
        diagnostics["reference_sample_identity"] = canonical_sha256({"samples":reference_source.samples})
    else:
        moments, diagnostics = prepare_gxe_moments(op, annotations, risk, contexts, method,
                                                   liability_sd=liability_sd, **common, **options)
    if adjustment_bytes:
        diagnostics["genotype_ancestry_adjustment"] = op.diagnostics
        diagnostics["ancestry_workspace_bytes"] = adjustment_bytes
        diagnostics["peak_planned_workspace_bytes"] += adjustment_bytes
        if reference_genotype_covariates is not None:
            diagnostics["reference_genotype_ancestry_adjustment"] = ref.diagnostics
    axis = source.variants if variant_indices is None else source.variants.subset(variant_indices)
    return make_gxe_artifact(moments, variant_axis=axis, annotation_names=annotation_names,
        context_names=context_names, sample_identity=canonical_sha256({"samples":[source.samples[i] for i in rows]}),
        genotype_scale_identity=scale.identity, risk=risk, contexts=contexts, liability_sd=liability_sd,
        diagnostics=diagnostics)
