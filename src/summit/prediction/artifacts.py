"""Closed, bounded model artifacts with completion-last, no-replace publication."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ._validation import array, canonical, closed, digest, identifier, metadata, psd
from .priors import ResponseGeometry
from .spec import GenotypeScale, VariantAxis

KIND = "summit.prediction.models"
VERSION = 1


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, *, max_bytes=128*2**20):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("JSON input is a symlink or exceeds the metadata limit")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite JSON value")
    with path.open() as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=bad_constant)


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(canonical(value)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def array_record(path):
    a = np.load(path, allow_pickle=False, mmap_mode="r", max_header_size=16384)
    return dict(file=path.name, shape=list(a.shape), dtype=a.dtype.str,
                bytes=path.stat().st_size, sha256=file_digest(path))


def load_array(root, record, *, expected_shape=None, expected_dtype=None):
    closed(record, ("file", "shape", "dtype", "bytes", "sha256"), name="array record")
    name = record["file"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npy"):
        raise ValueError("unsafe array member path")
    path = Path(root)/name
    if path.is_symlink():
        raise ValueError("array members cannot be symlinks")
    before = path.stat()
    shape = record["shape"]
    if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError("invalid array shape")
    dtype = np.dtype(record["dtype"])
    if dtype.kind not in "fiu" or dtype.hasobject or dtype.itemsize > 8:
        raise ValueError("unsupported artifact dtype")
    size = int(np.prod(shape, dtype=object))*dtype.itemsize
    if before.st_size != record["bytes"] or not size+10 <= before.st_size <= size+16384:
        raise ValueError("array byte length disagrees with declared dimensions")
    if expected_shape is not None and tuple(shape) != tuple(expected_shape):
        raise ValueError("array shape disagrees with model axes")
    if expected_dtype is not None and dtype != np.dtype(expected_dtype):
        raise ValueError("array dtype disagrees with model contract")
    if file_digest(path) != record["sha256"]:
        raise ValueError("array checksum mismatch")
    a = np.load(path, allow_pickle=False, mmap_mode="r", max_header_size=16384)
    if a.shape != tuple(shape) or a.dtype != dtype:
        raise ValueError("NPY header disagrees with manifest")
    for begin in range(0, a.shape[0] if a.ndim else 1, 8192):
        if not np.all(np.isfinite(a[begin:begin+8192] if a.ndim else a)):
            raise ValueError("artifact contains nonfinite numeric values")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("array changed during validation")
    return a


@dataclass(frozen=True)
class PredictionModel:
    trait_id: str
    model_id: str
    variants: VariantAxis
    scale: GenotypeScale
    weights: np.ndarray
    fixed_coefficients: np.ndarray
    covariance: np.ndarray
    context_spec: dict
    fixed_spec: dict
    phenotype_spec: dict
    prior_spec: dict
    convergence: dict
    provenance: dict
    geometry: ResponseGeometry | None = None
    identity: str = ""

    def __post_init__(self):
        identifier(self.trait_id)
        identifier(self.model_id)
        w = np.asarray(self.weights)
        if w.ndim != 2 or w.shape[0] != len(self.variants.ids) or w.shape[1] == 0 or w.dtype != np.float64:
            raise ValueError("weights must be FP64 and match the variant/context axes")
        if self.scale.variant_identity != self.variants.identity or len(self.scale.mean) != len(w):
            raise ValueError("model genotype scale does not match its variant axis")
        covariance = psd(self.covariance)
        if covariance.shape != (w.shape[1], w.shape[1]):
            raise ValueError("model covariance and weight dimensions differ")
        if self.geometry is not None and self.geometry.omega.shape != covariance.shape:
            raise ValueError("model geometry dimension differs")
        if self.geometry is not None and covariance[0, 0] > 0:
            # Restricted/profiled priors can change a and b. Their orthogonal
            # response must use the fitted prior's b/a, not the parent's b/a.
            # At zero fitted baseline variance, retain the explicitly supplied
            # positive-baseline anchor; the fitted baseline weights are zero.
            object.__setattr__(self, "geometry", ResponseGeometry(covariance,
                self.geometry.metric, self.geometry.reference, self.geometry.anchor))
        if self.convergence.get("converged") is not True:
            raise ValueError("a scoring model must have verified convergence")
        for name in ("true_residual_norm", "threshold", "fixed_projection"):
            value = self.convergence.get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError("scoring model lacks a finite true-convergence certificate")
        if self.convergence["true_residual_norm"] > self.convergence["threshold"]:
            raise ValueError("scoring model failed its true-residual rule")
        names = self.context_spec.get("names", ())
        fixed_names = self.fixed_spec.get("names", [term["name"] for term in self.fixed_spec.get("terms", [])])
        if len(names) != w.shape[1] or len(set(names)) != len(names) or len(fixed_names) != len(self.fixed_coefficients) or len(set(fixed_names)) != len(fixed_names):
            raise ValueError("model context/fixed feature identities disagree with coefficients")
        if not isinstance(self.phenotype_spec.get("units"), str) or not self.phenotype_spec["units"]:
            raise ValueError("model phenotype units must be declared")
        # File arrays already passed bounded validation; in-memory callers must
        # also supply finite immutable effects, without retaining a mutable view.
        if isinstance(self.weights, np.memmap):
            if self.weights.flags.writeable:
                raise ValueError("model mmap must be read-only")
            w = self.weights
            stat = Path(w.filename).stat()
            object.__setattr__(self, "_weight_file_state", (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        else:
            w = array(w, name="weights", ndim=2)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "fixed_coefficients", array(self.fixed_coefficients, name="fixed coefficients", ndim=1))
        for name in ("context_spec", "fixed_spec", "phenotype_spec", "prior_spec", "convergence", "provenance"):
            object.__setattr__(self, name, metadata(getattr(self, name)))

    @property
    def key(self):
        return (self.trait_id, self.model_id)

    def raw_weights(self):
        """Explicit export: returns an allocated raw-weight matrix and Q offsets."""
        weights = self.scale.inverse_scale[:, None]*self.weights
        offsets = -(self.scale.mean @ weights)
        return weights, offsets

    def check(self):
        if isinstance(self.weights, np.memmap):
            path = Path(self.weights.filename)
            state = path.stat()
            if path.is_symlink() or (state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns) != self._weight_file_state:
                raise RuntimeError("posterior weight file changed after model loading")


class ModelWriter:
    """Sequential weight-block sink; no genome-wide weight matrix in RAM."""
    def __init__(self, path, traits, source, result, provenance):
        self.path = Path(path)
        self.path.mkdir(parents=False, exist_ok=False)
        self.handles, self.progress, self.entries = {}, {}, []
        self.traits = tuple(traits)
        self.source, self.result = source, result
        self.provenance = metadata(provenance)
        try:
            for ti, t in enumerate(traits):
                for ci, c in enumerate(t.candidates):
                    key = (t.id, c.id)
                    name = f"weights-{ti}-{ci}.npy"
                    handle = (self.path/name).open("xb")
                    np.lib.format.write_array_header_2_0(handle, dict(descr=np.dtype("<f8").str,
                        fortran_order=False, shape=(len(t.variants), t.phi.shape[1])))
                    self.handles[key] = handle
                    self.progress[key] = 0
        except BaseException:
            self.close()
            raise

    def write(self, key, start, stop, values):
        t = next(t for t in self.traits if t.id == key[0])
        if self.progress[key] != start or stop > len(t.variants) or stop <= start:
            raise ValueError("weight shards must be written once in increasing variant order")
        values = np.asarray(values, dtype="<f8", order="C")
        if values.shape != (stop-start, t.phi.shape[1]) or not np.all(np.isfinite(values)):
            raise ValueError("invalid posterior weight block")
        self.handles[key].write(values.tobytes())
        self.progress[key] = stop

    def close(self):
        for handle in self.handles.values():
            if not handle.closed:
                handle.close()

    def finish(self, run_report):
        for t in self.traits:
            for c in t.candidates:
                key = (t.id, c.id)
                if self.progress[key] != len(t.variants):
                    raise ValueError("incomplete weight artifact")
                handle = self.handles[key]
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        trait_records = []
        for ti, t in enumerate(self.traits):
            def save(name, values):
                path = self.path/f"{name}-{ti}.npy"
                with path.open("xb") as handle:
                    np.save(handle, values, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                return array_record(path)
            geometry = None if t.geometry is None else dict(omega=t.geometry.omega.tolist(),
                metric=t.geometry.metric.tolist(), reference=t.geometry.reference, anchor=t.geometry.anchor)
            entry = dict(id=t.id, variants=self.source.variants.subset(t.variants).to_dict(),
                scale=dict(mean=save("mean", t.scale.mean), inverse_scale=save("inverse-scale", t.scale.inverse_scale),
                    variant_identity=t.scale.variant_identity, sample_identity=t.scale.sample_identity,
                    provenance=t.scale.provenance, ddof=t.scale.ddof, arithmetic=t.scale.arithmetic),
                context_spec=t.context_spec, fixed_spec=t.fixed_spec, phenotype_spec=t.phenotype_spec,
                geometry=geometry, models=[])
            for ci, c in enumerate(t.candidates):
                key = (t.id, c.id)
                entry["models"].append(dict(id=c.id, weights=array_record(self.path/f"weights-{ti}-{ci}.npy"),
                    fixed_coefficients=self.result.fixed_coefficients[key].tolist(), covariance=c.covariance.tolist(),
                    prior_spec=c.specification, convergence=self.result.reports[key]))
            trait_records.append(entry)
        manifest = dict(kind=KIND, schema_version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(), traits=trait_records,
                        provenance=self.provenance, run_report=run_report)
        write_json(self.path/"manifest.json", manifest)
        write_json(self.path/"COMPLETE.json", dict(kind=KIND, schema_version=VERSION,
                   manifest_sha256=file_digest(self.path/"manifest.json")))
        _sync_directory(self.path)
        return self.path


def load_prediction_models(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("model root cannot be a symlink")
    complete = read_json(path/"COMPLETE.json", max_bytes=4096)
    closed(complete, ("kind", "schema_version", "manifest_sha256"), name="completion")
    if complete["kind"] != KIND or complete["schema_version"] != VERSION:
        raise ValueError("unsupported prediction artifact kind/version")
    if file_digest(path/"manifest.json") != complete["manifest_sha256"]:
        raise ValueError("model manifest checksum mismatch")
    manifest = read_json(path/"manifest.json")
    closed(manifest, ("kind", "schema_version", "created_utc", "traits", "provenance", "run_report"), name="model manifest")
    if manifest["kind"] != KIND or manifest["schema_version"] != VERSION:
        raise ValueError("unsupported prediction artifact kind/version")
    models, seen_traits = [], set()
    for t in manifest["traits"]:
        closed(t, ("id", "variants", "scale", "context_spec", "fixed_spec", "phenotype_spec", "geometry", "models"), name="trait model")
        if t["id"] in seen_traits:
            raise ValueError("duplicate trait ID in artifact")
        seen_traits.add(t["id"])
        closed(t["variants"], ("ids", "chromosome", "position", "counted", "other", "genome_build"), name="variant axis")
        axis = VariantAxis(**t["variants"])
        s = t["scale"]
        closed(s, ("mean", "inverse_scale", "variant_identity", "sample_identity", "provenance", "ddof", "arithmetic"), name="model scale")
        scale = GenotypeScale(load_array(path, s["mean"], expected_shape=(len(axis.ids),), expected_dtype="float64"),
            load_array(path, s["inverse_scale"], expected_shape=(len(axis.ids),), expected_dtype="float64"),
            **{k: s[k] for k in ("variant_identity", "sample_identity", "provenance", "ddof", "arithmetic")})
        geometry = t["geometry"]
        if geometry is not None:
            closed(geometry, ("omega", "metric", "reference", "anchor"), name="geometry")
            geometry = ResponseGeometry(**geometry)
        seen_models = set()
        for c in t["models"]:
            closed(c, ("id", "weights", "fixed_coefficients", "covariance", "prior_spec", "convergence"), name="candidate model")
            if c["id"] in seen_models:
                raise ValueError("duplicate model ID in artifact")
            seen_models.add(c["id"])
            q = len(c["covariance"])
            weights = load_array(path, c["weights"], expected_shape=(len(axis.ids), q), expected_dtype="float64")
            models.append(PredictionModel(t["id"], c["id"], axis, scale, weights, c["fixed_coefficients"],
                c["covariance"], t["context_spec"], t["fixed_spec"], t["phenotype_spec"], c["prior_spec"],
                c["convergence"], manifest["provenance"], geometry,
                digest([complete["manifest_sha256"], t["id"], c["id"]])))
    if not models:
        raise ValueError("empty model bundle")
    return models


def write_genotype_scale(path, scale):
    path = Path(path)
    path.mkdir(exist_ok=False)
    arrays = {}
    for name in ("mean", "inverse_scale"):
        target = path/f"{name}.npy"
        with target.open("xb") as handle:
            np.save(handle, getattr(scale, name), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        arrays[name] = array_record(target)
    manifest = dict(kind="summit.prediction.scale", schema_version=1, **arrays,
        variant_identity=scale.variant_identity, sample_identity=scale.sample_identity,
        provenance=scale.provenance, ddof=scale.ddof, arithmetic=scale.arithmetic)
    write_json(path/"manifest.json", manifest)
    write_json(path/"COMPLETE.json", dict(manifest_sha256=file_digest(path/"manifest.json")))
    _sync_directory(path)


def load_genotype_scale(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("scale root cannot be a symlink")
    complete = read_json(path/"COMPLETE.json", max_bytes=4096)
    closed(complete, ("manifest_sha256",), name="scale completion")
    if file_digest(path/"manifest.json") != complete["manifest_sha256"]:
        raise ValueError("scale manifest checksum mismatch")
    s = read_json(path/"manifest.json")
    closed(s, ("kind", "schema_version", "mean", "inverse_scale", "variant_identity", "sample_identity", "provenance", "ddof", "arithmetic"), name="scale")
    if s["kind"] != "summit.prediction.scale" or s["schema_version"] != 1:
        raise ValueError("unsupported scale kind/version")
    mean = load_array(path, s["mean"], expected_dtype="float64")
    inv = load_array(path, s["inverse_scale"], expected_shape=mean.shape, expected_dtype="float64")
    return GenotypeScale(mean, inv, **{k: s[k] for k in ("variant_identity", "sample_identity", "provenance", "ddof", "arithmetic")})
