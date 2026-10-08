"""Copy an authenticated legacy model bundle into the compact shared-axis format.

The larger legacy JSON allowance is explicit and used only by this conversion
of locally produced artifacts. Ordinary model loading keeps its existing bound.
Inputs are never rewritten; every referenced numeric member is validated.
"""
import argparse
import os
from pathlib import Path
import shutil

from summit.prediction.artifacts import (read_json, write_json, file_digest,
    json_record, load_array, load_prediction_models, KIND, VERSION, _sync_directory)
from summit.prediction._validation import digest, closed
from summit.prediction.spec import VariantAxis


def repack(source, destination, *, legacy_limit_bytes=128*2**20):
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        raise ValueError("source root cannot be a symlink")
    complete = read_json(source / "COMPLETE.json", max_bytes=4096)
    closed(complete,("kind","schema_version","manifest_sha256"),name="completion")
    if complete["kind"] != KIND or complete["schema_version"] != 1:
        raise ValueError("converter requires an authenticated version-1 model")
    if file_digest(source / "manifest.json") != complete["manifest_sha256"]:
        raise ValueError("source manifest checksum mismatch")
    manifest = read_json(source / "manifest.json", max_bytes=legacy_limit_bytes)
    closed(manifest,("kind","schema_version","created_utc","traits","provenance","run_report"),name="legacy model")
    if manifest["kind"] != KIND or manifest["schema_version"] != 1:
        raise ValueError("legacy manifest version mismatch")
    destination.mkdir(parents=True,exist_ok=False)
    def copy(record):
        target = destination / record["file"]
        if target.exists():
            if file_digest(target) != record["sha256"]:
                raise ValueError("inconsistent reused numeric member")
            return
        with (source / record["file"]).open("rb") as original, target.open("xb") as handle:
            shutil.copyfileobj(original,handle,1024*1024)
            handle.flush()
            os.fsync(handle.fileno())
    axes, scales = {}, {}
    for trait in manifest["traits"]:
        closed(trait,("id","variants","scale","context_spec","fixed_spec","phenotype_spec","geometry","models"),name="trait")
        axis = VariantAxis(**trait["variants"])
        if axis.identity not in axes:
            path = destination / f"variant-axis-{len(axes)}.json"
            write_json(path, axis.to_dict())
            axes[axis.identity] = json_record(path)
        trait["variants"] = axis.identity
        scale = trait["scale"]
        identity = digest({**scale,"mean":scale["mean"]["sha256"],"inverse_scale":scale["inverse_scale"]["sha256"]})
        for name in ("mean","inverse_scale"):
            record = scale[name]
            load_array(source,record,expected_shape=(len(axis.ids),),expected_dtype="float64")
            if identity not in scales:
                copy(record)
        if identity not in scales:
            scales[identity] = scale
        trait["scale"] = scales[identity]
        for model in trait["models"]:
            record = model["weights"]
            load_array(source,record,expected_shape=(len(axis.ids),len(model["covariance"])),expected_dtype="float64")
            copy(record)
    manifest.update(schema_version=VERSION,variant_axes=axes)
    manifest["provenance"] = dict(manifest["provenance"],
        legacy_manifest_sha256=complete["manifest_sha256"],
        conversion="shared variant axes and affine scales; numeric model arrays unchanged")
    write_json(destination / "manifest.json",manifest)
    write_json(destination / "COMPLETE.json",dict(kind=KIND,schema_version=VERSION,
        manifest_sha256=file_digest(destination / "manifest.json")))
    _sync_directory(destination)
    loaded = load_prediction_models(destination)
    return dict(models=len(loaded),old_manifest_bytes=(source / "manifest.json").stat().st_size,
        new_manifest_bytes=(destination / "manifest.json").stat().st_size,
        shared_variant_axes=len(axes),shared_scales=len(scales))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("--legacy-metadata-limit-mib",type=int,default=128)
    a=p.parse_args()
    print(repack(a.source,a.destination,legacy_limit_bytes=a.legacy_metadata_limit_mib*2**20))


if __name__=="__main__":
    main()
