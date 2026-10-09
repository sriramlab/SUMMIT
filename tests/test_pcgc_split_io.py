import json
from dataclasses import replace

import numpy as np
import pytest

from summit.context.spec import canonical_json, canonical_sha256
from summit.pcgc import split_io
from summit.pcgc.artifacts import make_artifact, write_artifact
from summit.pcgc.gxe import prepare_gxe_moments
from summit.pcgc.gxe_io import make_gxe_artifact, write_gxe_artifact
from summit.pcgc.research import exact_moments
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from test_pcgc_io import fixture


def prepared(contextual=False):
    _, x, axis, risk, scale = fixture(64, 48)
    annotations = np.ones((len(axis.ids), 1))
    common = dict(variant_axis=axis, annotation_names=["all"],
                  sample_identity="a" * 64, genotype_scale_identity=scale.identity, risk=risk)
    if not contextual:
        return make_artifact(exact_moments(x, annotations, risk), diagnostics={}, **common)
    contexts = np.column_stack((np.ones(len(x)), np.linspace(-1, 1, len(x))))
    moments, diagnostics = prepare_gxe_moments(
        ArraySequentialGenotypeOperator(x), annotations, risk, contexts,
        liability_sd=1., probes=31, native=False,
    )
    return make_gxe_artifact(moments, context_names=["intercept", "E"],
                             contexts=contexts, liability_sd=1., diagnostics=diagnostics, **common)


def changed_header(artifact, field, value):
    header = json.loads(canonical_json(artifact.manifest))
    header[field] = value
    header.pop("manifest_hash")
    header["manifest_hash"] = canonical_sha256(header)
    return type(artifact)(artifact.moments, header)


@pytest.mark.parametrize("contextual", [False, True])
def test_separate_files_contain_disjoint_arrays_and_can_be_renamed(tmp_path, contextual):
    artifact = prepared(contextual)
    summary, reference = split_io.write_split_artifact(artifact, *split_io.output_paths(tmp_path/"disease"))
    with np.load(summary) as s, np.load(reference) as r:
        assert "rhs_rows" in s.files and "rhs_rows" not in r.files
        assert "ldscores" in r.files and "ldscores" not in s.files
        assert set(s.files) & set(r.files) == {"manifest_json"}
        assert set(s.files) | set(r.files) == set(artifact.manifest["array_hashes"]) | {"manifest_json"}
    moved = tmp_path/"elsewhere"
    moved.mkdir()
    summary = summary.rename(moved/"trait.npz")
    reference = reference.rename(tmp_path/"reference-with-another-name.npz")
    loaded = split_io.load_fit_artifact(summary, reference)
    assert loaded.manifest == artifact.manifest
    np.testing.assert_array_equal(loaded.moments.rhs_rows, artifact.moments.rhs_rows)
    np.testing.assert_array_equal(loaded.moments.ldscores, artifact.moments.ldscores)
    with pytest.raises(ValueError, match="require --ldscores"):
        split_io.load_fit_artifact(summary)
    with pytest.raises(ValueError, match="summary statistics with --h2"):
        split_io.load_fit_artifact(reference, summary)
    combined = (write_gxe_artifact if contextual else write_artifact)(artifact, tmp_path/"old.binary.npz")
    assert split_io.load_fit_artifact(combined).manifest == artifact.manifest
    with pytest.raises(ValueError, match="summary statistics with --h2"):
        split_io.load_fit_artifact(combined, reference)


@pytest.mark.parametrize("field,value", [
    ("sample_identity", "b" * 64),
    ("genotype_scale_identity", "b" * 64),
    ("risk_identity", "b" * 64),
    ("context_identity", "b" * 64),
    ("liability_sd_identity", "b" * 64),
    ("context_names", ["intercept", "different_exposure"]),
    ("annotation_names", ["different_annotation"]),
])
def test_equal_dimension_mismatches_fail_before_numeric_arrays_are_read(tmp_path, monkeypatch, field, value):
    artifact = prepared(True)
    summary, _ = split_io.write_split_artifact(artifact, *split_io.output_paths(tmp_path/"first"))
    _, wrong_reference = split_io.write_split_artifact(
        changed_header(artifact, field, value), *split_io.output_paths(tmp_path/"second"))
    with np.load(summary) as archive:
        archive_type = type(archive)
    original_get = archive_type.__getitem__
    def metadata_only(self, key):
        assert key == "manifest_json", "numeric array was loaded before compatibility admission"
        return original_get(self, key)
    monkeypatch.setattr(archive_type, "__getitem__", metadata_only)
    with pytest.raises(ValueError, match=field):
        split_io.load_fit_artifact(summary, wrong_reference)


@pytest.mark.parametrize("change", ["alleles", "variant_order", "genome_build", "reference_values", "method"])
def test_reference_variant_and_realization_checks(tmp_path, change):
    artifact = prepared()
    summary, _ = split_io.write_split_artifact(artifact, *split_io.output_paths(tmp_path/"first"))
    header = json.loads(canonical_json(artifact.manifest))
    if change in ("alleles", "variant_order", "genome_build"):
        axis = header["variant_axis"]
        if change == "alleles":
            axis["counted"], axis["other"] = axis["other"], axis["counted"]
        elif change == "variant_order":
            axis["ids"] = axis["ids"][::-1]
        else:
            axis["genome_build"] = "other_build"
        other = changed_header(artifact, "variant_axis", axis)
    else:
        from summit.context.spec import array_sha256
        moments = replace(artifact.moments, **(
            {"ldscores": artifact.moments.ldscores + .01} if change == "reference_values" else {"method": "pcgc-inverse"}
        ))
        header["method"] = moments.method
        header["array_hashes"]["ldscores"] = array_sha256(moments.ldscores)
        header.pop("manifest_hash")
        header["manifest_hash"] = canonical_sha256(header)
        other = type(artifact)(moments, header)
    _, reference = split_io.write_split_artifact(other, *split_io.output_paths(tmp_path/"second"))
    with pytest.raises(ValueError, match="incompatible PCGC reference"):
        split_io.load_fit_artifact(summary, reference)


@pytest.mark.parametrize("side,change", [
    ("summary", "values"), ("reference", "values"),
    ("summary", "extra"), ("reference", "missing"),
    ("summary", "metadata"), ("reference", "metadata"),
])
def test_corrupt_files_fail_checks(tmp_path, side, change):
    summary, reference = split_io.write_split_artifact(prepared(), *split_io.output_paths(tmp_path/"original"))
    source = summary if side == "summary" else reference
    with np.load(source) as archive:
        arrays = {name: archive[name] for name in archive.files}
    name = "rhs_rows" if side == "summary" else "ldscores"
    if change == "values": arrays[name] = arrays[name] + .1
    elif change == "extra": arrays["ldscores"] = np.ones((48, 1))
    elif change == "missing": del arrays[name]
    else:
        header = json.loads(str(arrays["manifest_json"].item()))
        header["schema_version"] = 99
        arrays["manifest_json"] = np.asarray(json.dumps(header))
    damaged = tmp_path/"damaged.npz"
    np.savez_compressed(damaged, **arrays)
    with pytest.raises(ValueError, match="checksum|file role"):
        split_io.load_fit_artifact(damaged if side == "summary" else summary,
                                   reference if side == "summary" else damaged)


def test_no_overwrite_and_publication_failure_preserve_other_files(tmp_path, monkeypatch):
    artifact = prepared()
    summary, reference = split_io.output_paths(tmp_path/"disease")
    reference.write_text("existing reference")
    with pytest.raises(FileExistsError):
        split_io.write_split_artifact(artifact, summary, reference)
    assert reference.read_text() == "existing reference" and not summary.exists()
    new_summary, new_reference = split_io.output_paths(tmp_path/"new")
    real_link = split_io.os.link
    def publish_race(source, destination):
        if destination == new_summary:
            new_summary.write_text("another writer's output")
            raise FileExistsError(destination)
        real_link(source, destination)
    monkeypatch.setattr(split_io.os, "link", publish_race)
    with pytest.raises(FileExistsError):
        split_io.write_split_artifact(artifact, new_summary, new_reference)
    assert new_summary.read_text() == "another writer's output"
    assert not new_reference.exists()
    assert not list(tmp_path.glob(".*"))


def test_incomplete_serialization_publishes_neither_file(tmp_path, monkeypatch):
    artifact = prepared()
    real_save = split_io.np.savez_compressed
    calls = 0
    def disk_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("simulated disk failure")
        return real_save(*args, **kwargs)
    monkeypatch.setattr(split_io.np, "savez_compressed", disk_error)
    with pytest.raises(OSError, match="disk failure"):
        split_io.write_split_artifact(artifact, *split_io.output_paths(tmp_path/"disease"))
    assert list(tmp_path.iterdir()) == []


def test_preparation_rejects_fit_inputs_and_existing_outputs_before_genotypes(tmp_path, monkeypatch):
    from summit.cli import build_parser
    from summit.pcgc import cli
    def unexpected_prepare(args):
        pytest.fail("genotypes were opened before output validation")
    monkeypatch.setattr(cli, "prepare", unexpected_prepare)
    prefix = tmp_path/"prepared"
    argv = ["--binary-method", "pcgc", "--make-binary-sumstats", "unused.tsv", "--out", str(prefix)]
    with pytest.raises(ValueError, match="belongs to inference"):
        args = [*argv, "--ldscores", "unused.npz"]
        cli.run(build_parser().parse_args(args), args)
    reference = split_io.output_paths(prefix)[1]
    reference.write_text("keep")
    with pytest.raises(FileExistsError):
        cli.run(build_parser().parse_args(argv), argv)
    assert reference.read_text() == "keep"


def test_combined_contextual_output_remains_available(tmp_path, monkeypatch):
    from summit.cli import build_parser
    from summit.pcgc import cli
    artifact = prepared(True)
    monkeypatch.setattr(cli, "prepare", lambda args: artifact)
    prefix = tmp_path/"combined"
    argv = ["--binary-method", "pcgc", "--make-binary-sumstats", "unused.tsv",
            "--binary-context-columns", "E", "--binary-output-format", "combined", "--out", str(prefix)]
    cli.run(build_parser().parse_args(argv), argv)
    assert split_io.load_fit_artifact(tmp_path/"combined.binary.npz").manifest == artifact.manifest
    assert not split_io.output_paths(prefix)[0].exists()
