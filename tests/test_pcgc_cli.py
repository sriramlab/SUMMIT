import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from summit.pcgc.cli import add_arguments, run, sample_table, annotation_table, population_scale
from summit.pcgc.artifacts import make_artifact, write_artifact
from summit.pcgc.research import exact_moments
from test_pcgc_io import fixture


def parser():
    p = argparse.ArgumentParser()
    add_arguments(p)
    for name in ("geno", "out", "h2", "njack", "annot"):
        p.add_argument("--"+name)
    p.add_argument("--num-threads", type=int)
    return p


def test_cli_fit_dispatch_and_rejection_of_incompatible_flags(tmp_path):
    _, x, axis, risk, scale = fixture()
    artifact = make_artifact(exact_moments(x, np.ones((len(axis.ids), 1)), risk), variant_axis=axis,
        annotation_names=["all"], sample_identity="a"*64, genotype_scale_identity=scale.identity, risk=risk, diagnostics={})
    path = write_artifact(artifact, tmp_path/"prepared.npz")
    argv = ["--binary-method", "pcgc", "--h2", str(path), "--out", str(tmp_path/"fit")]
    run(parser().parse_args(argv), argv)
    assert (tmp_path/"fit.binary.json").exists()
    assert json.loads((tmp_path/"fit.binary.json").read_text())["qualification"] == "point_estimate_only"
    with pytest.raises(FileExistsError):
        run(parser().parse_args(argv), argv)
    for unsupported in ("--weight-mode", "--rg", "--covar", "--ldscores"):
        with pytest.raises(ValueError, match="unsupported"):
            run(parser().parse_args(argv), argv+[unsupported, "x"])
    wrong = argv.copy()
    wrong[1] = "pcgc-inverse"
    wrong[5] = str(tmp_path/"wrong_method")
    with pytest.raises(ValueError, match="disagrees"):
        run(parser().parse_args(wrong+["--binary-research"]), wrong+["--binary-research"])
    with pytest.raises(ValueError, match="qualification"):
        run(parser().parse_args(wrong), wrong)
    with pytest.raises(ValueError, match="qualification"):
        run(parser().parse_args(argv+["--njack", "8"]), argv+["--njack", "8"])
    fractional = [*argv[:5], str(tmp_path/"fractional"), "--binary-research", "--njack", "3.5"]
    with pytest.raises(ValueError, match="integer"):
        run(parser().parse_args(fractional), fractional)


def test_sample_alignment_by_ids_and_unknown_id_rejection(tmp_path):
    source = SimpleNamespace(samples=(("f0", "i0"), ("f1", "i1"), ("f2", "i2")))
    path = tmp_path/"sample.tsv"
    path.write_text("FID IID Y C\nf2 i2 1 .2\nf0 i0 0 .7\n")
    rows, table = sample_table(path, source)
    np.testing.assert_array_equal(rows, [0, 2])
    np.testing.assert_allclose(table.C, [.7, .2])
    path.write_text("FID IID Y\nf4 i4 0\n")
    with pytest.raises(ValueError, match="unknown"):
        sample_table(path, source)
    path.write_text("FID IID Y\nf0 i0 0\nf0 i0 1\n")
    with pytest.raises(ValueError, match="duplicate"):
        sample_table(path, source)
    path.write_text("FID IID Y Y\nf0 i0 0 1\n")
    with pytest.raises(ValueError, match="duplicate column"):
        sample_table(path, source)


def test_annotation_order_is_aligned_and_missing_ids_fail(tmp_path):
    source = SimpleNamespace(variants=SimpleNamespace(ids=("a", "b", "c")))
    path = tmp_path/"a.tsv"
    path.write_text("SNP all rare\nc 1 .2\na 1 .5\nb 1 0\n")
    a, names = annotation_table(path, source)
    assert names == ["all", "rare"]
    np.testing.assert_allclose(a[:, 1], [.5, 0, .2])
    path.write_text("SNP all\na 1\nb 1\n")
    with pytest.raises(ValueError, match="exactly once"):
        annotation_table(path, source)


def test_population_scale_table_preserves_allele_and_variant_axes(tmp_path):
    _, _, axis, _, scale = fixture()
    source = SimpleNamespace(variants=axis)
    path = tmp_path/"scale.tsv"
    table = pd.DataFrame(dict(SNP=axis.ids, A1=axis.counted, A2=axis.other,
                              MEAN=scale.mean, INV_SD=scale.inverse_scale)).iloc[::-1]
    table.to_csv(path, sep="\t", index=False)
    loaded = population_scale(path, source)
    np.testing.assert_array_equal(loaded.mean, scale.mean)
    np.testing.assert_allclose(loaded.inverse_scale, scale.inverse_scale)
    assert loaded.variant_identity == axis.identity
    table["A1"], table["A2"] = table.A2.copy(), table.A1.copy()
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="counted"):
        population_scale(path, source)


def test_actual_summit_cli_prepares_and_fits_binary_artifact(tmp_path, monkeypatch):
    native = pytest.importorskip("summit.gxeldcore")
    if getattr(native, "prediction_execution_version", 0) < 2:
        pytest.skip("native build lacks shared genotype source")
    import sys
    from bed_reader import to_bed
    from summit import cli
    from prediction_helpers import prediction_threads
    from summit.prediction.artifacts import write_genotype_scale
    raw, _, axis, risk, scale = fixture(128, 256)
    samples = [(f"f{i}", f"i{i}") for i in range(len(raw))]
    to_bed(tmp_path/"test.bed", raw, count_A1=True, num_threads=1, properties={
        "fid": [s[0] for s in samples], "iid": [s[1] for s in samples],
        "sid": list(axis.ids), "chromosome": list(axis.chromosome), "bp_position": list(axis.position),
        "allele_1": list(axis.counted), "allele_2": list(axis.other)})
    pd.DataFrame(dict(FID=[s[0] for s in samples], IID=[s[1] for s in samples],
                      Y=(risk.z > 0).astype(int), RISK=risk.population_risk)).iloc[::-1].to_csv(tmp_path/"samples.tsv", sep="\t", index=False)
    write_genotype_scale(tmp_path/"scale", scale)
    monkeypatch.setattr(sys, "argv", ["summit", "--binary-method", "pcgc", "--make-binary-sumstats", str(tmp_path/"samples.tsv"),
        "--geno", str(tmp_path/"test.bed"), "--binary-scale", str(tmp_path/"scale"), "--binary-genome-build", "test",
        "--binary-prevalence", ".1", "--binary-risk-column", "RISK", "--binary-probes", "61", "--num-threads", str(prediction_threads()),
        "--out", str(tmp_path/"prepared"), "--_binary-research"])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["summit", "--binary-method", "pcgc", "--h2", str(tmp_path/"prepared.binary.npz"),
        "--njack", "8", "--out", str(tmp_path/"fit"), "--_binary-research"])
    assert cli.main() == 0
    assert (tmp_path/"fit.binary.json").exists()
