import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from summit.pcgc.cli import run, sample_table, annotation_table, population_scale
from summit.pcgc.artifacts import make_artifact, write_artifact
from summit.pcgc.research import exact_moments
from test_pcgc_io import fixture


def parser():
    from summit.cli import build_parser
    return build_parser()


def test_cli_fit_dispatch_and_rejection_of_incompatible_flags(tmp_path):
    _, x, axis, risk, scale = fixture(64, 257)
    artifact = make_artifact(exact_moments(x, np.ones((len(axis.ids), 1)), risk), variant_axis=axis,
        annotation_names=["all"], sample_identity="a"*64, genotype_scale_identity=scale.identity, risk=risk, diagnostics={})
    path = write_artifact(artifact, tmp_path/"prepared.npz")
    argv = ["--binary-method", "pcgc", "--h2", str(path), "--out", str(tmp_path/"fit")]
    run(parser().parse_args(argv), argv)
    assert (tmp_path/"fit.binary.json").exists()
    result = json.loads((tmp_path/"fit.binary.json").read_text())
    assert result["uncertainty_status"] == "estimated"
    assert result["jackknife_blocks"] == 200
    assert result["schema_version"] == 2
    assert "qualification" not in result
    with pytest.raises(FileExistsError):
        run(parser().parse_args(argv), argv)
    for unsupported in ("--weight-mode", "--rg", "--covar", "--ldscores"):
        with pytest.raises(ValueError, match="unsupported"):
            run(parser().parse_args(argv), argv+[unsupported, "x"])
    wrong = argv.copy()
    wrong[1] = "pcgc-inverse"
    wrong[5] = str(tmp_path/"wrong_method")
    with pytest.raises(ValueError, match="disagrees"):
        run(parser().parse_args(wrong), wrong)
    for count in ("3.5", "chr", "0", "1", "258"):
        invalid = [*argv[:5], str(tmp_path/"invalid"), "--njack", count]
        with pytest.raises(ValueError, match="integer|exceeds"):
            run(parser().parse_args(invalid), invalid)


@pytest.mark.parametrize("method", ["liability", "pcgc", "pcgc-inverse", "pcgc-basis", "pcgc-ld"])
def test_all_methods_report_component_and_total_se_without_a_gate(tmp_path, method):
    from summit.pcgc.research import exact_external_ld, external_ld_moments
    from summit.sumstats.binary import prepare_binary_risk
    from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
    _, x, axis, risk, scale = fixture(128, 257)
    if method == "liability":
        risk = prepare_binary_risk(risk.z > 0, .1)
    # Overlapping weights exercise covariance between components, unequal
    # block sizes, and the annotation reduction path at the 200-block default.
    a = np.column_stack([np.ones(len(axis.ids)), np.linspace(.1, 1, len(axis.ids))])
    kwargs = {"sensitivity": risk.sensitivity} if method == "pcgc-basis" else {}
    moments = exact_moments(x, a, risk, method, **kwargs)
    if method == "pcgc-ld":
        reference = np.random.default_rng(953).normal(size=(512, len(a)))
        moments = external_ld_moments(moments.rhs_rows, a, risk, exact_external_ld(reference, a))
    artifact = make_artifact(moments, variant_axis=axis, annotation_names=["all", "weighted"],
        sample_identity="a"*64, genotype_scale_identity=scale.identity, risk=risk, diagnostics={})
    path = write_artifact(artifact, tmp_path/"prepared.npz")
    for count in (200, 8):
        prefix = tmp_path/f"fit{count}"
        argv = ["--binary-method", method, "--h2", str(path), "--out", str(prefix)]
        if count != 200:
            argv += ["--njack=8"]
        run(parser().parse_args(argv), argv)
        result = json.loads(prefix.with_suffix(".binary.json").read_text())
        design = JackknifeDesign.from_trace_view(SimpleNamespace(nsnps=len(a)), JackknifeSpec.parse(count))
        # Recompute each deletion directly, without the production reducer.
        loo = np.array([np.linalg.solve(*moments.equations(design.unit_id != b)) for b in range(count)])
        centered = loo - loo.mean(axis=0)
        cov = (count-1)/count * centered.T @ centered
        np.testing.assert_allclose(result["jackknife_replicates"], loo, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result["conditional_standard_errors"], np.sqrt(cov.diagonal()), rtol=1e-10)
        np.testing.assert_allclose(result["conditional_total_standard_error"], np.sqrt(cov.sum()), rtol=1e-10)
        np.testing.assert_allclose(result["marginal_total_standard_error"], np.sqrt(cov.sum())/(1+risk.covariate_variance), rtol=1e-10)
        assert result["uncertainty_status"] == "estimated"
        assert result["jackknife_blocks"] == count


def test_small_artifact_requires_an_explicit_smaller_block_count(tmp_path):
    _, x, axis, risk, scale = fixture()
    moments = exact_moments(x, np.ones((len(axis.ids), 1)), risk)
    artifact = make_artifact(moments, variant_axis=axis, annotation_names=["all"],
        sample_identity="a"*64, genotype_scale_identity=scale.identity, risk=risk, diagnostics={})
    path = write_artifact(artifact, tmp_path/"prepared.npz")
    argv = ["--binary-method", "pcgc", "--h2", str(path), "--out", str(tmp_path/"fit")]
    with pytest.raises(ValueError, match="choose a smaller block count"):
        run(parser().parse_args(argv), argv)
    assert not (tmp_path/"fit.binary.json").exists()
    run(parser().parse_args(argv+["--njack", "8"]), argv+["--njack", "8"])


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
    from summit.pcgc.artifacts import load_artifact
    from summit.entrypoint import main as entry_main
    common = ["--binary-method", "pcgc", "--make-binary-sumstats", str(tmp_path/"samples.tsv"),
        "--geno", str(tmp_path/"test.bed"), "--binary-scale", str(tmp_path/"scale"),
        "--binary-prevalence", ".1", "--binary-risk-column", "RISK", "--num-threads", str(prediction_threads())]
    options = ["--nvecs", "61", "--seed", "81", "--block-size", "37", "--memory-gib", "1"]
    assert entry_main([*common, *options, "--genome-build", "test", "--out", str(tmp_path/"prepared")]) == 0
    # A build label has no numerical effect. Test an unlabeled TSV scale too.
    pd.DataFrame(dict(SNP=axis.ids, A1=axis.counted, A2=axis.other,
                      MEAN=scale.mean, INV_SD=scale.inverse_scale)).to_csv(tmp_path/"scale.tsv", sep="\t", index=False)
    unlabeled = list(common)
    unlabeled[unlabeled.index("--binary-scale")+1] = str(tmp_path/"scale.tsv")
    assert entry_main([*unlabeled, *options, "--out", str(tmp_path/"unlabeled")]) == 0
    labeled, unlabeled = [load_artifact(tmp_path/(name+".binary.npz")) for name in ("prepared", "unlabeled")]
    for field in ("rhs_rows", "ldscores", "annotations", "same_person"):
        np.testing.assert_allclose(getattr(labeled.moments, field), getattr(unlabeled.moments, field), rtol=1e-14, atol=1e-14)
    monkeypatch.setattr(sys, "argv", ["summit", "--binary-method", "pcgc", "--h2", str(tmp_path/"prepared.binary.npz"),
        "--out", str(tmp_path/"fit")])
    assert cli.main() == 0
    assert (tmp_path/"fit.binary.json").exists()
    result = json.loads((tmp_path/"fit.binary.json").read_text())
    assert result["jackknife_blocks"] == 200
    assert np.isfinite(result["marginal_total_standard_error"])


def test_root_cli_documents_binary_default_and_removes_research_flags():
    from summit.cli import build_parser
    root = build_parser()
    assert root.parse_args([]).njack == "chr"
    assert "200 contiguous SNP blocks" in " ".join(root.format_help().split())
    for flag in ("--binary-research", "--_binary-research"):
        with pytest.raises(SystemExit):
            root.parse_args([flag])
