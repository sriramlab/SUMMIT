from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from test_prediction_io import write_bed
from summit.prediction._validation import digest
from summit.prediction.artifacts import write_json, write_genotype_scale, load_prediction_models, read_json
from summit.prediction.genotype import FileGenotypeSource, estimate_scale
from summit.prediction.features import fit_contexts, evaluate_contexts, evaluate_fixed
from summit.prediction.cli import main


def test_cli_full_pipeline_and_frozen_feature_transform(tmp_path, capsys):
    source, old_traits, bed = write_bed(tmp_path)
    rng = np.random.default_rng(232)
    samples = pd.DataFrame(source.samples, columns=["FID", "IID"])
    context = samples.assign(age=rng.normal(50, 10, len(samples)), group=np.arange(len(samples)) % 3)
    fixed = samples.assign(pc=rng.normal(size=len(samples)))
    outcomes = samples.assign(y=rng.normal(size=len(samples)), r=.5+np.arange(len(samples))/len(samples))
    rows = old_traits[0].rows
    calibration, context_spec, metric = fit_contexts(context.iloc[rows],
        [dict(name="age", kind="continuous"), dict(name="group", kind="categorical", categories=[0, 1, 2], reference_category=0)])
    np.testing.assert_allclose(evaluate_contexts(context_spec, context.iloc[rows]), calibration)
    fixed_spec = dict(kind="summit.prediction.fixed", schema_version=1, terms=[
        dict(name="intercept", factors=[]), dict(name="pc", factors=[dict(source="covariate", name="pc", power=1)]),
        dict(name="age_squared", factors=[dict(source="context", name="age", power=2)])])
    evaluate_fixed(fixed_spec, fixed.iloc[rows], calibration, context_spec["names"])
    context.to_csv(tmp_path/"contexts.tsv", sep="\t", index=False)
    fixed.to_csv(tmp_path/"fixed.tsv", sep="\t", index=False)
    outcomes.to_csv(tmp_path/"outcomes.tsv", sep="\t", index=False)
    samples.iloc[rows].to_csv(tmp_path/"discovery.keep", sep="\t", index=False)
    samples.to_csv(tmp_path/"score.keep", sep="\t", index=False)
    write_json(tmp_path/"context.json", context_spec)
    write_json(tmp_path/"fixed.json", fixed_spec)
    with FileGenotypeSource(bed, genome_build="GRCh37") as file_source:
        scale = estimate_scale(file_source, rows, np.arange(len(source.variants.ids)), block_size=9)
    write_genotype_scale(tmp_path/"scale", scale)
    q = len(context_spec["names"])
    omega = np.diag(np.r_[.3, np.full(q-1, .07)])
    write_json(tmp_path/"prior.json", dict(kind="summit.prediction.prior", schema_version=1,
        covariance=omega.tolist(), context_identity=digest(context_spec), scale_identity=scale.identity,
        provenance={"source": "synthetic admitted PSD"}, geometry=dict(metric=metric[1:, 1:].tolist(), reference="synthetic", anchor="context_zero")))
    write_json(tmp_path/"residual.json", dict(kind="summit.prediction.residual", schema_version=1,
        file="outcomes.tsv", column="r", units="model_phenotype_variance", floor=.1, provenance={"source": "synthetic"}))
    trait = dict(id="trait", phenotype=dict(file="outcomes.tsv", column="y", units="test"), samples="discovery.keep",
        contexts="contexts.tsv", context_spec="context.json", covariates="fixed.tsv", fixed_spec="fixed.json",
        genotype_scale="scale", architecture_prior="prior.json", residual_spec="residual.json",
        candidates=[dict(id="full", operation="common_scale", kappa=.5),
                    dict(id="amplification", operation="separate_scales", kappa_a=.5, kappa_h=0)])
    write_json(tmp_path/"fit.json", dict(kind="summit.prediction.fit_spec", schema_version=1,
        genotypes=dict(geno=bed.name, genome_build="GRCh37"), traits=[trait], solver=dict(rtol=1e-10)))
    assert main(["plan", "--spec", str(tmp_path/"fit.json"), "--memory-gib", "1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["models"] == 2
    assert not (tmp_path/"fit").exists()
    assert main(["fit", "--spec", str(tmp_path/"fit.json"), "--out", str(tmp_path/"fit"),
        "--genotype-storage", "compact", "--block-size", "9", "--memory-gib", "1"]) == 0
    capsys.readouterr()
    models = load_prediction_models(tmp_path/"fit")
    write_json(tmp_path/"score.json", dict(kind="summit.prediction.score_spec", schema_version=1,
        genotypes=dict(geno=bed.name, genome_build="GRCh37"), traits=[dict(id="trait", samples="score.keep", contexts="contexts.tsv", covariates="fixed.tsv")]))
    assert main(["score", "--models", str(tmp_path/"fit"), "--spec", str(tmp_path/"score.json"),
        "--out", str(tmp_path/"scores"), "--block-size", "7", "--memory-gib", "1"]) == 0
    assert (tmp_path/"scores"/"COMPLETE.json").is_file()
    capsys.readouterr()
    assert main(["inspect", str(tmp_path/"fit")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["models"]) == 2
    novel = context.copy()
    novel.loc[0, "group"] = 9
    with pytest.raises(ValueError, match="unknown categorical"):
        evaluate_contexts(context_spec, novel)


def test_json_unknown_duplicate_nonfinite_and_unsafe_array(tmp_path):
    from summit.prediction.artifacts import load_array
    bad = tmp_path/"bad.json"
    bad.write_text('{"x": 1, "x": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(bad)
    nonfinite = tmp_path/"nonfinite.json"
    nonfinite.write_text('{"x": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        read_json(nonfinite)
    with pytest.raises(ValueError, match="unsafe"):
        load_array(tmp_path, dict(file="../bad.npy", shape=[2], dtype="<f8", bytes=20, sha256="a"))
