from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from test_prediction_core import fixture
from summit.prediction.api import fit_prediction
from summit.prediction.artifacts import load_prediction_models, load_genotype_scale, write_genotype_scale
from summit.prediction.batch import plan_prediction
from summit.prediction.genotype import ArrayGenotypeSource, FileGenotypeSource, standardize, estimate_scale
from summit.prediction.score import ScoreInput, score_prediction
from summit.prediction.spec import VariantAxis


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_model_reload_score_raw_conversion_and_allele_reorder(tmp_path, backend):
    source, traits = fixture()
    models = fit_prediction(traits, source, output=tmp_path/"fit",
        plan=plan_prediction(traits, source, storage="compact", block_size=9, rhs_columns=6), backend=backend)
    inputs = {t.id: ScoreInput(t.rows, t.phi, t.fixed, t.context_spec, t.fixed_spec) for t in traits}
    scores = score_prediction(models, source, inputs, backend=backend, block_size=11, rhs_columns=6)
    for m in models:
        t = next(t for t in traits if t.id == m.trait_id)
        raw = source.values[np.ix_(t.rows, t.variants)].astype(float)
        g = standardize(raw, m.scale.mean, m.scale.inverse_scale)
        np.testing.assert_allclose(scores.components[m.key], g @ m.weights, atol=2e-14)
        raw = np.where(raw == -127, m.scale.mean, raw)
        w, offset = m.raw_weights()
        np.testing.assert_allclose(raw @ w + offset, scores.components[m.key], atol=2e-14)
        if m.covariance[0, 0] > 0:
            np.testing.assert_allclose(m.geometry.gamma, m.covariance[1:, 0]/m.covariance[0, 0], atol=1e-14)
    # Reverse source variants and swap counted alleles, including missingness.
    order = np.arange(len(source.variants.ids))[::-1]
    axis = source.variants.subset(order)
    swapped = VariantAxis(axis.ids, axis.chromosome, axis.position, axis.other, axis.counted, axis.genome_build)
    values = source.values[:, order]
    values = np.where(values == -127, -127, 2-values)
    other = ArrayGenotypeSource(values, source.samples, swapped, hard_calls=True)
    rescored = score_prediction(models, other, inputs, backend=backend, block_size=7)
    for key in scores.genetic:
        np.testing.assert_allclose(scores.genetic[key], rescored.genetic[key], atol=3e-14)
    assert scores.report["ledger"]["source_variants"] == len(source.variants.ids)
    with pytest.raises(FileExistsError):
        fit_prediction(traits, source, output=tmp_path/"fit", backend=backend)
    with pytest.raises(ValueError, match="recipe"):
        score_prediction(models, source, {**inputs, traits[0].id: replace(inputs[traits[0].id], context_spec={"wrong": True})}, backend=backend)


def test_artifact_rejects_corruption_and_incomplete_bundle(tmp_path):
    source, traits = fixture()
    fit_prediction(traits, source, output=tmp_path/"fit", backend="numpy")
    target = tmp_path/"fit"/"weights-0-0.npy"
    with target.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        load_prediction_models(tmp_path/"fit")
    incomplete = tmp_path/"incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        load_prediction_models(incomplete)


def test_scale_roundtrip_and_sample_order_rejection(tmp_path):
    source, traits = fixture()
    scale = traits[0].scale
    write_genotype_scale(tmp_path/"scale", scale)
    loaded = load_genotype_scale(tmp_path/"scale")
    assert loaded.identity == scale.identity
    with pytest.raises(ValueError, match="sample/order"):
        plan_prediction([replace(traits[0], rows=traits[0].rows[::-1])], source)


def write_bed(tmp_path):
    from bed_reader import to_bed
    source, traits = fixture()
    properties = dict(fid=[x[0] for x in source.samples], iid=[x[1] for x in source.samples],
        sid=list(source.variants.ids), chromosome=list(source.variants.chromosome),
        bp_position=list(source.variants.position), allele_1=list(source.variants.counted), allele_2=list(source.variants.other))
    path = tmp_path/"calls.bed"
    calls = np.where(source.values == -127, np.nan, source.values.astype(float))
    to_bed(path, calls, properties=properties, count_A1=True, num_threads=1)
    return source, traits, path


def test_native_bed_reads_orientation_missingness_masks_and_mutation(tmp_path):
    array_source, traits, path = write_bed(tmp_path)
    with FileGenotypeSource(path, genome_build="GRCh37") as source:
        assert source.variants.identity == array_source.variants.identity
        rows, variants = traits[1].rows, traits[1].variants
        source.prepare(rows, 100, 1)
        calls = source.read(variants)
        np.testing.assert_array_equal(calls, array_source.values[np.ix_(rows, variants)])
        source.prepare(np.array([rows[0]], dtype=np.int64), 100, 1)
        single = source.read(variants)
        np.testing.assert_array_equal(single, array_source.values[np.ix_([rows[0]], variants)])
        with path.open("r+b") as handle:
            handle.seek(3)
            original = handle.read(1)
            handle.seek(3)
            handle.write(bytes([original[0] ^ 2]))
        with pytest.raises(RuntimeError, match="modified"):
            source.read(variants)


def test_bed_fit_and_score_match_array_source(tmp_path):
    array_source, traits, path = write_bed(tmp_path)
    with FileGenotypeSource(path, genome_build="GRCh37") as source:
        bed_traits = [replace(t, scale=estimate_scale(source, t.rows, t.variants, block_size=8)) for t in traits]
        native_models = fit_prediction(bed_traits, source, output=tmp_path/"bed-fit",
            plan=plan_prediction(bed_traits, source, storage="compact", block_size=8, rhs_columns=6))
    array_models = fit_prediction(traits, array_source, output=tmp_path/"array-fit", backend="numpy")
    for a, b in zip(native_models, array_models):
        np.testing.assert_allclose(a.weights, b.weights, atol=1e-12, rtol=1e-12)
