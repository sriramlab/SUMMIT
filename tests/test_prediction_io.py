from __future__ import annotations

from dataclasses import replace
import json
import numpy as np
import pytest

from prediction_helpers import prediction_threads

from test_prediction_core import fixture
from summit.prediction.api import fit_prediction
from summit.prediction.artifacts import load_prediction_models, load_genotype_scale, write_genotype_scale
from summit.prediction.batch import plan_prediction
from summit.prediction.genotype import ArrayGenotypeSource, FileGenotypeSource, standardize, estimate_scale
from summit.prediction.score import ScoreInput, score_prediction
from summit.prediction.spec import VariantAxis


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_cross_trait_scoring_shares_affine_groups(tmp_path, backend):
    source, traits = fixture()
    traits = [replace(t, candidates=t.candidates[:1]) for t in traits]
    traits.append(replace(traits[0], id="other", y=traits[0].y[::-1]))
    models = fit_prediction(traits, source, output=tmp_path/"fit", backend=backend,
        plan=plan_prediction(traits, source, storage="packed", block_size=9,
                             threads=prediction_threads()))
    manifest = json.loads((tmp_path / "fit/manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert len(manifest["variant_axes"]) == 2
    assert manifest["traits"][0]["variants"] == manifest["traits"][2]["variants"]
    assert manifest["traits"][0]["scale"] == manifest["traits"][2]["scale"]
    assert models[0].variants is models[2].variants
    assert models[0].scale is models[2].scale
    inputs = {t.id: ScoreInput(t.rows, t.phi, t.fixed, t.context_spec, t.fixed_spec)
              for t in traits}
    batch = score_prediction(models, source, inputs, backend=backend, block_size=7,
                             rhs_columns=6, threads=prediction_threads())
    assert batch.report["affine_groups"] == 2
    assert batch.report["ledger"]["traversals"] == {"scoring": 1}
    # The requested block need not fit when adaptive scoring is explicitly
    # enabled. Admit exactly three columns using the same workspace contract.
    nrows=len(np.unique(np.concatenate([value.rows for value in inputs.values()])))
    nmax=max(len(value.rows) for value in inputs.values())
    limited=(batch.report['allocated_output_bytes']+batch.report['model_and_axis_bytes']
        +64*nmax*6+256*2**20+32*nrows*3)
    with pytest.raises(MemoryError,match='bounded blocks exceed budget'):
        score_prediction(models,source,inputs,backend=backend,block_size=7,rhs_columns=6,
            threads=prediction_threads(),memory_bytes=limited)
    adaptive=score_prediction(models,source,inputs,backend=backend,block_size=7,rhs_columns=6,
        threads=prediction_threads(),memory_bytes=limited,adaptive_blocks=True)
    assert adaptive.report['block_size']==3 and adaptive.report['requested_block_size']==7
    assert (adaptive.report['allocated_output_bytes']+adaptive.report['estimated_scratch_bytes']
        +adaptive.report['model_and_axis_bytes'])<=limited
    for key in batch.components:
        np.testing.assert_allclose(adaptive.components[key],batch.components[key],rtol=2e-13,atol=2e-14)
        np.testing.assert_allclose(adaptive.prediction[key],batch.prediction[key],rtol=2e-13,atol=2e-14)
    with pytest.raises(MemoryError,match='bounded blocks exceed budget'):
        score_prediction(models,source,inputs,backend=backend,block_size=7,rhs_columns=6,
            threads=prediction_threads(),memory_bytes=limited-32*nrows*3,adaptive_blocks=True)
    component_inputs = {key:replace(value,fixed=np.empty((len(value.rows),0)))
                        for key,value in inputs.items()}
    component_scores = score_prediction(models,source,component_inputs,backend=backend,
        block_size=8,threads=prediction_threads(),components_only=True)
    assert not component_scores.genetic and not component_scores.prediction
    assert component_scores.report["allocated_output_bytes"] < batch.report["allocated_output_bytes"]
    for key, values in batch.components.items():
        np.testing.assert_allclose(component_scores.components[key],values,rtol=2e-13,atol=2e-14)
    for model in models:
        one = score_prediction([model], source, {model.trait_id: inputs[model.trait_id]},
                               backend=backend, block_size=11, threads=prediction_threads())
        np.testing.assert_allclose(batch.components[model.key], one.components[model.key],
                                   rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(batch.prediction[model.key], one.prediction[model.key],
                                   rtol=2e-13, atol=2e-14)


def test_shared_axis_legacy_loading_conversion_and_member_authentication(tmp_path):
    from summit.prediction.artifacts import file_digest, json_record
    from scripts.epistasis.repack_prediction_artifact import repack
    source, traits = fixture()
    traits = [replace(traits[0],candidates=traits[0].candidates[:1]),
              replace(traits[0],id="new",candidates=traits[0].candidates[:1])]
    original = fit_prediction(traits,source,output=tmp_path/"legacy",backend="numpy")
    path = tmp_path/"legacy/manifest.json"
    manifest = json.loads(path.read_text())
    for trait in manifest["traits"]:
        trait["variants"] = json.loads((path.parent/manifest["variant_axes"][trait["variants"]]["file"]).read_text())
    manifest.pop("variant_axes")
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest))
    complete = json.loads((path.parent/"COMPLETE.json").read_text())
    complete.update(schema_version=1,manifest_sha256=file_digest(path))
    (path.parent/"COMPLETE.json").write_text(json.dumps(complete))
    legacy = load_prediction_models(path.parent)
    report = repack(path.parent,tmp_path/"compact")
    assert report["models"] == 2 and report["shared_variant_axes"] == report["shared_scales"] == 1
    compact = load_prediction_models(tmp_path/"compact")
    for old,new,expected in zip(legacy,compact,original):
        np.testing.assert_array_equal(old.weights,new.weights)
        np.testing.assert_array_equal(new.weights,expected.weights)
        assert old.variants.identity == new.variants.identity
        assert old.scale.identity == new.scale.identity
    with pytest.raises(FileExistsError):
        repack(path.parent,tmp_path/"compact")
    axis = tmp_path/"compact/variant-axis-0.json"
    axis.write_text(axis.read_text().replace('rs2','zz2'))
    with pytest.raises(ValueError,match="checksum"):
        load_prediction_models(tmp_path/"compact")
    path = tmp_path/"compact/manifest.json"
    manifest = json.loads(path.read_text())
    key = next(iter(manifest["variant_axes"]))
    manifest["variant_axes"][key] = json_record(axis)
    path.write_text(json.dumps(manifest))
    complete = json.loads((path.parent/"COMPLETE.json").read_text())
    complete["manifest_sha256"] = file_digest(path)
    (path.parent/"COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError,match="axis identity"):
        load_prediction_models(path.parent)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_model_reload_score_raw_conversion_and_allele_reorder(tmp_path, backend):
    source, traits = fixture()
    models = fit_prediction(traits, source, output=tmp_path/"fit",
        plan=plan_prediction(traits, source, storage="compact", block_size=9, rhs_columns=6, threads=prediction_threads()), backend=backend)
    inputs = {t.id: ScoreInput(t.rows, t.phi, t.fixed, t.context_spec, t.fixed_spec) for t in traits}
    scores = score_prediction(models, source, inputs, backend=backend, block_size=11, rhs_columns=6, threads=prediction_threads())
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
    rescored = score_prediction(models, other, inputs, backend=backend, block_size=7, threads=prediction_threads())
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
        source.prepare(rows, 100, prediction_threads())
        calls = source.read(variants)
        np.testing.assert_array_equal(calls, array_source.values[np.ix_(rows, variants)])
        source.prepare(np.array([rows[0]], dtype=np.int64), 100, prediction_threads())
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
        bed_traits = [replace(t, scale=estimate_scale(source, t.rows, t.variants, block_size=8, threads=prediction_threads())) for t in traits]
        native_models = fit_prediction(bed_traits, source, output=tmp_path/"bed-fit",
            plan=plan_prediction(bed_traits, source, storage="compact", block_size=8, rhs_columns=6, threads=prediction_threads()))
    array_models = fit_prediction(traits, array_source, output=tmp_path/"array-fit", backend="numpy")
    for a, b in zip(native_models, array_models):
        np.testing.assert_allclose(a.weights, b.weights, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("build", [None, "GRCh37"])
def test_optional_build_preserves_position_and_allele_checks(build):
    from summit.prediction.score import align_variants
    source, _ = fixture()
    model = replace(source.variants, genome_build=build)
    for label in (None, build):
        rows, cols, flips = align_variants(model, replace(model, genome_build=label))
        np.testing.assert_array_equal(rows, cols)
        assert not flips.any()
    if build is not None:
        with pytest.raises(ValueError, match="genome build"):
            align_variants(model, replace(model, genome_build="GRCh38"))
    with pytest.raises(ValueError, match="genomic position"):
        align_variants(model, replace(model, genome_build=None,
                                      position=(model.position[0]+1, *model.position[1:])))
    bad = next(a for a in "ACGT" if a not in (model.counted[0], model.other[0]))
    with pytest.raises(ValueError, match="allele mismatch"):
        align_variants(model, replace(model, genome_build=None,
                                      counted=(bad, *model.counted[1:])))
