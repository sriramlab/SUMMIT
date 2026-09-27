import copy
import numpy as np
import pytest
from prediction_helpers import prediction_threads

from summit.pcgc.artifacts import make_artifact, write_artifact, load_artifact, BinaryArtifact
from summit.pcgc.genotype import prepare_from_source
from summit.pcgc.reference import prepare_moments
from summit.pcgc.reference import population_ld_reference
from summit.pcgc.moments import external_ld_moments
from summit.pcgc.research import exact_moments
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.spec import GenotypeScale, VariantAxis
from summit.sumstats.binary import prepare_binary_risk
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator


def fixture(n=32, m=24):
    rng = np.random.default_rng(6139)
    raw = rng.binomial(2, .3, (n, m)).astype(float)
    scale_mean = np.full(m, .6)
    scale_inv = np.full(m, 1/np.sqrt(.42))
    raw[1, 2] = np.nan
    x = np.nan_to_num((raw-scale_mean)*scale_inv, nan=0.)
    axis = VariantAxis(tuple(f"rs{i}" for i in range(m)), ("1",)*m, tuple(range(1, m+1)), ("A",)*m, ("G",)*m, "test")
    risk = prepare_binary_risk(np.arange(n) % 2, .1, population_risk=np.linspace(.02, .25, n))
    scale = GenotypeScale(scale_mean, scale_inv, axis.identity, "declared_population",
                          {"population_scale": True, "source": "simulation"}, ddof=0)
    return raw, x, axis, risk, scale


def test_typed_artifact_roundtrip_tampering_and_no_overwrite(tmp_path):
    raw, x, axis, risk, scale = fixture()
    m = exact_moments(x, np.ones((len(axis.ids), 1)), risk)
    artifact = make_artifact(m, variant_axis=axis, annotation_names=["all"], sample_identity="a"*64,
                             genotype_scale_identity=scale.identity, risk=risk, diagnostics={"reference": "exact_test"})
    path = tmp_path/"binary.npz"
    write_artifact(artifact, path)
    loaded = load_artifact(path)
    np.testing.assert_array_equal(loaded.moments.rhs_rows, m.rhs_rows)
    assert loaded.manifest == artifact.manifest
    with pytest.raises(FileExistsError):
        write_artifact(artifact, path)
    header = copy.deepcopy(dict(artifact.manifest))
    header["n_samples"] += 1
    with pytest.raises(ValueError, match="sample count"):
        BinaryArtifact(m, header)
    header = copy.deepcopy(dict(artifact.manifest))
    header["risk"] = {"population_prevalence": .5}
    with pytest.raises(ValueError, match="metadata checksum"):
        BinaryArtifact(m, header)
    np.savez(tmp_path/"ordinary.npz", beta=np.zeros(3), se=np.ones(3))
    with pytest.raises(ValueError, match="contract"):
        load_artifact(tmp_path/"ordinary.npz")


def test_legacy_archive_remains_readable_but_rejects_uncertainty(tmp_path):
    from dataclasses import replace
    from summit.context.spec import array_sha256, canonical_json, canonical_sha256
    from summit.pcgc.artifacts import ARRAYS
    from summit.pcgc.moments import LEGACY_DIAGONAL, fit_moments
    _,x,axis,risk,scale=fixture()
    m=exact_moments(x,np.ones((len(axis.ids),1)),risk)
    artifact=make_artifact(m,variant_axis=axis,annotation_names=['all'],sample_identity='a'*64,
                           genotype_scale_identity=scale.identity,risk=risk,diagnostics={})
    f2=(x*risk.sensitivity[:,None])**2
    old=replace(m,ldscores=m.ldscores+f2.T @ (f2 @ m.annotations)/len(x)**2,ldscore_contract=LEGACY_DIAGONAL)
    header=copy.deepcopy(dict(artifact.manifest))
    header['schema_version']=1
    header.pop('ldscore_contract')
    header.pop('manifest_hash')
    header['array_hashes']={k:array_sha256(getattr(old,k)) for k in ARRAYS}
    header['manifest_hash']=canonical_sha256(header)
    path=tmp_path/'legacy.npz'
    np.savez(path,manifest_json=np.asarray(canonical_json(header)),**{k:getattr(old,k) for k in ARRAYS})
    loaded=load_artifact(path).moments
    np.testing.assert_allclose(fit_moments(loaded)['conditional_components'],fit_moments(m)['conditional_components'])
    with pytest.raises(ValueError,match='regenerate'):
        fit_moments(loaded,block_ids=np.arange(len(axis.ids)) % 4)


@pytest.mark.parametrize("method", ["pcgc", "pcgc-inverse", "pcgc-basis"])
def test_bed_native_two_pass_path_matches_array_python(tmp_path, method):
    native = pytest.importorskip("summit.gxeldcore")
    if getattr(native, "prediction_execution_version", 0) < 2:
        pytest.skip("native build lacks shared genotype source")
    from bed_reader import to_bed
    raw, x, axis, risk, scale = fixture(128, 256)
    path = tmp_path/"fixture.bed"
    n = len(raw)
    to_bed(path, raw, count_A1=True, num_threads=1, properties={
        "fid": [f"f{i}" for i in range(n)], "iid": [f"i{i}" for i in range(n)],
        "sid": list(axis.ids), "chromosome": list(axis.chromosome), "bp_position": list(axis.position),
        "allele_1": list(axis.counted), "allele_2": list(axis.other)})
    options = dict(probes=37, seed=17, block_size=41, threads=prediction_threads())
    if method == "pcgc-basis":
        options.update(basis=np.column_stack([np.ones(n), risk.sensitivity]), coefficients=np.array([0., 1.]))
    with FileGenotypeSource(path, genome_build="test") as source:
        artifact = prepare_from_source(source, scale, np.arange(n), risk, np.ones((len(axis.ids), 1)),
            annotation_names=["all"], method=method, native=True, **options)
    expected, _ = prepare_moments(ArraySequentialGenotypeOperator(x), np.ones((len(axis.ids), 1)),
                                   risk, method, native=False, **options)
    for name in ("rhs_rows", "ldscores", "same_person"):
        np.testing.assert_allclose(getattr(artifact.moments, name), getattr(expected, name), rtol=5e-12, atol=5e-10)
    assert artifact.manifest["diagnostics"]["genotype_passes"] == 2
    execution = artifact.manifest["diagnostics"]["reference_execution"]
    assert execution["probes"]["root_seed"] == 17
    assert execution["probes"]["probe_count"] == 37
    assert execution["native_build"]["blas_vendor"] in ("OPENBLAS", "BLIS", "OpenBLAS")


def test_fractional_pgen_two_pass_path_preserves_scale_and_missingness(tmp_path):
    native = pytest.importorskip("summit.gxeldcore")
    if getattr(native, "prediction_execution_version", 0) < 2:
        pytest.skip("native build lacks shared genotype source")
    pgenlib = pytest.importorskip("pgenlib")
    threads = prediction_threads()
    raw, _, axis, risk, scale = fixture(128, 256)
    rng = np.random.default_rng(8584)
    alt = rng.uniform(0, 2, raw.T.shape).astype(np.float32)
    alt[2, 1] = -9
    prefix = tmp_path/"dosages"
    with pgenlib.PgenWriter(bytes(prefix.with_suffix(".pgen")), sample_ct=len(raw), variant_ct=len(axis.ids),
                           nonref_flags=False, dosage_present=True) as writer:
        writer.append_dosages_batch(np.ascontiguousarray(alt))
    prefix.with_suffix(".psam").write_text("#FID\tIID\n"+"".join(f"f{i}\ti{i}\n" for i in range(len(raw))))
    prefix.with_suffix(".pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n"+
        "".join(f"1\t{j+1}\trs{j}\tA\tG\n" for j in range(len(axis.ids))))
    with FileGenotypeSource(prefix, genome_build="test") as source:
        source.prepare(np.arange(len(raw)), len(axis.ids), threads)
        decoded = source.read(np.arange(len(axis.ids)))
        expected_raw = 2-alt.T
        expected_raw[alt.T == -9] = -127
        np.testing.assert_allclose(decoded, expected_raw, rtol=0, atol=1/16384)
        x = np.where(decoded == -127, 0, (decoded-scale.mean)*scale.inverse_scale)
        artifact = prepare_from_source(source, scale, np.arange(len(raw)), risk, np.ones((len(axis.ids), 1)),
                                       annotation_names=["all"], probes=31, seed=73, block_size=47, threads=threads)
    expected, _ = prepare_moments(ArraySequentialGenotypeOperator(x, genotype_format="pgen"),
                                 np.ones((len(axis.ids), 1)), risk, probes=31, seed=73, block_size=47, native=False)
    np.testing.assert_allclose(artifact.moments.ldscores, expected.ldscores, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(artifact.moments.rhs_rows, expected.rhs_rows, rtol=1e-11, atol=1e-9)


def test_external_bed_reference_uses_one_study_and_two_reference_passes(tmp_path):
    native = pytest.importorskip("summit.gxeldcore")
    if getattr(native, "prediction_execution_version", 0) < 2:
        pytest.skip("native build lacks shared genotype source")
    from bed_reader import to_bed
    raw, x, axis, risk, scale = fixture(128, 256)
    ref_raw = np.random.default_rng(944).binomial(2, .3, raw.shape).astype(float)
    for label, values in (("study", raw), ("reference", ref_raw)):
        to_bed(tmp_path/f"{label}.bed", values, count_A1=True, num_threads=1, properties={
            "fid": [label]*len(raw), "iid": [f"i{i}" for i in range(len(raw))],
            "sid": list(axis.ids), "chromosome": list(axis.chromosome), "bp_position": list(axis.position),
            "allele_1": list(axis.counted), "allele_2": list(axis.other)})
    with FileGenotypeSource(tmp_path/"study.bed", genome_build="test") as study, \
         FileGenotypeSource(tmp_path/"reference.bed", genome_build="test") as reference:
        artifact = prepare_from_source(study, scale, np.arange(len(raw)), risk, np.ones((len(axis.ids), 1)),
            annotation_names=["all"], method="pcgc-ld", reference_source=reference, probes=61, seed=39,
            block_size=31, threads=prediction_threads())
    a = np.ones((len(axis.ids), 1))
    ref_x = (ref_raw-scale.mean)*scale.inverse_scale
    population_ld, _ = population_ld_reference(ArraySequentialGenotypeOperator(ref_x), a,
                                                probes=61, seed=39, block_size=31, native=False)
    expected = external_ld_moments(exact_moments(x, a, risk).rhs_rows, a, risk, population_ld)
    for name in ("ldscores", "rhs_rows", "same_person"):
        np.testing.assert_allclose(getattr(artifact.moments, name), getattr(expected, name), rtol=1e-11, atol=2e-9)
    assert artifact.manifest["diagnostics"]["study_genotype_passes"] == 1
    assert artifact.manifest["diagnostics"]["reference_genotype_passes"] == 2
    assert artifact.manifest["diagnostics"]["reference_execution"]["probes"]["root_seed"] == 39


def test_noncontiguous_variant_subset_keeps_population_scale_and_artifact_axis(tmp_path):
    pytest.importorskip('summit.gxeldcore')
    from bed_reader import to_bed
    raw, x, axis, risk, scale = fixture(128, 256)
    path = tmp_path/'subset.bed'
    to_bed(path, raw, count_A1=True, num_threads=1, properties={
        'fid':[f'f{i}' for i in range(len(raw))], 'iid':[f'i{i}' for i in range(len(raw))],
        'sid':list(axis.ids), 'chromosome':list(axis.chromosome), 'bp_position':list(axis.position),
        'allele_1':list(axis.counted), 'allele_2':list(axis.other)})
    selected = np.arange(0, len(axis.ids), 3)
    a = np.ones((len(selected),1))
    with FileGenotypeSource(path, genome_build='test') as source:
        artifact = prepare_from_source(source, scale, np.arange(len(raw)), risk, a,
            annotation_names=['all'], variant_indices=selected, probes=67, seed=715, block_size=31,
            threads=prediction_threads())
    expected, _ = prepare_moments(ArraySequentialGenotypeOperator(x[:,selected]), a, risk,
                                  native=False, probes=67, seed=715, block_size=31)
    assert tuple(artifact.manifest['variant_axis']['ids']) == axis.subset(selected).ids
    assert artifact.manifest['schema_version'] == 2
    np.testing.assert_allclose(artifact.moments.ldscores, expected.ldscores, atol=1e-11)
    np.testing.assert_allclose(artifact.moments.rhs_rows, expected.rhs_rows, atol=1e-8)


def test_decoder_allocation_waits_for_admitted_block_width(tmp_path, monkeypatch):
    pytest.importorskip('summit.gxeldcore')
    from bed_reader import to_bed
    from summit.pcgc.genotype import scaled_file_operator
    from summit.pcgc.rank_one import plan_pcgc_reference
    raw,x,axis,risk,scale=fixture(128,256)
    path=tmp_path/'admission.bed'
    to_bed(path,raw,count_A1=True,num_threads=1,properties={
        'fid':[f'f{i}' for i in range(len(raw))], 'iid':[f'i{i}' for i in range(len(raw))],
        'sid':list(axis.ids), 'chromosome':list(axis.chromosome), 'bp_position':list(axis.position),
        'allele_1':list(axis.counted), 'allele_2':list(axis.other)})
    a=np.ones((len(axis.ids),1))
    threads=prediction_threads()
    full=plan_pcgc_reference(num_samples=128,num_variants=256,num_annotations=1,probes=31,block_size=41,threads=threads)
    budget=full.peak_resident_bytes-1
    bounded=plan_pcgc_reference(num_samples=128,num_variants=256,num_annotations=1,probes=31,block_size=41,memory_bytes=budget,threads=threads)
    with FileGenotypeSource(path,genome_build='test') as source:
        calls=[]
        original=source.prepare
        def record(rows,width,threads):
            calls.append(width)
            return original(rows,width,threads)
        monkeypatch.setattr(source,'prepare',record)
        op=scaled_file_operator(source,scale,np.arange(128),block_size=41,threads=threads)
        assert calls==[]
        with pytest.raises(MemoryError):
            prepare_moments(op,a,risk,probes=31,block_size=41,memory_bytes=1,threads=threads)
        assert calls==[]
        actual,diagnostics=prepare_moments(op,a,risk,probes=31,block_size=41,memory_bytes=budget,threads=threads)
        assert calls==[bounded.tiling['variant_block_width']]
        assert diagnostics['genotype_passes']==2
    expected,_=prepare_moments(ArraySequentialGenotypeOperator(x),a,risk,probes=31,block_size=41,native=False)
    np.testing.assert_allclose(actual.ldscores,expected.ldscores,rtol=1e-11,atol=1e-11)
