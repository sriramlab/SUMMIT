from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from summit.context.fit import ContextRankError
from summit.epistasis.oracle import (
    selected_kernels, explicit_pair_features, group_kernel, dense_summary,
    gaussian_quadratic_covariance,
)
from summit.epistasis.prepare import SelectedStudy, fit_scale, block_jackknife
from summit.epistasis.summary import fit_epistasis, load_summary, write_summary
from summit.prediction.genotype import ArrayGenotypeSource
from summit.prediction.spec import VariantAxis
from summit.epistasis.models import annotation_weights, target_design
from summit.epistasis.pairs import (prepare_pair_scores, pair_tests, PairScores,
                                   write_pair_scores, load_pair_scores, combine_independent_pair_scores)
from summit.epistasis.groups import GroupKernel, remaining_genome_weights, exact_operator_summary


def fixture(seed=14, n=72, m=14, backend="numpy", block_size=5, threads=1):
    rng = np.random.default_rng(seed)
    raw = rng.binomial(2, np.linspace(.2, .5, m), (n, m)).astype(float)
    raw[0, 3] = np.nan
    axis = VariantAxis(tuple(f"v{i}" for i in range(m)), ("1",)*m,
                       tuple(range(1, m+1)), ("A",)*m, ("G",)*m)
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis)
    scale = fit_scale(source, np.arange(n), block_size=block_size, threads=threads)
    x = (raw - scale.mean) * scale.inverse_scale
    x[np.isnan(x)] = 0
    e = np.column_stack([np.ones(n), x[:, 0], x[:, 0]])
    w = np.ones((m, 3))
    w[:, 2] = np.linspace(0, 1, m)
    w[0, 1:] = 0
    fixed = np.column_stack([np.ones(n), x[:, 0], rng.normal(size=n)])
    y = rng.normal(size=(n, 2))
    study = SelectedStudy(source, np.arange(n), scale, fixed_effects=fixed,
        modifiers=e, weights=w, component_names=("additive", "epi_all", "epi_weighted"),
        definitions={"target": "v0", "exclusions": ["v0"]}, backend=backend,
        block_size=block_size, threads=threads)
    return study, x, e, w, fixed, y


def test_streamed_exact_moments_and_analytic_covariance():
    study, x, e, w, fixed, y = fixture()
    reference = study.reference(exact=True)
    observed, _ = study.summarize(reference, y, trait_names=("a", "b"))
    kernels, p = selected_kernels(x, e, w, fixed)
    expected = dense_summary(kernels, y, component_names=observed.component_names,
        trait_names=observed.trait_names, residual_rank=p.residual_rank, metadata={"oracle": True})
    for key in ("matrix", "rhs", "traces", "cubic"):
        np.testing.assert_allclose(getattr(observed, key), getattr(expected, key), rtol=2e-12, atol=2e-10)
    assert study.stream.ledger.traversals == {"reference_source": 1, "reference_target": 1,
                                             "trait_actions": 1, "trait_analytic_moments": 1}
    fit = fit_epistasis(observed)
    v = np.einsum("a,aij->ij", fit["coefficients"], kernels)
    # This checks the phenotype plug-in, not the different Gaussian expectation.
    a = kernels @ (p.projector @ y[:, 0])
    cq = 2 * a @ v @ a.T
    inv = np.linalg.inv(observed.matrix)
    np.testing.assert_allclose(fit["covariance"], inv @ cq @ inv.T, atol=1e-12)
    np.testing.assert_allclose(fit["variance_contributions"].sum(), observed.rhs[-1, 0]/p.residual_rank)
    assert np.isclose(fit["proportions"].sum(), 1)


@pytest.mark.parametrize("within", [False, True])
def test_group_identity_overlap_self_removal_symmetry(within):
    _, x, _, _, fixed, _ = fixture()
    a = np.r_[np.ones(7), np.zeros(7)]
    b = a if within else np.linspace(0, 2, 14)
    f, pairs, mass = explicit_pair_features(x, a, b, within=within)
    assert all(i < j for i, j in pairs)
    p = np.eye(len(x)) - fixed @ np.linalg.pinv(fixed)
    expected = (p @ f) @ (p @ f).T
    k = group_kernel(x, a, b, fixed, within=within)
    np.testing.assert_allclose(k, expected, atol=1e-13)
    np.testing.assert_allclose(k, group_kernel(x, b, a, fixed, within=within), atol=1e-13)
    assert mass > 0
    # Projection before Hadamard multiplication defines another kernel.
    ka = p @ ((x*a) @ x.T / a.sum()) @ p
    kb = p @ ((x*b) @ x.T / b.sum()) @ p
    assert not np.allclose(k, ka*kb)
    operator = GroupKernel(x, a, b, fixed_effects=fixed, within=within)
    rhs = np.random.default_rng(552).normal(size=(len(x), 3))
    np.testing.assert_allclose(operator.apply(rhs), k @ rhs, atol=1e-12)
    np.testing.assert_allclose(operator.apply(rhs[:, 0]), k @ rhs[:, 0], atol=1e-12)
    assert np.array_equal(remaining_genome_weights(2*a), a == 0)


def test_batch_tiling_and_allele_recoding():
    study, x, e, w, fixed, y = fixture(block_size=3)
    r = study.reference(exact=True)
    s, _ = study.summarize(r, y, trait_names=("a", "b"))
    other, *_ = fixture(block_size=7)
    rr = other.reference(exact=True)
    np.testing.assert_allclose(r.matrix, rr.matrix, atol=1e-11)
    one, _ = other.summarize(rr, y[:, 1], trait_names=("b",))
    np.testing.assert_allclose(s.cubic[1], one.cubic[0], atol=2e-10)
    k, _ = selected_kernels(x, e, w, fixed)
    x[:, 0] *= -1
    e[:, 1:] *= -1
    recoded, _ = selected_kernels(x, e, w, fixed)
    np.testing.assert_allclose(k, recoded, atol=1e-13)
    perm = np.random.default_rng(24).permutation(len(x))
    permuted, _ = selected_kernels(x[perm], e[perm], w, fixed[perm])
    np.testing.assert_allclose(recoded[:, perm][:, :, perm], permuted, atol=1e-13)


def test_saved_summary_fresh_process_without_genotypes(tmp_path):
    study, *_, y = fixture()
    summary, _ = study.summarize(study.reference(exact=True), y, trait_names=("a", "b"))
    path = write_summary(summary, tmp_path / "study.npz")
    restored = load_summary(path)
    np.testing.assert_array_equal(summary.cubic, restored.cubic)
    with pytest.raises(FileExistsError):
        write_summary(summary, path)
    # Child has only the published summary; no input files or inherited arrays.
    code = ("import sys; sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']; "
            "from summit.epistasis import load_summary,fit_epistasis; s=load_summary(sys.argv[1]); print(fit_epistasis(s)['rank'])")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "4"
    # Corruption must be rejected before any inference is attempted.
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["rhs"] = arrays["rhs"].copy()
    arrays["rhs"][0, 0] += 1
    np.savez(tmp_path/"corrupt.npz", **arrays)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_summary(tmp_path/"corrupt.npz")


def test_random_reference_and_block_deletions():
    study, *_, y = fixture()
    exact = study.reference(exact=True)
    random = study.reference(nvecs=8192, seed=33)
    np.testing.assert_allclose(random.matrix, exact.matrix, rtol=.07, atol=1)
    summary, rows = study.summarize(exact, y, trait_names=("a", "b"), retain_variant_rows=True)
    blocks = np.arange(study.m) % 4
    result = block_jackknife(exact, summary, rows, blocks)
    assert result["failures"] == [None]*4
    from dataclasses import replace
    mismatched = replace(summary, metadata=dict(summary.metadata, definition_hash="different"))
    with pytest.raises(ValueError, match="definitions disagree"):
        block_jackknife(exact, mismatched, rows, blocks)
    # Independent reconstruction of every declared frozen-source deletion.
    for label in range(4):
        keep = blocks != label
        mass = study.weights[keep].sum(axis=0)
        d = exact.directional[keep].sum(axis=0)
        t = summary.matrix.copy()
        t[:-1, :-1] = (d+d.T)/(2*np.outer(mass, mass))
        t[:-1, -1] = t[-1, :-1] = exact.information[keep].sum(axis=0)/mass
        q = np.r_[rows[keep, :, 0].sum(axis=0)/mass, summary.rhs[-1, 0]]
        np.testing.assert_allclose(result["coefficients"][label], np.linalg.solve(t, q), atol=1e-12)


def test_rank_failure_negative_estimates_and_invalid_inputs():
    study, x, e, w, fixed, y = fixture()
    e[:, 2] = e[:, 1]
    w[:, 2] = w[:, 1]
    kernels, p = selected_kernels(x, e, w, fixed)
    s = dense_summary(kernels, y, component_names=(*study.names, "residual"),
                      trait_names=("a", "b"), residual_rank=p.residual_rank, metadata={"test": True})
    with pytest.raises(ContextRankError):
        fit_epistasis(s)
    with pytest.raises(ValueError, match="positive mass"):
        selected_kernels(x, e, np.zeros_like(w), fixed)
    with pytest.raises(ValueError, match="no nonself"):
        group_kernel(x, np.eye(1, len(w))[0], np.eye(1, len(w))[0], fixed)
    study.memory_bytes = 1
    with pytest.raises(MemoryError, match="budget"):
        study.reference(exact=True)


def test_gaussian_covariance_ingredients_by_independent_formula():
    study, x, e, w, fixed, _ = fixture()
    k, p = selected_kernels(x, e, w, fixed)
    v = .2*k[0] + .1*k[1] + .05*k[2] + .7*k[-1]
    observed = gaussian_quadratic_covariance(k, v)
    expected = np.array([[2*np.trace(v @ a @ v @ b) for b in k] for a in k])
    np.testing.assert_allclose(observed, expected, atol=2e-12)


def test_target_design_exclusions_main_effects_and_score_distinction():
    study, x, *_ = fixture()
    annotations = annotation_weights(study.source.variants.ids, {"region": {"v1": 2., "v2": 1.}})
    design = target_design(study.source, study.rows, study.scale,
        components=[dict(name="target", target="v0", background="region")],
        annotations=annotations, additive_annotations=["all", "region"], local_variants=["v3"])
    assert design["weights"][0, -1] == 0
    np.testing.assert_allclose(design["modifiers"][:, -1], x[:, 0])
    np.testing.assert_allclose(design["fixed_effects"][:, -1], x[:, 0])
    with pytest.raises(ValueError, match="overlaps"):
        target_design(study.source, study.rows, study.scale,
            components=[dict(name="score", score={"v0": 1., "v1": -1.}, background="all")],
            annotations=annotations, additive_annotations=["all"])
    with pytest.raises(ValueError, match="unknown annotation"):
        annotation_weights(study.source.variants.ids, {"bad": {"unknown": 1.}})


def test_pair_gls_signed_recoding_and_exact_rank_one_tail():
    study, x, _, _, fixed, y = fixture()
    v = .7*np.eye(len(x)) + .3*x@x.T/x.shape[1]
    pairs = (("v0", "v1"), ("v0", "v2"))
    score = prepare_pair_scores(x, study.source.variants, pairs, y, fixed_effects=fixed,
        covariance_solve=lambda z: np.linalg.solve(v, z), covariance_identity="known_v",
        covariance_known=True, trait_names=("a", "b"), sample_identity="samples")
    f = np.column_stack([x[:, 0]*x[:, 1], x[:, 0]*x[:, 2]])
    c = np.column_stack([fixed, x[:, 0], x[:, 1], x[:, 2]])
    vi = np.linalg.inv(v)
    r = vi-vi@c@np.linalg.pinv(c.T@vi@c)@c.T@vi
    np.testing.assert_allclose(score.scores, f.T@r@y, atol=2e-12)
    np.testing.assert_allclose(score.information, f.T@r@f, atol=2e-12)
    tests = pair_tests(score, burden_weights=[1., -1.])
    assert tests["adaptive_bonferroni_p"] >= min(tests["joint_p"], tests["burden_p"], tests["sparse_bonferroni_p"])
    x[:, 1] *= -1
    flipped = prepare_pair_scores(x, study.source.variants, pairs, y, fixed_effects=fixed,
        covariance_solve=lambda z: np.linalg.solve(v, z), covariance_identity="known_v",
        covariance_known=True, trait_names=("a", "b"), sample_identity="samples")
    np.testing.assert_allclose(flipped.scores, score.scores * np.array([-1, 1])[:, None], atol=1e-12)
    np.testing.assert_allclose(pair_tests(flipped)["marginal_p_two_sided"], tests["marginal_p_two_sided"], atol=1e-13)


def test_native_selected_matches_numpy_when_extension_is_loaded():
    try:
        from summit import gxeldcore
    except ImportError:
        pytest.skip("native extension is not loaded")
    assert gxeldcore.build_info()["gemm_integrity_enabled"]
    from epistasis_helpers import epistasis_threads
    threads = epistasis_threads()
    study, *_, y = fixture(backend="native", threads=threads)
    native, _ = study.summarize(study.reference(exact=True), y, trait_names=("a", "b"))
    reference, *_, y = fixture(threads=threads)
    numpy, _ = reference.summarize(reference.reference(exact=True), y, trait_names=("a", "b"))
    for key in ("matrix", "rhs", "cubic"):
        np.testing.assert_allclose(getattr(native, key), getattr(numpy, key), atol=2e-10)
    native_random = study.reference(nvecs=64, seed=87)
    numpy_random = reference.reference(nvecs=64, seed=87)
    np.testing.assert_allclose(native_random.matrix, numpy_random.matrix, atol=2e-10)


def test_heteroskedastic_extension_matches_explicit_projected_residual_kernels():
    study, x, e, w, fixed, y = fixture()
    extended = SelectedStudy(study.source, study.rows, study.scale,
        fixed_effects=fixed, modifiers=e[:, :2], weights=w[:, :2],
        component_names=study.names[:2], definitions={"extension": "heteroskedastic"},
        residual_basis=np.column_stack([e[:, 1]**2, np.ones(len(x))]),
        residual_names=("residual_target_square", "residual"), backend="numpy")
    reference = extended.reference(exact=True)
    summary, rows = extended.summarize(reference, y, trait_names=("a", "b"), retain_variant_rows=True)
    k, projection = selected_kernels(x, e[:, :2], w[:, :2], fixed)
    p = projection.projector
    k = np.concatenate([k[:-1], ((p*e[:, 1]**2)@p)[None], k[-1:]])
    expected = dense_summary(k, y, component_names=summary.component_names,
        trait_names=summary.trait_names, residual_rank=projection.residual_rank, metadata={"oracle": True})
    for key in ("matrix", "rhs", "traces", "cubic"):
        np.testing.assert_allclose(getattr(summary, key), getattr(expected, key), rtol=2e-12, atol=3e-10)
    jk = block_jackknife(reference, summary, rows, np.arange(len(w))%3)
    assert jk["coefficients"].shape == (3, 4)


def test_pair_artifacts_and_common_effect_cohort_combination(tmp_path):
    metadata = dict(covariance_known=True, covariance_identity="V", sample_identity="cohort1",
        variants={"ids": ["a", "b", "c"], "counted": ["A"]*3, "other": ["G"]*3},
        inference="exact_gaussian_known_covariance", overlap_contract="mutually_disjoint_cohorts",
        effect_units="raw_y_per_common_scaled_product", genotype_scale_contract="fixed_common_affine")
    first = PairScores([[1.], [-2.]], [[2., .2], [.2, 3.]], (("a", "b"), ("a", "c")), ("y",), metadata)
    second = PairScores([[3.], [1.]], [[4., .1], [.1, 2.]], first.pair_ids, first.trait_names,
                        dict(metadata, sample_identity="cohort2"))
    restored = load_pair_scores(write_pair_scores(first, tmp_path/"pairs.npz"))
    np.testing.assert_array_equal(restored.scores, first.scores)
    combined = combine_independent_pair_scores([first, second])
    np.testing.assert_array_equal(combined.scores, [[4.], [-1.]])
    np.testing.assert_allclose(combined.information, [[6., .3], [.3, 5.]])
    with pytest.raises(ValueError, match="duplicate cohort"):
        combine_independent_pair_scores([first, first])
    with pytest.raises(ValueError, match="duplicate cohort"):
        combine_independent_pair_scores([combined, second])
    with pytest.raises(ValueError, match="invalid trait"):
        pair_tests(first, trait=-1)
    with pytest.raises(ValueError, match="contract"):
        combine_independent_pair_scores([first, PairScores(second.scores, second.information,
            second.pair_ids, second.trait_names, dict(second.metadata, overlap_contract="unknown"))])


def test_cli_manifest_native_preparation_and_inference(tmp_path):
    try:
        from summit import gxeldcore
    except ImportError:
        pytest.skip("native extension is not loaded")
    import json
    from bed_reader import to_bed
    from epistasis_helpers import entrypoint as main
    n, m = 80, 12
    rng = np.random.default_rng(821)
    raw = rng.binomial(2, .4, (n, m)).astype(float)
    ids = [str(i) for i in range(n)]
    to_bed(tmp_path/"input.bed", raw, properties=dict(fid=ids, iid=ids,
        sid=[f"v{i}" for i in range(m)], chromosome=["1"]*m,
        bp_position=np.arange(1, m+1), allele_1=["A"]*m, allele_2=["G"]*m))
    (tmp_path/"samples.tsv").write_text("FID IID\n"+"".join(f"{i} {i}\n" for i in range(n)))
    y = rng.normal(size=(n, 2))
    (tmp_path/"pheno.tsv").write_text("FID IID a b\n"+"".join(f"{i} {i} {y[i,0]} {y[i,1]}\n" for i in range(n)))
    manifest = dict(kind="summit.epistasis.prepare", schema_version=1, genotypes={"geno": "input.bed"},
        samples="samples.tsv", phenotypes={"file": "pheno.tsv", "columns": ["a", "b"], "unit": "simulation_units"},
        annotations={"a": {"v0": 1., "v1": 1.}, "b": {"v2": 1., "v3": 2.}},
        jobs=[dict(id="v0", additive_annotations=["all"],
            components=[dict(name="epi", target="v0", background="all")]),
            dict(id="groups", additive_annotations=["all", "a", "b"], groups=[
                dict(name="cross", mode="cross", left="a", right="b"),
                dict(name="within", mode="within", left="b"),
                dict(name="remainder", mode="remainder", left="a")]),
            dict(id="pairs", additive_annotations=["all"], pairs=[["v0", "v2"], ["v0", "v3"]],
                 nuisance=dict(additive_coefficients={"all": .3}, residual_variance=.7,
                               known=False, identity="declared_plugin"), burden_weights=[1., -1.]),
            dict(id="linear", additive_annotations=["all"], pairs=[["v0", "v2"], ["v0", "v3"]],
                 inference=dict(method="linear_exact",main_effects="all_genotypes",save_reference=True),burden_weights=[1.,-1.]),
            dict(id="linear_components",additive_annotations=["all","a","b"],
                 components=[dict(name="a",target="v0",background="a"),dict(name="b",target="v0",background="b")],
                 inference=dict(method="linear_exact",main_effects="all_genotypes")),
            dict(id="estimated", additive_annotations=["all"],
                 components=[dict(name="epi",target="v0",background="all")],
                 inference=dict(method="reml_bootstrap",bootstrap_draws=19)),
            dict(id="estimated_pairs",additive_annotations=["all"],pairs=[["v0","v2"],["v0","v3"]],
                 inference=dict(method="reml_bootstrap",bootstrap_draws=19),burden_weights=[1.,-1.])])
    manifest["jobs"].extend(dict(id=f"batch{i}",additive_annotations=["all"],
        components=[dict(name="epi",target=f"v{i}",background="all")],
        inference=dict(method="linear_exact",feature_sketch_dimensions=8,main_effects="all_genotypes")) for i in range(2))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    assert main(["epistasis", "prepare", str(tmp_path/"manifest.json"), "--out", str(tmp_path/"prepared"), "--exact"]) == 0
    summary_path = tmp_path/"prepared/v0.epistasis.npz"
    assert json.loads((tmp_path/"prepared/preparation.json").read_text())["shared_target_source_passes"]==1
    summary = load_summary(summary_path)
    assert summary.component_names == ("additive:all", "epi", "residual")
    assert main(["epistasis", "fit", str(summary_path), "--out", str(tmp_path/"fit.json")]) == 0
    result = json.loads((tmp_path/"fit.json").read_text())
    assert len(result["fits"]) == 2
    assert all(f["rank"] == 3 for f in result["fits"])
    group = load_summary(tmp_path/"prepared/groups.epistasis.npz")
    assert group.component_names[-4:] == ("cross", "within", "remainder", "residual")
    assert group.metadata["reference_probe_axis"] == "deterministic_sample_identity"
    assert fit_epistasis(group)["rank"] == 7
    pair_path = tmp_path/"prepared/pairs.pairs.npz"
    pair_summary = load_pair_scores(pair_path)
    assert not pair_summary.metadata["covariance_known"]
    assert main(["epistasis", "fit", str(pair_path), "--out", str(tmp_path/"pair_fit.json")]) == 0
    pair_fit = json.loads((tmp_path/"pair_fit.json").read_text())
    assert len(pair_fit["fits"]) == 2
    assert pair_fit["fits"][0]["inference"] == "plugin_gaussian_score"
    assert 0 <= pair_fit["fits"][0]["adaptive_bonferroni_p"] <= 1
    for filename in ("linear.linear-score.npz","linear_components.linear-score.npz","estimated.bootstrap-score.npz","estimated_pairs.bootstrap-score.npz"):
        assert main(["epistasis","fit",str(tmp_path/"prepared"/filename),"--out",str(tmp_path/(filename+".json"))])==0
    multiple=json.loads((tmp_path/"linear_components.linear-score.npz.json").read_text())["fits"][0]
    assert len(multiple["tests"])==2
    assert 0<=multiple["global_combined_kernel_test"]["kernel_p"]<=1
    estimated=json.loads((tmp_path/"estimated.bootstrap-score.npz.json").read_text())
    assert estimated["fits"][0]["minimum_p"]==.05
    assert estimated["fits"][0]["trait_unit"]=="simulation_units"
    pairs_fit=json.loads((tmp_path/"estimated_pairs.bootstrap-score.npz.json").read_text())
    assert len(pairs_fit["fits"][0]["pair_tests"]["signed_scores"])==2
    intervals=pairs_fit["fits"][0]["pair_tests"]["adaptive_monte_carlo_interval"]
    assert 0<=intervals[0]<=intervals[1]<=1
    # Cohort inputs are disposable test fixtures. Fresh-process inference sees
    # only the summaries, after all individual-level fixture files are removed.
    import subprocess,sys
    for name in ("input.bed","input.bim","input.fam","pheno.tsv","samples.tsv"):
        (tmp_path/name).unlink()
    launcher=Path(__file__).resolve().parents[1]/"scripts/epistasis/checkout_python.py"
    # Bound OpenMP may leave the caller on one CPU. New interpreters need
    # the full recorded allocation before importing numerical libraries.
    import os
    prefix = []
    if hasattr(os, "sched_getaffinity"):
        cpus = getattr(sys.modules.get("workflow"), "_PRE_NUMERICAL_CPU_AFFINITY",
                       tuple(sorted(os.sched_getaffinity(0))))
        prefix = ["taskset", "-c", ",".join(map(str, cpus))]
    for filename in ("linear.linear-score.npz","estimated.bootstrap-score.npz","estimated_pairs.bootstrap-score.npz"):
        subprocess.run(prefix + [sys.executable,str(launcher),"summit.epistasis.cli","fit",
            str(tmp_path/"prepared"/filename),"--out",str(tmp_path/("fresh_"+filename+".json"))],check=True,capture_output=True)


def test_bounded_pair_covariance_matches_explicit_gls():
    from summit.epistasis.pairs import prepare_bounded_pair_summary
    original, x, _, _, _, y = fixture()
    annotations = annotation_weights(original.source.variants.ids, {"region": {"v1": 2., "v3": 1.}})
    design = target_design(original.source, original.rows, original.scale, components=[],
        annotations=annotations, additive_annotations=["all", "region"], allow_additive_only=True,
        local_variants=["v4"])
    study = SelectedStudy(original.source, original.rows, original.scale, **design, backend="numpy")
    nuisance = dict(additive_coefficients={"all": .3, "region": .2}, residual_variance=.7,
                    known=True, identity="fixed_parameter_validation")
    pairs = [("v0", "v1"), ("v2", "v3")]
    summary = prepare_bounded_pair_summary(study, pairs, y, nuisance, trait_names=("a", "b"))
    v = .7*np.eye(len(x)) + .3*x@x.T/x.shape[1] + .2*(x*annotations["region"])@x.T/3
    expected = prepare_pair_scores(x, original.source.variants, pairs, y,
        fixed_effects=design["fixed_effects"], covariance_solve=lambda z: np.linalg.solve(v, z),
        covariance_identity="fixed_parameter_validation", covariance_known=True,
        trait_names=("a", "b"), sample_identity="test")
    np.testing.assert_allclose(summary.scores, expected.scores, atol=2e-12)
    np.testing.assert_allclose(summary.information, expected.information, atol=2e-12)
    with pytest.raises(ValueError, match="coefficient for every"):
        prepare_bounded_pair_summary(study, pairs, y, dict(nuisance, additive_coefficients={"all": .3}), trait_names=("a", "b"))


def test_snp_permutation_negative_estimates_and_covariance_diagnostics():
    study, x, e, w, fixed, y = fixture()
    k, p = selected_kernels(x, e, w, fixed)
    order = np.random.default_rng(71).permutation(x.shape[1])
    permuted, _ = selected_kernels(x[:, order], e, w[order], fixed)
    np.testing.assert_allclose(k, permuted, atol=1e-13)
    summary = dense_summary(k, y, component_names=(*study.names, "residual"), trait_names=("a", "b"),
                            residual_rank=p.residual_rank, metadata={"test": True})
    fit = fit_epistasis(summary)
    np.testing.assert_allclose(fit["coefficients"], np.linalg.solve(summary.matrix, summary.rhs[:, 0]))
    assert np.any(fit["coefficients"] < 0)
    # A pathological plug-in covariance must remain visible, never clipped.
    from dataclasses import replace
    invalid = fit_epistasis(replace(summary, cubic=-summary.cubic))
    assert not invalid["covariance_valid"]
    assert np.isnan(invalid["standard_errors"]).all()
    assert np.min(invalid["covariance_eigenvalues"]) < 0


def test_group_summary_and_fit_without_dense_kernel_storage():
    study, x, e, w, fixed, y = fixture()
    a = np.r_[np.ones(5), np.zeros(9)]
    b = remaining_genome_weights(a)
    group = GroupKernel(x, a, b, fixed_effects=fixed)
    base, p = selected_kernels(x, e[:, :1], w[:, :1], fixed)
    px = p.projector@x
    operators = [lambda z: px@(px.T@z)/x.shape[1], group.apply, lambda z: p.projector@z]
    names = ("additive", "set_by_remainder", "residual")
    summary = exact_operator_summary(operators, y, fixed_effects=fixed, component_names=names,
        trait_names=("a", "b"), metadata={"groups": "prespecified"}, sample_tile=11)
    kernels = np.stack([base[0], group_kernel(x, a, b, fixed), p.projector])
    expected = dense_summary(kernels, y, component_names=names, trait_names=("a", "b"),
                             residual_rank=p.residual_rank, metadata={"test": True})
    for key in ("matrix", "rhs", "cubic"):
        np.testing.assert_allclose(getattr(summary, key), getattr(expected, key), atol=2e-10)
    np.testing.assert_allclose(fit_epistasis(summary)["coefficients"], fit_epistasis(expected)["coefficients"], atol=2e-10)


def test_group_preparation_matches_explicit_weighted_pair_moments():
    from summit.epistasis.groups import prepare_group_summary
    original, x, _, _, _, y = fixture()
    annotations = annotation_weights(original.source.variants.ids, {
        "a": {"v0": 1., "v1": 2., "v2": .5},
        "b": {"v2": 1., "v3": .7, "v4": 2.}})
    design = target_design(original.source, original.rows, original.scale, components=[],
        annotations=annotations, additive_annotations=["all", "a", "b"], allow_additive_only=True)
    study = SelectedStudy(original.source, original.rows, original.scale, **design, backend="numpy")
    groups = [dict(name="cross", mode="cross", left="a", right="b"),
              dict(name="within", mode="within", left="a"),
              dict(name="remainder", mode="remainder", left="b")]
    summary = prepare_group_summary(study, groups, annotations, y, trait_names=("a", "b"))
    base, projection = selected_kernels(x, design["modifiers"], design["weights"], design["fixed_effects"])
    extra = [group_kernel(x, annotations["a"], annotations["b"], design["fixed_effects"]),
             group_kernel(x, annotations["a"], annotations["a"], design["fixed_effects"], within=True),
             group_kernel(x, annotations["b"], remaining_genome_weights(annotations["b"]), design["fixed_effects"])]
    k = np.concatenate([base[:-1], np.stack(extra), base[-1:]])
    expected = dense_summary(k, y, component_names=summary.component_names, trait_names=("a", "b"),
        residual_rank=projection.residual_rank, metadata={"test": True})
    for key in ("matrix", "rhs", "cubic", "traces"):
        np.testing.assert_allclose(getattr(summary, key), getattr(expected, key), atol=3e-10)
    with pytest.raises(ValueError, match="additive nuisance"):
        prepare_group_summary(study, [dict(name="bad", mode="within", left="unadjusted")],
            dict(annotations, unadjusted=np.ones(x.shape[1])), y, trait_names=("a", "b"))
