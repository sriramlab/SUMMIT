"""Person-level cross-block covariance: dense oracles, streaming, fit and files."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr, ndtri

from summit.pcgc.crossblock import (
    CrossBlockAccumulator, CrossBlockMoments, canonical_blocks, context_pair_sums,
    contiguous_blocks, crossblock_covariance, _center, METHOD,
)
from summit.pcgc.gxe import context_pairs, fit_gxe, prepare_gxe_moments
from summit.sumstats.binary import fit_binary_risk, prepare_binary_risk
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from prediction_helpers import prediction_threads


def design(n=72, m=31, q=3, seed=8113):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m))
    phi = np.column_stack((np.ones(n), rng.uniform(-1, 1, (n, q-1))))
    a = np.column_stack((np.ones(m), np.where(np.arange(m) < 20, np.linspace(.3, 1.7, m), 0.)))
    y = (np.arange(n) % 3 == 0).astype(float)
    risk = prepare_binary_risk(y, .08, population_risk=ndtr(-1.4+.3*phi[:, 1]))
    blocks = np.array([3, 3, 3, 1, 1, 0, 0, 0, 0, 2]*4)[:m]
    return x, phi, a, risk, blocks


def dense_kernels(x, a, features, select):
    """Participant-pair kernels K_c(i,l) restricted to selected target SNPs."""
    q = features.shape[1]
    mass = a.sum(0)
    out = []
    for k in range(a.shape[1]):
        G = (x[:, select]*a[select, k]) @ x[:, select].T/mass[k]
        for u, v in context_pairs(q):
            h = np.outer(features[:, u], features[:, v])
            if u != v:
                h = h+h.T
            K = G*h
            np.fill_diagonal(K, 0.)
            out.append(K)
    return out


def accumulate(x, a, features, response, cases, blocks, width, **extra):
    acc = CrossBlockAccumulator(a, features, response, cases, blocks, **extra)
    for start in range(0, x.shape[1], width):
        stop = min(x.shape[1], start+width)
        acc.add_block(start, stop, x[:, start:stop])
    return acc.finalize()


@pytest.mark.parametrize("width", [1, 4, 31])
def test_person_rows_match_dense_participant_kernels(width):
    x, phi, a, risk, blocks = design()
    psi, r, cases = phi*risk.sensitivity[:, None], risk.z, risk.z > 0
    cb = accumulate(x, a, psi, r, cases, blocks, width)
    ids = canonical_blocks(blocks)
    S = context_pair_sums(psi)
    p = len(context_pairs(phi.shape[1]))
    # Independent context sums: sum over l != i of h_p(i,l) h_r(i,l).
    h = dense_kernels(np.ones((len(x), 1)), np.ones((1, 1)), psi, np.ones(1, bool))
    np.testing.assert_allclose(S, np.einsum('pil,ril->ipr', h, h), rtol=1e-12, atol=1e-13)
    Sc = _center(S.reshape(len(x), -1), cases).reshape(S.shape)
    total = 0
    for F in range(ids.max()+1):
        K = dense_kernels(x, a, psi, ids == F)
        alpha = np.stack([r*(Kc @ r) for Kc in K], 1)
        total = total+alpha
        ac = _center(alpha, cases)
        np.testing.assert_allclose(cb.block_gram[F], ac.T @ ac, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(cb.block_context[F], np.einsum('ic,ipr->cpr', ac, Sc), rtol=1e-11, atol=1e-11)
    tc = _center(total, cases)
    np.testing.assert_allclose(cb.total_gram, tc.T @ tc, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(cb.context_total, S.sum(0), rtol=1e-12)
    np.testing.assert_allclose(cb.context_fourth, np.einsum('ipr,ist->iprst', Sc, Sc).sum(0), rtol=1e-11)
    assert cb.block_gram.shape == (4, 2*p, 2*p)


def test_between_block_covariance_is_the_off_diagonal_person_sum():
    x, phi, a, risk, blocks = design()
    psi, r, cases = phi*risk.sensitivity[:, None], risk.z, risk.z > 0
    cb = accumulate(x, a, psi, r, cases, blocks, 6)
    ids = canonical_blocks(blocks)
    B, C = ids.max()+1, cb.component_count
    p = len(context_pairs(phi.shape[1]))
    pc = np.arange(C) % p
    rng = np.random.default_rng(5)
    theta = rng.normal(size=C)*.1
    Kfull = dense_kernels(x, a, psi, np.ones(len(ids), bool))
    H = np.array([[np.sum(Kc*Kd) for Kd in Kfull] for Kc in Kfull])
    HF = np.array([[[np.sum(Kc*Kd) for Kd in Kfull] for Kc in dense_kernels(x, a, psi, ids == F)] for F in range(B)])
    S = context_pair_sums(psi)
    rows = []
    K_ = a.shape[1]
    for F in range(B):
        K = dense_kernels(x, a, psi, ids == F)
        alpha = np.stack([r*(Kc @ r) for Kc in K], 1)
        # Independent least-squares fit of H_F[(a,p),(b,r)] = g[a,b] S_tot[p,r].
        design_ = np.zeros((C*C, K_*K_))
        for c1 in range(C):
            for c2 in range(C):
                design_[c1*C+c2, (c1//p)*K_+c2//p] = S.sum(0)[c1 % p, c2 % p]
        g = np.linalg.lstsq(design_, HF[F].ravel(), rcond=None)[0].reshape(K_, K_)
        lam = g[np.arange(C)[:, None]//p, np.arange(C)[None, :]//p]*S[:, pc][:, :, pc]
        rows.append(_center(alpha-lam @ theta, cases))
    offdiagonal = 4*sum(rows[F].T @ rows[G] for F in range(B) for G in range(B) if F != G)
    between, nuisance, details = crossblock_covariance(cb, theta, H, HF)
    np.testing.assert_allclose(between, (offdiagonal+offdiagonal.T)/2, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(nuisance, 0.)
    assert np.isclose(details["person_total_trace"]-details["within_block_trace"], details["between_block_trace"])


def test_risk_fit_jacobian_matches_finite_differences():
    x, phi, a, _, blocks = design(n=64, m=25, q=2)
    y = (np.arange(64) % 3 == 0).astype(float)
    cov = phi[:, 1:]
    risk = fit_binary_risk(y, .08, cov)
    from summit.pcgc.sampling import nuisance_derivatives
    design_, influence, log_weight, derivative = nuisance_derivatives(risk, cov, "pcgc", np.ones(64))
    cb = accumulate(x, a, phi*risk.sensitivity[:, None], risk.z, risk.z > 0, blocks[:25], 7,
                    influence=influence, response_derivative=derivative, log_weight=log_weight)
    for r_ in range(design_.shape[1]):
        values = []
        for sign in (-1, 1):
            alt = prepare_binary_risk(y, risk.population_prevalence,
                                      population_risk=ndtr(ndtri(risk.population_risk)+sign*1e-5*design_[:, r_]))
            K = dense_kernels(x, a, phi*alt.sensitivity[:, None], np.ones(25, bool))
            values.append(np.array([alt.z @ Kc @ alt.z for Kc in K]))
        np.testing.assert_allclose((values[1]-values[0])/2e-5, cb.nuisance_jacobian[:, r_], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("method", ["pcgc", "pcgc-inverse", "pcgc-basis"])
def test_streamed_preparation_matches_direct_accumulator_and_fit(native, method):
    x, phi, a, risk, _ = design(n=90, m=47, q=2)
    sd = 1+.1*phi[:, 1]**2
    options = dict(probes=67, seed=21, block_size=11, native=native, threads=prediction_threads(), liability_sd=sd)
    if method == "pcgc-basis":
        options.update(basis=risk.sensitivity[:, None], coefficients=[1.])
    plain, _ = prepare_gxe_moments(Operator(x), a, risk, phi, method, **options)
    moments, diagnostics = prepare_gxe_moments(Operator(x), a, risk, phi, method, crossblock_blocks=6, **options)
    np.testing.assert_array_equal(moments.ldscores, plain.ldscores)
    np.testing.assert_array_equal(moments.rhs_rows, plain.rhs_rows)
    w = risk.sensitivity/sd
    features = phi if method == "pcgc-inverse" else phi*w[:, None]
    response = risk.z/w if method == "pcgc-inverse" else risk.z
    blocks = contiguous_blocks(47, 6)
    direct = accumulate(x, a, features, response, risk.z > 0, blocks, 47)
    for name in ("block_gram", "block_context", "total_gram", "total_context", "context_fourth", "context_total"):
        np.testing.assert_allclose(getattr(moments.crossblock_moments, name), getattr(direct, name), rtol=1e-10, atol=1e-10)
    assert diagnostics["person_crossblock"]["blocks"] == 6
    result = fit_gxe(moments, block_ids=blocks+10)
    assert result["uncertainty_method"] == METHOD and result["crossblock_status"] == "applied"
    jk = np.asarray(result["snp_block_covariance"])
    H, _ = moments.equations()
    from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
    directed, _, mass = moments.raw_equations()
    bd, _, _ = reduce_generalized_gxe_reference_for_inference(directional_ldscores=moments.ldscores,
        annotations=a, variant_block_ids=blocks, block_labels=tuple(map(str, range(6))), expected_directed_numerator=directed)
    between, _, _ = crossblock_covariance(direct, np.asarray(result["components"]), H,
                                          len(x)**2*bd/np.outer(mass, mass)[None])
    Hinv = np.linalg.inv(H)
    np.testing.assert_allclose(result["covariance"], jk+6/5*Hinv @ between @ Hinv.T, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(result["person_crossblock_covariance"], 6/5*Hinv @ between @ Hinv.T, rtol=1e-9, atol=1e-14)
    other = fit_gxe(moments, block_ids=np.arange(47) % 6)
    assert other["crossblock_status"] == "not_applied_inference_blocks_differ_from_preparation_blocks"
    np.testing.assert_allclose(other["covariance"], other["snp_block_covariance"])


def test_fitted_risks_require_covariates_and_propagate_risk_fit():
    x, phi, a, _, _ = design(n=90, m=47, q=2)
    y = (np.arange(90) % 3 == 0).astype(float)
    risk = fit_binary_risk(y, .08, phi[:, 1:])
    with pytest.raises(ValueError, match="original risk covariates"):
        prepare_gxe_moments(Operator(x), a, risk, phi, liability_sd=1., native=False, probes=31, crossblock_blocks=5)
    moments, _ = prepare_gxe_moments(Operator(x), a, risk, phi, liability_sd=1., native=False, probes=31,
                                      crossblock_blocks=5, risk_covariates=phi[:, 1:])
    assert moments.crossblock_moments.has_nuisance
    result = fit_gxe(moments, block_ids=contiguous_blocks(47, 5))
    assert result["crossblock_diagnostics"]["risk_fit_uncertainty_included"]
    assert np.asarray(result["risk_fit_covariance"]).shape == (6, 6)
    assert result["uncertainty_conditioning"].startswith("fixed_population_prevalence")


def test_sampling_result_is_kept_separate_from_calibrated_intervals():
    x, phi, a, risk, _ = design(n=90, m=47, q=2)
    moments, _ = prepare_gxe_moments(Operator(x), a[:, :1], risk, phi, liability_sd=1., native=False, probes=31,
                                      crossblock_blocks=5, sampling_partners=8, sampling_seed=3)
    result = fit_gxe(moments, block_ids=contiguous_blocks(47, 5), native=False)
    assert result["uncertainty_method"] == METHOD
    assert result["individual_sampling"]["uncertainty_method"] == "individual_sampling_fixed_genome_v1"
    plain = fit_gxe(moments, native=False)
    assert plain["uncertainty_method"] == "individual_sampling_fixed_genome_v1"


def test_moments_reject_inconsistent_axes():
    x, phi, a, risk, blocks = design()
    cb = accumulate(x, a, phi*risk.sensitivity[:, None], risk.z, risk.z > 0, blocks, 5)
    values = {name: getattr(cb, name) for name in ("block_ids", "block_gram", "block_context", "total_gram",
                                                  "total_context", "context_fourth", "context_total")}
    with pytest.raises(ValueError, match="first appearance"):
        CrossBlockMoments(**dict(values, block_ids=np.asarray(values["block_ids"])[::-1].copy()), n_samples=len(x))
    with pytest.raises(ValueError, match="axes"):
        CrossBlockMoments(**dict(values, total_gram=np.zeros((2, 2))), n_samples=len(x))
    with pytest.raises(ValueError, match="incomplete"):
        CrossBlockMoments(**values, n_samples=len(x), influence_gram=np.eye(1))
    acc = CrossBlockAccumulator(a, phi, risk.z, risk.z > 0, blocks)
    acc.add_block(0, 5, x[:, :5])
    with pytest.raises(ValueError, match="once"):
        acc.add_block(3, 8, x[:, 3:8])
    with pytest.raises(ValueError, match="incomplete"):
        acc.finalize()


def test_artifacts_round_trip_cross_block_moments(tmp_path):
    from test_pcgc_io import fixture
    from summit.pcgc.gxe_io import make_gxe_artifact, write_gxe_artifact, load_gxe_artifact, GxEArtifact
    from summit.pcgc.split_io import write_split_artifact, load_split_artifact
    _, x, axis, _, scale = fixture(83, 65)
    phi = np.column_stack((np.ones(len(x)), np.linspace(-1, 1, len(x))))
    y = (np.arange(len(x)) % 3 == 0).astype(float)
    risk = fit_binary_risk(y, .1, phi[:, 1:])
    moments, d = prepare_gxe_moments(Operator(x), np.ones((65, 1)), risk, phi, liability_sd=1., native=False,
                                     probes=61, crossblock_blocks=7, risk_covariates=phi[:, 1:])
    artifact = make_gxe_artifact(moments, variant_axis=axis, annotation_names=["all"], context_names=["intercept", "e"],
        sample_identity="a"*64, genotype_scale_identity=scale.identity, risk=risk, contexts=phi, liability_sd=1., diagnostics=d)
    assert {"person_crossblock_covariance", "person_crossblock_risk_fit"} <= set(artifact.manifest["features"])
    combined = load_gxe_artifact(write_gxe_artifact(artifact, tmp_path/"c.binary.npz"))
    split = load_split_artifact(*write_split_artifact(artifact, tmp_path/"s.binary.sumstats.npz", tmp_path/"s.binary.ldscores.npz"))
    blocks = contiguous_blocks(65, 7)
    expected = fit_gxe(moments, block_ids=blocks)
    for loaded in (combined, split):
        assert loaded.moments.crossblock_moments.block_ids.dtype.kind == "i"
        np.testing.assert_array_equal(fit_gxe(loaded.moments, block_ids=blocks)["covariance"], expected["covariance"])
    from dataclasses import replace
    tampered = replace(moments.crossblock_moments, total_gram=moments.crossblock_moments.total_gram*1.01)
    with pytest.raises(ValueError, match="checksum"):
        GxEArtifact(replace(moments, crossblock_moments=tampered), artifact.manifest)


def test_cli_prepares_cross_block_moments_and_fits_matching_blocks(tmp_path):
    from summit.entrypoint import main
    from summit.pcgc.split_io import load_split_artifact, output_paths
    from test_pcgc_gxe_io import write_bed
    from test_pcgc_io import fixture
    raw, x, axis, risk, scale = fixture(144, 128)
    e = np.random.default_rng(2719).uniform(-1, 1, len(raw))
    write_bed(tmp_path/"study.bed", raw, axis)
    pd.DataFrame(dict(FID=["study"]*len(raw), IID=[f"i{i}" for i in range(len(raw))], Y=(risk.z > 0).astype(int),
                      E=e, AGE=np.cos(np.arange(len(raw))))).to_csv(tmp_path/"people.tsv", sep="\t", index=False)
    pd.DataFrame(dict(SNP=axis.ids, A1=axis.counted, A2=axis.other, MEAN=scale.mean, INV_SD=scale.inverse_scale)).to_csv(
        tmp_path/"scale.tsv", sep="\t", index=False)
    base = ["--binary-method", "pcgc", "--make-binary-sumstats", str(tmp_path/"people.tsv"), "--geno", str(tmp_path/"study.bed"),
            "--binary-scale", str(tmp_path/"scale.tsv"), "--binary-prevalence", ".1", "--binary-context-columns", "E",
            "--binary-covariates", "AGE", "--binary-unit-liability", "--nvecs", "64", "--seed", "8", "--block-size", "37",
            "--memory-gib", "1", "--num-threads", str(prediction_threads())]
    assert main([*base, "--out", str(tmp_path/"plain")]) == 0
    assert load_split_artifact(*output_paths(tmp_path/"plain")).moments.crossblock_moments is None
    assert main(["--binary-method", "pcgc", "--h2", str(output_paths(tmp_path/"plain")[0]), "--ldscores",
                 str(output_paths(tmp_path/"plain")[1]), "--njack", "8", "--out", str(tmp_path/"plain_fit")]) == 0
    plain = json.loads((tmp_path/"plain_fit.binary.json").read_text())
    assert plain["uncertainty_calibration_status"].startswith("not_calibrated") and "crossblock_status" not in plain
    assert main([*base, "--binary-crossblock-blocks", "128", "--out", str(tmp_path/"default")]) == 0
    artifact = load_split_artifact(*output_paths(tmp_path/"default"))
    assert artifact.moments.crossblock_moments.block_count == 128 and artifact.moments.crossblock_moments.has_nuisance
    assert main(["--binary-method", "pcgc", "--h2", str(output_paths(tmp_path/"default")[0]), "--ldscores",
                 str(output_paths(tmp_path/"default")[1]), "--out", str(tmp_path/"fit")]) == 0
    result = json.loads((tmp_path/"fit.binary.json").read_text())
    assert result["uncertainty_method"] == METHOD and result["jackknife_blocks"] == 128
    assert main(["--binary-method", "pcgc", "--h2", str(output_paths(tmp_path/"default")[0]), "--ldscores",
                 str(output_paths(tmp_path/"default")[1]), "--njack", "8", "--out", str(tmp_path/"fit8")]) == 0
    other = json.loads((tmp_path/"fit8.binary.json").read_text())
    assert other["crossblock_status"].startswith("not_applied") and other["uncertainty_method"] != METHOD
    assert main([*base, "--binary-crossblock-blocks", "8", "--out", str(tmp_path/"eight")]) == 0
    assert main(["--binary-method", "pcgc", "--h2", str(output_paths(tmp_path/"eight")[0]), "--ldscores",
                 str(output_paths(tmp_path/"eight")[1]), "--njack", "8", "--out", str(tmp_path/"fit_eight")]) == 0
    # With 144 people the partition matches; the tiny sample may still make
    # the between-block estimate too noisy, which is reported, not hidden.
    eight = json.loads((tmp_path/"fit_eight.binary.json").read_text())
    assert eight["crossblock_status"] in ("applied", "not_applied_nonpositive_calibrated_variance")
    assert "person_crossblock_covariance" in eight and eight["jackknife_blocks"] == 8
    assert main([*base, "--binary-crossblock-blocks", "0", "--out", str(tmp_path/"off")]) == 0
    assert load_split_artifact(*output_paths(tmp_path/"off")).moments.crossblock_moments is None
    with pytest.raises(SystemExit):
        main([*base, "--binary-crossblock-blocks", "1", "--out", str(tmp_path/"bad")])
    assert not output_paths(tmp_path/"bad")[0].exists()


def test_small_simulation_restores_null_interaction_covariance_variance():
    """Fast end-to-end check: rare disease, large N/M, fixed genotypes.

    The SNP-block jackknife misses about half of the variance of the null
    additive-by-context covariance; adding the between-block person term
    restores it. 200 replicates give a variance-ratio MCSE of about 0.1.
    """
    from threadpoolctl import threadpool_limits
    from scripts.pcgc import calibrate_gxe_intervals as sim
    spec = dict(design="fixed", n=4096, m=64, q=2, K=.05, h=.5, ld="none", annotations="one", risk="supplied",
                risk_slope=.3, case_fraction=.5, blocks=16, seed=11)
    setting = sim.Setting(spec)
    with threadpool_limits(1):
        rows = [sim.run_replicate(setting, i, {}) for i in range(200)]
    assert not any("failure" in r for r in rows)
    est = np.array([r["estimate"] for r in rows])
    jk = np.array([r["jk_var"] for r in rows])
    calibrated = jk+np.array([r["between_var"] for r in rows])
    empirical = est.var(0, ddof=1)
    null = 2   # Omega_01, the additive-by-context covariance, is zero
    assert setting.truth[null] == 0
    assert jk[:, null].mean()/empirical[null] < .7
    assert .8 < calibrated[:, null].mean()/empirical[null] < 1.25
    rejection = np.mean(np.abs(est[:, null]) > 1.96*np.sqrt(calibrated[:, null]))
    assert rejection < .1 < np.mean(np.abs(est[:, null]) > 1.96*np.sqrt(jk[:, null]))


def test_external_ld_preparation_records_study_rows():
    from summit.pcgc.gxe import prepare_gxe_external, EXTERNAL_CONTRACT, plan_gxe_reference
    x, phi, a, risk, _ = design(n=90, m=47, q=2)
    reference = np.random.default_rng(4).normal(size=(120, 47))
    common = dict(liability_sd=1., factorization_contract=EXTERNAL_CONTRACT, probes=31, seed=3, block_size=9, native=False)
    moments, diagnostics = prepare_gxe_external(Operator(x), Operator(reference), a, risk, phi, crossblock_blocks=5, **common)
    direct = accumulate(x, a, phi*risk.sensitivity[:, None], risk.z, risk.z > 0, contiguous_blocks(47, 5), 47)
    np.testing.assert_allclose(moments.crossblock_moments.total_gram, direct.total_gram, rtol=1e-10, atol=1e-10)
    assert diagnostics["person_crossblock"]["blocks"] == 5
    assert fit_gxe(moments, block_ids=contiguous_blocks(47, 5))["crossblock_status"] in (
        "applied", "not_applied_nonpositive_calibrated_variance")
    small = dict(num_samples=1000, num_variants=500, num_contexts=2, num_annotations=1, memory_bytes=2**28)
    assert (plan_gxe_reference(**small, crossblock_blocks=200)["reserved_result_and_input_bytes"]
            > plan_gxe_reference(**small)["reserved_result_and_input_bytes"])
