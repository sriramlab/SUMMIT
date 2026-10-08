import numpy as np
import pytest
from scipy.stats import chi2

from summit.epistasis.quadratic import quadratic_sf, quadratic_quantile, moment_spectrum
from summit.epistasis.null import GaussianNullReference
from summit.epistasis.score import (
    prepare_linear_scores,
    linear_score_tests,
    write_linear_scores,
    load_linear_scores,
    refit_bootstrap,
)
from summit.epistasis.features import (
    prepare_feature_reference,
    _probe,
    write_feature_reference,
    load_feature_reference,
)
from summit.epistasis.models import annotation_weights, target_design
from summit.epistasis.prepare import SelectedStudy, fit_scale
from summit.prediction.genotype import ArrayGenotypeSource
from summit.prediction.spec import VariantAxis


def test_quadratic_positive_negative_and_indefinite_distributions():
    for sign in (-1, 1):
        result = quadratic_sf(sign * 18.0, np.full(8, sign * 2.0))
        expected = chi2.sf(9, 8) if sign == 1 else chi2.cdf(9, 8)
        assert result["p"] == expected
    # Difference of identically distributed quadratic forms is symmetric.
    values = np.r_[np.linspace(0.5, 2, 12), -np.linspace(0.5, 2, 12)]
    assert abs(quadratic_sf(0.0, values)["p"] - 0.5) < 1e-10
    rng = np.random.default_rng(819)
    lam = np.r_[np.linspace(0.1, 1, 18), -np.linspace(0.1, 0.6, 6)]
    draws = (rng.normal(size=(50000, len(lam))) ** 2) @ lam
    for p in (0.05, 0.5, 0.95):
        q = quadratic_quantile(p, lam)
        assert abs(np.mean(draws <= q) - p) < 0.009
    with pytest.raises(ValueError):
        quadratic_sf(1, [np.nan])


def test_moment_estimate_is_indefinite_quadratic_form():
    rng = np.random.default_rng(120)
    f = rng.normal(size=(50, 9))
    k = np.stack([f @ f.T / 9, np.eye(50)])
    t = np.einsum("aij,bji->ab", k, k)
    v = 0.2 * k[0] + 0.8 * k[1]
    result = moment_spectrum(k, t, v, 0)
    assert result["eigenvalues"].min() < 0
    assert result["mean"] == pytest.approx(0.2)
    y = rng.normal(size=50)
    q = np.einsum("i,aij,j->a", y, k, y)
    assert y @ result["contrast"] @ y == pytest.approx(np.linalg.solve(t, q)[0])


def test_linear_unknown_scale_matches_exact_regression_and_roundtrip(tmp_path):
    from scipy.stats import f as f_dist

    rng = np.random.default_rng(714)
    n = 96
    x = rng.binomial(2, 0.3, (n, 3)).astype(float)
    fixed = np.column_stack([np.ones(n), x])
    features = np.column_stack([x[:, 0] * x[:, 1], x[:, 0] * x[:, 2]])
    y = features @ np.array([0.2, -0.4]) + rng.normal(size=n)
    summary = prepare_linear_scores(
        features,
        y,
        fixed,
        feature_names=("a", "b"),
        trait_names=("y",),
        metadata={"test": True},
    )
    result = linear_score_tests(
        load_linear_scores(write_linear_scores(summary, tmp_path / "linear.npz")),
        burden=[1, -1],
    )
    full = np.column_stack([fixed, features])
    b = np.linalg.lstsq(full, y, rcond=None)[0]
    residual = y - full @ b
    null_residual = y - fixed @ np.linalg.lstsq(fixed, y, rcond=None)[0]
    f = ((null_residual @ null_residual - residual @ residual) / 2) / (
        residual @ residual / (n - full.shape[1])
    )
    assert result["joint_p"] == pytest.approx(
        f_dist.sf(f, 2, n - full.shape[1]), rel=1e-10
    )
    np.testing.assert_allclose(result["joint_beta"], b[-2:], atol=1e-12)
    one = prepare_linear_scores(
        features[:, :1],
        y,
        fixed,
        feature_names=("a",),
        trait_names=("y",),
        metadata={"test": True},
    )
    test = linear_score_tests(one)
    assert test["kernel_p"] == pytest.approx(test["joint_p"], abs=2e-10)
    assert result["adaptive_bonferroni_p"] >= min(
        result["kernel_p"], result["burden_p"], result["sparse_bonferroni_p"]
    )


def test_reml_objective_efficient_score_and_refitted_bootstrap():
    from scipy.linalg import null_space
    from scipy.optimize import minimize

    rng = np.random.default_rng(347)
    n = 64
    x = rng.normal(size=(n, 20))
    fixed = np.column_stack([np.ones(n), x[:, 0]])
    g = x @ x.T / 20
    reference = GaussianNullReference(g, fixed, identity="test")
    y = np.sqrt(0.3 / 20) * x @ rng.normal(size=20) + np.sqrt(0.7) * rng.normal(size=n)
    z = reference.transform(y)
    fit = reference.fit(z)

    def objective(logtheta):
        theta = np.exp(logtheta)
        variance = theta[0] * reference.eigenvalues + theta[1]
        return np.log(variance).sum() + np.sum(z * z / variance)

    independent = minimize(objective, np.log([0.3, 0.7]), method="BFGS")
    assert fit["objective"] + reference.rank <= independent.fun + 1e-7
    f = x[:, 0, None] * x[:, 1:]
    k = reference.kernel(f @ f.T / 19)
    score = reference.efficient_score(z, k, fit, retain_contrast=True)
    contrast = score["contrast"]
    w = z / np.sqrt(fit["variance"])
    assert score["statistic"] == pytest.approx(
        w @ contrast @ w - np.trace(contrast), abs=1e-11
    )
    derivatives = np.column_stack(
        [1 / fit["variance"], reference.eigenvalues / fit["variance"]]
    )
    np.testing.assert_allclose(np.diag(contrast) @ derivatives, 0, atol=1e-10)
    boot = refit_bootstrap(reference, [k], y, draws=19, seed=80)
    assert boot["p"][0] >= 1 / 20
    assert boot["bootstrap_statistics"].shape == (19, 1)
    assert "REML refit every draw" in boot["stochastic_contract"]


def test_feature_and_phenotype_units_preserve_score_inference():
    from dataclasses import replace
    from summit.epistasis.pairs import PairScores, pair_tests

    rng = np.random.default_rng(196)
    f = rng.normal(size=(90, 3))
    y = f @ np.array([0.1, -0.2, 0.1]) + rng.normal(size=90)
    fixed = np.ones((90, 1))
    summary = prepare_linear_scores(
        f, y, fixed, feature_names=("a", "b", "c"), trait_names=("y",), metadata={}
    )
    expected = linear_score_tests(summary, burden=[1, 1, 1])
    for feature_unit, trait_unit in ((1e-9, 1.0), (1e9, 1.0), (1.0, 1e-9), (1.0, 1e9)):
        converted = prepare_linear_scores(
            f * feature_unit,
            y * trait_unit,
            fixed,
            feature_names=summary.feature_names,
            trait_names=summary.trait_names,
            metadata={},
        )
        result = linear_score_tests(converted, burden=[1, 1, 1])
        for key in ("kernel_p", "joint_p", "burden_p", "sparse_bonferroni_p"):
            assert result[key] == pytest.approx(expected[key], abs=1e-9)
        np.testing.assert_allclose(
            result["joint_beta"] * feature_unit / trait_unit,
            expected["joint_beta"],
            atol=1e-12,
        )
    pair = PairScores(
        summary.scores,
        summary.information,
        (("a", "b"), ("a", "c"), ("b", "c")),
        ("y",),
        dict(inference="exact_gaussian_known_covariance"),
    )
    reference = pair_tests(pair)
    # y*=u*y and known V*=u²V changes s by 1/u and H by 1/u².
    for unit in (1e-9, 1e9):
        result = pair_tests(
            replace(
                pair,
                scores=pair.scores / unit,
                information=pair.information / unit**2,
            )
        )
        np.testing.assert_allclose(
            result["marginal_p_two_sided"],
            reference["marginal_p_two_sided"],
            atol=1e-12,
        )
        assert result["joint_p"] == pytest.approx(reference["joint_p"], abs=1e-12)


def test_multiple_nuisance_reml_matches_one_kernel_and_whitened_score():
    from summit.epistasis.null import GeneralGaussianNullReference

    rng = np.random.default_rng(91)
    n = 36
    x = rng.normal(size=(n, 8))
    g = x @ x.T / 8
    fixed = np.ones((n, 1))
    y = x @ rng.normal(size=8) / 4 + rng.normal(size=n)
    one = GaussianNullReference(g, fixed, identity="one")
    many = GeneralGaussianNullReference(
        [g, np.eye(n)], fixed, names=("additive", "residual"), identity="many"
    )
    fit = one.fit(one.transform(y))
    fit2 = many.fit(many.transform(y))
    np.testing.assert_allclose(
        fit2["coefficients"],
        [fit["genetic_variance"], fit["residual_variance"]],
        atol=1e-5,
    )
    k = x[:, 0, None] * x[:, 1:]
    first = one.efficient_score(one.transform(y), one.kernel(k @ k.T), fit)
    second = many.efficient_score(many.transform(y), many.kernel(k @ k.T), fit2)
    assert first["statistic"] == pytest.approx(second["statistic"], abs=2e-5)
    boot = refit_bootstrap(many, [many.kernel(k @ k.T)], y, draws=19, seed=34)
    assert boot["p"][0] >= 0.05


def test_implicit_nonself_pair_sketch_draw_and_exact_group_features(tmp_path):
    from summit.epistasis.oracle import explicit_pair_features

    rng = np.random.default_rng(417)
    n, m = 56, 9
    raw = rng.binomial(2, 0.4, (n, m))
    axis = VariantAxis(
        tuple(f"v{i}" for i in range(m)),
        ("1",) * m,
        tuple(range(1, m + 1)),
        ("A",) * m,
        ("G",) * m,
    )
    source = ArrayGenotypeSource(
        raw, [(str(i), str(i)) for i in range(n)], axis, hard_calls=True
    )
    scale = fit_scale(source, np.arange(n))
    x = (raw - scale.mean) * scale.inverse_scale
    annotations = annotation_weights(
        axis.ids,
        {
            "a": {"v0": 1.0, "v1": 2.0, "v2": 1.0},
            "b": {"v1": 0.5, "v3": 1.0, "v4": 2.0},
        },
    )
    design = target_design(
        source,
        np.arange(n),
        scale,
        components=[],
        annotations=annotations,
        additive_annotations=["all", "a", "b"],
        allow_additive_only=True,
    )
    study = SelectedStudy(
        source, np.arange(n), scale, **design, backend="numpy", block_size=3
    )
    for mode in ("cross", "within", "remainder"):
        group = dict(name=mode, mode=mode, left="a")
        if mode == "cross":
            group["right"] = "b"
        job = dict(id="test", groups=[group], additive_annotations=["all", "a", "b"])
        exact = prepare_feature_reference(
            study,
            job,
            annotations,
            main_effects="tested_variants",
            dominance="tested_variants",
        )
        a = annotations["a"]
        b = (
            annotations["b"]
            if mode == "cross"
            else a
            if mode == "within"
            else (a == 0).astype(float)
        )
        f, _, _ = explicit_pair_features(x, a, b, within=mode == "within")
        np.testing.assert_allclose(
            exact.features @ exact.features.T, f @ f.T, atol=2e-12
        )
        sketched = prepare_feature_reference(
            study, job, annotations, sketch_dimensions=8, seed=16, bank=2
        )
        z = _probe(study, np.arange(m), 8, 16, 2, "left")
        zz = _probe(study, np.arange(m), 8, 16, 2, "right")
        from summit.context.spec import array_sha256

        if array_sha256(a[np.argsort(axis.ids)]) > array_sha256(
            b[np.argsort(axis.ids)]
        ):
            a, b = b, a
        expected = np.zeros_like(sketched.features)
        mass = a.sum() * b.sum() - a @ b
        for i in range(m):
            for j in range(m):
                if i != j:
                    expected += (
                        x[:, i, None]
                        * x[:, j, None]
                        * np.sqrt(a[i] * b[j])
                        * (z[i] * zz[j])[None, :]
                        / np.sqrt(8 * mass)
                    )
        np.testing.assert_allclose(sketched.features, expected, atol=2e-12)
    restored = load_feature_reference(
        write_feature_reference(exact, tmp_path / "reference.npz"),
        compatibility_id=exact.metadata["compatibility_id"],
    )
    np.testing.assert_array_equal(restored.features, exact.features)
    with pytest.raises(ValueError, match="compatibility"):
        load_feature_reference(tmp_path / "reference.npz", compatibility_id="different")


def test_cohort_allele_units_and_overlap_score_experiment():
    from dataclasses import replace
    from summit.epistasis.pairs import prepare_pair_scores
    from summit.epistasis.cohorts import (
        harmonize_pair_scores,
        combine_score_experiments,
    )

    rng = np.random.default_rng(829)
    raw = rng.binomial(2, 0.3, (100, 3)).astype(float)
    axis = VariantAxis(
        ("a", "b", "c"),
        ("1",) * 3,
        (1, 2, 3),
        ("A",) * 3,
        ("G",) * 3,
        genome_build="GRCh37",
    )
    canonical = {
        v: dict(chromosome="1", position=i + 1, counted="A", other="G")
        for i, v in enumerate(axis.ids)
    }
    y = raw[:, 0] * raw[:, 1] * 0.2 + rng.normal(size=100)
    pairs = [("a", "b"), ("a", "c")]

    def prepare(data, axis, y, multiplier, identity):
        mean = data.mean(axis=0)
        scale = 1 / np.std(data, axis=0)
        s = prepare_pair_scores(
            (data - mean) * scale,
            axis,
            pairs[::-1],
            y[:, None],
            fixed_effects=np.ones((len(y), 1)),
            covariance_solve=lambda z: z * multiplier**2,
            covariance_known=True,
            covariance_identity="known",
            trait_names=("y",),
            sample_identity=identity,
        )
        s = replace(
            s,
            metadata=dict(
                s.metadata,
                genome_build="GRCh37",
                trait_unit="input",
                inverse_scales=dict(zip(axis.ids, scale)),
            ),
        )
        return harmonize_pair_scores(
            s,
            canonical,
            genome_build="GRCh37",
            trait_unit="output",
            phenotype_multiplier=multiplier,
        )

    first, _ = prepare(raw[:50], axis, y[:50], 1.0, "first")
    flipped = raw[50:].copy()
    flipped[:, 0] = 2 - flipped[:, 0]
    reversed_axis = VariantAxis(
        axis.ids,
        axis.chromosome,
        axis.position,
        ("G", "A", "A"),
        ("A", "G", "G"),
        genome_build="GRCh37",
    )
    second, _ = prepare(flipped, reversed_axis, y[50:] / 100, 100.0, "second")
    expected, _ = prepare(raw[50:], axis, y[50:], 1.0, "expected")
    np.testing.assert_allclose(second.scores, expected.scores, atol=1e-12)
    np.testing.assert_allclose(second.information, expected.information, atol=1e-12)
    combined = combine_score_experiments([first, second], independent=True)
    np.testing.assert_allclose(
        combined.scores, first.scores + second.scores, atol=1e-12
    )
    np.testing.assert_allclose(
        combined.information, first.information + second.information, atol=1e-12
    )
    from scipy.linalg import block_diag

    pooled_fixed = block_diag(
        np.column_stack([np.ones(50), raw[:50]]),
        np.column_stack([np.ones(50), raw[50:]]),
    )
    pooled = prepare_pair_scores(
        raw,
        axis,
        pairs,
        y,
        fixed_effects=pooled_fixed,
        covariance_solve=lambda z: z,
        covariance_identity="known",
        covariance_known=True,
        trait_names=("y",),
        sample_identity="pooled",
    )
    np.testing.assert_allclose(combined.scores, pooled.scores, atol=1e-12)
    np.testing.assert_allclose(combined.information, pooled.information, atol=1e-12)
    # Correlated Gaussian score experiments: compare with direct GLS on stacked scores.
    from scipy.linalg import block_diag

    c = block_diag(first.information, second.information)
    cross = (
        0.2
        * np.linalg.cholesky(first.information)
        @ np.linalg.cholesky(second.information).T
    )
    c[:2, 2:] = cross
    c[2:, :2] = cross.T
    overlap = combine_score_experiments([first, second], cross_score_covariance=c)
    a = np.vstack([first.information, second.information])
    np.testing.assert_allclose(overlap.information, a.T @ np.linalg.solve(c, a))
    with pytest.raises(ValueError):
        combine_score_experiments([first, second])


def test_pairwise_ld_does_not_identify_interaction_score_covariance():
    # Uniform independent four signs versus uniform even parity. All means and
    # pairwise LD agree; Cov(x1*x2,x3*x4) differs by one. Diploid construction
    # sums two independent haplotypes, giving valid 0/1/2 dosages.
    import itertools

    h = np.array(list(itertools.product((-1.0, 1.0), repeat=4)))
    parity = h[np.prod(h, axis=1) == 1]

    def moments(h):
        x = (h[:, None, :] + h[None, :, :]).reshape(-1, 4) / np.sqrt(2)
        return x.T @ x / len(x), np.mean(np.prod(x, axis=1))

    ld, product = moments(h)
    ld2, product2 = moments(parity)
    np.testing.assert_allclose(ld, ld2, atol=1e-14)
    assert product == pytest.approx(0)
    assert product2 == pytest.approx(0.5)


def test_annihilated_pair_does_not_invalidate_remaining_kernel():
    rng = np.random.default_rng(971)
    f = rng.normal(size=(80, 3))
    f[:, 1] = 1
    y = rng.normal(size=80)
    fixed = np.ones((80, 1))
    full = prepare_linear_scores(
        f, y, fixed, feature_names=("a", "zero", "b"), trait_names=("y",), metadata={}
    )
    reduced = prepare_linear_scores(
        f[:, [0, 2]],
        y,
        fixed,
        feature_names=("a", "b"),
        trait_names=("y",),
        metadata={},
    )
    a = linear_score_tests(full, burden=[0, 1, 0])
    b = linear_score_tests(reduced)
    assert a["kernel_p"] == pytest.approx(b["kernel_p"], abs=1e-10)
    assert a["unidentifiable_feature_names"] == ["zero"]
    assert not a["burden_identifiable"]
    assert np.isnan(a["marginal_p"][1])


def test_group_sketch_recoding_permutation_and_independent_bank_moments():
    from summit.epistasis.sketch import independent_bank_moments
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis

    rng = np.random.default_rng(816)
    raw = rng.binomial(2, 0.4, (48, 8))

    def reference(raw, order, flip=False, bank=0, exchange=False, tile=3):
        axis = VariantAxis(
            tuple(f"v{i}" for i in order),
            ("1",) * 8,
            tuple(int(i + 1) for i in order),
            tuple("G" if flip and i == 0 else "A" for i in order),
            tuple("A" if flip and i == 0 else "G" for i in order),
        )
        source = ArrayGenotypeSource(
            raw, [(str(i), str(i)) for i in range(48)], axis, hard_calls=True
        )
        scale = fit_scale(source, np.arange(48))
        ann = annotation_weights(
            axis.ids,
            {"a": {"v0": 1.0, "v1": 2.0, "v2": 1.0}, "b": {"v1": 1.0, "v3": 1.0}},
        )
        design = target_design(
            source,
            np.arange(48),
            scale,
            components=[],
            annotations=ann,
            additive_annotations=["all", "a", "b"],
            allow_additive_only=True,
        )
        study = SelectedStudy(
            source, np.arange(48), scale, **design, backend="numpy", block_size=tile
        )
        job = dict(
            id="cross",
            additive_annotations=["all", "a", "b"],
            groups=[
                dict(
                    name="cross",
                    mode="cross",
                    left="b" if exchange else "a",
                    right="a" if exchange else "b",
                )
            ],
        )
        return prepare_feature_reference(
            study, job, ann, sketch_dimensions=8, bank=bank
        )

    first = reference(raw, np.arange(8))
    exchanged = reference(raw, np.arange(8), exchange=True, tile=5)
    np.testing.assert_allclose(first.features, exchanged.features, atol=1e-13)
    recoded = raw.copy()
    recoded[:, 0] = 2 - recoded[:, 0]
    order = rng.permutation(8)
    recoded = reference(recoded[:, order], order, flip=True, tile=2)
    np.testing.assert_allclose(first.features, recoded.features, atol=1e-13)
    refs = [reference(raw, np.arange(8), bank=i) for i in range(3)]
    y = rng.normal(size=(48, 2))
    result = independent_bank_moments(refs, y)
    u = thin_rank_revealing_fixed_effect_basis(first.fixed_effects)
    ks = []
    for r in refs:
        f = r.features - u @ (u.T @ r.features)
        ks.append(f @ f.T)
    import itertools

    t = sum(np.trace(ks[a] @ ks[b]) for a, b in itertools.permutations(range(3), 2)) / 6
    cubic = (
        sum(
            np.einsum("it,ij,jt->t", y, ks[a] @ ks[b] @ ks[c], y)
            for a, b, c in itertools.permutations(range(3))
        )
        / 6
    )
    assert result["matrix"][0, 0] == pytest.approx(t)
    np.testing.assert_allclose(result["cubic"][:, 0, 0, 0], cubic, atol=1e-10)


def test_conditional_followup_matches_joint_regression_with_other_interactions():
    from summit.epistasis.score import (
        conditional_feature_summary,
        prespecified_followup,
    )

    rng = np.random.default_rng(765)
    f = rng.normal(size=(80, 3))
    f[:, 1] += 0.7 * f[:, 0]
    fixed = np.ones((80, 1))
    y = 2 * f[:, 1] + rng.normal(size=80)
    summary = prepare_linear_scores(
        f, y, fixed, feature_names=("a", "b", "c"), trait_names=("y",), metadata={}
    )
    conditional = linear_score_tests(conditional_feature_summary(summary, [0]))
    explicit = linear_score_tests(
        prepare_linear_scores(
            f[:, :1],
            y,
            np.column_stack([fixed, f[:, 1:]]),
            feature_names=("a",),
            trait_names=("y",),
            metadata={},
        )
    )
    assert conditional["joint_p"] == pytest.approx(explicit["joint_p"], abs=1e-10)
    follow = prespecified_followup(summary, {"first": [0, 1], "second": [1, 2]})
    assert follow["pair_universe"] == 3
    for group in follow["groups"]:
        assert np.all(group["gated_pair_p"] >= group["pair_adjusted_p"])


def test_saved_cohort_and_followup_manifest_commands(tmp_path):
    import json
    from dataclasses import replace
    from epistasis_helpers import cli as main
    from summit.epistasis.pairs import (
        prepare_pair_scores,
        write_pair_scores,
        load_pair_scores,
    )

    rng = np.random.default_rng(313)
    axis = VariantAxis(
        ("a", "b", "c"),
        ("1",) * 3,
        (1, 2, 3),
        ("A",) * 3,
        ("G",) * 3,
        genome_build="GRCh37",
    )
    pairs = [("a", "b"), ("a", "c")]
    summaries = []
    for cohort in range(2):
        x = rng.binomial(2, 0.35, (64, 3)).astype(float)
        y = rng.normal(size=64)
        s = prepare_pair_scores(
            x,
            axis,
            pairs,
            y,
            fixed_effects=np.ones((64, 1)),
            covariance_solve=lambda v: v,
            covariance_identity="known",
            covariance_known=True,
            trait_names=("y",),
            sample_identity=str(cohort),
        )
        s = replace(
            s,
            metadata=dict(
                s.metadata,
                genome_build="GRCh37",
                trait_unit="test",
                inverse_scales=dict.fromkeys(axis.ids, 1.0),
            ),
        )
        write_pair_scores(s, tmp_path / f"cohort{cohort}.npz")
        summaries.append(s)
    manifest = dict(
        kind="summit.epistasis.combine",
        schema_version=1,
        genome_build="GRCh37",
        trait_unit="test",
        independent=True,
        canonical_variants={
            v: dict(chromosome="1", position=i + 1, counted="A", other="G")
            for i, v in enumerate(axis.ids)
        },
        cohorts=[
            dict(summary=f"cohort{i}.npz", phenotype_multiplier=1.0) for i in range(2)
        ],
    )
    path = tmp_path / "cohorts.json"
    path.write_text(json.dumps(manifest))
    assert main(["combine", str(path), "--out", str(tmp_path / "combined.npz")]) == 0
    combined = load_pair_scores(tmp_path / "combined.npz")
    np.testing.assert_allclose(combined.scores, sum(s.scores for s in summaries))
    assert (
        main(
            ["fit", str(tmp_path / "combined.npz"), "--out", str(tmp_path / "fit.json")]
        )
        == 0
    )
    result = json.loads((tmp_path / "fit.json").read_text())["fits"][0]
    assert result["estimand"] == "common_signed_pair_effect"
    assert "heterogeneity_p" in result["cohort_tests"]
    f = np.column_stack([x[:, 0] * x[:, 1], x[:, 0] * x[:, 2]])
    linear = prepare_linear_scores(
        f,
        y,
        np.column_stack([np.ones(64), x]),
        feature_names=("ab", "ac"),
        trait_names=("y",),
        metadata={},
    )
    write_linear_scores(linear, tmp_path / "linear.npz")
    path = tmp_path / "followup.json"
    path.write_text(
        json.dumps(
            dict(
                kind="summit.epistasis.followup",
                schema_version=1,
                summary="linear.npz",
                groups=dict(first=[0], second=[1]),
            )
        )
    )
    assert (
        main(["followup", str(path), "--out", str(tmp_path / "followup_result.json")])
        == 0
    )
    assert (
        json.loads((tmp_path / "followup_result.json").read_text())["results"][0][
            "pair_universe"
        ]
        == 2
    )


def test_shared_variance_combination_is_block_diagonal_not_common_effect():
    from summit.epistasis.oracle import dense_summary
    from summit.epistasis.cohorts import combine_shared_variance_summaries
    from scipy.linalg import block_diag

    rng = np.random.default_rng(386)
    contract = dict(
        effect_distribution="independent_cohort_effects_shared_variances",
        phenotype_unit="raw_y",
        component_units={"epistasis": "declared_pair_kernel_per_mass", "residual": "I"},
    )
    summaries = []
    kernels = []
    phenotypes = []
    for i, n in enumerate((24, 32)):
        f = rng.normal(size=(n, 5))
        k = np.stack([f @ f.T / 5, np.eye(n)])
        y = rng.normal(size=n)
        kernels.append(k)
        phenotypes.append(y)
        summaries.append(
            dense_summary(
                k,
                y,
                component_names=("epistasis", "residual"),
                trait_names=("y",),
                residual_rank=n,
                metadata=dict(
                    cohort_variance_contract=contract,
                    overlap_contract="mutually_disjoint_cohorts",
                    sample_hash=str(i),
                ),
            )
        )
    combined = combine_shared_variance_summaries(summaries)
    expected = dense_summary(
        np.stack([block_diag(kernels[0][j], kernels[1][j]) for j in range(2)]),
        np.r_[*phenotypes],
        component_names=("epistasis", "residual"),
        trait_names=("y",),
        residual_rank=56,
        metadata={"test": True},
    )
    for key in ("matrix", "rhs", "cubic", "traces"):
        np.testing.assert_allclose(
            getattr(combined, key), getattr(expected, key), atol=1e-10
        )


def test_independent_target_batch_shares_unprojected_sources_only():
    from summit.epistasis.features import prepare_shared_target_sources

    rng = np.random.default_rng(471)
    raw = rng.binomial(2, 0.4, (60, 12))
    axis = VariantAxis(
        tuple(f"v{i}" for i in range(12)),
        ("1",) * 12,
        tuple(range(1, 13)),
        ("A",) * 12,
        ("G",) * 12,
    )
    source = ArrayGenotypeSource(
        raw, [(str(i), str(i)) for i in range(60)], axis, hard_calls=True
    )
    scale = fit_scale(source, np.arange(60))
    ann = annotation_weights(axis.ids, {})
    studies = []
    jobs = []
    for i in range(2):
        job = dict(
            id=f"t{i}",
            additive_annotations=["all"],
            components=[dict(name="epi", target=f"v{i}", background="all")],
            local_variants=[f"v{i+3}"],
        )
        design = target_design(
            source,
            np.arange(60),
            scale,
            annotations=ann,
            components=job["components"],
            additive_annotations=["all"],
            local_variants=job["local_variants"],
        )
        studies.append(
            SelectedStudy(
                source, np.arange(60), scale, **design, backend="numpy", block_size=4
            )
        )
        jobs.append(job)
    shared = prepare_shared_target_sources(
        studies[0], [s.weights[:, 1] for s in studies], dimensions=8
    )
    for study, job in zip(studies, jobs):
        batched = prepare_feature_reference(
            study, job, ann, sketch_dimensions=8, shared_sources=shared
        )
        alone = prepare_feature_reference(study, job, ann, sketch_dimensions=8)
        np.testing.assert_array_equal(batched.fixed_effects, alone.fixed_effects)
        np.testing.assert_allclose(batched.features, alone.features, atol=1e-14)
        assert batched.metadata["genotype_passes"]["interaction_feature_sketch"] == 0


def test_native_implicit_sketch_matches_numpy_across_tilings():
    import os

    try:
        from summit import gxeldcore
    except ImportError:
        pytest.skip("native module not loaded")
    from epistasis_helpers import epistasis_threads
    threads = epistasis_threads()
    rng = np.random.default_rng(879)
    raw = rng.binomial(2, 0.4, (72, 14)).astype(float)
    raw[3, 5] = np.nan
    axis = VariantAxis(
        tuple(f"v{i}" for i in range(14)),
        ("1",) * 14,
        tuple(range(1, 15)),
        ("A",) * 14,
        ("G",) * 14,
    )
    results = []
    for backend, tile in (("numpy", 3), ("native", 5)):
        source = ArrayGenotypeSource(
            raw, [(str(i), str(i)) for i in range(72)], axis, hard_calls=True
        )
        scale = fit_scale(source, np.arange(72), threads=threads)
        ann = annotation_weights(
            axis.ids,
            {
                "a": {f"v{i}": 1.0 for i in range(5)},
                "b": {f"v{i}": 1.0 for i in range(2, 9)},
            },
        )
        design = target_design(
            source,
            np.arange(72),
            scale,
            components=[],
            annotations=ann,
            additive_annotations=["all", "a", "b"],
            allow_additive_only=True,
            threads=threads,
        )
        study = SelectedStudy(
            source,
            np.arange(72),
            scale,
            **design,
            backend=backend,
            threads=threads,
            block_size=tile,
        )
        job = dict(
            id="cross",
            additive_annotations=["all", "a", "b"],
            groups=[dict(name="cross", mode="cross", left="a", right="b")],
        )
        results.append(
            prepare_feature_reference(
                study, job, ann, sketch_dimensions=16, seed=77
            ).features
        )
    np.testing.assert_allclose(*results, atol=2e-12)


def test_per_probe_matrices_exact_draws_saved_uncertainty_and_uniform_bound(tmp_path):
    from test_epistasis import fixture
    from summit.epistasis.oracle import selected_kernels
    from summit.epistasis.summary import write_summary, load_summary, fit_epistasis
    from summit.epistasis.probe import uniform_reference_bound
    from summit.ldscore.generalized_gxe_variant import generate_global_variant_probes

    study, x, e, w, fixed, y = fixture()
    reference = study.reference(nvecs=32, seed=17)
    _, projection = selected_kernels(x, e, w, fixed)
    features = [
        (projection.projector @ (x * e[:, a, None])) * np.sqrt(w[:, a] / w[:, a].sum())
        for a in range(w.shape[1])
    ]
    z = generate_global_variant_probes(np.arange(len(w)), np.arange(32), root_seed=17)
    for a in range(study.c):
        for b in range(study.c):
            expected = (
                np.sum((features[a].T @ features[b] @ z) ** 2, axis=0)
                + np.sum((features[b].T @ features[a] @ z) ** 2, axis=0)
            ) / 2
            np.testing.assert_allclose(
                reference.probe_matrices[:, a, b], expected, atol=1e-10
            )
    np.testing.assert_allclose(
        reference.probe_matrices.mean(axis=0), reference.matrix, atol=1e-10
    )
    summary, _ = study.summarize(reference, y, trait_names=("a", "b"))
    restored = load_summary(write_summary(summary, tmp_path / "random.npz"))
    fit = fit_epistasis(restored)
    assert np.all(fit["probe_uncertainty"]["standard_errors"] >= 0)
    assert not fit["probe_uncertainty"]["included_in_fame_se"]
    bound = uniform_reference_bound(
        np.diag([2.0, 1.0]), genetic_count=1, probes=100000, failure_probability=0.01
    )
    assert bound["informative"]
    assert bound["relative_coefficient_energy_bound"] < 0.1
    assert (
        uniform_reference_bound(np.eye(2), genetic_count=1, probes=4)["informative"]
        is False
    )
