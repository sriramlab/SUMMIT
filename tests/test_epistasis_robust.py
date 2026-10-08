"""Reference compatibility and robust conditional-mean epistasis inference."""
import json
import numpy as np
import pytest
from epistasis_helpers import epistasis_threads


@pytest.mark.parametrize("nested_draws", [0, 2])
def test_direction_confirmation_driver_full_and_nested(
    tmp_path, monkeypatch, nested_draws
):
    import csv
    import sys
    from scripts.epistasis import direction_continuation as driver
    from epistasis_helpers import cli
    monkeypatch.setattr(driver,'epistasis_main',cli)
    score_frozen=driver.score_frozen
    monkeypatch.setattr(driver,'score_frozen',lambda *a,**k:score_frozen(*a,**dict(k,threads=epistasis_threads())))

    seen = []
    prepare = driver.prepare_robust_scores

    def capture(features, outcomes, *args, **kwargs):
        seen.append(np.asarray(outcomes).copy())
        return prepare(features, outcomes, *args, **kwargs)

    monkeypatch.setattr(driver, "prepare_robust_scores", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "direction_continuation",
            "--out",
            str(tmp_path / "results"),
            "--scratch",
            str(tmp_path / "scratch"),
            "--replicates",
            "1",
            "--nested-draws",
            str(nested_draws),
            "--training-samples",
            "128",
            "--test-samples",
            "512",
            "--variants",
            "128",
        ],
    )
    driver.main()
    with (tmp_path / "results/replicates.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == (4 * nested_draws if nested_draws else 16)
    assert all(row["failed"] == "False" for row in rows)
    assert all(0 <= float(row["p"]) <= 1 for row in rows)
    assert len(seen) == (4 if nested_draws else 16)
    if nested_draws:
        assert {row["setting"] for row in rows} == {"null"}
        assert seen[0].shape == (512, nested_draws)
        assert not np.array_equal(seen[0][:, 0], seen[0][:, 1])
    else:
        assert len({row["setting"] for row in rows}) == 4
        assert seen[0].shape == (512,)
    for start in range(0, len(seen), 4):
        for outcomes in seen[start + 1 : start + 4]:
            np.testing.assert_array_equal(outcomes, seen[start])


def test_positive_gamma_tail_independent_integral():
    from summit.epistasis.quadratic import quadratic_sf
    from scipy.integrate import quad
    from scipy.stats import chi2

    # Sum chi2_1 and twice chi2_2: integrate the exponential survival
    # conditional on Z^2. Includes moderate and very small tail values.
    for value in (1.0, 9.0, 30.0, 90.0):
        expected = (
            chi2.sf(value, 1)
            + quad(
                lambda z: 2
                * np.exp(-z * z / 2)
                / np.sqrt(2 * np.pi)
                * np.exp(-(value - z * z) / 4),
                0,
                np.sqrt(value),
                epsabs=1e-13,
            )[0]
        )
        got = quadratic_sf(value, [1.0, 2.0], multiplicities=[1, 2], atol=1e-11)
        assert abs(got["p"] - expected) <= got["absolute_error"] + 1e-13


@pytest.mark.parametrize("mode", ["within", "cross", "overlap", "remainder"])
@pytest.mark.parametrize("dimensions", [None, 8])
def test_public_saved_group_reference_rejects_changed_membership(
    tmp_path, mode, dimensions
):
    from bed_reader import to_bed
    from epistasis_helpers import cli as main

    n, m = 80, 10
    rng = np.random.default_rng(516)
    raw = rng.binomial(2, 0.4, (n, m)).astype(float)
    ids = list(map(str, range(n)))
    to_bed(
        tmp_path / "input.bed",
        raw,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=[f"v{i}" for i in range(m)],
            chromosome=["1"] * m,
            bp_position=np.arange(1, m + 1),
            allele_1=["A"] * m,
            allele_2=["G"] * m,
        ),
    )
    (tmp_path / "samples.tsv").write_text(
        "FID IID\n" + "".join(f"{i} {i}\n" for i in ids)
    )
    y = rng.normal(size=n)
    (tmp_path / "y.tsv").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v}\n" for i, v in zip(ids, y))
    )
    group = dict(name="test", mode="cross" if mode == "overlap" else mode, left="A")
    if mode in ("cross", "overlap"):
        group["right"] = "B"
    settings = dict(
        method="linear_exact", main_effects="all_genotypes", save_reference=True
    )
    if dimensions is not None:
        settings["feature_sketch_dimensions"] = dimensions
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="input.bed"),
        samples="samples.tsv",
        phenotypes=dict(file="y.tsv", columns=["y"]),
        annotations=dict(
            A=dict(v1=1.0, v2=2.0),
            B=dict(v2=1.0, v7=1.0) if mode == "overlap" else dict(v6=1.0, v7=1.0),
        ),
        jobs=[
            dict(
                id="group",
                additive_annotations=["all"],
                groups=[group],
                inference=settings,
            )
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    assert main(["prepare", str(manifest), "--out", str(tmp_path / "original")]) == 0
    settings.pop("save_reference")
    settings["reference"] = "original/group.cohort-reference.npz"
    # Equivalent resolved weights, different mapping order/explicit zero.
    spec["annotations"]["A"] = dict(v2=2.0, v0=0.0, v1=1.0)
    manifest.write_text(json.dumps(spec))
    assert main(["prepare", str(manifest), "--out", str(tmp_path / "equivalent")]) == 0
    # An intact old array is not proof that its scientific definition matches.
    reference_path = tmp_path / "original/group.cohort-reference.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    old = json.loads(str(arrays["manifest"]))
    old["schema_version"] = 1
    arrays["manifest"] = np.array(json.dumps(old))
    np.savez(tmp_path / "old.npz", **arrays)
    from summit.epistasis.features import load_feature_reference

    with pytest.raises(ValueError, match="prepare a new version-2"):
        load_feature_reference(
            tmp_path / "old.npz", compatibility_id=old["metadata"]["compatibility_id"]
        )
    for label, changed in [
        ("membership", dict(v3=1.0, v4=2.0)),
        ("weight", dict(v1=2.0, v2=1.0)),
    ]:
        spec["annotations"]["A"] = changed
        manifest.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="compatibility"):
            main(["prepare", str(manifest), "--out", str(tmp_path / label)])


def test_hc3_independent_dense_reference_and_recoding(tmp_path):
    from summit.epistasis.robust import (
        prepare_robust_scores,
        robust_score_tests,
        write_robust_scores,
        load_robust_scores,
    )

    rng = np.random.default_rng(717)
    n = 173
    x = rng.binomial(2, [0.2, 0.35, 0.4], (n, 3)).astype(float)
    c = np.column_stack([np.ones(n), x, x == 1])
    f = np.column_stack([x[:, 0] * x[:, 1], x[:, 0] * x[:, 2]])
    y = (
        c @ rng.normal(size=(7, 2))
        + f @ np.array([[0.2, -0.1], [-0.3, 0.4]])
        + rng.normal(size=(n, 2)) * (0.4 + x[:, 0, None])
    )
    s = prepare_robust_scores(
        f, y, c, feature_names=("a", "b"), trait_names=("u", "v"), metadata={}
    )
    d = np.column_stack([c, f])
    di = np.linalg.pinv(d)
    coefficients = di @ y
    residual = y - d @ coefficients
    leverage = np.diag(d @ di)
    for j in range(2):
        expected = (di * ((residual[:, j] / (1 - leverage)) ** 2)) @ di.T
        got = robust_score_tests(s, trait=j, burden=[1, 1])
        np.testing.assert_allclose(got["beta"], coefficients[-2:, j], atol=1e-11)
        np.testing.assert_allclose(
            got["standard_errors"] ** 2, np.diag(expected)[-2:], rtol=1e-11
        )
    flipped = f.copy()
    flipped[:, 0] *= -1
    sf = prepare_robust_scores(
        flipped, y, c, feature_names=("a", "b"), trait_names=("u", "v"), metadata={}
    )
    got = robust_score_tests(s)
    recoded = robust_score_tests(sf)
    np.testing.assert_allclose(recoded["beta"], got["beta"] * [-1, 1], atol=1e-12)
    assert recoded["kernel_p"] == pytest.approx(got["kernel_p"], abs=1e-12)
    path = write_robust_scores(s, tmp_path / "scores.npz")
    loaded = load_robust_scores(path)
    np.testing.assert_array_equal(loaded.score_covariance, s.score_covariance)
    duplicate = prepare_robust_scores(
        np.column_stack([f, f[:, 0]]),
        y,
        c,
        feature_names=("a", "b", "duplicate_a"),
        trait_names=("u", "v"),
        metadata={},
    )
    repeated = robust_score_tests(duplicate, weights=[0.5, 1, 0.5])
    np.testing.assert_allclose(repeated["kernel_p"], got["kernel_p"], atol=1e-9)
    assert repeated["interaction_rank"] == 2
    assert repeated["estimable_coefficients"].tolist() == [False, True, False]
    assert np.isnan(repeated["beta"][[0, 2]]).all()
    assert repeated["beta"][1] == pytest.approx(got["beta"][1])
    contrasts = robust_score_tests(
        duplicate, contrasts={"sum": [1, 0, 1], "difference": [1, 0, -1]}
    )["coefficient_contrasts"]
    assert contrasts[0]["beta"] == pytest.approx(got["beta"][0])
    assert contrasts[0]["standard_error"] == pytest.approx(got["standard_errors"][0])
    assert contrasts[1]["p"] is None


def test_scalar_wild_refits_and_unknown_mean():
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests

    rng = np.random.default_rng(816)
    n = 131
    c = np.column_stack([np.ones(n), rng.normal(size=n)])
    f = rng.normal(size=(n, 1))
    y = rng.normal(size=n) * (1 + abs(c[:, 1]))
    args = dict(
        feature_names=("f",), trait_names=("y",), metadata={}, wild_draws=99, seed=191
    )
    s = prepare_robust_scores(f, y, c, **args)
    shifted = prepare_robust_scores(f, y + c @ [317.0, -4.0], c, **args)
    result = robust_score_tests(s)
    other = robust_score_tests(shifted)
    assert result["wild_bootstrap"] == other["wild_bootstrap"]
    assert 0.01 <= result["wild_bootstrap"]["p"] <= 1
    # Independently reproduce every scalar restricted wild refit in full design.
    null = c @ np.linalg.lstsq(c, y, rcond=None)[0]
    hc = np.diag(c @ np.linalg.pinv(c))
    d = np.column_stack([c, f])
    di = np.linalg.pinv(d)
    leverage = np.diag(d @ di)
    observed = abs(result["beta"][0] / result["standard_errors"][0])
    hits = 0
    rng = np.random.default_rng(191)
    for begin in (0, 64):
        draws = min(64, 99 - begin)
        signs = 2 * rng.integers(0, 2, (n, draws)) - 1
        for j in range(draws):
            boot = (y - null) / np.sqrt(1 - hc) * signs[:, j]
            bhat = di @ boot
            e = boot - d @ bhat
            variance = np.sum((di[-1] * e / (1 - leverage)) ** 2)
            hits += abs(bhat[-1]) / np.sqrt(variance) >= observed
    assert result["wild_bootstrap"]["exceedances"] == hits
    both = prepare_robust_scores(
        f, np.column_stack([y, 2 * y]), c, **dict(args, trait_names=("u", "v"))
    )
    assert (
        robust_score_tests(both, trait=1)["wild_bootstrap"] == result["wild_bootstrap"]
    )


def test_interaction_absorbed_by_main_effects_is_not_numerically_tested():
    from summit.epistasis.robust import prepare_robust_scores

    rng = np.random.default_rng(291)
    c = np.column_stack([np.ones(100), rng.normal(size=100)])
    with pytest.raises(ValueError, match="absorbed"):
        prepare_robust_scores(
            c[:, 1, None],
            rng.normal(size=100),
            c,
            feature_names=("f",),
            trait_names=("y",),
            metadata={},
        )


def test_integer_dosages_do_not_identify_heterozygosity():
    from test_epistasis import fixture
    from summit.epistasis.models import target_design

    study, *_ = fixture()
    assert not study.source.hard_calls
    with pytest.raises(ValueError, match="declared hard-call source"):
        target_design(
            study.source,
            study.rows,
            study.scale,
            components=[],
            annotations={"all": np.ones(study.m)},
            additive_annotations=["all"],
            allow_additive_only=True,
            dominance_variants=["v0"],
        )


def test_pairwise_ld_does_not_identify_interaction_score_covariance():
    # Both panels have independent Bernoulli(.5) alleles at each locus and
    # HWE diploid genotype margins, with exactly the same pairwise LD.
    from itertools import product

    hap = np.array(list(product([-1.0, 1.0], repeat=3)))
    parity = np.column_stack([hap, np.prod(hap, axis=1)])
    independent = np.array(list(product([-1.0, 1.0], repeat=4)))

    def diploid(h):
        return (h[:, None, :] + h[None, :, :]).reshape(-1, 4) / np.sqrt(2)

    a, b = map(diploid, (parity, independent))
    np.testing.assert_allclose(a.T @ a / len(a), b.T @ b / len(b), atol=1e-15)
    assert np.mean(np.prod(a, axis=1)) == pytest.approx(0.5)
    assert np.mean(np.prod(b, axis=1)) == pytest.approx(0.0)


def test_robust_public_cli_saved_fit(tmp_path):
    from bed_reader import to_bed
    from epistasis_helpers import cli as main

    n, m = 240, 300
    rng = np.random.default_rng(201)
    raw = rng.binomial(2, 0.4, (n, m)).astype(float)
    ids = list(map(str, range(n)))
    to_bed(
        tmp_path / "input.bed",
        raw,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=[f"v{i}" for i in range(m)],
            chromosome=["1"] * m,
            bp_position=np.arange(1, m + 1),
            allele_1=["A"] * m,
            allele_2=["G"] * m,
        ),
    )
    (tmp_path / "samples.tsv").write_text(
        "FID IID\n" + "".join(f"{i} {i}\n" for i in ids)
    )
    y = raw[:, 0] + 2 * (raw[:, 1] == 1) + rng.normal(size=n) * (1 + raw[:, 0])
    (tmp_path / "y.tsv").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v}\n" for i, v in zip(ids, y))
    )
    # Additional dense local source deliberately uses the reverse sample order.
    to_bed(
        tmp_path / "dense.bed",
        np.ascontiguousarray(raw[::-1, 2:4]),
        properties=dict(
            fid=ids[::-1],
            iid=ids[::-1],
            sid=["dense2", "dense3"],
            chromosome=["1"] * 2,
            bp_position=[3, 4],
            allele_1=["A"] * 2,
            allele_2=["G"] * 2,
        ),
    )
    (tmp_path / "dense-variants.txt").write_text("dense2\ndense3\n")
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="input.bed"),
        samples="samples.tsv",
        fixed_genotypes=[
            dict(
                name="dense_local",
                genotypes=dict(geno="dense.bed"),
                variants="dense-variants.txt",
                dominance=True,
            )
        ],
        phenotypes=dict(file="y.tsv", columns=["y"], unit="simulated"),
        annotations={},
        jobs=[
            dict(
                id="pair",
                additive_annotations=["all"],
                pairs=[["v0", "v1"]],
                inference=dict(
                    method="robust_mean",
                    dominance="tested_variants",
                    wild_draws=99,
                    save_reference=True,
                ),
            )
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    main(["prepare", str(manifest), "--out", str(tmp_path / "prepared")])
    # Inference never calls a genotype reader or accesses the phenotype files.
    from unittest.mock import patch

    original = (tmp_path / "prepared/pair.robust-score.npz").read_bytes()
    with patch(
        "summit.epistasis.inference_workflow.prepare_feature_reference",
        side_effect=AssertionError("unnecessary preparation"),
    ):
        main(
            ["prepare", str(manifest), "--out", str(tmp_path / "prepared"), "--resume"]
        )
    assert (tmp_path / "prepared/pair.robust-score.npz").read_bytes() == original
    with patch(
        "summit.prediction.genotype.source_from_spec",
        side_effect=AssertionError("genotype access"),
    ):
        main(
            [
                "fit",
                str(tmp_path / "prepared/pair.robust-score.npz"),
                "--out",
                str(tmp_path / "fit.json"),
            ]
        )
    result = json.loads((tmp_path / "fit.json").read_text())["fits"][0]
    assert result["diagnostics"]["n_samples"] == n
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests

    x = (raw - raw.mean(0)) / np.sqrt(raw.mean(0) * (1 - raw.mean(0) / 2))
    expected = robust_score_tests(
        prepare_robust_scores(
            (x[:, 0] * x[:, 1])[:, None],
            y,
            np.column_stack([np.ones(n), raw[:, :4], raw[:, :4] == 1]),
            feature_names=("f",),
            trait_names=("y",),
            metadata={},
        )
    )
    np.testing.assert_allclose(result["beta"], expected["beta"], atol=1e-12)
    np.testing.assert_allclose(
        result["standard_errors"], expected["standard_errors"], atol=1e-12
    )
    assert result["wild_bootstrap"]["draws"] == 99
    follow = dict(
        kind="summit.epistasis.followup",
        schema_version=1,
        summary="prepared/pair.robust-score.npz",
        groups={"local": [0]},
    )
    (tmp_path / "follow.json").write_text(json.dumps(follow))
    main(
        [
            "followup",
            str(tmp_path / "follow.json"),
            "--out",
            str(tmp_path / "follow-result.json"),
        ]
    )
    assert (
        json.loads((tmp_path / "follow-result.json").read_text())["results"][0][
            "pair_universe"
        ]
        == 1
    )
    from summit.epistasis.robust import load_robust_scores, robust_followup

    saved = load_robust_scores(tmp_path / "prepared/pair.robust-score.npz")
    assert robust_followup(saved, {"local": [0]}, trait="y")["pair_universe"] == 1
    # Changed cohort outcomes may not reuse a previously completed summary.
    (tmp_path / "y.tsv").write_text(
        "FID IID y\n"
        + "".join(f"{i} {i} {v+raw[k,2]}\n" for k, (i, v) in enumerate(zip(ids, y)))
    )
    with pytest.raises(ValueError, match="completed summary"):
        main(
            ["prepare", str(manifest), "--out", str(tmp_path / "prepared"), "--resume"]
        )


def test_independent_direction_training_and_confirmation_cli(tmp_path):
    from bed_reader import to_bed
    from epistasis_helpers import cli as main
    from summit.prediction.artifacts import load_prediction_models
    from summit.prediction.genotype import FileGenotypeSource
    from summit.epistasis.directions import score_frozen

    rng = np.random.default_rng(610)
    n, m = 220, 250
    raw = rng.binomial(
        2, np.r_[np.full(100, 0.25), np.full(n - 100, 0.55)][:, None], (n, m)
    ).astype(float)
    raw[rng.random((n, m)) < 0.025] = np.nan
    ids = list(map(str, range(n)))
    to_bed(
        tmp_path / "input.bed",
        raw,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=[f"v{i}" for i in range(m)],
            chromosome=["1"] * m,
            bp_position=np.arange(1, m + 1),
            allele_1=["A"] * m,
            allele_2=["G"] * m,
        ),
    )
    for name, selected in [("train", range(100)), ("test", range(100, n))]:
        (tmp_path / f"{name}.tsv").write_text(
            "FID IID\n" + "".join(f"{i} {i}\n" for i in selected)
        )
    full = np.nan_to_num(raw, nan=0.8)
    y = full[:, 1] + (full[:, 0] - 0.8) * (full[:, 2] - 0.8) + rng.normal(size=n)
    (tmp_path / "y.tsv").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v}\n" for i, v in zip(ids, y))
    )
    (tmp_path / "variants.txt").write_text(
        "\n".join(f"v{i}" for i in range(1, m)) + "\n"
    )
    train = dict(
        kind="summit.epistasis.train_direction",
        schema_version=1,
        genotypes=dict(geno="input.bed"),
        samples="train.tsv",
        phenotype=dict(file="y.tsv", column="y", unit="simulated"),
        target="v0",
        variants="variants.txt",
        prior=dict(additive=1.0, interaction=1.0, residual=1.0),
        storage="packed",
        solver=dict(rtol=1e-8),
    )
    (tmp_path / "train.json").write_text(json.dumps(train))
    main(
        [
            "train-direction",
            str(tmp_path / "train.json"),
            "--out",
            str(tmp_path / "trained"),
        ]
    )
    main(
        [
            "train-direction",
            str(tmp_path / "train.json"),
            "--out",
            str(tmp_path / "trained"),
            "--resume",
        ]
    )
    # Interrupt the actual native CG path after a checkpoint, then resume it.
    from summit.prediction.operator import GenotypeOperator
    from unittest.mock import patch

    original_apply = GenotypeOperator.apply
    calls = []

    def interrupted(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("simulated interruption")
        return original_apply(self, *args, **kwargs)

    with patch.object(GenotypeOperator, "apply", interrupted):
        with pytest.raises(InterruptedError):
            main(
                [
                    "train-direction",
                    str(tmp_path / "train.json"),
                    "--out",
                    str(tmp_path / "interrupted"),
                ]
            )
    assert (tmp_path / "interrupted/solver.npz").exists()
    main(
        [
            "train-direction",
            str(tmp_path / "train.json"),
            "--out",
            str(tmp_path / "interrupted"),
            "--resume",
        ]
    )
    np.testing.assert_allclose(
        load_prediction_models(tmp_path / "interrupted/models")[0].weights,
        load_prediction_models(tmp_path / "trained/models")[0].weights,
        rtol=1e-9,
        atol=1e-10,
    )
    # A crash after model export but before wrapper export must still bind the
    # solver settings when the existing model is resumed.
    (tmp_path / "interrupted/direction.json").unlink()
    changed_solver = dict(train, solver=dict(rtol=1e-5))
    (tmp_path / "changed_solver.json").write_text(json.dumps(changed_solver))
    with pytest.raises(ValueError, match="solver settings changed"):
        main(
            [
                "train-direction",
                str(tmp_path / "changed_solver.json"),
                "--out",
                str(tmp_path / "interrupted"),
                "--resume",
            ]
        )
    definitions = [
        dict(name="pgs", direction="trained/direction.json", component=0, adjust=True),
        dict(name="interaction", direction="trained/direction.json", component=1),
    ]
    models = {m.model_id:m for m in load_prediction_models(tmp_path / "trained/models")}
    model = models["prespecified"]
    additive = models["additive_null"]
    with FileGenotypeSource(tmp_path / "input.bed") as source:
        scores, adjust, _ = score_frozen(
            definitions,
            tmp_path,
            source,
            np.arange(100, n),
            threads=epistasis_threads(),
            block_size=31,
            memory_bytes=2**30,
        )
        # Read the source's declared counted allele, not bed_reader defaults.
        source.prepare(np.arange(100, n), m, 1)
        g = source.read(np.arange(1, m)).astype(float)
        g = np.where(g == -127, model.scale.mean, g)
        expected = ((g - model.scale.mean) * model.scale.inverse_scale) @ model.weights
        expected_additive = ((g-additive.scale.mean)*additive.scale.inverse_scale) @ additive.weights[:,0]
        np.testing.assert_allclose(adjust[:, 0], expected_additive, atol=1e-10)
        np.testing.assert_array_equal(additive.weights[:,1],0)
        # Independently solve the additive-only primal ridge system. The
        # interaction learner must not alter these nuisance weights.
        source.prepare(np.arange(100),m,1)
        raw_training = source.read(np.arange(m)).astype(float)
        target = raw_training[:,0].copy()
        target[target==-127] = target[target!=-127].mean()
        c0 = np.column_stack([np.ones(100),target])
        z = np.where(raw_training[:,1:]==-127,additive.scale.mean,raw_training[:,1:])
        z = (z-additive.scale.mean)*additive.scale.inverse_scale
        z -= c0 @ np.linalg.lstsq(c0,z,rcond=None)[0]
        py = y[:100]-c0@np.linalg.lstsq(c0,y[:100],rcond=None)[0]
        independent = np.linalg.solve(z.T@z+(m-1)*np.eye(m-1),z.T@py)
        np.testing.assert_allclose(additive.weights[:,0],independent,atol=1e-9,rtol=1e-7)
        np.testing.assert_allclose(
            scores["interaction"]["values"], expected[:, 1], atol=1e-10
        )
        # Older frozen records selected both components from the joint model.
        # Keep their original numerical definition when loading old artifacts.
        legacy = json.loads((tmp_path / "trained/direction.json").read_text())
        legacy.pop("additive_model_identity")
        (tmp_path / "trained/legacy.json").write_text(json.dumps(legacy))
        _, legacy_adjust, _ = score_frozen(
            [dict(name="pgs", direction="trained/legacy.json", component=0, adjust=True)],
            tmp_path, source, np.arange(100, n), threads=epistasis_threads(),
            block_size=31, memory_bytes=2**30,
        )
        np.testing.assert_allclose(legacy_adjust[:, 0], expected[:, 0], atol=1e-10)
        # Frozen local main columns span exactly the saved prediction function,
        # including missing entries and a different confirmation allele frequency.
        from summit.epistasis.models import target_design
        from summit.epistasis.prepare import fit_scale

        names = ["v1", "v2", "v3"]
        repeated, _, _ = score_frozen(
            definitions,
            tmp_path,
            source,
            np.arange(100, n),
            threads=epistasis_threads(),
            block_size=17,
            memory_bytes=2**30,
            main_variants=names,
        )
        np.testing.assert_allclose(
            repeated["interaction"]["values"], expected[:, 1], atol=1e-10
        )
        scale = fit_scale(
            source, np.arange(100, n), threads=epistasis_threads(), block_size=31, memory_bytes=2**30
        )
        design = target_design(
            source,
            np.arange(100, n),
            scale,
            components=[],
            annotations={"all": np.ones(m)},
            additive_annotations=["all"],
            allow_additive_only=True,
            local_variants=names,
            main_imputation=repeated["interaction"]["main_imputation"],
            block_size=31,
        )
        actual = design["fixed_effects"][:, 1:]
        source.prepare(np.arange(100, n), 31, 1)
        dose = source.read(np.arange(1, 4)).astype(float)
        dose = np.where(dose == -127, model.scale.mean[:3], dose)
        np.testing.assert_allclose(
            actual, (dose - scale.mean[1:4]) * scale.inverse_scale[1:4], atol=1e-12
        )
        from summit.epistasis.prepare import SelectedStudy
        from summit.epistasis.features import prepare_feature_reference

        study = SelectedStudy(
            source, np.arange(100, n), scale, **design, backend="numpy"
        )
        job = dict(id="pair", pairs=[["v1", "v2"]], additive_annotations=["all"])
        ref = prepare_feature_reference(study, job, {"all": np.ones(m)})
        np.testing.assert_allclose(
            ref.features[:, 0], actual[:, 0] * actual[:, 1], atol=1e-12
        )
        with pytest.raises(ValueError, match="overlap"):
            score_frozen(
                definitions,
                tmp_path,
                source,
                np.arange(90, 120),
                threads=epistasis_threads(),
                block_size=31,
                memory_bytes=2**30,
            )
    to_bed(
        tmp_path / "flipped.bed",
        2 - raw,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=[f"v{i}" for i in range(m)],
            chromosome=["1"] * m,
            bp_position=np.arange(1, m + 1),
            allele_1=["G"] * m,
            allele_2=["A"] * m,
        ),
    )
    with FileGenotypeSource(tmp_path / "flipped.bed") as source:
        flipped, _, _ = score_frozen(
            definitions,
            tmp_path,
            source,
            np.arange(100, n),
            threads=epistasis_threads(),
            block_size=19,
            memory_bytes=2**30,
            main_variants=names,
        )
        np.testing.assert_allclose(
            flipped["interaction"]["values"],
            scores["interaction"]["values"],
            atol=1e-10,
        )
        for name, mu in repeated["interaction"]["main_imputation"].items():
            assert flipped["interaction"]["main_imputation"][name] == pytest.approx(
                2 - mu
            )
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="input.bed"),
        samples="test.tsv",
        phenotypes=dict(file="y.tsv", columns=["y"], unit="simulated"),
        annotations=dict(target=dict(v0=1)),
        frozen_scores=definitions,
        jobs=[
            dict(
                id="confirmation",
                additive_annotations=["all"],
                local_variants=["v0", "v1", "v2", "v3"],
                dominance_variants=["v0"],
                components=[
                    dict(
                        name="trained_direction",
                        frozen_score="interaction",
                        background="target",
                    )
                ],
                inference=dict(
                    method="robust_mean", main_effects="declared", save_reference=True
                ),
            )
        ],
    )
    (tmp_path / "test.json").write_text(json.dumps(spec))
    main(["prepare", str(tmp_path / "test.json"), "--out", str(tmp_path / "prepared")])
    main(
        [
            "fit",
            str(tmp_path / "prepared/confirmation.robust-score.npz"),
            "--out",
            str(tmp_path / "fit.json"),
        ]
    )
    assert (
        json.loads((tmp_path / "fit.json").read_text())["fits"][0]["diagnostics"][
            "n_samples"
        ]
        == 120
    )


def test_robust_cohort_affine_units_and_pooled_point_estimate(tmp_path):
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
    from summit.epistasis.robust_cohorts import (
        harmonize_robust_pairs,
        combine_robust_scores,
    )
    from summit.prediction.spec import VariantAxis
    from scipy.linalg import block_diag

    rng = np.random.default_rng(933)
    summaries = []
    all_f = []
    all_y = []
    all_c = []
    canonical = {
        "a": dict(chromosome="1", position=1, counted="A", other="G"),
        "b": dict(chromosome="1", position=2, counted="C", other="T"),
    }
    for j, n in enumerate((173, 229)):
        g = rng.binomial(2, 0.4, (n, 2)).astype(float)
        y = (
            0.3 * g[:, 0] * g[:, 1]
            + g @ [0.5, -0.2]
            + rng.normal(size=n) * (1 + g[:, 0])
        )
        observed = g.copy()
        if j:
            observed[:, 0] = 2 - observed[:, 0]
        inverse = np.array([1.3, 2.1])
        x = (observed - observed.mean(axis=0)) * inverse
        c = np.column_stack([np.ones(n), x])
        f = (x[:, 0] * x[:, 1])[:, None]
        axis = VariantAxis(
            ("a", "b"),
            ("1", "1"),
            (1, 2),
            ("G", "C") if j else ("A", "C"),
            ("A", "T") if j else ("G", "T"),
            "GRCh37",
        )
        meta = dict(
            pair_ids=[["a", "b"]],
            genome_build="GRCh37",
            trait_unit="unit" + str(j),
            variants=axis.to_dict(),
            inverse_scales=dict(a=inverse[0], b=inverse[1]),
            fixed_main_effect_variants=["a", "b"],
            sample_hash=str(j),
        )
        s = prepare_robust_scores(
            f,
            y / (j + 1),
            c,
            feature_names=("pair",),
            trait_names=("y",),
            metadata=meta,
        )
        from summit.epistasis.robust import write_robust_scores

        write_robust_scores(s, tmp_path / f"cohort{j}.npz")
        aligned, transform = harmonize_robust_pairs(
            s,
            canonical,
            genome_build="GRCh37",
            trait_unit="common",
            phenotype_multiplier=j + 1,
        )
        summaries.append(aligned)
        all_f.append((g[:, 0] * g[:, 1])[:, None])
        all_y.append(y)
        all_c.append(c)
    combined = combine_robust_scores(summaries, independent=True)
    pooled = np.column_stack([block_diag(*all_c), np.vstack(all_f)])
    expected = np.linalg.lstsq(pooled, np.concatenate(all_y), rcond=None)[0][-1]
    assert robust_score_tests(combined)["beta"][0] == pytest.approx(expected, abs=1e-12)
    full = block_diag(*[s.score_covariance[0] for s in summaries])[None]
    overlap = combine_robust_scores(summaries, cross_score_covariance=full)
    np.testing.assert_allclose(overlap.score_covariance, combined.score_covariance)
    with pytest.raises(ValueError, match="declare independent"):
        combine_robust_scores(summaries)
    # The public summary-only dispatcher must retain HC3 meat and apply its
    # mean-score unit transformation, not the older known-V GLS transformation.
    from epistasis_helpers import cli as main

    spec = dict(
        kind="summit.epistasis.combine",
        schema_version=1,
        cohorts=[
            dict(summary=f"cohort{j}.npz", phenotype_multiplier=j + 1) for j in range(2)
        ],
        canonical_variants=canonical,
        genome_build="GRCh37",
        trait_unit="common",
        independent=True,
    )
    (tmp_path / "cohorts.json").write_text(json.dumps(spec))
    main(
        [
            "combine",
            str(tmp_path / "cohorts.json"),
            "--out",
            str(tmp_path / "combined.npz"),
        ]
    )
    main(
        [
            "fit",
            str(tmp_path / "combined.npz"),
            "--out",
            str(tmp_path / "combined.json"),
        ]
    )
    result = json.loads((tmp_path / "combined.json").read_text())["fits"][0]
    np.testing.assert_allclose(result["beta"], robust_score_tests(combined)["beta"])
    np.testing.assert_allclose(
        result["standard_errors"], robust_score_tests(combined)["standard_errors"]
    )


def test_nuisance_only_unit_leverage_and_essential_row():
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests

    rng = np.random.default_rng(4421)
    n = 1200
    f = rng.normal(size=(n, 1))
    y = rng.normal(size=n)
    c = np.column_stack([np.ones(n), np.arange(n) == 0])

    def fit(f, y, c):
        return prepare_robust_scores(
            f, y, c, feature_names=("f",), trait_names=("y",), metadata={}
        )

    expected = robust_score_tests(fit(f[1:], y[1:], c[1:, :1]))
    for outcome in (y[0], 0.0, 1e6, 1e200):
        yy = y.copy()
        yy[0] = outcome
        summary = fit(f, yy, c)
        actual = robust_score_tests(summary)
        assert summary.metadata["nuisance_saturated_rows"] == 1
        np.testing.assert_allclose(actual["beta"], expected["beta"], atol=2e-11)
        np.testing.assert_allclose(
            actual["standard_errors"], expected["standard_errors"], atol=2e-11
        )
    with pytest.raises(ValueError, match="residual support"):
        fit(c[:, 1:], y, c[:, :1])


def test_absorbed_coordinate_preserves_kernel_and_original_family():
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests

    rng = np.random.default_rng(12211)
    n = 1200
    c = np.column_stack([np.ones(n), rng.normal(size=n)])
    f = rng.normal(size=(n, 1))
    y = rng.normal(size=n)

    def fit(f):
        return robust_score_tests(
            prepare_robust_scores(
                f,
                y,
                c,
                feature_names=tuple(str(j) for j in range(f.shape[1])),
                trait_names=("y",),
                metadata={},
            )
        )

    a = fit(f)
    b = fit(np.column_stack([f, c[:, 1]]))
    assert b["interaction_rank"] == 1
    assert b["kernel_p"] == pytest.approx(a["kernel_p"], abs=1e-9)
    assert b["sparse_bonferroni_p"] == pytest.approx(
        min(1, 2 * a["sparse_bonferroni_p"])
    )


def test_actual_hc3_ratio_distribution_and_many_covariate_unbiasedness():
    from summit.epistasis.robust_reference import (
        hc3_ratio_reference,
        many_covariate_reference,
    )
    from scipy.stats import norm

    rng = np.random.default_rng(74261)
    n = 120
    c = np.column_stack([np.ones(n), rng.normal(size=(n, 16))])
    f = rng.normal(size=n)
    omega = 0.3 + f * f
    ref = hc3_ratio_reference(f, c, omega, norm.isf(0.025))
    epsilon = np.sqrt(omega[:, None]) * rng.normal(size=(n, 20000))
    a, b = ref["coefficient_influence"], ref["denominator_form"]
    t = (a @ epsilon) / np.sqrt(np.sum(epsilon * (b @ epsilon), axis=0))
    assert abs(np.mean(abs(t) >= norm.isf(0.025)) - ref["p"]) < 0.005
    assert ref["probability_bracket"][1] - ref["probability_bracket"][0] < 1e-7
    means = c @ rng.normal(size=c.shape[1])
    estimates = many_covariate_reference(f, epsilon + means[:, None], c)
    truth = np.dot(a * a, omega)
    for name in ("leave_out", "hadamard"):
        values = estimates[name]
        assert abs(values.mean() - truth) < 5 * values.std() / np.sqrt(len(values))


def test_public_independent_orthogonal_native_matches_dense_ridge(tmp_path):
    from bed_reader import to_bed
    from epistasis_helpers import cli as main
    from summit.epistasis.features import load_feature_reference
    from summit.epistasis.robust import load_robust_scores, prepare_robust_scores
    from summit.prediction.genotype import FileGenotypeSource
    from summit.epistasis.prepare import fit_scale

    rng = np.random.default_rng(541778)
    n, m = 200, 32
    raw = rng.binomial(2, 0.3, (n, m)).astype(float)
    ids = list(map(str, range(n)))
    to_bed(
        tmp_path / "input.bed",
        raw,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=[f"v{i}" for i in range(m)],
            chromosome=["1"] * m,
            bp_position=np.arange(1, m + 1),
            allele_1=["A"] * m,
            allele_2=["G"] * m,
        ),
    )
    for name, selected in [("all", range(n)), ("train", range(100))]:
        (tmp_path / (name + ".tsv")).write_text(
            "FID IID\n" + "".join(f"{i} {i}\n" for i in selected)
        )
    y = raw @ rng.normal(size=m) / np.sqrt(m) + rng.normal(size=n)
    (tmp_path / "y.tsv").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v:.17g}\n" for i, v in enumerate(y))
    )
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="input.bed"),
        samples="all.tsv",
        phenotypes=dict(file="y.tsv", columns=["y"], unit="simulated"),
        annotations={},
        jobs=[
            dict(
                id="pair",
                additive_annotations=["all"],
                pairs=[["v0", "v1"]],
                inference=dict(
                    method="orthogonal_mean",
                    main_effects="tested_variants",
                    dominance="tested_variants",
                    training_samples="train.tsv",
                    ridge_variance=1.0,
                    residual_variance=1.0,
                    solver=dict(rtol=1e-10),
                    storage="packed",
                    save_reference=True,
                ),
            )
        ],
    )
    (tmp_path / "input.json").write_text(json.dumps(spec))
    main(["prepare", str(tmp_path / "input.json"), "--out", str(tmp_path / "out")])
    path = tmp_path / "out/pair.cohort-reference.npz"
    with np.load(path) as a:
        identity = json.loads(str(a["manifest"]))["metadata"]["compatibility_id"]
    ref = load_feature_reference(path, compatibility_id=identity)
    with FileGenotypeSource(tmp_path / "input.bed") as source:
        from summit.prediction.genotype import estimate_scale

        scale = estimate_scale(
            source,
            np.arange(100),
            np.arange(m),
            threads=epistasis_threads(),
            block_size=32,
            memory_bytes=2**30,
        )
        source.prepare(np.arange(n), m, 1)
        g = source.read(np.arange(m)).astype(float)
        x = (g - scale.mean) * scale.inverse_scale
    c = ref.fixed_effects
    response = np.column_stack([y, ref.features])
    ci = np.linalg.pinv(c[:100])
    px = x[:100] - c[:100] @ (ci @ x[:100])
    py = response[:100] - c[:100] @ (ci @ response[:100])
    effects = np.linalg.solve(px.T @ px + m * np.eye(m), px.T @ py)
    fixed = ci @ (response[:100] - x[:100] @ effects)
    residual = response[100:] - x[100:] @ effects - c[100:] @ fixed
    expected = prepare_robust_scores(
        residual[:, 1:],
        residual[:, :1],
        c[100:],
        feature_names=ref.metadata["feature_names"],
        trait_names=("y",),
        metadata={},
    )
    got = load_robust_scores(tmp_path / "out/pair.robust-score.npz")
    np.testing.assert_allclose(got.scores, expected.scores, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(got.information, expected.information, rtol=1e-8)
    np.testing.assert_allclose(
        got.score_covariance, expected.score_covariance, rtol=1e-8
    )
    assert got.metadata["nuisance_training_n"] == 100


def test_robust_artifact_feature_span_and_singular_conditional_covariance():
    from dataclasses import replace
    from summit.epistasis.robust import RobustScoreSummary, robust_score_tests

    valid = RobustScoreSummary(
        scores=np.array([[1.0], [1.0]]),
        information=np.ones((2, 2)),
        score_covariance=np.ones((1, 2, 2)),
        feature_names=("a", "duplicate"),
        trait_names=("y",),
        metadata=dict(method="fixture", inference="Gaussian limit"),
    )
    assert np.isfinite(robust_score_tests(valid)["kernel_p"])
    with pytest.raises(ValueError, match="outside the feature span"):
        replace(valid, scores=np.array([[1.0], [-1.0]]))
    with pytest.raises(ValueError, match="outside the feature span"):
        replace(valid, score_covariance=np.eye(2)[None])
    singular = replace(
        valid,
        information=np.eye(2),
        scores=np.array([[1.0], [0.0]]),
        score_covariance=np.diag([1.0, 0.0])[None],
        metadata=dict(
            valid.metadata, component_index=[0, 0], component_names=["panel"]
        ),
    )
    result = robust_score_tests(singular)
    assert np.isfinite(result["kernel_p"])
    assert result["conditional_components"][0]["p"] is None
    assert np.isnan(result["conditional_p"][1])
    assert robust_score_tests(singular, weights=[0, 1])["kernel_p"] is None
