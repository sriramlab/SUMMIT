import json
import numpy as np
import pytest
from summit.epistasis.trans import trans_membership
from summit.prediction.spec import VariantAxis


def test_stratified_region_and_pairs_share_complete_nuisance_span(tmp_path):
    from dataclasses import replace
    from bed_reader import to_bed
    from epistasis_helpers import cli as main
    from summit.epistasis.features import (
        load_feature_reference,
        write_feature_reference,
    )
    from summit.epistasis.robust import load_robust_scores, robust_score_tests

    rng = np.random.default_rng(730191)
    n = 4096
    strata = rng.integers(2, size=n)
    g = np.column_stack(
        [
            rng.binomial(2, 0.2 + 0.2 * strata),
            rng.binomial(2, 0.2 + 0.25 * strata),
            rng.binomial(2, 0.3, size=n),
        ]
    ).astype(float)
    mean = (1 + 3 * strata) * g[:, 1] + 0.5 * g[:, 2]
    y = mean + rng.normal(size=n)
    ids = list(map(str, range(n)))
    to_bed(
        tmp_path / "g.bed",
        g,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=["t", "b", "c"],
            chromosome=["1", "2", "2"],
            bp_position=[100, 100, 200],
            allele_1=["A"] * 3,
            allele_2=["G"] * 3,
        ),
    )
    (tmp_path / "samples").write_text("FID IID\n" + "".join(f"{i} {i}\n" for i in ids))
    (tmp_path / "strata").write_text(
        "FID IID stratum\n" + "".join(f"{i} {i} s{v}\n" for i, v in enumerate(strata))
    )
    (tmp_path / "y").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v:.17g}\n" for i, v in enumerate(y))
    )
    settings = dict(
        method="robust_mean",
        sampling_model="iid_population_projection",
        main_effects="tested_variants",
        dominance="tested_variants",
        save_reference=True,
    )
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="g.bed"),
        samples="samples",
        phenotypes=dict(file="y", columns=["y"], unit="simulation"),
        strata=dict(file="strata", column="stratum"),
        annotations=dict(B={"b": 1.0, "c": 1.0}),
        jobs=[
            dict(
                id="region",
                trans_target="t",
                additive_annotations=["all"],
                components=[dict(name="trans", target="t", background="B")],
                inference=settings,
            ),
            dict(
                id="pairs",
                trans_target="t",
                additive_annotations=["all"],
                pairs=[["t", "b"], ["t", "c"]],
                inference=settings,
            ),
        ],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(spec))
    main(["prepare", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "out")])

    # Independent complete joint regression in study genotype units.
    mu = g.mean(0)
    x = (g - mu) / np.sqrt(mu * (1 - mu / 2))
    nuisance = np.column_stack([np.ones(n), x, g == 1])
    nuisance = np.column_stack([nuisance * (strata == s)[:, None] for s in (0, 1)])
    f = x[:, :1] * x[:, 1:]
    design = np.column_stack([nuisance, f])
    inverse = np.linalg.pinv(design)
    coefficients = inverse @ y
    residual = y - design @ coefficients
    leverage = np.einsum("ij,ji->i", design, inverse)
    errors = np.sqrt(np.sum((inverse[-2:] * (residual / (1 - leverage))) ** 2, axis=1))

    fits = []
    for name, factor in (("pairs", 1.0), ("region", np.sqrt(2))):
        saved = load_robust_scores(tmp_path / f"out/{name}.robust-score.npz")
        fit = robust_score_tests(saved)
        np.testing.assert_allclose(
            fit["beta"] / factor, coefficients[-2:], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(fit["standard_errors"] / factor, errors, rtol=1e-10)
        assert saved.metadata["fixed_rank"] == 14
        assert not saved.metadata["outside_confirmation_design"]
        fits.append(fit)
    assert fits[0]["kernel_p"] == pytest.approx(fits[1]["kernel_p"], abs=1e-10)

    reuse = dict(
        kind="summit.epistasis.prepare_traits",
        schema_version=1,
        reference="out/region.cohort-reference.npz",
        samples="samples",
        phenotypes=spec["phenotypes"],
    )
    (tmp_path / "reuse.json").write_text(json.dumps(reuse))
    main(
        [
            "prepare-traits",
            str(tmp_path / "reuse.json"),
            "--out",
            str(tmp_path / "reused.npz"),
        ]
    )
    reused = load_robust_scores(tmp_path / "reused.npz")
    np.testing.assert_allclose(
        robust_score_tests(reused)["beta"], fits[1]["beta"], rtol=1e-12
    )
    path = tmp_path / reuse["reference"]
    with np.load(path, allow_pickle=False) as archive:
        identity = json.loads(str(archive["manifest"]))["metadata"]["compatibility_id"]
    reference = load_feature_reference(path, compatibility_id=identity)
    meta = dict(reference.metadata)
    meta["definitions"] = dict(meta["definitions"])
    old_strata = dict(meta["definitions"]["fixed_effect_strata"])
    old_strata.pop("preparation")
    meta["definitions"]["fixed_effect_strata"] = old_strata
    write_feature_reference(replace(reference, metadata=meta), tmp_path / "legacy.npz")
    reuse["reference"] = "legacy.npz"
    (tmp_path / "reuse.json").write_text(json.dumps(reuse))
    with pytest.raises(ValueError, match="older stratified"):
        main(
            [
                "prepare-traits",
                str(tmp_path / "reuse.json"),
                "--out",
                str(tmp_path / "bad.npz"),
            ]
        )


def test_trans_axes_are_real_and_exclusions_are_interaction_only():
    axis = VariantAxis(
        ("a", "b", "c"),
        ("1", "1", "2"),
        np.array([100, 200, 300]),
        ("A",) * 3,
        ("G",) * 3,
    )
    assert trans_membership(axis, "a", ["c"])["chromosomes"] == ["2"]
    with pytest.raises(ValueError, match="target chromosome"):
        trans_membership(axis, "a", ["b", "c"])
    from summit.epistasis.trans import validate_trans_job

    # target_design always excludes the target itself, even if annotated.
    checked = validate_trans_job(
        dict(
            trans_target="a",
            components=[dict(target="a", background="B")],
            inference=dict(method="robust_mean"),
        ),
        axis,
        {"B": np.array([1.0, 0.0, 1.0])},
        {},
    )
    assert checked["components"][0]["member_count"] == 1


def test_public_trans_sampling_and_saved_fit(tmp_path, monkeypatch):
    from bed_reader import to_bed
    from epistasis_helpers import cli as main
    from summit.epistasis.robust import load_robust_scores, robust_score_tests

    rng = np.random.default_rng(13694)
    n, m = 240, 6
    g = rng.binomial(2, 0.3, (n, m)).astype(float)
    ids = list(map(str, range(n)))
    to_bed(
        tmp_path / "g.bed",
        g,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=[f"v{i}" for i in range(m)],
            chromosome=["1"] * 3 + ["2"] * 3,
            bp_position=np.arange(m) + 100,
            allele_1=["A"] * m,
            allele_2=["G"] * m,
        ),
    )
    (tmp_path / "samples").write_text("FID IID\n" + "".join(f"{i} {i}\n" for i in ids))
    y = g[:, 1] + rng.normal(size=n)
    (tmp_path / "y").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v}\n" for i, v in zip(ids, y))
    )
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="g.bed"),
        samples="samples",
        phenotypes=dict(file="y", columns=["y"], unit="simulation"),
        annotations={},
        jobs=[
            dict(
                id="t",
                additive_annotations=["all"],
                trans_target="v0",
                pairs=[["v0", "v3"]],
                local_variants=["v1"],
                inference=dict(
                    method="robust_mean",
                    sampling_model="iid_population_projection",
                    save_reference=True,
                ),
            )
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    main(["prepare", str(path), "--out", str(tmp_path / "out")])
    saved = load_robust_scores(tmp_path / "out/t.robust-score.npz")
    assert saved.metadata["sampling_model"] == "iid_population_projection"
    assert "population" in robust_score_tests(saved)["estimand"]
    (tmp_path / "reverse_samples").write_text(
        "FID IID\n" + "".join(f"{i} {i}\n" for i in reversed(ids))
    )
    reuse = dict(
        kind="summit.epistasis.prepare_traits",
        schema_version=1,
        reference="out/t.cohort-reference.npz",
        samples="reverse_samples",
        phenotypes=spec["phenotypes"],
    )
    (tmp_path / "reuse.json").write_text(json.dumps(reuse))
    with monkeypatch.context() as patch:

        def unavailable(*args, **kwargs):
            raise AssertionError("genotype input unavailable")

        patch.setattr("summit.prediction.genotype.source_from_spec", unavailable)
        main(
            [
                "prepare-traits",
                str(tmp_path / "reuse.json"),
                "--out",
                str(tmp_path / "reused.npz"),
            ]
        )
    reused = load_robust_scores(tmp_path / "reused.npz")
    for field in ("scores", "information", "score_covariance"):
        np.testing.assert_allclose(
            getattr(saved, field), getattr(reused, field), rtol=1e-12, atol=1e-12
        )
    assert reused.metadata["genotype_passes_this_preparation"] == 0
    assert "cohort_sample_tokens" not in reused.metadata
    (tmp_path / "reverse_samples").write_text(
        "FID IID\n" + "".join(f"{i} {i}\n" for i in ids[:-1])
    )
    with pytest.raises(ValueError, match="sample mask"):
        main(
            [
                "prepare-traits",
                str(tmp_path / "reuse.json"),
                "--out",
                str(tmp_path / "bad_reuse.npz"),
            ]
        )
    main(
        [
            "fit",
            str(tmp_path / "out/t.robust-score.npz"),
            "--out",
            str(tmp_path / "fit.json"),
        ]
    )
    main(["prepare", str(path), "--out", str(tmp_path / "out"), "--resume"])
    strata_path = tmp_path / "strata"
    strata_path.write_text(
        "FID IID group\n"
        + "".join(f"{i} {i} {'A' if int(i)<120 else 'B'}\n" for i in ids)
    )
    spec["strata"] = dict(file="strata", column="group")
    path.write_text(json.dumps(spec))
    main(["prepare", str(path), "--out", str(tmp_path / "stratified")])
    stratified = load_robust_scores(tmp_path / "stratified/t.robust-score.npz")
    assert stratified.metadata["fixed_rank"] > saved.metadata["fixed_rank"]
    # Editing only stratum membership must invalidate a completed artifact.
    strata_path.write_text(
        "FID IID group\n"
        + "".join(f"{i} {i} {'A' if int(i)%2 else 'B'}\n" for i in ids)
    )
    with pytest.raises(ValueError, match="does not match|changed"):
        main(["prepare", str(path), "--out", str(tmp_path / "stratified"), "--resume"])
    spec["jobs"][0]["pairs"] = [["v0", "v1"]]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="target chromosome"):
        main(["prepare", str(path), "--out", str(tmp_path / "wrong")])


def test_population_mode_does_not_relabel_conditional_bootstrap():
    from summit.epistasis.robust import prepare_robust_scores

    rng = np.random.default_rng(7135)
    with pytest.raises(ValueError, match="correct conditional mean"):
        prepare_robust_scores(
            rng.normal(size=(40, 1)),
            rng.normal(size=40),
            np.ones((40, 1)),
            feature_names=("f",),
            trait_names=("y",),
            metadata={},
            wild_draws=99,
            sampling_model="iid_population_projection",
        )


def test_stratification_includes_main_slopes_and_budget():
    from summit.epistasis.trans import stratified_fixed_effects

    c = np.column_stack([np.ones(20), np.arange(20)])
    labels = np.array(["A"] * 10 + ["B"] * 10)
    stratified, record = stratified_fixed_effects(c, labels, memory_bytes=2**30)
    np.testing.assert_array_equal(stratified[:10, :2], c[:10])
    np.testing.assert_array_equal(stratified[10:, 2:], c[10:])
    assert np.count_nonzero(stratified[:10, 2:]) == 0
    assert record["counts"] == [10, 10]
    with pytest.raises(MemoryError):
        stratified_fixed_effects(c, labels, memory_bytes=1)
