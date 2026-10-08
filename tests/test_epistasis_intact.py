"""Independent regression references and complete-row matched-learning checks."""
from argparse import Namespace
import numpy as np
import pandas as pd


def test_joint_hc3_against_complete_regression_and_units():
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
    from scripts.epistasis.intact_validation import projection_truth

    rng = np.random.default_rng(381174)
    n = 1200
    g = rng.binomial(2, 0.35, (n, 8)).astype(float)
    # Dependence, all constituent means, non-orthogonal directions and covariance.
    g[:, 2] = np.where(rng.random(n) < 0.7, g[:, 1], g[:, 2])
    c = np.column_stack([np.ones(n), g, g == 1])
    f = g[:, 0, None] * np.column_stack([g[:, 1] - 0.7, 0.8 * g[:, 1] + g[:, 2] - 1.26])
    mean = c @ rng.normal(size=c.shape[1]) + f @ np.array([0.2, -0.1])
    y = mean + np.sqrt(0.4 + g[:, 0] ** 2) * rng.standard_t(5, n) / np.sqrt(5 / 3)
    design = np.column_stack([c, f])
    inv = np.linalg.inv(design.T @ design)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coef
    h = np.einsum("ij,jk,ik->i", design, inv, design)
    influence = (inv @ design.T) * (residual / (1 - h))[None, :]
    expected = influence @ influence.T
    for units in (np.ones(2), np.array([-3.0, 0.02])):
        s = prepare_robust_scores(
            f * units, y, c, feature_names=("a", "b"), trait_names=("y",), metadata={}
        )
        got = robust_score_tests(s)
        np.testing.assert_allclose(got["beta"] * units, coef[-2:], atol=1e-12)
        covariance = (
            np.linalg.inv(s.information)
            @ s.score_covariance[0]
            @ np.linalg.inv(s.information)
        )
        np.testing.assert_allclose(
            covariance * units[:, None] * units[None, :], expected[-2:, -2:], atol=2e-12
        )
        np.testing.assert_allclose(
            got["coefficient_covariance"], covariance, atol=2e-12
        )
        assert got["joint_df"] == 2
        np.testing.assert_allclose(
            projection_truth(f * units, c, mean) * units, [0.2, -0.1], atol=1e-12
        )
    assert abs(expected[-2, -1]) > 0.01 * expected[-1, -1]


def test_intact_full_learning_preserves_rows_and_truth(tmp_path, monkeypatch):
    from scripts.epistasis import intact_validation as driver
    from epistasis_helpers import cli
    monkeypatch.setattr(driver,'cli',cli)
    from summit.prediction.spec import VariantAxis

    rng = np.random.default_rng(83219)
    n, ma, mb = 900, 16, 12
    raw = rng.binomial(2, 0.35, (n, ma + mb)).astype(float)
    # Deliberately retain cross-chromosome dependence and background missingness.
    raw[:, ma] = np.where(rng.random(n) < 0.6, raw[:, 0], raw[:, ma])
    missing = rng.random(raw.shape) < 0.001
    missing[:, 0] = False
    raw[missing] = -127
    cov = rng.normal(size=(n, 15))
    axis = VariantAxis(
        tuple(f"v{i}" for i in range(ma + mb)),
        ("12",) * ma + ("5",) * mb,
        tuple(range(1, ma + mb + 1)),
        ("A",) * (ma + mb),
        ("C",) * (ma + mb),
        genome_build="GRCh37",
    )
    monkeypatch.setattr(
        driver, "intact_panel", lambda *args: (raw, cov, axis, dict(local_size=ma))
    )
    args = Namespace(
        out=tmp_path / "results",
        scratch=tmp_path,
        genotypes="unused",
        covariates="unused",
        target="12:66358347",
        background_chromosome="5",
        panel_seed=271,
        architecture_seed=318,
        seed=712,
        training_samples=256,
        test_samples=512,
        replicates=1,
        settings="finite,mixed",
        sampling="fixed,population",
        adjustments="pgs,supplied",
    )
    args.nested_models = 1
    args.nested_draws = 2
    args.nested_settings = "finite"
    driver.run(args)
    results = pd.read_csv(args.out / "replicates.csv")
    assert len(results) == 2 * 2 * 2 * 4
    assert not results.failed.any(), results.get("error")
    finite = results.query("setting == 'finite' and method != 'joint'")
    assert np.max(abs(finite.truth)) < 1e-9
    matched = results.query("setting == 'mixed'")
    assert np.allclose(matched.reference_signal_variance, 0.02)
    assert matched.realized_test_signal_variance.nunique() == 2
    # Missing background imputation differs between the generating donor law
    # and each training model; its nonzero projection remains in saved truths.
    oracle = matched.query("adjustment == 'supplied' and method == 'oracle'")
    assert np.isfinite(oracle.truth).all()
    nested = pd.read_csv(args.out / "frozen_replicates.csv")
    assert len(nested) == 4 and not nested.failed.any()
    assert nested.model_identity.nunique() == 1


def test_coverage_reducer_counts_numpy_booleans(tmp_path):
    from scripts.epistasis.intact_validation import reduce_records

    records = [
        dict(
            sampling="fixed",
            setting="finite",
            adjustment="pgs",
            method="learned",
            failed=False,
            p=0.1,
            outside_scope="",
            estimate=0.0,
            truth=0.0,
            se=1.0,
            coverage=np.bool_(i < 9),
            alignment_squared=0.2,
        )
        for i in range(10)
    ]
    # The joint row forces the mixed object column that exposed the old reducer.
    records.append(dict(records[0], method="joint", coverage=np.nan, estimate=np.nan))
    table = reduce_records(records, tmp_path)
    assert table.query("method == 'learned'").coverage.iloc[0] == 0.9


def test_fixed_batched_draws_equal_independent_regressions():
    from scripts.epistasis.intact_validation import frozen_fixed

    rng = np.random.default_rng(213)
    n = 240
    x = rng.normal(size=(n, 3))
    c = np.column_stack([np.ones(n), x])
    f = (x[:, 0] * x[:, 1])[:, None]
    mean = x.sum(1)
    rows = frozen_fixed(np.random.default_rng(17), f, c, mean, x, 3, np.zeros(1))
    noise = np.random.default_rng(17).normal(size=(n, 3))
    y = mean[:, None] + np.sqrt(0.4 + 0.6 * x[:, 0, None] ** 2) * noise
    design = np.column_stack([c, f])
    inverse = np.linalg.inv(design.T @ design)
    for j, row in enumerate(rows):
        b = np.linalg.lstsq(design, y[:, j], rcond=None)[0]
        h = np.sum((design @ inverse) * design, axis=1)
        influence = (inverse @ design.T)[-1] * (y[:, j] - design @ b) / (1 - h)
        np.testing.assert_allclose(
            [row["estimate"], row["se"]], [b[-1], np.linalg.norm(influence)], atol=1e-12
        )
