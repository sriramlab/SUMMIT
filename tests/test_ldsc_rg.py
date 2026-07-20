from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
from summit.inference.ldsc_rg import fit_constrained_cov_ldsc_irwls
from summit.inference.rgcore import InterceptFit, fit_rg
from summit.inference.sumcore import Sumcore
from summit.inference.trace import TraceView


def _manual_cov_irwls(
    design,
    response,
    ref_ld_total,
    weight_ld,
    *,
    n1,
    n2,
    m_annot,
    h1,
    h2,
    intercept,
    keep=None,
    iters=3,
):
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    ref_ld_total = np.asarray(ref_ld_total, dtype=np.float64)
    weight_ld = np.asarray(weight_ld, dtype=np.float64)
    if keep is None:
        keep = np.ones(response.size, dtype=bool)
    x = design[keep]
    y = response[keep]
    gamma = np.linalg.lstsq(x, y, rcond=None)[0]
    path = [gamma.copy()]
    m_total = float(np.sum(m_annot))

    final_weights = None
    final_lhs = None
    final_rhs = None
    for _ in range(iters):
        ld = np.maximum(ref_ld_total[keep], 1.0)
        overcount = np.maximum(weight_ld[keep], 1.0)
        a = 1.0 + n1 * np.clip(h1, 0.0, 1.0) * ld / m_total
        b = 1.0 + n2 * np.clip(h2, 0.0, 1.0) * ld / m_total
        c = intercept + np.sqrt(n1 * n2) * np.clip(gamma.sum(), -1.0, 1.0) * ld / m_total
        weights = 1.0 / (overcount * (a * b + c * c))
        root_w = np.sqrt(weights)
        gamma = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
        final_weights = weights
        final_lhs = (x * root_w[:, None]).T @ (x * root_w[:, None])
        final_rhs = (x * root_w[:, None]).T @ (y * root_w)
        path.append(gamma.copy())
    return gamma, final_weights, final_lhs, final_rhs, np.asarray(path)


@pytest.mark.parametrize("iters", [1, 3])
def test_cov_ldsc_matches_independent_partitioned_irwls(iters):
    design = np.asarray(
        [
            [0.10, 0.90],
            [0.35, 0.30],
            [0.80, 0.15],
            [0.20, 1.50],
            [1.70, 0.40],
            [0.65, 1.10],
            [2.10, 0.25],
        ]
    )
    response = design @ np.asarray([0.09, -0.035]) + np.asarray(
        [0.015, -0.025, 0.010, 0.030, -0.020, 0.012, -0.018]
    )
    ref_ld_total = np.asarray([0.2, 0.8, 1.3, 2.4, 3.1, 1.7, 4.0])
    weight_ld = np.asarray([7.0, 0.3, 2.2, 5.0, 1.4, 8.0, 3.3])
    m_annot = np.asarray([140.0, 260.0])
    kwargs = dict(
        n1=125.0,
        n2=175.0,
        m_annot=m_annot,
        h1=0.31,
        h2=0.47,
        intercept=-0.08,
        iters=iters,
    )
    expected = _manual_cov_irwls(
        design, response, ref_ld_total, weight_ld, **kwargs
    )
    observed = fit_constrained_cov_ldsc_irwls(
        design,
        response,
        ref_ld_total,
        weight_ld,
        n1_scale=kwargs["n1"],
        n2_scale=kwargs["n2"],
        m_annot=m_annot,
        h1_total=kwargs["h1"],
        h2_total=kwargs["h2"],
        intercept=kwargs["intercept"],
        irwls_iters=iters,
    )

    np.testing.assert_allclose(observed.gamma, expected[0], rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(observed.weights, expected[1], rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(observed.lhs, expected[2], rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(observed.rhs, expected[3], rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(observed.gamma_path, expected[4], rtol=3e-14, atol=3e-14)

    # Guard both important distinctions: the regression design is not floored,
    # and the explicit overcounting LD is not replaced by total reference LD.
    wrong_design = _manual_cov_irwls(
        np.maximum(design, 1.0), response, ref_ld_total, weight_ld, **kwargs
    )[0]
    wrong_weight_ld = _manual_cov_irwls(
        design, response, ref_ld_total, ref_ld_total, **kwargs
    )[0]
    assert not np.allclose(observed.gamma, wrong_design, rtol=1e-5, atol=1e-8)
    assert not np.allclose(observed.gamma, wrong_weight_ld, rtol=1e-5, atol=1e-8)


def test_single_component_uses_aggregate_constrained_initializer():
    ref_ld = np.asarray([0.4, 0.9, 1.3, 2.1, 3.0])
    n1 = 80.0
    n2 = 125.0
    m_annot = np.asarray([240.0])
    design = np.sqrt(n1 * n2) * ref_ld[:, None] / m_annot[0]
    response = np.asarray([0.05, 0.20, -0.03, 0.35, 0.42])
    observed = fit_constrained_cov_ldsc_irwls(
        design,
        response,
        ref_ld,
        np.asarray([1.2, 3.0, 2.2, 5.0, 7.0]),
        n1_scale=n1,
        n2_scale=n2,
        m_annot=m_annot,
        h1_total=0.3,
        h2_total=0.4,
        intercept=0.06,
        irwls_iters=1,
    )
    expected = m_annot.sum() * response.sum() / (
        np.sqrt(n1 * n2) * ref_ld.sum()
    )
    np.testing.assert_allclose(observed.gamma_path[0, 0], expected, rtol=0.0, atol=1e-15)
    assert observed.initialization == "aggregate_cov_ldsc"


def _integrated_fixture():
    m = 15
    k = 2
    idx = np.arange(m, dtype=np.float64)
    snps = np.asarray([f"rs{i + 1}" for i in range(m)])
    annot = np.column_stack(
        [np.ones(m), ((np.arange(m) % 4) != 0).astype(np.float64)]
    )
    ldscores = np.column_stack(
        [0.45 + 0.08 * idx, 0.20 + 0.025 * idx + 0.11 * (idx % 3)]
    )
    weight_ld = 0.55 + 0.29 * (idx % 5) + 0.03 * idx
    n1 = 90.0
    n2 = 130.0
    m_annot = annot.sum(axis=0, dtype=np.float64)
    design = np.sqrt(n1 * n2) * ldscores / m_annot[None, :]
    c_full = 0.075
    y = c_full + design @ np.asarray([0.042, -0.011]) + 0.018 * np.cos(idx)

    trace = TraceView(
        snps=snps,
        chr=np.repeat(np.arange(1, 4), 5),
        bp=np.arange(m, dtype=np.int64) * 1000 + 50_000,
        annot=annot,
        annot_header=np.asarray(["base", "shared"]),
        ldscores=ldscores,
        ldscores_reg_w=weight_ld[:, None],
    )
    jk = JackknifeDesign.from_trace_view(trace, JackknifeSpec.parse(3))
    R = jk.nrep
    c_reps = np.asarray([0.045, 0.095, 0.060, c_full])
    h1_components = np.tile(np.asarray([0.24, 0.08]), (R + 1, 1))
    h2_components = np.tile(np.asarray([0.35, 0.10]), (R + 1, 1))
    # Deliberately vary the delete plug-ins; this makes using only the full h2
    # values detectable in the exact-refit comparison.
    h1_components[:R] += np.asarray([[0.01, -0.005], [-0.015, 0.008], [0.005, 0.004]])
    h2_components[:R] += np.asarray([[-0.01, 0.006], [0.012, -0.004], [0.007, 0.003]])

    def h2_stub(components):
        totals = components.sum(axis=1)
        return SimpleNamespace(
            weight_mode="ldsc",
            prepared=SimpleNamespace(
                trace_view=trace,
                jackknife=jk,
                active_mask=np.ones(m, dtype=bool),
            ),
            sigma_reps=np.column_stack(
                [components, 1.0 - totals, totals]
            ),
            h2_reps=np.column_stack([components, totals]),
        )

    prepared = SimpleNamespace(
        trace_view=trace,
        jackknife=jk,
        y=y,
        active_mask=np.ones(m, dtype=bool),
        unit_sizes=jk.unit_sizes(dtype=np.float64),
        n1_scale=n1,
        n2_scale=n2,
    )
    intercept = InterceptFit(
        trace_view=trace,
        matched1=None,
        matched2=None,
        jackknife=jk,
        active_mask=np.ones(m, dtype=bool),
        unit_sizes=jk.unit_sizes(dtype=np.float64),
        ld=ldscores.sum(axis=1),
        y=y,
        c_reps=c_reps,
        c=np.asarray([c_full, 0.01]),
        info={"fixed": False, "source": "summit_intercept_regression"},
    )
    return (
        prepared,
        h2_stub(h1_components),
        h2_stub(h2_components),
        intercept,
        m_annot,
        design,
        ldscores.sum(axis=1),
        weight_ld,
    )


def test_fit_rg_ldsc_refits_weights_with_matching_h2_and_intercept_replicates():
    (
        prepared,
        h2_fit1,
        h2_fit2,
        intercept,
        m_annot,
        design,
        ref_ld_total,
        weight_ld,
    ) = _integrated_fixture()
    observed = fit_rg(
        prepared,
        h2_fit1,
        h2_fit2,
        intercept,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
        ldsc_irwls_iters=3,
        ldsc_irwls_tol=0.0,
    )

    jk = prepared.jackknife
    for r in range(jk.nrep + 1):
        keep = (
            np.ones(prepared.trace_view.nsnps, dtype=bool)
            if r == jk.nrep
            else (jk.D[r, jk.unit_id] < 0.5)
        )
        direct = fit_constrained_cov_ldsc_irwls(
            design,
            prepared.y - intercept.c_reps[r],
            ref_ld_total,
            weight_ld,
            n1_scale=prepared.n1_scale,
            n2_scale=prepared.n2_scale,
            m_annot=m_annot,
            h1_total=h2_fit1.h2_reps[r, -1],
            h2_total=h2_fit2.h2_reps[r, -1],
            intercept=intercept.c_reps[r],
            keep=keep,
            irwls_iters=3,
        )
        np.testing.assert_allclose(
            observed.gamma_reps[r], direct.gamma, rtol=3e-13, atol=3e-13
        )

    np.testing.assert_allclose(
        observed.rg_reps,
        observed.gamma_reps
        / np.sqrt(
            h2_fit1.sigma_reps[:, :2] * h2_fit2.sigma_reps[:, :2]
        ),
    )
    assert observed.weight_mode == "ldsc"
    assert observed.rg_se_method == "jackknife"
    assert observed.weight_info["intercept_replicates"] == "summit_delete_refits"
    assert observed.weight_info["h2_plugins"] == "matching_ldsc_delete_refits"
    assert observed.weight_info["jackknife_weights"] == "refit_irwls_per_replicate"
    assert observed.weight_info["n_failed_replicates"] == 0


def test_ldsc_rg_rejects_he_only_se_formulas():
    prepared, h2_fit1, h2_fit2, intercept, m_annot, *_rest = _integrated_fixture()
    with pytest.raises(ValueError, match="jackknife only"):
        fit_rg(
            prepared,
            h2_fit1,
            h2_fit2,
            intercept,
            weight_mode="ldsc",
            ldsc_m_annot=m_annot,
            rg_se_method="robust",
        )


def test_external_summit_intercept_stays_fixed_in_all_covariance_refits():
    prepared, h2_fit1, h2_fit2, intercept, m_annot, *_rest = _integrated_fixture()
    fixed_c = 0.037
    fixed = replace(
        intercept,
        c_reps=np.full(prepared.jackknife.nrep + 1, fixed_c),
        c=np.asarray([fixed_c, 0.0]),
        info={"fixed": True, "source": "pheno-rg"},
    )
    observed = fit_rg(
        prepared,
        h2_fit1,
        h2_fit2,
        fixed,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
        ldsc_irwls_iters=2,
    )
    np.testing.assert_array_equal(observed.intercept.c_reps, fixed.c_reps)
    assert observed.weight_info["intercept_fixed"] is True
    assert observed.weight_info["intercept_source"] == "pheno-rg"
    assert observed.weight_info["intercept_replicates"] == "fixed_across_replicates"


def test_ldsc_rg_is_symmetric_to_trait_order_and_antisymmetric_to_trait2_sign():
    prepared, h2_fit1, h2_fit2, intercept, m_annot, *_rest = _integrated_fixture()
    forward = fit_rg(
        prepared,
        h2_fit1,
        h2_fit2,
        intercept,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
        ldsc_irwls_iters=3,
    )

    swapped_prepared = SimpleNamespace(
        **{
            **prepared.__dict__,
            "n1_scale": prepared.n2_scale,
            "n2_scale": prepared.n1_scale,
        }
    )
    swapped = fit_rg(
        swapped_prepared,
        h2_fit2,
        h2_fit1,
        intercept,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
        ldsc_irwls_iters=3,
    )
    np.testing.assert_allclose(swapped.gamma_reps, forward.gamma_reps, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(swapped.rg_reps, forward.rg_reps, rtol=0.0, atol=2e-15)

    signed_prepared = SimpleNamespace(
        **{**prepared.__dict__, "y": -np.asarray(prepared.y)}
    )
    signed_intercept = replace(
        intercept,
        y=-np.asarray(intercept.y),
        c_reps=-np.asarray(intercept.c_reps),
        c=np.asarray([-intercept.c[0], intercept.c[1]]),
    )
    signed = fit_rg(
        signed_prepared,
        h2_fit1,
        h2_fit2,
        signed_intercept,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
        ldsc_irwls_iters=3,
    )
    np.testing.assert_allclose(signed.gamma_reps, -forward.gamma_reps, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(signed.rg_reps, -forward.rg_reps, rtol=0.0, atol=2e-14)


def test_identical_trait_covariance_has_univariate_weight_fixed_point():
    n_scale = 250.0
    m_annot = np.asarray([500.0])
    ref_ld = np.asarray([0.4, 0.8, 1.0, 1.7, 2.4, 3.1])
    design = n_scale * ref_ld[:, None] / m_annot[0]
    h = 0.32
    response = design[:, 0] * h
    fit = fit_constrained_cov_ldsc_irwls(
        design,
        response,
        ref_ld,
        ref_ld,
        n1_scale=n_scale,
        n2_scale=n_scale,
        m_annot=m_annot,
        h1_total=h,
        h2_total=h,
        intercept=1.0,
        initial_gamma=np.asarray([h]),
        irwls_iters=3,
    )
    np.testing.assert_allclose(fit.gamma_path, h, rtol=0.0, atol=2e-15)
    mu = 1.0 + n_scale * h * np.maximum(ref_ld, 1.0) / m_annot.sum()
    expected_weights = 1.0 / (2.0 * mu * mu * np.maximum(ref_ld, 1.0))
    np.testing.assert_allclose(fit.weights, expected_weights, rtol=2e-15, atol=2e-15)


def test_ldsc_rg_rejects_misordered_delete_plugin_design():
    prepared, h2_fit1, h2_fit2, intercept, m_annot, *_rest = _integrated_fixture()
    wrong_jk = replace(
        prepared.jackknife,
        D=np.asarray(prepared.jackknife.D)[::-1].copy(),
    )
    bad_h2 = SimpleNamespace(
        **{
            **h2_fit1.__dict__,
            "prepared": SimpleNamespace(
                trace_view=prepared.trace_view,
                jackknife=wrong_jk,
                active_mask=prepared.active_mask,
            ),
        }
    )
    with pytest.raises(ValueError, match="jackknife deletion design/order"):
        fit_rg(
            prepared,
            bad_h2,
            h2_fit2,
            intercept,
            weight_mode="ldsc",
            ldsc_m_annot=m_annot,
        )


def test_ldsc_rg_reported_points_are_full_replicates_with_matching_denominators():
    prepared, h2_fit1, h2_fit2, intercept, m_annot, *_rest = _integrated_fixture()
    observed = fit_rg(
        prepared,
        h2_fit1,
        h2_fit2,
        intercept,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
    )
    full = prepared.jackknife.nrep
    np.testing.assert_allclose(observed.gamma[:, 0], observed.gamma_reps[full])
    np.testing.assert_allclose(observed.rg[:, 0], observed.rg_reps[full])
    np.testing.assert_allclose(
        observed.rg_reps,
        observed.gamma_reps
        / np.sqrt(h2_fit1.sigma_reps[:, :2] * h2_fit2.sigma_reps[:, :2]),
    )
    np.testing.assert_allclose(
        observed.rg_total[0],
        observed.gamma_reps[full].sum()
        / np.sqrt(h2_fit1.h2_reps[full, -1] * h2_fit2.h2_reps[full, -1]),
    )


def _write_sumcore_ldsc_fixture(tmp_path):
    """Write a small, nondegenerate score-scale rg fixture to real input files."""
    m = 32
    idx = np.arange(m, dtype=np.float64)
    chromosome = np.repeat(np.arange(1, 5), m // 4)
    within_chr = np.tile(np.arange(m // 4), 4)
    bp = 100_000 + 2_000 * within_chr
    snps = np.asarray(
        [f"rs{chrom}_{pos}" for chrom, pos in zip(chromosome, bp)]
    )

    # One-component reference LD with enough nonlinear variation that the
    # fitted intercept changes across chromosome deletes.
    ld = 0.55 + 0.085 * idx + 0.14 * (idx % 4) + 0.06 * np.sin(0.7 * idx)
    annot = pd.DataFrame(
        {
            "CHR": chromosome,
            "BP": bp,
            "SNP": snps,
            "CM": np.zeros(m),
            "Base": np.ones(m),
        }
    )
    annot_path = tmp_path / "fixture.annot.tsv"
    annot.to_csv(annot_path, sep="\t", index=False)

    ld_path = tmp_path / "fixture.ldscore.tsv"
    pd.DataFrame(
        {
            "CHR": chromosome,
            "BP": bp,
            "SNP": snps,
            "CM": np.zeros(m),
            "Base": ld,
        }
    ).to_csv(ld_path, sep="\t", index=False)

    # Deliberately store the scalar weight LD in reverse order. Sumcore must
    # align it by SNP (and validate CHR/BP), not assume positional agreement.
    weight_path = tmp_path / "fixture.weights.ldscore.tsv"
    pd.DataFrame(
        {
            "CHR": chromosome,
            "BP": bp,
            "SNP": snps,
            "wLD": 0.45 + 0.23 * (idx % 5) + 0.025 * idx,
        }
    ).iloc[::-1].to_csv(weight_path, sep="\t", index=False)

    def score_z_to_beta(z_star, n_obs, se):
        # Invert SUMMIT's exact score transformation
        # z*^2 = (N-1)b^2 / {b^2 + (N-2)se^2} (cov_rank=0).
        z_star = np.asarray(z_star, dtype=np.float64)
        n_star = float(n_obs - 1)
        nu = float(n_obs - 2)
        z2 = z_star * z_star
        return np.sign(z_star) * float(se) * np.sqrt(z2 * nu / (n_star - z2))

    def write_trait(name, *, n_obs, h2, perturbation, sign=1.0):
        n_scale = float(n_obs - 1)
        score2 = 1.0 + n_scale * ld * float(h2) / float(m) + perturbation
        assert np.all(score2 > 0.0)
        z_star = float(sign) * np.sqrt(score2)
        se = 0.04
        path = tmp_path / f"{name}.tsv"
        pd.DataFrame(
            {
                "SNP": snps,
                "A1": np.repeat("A", m),
                "A2": np.repeat("G", m),
                "N": np.repeat(n_obs, m),
                "BETA": score_z_to_beta(z_star, n_obs, se),
                "SE": np.repeat(se, m),
            }
        ).to_csv(path, sep="\t", index=False)
        return path

    trait1 = write_trait(
        "trait1",
        n_obs=900,
        h2=0.24,
        perturbation=0.10 * np.sin(0.9 * idx) + 0.025 * (idx % 3),
    )
    trait2 = write_trait(
        "trait2",
        n_obs=1150,
        h2=0.36,
        perturbation=0.08 * np.cos(0.55 * idx) - 0.02 * (idx % 4),
    )
    trait2_negative = write_trait(
        "trait2_negative",
        n_obs=1150,
        h2=0.36,
        perturbation=0.08 * np.cos(0.55 * idx) - 0.02 * (idx % 4),
        sign=-1.0,
    )
    return ld_path, annot_path, weight_path, trait1, trait2, trait2_negative


def test_sumcore_ldsc_rg_uses_fitted_summit_intercept_delete_refits(tmp_path):
    (
        ld_path,
        annot_path,
        weight_path,
        trait1,
        trait2,
        trait2_negative,
    ) = _write_sumcore_ldsc_fixture(tmp_path)

    def run(first, second):
        return Sumcore(
            rg=f"{first},{second}",
            ldscores=str(ld_path),
            ldscores_w=str(weight_path),
            annot=str(annot_path),
            njack="chr",
            chisq_threshold=None,
            intercept_chisq_thr=None,
            align_alleles=True,
            weight_mode="ldsc",
            ldsc_irwls_iters=3,
            ldsc_irwls_tol=0.0,
        )._run()

    forward = run(trait1, trait2)
    intercept = forward["intercept"]
    rg_fit = forward["rg_fit"]
    h2_fit1 = forward["h2_fit1"]
    h2_fit2 = forward["h2_fit2"]

    # --weight-mode ldsc changes h2/covariance weighting, not SUMMIT's
    # intercept/refit semantics. No external InterceptFit was supplied here.
    assert intercept.info.get("fixed", False) is False
    assert intercept.info["weight_mode"] == "score"
    assert rg_fit.weight_info["intercept_fixed"] is False
    assert rg_fit.weight_info["intercept_replicates"] == "summit_delete_refits"
    assert rg_fit.weight_info["h2_plugins"] == "matching_ldsc_delete_refits"
    assert rg_fit.weight_info["jackknife_weights"] == "refit_irwls_per_replicate"
    assert h2_fit1.weight_info["jackknife_weights"] == "refit_irwls_per_replicate"
    assert h2_fit2.weight_info["jackknife_weights"] == "refit_irwls_per_replicate"
    assert np.ptp(intercept.c_reps) > 1e-8

    for value in (
        intercept.c,
        intercept.c_reps,
        h2_fit1.h2,
        h2_fit2.h2,
        rg_fit.gamma,
        rg_fit.rg,
        rg_fit.gamma_total,
        rg_fit.rg_total,
        rg_fit.gamma_reps,
        rg_fit.rg_reps,
    ):
        assert np.isfinite(value).all()

    # Independently reconstruct every full/delete covariance IRWLS fit from
    # the objects produced by Sumcore. This catches use of a full-sample c or
    # h2 plug-in in a delete replicate.
    prepared = rg_fit.prepared
    jk = prepared.jackknife
    ldscores = np.asarray(prepared.trace_view.ldscores, dtype=np.float64)
    weight_ld = np.asarray(
        prepared.trace_view.ldscores_reg_w[:, 0], dtype=np.float64
    )
    m_annot = np.asarray(rg_fit.weight_info["m_annot"], dtype=np.float64)
    design = (
        np.sqrt(prepared.n1_scale * prepared.n2_scale)
        * ldscores
        / m_annot[None, :]
    )
    total_ld = ldscores.sum(axis=1)
    for r in range(jk.nrep + 1):
        keep = (
            np.asarray(prepared.active_mask, dtype=bool)
            if r == jk.nrep
            else (
                np.asarray(prepared.active_mask, dtype=bool)
                & (jk.D[r, jk.unit_id] < 0.5)
            )
        )
        direct = fit_constrained_cov_ldsc_irwls(
            design,
            prepared.y - intercept.c_reps[r],
            total_ld,
            weight_ld,
            n1_scale=prepared.n1_scale,
            n2_scale=prepared.n2_scale,
            m_annot=m_annot,
            h1_total=h2_fit1.h2_reps[r, -1],
            h2_total=h2_fit2.h2_reps[r, -1],
            intercept=intercept.c_reps[r],
            keep=keep,
            irwls_iters=3,
            irwls_tol=0.0,
        )
        np.testing.assert_allclose(
            rg_fit.gamma_reps[r], direct.gamma, rtol=3e-13, atol=3e-13
        )

    swapped = run(trait2, trait1)
    np.testing.assert_allclose(
        swapped["intercept"].c_reps, intercept.c_reps, rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        swapped["rg_fit"].gamma_reps, rg_fit.gamma_reps, rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        swapped["rg_fit"].rg_reps, rg_fit.rg_reps, rtol=0.0, atol=2e-14
    )

    signed = run(trait1, trait2_negative)
    np.testing.assert_allclose(
        signed["intercept"].c_reps, -intercept.c_reps, rtol=0.0, atol=3e-14
    )
    np.testing.assert_allclose(
        signed["rg_fit"].gamma_reps, -rg_fit.gamma_reps, rtol=0.0, atol=3e-14
    )
    np.testing.assert_allclose(
        signed["rg_fit"].rg_reps, -rg_fit.rg_reps, rtol=0.0, atol=3e-14
    )


def test_sumcore_ldsc_rg_accepts_stricter_intercept_only_snp_filter(tmp_path):
    ld_path, annot_path, weight_path, trait1, trait2, _ = (
        _write_sumcore_ldsc_fixture(tmp_path)
    )
    result = Sumcore(
        rg=f"{trait1},{trait2}",
        ldscores=str(ld_path),
        ldscores_w=str(weight_path),
        annot=str(annot_path),
        njack="chr",
        chisq_threshold=None,
        intercept_chisq_thr=20.0,
        align_alleles=True,
        weight_mode="ldsc",
        ldsc_irwls_iters=3,
    )._run()
    intercept_active = np.asarray(result["intercept"].active_mask, dtype=bool)
    covariance_active = np.asarray(
        result["rg_fit"].prepared.active_mask, dtype=bool
    )
    assert np.sum(intercept_active) < np.sum(covariance_active)
    assert np.isfinite(result["intercept"].c_reps).all()
    assert np.isfinite(result["rg_fit"].gamma_reps).all()
