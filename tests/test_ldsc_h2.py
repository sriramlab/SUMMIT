from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from summit.inference.h2core import fit_h2, prepare_h2
from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
from summit.inference.ldsc_h2 import (
    fit_constrained_ldsc_irwls,
    read_ldsc_weight_ld_aligned,
    resolve_ldsc_reference_moments,
)
from summit.inference.trace import TraceView
from summit.sumstats.sumstats import MatchedSumstats


def _manual_irwls(
    design,
    q,
    ref_ld_total,
    weight_ld,
    *,
    n_scale,
    m_annot,
    keep=None,
    iters=3,
):
    """Independent transcription of the documented constrained-LDSC equations."""
    design = np.asarray(design, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ref_ld_total = np.asarray(ref_ld_total, dtype=np.float64)
    weight_ld = np.asarray(weight_ld, dtype=np.float64)
    m_annot = np.asarray(m_annot, dtype=np.float64)
    if keep is None:
        keep = np.ones(q.size, dtype=bool)
    else:
        keep = np.asarray(keep, dtype=bool)

    m_total = float(m_annot.sum())
    if design.shape[1] == 1:
        h_total = m_total * q[keep].sum() / (float(n_scale) * ref_ld_total[keep].sum())
        h = np.asarray([h_total], dtype=np.float64)
    else:
        h = np.linalg.lstsq(design[keep], q[keep], rcond=None)[0]
    path = [h.copy()]

    final_lhs = None
    final_rhs = None
    final_weights = None
    for _ in range(int(iters)):
        h_for_weights = float(np.clip(h.sum(), 0.0, 1.0))
        mean = 1.0 + h_for_weights * (float(n_scale) / m_total) * np.maximum(ref_ld_total, 1.0)
        mean = np.maximum(mean, 1e-3)
        weights = 1.0 / (2.0 * mean**2 * np.maximum(weight_ld, 1.0))

        x = design[keep]
        w = weights[keep]
        y = q[keep]
        lhs = x.T @ (w[:, None] * x)
        rhs = x.T @ (w * y)
        h = np.linalg.solve(lhs, rhs)

        final_lhs = lhs
        final_rhs = rhs
        final_weights = weights
        path.append(h.copy())

    return {
        "h": h,
        "lhs": final_lhs,
        "rhs": final_rhs,
        "weights": final_weights,
        "path": np.asarray(path),
    }


@pytest.mark.parametrize("iters", [1, 3])
def test_k1_matches_manual_irwls_with_separate_weight_ld_and_unfloored_design(iters):
    design = np.asarray([[0.05], [0.20], [0.55], [1.40], [2.20], [3.10]])
    q = np.asarray([0.15, 0.32, 0.05, 0.70, 1.20, 0.90])
    ref_ld_total = np.asarray([0.20, 0.80, 1.30, 2.40, 3.20, 1.80])
    # Deliberately distinct from ref_ld_total. Values below one also exercise the
    # overcounting-LD floor without conflating it with the regression design.
    weight_ld = np.asarray([6.0, 0.1, 3.5, 1.2, 8.0, 2.2])
    m_annot = np.asarray([80.0])
    n_scale = 120.0

    expected = _manual_irwls(
        design,
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=m_annot,
        iters=iters,
    )
    observed = fit_constrained_ldsc_irwls(
        design,
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=m_annot,
        irwls_iters=iters,
    )

    np.testing.assert_allclose(observed.h, expected["h"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.lhs, expected["lhs"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.rhs, expected["rhs"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.weights, expected["weights"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.h_path, expected["path"], rtol=2e-14, atol=2e-14)
    assert observed.n_iter == iters

    # These alternatives are intentionally wrong. They ensure that the test is
    # sensitive both to substituting total/reference LD for wLD and to flooring X.
    wrong_wld = _manual_irwls(
        design,
        q,
        ref_ld_total,
        ref_ld_total,
        n_scale=n_scale,
        m_annot=m_annot,
        iters=iters,
    )["h"]
    floored_design = _manual_irwls(
        np.maximum(design, 1.0),
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=m_annot,
        iters=iters,
    )["h"]
    assert not np.allclose(observed.h, wrong_wld, rtol=1e-5, atol=1e-8)
    assert not np.allclose(observed.h, floored_design, rtol=1e-5, atol=1e-8)


def test_kgt1_uses_experimental_unweighted_initialization_and_matches_manual_updates():
    design = np.asarray(
        [
            [0.20, 1.10],
            [0.60, 0.40],
            [1.40, 0.30],
            [0.30, 2.00],
            [2.00, 0.70],
            [0.80, 1.30],
        ]
    )
    q = design @ np.asarray([0.15, 0.08]) + np.asarray([-0.02, 0.03, 0.01, -0.01, 0.02, -0.03])
    ref_ld_total = np.asarray([0.50, 1.20, 1.70, 2.10, 2.80, 1.40])
    weight_ld = np.asarray([2.0, 4.0, 1.5, 7.0, 3.0, 5.0])
    m_annot = np.asarray([120.0, 280.0])
    n_scale = 150.0

    expected = _manual_irwls(
        design,
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=m_annot,
        iters=3,
    )
    observed = fit_constrained_ldsc_irwls(
        design,
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=m_annot,
        irwls_iters=3,
    )

    expected_h0 = np.linalg.lstsq(design, q, rcond=None)[0]
    np.testing.assert_allclose(observed.h_path[0], expected_h0, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.h_path, expected["path"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.h, expected["h"], rtol=2e-14, atol=2e-14)


def test_reference_m_is_fixed_when_the_fitted_snp_axis_is_filtered():
    design = np.asarray([[0.25], [0.60], [1.10], [1.70], [2.40], [3.20]])
    q = np.asarray([0.10, 0.40, 0.30, 0.80, 1.10, 0.70])
    ref_ld_total = np.asarray([0.40, 0.90, 1.30, 2.00, 2.80, 3.50])
    weight_ld = np.asarray([1.2, 2.0, 3.0, 4.0, 5.0, 6.0])
    keep = np.asarray([True, False, True, False, True, False])
    n_scale = 90.0
    fixed_reference_m = np.asarray([240.0])

    observed = fit_constrained_ldsc_irwls(
        design,
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=fixed_reference_m,
        keep=keep,
        irwls_iters=3,
    )
    expected = _manual_irwls(
        design,
        q,
        ref_ld_total,
        weight_ld,
        n_scale=n_scale,
        m_annot=fixed_reference_m,
        keep=keep,
        iters=3,
    )

    expected_h0 = fixed_reference_m.sum() * q[keep].sum() / (n_scale * ref_ld_total[keep].sum())
    np.testing.assert_allclose(observed.h_path[0, 0], expected_h0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(observed.h_path, expected["path"], rtol=2e-14, atol=2e-14)

    # A delete-specific M would be a different estimand. Keep an explicit guard
    # against accidentally replacing the fixed reference mass with retained count.
    wrong_delete_specific_h0 = keep.sum() * q[keep].sum() / (n_scale * ref_ld_total[keep].sum())
    assert not np.isclose(observed.h_path[0, 0], wrong_delete_specific_h0)


def _make_prepared_fixture():
    m = 12
    idx = np.arange(m, dtype=np.float64)
    snps = np.asarray([f"rs{i + 1}" for i in range(m)])
    annot = np.column_stack(
        [
            np.ones(m),
            ((np.arange(m) % 3) != 0).astype(np.float64),
        ]
    )
    ldscores = np.column_stack(
        [
            0.55 + 0.11 * idx,
            0.25 + 0.035 * idx + 0.12 * (np.arange(m) % 4),
        ]
    )
    weight_ld = 0.65 + 0.31 * (np.arange(m) % 5) + 0.04 * idx
    n_scale = 80.0
    m_annot = annot.sum(axis=0, dtype=np.float64)
    design = n_scale * ldscores / m_annot[None, :]
    q = design @ np.asarray([0.055, 0.018]) + 0.025 * np.sin(idx)
    y = q + 1.0

    trace = TraceView(
        snps=snps,
        chr=np.repeat(np.arange(1, 4), 4),
        bp=np.arange(m, dtype=np.int64) * 1000 + 100_000,
        annot=annot,
        annot_header=np.asarray(["base", "shared"]),
        ldscores=ldscores,
        ldscores_reg_w=weight_ld[:, None],
    )
    matched = MatchedSumstats(
        snps=snps,
        z=np.sqrt(y),
        chi2=y.copy(),
        beta=np.zeros(m),
        se=np.ones(m),
        n=np.full(m, n_scale + 1.0),
        a1=np.repeat("A", m),
        a2=np.repeat("G", m),
        nsamp=n_scale + 1.0,
        n_scale=n_scale,
        cov_rank=0,
        cov_rank_source="test",
        name="synthetic",
    )
    jackknife = JackknifeDesign.from_trace_view(trace, JackknifeSpec.parse(3))
    prepared = prepare_h2(
        trace,
        matched,
        jackknife,
        summary_y=y,
        summary_y_info={"mode": "test", "n_scale": n_scale, "n_nonfinite": 0},
    )
    overlap = annot.T @ annot
    return prepared, m_annot, overlap, design, q, ldscores.sum(axis=1), weight_ld


def _fit_integrated_ldsc(prepared, m_annot, overlap, *, enrich_mode="non-overlap"):
    return fit_h2(
        prepared,
        weight_mode="ldsc",
        ldsc_m_annot=m_annot,
        ldsc_overlap_matrix=overlap,
        ldsc_source_nsnps=prepared.trace_view.nsnps,
        ldsc_irwls_iters=3,
        ldsc_irwls_tol=0.0,
        enrich_mode=enrich_mode,
        report_tau=True,
        allow_neg_enr=True,
    )


def test_each_delete_block_is_an_exact_irwls_refit():
    prepared, m_annot, overlap, design, q, ref_ld_total, weight_ld = _make_prepared_fixture()
    observed = _fit_integrated_ldsc(prepared, m_annot, overlap, enrich_mode="non-overlap")
    jk = prepared.jackknife
    valid = np.asarray(prepared.active_mask, dtype=bool)

    for r in range(jk.nrep + 1):
        keep = valid if r == jk.nrep else (valid & (jk.D[r, jk.unit_id] < 0.5))
        direct = fit_constrained_ldsc_irwls(
            design,
            q,
            ref_ld_total,
            weight_ld,
            n_scale=prepared.n_scale,
            m_annot=m_annot,
            keep=keep,
            irwls_iters=3,
        )
        np.testing.assert_allclose(observed.sigma_reps[r, :2], direct.h, rtol=2e-13, atol=2e-13)
        expected_overlap = overlap @ (direct.h / m_annot)
        np.testing.assert_allclose(observed.h2_reps[r, :2], expected_overlap, rtol=2e-13, atol=2e-13)

    assert observed.weight_mode == "ldsc"
    assert observed.weight_info["jackknife_weights"] == "refit_irwls_per_replicate"
    assert observed.weight_info["n_failed_replicates"] == 0


def test_overlapping_annotation_h2_and_enrichment_mapping():
    prepared, m_annot, overlap, _design, _q, _ref_ld_total, _weight_ld = _make_prepared_fixture()
    observed = _fit_integrated_ldsc(prepared, m_annot, overlap, enrich_mode="overlap")

    raw_h = observed.sigma_reps[:, :2]
    expected_tau = raw_h / m_annot[None, :]
    expected_h2_overlap = np.einsum("kl,rl->rk", overlap, expected_tau)
    expected_total = raw_h.sum(axis=1)
    prop = m_annot / float(prepared.trace_view.nsnps)
    expected_enrichment = (expected_h2_overlap / expected_total[:, None]) / prop[None, :]

    np.testing.assert_allclose(observed.tau_reps, expected_tau, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.h2_reps[:, :2], expected_h2_overlap, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.h2_reps[:, -1], expected_total, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(observed.enrich_reps, expected_enrichment, rtol=2e-14, atol=2e-14)
    assert observed.enrich_mode_used == "overlap"


def test_he_remains_the_default_and_matches_explicit_he_numerically():
    prepared, _m_annot, _overlap, _design, _q, _ref_ld_total, _weight_ld = _make_prepared_fixture()
    default = fit_h2(
        prepared,
        enrich_mode="both",
        report_tau=True,
        allow_neg_enr=True,
    )
    explicit = fit_h2(
        prepared,
        weight_mode="he",
        enrich_mode="both",
        report_tau=True,
        allow_neg_enr=True,
    )

    array_fields = (
        "sigma_reps",
        "h2_reps",
        "enrich_reps",
        "enrich_overlap_reps",
        "enrich_nonoverlap_reps",
        "tau_reps",
        "tau_star_reps",
        "sigmas",
        "h2",
        "enrich",
        "enrich_overlap",
        "enrich_nonoverlap",
        "tau",
        "tau_star",
    )
    for name in array_fields:
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))

    assert default.weight_mode == explicit.weight_mode == "he"
    assert default.weight_info is explicit.weight_info is None
    assert default.sigma_reps.shape == (prepared.jackknife.nrep + 1, prepared.trace_view.nbins + 2)
    assert default.h2_reps.shape == (prepared.jackknife.nrep + 1, prepared.trace_view.nbins + 1)


def test_nonfinite_response_is_allowed_outside_keep():
    design = np.asarray([[0.5], [1.0], [1.5], [2.0]])
    q = np.asarray([0.1, np.nan, 0.4, 0.7])
    keep = np.asarray([True, False, True, True])
    observed = fit_constrained_ldsc_irwls(
        design,
        q,
        ref_ld_total=np.asarray([0.5, 1.0, 1.5, 2.0]),
        weight_ld=np.asarray([1.2, -0.3, 2.0, 3.0]),
        n_scale=100.0,
        m_annot=np.asarray([50.0]),
        keep=keep,
        irwls_iters=3,
    )
    direct = fit_constrained_ldsc_irwls(
        design[keep],
        q[keep],
        ref_ld_total=np.asarray([0.5, 1.5, 2.0]),
        weight_ld=np.asarray([1.2, 2.0, 3.0]),
        n_scale=100.0,
        m_annot=np.asarray([50.0]),
        irwls_iters=3,
    )
    np.testing.assert_allclose(observed.h, direct.h, rtol=2e-14, atol=2e-14)


def test_near_collinear_design_uses_direct_svd_wls():
    x = np.linspace(0.2, 3.0, 40)
    design = np.column_stack([x, x + 1e-7 * np.sin(np.arange(x.size))])
    q = design @ np.asarray([0.4, -0.2]) + 1e-8 * np.cos(np.arange(x.size))
    ref = 0.8 + x
    wld = 1.3 + 0.2 * x
    initial = np.asarray([0.05, 0.03])

    observed = fit_constrained_ldsc_irwls(
        design,
        q,
        ref,
        wld,
        n_scale=200.0,
        m_annot=np.asarray([100.0, 120.0]),
        initial_h=initial,
        irwls_iters=1,
    )
    h_for_weights = np.clip(initial.sum(), 0.0, 1.0)
    mean = 1.0 + h_for_weights * (200.0 / 220.0) * np.maximum(ref, 1.0)
    weights = 1.0 / (2.0 * mean**2 * np.maximum(wld, 1.0))
    expected = np.linalg.lstsq(
        np.sqrt(weights)[:, None] * design,
        np.sqrt(weights) * q,
        rcond=None,
    )[0]
    np.testing.assert_allclose(observed.h, expected, rtol=1e-11, atol=1e-9)
    assert observed.rank == 2
    assert observed.condition_number > 1e6


def test_weight_ld_alignment_retains_finite_values_below_one(tmp_path):
    frame = pd.DataFrame(
        {
            "CHR": [1, 1, 1],
            "SNP": ["rs3", "rs1", "rs_extra"],
            "BP": [300, 100, 999],
            "L2": [-0.4, 0.7, 2.0],
        }
    )
    path = tmp_path / "weights.ldscore.tsv"
    frame.to_csv(path, sep="\t", index=False)
    values, present = read_ldsc_weight_ld_aligned(path, ["rs1", "rs2", "rs3"])
    np.testing.assert_array_equal(present, [True, False, True])
    assert values[0, 0] == pytest.approx(0.7)
    assert np.isnan(values[1, 0])
    assert values[2, 0] == pytest.approx(-0.4)


def test_weight_ld_alignment_rejects_coordinate_mismatch(tmp_path):
    frame = pd.DataFrame(
        {
            "CHR": [1, 2],
            "SNP": ["rs1", "rs2"],
            "BP": [100, 999],
            "L2": [1.2, 1.4],
        }
    )
    path = tmp_path / "bad_coordinates.ldscore.tsv"
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="CHR/BP mismatch"):
        read_ldsc_weight_ld_aligned(
            path,
            ["rs1", "rs2"],
            target_chr=[1, 2],
            target_bp=[100, 200],
        )


def test_reference_moments_use_full_annotation_and_validate_m(tmp_path):
    annot = pd.DataFrame(
        {
            "CHR": [1, 1, 2, 2],
            "BP": [100, 200, 100, 200],
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "CM": [0.0, 0.0, 0.0, 0.0],
            "base": [1.0, 1.0, 1.0, 1.0],
            "cat": [0.0, 1.0, 1.0, 0.0],
        }
    )
    path = tmp_path / "full.annot.tsv"
    annot.to_csv(path, sep="\t", index=False)
    # The fitted regression axis contains only a subset; reference moments must
    # still come from all four effect-reference rows.
    trace_a = annot.loc[[0, 2], ["base", "cat"]].to_numpy()
    moments = resolve_ldsc_reference_moments(
        annot_path=path,
        trace_annot=trace_a,
        trace_header=np.asarray(["base", "cat"]),
        m_override=np.asarray([4.0, 2.0]),
    )
    np.testing.assert_array_equal(moments.m_annot, [4.0, 2.0])
    np.testing.assert_array_equal(moments.overlap_matrix, [[4.0, 2.0], [2.0, 2.0]])
    assert moments.source_nsnps == 4

    with pytest.raises(ValueError, match="different effect-SNP universes"):
        resolve_ldsc_reference_moments(
            annot_path=path,
            trace_annot=trace_a,
            trace_header=np.asarray(["base", "cat"]),
            m_override=np.asarray([3.0, 2.0]),
        )
