from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from summit.inference.sumcore import Sumcore
from summit.manifest.rg_manifest_fast import dispatch_rg_manifest_fast


def _write_table(path, data) -> str:
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return str(path)


def _beta_for_exact_score_z(z_star, *, n: float, se: float) -> np.ndarray:
    """Invert SUMMIT's covariate-free exact score-scale transformation."""
    z_star = np.asarray(z_star, dtype=np.float64)
    n_scale = float(n - 1.0)
    resid_df = float(n - 2.0)
    z2 = z_star * z_star
    return float(se) * z_star * np.sqrt(resid_df / (n_scale - z2))


def test_fast_multimodel_fixed_intercept_matches_separate_sumcore_he_fits(tmp_path):
    m = 72
    idx = np.arange(m, dtype=np.float64)
    chrom = np.repeat(np.arange(1, 5), m // 4)
    bp = 100_000 + 100 * np.arange(m)
    snp = np.asarray([f"rs{i:04d}" for i in range(m)])

    base = np.ones(m, dtype=np.float64)
    immune = ((np.arange(m) % 5) < 3).astype(np.float64)
    neural = (((7 * np.arange(m)) % 11) < 5).astype(np.float64)
    ld_base = 0.85 + 0.015 * idx + 0.08 * (idx % 4)
    ld_immune = 0.20 + 0.55 * immune + 0.025 * (idx % 7)
    ld_neural = 0.25 + 0.60 * neural + 0.020 * ((3 * idx) % 5)

    union_ld = _write_table(
        tmp_path / "union.ldscore",
        {
            "CHR": chrom,
            "BP": bp,
            "SNP": snp,
            "base": ld_base,
            "immune": ld_immune,
            "neural": ld_neural,
        },
    )
    union_annot = _write_table(
        tmp_path / "union.annot",
        {
            "CHR": chrom,
            "BP": bp,
            "SNP": snp,
            "base": base,
            "immune": immune,
            "neural": neural,
        },
    )

    n = 20_000.0
    se = 0.04
    z1 = np.sqrt(1.0 + 5.0 * ld_base + 2.2 * ld_immune + 1.8 * ld_neural)
    z2 = np.sqrt(1.0 + 4.3 * ld_base + 1.7 * ld_immune + 2.4 * ld_neural)
    trait1 = _write_table(
        tmp_path / "trait1.tsv",
        {
            "ID": snp,
            "A1": "A",
            "A2": "C",
            "N": n,
            "BETA": _beta_for_exact_score_z(z1, n=n, se=se),
            "SE": se,
        },
    )
    trait2 = _write_table(
        tmp_path / "trait2.tsv",
        {
            "ID": snp,
            "A1": "A",
            "A2": "C",
            "N": n,
            "BETA": _beta_for_exact_score_z(z2, n=n, se=se),
            "SE": se,
        },
    )

    model_manifest = _write_table(
        tmp_path / "models.tsv",
        {
            "model": ["immune_model", "neural_model"],
            "bins": ["base,immune", "base,neural"],
            "aliases": ["base,focal", "base,focal"],
        },
    )
    fixed_intercept = 0.125
    manifest = pd.DataFrame(
        [
            {
                "row_id": 1,
                "phen1": "trait1",
                "phen2": "trait2",
                "sumstats1": trait1,
                "sumstats2": trait2,
                "intercept_rg": fixed_intercept,
                "cov_rank1": None,
                "cov_rank2": None,
                "out_stem": "trait1__trait2",
            }
        ]
    )
    trait_meta = {
        trait1: {"phen": "trait1", "cov_rank": None},
        trait2: {"phen": "trait2", "cov_rank": None},
    }
    outdir = tmp_path / "fast"
    args = SimpleNamespace(
        adjust_delta=False,
        align_alleles=True,
        allow_neg_enr=False,
        annot=union_annot,
        bim=None,
        chisq_action="none",
        clip_nonfinite_vals=False,
        enrich_mode="auto",
        jack_mode="mean",
        keep_ambiguous=False,
        ldscores=union_ld,
        ldscores_reg=None,
        max_chisq=None,
        njack="chr",
        out=str(outdir),
        rg_se_method="jackknife",
        rg_model_manifest=model_manifest,
        rg_fast_no_pair_logs=True,
        verbose=False,
        write_jack=True,
        write_normeq=False,
    )
    dispatch_rg_manifest_fast(args, None, manifest, trait_meta, 0)

    observed = pd.read_csv(outdir / "manifest.results.tsv", sep="\t").set_index(
        "model"
    )
    assert list(observed.index) == ["immune_model", "neural_model"]
    assert set(observed["estimator"]) == {"he"}

    model_inputs = {
        "immune_model": (immune, ld_immune),
        "neural_model": (neural, ld_neural),
    }
    for model, (focal_annot, focal_ld) in model_inputs.items():
        regular_ld = _write_table(
            tmp_path / f"{model}.ldscore",
            {
                "CHR": chrom,
                "BP": bp,
                "SNP": snp,
                "base": ld_base,
                "focal": focal_ld,
            },
        )
        regular_annot = _write_table(
            tmp_path / f"{model}.annot",
            {
                "CHR": chrom,
                "BP": bp,
                "SNP": snp,
                "base": base,
                "focal": focal_annot,
            },
        )
        direct = Sumcore(
            rg=f"{trait1},{trait2}",
            ldscores=regular_ld,
            annot=regular_annot,
            njack="chr",
            chisq_threshold=None,
            chisq_action="none",
            align_alleles=True,
            intercept_rg=fixed_intercept,
            intercept_rg_source="manifest",
            rg_se_method="jackknife",
            weight_mode="he",
        )._run()

        h2_fit1 = direct["h2_fit1"]
        h2_fit2 = direct["h2_fit2"]
        intercept = direct["intercept"]
        rg_fit = direct["rg_fit"]
        expected = {
            "n_snps": m,
            "h2_trait1": h2_fit1.h2[-1, 0],
            "h2_trait1_se": h2_fit1.h2[-1, 1],
            "h2_trait2": h2_fit2.h2[-1, 0],
            "h2_trait2_se": h2_fit2.h2[-1, 1],
            "intercept_c": intercept.c[0],
            "intercept_c_se": intercept.c[1],
            "gamma_g_total": rg_fit.gamma_total[0],
            "gamma_g_total_se": rg_fit.gamma_total[1],
            "rg_total": rg_fit.rg_total[0],
            "rg_total_se": rg_fit.rg_total[1],
            "gamma_g__base": rg_fit.gamma[0, 0],
            "gamma_g__base_se": rg_fit.gamma[0, 1],
            "rg__base": rg_fit.rg[0, 0],
            "rg__base_se": rg_fit.rg[0, 1],
            "gamma_g__focal": rg_fit.gamma[1, 0],
            "gamma_g__focal_se": rg_fit.gamma[1, 1],
            "rg__focal": rg_fit.rg[1, 0],
            "rg__focal_se": rg_fit.rg[1, 1],
        }
        observed_values = observed.loc[model, list(expected)].to_numpy(dtype=float)
        expected_values = np.asarray(list(expected.values()), dtype=np.float64)
        assert np.isfinite(expected_values).all()
        assert np.isfinite(observed_values).all()
        np.testing.assert_allclose(
            observed_values, expected_values, rtol=3e-11, atol=3e-11
        )

        total_rg_reps = np.sum(rg_fit.gamma_reps, axis=1) / np.sqrt(
            h2_fit1.h2_reps[:, -1] * h2_fit2.h2_reps[:, -1]
        )
        expected_reps = np.column_stack(
            [
                intercept.c_reps,
                rg_fit.gamma_reps,
                np.sum(rg_fit.gamma_reps, axis=1),
                rg_fit.rg_reps,
                total_rg_reps,
            ]
        )
        observed_reps = np.loadtxt(
            str(observed.loc[model, "out_prefix"]) + ".rg.jack",
            comments="#",
            usecols=range(1, 8),
        )
        assert np.isfinite(expected_reps).all()
        assert np.isfinite(observed_reps).all()
        np.testing.assert_allclose(
            observed_reps, expected_reps, rtol=5e-9, atol=5e-10
        )
