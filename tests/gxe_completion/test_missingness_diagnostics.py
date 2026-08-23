from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.logger import Logger


def _missing_inputs(root: Path) -> tuple[Path, Path, np.ndarray]:
    rng = np.random.default_rng(7719)
    n, m = 40, 8
    environment_values = np.linspace(-2.0, 2.0, n)
    genotype = rng.binomial(2, rng.uniform(0.18, 0.42, m), size=(n, m)).astype(float)
    missing = environment_values > 0.0
    genotype[missing, 0] = np.nan
    genotype[[1, 7], 1] = np.nan
    prefix = root / "missing-genotype"
    to_bed(str(prefix) + ".bed", genotype)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    environment = root / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "E": environment_values,
        }
    ).to_csv(environment, sep="\t", index=False)
    return prefix, environment, missing


def _estimator(
    prefix: Path, environment: Path, output: Path, backend: str
) -> GenomewideEnvLDScore:
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        annot_path=None,
        out_path=str(output),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=5,
        step_size=8,
        seed=11,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend=backend,
    )


def test_mean_imputation_retains_call_rate_and_environment_association(
    tmp_path: Path,
) -> None:
    prefix, environment, missing = _missing_inputs(tmp_path)
    estimator = _estimator(prefix, environment, tmp_path / "python", "python")
    try:
        genotype = estimator._read_genotype_block(0, estimator.nsnps)
        # Mean imputation occurs before centering, so imputed entries are zero
        # on the centered/scaled genotype feature.
        np.testing.assert_array_equal(genotype[missing, 0], 0.0)
        estimator.inv_sqrt_resvar_x_all, estimator.inv_sqrt_resvar_w_all = (
            estimator._precompute_residual_variances()
        )
        diagnostics = estimator.feature_diagnostics
        assert diagnostics["genotype_missing_calls"] == int(missing.sum()) + 2
        assert diagnostics["genotype_variants_with_missing_calls"] == 2
        assert diagnostics["minimum_genotype_call_rate"] == 0.5
        assert diagnostics["maximum_missing_environment_correlation"] > 0.8
        assert diagnostics["maximum_missing_phenotype_correlation"] == 0.0
        assert diagnostics["missingness_warning"] is True
        assert diagnostics["mean_imputation_validity"] == "requires_sensitivity_analysis"
    finally:
        estimator.close()


def test_direct_backend_falls_back_for_diagnostic_missingness(
    tmp_path: Path,
) -> None:
    prefix, environment, _ = _missing_inputs(tmp_path)
    estimator = _estimator(prefix, environment, tmp_path / "direct", "direct")
    try:
        estimator.inv_sqrt_resvar_x_all, estimator.inv_sqrt_resvar_w_all = (
            estimator._precompute_residual_variances()
        )
        assert estimator.native_backend == "python"
        assert estimator.feature_diagnostics["genotype_missing_calls"] > 0
        assert estimator.feature_diagnostics["missingness_warning"] is True
    finally:
        estimator.close()
