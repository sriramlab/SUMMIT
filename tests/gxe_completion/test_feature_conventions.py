from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _validate_feature_convention_metadata,
)
from summit.logger import Logger


def _inputs(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(20260814)
    genotype = rng.binomial(2, rng.uniform(0.15, 0.45, 9), size=(29, 9)).astype(float)
    prefix = root / "genotype"
    to_bed(str(prefix) + ".bed", genotype)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    environment = root / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "E": rng.normal(size=genotype.shape[0]),
        }
    ).to_csv(environment, sep="\t", index=False)
    return prefix, environment


def _estimator(
    prefix: Path, environment: Path, output: Path, mode: str
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
        step_size=9,
        seed=7,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode=mode,
        genotype_scale=(
            "sample"
            if mode in {"standardized", "standardized_projected"}
            else "hwe"
        ),
        native_backend="python",
    )


@pytest.mark.parametrize(
    ("canonical", "legacy", "internal"),
    [
        ("standardized_projected", "standardized", "standardized"),
        ("raw_projected", "genie", "genie"),
    ],
)
def test_canonical_feature_modes_are_exact_aliases(
    tmp_path: Path, canonical: str, legacy: str, internal: str
) -> None:
    prefix, environment = _inputs(tmp_path)
    left = _estimator(prefix, environment, tmp_path / "canonical", canonical)
    right = _estimator(prefix, environment, tmp_path / "legacy", legacy)
    try:
        genotype = left._read_genotype_block(0, left.nsnps)
        left.inv_sqrt_resvar_x_all, left.inv_sqrt_resvar_w_all = (
            left._precompute_residual_variances()
        )
        right.inv_sqrt_resvar_x_all, right.inv_sqrt_resvar_w_all = (
            right._precompute_residual_variances()
        )
        assert left.kernel_mode == right.kernel_mode == internal
        assert left.feature_convention == right.feature_convention == canonical
        for interaction in (False, True):
            method = (
                "_prepare_interaction_block"
                if interaction
                else "_prepare_additive_block"
            )
            np.testing.assert_array_equal(
                getattr(left, method)(0, left.nsnps, G=genotype),
                getattr(right, method)(0, right.nsnps, G=genotype),
            )
    finally:
        left.close()
        right.close()


def test_feature_convention_metadata_is_versioned_and_fails_closed() -> None:
    assert _validate_feature_convention_metadata(
        {
            "kernel_mode": "standardized",
            "feature_convention": "standardized_projected",
            "feature_convention_version": 1,
        }
    ) == "standardized_projected"
    assert _validate_feature_convention_metadata(
        {"kernel_mode": "genie"}
    ) == "raw_projected"
    with pytest.raises(ValueError, match="different GxE reference"):
        _validate_feature_convention_metadata(
            {
                "kernel_mode": "genie",
                "feature_convention": "standardized_projected",
                "feature_convention_version": 1,
            }
        )
    with pytest.raises(ValueError, match="feature_convention_version"):
        _validate_feature_convention_metadata(
            {
                "kernel_mode": "genie",
                "feature_convention": "raw_projected",
                "feature_convention_version": 2,
            }
        )
