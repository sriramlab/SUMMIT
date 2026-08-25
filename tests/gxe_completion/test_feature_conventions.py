from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
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
            if mode == "standardized_projected"
            else "hwe"
        ),
        native_backend="python",
    )


@pytest.mark.parametrize(
    ("canonical", "stored"),
    [
        ("standardized_projected", "standardized_projected"),
        ("raw_projected", "raw_projected"),
    ],
)
def test_canonical_feature_modes_select_the_expected_kernel(
    tmp_path: Path, canonical: str, stored: str
) -> None:
    prefix, environment = _inputs(tmp_path)
    estimator = _estimator(prefix, environment, tmp_path / "canonical", canonical)
    try:
        assert estimator.kernel_mode == stored
        assert estimator.feature_convention == canonical
    finally:
        estimator.close()


@pytest.mark.parametrize("alias", ["standardized", "genie"])
def test_development_feature_mode_aliases_are_rejected(
    tmp_path: Path, alias: str
) -> None:
    prefix, environment = _inputs(tmp_path)
    with pytest.raises(ValueError, match="kernel_mode must be"):
        _estimator(prefix, environment, tmp_path / alias, alias)
