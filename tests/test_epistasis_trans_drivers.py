"""Final real-block driver's full and nested paths, using bounded donor fixtures."""
from argparse import Namespace
import os
import numpy as np
import pandas as pd
import pytest
from summit.prediction.spec import VariantAxis


@pytest.mark.skipif(
    os.environ.get("OMP_NUM_THREADS", "1") != "1",
    reason="single-thread validation driver",
)
def test_trans_complete_retraining_and_nested_population(tmp_path, monkeypatch):
    from scripts.epistasis import trans_validation
    from epistasis_helpers import cli
    monkeypatch.setattr(trans_validation,'cli',cli)

    rng = np.random.default_rng(485917)
    ma, mb = 32, 96
    pools = [rng.binomial(2, 0.35, (1200, k)).astype(float) for k in (ma, mb)]
    ids = (
        ("12:66358347",)
        + tuple(f"local{i}" for i in range(ma - 1))
        + tuple(f"distal{i}" for i in range(mb))
    )
    axis = VariantAxis(
        ids,
        ("12",) * ma + ("7",) * mb,
        tuple(range(1, ma + mb + 1)),
        ("C",) * (ma + mb),
        ("T",) * (ma + mb),
        genome_build="GRCh37",
    )
    monkeypatch.setattr(
        trans_validation,
        "population_blocks",
        lambda *args: (
            pools,
            axis,
            {"source_identity": "synthetic authentic-axis fixture", "blocks": []},
        ),
    )
    args = Namespace(
        out=tmp_path / "output",
        scratch=tmp_path,
        genotypes=tmp_path / "not_read.bed",
        seed=941672,
        genotype_seed=17,
        background_chromosome="7",
        replicates=1,
        training_samples=128,
        test_samples=256,
        nested_models=1,
        nested_draws=2,
        omit_causal_main=True,
    )
    trans_validation.run(args)
    full = pd.read_csv(args.out / "replicates.csv")
    nested = pd.read_csv(args.out / "population_resampling.csv")
    assert len(full) == 16 and not full.failed.any()
    assert len(nested) == 4 and not nested.failed.any()
    assert np.isfinite(full[["beta0", "se0", "truth0", "p"]]).all().all()
