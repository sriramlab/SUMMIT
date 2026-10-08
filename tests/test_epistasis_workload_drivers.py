"""Exercise complete final drivers, including a real process restart boundary."""
import json
import sys
import os
import numpy as np
from bed_reader import to_bed


def test_whole_trans_subsample_and_process_checkpoint(tmp_path, monkeypatch):
    from scripts.epistasis import whole_panel
    from epistasis_helpers import epistasis_threads
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS'):
        monkeypatch.setenv(key,str(epistasis_threads()))

    if whole_panel.native_module().build_info().get('private_blas_backend') != 'upstream_blis':
        # A leftover path must not switch child processes away from the
        # backend actually loaded and used by their parent.
        monkeypatch.setenv('SUMMIT_PRIVATE_NATIVE_DIR', str(tmp_path/'unused_private_build'))

    rng = np.random.default_rng(985372)
    n, m = 512, 32
    g = rng.binomial(2, 0.3, (n, m)).astype(float)
    ids = list(map(str, range(n)))
    variants = ["12:66358347"] + [f"v{i}" for i in range(1, m)]
    to_bed(
        tmp_path / "input.bed",
        g,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=variants,
            chromosome=["12"] * 8 + ["1"] * (m - 8),
            bp_position=[66358347 + i for i in range(m)],
            allele_1=["C"] * m,
            allele_2=["T"] * m,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whole_panel",
            "--genotypes",
            str(tmp_path / "input.bed"),
            "--out",
            str(tmp_path / "run"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--samples",
            "384",
            "--training-samples",
            "192",
            "--num-threads",
            os.environ.get("OMP_NUM_THREADS", "1"),
            "--trans",
            "--local-min-cell",
            "5",
            "--interrupt-training",
            "--sampling-model",
            "iid_population_projection",
        ],
    )
    whole_panel.main()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["n"] == 384 and receipt["source_n"] == 512
    assert receipt["interaction_markers"] == 24
    assert receipt["sampling_model"] == "iid_population_projection"
    interruption = json.loads((tmp_path / "run/interruption.json").read_text())
    assert interruption["iteration"] >= 2 and interruption["exit"] == 75
    trained = json.loads((tmp_path / "run/trained/models/manifest.json").read_text())
    assert trained["run_report"]["resumed_solver"]
    assert (tmp_path / "run/summary_only/fit.json").exists()
    from scripts.epistasis import whole_trans_population
    import pandas as pd

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whole_trans_population",
            "--inputs",
            str(tmp_path / "run"),
            "--out",
            str(tmp_path / "population"),
            "--draws",
            "2",
            "--num-threads",
            os.environ.get("OMP_NUM_THREADS", "1"),
            "--sampling",
            "intact_rows",
        ],
    )
    whole_trans_population.main()
    population = pd.read_csv(tmp_path / "population/replicates.csv")
    assert len(population) == 12 and not population.failed.any()
    population_design = json.loads((tmp_path / "population/design.json").read_text())
    assert population_design["genotype_passes_per_replicate"] == 0
    assert population_design["native_score_decomposition_error"] < 1e-10
    from scripts.epistasis.pilot_design_validation import run

    run(
        tmp_path / "run",
        tmp_path / "pilot",
        draws=2,
        reference="confirmed/learned.cohort-reference.npz",
    )
    validation = pd.read_csv(tmp_path / "pilot/replicates.csv")
    assert len(validation) == 6 and not validation.failed.any()
