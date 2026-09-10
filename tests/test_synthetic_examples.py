"""Public examples must generate their inputs without a participant dataset."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import open_bed


ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative):
    spec = spec_from_file_location(name, ROOT / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_example_ids_and_summary_statistics(tmp_path):
    generator = module_at("example_generator", "example/prepare_example_inputs.py")
    output = tmp_path / "synthetic"
    generator.generate(output)
    generator.generate(output)
    with open_bed(str(output / "small.bed"), count_A1=True) as reader:
        assert reader.shape == (generator.SAMPLES, generator.VARIANTS)
        assert all(str(iid).startswith("SIM") for iid in reader.iid)
        calls = reader.read(index=np.s_[:, :2])
    cov = pd.read_csv(output / "small.cov", sep="\t")
    y = pd.read_csv(output / "traits.tsv", sep="\t").trait_a.to_numpy()
    z = np.column_stack((np.ones(len(y)), cov[["cov1", "cov2"]]))
    summary = pd.read_csv(output / "trait_a.sumstats", sep="\t")
    # Compare the exported beta/SE ratio with an independently fitted OLS model.
    for j in range(2):
        design = np.column_stack((z, calls[:, j]))
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ beta
        se = np.sqrt(residual @ residual / (len(y) - design.shape[1])
                     * np.linalg.inv(design.T @ design)[-1, -1])
        np.testing.assert_allclose(summary.BETA[j] / summary.SE[j], beta[-1] / se, rtol=1e-10)
    existing = tmp_path / "other"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        generator.generate(existing)


def test_repository_checker_reports_categories_without_values():
    checker = module_at("content_check", "scripts/check_repository_content.py")
    assert checker.findings("example/input.bed", b"")
    assert checker.findings("notes.md", b"1000001 1000001 0.5\n") == ["numeric participant row"]
    assert checker.findings("notes.md", b"SIM000001 SIM000001 0.5\n") == []
    token = b"ghp_" + b"a" * 36
    assert checker.findings("notes.md", token) == ["access token"]
