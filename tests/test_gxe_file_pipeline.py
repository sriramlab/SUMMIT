from __future__ import annotations

import json
import math
import types

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.inference.gxe import fit_from_files
from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _orthonormalize_columns,
    _validate_plink_bed_shape,
)
from summit.logger import Logger


def test_plink_shape_preflight_rejects_truncated_bed(tmp_path):
    prefix = tmp_path / "shape"
    to_bed(str(prefix) + ".bed", np.zeros((5, 3), dtype=np.float64))
    assert _validate_plink_bed_shape(str(prefix)) == (5, 3)
    bed_path = tmp_path / "shape.bed"
    bed_path.write_bytes(bed_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Invalid or truncated"):
        _validate_plink_bed_shape(str(prefix))


def test_file_bundle_reconstructs_explicit_individual_level_fit(tmp_path):
    rng = np.random.default_rng(8128)
    n, m = 37, 13
    raw = rng.binomial(2, rng.uniform(0.12, 0.44, size=m), size=(n, m)).astype(np.float64)
    prefix = tmp_path / "tiny"
    to_bed(str(prefix) + ".bed", raw)

    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env_raw = rng.normal(size=n)
    cov_raw = rng.normal(size=n)
    y_raw = rng.normal(size=n)
    env_path = tmp_path / "tiny.env"
    cov_path = tmp_path / "tiny.cov"
    pheno_path = tmp_path / "tiny.pheno"
    ids.assign(E=env_raw).to_csv(env_path, sep="\t", index=False)
    ids.assign(C=cov_raw, C_DUP=cov_raw).to_csv(cov_path, sep="\t", index=False)
    ids.assign(Y=y_raw).to_csv(pheno_path, sep="\t", index=False)
    annot = np.column_stack([np.linspace(0.2, 1.0, m), 0.25 + (np.arange(m) % 2) * 0.75])
    annot_path = tmp_path / "tiny.annot"
    pd.DataFrame(
        {
            "CHR": bim[0].astype(str),
            "SNP": bim[1].astype(str),
            "BP": bim[3].astype(int),
            "bin_a": annot[:, 0],
            "bin_b": annot[:, 1],
        }
    ).to_csv(annot_path, sep="\t", index=False)

    out = tmp_path / "bundle"
    obj = GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(env_path),
        annot_path=str(annot_path),
        out_path=str(out),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=str(cov_path),
        pheno_path=str(pheno_path),
        num_vecs=m,
        step_size=m,
        seed=1,
        verbose=False,
        dtype="float64",
        kernel_mode="raw_projected",
        genotype_scale="hwe",
        target_xz_mem=0.01,
    )

    # A complete orthogonal probe panel makes the stochastic score identity
    # exact: (1/V) ZZ' = I.  This is an independent file-level oracle, not a
    # tolerance-based Monte-Carlo test.
    def exact_probes(self, L, v_count, blk_start, v_start):
        assert v_count == m and v_start == 0
        return np.asfortranarray(
            math.sqrt(m) * np.eye(m)[blk_start:blk_start + L]
        )

    obj._generate_random_block = types.MethodType(exact_probes, obj)
    original_read = obj._read_genotype_block
    read_count = 0

    def counted_read(self, start, end):
        nonlocal read_count
        read_count += 1
        return original_read(start, end)

    obj._read_genotype_block = types.MethodType(counted_read, obj)
    obj._compute_ldscore()
    # Feature diagnostics, one global source pass, and one global target pass.
    assert read_count == 3

    fitted, _ = fit_from_files(
        str(out) + ".gxe.ref.json",
        str(out) + ".gxe.moments.json",
        str(out) + ".gxe.gwas.tsv.gz",
        str(out) + ".gxe.gwis.tsv.gz",
        njack=3,
        max_condition=1e16,
    )

    e = (env_raw - env_raw.mean()) / env_raw.std(ddof=1)
    c = (cov_raw - cov_raw.mean()) / cov_raw.std(ddof=1)
    q = _orthonormalize_columns(np.column_stack([c, c, e]))
    qfull = _orthonormalize_columns(np.column_stack([np.ones(n), q]))
    pmat = np.eye(n) - qfull @ qfull.T
    dosage_mean = raw.mean(axis=0)
    hwe_sd = np.sqrt(dosage_mean * (1.0 - 0.5 * dosage_mean))
    x0 = (raw - dosage_mean) / hwe_sd
    x = pmat @ x0
    w = pmat @ (e[:, None] * x0)
    y = pmat @ y_raw
    residual_rank = n - qfull.shape[1]
    y *= np.sqrt(residual_rank / np.dot(y, y))
    masses = annot.sum(axis=0)
    kernels = []
    for col in range(annot.shape[1]):
        kernels.append((x * annot[:, col]) @ x.T / masses[col])
    for col in range(annot.shape[1]):
        kernels.append((w * annot[:, col]) @ w.T / masses[col])
    kernels.extend([pmat @ np.diag(e * e) @ pmat, pmat])
    lhs = np.asarray([[np.trace(a @ b) for b in kernels] for a in kernels])
    rhs = np.asarray([y @ a @ y for a in kernels])
    direct = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(fitted.coefficients, direct, rtol=2e-9, atol=2e-9)
    assert fitted.coefficient_standard_errors is not None
    assert fitted.proportion_standard_errors is not None
    assert fitted.jackknife_coefficients.shape == (3, len(direct))
    assert fitted.jackknife_proportions.shape == (3, len(direct))
    assert fitted.jackknife_block_labels == (
        "block_0001",
        "block_0002",
        "block_0003",
    )

    generated = list(tmp_path.glob("bundle.g*"))
    assert generated
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in generated)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        obj._compute_ldscore()

    ref_path = tmp_path / "bundle.gxe.ref.json"
    moments_path = tmp_path / "bundle.gxe.moments.json"
    gwas_path = tmp_path / "bundle.gxe.gwas.tsv.gz"
    gwis_path = tmp_path / "bundle.gxe.gwis.tsv.gz"
    generated_reference = json.loads(ref_path.read_text())
    assert generated_reference["randomization"]["step_size"] == m
    assert generated_reference["randomization"]["probe_tiles"] == [[0, m]]

    # Equivalent summary files remain portable across paths; compatibility is
    # established from their actual variant axes and statistical metadata.
    copied_gwas = tmp_path / "copied.gwas.tsv.gz"
    copied_gwas.write_bytes(gwas_path.read_bytes())
    copied_fit, _ = fit_from_files(
        ref_path,
        moments_path,
        copied_gwas,
        gwis_path,
        njack=3,
        max_condition=1e16,
    )
    np.testing.assert_array_equal(copied_fit.coefficients, fitted.coefficients)

    different_reference = json.loads(ref_path.read_text())
    different_reference["kernel_mode"] = "standardized_projected"
    different_reference_path = tmp_path / "different.gxe.ref.json"
    different_reference_path.write_text(json.dumps(different_reference))
    with pytest.raises(ValueError, match="feature convention"):
        fit_from_files(
            different_reference_path,
            moments_path,
            gwas_path,
            gwis_path,
            njack=3,
            max_condition=1e16,
        )

    bad_gwas = pd.read_csv(gwas_path, sep=r"\s+")
    bad_gwas["N"] += 1
    bad_gwas_path = tmp_path / "bad.gwas.tsv.gz"
    bad_gwas.to_csv(bad_gwas_path, sep="\t", index=False, compression="gzip")
    with pytest.raises(ValueError, match="N does not match"):
        fit_from_files(
            ref_path,
            moments_path,
            bad_gwas_path,
            gwis_path,
            njack=3,
            max_condition=1e16,
        )

    missing_mode = pd.read_csv(gwas_path, sep=r"\s+")
    missing_mode.loc[0, "SCORE_MODE"] = None
    missing_mode_path = tmp_path / "missing-mode.gwas.tsv.gz"
    missing_mode.to_csv(missing_mode_path, sep="\t", index=False, compression="gzip")
    with pytest.raises(ValueError, match="marginal cross-products"):
        fit_from_files(
            ref_path,
            moments_path,
            missing_mode_path,
            gwis_path,
            njack=3,
            max_condition=1e16,
        )
