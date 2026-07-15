from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pgenlib
import pytest
from bed_reader import to_bed

from summit import gwldcore
from summit.ldscore.genotype_source import (
    PgenBlockReader,
    read_psam_sample_ids,
    read_pvar_variants,
    resolve_genotype_input,
)
from summit.ldscore.gw_ldscore import GenomewideLDScore, read_cov


class _Log:
    def __init__(self):
        self.messages = []

    def _log(self, message):
        self.messages.append(str(message))

    def _save_log(self, path):
        self.saved_path = str(path)


def _write_metadata(prefix: Path, sample_ct: int, variant_ct: int) -> None:
    with prefix.with_suffix(".psam").open("w", encoding="utf-8") as handle:
        handle.write("#FID\tIID\n")
        for sample_idx in range(sample_ct):
            handle.write(f"f{sample_idx}\ti{sample_idx}\n")
    with prefix.with_suffix(".pvar").open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for variant_idx in range(variant_ct):
            handle.write(f"1\t{100 + variant_idx}\trs{variant_idx}\tA\tG\n")


def _write_hardcall_pgen(prefix: Path, alt_counts: np.ndarray) -> None:
    alt_counts = np.ascontiguousarray(alt_counts, dtype=np.int8)
    variant_ct, sample_ct = alt_counts.shape
    with pgenlib.PgenWriter(
        bytes(prefix.with_suffix(".pgen")),
        sample_ct=sample_ct,
        variant_ct=variant_ct,
        nonref_flags=False,
    ) as writer:
        writer.append_biallelic_batch(alt_counts)
    _write_metadata(prefix, sample_ct, variant_ct)


def _write_dosage_pgen(prefix: Path, alt_dosages: np.ndarray) -> None:
    alt_dosages = np.ascontiguousarray(alt_dosages, dtype=np.float32)
    variant_ct, sample_ct = alt_dosages.shape
    with pgenlib.PgenWriter(
        bytes(prefix.with_suffix(".pgen")),
        sample_ct=sample_ct,
        variant_ct=variant_ct,
        nonref_flags=False,
        dosage_present=True,
    ) as writer:
        writer.append_dosages_batch(alt_dosages)
    _write_metadata(prefix, sample_ct, variant_ct)


def _write_matching_bed(prefix: Path, alt_counts: np.ndarray) -> None:
    variant_ct, sample_ct = alt_counts.shape
    values = alt_counts.T.astype(np.float64)
    values[values == -9] = np.nan
    properties = {
        "fid": [f"f{i}" for i in range(sample_ct)],
        "iid": [f"i{i}" for i in range(sample_ct)],
        "father": ["0"] * sample_ct,
        "mother": ["0"] * sample_ct,
        "sex": [0] * sample_ct,
        "pheno": ["-9"] * sample_ct,
        "chromosome": ["1"] * variant_ct,
        "sid": [f"rs{i}" for i in range(variant_ct)],
        "cm_position": [0.0] * variant_ct,
        "bp_position": [100 + i for i in range(variant_ct)],
        "allele_1": ["G"] * variant_ct,
        "allele_2": ["A"] * variant_ct,
    }
    to_bed(prefix.with_suffix(".bed"), values, properties=properties, count_A1=True)


def _hardcalls() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
            [0, -9, 1, 2, -9, 0, 1, 2, 0, 1, -9, 2],
            [0] * 12,
            [2] * 12,
            [-9] * 12,
            [0, 0, 0, 1, 1, 1, 2, 2, 2, -9, 0, 2],
        ],
        dtype=np.int8,
    )


def _numpy_standardize(raw: np.ndarray, ddof: int = 1) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    out = np.zeros_like(raw)
    for variant_idx in range(raw.shape[1]):
        col = raw[:, variant_idx]
        observed = col != -9
        values = col[observed]
        if values.size == 0:
            continue
        mean = float(values.mean())
        centered = values - mean
        denom = values.size - ddof
        inv_sd = 1.0
        if denom > 0:
            variance = float(centered @ centered) / float(denom)
            if variance > 0.0:
                inv_sd = 1.0 / np.sqrt(variance)
        out[observed, variant_idx] = centered * inv_sd
    return out


def _run_gwld(genotype_path: Path, out: Path, covar_path: Path | None = None):
    estimator = GenomewideLDScore(
        bed_path=str(genotype_path),
        annot_path=None,
        out_path=str(out),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        covar_path=None if covar_path is None else str(covar_path),
        num_vecs=512,
        step_size=3,
        seed=9182,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        use_mailman=False,
        impute_method="mean",
    )
    try:
        estimator._compute_ldscore()
        return estimator.gwldscore.copy(), estimator.row_sel
    finally:
        estimator.close()


def test_resolve_requires_explicit_suffix_when_both_trios_exist(tmp_path):
    prefix = tmp_path / "both"
    for suffix in (".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam"):
        prefix.with_suffix(suffix).touch()
    with pytest.raises(ValueError, match="Both BED and PGEN"):
        resolve_genotype_input(str(prefix))
    assert resolve_genotype_input(str(prefix.with_suffix(".bed"))).format == "bed"
    assert resolve_genotype_input(str(prefix.with_suffix(".pgen"))).format == "pgen"


def test_pvar_psam_parsing_and_multiallelic_rejection(tmp_path):
    prefix = tmp_path / "meta"
    _write_metadata(prefix, sample_ct=3, variant_ct=2)
    samples = read_psam_sample_ids(str(prefix.with_suffix(".psam")))
    variants = read_pvar_variants(str(prefix.with_suffix(".pvar")))
    assert samples.to_dict("list") == {
        "FID": ["f0", "f1", "f2"],
        "IID": ["i0", "i1", "i2"],
    }
    assert variants[["SNP", "A1", "A2"]].to_dict("list") == {
        "SNP": ["rs0", "rs1"],
        "A1": ["G", "G"],
        "A2": ["A", "A"],
    }
    text = prefix.with_suffix(".pvar").read_text(encoding="utf-8")
    duplicate_text = text.replace("rs1", "rs0")
    prefix.with_suffix(".pvar").write_text(duplicate_text, encoding="utf-8")
    assert read_pvar_variants(str(prefix.with_suffix(".pvar")))["SNP"].tolist() == [
        "rs0", "rs0"
    ]
    prefix.with_suffix(".pvar").write_text(text.replace("\tG\n", "\tG,T\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Multiallelic"):
        read_pvar_variants(str(prefix.with_suffix(".pvar")))


def test_duplicate_variant_ids_require_positional_annotation_alignment(tmp_path):
    prefix = tmp_path / "duplicate_ids"
    alt = _hardcalls()
    _write_hardcall_pgen(prefix, alt)
    pvar_path = prefix.with_suffix(".pvar")
    pvar_path.write_text(
        pvar_path.read_text(encoding="utf-8").replace("rs1", "rs0"),
        encoding="utf-8",
    )
    annotation = pd.DataFrame({
        "CHR": ["1"] * alt.shape[0],
        "BP": [100 + idx for idx in range(alt.shape[0])],
        "SNP": ["rs0", "rs2", "rs0", "rs3", "rs4", "rs5", "rs6"],
        "CM": [0.0] * alt.shape[0],
        "base": [1] * alt.shape[0],
    })
    annotation_path = tmp_path / "duplicate.annot"
    annotation.to_csv(annotation_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="Cannot reorder annotations"):
        GenomewideLDScore(
            bed_path=str(prefix.with_suffix(".pgen")),
            annot_path=str(annotation_path),
            out_path=str(tmp_path / "out"),
            log=_Log(),
            rand_dist="rademacher",
            low_level={"numa_mode": None},
            num_vecs=8,
            step_size=3,
            seed=1,
            dtype="float64",
            num_threads=1,
            use_mailman=False,
            impute_method="mean",
            ddof=1,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pgen_reader_ref_dosage_subset_partial_block_and_buffer_reuse(tmp_path, dtype):
    prefix = tmp_path / "hard"
    alt = _hardcalls()
    _write_hardcall_pgen(prefix, alt)
    subset = np.asarray([0, 3, 6, 10], dtype=np.int64)
    reader = PgenBlockReader(
        str(prefix.with_suffix(".pgen")),
        raw_sample_ct=alt.shape[1],
        variant_ct=alt.shape[0],
        sample_subset=subset,
        step_size=3,
        dtype=dtype,
        ddof=1,
        standardize_threads=2,
    )
    try:
        first = reader.read_standardized_block(0, 3)
        ptr = first.__array_interface__["data"][0]
        expected_raw = np.where(alt[:3, subset].T == -9, -9, 2 - alt[:3, subset].T)
        np.testing.assert_allclose(first, _numpy_standardize(expected_raw), rtol=2e-6, atol=2e-6)
        final = reader.read_standardized_block(6, 7)
        assert final.shape == (subset.size, 1)
        assert final.flags.f_contiguous
        assert final.__array_interface__["data"][0] == ptr
        assert reader.blocks_read == 2
        assert reader.variants_read == 4
    finally:
        reader.close()


def test_native_standardizer_rejects_corrupt_nonmissing_dosage():
    geno = np.asfortranarray(np.asarray([[0.0], [1.0], [2.01]], dtype=np.float64))
    with pytest.raises(RuntimeError, match="out-of-range"):
        gwldcore.standardize_dosage_inplace(geno, ddof=1, missing_value=-9.0)


def test_covariate_join_preserves_literal_na_sample_id(tmp_path):
    sample_ids = pd.DataFrame({
        "FID": ["NA", "f1", "f2"],
        "IID": ["i0", "i1", "i2"],
    })
    covar = tmp_path / "covar_na_id.tsv"
    covar.write_text(
        "FID\tIID\tage\nNA\ti0\t20\nf1\ti1\t30\nf2\ti2\t40\n",
        encoding="utf-8",
    )
    C, R, keep = read_cov(
        str(covar),
        fam_filename=None,
        sample_ids=sample_ids,
        logger=_Log(),
    )
    np.testing.assert_array_equal(keep, np.arange(3))
    assert C.shape == (3, 1)
    assert R.shape == (1, 3)


def test_hardcall_pgen_and_bed_have_identical_end_to_end_scores_with_covariates(tmp_path):
    alt = _hardcalls()
    pgen_prefix = tmp_path / "hard_pgen"
    bed_prefix = tmp_path / "hard_bed"
    _write_hardcall_pgen(pgen_prefix, alt)
    _write_matching_bed(bed_prefix, alt)

    covar = tmp_path / "covar.tsv"
    cov = pd.DataFrame({
        "FID": [f"f{i}" for i in range(alt.shape[1])],
        "IID": [f"i{i}" for i in range(alt.shape[1])],
        "age": np.arange(alt.shape[1], dtype=float),
        "batch": np.asarray([0, 1] * (alt.shape[1] // 2), dtype=float),
    })
    cov.loc[5, "age"] = np.nan
    cov.to_csv(covar, sep="\t", index=False, na_rep="NA")

    pgen_scores, pgen_rows = _run_gwld(
        pgen_prefix.with_suffix(".pgen"), tmp_path / "pgen_out", covar
    )
    bed_scores, bed_rows = _run_gwld(
        bed_prefix.with_suffix(".bed"), tmp_path / "bed_out", covar
    )
    np.testing.assert_array_equal(pgen_rows, bed_rows)
    np.testing.assert_array_equal(pgen_scores, bed_scores)
    np.testing.assert_array_equal(
        np.atleast_1d(np.loadtxt(tmp_path / "pgen_out.gw.M")),
        np.asarray([alt.shape[0]], dtype=np.float64),
    )


def test_fractional_pgen_estimator_targets_exact_dosage_matrix_ld(tmp_path):
    rng = np.random.default_rng(77)
    alt = rng.uniform(0.0, 2.0, size=(6, 30)).astype(np.float32)
    alt[1, ::7] = -9.0
    alt[2, :] = 0.875
    alt[3, :] = -9.0
    alt[4, :] = np.linspace(0.125, 1.875, alt.shape[1], dtype=np.float32)
    prefix = tmp_path / "dosage"
    _write_dosage_pgen(prefix, alt)

    raw_ref = np.empty_like(alt)
    with pgenlib.PgenReader(
        bytes(prefix.with_suffix(".pgen")),
        raw_sample_ct=alt.shape[1],
        variant_ct=alt.shape[0],
    ) as reader:
        reader.read_dosages_range(0, alt.shape[0], raw_ref, allele_idx=0)
    standardized = _numpy_standardize(raw_ref.T)
    d = float(alt.shape[1] - 1)
    ss = np.sum(standardized * standardized, axis=0)
    residual_variance = ss / d
    inv = 1.0 / np.sqrt(np.maximum(residual_variance, 1e-10))
    x = standardized * inv[None, :]
    correlation = (x.T @ x) / d
    exact = np.sum(correlation * correlation, axis=1) - alt.shape[0] / d

    estimator = GenomewideLDScore(
        bed_path=str(prefix.with_suffix(".pgen")),
        annot_path=None,
        out_path=str(tmp_path / "dosage_out"),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=100_000,
        step_size=3,
        seed=409,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.1,
        use_mailman=False,
        impute_method="mean",
    )
    try:
        estimator._compute_ldscore()
        observed = estimator.gwldscore[:, 0]
    finally:
        estimator.close()
    probe_variance = 2.0 * (
        np.square(np.sum(correlation * correlation, axis=1))
        - np.sum(np.power(correlation, 4), axis=1)
    )
    probe_se = np.sqrt(np.maximum(probe_variance, 0.0) / estimator.nvecs)
    error = np.abs(observed - exact)
    assert np.all(error <= 5.0 * probe_se + 1e-10), (
        f"fractional-dosage probe error exceeded five Monte Carlo SEs: "
        f"max error={error.max():.6g}, max SE={probe_se.max():.6g}"
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"impute_method": "hwe"}, "HWE imputation"),
        ({"device": "cuda"}, "CUDA"),
        ({"correct_skew": True}, "skew correction"),
        ({"write_kmoments": True}, "K-moment"),
        ({"ddof": 2}, "requires ddof=1"),
    ],
)
def test_pgen_rejects_bed_only_modes(tmp_path, kwargs, message):
    prefix = tmp_path / "unsupported"
    _write_hardcall_pgen(prefix, _hardcalls())
    base = dict(
        bed_path=str(prefix.with_suffix(".pgen")),
        annot_path=None,
        out_path=str(tmp_path / "out"),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=8,
        step_size=3,
        seed=1,
        dtype="float64",
        num_threads=1,
        use_mailman=False,
        impute_method="mean",
        ddof=1,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        GenomewideLDScore(**base)
