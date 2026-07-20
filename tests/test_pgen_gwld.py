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
    read_aligned_annotations,
    read_psam_sample_ids,
    read_pvar_variants,
    resolve_genotype_input,
)
from summit.ldscore.gw_ldscore import GenomewideLDScore, read_cov
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.win_ldscore import WindowedLDScore


class _Log:
    def __init__(self):
        self.messages = []

    def _log(self, message):
        self.messages.append(str(message))

    def _save_log(self, path):
        self.saved_path = str(path)


@pytest.mark.parametrize("full", [True, False])
def test_shared_annotation_reader_supports_chromosome_split_specs(tmp_path, full):
    variants = pd.DataFrame(
        {
            "CHR": ["1", "1", "2", "2"],
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "BP": [101, 102, 201, 202],
        }
    )
    expected = np.asarray([[1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8]])
    for chrom, rows in ((1, slice(0, 2)), (2, slice(2, 4))):
        path = tmp_path / f"split{chrom}.annot"
        if full:
            frame = variants.iloc[rows].copy()
            frame.insert(3, "CM", 0.0)
            frame["base"] = expected[rows, 0]
            frame["fractional"] = expected[rows, 1]
            frame.to_csv(path, sep="\t", index=False)
        else:
            np.savetxt(path, expected[rows], delimiter="\t")

    names, observed, continuous = read_aligned_annotations(
        str(tmp_path / "split@.annot"), variants, source_label="test variants"
    )
    np.testing.assert_array_equal(observed, expected)
    assert names == (["base", "fractional"] if full else ["L2_0", "L2_1"])
    assert continuous is True


def test_single_annotation_gxe_sketch_applies_fractional_weights():
    estimator = object.__new__(GenomewideEnvLDScore)
    estimator.nbins = 1
    W = np.asarray([[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]])
    Z = np.asarray([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25]])
    annot = np.asarray([[0.25], [0.0], [1.0]])
    observed = np.zeros((W.shape[0], Z.shape[1]))
    estimator._accumulate_sketch_block(observed, W, Z, annot)
    expected = (W * np.sqrt(annot[:, 0])[None, :]) @ Z
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


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


def _run_windowed(genotype_path: Path, out: Path, covar_path: Path | None = None):
    estimator = WindowedLDScore(
        bed_path=str(genotype_path),
        annot_path=None,
        out_path=str(out),
        covar_path=None if covar_path is None else str(covar_path),
        log=_Log(),
        ld_wind_kb=1_000.0,
        step_size=3,
        seed=9182,
        dtype="float64",
        num_threads=2,
        impute_method="mean",
        panel_cols=2,
        cache_mb=16,
    )
    try:
        estimator._compute_ldscore()
        return estimator.win_ldscore.copy()
    finally:
        estimator.close()


def _run_gxe(
    genotype_path: Path,
    env_path: Path,
    out: Path,
    covar_path: Path | None = None,
):
    estimator = GenomewideEnvLDScore(
        bed_path=str(genotype_path),
        env_path=str(env_path),
        annot_path=None,
        out_path=str(out),
        covar_path=None if covar_path is None else str(covar_path),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=256,
        step_size=3,
        seed=773,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        impute_method="mean",
    )
    try:
        estimator._compute_ldscore()
        return estimator.gxe_ldscore.copy(), estimator.gee_ldscore.copy(), estimator.row_sel.copy()
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


def test_duplicate_variant_ids_are_rejected_before_annotation_alignment(tmp_path):
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

    with pytest.raises(ValueError, match="duplicate SNP IDs"):
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
    pgen_mc = pd.read_csv(tmp_path / "pgen_out.gw.mc.tsv", sep="\t")
    bed_mc = pd.read_csv(tmp_path / "bed_out.gw.mc.tsv", sep="\t")
    pd.testing.assert_frame_equal(pgen_mc, bed_mc, check_exact=True)
    assert not (tmp_path / "pgen_out.gw.mcvar.gz").exists()


@pytest.mark.parametrize("genotype_format", ["bed", "pgen"])
def test_partitioned_mc_diagnostic_and_optional_per_snp_output(
    tmp_path, genotype_format
):
    alt = _hardcalls()
    prefix = tmp_path / f"mc_{genotype_format}"
    if genotype_format == "bed":
        _write_matching_bed(prefix, alt)
        genotype_path = prefix.with_suffix(".bed")
    else:
        _write_hardcall_pgen(prefix, alt)
        genotype_path = prefix.with_suffix(".pgen")

    annotation = pd.DataFrame({
        "CHR": ["1"] * alt.shape[0],
        "BP": [100 + idx for idx in range(alt.shape[0])],
        "SNP": [f"rs{idx}" for idx in range(alt.shape[0])],
        "CM": [0.0] * alt.shape[0],
        "base": np.ones(alt.shape[0], dtype=int),
        "subset": np.asarray([1, 0, 1, 0, 0, 1, 0], dtype=int),
    })
    annotation_path = tmp_path / f"mc_{genotype_format}.annot"
    annotation.to_csv(annotation_path, sep="\t", index=False)
    out = tmp_path / f"mc_{genotype_format}_out"
    estimator = GenomewideLDScore(
        bed_path=str(genotype_path),
        annot_path=str(annotation_path),
        out_path=str(out),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=64,
        step_size=3,
        seed=487,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        use_mailman=False,
        impute_method="mean",
        write_ld_mc_var=True,
    )
    try:
        estimator._compute_ldscore()
        assert estimator.ld_mc_variance.shape == (alt.shape[0], 2)
        np.testing.assert_allclose(
            estimator.ld_mc_diagnostic["integrated_mc_variance"],
            estimator.ld_mc_variance.sum(axis=0),
            rtol=2e-14,
            atol=2e-14,
        )
        assert np.all(estimator.ld_mc_variance >= 0.0)
    finally:
        estimator.close()

    diagnostic = pd.read_csv(f"{out}.gw.mc.tsv", sep="\t")
    assert diagnostic["annotation"].tolist() == ["base", "subset"]
    assert np.all(diagnostic["integrated_mc_variance"] >= 0.0)
    per_snp = pd.read_csv(f"{out}.gw.mcvar.gz", sep="\t")
    assert per_snp.columns.tolist() == [
        "CHR", "SNP", "BP",
        "base_MC_VAR", "base_MC_SE", "base_MC_CI95_LO", "base_MC_CI95_HI",
        "subset_MC_VAR", "subset_MC_SE", "subset_MC_CI95_LO", "subset_MC_CI95_HI",
    ]
    for name in ("base", "subset"):
        np.testing.assert_allclose(
            per_snp[f"{name}_MC_SE"],
            np.sqrt(per_snp[f"{name}_MC_VAR"]),
            rtol=2e-10,
            atol=2e-12,
        )
        center = (
            per_snp[f"{name}_MC_CI95_LO"]
            + per_snp[f"{name}_MC_CI95_HI"]
        ) / 2.0
        score = pd.read_csv(f"{out}.gw.ldscore.gz", sep="\t")[name]
        np.testing.assert_allclose(center, score, rtol=0.0, atol=6e-7)


def test_mailman_mc_aggregate_matches_optional_per_snp_sum(tmp_path):
    alt = _hardcalls()
    prefix = tmp_path / "mc_mailman"
    _write_matching_bed(prefix, alt)
    estimator = GenomewideLDScore(
        bed_path=str(prefix.with_suffix(".bed")),
        annot_path=None,
        out_path=str(tmp_path / "mc_mailman_out"),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=64,
        step_size=3,
        seed=991,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        use_mailman=True,
        impute_method="hwe",
        write_ld_mc_var=True,
    )
    try:
        estimator._compute_ldscore()
        mailman_score = estimator.gwldscore.copy()
        mailman_mc_var = estimator.ld_mc_variance.copy()
        mailman_integrated = estimator.ld_mc_diagnostic[
            "integrated_mc_variance"
        ].to_numpy(copy=True)
        np.testing.assert_allclose(
            estimator.ld_mc_diagnostic["integrated_mc_variance"],
            estimator.ld_mc_variance.sum(axis=0),
            rtol=2e-13,
            atol=2e-13,
        )
    finally:
        estimator.close()

    dense = GenomewideLDScore(
        bed_path=str(prefix.with_suffix(".bed")),
        annot_path=None,
        out_path=str(tmp_path / "mc_dense_hwe_out"),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=64,
        step_size=3,
        seed=991,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        use_mailman=False,
        impute_method="hwe",
        write_ld_mc_var=True,
    )
    try:
        dense._compute_ldscore()
        np.testing.assert_allclose(dense.gwldscore, mailman_score, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            dense.ld_mc_variance, mailman_mc_var, rtol=2e-11, atol=2e-12
        )
    finally:
        dense.close()

    aggregate_only = GenomewideLDScore(
        bed_path=str(prefix.with_suffix(".bed")),
        annot_path=None,
        out_path=str(tmp_path / "mc_mailman_aggregate_only_out"),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=64,
        step_size=3,
        seed=991,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        use_mailman=True,
        impute_method="hwe",
    )
    try:
        aggregate_only._compute_ldscore()
        np.testing.assert_array_equal(aggregate_only.gwldscore, mailman_score)
        np.testing.assert_allclose(
            aggregate_only.ld_mc_diagnostic["integrated_mc_variance"],
            mailman_integrated,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        aggregate_only.close()

    mailman_without_mc = GenomewideLDScore(
        bed_path=str(prefix.with_suffix(".bed")),
        annot_path=None,
        out_path=str(tmp_path / "mc_mailman_disabled_out"),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=64,
        step_size=3,
        seed=991,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        use_mailman=True,
        impute_method="hwe",
        estimate_mc_noise=False,
    )
    try:
        mailman_without_mc._compute_ldscore()
        np.testing.assert_array_equal(mailman_without_mc.gwldscore, mailman_score)
        assert not (tmp_path / "mc_mailman_disabled_out.gw.mc.tsv").exists()
    finally:
        mailman_without_mc.close()


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


def test_hardcall_pgen_and_bed_have_identical_windowed_scores(tmp_path):
    alt = _hardcalls()
    pgen_prefix = tmp_path / "window_pgen"
    bed_prefix = tmp_path / "window_bed"
    _write_hardcall_pgen(pgen_prefix, alt)
    _write_matching_bed(bed_prefix, alt)
    covar = tmp_path / "window_covar.tsv"
    cov = pd.DataFrame({
        "FID": [f"f{i}" for i in range(alt.shape[1])],
        "IID": [f"i{i}" for i in range(alt.shape[1])],
        "age": np.linspace(20.0, 70.0, alt.shape[1]),
    })
    cov.loc[5, "age"] = np.nan
    cov.to_csv(covar, sep="\t", index=False, na_rep="NA")

    pgen_score = _run_windowed(
        pgen_prefix.with_suffix(".pgen"), tmp_path / "window_pgen_out", covar
    )
    bed_score = _run_windowed(
        bed_prefix.with_suffix(".bed"), tmp_path / "window_bed_out", covar
    )
    np.testing.assert_allclose(pgen_score, bed_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.atleast_1d(np.loadtxt(tmp_path / "window_pgen_out.win.M_5_50")),
        np.atleast_1d(np.loadtxt(tmp_path / "window_bed_out.win.M_5_50")),
        rtol=0.0,
        atol=0.0,
    )


def test_small_pgen_window_auto_panel_survives_maf_pass(tmp_path):
    prefix = tmp_path / "small_window"
    _write_hardcall_pgen(prefix, _hardcalls())
    estimator = WindowedLDScore(
        bed_path=str(prefix.with_suffix(".pgen")),
        annot_path=None,
        out_path=str(tmp_path / "small_window_out"),
        log=_Log(),
        ld_wind_kb=1.0,
        step_size=1000,
        seed=11,
        dtype="float64",
        num_threads=1,
        impute_method="mean",
        panel_cols=None,
        cache_mb=0,
    )
    try:
        estimator._compute_ldscore()
        assert estimator.win_ldscore.shape == (_hardcalls().shape[0], 1)
        assert estimator._pgen_reader.block_capacity == _hardcalls().shape[0]
    finally:
        estimator.close()


def test_windowed_partitioned_scores_match_exact_bp_mask_and_covariate_df(tmp_path):
    rng = np.random.default_rng(381)
    alt = rng.integers(0, 3, size=(10, 24), dtype=np.int8)
    # Ensure every column is nonconstant after projection.
    alt[:, :3] = np.asarray([0, 1, 2], dtype=np.int8)
    pgen_prefix = tmp_path / "window_oracle_pgen"
    bed_prefix = tmp_path / "window_oracle_bed"
    _write_hardcall_pgen(pgen_prefix, alt)
    _write_matching_bed(bed_prefix, alt)

    bp = np.asarray([100, 150, 200, 201, 300, 301, 450, 451, 700, 901])
    # Rewriting without the VCF meta line keeps a valid #CHROM header.
    with pgen_prefix.with_suffix(".pvar").open("w", encoding="utf-8") as handle:
        handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for idx, pos in enumerate(bp):
            handle.write(f"1\t{int(pos)}\trs{idx}\tA\tG\n")
    bim = pd.read_csv(bed_prefix.with_suffix(".bim"), sep=r"\s+", header=None)
    bim.iloc[:, 3] = bp
    bim.to_csv(bed_prefix.with_suffix(".bim"), sep="\t", header=False, index=False)

    annotation = np.column_stack([
        np.ones(len(bp), dtype=float),
        np.asarray([1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=float),
    ])
    annot_path = tmp_path / "window_oracle.annot"
    pd.DataFrame({
        "CHR": ["chr1"] * len(bp),
        "BP": bp,
        "SNP": [f"rs{i}" for i in range(len(bp))],
        "CM": np.zeros(len(bp)),
        "base": annotation[:, 0],
        "subset": annotation[:, 1],
    }).to_csv(annot_path, sep="\t", index=False)

    age = np.linspace(-1.5, 2.0, alt.shape[1])
    covar_path = tmp_path / "window_oracle.covar"
    pd.DataFrame({
        "FID": [f"f{i}" for i in range(alt.shape[1])],
        "IID": [f"i{i}" for i in range(alt.shape[1])],
        "age": age,
    }).sample(frac=1.0, random_state=4).to_csv(covar_path, sep="\t", index=False)

    observed = []
    for genotype_path, out_name in (
        (pgen_prefix.with_suffix(".pgen"), "pgen"),
        (bed_prefix.with_suffix(".bed"), "bed"),
    ):
        estimator = WindowedLDScore(
            bed_path=str(genotype_path),
            annot_path=str(annot_path),
            out_path=str(tmp_path / out_name),
            covar_path=str(covar_path),
            log=_Log(),
            ld_wind_kb=0.1,
            step_size=3,
            seed=29,
            dtype="float64",
            num_threads=1,
            impute_method="mean",
            panel_cols=2,
            cache_mb=0,
        )
        try:
            estimator._compute_ldscore()
            observed.append(estimator.win_ldscore.copy())
            assert estimator.corr_dim == alt.shape[1] - 2
        finally:
            estimator.close()

    raw = (2.0 - alt.T.astype(np.float64))
    raw -= raw.mean(axis=0, keepdims=True)
    raw /= raw.std(axis=0, ddof=0, keepdims=True)
    design = np.column_stack([np.ones(alt.shape[1]), age])
    q, _ = np.linalg.qr(design, mode="reduced")
    x = raw - q @ (q.T @ raw)
    x -= x.mean(axis=0, keepdims=True)
    x *= np.sqrt(float(x.shape[0]) / np.sum(x * x, axis=0)).reshape(1, -1)
    corr = (x.T @ x) / float(x.shape[0])
    corr_dim = x.shape[0] - q.shape[1]
    unbiased = corr * corr - (1.0 - corr * corr) / float(corr_dim - 1)

    window_bp = 100.0
    mask = (
        np.abs(bp[:, None].astype(np.int64) - bp[None, :].astype(np.int64))
        <= window_bp
    ).astype(float)
    expected = (unbiased * mask) @ annotation

    np.testing.assert_allclose(observed[0], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(observed[1], expected, rtol=2e-12, atol=2e-12)


def test_windowed_large_bp_exact_boundary_matches_bed_pgen_and_oracle(tmp_path):
    rng = np.random.default_rng(902)
    alt = rng.integers(0, 3, size=(5, 24), dtype=np.int8)
    alt[:, :3] = np.asarray([0, 1, 2], dtype=np.int8)
    bp = np.asarray(
        [
            1_000_000_000,
            1_029_300_674,
            1_049_300_674,  # exactly 20 Mb from the preceding variant
            1_049_300_675,  # one BP outside that pair's window
            1_080_000_000,
        ],
        dtype=np.int64,
    )
    pgen_prefix = tmp_path / "boundary_pgen"
    bed_prefix = tmp_path / "boundary_bed"
    _write_hardcall_pgen(pgen_prefix, alt)
    _write_matching_bed(bed_prefix, alt)
    with pgen_prefix.with_suffix(".pvar").open("w", encoding="utf-8") as handle:
        handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for idx, pos in enumerate(bp):
            handle.write(f"1\t{int(pos)}\trs{idx}\tA\tG\n")
    bim = pd.read_csv(bed_prefix.with_suffix(".bim"), sep=r"\s+", header=None)
    bim.iloc[:, 3] = bp
    bim.to_csv(bed_prefix.with_suffix(".bim"), sep="\t", header=False, index=False)

    observed = []
    for genotype_path, label in (
        (pgen_prefix.with_suffix(".pgen"), "pgen"),
        (bed_prefix.with_suffix(".bed"), "bed"),
    ):
        estimator = WindowedLDScore(
            bed_path=str(genotype_path),
            annot_path=None,
            out_path=str(tmp_path / f"boundary_{label}"),
            log=_Log(),
            ld_wind_kb=20_000.0,
            step_size=2,
            seed=51,
            dtype="float64",
            num_threads=1,
            impute_method="mean",
            panel_cols=1,
            cache_mb=0,
        )
        try:
            estimator._compute_ldscore()
            observed.append(estimator.win_ldscore[:, 0].copy())
        finally:
            estimator.close()

    x = 2.0 - alt.T.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, ddof=0, keepdims=True)
    corr = (x.T @ x) / float(x.shape[0])
    # Centering consumes one residual dimension; SUMMIT's finite-sample null
    # divides by corr_dim - 1 = N - 2 in the no-covariate case.
    unbiased = corr * corr - (1.0 - corr * corr) / float(x.shape[0] - 2)
    mask = (
        np.abs(bp[:, None].astype(np.int64) - bp[None, :].astype(np.int64))
        <= 20_000_000
    )
    expected = (unbiased * mask).sum(axis=1)

    np.testing.assert_allclose(observed[0], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(observed[1], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(observed[0], observed[1], rtol=2e-12, atol=2e-12)


def test_hardcall_pgen_and_bed_have_identical_gxe_scores(tmp_path):
    alt = _hardcalls()
    pgen_prefix = tmp_path / "gxe_pgen"
    bed_prefix = tmp_path / "gxe_bed"
    _write_hardcall_pgen(pgen_prefix, alt)
    _write_matching_bed(bed_prefix, alt)
    env_path = tmp_path / "environment.tsv"
    pd.DataFrame({
        "FID": [f"f{i}" for i in range(alt.shape[1])],
        "IID": [f"i{i}" for i in range(alt.shape[1])],
        "environment": np.asarray([-1.2, 0.4, 1.1, -0.7, 0.2, 1.5] * 2),
    }).to_csv(env_path, sep="\t", index=False)
    covar_path = tmp_path / "gxe_covar.tsv"
    pd.DataFrame({
        "FID": [f"f{i}" for i in range(alt.shape[1])],
        "IID": [f"i{i}" for i in range(alt.shape[1])],
        "age": np.linspace(20.0, 70.0, alt.shape[1]),
        "batch": np.asarray([0.0, 1.0] * (alt.shape[1] // 2)),
    }).to_csv(covar_path, sep="\t", index=False)

    pgen_xw, pgen_ww, pgen_rows = _run_gxe(
        pgen_prefix.with_suffix(".pgen"), env_path, tmp_path / "gxe_pgen_out", covar_path
    )
    bed_xw, bed_ww, bed_rows = _run_gxe(
        bed_prefix.with_suffix(".bed"), env_path, tmp_path / "gxe_bed_out", covar_path
    )
    np.testing.assert_array_equal(pgen_rows, bed_rows)
    np.testing.assert_allclose(pgen_xw, bed_xw, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(pgen_ww, bed_ww, rtol=2e-12, atol=2e-12)


def test_fractional_pgen_gxe_matches_independent_oracle_with_shuffled_sample_files(tmp_path):
    rng = np.random.default_rng(9201)
    sample_ct = 36
    variant_ct = 6
    alt = rng.uniform(0.05, 1.95, size=(variant_ct, sample_ct)).astype(np.float32)
    alt[1, ::9] = -9.0
    alt[4, 3::11] = -9.0
    prefix = tmp_path / "fractional_gxe"
    _write_dosage_pgen(prefix, alt)

    annot = np.column_stack([
        np.ones(variant_ct, dtype=float),
        np.asarray([1, 0, 1, 0, 0, 1], dtype=float),
    ])
    annot_path = tmp_path / "fractional_gxe.annot"
    pd.DataFrame({
        "CHR": ["1"] * variant_ct,
        "BP": np.arange(100, 100 + variant_ct),
        "SNP": [f"rs{i}" for i in range(variant_ct)],
        "CM": np.zeros(variant_ct),
        "base": annot[:, 0],
        "subset": annot[:, 1],
    }).to_csv(annot_path, sep="\t", index=False)

    env_values = rng.normal(size=sample_ct)
    age_values = np.linspace(18.0, 78.0, sample_ct) + rng.normal(scale=2.0, size=sample_ct)
    env_missing = 7
    cov_missing = 19
    env_df = pd.DataFrame({
        "FID": [f"f{i}" for i in range(sample_ct)],
        "IID": [f"i{i}" for i in range(sample_ct)],
        "environment": env_values,
    })
    env_df.loc[env_missing, "environment"] = np.nan
    cov_df = pd.DataFrame({
        "FID": [f"f{i}" for i in range(sample_ct)],
        "IID": [f"i{i}" for i in range(sample_ct)],
        "age": age_values,
    })
    cov_df.loc[cov_missing, "age"] = np.nan
    env_path = tmp_path / "fractional_gxe.env"
    cov_path = tmp_path / "fractional_gxe.covar"
    env_df.sample(frac=1.0, random_state=8).to_csv(
        env_path, sep="\t", index=False, na_rep="NA"
    )
    cov_df.sample(frac=1.0, random_state=13).to_csv(
        cov_path, sep="\t", index=False, na_rep="NA"
    )

    estimator = GenomewideEnvLDScore(
        bed_path=str(prefix.with_suffix(".pgen")),
        env_path=str(env_path),
        annot_path=str(annot_path),
        out_path=str(tmp_path / "fractional_gxe_out"),
        covar_path=str(cov_path),
        log=_Log(),
        rand_dist="rademacher",
        low_level={"numa_mode": None},
        num_vecs=50_000,
        step_size=4,
        seed=671,
        dtype="float64",
        num_threads=2,
        ddof=1,
        target_xz_mem=0.01,
        impute_method="mean",
    )
    try:
        estimator._compute_ldscore()
        observed_xw = estimator.gxe_ldscore.copy()
        observed_ww = estimator.gee_ldscore.copy()
        observed_rows = estimator.row_sel.copy()
        df_corr = int(estimator.df_corr)
    finally:
        estimator.close()

    keep = np.ones(sample_ct, dtype=bool)
    keep[[env_missing, cov_missing]] = False
    keep_idx = np.flatnonzero(keep)
    np.testing.assert_array_equal(observed_rows, keep_idx)

    ref_variant_major = np.empty_like(alt)
    with pgenlib.PgenReader(
        bytes(prefix.with_suffix(".pgen")),
        raw_sample_ct=sample_ct,
        variant_ct=variant_ct,
    ) as reader:
        reader.read_dosages_range(
            0, variant_ct, ref_variant_major, allele_idx=0, sample_maj=0
        )
    raw = ref_variant_major[:, keep_idx].T.astype(np.float64)
    G = np.zeros_like(raw)
    for j in range(variant_ct):
        col = raw[:, j]
        valid = col != -9.0
        mean = float(col[valid].mean())
        filled = np.where(valid, col, mean)
        G[:, j] = (filled - mean) / filled.std(ddof=1)

    env = env_values[keep_idx].astype(np.float64)
    env = (env - env.mean()) / env.std(ddof=1)
    age = age_values[keep_idx].astype(np.float64)
    age = (age - age.mean()) / age.std(ddof=1)
    q, r = np.linalg.qr(np.column_stack([age, env]), mode="reduced")
    q = q[:, np.abs(np.diag(r)) > 1e-10]
    assert df_corr == len(keep_idx) - q.shape[1] - 1

    X = G - q @ (q.T @ G)
    X -= X.mean(axis=0, keepdims=True)
    X *= np.sqrt(float(df_corr) / np.sum(X * X, axis=0)).reshape(1, -1)
    W = G * env[:, None]
    W -= q @ (q.T @ W)
    W -= W.mean(axis=0, keepdims=True)
    W *= np.sqrt(float(df_corr) / np.sum(W * W, axis=0)).reshape(1, -1)

    r_xw = (X.T @ W) / float(df_corr)
    r_ww = (W.T @ W) / float(df_corr)
    null = annot.sum(axis=0, keepdims=True) / float(df_corr)
    expected_xw = (r_xw * r_xw) @ annot - null
    expected_ww = (r_ww * r_ww) @ annot - null

    def probe_se(corr):
        out = np.empty((variant_ct, annot.shape[1]), dtype=float)
        for k in range(annot.shape[1]):
            b2 = corr * corr * annot[:, k].reshape(1, -1)
            variance = 2.0 * (
                np.square(b2.sum(axis=1)) - np.square(b2).sum(axis=1)
            )
            out[:, k] = np.sqrt(np.maximum(variance, 0.0) / estimator.nvecs)
        return out

    se_xw = probe_se(r_xw)
    se_ww = probe_se(r_ww)
    assert np.all(np.abs(observed_xw - expected_xw) <= 6.0 * se_xw + 2e-9)
    assert np.all(np.abs(observed_ww - expected_ww) <= 6.0 * se_ww + 2e-9)


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
