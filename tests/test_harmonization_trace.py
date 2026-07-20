from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from summit.inference.trace import Trace
from summit.inference.sumcore import Sumcore
from summit.manifest.rg_manifest_fast import dispatch_rg_manifest_fast
from summit.sumstats.sumstats import (
    Sumstats,
    encode_dna_alleles,
    harmonize_allele_codes,
)


def _write_table(path, data):
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return str(path)


def test_vectorized_harmonization_orients_all_nonambiguous_cases():
    ref1 = encode_dna_alleles(["A", "A", "A", "A", "A", "I", "C"])
    ref2 = encode_dna_alleles(["C", "C", "C", "C", "T", "D", "C"])
    alt1 = encode_dna_alleles(["A", "C", "T", "G", "A", "I", "C"])
    alt2 = encode_dna_alleles(["C", "A", "G", "T", "T", "D", "C"])

    keep, flip = harmonize_allele_codes(ref1, ref2, alt1, alt2)
    np.testing.assert_array_equal(keep, [True, True, True, True, False, False, False])
    np.testing.assert_array_equal(flip, [False, True, False, True, False, False, False])


def test_retained_ambiguous_variants_use_literal_label_orientation():
    ref1 = encode_dna_alleles(["A", "A", "C", "C"])
    ref2 = encode_dna_alleles(["T", "T", "G", "G"])
    alt1 = encode_dna_alleles(["A", "T", "C", "G"])
    alt2 = encode_dna_alleles(["T", "A", "G", "C"])

    keep, flip = harmonize_allele_codes(
        ref1, ref2, alt1, alt2, drop_ambiguous=False
    )
    np.testing.assert_array_equal(keep, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(flip, [False, True, False, True])


def test_sumstats_requires_named_alleles_in_safe_mode(tmp_path):
    path = _write_table(
        tmp_path / "unnamed.tsv",
        {
            "ID": ["rs1"],
            "X": ["A"],
            "Y": ["C"],
            "N": [1000],
            "BETA": [0.1],
            "SE": [0.05],
        },
    )
    with pytest.raises(RuntimeError, match="needs named A1/A2"):
        Sumstats.from_file(path, require_alleles=True)

    # This is the explicit legacy/already-aligned escape hatch.
    ss = Sumstats.from_file(path, require_alleles=False)
    assert ss.nsnps == 1


def test_matched_materialization_orients_effects_in_place():
    ss = Sumstats(
        snps=np.array(["rs1", "rs2"]),
        z=np.array([1.0, 2.0]),
        beta=np.array([0.1, 0.2]),
        se=np.array([0.1, 0.1]),
        n=np.array([1000.0, 1000.0]),
        nsamp=1000.0,
        n_scale=999.0,
        cov_rank=0,
        cov_rank_source="test",
        a1=np.array(["A", "C"]),
        a2=np.array(["G", "T"]),
        name="test",
    )
    trace = SimpleNamespace(snps=ss.snps, nsnps=2)
    aligned = ss.align_to_trace(trace)
    matched = aligned.materialize(
        np.ones(2, dtype=bool),
        compute_diagnostics=False,
        allele_flip_mask=np.array([False, True]),
    )
    np.testing.assert_allclose(matched.z, [1.0, -2.0])
    np.testing.assert_allclose(matched.beta, [0.1, -0.2])
    np.testing.assert_array_equal(matched.a1, ["A", "T"])
    np.testing.assert_array_equal(matched.a2, ["G", "C"])
    # Immutable source arrays remain unchanged for reuse by manifest pairs.
    np.testing.assert_allclose(ss.z, [1.0, 2.0])
    np.testing.assert_array_equal(ss.a1, ["A", "C"])


def test_agreeing_alleles_preserve_the_existing_matched_arrays_bitwise():
    kwargs = dict(
        snps=np.array(["rs1", "rs2"]),
        z=np.array([1.25, -0.75]),
        beta=np.array([0.125, -0.075]),
        se=np.array([0.1, 0.1]),
        n=np.array([1000.0, 1000.0]),
        nsamp=1000.0,
        n_scale=999.0,
        cov_rank=0,
        cov_rank_source="test",
        a1=np.array(["A", "C"]),
        a2=np.array(["G", "T"]),
        name="test",
    )
    ss1 = Sumstats(**kwargs)
    ss2 = Sumstats(**kwargs)
    trace = SimpleNamespace(snps=ss1.snps, nsnps=2)
    aligned1 = ss1.align_to_trace(trace)
    aligned2 = ss2.align_to_trace(trace)

    core = object.__new__(Sumcore)
    core.drop_ambiguous = True
    core.log = None
    base = np.ones(2, dtype=bool)
    keep, flip = core._apply_allele_alignment_filter(aligned1, aligned2, base)
    np.testing.assert_array_equal(keep, base)
    np.testing.assert_array_equal(flip, np.zeros(2, dtype=bool))

    old = aligned2.materialize(base, compute_diagnostics=False)
    new = aligned2.materialize(
        keep, compute_diagnostics=False, allele_flip_mask=flip
    )
    for field in ("z", "chi2", "beta", "se", "n", "a1", "a2"):
        np.testing.assert_array_equal(getattr(new, field), getattr(old, field))


def test_trace_rejects_duplicate_primary_snp_ids(tmp_path):
    ld = _write_table(
        tmp_path / "dup.ldscore",
        {"CHR": [1, 1], "BP": [100, 101], "SNP": ["rs1", "rs1"], "L2": [1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="duplicate SNP IDs"):
        Trace(ldscores=ld)


@pytest.mark.parametrize("which", ["reg", "weight"])
def test_trace_rejects_duplicate_auxiliary_snp_ids(tmp_path, which):
    ld = _write_table(
        tmp_path / "main.ldscore",
        {"CHR": [1, 1], "BP": [100, 200], "SNP": ["rs1", "rs2"], "L2": [1.0, 2.0]},
    )
    aux = _write_table(
        tmp_path / f"dup_{which}.ldscore",
        {"CHR": [1, 1], "BP": [100, 101], "SNP": ["rs1", "rs1"], "L2": [2.0, 3.0]},
    )
    kwargs = {"ldscores_reg": aux} if which == "reg" else {"ldscores_reg_w": aux}
    with pytest.raises(ValueError, match="duplicate SNP IDs"):
        Trace(ldscores=ld, **kwargs)


@pytest.mark.parametrize("which", ["reg", "weight"])
def test_trace_rejects_coordinate_mismatch_in_auxiliary_ld(tmp_path, which):
    ld = _write_table(
        tmp_path / "main.ldscore",
        {"CHR": [1, 1], "BP": [100, 200], "SNP": ["rs1", "rs2"], "L2": [1.0, 2.0]},
    )
    aux = _write_table(
        tmp_path / f"{which}.ldscore",
        {"CHR": [1, 1], "BP": [100, 999], "SNP": ["rs1", "rs2"], "L2": [2.0, 3.0]},
    )
    kwargs = {"ldscores_reg": aux} if which == "reg" else {"ldscores_reg_w": aux}
    with pytest.raises(ValueError, match="CHR/BP mismatch"):
        Trace(ldscores=ld, **kwargs)


def test_trace_rejects_annotation_duplicate_and_coordinate_mismatch(tmp_path):
    ld = _write_table(
        tmp_path / "main.ldscore",
        {"CHR": [1, 1], "BP": [100, 200], "SNP": ["rs1", "rs2"], "L2": [1.0, 2.0]},
    )
    dup = _write_table(
        tmp_path / "dup.annot",
        {"CHR": [1, 1], "BP": [100, 100], "SNP": ["rs1", "rs1"], "A": [1, 1]},
    )
    with pytest.raises(ValueError, match="duplicate SNP IDs"):
        Trace(ldscores=ld, annot=dup)

    mismatch = _write_table(
        tmp_path / "mismatch.annot",
        {"CHR": [1, 1], "BP": [100, 999], "SNP": ["rs1", "rs2"], "A": [1, 1]},
    )
    with pytest.raises(ValueError, match="CHR/BP mismatch"):
        Trace(ldscores=ld, annot=mismatch)


def test_trace_indexed_joins_preserve_values_when_coordinates_agree(tmp_path):
    ld = _write_table(
        tmp_path / "main.ldscore",
        {"CHR": [1, 1], "BP": [200, 100], "SNP": ["rs2", "rs1"], "L2": [20.0, 10.0]},
    )
    reg = _write_table(
        tmp_path / "reg.ldscore",
        {"CHR": [1, 1], "BP": [100, 200], "SNP": ["rs1", "rs2"], "R": [100.0, 200.0]},
    )
    weight = _write_table(
        tmp_path / "weight.ldscore",
        {"CHR": [1, 1], "BP": [200, 100], "SNP": ["rs2", "rs1"], "W": [2000.0, 1000.0]},
    )
    annot = _write_table(
        tmp_path / "full.annot",
        {"CHR": [1, 1], "BP": [200, 100], "SNP": ["rs2", "rs1"], "A": [2.0, 1.0]},
    )

    trace = Trace(
        ldscores=ld,
        ldscores_reg=reg,
        ldscores_reg_w=weight,
        annot=annot,
    )
    np.testing.assert_array_equal(trace.snps, ["rs1", "rs2"])
    np.testing.assert_allclose(trace.ldscores[:, 0], [10.0, 20.0])
    np.testing.assert_allclose(trace.ldscores_reg[:, 0], [100.0, 200.0])
    np.testing.assert_allclose(trace.ldscores_reg_w[:, 0], [1000.0, 2000.0])
    np.testing.assert_allclose(trace.annot[:, 0], [1.0, 2.0])


def test_fast_manifest_harmonization_matches_prealigned_effects(tmp_path):
    m = 60
    chrom = np.repeat([1, 2, 3], m // 3)
    bp = np.arange(1, m + 1) * 100
    snp = np.asarray([f"rs{i}" for i in range(m)])
    ldval = np.linspace(1.0, 3.0, m)
    ld = _write_table(
        tmp_path / "main.ldscore",
        {"CHR": chrom, "BP": bp, "SNP": snp, "L2": ldval},
    )
    annot = _write_table(
        tmp_path / "main.annot",
        {"CHR": chrom, "BP": bp, "SNP": snp, "base": np.ones(m)},
    )

    n = np.full(m, 10_000)
    se = np.full(m, 0.05)
    z = np.sqrt(1.0 + 8.0 * ldval)
    beta = z * se
    trait1 = _write_table(
        tmp_path / "trait1.tsv",
        {"ID": snp, "A1": "A", "A2": "C", "N": n, "BETA": beta, "SE": se},
    )
    trait2 = _write_table(
        tmp_path / "trait2.tsv",
        {"ID": snp, "A1": "A", "A2": "C", "N": n, "BETA": beta, "SE": se},
    )
    swapped = _write_table(
        tmp_path / "trait2_swapped.tsv",
        {"ID": snp, "A1": "C", "A2": "A", "N": n, "BETA": -beta, "SE": se},
    )

    manifest = pd.DataFrame(
        [
            {
                "row_id": 1, "phen1": "t1", "phen2": "t2",
                "sumstats1": trait1, "sumstats2": trait2,
                "intercept_rg": 0.0, "cov_rank1": None, "cov_rank2": None,
                "out_stem": "t1__t2",
            },
            {
                "row_id": 2, "phen1": "t1", "phen2": "t2_swapped",
                "sumstats1": trait1, "sumstats2": swapped,
                "intercept_rg": 0.0, "cov_rank1": None, "cov_rank2": None,
                "out_stem": "t1__t2_swapped",
            },
        ]
    )
    trait_meta = {
        trait1: {"phen": "t1", "cov_rank": None},
        trait2: {"phen": "t2", "cov_rank": None},
        swapped: {"phen": "t2_swapped", "cov_rank": None},
    }
    args = SimpleNamespace(
        adjust_delta=False,
        align_alleles=True,
        allow_neg_enr=False,
        annot=annot,
        bim=None,
        chisq_action="none",
        clip_nonfinite_vals=False,
        enrich_mode="auto",
        jack_mode="mean",
        keep_ambiguous=False,
        ldscores=ld,
        ldscores_reg=None,
        max_chisq=None,
        njack="chr",
        out=str(tmp_path / "fast_out"),
        rg_se_method="jackknife",
        verbose=False,
        write_jack=False,
        write_normeq=False,
    )

    dispatch_rg_manifest_fast(args, None, manifest, trait_meta, 0)
    result = pd.read_csv(tmp_path / "fast_out" / "manifest.results.tsv", sep="\t")
    cols = [
        "n_snps", "h2_trait1", "h2_trait1_se", "h2_trait2", "h2_trait2_se",
        "gamma_g_total", "gamma_g_total_se", "rg_total", "rg_total_se",
    ]
    np.testing.assert_allclose(
        result.loc[0, cols].to_numpy(dtype=float),
        result.loc[1, cols].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
