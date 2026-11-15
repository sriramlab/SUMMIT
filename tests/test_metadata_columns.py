"""LD-score and annotation metadata matched by header name: the leading run of CHR, BP, SNP and CM columns in any
order (CHR, BP and SNP required, CM optional), data columns after it. Canonical and reordered copies of a file give
identical results in every reader; a metadata name after a data column, a missing BP or a duplicated SNP is refused."""
import numpy as np
import pandas as pd
import pytest

from summit import utils
from summit.inference.ldsc_h2 import read_ldsc_weight_ld_aligned, resolve_ldsc_reference_moments
from summit.inference.trace import Trace
from summit.ldscore.genotype_source import read_aligned_annotations
from summit.relatives.moments import read_annotation_classes

M = 30
LAYOUTS = [["CHR", "BP", "SNP", "CM"], ["CHR", "SNP", "CM", "BP"], ["SNP", "CHR", "BP"], ["CHR", "BP", "SNP"]]


def _meta():
    i = np.arange(M)
    return pd.DataFrame({"CHR": i // 15 + 1, "BP": 1000 + 100 * i, "SNP": [f"rs{j}" for j in i], "CM": 0.01 * i})


def _write(path, layout, data):
    df = pd.concat([_meta()[layout], data], axis=1)
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def _ld(k=2):
    i = np.arange(M)
    return pd.DataFrame({f"L2_{j}": 1.0 + 0.1 * ((i * (j + 3)) % 7) for j in range(k)})


def _annot(k=2):
    return pd.DataFrame({f"A{j}": (np.arange(M) % k == j).astype(float) for j in range(k)})


def test_split_rule():
    split = utils._split_snp_metadata
    for layout in LAYOUTS:
        assert split(layout + ["L2"], source="x") == (layout, ["L2"])
    assert split(["A0", "A1"], source="x", allow_thin=True) == ([], ["A0", "A1"])
    assert split(["SNP", "A0"], source="x", required=("SNP",)) == (["SNP"], ["A0"])
    with pytest.raises(ValueError, match="after the first data column"):
        split(["CHR", "BP", "SNP", "L2", "CM"], source="x")
    with pytest.raises(ValueError, match=r"missing \['BP'\]"):
        split(["CHR", "SNP", "L2"], source="x")
    with pytest.raises(ValueError, match="duplicated"):
        split(["CHR", "BP", "SNP", "SNP.1", "L2"], source="x")         # pandas' renaming of a second SNP column
    with pytest.raises(ValueError, match="after the first data column"):
        split(["A0", "SNP"], source="x", allow_thin=True)
    with pytest.raises(ValueError, match="leading run of columns named CHR, BP, SNP and CM"):
        split(["CHR", "BP", "L2"], source="x")


def _trace(tmp_path, tag, ld_layout, annot_layout):
    ld = _write(tmp_path / f"{tag}.ld.tsv", ld_layout, _ld())
    an = _write(tmp_path / f"{tag}.annot.tsv", annot_layout, _annot())
    reg = _write(tmp_path / f"{tag}.reg.tsv", ld_layout, _ld(1))
    return Trace(ldscores=ld, ldscores_reg=reg, ldscores_reg_w=reg, annot=an)


@pytest.mark.parametrize("layout", LAYOUTS[1:])
def test_trace_reads_reordered_copies_identically(tmp_path, layout):
    canon = _trace(tmp_path, "canon", LAYOUTS[0], LAYOUTS[0])
    other = _trace(tmp_path, "other", layout, layout)
    for name in ("snps", "chr", "bp", "annot", "annot_header", "ldscores", "ldscores_reg", "ldscores_reg_w"):
        np.testing.assert_array_equal(getattr(other, name), getattr(canon, name))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_other_readers_read_reordered_copies_identically(tmp_path, layout):
    meta = _meta()
    w0 = read_ldsc_weight_ld_aligned(_write(tmp_path / "w0.tsv", LAYOUTS[0], _ld(1)), meta.SNP.to_numpy(),
                                     target_chr=meta.CHR.to_numpy(), target_bp=meta.BP.to_numpy())
    w1 = read_ldsc_weight_ld_aligned(_write(tmp_path / "w1.tsv", layout, _ld(1)), meta.SNP.to_numpy(),
                                     target_chr=meta.CHR.to_numpy(), target_bp=meta.BP.to_numpy())
    for a, b in zip(w0, w1):
        np.testing.assert_array_equal(a, b)
    a0 = _write(tmp_path / "a0.tsv", LAYOUTS[0], _annot())
    a1 = _write(tmp_path / "a1.tsv", layout, _annot())
    kw = dict(trace_annot=_annot().to_numpy(), trace_header=["A0", "A1"])
    r0 = resolve_ldsc_reference_moments(annot_path=a0, **kw)
    r1 = resolve_ldsc_reference_moments(annot_path=a1, **kw)
    np.testing.assert_array_equal(r0.m_annot, r1.m_annot)
    np.testing.assert_array_equal(r0.overlap_matrix, r1.overlap_matrix)
    g0 = read_aligned_annotations(a0, meta[["CHR", "SNP", "BP"]])
    g1 = read_aligned_annotations(a1, meta[["CHR", "SNP", "BP"]])
    assert g0[0] == g1[0] and g0[2] == g1[2]
    np.testing.assert_array_equal(g0[1], g1[1])
    c0 = read_annotation_classes(a0, meta.SNP.to_numpy())
    c1 = read_annotation_classes(a1, meta.SNP.to_numpy())
    np.testing.assert_array_equal(c0[0], c1[0])
    assert c0[1] == c1[1]


def _bad(tmp_path, kind):
    data, meta = _ld(1), _meta()
    if kind == "cm_after_data":
        df = pd.concat([meta[["CHR", "BP", "SNP"]], data, meta[["CM"]]], axis=1)
    elif kind == "missing_bp":
        df = pd.concat([meta[["CHR", "SNP"]], data], axis=1)
    else:
        df = pd.concat([meta[["CHR", "BP", "SNP"]], meta[["SNP"]], data], axis=1)
    path = tmp_path / f"{kind}.tsv"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


@pytest.mark.parametrize("kind", ["cm_after_data", "missing_bp", "duplicated_snp"])
def test_readers_refuse_malformed_metadata(tmp_path, kind):
    bad = _bad(tmp_path, kind)
    good = _write(tmp_path / "good.tsv", LAYOUTS[0], _ld(1))
    meta = _meta()
    with pytest.raises(ValueError, match="Malformed"):
        Trace(ldscores=bad)
    with pytest.raises(ValueError, match="Malformed"):
        Trace(ldscores=good, annot=bad)
    with pytest.raises(ValueError, match="Malformed"):
        read_ldsc_weight_ld_aligned(bad, meta.SNP.to_numpy())
    with pytest.raises(ValueError, match="Malformed"):
        resolve_ldsc_reference_moments(annot_path=bad, trace_annot=np.ones((M, 1)), trace_header=["L2_0"])
    with pytest.raises(ValueError, match="Malformed"):
        read_aligned_annotations(bad, meta[["CHR", "SNP", "BP"]])
    if kind != "missing_bp":                                     # the class reader needs SNP only
        with pytest.raises(ValueError, match="Malformed"):
            read_annotation_classes(bad, meta.SNP.to_numpy())


def test_thin_annotations_are_unchanged(tmp_path):
    ld = _write(tmp_path / "ld.tsv", LAYOUTS[0], _ld())
    thin = tmp_path / "thin.txt"
    np.savetxt(thin, _annot().to_numpy(), fmt="%g")
    headed = tmp_path / "headed.txt"
    _annot().to_csv(headed, sep=" ", index=False)
    t0 = Trace(ldscores=ld, annot=str(thin))
    t1 = Trace(ldscores=ld, annot=str(headed))
    np.testing.assert_array_equal(t0.annot, _annot().to_numpy())
    np.testing.assert_array_equal(t1.annot, _annot().to_numpy())
    assert list(t1.annot_header) == ["A0", "A1"]
