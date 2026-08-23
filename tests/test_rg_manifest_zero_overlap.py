from pathlib import Path

import pytest

from summit.manifest.rg_manifest_builder import build_rg_manifest


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    phen = tmp_path / "phen.tsv"
    phen.write_text(
        "FID\tIID\ttrait_a\ttrait_b\n"
        "1\t1\t1\t-9\n"
        "2\t2\t3\t-9\n"
        "3\t3\t-9\t2\n"
        "4\t4\t-9\t4\n"
    )
    phen_list = tmp_path / "phen_list.txt"
    phen_list.write_text("trait_a\ntrait_b\n")

    mapping = tmp_path / "sumstats.tsv"
    rows = ["phen\tsumstats"]
    for trait in ("trait_a", "trait_b"):
        path = tmp_path / f"{trait}.sumstat"
        path.write_text("SNP\tA1\tA2\tN\tBETA\tSE\tZ\n")
        rows.append(f"{trait}\t{path}")
    mapping.write_text("\n".join(rows) + "\n")
    return phen, phen_list, mapping


def test_zero_overlap_requires_explicit_opt_in(tmp_path: Path):
    phen, phen_list, mapping = _write_inputs(tmp_path)

    with pytest.raises(RuntimeError, match="Non-positive overlap counts"):
        build_rg_manifest(
            output_path=str(tmp_path / "strict.tsv"),
            phen_source=str(phen),
            sumstats_source=str(mapping),
            phen_list_path=str(phen_list),
            all_pairwise=True,
        )


def test_zero_overlap_has_exact_zero_overlap_covariance_when_allowed(tmp_path: Path):
    phen, phen_list, mapping = _write_inputs(tmp_path)

    result = build_rg_manifest(
        output_path=str(tmp_path / "allowed.tsv"),
        phen_source=str(phen),
        sumstats_source=str(mapping),
        phen_list_path=str(phen_list),
        all_pairwise=True,
        allow_zero_overlap=True,
    )

    assert result.shape[0] == 1
    assert int(result.loc[0, "n_overlap"]) == 0
    assert float(result.loc[0, "overlap_covariance"]) == 0.0
