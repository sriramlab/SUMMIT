from __future__ import annotations

import json
from pathlib import Path

from summit.ldscore.generalized_gxe_reference_v1 import (
    write_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_variant_cli import build_parser, main
from test_generalized_gxe_reference_v1 import _artifact


def _plan_arguments(blocks: int) -> list[str]:
    return [
        "plan",
        "--samples",
        "200",
        "--variants",
        "1000",
        "--basis",
        "3",
        "--annotations",
        "2",
        "--probes",
        "16",
        "--jackknife-blocks",
        str(blocks),
        "--memory-bytes",
        str(2 * 1024**3),
        "--genotype-format",
        "bed",
        "--threads",
        "2",
        "--omit-per-variant-panel",
    ]


def test_help_names_variant_probe_two_pass_estimator() -> None:
    help_text = " ".join(build_parser().format_help().split())
    assert "summit-generalized-gxe-variant-ldscore" in help_text
    assert "variant-probe" in help_text
    assert "exactly-two-pass" in help_text
    assert "sample-probe contextual action estimator" in help_text


def test_dry_run_reports_exact_two_passes_and_canonical_jackknife(
    capsys,
) -> None:
    assert main(_plan_arguments(10)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["probe_axis"] == "variant"
    assert result["planned_reference_genotype_passes"] == 2
    assert result["work_plan"]["descriptor"]["planned_complete_passes"] == 2
    assert result["jackknife_method"] == (
        "frozen_full_genome_variant_ldscore_delete_block_v1"
    )


def test_inspect_validates_and_summarizes_closed_artifact(
    tmp_path: Path, capsys
) -> None:
    path = write_generalized_gxe_variant_reference_v1(
        _artifact(omit_panel=True), tmp_path / "reference"
    )
    assert main(["inspect", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["kind"] == (
        "summit.generalized_gxe.variant_ldscore_reference"
    )
    assert result["dimensions"] == {
        "N": 7,
        "M": 5,
        "Q": 3,
        "P": 6,
        "K": 2,
        "C": 12,
        "J": 2,
        "B": 11,
    }
    assert result["pass_ledger"]["observed_reference_genotype_passes"] == 2
    assert result["per_variant_panel"]["storage"] == "omitted"
