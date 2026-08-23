from __future__ import annotations

import math

import pytest

from summit import cli, utils
from summit.ldscore.gw_ldscore import _bounded_vtile_sizes, _resolve_mailman_selection


def test_memory_budget_parser_accepts_auto_and_positive_numbers():
    assert utils.parse_memory_budget("auto") == "auto"
    assert utils.parse_memory_budget("7.5") == 7.5
    for value in ("0", "-1", "nan", "nope"):
        with pytest.raises(ValueError):
            utils.parse_memory_budget(value)


def test_auto_memory_budget_keeps_fractional_and_fixed_margin(monkeypatch):
    monkeypatch.setattr(
        utils,
        "available_memory_bytes",
        lambda: (20 * 1024**3, {"limiting_source": "test", "candidates_bytes": {"test": 20 * 1024**3}}),
    )
    budget, evidence = utils.resolve_memory_budget_gib("auto")
    assert math.isclose(budget, 13.0)
    assert evidence["mode"] == "auto"
    assert evidence["limiting_source"] == "test"


def test_cli_defaults_to_auto_memory_and_semantic_mailman_selection():
    defaults = cli.build_parser().parse_args([])
    assert defaults.target_xz_mem == "auto"
    assert defaults.target_mem is None
    assert defaults.gxe_total_memory_gib == "auto"
    assert defaults.use_mailman == "auto"

    explicit = cli.build_parser().parse_args(
        ["--gxe-total-memory-gib", "24.5"]
    )
    assert explicit.gxe_total_memory_gib == 24.5


def test_mailman_auto_is_small_probe_and_hwe_only():
    assert _resolve_mailman_selection("auto", nvecs=10, impute_method="hwe") == (
        True,
        "auto",
    )
    assert _resolve_mailman_selection("auto", nvecs=11, impute_method="hwe") == (
        False,
        "auto",
    )
    assert _resolve_mailman_selection("auto", nvecs=10, impute_method="mean") == (
        False,
        "auto",
    )
    assert _resolve_mailman_selection(True, nvecs=1024, impute_method="hwe") == (
        True,
        "explicit",
    )


def test_ordinary_probe_tiles_never_exceed_memory_derived_ceiling():
    assert _bounded_vtile_sizes(1_024, 1) == [1] * 1_024
    sizes = _bounded_vtile_sizes(1_024, 130)
    assert sum(sizes) == 1_024
    assert max(sizes) <= 130
    assert sizes[:-1] == [128] * 7
