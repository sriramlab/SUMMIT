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


@pytest.mark.parametrize("available_gib, expected_gib", [(8, 4), (10, 6), (20, 13)])
def test_auto_memory_budget_keeps_fractional_and_fixed_margin(
    monkeypatch, available_gib, expected_gib
):
    available = available_gib * 1024**3
    monkeypatch.setattr(
        utils,
        "available_memory_bytes",
        lambda: (available, {"limiting_source": "test", "candidates_bytes": {"test": available}}),
    )
    budget, evidence = utils.resolve_memory_budget_gib("auto")
    assert math.isclose(budget, expected_gib)
    assert evidence["mode"] == "auto"
    assert evidence["limiting_source"] == "test"
    assert evidence["reserve_gib"] == 4.0


@pytest.mark.parametrize("available_gib", [0.5, 1, 2, 3, 4, 6])
def test_auto_memory_budget_accepts_small_allocations_with_headroom(
    monkeypatch, available_gib
):
    available = int(available_gib * 1024**3)
    monkeypatch.setattr(
        utils, "available_memory_bytes",
        lambda: (available, {"limiting_source": "host_available"}),
    )
    budget, evidence = utils.resolve_memory_budget_gib("auto")
    assert budget == available_gib / 2
    assert evidence["available_gib"] == available_gib
    assert evidence["reserve_gib"] == available_gib / 2
    assert evidence["requested_reserve_gib"] == 4.0


def test_auto_memory_budget_still_rejects_insufficient_memory(monkeypatch):
    monkeypatch.setattr(
        utils, "available_memory_bytes",
        lambda: (256 * 1024**2, {"limiting_source": "RLIMIT_AS"}),
    )
    with pytest.raises(RuntimeError, match="less than 0.25 GiB") as error:
        utils.resolve_memory_budget_gib("auto")
    assert "available=0.250 GiB" in str(error.value)
    assert "limiting_source='RLIMIT_AS'" in str(error.value)
    assert "reserve=0.125 GiB" in str(error.value)


def test_explicit_memory_budget_does_not_probe_available_memory(monkeypatch):
    def reject_probe():
        pytest.fail("an explicit panel budget must not invoke auto budgeting")

    monkeypatch.setattr(utils, "available_memory_bytes", reject_probe)
    assert utils.resolve_memory_budget_gib(0.125) == (
        0.125, {"mode": "explicit", "resolved_gib": 0.125}
    )


def test_cli_defaults_to_auto_memory_and_semantic_mailman_selection():
    defaults = cli.build_parser().parse_args([])
    assert defaults.memory_gib == "auto"
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
