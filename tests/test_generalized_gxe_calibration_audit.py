from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


RUNNER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generalized_gxe"
    / "run_simulation_benchmarks.py"
)
sys.path.insert(0, str(RUNNER_SCRIPT.parent))
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "generalized_gxe_simulation_runner", RUNNER_SCRIPT
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def _gate_fixture(null_rate: float, trait_count: int = 100) -> dict:
    diagonal_summary = {
        **{
            f"omega_{index}_{index}": {"null_rejection_rate_5pct": 0.9}
            for index in range(3)
        },
        **{
            name: {
                "null_rejection_rate_5pct": null_rate,
                "empirical_error_sd_over_mean_jackknife_se": 1.0,
            }
            for name in ("omega_0_1", "omega_0_2", "omega_1_2")
        },
    }
    return {
        "diagonal_two_environment": {
            "trait_count": trait_count,
            "summary": diagonal_summary,
        },
        "offdiagonal_two_environment": {
            "trait_count": trait_count,
            "summary": {
                "omega_1_2": {
                    "mean_estimate": 0.18,
                    "mean_truth": 0.18,
                    "bias": 0.0,
                    "null_rejection_rate_5pct": 0.9,
                }
            },
            "omega_1_2_available_in_diagonal_restriction": False,
        },
    }


def test_hundred_replicate_gate_is_dynamic_and_tightens_null_ceiling() -> None:
    passing = RUNNER._stop_gates(_gate_fixture(0.10), 100)
    assert passing["all_passed"]
    assert passing["diagonal"]["requested_replicates"] == 100
    assert passing["diagonal"]["all_requested_fits_completed"]
    assert passing["diagonal"]["offdiagonal_false_positive_rate_ceiling"] == 0.10
    assert passing["diagonal"][
        "all_null_empirical_se_ratios_between_0p75_and_1p35"
    ]

    failing = RUNNER._stop_gates(_gate_fixture(0.11), 100)
    assert not failing["diagonal"][
        "maximum_offdiagonal_false_positive_rate_within_ceiling"
    ]
    assert not failing["all_passed"]

    ratio_failure_fixture = _gate_fixture(0.05)
    ratio_failure_fixture["diagonal_two_environment"]["summary"]["omega_1_2"][
        "empirical_error_sd_over_mean_jackknife_se"
    ] = 1.36
    ratio_failure = RUNNER._stop_gates(ratio_failure_fixture, 100)
    assert not ratio_failure["diagonal"][
        "all_null_empirical_se_ratios_between_0p75_and_1p35"
    ]
    assert not ratio_failure["all_passed"]


def test_reused_trait_checks_concrete_ordered_axes_and_block_sums() -> None:
    axes = SimpleNamespace(
        n=5,
        m=3,
        sample_ids=("a", "b", "c", "d", "e"),
        variant_ids=("v1", "v2", "v3"),
        counted_alleles=("A", "C", "G"),
        other_alleles=("T", "G", "A"),
    )
    annotations = np.ones((axes.m, 1))
    block_ids = np.asarray([0, 0, 1], dtype=np.int64)
    block_labels = ("block_0", "block_1")
    residual_names = ("residual",)
    trait = SimpleNamespace(
        n_samples=axes.n,
        n_variants=axes.m,
        trait_ids=("t1", "t2"),
        residual_names=residual_names,
        component_index=SimpleNamespace(annotation_names=("all_variants",)),
        group_ids=block_labels,
        annotation_masses=np.asarray([3.0]),
        group_variant_counts=np.asarray([2, 1]),
    )
    arguments = {
        "trait": trait,
        "axes": axes,
        "annotations": annotations,
        "inference_block_ids": block_ids,
        "inference_block_labels": block_labels,
        "trait_names": ("t1", "t2"),
        "residual_names": residual_names,
    }
    RUNNER._validate_reused_trait(**arguments)

    wrong_trait = SimpleNamespace(**trait.__dict__)
    wrong_trait.trait_ids = ("t1", "wrong")
    changed = dict(arguments, trait=wrong_trait)
    with np.testing.assert_raises_regex(ValueError, "trait_ids"):
        RUNNER._validate_reused_trait(**changed)
