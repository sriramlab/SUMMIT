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


def test_reused_trait_requires_the_exact_phenotype_batch_digest() -> None:
    axes = SimpleNamespace(
        n=5,
        m=3,
        sample_ids=("a", "b", "c", "d", "e"),
        variant_ids=("v1", "v2", "v3"),
        counted_alleles=("A", "C", "G"),
        other_alleles=("T", "G", "A"),
    )
    basis = np.column_stack(
        [np.ones(axes.n), np.linspace(-1.0, 1.0, axes.n)]
    )
    fixed = np.column_stack([np.ones(axes.n) / np.sqrt(axes.n)])
    annotations = np.ones((axes.m, 1))
    block_ids = np.asarray([0, 0, 1], dtype=np.int64)
    block_labels = ("block_0", "block_1")
    phenotypes = np.arange(10, dtype=np.float64).reshape(axes.n, 2)
    residual_basis, residual_names, _ = RUNNER.symmetric_context_residual_basis(
        basis
    )
    scale_digest = "a" * 64
    reference_run = SimpleNamespace(
        artifact=SimpleNamespace(
            scale_plan=SimpleNamespace(digest=scale_digest)
        )
    )
    retained_samples = np.arange(axes.n, dtype=np.int64)
    retained_variants = np.arange(axes.m, dtype=np.int64)
    identity = {
        "sample_order_sha256": RUNNER.canonical_sha256(
            {"ordered_iids": list(axes.sample_ids)}
        ),
        "variant_order_allele_sha256": (
            RUNNER.contextual_variant_order_allele_sha256_v1(
                retained_variants,
                axes.variant_ids,
                axes.counted_alleles,
                axes.other_alleles,
                np.ones(axes.m, dtype=np.uint8),
            )
        ),
        "retained_sample_map_sha256": RUNNER.array_sha256(retained_samples),
        "retained_variant_order_sha256": RUNNER.array_sha256(retained_variants),
        "fixed_effect_spec_sha256": RUNNER.array_sha256(fixed),
        "basis_specification_sha256": RUNNER.array_sha256(basis),
        "basis_calibration_sha256": RUNNER.array_sha256(basis.T @ basis),
        "fixed_basis_sha256": RUNNER.array_sha256(fixed),
        "evaluated_phi_sha256": RUNNER.array_sha256(basis),
        "genotype_scale_plan_sha256": scale_digest,
        "missingness_sha256": RUNNER.zero_missingness_sha256(axes.m, axes.n),
        "annotation_map_sha256": RUNNER.array_sha256(annotations),
        "group_map_sha256": RUNNER.array_sha256(block_ids),
        "phenotype_batch_sha256": RUNNER.array_sha256(phenotypes),
        "residual_basis_sha256": RUNNER.array_sha256(residual_basis),
    }
    trait = SimpleNamespace(
        manifest={"identity": identity},
        n_samples=axes.n,
        n_variants=axes.m,
        trait_ids=("t1", "t2"),
        residual_names=residual_names,
        component_index=SimpleNamespace(annotation_names=("all_variants",)),
        group_ids=block_labels,
        scale_plan=SimpleNamespace(digest=scale_digest),
    )
    arguments = {
        "trait": trait,
        "axes": axes,
        "reference_run": reference_run,
        "basis": basis,
        "fixed": fixed,
        "annotations": annotations,
        "block_ids": block_ids,
        "block_labels": block_labels,
        "phenotypes": phenotypes,
        "trait_names": ("t1", "t2"),
        "residual_basis": residual_basis,
        "residual_names": residual_names,
    }
    RUNNER._validate_reused_trait(**arguments)

    changed = dict(arguments)
    changed["phenotypes"] = phenotypes + 1.0
    with np.testing.assert_raises_regex(ValueError, "phenotype_batch_sha256"):
        RUNNER._validate_reused_trait(**changed)
