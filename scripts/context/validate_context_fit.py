#!/usr/bin/env python3
"""Validate contextual fitting against explicit dense normal equations.

Four deterministic matched-cohort simulations exercise rank-one genetic
amplification, rank-two genetic heterogeneity, residual-only
heteroskedasticity, and two disjoint annotations.  Exact contextual reference
moments and trait summaries are fit through the public summary-only API, then
compared with a NumPy solve of normal equations formed from dense kernels.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    coefficients_to_omegas,
    common_scale_features,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    fit_context_model,
    omegas_to_coefficients,
    project_normalize_phenotype,
    rank_revealing_projector,
    symmetric_rank_diagnostics,
)


OUTPUT_STEM = "04_context_fit_validation"


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    label: str
    annotation_names: tuple[str, ...]
    true_omegas: np.ndarray
    residual_kind: str
    residual_coefficients: np.ndarray


@dataclass(frozen=True)
class CaseFixture:
    definition: CaseDefinition
    genotype: np.ndarray
    basis: np.ndarray
    fixed: np.ndarray
    projector: Any
    annotations: np.ndarray
    components: ContextComponentIndex
    residual_basis: np.ndarray
    residual_names: tuple[str, ...]
    phenotype: np.ndarray
    normalized_phenotype: np.ndarray
    genetic_kernels: np.ndarray
    residual_kernels: np.ndarray
    loo_group_ids: tuple[str, ...]
    context_grid: np.ndarray
    basis_metric: np.ndarray
    generating_covariance_minimum_eigenvalue: float
    phenotype_variance_normalization_multiplier: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=120, help="Samples per case.")
    parser.add_argument("--m", type=int, default=160, help="Variants per case.")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=64)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n < 32:
        parser.error("--n must be at least 32")
    if args.m < 24:
        parser.error("--m must be at least 24")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.loo_groups < 3 or args.loo_groups > args.m // 2:
        parser.error("--loo-groups must satisfy 3 <= groups <= M/2")
    if args.block_size < 1:
        parser.error("--block-size must be positive")


def _peak_rss_bytes() -> int:
    """Return the process resident-set high-water mark in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _timed(function: Callable[[], Any]) -> tuple[Any, float, int]:
    started = time.perf_counter()
    result = function()
    return result, float(time.perf_counter() - started), _peak_rss_bytes()


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True, dtype=np.float64)
    scales = np.std(centered, axis=0, ddof=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise RuntimeError("Synthetic fixture generated an invalid column scale.")
    return np.asarray(centered / scales[None, :], dtype=np.float64)


def _case_definitions() -> tuple[CaseDefinition, ...]:
    rank_one_loading = np.asarray([0.55, 0.28], dtype=np.float64)
    annotation_one_loading = np.asarray([0.34, 0.14], dtype=np.float64)
    return (
        CaseDefinition(
            name="rank_one_amplification",
            label="Rank-one",
            annotation_names=("all",),
            true_omegas=np.asarray(
                [np.outer(rank_one_loading, rank_one_loading)], dtype=np.float64
            ),
            residual_kind="constant",
            residual_coefficients=np.asarray([0.62], dtype=np.float64),
        ),
        CaseDefinition(
            name="rank_two_heterogeneity",
            label="Rank-two",
            annotation_names=("all",),
            true_omegas=np.asarray([[[0.30, 0.06], [0.06, 0.16]]], dtype=np.float64),
            residual_kind="constant",
            residual_coefficients=np.asarray([0.54], dtype=np.float64),
        ),
        CaseDefinition(
            name="residual_heteroskedasticity_only",
            label="Residual-only",
            annotation_names=("all",),
            true_omegas=np.asarray([[[0.35, 0.0], [0.0, 0.0]]], dtype=np.float64),
            residual_kind="quadratic",
            residual_coefficients=np.asarray([0.45, 0.20], dtype=np.float64),
        ),
        CaseDefinition(
            name="two_disjoint_annotations",
            label="K=2 disjoint",
            annotation_names=("even", "odd"),
            true_omegas=np.asarray(
                [
                    np.outer(annotation_one_loading, annotation_one_loading),
                    [[0.12, -0.025], [-0.025, 0.16]],
                ],
                dtype=np.float64,
            ),
            residual_kind="constant",
            residual_coefficients=np.asarray([0.585], dtype=np.float64),
        ),
    )


def _annotations(m: int, names: tuple[str, ...]) -> np.ndarray:
    weights = np.zeros((m, len(names)), dtype=np.float64)
    weights[np.arange(m), np.arange(m) % len(names)] = 1.0
    return weights


def _residual_basis(
    context: np.ndarray, kind: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    if kind == "constant":
        return np.ones((context.size, 1), dtype=np.float64), ("constant",)
    if kind == "quadratic":
        return (
            np.column_stack(
                [np.ones(context.size, dtype=np.float64), context * context]
            ),
            ("constant", "context_squared"),
        )
    raise ValueError(f"Unknown residual basis kind {kind!r}.")


def _sample_covariance(
    rng: np.random.Generator, covariance: np.ndarray, fixed: np.ndarray
) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-10 * scale:
        raise RuntimeError(
            "Synthetic generating covariance is not positive semidefinite."
        )
    stochastic = eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0.0)) * rng.standard_normal(eigenvalues.size)
    )
    fixed_signal = fixed @ np.linspace(0.05, 0.15, fixed.shape[1])
    return (
        np.asarray(stochastic + fixed_signal, dtype=np.float64),
        float(np.min(eigenvalues)),
    )


def _make_fixture(
    definition: CaseDefinition,
    *,
    case_index: int,
    args: argparse.Namespace,
) -> CaseFixture:
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, case_index]))
    genotype = _standardize_columns(rng.standard_normal((args.n, args.m)))
    context = _standardize_columns(rng.uniform(-1.0, 1.0, size=(args.n, 1)))[:, 0]
    nuisance = _standardize_columns(rng.standard_normal((args.n, 2)))
    basis = np.column_stack([np.ones(args.n, dtype=np.float64), context])
    fixed = np.column_stack([np.ones(args.n, dtype=np.float64), context, nuisance])
    projector = rank_revealing_projector(fixed)
    components = ContextComponentIndex(definition.annotation_names, ContextPairIndex(2))
    annotations = _annotations(args.m, definition.annotation_names)
    residual_basis, residual_names = _residual_basis(context, definition.residual_kind)
    features = common_scale_features(genotype, basis, projector.projector)
    genetic_kernels = dense_genetic_kernels(features, annotations, components)
    residual_kernels = dense_residual_kernels(projector.projector, residual_basis)
    true_genetic = omegas_to_coefficients(definition.true_omegas, components)
    covariance = np.einsum(
        "a,aij->ij", true_genetic, genetic_kernels, optimize=True
    ) + np.einsum(
        "a,aij->ij",
        definition.residual_coefficients,
        residual_kernels,
        optimize=True,
    )
    phenotype, minimum_eigenvalue = _sample_covariance(rng, covariance, fixed)
    projected_phenotype = projector.projector @ phenotype
    normalization_multiplier = projector.residual_rank / float(
        projected_phenotype @ projected_phenotype
    )
    normalized = project_normalize_phenotype(phenotype, projector)
    group_ids = tuple(f"block:{variant % args.loo_groups}" for variant in range(args.m))
    grid_values = np.linspace(-1.75, 1.75, 15)
    context_grid = np.column_stack(
        [np.ones(grid_values.size, dtype=np.float64), grid_values]
    )
    return CaseFixture(
        definition=definition,
        genotype=genotype,
        basis=np.asarray(basis, dtype=np.float64),
        fixed=np.asarray(fixed, dtype=np.float64),
        projector=projector,
        annotations=annotations,
        components=components,
        residual_basis=np.asarray(residual_basis, dtype=np.float64),
        residual_names=residual_names,
        phenotype=phenotype,
        normalized_phenotype=normalized,
        genetic_kernels=genetic_kernels,
        residual_kernels=residual_kernels,
        loo_group_ids=group_ids,
        context_grid=np.asarray(context_grid, dtype=np.float64),
        basis_metric=np.asarray(basis.T @ basis / args.n, dtype=np.float64),
        generating_covariance_minimum_eigenvalue=minimum_eigenvalue,
        phenotype_variance_normalization_multiplier=normalization_multiplier,
    )


def _identity_hashes(fixture: CaseFixture) -> dict[str, str]:
    return {
        "basis_hash": array_sha256(fixture.basis),
        "fixed_effect_hash": array_sha256(fixture.fixed),
        "variant_hash": canonical_sha256(
            {"ordered_synthetic_variants": list(range(fixture.genotype.shape[1]))}
        ),
    }


def _phase_times(value: Any) -> dict[str, float]:
    raw = getattr(value, "phase_times_seconds", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for name, duration in raw.items():
        if isinstance(duration, bool):
            continue
        try:
            converted = float(duration)
        except (TypeError, ValueError):
            continue
        if np.isfinite(converted) and converted >= 0.0:
            result[str(name)] = converted
    return result


def _maximum_absolute(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0))


def _maximum_scale_aware(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    scale = np.maximum(1.0, np.maximum(np.abs(left_array), np.abs(right_array)))
    return float(np.max(np.abs(left_array - right_array) / scale, initial=0.0))


def _covariance_surfaces(omegas: np.ndarray, context_grid: np.ndarray) -> np.ndarray:
    return np.asarray(
        [context_grid @ omega @ context_grid.T for omega in omegas],
        dtype=np.float64,
    )


def _context_output_surfaces(output: dict[str, Any] | None) -> np.ndarray:
    """Extract public covariance surfaces across fit-schema representations."""
    if output is None:
        raise RuntimeError("Context fitting did not return requested derived outputs.")
    if "covariance_surfaces" in output:
        return np.asarray(output["covariance_surfaces"], dtype=np.float64)
    annotations = output.get("annotations")
    if isinstance(annotations, list):
        try:
            return np.asarray(
                [entry["covariance_surface"] for entry in annotations],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Malformed annotation-level context outputs.") from exc
    raise RuntimeError("Context fit output does not contain covariance surfaces.")


def _dense_solve(fixture: CaseFixture) -> tuple[Any, Any, np.ndarray, float]:
    equations = dense_normal_equations(
        fixture.genetic_kernels,
        fixture.residual_kernels,
        fixture.normalized_phenotype,
        fixture.components.names,
        fixture.residual_names,
    )
    diagnostics = symmetric_rank_diagnostics(equations.matrix)
    if diagnostics.rank != equations.matrix.shape[0]:
        raise RuntimeError(
            f"Dense case {fixture.definition.name!r} is rank deficient: "
            f"rank {diagnostics.rank}/{equations.matrix.shape[0]}."
        )
    coefficients = np.linalg.solve(
        0.5 * (equations.matrix + equations.matrix.T), equations.rhs
    )
    denominator = max(float(np.linalg.norm(equations.rhs)), 1.0)
    relative_residual = float(
        np.linalg.norm(equations.matrix @ coefficients - equations.rhs) / denominator
    )
    return equations, diagnostics, coefficients, relative_residual


def _run_case(
    definition: CaseDefinition,
    *,
    case_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    fixture, fixture_seconds, fixture_peak = _timed(
        lambda: _make_fixture(definition, case_index=case_index, args=args)
    )
    hashes = _identity_hashes(fixture)

    summary, summary_seconds, summary_peak = _timed(
        lambda: build_context_trait_summary(
            genotype=fixture.genotype,
            basis=fixture.basis,
            phenotype=fixture.phenotype,
            projector=fixture.projector,
            annotations=fixture.annotations,
            component_index=fixture.components,
            residual_basis=fixture.residual_basis,
            residual_names=fixture.residual_names,
            loo_groups=fixture.loo_group_ids,
            genotype_scaling="sample_sd_ddof1_pre_scaled_input",
            block_size=args.block_size,
            **hashes,
        )
    )
    reference, reference_seconds, reference_peak = _timed(
        lambda: build_context_reference(
            genotype=fixture.genotype,
            basis=fixture.basis,
            projector=fixture.projector,
            annotations=fixture.annotations,
            component_index=fixture.components,
            loo_groups=fixture.loo_group_ids,
            genotype_scaling="sample_sd_ddof1_pre_scaled_input",
            gram_method="exact",
            same_person_method="exact",
            probe_tile_size=8,
            **hashes,
        )
    )
    (
        (
            dense_equations,
            dense_diagnostics,
            dense_coefficients,
            dense_relative_residual,
        ),
        dense_seconds,
        dense_peak,
    ) = _timed(lambda: _dense_solve(fixture))

    jackknife_groups = tuple(dict.fromkeys(fixture.loo_group_ids))
    fit, fit_seconds, fit_peak = _timed(
        lambda: fit_context_model(
            reference,
            summary,
            loo_groups=jackknife_groups,
            context_grid=fixture.context_grid,
            basis_metric=fixture.basis_metric,
            project_psd=False,
            annotations_disjoint=True,
        )
    )

    p_genetic = len(fixture.components)
    dense_omegas = coefficients_to_omegas(
        dense_coefficients[:p_genetic], fixture.components
    )
    summary_omegas = np.asarray(fit.raw_omegas, dtype=np.float64)
    scaled_true_omegas = (
        fixture.phenotype_variance_normalization_multiplier * definition.true_omegas
    )
    scaled_residual_coefficients = (
        fixture.phenotype_variance_normalization_multiplier
        * definition.residual_coefficients
    )
    true_surfaces = _covariance_surfaces(scaled_true_omegas, fixture.context_grid)
    dense_surfaces = _covariance_surfaces(dense_omegas, fixture.context_grid)
    raw_omega_surfaces = _covariance_surfaces(summary_omegas, fixture.context_grid)
    summary_surfaces = _context_output_surfaces(fit.context_outputs)
    true_coefficients = np.concatenate(
        [
            omegas_to_coefficients(scaled_true_omegas, fixture.components),
            scaled_residual_coefficients,
        ]
    )
    summary_coefficients = np.asarray(fit.solve.coefficients, dtype=np.float64)
    jackknife_covariance = np.asarray(fit.jackknife_covariance, dtype=np.float64)
    jackknife_eigenvalues = np.linalg.eigvalsh(
        0.5 * (jackknife_covariance + jackknife_covariance.T)
    )
    jackknife_surface_internal_errors = []
    for replicate_coefficients, output in zip(
        fit.jackknife_coefficients, fit.jackknife_context_outputs
    ):
        replicate_omegas = coefficients_to_omegas(
            np.asarray(replicate_coefficients[:p_genetic], dtype=np.float64),
            fixture.components,
        )
        direct_surfaces = _covariance_surfaces(replicate_omegas, fixture.context_grid)
        jackknife_surface_internal_errors.append(
            _maximum_absolute(_context_output_surfaces(output), direct_surfaces)
        )
    fit_diagnostics = fit.solve.diagnostics

    discrepancies = {
        "normal_matrix_max_abs": _maximum_absolute(
            fit.equations.matrix, dense_equations.matrix
        ),
        "normal_matrix_scale_aware": _maximum_scale_aware(
            fit.equations.matrix, dense_equations.matrix
        ),
        "rhs_max_abs": _maximum_absolute(fit.equations.rhs, dense_equations.rhs),
        "rhs_scale_aware": _maximum_scale_aware(fit.equations.rhs, dense_equations.rhs),
        "traces_max_abs": _maximum_absolute(
            fit.equations.traces, dense_equations.traces
        ),
        "coefficients_max_abs": _maximum_absolute(
            summary_coefficients, dense_coefficients
        ),
        "coefficients_scale_aware": _maximum_scale_aware(
            summary_coefficients, dense_coefficients
        ),
        "surfaces_max_abs": _maximum_absolute(summary_surfaces, dense_surfaces),
        "surfaces_scale_aware": _maximum_scale_aware(summary_surfaces, dense_surfaces),
        "derived_surfaces_vs_raw_omegas_max_abs": _maximum_absolute(
            summary_surfaces, raw_omega_surfaces
        ),
        "jackknife_derived_surfaces_vs_coefficients_max_abs": max(
            jackknife_surface_internal_errors, default=0.0
        ),
        "dense_coefficients_vs_generating_max_abs": _maximum_absolute(
            dense_coefficients, true_coefficients
        ),
        "dense_surfaces_vs_generating_max_abs": _maximum_absolute(
            dense_surfaces, true_surfaces
        ),
    }
    generating_omega_eigenvalues = np.asarray(
        [np.linalg.eigvalsh(omega) for omega in scaled_true_omegas],
        dtype=np.float64,
    )
    dense_omega_eigenvalues = np.asarray(
        [np.linalg.eigvalsh(omega) for omega in dense_omegas], dtype=np.float64
    )
    summary_omega_eigenvalues = np.asarray(
        [np.linalg.eigvalsh(omega) for omega in summary_omegas], dtype=np.float64
    )

    return {
        "name": definition.name,
        "label": definition.label,
        "dimensions": {
            "n": args.n,
            "m": args.m,
            "q": 2,
            "k": len(definition.annotation_names),
            "h": len(fixture.residual_names),
            "p_genetic": p_genetic,
            "p_total": dense_coefficients.size,
        },
        "component_names": list(dense_equations.component_names),
        "annotation_names": list(definition.annotation_names),
        "residual_names": list(fixture.residual_names),
        "generating_coefficients": true_coefficients.tolist(),
        "dense_coefficients": dense_coefficients.tolist(),
        "summary_coefficients": summary_coefficients.tolist(),
        "generating_omegas_before_phenotype_normalization": (
            definition.true_omegas.tolist()
        ),
        "generating_omegas": scaled_true_omegas.tolist(),
        "dense_omegas": dense_omegas.tolist(),
        "summary_omegas": summary_omegas.tolist(),
        "context_grid": fixture.context_grid.tolist(),
        "surfaces": {
            "generating": true_surfaces.tolist(),
            "dense": dense_surfaces.tolist(),
            "summary": summary_surfaces.tolist(),
        },
        "discrepancies": discrepancies,
        "mechanism_diagnostics": {
            "generating_omega_eigenvalues": generating_omega_eigenvalues.tolist(),
            "dense_omega_eigenvalues": dense_omega_eigenvalues.tolist(),
            "summary_omega_eigenvalues": summary_omega_eigenvalues.tolist(),
            "generating_omega_ranks": [
                int(np.sum(eigenvalues > 1.0e-12))
                for eigenvalues in generating_omega_eigenvalues
            ],
        },
        "normal_equation_diagnostics": {
            "dense_rank": dense_diagnostics.rank,
            "summary_rank": fit_diagnostics.rank,
            "dimension": dense_equations.matrix.shape[0],
            "dense_condition_number": dense_diagnostics.condition_number,
            "summary_condition_number": fit_diagnostics.condition_number,
            "dense_minimum_eigenvalue": float(np.min(dense_diagnostics.eigenvalues)),
            "summary_minimum_eigenvalue": float(np.min(fit_diagnostics.eigenvalues)),
            "dense_relative_solve_residual": dense_relative_residual,
            "summary_relative_solve_residual": float(fit.solve.relative_residual),
            "generating_covariance_minimum_eigenvalue": (
                fixture.generating_covariance_minimum_eigenvalue
            ),
            "phenotype_variance_normalization_multiplier": (
                fixture.phenotype_variance_normalization_multiplier
            ),
        },
        "jackknife": {
            "groups": list(fit.jackknife_groups),
            "replicate_count": int(fit.jackknife_coefficients.shape[0]),
            "covariance_eigenvalues": jackknife_eigenvalues.tolist(),
            "minimum_covariance_eigenvalue": float(np.min(jackknife_eigenvalues)),
            "maximum_covariance_eigenvalue": float(
                np.max(jackknife_eigenvalues, initial=0.0)
            ),
            "standard_errors": np.asarray(fit.standard_errors).tolist(),
        },
        "runtime_seconds": {
            "fixture_and_dense_kernels": fixture_seconds,
            "trait_summary": summary_seconds,
            "reference": reference_seconds,
            "dense_normal_equation_and_solve": dense_seconds,
            "summary_fit": fit_seconds,
            "total": float(time.perf_counter() - total_started),
            "trait_summary_internal": _phase_times(summary),
            "reference_internal": _phase_times(reference),
            "fit_internal": _phase_times(fit),
        },
        "absolute_peak_rss_bytes": {
            "after_fixture": fixture_peak,
            "after_trait_summary": summary_peak,
            "after_reference": reference_peak,
            "after_dense_solve": dense_peak,
            "after_summary_fit": fit_peak,
            "trait_summary_reported": int(summary.peak_rss_bytes),
            "reference_reported": int(reference.peak_rss_bytes),
            "fit_reported": int(fit.peak_rss_bytes),
        },
    }


def _plot(records: list[dict[str, Any]], output_dir: Path) -> None:
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)

    all_dense: list[float] = []
    all_summary: list[float] = []
    for index, record in enumerate(records):
        dense = np.asarray(record["dense_coefficients"], dtype=np.float64)
        summary = np.asarray(record["summary_coefficients"], dtype=np.float64)
        all_dense.extend(dense.tolist())
        all_summary.extend(summary.tolist())
        axes[0].scatter(
            dense,
            summary,
            s=22,
            color=colors[index],
            alpha=0.85,
            label=record["label"],
        )
    bounds = np.asarray([*all_dense, *all_summary], dtype=np.float64)
    span = max(float(np.ptp(bounds)), 1.0e-6)
    lower = float(np.min(bounds) - 0.06 * span)
    upper = float(np.max(bounds) + 0.06 * span)
    axes[0].plot([lower, upper], [lower, upper], color="0.25", linestyle="--")
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xlabel("Dense coefficient")
    axes[0].set_ylabel("Summary-fit coefficient")
    axes[0].legend(frameon=False, fontsize=7)

    positions = np.arange(len(records), dtype=np.float64)
    coefficient_errors = np.asarray(
        [record["discrepancies"]["coefficients_max_abs"] for record in records]
    )
    surface_errors = np.asarray(
        [record["discrepancies"]["surfaces_max_abs"] for record in records]
    )
    positive = np.concatenate(
        [
            coefficient_errors[coefficient_errors > 0.0],
            surface_errors[surface_errors > 0.0],
        ]
    )
    floor = (
        float(np.min(positive)) * 0.5
        if positive.size
        else float(np.finfo(np.float64).eps)
    )
    axes[1].plot(
        positions,
        np.maximum(coefficient_errors, floor),
        "o-",
        color="#1f6f8b",
        label="coefficients",
    )
    axes[1].plot(
        positions,
        np.maximum(surface_errors, floor),
        "s--",
        color="#d97706",
        label="surfaces",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum absolute discrepancy")
    axes[1].set_xticks(positions, [record["label"] for record in records])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=8)

    line_index = 0
    for case_index, record in enumerate(records):
        grid = np.asarray(record["context_grid"], dtype=np.float64)[:, 1]
        generating = np.asarray(record["surfaces"]["generating"], dtype=np.float64)
        summary = np.asarray(record["surfaces"]["summary"], dtype=np.float64)
        for annotation_index, annotation_name in enumerate(record["annotation_names"]):
            color = colors[line_index % len(colors)]
            label = record["label"]
            if len(record["annotation_names"]) > 1:
                label = f"{label}: {annotation_name}"
            axes[2].plot(
                grid,
                np.diag(summary[annotation_index]),
                color=color,
                linewidth=1.6,
                label=label,
            )
            axes[2].plot(
                grid,
                np.diag(generating[annotation_index]),
                color=color,
                linewidth=1.1,
                linestyle="--",
                alpha=0.8,
            )
            line_index += 1
    axes[2].axhline(0.0, color="0.7", linewidth=0.8)
    axes[2].set_xlabel("Context value")
    axes[2].set_ylabel("Genetic variance surface diagonal")
    mechanism_legend = axes[2].legend(frameon=False, fontsize=6, loc="best")
    axes[2].add_artist(mechanism_legend)
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="0.25", label="summary fit"),
            Line2D([0], [0], color="0.25", linestyle="--", label="generating target"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Matched-cohort contextual fit validation")
    figure.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300)
    figure.savefig(output_dir / f"{OUTPUT_STEM}.pdf")
    plt.close(figure)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    records = [
        _run_case(definition, case_index=index, args=args)
        for index, definition in enumerate(_case_definitions())
    ]
    aggregate = {
        "maximum_coefficient_discrepancy": max(
            record["discrepancies"]["coefficients_max_abs"] for record in records
        ),
        "maximum_surface_discrepancy": max(
            record["discrepancies"]["surfaces_max_abs"] for record in records
        ),
        "all_normal_equations_full_rank": all(
            record["normal_equation_diagnostics"]["summary_rank"]
            == record["normal_equation_diagnostics"]["dimension"]
            for record in records
        ),
        "maximum_normal_equation_condition_number": max(
            record["normal_equation_diagnostics"]["summary_condition_number"]
            for record in records
        ),
        "minimum_jackknife_covariance_eigenvalue": min(
            record["jackknife"]["minimum_covariance_eigenvalue"] for record in records
        ),
    }
    aggregate["dense_equivalence_passed"] = bool(
        aggregate["maximum_coefficient_discrepancy"] <= 1.0e-10
        and aggregate["maximum_surface_discrepancy"] <= 1.0e-10
        and aggregate["all_normal_equations_full_rank"]
    )
    payload = {
        "kind": "summit.context.fit_validation",
        "schema_version": 1,
        "seed": args.seed,
        "configuration": {
            "n": args.n,
            "m": args.m,
            "loo_groups": args.loo_groups,
            "block_size": args.block_size,
            "matched_reference_and_study": True,
            "reference_gram_method": "exact",
            "reference_same_person_method": "exact",
            "genotype_scaling": "sample_sd_ddof1_pre_scaled_input",
        },
        "aggregate": aggregate,
        "total_runtime_seconds": float(time.perf_counter() - total_started),
        "absolute_peak_rss_bytes": _peak_rss_bytes(),
        "cases": records,
    }
    (args.output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(records, args.output_dir)


if __name__ == "__main__":
    main()
