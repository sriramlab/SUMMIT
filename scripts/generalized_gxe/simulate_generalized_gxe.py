#!/usr/bin/env python3
"""Simulate generalized GxE phenotypes in one fixed-genotype traversal.

For SNP j, the simulator draws a Q-vector

    beta_j ~ N(0, Omega / M)

and generates

    y_i = sum_q phi_iq * (G beta_q)_i + C_i gamma + phi_i' u_i,
    u_i ~ N(0, Psi).

All contexts use the same mean-imputed, sample-SD genotype scale.  Replicates
share one environment realization so their phenotype summaries can be formed
by one native multi-phenotype traversal.  By default, only ``Psi[0,0]`` is
nonzero and the noise is homoskedastic.  ``--psi`` instead permits a full PSD
contextual residual covariance.  The saved inference design includes the
rank-reduced span of symmetric columns ``eta_qr * phi_q * phi_r`` and records
the corresponding normalized truth coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from bed_reader import open_bed

from workflow import (
    canonical_json,
    fixed_basis,
    rank_reduced_symmetric_context_residual_basis,
    read_plink_axes,
    standardized,
    symmetric_context_residual_basis,
)


SCHEMA = "summit.generalized_gxe.simulation_batch_v1"


def validate_omega(value: Any, q: int) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(value, dtype=np.float64)
    if omega.shape != (q, q) or not np.all(np.isfinite(omega)):
        raise ValueError(f"Omega must be a finite {q}-by-{q} matrix")
    scale = max(float(np.max(np.abs(omega), initial=0.0)), 1.0)
    if not np.allclose(omega, omega.T, rtol=0.0, atol=1.0e-12 * scale):
        raise ValueError("Omega must be symmetric")
    omega = 0.5 * (omega + omega.T)
    eigenvalues, eigenvectors = np.linalg.eigh(omega)
    if eigenvalues[0] < -1.0e-12 * scale:
        raise ValueError("Omega must be positive semidefinite")
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    reconstructed = root @ root.T
    if not np.allclose(reconstructed, omega, rtol=2.0e-13, atol=2.0e-13 * scale):
        raise RuntimeError("Omega square root did not reconstruct the input")
    return omega, root


def generate_environment(
    n: int,
    seed_sequence: np.random.SeedSequence,
    environment_type: str = "gaussian_gaussian",
    correlation: float = 0.0,
) -> np.ndarray:
    """Generate two centered, unit-SD continuous or binary contexts."""
    if environment_type not in {
        "gaussian_gaussian",
        "gaussian_binary",
        "binary_binary",
    }:
        raise ValueError("unsupported environment type")
    if not np.isfinite(correlation) or abs(correlation) >= 1.0:
        raise ValueError("environment correlation must lie strictly between -1 and 1")
    rng = np.random.default_rng(seed_sequence)
    first = standardized(rng.standard_normal(n))
    second = rng.standard_normal(n)
    second -= np.mean(second, dtype=np.float64)
    second -= first * (first @ second) / (first @ first)
    second = standardized(second)
    second = correlation * first + np.sqrt(1.0 - correlation**2) * second
    if environment_type == "gaussian_binary":
        second = standardized((second > 0.0).astype(np.float64))
    elif environment_type == "binary_binary":
        first = standardized((first > 0.0).astype(np.float64))
        second = standardized((second > 0.0).astype(np.float64))
    environment = np.asfortranarray(np.column_stack([first, second]))
    if any(np.unique(environment[:, index]).size < 2 for index in range(2)):
        raise RuntimeError("an environment realization is constant")
    diagonal = np.diag(environment.T @ environment)
    if not np.allclose(diagonal, n - 1.0, rtol=2.0e-13, atol=2.0e-10):
        raise RuntimeError("environment calibration failed")
    return environment


def standardize_genotype_block(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-impute and sample-standardize one A1-dosage block in place."""
    values = np.asarray(raw, dtype=np.float64, order="F")
    observed = np.sum(~np.isnan(values), axis=0, dtype=np.int64)
    if np.any(observed < 2):
        raise ValueError("a genotype column has fewer than two observed calls")
    # Hard calls are integers.  Use the same integer sufficient statistics as
    # the native decoder so the sealed affine vectors agree bit for bit.
    totals = np.asarray(
        np.nansum(values, axis=0, dtype=np.float64), dtype=np.int64
    )
    total_squares = np.asarray(
        np.nansum(values * values, axis=0, dtype=np.float64), dtype=np.int64
    )
    means = totals.astype(np.float64) / observed
    compact_totals = 2 * observed - totals
    compact_total_squares = 4 * observed - 4 * totals + total_squares
    compact_means = compact_totals.astype(np.float64) / observed
    # DirectContext decodes the compact BED allele and returns
    # compact_mean-compact_value, which is the centered BIM-A1 dosage.
    values[:] = compact_means - (2.0 - values)
    np.nan_to_num(values, copy=False, nan=0.0)
    sums_of_squares = compact_total_squares.astype(np.float64) - (
        compact_totals.astype(np.float64)
        * compact_totals.astype(np.float64)
        / observed
    )
    if np.any(sums_of_squares <= 0.0):
        raise ValueError("a genotype column is monomorphic after sample selection")
    inverse = np.sqrt((values.shape[0] - 1.0) / sums_of_squares)
    values *= inverse
    missing = values.shape[0] - observed
    return values, means, inverse, missing


def _parse_omega(token: str, q: int) -> np.ndarray:
    path = Path(token)
    text = path.read_text() if path.is_file() else token
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("--omega must be JSON or a path to JSON") from exc
    return validate_omega(value, q)[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--causal-fraction", type=float, default=1.0)
    parser.add_argument("--residual-variance", type=float, default=0.4)
    parser.add_argument(
        "--psi",
        help=(
            "optional Q=3 PSD residual-effect covariance as JSON or a JSON file; "
            "defaults to homoskedastic residual variance in the intercept component"
        ),
    )
    parser.add_argument(
        "--omega",
        default="[[0.2,0,0],[0,0.2,0],[0,0,0.2]]",
        help="Q=3 PSD effect-covariance matrix as JSON or a JSON file",
    )
    parser.add_argument(
        "--fixed-environment-effects", type=float, nargs=2, default=(0.25, -0.15)
    )
    parser.add_argument(
        "--environment-type",
        choices=("gaussian_gaussian", "gaussian_binary", "binary_binary"),
        default="gaussian_gaussian",
    )
    parser.add_argument("--environment-correlation", type=float, default=0.0)
    parser.add_argument("--label", default="diagonal_two_environment")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {args.output}")
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if not (0.0 < args.causal_fraction <= 1.0):
        raise ValueError("--causal-fraction must lie in (0,1]")
    if not np.isfinite(args.residual_variance) or args.residual_variance <= 0.0:
        raise ValueError("--residual-variance must be finite and positive")
    axes = read_plink_axes(args.prefix)
    n, m, q, r = axes.n, axes.m, 3, args.replicates
    omega, omega_root = validate_omega(_parse_omega(args.omega, q), q)
    if args.psi is None:
        psi = np.diag([args.residual_variance, 0.0, 0.0])
    else:
        psi = _parse_omega(args.psi, q)
    psi, psi_root = validate_omega(psi, q)
    seed_sequence = np.random.SeedSequence(args.seed)
    environment_seed, effect_seed, causal_seed, noise_seed = seed_sequence.spawn(4)
    environment = generate_environment(
        n,
        environment_seed,
        args.environment_type,
        args.environment_correlation,
    )
    basis = np.asfortranarray(
        np.column_stack([np.ones(n, dtype=np.float64), environment])
    )
    if "binary" in args.environment_type:
        residual_basis, residual_names, residual_pairs = (
            rank_reduced_symmetric_context_residual_basis(
                basis, ("intercept", "environment_1", "environment_2")
            )
        )
    else:
        residual_basis, residual_names, residual_pairs = (
            symmetric_context_residual_basis(basis)
        )
    fixed_design = np.column_stack([np.ones(n, dtype=np.float64), environment])
    fixed = fixed_basis(fixed_design)
    residual_rank = n - fixed.shape[1]
    fixed_effect = environment @ np.asarray(args.fixed_environment_effects)

    effect_rng = np.random.default_rng(effect_seed)
    innovations = effect_rng.standard_normal((m, r, q))
    causal_rng = np.random.default_rng(causal_seed)
    if args.causal_fraction < 1.0:
        causal = causal_rng.random((m, r)) < args.causal_fraction
        counts = np.sum(causal, axis=0, dtype=np.int64)
        if np.any(counts == 0):
            raise RuntimeError("causal-fraction draw left a replicate with no causal SNPs")
    else:
        causal = None
        counts = np.full(r, m, dtype=np.int64)

    component_scores = np.zeros((n, r, q), dtype=np.float64)
    realized_effect_covariance = np.zeros((r, q, q), dtype=np.float64)
    means = np.empty(m, dtype=np.float64)
    inverse_scales = np.empty(m, dtype=np.float64)
    missing_counts = np.empty(m, dtype=np.int64)
    blocks = 0
    visits = 0
    started = time.perf_counter()
    with open_bed(str(axes.prefix) + ".bed", count_A1=True, num_threads=1) as bed:
        if (bed.iid_count, bed.sid_count) != (n, m):
            raise RuntimeError("BED dimensions changed after preflight")
        for start in range(0, m, args.chunk_size):
            stop = min(m, start + args.chunk_size)
            raw = bed.read(index=np.s_[:, start:stop], dtype="float64", order="F")
            genotype, block_mean, block_inverse, block_missing = standardize_genotype_block(raw)
            width = stop - start
            block_effects = innovations[start:stop] @ omega_root.T
            if causal is not None:
                # Preserve Omega as the total per-replicate covariance by
                # dividing by that replicate's realized causal count.
                block_effects *= causal[start:stop, :, None]
                block_effects /= np.sqrt(counts[None, :, None])
            else:
                block_effects /= np.sqrt(float(m))
            realized_effect_covariance += np.einsum(
                "brq,brs->rqs", block_effects, block_effects, optimize=True
            )
            scores = genotype @ block_effects.reshape(width, r * q)
            component_scores += scores.reshape(n, r, q)
            means[start:stop] = block_mean
            inverse_scales[start:stop] = block_inverse
            missing_counts[start:stop] = block_missing
            visits += width
            blocks += 1
    genotype_seconds = time.perf_counter() - started
    if visits != m or blocks != (m + args.chunk_size - 1) // args.chunk_size:
        raise RuntimeError("simulator did not complete exactly one genotype traversal")

    genetic = np.einsum("nq,nrq->nr", basis, component_scores, optimize=True)
    noise_rng = np.random.default_rng(noise_seed)
    residual_innovations = noise_rng.standard_normal((n, r, q))
    residual_components = residual_innovations @ psi_root.T
    noise = np.einsum(
        "nq,nrq->nr", basis, residual_components, optimize=True
    )
    phenotypes = genetic + fixed_effect[:, None] + noise
    projected = phenotypes - fixed @ (fixed.T @ phenotypes)
    phenotype_variances = np.sum(projected * projected, axis=0) / residual_rank
    if np.any(phenotype_variances <= 0.0):
        raise RuntimeError("a simulated phenotype has zero projected variance")
    normalized_omega = omega[None, :, :] / phenotype_variances[:, None, None]
    normalized_residual = psi[0, 0] / phenotype_variances
    normalized_residual_coefficients = np.column_stack(
        [
            psi[left, right] / phenotype_variances
            for left, right in residual_pairs
        ]
    )

    projected_components = np.empty_like(component_scores)
    for replicate in range(r):
        contextual = basis * component_scores[:, replicate, :]
        projected_components[:, replicate, :] = contextual - fixed @ (
            fixed.T @ contextual
        )
    realized_component_covariance = np.einsum(
        "nrq,nrs->rqs", projected_components, projected_components, optimize=True
    ) / residual_rank
    realized_genetic_variance = np.sum(
        realized_component_covariance, axis=(1, 2)
    )
    trait_names = np.asarray(
        [f"{args.label}_replicate_{index:03d}" for index in range(r)]
    )
    args.output.mkdir(parents=True, mode=0o700)
    array_path = args.output / "simulation_batch.npz"
    np.savez_compressed(
        array_path,
        sample_ids=np.asarray(axes.sample_ids),
        trait_names=trait_names,
        basis=basis,
        fixed_basis=fixed,
        residual_basis=residual_basis,
        residual_names=np.asarray(residual_names),
        residual_pairs=np.asarray(residual_pairs, dtype=np.int64),
        phenotypes=np.asfortranarray(phenotypes),
        environment=environment,
        normalized_omega=normalized_omega,
        normalized_residual_variance=normalized_residual,
        normalized_residual_coefficients=normalized_residual_coefficients,
        phenotype_projected_variance=phenotype_variances,
        realized_effect_covariance=realized_effect_covariance,
        realized_component_covariance=realized_component_covariance,
        realized_genetic_variance=realized_genetic_variance,
        affine_mean=means,
        affine_inverse_scale=inverse_scales,
        missing_counts=missing_counts,
    )
    manifest = {
        "schema": SCHEMA,
        "label": args.label,
        "model": {
            "equation": "y=sum_q phi_q*(G beta_q)+C gamma+epsilon",
            "effect_distribution": "beta_j~N(0,Omega/M_causal) on causal SNPs",
            "basis_names": ["intercept", "environment_1", "environment_2"],
            "omega": omega.tolist(),
            "omega_eigenvalues": np.linalg.eigvalsh(omega).tolist(),
            "residual_variance": args.residual_variance,
            "psi": psi.tolist(),
            "psi_eigenvalues": np.linalg.eigvalsh(psi).tolist(),
            "environment_type": args.environment_type,
            "requested_environment_correlation": args.environment_correlation,
            "realized_environment_correlation": float(
                np.corrcoef(environment, rowvar=False)[0, 1]
            ),
            "fixed_environment_effects": list(args.fixed_environment_effects),
            "phenotype_inference_normalization": "project_then_unit_residual_variance_v1",
            "residual_nuisance_basis": (
                "eta_qr_phi_q_phi_r_for_all_symmetric_context_pairs"
            ),
            "residual_names": list(residual_names),
            "residual_true_coefficients": [
                float(psi[left, right]) for left, right in residual_pairs
            ],
        },
        "dimensions": {
            "N": n,
            "M": m,
            "Q": q,
            "replicates": r,
            "fixed_rank": fixed.shape[1],
            "residual_rank": residual_rank,
        },
        "randomization": {
            "root_seed": args.seed,
            "seed_spawn_keys": {
                "environment": list(environment_seed.spawn_key),
                "effects": list(effect_seed.spawn_key),
                "causal_mask": list(causal_seed.spawn_key),
                "noise": list(noise_seed.spawn_key),
            },
            "environment_shared_across_replicates": True,
            "effect_and_noise_draws_independent_across_replicates": True,
        },
        "genotype": {
            "prefix": str(axes.prefix),
            "scale": "mean_imputed_sample_sd_ddof1_common_across_contexts",
            "counted_allele": "BIM_A1",
            "causal_fraction": args.causal_fraction,
            "causal_counts": counts.tolist(),
            "complete_traversals": 1,
            "variant_visits": visits,
            "decoded_blocks": blocks,
            "chunk_size": args.chunk_size,
            "missing_calls": int(np.sum(missing_counts, dtype=np.int64)),
        },
        "runtime_seconds": {"genotype_traversal_and_effect_gemm": genotype_seconds},
        "outputs": {
            "arrays": str(array_path),
        },
        "diagnostics": {
            "maximum_environment_diagonal_gram_error": float(
                np.max(np.abs(np.diag(environment.T @ environment) - (n - 1.0)))
            ),
            "realized_environment_correlation": float(
                np.corrcoef(environment, rowvar=False)[0, 1]
            ),
            "maximum_effect_covariance_absolute_error": float(
                np.max(np.abs(realized_effect_covariance - omega[None, :, :]))
            ),
            "mean_realized_genetic_variance": float(np.mean(realized_genetic_variance)),
            "mean_projected_phenotype_variance": float(np.mean(phenotype_variances)),
        },
    }
    manifest_path = args.output / "simulation_manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n")
    print(
        canonical_json(
            {
                "manifest": str(manifest_path),
                "arrays": str(array_path),
                "runtime_seconds": genotype_seconds,
                "complete_genotype_traversals": 1,
                "variant_visits": visits,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
