#!/usr/bin/env python3
"""Validate disjoint annotation-specific contextual covariance.

The default path uses deterministic synthetic MAF/LD-like partitions and
emits aggregate diagnostics only.  It checks annotation-major/pair-minor dense
algebra, component and total recovery, lossless grouped approximate-LOO
contributions, full-jackknife annotation contrasts, matched independent
reference transport, rank/imbalance failures, and K/Q resource scaling.  An
opt-in real-data path uses a small protected PLINK subset and writes no sample
or variant rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri


OUTPUT_STEM = "08_context_annotations_validation"
PC_COLUMNS = tuple(f"f.22009.0.{index}" for index in range(1, 6))


@dataclass(frozen=True)
class VariantArchitecture:
    annotation_names: tuple[str, ...]
    annotations: np.ndarray
    allele_frequencies: np.ndarray
    ld_correlations: np.ndarray
    variant_chromosomes: np.ndarray
    variant_positions: np.ndarray


@dataclass(frozen=True)
class SyntheticCohort:
    genotype: np.ndarray
    basis: np.ndarray
    fixed_effects: np.ndarray
    phenotype: np.ndarray | None
    pc: np.ndarray


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    q: int
    k: int
    architecture: VariantArchitecture
    reference: SyntheticCohort
    study: SyntheticCohort
    true_omegas: np.ndarray
    loo_groups: tuple[str, ...]
    sparse: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--reference-n", type=int, default=256)
    parser.add_argument("--m", type=int, default=192)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--transport-replicates", type=int, default=12)
    parser.add_argument(
        "--probe-counts", type=int, nargs="+", default=(8, 32), metavar="B"
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--real-traits",
        action="store_true",
        help="Run an aggregate-only real MAF/LD-bin BMI/CRP numerical sanity.",
    )
    parser.add_argument("--geno-prefix", type=Path, default=None)
    parser.add_argument("--phenotype-root", type=Path, default=None)
    parser.add_argument(
        "--covariate-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--real-n", type=int, default=256)
    parser.add_argument("--real-m", type=int, default=240)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n < 96 or args.reference_n < 96:
        parser.error("--n and --reference-n must be at least 96")
    if args.m < 96 or args.m % 8:
        parser.error("--m must be at least 96 and divisible by 8")
    if args.loo_groups < 12 or args.m % args.loo_groups:
        parser.error("--loo-groups must be >=12 and divide --m exactly")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.transport_replicates < 6:
        parser.error("--transport-replicates must be at least 6")
    probe_counts = tuple(sorted({int(value) for value in args.probe_counts}))
    if not probe_counts or probe_counts[0] < 2 or probe_counts[-1] > 256:
        parser.error("--probe-counts must contain integers in [2,256]")
    args.probe_counts = probe_counts
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.real_traits:
        if any(value is None for value in (args.geno_prefix, args.phenotype_root, args.covariate_file)):
            parser.error("--real-traits requires --geno-prefix, --phenotype-root, and --covariate-file")
        if args.real_n < 128:
            parser.error("--real-n must be at least 128")
        if args.real_m < 96 or args.real_m % 8 or args.real_m % args.loo_groups:
            parser.error("--real-m must be >=96 and divisible by 8 and --loo-groups")


def _output_paths(output_dir: Path, *, include_real: bool) -> tuple[Path, ...]:
    labels = ["recovery", "loo_contrasts", "transport", "resources"]
    if include_real:
        labels.append("real_traits")
    names = [f"{OUTPUT_STEM}.json"]
    for label in labels:
        names.extend([f"{OUTPUT_STEM}_{label}.png", f"{OUTPUT_STEM}_{label}.pdf"])
    return tuple(output_dir / name for name in names)


def _prepare_output_directory(output_dir: Path, *, include_real: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    existing = [
        path.name
        for path in _output_paths(output_dir, include_real=include_real)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing validation outputs: " + ", ".join(existing)
        )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _save_figure(figure: Any, output_dir: Path, label: str) -> None:
    for suffix in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}_{label}.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        path.chmod(0o600)
    plt.close(figure)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _standardize(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("Encountered a degenerate numeric column.")
    return np.asarray(centered / scale, dtype=np.float64)


def _resource_estimate(
    *,
    n: int,
    m: int,
    q: int,
    k: int,
    residual_count: int,
    probe_tile: int,
    loo_groups: int,
) -> dict[str, Any]:
    pair_count = q * (q + 1) // 2
    p_genetic = k * pair_count
    p_total = p_genetic + residual_count
    float_bytes = np.dtype(np.float64).itemsize
    reference_snp_bytes = m * p_genetic * p_genetic * float_bytes
    trait_snp_bytes = m * p_genetic * (2 + residual_count) * float_bytes
    grouped_reference_bytes = loo_groups * p_genetic * p_genetic * float_bytes
    grouped_trait_bytes = loo_groups * p_genetic * (2 + residual_count) * float_bytes
    action_bytes = n * probe_tile * p_genetic * float_bytes
    return {
        "q": q,
        "k": k,
        "pair_count": pair_count,
        "p_genetic": p_genetic,
        "p_total": p_total,
        "normal_matrix_shape": [p_total, p_total],
        "normal_matrix_bytes": p_total * p_total * float_bytes,
        "reference_snp_contribution_bytes": reference_snp_bytes,
        "trait_snp_contribution_bytes": trait_snp_bytes,
        "lossless_grouped_reference_bytes": grouped_reference_bytes,
        "lossless_grouped_trait_bytes": grouped_trait_bytes,
        "grouping_compression_ratio": float(
            (reference_snp_bytes + trait_snp_bytes)
            / max(grouped_reference_bytes + grouped_trait_bytes, 1)
        ),
        "probe_action_buffer_bytes": action_bytes,
        "jackknife_coefficient_bytes": loo_groups * p_total * float_bytes,
        "joint_jackknife_covariance_bytes": p_total * p_total * float_bytes,
        "source_genotype_products_per_probe_tile": q,
        "target_genotype_products_per_probe_tile": k * q,
        "interpretation": (
            "correctness-first float64 estimates; production peak also depends "
            "on decoder, projection, probe, and protected-GEMM workspaces"
        ),
    }


def _resource_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        _resource_estimate(
            n=args.reference_n,
            m=args.m,
            q=q,
            k=k,
            residual_count=1,
            probe_tile=min(max(args.probe_counts), 32),
            loo_groups=args.loo_groups,
        )
        for k in (1, 4, 8)
        for q in (2, 3, 4)
    ]


def _annotation_names(k: int) -> tuple[str, ...]:
    if k == 1:
        return ("all",)
    if k == 2:
        return ("lower_maf", "higher_maf")
    maf_labels = (
        ("low_maf", "high_maf")
        if k == 4
        else ("low_maf", "midlow_maf", "midhigh_maf", "high_maf")
    )
    ld_labels = ("low_ld", "high_ld")
    names = tuple(f"{maf}_{ld}" for maf in maf_labels for ld in ld_labels)
    return names[:k]


def _variant_architecture(m: int, k: int, *, seed: int) -> VariantArchitecture:
    if k not in (1, 2, 4, 8) or m % k:
        raise ValueError("Synthetic annotations require K in {1,2,4,8} dividing M.")
    rng = np.random.default_rng(seed)
    names = _annotation_names(k)
    annotations = np.zeros((m, k), dtype=np.float64)
    per_bin = m // k
    frequencies = np.empty(m, dtype=np.float64)
    correlations = np.empty(m, dtype=np.float64)
    if k == 1:
        maf_intervals = ((0.06, 0.48),)
        ld_levels = (0.30,)
    elif k == 2:
        maf_intervals = ((0.05, 0.20), (0.20, 0.48))
        ld_levels = (0.30, 0.30)
    else:
        selected_bands = (
            ((0.05, 0.20), (0.20, 0.48))
            if k == 4
            else ((0.05, 0.12), (0.12, 0.22), (0.22, 0.34), (0.34, 0.48))
        )
        maf_intervals = tuple(interval for interval in selected_bands for _ in range(2))
        ld_levels = tuple(level for _ in selected_bands for level in (0.08, 0.68))
    for annotation_index in range(k):
        selected = np.arange(annotation_index, m, k, dtype=np.int64)
        if selected.size != per_bin:
            raise RuntimeError("Interleaved annotation construction lost bin balance.")
        annotations[selected, annotation_index] = 1.0
        low, high = maf_intervals[annotation_index]
        frequencies[selected] = rng.uniform(low, high, size=per_bin)
        correlations[selected] = ld_levels[annotation_index]
    chromosomes = 1 + np.arange(m, dtype=np.int64) * 22 // m
    positions = np.arange(m, dtype=np.int64) * 100_000 + 1
    return VariantArchitecture(
        annotation_names=names,
        annotations=annotations,
        allele_frequencies=frequencies,
        ld_correlations=correlations,
        variant_chromosomes=chromosomes,
        variant_positions=positions,
    )


def _validate_disjoint_partition(
    architecture: VariantArchitecture, loo_groups: Sequence[str]
) -> dict[str, Any]:
    annotations = np.asarray(architecture.annotations, dtype=np.float64)
    if annotations.ndim != 2 or annotations.shape[1] != len(
        architecture.annotation_names
    ):
        raise ValueError("Annotation matrix and ordered names are incompatible.")
    if not np.all(np.isfinite(annotations)) or np.any(
        (annotations != 0.0) & (annotations != 1.0)
    ):
        raise ValueError("Disjoint partition weights must be finite zero/one values.")
    memberships = np.sum(annotations, axis=1)
    if not np.all(memberships == 1.0):
        raise ValueError("Every retained variant must belong to exactly one partition.")
    masses = np.sum(annotations, axis=0)
    if np.any(masses <= 0.0):
        raise ValueError("Every annotation partition must have positive mass.")
    groups = np.asarray(tuple(str(value) for value in loo_groups), dtype=object)
    if groups.shape != (annotations.shape[0],):
        raise ValueError("LOO groups must contain one label per ordered variant.")
    ordered_groups = tuple(dict.fromkeys(groups.tolist()))
    group_counts = np.asarray([np.sum(groups == value) for value in ordered_groups])
    group_bin_counts = np.asarray(
        [np.sum(annotations[groups == value], axis=0) for value in ordered_groups]
    )
    payload = {
        "mode": "disjoint_exhaustive_binary_partition",
        "annotation_names": architecture.annotation_names,
        "masses": masses,
        "variant_count": int(annotations.shape[0]),
        "partition_hash": _canonical_hash(
            {
                "names": architecture.annotation_names,
                "weights_sha256": hashlib.sha256(
                    np.ascontiguousarray(annotations).tobytes()
                ).hexdigest(),
            }
        ),
        "loo_group_count": len(ordered_groups),
        "loo_group_variant_count_minimum": int(np.min(group_counts)),
        "loo_group_variant_count_maximum": int(np.max(group_counts)),
        "minimum_remaining_bin_mass": float(np.min(masses[None, :] - group_bin_counts)),
        "maximum_group_bin_count_imbalance": int(
            np.max(np.ptp(group_bin_counts, axis=0), initial=0)
        ),
    }
    payload["gate"] = bool(
        payload["loo_group_variant_count_maximum"]
        - payload["loo_group_variant_count_minimum"]
        <= 1
        and payload["minimum_remaining_bin_mass"] > 0.0
    )
    return payload


def _balanced_physical_groups(m: int, group_count: int) -> tuple[str, ...]:
    if m % group_count:
        raise ValueError("Physical group count must divide M in this validation.")
    per_group = m // group_count
    return tuple(f"block:{index // per_group}" for index in range(m))


def _simulate_genotype(
    architecture: VariantArchitecture,
    n: int,
    *,
    seed: int,
    pc: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = architecture.annotations.shape[0]
    genotype = np.empty((n, m), dtype=np.float64)
    memberships = np.argmax(architecture.annotations, axis=1)
    for annotation_index in range(len(architecture.annotation_names)):
        selected = np.flatnonzero(memberships == annotation_index)
        latent = np.empty((2, n, selected.size), dtype=np.float64)
        rho = float(architecture.ld_correlations[selected[0]])
        innovation_scale = math.sqrt(max(1.0 - rho * rho, 0.0))
        for haplotype in range(2):
            innovations = rng.normal(size=(n, selected.size))
            latent[haplotype, :, 0] = innovations[:, 0]
            for offset in range(1, selected.size):
                latent[haplotype, :, offset] = (
                    rho * latent[haplotype, :, offset - 1]
                    + innovation_scale * innovations[:, offset]
                )
        thresholds = ndtri(1.0 - architecture.allele_frequencies[selected])
        genotype[:, selected] = np.sum(
            latent > thresholds[None, None, :], axis=0, dtype=np.float64
        )
    population_loadings = rng.normal(scale=0.10, size=m)
    genotype += pc[:, None] * population_loadings[None, :]
    return _standardize(genotype)


def _simulate_basis_and_fixed(
    n: int, q: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    pc = _standardize(latent[:, 2])
    z1 = _standardize(0.40 * pc + math.sqrt(0.84) * latent[:, 0])
    z2 = _standardize(0.55 * z1 + math.sqrt(1.0 - 0.55**2) * latent[:, 1])
    columns = [np.ones(n, dtype=np.float64), z1, z2, _standardize(z1 * z2)]
    basis = np.column_stack(columns[:q])
    fixed = np.column_stack(
        [
            np.ones(n, dtype=np.float64),
            basis[:, 1:],
            pc,
            pc[:, None] * basis[:, 1:],
        ]
    )
    return np.asarray(basis), np.asarray(fixed), np.asarray(pc)


def _truth_omegas(k: int, q: int) -> np.ndarray:
    result = np.zeros((k, q, q), dtype=np.float64)
    base = np.linspace(0.48, -0.12, q)
    orthogonal = np.linspace(0.05, 0.34, q)
    for annotation_index in range(k):
        if k > 1 and annotation_index == k - 1:
            continue
        scale = 0.30 + 0.12 * annotation_index
        loading = np.roll(base, annotation_index % q)
        omega = scale * np.outer(loading, loading)
        if annotation_index % 2:
            second = np.roll(orthogonal, (annotation_index + 1) % q)
            omega += 0.12 * np.outer(second, second)
        result[annotation_index] = omega
    if k == 1:
        result[0] += 0.10 * np.eye(q)
    return result


def _simulate_fixture(
    *,
    name: str,
    n: int,
    reference_n: int,
    m: int,
    q: int,
    k: int,
    loo_groups: int,
    seed: int,
    sparse: bool = False,
) -> SyntheticFixture:
    architecture = _variant_architecture(m, k, seed=seed + 1)
    study_basis, study_fixed, study_pc = _simulate_basis_and_fixed(n, q, seed=seed + 2)
    reference_basis, reference_fixed, reference_pc = _simulate_basis_and_fixed(
        reference_n, q, seed=seed + 3
    )
    study_genotype = _simulate_genotype(architecture, n, seed=seed + 4, pc=study_pc)
    reference_genotype = _simulate_genotype(
        architecture, reference_n, seed=seed + 5, pc=reference_pc
    )
    omegas = _truth_omegas(k, q)
    rng = np.random.default_rng(seed + 6)
    effects = np.zeros((m, q), dtype=np.float64)
    memberships = np.argmax(architecture.annotations, axis=1)
    for annotation_index in range(k):
        selected = np.flatnonzero(memberships == annotation_index)
        active = selected
        if sparse and np.any(omegas[annotation_index]):
            active = selected[: max(2, selected.size // 10)]
        if not np.any(omegas[annotation_index]):
            continue
        effects[active] = rng.multivariate_normal(
            np.zeros(q),
            omegas[annotation_index] / float(active.size),
            size=active.size,
        )
    genetic = np.sum(study_basis * (study_genotype @ effects), axis=1, dtype=np.float64)
    phenotype = genetic + 0.55 * rng.normal(size=n)
    groups = _balanced_physical_groups(m, loo_groups)
    return SyntheticFixture(
        name=name,
        q=q,
        k=k,
        architecture=architecture,
        reference=SyntheticCohort(
            genotype=reference_genotype,
            basis=reference_basis,
            fixed_effects=reference_fixed,
            phenotype=None,
            pc=reference_pc,
        ),
        study=SyntheticCohort(
            genotype=study_genotype,
            basis=study_basis,
            fixed_effects=study_fixed,
            phenotype=np.asarray(phenotype),
            pc=study_pc,
        ),
        true_omegas=omegas,
        loo_groups=groups,
        sparse=sparse,
    )


def _scale_aware_error(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Cannot compare arrays with shapes {left_array.shape} and "
            f"{right_array.shape}."
        )
    scale = np.maximum(1.0, np.maximum(np.abs(left_array), np.abs(right_array)))
    return float(np.max(np.abs(left_array - right_array) / scale, initial=0.0))


def _first_attr(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise AttributeError(
        f"{type(value).__name__} exposes none of the expected fields {names}."
    )


def _component_index(fixture: SyntheticFixture) -> Any:
    from summit.context import ContextComponentIndex, ContextPairIndex

    return ContextComponentIndex(
        fixture.architecture.annotation_names, ContextPairIndex(fixture.q)
    )


def _synthetic_identity(fixture: SyntheticFixture) -> dict[str, str]:
    architecture = fixture.architecture
    variant_hash = _canonical_hash(
        {
            "simulation": "maf_ld_like_v1",
            "variant_count": int(architecture.annotations.shape[0]),
            "chromosomes_sha256": hashlib.sha256(
                np.ascontiguousarray(architecture.variant_chromosomes).tobytes()
            ).hexdigest(),
            "positions_sha256": hashlib.sha256(
                np.ascontiguousarray(architecture.variant_positions).tobytes()
            ).hexdigest(),
        }
    )
    return {
        "basis_hash": _canonical_hash(
            {
                "basis": "intercept_z1_z2_product_prefix",
                "q": fixture.q,
                "feature_mode": "raw_projected",
            }
        ),
        "fixed_effect_hash": _canonical_hash(
            {
                "fixed_effect_spec": "basis_pc_and_pc_by_nonconstant_basis",
                "q": fixture.q,
            }
        ),
        "variant_hash": variant_hash,
    }


def _build_core_objects(
    fixture: SyntheticFixture,
    *,
    block_size: int,
    fit_model: bool = True,
) -> tuple[Any, Any, Any | None, Any, Any]:
    """Build the existing SNP-level oracle objects used for parity only."""
    from summit.context import (
        build_context_reference,
        build_context_trait_summary,
        fit_context_model,
        rank_revealing_projector,
    )

    component_index = _component_index(fixture)
    identity = _synthetic_identity(fixture)
    reference_projector = rank_revealing_projector(fixture.reference.fixed_effects)
    study_projector = rank_revealing_projector(fixture.study.fixed_effects)
    reference = build_context_reference(
        genotype=fixture.reference.genotype,
        basis=fixture.reference.basis,
        projector=reference_projector,
        annotations=fixture.architecture.annotations,
        component_index=component_index,
        basis_hash=identity["basis_hash"],
        fixed_effect_hash=identity["fixed_effect_hash"],
        variant_hash=identity["variant_hash"],
        loo_groups=fixture.loo_groups,
        genotype_scaling="synthetic_population_unit_variance",
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=min(block_size, 32),
    )
    summary = build_context_trait_summary(
        genotype=fixture.study.genotype,
        basis=fixture.study.basis,
        phenotype=fixture.study.phenotype,
        projector=study_projector,
        annotations=fixture.architecture.annotations,
        component_index=component_index,
        residual_basis=np.ones((fixture.study.genotype.shape[0], 1)),
        residual_names=("identity",),
        basis_hash=identity["basis_hash"],
        fixed_effect_hash=identity["fixed_effect_hash"],
        variant_hash=identity["variant_hash"],
        loo_groups=fixture.loo_groups,
        genotype_scaling="synthetic_population_unit_variance",
        block_size=block_size,
    )
    fit = None
    if fit_model:
        fit = fit_context_model(
            reference,
            summary,
            loo_groups=tuple(dict.fromkeys(fixture.loo_groups)),
            context_grid=fixture.study.basis[: min(32, fixture.study.basis.shape[0])],
            basis_metric=(fixture.reference.basis.T @ fixture.reference.basis)
            / fixture.reference.basis.shape[0],
            project_psd=False,
            annotations_disjoint=True,
        )
    return reference, summary, fit, reference_projector, study_projector


def _group_sums(
    values: np.ndarray, groups: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    group_array = np.asarray(tuple(str(value) for value in groups), dtype=object)
    unique = tuple(dict.fromkeys(group_array.tolist()))
    result = np.stack(
        [
            np.sum(values[group_array == group], axis=0, dtype=np.float64)
            for group in unique
        ]
    )
    return result, unique


def _dense_oracle_payload(
    fixture: SyntheticFixture,
    reference: Any,
    summary: Any,
    reference_projector: Any,
    study_projector: Any,
) -> dict[str, Any]:
    from summit.context import (
        common_scale_features,
        dense_genetic_kernels,
        exact_same_person_matrix,
        kernel_gram,
        kernel_rhs,
        kernel_traces,
        project_normalize_phenotype,
    )

    components = reference.component_index
    weights = fixture.architecture.annotations
    reference_features = common_scale_features(
        fixture.reference.genotype,
        fixture.reference.basis,
        reference_projector.projector,
    )
    reference_kernels = dense_genetic_kernels(reference_features, weights, components)
    study_features = common_scale_features(
        fixture.study.genotype,
        fixture.study.basis,
        study_projector.projector,
    )
    study_kernels = dense_genetic_kernels(study_features, weights, components)
    normalized_y = project_normalize_phenotype(fixture.study.phenotype, study_projector)
    dense_reference_gram = kernel_gram(reference_kernels)
    dense_same_person = exact_same_person_matrix(reference_kernels)
    dense_rhs = kernel_rhs(study_kernels, normalized_y)
    dense_traces = kernel_traces(study_kernels)
    off_diagonal_errors: list[float] = []
    for component in components.entries:
        if component.q == component.r:
            continue
        selected = weights[:, component.annotation_index]
        mass = float(np.sum(selected))
        expected = (
            (study_features[component.q] * selected[None, :])
            @ study_features[component.r].T
            + (study_features[component.r] * selected[None, :])
            @ study_features[component.q].T
        ) / mass
        off_diagonal_errors.append(
            _scale_aware_error(study_kernels[component.index], expected)
        )
    return {
        "reference_gram_discrepancy": _scale_aware_error(
            reference.gram, dense_reference_gram
        ),
        "same_person_discrepancy": _scale_aware_error(
            reference.same_person, dense_same_person
        ),
        "trait_rhs_discrepancy": _scale_aware_error(summary.genetic_rhs, dense_rhs),
        "trait_trace_discrepancy": _scale_aware_error(
            summary.genetic_traces, dense_traces
        ),
        "maximum_offdiagonal_factor_discrepancy": float(
            max(off_diagonal_errors, default=0.0)
        ),
        "component_order": list(components.names),
        "gate": bool(
            _scale_aware_error(reference.gram, dense_reference_gram) < 2.0e-11
            and _scale_aware_error(reference.same_person, dense_same_person) < 2.0e-11
            and _scale_aware_error(summary.genetic_rhs, dense_rhs) < 2.0e-11
            and _scale_aware_error(summary.genetic_traces, dense_traces) < 2.0e-11
            and max(off_diagonal_errors, default=0.0) < 2.0e-11
        ),
    }


def _manual_grouped_payload(
    fixture: SyntheticFixture, reference: Any, summary: Any, fit: Any
) -> dict[str, Any]:
    from summit.context import (
        reference_moments_after_deleting_groups,
        trait_moments_after_deleting_groups,
    )

    reference_groups, unique = _group_sums(
        reference.gram_numerator_contributions, fixture.loo_groups
    )
    rhs_groups, rhs_unique = _group_sums(
        summary.rhs_numerator_contributions, fixture.loo_groups
    )
    trace_groups, trace_unique = _group_sums(
        summary.trace_numerator_contributions, fixture.loo_groups
    )
    cross_groups, cross_unique = _group_sums(
        summary.genetic_residual_numerator_contributions, fixture.loo_groups
    )
    annotation_groups, annotation_unique = _group_sums(
        fixture.architecture.annotations, fixture.loo_groups
    )
    if not (unique == rhs_unique == trace_unique == cross_unique == annotation_unique):
        raise RuntimeError("Independent grouped accumulators disagree on group order.")
    component_annotation = np.asarray(
        [entry.annotation_index for entry in reference.component_index.entries]
    )
    reference_errors: list[float] = []
    trait_errors: list[float] = []
    for group_index, group in enumerate(unique):
        remaining_annotation = (
            reference.annotation_masses - annotation_groups[group_index]
        )
        remaining_component = remaining_annotation[component_annotation]
        expected_reference = (
            np.sum(reference_groups, axis=0) - reference_groups[group_index]
        ) / np.outer(remaining_component, remaining_component)
        expected_reference = 0.5 * (expected_reference + expected_reference.T)
        observed_reference = reference_moments_after_deleting_groups(
            reference, (group,)
        )
        reference_errors.append(
            max(
                _scale_aware_error(observed_reference.gram, expected_reference),
                _scale_aware_error(
                    observed_reference.annotation_masses, remaining_annotation
                ),
                _scale_aware_error(
                    observed_reference.same_person, reference.same_person
                ),
            )
        )
        expected_rhs = (np.sum(rhs_groups, axis=0) - rhs_groups[group_index]) / (
            remaining_component
        )
        expected_trace = (
            np.sum(trace_groups, axis=0) - trace_groups[group_index]
        ) / remaining_component
        expected_cross = (
            np.sum(cross_groups, axis=0) - cross_groups[group_index]
        ) / remaining_component[:, None]
        observed_trait = trait_moments_after_deleting_groups(summary, (group,))
        trait_errors.append(
            max(
                _scale_aware_error(observed_trait.genetic_rhs, expected_rhs),
                _scale_aware_error(observed_trait.genetic_traces, expected_trace),
                _scale_aware_error(observed_trait.genetic_residual, expected_cross),
                _scale_aware_error(
                    observed_trait.annotation_masses, remaining_annotation
                ),
            )
        )
    snp_bytes = int(
        reference.gram_numerator_contributions.nbytes
        + summary.rhs_numerator_contributions.nbytes
        + summary.trace_numerator_contributions.nbytes
        + summary.genetic_residual_numerator_contributions.nbytes
    )
    grouped_bytes = int(
        reference_groups.nbytes
        + rhs_groups.nbytes
        + trace_groups.nbytes
        + cross_groups.nbytes
        + annotation_groups.nbytes
    )
    return {
        "group_order": list(unique),
        "group_count": len(unique),
        "group_annotation_masses": annotation_groups.tolist(),
        "group_variant_counts": [
            int(sum(value == group for value in fixture.loo_groups)) for group in unique
        ],
        "full_reference_reconstruction_discrepancy": _scale_aware_error(
            np.sum(reference_groups, axis=0),
            reference.gram
            * np.outer(
                reference.annotation_masses[component_annotation],
                reference.annotation_masses[component_annotation],
            ),
        ),
        "full_trait_rhs_reconstruction_discrepancy": _scale_aware_error(
            np.sum(rhs_groups, axis=0),
            summary.genetic_rhs * summary.annotation_masses[component_annotation],
        ),
        "maximum_deleted_reference_discrepancy": float(
            max(reference_errors, default=0.0)
        ),
        "maximum_deleted_trait_discrepancy": float(max(trait_errors, default=0.0)),
        "snp_level_bytes": snp_bytes,
        "grouped_bytes": grouped_bytes,
        "compression_ratio": float(snp_bytes / max(grouped_bytes, 1)),
        "loo_coefficient_shape": list(np.asarray(fit.loo_coefficients).shape),
        "gate": bool(
            max(reference_errors, default=0.0) < 2.0e-11
            and max(trait_errors, default=0.0) < 2.0e-11
        ),
    }


def _jackknife_se(values: object) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("Jackknife values must contain at least two replicates.")
    centered = array - np.mean(array)
    return float(np.sqrt((array.size - 1.0) / array.size * (centered @ centered)))


def _manual_contrast_payload(fixture: SyntheticFixture, fit: Any) -> dict[str, Any]:
    pair_count = fixture.q * (fixture.q + 1) // 2
    p_genetic = fixture.k * pair_count
    if fixture.k < 2:
        return {
            "status": "not_applicable_single_annotation",
            "gate": True,
        }
    component = 0
    left_index = component
    right_index = pair_count + component
    contrast = np.zeros(fit.raw_coefficients.size, dtype=np.float64)
    contrast[left_index] = 1.0
    contrast[right_index] = -1.0
    loo_values = np.asarray(fit.loo_coefficients) @ contrast
    estimate = float(np.asarray(fit.raw_coefficients) @ contrast)
    covariance_se = float(
        np.sqrt(max(float(contrast @ fit.jackknife_covariance @ contrast), 0.0))
    )
    direct_se = _jackknife_se(loo_values)
    marginal_only_variance = float(
        fit.jackknife_covariance[left_index, left_index]
        + fit.jackknife_covariance[right_index, right_index]
    )
    marginal_only_se = math.sqrt(max(marginal_only_variance, 0.0))
    cross_covariance = float(fit.jackknife_covariance[left_index, right_index])
    return {
        "annotation_left": fixture.architecture.annotation_names[0],
        "annotation_right": fixture.architecture.annotation_names[1],
        "component": "basis:0,0",
        "estimate": estimate,
        "standard_error": covariance_se,
        "direct_loo_standard_error": direct_se,
        "full_joint_covariance_cross_term": cross_covariance,
        "marginal_only_standard_error": marginal_only_se,
        "marginal_only_se_difference": abs(marginal_only_se - covariance_se),
        "loo_minimum": float(np.min(loo_values)),
        "loo_maximum": float(np.max(loo_values)),
        "covariance_formula_discrepancy": abs(covariance_se - direct_se),
        "gate": bool(abs(covariance_se - direct_se) < 2.0e-11),
    }


def _fit_payload(fixture: SyntheticFixture, fit: Any) -> dict[str, Any]:
    from summit.context import rank_revealing_projector

    fitted = np.asarray(fit.raw_omegas, dtype=np.float64)
    projector = rank_revealing_projector(fixture.study.fixed_effects)
    projected_phenotype = projector.projector @ fixture.study.phenotype
    normalization_squared = projector.residual_rank / float(
        projected_phenotype @ projected_phenotype
    )
    truth = normalization_squared * np.asarray(fixture.true_omegas, dtype=np.float64)
    flattened_truth = truth.ravel()
    flattened_fit = fitted.ravel()
    truth_norm = float(np.linalg.norm(flattened_truth))
    fit_norm = float(np.linalg.norm(flattened_fit))
    cosine = (
        None
        if truth_norm == 0.0 or fit_norm == 0.0
        else float(flattened_truth @ flattened_fit / (truth_norm * fit_norm))
    )
    minimum_eigenvalues = [
        float(np.min(np.linalg.eigvalsh(0.5 * (omega + omega.T)))) for omega in fitted
    ]
    trace_contributions = []
    pair_count = fixture.q * (fixture.q + 1) // 2
    for annotation_index in range(fixture.k):
        selected = slice(
            annotation_index * pair_count, (annotation_index + 1) * pair_count
        )
        trace_contributions.append(
            float(
                np.dot(
                    fit.genetic_coefficients[selected],
                    fit.equations.traces[selected],
                )
            )
        )
    return {
        "raw_omegas": fitted.tolist(),
        "expected_normalized_generating_omegas": truth.tolist(),
        "phenotype_normalization_squared": float(normalization_squared),
        "omega_rmse": float(np.sqrt(np.mean((fitted - truth) ** 2))),
        "omega_cosine": cosine,
        "minimum_raw_omega_eigenvalues": minimum_eigenvalues,
        "annotation_trace_contributions": trace_contributions,
        "total_genetic_trace_contribution": float(sum(trace_contributions)),
        "rank": int(fit.solve.rank),
        "dimension": int(fit.raw_coefficients.size),
        "condition_number": float(fit.solve.condition_number),
        "relative_residual": float(fit.solve.relative_residual),
        "jackknife_replicates": int(fit.loo_coefficients.shape[0]),
        "finite": bool(
            np.all(np.isfinite(fit.raw_coefficients))
            and np.all(np.isfinite(fit.jackknife_covariance))
        ),
    }


def _build_reference_for_cohort(
    fixture: SyntheticFixture,
    cohort: SyntheticCohort,
    *,
    block_size: int,
) -> Any:
    from summit.context import build_context_reference, rank_revealing_projector

    identity = _synthetic_identity(fixture)
    projector = rank_revealing_projector(cohort.fixed_effects)
    return build_context_reference(
        genotype=cohort.genotype,
        basis=cohort.basis,
        projector=projector,
        annotations=fixture.architecture.annotations,
        component_index=_component_index(fixture),
        basis_hash=identity["basis_hash"],
        fixed_effect_hash=identity["fixed_effect_hash"],
        variant_hash=identity["variant_hash"],
        loo_groups=fixture.loo_groups,
        genotype_scaling="synthetic_population_unit_variance",
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=min(block_size, 32),
    )


def _independent_reference_cohort(
    fixture: SyntheticFixture, n: int, *, seed: int
) -> SyntheticCohort:
    basis, fixed, pc = _simulate_basis_and_fixed(n, fixture.q, seed=seed)
    genotype = _simulate_genotype(fixture.architecture, n, seed=seed + 1, pc=pc)
    return SyntheticCohort(
        genotype=genotype,
        basis=basis,
        fixed_effects=fixed,
        phenotype=None,
        pc=pc,
    )


def _build_summary_for_fixture(
    fixture: SyntheticFixture, *, block_size: int
) -> tuple[Any, Any]:
    from summit.context import build_context_trait_summary, rank_revealing_projector

    identity = _synthetic_identity(fixture)
    projector = rank_revealing_projector(fixture.study.fixed_effects)
    summary = build_context_trait_summary(
        genotype=fixture.study.genotype,
        basis=fixture.study.basis,
        phenotype=fixture.study.phenotype,
        projector=projector,
        annotations=fixture.architecture.annotations,
        component_index=_component_index(fixture),
        residual_basis=np.ones((fixture.study.genotype.shape[0], 1)),
        residual_names=("identity",),
        basis_hash=identity["basis_hash"],
        fixed_effect_hash=identity["fixed_effect_hash"],
        variant_hash=identity["variant_hash"],
        loo_groups=fixture.loo_groups,
        genotype_scaling="synthetic_population_unit_variance",
        block_size=block_size,
    )
    return summary, projector


def _transport_validation(
    fixture: SyntheticFixture, *, args: argparse.Namespace
) -> dict[str, Any]:
    from summit.context import fit_context_model, transfer_reference_gram

    reference_sizes = tuple(
        dict.fromkeys(
            (
                max(96, args.n // 2),
                max(args.reference_n, max(96, args.n // 2) + 32),
            )
        )
    )
    correct_gram_errors: dict[int, list[float]] = {
        value: [] for value in reference_sizes
    }
    blind_gram_errors: dict[int, list[float]] = {value: [] for value in reference_sizes}
    coefficient_errors: dict[int, list[np.ndarray]] = {
        value: [] for value in reference_sizes
    }
    exact_a_values: list[np.ndarray] = []
    exact_b_values: list[np.ndarray] = []
    exact_conditions: list[float] = []
    for replicate in range(args.transport_replicates):
        replicate_fixture = _simulate_fixture(
            name=f"transport_replicate_{replicate}",
            n=args.n,
            reference_n=reference_sizes[0],
            m=fixture.architecture.annotations.shape[0],
            q=fixture.q,
            k=fixture.k,
            loo_groups=args.loo_groups,
            seed=args.seed + 30_000 + replicate * 1009,
        )
        phenotype_a, _ = _resample_fixture_phenotype(
            replicate_fixture, seed=args.seed + 31_000 + replicate * 1013
        )
        phenotype_b, _ = _resample_fixture_phenotype(
            replicate_fixture, seed=args.seed + 32_000 + replicate * 1019
        )
        summary_a, _ = _build_summary_for_fixture(
            phenotype_a, block_size=args.block_size
        )
        summary_b, _ = _build_summary_for_fixture(
            phenotype_b, block_size=args.block_size
        )
        exact_reference = _build_reference_for_cohort(
            replicate_fixture,
            replicate_fixture.study,
            block_size=args.block_size,
        )
        exact_fit_a = fit_context_model(
            exact_reference,
            summary_a,
            loo_groups=tuple(dict.fromkeys(replicate_fixture.loo_groups)),
            project_psd=False,
            annotations_disjoint=True,
        )
        exact_fit_b = fit_context_model(
            exact_reference,
            summary_b,
            loo_groups=tuple(dict.fromkeys(replicate_fixture.loo_groups)),
            project_psd=False,
            annotations_disjoint=True,
        )
        exact_a_values.append(np.asarray(exact_fit_a.raw_coefficients))
        exact_b_values.append(np.asarray(exact_fit_b.raw_coefficients))
        exact_conditions.append(float(exact_fit_a.solve.condition_number))
        exact_study_gram = exact_reference.gram
        for reference_n in reference_sizes:
            cohort = _independent_reference_cohort(
                replicate_fixture,
                reference_n,
                seed=args.seed + 33_000 + reference_n * 17 + replicate * 1031,
            )
            reference = _build_reference_for_cohort(
                replicate_fixture, cohort, block_size=args.block_size
            )
            correct = transfer_reference_gram(
                reference.gram,
                reference.same_person,
                reference_n=reference.n_samples,
                study_n=summary_a.n_samples,
            )
            blind = reference.gram * (summary_a.n_samples / reference.n_samples) ** 2
            scale = max(
                float(np.sqrt(np.mean(exact_study_gram * exact_study_gram))),
                np.finfo(np.float64).tiny,
            )
            correct_gram_errors[reference_n].append(
                float(np.sqrt(np.mean((correct - exact_study_gram) ** 2)) / scale)
            )
            blind_gram_errors[reference_n].append(
                float(np.sqrt(np.mean((blind - exact_study_gram) ** 2)) / scale)
            )
            fit = fit_context_model(
                reference,
                summary_a,
                loo_groups=tuple(dict.fromkeys(replicate_fixture.loo_groups)),
                project_psd=False,
                annotations_disjoint=True,
            )
            coefficient_errors[reference_n].append(
                np.asarray(fit.raw_coefficients - exact_fit_a.raw_coefficients)
            )

    exact_a_array = np.asarray(exact_a_values)
    exact_b_array = np.asarray(exact_b_values)
    phenotype_delta = (exact_a_array - exact_b_array) / math.sqrt(2.0)
    phenotype_rms = float(
        np.sqrt(np.mean(np.sum(phenotype_delta[:, :-1] ** 2, axis=1)))
    )
    records: list[dict[str, Any]] = []
    for reference_n in reference_sizes:
        delta = np.asarray(coefficient_errors[reference_n])
        genetic_rms = float(np.sqrt(np.mean(np.sum(delta[:, :-1] ** 2, axis=1))))
        records.append(
            {
                "reference_n": reference_n,
                "replicates": args.transport_replicates,
                "correct_transfer_error_mean": float(
                    np.mean(correct_gram_errors[reference_n])
                ),
                "correct_transfer_error_median": float(
                    np.median(correct_gram_errors[reference_n])
                ),
                "correct_transfer_error_sd": float(
                    np.std(correct_gram_errors[reference_n], ddof=1)
                ),
                "blind_squared_n_error_mean": float(
                    np.mean(blind_gram_errors[reference_n])
                ),
                "blind_squared_n_error_median": float(
                    np.median(blind_gram_errors[reference_n])
                ),
                "fit_genetic_vector_rms_from_in_study": genetic_rms,
                "fit_rms_over_phenotype_sampling_rms": float(
                    genetic_rms / max(phenotype_rms, np.finfo(np.float64).tiny)
                ),
                "coefficient_bias": np.mean(delta, axis=0).tolist(),
            }
        )
    correct_improves = bool(
        records[-1]["correct_transfer_error_mean"]
        < records[0]["correct_transfer_error_mean"]
    )
    fit_improves = bool(
        records[-1]["fit_genetic_vector_rms_from_in_study"]
        < records[0]["fit_genetic_vector_rms_from_in_study"]
    )
    correct_beats_blind = bool(
        np.mean(
            [
                record["correct_transfer_error_mean"]
                < record["blind_squared_n_error_mean"]
                for record in records
            ]
        )
        >= 0.5
    )
    return {
        "design": {
            "same_study_genotype_and_phenotype_within_each_reference_comparison": True,
            "study_replicates_are_independent": True,
            "independent_matched_population_references": True,
            "reference_sizes": list(reference_sizes),
            "phenotype_sampling_error_separated_from_reference_error": True,
            "blind_comparator": "reference_gram_times_(study_n/reference_n)^2",
        },
        "exact_in_study_fit": {
            "maximum_condition_number": float(np.max(exact_conditions)),
            "phenotype_sampling_genetic_vector_rms": phenotype_rms,
        },
        "records": records,
        "correct_transfer_improves_with_reference_n": correct_improves,
        "fit_approaches_in_study_fit": fit_improves,
        "correct_transfer_beats_blind_squared_n_in_at_least_half_sizes": (
            correct_beats_blind
        ),
        "gate": bool(correct_improves and fit_improves and correct_beats_blind),
    }


def _failure_contract(args: argparse.Namespace) -> dict[str, Any]:
    from summit.context import (
        ContextRankError,
        build_disjoint_annotation_partition,
        fit_annotation_context_model,
        group_context_reference,
        group_context_trait_summary,
    )

    base = _variant_architecture(args.m, 4, seed=args.seed + 41_000)
    groups = _balanced_physical_groups(args.m, args.loo_groups)
    variant_hash = _canonical_hash({"failure_fixture": "ordered_variants", "m": args.m})

    def locally_rejected(annotations: np.ndarray, names: tuple[str, ...]) -> bool:
        candidate = VariantArchitecture(
            annotation_names=names,
            annotations=annotations,
            allele_frequencies=base.allele_frequencies,
            ld_correlations=base.ld_correlations,
            variant_chromosomes=base.variant_chromosomes,
            variant_positions=base.variant_positions,
        )
        try:
            _validate_disjoint_partition(candidate, groups)
        except ValueError:
            return True
        return False

    def publicly_rejected(
        annotations: np.ndarray,
        names: tuple[str, ...],
        group_values: Sequence[str] = groups,
    ) -> bool:
        try:
            build_disjoint_annotation_partition(
                annotations,
                names,
                variant_hash=variant_hash,
                loo_groups=group_values,
                source="negative_path_validation",
            )
        except ValueError:
            return True
        return False

    overlap = np.array(base.annotations, copy=True)
    overlap[0, 1] = 1.0
    uncovered = np.array(base.annotations, copy=True)
    uncovered[0] = 0.0
    empty = np.array(base.annotations, copy=True)
    empty[:, 0] = 0.0
    empty[np.argmax(empty[:, 1:], axis=1) < 0, 1] = 1.0
    nonbinary = np.array(base.annotations, copy=True)
    nonbinary[0, :] = 0.0
    nonbinary[0, :2] = 0.5
    unbalanced_groups = tuple(
        "large_group" if index < args.m - 1 else "singleton_group"
        for index in range(args.m)
    )
    single_support_groups = np.asarray(
        [f"annotation_group:{index}" for index in np.argmax(base.annotations, axis=1)],
        dtype=object,
    )
    failures = {
        "local_overlap_rejected": locally_rejected(overlap, base.annotation_names),
        "public_overlap_rejected": publicly_rejected(overlap, base.annotation_names),
        "local_uncovered_rejected": locally_rejected(uncovered, base.annotation_names),
        "public_uncovered_rejected": publicly_rejected(
            uncovered, base.annotation_names
        ),
        "local_empty_bin_rejected": locally_rejected(empty, base.annotation_names),
        "public_empty_bin_rejected": publicly_rejected(empty, base.annotation_names),
        "local_nonbinary_weights_rejected": locally_rejected(
            nonbinary, base.annotation_names
        ),
        "public_nonbinary_weights_rejected": publicly_rejected(
            nonbinary, base.annotation_names
        ),
        "public_unbalanced_groups_rejected": publicly_rejected(
            base.annotations, base.annotation_names, unbalanced_groups
        ),
        "public_single_group_annotation_support_rejected": publicly_rejected(
            base.annotations,
            base.annotation_names,
            tuple(str(value) for value in single_support_groups),
        ),
    }

    rank_fixture = _simulate_fixture(
        name="duplicated_basis_rank_failure",
        n=96,
        reference_n=96,
        m=96,
        q=2,
        k=2,
        loo_groups=12,
        seed=args.seed + 42_000,
    )
    duplicate_reference_basis = np.column_stack(
        [rank_fixture.reference.basis[:, 0], rank_fixture.reference.basis[:, 0]]
    )
    duplicate_study_basis = np.column_stack(
        [rank_fixture.study.basis[:, 0], rank_fixture.study.basis[:, 0]]
    )
    rank_fixture = SyntheticFixture(
        name=rank_fixture.name,
        q=2,
        k=2,
        architecture=rank_fixture.architecture,
        reference=SyntheticCohort(
            genotype=rank_fixture.reference.genotype,
            basis=duplicate_reference_basis,
            fixed_effects=np.column_stack([np.ones(96), rank_fixture.reference.pc]),
            phenotype=None,
            pc=rank_fixture.reference.pc,
        ),
        study=SyntheticCohort(
            genotype=rank_fixture.study.genotype,
            basis=duplicate_study_basis,
            fixed_effects=np.column_stack([np.ones(96), rank_fixture.study.pc]),
            phenotype=rank_fixture.study.phenotype,
            pc=rank_fixture.study.pc,
        ),
        true_omegas=rank_fixture.true_omegas,
        loo_groups=rank_fixture.loo_groups,
        sparse=False,
    )
    rank_failure_rejected = False
    rank_failure_type: str | None = None
    try:
        reference, summary, _, _, _ = _build_core_objects(
            rank_fixture, block_size=args.block_size, fit_model=False
        )
        partition = _public_partition(rank_fixture)
        grouped_reference = group_context_reference(reference, partition)
        grouped_summary = group_context_trait_summary(summary, partition)
        fit_annotation_context_model(
            grouped_reference, grouped_summary, partition=partition
        )
    except (ContextRankError, ValueError) as exc:
        rank_failure_rejected = True
        rank_failure_type = type(exc).__name__
    failures["public_duplicated_basis_rank_failure_rejected"] = rank_failure_rejected
    failures["duplicated_basis_exception"] = rank_failure_type

    identity_fixture = _simulate_fixture(
        name="partition_identity_failure",
        n=96,
        reference_n=96,
        m=96,
        q=1,
        k=2,
        loo_groups=12,
        seed=args.seed + 43_000,
    )
    identity_reference, _, _, _, _ = _build_core_objects(
        identity_fixture, block_size=args.block_size, fit_model=False
    )
    wrong_annotations = identity_fixture.architecture.annotations[:, ::-1]
    wrong_names = tuple(reversed(identity_fixture.architecture.annotation_names))
    wrong_partition = build_disjoint_annotation_partition(
        wrong_annotations,
        wrong_names,
        variant_hash=_synthetic_identity(identity_fixture)["variant_hash"],
        loo_groups=identity_fixture.loo_groups,
        source="wrong_annotation_order_negative_path",
    )
    identity_rejected = False
    try:
        group_context_reference(identity_reference, wrong_partition)
    except ValueError:
        identity_rejected = True
    failures["public_partition_identity_mismatch_rejected"] = identity_rejected
    failures["gate"] = bool(
        all(value for key, value in failures.items() if key.endswith("_rejected"))
    )
    return failures


def _performance_validation(args: argparse.Namespace) -> list[dict[str, Any]]:
    from summit.context import (
        ContextComponentIndex,
        ContextPairIndex,
        common_scale_features,
        context_kernel_actions,
        rank_revealing_projector,
    )

    records: list[dict[str, Any]] = []
    n = min(args.n, 128)
    m = args.m
    for k in (1, 4, 8):
        architecture = _variant_architecture(m, k, seed=args.seed + 50_000 + k)
        for q in (2, 3, 4):
            basis, fixed, pc = _simulate_basis_and_fixed(
                n, q, seed=args.seed + 51_000 + 10 * k + q
            )
            genotype = _simulate_genotype(
                architecture,
                n,
                seed=args.seed + 52_000 + 10 * k + q,
                pc=pc,
            )
            projector = rank_revealing_projector(fixed)
            components = ContextComponentIndex(
                architecture.annotation_names, ContextPairIndex(q)
            )
            feature_started = time.perf_counter()
            features = common_scale_features(genotype, basis, projector.projector)
            feature_seconds = time.perf_counter() - feature_started
            for probe_count in args.probe_counts:
                rng = np.random.default_rng(
                    args.seed + 53_000 + 100 * k + 10 * q + probe_count
                )
                probes = np.asarray(
                    rng.integers(0, 2, size=(n, probe_count)) * 2 - 1,
                    dtype=np.float64,
                )
                started = time.perf_counter()
                actions = context_kernel_actions(
                    genotype,
                    basis,
                    projector.projector,
                    architecture.annotations,
                    components,
                    probes,
                )
                elapsed = time.perf_counter() - started
                predicted = _resource_estimate(
                    n=n,
                    m=m,
                    q=q,
                    k=k,
                    residual_count=1,
                    probe_tile=probe_count,
                    loo_groups=args.loo_groups,
                )
                records.append(
                    {
                        "k": k,
                        "q": q,
                        "probe_count": probe_count,
                        "p_genetic": len(components),
                        "feature_seconds": float(feature_seconds),
                        "kernel_action_seconds": float(elapsed),
                        "action_shape": list(actions.shape),
                        "observed_action_bytes": int(actions.nbytes),
                        "predicted_action_bytes": int(
                            predicted["probe_action_buffer_bytes"]
                        ),
                        "byte_formula_matches": bool(
                            actions.nbytes == predicted["probe_action_buffer_bytes"]
                        ),
                        "finite": bool(np.all(np.isfinite(actions))),
                    }
                )
            del features
    return records


def _public_partition(fixture: SyntheticFixture) -> Any:
    from summit.context import build_disjoint_annotation_partition

    identity = _synthetic_identity(fixture)
    definitions = tuple(
        {
            "kind": "synthetic_maf_ld_like_bin",
            "index": index,
            "label": name,
            "maf_interval": [
                float(
                    np.min(
                        fixture.architecture.allele_frequencies[
                            fixture.architecture.annotations[:, index] == 1.0
                        ]
                    )
                ),
                float(
                    np.max(
                        fixture.architecture.allele_frequencies[
                            fixture.architecture.annotations[:, index] == 1.0
                        ]
                    )
                ),
            ],
            "ld_correlation": float(
                np.mean(
                    fixture.architecture.ld_correlations[
                        fixture.architecture.annotations[:, index] == 1.0
                    ]
                )
            ),
        }
        for index, name in enumerate(fixture.architecture.annotation_names)
    )
    return build_disjoint_annotation_partition(
        fixture.architecture.annotations,
        fixture.architecture.annotation_names,
        definitions=definitions,
        variant_hash=identity["variant_hash"],
        loo_groups=fixture.loo_groups,
        source="deterministic_synthetic_maf_ld_like_v1",
        source_digest=_canonical_hash(
            {
                "maf_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        fixture.architecture.allele_frequencies
                    ).tobytes()
                ).hexdigest(),
                "ld_sha256": hashlib.sha256(
                    np.ascontiguousarray(fixture.architecture.ld_correlations).tobytes()
                ).hexdigest(),
            }
        ),
    )


def _numeric_variant_axis_paths(value: Any, n_variants: int) -> list[str]:
    found: list[str] = []
    visited: set[int] = set()

    def visit(item: Any, path: str) -> None:
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, np.ndarray):
            if n_variants in item.shape:
                found.append(path)
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, f"{path}.{key}")
            return
        if isinstance(item, (tuple, list)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
            return
        fields = getattr(item, "__dataclass_fields__", None)
        if fields is not None:
            for name in fields:
                visit(getattr(item, name), f"{path}.{name}")

    visit(value, type(value).__name__)
    return found


def _public_grouped_validation(
    fixture: SyntheticFixture,
    core_reference: Any,
    core_summary: Any,
    core_fit: Any,
    reference_projector: Any,
    study_projector: Any,
    *,
    block_size: int,
) -> tuple[dict[str, Any], Any, Any, Any]:
    from summit.context import (
        build_grouped_context_reference,
        build_grouped_context_trait_summary,
        derive_annotation_contrast,
        derive_annotation_total,
        fit_annotation_context_model,
        group_context_reference,
        group_context_trait_summary,
        validate_disjoint_annotation_partition,
    )

    partition = _public_partition(fixture)
    validation = validate_disjoint_annotation_partition(partition)
    grouped_reference = group_context_reference(core_reference, partition)
    grouped_summary = group_context_trait_summary(core_summary, partition)
    identity = _synthetic_identity(fixture)
    direct_reference = build_grouped_context_reference(
        partition=partition,
        genotype=fixture.reference.genotype,
        basis=fixture.reference.basis,
        projector=reference_projector,
        basis_hash=identity["basis_hash"],
        fixed_effect_hash=identity["fixed_effect_hash"],
        variant_hash=identity["variant_hash"],
        genotype_scaling="synthetic_population_unit_variance",
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=min(block_size, 32),
    )
    direct_summary = build_grouped_context_trait_summary(
        partition=partition,
        genotype=fixture.study.genotype,
        basis=fixture.study.basis,
        phenotype=fixture.study.phenotype,
        projector=study_projector,
        residual_basis=np.ones((fixture.study.genotype.shape[0], 1)),
        residual_names=("identity",),
        basis_hash=identity["basis_hash"],
        fixed_effect_hash=identity["fixed_effect_hash"],
        variant_hash=identity["variant_hash"],
        genotype_scaling="synthetic_population_unit_variance",
        block_size=block_size,
    )
    context_grid = fixture.study.basis[: min(32, fixture.study.basis.shape[0])]
    basis_metric = fixture.reference.basis.T @ fixture.reference.basis
    basis_metric /= fixture.reference.basis.shape[0]
    grouped_fit = fit_annotation_context_model(
        direct_reference,
        direct_summary,
        partition=partition,
        context_grid=context_grid,
        basis_metric=basis_metric,
        project_psd=False,
    )
    reference_errors = {
        "gram": _scale_aware_error(direct_reference.gram, grouped_reference.gram),
        "same_person": _scale_aware_error(
            direct_reference.same_person, grouped_reference.same_person
        ),
        "grouped_gram": _scale_aware_error(
            _first_attr(
                direct_reference,
                "grouped_gram_numerator_contributions",
                "group_gram_numerator_contributions",
                "gram_numerator_contributions",
            ),
            _first_attr(
                grouped_reference,
                "grouped_gram_numerator_contributions",
                "group_gram_numerator_contributions",
                "gram_numerator_contributions",
            ),
        ),
    }
    summary_errors = {
        "rhs": _scale_aware_error(
            direct_summary.genetic_rhs, grouped_summary.genetic_rhs
        ),
        "trace": _scale_aware_error(
            direct_summary.genetic_traces, grouped_summary.genetic_traces
        ),
        "grouped_rhs": _scale_aware_error(
            _first_attr(
                direct_summary,
                "grouped_rhs_numerator_contributions",
                "group_rhs_numerator_contributions",
                "rhs_numerator_contributions",
            ),
            _first_attr(
                grouped_summary,
                "grouped_rhs_numerator_contributions",
                "group_rhs_numerator_contributions",
                "rhs_numerator_contributions",
            ),
        ),
        "grouped_trace": _scale_aware_error(
            _first_attr(
                direct_summary,
                "grouped_trace_numerator_contributions",
                "group_trace_numerator_contributions",
                "trace_numerator_contributions",
            ),
            _first_attr(
                grouped_summary,
                "grouped_trace_numerator_contributions",
                "group_trace_numerator_contributions",
                "trace_numerator_contributions",
            ),
        ),
        "grouped_genetic_residual": _scale_aware_error(
            _first_attr(
                direct_summary,
                "grouped_genetic_residual_numerator_contributions",
                "group_genetic_residual_numerator_contributions",
                "genetic_residual_numerator_contributions",
            ),
            _first_attr(
                grouped_summary,
                "grouped_genetic_residual_numerator_contributions",
                "group_genetic_residual_numerator_contributions",
                "genetic_residual_numerator_contributions",
            ),
        ),
    }
    fit_errors = {
        "raw_coefficients": _scale_aware_error(
            grouped_fit.raw_coefficients, core_fit.raw_coefficients
        ),
        "loo_coefficients": _scale_aware_error(
            grouped_fit.loo_coefficients, core_fit.loo_coefficients
        ),
        "joint_jackknife_covariance": _scale_aware_error(
            grouped_fit.jackknife_covariance, core_fit.jackknife_covariance
        ),
    }
    contrast_payload: dict[str, Any]
    if fixture.k < 2:
        contrast_payload = {"status": "not_applicable_single_annotation", "gate": True}
    else:
        left_name, right_name = fixture.architecture.annotation_names[:2]
        pair_count = fixture.q * (fixture.q + 1) // 2
        left = np.arange(pair_count)
        right = np.arange(pair_count, 2 * pair_count)
        contrast = derive_annotation_contrast(
            grouped_fit,
            left_name,
            right_name,
            scale="total_component",
            reference=direct_reference,
            summary=direct_summary,
            context_grid=context_grid,
            basis_metric=basis_metric,
        )
        expected_values = (
            grouped_fit.loo_coefficients[:, left]
            - grouped_fit.loo_coefficients[:, right]
        )
        expected_covariance = (
            (expected_values.shape[0] - 1.0)
            / expected_values.shape[0]
            * (
                (expected_values - np.mean(expected_values, axis=0)).T
                @ (expected_values - np.mean(expected_values, axis=0))
            )
        )
        per_mass = derive_annotation_contrast(
            grouped_fit,
            left_name,
            right_name,
            scale="per_annotation_mass",
            reference=direct_reference,
            summary=direct_summary,
        )
        contrast_covariance = _first_attr(
            contrast, "jackknife_covariance", "covariance"
        )
        contrast_surface = _first_attr(
            contrast, "covariance_surface_difference", "surface_difference"
        )
        trace_difference = _first_attr(
            contrast, "trace_contribution_difference", "trace_difference"
        )
        trace_standard_error = _first_attr(
            contrast,
            "trace_contribution_standard_error",
            "trace_standard_error",
        )
        trace_status = getattr(
            contrast,
            "trace_contribution_status",
            "defined_from_deleted_group_traces"
            if "defined_with_deleted_group_traces" in contrast.status
            else contrast.status,
        )
        contrast_payload = {
            "left": left_name,
            "right": right_name,
            "coefficient_difference": np.asarray(
                contrast.coefficient_difference
            ).tolist(),
            "standard_errors": np.asarray(contrast.standard_errors).tolist(),
            "joint_covariance_discrepancy": _scale_aware_error(
                contrast_covariance, expected_covariance
            ),
            "loo_difference_discrepancy": _scale_aware_error(
                contrast.loo_coefficient_differences, expected_values
            ),
            "surface_shape": list(np.asarray(contrast_surface).shape),
            "trace_contribution_difference": float(trace_difference),
            "trace_contribution_standard_error": float(trace_standard_error),
            "trace_contribution_status": trace_status,
            "per_mass_coefficient_difference": np.asarray(
                per_mass.coefficient_difference
            ).tolist(),
            "per_mass_loo_replicates": int(
                np.asarray(per_mass.loo_coefficient_differences).shape[0]
            ),
            "gate": bool(
                _scale_aware_error(contrast_covariance, expected_covariance) < 2.0e-11
                and _scale_aware_error(
                    contrast.loo_coefficient_differences, expected_values
                )
                < 2.0e-11
                and trace_status == "defined_from_deleted_group_traces"
            ),
        }
    total = derive_annotation_total(
        grouped_fit, reference=direct_reference, summary=direct_summary
    )
    pair_count = fixture.q * (fixture.q + 1) // 2
    expected_total = np.sum(
        grouped_fit.genetic_coefficients.reshape(fixture.k, pair_count), axis=0
    )
    expected_total_loo = np.sum(
        grouped_fit.loo_coefficients[:, : fixture.k * pair_count].reshape(
            grouped_fit.loo_coefficients.shape[0], fixture.k, pair_count
        ),
        axis=1,
    )
    total_coefficients = _first_attr(total, "coefficients", "coefficient_total")
    total_loo = _first_attr(total, "loo_coefficients", "loo_coefficient_totals")
    total_trace = _first_attr(total, "trace_contribution", "trace_total")
    total_payload = {
        "coefficients": np.asarray(total_coefficients).tolist(),
        "standard_errors": np.asarray(total.standard_errors).tolist(),
        "trace_contribution": float(total_trace),
        "coefficient_discrepancy": _scale_aware_error(
            total_coefficients, expected_total
        ),
        "loo_discrepancy": _scale_aware_error(total_loo, expected_total_loo),
        "gate": bool(
            _scale_aware_error(total_coefficients, expected_total) < 2.0e-11
            and _scale_aware_error(total_loo, expected_total_loo) < 2.0e-11
        ),
    }
    variant_axis_paths = _numeric_variant_axis_paths(
        (direct_reference, direct_summary), fixture.architecture.annotations.shape[0]
    )
    result = {
        "partition": {
            "status": validation["status"],
            "digest": partition.digest,
            "annotation_names": list(partition.annotation_names),
            "annotation_masses": np.asarray(partition.annotation_masses).tolist(),
            "group_count": len(
                getattr(
                    partition.group_balance, "loo_group_ids", partition.group_labels
                )
            ),
            "group_variant_counts": np.asarray(
                partition.group_balance.group_variant_counts
            ).tolist(),
            "minimum_retained_annotation_mass": float(
                np.min(partition.group_balance.minimum_retained_annotation_mass)
            ),
            "equal_group_delete_one_compatible": bool(
                getattr(
                    partition.group_balance,
                    "equal_group_delete_one_compatible",
                    partition.group_balance.total_groups_balanced
                    and partition.group_balance.every_deletion_retains_each_annotation,
                )
            ),
        },
        "direct_vs_adapter_reference_discrepancies": reference_errors,
        "direct_vs_adapter_summary_discrepancies": summary_errors,
        "grouped_vs_snp_fit_discrepancies": fit_errors,
        "grouped_numeric_variant_axis_paths": variant_axis_paths,
        "contrast": contrast_payload,
        "total": total_payload,
    }
    all_errors = list(reference_errors.values()) + list(summary_errors.values())
    all_errors += list(fit_errors.values())
    result["gate"] = bool(
        max(all_errors, default=0.0) < 2.0e-11
        and not variant_axis_paths
        and contrast_payload["gate"]
        and total_payload["gate"]
        and getattr(
            partition.group_balance,
            "equal_group_delete_one_compatible",
            partition.group_balance.total_groups_balanced
            and partition.group_balance.every_deletion_retains_each_annotation,
        )
    )
    return result, direct_reference, direct_summary, grouped_fit


def _public_resource_validation(args: argparse.Namespace) -> dict[str, Any]:
    from summit.context import AnnotationResourceRequest, estimate_annotation_resources

    records: list[dict[str, Any]] = []
    for independent in _resource_grid(args):
        fields = set(AnnotationResourceRequest.__dataclass_fields__)
        request_values: dict[str, Any] = {
            "n_reference": args.reference_n,
            "n_study": args.n,
            "n_variants": args.m,
            "q": int(independent["q"]),
            "k": int(independent["k"]),
            "h": 1,
            "probe_tile_size": min(max(args.probe_counts), 32),
        }
        aliases = {
            "loo_groups": args.loo_groups,
            "loo_group_count": args.loo_groups,
            "gram_probes": max(args.probe_counts),
            "gram_probe_count": max(args.probe_counts),
            "same_person_probes": max(args.probe_counts),
            "same_person_probe_count": max(args.probe_counts),
            "memory_cap_bytes": 8_000_000_000,
            "maximum_action_bytes": 8_000_000_000,
            "maximum_p_genetic": 100,
            "dtype": "float64",
            "dtype_bytes": 8,
            "reference_method": "hutchinson",
        }
        request_values.update(
            {name: value for name, value in aliases.items() if name in fields}
        )
        request = AnnotationResourceRequest(**request_values)
        observed = estimate_annotation_resources(request)

        def public_value(
            direct: str, mapping: str, nested: str, *, default: Any = None
        ) -> Any:
            if hasattr(observed, direct):
                return getattr(observed, direct)
            values = getattr(observed, mapping, {})
            return values.get(nested, default)

        pair_count = getattr(observed, "pair_count", independent["pair_count"])
        normal_bytes = public_value(
            "normal_matrix_bytes", "buffer_bytes", "normal_matrix"
        )
        action_bytes = public_value(
            "action_buffer_bytes", "buffer_bytes", "operator_action_tile"
        )
        snp_reference = public_value(
            "snp_reference_loo_bytes",
            "storage_bytes",
            "snp_reference_contribution_bytes",
        )
        grouped_reference = public_value(
            "grouped_reference_loo_bytes",
            "storage_bytes",
            "grouped_reference_contribution_bytes",
        )
        snp_trait = public_value(
            "snp_trait_loo_bytes", "storage_bytes", "snp_trait_contribution_bytes"
        )
        grouped_trait = public_value(
            "grouped_trait_loo_bytes",
            "storage_bytes",
            "grouped_trait_contribution_bytes",
        )
        operation_counts = getattr(observed, "operation_counts", {})
        tiles = max(int(operation_counts.get("probe_tiles", 1)), 1)
        source_gemms = getattr(
            observed,
            "source_gemms_per_gram_probe_tile",
            int(operation_counts.get("source_genotype_products", 0)) // tiles,
        )
        target_gemms = getattr(
            observed,
            "target_gemms_per_gram_probe_tile",
            int(operation_counts.get("annotation_target_genotype_products", 0))
            // tiles,
        )
        within_limits = getattr(
            observed,
            "within_limits",
            getattr(observed, "within_memory_cap", False),
        )
        limit_reasons = getattr(
            observed,
            "limit_reasons",
            (() if within_limits else (str(getattr(observed, "verdict", "unknown")),)),
        )
        discrepancies = {
            "pair_count": abs(int(pair_count) - int(independent["pair_count"])),
            "p_genetic": abs(int(observed.p_genetic) - int(independent["p_genetic"])),
            "p_total": abs(int(observed.p_total) - int(independent["p_total"])),
            "normal_matrix_bytes": abs(
                int(normal_bytes) - int(independent["normal_matrix_bytes"])
            ),
            "action_buffer_bytes": abs(
                int(action_bytes) - int(independent["probe_action_buffer_bytes"])
            ),
            "snp_reference_loo_bytes": abs(
                int(snp_reference)
                - int(independent["reference_snp_contribution_bytes"])
            ),
            "grouped_reference_loo_bytes": abs(
                int(grouped_reference)
                - int(independent["lossless_grouped_reference_bytes"])
            ),
            "snp_trait_loo_bytes": abs(
                int(snp_trait) - int(independent["trait_snp_contribution_bytes"])
            ),
            "grouped_trait_loo_bytes": abs(
                int(grouped_trait) - int(independent["lossless_grouped_trait_bytes"])
            ),
        }
        records.append(
            {
                **independent,
                "public": {
                    "within_limits": bool(within_limits),
                    "limit_reasons": list(limit_reasons),
                    "condition_status": observed.condition_status,
                    "source_gemms_per_gram_probe_tile": int(source_gemms),
                    "target_gemms_per_gram_probe_tile": int(target_gemms),
                },
                "absolute_discrepancies": discrepancies,
                "gate": bool(
                    max(discrepancies.values(), default=0) == 0
                    and source_gemms
                    == independent["source_genotype_products_per_probe_tile"]
                    and target_gemms
                    == independent["target_genotype_products_per_probe_tile"]
                ),
            }
        )
    return {"records": records, "gate": bool(all(item["gate"] for item in records))}


def _resample_fixture_phenotype(
    fixture: SyntheticFixture, *, seed: int
) -> tuple[SyntheticFixture, np.ndarray]:
    from summit.context import rank_revealing_projector

    rng = np.random.default_rng(seed)
    m = fixture.architecture.annotations.shape[0]
    effects = np.zeros((m, fixture.q), dtype=np.float64)
    memberships = np.argmax(fixture.architecture.annotations, axis=1)
    for annotation_index in range(fixture.k):
        selected = np.flatnonzero(memberships == annotation_index)
        active = selected
        if fixture.sparse and np.any(fixture.true_omegas[annotation_index]):
            active = selected[: max(2, selected.size // 10)]
        if not np.any(fixture.true_omegas[annotation_index]):
            continue
        effects[active] = rng.multivariate_normal(
            np.zeros(fixture.q),
            fixture.true_omegas[annotation_index] / float(active.size),
            size=active.size,
        )
    genetic = np.sum(
        fixture.study.basis * (fixture.study.genotype @ effects),
        axis=1,
        dtype=np.float64,
    )
    phenotype = genetic + 0.55 * rng.normal(size=genetic.size)
    projector = rank_revealing_projector(fixture.study.fixed_effects)
    projected = projector.projector @ phenotype
    normalization_squared = projector.residual_rank / float(projected @ projected)
    expected_normalized_omegas = normalization_squared * fixture.true_omegas
    updated = SyntheticFixture(
        name=fixture.name,
        q=fixture.q,
        k=fixture.k,
        architecture=fixture.architecture,
        reference=fixture.reference,
        study=SyntheticCohort(
            genotype=fixture.study.genotype,
            basis=fixture.study.basis,
            fixed_effects=fixture.study.fixed_effects,
            phenotype=phenotype,
            pc=fixture.study.pc,
        ),
        true_omegas=fixture.true_omegas,
        loo_groups=fixture.loo_groups,
        sparse=fixture.sparse,
    )
    return updated, expected_normalized_omegas


def _empirical_recovery_validation(
    fixture: SyntheticFixture,
    grouped_reference: Any,
    partition: Any,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from summit.context import (
        build_grouped_context_trait_summary,
        fit_annotation_context_model,
        omegas_to_coefficients,
        rank_revealing_projector,
    )

    identity = _synthetic_identity(fixture)
    estimates: list[np.ndarray] = []
    expected_values: list[np.ndarray] = []
    standard_errors: list[np.ndarray] = []
    rank_values: list[int] = []
    condition_values: list[float] = []
    replicate_count = args.transport_replicates
    for replicate in range(replicate_count):
        sampled, expected_omegas = _resample_fixture_phenotype(
            fixture, seed=args.seed + 60_000 + replicate * 127
        )
        projector = rank_revealing_projector(sampled.study.fixed_effects)
        summary = build_grouped_context_trait_summary(
            partition=partition,
            genotype=sampled.study.genotype,
            basis=sampled.study.basis,
            phenotype=sampled.study.phenotype,
            projector=projector,
            residual_basis=np.ones((sampled.study.genotype.shape[0], 1)),
            residual_names=("identity",),
            basis_hash=identity["basis_hash"],
            fixed_effect_hash=identity["fixed_effect_hash"],
            variant_hash=identity["variant_hash"],
            genotype_scaling="synthetic_population_unit_variance",
            block_size=args.block_size,
        )
        fit = fit_annotation_context_model(
            grouped_reference, summary, partition=partition, project_psd=False
        )
        estimates.append(np.asarray(fit.genetic_coefficients))
        expected_values.append(
            omegas_to_coefficients(expected_omegas, fit.component_index)
        )
        standard_errors.append(
            np.asarray(fit.standard_errors)[: len(fit.component_index)]
        )
        rank_values.append(int(fit.solve.rank))
        condition_values.append(float(fit.solve.condition_number))
    estimates_array = np.asarray(estimates)
    expected_array = np.asarray(expected_values)
    se_array = np.asarray(standard_errors)
    valid_se = np.isfinite(se_array) & (se_array > 0.0)
    nonnull = np.any(np.abs(expected_array) > 1.0e-12, axis=0)
    covered = np.abs(estimates_array - expected_array) <= 1.96 * se_array
    pair_count = fixture.q * (fixture.q + 1) // 2
    null_indices = np.arange((fixture.k - 1) * pair_count, fixture.k * pair_count)
    null_valid = valid_se[:, null_indices]
    null_z = np.divide(
        estimates_array[:, null_indices],
        se_array[:, null_indices],
        out=np.full_like(estimates_array[:, null_indices], np.nan),
        where=null_valid,
    )
    nonnull_valid = valid_se[:, nonnull]
    coverage_rate = float(
        np.mean(covered[:, nonnull][nonnull_valid]) if np.any(nonnull_valid) else np.nan
    )
    null_exceedance = float(
        np.mean(np.abs(null_z[np.isfinite(null_z)]) > 1.96)
        if np.any(np.isfinite(null_z))
        else np.nan
    )
    mean_bias = np.mean(estimates_array - expected_array, axis=0)
    return {
        "replicates": replicate_count,
        "reference_for_repeated_null_audit": (
            "exact_in_study_T_grouped_by_frozen_physical_blocks"
        ),
        "mechanisms": [
            "annotation_specific_amplification",
            "annotation_specific_context_effect_heterogeneity",
            "one_exact_null_annotation",
            "maf_ld_like_genotype_bins",
        ],
        "mean_component_bias": mean_bias.tolist(),
        "component_bias_rmse": float(np.sqrt(np.mean(mean_bias**2))),
        "nonnull_nominal_95_percent_coverage": coverage_rate,
        "null_bin_abs_z_above_1.96_fraction": null_exceedance,
        "null_bin_z_median": float(np.nanmedian(null_z)),
        "null_bin_z_90_percent_absolute_quantile": float(
            np.nanquantile(np.abs(null_z), 0.90)
        ),
        "fit_rank_minimum": min(rank_values),
        "fit_rank_maximum": max(rank_values),
        "condition_number_median": float(np.median(condition_values)),
        "interpretation": (
            "descriptive small-replicate audit; the |z|>1.96 fraction is not a "
            "declared or calibrated family-wise false-positive rate"
        ),
        "gate": bool(
            np.all(np.isfinite(estimates_array))
            and np.all(np.isfinite(se_array))
            and np.isfinite(coverage_rate)
            and np.isfinite(null_exceedance)
            and min(rank_values) == estimates_array.shape[1] + 1
        ),
    }


def _read_real_annotation_fixture(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import pandas as pd
        from bed_reader import open_bed
    except ImportError as exc:
        raise RuntimeError("--real-traits requires pandas and bed-reader.") from exc

    required = [
        Path(f"{args.geno_prefix}.bed"),
        Path(f"{args.geno_prefix}.bim"),
        Path(f"{args.geno_prefix}.fam"),
        args.covariate_file,
        args.phenotype_root / "bmi.pheno",
        args.phenotype_root / "c_reactive_prot.pheno",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required real-data inputs are absent: {missing}")

    def read_table(path: Path, usecols: Sequence[str]) -> Any:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            usecols=list(usecols),
            dtype={"FID": str, "IID": str},
        )
        if frame[["FID", "IID"]].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate identifiers.")
        return frame

    fam = pd.read_csv(
        Path(f"{args.geno_prefix}.fam"),
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype=str,
    )
    if fam.duplicated().any():
        raise ValueError("PLINK FAM contains duplicate identifiers.")
    fam["bed_row"] = np.arange(len(fam), dtype=np.int64)
    covariates = read_table(
        args.covariate_file, ("FID", "IID", "sex", "age", *PC_COLUMNS)
    )
    selected = fam.merge(covariates, on=["FID", "IID"], validate="one_to_one")
    for trait in ("bmi", "c_reactive_prot"):
        frame = read_table(
            args.phenotype_root / f"{trait}.pheno", ("FID", "IID", "pheno")
        ).rename(columns={"pheno": trait})
        selected = selected.merge(frame, on=["FID", "IID"], validate="one_to_one")
    numeric_names = ["sex", "age", *PC_COLUMNS, "bmi", "c_reactive_prot"]
    numeric = selected[numeric_names].to_numpy(dtype=np.float64)
    complete = np.all(np.isfinite(numeric), axis=1) & np.all(numeric != -9.0, axis=1)
    intersection = selected.loc[complete].copy()
    if len(intersection) < args.real_n:
        raise ValueError(
            f"Only {len(intersection)} complete cases remain for --real-n={args.real_n}."
        )
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 80_000]))
    chosen_rows = np.sort(
        rng.choice(len(intersection), size=args.real_n, replace=False)
    )
    selected = (
        intersection.iloc[chosen_rows].sort_values("bed_row").reset_index(drop=True)
    )

    bed = open_bed(Path(f"{args.geno_prefix}.bed"))
    candidate_count = min(int(bed.sid_count), max(4 * args.real_m, 384))
    candidate_indices = np.unique(
        np.linspace(0, int(bed.sid_count) - 1, candidate_count, dtype=np.int64)
    )
    raw = bed.read(
        index=(selected["bed_row"].to_numpy(dtype=np.int64), candidate_indices),
        dtype="float64",
        order="C",
    )
    missing_rate = np.mean(np.isnan(raw), axis=0)
    means = np.nanmean(raw, axis=0)
    allele_frequency = means / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    variance = np.nanvar(raw, axis=0, ddof=1)
    valid = (
        np.isfinite(means)
        & np.isfinite(variance)
        & (variance > np.finfo(np.float64).eps)
        & (missing_rate <= 0.05)
        & (maf >= 0.05)
        & (maf <= 0.5)
    )
    raw = raw[:, valid]
    candidate_indices = candidate_indices[valid]
    maf = maf[valid]
    means = means[valid]
    if raw.shape[1] < args.real_m:
        raise ValueError(
            f"Only {raw.shape[1]} deterministic candidate variants passed QC."
        )
    missing_values = np.isnan(raw)
    if np.any(missing_values):
        raw[missing_values] = np.broadcast_to(means[None, :], raw.shape)[missing_values]
    standardized_candidates = _standardize(raw)
    correlation = (
        standardized_candidates.T @ standardized_candidates / (args.real_n - 1.0)
    )
    ld_proxy = np.sum(correlation * correlation, axis=1) - 1.0
    ld_split = float(np.median(ld_proxy))
    maf_split = 0.20
    target_counts = np.full(4, args.real_m // 4, dtype=np.int64)
    target_counts[: args.real_m % 4] += 1
    bin_index = 2 * (maf >= maf_split).astype(np.int64) + (ld_proxy >= ld_split).astype(
        np.int64
    )
    selected_columns: list[int] = []
    available_counts = []
    for index in range(4):
        available = np.flatnonzero(bin_index == index)
        available_counts.append(int(available.size))
        if available.size < target_counts[index]:
            raise ValueError(
                "The deterministic real candidate panel does not support the "
                f"declared four bins: bin {index} has {available.size}, needs "
                f"{target_counts[index]}. Increase --real-m candidate coverage."
            )
        selected_columns.extend(available[: target_counts[index]].tolist())
    selected_columns_array = np.asarray(selected_columns, dtype=np.int64)
    genotype = standardized_candidates[:, selected_columns_array]
    chosen_maf = maf[selected_columns_array]
    chosen_ld = ld_proxy[selected_columns_array]
    chosen_variant_indices = candidate_indices[selected_columns_array]
    groups = tuple(f"block:{index % args.loo_groups}" for index in range(args.real_m))
    variant_hash = _canonical_hash(
        {
            "selected_index_sha256": hashlib.sha256(
                np.ascontiguousarray(chosen_variant_indices).tobytes()
            ).hexdigest(),
            "ordered": True,
        }
    )
    fixture = {
        "selected": selected,
        "genotype": genotype,
        "maf": chosen_maf,
        "ld_proxy": chosen_ld,
        "loo_groups": groups,
        "variant_hash": variant_hash,
        "ld_edges": (
            float(np.min(chosen_ld)),
            ld_split,
            float(np.max(chosen_ld)),
        ),
    }
    diagnostics = {
        "source_sample_count": int(bed.iid_count),
        "source_variant_count": int(bed.sid_count),
        "complete_case_intersection_count": int(len(intersection)),
        "selected_sample_count": args.real_n,
        "selected_variant_count": args.real_m,
        "candidate_variant_count": int(candidate_count),
        "candidate_qc_pass_count": int(raw.shape[1]),
        "candidate_bin_counts": available_counts,
        "selected_bin_counts": target_counts.tolist(),
        "maf_edges": [0.05, maf_split, 0.5],
        "ld_proxy_edges": [
            float(np.min(chosen_ld)),
            ld_split,
            float(np.max(chosen_ld)),
        ],
        "ld_proxy_definition": (
            "sum of within-candidate-panel sample r-squared excluding self; "
            "numerical sanity proxy, not production LD-score annotation"
        ),
        "bin_edge_calibration": "fixed_once_on_reference_candidate_panel",
        "sample_selection": "seeded_complete_case_subset_sorted_in_memory",
        "variant_selection": "evenly_spaced_candidates_qc_then_balanced_four_bins",
        "ordered_variant_hash": variant_hash,
    }
    return fixture, diagnostics


def _real_annotation_sanity(args: argparse.Namespace) -> dict[str, Any]:
    from summit.context import (
        build_grouped_context_reference,
        build_grouped_context_trait_summary,
        build_maf_ld_partition,
        derive_annotation_contrast,
        derive_annotation_total,
        fit_annotation_context_model,
        rank_revealing_projector,
    )

    fixture, diagnostics = _read_real_annotation_fixture(args)
    selected = fixture["selected"]
    genotype = np.asarray(fixture["genotype"])
    age = _standardize(selected["age"].to_numpy(dtype=np.float64))
    sex = _standardize(selected["sex"].to_numpy(dtype=np.float64))
    pcs = _standardize(selected[list(PC_COLUMNS)].to_numpy(dtype=np.float64))
    basis = np.column_stack([np.ones(args.real_n), age])
    fixed = np.column_stack([np.ones(args.real_n), age, sex, pcs, age * pcs[:, 0]])
    projector = rank_revealing_projector(fixed)
    basis_hash = _canonical_hash(
        {
            "basis": "real_age_standardized_reference_calibration_v1",
            "q": 2,
        }
    )
    fixed_effect_hash = _canonical_hash(
        {
            "fixed_effects": "intercept_age_sex_pc1_to_pc5_age_by_pc1",
            "version": 1,
        }
    )
    source_digest = _canonical_hash(
        {
            "maf_sha256": hashlib.sha256(
                np.ascontiguousarray(fixture["maf"]).tobytes()
            ).hexdigest(),
            "ld_proxy_sha256": hashlib.sha256(
                np.ascontiguousarray(fixture["ld_proxy"]).tobytes()
            ).hexdigest(),
        }
    )
    partition = build_maf_ld_partition(
        fixture["maf"],
        fixture["ld_proxy"],
        maf_edges=(0.05, 0.20, 0.5),
        ld_edges=fixture["ld_edges"],
        variant_hash=fixture["variant_hash"],
        loo_groups=fixture["loo_groups"],
        source="caller_supplied_genotype_panel_maf_ld_proxy",
        source_digest=source_digest,
    )
    reference = build_grouped_context_reference(
        partition=partition,
        genotype=genotype,
        basis=basis,
        projector=projector,
        basis_hash=basis_hash,
        fixed_effect_hash=fixed_effect_hash,
        variant_hash=fixture["variant_hash"],
        genotype_scaling="study_mean_imputed_and_unit_variance",
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=min(args.block_size, 32),
    )
    context_grid = np.column_stack([np.ones(25), np.linspace(-2.0, 2.0, 25)])
    basis_metric = basis.T @ basis / args.real_n
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trait in ("bmi", "c_reactive_prot"):
        summary = build_grouped_context_trait_summary(
            partition=partition,
            genotype=genotype,
            basis=basis,
            phenotype=selected[trait].to_numpy(dtype=np.float64),
            projector=projector,
            residual_basis=np.ones((args.real_n, 1)),
            residual_names=("identity",),
            basis_hash=basis_hash,
            fixed_effect_hash=fixed_effect_hash,
            variant_hash=fixture["variant_hash"],
            genotype_scaling="study_mean_imputed_and_unit_variance",
            block_size=args.block_size,
        )
        fit = fit_annotation_context_model(
            reference,
            summary,
            partition=partition,
            context_grid=context_grid,
            basis_metric=basis_metric,
            project_psd=False,
        )
        total = derive_annotation_total(fit, reference=reference, summary=summary)
        pair_count = 3
        trace_contributions = []
        surfaces = []
        intercept_variance_se = []
        for index, omega in enumerate(np.asarray(fit.raw_omegas)):
            selected_components = slice(index * pair_count, (index + 1) * pair_count)
            trace_contributions.append(
                float(
                    fit.genetic_coefficients[selected_components]
                    @ fit.equations.traces[selected_components]
                )
            )
            surfaces.append(np.diag(context_grid @ omega @ context_grid.T).tolist())
            intercept_variance_se.append(float(fit.standard_errors[index * pair_count]))
        pairwise = []
        for left_index in range(len(partition.annotation_names) - 1):
            right_index = left_index + 1
            contrast = derive_annotation_contrast(
                fit,
                partition.annotation_names[left_index],
                partition.annotation_names[right_index],
                scale="total_component",
                reference=reference,
                summary=summary,
                context_grid=context_grid,
                basis_metric=basis_metric,
            )
            contrast_trace = _first_attr(
                contrast, "trace_contribution_difference", "trace_difference"
            )
            contrast_trace_se = _first_attr(
                contrast,
                "trace_contribution_standard_error",
                "trace_standard_error",
            )
            pairwise.append(
                {
                    "left": partition.annotation_names[left_index],
                    "right": partition.annotation_names[right_index],
                    "intercept_variance_difference": float(
                        contrast.coefficient_difference[0]
                    ),
                    "intercept_variance_difference_se": float(
                        contrast.standard_errors[0]
                    ),
                    "trace_contribution_difference": float(contrast_trace),
                    "trace_contribution_difference_se": float(contrast_trace_se),
                }
            )
        records.append(
            {
                "trait": trait,
                "rank": int(fit.solve.rank),
                "dimension": int(fit.raw_coefficients.size),
                "condition_number": float(fit.solve.condition_number),
                "relative_residual": float(fit.solve.relative_residual),
                "raw_omegas": np.asarray(fit.raw_omegas).tolist(),
                "intercept_variance_standard_errors": intercept_variance_se,
                "annotation_trace_contributions": trace_contributions,
                "total_coefficients": np.asarray(
                    _first_attr(total, "coefficients", "coefficient_total")
                ).tolist(),
                "total_standard_errors": np.asarray(total.standard_errors).tolist(),
                "total_trace_contribution": float(
                    _first_attr(total, "trace_contribution", "trace_total")
                ),
                "age_grid_standardized": context_grid[:, 1].tolist(),
                "annotation_variance_surfaces": surfaces,
                "adjacent_annotation_contrasts": pairwise,
                "loo_replicates": int(fit.loo_coefficients.shape[0]),
                "finite": bool(
                    np.all(np.isfinite(fit.raw_coefficients))
                    and np.all(np.isfinite(fit.jackknife_covariance))
                ),
            }
        )
    return {
        "purpose": "privacy_preserving_numerical_sanity_not_significance_gate",
        "diagnostics": diagnostics,
        "partition": {
            "annotation_names": list(partition.annotation_names),
            "annotation_masses": np.asarray(partition.annotation_masses).tolist(),
            "loo_group_count": len(
                getattr(
                    partition.group_balance, "loo_group_ids", partition.group_labels
                )
            ),
            "minimum_retained_annotation_mass": float(
                np.min(partition.group_balance.minimum_retained_annotation_mass)
            ),
            "digest": partition.digest,
        },
        "records": records,
        "elapsed_seconds": float(time.perf_counter() - started),
        "gate": bool(all(record["finite"] for record in records)),
        "privacy": {
            "identifiers_written": False,
            "sample_rows_written": False,
            "variant_rows_written": False,
            "outputs": "aggregate_counts_hashes_fit_diagnostics_and_surfaces_only",
        },
    }


def _run_synthetic_case(
    fixture: SyntheticFixture, *, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    local_partition = _validate_disjoint_partition(
        fixture.architecture, fixture.loo_groups
    )
    (
        core_reference,
        core_summary,
        core_fit,
        reference_projector,
        study_projector,
    ) = _build_core_objects(fixture, block_size=args.block_size, fit_model=True)
    assert core_fit is not None
    dense = _dense_oracle_payload(
        fixture,
        core_reference,
        core_summary,
        reference_projector,
        study_projector,
    )
    independent_grouping = _manual_grouped_payload(
        fixture, core_reference, core_summary, core_fit
    )
    manual_contrast = _manual_contrast_payload(fixture, core_fit)
    (
        public_grouped,
        grouped_reference,
        grouped_summary,
        grouped_fit,
    ) = _public_grouped_validation(
        fixture,
        core_reference,
        core_summary,
        core_fit,
        reference_projector,
        study_projector,
        block_size=args.block_size,
    )
    fit_payload = _fit_payload(fixture, grouped_fit)
    result = {
        "name": fixture.name,
        "q": fixture.q,
        "k": fixture.k,
        "n_study": fixture.study.genotype.shape[0],
        "n_reference": fixture.reference.genotype.shape[0],
        "m": fixture.architecture.annotations.shape[0],
        "sparse_high_leverage_architecture": fixture.sparse,
        "local_partition_oracle": local_partition,
        "dense_oracle": dense,
        "independent_grouped_oracle": independent_grouping,
        "manual_joint_contrast": manual_contrast,
        "public_grouped_api": public_grouped,
        "fit": fit_payload,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result["gate"] = bool(
        local_partition["gate"]
        and dense["gate"]
        and independent_grouping["gate"]
        and manual_contrast["gate"]
        and public_grouped["gate"]
        and fit_payload["finite"]
        and fit_payload["rank"] == fit_payload["dimension"]
        and fit_payload["relative_residual"] < 1.0e-8
    )
    plot = {
        "name": fixture.name,
        "k": fixture.k,
        "q": fixture.q,
        "truth": np.asarray(fit_payload["expected_normalized_generating_omegas"]),
        "fitted": np.asarray(fit_payload["raw_omegas"]),
        "contrast": manual_contrast,
    }
    del grouped_reference, grouped_summary
    return result, plot


def _plot_recovery(plot_records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        1,
        len(plot_records),
        figsize=(4.3 * len(plot_records), 4),
        constrained_layout=True,
    )
    if len(plot_records) == 1:
        axes = [axes]
    for axis, record in zip(axes, plot_records, strict=True):
        truth = np.asarray(record["truth"], dtype=np.float64)
        fitted = np.asarray(record["fitted"], dtype=np.float64)
        x = truth.ravel()
        y = fitted.ravel()
        limit = max(float(np.max(np.abs(np.concatenate([x, y])))), 0.05)
        axis.axline((0.0, 0.0), slope=1.0, color="0.45", linestyle="--", linewidth=1)
        axis.scatter(
            x,
            y,
            c=np.repeat(np.arange(truth.shape[0]), truth.shape[1] ** 2),
            s=22,
            cmap="viridis",
        )
        axis.set_xlim(-1.08 * limit, 1.08 * limit)
        axis.set_ylim(-1.08 * limit, 1.08 * limit)
        mechanism = {
            1: "common covariance",
            2: "amplification + null bin",
            4: "sparse heterogeneous bins",
        }[int(record["k"])]
        axis.set_title(f"K={record['k']}, Q={record['q']}: {mechanism}")
        axis.set_xlabel("expected normalized coefficient")
        axis.set_ylabel("fitted raw coefficient")
        axis.grid(alpha=0.18)
    figure.suptitle("Annotation-specific covariance recovery (single phenotype)")
    _save_figure(figure, output_dir, "recovery")


def _plot_loo_contrasts(
    synthetic: Sequence[Mapping[str, Any]],
    empirical: Mapping[str, Any],
    output_dir: Path,
) -> None:
    usable = [
        item
        for item in synthetic
        if item["manual_joint_contrast"].get("status")
        != "not_applicable_single_annotation"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = [f"K={item['k']}, Q={item['q']}" for item in usable]
    estimates = np.asarray(
        [item["manual_joint_contrast"]["estimate"] for item in usable]
    )
    full = np.asarray(
        [item["manual_joint_contrast"]["standard_error"] for item in usable]
    )
    marginal = np.asarray(
        [
            item["manual_joint_contrast"]["marginal_only_standard_error"]
            for item in usable
        ]
    )
    positions = np.arange(len(usable))
    axes[0].errorbar(
        positions - 0.06,
        estimates,
        yerr=1.96 * full,
        fmt="o",
        capsize=3,
        label="full joint JK",
    )
    axes[0].errorbar(
        positions + 0.06,
        estimates,
        yerr=1.96 * marginal,
        fmt="x",
        capsize=3,
        label="marginals only",
    )
    axes[0].axhline(0.0, color="0.4", linewidth=1)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("bin 1 − bin 2 intercept variance")
    axes[0].set_title("Joint covariance matters for contrasts")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.18)

    metrics = [
        float(empirical["nonnull_nominal_95_percent_coverage"]),
        float(empirical["null_bin_abs_z_above_1.96_fraction"]),
    ]
    axes[1].bar(
        ["nonnull\ncoverage", "null |z|>1.96\n(descriptive)"],
        metrics,
        color=["#4C78A8", "#F58518"],
    )
    axes[1].axhline(0.95, color="#4C78A8", linestyle="--", linewidth=1)
    axes[1].axhline(0.05, color="#F58518", linestyle=":", linewidth=1)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("fraction")
    axes[1].set_title(
        f"Repeated phenotype audit (R={empirical['replicates']}; descriptive)"
    )
    axes[1].grid(axis="y", alpha=0.18)
    figure.suptitle("Approximate grouped jackknife diagnostics")
    _save_figure(figure, output_dir, "loo_contrasts")


def _plot_transport(transport: Mapping[str, Any], output_dir: Path) -> None:
    records = transport["records"]
    n_values = np.asarray([record["reference_n"] for record in records])
    correct = np.asarray([record["correct_transfer_error_mean"] for record in records])
    blind = np.asarray([record["blind_squared_n_error_mean"] for record in records])
    fit_error = np.asarray(
        [record["fit_genetic_vector_rms_from_in_study"] for record in records]
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    axes[0].plot(n_values, correct, "o-", label="N and N(N−1) transfer")
    axes[0].plot(n_values, blind, "s--", label="blind squared-N")
    axes[0].set_xlabel("reference N")
    axes[0].set_ylabel("mean scale-aware Gram error")
    axes[0].set_title("Matched independent-reference transport")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18)
    axes[1].plot(n_values, fit_error, "o-", color="#54A24B")
    axes[1].set_xlabel("reference N")
    axes[1].set_ylabel("relative coefficient difference")
    axes[1].set_title("Approach to exact in-study-T fit")
    axes[1].grid(alpha=0.18)
    _save_figure(figure, output_dir, "transport")


def _plot_resources(
    resources: Mapping[str, Any],
    performance: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    records = resources["records"]
    for q in (2, 3, 4):
        selected = [record for record in records if record["q"] == q]
        p = np.asarray([record["p_genetic"] for record in selected])
        snp = np.asarray(
            [record["reference_snp_contribution_bytes"] for record in selected]
        )
        grouped = np.asarray(
            [record["lossless_grouped_reference_bytes"] for record in selected]
        )
        axes[0].plot(p, snp / 2**20, "o-", label=f"SNP Q={q}")
        axes[0].plot(p, grouped / 2**20, "o--", label=f"grouped Q={q}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("genetic components P_g")
    axes[0].set_ylabel("reference LOO storage (MiB)")
    axes[0].set_title("Lossless grouping removes the M axis")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(alpha=0.18)
    for probe in sorted({int(item["probe_count"]) for item in performance}):
        selected = [item for item in performance if item["probe_count"] == probe]
        p = np.asarray([item["p_genetic"] for item in selected])
        seconds = np.asarray([item["kernel_action_seconds"] for item in selected])
        order = np.argsort(p)
        axes[1].plot(p[order], seconds[order], "o-", label=f"B={probe}")
    axes[1].set_xlabel("genetic components P_g")
    axes[1].set_ylabel("kernel-action seconds")
    axes[1].set_title("Correctness-backend action scaling")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    _save_figure(figure, output_dir, "resources")


def _plot_real(real: Mapping[str, Any], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        1,
        len(real["records"]),
        figsize=(5 * len(real["records"]), 4),
        squeeze=False,
        constrained_layout=True,
    )
    names = real["partition"]["annotation_names"]
    for axis, record in zip(axes[0], real["records"], strict=True):
        grid = np.asarray(record["age_grid_standardized"])
        for name, surface in zip(
            names, record["annotation_variance_surfaces"], strict=True
        ):
            axis.plot(grid, surface, label=name)
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_xlabel("standardized age")
        axis.set_ylabel("raw fitted covariance surface diagonal")
        axis.set_title(f"{record['trait']}\ncondition={record['condition_number']:.2g}")
        axis.grid(alpha=0.18)
    axes[0, -1].legend(frameon=False, fontsize=7)
    figure.suptitle(
        "Real-trait MAF/LD-bin numerical sanity (not significance evidence)"
    )
    _save_figure(figure, output_dir, "real_traits")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    fixtures = (
        _simulate_fixture(
            name="k1_q1_common_covariance",
            n=args.n,
            reference_n=args.reference_n,
            m=args.m,
            q=1,
            k=1,
            loo_groups=args.loo_groups,
            seed=args.seed + 100,
        ),
        _simulate_fixture(
            name="k2_q2_amplification_and_null",
            n=args.n,
            reference_n=args.reference_n,
            m=args.m,
            q=2,
            k=2,
            loo_groups=args.loo_groups,
            seed=args.seed + 200,
        ),
        _simulate_fixture(
            name="k4_q4_sparse_heterogeneous",
            n=args.n,
            reference_n=args.reference_n,
            m=args.m,
            q=4,
            k=4,
            loo_groups=args.loo_groups,
            seed=args.seed + 300,
            sparse=True,
        ),
    )
    synthetic: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    grouped_for_empirical: Any | None = None
    partition_for_empirical: Any | None = None
    for fixture in fixtures:
        result, plot = _run_synthetic_case(fixture, args=args)
        synthetic.append(result)
        plot_records.append(plot)
        if fixture.k == 2:
            reference = _build_reference_for_cohort(
                fixture, fixture.study, block_size=args.block_size
            )
            partition_for_empirical = _public_partition(fixture)
            from summit.context import group_context_reference

            grouped_for_empirical = group_context_reference(
                reference, partition_for_empirical
            )
    if grouped_for_empirical is None or partition_for_empirical is None:
        raise AssertionError("K=2 empirical recovery fixture was not constructed.")
    empirical = _empirical_recovery_validation(
        fixtures[1],
        grouped_for_empirical,
        partition_for_empirical,
        args=args,
    )
    transport_fixture = fixtures[1]
    transport = _transport_validation(transport_fixture, args=args)
    failures = _failure_contract(args)
    resources = _public_resource_validation(args)
    performance = _performance_validation(args)
    performance_gate = bool(
        all(
            record["finite"] and record["byte_formula_matches"]
            for record in performance
        )
    )
    k8_partition = _validate_disjoint_partition(
        _variant_architecture(args.m, 8, seed=args.seed + 70_000),
        _balanced_physical_groups(args.m, args.loo_groups),
    )
    synthetic_gate = bool(
        all(record["gate"] for record in synthetic)
        and empirical["gate"]
        and transport["gate"]
        and failures["gate"]
        and resources["gate"]
        and performance_gate
        and k8_partition["gate"]
    )
    real: dict[str, Any] | None = None
    if args.real_traits:
        if not synthetic_gate:
            raise RuntimeError(
                "The protected real-trait sanity is gated on all synthetic checks."
            )
        real = _real_annotation_sanity(args)
    gates = {
        "synthetic_cases": bool(all(record["gate"] for record in synthetic)),
        "empirical_recovery_finite": bool(empirical["gate"]),
        "matched_reference_transport": bool(transport["gate"]),
        "strict_failures": bool(failures["gate"]),
        "public_resource_formula": bool(resources["gate"]),
        "performance_action_shapes": performance_gate,
        "k8_partition_stress": bool(k8_partition["gate"]),
        "real_numerical_sanity": None if real is None else bool(real["gate"]),
    }
    overall = bool(synthetic_gate and (real is None or real["gate"]))
    payload = {
        "status": "PASS" if overall else "FAIL",
        "stage": "08_disjoint_annotation_contextual_covariance",
        "scope": {
            "default": "deterministic synthetic correctness_and_scaling_gate",
            "real": (
                "not_requested"
                if real is None
                else "small_privacy_preserving_numerical_sanity_not_significance_gate"
            ),
            "overlapping_annotations": "out_of_scope_not_interpreted_as_covariances",
            "jackknife": (
                "summary_only_approximate_delete_group_contributions_with_full_joint_covariance"
            ),
        },
        "configuration": {
            "n": args.n,
            "reference_n": args.reference_n,
            "m": args.m,
            "loo_groups": args.loo_groups,
            "probe_counts": list(args.probe_counts),
            "transport_replicates": args.transport_replicates,
            "seed": args.seed,
        },
        "gates": gates,
        "synthetic_cases": synthetic,
        "empirical_recovery": empirical,
        "matched_reference_transport": transport,
        "failure_contract": failures,
        "resource_validation": resources,
        "performance": performance,
        "k8_partition_stress": k8_partition,
        "real_traits": real,
        "elapsed_seconds": float(time.perf_counter() - started),
        "privacy": {
            "individual_identifiers_written": False,
            "sample_rows_written": False,
            "variant_rows_written": False,
            "file_mode": "0600",
            "directory_mode": "0700",
        },
    }
    _plot_recovery(plot_records, args.output_dir)
    _plot_loo_contrasts(synthetic, empirical, args.output_dir)
    _plot_transport(transport, args.output_dir)
    _plot_resources(resources, performance, args.output_dir)
    if real is not None:
        _plot_real(real, args.output_dir)
    return payload


def main() -> None:
    os.umask(0o077)
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    _prepare_output_directory(args.output_dir, include_real=args.real_traits)
    payload = run_validation(args)
    output = args.output_dir / f"{OUTPUT_STEM}.json"
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "gates": payload["gates"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
