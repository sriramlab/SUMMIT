from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_microcases(out: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    # Projection-order counterexample.
    p = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float64)
    g = np.array([[1.0], [0.0]], dtype=np.float64)
    phi = np.array([1.0, 2.0], dtype=np.float64)
    correct = p @ (phi[:, None] * g)
    wrong_project_first = p @ (phi[:, None] * (p @ g))
    wrong_no_outer_projection = phi[:, None] * (p @ g)
    path = out / "micro_projection_order.npz"
    np.savez_compressed(
        path,
        projector=p,
        genotype=g,
        phi=phi,
        correct_P_D_G=correct,
        wrong_P_D_P_G=wrong_project_first,
        wrong_D_P_G=wrong_no_outer_projection,
    )
    records.append({"file": path.name, "sha256": sha256(path), "kind": "projection_order"})

    # Directional factors and Omega packing.
    f0 = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    f1 = np.array([2.0, 1.0, -1.0], dtype=np.float64)
    y = np.array([1.0, 0.5, -1.0], dtype=np.float64)
    d01 = np.outer(f0, f1)
    d10 = np.outer(f1, f0)
    kernels = np.stack([np.outer(f0, f0), np.outer(f1, f1), d01 + d10])
    gram = np.einsum("cij,dij->cd", kernels, kernels)
    diag = np.diagonal(kernels, axis1=1, axis2=2)
    same_person = diag @ diag.T
    s0, s1 = float(f0 @ y), float(f1 @ y)
    rhs = np.array([s0 * s0, s1 * s1, 2.0 * s0 * s1], dtype=np.float64)
    omega_packed = np.array([0.7, 1.2, -0.4], dtype=np.float64)
    omega_matrix = np.array([[0.7, -0.4], [-0.4, 1.2]], dtype=np.float64)
    surface_phi = np.array([1.0, 2.0], dtype=np.float64)
    surface = float(surface_phi @ omega_matrix @ surface_phi)
    gram_00_01_terms = np.array([
        np.sum(kernels[0] * d01), np.sum(kernels[0] * d10)
    ])
    gram_01_01_terms = np.array([
        np.sum(d01 * d01), np.sum(d01 * d10),
        np.sum(d10 * d01), np.sum(d10 * d10),
    ])
    path = out / "micro_directional_factors.npz"
    np.savez_compressed(
        path,
        f0=f0,
        f1=f1,
        phenotype=y,
        directional_01=d01,
        directional_10=d10,
        kernels_pair_order_00_11_01=kernels,
        gram=gram,
        same_person=same_person,
        rhs=rhs,
        gram_00_01_directional_terms=gram_00_01_terms,
        gram_01_01_four_directional_terms=gram_01_01_terms,
        omega_packed=omega_packed,
        omega_matrix=omega_matrix,
        surface_phi=surface_phi,
        surface_value=np.asarray(surface),
    )
    records.append({"file": path.name, "sha256": sha256(path), "kind": "directional_factors"})

    # Sample-count versus residual-rank transfer.
    t_ref = np.array([[10.0, 2.0], [2.0, 5.0]], dtype=np.float64)
    d_ref = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    n_ref, n_study, residual_rank = 5, 7, 3
    expected = (n_study / n_ref) * d_ref + (
        n_study * (n_study - 1) / (n_ref * (n_ref - 1))
    ) * (t_ref - d_ref)
    wrong = (residual_rank / n_ref) * d_ref + (
        residual_rank * (residual_rank - 1) / (n_ref * (n_ref - 1))
    ) * (t_ref - d_ref)
    path = out / "micro_transfer_n_vs_r.npz"
    np.savez_compressed(
        path,
        T_reference=t_ref,
        D_reference=d_ref,
        N_reference=np.asarray(n_ref),
        N_study=np.asarray(n_study),
        residual_rank=np.asarray(residual_rank),
        expected_using_N=expected,
        wrong_using_residual_rank=wrong,
    )
    records.append({"file": path.name, "sha256": sha256(path), "kind": "population_transfer"})

    # Identical overlapping annotations: raw system is rank deficient.
    g2 = np.array([[1.0, 0.5, -1.0], [0.0, -1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float64)
    weights = np.ones((3, 2), dtype=np.float64)
    base_kernel = g2 @ g2.T / 3.0
    overlap_kernels = np.stack([base_kernel, base_kernel])
    overlap_gram = np.einsum("cij,dij->cd", overlap_kernels, overlap_kernels)
    eig = np.linalg.eigvalsh(overlap_gram)
    path = out / "micro_rank_deficient_overlap.npz"
    np.savez_compressed(
        path,
        genotype=g2,
        annotation_weights=weights,
        kernels=overlap_kernels,
        gram=overlap_gram,
        eigenvalues=eig,
        numerical_rank=np.asarray(np.linalg.matrix_rank(overlap_gram)),
    )
    records.append({"file": path.name, "sha256": sha256(path), "kind": "rank_deficient_overlap"})

    # Nontrivial overlapping annotations: direct grouped TN and group-restricted
    # actions must produce the same fixed-probe grouped numerators.
    rng = np.random.default_rng(20260820)
    n, m, q_count, k_count, j_count, b_count = 9, 11, 3, 3, 3, 5
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    scale = genotype.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    genotype /= scale
    env = rng.normal(size=n)
    basis = np.column_stack([np.ones(n), env, env * env - np.mean(env * env)])
    fixed = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    fixed_basis, _ = np.linalg.qr(fixed, mode="reduced")
    projector = np.eye(n) - fixed_basis @ fixed_basis.T
    annotations = rng.uniform(0.15, 1.0, size=(m, k_count))
    annotations[::2, 2] = 0.0
    annotations[1::3, 0] = 0.0
    groups = np.arange(m, dtype=np.int64) % j_count
    probes = (2 * rng.integers(0, 2, size=(n, b_count)) - 1).astype(np.float64)
    pairs = [(q, q) for q in range(q_count)] + [
        (q, r) for q in range(q_count) for r in range(q + 1, q_count)
    ]
    component_annotation = np.repeat(np.arange(k_count, dtype=np.int64), len(pairs))
    component_pair = np.tile(np.arange(len(pairs), dtype=np.int64), k_count)
    masses = annotations.sum(axis=0)
    projected_probes = projector @ probes
    source = np.stack([
        genotype.T @ (basis[:, q, None] * projected_probes)
        for q in range(q_count)
    ])
    targets = np.empty((k_count, q_count, n, b_count), dtype=np.float64)
    for k in range(k_count):
        for q in range(q_count):
            targets[k, q] = genotype @ (annotations[:, k, None] * source[q])
    c_count = len(component_annotation)
    raw_actions = np.empty((c_count, n, b_count), dtype=np.float64)
    for c, (k, pair_index) in enumerate(zip(component_annotation, component_pair)):
        q, r = pairs[int(pair_index)]
        raw = basis[:, q, None] * targets[k, r]
        if q != r:
            raw = raw + basis[:, r, None] * targets[k, q]
        raw_actions[c] = projector @ raw
    normalized_actions = raw_actions / masses[component_annotation, None, None]
    gram = np.einsum("cnb,dnb->cd", normalized_actions, normalized_actions) / b_count

    direct = np.zeros((j_count, c_count, c_count), dtype=np.float64)
    contractions = np.empty((q_count, c_count, m, b_count), dtype=np.float64)
    for q in range(q_count):
        for d in range(c_count):
            contractions[q, d] = genotype.T @ (basis[:, q, None] * raw_actions[d])
    for g in range(j_count):
        mask = groups == g
        directional = np.empty((c_count, c_count), dtype=np.float64)
        for c, (k, pair_index) in enumerate(zip(component_annotation, component_pair)):
            q, r = pairs[int(pair_index)]
            term = source[r] * contractions[q]
            if q != r:
                term = term + source[q] * contractions[r]
            directional[c] = np.sum(annotations[mask, k, None] * term[:, mask, :], axis=(1, 2))
        direct[g] = (directional + directional.T) / (2.0 * b_count)

    restricted = np.zeros_like(direct)
    for g in range(j_count):
        mask = groups == g
        group_targets = np.zeros_like(targets)
        for k in range(k_count):
            for q in range(q_count):
                group_targets[k, q] = genotype[:, mask] @ (
                    annotations[mask, k, None] * source[q, mask]
                )
        group_actions = np.empty_like(raw_actions)
        for c, (k, pair_index) in enumerate(zip(component_annotation, component_pair)):
            q, r = pairs[int(pair_index)]
            raw = basis[:, q, None] * group_targets[k, r]
            if q != r:
                raw = raw + basis[:, r, None] * group_targets[k, q]
            group_actions[c] = raw
        cross = np.einsum("cnb,dnb->cd", group_actions, raw_actions) / b_count
        restricted[g] = (cross + cross.T) / 2.0

    path = out / "micro_overlap_grouped_algorithms.npz"
    np.savez_compressed(
        path,
        genotype=genotype,
        basis=basis,
        fixed_basis=fixed_basis,
        projector=projector,
        annotations=annotations,
        group_index=groups,
        sample_probes=probes,
        pair_q=np.asarray([q for q, _ in pairs], dtype=np.int64),
        pair_r=np.asarray([r for _, r in pairs], dtype=np.int64),
        component_annotation=component_annotation,
        component_pair=component_pair,
        annotation_masses=masses,
        raw_actions=raw_actions,
        gram=gram,
        direct_group_gram_num=direct,
        group_restricted_gram_num=restricted,
    )
    records.append({
        "file": path.name,
        "sha256": sha256(path),
        "kind": "overlap_grouped_algorithms",
    })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    sys.path.insert(0, args.source_root)

    from summit.context.spec import ContextPairIndex, ContextComponentIndex
    from summit.context.oracle import (
        common_scale_features,
        dense_genetic_kernels,
        exact_same_person_matrix,
        kernel_gram,
        rank_revealing_projector,
        transfer_reference_gram,
    )
    from summit.context.reference import (
        build_context_reference,
        reference_moments_after_deleting_groups,
    )
    from summit.context.summary import (
        build_context_trait_summary,
        trait_moments_after_deleting_groups,
    )
    from summit.context.fit import fit_context_model

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("fixture_q*_k*.npz"):
        old.unlink()
    for old in out.glob("micro_*.npz"):
        old.unlink()

    zero_hash = "0" * 64
    manifest: dict[str, object] = {
        "generator": "snapshot_python_oracle",
        "source_commit": "251f197950775ca891f244dedc109476b2ad43b4",
        "fixtures": [],
        "microcases": [],
    }

    for q_count in range(1, 5):
        rng = np.random.default_rng(1000 + q_count)
        n = 64 if q_count >= 3 else 48
        m = 96 if q_count >= 3 else 72
        allele_frequency = rng.uniform(0.1, 0.5, size=m)
        raw = rng.binomial(2, allele_frequency, size=(n, m)).astype(np.float64)
        means = raw.mean(axis=0)
        scales = raw.std(axis=0, ddof=0)
        scales[scales < 1e-12] = 1.0
        genotype = (raw - means) / scales

        environment = rng.normal(size=n)
        columns = [np.ones(n)]
        if q_count >= 2:
            columns.append(environment)
        if q_count >= 3:
            columns.append(environment * environment - np.mean(environment * environment))
        if q_count >= 4:
            columns.append(np.sin(environment) + 0.2 * environment)
        phi = np.column_stack(columns)

        fixed = np.column_stack([np.ones(n), rng.normal(size=n)])
        projector = rank_revealing_projector(fixed)
        k_count = 3 if q_count == 4 else 2
        annotations = np.zeros((m, k_count), dtype=np.float64)
        groups = np.asarray([f"g{j % 4}" for j in range(m)])
        group_index = np.asarray([j % 4 for j in range(m)], dtype=np.int64)
        for j in range(m):
            annotations[j, (j // 4) % k_count] = 1.0

        pair_index = ContextPairIndex(q_count)
        components = ContextComponentIndex(tuple(f"a{k}" for k in range(k_count)), pair_index)
        sample_probes = (rng.integers(0, 2, size=(n, 7)) * 2 - 1).astype(np.float64)
        variant_probes = (rng.integers(0, 2, size=(m, 7)) * 2 - 1).astype(np.float64)
        phenotype_raw = rng.normal(size=n)
        residual_basis = np.column_stack([np.ones(n), environment])

        features = common_scale_features(genotype, phi, projector.projector)
        kernels = dense_genetic_kernels(features, annotations, components)
        exact_gram = kernel_gram(kernels)
        exact_d = exact_same_person_matrix(kernels)

        reference = build_context_reference(
            genotype=genotype,
            basis=phi,
            projector=projector,
            annotations=annotations,
            component_index=components,
            basis_hash=zero_hash,
            fixed_effect_hash=zero_hash,
            variant_hash=zero_hash,
            loo_groups=groups,
            genotype_scaling="pre_scaled_input",
            gram_method="hutchinson",
            gram_probes=sample_probes,
            same_person_method="ustat",
            variant_probes=variant_probes,
            probe_tile_size=3,
            contribution_storage="loo_grouped",
        )
        summary = build_context_trait_summary(
            genotype=genotype,
            basis=phi,
            phenotype=phenotype_raw,
            projector=projector,
            annotations=annotations,
            component_index=components,
            residual_basis=residual_basis,
            residual_names=("residual", "context_residual"),
            basis_hash=zero_hash,
            fixed_effect_hash=zero_hash,
            variant_hash=zero_hash,
            loo_groups=groups,
            genotype_scaling="pre_scaled_input",
            block_size=13,
            contribution_storage="loo_grouped",
        )
        fit = fit_context_model(reference, summary, project_psd=False)

        component_masses = np.asarray([
            reference.annotation_masses[c.annotation_index] for c in components.entries
        ])
        transferred_n = n + 7
        transferred = transfer_reference_gram(
            reference.gram,
            reference.same_person,
            reference_n=n,
            study_n=transferred_n,
        )

        deletion_ref = []
        deletion_trait_rhs = []
        for group in reference.loo_group_ids:
            deletion_ref.append(reference_moments_after_deleting_groups(reference, [group]).gram)
            deletion_trait_rhs.append(trait_moments_after_deleting_groups(summary, [group]).genetic_rhs)
        deletion_coefficients = np.asarray(fit.loo_coefficients)

        filename = f"fixture_q{q_count}_k{k_count}.npz"
        path = out / filename
        np.savez_compressed(
            path,
            genotype=genotype,
            genotype_mean=means,
            genotype_scale=scales,
            phi=phi,
            fixed_effects=fixed,
            fixed_basis=projector.fixed_basis,
            projector=projector.projector,
            residual_rank=np.asarray(projector.residual_rank, dtype=np.int64),
            annotations=annotations,
            group_index=group_index,
            sample_probes=sample_probes,
            variant_probes=variant_probes,
            phenotype_raw=phenotype_raw,
            residual_basis=residual_basis,
            pair_q=np.asarray([x.q for x in pair_index.entries], dtype=np.int64),
            pair_r=np.asarray([x.r for x in pair_index.entries], dtype=np.int64),
            pair_eta=np.asarray([x.kernel_factor for x in pair_index.entries], dtype=np.int64),
            component_annotation=np.asarray([x.annotation_index for x in components.entries], dtype=np.int64),
            component_pair=np.asarray([x.pair_index for x in components.entries], dtype=np.int64),
            features=features,
            dense_kernels=kernels,
            exact_gram=exact_gram,
            exact_same_person=exact_d,
            hutch_gram=reference.gram,
            ustat_same_person=reference.same_person,
            group_gram_unnormalized_num=reference.gram_numerator_contributions,
            annotation_masses=reference.annotation_masses,
            group_annotation_masses=reference.group_annotation_masses,
            group_variant_counts=reference.group_variant_counts,
            trait_genetic_rhs=summary.genetic_rhs,
            trait_genetic_traces=summary.genetic_traces,
            trait_genetic_residual=summary.genetic_residual,
            residual_rhs=summary.residual_rhs,
            residual_traces=summary.residual_traces,
            residual_gram=summary.residual_gram,
            group_rhs_unnormalized_num=summary.rhs_numerator_contributions,
            group_trace_unnormalized_num=summary.trace_numerator_contributions,
            group_genetic_residual_num=summary.genetic_residual_numerator_contributions,
            transferred_study_n=np.asarray(transferred_n, dtype=np.int64),
            transferred_gram=transferred,
            raw_coefficients=fit.raw_coefficients,
            raw_omegas=fit.raw_omegas,
            deletion_reference_grams=np.asarray(deletion_ref),
            deletion_trait_rhs=np.asarray(deletion_trait_rhs),
            deletion_coefficients=deletion_coefficients,
            component_masses=component_masses,
        )
        manifest["fixtures"].append({
            "file": filename,
            "sha256": sha256(path),
            "N": n,
            "M": m,
            "Q": q_count,
            "K": k_count,
            "P": len(pair_index),
            "C": len(components),
            "J": 4,
            "B_T": 7,
            "B_D": 7,
            "raw_offdiag_min": None if q_count == 1 else float(np.min([
                fit.raw_omegas[k, a, b]
                for k in range(k_count)
                for a in range(q_count)
                for b in range(a + 1, q_count)
            ])),
        })

    manifest["microcases"] = build_microcases(out)
    manifest_path = out / "fixtures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
