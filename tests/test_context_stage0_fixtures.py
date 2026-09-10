from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    build_context_reference,
    build_context_trait_summary,
    coefficients_to_omegas,
    common_scale_features,
    context_covariance_surface,
    context_kernel_actions,
    dense_genetic_kernels,
    exact_same_person_matrix,
    fit_context_model,
    hutchinson_gram,
    kernel_gram,
    kernel_rhs,
    rank_revealing_projector,
    reference_moments_after_deleting_groups,
    symmetric_rank_diagnostics,
    trait_moments_after_deleting_groups,
    transfer_reference_gram,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "context_native_stage0"
SOURCE_COMMIT = "251f197950775ca891f244dedc109476b2ad43b4"
ATOL = 1.0e-9
RTOL = 1.0e-11
ZERO_HASH = "0" * 64

# Keep numerical fixtures independent of their local manifest. Documentation
# can be edited without changing the numerical reference data.
PACKAGE_FILES = {
    "fixture_q1_k2.npz": (
        78679,
        "e146a8d9c1f63dc51a34dd409baed669217f0d38c5f271e08dc8ddb025cde96a",
    ),
    "fixture_q2_k2.npz": (
        155724,
        "735f8123458f02bc7e1e66dc51f4a2de64e346ba948e7d7e1ac899ecdca655f1",
    ),
    "fixture_q3_k2.npz": (
        446445,
        "339a8f78a341cc6124c8b457a5bd5e49fdf25dc3937c71179440648b6539d1c5",
    ),
    "fixture_q4_k3.npz": (
        923111,
        "49153e514bb00003c15a713ab35b812b10a89203588261701534a10d85ffa997",
    ),
    "fixtures_manifest.json": (
        2280,
        "d888768093eb811228a15999a203ed627a7013508b41bd4cc8d884342e1252ed",
    ),
    "micro_directional_factors.npz": (
        3272,
        "415f6170b495862ec3fc2ac07174aea334e845e476f8562d9373f1bfe2ef284c",
    ),
    "micro_overlap_grouped_algorithms.npz": (
        22168,
        "b745a3c6c3fdf3a2375f31cf7a2ea5fa0f1ca0c0d298bd71f6ef663360a6f394",
    ),
    "micro_projection_order.npz": (
        1228,
        "6e52f9c4a0c7f65835823c5a70b23eb3a9216402a60915b2fdce0b37d87b66f5",
    ),
    "micro_rank_deficient_overlap.npz": (
        1269,
        "d5400c55f5ed10f0aba94df46f86ad4102a45cfc6a2a925be0271d700b0f8daf",
    ),
    "micro_transfer_n_vs_r.npz": (
        1480,
        "4700032104b95bf5c5b9a579e93fc27adf82c1eca8aafc45cded9f103fedabc8",
    ),
}

# The diagonal-first, then lexicographic off-diagonal order is an API contract.
PAIR_DIGESTS = {
    1: "7ad52c53d2e60a48ee481335655da773de7c368f19058749ce22b428f31ee651",
    2: "ed5af052fb162b216aec1377053029d6781f1b2b8d5371ac5acff97086c7f2e1",
    3: "a9895e7060834060a736dd4e8dd7aa1abb89811baf57796ec0365254e16e8723",
    4: "6b43a6f4f98014960db86bd354288b6dbb2262eb0e5c45246e7290eeb3464c24",
}

MAIN_FIXTURES = (
    ("fixture_q1_k2.npz", 1, 2, 48, 72, 1, 2),
    ("fixture_q2_k2.npz", 2, 2, 48, 72, 3, 6),
    ("fixture_q3_k2.npz", 3, 2, 64, 96, 6, 12),
    ("fixture_q4_k3.npz", 4, 3, 64, 96, 10, 30),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(actual: object, expected: object, label: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=RTOL,
        atol=ATOL,
        err_msg=label,
    )


def _component_masses(
    annotation_masses: np.ndarray, components: ContextComponentIndex
) -> np.ndarray:
    return np.asarray(
        [
            annotation_masses[component.annotation_index]
            for component in components.entries
        ],
        dtype=np.float64,
    )


def test_imported_stage0_fixture_package_is_byte_exact() -> None:
    assert FIXTURE_ROOT.is_dir()
    for filename, (expected_size, expected_hash) in PACKAGE_FILES.items():
        path = FIXTURE_ROOT / filename
        assert path.is_file(), filename
        assert path.stat().st_size == expected_size, filename
        assert _sha256(path) == expected_hash, filename

    manifest = json.loads((FIXTURE_ROOT / "fixtures_manifest.json").read_text())
    assert manifest["generator"] == "snapshot_python_oracle"
    assert manifest["source_commit"] == SOURCE_COMMIT
    records = manifest["fixtures"] + manifest["microcases"]
    assert {record["file"] for record in records} == set(PACKAGE_FILES) - {
        "fixtures_manifest.json",
    }
    for record in records:
        assert record["sha256"] == PACKAGE_FILES[record["file"]][1]


@pytest.mark.parametrize(
    ("filename", "q_count", "k_count", "n", "m", "p_count", "c_count"),
    MAIN_FIXTURES,
)
def test_active_oracle_replays_frozen_q1_to_q4(
    filename: str,
    q_count: int,
    k_count: int,
    n: int,
    m: int,
    p_count: int,
    c_count: int,
) -> None:
    with np.load(FIXTURE_ROOT / filename, allow_pickle=False) as fixture:
        genotype = fixture["genotype"]
        basis = fixture["phi"]
        annotations = fixture["annotations"]
        group_index = fixture["group_index"]
        groups = tuple(f"g{int(index)}" for index in group_index)

        assert genotype.shape == (n, m)
        assert basis.shape == (n, q_count)
        assert annotations.shape == (m, k_count)
        pair_index = ContextPairIndex(q_count)
        components = ContextComponentIndex(
            tuple(f"a{index}" for index in range(k_count)), pair_index
        )
        assert len(pair_index) == p_count
        assert len(components) == c_count
        assert pair_index.digest == PAIR_DIGESTS[q_count]
        np.testing.assert_array_equal(
            fixture["pair_q"], [entry.q for entry in pair_index.entries]
        )
        np.testing.assert_array_equal(
            fixture["pair_r"], [entry.r for entry in pair_index.entries]
        )
        np.testing.assert_array_equal(
            fixture["pair_eta"], [entry.kernel_factor for entry in pair_index.entries]
        )
        np.testing.assert_array_equal(
            fixture["component_annotation"],
            [entry.annotation_index for entry in components.entries],
        )
        np.testing.assert_array_equal(
            fixture["component_pair"],
            [entry.pair_index for entry in components.entries],
        )

        projector = rank_revealing_projector(fixture["fixed_effects"])
        assert projector.residual_rank == int(fixture["residual_rank"])
        _assert_close(projector.projector, fixture["projector"], "projector")
        _assert_close(projector.fixed_basis, fixture["fixed_basis"], "fixed basis")

        features = common_scale_features(genotype, basis, projector.projector)
        kernels = dense_genetic_kernels(features, annotations, components)
        _assert_close(features, fixture["features"], "common-scale features")
        _assert_close(kernels, fixture["dense_kernels"], "dense kernels")
        _assert_close(kernel_gram(kernels), fixture["exact_gram"], "exact Gram")
        _assert_close(
            exact_same_person_matrix(kernels),
            fixture["exact_same_person"],
            "exact same-person matrix",
        )

        reference = build_context_reference(
            genotype=genotype,
            basis=basis,
            projector=projector,
            annotations=annotations,
            component_index=components,
            basis_hash=ZERO_HASH,
            fixed_effect_hash=ZERO_HASH,
            variant_hash=ZERO_HASH,
            loo_groups=groups,
            genotype_scaling="pre_scaled_input",
            gram_method="hutchinson",
            gram_probes=fixture["sample_probes"],
            same_person_method="ustat",
            variant_probes=fixture["variant_probes"],
            probe_tile_size=3,
            contribution_storage="loo_grouped",
        )
        _assert_close(reference.gram, fixture["hutch_gram"], "fixed-probe Gram")
        _assert_close(
            reference.same_person,
            fixture["ustat_same_person"],
            "same-person U-statistic",
        )
        _assert_close(
            reference.gram_numerator_contributions,
            fixture["group_gram_unnormalized_num"],
            "grouped Gram numerators",
        )
        _assert_close(
            reference.annotation_masses,
            fixture["annotation_masses"],
            "annotation masses",
        )
        _assert_close(
            reference.group_annotation_masses,
            fixture["group_annotation_masses"],
            "group annotation masses",
        )
        np.testing.assert_array_equal(
            reference.group_variant_counts, fixture["group_variant_counts"]
        )
        assert reference.manifest["component_index_hash"] == components.digest

        component_masses = _component_masses(reference.annotation_masses, components)
        _assert_close(
            component_masses, fixture["component_masses"], "component masses"
        )
        reconstructed_gram = np.sum(
            reference.gram_numerator_contributions, axis=0, dtype=np.float64
        ) / np.outer(component_masses, component_masses)
        _assert_close(reconstructed_gram, reference.gram, "grouped Gram reconstruction")

        summary = build_context_trait_summary(
            genotype=genotype,
            basis=basis,
            phenotype=fixture["phenotype_raw"],
            projector=projector,
            annotations=annotations,
            component_index=components,
            residual_basis=fixture["residual_basis"],
            residual_names=("residual", "context_residual"),
            basis_hash=ZERO_HASH,
            fixed_effect_hash=ZERO_HASH,
            variant_hash=ZERO_HASH,
            loo_groups=groups,
            genotype_scaling="pre_scaled_input",
            block_size=13,
            contribution_storage="loo_grouped",
        )
        trait_fields = {
            "genetic_rhs": "trait_genetic_rhs",
            "genetic_traces": "trait_genetic_traces",
            "genetic_residual": "trait_genetic_residual",
            "residual_rhs": "residual_rhs",
            "residual_traces": "residual_traces",
            "residual_gram": "residual_gram",
            "rhs_numerator_contributions": "group_rhs_unnormalized_num",
            "trace_numerator_contributions": "group_trace_unnormalized_num",
            "genetic_residual_numerator_contributions": (
                "group_genetic_residual_num"
            ),
        }
        for observed_name, fixture_name in trait_fields.items():
            _assert_close(
                getattr(summary, observed_name),
                fixture[fixture_name],
                observed_name,
            )
        _assert_close(
            summary.group_annotation_masses,
            fixture["group_annotation_masses"],
            "trait group annotation masses",
        )
        np.testing.assert_array_equal(
            summary.group_variant_counts, fixture["group_variant_counts"]
        )

        transferred = transfer_reference_gram(
            reference.gram,
            reference.same_person,
            reference_n=n,
            study_n=int(fixture["transferred_study_n"]),
        )
        _assert_close(transferred, fixture["transferred_gram"], "population transfer")
        _assert_close(
            reference.transferred_gram(int(fixture["transferred_study_n"])),
            fixture["transferred_gram"],
            "reference population transfer",
        )

        for group_position, group in enumerate(reference.loo_group_ids):
            deleted_reference = reference_moments_after_deleting_groups(
                reference, (group,)
            )
            deleted_trait = trait_moments_after_deleting_groups(summary, (group,))
            _assert_close(
                deleted_reference.gram,
                fixture["deletion_reference_grams"][group_position],
                f"deleted reference Gram {group}",
            )
            _assert_close(
                deleted_reference.same_person,
                fixture["ustat_same_person"],
                f"deleted reference same-person {group}",
            )
            _assert_close(
                deleted_trait.genetic_rhs,
                fixture["deletion_trait_rhs"][group_position],
                f"deleted trait RHS {group}",
            )

            retained_annotation_masses = (
                fixture["annotation_masses"]
                - fixture["group_annotation_masses"][group_position]
            )
            retained_component_masses = retained_annotation_masses[
                fixture["component_annotation"]
            ]
            expected_trace = (
                fixture["component_masses"] * fixture["trait_genetic_traces"]
                - fixture["group_trace_unnormalized_num"][group_position]
            ) / retained_component_masses
            expected_cross = (
                fixture["component_masses"][:, None]
                * fixture["trait_genetic_residual"]
                - fixture["group_genetic_residual_num"][group_position]
            ) / retained_component_masses[:, None]
            _assert_close(
                deleted_trait.genetic_traces,
                expected_trace,
                f"deleted trait traces {group}",
            )
            _assert_close(
                deleted_trait.genetic_residual,
                expected_cross,
                f"deleted trait genetic-residual block {group}",
            )
            _assert_close(
                deleted_trait.residual_rhs,
                fixture["residual_rhs"],
                f"deleted residual RHS {group}",
            )
            _assert_close(
                deleted_trait.residual_traces,
                fixture["residual_traces"],
                f"deleted residual traces {group}",
            )
            _assert_close(
                deleted_trait.residual_gram,
                fixture["residual_gram"],
                f"deleted residual Gram {group}",
            )

        fit = fit_context_model(
            reference,
            summary,
            project_psd=False,
            context_grid=basis,
        )
        _assert_close(fit.raw_coefficients, fixture["raw_coefficients"], "raw fit")
        _assert_close(fit.raw_omegas, fixture["raw_omegas"], "raw Omegas")
        _assert_close(
            fit.loo_coefficients,
            fixture["deletion_coefficients"],
            "all leave-group-out fits",
        )
        assert fit.solve.rank == fit.raw_coefficients.size
        assert fit.solve.diagnostics.rank == fit.raw_coefficients.size

        assert fit.context_outputs is not None
        for annotation_index, output in enumerate(
            fit.context_outputs["annotations"]
        ):
            expected_omega = fixture["raw_omegas"][annotation_index]
            expected_surface = context_covariance_surface(expected_omega, basis)
            _assert_close(output["omega"], expected_omega, "fit Omega output")
            _assert_close(
                output["covariance_surface"],
                expected_surface,
                "fit covariance surface",
            )

        for group_position, output in enumerate(fit.jackknife_context_outputs):
            expected_omegas = coefficients_to_omegas(
                fixture["deletion_coefficients"][group_position, :c_count],
                components,
            )
            for annotation_index, annotation_output in enumerate(
                output["annotations"]
            ):
                expected_surface = context_covariance_surface(
                    expected_omegas[annotation_index], basis
                )
                _assert_close(
                    annotation_output["covariance_surface"],
                    expected_surface,
                    f"leave-group-out covariance surface {group_position}",
                )


def test_projection_order_microcase_uses_p_d_g() -> None:
    with np.load(
        FIXTURE_ROOT / "micro_projection_order.npz", allow_pickle=False
    ) as fixture:
        observed = common_scale_features(
            fixture["genotype"],
            fixture["phi"][:, None],
            fixture["projector"],
        )[0]
        _assert_close(observed, fixture["correct_P_D_G"], "P D(phi) G")
        assert not np.allclose(observed, fixture["wrong_P_D_P_G"])
        assert not np.allclose(observed, fixture["wrong_D_P_G"])


def test_directional_factor_and_surface_microcase() -> None:
    with np.load(
        FIXTURE_ROOT / "micro_directional_factors.npz", allow_pickle=False
    ) as fixture:
        features = np.stack([fixture["f0"], fixture["f1"]])[:, :, None]
        components = ContextComponentIndex(("all",), ContextPairIndex(2))
        kernels = dense_genetic_kernels(features, np.ones((1, 1)), components)
        _assert_close(
            kernels,
            fixture["kernels_pair_order_00_11_01"],
            "directional-factor kernels",
        )
        _assert_close(kernel_gram(kernels), fixture["gram"], "directional Gram")
        _assert_close(
            exact_same_person_matrix(kernels),
            fixture["same_person"],
            "directional same-person matrix",
        )
        _assert_close(
            kernel_rhs(kernels, fixture["phenotype"]),
            fixture["rhs"],
            "directional RHS",
        )

        directional_01 = np.outer(fixture["f0"], fixture["f1"])
        directional_10 = np.outer(fixture["f1"], fixture["f0"])
        _assert_close(
            directional_01, fixture["directional_01"], "direction q=0,r=1"
        )
        _assert_close(
            directional_10, fixture["directional_10"], "direction q=1,r=0"
        )
        _assert_close(
            [
                np.sum(kernels[0] * directional_01),
                np.sum(kernels[0] * directional_10),
            ],
            fixture["gram_00_01_directional_terms"],
            "two directional terms",
        )
        _assert_close(
            [
                np.sum(directional_01 * directional_01),
                np.sum(directional_01 * directional_10),
                np.sum(directional_10 * directional_01),
                np.sum(directional_10 * directional_10),
            ],
            fixture["gram_01_01_four_directional_terms"],
            "four directional terms",
        )

        omega = coefficients_to_omegas(fixture["omega_packed"], components)[0]
        _assert_close(omega, fixture["omega_matrix"], "Omega packing")
        surface = context_covariance_surface(
            omega, fixture["surface_phi"][None, :]
        )
        _assert_close(surface[0, 0], fixture["surface_value"], "Omega surface")


def test_population_transfer_microcase_uses_sample_count_not_rank() -> None:
    with np.load(
        FIXTURE_ROOT / "micro_transfer_n_vs_r.npz", allow_pickle=False
    ) as fixture:
        observed = transfer_reference_gram(
            fixture["T_reference"],
            fixture["D_reference"],
            reference_n=int(fixture["N_reference"]),
            study_n=int(fixture["N_study"]),
        )
        _assert_close(observed, fixture["expected_using_N"], "N-based transfer")
        assert int(fixture["residual_rank"]) != int(fixture["N_study"])
        assert not np.allclose(observed, fixture["wrong_using_residual_rank"])


def test_rank_deficient_overlap_microcase_reports_rank_one() -> None:
    with np.load(
        FIXTURE_ROOT / "micro_rank_deficient_overlap.npz", allow_pickle=False
    ) as fixture:
        components = ContextComponentIndex(
            ("first", "second"), ContextPairIndex(1)
        )
        features = fixture["genotype"][None, :, :]
        kernels = dense_genetic_kernels(
            features, fixture["annotation_weights"], components
        )
        gram = kernel_gram(kernels)
        diagnostics = symmetric_rank_diagnostics(gram)
        _assert_close(kernels, fixture["kernels"], "overlap kernels")
        _assert_close(gram, fixture["gram"], "rank-deficient overlap Gram")
        _assert_close(
            diagnostics.eigenvalues,
            fixture["eigenvalues"],
            "rank-deficient eigenvalues",
        )
        assert diagnostics.rank == int(fixture["numerical_rank"]) == 1
        assert diagnostics.null_space.shape == (2, 1)


def test_overlap_grouped_algorithms_microcase_replays_active_oracle() -> None:
    with np.load(
        FIXTURE_ROOT / "micro_overlap_grouped_algorithms.npz", allow_pickle=False
    ) as fixture:
        q_count = fixture["basis"].shape[1]
        k_count = fixture["annotations"].shape[1]
        components = ContextComponentIndex(
            tuple(f"a{index}" for index in range(k_count)),
            ContextPairIndex(q_count),
        )
        projector = rank_revealing_projector(fixture["fixed_basis"])
        _assert_close(projector.projector, fixture["projector"], "overlap projector")

        component_masses = fixture["annotation_masses"][
            fixture["component_annotation"]
        ]
        actions = context_kernel_actions(
            fixture["genotype"],
            fixture["basis"],
            projector.projector,
            fixture["annotations"],
            components,
            fixture["sample_probes"],
        )
        _assert_close(
            actions * component_masses[:, None, None],
            fixture["raw_actions"],
            "overlap raw actions",
        )
        _assert_close(hutchinson_gram(actions), fixture["gram"], "overlap Gram")

        groups = tuple(f"g{int(index)}" for index in fixture["group_index"])
        reference = build_context_reference(
            genotype=fixture["genotype"],
            basis=fixture["basis"],
            projector=projector,
            annotations=fixture["annotations"],
            component_index=components,
            basis_hash=ZERO_HASH,
            fixed_effect_hash=ZERO_HASH,
            variant_hash=ZERO_HASH,
            loo_groups=groups,
            genotype_scaling="pre_scaled_input",
            gram_method="hutchinson",
            gram_probes=fixture["sample_probes"],
            same_person_method="exact",
            probe_tile_size=3,
            contribution_storage="loo_grouped",
        )
        _assert_close(reference.gram, fixture["gram"], "grouped overlap Gram")
        _assert_close(
            reference.gram_numerator_contributions,
            fixture["direct_group_gram_num"],
            "direct grouped TN numerators",
        )
        _assert_close(
            reference.gram_numerator_contributions,
            fixture["group_restricted_gram_num"],
            "group-restricted action numerators",
        )
        _assert_close(
            fixture["direct_group_gram_num"],
            fixture["group_restricted_gram_num"],
            "grouped attribution equivalence",
        )
        reconstructed = np.sum(
            reference.gram_numerator_contributions, axis=0, dtype=np.float64
        ) / np.outer(component_masses, component_masses)
        _assert_close(reconstructed, fixture["gram"], "overlap group reconstruction")
