from __future__ import annotations

import numpy as np
import pytest

from summit.context.oracle import (
    common_scale_features,
    context_kernel_actions,
    dense_genetic_kernels,
    hutchinson_gram,
    rank_revealing_projector,
    scale_aware_max_discrepancy,
)
from summit.context.spec import ContextComponentIndex, ContextPairIndex


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_operator_actions_equal_explicit_dense_actions_and_tile_exactly(q: int) -> None:
    rng = np.random.default_rng(400 + q)
    n, m, b = 20, 15, 7
    genotype = rng.normal(size=(n, m))
    phi = np.column_stack([np.ones(n), rng.normal(size=(n, max(q - 1, 0)))])
    fixed = np.column_stack([np.ones(n), phi[:, 1:], rng.normal(size=n)])
    projector = rank_revealing_projector(fixed).projector
    annotations = np.column_stack([np.linspace(0.2, 1.0, m), np.linspace(1.0, 0.2, m)])
    components = ContextComponentIndex(("a", "b"), ContextPairIndex(q))
    probes = rng.choice([-1.0, 1.0], size=(n, b))
    features = common_scale_features(genotype, phi, projector)
    kernels = dense_genetic_kernels(features, annotations, components)
    dense_actions = np.einsum("aij,jv->aiv", kernels, probes, optimize=True)
    actions = context_kernel_actions(
        genotype, phi, projector, annotations, components, probes
    )
    assert scale_aware_max_discrepancy(actions, dense_actions) < 1e-12
    tiled = np.concatenate(
        [
            context_kernel_actions(
                genotype, phi, projector, annotations, components, probes[:, start:stop]
            )
            for start, stop in ((0, 2), (2, 5), (5, 7))
        ],
        axis=2,
    )
    np.testing.assert_allclose(tiled, actions, rtol=3e-14, atol=3e-13)
    np.testing.assert_allclose(
        hutchinson_gram(actions),
        np.einsum("aiv,biv->ab", dense_actions, dense_actions) / b,
        rtol=3e-14,
        atol=3e-13,
    )


def test_shared_probe_hutchinson_mean_targets_dense_signed_gram() -> None:
    rng = np.random.default_rng(912)
    n, m, q = 20, 15, 2
    genotype = rng.normal(size=(n, m))
    phi = np.column_stack([np.ones(n), rng.normal(size=n)])
    projector = rank_revealing_projector(
        np.column_stack([np.ones(n), phi[:, 1]])
    ).projector
    annotations = np.ones((m, 1))
    components = ContextComponentIndex(("all",), ContextPairIndex(q))
    features = common_scale_features(genotype, phi, projector)
    kernels = dense_genetic_kernels(features, annotations, components)
    dense_gram = np.einsum("aij,bij->ab", kernels, kernels)
    estimates = []
    for _ in range(1200):
        probes = rng.choice([-1.0, 1.0], size=(n, 8))
        actions = np.einsum("aij,jv->aiv", kernels, probes, optimize=True)
        estimates.append(hutchinson_gram(actions))
    empirical = np.mean(estimates, axis=0)
    relative = np.abs(empirical - dense_gram) / np.maximum(1.0, np.abs(dense_gram))
    assert float(np.max(relative)) < 0.025
    assert np.min(kernels[2]) < 0.0


def test_hutchinson_requires_a_probe() -> None:
    with pytest.raises(ValueError, match="probe"):
        hutchinson_gram(np.empty((2, 3, 0)))
