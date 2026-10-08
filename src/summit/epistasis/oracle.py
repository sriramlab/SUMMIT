"""Independent explicit-feature/dense references; bounded numerical work only."""
from __future__ import annotations

import numpy as np

from summit.context.oracle import rank_revealing_projector, validate_annotations
from .summary import EpistasisSummary


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite matrix")
    if value.shape[0] > 2048:
        raise ValueError("dense oracle is limited to 2048 samples")
    return value


def selected_kernels(genotype, modifiers, weights, fixed_effects):
    """One independent-effect kernel per modifier/weight column, plus P."""
    x = _matrix(genotype, "genotype")
    e = _matrix(modifiers, "modifiers")
    w, masses = validate_annotations(weights, x.shape[1], e.shape[1])
    p = rank_revealing_projector(fixed_effects)
    if e.shape[0] != len(x) or len(p.projector) != len(x):
        raise ValueError("sample axes differ")
    kernels = []
    for a in range(e.shape[1]):
        features = p.projector @ (x * e[:, a, None])
        kernels.append((features * w[:, a]) @ features.T / masses[a])
    return np.stack([*kernels, p.projector]), p


def explicit_pair_features(genotype, left_weights, right_weights, *, within=False):
    """Unique unordered pairs; overlapping cross sets add the two edge weights.

    Cross weight for i<j is a_i b_j + a_j b_i. Within weight is a_i a_j.
    The latter differs by a constant factor from cross(a,a), canceled by mass.
    No self-pair is included. Columns include sqrt(weight/mass).
    """
    x = _matrix(genotype, "genotype")
    w, _ = validate_annotations(np.column_stack([left_weights, right_weights]), x.shape[1])
    a, b = w.T
    if within and not np.array_equal(a, b):
        raise ValueError("within-set weights must agree")
    pairs, weights = [], []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            weight = a[i] * b[j] + (0 if within else a[j] * b[i])
            if weight > 0:
                pairs.append((i, j))
                weights.append(weight)
    if not pairs:
        raise ValueError("group contains no nonself pairs")
    if len(pairs) * len(x) > 8_000_000:
        raise ValueError("explicit pair oracle capacity exceeded")
    mass = float(sum(weights))
    features = np.column_stack([x[:, i] * x[:, j] for i, j in pairs])
    return features * np.sqrt(np.asarray(weights) / mass), tuple(pairs), mass


def group_kernel(genotype, left_weights, right_weights, fixed_effects, *, within=False):
    """Hadamard identity before projection, including overlap/self correction."""
    x = _matrix(genotype, "genotype")
    w, masses = validate_annotations(np.column_stack([left_weights, right_weights]), x.shape[1])
    a, b = w.T
    if within and not np.array_equal(a, b):
        raise ValueError("within-set weights must agree")
    mass = masses.prod() - a @ b
    if mass <= 0:
        raise ValueError("group contains no nonself pairs")
    numerator = ((x * a) @ x.T) * ((x * b) @ x.T)
    squares = x * x
    numerator -= (squares * (a * b)) @ squares.T
    p = rank_revealing_projector(fixed_effects).projector
    # For within sets, numerator and mass both count each pair twice.
    return p @ (numerator / mass) @ p


def dense_summary(kernels, phenotypes, *, component_names, trait_names,
                  residual_rank, metadata):
    kernels = np.asarray(kernels, dtype=float)
    y = np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if kernels.ndim != 3 or kernels.shape[1:] != (len(y), len(y)) or len(y) > 2048:
        raise ValueError("incompatible or oversized dense kernels")
    if not np.all(np.isfinite(kernels)) or not np.all(np.isfinite(y)):
        raise ValueError("nonfinite dense inputs")
    if not np.allclose(kernels, kernels.transpose(0, 2, 1), rtol=1e-10, atol=1e-10):
        raise ValueError("kernels must be symmetric")
    p = kernels[-1]
    if not np.allclose(p @ p, p, atol=1e-10):
        raise ValueError("last kernel must be a projector")
    y = p @ y
    actions = kernels @ y
    q = np.einsum("nt,ant->at", y, actions)
    t = np.einsum("aij,bji->ab", kernels, kernels)
    cubic = np.empty((y.shape[1], len(kernels), len(kernels), len(kernels)))
    for h in range(y.shape[1]):
        for a in range(len(kernels)):
            for c in range(len(kernels)):
                for b in range(len(kernels)):
                    cubic[h, a, c, b] = actions[a, :, h] @ kernels[c] @ actions[b, :, h]
    return EpistasisSummary(t, q, np.trace(kernels, axis1=1, axis2=2), cubic,
                           tuple(component_names), tuple(trait_names), len(y), residual_rank, metadata)


def gaussian_quadratic_covariance(kernels, covariance):
    """Exact conditional Gaussian Cov(q_a,q_b), for an externally given V."""
    kernels = np.asarray(kernels)
    vk = np.asarray(covariance) @ kernels
    return 2 * np.einsum("aij,bji->ab", vk, vk)
