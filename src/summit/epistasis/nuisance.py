"""Finite covariate-dependent main effects for supplied trans mean tests."""
import numpy as np
from summit.context.spec import array_sha256


def varying_main_effects(fixed, covariates, mains, *, covariate_names, main_names,
                         memory_bytes):
    """Add Z_j Z_k and Z_j times constituent/predictor main effects.

    This is an explicit finite mean span, not an unrestricted genotype function.
    It contains no product between the target and a trans score. No outcomes
    enter construction; coefficients are estimated jointly during inference.
    """
    c, z, x = (np.asarray(a, dtype=float) for a in (fixed, covariates, mains))
    if (any(a.ndim != 2 or not np.all(np.isfinite(a)) for a in (c, z, x))
            or len({len(a) for a in (c, z, x)}) != 1
            or z.shape[1] != len(covariate_names) or x.shape[1] != len(main_names)
            or not covariate_names or not main_names
            or len(set(covariate_names)) != len(covariate_names)
            or len(set(main_names)) != len(main_names)):
        raise ValueError("varying main effects require finite aligned, named columns")
    k, q = z.shape[1], x.shape[1]
    extra = k * (k + 1) // 2 + k * q
    if 8 * len(c) * (4 * (c.shape[1] + extra) + k + q) + 64 * 2**20 > memory_bytes:
        raise MemoryError("covariate-dependent main effects exceed the cohort memory budget")
    result = np.empty((len(c), c.shape[1] + extra))
    result[:, :c.shape[1]] = c
    column = c.shape[1]
    for i in range(k):
        for j in range(i, k):
            result[:, column] = z[:, i] * z[:, j]
            column += 1
    for i in range(k):
        for j in range(q):
            result[:, column] = z[:, i] * x[:, j]
            column += 1
    return result, dict(
        method="finite_quadratic_covariates_and_varying_constituent_mains_v1",
        covariates=list(covariate_names), mains=list(main_names),
        covariate_hash=array_sha256(z), main_hash=array_sha256(x),
        added_columns=extra,
    )


def select_structure_covariates(spec, values):
    names = spec.get("varying_effects", [])
    if (not isinstance(names, list) or len(set(names)) != len(names)
            or any(name not in spec["columns"] for name in names)):
        raise ValueError("varying_effects must select distinct supplied covariate columns")
    return names, values[:, [spec["columns"].index(name) for name in names]]
