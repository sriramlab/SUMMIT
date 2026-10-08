"""Independent-bank moment research reference; separate from score inference.

E[Khat]=K does not imply E[tr(Khat²)]=tr(K²). Three independent
banks allow unbiased products through degree three. Inverting their random
normal matrix still induces bias; this module does not label that fit exact.
"""
import itertools
import numpy as np
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.spec import canonical_sha256


def independent_bank_moments(references, phenotypes):
    references = tuple(references)
    if len(references) != 3:
        raise ValueError("three independently drawn feature banks required")
    first = references[0]
    banks = []
    contracts = []
    for reference in references:
        m = reference.metadata
        if m.get("sketch_dimensions") is None:
            raise ValueError(
                "independent-bank calculation requires random feature banks"
            )
        banks.append((m["seed"], m["bank"]))
        contracts.append(
            canonical_sha256(
                {
                    k: v
                    for k, v in m.items()
                    if k
                    not in (
                        "bank",
                        "seed",
                        "feature_names",
                        "genotype_passes",
                        "compatibility_id",
                    )
                }
            )
        )
        if not np.array_equal(reference.fixed_effects, first.fixed_effects):
            raise ValueError("bank projections disagree")
    if len(set(banks)) != 3 or len(set(contracts)) != 1:
        raise ValueError(
            "banks must be independent with matching scientific definitions"
        )
    y = np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or len(y) != len(first.features) or not np.all(np.isfinite(y)):
        raise ValueError("invalid phenotype axes")
    u = thin_rank_revealing_fixed_effect_basis(first.fixed_effects)
    y = y - u @ (u.T @ y)
    names = first.metadata["component_names"]
    c = len(names)
    fs = []
    ky = []
    for ref in references:
        f = ref.features - u @ (u.T @ ref.features)
        parts = [f[:, ref.component_index == i] for i in range(c)]
        fs.append(parts)
        ky.append([a @ (a.T @ y) for a in parts])
    t = np.zeros((c, c))
    q = np.zeros((c, y.shape[1]))
    cubic = np.zeros((y.shape[1], c, c, c))
    for bank in range(3):
        for a in range(c):
            q[a] += np.sum(y * ky[bank][a], axis=0) / 3
    for left, right in itertools.permutations(range(3), 2):
        for a in range(c):
            for b in range(c):
                t[a, b] += np.sum((fs[left][a].T @ fs[right][b]) ** 2) / 6
    for left, middle, right in itertools.permutations(range(3)):
        for a in range(c):
            for d in range(c):
                l = fs[middle][d].T @ ky[left][a]
                for b in range(c):
                    r = fs[middle][d].T @ ky[right][b]
                    cubic[:, a, d, b] += np.sum(l * r, axis=0) / 6
    return dict(
        matrix=t,
        rhs=q,
        cubic=cubic,
        component_names=names,
        method="three_independent_bank_unbiased_moments_v1",
        limitation="moments unbiased over sketches; inverse normal equations and plug-in inference are not thereby unbiased or calibrated",
    )
