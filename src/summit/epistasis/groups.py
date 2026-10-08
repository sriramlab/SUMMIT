"""Exact group-kernel actions without materializing SNP-pair features.

This correctness implementation uses O(N |A| |B| R) work for R right-hand
sides. It is a foundation for bounded groups, not a biobank throughput claim.
The later implicit pair sketches are deliberately not substituted here.
"""
from __future__ import annotations

import numpy as np

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.oracle import validate_annotations
from summit.context.spec import owned_readonly_array


class GroupKernel:
    """Cross, overlapping-cross or within-set unordered independent pair effects.

    Cross pair weight a_i b_j+a_j b_i for i<j; self-pairs removed.
    Within weights a_i a_j; division by their mass cancels the two orientations.
    ``matmul`` and ``transpose_matmul`` can be protected SUMMIT operators.
    The common scaled genotype panel is caller-owned and is never reprojected
    before the product. Large panels should use the selected streamed path
    until an implicit group reference is separately qualified.
    """
    def __init__(self, genotype, left_weights, right_weights, *, fixed_effects,
                 within=False, matmul=None, transpose_matmul=None):
        x = np.asarray(genotype, dtype=float)
        if x.ndim != 2 or not np.all(np.isfinite(x)):
            raise ValueError("common-scaled genotypes must be a finite matrix")
        weights, masses = validate_annotations(np.column_stack([left_weights, right_weights]), x.shape[1])
        a, b = weights.T
        if within and not np.array_equal(a, b):
            raise ValueError("within-set weights must agree")
        mass = masses.prod()-a@b
        if mass <= 0:
            raise ValueError("group contains no nonself pairs")
        self.genotype = owned_readonly_array(x)
        self.a, self.b = a.copy(), b.copy()
        self.a.setflags(write=False)
        self.b.setflags(write=False)
        self.oriented_mass = float(mass)
        self.pair_mass = float(mass/2 if within else mass)
        self.u = thin_rank_revealing_fixed_effect_basis(fixed_effects)
        if len(self.u) != len(x):
            raise ValueError("group fixed effects use different samples")
        self.nn = matmul or (lambda a, b: a @ b)
        self.tn = transpose_matmul or (lambda a, b: a.T @ b)

    def _project(self, z):
        return z - self.nn(self.u, self.tn(self.u, z)) if self.u.shape[1] else z.copy()

    def apply(self, right):
        z = np.asarray(right, dtype=float)
        vector = z.ndim == 1
        if vector:
            z = z[:, None]
        if z.ndim != 2 or len(z) != len(self.genotype) or not np.all(np.isfinite(z)):
            raise ValueError("invalid group-kernel right-hand side")
        z = self._project(z)
        x, a, b = self.genotype, self.a, self.b
        # Choose the cheaper outer set without changing the symmetric model.
        if np.count_nonzero(a) > np.count_nonzero(b):
            a, b = b, a
        members = np.flatnonzero(b)
        xb = np.asfortranarray(x[:, members])
        result = np.zeros_like(z)
        for i in np.flatnonzero(a):
            modified = x[:, i, None]*z
            score = self.tn(xb, modified)
            result += a[i]*x[:, i, None]*self.nn(xb, b[members, None]*score)
        overlap = np.flatnonzero(a*b)
        if len(overlap):
            squares = np.asfortranarray(x[:, overlap]**2)
            result -= self.nn(squares, (a[overlap]*b[overlap])[:, None]*self.tn(squares, z))
        result = self._project(result/self.oriented_mass)
        return result[:, 0] if vector else result


def remaining_genome_weights(group_weights):
    """Set complement by membership, not 1 minus a possibly weighted annotation."""
    values = np.asarray(group_weights, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("group weights must be a finite nonnegative vector")
    return (values == 0).astype(float)


def exact_operator_summary(operators, phenotypes, *, fixed_effects, component_names,
                           trait_names, metadata, sample_tile=32):
    """Bounded exact group preparation, using sample identity columns in tiles.

    This explicitly named correctness method is neither the generalized
    variant-probe reference nor its deletion scheme. It does not form dense
    kernels, but its work scales at least quadratically in N; N is capped at
    2048. The last operator must apply the declared residual projector.
    """
    from .summary import EpistasisSummary
    if type(sample_tile) is not int or sample_tile < 1:
        raise ValueError("sample_tile must be a positive integer")
    y = np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or len(y) > 2048 or not np.all(np.isfinite(y)):
        raise ValueError("exact group summary requires at most 2048 finite sample rows")
    u = thin_rank_revealing_fixed_effect_basis(fixed_effects)
    if len(u) != len(y):
        raise ValueError("fixed effects and phenotypes use different samples")
    project = lambda z: z-u@(u.T@z)
    y = project(y)
    n, t = y.shape
    k = len(operators)
    if k != len(component_names) or not k:
        raise ValueError("operator names do not match")
    matrix, traces = np.zeros((k, k)), np.zeros(k)
    for start in range(0, n, sample_tile):
        stop = min(n, start+sample_tile)
        identities = np.zeros((n, stop-start))
        identities[np.arange(start, stop), np.arange(stop-start)] = 1
        block = np.stack([op(identities) for op in operators])
        if block.shape != (k, n, stop-start) or not np.all(np.isfinite(block)):
            raise ValueError("invalid kernel operator output")
        if not np.allclose(block[-1], project(identities), atol=1e-10):
            raise ValueError("last operator must equal the residual projector")
        matrix += np.einsum("aij,bij->ab", block, block)
        traces += block[:, np.arange(start, stop), np.arange(stop-start)].sum(axis=1)
    actions = np.stack([op(y) for op in operators])
    q = np.einsum("nt,ant->at", y, actions)
    cubic = np.empty((t, k, k, k))
    for c, op in enumerate(operators):
        for b in range(k):
            cubic[:, :, c, b] = np.einsum("ant,nt->ta", actions, op(actions[b]))
    return EpistasisSummary(matrix, q, traces, cubic, tuple(component_names), tuple(trait_names),
                           n, n-u.shape[1], dict(metadata, method="exact_tiled_operator_oracle_v1",
                           reference_probe_axis="deterministic_sample_identity", sample_tile=sample_tile))


def prepare_group_summary(study, groups, annotations, phenotypes, *, trait_names,
                          phenotype_scale="raw"):
    """Bounded exact preparation using a shared additive-only SelectedStudy.

    Genotypes are decoded once on the common scale. Group jobs require explicit
    additive nuisances for both sets (unless a set is the genome). No automatic
    local exclusion or projection is inferred from a gene/region label.
    This method has a deterministic work cap and is not a randomized reference.
    """
    from summit.prediction._validation import closed
    if max(study.n, study.m) > 2048 or not groups or study.h != 1:
        raise ValueError("exact group preparation requires nonempty groups, iid residual and N,M <= 2048")
    if not np.array_equal(study.modifiers, np.ones_like(study.modifiers)):
        raise ValueError("group preparation requires an additive-only study")
    y = np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or len(y) != study.n or not np.all(np.isfinite(y)):
        raise ValueError("phenotypes must be finite and sample aligned")
    additive = study.metadata["definitions"]["additive_annotations"]
    definitions, weights = [], []
    for group in groups:
        closed(group, ("name", "left", "mode"), ("right",), name="group component")
        mode = group["mode"]
        if mode not in ("cross", "within", "remainder"):
            raise ValueError("group mode must be cross, within or remainder")
        if ("right" in group) != (mode == "cross"):
            raise ValueError("only cross groups require a right annotation")
        names = [group["left"]] + ([group["right"]] if mode == "cross" else [])
        if any(name not in annotations for name in names):
            raise ValueError("unknown group annotation")
        if any(name not in additive for name in names):
            raise ValueError("declare each group annotation as an additive nuisance")
        a = annotations[group["left"]]
        b = (annotations[group["right"]] if mode == "cross" else
             a if mode == "within" else remaining_genome_weights(a))
        weights.append((a, b))
        definitions.append(dict(group, self_pairs=False, unordered=True))
    names = (*study.names, *(g["name"] for g in groups), "residual")
    if len(set(names)) != len(names):
        raise ValueError("duplicate group/component names")
    # Exact identity actions dominate: guard an accidental large Cartesian job.
    work = study.n**2 * sum(np.count_nonzero(a)*np.count_nonzero(b) for a, b in weights)
    if work > 3_000_000_000:
        raise ValueError("exact group work exceeds the bounded correctness limit")
    k, t = len(names), y.shape[1]
    planned = 8*(study.n*study.m*(len(groups)+4) + 4*study.n*k*t
                 + 2*t*k**3 + 4*study.n*k*32) + 64*2**20
    if planned > study.memory_bytes:
        raise MemoryError("exact group workspace exceeds the declared memory budget")
    study.nn.begin_execution()
    study.tn.begin_execution()
    x = np.empty((study.n, study.m), order="F")
    for start, stop, block in study._blocks("exact_group_genotype"):
        x[:, start:stop] = block
    projected = study.project(x)
    operators = []
    for a in range(study.c):
        def additive_op(z, a=a):
            result = study._nn(projected, study.weights[:, a, None]*study._tn(projected, z))/study.masses[a]
            study.nn.finish_execution()
            return result
        operators.append(additive_op)
    masses = []
    for group, (a, b) in zip(groups, weights):
        kernel = GroupKernel(x, a, b, fixed_effects=study.fixed,
            within=group["mode"] == "within", matmul=study._nn, transpose_matmul=study._tn)
        def group_op(z, kernel=kernel):
            result = kernel.apply(z)
            study.nn.finish_execution()
            return result
        operators.append(group_op)
        masses.append(kernel.pair_mass)
    operators.append(study.project)
    multiplier = np.ones(y.shape[1])
    if phenotype_scale == "residual_variance":
        residual = study.project(y)
        ss = np.einsum("nt,nt->t", residual, residual)
        if np.any(ss <= np.finfo(float).eps*np.einsum("nt,nt->t", y, y)):
            raise ValueError("zero residual phenotype variance")
        multiplier = np.sqrt(study.rank/ss)
        y = y*multiplier
    elif phenotype_scale != "raw":
        raise ValueError("phenotype_scale must be raw or residual_variance")
    summary = exact_operator_summary(operators, y, fixed_effects=study.fixed,
        component_names=names, trait_names=trait_names, metadata=dict(study.metadata,
        genetic_count=len(names)-1, groups=definitions, unordered_pair_masses=masses,
        component_names=names[:-1], masses=[*study.masses, *masses],
        group_workspace_plan_bytes=planned, group_identity_work_product=int(work),
        trait=dict(phenotype_scale=phenotype_scale, multiplier=multiplier.tolist())))
    study.nn.finish_execution()
    return summary
