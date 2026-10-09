"""Two-pass, study-specific ordered cross-trait PCGC references.

Shares genotype decoding, features and global source sketches across pairs.
No block assignment enters preparation. No N by N or M by M matrices.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from summit.ldscore.generalized_gxe_pass1 import _peak_rss_bytes

from summit.context.spec import array_sha256, canonical_sha256
from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator, ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator, ProtectedTNOperator
from summit.ldscore.generalized_gxe_variant import (
    GlobalVariantProbeSpec, native_global_variant_probes, plan_generalized_gxe_variant_work,
    GeneralizedGxEPlanInputs,
)
from .cross_gxe_model import DiagonalRemoval
from .gxe import context_pairs
from .moments import annotations_array


class Products:
    def __init__(self, native, threads):
        self.left = (ProtectedNNOperator if native else NumpyNNOperator)(threads=threads)
        self.right = (ProtectedTNOperator if native else NumpyTNOperator)(threads=threads)
        self.seconds = 0.
        self.left.begin_execution()

    def nn(self, a, b):
        start = perf_counter()
        result = self.left.matmul(np.asfortranarray(a), np.asfortranarray(b))
        self.seconds += perf_counter()-start
        return result

    def tn(self, a, b):
        start = perf_counter()
        result = self.right.matmul_tn(np.asfortranarray(a), np.asfortranarray(b))
        self.seconds += perf_counter()-start
        return result

    def drain(self):
        # Consume bounded telemetry after each block; do not retain large logs.
        value = self.left.finish_execution()
        return value.get("gemm_status", {})


def project(value, u, products):
    if not u.shape[1]:
        return np.asfortranarray(value)
    return np.asfortranarray(value-products.nn(u, products.tn(u, value)))


def row_selector(indices):
    """Preserve column-major views for consecutive participant rows."""
    if not len(indices):
        return slice(0, 0)
    first = int(indices[0])
    if np.array_equal(indices, np.arange(first, first+len(indices))):
        return slice(first, first+len(indices))
    return indices


def pair_layout(qa, qb, within):
    pairs = context_pairs(qa) if within else tuple((u, v) for u in range(qa) for v in range(qb))
    return tuple(((u, v),) if not within or u == v else ((u, v), (v, u)) for u, v in pairs)


@dataclass
class CrossPairRows:
    left: int
    right: int
    layout: tuple
    overlap: int
    sample_product: int
    ldscores: np.ndarray
    rhs_rows: np.ndarray
    probe_grams: np.ndarray

    @property
    def p(self):
        return len(self.layout)


@dataclass
class CrossGxEMoments:
    annotations: np.ndarray
    traits: tuple
    pairs: tuple
    signed_scores: tuple
    identity: dict
    diagnostics: dict

    def validate(self):
        m, k = self.annotations.shape
        annotations_array(self.annotations, m)
        if len(self.traits) != len(self.signed_scores):
            raise ValueError("trait score axes disagree")
        for t, s in zip(self.traits, self.signed_scores):
            q = len(t["context_names"])
            if s.shape != (m, q) or not np.isfinite(s).all():
                raise ValueError("signed score axes disagree")
        keys = set()
        probe_counts = set()
        for pair in self.pairs:
            if not 0 <= pair.left <= pair.right < len(self.traits) or (pair.left, pair.right) in keys:
                raise ValueError("invalid or duplicated trait orientation")
            keys.add((pair.left, pair.right))
            qa, qb = [len(self.traits[i]["context_names"]) for i in (pair.left, pair.right)]
            if pair.layout != pair_layout(qa, qb, pair.left == pair.right):
                raise ValueError("ordered context axes disagree")
            p = pair.p
            if pair.ldscores.shape != (m, p, k*p) or pair.rhs_rows.shape != (m, p):
                raise ValueError("cross-trait reference and response axes disagree")
            if pair.probe_grams.ndim != 3 or pair.probe_grams.shape[1:] != (k*p, k*p):
                raise ValueError("cross-trait reference draw axes disagree")
            probe_counts.add(len(pair.probe_grams))
            if len(pair.probe_grams) < 2 or pair.overlap < 0 or pair.sample_product < 9:
                raise ValueError("invalid cross-trait sample/probe metadata")
            if not all(np.isfinite(a).all() for a in (pair.ldscores, pair.rhs_rows, pair.probe_grams)):
                raise ValueError("cross-trait moments contain nonfinite values")
        if len(probe_counts) != 1:
            raise ValueError("joint reference requires identical probe counts across pairs")
        if not all((t, t) in keys for t in range(len(self.traits))):
            raise ValueError("joint preparation requires all within-trait denominators")


def plan_cross_reference(n, m, traits, pairs, *, annotations, probes, block_size,
                         probe_tile, memory_bytes, native=True, extra_bytes=0, threads=1, genotype_format="bed"):
    """Conservative total working-set admission, including resident artifacts.

    Reuses the mature reference planner for decoder/native workspace; additional
    cross-trait arrays are budgeted explicitly. All source sketches remain
    resident; probe tiling bounds product workspace, never source scope.
    """
    if any(type(v) is not int or v < 1 for v in (n, m, annotations, block_size, probe_tile, memory_bytes)):
        raise ValueError("cross-trait planner sizes must be positive integers")
    if type(probes) is not int or probes < 2:
        raise ValueError("at least two global variant probes are required")
    k = annotations
    nq = sum(len(t.samples)*t.q for t in traits)
    ps = [len(pair_layout(traits[a].q, traits[b].q, a == b)) for a, b in pairs]
    overlap = [len(set(traits[a].samples) & set(traits[b].samples)) for a, b in pairs]
    max_rank = max((traits[a].fixed_basis.shape[1]*traits[b].fixed_basis.shape[1] for a, b in pairs), default=0)
    max_overlap_rank = max((o*traits[a].fixed_basis.shape[1]*traits[b].fixed_basis.shape[1]
                            for o, (a, b) in zip(overlap, pairs)), default=0)
    outputs = m*(k*sum(p*p for p in ps)+sum(ps)+sum(t.q for t in traits))+probes*k*k*sum(p*p for p in ps)
    sketches = (n+nq)*k*probes
    diagonals = sum(o*(k*p+1) for o, p in zip(overlap, ps))
    # Native GEMM accepts/copies F-order operands, including projection and
    # correction inputs. Include coexistence during artifact serialization.
    baseline = _peak_rss_bytes()
    fixed = baseline+8*(2*outputs+sketches+3*diagonals+4*max_overlap_rank+4*max_rank**2+
               m*k+sum(len(t.samples)*(3*t.q+3*t.fixed_basis.shape[1]+8) for t in traits))+extra_bytes
    b, tile = min(m, block_size), min(probes, probe_tile)
    while True:
        workspace = 8*(6*n*b+3*nq*b+4*nq*k*tile+
                       4*b*k*tile*sum(t.q*t.q for t in traits)+4*b*tile*k)
        remaining = memory_bytes-fixed-workspace
        if remaining > 0:
            try:
                base = plan_generalized_gxe_variant_work(GeneralizedGxEPlanInputs(
                    num_samples=n, num_variants=m, num_basis=1, num_annotations=1,
                    num_probes=probes, memory_limit_bytes=remaining,
                    preferred_variant_block_width=b, preferred_source_probe_tile_width=tile,
                    genotype_format=genotype_format, threads=threads))
                break
            except (MemoryError, ValueError):
                pass
        if b > 1:
            b = max(1, b//2)
        elif tile > 1:
            tile = max(1, tile//2)
        else:
            raise MemoryError("cross-trait resident sketches, rows and overlap workspace exceed memory budget")
    return dict(block_size=min(b, base.tiling["variant_block_width"]), probe_tile=min(tile, base.tiling["source_probe_tile_width"]),
                baseline_rss_bytes=baseline, resident_artifact_bytes=8*outputs, source_sketch_bytes=8*sketches,
                overlap_diagonal_bytes=8*diagonals, fixed_reserved_bytes=fixed,
                cross_workspace_bytes=workspace, native_decoder_plan=base.to_dict(),
                peak_planned_bytes=fixed+workspace+base.peak_resident_bytes,
                memory_budget_bytes=memory_bytes, reference_genotype_passes=2)


def prepare_cross_gxe(operator, annotations, traits, master_samples, *,
                      pairs=None, probes=256, seed=0, block_size=256, probe_tile=32,
                      memory_bytes=2**30, native=True, threads=1, extra_bytes=0):
    """Joint reference/trait preparation; selection checks precede genotype I/O."""
    start_time = perf_counter()
    traits = tuple(traits)
    master_samples = tuple(tuple(p) for p in master_samples)
    n, m = operator.num_samples, operator.num_variants
    if not traits or len({t.name for t in traits}) != len(traits):
        raise ValueError("trait names must be unique")
    if len(master_samples) != n or len(set(master_samples)) != n:
        raise ValueError("master genotype identities must be unique and match its sample axis")
    lookup = {p: i for i, p in enumerate(master_samples)}
    if set().union(*(set(t.samples) for t in traits)) != set(master_samples):
        raise ValueError("master genotype axis must be exactly the union of trait identities")
    rows = [np.array([lookup[p] for p in t.samples], dtype=np.int64) for t in traits]
    if pairs is None:
        pairs = [(a, b) for a in range(len(traits)) for b in range(a+1, len(traits))]
    pairs = tuple(sorted(set([(a, a) for a in range(len(traits))]+[tuple(p) for p in pairs])))
    if any(len(p) != 2 or not 0 <= p[0] <= p[1] < len(traits) for p in pairs):
        raise ValueError("request pairs in declared forward trait order")
    a = annotations_array(annotations, m)
    overlap_indices = []
    for left, right in pairs:
        _, ia, ib = np.intersect1d(rows[left], rows[right], return_indices=True)
        if left != right and len(ia) and any(
                traits[t].metadata["selection"] != "outcome_independent" for t in (left, right)):
            raise ValueError("ascertained cross-trait overlap requires a joint selection model; unsupported")
        if np.any(1-np.sum(traits[left].fixed_basis[ia]**2,axis=1)-
                  np.sum(traits[right].fixed_basis[ib]**2,axis=1) <= 1e-8):
            raise ValueError("overlap projection has excessive leverage; diagonal removal is unstable")
        overlap_indices.append((ia, ib))
    row_views = [row_selector(index) for index in rows]
    overlap_views = [(row_selector(ia), row_selector(ib)) for ia, ib in overlap_indices]
    plan = plan_cross_reference(n, m, traits, pairs, annotations=a.shape[1], probes=probes,
                               block_size=block_size, probe_tile=probe_tile,
                               memory_bytes=memory_bytes, native=native, extra_bytes=extra_bytes,
                               threads=threads, genotype_format=operator.genotype_format)
    block_size, tile = plan["block_size"], plan["probe_tile"]
    if hasattr(operator, "configure_block_width"):
        operator.configure_block_width(block_size)
    products = Products(native, threads)
    k = a.shape[1]
    spec = GlobalVariantProbeSpec(root_seed=seed, probe_offset=0, probe_count=probes)
    response = [project(t.response[:, None], t.fixed_basis, products)[:, 0] for t in traits]
    base_sketch = np.zeros((n, k*probes), order="F")
    sketches = None
    summaries, diagonals = [], []
    for (left, right), (ia, ib) in zip(pairs, overlap_indices):
        layout = pair_layout(traits[left].q, traits[right].q, left == right)
        p = len(layout)
        summaries.append(CrossPairRows(left, right, layout, len(ia),
            len(traits[left].samples)*len(traits[right].samples),
            np.zeros((m, p, k*p)), np.empty((m, p)), np.zeros((probes, k*p, k*p))))
        diagonals.append(np.zeros((len(ia), k*p), order="F"))
    scores = [np.empty((m, t.q)) for t in traits]
    timings = dict(pass1_sketch_seconds=0., source_transform_seconds=0., pass2_reference_seconds=0., feature_seconds=0., diagonal_seconds=0., reference_seconds=0.,
                   score_seconds=0., overlap_row_seconds=0., overlap_solve_seconds=0.,
                   genotype_read_transform_seconds=0., reference_draw_aggregation_seconds=0.)
    correction, response_correction = [], []

    def features(x):
        values = []
        for t, index in zip(traits, row_views):
            base = x[index]
            values.append([project(base*(t.sensitivity*t.contexts[:, q])[:, None],
                                   t.fixed_basis, products) for q in range(t.q)])
        return values

    for pass_number in (1, 2):
        operator.begin_pass(pass_number)
        for first in range(0, m, block_size):
            last = min(m, first+block_size)
            read_start = perf_counter()
            x = operator.read_block(first, last).values
            timings["genotype_read_transform_seconds"] += perf_counter()-read_start
            tick = perf_counter()
            f = features(x)
            timings["feature_seconds"] += perf_counter()-tick
            weights = a[first:last]
            if pass_number == 1:
                tick = perf_counter()
                for pair, (ia, ib), diagonal in zip(summaries, overlap_views, diagonals):
                    if pair.overlap:
                        for p, orientations in enumerate(pair.layout):
                            value = sum(f[pair.left][u][ia]*f[pair.right][v][ib] for u, v in orientations)
                            diagonal[:, p::pair.p] += products.nn(value, weights)
                timings["diagonal_seconds"] += perf_counter()-tick
                tick = perf_counter()
                for begin in range(0, probes, tile):
                    end = min(probes, begin+tile)
                    z = (native_global_variant_probes(np.arange(first, last), np.arange(begin, end),
                         root_seed=seed, threads=threads) if native else
                         GlobalVariantProbeSpec(seed, begin, end-begin).generate(np.arange(first, last)))
                    az = (np.sqrt(weights)[:, :, None]*z[:, None, :]).reshape(last-first, -1)
                    contribution = products.nn(x, az)
                    for annotation in range(k):
                        base_sketch[:, annotation*probes+begin:annotation*probes+end] += contribution[:, annotation*(end-begin):(annotation+1)*(end-begin)]
                    del contribution
                phase_seconds = perf_counter()-tick
                timings["reference_seconds"] += phase_seconds
                timings["pass1_sketch_seconds"] += phase_seconds
            else:
                tick = perf_counter()
                for t, trait in enumerate(traits):
                    for q in range(trait.q):
                        scores[t][first:last, q] = products.tn(f[t][q], response[t][:, None])[:, 0]
                timings["score_seconds"] += perf_counter()-tick
                tick = perf_counter()
                for index, (pair, (ia, ib)) in enumerate(zip(summaries, overlap_views)):
                    for p, orientations in enumerate(pair.layout):
                        rhs = sum(scores[pair.left][first:last, u]*scores[pair.right][first:last, v] for u, v in orientations)
                        if pair.overlap:
                            d = sum(f[pair.left][u][ia]*f[pair.right][v][ib] for u, v in orientations)
                            rhs -= products.tn(d, response_correction[index])[:, 0]
                            pair.ldscores[first:last, p] -= products.tn(d, correction[index])
                        pair.rhs_rows[first:last, p] = rhs
                timings["overlap_row_seconds"] += perf_counter()-tick
                tick = perf_counter()
                for begin in range(0, probes, tile):
                    end = min(probes, begin+tile)
                    panels = []
                    for t, trait in enumerate(traits):
                        source = np.concatenate([s[:, annotation*probes+begin:annotation*probes+end]
                                                 for s in sketches[t] for annotation in range(k)], axis=1)
                        target = np.concatenate(f[t], axis=1)
                        panels.append(products.tn(target, source).reshape(trait.q, last-first, trait.q, k, end-begin))
                    for pair in summaries:
                        left, right = panels[pair.left], panels[pair.right]
                        for p, target in enumerate(pair.layout):
                            for s, source in enumerate(pair.layout):
                                product = sum(left[u, :, c]*right[v, :, d] for u, v in target for c, d in source)
                                pair.ldscores[first:last, p, s::pair.p] += product.sum(axis=2)/probes
                                aggregation_start = perf_counter()
                                for annotation in range(k):
                                    draw = products.tn(weights, product[:, annotation])
                                    pair.probe_grams[begin:end, p::pair.p, annotation*pair.p+s] += draw.T
                                timings["reference_draw_aggregation_seconds"] += perf_counter()-aggregation_start
                phase_seconds = perf_counter()-tick
                timings["reference_seconds"] += phase_seconds
                timings["pass2_reference_seconds"] += phase_seconds
            products.drain()
        operator.finish_pass()
        if pass_number == 1:
            tick = perf_counter()
            sketches = []
            for t, index in zip(traits, row_views):
                trait_sketches = []
                for q in range(t.q):
                    transformed = np.empty((len(t.samples), k*probes), order="F")
                    for begin in range(0, probes, tile):
                        end = min(probes, begin+tile)
                        value = np.concatenate([base_sketch[index, annotation*probes+begin:annotation*probes+end]
                                                for annotation in range(k)], axis=1)
                        value = value*(t.sensitivity*t.contexts[:,q])[:,None]
                        projected = project(value, t.fixed_basis, products)
                        for annotation in range(k):
                            transformed[:, annotation*probes+begin:annotation*probes+end] = projected[:, annotation*(end-begin):(annotation+1)*(end-begin)]
                        del value, projected
                    trait_sketches.append(transformed)
                sketches.append(trait_sketches)
            del base_sketch
            phase_seconds = perf_counter()-tick
            timings["reference_seconds"] += phase_seconds
            timings["source_transform_seconds"] += phase_seconds
            tick = perf_counter()
            for pair, (ia, ib), diagonal in zip(summaries, overlap_indices, diagonals):
                remover = DiagonalRemoval(traits[pair.left].fixed_basis[ia],
                                          traits[pair.right].fixed_basis[ib], products)
                correction.append(remover.apply(diagonal))
                response_correction.append(remover.apply((response[pair.left][ia]*response[pair.right][ib])[:, None]))
            del diagonals
            timings["overlap_solve_seconds"] = perf_counter()-tick
    for pair in summaries:
        pair.ldscores /= pair.sample_product
        mass = np.repeat(a.sum(0), pair.p)
        pair.probe_grams /= np.outer(mass, mass)[None]
    products.drain()
    diagnostics = dict(plan=plan, timings={**timings, "protected_gemm_seconds": products.seconds,
        "decode_scale_seconds": operator.decode_seconds, "total_seconds": perf_counter()-start_time},
        peak_rss_bytes=_peak_rss_bytes(),
        genotype_passes=operator.observed_passes, variant_visits=operator.observed_variant_visits,
        genotype_scale_identity=operator.genotype_scale_id, probes=spec.to_metadata(),
        native=native, threads=threads, gemm_calls=products.left.calls+products.right.calls,
        repaired_columns=products.left.repaired_columns+products.right.repaired_columns)
    identity = dict(master_sample_sha256=canonical_sha256({"samples": master_samples}),
                    genotype_scale_identity=operator.genotype_scale_id,
                    annotations_sha256=array_sha256(a), reference_probes=spec.to_metadata(),
                    estimator="projected_diagonal_annihilation_pcgc_v1")
    result = CrossGxEMoments(a, tuple(t.identity() for t in traits), tuple(summaries),
                            tuple(scores), identity, diagnostics)
    result.validate()
    return result
