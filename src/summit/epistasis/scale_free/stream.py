"""One genotype traversal for every scale-free analysis of a plan.

Each block is decoded once by the shared ``RawBlockStream`` and standardized
per individual group by the native ``StandardizedBlock``. Group scores
``S_g = X_g^T Y_g`` go through SUMMIT's protected TN product. The arithmetic
is that of one ungrouped product, and only the per-group outputs are kept.
After normalization by the group's class size, scores enter G x G kernels

    K_m[g, h] += sum_j w_j S~_j(c1, g) S~_j(c2, h)

for each statistic m = (c1, c2) and background annotation. Per jackknife block,
the between-group part sum_{g != h} K is also kept. Nothing of size M x G x q is
stored.

The kernels are many small products, for which the native protected TN is
dominated by per-call overhead. They are formed with batched BLAS on scores
augmented by a sum-over-groups row, so each product carries its own row and
column checksums (Huang-Abraham). Mismatching kernels are recomputed, or
rejected if the mismatch persists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

PAIRS = {'gy': ('gy', 'gy'), 'll': ('lo', 'lo'), 'hh': ('hi', 'hi'), 'lh': ('lo', 'hi'), 'yy': ('y', 'y')}


@dataclass(frozen=True)
class Background:
    """Background variant axis: weights per annotation and jackknife blocks.

    variants: strictly increasing source variant indices. weights: (M, A),
    column 0 is the complete background ('all'). Further columns are
    annotations with 0 <= w <= weights[:, 0]; their complements ('rest') are
    derived. blocks: nondecreasing jackknife block per variant, never spanning
    chromosomes.
    """
    variants: np.ndarray
    chromosomes: np.ndarray
    weights: np.ndarray
    names: tuple
    blocks: np.ndarray

    @property
    def n_blocks(self):
        return int(self.blocks[-1]) + 1

    @property
    def block_chromosomes(self):
        first = np.r_[0, np.flatnonzero(np.diff(self.blocks)) + 1]
        return self.chromosomes[first]


def background_axis(source, *, variants=None, annotations=None, names=(), n_blocks=200):
    """Contiguous jackknife blocks of near-equal size within each chromosome."""
    m_all = len(source.variants.ids)
    variants = np.arange(m_all, dtype=np.int64) if variants is None else np.asarray(variants, dtype=np.int64)
    if not len(variants) or np.any(np.diff(variants) <= 0) or variants[0] < 0 or variants[-1] >= m_all:
        raise ValueError('background variants must be nonempty, unique, sorted source indices')
    chromosomes = np.asarray(source.variants.chromosome, dtype=object)[variants].astype(str)
    weights = np.ones((len(variants), 1))
    if annotations is not None:
        extra = np.asarray(annotations, dtype=np.float64).reshape(len(variants), -1)
        if len(names) != extra.shape[1] or len(set(names)) != len(names) or 'all' in names or 'rest' in names:
            raise ValueError('annotation names must be unique, one per column, and not all/rest')
        if not np.all(np.isfinite(extra)) or np.any(extra < 0) or np.any(extra > 1):
            raise ValueError('annotation weights must lie in [0, 1]')
        weights = np.column_stack([weights, extra])
    if type(n_blocks) is not int or n_blocks < 2:
        raise ValueError('at least two jackknife blocks are required')
    change = np.r_[True, chromosomes[1:] != chromosomes[:-1]]
    if len(set(chromosomes[change])) != change.sum():
        raise ValueError('background variants must be grouped by chromosome')
    target = len(variants) / n_blocks
    blocks = np.empty(len(variants), dtype=np.int64)
    starts = np.r_[np.flatnonzero(change), len(variants)]
    next_id = 0
    for a, b in zip(starts[:-1], starts[1:]):
        k = max(1, int(round((b - a) / target)))
        blocks[a:b] = next_id + (np.arange(b - a) * k) // (b - a)
        next_id += k
    return Background(variants=variants, chromosomes=chromosomes, weights=weights,
                      names=('all',) + tuple(names), blocks=blocks)


@dataclass
class Accumulation:
    """Unnormalized kernels; divide by the analysis' background weight."""
    names: tuple
    kernels: dict                 # pair -> (A, P_pair, G, G)
    between: dict                 # pair -> (A, P_pair, n_blocks): sum_{g != h} per block
    block_weight: np.ndarray      # (A, n_blocks)
    block_chromosomes: np.ndarray
    keep: dict                    # pair -> (P_pair, n_blocks) background block used by the analysis
    diagnostics: dict = field(default_factory=dict)


def checked_gram(left, right, *, tolerance=1e-9):
    """Batched kernels left @ right^T verified by an augmented checksum row.

    left, right: (P, G + 1, b) whose last row is the sum over the first G rows,
    formed when the scores were written. The product then carries its own row
    and column checksums (Huang-Abraham). Rounding discrepancies are below
    about 2 (b + G) eps sqrt(b) sum_h |K_gh|. The tolerance tolerance * sqrt(b)
    is more than 1000 times that bound, and any larger discrepancy is treated
    as a faulty product. Returns ((P, G, G) kernels, recomputed slices).
    """
    full = np.matmul(left, right.transpose(0, 2, 1))
    width = left.shape[2]
    repaired = 0
    for p in np.flatnonzero(_checksum_failures(full, width, tolerance)):
        full[p] = np.dot(left[p], right[p].T)
        if _checksum_failures(full[p:p + 1], width, tolerance)[0]:
            raise RuntimeError('kernel product failed its checksum twice; stop and check the host')
        repaired += 1
    return full[:, :-1, :-1], repaired


def _checksum_failures(full, width, tolerance):
    kernel = full[:, :-1, :-1]
    size = np.abs(kernel)
    bound = tolerance * np.sqrt(width)
    rows, cols = full[:, :-1, -1], full[:, -1, :-1]
    return (np.any(np.abs(kernel.sum(axis=2) - rows) > bound * (size.sum(axis=2) + np.abs(rows)), axis=1)
            | np.any(np.abs(kernel.sum(axis=1) - cols) > bound * (size.sum(axis=1) + np.abs(cols)), axis=1))


def _segments(block_ids):
    cut = np.r_[0, np.flatnonzero(np.diff(block_ids)) + 1, len(block_ids)]
    return [(int(a), int(b), int(block_ids[a])) for a, b in zip(cut[:-1], cut[1:])]


def plan_block_size(plan, n_rows, *, memory_bytes, block_size):
    """Largest block within budget: score buffer, raw block and kernels."""
    q, g = plan.n_columns, plan.n_groups
    fixed = sum(x.nbytes for x in plan.blocks) + 8 * (g + 1) ** 2 * (q + 3 * 64) + 256 * 2**20
    per_variant = 8 * q * (g + 1) + 2 * n_rows + 8 * 64 * (g + 1)
    width = int((memory_bytes - fixed) // per_variant)
    if width < 64:
        raise MemoryError('scale-free pass exceeds the memory budget; split the analyses or raise --memory-gib')
    return min(block_size, width)


def accumulate(source, plan, scale, background, *, block_size=1024, threads=1, native=True,
               batch=64, progress=None):
    """Traverse the background once and return the kernel accumulation.

    scale: GenotypeScale over background.variants (positions match), fitted
    on the pass rows or supplied from a population reference.
    """
    from summit.ldscore.matrix_products import MatrixProducts
    from summit.prediction.genotype import RawBlockStream, StandardizedBlock, native_module

    if len(scale.mean) != len(background.variants):
        raise ValueError('genotype scale must align to the background variants')
    module = native_module() if native else None
    products = MatrixProducts(native=native, threads=threads)
    products.left.begin_execution()
    standardizer = StandardizedBlock(module, threads)
    stream = RawBlockStream(source, plan.rows, background.variants, block_size=block_size,
                            threads=threads, native=module)
    if not np.array_equal(stream.rows, plan.rows) or not np.array_equal(stream.variants, background.variants):
        raise RuntimeError('genotype stream axes differ from the plan')
    g_count, q = plan.n_groups, plan.n_columns
    a_count, nb = background.weights.shape[1], background.n_blocks
    sizes = {p: plan.kinds[PAIRS[p][0]].stop - plan.kinds[PAIRS[p][0]].start for p in PAIRS}
    pairs = [p for p in PAIRS if sizes[p]]
    kernels = {p: np.zeros((a_count, sizes[p], g_count, g_count)) for p in pairs}
    between = {p: np.zeros((a_count, sizes[p], nb)) for p in pairs}
    block_weight = np.zeros((a_count, nb))
    block_chrom = background.block_chromosomes
    keep = {}
    for p in pairs:
        excluded = [set(plan.analyses[a]['exclude_chromosomes']) for a, _ in plan.keys[PAIRS[p][0]]]
        keep[p] = np.array([[c not in ex for c in block_chrom] for ex in excluded], dtype=bool)
    inverse_counts = 1.0 / plan.counts
    group_rows = [plan.order[plan.bounds[g]:plan.bounds[g + 1]] for g in range(g_count)]
    scores = np.empty((q, g_count + 1, block_size))
    repaired, kernel_seconds, score_seconds, started = 0, 0., 0., perf_counter()
    for start, variants, raw in stream.blocks('scale_free_scores'):
        width = len(variants)
        local = np.arange(width, dtype=np.int64)
        mean = scale.mean[start:start + width]
        inverse = scale.inverse_scale[start:start + width]
        tick = perf_counter()
        for g in range(g_count):
            x = standardizer.prepare(raw, group_rows[g], local, mean, inverse)
            s = products.tn(x, plan.blocks[g])
            np.multiply(s.T, inverse_counts[g][:, None], out=scores[:, g, :width])
        np.sum(scores[:, :g_count, :width], axis=1, out=scores[:, g_count, :width])
        score_seconds += perf_counter() - tick
        tick = perf_counter()
        for j0, j1, block in _segments(background.blocks[start:start + width]):
            for a in range(a_count):
                w = background.weights[start + j0:start + j1, a]
                total = w.sum()
                block_weight[a, block] += total
                if total == 0:
                    continue
                unit = bool(np.all(w == 1))
                for p in pairs:
                    left_kind, right_kind = PAIRS[p]
                    ls, rs = plan.kinds[left_kind], plan.kinds[right_kind]
                    for b0 in range(0, sizes[p], batch):
                        b1 = min(sizes[p], b0 + batch)
                        left = scores[ls.start + b0:ls.start + b1, :, j0:j1]
                        right = scores[rs.start + b0:rs.start + b1, :, j0:j1]
                        if not unit:
                            left = left * w
                        product, fixed = checked_gram(left, right)
                        repaired += fixed
                        use = keep[p][b0:b1, block]
                        np.add(kernels[p][a, b0:b1], product, out=kernels[p][a, b0:b1], where=use[:, None, None])
                        off = product.sum(axis=(1, 2)) - np.trace(product, axis1=1, axis2=2)
                        between[p][a, b0:b1, block] += np.where(use, off, 0.)
        kernel_seconds += perf_counter() - tick
        if progress is not None:
            progress(start + width, len(background.variants), perf_counter() - started)
    execution = products.drain()
    diagnostics = dict(variants=int(len(background.variants)), rows=int(len(plan.rows)), groups=g_count,
                       columns=q, block_size=int(block_size), score_seconds=score_seconds,
                       kernel_seconds=kernel_seconds, kernel_checksum_repairs=int(repaired),
                       protected_tn_calls=execution['protected_tn_calls'],
                       protected_repaired_columns=execution['repaired_columns'],
                       native=bool(native), threads=int(threads))
    return Accumulation(names=background.names, kernels=kernels, between=between, block_weight=block_weight,
                        block_chromosomes=block_chrom, keep=keep, diagnostics=diagnostics)
