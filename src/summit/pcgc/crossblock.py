"""Person-level covariance between SNP-block PCGC estimating equations.

The SNP-block jackknife treats target SNP blocks as independent. They are
not: every block's estimating equation is a sum over the same participants.
A participant's threshold noise (whose variance depends on the genetic value),
genetic value, context and risk weights enter every block, so block equations
covary through people. The jackknife misses that covariance.

This module records, in one genotype traversal, each participant's Hajek row
for each SNP block. At fit time the between-block part of their covariance
is added to the SNP-block jackknife covariance. The within-block part is
already in the jackknife and is not added again. See
reports/pcgc_development/pcgc_gxe_binary_intervals.md for the derivation.
"""
from dataclasses import dataclass
import numpy as np

from summit.sumstats.binary import finite_array, readonly

METHOD = "snp_block_jackknife_plus_person_crossblock_v1"
PERSON_ROW_MODEL = "separable_context_genotype_kernel_products_v1"


def _orientations(q):
    from .gxe import context_pairs
    return [((u, v),) if u == v else ((u, v), (v, u)) for u, v in context_pairs(q)]


def context_pair_sums(features):
    """S[i,p,r] = sum over l != i of h_p(i,l) h_r(i,l), with h_p the context kernel.

    h_p(i,l) = psi_iu psi_lv (+ psi_iv psi_lu when u != v) for pair p = (u, v).
    """
    psi = finite_array("context features", features, 2)
    n, q = psi.shape
    orient = _orientations(q)
    total = psi.T @ psi
    out = np.zeros((n, len(orient), len(orient)))
    for p, left in enumerate(orient):
        for r, right in enumerate(orient[p:], start=p):
            value = np.zeros(n)
            for s, t in left:
                for s2, t2 in right:
                    value += psi[:, s]*psi[:, s2]*(total[t, t2]-psi[:, t]*psi[:, t2])
            out[:, p, r] = out[:, r, p] = value
    return out


def _center(values, cases):
    """Subtract case and control means (fixed case-control counts)."""
    values = np.asarray(values)
    flat = values.reshape(len(values), -1)
    count = int(cases.sum())
    if min(count, len(cases)-count) < 4:
        raise ValueError("person cross-block covariance needs at least four cases and four controls")
    case_sum = cases.astype(float) @ flat
    means = np.stack(((flat.sum(axis=0)-case_sum)/(len(cases)-count), case_sum/count))
    return (flat-means[cases.astype(np.int64)]).reshape(values.shape)


def contiguous_blocks(num_variants, count):
    """The equal-count contiguous partition used by binary --njack."""
    from types import SimpleNamespace
    from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
    if type(count) is not int or not 2 <= count <= num_variants:
        raise ValueError("person cross-block covariance needs 2 <= blocks <= SNPs")
    view = SimpleNamespace(nsnps=num_variants)
    return np.asarray(JackknifeDesign.from_trace_view(view, JackknifeSpec.parse(count)).unit_id, dtype=np.int64)


def canonical_blocks(block_ids):
    """Relabel a partition by first appearance; equal partitions compare equal."""
    ids = np.asarray(block_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("block IDs must be an integer vector on the SNP axis")
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse].astype(np.int64)


@dataclass(frozen=True)
class CrossBlockMoments:
    """Sufficient statistics for the between-block person covariance.

    Components c use SUMMIT's annotation-major context-pair order. With
    alpha_F(i) the response-side Hajek row of block F and S(i) the context
    pair sums, all arrays are stratum-centered sums over participants:

    block_gram[F]     sum_i alpha_F alpha_F.T
    block_context[F]  sum_i alpha_F (x) S(i)
    total_*           the same for alpha = sum_F alpha_F
    context_fourth    sum_i S(i) (x) S(i);  context_total: uncentered sum_i S(i)

    Optional nuisance arrays propagate a same-study probit risk fit.
    """
    block_ids: np.ndarray
    block_gram: np.ndarray
    block_context: np.ndarray
    total_gram: np.ndarray
    total_context: np.ndarray
    context_fourth: np.ndarray
    context_total: np.ndarray
    n_samples: int
    nuisance_jacobian: np.ndarray | None = None
    nuisance_context: np.ndarray | None = None
    score_influence: np.ndarray | None = None
    context_influence: np.ndarray | None = None
    influence_gram: np.ndarray | None = None

    def __post_init__(self):
        ids = canonical_blocks(self.block_ids)
        if not np.array_equal(ids, np.asarray(self.block_ids)):
            raise ValueError("cross-block IDs must be labelled by first appearance")
        b = int(ids.max())+1
        if b < 2:
            raise ValueError("person cross-block covariance needs at least two blocks")
        gram = finite_array("cross-block gram", self.block_gram, 3)
        c = gram.shape[1]
        context = finite_array("cross-block context", self.block_context, 4)
        p = context.shape[2]
        if gram.shape != (b, c, c) or context.shape != (b, c, p, p) or c % p:
            raise ValueError("cross-block moment axes disagree")
        shapes = dict(total_gram=(c, c), total_context=(c, p, p), context_fourth=(p, p, p, p), context_total=(p, p))
        for name, shape in shapes.items():
            if finite_array(name, getattr(self, name)).shape != shape:
                raise ValueError("cross-block moment axes disagree")
        if type(self.n_samples) is not int or self.n_samples < 8:
            raise ValueError("invalid cross-block sample count")
        nuisance = ("nuisance_jacobian", "nuisance_context", "score_influence", "context_influence", "influence_gram")
        present = [getattr(self, name) is not None for name in nuisance]
        if any(present) and not all(present):
            raise ValueError("incomplete cross-block nuisance moments")
        if all(present):
            r = np.asarray(self.influence_gram).shape[0]
            expected = dict(nuisance_jacobian=(c, r), nuisance_context=(r, p, p), score_influence=(c, r),
                            context_influence=(p, p, r), influence_gram=(r, r))
            for name, shape in expected.items():
                if finite_array(name, getattr(self, name)).shape != shape:
                    raise ValueError("cross-block nuisance axes disagree")
                object.__setattr__(self, name, readonly(getattr(self, name)))
        for name in ("block_gram", "block_context", *shapes):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        value = np.array(ids, dtype=np.int64)
        value.setflags(write=False)
        object.__setattr__(self, "block_ids", value)

    @property
    def block_count(self):
        return self.block_gram.shape[0]

    @property
    def component_count(self):
        return self.block_gram.shape[1]

    @property
    def has_nuisance(self):
        return self.influence_gram is not None

    def matches(self, block_ids):
        ids = np.asarray(block_ids)
        return ids.shape == self.block_ids.shape and np.array_equal(canonical_blocks(ids), self.block_ids)


class CrossBlockAccumulator:
    """Stream genotype blocks once; keep only N x C open-block buffers."""

    def __init__(self, annotations, features, response, cases, block_ids, *, nn=None, tn=None,
                 influence=None, response_derivative=None, log_weight=None):
        from .gxe import context_pairs
        a = finite_array("annotations", annotations, 2)
        self.features = finite_array("context features", features, 2)
        self.response = finite_array("responses", response, 1)
        n, q = self.features.shape
        if len(self.response) != n or np.asarray(cases).shape != (n,):
            raise ValueError("cross-block person axes disagree")
        self.cases = np.asarray(cases, dtype=bool)
        ids = canonical_blocks(block_ids)
        if len(ids) != len(a):
            raise ValueError("cross-block IDs must cover every SNP")
        self.block_ids = ids
        self.block_sizes = np.bincount(ids)
        if len(self.block_sizes) < 2:
            raise ValueError("person cross-block covariance needs at least two blocks")
        self.weights = a/a.sum(axis=0)
        self.pairs = context_pairs(q)
        self.k, self.p = a.shape[1], len(self.pairs)
        self.c = self.k*self.p
        from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator
        from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator
        self.nn = NumpyNNOperator() if nn is None else nn
        self.tn = NumpyTNOperator() if tn is None else tn
        self.weighted_response = np.asfortranarray(self.features*self.response[:, None])
        self.S = context_pair_sums(self.features)
        self.S_centered = _center(self.S.reshape(n, -1), self.cases).reshape(self.S.shape)
        b = len(self.block_sizes)
        self.block_gram = np.zeros((b, self.c, self.c))
        self.block_context = np.zeros((b, self.c, self.p, self.p))
        self.actions = np.zeros((n, self.c))
        self.open = {}
        self.seen = np.zeros(b, dtype=np.int64)
        self.visited = np.zeros(len(a), dtype=bool)
        self.nuisance = influence is not None
        if self.nuisance:
            self.influence = finite_array("risk influence", influence, 2)
            self.response_derivative = finite_array("response derivative", response_derivative, 2)
            self.log_weight = finite_array("log feature weight derivative", log_weight, 2)
            if not (self.influence.shape == self.response_derivative.shape == self.log_weight.shape) or len(self.influence) != n:
                raise ValueError("risk nuisance axes disagree")

    def add_block(self, start, stop, x):
        """Add SNPs [start, stop); x holds their population-scaled genotypes."""
        x = np.asfortranarray(x, dtype=np.float64)
        if x.shape != (len(self.response), stop-start) or np.any(self.visited[start:stop]):
            raise ValueError("cross-block genotype blocks must cover each SNP once")
        self.visited[start:stop] = True
        # s_jq = sum_l x_lj psi_lq r_l over all participants.
        scores = self.tn.matmul_tn(x, self.weighted_response)
        ids = self.block_ids[start:stop]
        cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, stop-start]
        for first, last in zip(cuts[:-1], cuts[1:]):
            block = int(ids[first])
            w = np.asfortranarray(self.weights[start+first:start+last])
            xs = x[:, first:last]
            s = scores[first:last]
            rhs = np.asfortranarray((w[:, :, None]*s[:, None, :]).reshape(last-first, -1))
            values = self.nn.matmul(xs, rhs).reshape(len(xs), self.k, s.shape[1])
            diagonal = self.nn.matmul(np.asfortranarray(xs*xs), w)
            out = self.open.setdefault(block, np.zeros((len(xs), self.c)))
            psi, r = self.features, self.response
            for a in range(self.k):
                for j, (u_, v_) in enumerate(self.pairs):
                    col = a*self.p+j
                    out[:, col] += psi[:, u_]*values[:, a, v_]
                    if u_ != v_:
                        out[:, col] += psi[:, v_]*values[:, a, u_]
                    out[:, col] -= (1 if u_ == v_ else 2)*psi[:, u_]*psi[:, v_]*r*diagonal[:, a]
            self.seen[block] += last-first
            if self.seen[block] == self.block_sizes[block]:
                self._close(block)

    def _close(self, block):
        actions = self.open.pop(block)
        alpha = _center(self.response[:, None]*actions, self.cases)
        self.block_gram[block] = alpha.T @ alpha
        self.block_context[block] = np.tensordot(alpha, self.S_centered, axes=(0, 0))
        self.actions += actions

    def finalize(self):
        if self.open or not np.all(self.visited) or not np.array_equal(self.seen, self.block_sizes):
            raise ValueError("person cross-block traversal is incomplete")
        n = len(self.response)
        alpha_raw = self.response[:, None]*self.actions
        alpha = _center(alpha_raw, self.cases)
        flat = self.S_centered.reshape(n, -1)
        values = dict(block_ids=self.block_ids, block_gram=self.block_gram, block_context=self.block_context,
            total_gram=alpha.T @ alpha, total_context=np.tensordot(alpha, self.S_centered, axes=(0, 0)),
            context_fourth=(flat.T @ flat).reshape((self.p,)*4), context_total=self.S.sum(axis=0), n_samples=n)
        if self.nuisance:
            influence = _center(self.influence, self.cases)
            values.update(
                nuisance_jacobian=2*self.actions.T @ (self.response_derivative+self.response[:, None]*self.log_weight),
                nuisance_context=np.tensordot(self.log_weight, self.S, axes=(0, 0)),
                score_influence=alpha.T @ influence,
                context_influence=np.tensordot(self.S_centered, influence, axes=(0, 0)),
                influence_gram=influence.T @ influence)
        self.S = self.S_centered = None
        return CrossBlockMoments(**values)


class CrossBlockOperator:
    """Record person rows during the first existing genotype traversal."""

    def __init__(self, operator, accumulator):
        self.operator = operator
        self.accumulator = accumulator

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def read_block(self, start, stop):
        block = self.operator.read_block(start, stop)
        if self.operator.observed_passes == 1:
            self.accumulator.add_block(start, stop, block.values)
        return block


def _pair_index(c, p):
    return np.arange(c) % p


def _gram(gram, context, gamma, theta, fourth, p):
    """Stratum-centered sum_i psi_i psi_i.T for psi_i = alpha_i - Lambda_i theta."""
    c = len(theta)
    pc = _pair_index(c, p)
    weights = gamma*theta[None, :]
    grouped = np.zeros((c, p))
    for r in range(p):
        grouped[:, r] = weights[:, pc == r].sum(axis=1)
    # Lambda_i[d,e] = gamma[d,e] S_i[p(d),p(e)], so with grouped[d,r] summing
    # gamma[d,e] theta[e] over p(e)=r:
    #   cross[c,d] = sum_i alpha_c (Lambda_i theta)_d = sum_r context[c,p(d),r] grouped[d,r]
    #   quad[c,d]  = sum_i (Lambda_i theta)_c (Lambda_i theta)_d
    cross = np.einsum('cdr,dr->cd', context[:, pc, :], grouped)
    quad = np.einsum('cr,ds,crds->cd', grouped, grouped, fourth[pc][:, :, pc])
    return gram-cross-cross.T+quad


def genotype_factors(normal, context_total):
    """Separable genotype factor g[a,b] with H[(a,p),(b,r)] ~ g[a,b] S_tot[p,r].

    Least squares over context pairs, weighting by S_tot^2. A single ratio
    H/S_tot would be unstable for context pairs whose products average to
    nearly zero (for example a centered exposure against the intercept).
    """
    p = context_total.shape[0]
    c = normal.shape[-1]
    k = c//p
    values = np.asarray(normal).reshape(*normal.shape[:-2], k, p, k, p)
    weight = np.einsum('pr,pr->', context_total, context_total)
    if weight <= 0:
        raise ValueError("context pair sums vanish")
    return np.einsum('...apbr,pr->...ab', values, context_total)/weight


def _row_coefficients(factors, p):
    """gamma[c,d] = g[a(c),a(d)], so that Lambda_i[c,d] = gamma[c,d] S_i[p(c),p(d)]."""
    return np.repeat(np.repeat(factors, p, axis=-2), p, axis=-1)


def crossblock_covariance(moments, theta, normal, block_normals):
    """Added covariance of the estimating function, in normal-equation units.

    normal is the full normalized H; block_normals[F] is block F's target-row
    contribution to H. Returns (between-block part, nuisance part, details).
    """
    theta = finite_array("coefficients", theta, 1)
    H = finite_array("normal matrix", normal, 2)
    blocks = finite_array("block normal matrices", block_normals, 3)
    c, p = moments.component_count, moments.context_total.shape[0]
    if len(theta) != c or H.shape != (c, c) or blocks.shape != (moments.block_count, c, c):
        raise ValueError("cross-block covariance axes disagree")
    pc = _pair_index(c, p)
    total_gamma = _row_coefficients(genotype_factors(H, moments.context_total), p)
    block_gamma = _row_coefficients(genotype_factors(blocks, moments.context_total), p)
    total = _gram(moments.total_gram, moments.total_context, total_gamma, theta, moments.context_fourth, p)
    within = np.zeros_like(total)
    for F in range(moments.block_count):
        within += _gram(moments.block_gram[F], moments.block_context[F], block_gamma[F], theta,
                        moments.context_fourth, p)
    between = 4*(total-within)
    between = (between+between.T)/2
    nuisance = np.zeros_like(between)
    if moments.has_nuisance:
        weights = total_gamma*theta[None, :]
        grouped = np.stack([weights[:, pc == r].sum(axis=1) for r in range(p)], axis=1)
        jacobian = moments.nuisance_jacobian-4*np.einsum('cr,kcr->ck', grouped, moments.nuisance_context[:, pc, :])
        row_influence = moments.score_influence-np.einsum('cr,crk->ck', grouped, moments.context_influence[pc])
        cross = 2*row_influence @ jacobian.T
        nuisance = cross+cross.T+jacobian @ moments.influence_gram @ jacobian.T
        nuisance = (nuisance+nuisance.T)/2
    fitted = np.einsum('ab,pr->apbr', genotype_factors(H, moments.context_total), moments.context_total).reshape(c, c)
    details = dict(between_block_trace=float(np.trace(between)), within_block_trace=float(4*np.trace(within)),
                   person_total_trace=float(4*np.trace(total)),
                   separable_normal_relative_error=float(np.linalg.norm(fitted-H)/np.linalg.norm(H)))
    return between, nuisance, details


def resolve_blocks(blocks, num_variants):
    """None or 0 disables; an integer selects the binary --njack partition."""
    if blocks is None or (type(blocks) is int and blocks == 0):
        return None
    if type(blocks) is int:
        return contiguous_blocks(num_variants, blocks)
    ids = canonical_blocks(blocks)
    if len(ids) != num_variants or ids.max() < 1:
        raise ValueError("cross-block IDs must cover every SNP with at least two blocks")
    return ids


def crossblock_workspace_bytes(n, m, q, k, blocks, risk_rank=0, block_size=256):
    """Person buffers, context sums, per-block moments and one block of scratch."""
    p = q*(q+1)//2
    c = k*p
    b = min(m, block_size)
    people = n*(4*c+2*p*p+3*max(risk_rank, 0)+k*(q+1)+q+2*k+2)
    moments = blocks*(c*c+c*p*p)+2*(c*c+c*p*p)+p**4+p*p
    return 8*(people+moments+n*b+m+b*q)+16*1024**2


def make_accumulator(annotations, contexts, risk, sd, features, response, method, block_ids, *,
                     risk_covariates=None, nn=None, tn=None):
    """Accumulator for prepare_* ; fitted probit risks need their covariates."""
    from .sampling import nuisance_derivatives
    if risk.source == "ascertained_probit" and risk_covariates is None:
        raise ValueError("person cross-block covariance with fitted risks requires the original risk covariates")
    extra = {}
    if risk_covariates is not None:
        _, influence, log_weight, derivative = nuisance_derivatives(risk, risk_covariates, method, sd)
        extra = dict(influence=influence, response_derivative=derivative, log_weight=log_weight)
    return CrossBlockAccumulator(annotations, features, response, risk.z > 0, block_ids, nn=nn, tn=tn, **extra)


def calibrated_covariance(moments, theta, normal, block_directed, block_count):
    """SNP-block jackknife plus between-block person covariance and risk-fit terms.

    Returns the added coefficient covariance and diagnostics. block_directed are
    the raw per-block directed numerators used by fit_gxe's jackknife.
    """
    crossblock = moments.crossblock_moments
    mass = np.repeat(moments.annotations.sum(axis=0), len(moments.pairs))
    blocks = moments.n_samples**2*np.asarray(block_directed)/np.outer(mass, mass)[None]
    between, nuisance, details = crossblock_covariance(crossblock, theta, normal, blocks)
    inverse = np.linalg.inv(normal)
    # The delete-block jackknife's expectation omits between-block covariance
    # and is also reduced by it, by a factor 1/(B-1) for equal blocks.
    factor = block_count/(block_count-1)
    added_between = factor*inverse @ between @ inverse.T
    added_nuisance = inverse @ nuisance @ inverse.T
    added_between = (added_between+added_between.T)/2
    added_nuisance = (added_nuisance+added_nuisance.T)/2
    details.update(method=METHOD, person_row_model=PERSON_ROW_MODEL, blocks=int(block_count),
                   between_block_factor=factor, risk_fit_uncertainty_included=crossblock.has_nuisance,
                   pair_term="omitted_between_block_second_order_hoeffding")
    return added_between, added_nuisance, details


# Status labels written to every binary G x E fit with uncertainty. See
# reports/pcgc_development/pcgc_gxe_binary_intervals.md for the grid behind them.
JACKKNIFE_STATUS = "calibrated_in_simulation_for_N_over_M_up_to_1"
JACKKNIFE_NOTE = (
    "SNP-block jackknife. Calibrated in simulations at array scale (N/M <= 1, including real UK "
    "Biobank genotypes with 448,562 SNPs at N = 337,845). It omits covariance between blocks "
    "through shared participants, which matters only when N/M is large (about 8 or more); "
    "--binary-crossblock-blocks adds it. See reports/pcgc_development/pcgc_gxe_binary_intervals.md.")
CALIBRATION_STATUS = "variance_calibrated_in_simulation_wald_coverage_can_be_below_nominal"
CALIBRATION_NOTE = (
    "Opt-in correction for large N/M. With N/M = 8-128 in simulation the variance ratio was "
    "0.89-1.14 against 0.44-1.26 for the jackknife. At array scale (N/M <= 1, including real "
    "UK Biobank genotypes) the jackknife was calibrated and this correction added noise or "
    "over-corrected; see reports/pcgc_development/pcgc_gxe_binary_intervals.md.")
# Opt-in since 2026-10-17 (default from 2026-10-15): at realistic N/M the
# jackknife alone was calibrated and the correction did not improve coverage.
DEFAULT_ENABLED = False
