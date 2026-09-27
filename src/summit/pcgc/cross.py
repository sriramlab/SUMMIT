"""Research cross-trait moments on one aligned, unprojected master sample axis.

The declared selection design must preserve each cohort's marginal sampling
law: inclusion depends on that trait's case status alone. Shared controls drawn
only from joint eligibility do not automatically satisfy this contract.
"""
from dataclasses import dataclass
import numpy as np

from summit.context.cross_trait_gram import cohort_rows
from summit.context.cross_trait_uncertainty import paired_delta_covariances
from summit.sumstats.binary import finite_array, readonly
from .moments import BinaryMoments, fit_moments, method_vectors
from .reference import generalized_reference

SELECTION_CONTRACT = "marginal_case_status_sampling_v1"


@dataclass(frozen=True)
class QuantitativeResponse:
    """Known population residual scale, without sample variance normalization."""
    z: np.ndarray
    covariate_variance: float = 0.

    def __post_init__(self):
        z = finite_array("quantitative residual response", self.z, 1)
        if len(z) < 3 or not np.isfinite(self.covariate_variance) or self.covariate_variance < 0:
            raise ValueError("invalid quantitative response or covariate variance")
        object.__setattr__(self, "z", readonly(z))

    @property
    def n_samples(self):
        return len(self.z)

    @property
    def sensitivity(self):
        return np.ones(len(self.z))


def response_vectors(risk, method):
    if isinstance(risk, QuantitativeResponse):
        return risk.sensitivity, risk.z
    return method_vectors(risk, method)


@dataclass(frozen=True)
class PairMoments:
    left: BinaryMoments
    right: BinaryMoments
    cross: BinaryMoments
    covariance_scale: float
    overlap_count: int


def prepare_pair(operator, annotations, left, right, left_rows, right_rows, *,
                 selection_contract, method="pcgc", **options):
    if selection_contract != SELECTION_CONTRACT:
        raise ValueError("binary cross-trait selection must preserve both marginal case-status sampling laws")
    if method not in ("pcgc", "pcgc-inverse", "liability"):
        raise ValueError("cross-trait research currently supports exact-risk PCGC or inverse moments")
    n = operator.num_samples
    i, j = cohort_rows(left_rows, n), cohort_rows(right_rows, n)
    if len(i) != left.n_samples or len(j) != right.n_samples:
        raise ValueError("cross-trait cohort rows and risk axes disagree")
    if len(np.union1d(i, j)) != n:
        raise ValueError("master genotype axis must be the union of the two cohorts")
    phi, responses = np.zeros((n, 2)), np.zeros((n, 2))
    phi[i, 0], responses[i, 0] = response_vectors(left, method)
    phi[j, 1], responses[j, 1] = response_vectors(right, method)
    diagonal_responses = np.column_stack([responses**2, responses[:, 0]*responses[:, 1]])
    ref, scored, _ = generalized_reference(operator, annotations, phi, responses=responses,
                                           diagonal_responses=diagonal_responses, collect_diagonal_rows=True,
                                           diagonal_weights=np.column_stack([phi**4, (phi[:, 0]*phi[:, 1])**2]), **options)
    pairs = ref.pair_table
    p0, p1, pc = pairs.index((0, 0)), pairs.index((1, 1)), pairs.index((0, 1))
    k = len(ref.annotation_masses)
    columns = {p: [ref.component_table.index((a, p)) for a in range(k)] for p in (p0, p1, pc)}
    within = []
    for side, risk, p in ((0, left, p0), (1, right, p1)):
        ld = ref.directional_ldscores[:, p, columns[p]]*(n/risk.n_samples)**2
        ld -= scored.reference_diagonal_rows[:, side]/risk.n_samples**2
        sp = ref.same_person[np.ix_(columns[p], columns[p])]
        rhs = scored.scores[:, side]**2-scored.diagonals[:, side]
        within.append(BinaryMoments(annotations, ld, sp, rhs, risk.n_samples, method, risk.covariate_variance))
    # With P=I, F0.T F1 = X.T diag(phi0*phi1) X is symmetric. Therefore
    # L_(01,01)/2 - L_(00,11) recovers the ordered rectangular Frobenius
    # moment exactly, including overlap. This identity fails for arbitrary
    # projected contextual features and is not applied to that machinery.
    ld = .5*ref.directional_ldscores[:, pc, columns[pc]]-ref.directional_ldscores[:, p0, columns[p1]]
    ld -= scored.reference_diagonal_rows[:, 2]/n**2
    sp = .25*ref.same_person[np.ix_(columns[pc], columns[pc])]
    rhs = scored.scores[:, 0]*scored.scores[:, 1]-scored.diagonals[:, 2]
    cross = BinaryMoments(annotations, ld, sp, rhs, n, method)
    scale = np.sqrt((1+left.covariate_variance)*(1+right.covariate_variance))
    return PairMoments(*within, cross, float(scale), len(np.intersect1d(i, j)))


def correlation(values):
    values = np.asarray(values)
    valid = (values[..., 0].real > 0) & (values[..., 1].real > 0)
    denominator = np.sqrt(np.where(valid, values[..., 0]*values[..., 1], 1))
    return {"rg": np.where(valid, values[..., 2]/denominator, np.nan)}


def fit_pair(pair, *, block_ids=None):
    fits = [fit_moments(m, block_ids=block_ids) for m in (pair.left, pair.right, pair.cross)]
    point = np.array([fit["conditional_total"] for fit in fits])
    rg = float(correlation(point)["rg"])
    result = dict(conditional_heritabilities=point[:2].tolist(), conditional_covariance=float(point[2]),
                  marginal_covariance=float(point[2]/pair.covariance_scale),
                  genetic_correlation=rg if np.isfinite(rg) else None,
                  overlap_count=pair.overlap_count, selection_contract=SELECTION_CONTRACT,
                  uncertainty_status="not_requested")
    if block_ids is not None:
        deleted = np.column_stack([np.asarray(f["jackknife_replicates"]).sum(axis=1) for f in fits])
        variance = float(paired_delta_covariances(correlation, point, deleted)["rg"][0, 0])
        result.update(conditional_covariance_standard_error=fits[2]["conditional_total_standard_error"],
                      marginal_covariance_standard_error=fits[2]["conditional_total_standard_error"]/pair.covariance_scale,
                      genetic_correlation_standard_error=float(np.sqrt(max(0, variance))) if np.isfinite(variance) else None,
                      finite_ratio_deletions=int(np.isfinite(correlation(deleted)["rg"]).sum()),
                      jackknife_method=fits[2]['jackknife_method'],
                      uncertainty_conditioning=fits[2]['uncertainty_conditioning'],
                      uncertainty_status=fits[2]['uncertainty_status'])
    return result
