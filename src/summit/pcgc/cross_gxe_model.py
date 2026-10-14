"""Trait definitions and projection-aware overlap removal for cross-trait PCGC.

No participant-by-participant matrices are constructed. See the wiki guide
for the supported selection laws and conditional scales.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, ndtri

from summit.context.spec import array_sha256, canonical_sha256, freeze_context_mapping
from summit.sumstats.binary import finite_array, readonly, binary_phenotype, sampled_log_probabilities


# Threshold heteroscedasticity, delta=(eta^2-1)/2 at eta=Phi^-1(K), is
# positive (anticonservative SNP-block intervals) exactly when K < Phi(-1).
RARE_PREVALENCE = float(ndtr(-1.))


def _column_space(design, tolerance=1e-10):
    """Orthonormal basis of the numerical column space (rank-revealing SVD)."""
    if not design.shape[1]:
        return np.empty((len(design), 0))
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    return u[:, s > tolerance*max(design.shape)*s[0]] if s[0] > 0 else np.empty((len(design), 0))


def add_missing_columns(design, extra, *, tolerance=1e-8):
    """Append the columns of extra that are not already in span(design).

    Used to put every context column into a mean model (quantitative fixed
    effects or a fitted probit risk design) without duplicating columns the
    caller already supplied. Returns the augmented design and the number added.
    """
    design = np.asarray(design, dtype=float)
    extra = np.asarray(extra, dtype=float)
    if extra.ndim != 2 or design.ndim != 2 or len(extra) != len(design):
        raise ValueError("context and mean-model axes disagree")
    basis = _column_space(design)
    added = []
    for column in extra.T:
        norm = np.linalg.norm(column)
        if norm == 0:
            continue
        residual = column-basis@(basis.T@column)
        residual -= basis@(basis.T@residual)
        if np.linalg.norm(residual) > tolerance*norm:
            added.append(column)
            basis = np.column_stack((basis, residual/np.linalg.norm(residual)))
    if not added:
        return design, 0
    return np.column_stack([design, *added]), len(added)


def overlap_diagonal(ua, ub, *, minimum_relative_diagonal=1e-4, maximum_condition=1e8):
    """Same-person diagonal of P_A J P_B, with leverage and conditioning checks.

    The diagonal is 1 - h_A - h_B for combined leverage h_A + h_B. Entries are
    compared with the largest diagonal (relative threshold), and the condition
    number of D + V V' is bounded by (max d + ||V||^2) / min d.
    """
    a, b = np.asarray(ua), np.asarray(ub)
    if a.shape[0] != b.shape[0]:
        raise ValueError("overlap projection axes disagree")
    d = 1-np.sum(a*a, axis=1)-np.sum(b*b, axis=1)
    if not len(d):
        return d
    largest = float(np.max(d))
    if largest <= 0 or np.any(d <= minimum_relative_diagonal*largest):
        raise ValueError("overlap projection has excessive leverage; diagonal removal is unstable")
    if a.shape[1] and b.shape[1]:
        v = (a[:, :, None]*b[:, None, :]).reshape(len(a), a.shape[1]*b.shape[1])
        spread = float(np.linalg.eigvalsh(v.T@v)[-1])
        if (largest+spread)/float(np.min(d)) > maximum_condition:
            raise ValueError("overlap projection leverage makes the diagonal removal ill-conditioned")
    return d


def context_second_moment(contexts, liability_sd=1.):
    """Mean of phi phi' / s^2, used for the sample genetic-variance summary."""
    phi = np.asarray(contexts, dtype=float)/np.broadcast_to(np.asarray(liability_sd, dtype=float), (len(contexts),))[:, None]
    return (phi.T@phi/len(phi)).tolist()


@dataclass(frozen=True)
class CrossTrait:
    name: str
    samples: tuple
    contexts: np.ndarray
    context_names: tuple
    response: np.ndarray
    sensitivity: np.ndarray
    fixed_basis: np.ndarray
    metadata: dict

    def __post_init__(self):
        ids = tuple(tuple(p) for p in self.samples)
        if len(ids) < 3 or len(set(ids)) != len(ids) or any(
                len(p) != 2 or any(not isinstance(s, str) or not s for s in p) for p in ids):
            raise ValueError("traits require unique nonempty FID/IID identities")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("trait name must be nonempty")
        phi = finite_array("individual contexts", self.contexts, 2)
        y = finite_array("trait response", self.response, 1)
        w = finite_array("trait sensitivity", self.sensitivity, 1)
        u = finite_array("fixed-effect basis", self.fixed_basis, 2)
        n = len(ids)
        if phi.shape[0] != n or y.shape != (n,) or w.shape != (n,) or u.shape[0] != n:
            raise ValueError("trait sample axes disagree")
        q = phi.shape[1]
        if not 1 <= q < n or np.linalg.matrix_rank(phi) != q or np.any(w <= 0):
            raise ValueError("context basis must be full rank and sensitivity positive")
        names = tuple(self.context_names)
        if len(names) != q or len(set(names)) != q or any(not isinstance(s, str) or not s for s in names):
            raise ValueError("context names must be unique and match the basis")
        if u.shape[1] >= n or not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-11, rtol=1e-11):
            raise ValueError("fixed-effect basis must be orthonormal")
        if self.metadata.get("kind") not in ("binary", "quantitative"):
            raise ValueError("unknown trait type")
        if self.metadata.get("selection") not in ("outcome_independent", "own_case_status"):
            raise ValueError("unsupported selection law")
        if self.metadata["kind"] == "binary" and u.shape[1]:
            raise ValueError("binary features receive no OLS projection")
        if self.metadata["kind"] == "quantitative" and self.metadata["selection"] != "outcome_independent":
            raise ValueError("quantitative diagnosis-selected samples are unsupported")
        object.__setattr__(self, "samples", ids)
        object.__setattr__(self, "context_names", names)
        for key, val in (("contexts", phi), ("response", y), ("sensitivity", w), ("fixed_basis", u)):
            object.__setattr__(self, key, readonly(val))
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))

    @property
    def q(self):
        return self.contexts.shape[1]

    def identity(self):
        return dict(name=self.name, context_names=list(self.context_names), metadata=dict(self.metadata),
                    samples_sha256=canonical_sha256({"samples": self.samples}),
                    **{key+"_sha256": array_sha256(getattr(self, key)) for key in
                       ("contexts", "response", "sensitivity", "fixed_basis")})


def quantitative_trait(name, samples, y, contexts, *, context_names, population_sd,
                       fixed_effects=None, selection="outcome_independent",
                       context_main_effects=True):
    """Scale y by a known population SD and project a fixed mean model.

    fixed_effects=None means an intercept only. A supplied design includes
    any intercept; pass an n-by-0 array for no fixed design. By default the
    context columns also enter the projected design as main effects, unless
    already in its span: an omitted context main effect stays in the residual
    and is correlated with the context-weighted kernels.
    """
    y = finite_array("quantitative phenotype", y, 1)
    phi = finite_array("individual contexts", contexts, 2)
    sd = float(population_sd)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("declare a positive fixed quantitative population SD")
    if type(context_main_effects) is not bool:
        raise ValueError("context_main_effects must be a Boolean")
    if phi.shape[0] != len(y):
        raise ValueError("trait sample axes disagree")
    design = (np.ones((len(y), 1)) if fixed_effects is None else
              finite_array("quantitative fixed effects", fixed_effects, 2))
    if design.shape[0] != len(y) or design.shape[1] >= len(y):
        raise ValueError("fixed-effect axes disagree")
    if design.shape[1]:
        norm = np.linalg.norm(design, axis=0)
        if np.any(norm == 0) or np.linalg.matrix_rank(design/norm) != design.shape[1]:
            raise ValueError("fixed effects are rank deficient")
    added = 0
    if context_main_effects:
        design, added = add_missing_columns(design, phi)
        if design.shape[1] >= len(y):
            raise ValueError("fixed-effect axes disagree")
    u = np.empty((len(y), 0))
    if design.shape[1]:
        norm = np.linalg.norm(design, axis=0)
        u = np.linalg.qr(design/norm, mode="reduced")[0]
    return CrossTrait(name, tuple(samples), phi, tuple(context_names), y/sd,
                      np.ones(len(y)), u, dict(kind="quantitative", selection=selection,
                      population_sd=sd, sample_size=len(y), response_scale="supplied_population_sd",
                      nuisance_source="fixed_design_OLS", fixed_design_columns=int(design.shape[1]),
                      default_intercept=fixed_effects is None,
                      context_main_effects=context_main_effects, context_main_effects_added=added,
                      context_second_moment=context_second_moment(phi)))


def binary_trait(name, samples, y, contexts, *, context_names, liability_sd,
                 selection, population_risk=None, prevalence=None,
                 sampling_odds=None, risk_covariates=None,
                 allow_uncalibrated_supplied_risks=False):
    """Known risks or fitted probit; no automatic phenotype-dependent masking.

    sampling_odds is Pr(included|case)/Pr(included|control). For
    own-case-status sampling give it or the population prevalence, from
    which it is estimated as in existing PCGC. Outcome-independent sampling
    fixes it to 1 and uses neither.

    A fitted probit design always contains every context column (added unless
    already in the span of risk_covariates and the intercept). The score
    equations of that fit make the weighted contexts orthogonal to z.
    Own-case-status sampling requires such fitted risks: supplied risks, even
    the true ones, leave a sample-level weighted mean of z in each context
    stratum that multiplies the case-enriched genotype means; the resulting
    term is common to every SNP block and the jackknife misses it (null
    within-trait type-I error 8-9% instead of 5% in retrospective tests).
    allow_uncalibrated_supplied_risks=True keeps supplied risks there; the fit
    then marks that trait's within-trait intervals uncalibrated.
    """
    y = binary_phenotype(y)
    phi = finite_array("individual contexts", contexts, 2)
    if phi.shape[0] != y.size:
        raise ValueError("trait sample axes disagree")
    if type(allow_uncalibrated_supplied_risks) is not bool:
        raise ValueError("allow_uncalibrated_supplied_risks must be a Boolean")
    if prevalence is not None and (not np.isfinite(prevalence) or not 0 < prevalence < 1):
        raise ValueError("population prevalence must lie strictly inside (0,1)")
    sd = np.broadcast_to(finite_array("conditional liability SD", liability_sd), y.shape)
    if np.any(sd <= 0):
        raise ValueError("conditional liability SD must be positive")
    if selection not in ("outcome_independent", "own_case_status"):
        raise ValueError("unsupported binary selection; joint diagnosis and treatment restrictions need a joint law")
    if selection == "outcome_independent":
        if sampling_odds is not None and sampling_odds != 1:
            raise ValueError("outcome-independent selection requires sampling_odds=1")
        if prevalence is not None:
            raise ValueError("outcome-independent selection does not use a population prevalence; "
                             "the cohort case fraction enters the risk model directly")
        odds = 1.
    elif sampling_odds is None:
        if prevalence is None or not np.isfinite(prevalence) or not 0 < prevalence < 1:
            raise ValueError("own-case-status sampling needs prevalence or known sampling_odds")
        odds = y.mean()*(1-prevalence)/(prevalence*(1-y.mean()))
    else:
        if prevalence is not None:
            raise ValueError("give the population prevalence or known sampling_odds, not both")
        odds = float(sampling_odds)
    if not np.isfinite(odds) or odds <= 0:
        raise ValueError("sampling odds must be finite and positive")
    implied_prevalence = float(y.mean()/(odds*(1-y.mean())+y.mean()))
    source = "supplied"
    risk_identity = None
    risk_design = None
    added = 0
    if population_risk is None:
        if np.any(sd != 1):
            raise ValueError("fitted risks support unit conditional liability variance only")
        covariates = (np.empty((y.size, 0)) if risk_covariates is None else
                      finite_array("risk covariates", risk_covariates, 2))
        if covariates.shape[0] != y.size:
            raise ValueError("risk covariates must match the phenotype sample axis")
        design, added = add_missing_columns(np.column_stack((np.ones(y.size), covariates)), phi)
        risk_design = design[:, 1:]
        # Existing probit likelihood uses this same sampling-odds parameter.
        from summit.sumstats.binary import fit_binary_risk
        fitted = fit_binary_risk(y, implied_prevalence, risk_design if risk_design.shape[1] else None)
        population_risk = fitted.population_risk
        source = "fitted_probit"
        risk_identity = array_sha256(fitted.coefficients)
    elif risk_covariates is not None:
        raise ValueError("choose supplied risks or a fitted risk design")
    elif selection == "own_case_status" and not allow_uncalibrated_supplied_risks:
        raise ValueError("own-case-status sampling requires fitted probit risks whose design includes every "
                         "context column (omit population_risk): supplied risks over-reject null within-trait "
                         "components. Set allow_uncalibrated_supplied_risks to keep them; within-trait "
                         "intervals are then marked uncalibrated")
    k = finite_array("population risks", population_risk, 1)
    if k.shape != y.shape or np.any((k <= 0) | (k >= 1)):
        raise ValueError("population risks must be finite and strictly inside (0,1)")
    if source == "fitted_probit":
        calibration = "fitted_probit_contexts_in_design"
    elif selection == "own_case_status":
        calibration = "supplied_case_status_uncalibrated"
    else:
        calibration = "supplied_outcome_independent"
    # Population prevalence used for the rare-outcome calibration flag. With
    # outcome-independent sampling the sample risk mean estimates it.
    reference_prevalence = float(k.mean()) if selection == "outcome_independent" else implied_prevalence
    lp, lq, slope = sampled_log_probabilities(ndtri(k), -np.log(odds))
    z = np.where(y == 1, np.exp(.5*(lq-lp)), -np.exp(.5*(lp-lq)))
    w = slope*np.exp(.5*(lp+lq))/sd
    relative_w = w/np.max(w)
    return CrossTrait(name, tuple(samples), phi, tuple(context_names), z, w,
                      np.empty((len(y), 0)), dict(kind="binary", selection=selection,
                      sample_size=len(y),case_count=int(y.sum()),control_count=int(len(y)-y.sum()),
                      population_risk_range=[float(k.min()),float(k.max())],
                      liability_sd_range=[float(sd.min()),float(sd.max())],
                      sensitivity_effective_sample_size=float(np.sum(relative_w**2)**2/np.sum(relative_w**4)),
                      sampling_odds=float(odds), prevalence=prevalence, risk_source=source,
                      fitted_coefficients_sha256=risk_identity,
                      risk_design_sha256=(None if risk_design is None else array_sha256(risk_design)),
                      risk_design_context_columns_added=added, risk_calibration=calibration,
                      reference_prevalence=reference_prevalence,
                      risk_sha256=array_sha256(k), liability_sd_sha256=array_sha256(sd),
                      liability_scale="known_conditional_total_sd_common_units",
                      nuisance_source=source, context_second_moment=context_second_moment(phi, sd)))


class DiagonalRemoval:
    """Inverse of D(P_A J(.) P_B), via diagonal plus low rank Woodbury.

    The same inverse acts on all source diagonals and response products. It
    is never applied separately for every target SNP.
    """
    def __init__(self, ua, ub, products):
        self.products = products
        a, b = np.asarray(ua), np.asarray(ub)
        self.d = overlap_diagonal(a, b)
        self.v = np.asfortranarray((a[:, :, None]*b[:, None, :]).reshape(len(a), a.shape[1]*b.shape[1]))
        self.factor = None
        if len(self.d) and self.v.shape[1]:
            gram = products.tn(self.v, self.v/self.d[:, None])
            self.factor = cho_factor(np.eye(gram.shape[0])+gram, lower=True, check_finite=True)

    def apply(self, rhs):
        rhs = finite_array("overlap correction right hand side", rhs, 2)
        if len(rhs) != len(self.d):
            raise ValueError("overlap correction dimensions disagree")
        value = rhs/self.d[:, None]
        if self.factor is not None:
            small = cho_solve(self.factor, self.products.tn(self.v, value))
            value -= self.products.nn(self.v, small)/self.d[:, None]
        return np.asfortranarray(value)
