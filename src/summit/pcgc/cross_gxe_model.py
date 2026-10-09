"""Trait definitions and projection-aware overlap removal for cross-trait PCGC.

No participant-by-participant matrices are constructed. See the wiki guide
for the supported selection laws and conditional scales.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri

from summit.context.spec import array_sha256, canonical_sha256, freeze_context_mapping
from summit.sumstats.binary import finite_array, readonly, binary_phenotype, sampled_log_probabilities


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
                       fixed_effects=None, selection="outcome_independent"):
    """Scale y by a known population SD; fixed_effects includes any intercept."""
    y = finite_array("quantitative phenotype", y, 1)
    sd = float(population_sd)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("declare a positive fixed quantitative population SD")
    u = np.empty((len(y), 0))
    if fixed_effects is not None:
        design = finite_array("quantitative fixed effects", fixed_effects, 2)
        if design.shape[0] != len(y) or design.shape[1] >= len(y):
            raise ValueError("fixed-effect axes disagree")
        if design.shape[1]:
            norm = np.linalg.norm(design, axis=0)
            if np.any(norm == 0) or np.linalg.matrix_rank(design/norm) != design.shape[1]:
                raise ValueError("fixed effects are rank deficient")
            u = np.linalg.qr(design/norm, mode="reduced")[0]
    return CrossTrait(name, tuple(samples), contexts, tuple(context_names), y/sd,
                      np.ones(len(y)), u, dict(kind="quantitative", selection=selection,
                      population_sd=sd, response_scale="supplied_population_sd",
                      nuisance_source="fixed_design_OLS"))


def binary_trait(name, samples, y, contexts, *, context_names, liability_sd,
                 selection, population_risk=None, prevalence=None,
                 sampling_odds=None, risk_covariates=None):
    """Known risks or fitted probit; no automatic phenotype-dependent masking.

    sampling_odds is Pr(included|case)/Pr(included|control). If omitted for
    own-case-status sampling, estimate it from sample and supplied population
    prevalence, as in existing PCGC. Outcome-independent sampling fixes it to 1.
    """
    y = binary_phenotype(y)
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
        odds = 1.
    elif sampling_odds is None:
        if prevalence is None or not np.isfinite(prevalence) or not 0 < prevalence < 1:
            raise ValueError("own-case-status sampling needs prevalence or known sampling_odds")
        odds = y.mean()*(1-prevalence)/(prevalence*(1-y.mean()))
    else:
        odds = float(sampling_odds)
    if not np.isfinite(odds) or odds <= 0:
        raise ValueError("sampling odds must be finite and positive")
    source = "supplied"
    risk_identity = None
    if population_risk is None:
        if np.any(sd != 1):
            raise ValueError("fitted risks support unit conditional liability variance only")
        # Existing probit likelihood uses this same sampling-odds parameter.
        from summit.sumstats.binary import fit_binary_risk
        implied_prevalence = y.mean()/(odds*(1-y.mean())+y.mean())
        fitted = fit_binary_risk(y, implied_prevalence, risk_covariates)
        population_risk = fitted.population_risk
        source = "fitted_probit"
        risk_identity = array_sha256(fitted.coefficients)
    elif risk_covariates is not None:
        raise ValueError("choose supplied risks or a fitted risk design")
    k = finite_array("population risks", population_risk, 1)
    if k.shape != y.shape or np.any((k <= 0) | (k >= 1)):
        raise ValueError("population risks must be finite and strictly inside (0,1)")
    lp, lq, slope = sampled_log_probabilities(ndtri(k), -np.log(odds))
    z = np.where(y == 1, np.exp(.5*(lq-lp)), -np.exp(.5*(lp-lq)))
    w = slope*np.exp(.5*(lp+lq))/sd
    return CrossTrait(name, tuple(samples), contexts, tuple(context_names), z, w,
                      np.empty((len(y), 0)), dict(kind="binary", selection=selection,
                      sampling_odds=float(odds), prevalence=prevalence, risk_source=source,
                      fitted_coefficients_sha256=risk_identity,
                      risk_design_sha256=(None if risk_covariates is None else array_sha256(np.asarray(risk_covariates,dtype=float))),
                      risk_sha256=array_sha256(k), liability_sd_sha256=array_sha256(sd),
                      liability_scale="known_conditional_total_sd_common_units",
                      nuisance_source=source))


class DiagonalRemoval:
    """Inverse of D(P_A J(.) P_B), via diagonal plus low rank Woodbury.

    The same inverse acts on all source diagonals and response products. It
    is never applied separately for every target SNP.
    """
    def __init__(self, ua, ub, products, *, minimum_diagonal=1e-8):
        self.products = products
        a, b = np.asarray(ua), np.asarray(ub)
        if a.shape[0] != b.shape[0]:
            raise ValueError("overlap projection axes disagree")
        self.d = 1-np.sum(a*a, axis=1)-np.sum(b*b, axis=1)
        if np.any(self.d <= minimum_diagonal):
            raise ValueError("overlap projection has excessive leverage; diagonal removal is unstable")
        self.v = np.asfortranarray((a[:, :, None]*b[:, None, :]).reshape(len(a), a.shape[1]*b.shape[1]))
        self.factor = None
        if self.v.shape[1]:
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
