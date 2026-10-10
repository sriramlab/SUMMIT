"""Independent-individual conditional-mean scores with estimated HC3 covariance.

These are asymptotic mean-effect tests, not FAME variance-component tests.
See docs/epistasis_robust_derivation.md for the estimand and nuisance influence.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import json
import numpy as np
from scipy.stats import norm, chi2, beta as beta_dist
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.spec import (
    array_sha256,
    owned_readonly_array,
    freeze_context_mapping,
)
from .quadratic import quadratic_sf
from .summary import _publish_bundle


def _inverse(h):
    diagonal = np.diag(h)
    if np.any(diagonal <= 0):
        raise ValueError("unidentifiable interaction columns")
    scale = np.sqrt(diagonal)
    corr = h / scale[:, None] / scale[None, :]
    eig, vec = np.linalg.eigh(corr)
    if eig[0] <= eig[-1] * 1e-10:
        raise ValueError("unidentifiable or ill-conditioned interaction columns")
    return ((vec / eig) @ vec.T) / scale[:, None] / scale[None, :], float(
        eig[-1] / eig[0]
    )


def _span_inverse(h, *, require_resolved=True):
    """Scale-aware generalized inverse and original-coordinate estimability.

    Used only internally for projections; nonestimable coefficients are never
    reported. A PSD matrix can define a valid kernel test on a proper span.
    """
    diagonal = np.diag(h)
    if np.any(diagonal < -1e-12 * max(float(np.max(abs(h))), 1e-300)):
        raise ValueError("indefinite robust information or covariance")
    scale = np.sqrt(np.maximum(diagonal, 0))
    scale[scale == 0] = 1
    corr = h / scale[:, None] / scale[None, :]
    eig, vec = np.linalg.eigh((corr + corr.T) / 2)
    tolerance = max(float(eig[-1]), 1e-300) * 64 * len(h) * np.finfo(float).eps
    if eig[0] < -tolerance:
        raise ValueError("indefinite robust information or covariance")
    keep = eig > tolerance
    if require_resolved and np.any(keep & (eig < eig[-1] * 1e-10)):
        raise ValueError(
            "ill-conditioned identifiable interaction span; no numerical truncation performed"
        )
    if not np.any(keep):
        return np.zeros_like(h), np.zeros(len(h), dtype=bool), 0, np.inf
    basis = vec[:, keep]
    inverse = ((basis / eig[keep]) @ basis.T) / scale[:, None] / scale[None, :]
    estimable = np.sum(vec[:, ~keep] ** 2, axis=1) < 1e-10
    return inverse, estimable, int(keep.sum()), float(eig[keep][-1] / eig[keep][0])


@dataclass(frozen=True)
class RobustScoreSummary:
    scores: np.ndarray
    information: np.ndarray
    score_covariance: np.ndarray
    feature_names: tuple
    trait_names: tuple
    metadata: dict

    def __post_init__(self):
        for key in ("scores", "information", "score_covariance"):
            a = owned_readonly_array(getattr(self, key), dtype=float)
            if not np.all(np.isfinite(a)):
                raise ValueError("nonfinite robust score summary")
            object.__setattr__(self, key, a)
        for key in ("feature_names", "trait_names"):
            names = tuple(getattr(self, key))
            if (
                not names
                or len(set(names)) != len(names)
                or any(not isinstance(x, str) or not x for x in names)
            ):
                raise ValueError("invalid robust score axis")
            object.__setattr__(self, key, names)
        p, t = len(self.feature_names), len(self.trait_names)
        if (
            self.scores.shape != (p, t)
            or self.information.shape != (p, p)
            or self.score_covariance.shape != (t, p, p)
        ):
            raise ValueError("robust score axes disagree")
        for index, a in enumerate([self.information, *self.score_covariance]):
            if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
                raise ValueError("asymmetric robust information or covariance")
            _span_inverse(a, require_resolved=index == 0)
        # Stored scores and meat must belong to the same feature span as H.
        # Normalize first so validation is invariant to original feature units.
        scale = np.sqrt(np.maximum(np.diag(self.information), 0))
        scale[scale == 0] = 1
        corr = self.information / scale[:, None] / scale[None, :]
        ci, _, _, _ = _span_inverse(corr)
        projector = corr @ ci
        score = self.scores / scale[:, None]
        matrices = [score] + [
            v / scale[:, None] / scale[None, :] for v in self.score_covariance
        ]
        if any(
            np.linalg.norm(v - projector @ v)
            > 1e-8 * max(np.linalg.norm(v), np.finfo(float).tiny)
            for v in matrices
        ):
            raise ValueError("robust scores or covariance lie outside the feature span")
        if any(
            not isinstance(self.metadata.get(k), str) or not self.metadata[k]
            for k in ("method", "inference")
        ):
            raise ValueError(
                "robust summaries must identify their estimator and inference assumptions"
            )
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))


class RobustMeanGeometry:
    """Reusable finite-mean geometry with the same HC3 rules as score preparation.

    Construct with ``prepare_robust_geometry``. No outcomes or mutable input
    design are retained. Large products use the supplied SUMMIT NN/TN hooks.
    """

    def fit(self, phenotypes):
        y = np.asarray(phenotypes, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or len(y) != len(self.r) or not np.all(np.isfinite(y)):
            raise ValueError("finite sample-aligned outcomes required")
        active_y = y.copy()
        active_y[self.saturated] = 0
        py = active_y - self.nn(self.u, self.tn(self.u, active_y))
        scores = self.tn(self.r, py)
        coefficients = self.nn(self.inverse, scores)
        residual = py - self.nn(self.r, coefficients)
        residual[self.saturated] = 0
        meat = np.empty((y.shape[1], self.r.shape[1], self.r.shape[1]))
        for j in range(y.shape[1]):
            weighted = self.r * (residual[:, j] / self.denominator)[:, None]
            meat[j] = self.tn(weighted, weighted)
            del weighted
        return scores, coefficients, residual, meat, py


@dataclass(frozen=True)
class RobustNuisanceGeometry:
    basis: np.ndarray
    design_shape: tuple
    design_hash: str


def prepare_robust_nuisance(fixed_effects):
    """Freeze a nuisance span once for multiple feature panels or outcomes."""
    c = np.asarray(fixed_effects, dtype=float)
    basis = thin_rank_revealing_fixed_effect_basis(c)
    basis.setflags(write=False)
    return RobustNuisanceGeometry(basis, c.shape, array_sha256(c))


def prepare_robust_geometry(features, fixed_effects=None, *, nuisance=None, nn=None, tn=None):
    """Factor a declared design once for repeated outcome transformations.

    Original feature coordinates, rank decisions, and saturated-row rules
    agree with ``prepare_robust_scores``. Designs are not cached globally.
    """
    f = np.asarray(features, dtype=float)
    if (fixed_effects is None) == (nuisance is None):
        raise ValueError('supply fixed effects or a prepared nuisance span, exclusively')
    if nuisance is None:
        nuisance = prepare_robust_nuisance(fixed_effects)
    if (not isinstance(nuisance, RobustNuisanceGeometry) or f.ndim != 2
            or not f.shape[1] or len(f) != nuisance.design_shape[0]
            or not np.all(np.isfinite(f))):
        raise ValueError("robust inputs must be finite and sample aligned")
    geometry = RobustMeanGeometry()
    nn = nn or (lambda a, b: a @ b)
    tn = tn or (lambda a, b: a.T @ b)
    u = nuisance.basis
    r = f - nn(u, tn(u, f))
    r -= nn(u, tn(u, r))
    energy = np.sum(r * r, axis=0)
    absorbed = energy <= (
        64 * np.finfo(float).eps * max(nuisance.design_shape + f.shape)
    ) ** 2 * np.maximum(np.sum(f * f, axis=0), np.finfo(float).tiny)
    r[:, absorbed] = 0
    h = tn(r, r)
    inverse, estimable, rank, condition = _span_inverse(h)
    if rank == 0:
        raise ValueError("interaction feature span is absorbed by the nuisance mean")
    leverage_c = np.sum(u * u, axis=1)
    partial = np.sum(r * nn(r, inverse), axis=1)
    leverage = leverage_c + partial
    roundoff = 64 * np.finfo(float).eps * max(nuisance.design_shape + f.shape)
    saturated = (abs(1 - leverage_c) <= roundoff) & (abs(partial) <= roundoff**2)
    active = ~saturated
    if len(f) - u.shape[1] - rank < 3 or np.any(leverage[active] >= 1 - 1e-8):
        raise ValueError(
            "insufficient residual support for HC3 inference: essential or unresolved unit leverage"
        )
    r[saturated] = 0
    denominator = 1 - leverage
    denominator[saturated] = 1
    for name, value in dict(u=u, r=r, h=h, inverse=inverse, estimable=estimable,
            rank=rank, condition=condition, leverage_c=leverage_c, partial=partial,
            leverage=leverage, saturated=saturated, active=active,
            denominator=denominator, nn=nn, tn=tn).items():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        setattr(geometry, name, value)
    return geometry


def prepare_robust_scores(
    features,
    phenotypes,
    fixed_effects,
    *,
    feature_names,
    trait_names,
    metadata,
    nn=None,
    tn=None,
    wild_draws=0,
    seed=1,
    sampling_model="fixed_design_correct_mean",
):
    """Fit all declared mean terms; retain signed scores and HC3 meat separately.

    Live sample storage is O(N(p+k+traits)); no sample-square matrices. Wild
    refitting is available for a single supplied direction, with fixed design.
    """
    f, y, c = (
        np.asarray(v, dtype=float) for v in (features, phenotypes, fixed_effects)
    )
    if y.ndim == 1:
        y = y[:, None]
    if (
        f.ndim != 2
        or f.shape[1] == 0
        or y.ndim != 2
        or c.ndim != 2
        or len(f) != len(y)
        or len(c) != len(y)
        or not all(np.all(np.isfinite(v)) for v in (f, y, c))
    ):
        raise ValueError("robust inputs must be finite and sample aligned")
    if type(wild_draws) is not int or (wild_draws and not 99 <= wild_draws <= 99999):
        raise ValueError("wild draws must be zero or 99..99999")
    if wild_draws and f.shape[1] != 1:
        raise ValueError("wild refitting currently requires a single frozen direction")
    if sampling_model not in ("fixed_design_correct_mean", "iid_population_projection"):
        raise ValueError("unknown mean-inference sampling model")
    if wild_draws and sampling_model != "fixed_design_correct_mean":
        raise ValueError(
            "this wild implementation is qualified only for a correct conditional mean"
        )
    nn = nn or (lambda a, b: a @ b)
    tn = tn or (lambda a, b: a.T @ b)
    geometry = prepare_robust_geometry(f, c, nn=nn, tn=tn)
    u, r, h, inverse = geometry.u, geometry.r, geometry.h, geometry.inverse
    estimable, rank, condition = geometry.estimable, geometry.rank, geometry.condition
    leverage_c, partial, leverage = geometry.leverage_c, geometry.partial, geometry.leverage
    saturated, active, denominator = geometry.saturated, geometry.active, geometry.denominator
    s, coef, e, meat, py = geometry.fit(y)
    supported = np.diag(h) > 0
    normalized = r[:, supported] / np.sqrt(np.diag(h)[supported])
    diagnostics = dict(
        n_samples=len(y),
        fixed_rank=u.shape[1],
        feature_rank=rank,
        original_feature_count=f.shape[1],
        estimable_coefficients=estimable.tolist(),
        nuisance_saturated_rows=int(saturated.sum()),
        max_active_leverage=float(leverage[active].max()),
        max_partial_leverage=float(partial.max()),
        max_leverage=float(leverage.max()),
        information_condition=condition,
        minimum_feature_effective_support=float(
            np.min(1 / np.sum(normalized**4, axis=0))
        ),
        fixed_hash=array_sha256(c),
        feature_hash=array_sha256(f),
        method="independent_mean_HC3_span_v2",
        estimand="conditional interaction mean coefficients",
        covariance_assumption="independent individuals; correctly specified conditional mean; unrestricted individual variances",
        inference="asymptotic; covariance and nuisance mean estimated from the same phenotype",
        sampling_model=sampling_model,
    )
    if sampling_model == "iid_population_projection":
        diagnostics.update(
            estimand="coefficient of supplied interaction features in the population linear projection onto the declared finite main and interaction span, conditional on independent frozen training",
            covariance_assumption="IID sampled individuals/designs; finite score second moments and stable information; the conditional mean may differ from its finite linear projection",
            biological_null="a zero population projection is not implied by absence of biological interaction without additional separability/independence assumptions",
        )
    # Frozen before independent confirmation; this screen only addresses
    # numerical/asymptotic support, never adequacy of the conditional mean.
    outside = []
    for condition_failed, reason in (
        (len(y) < 1000, "N below 1000"),
        (
            (u.shape[1] + rank - saturated.sum()) / active.sum() > 0.05,
            "fitted rank exceeds 5% of N",
        ),
        (leverage[active].max() > 0.1, "maximum leverage exceeds .1"),
        (
            diagnostics["minimum_feature_effective_support"] < 100,
            "feature effective support below 100",
        ),
        (condition > 1e6, "normalized information condition exceeds 1e6"),
    ):
        if condition_failed:
            outside.append(reason)
    diagnostics["outside_confirmation_design"] = outside
    diagnostics[
        "scope_note"
    ] = "Passing design diagnostics does not verify independence, the mean model, or genome-wide tail calibration"
    if wild_draws:
        # Restricted mean estimate cancels under the common projection. Every
        # generated residual is projected and its full HC3 covariance refitted.
        records = []
        for trait in range(y.shape[1]):
            # Common draws across traits preserve batch-versus-separate results.
            rng = np.random.default_rng(seed)
            null_denominator = 1 - leverage_c
            null_denominator[saturated] = 1
            null_residual = py[:, trait] / np.sqrt(null_denominator)
            null_residual[saturated] = 0
            observed = float(abs(s[0, trait]) / np.sqrt(meat[trait, 0, 0]))
            hits = 0
            for begin in range(0, wild_draws, 64):
                count = min(64, wild_draws - begin)
                v = null_residual[:, None] * (
                    2 * rng.integers(0, 2, size=(len(y), count)) - 1
                )
                v -= nn(u, tn(u, v))
                sb = tn(r, v)
                eb = v - nn(r, nn(inverse, sb))
                vb = np.sum((r[:, 0, None] * eb / denominator[:, None]) ** 2, axis=0)
                if np.any(vb <= 0):
                    raise ArithmeticError("undefined wild-bootstrap refit")
                hits += int(np.sum(abs(sb[0]) / np.sqrt(vb) >= observed))
            records.append(
                dict(
                    p=(hits + 1) / (wild_draws + 1),
                    exceedances=hits,
                    draws=wild_draws,
                    minimum_p=1 / (wild_draws + 1),
                    seed=seed,
                    draw_rule="same fixed Rademacher sequence for every trait",
                    monte_carlo_interval=[
                        0.0
                        if not hits
                        else float(beta_dist.ppf(0.025, hits, wild_draws - hits + 1)),
                        1.0
                        if hits == wild_draws
                        else float(beta_dist.ppf(0.975, hits + 1, wild_draws - hits)),
                    ],
                    inference="restricted HC2 residual Rademacher wild bootstrap; mean and HC3 refitted; asymptotic",
                )
            )
        diagnostics["wild_bootstrap"] = records
    return RobustScoreSummary(
        s,
        h,
        meat,
        tuple(feature_names),
        tuple(trait_names),
        dict(metadata, **diagnostics),
    )


def robust_score_tests(summary, *, trait=0, weights=None, burden=None, contrasts=None):
    if isinstance(trait, str):
        trait = summary.trait_names.index(trait)
    if (
        isinstance(trait, bool)
        or not isinstance(trait, (int, np.integer))
        or not 0 <= trait < len(summary.trait_names)
    ):
        raise ValueError("invalid robust trait selection")
    s, h, v = (
        summary.scores[:, trait],
        summary.information,
        summary.score_covariance[trait],
    )
    inverse, estimable, rank, _ = _span_inverse(h)
    coef = inverse @ s
    cov = inverse @ v @ inverse
    variance = np.diag(cov)
    if np.any(estimable & (variance < 0)):
        raise ArithmeticError("negative variance for an identifiable coefficient")
    # Do not take square roots of roundoff in nonidentifiable coordinates:
    # their coefficients and uncertainties are unavailable, regardless of sign.
    se = np.sqrt(variance, out=np.zeros_like(variance),
        where=estimable & (variance > 0))
    z = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
    p = len(s)
    w = np.ones(p) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (p,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid robust kernel weights")
    lam = np.linalg.eigvalsh(np.sqrt(w)[:, None] * v * np.sqrt(w)[None, :])
    if lam[0] < -1e-10 * lam[-1]:
        raise ValueError("indefinite robust covariance")
    if lam[-1] <= 0:
        tail = dict(p=None, absolute_error=None, method="zero_information_kernel")
    else:
        tail = quadratic_sf(
            float(np.dot(w, s * s)),
            lam[lam > 64 * np.finfo(float).eps * p * lam[-1]],
            atol=1e-9,
        )
    marginal = 2 * norm.sf(abs(z))
    covariance_available = estimable & (se > 0)
    marginal[~covariance_available] = np.nan
    reported_coef, reported_se = coef.copy(), se.copy()
    reported_coef[~estimable] = np.nan
    reported_se[~covariance_available] = np.nan
    reported_covariance = cov.copy()
    reported_covariance[~covariance_available, :] = np.nan
    reported_covariance[:, ~covariance_available] = np.nan
    vinverse, _, vrank, _ = _span_inverse(v, require_resolved=False)
    if vrank == 0:
        raise ValueError("zero score covariance: inference is undefined")
    diagnostics = {
        k: summary.metadata[k]
        for k in (
            "n_samples",
            "fixed_rank",
            "feature_rank",
            "max_leverage",
            "max_active_leverage",
            "max_partial_leverage",
            "nuisance_saturated_rows",
            "original_feature_count",
            "estimable_coefficients",
            "information_condition",
            "minimum_feature_effective_support",
            "outside_confirmation_design",
            "scope_note",
            "phenotype_units",
            "trait_unit",
            "phenotypes_hash",
            "fixed_hash",
            "feature_hash",
            "compatibility_id",
            "preparation_identity",
            "covariance_assumption",
            "kernel_target",
            "finite_sample_covariance",
            "covariance_precision",
            "support",
            "nuisance_training_n",
            "confirmation_n",
            "nuisance_fit_identity",
            "sampling_model",
            "status",
            "biological_null",
            "reuse",
            "genotype_passes_this_preparation",
        )
        if k in summary.metadata
    }
    out = dict(
        method=summary.metadata["method"],
        trait=summary.trait_names[trait],
        feature_names=summary.feature_names,
        beta=reported_coef,
        standard_errors=reported_se,
        coefficient_covariance=reported_covariance,
        estimable_coefficients=estimable,
        coefficient_covariance_available=covariance_available,
        interaction_rank=rank,
        beta_interval_95=np.column_stack(
            [
                reported_coef - 1.959963984540054 * reported_se,
                reported_coef + 1.959963984540054 * reported_se,
            ]
        ),
        conditional_p=marginal,
        joint_wald=float(s @ vinverse @ s),
        joint_df=vrank,
        kernel_p=tail["p"],
        kernel_p_error=tail["absolute_error"],
        kernel_tail_method=tail["method"],
        sparse_bonferroni_p=min(
            1.0, float(p * np.min(np.where(covariance_available, marginal, 1.0)))
        ),
        inference=summary.metadata["inference"],
        estimand=summary.metadata.get("estimand", "declared mean-feature coefficient"),
        diagnostics=diagnostics,
        interpretation="conditional mean association on the declared phenotype scale; not causal epistasis or a variance-component estimate",
    )
    if summary.metadata.get("sampling_model") == "iid_population_projection":
        out[
            "interpretation"
        ] = "population linear-projection mean association on the declared phenotype scale; not a conditional-design or biological-epistasis guarantee"
    out["joint_p"] = float(chi2.sf(out["joint_wald"], vrank))
    components = summary.metadata.get("component_index")
    if components is not None:
        out["conditional_components"] = []
        for j, name in enumerate(summary.metadata["component_names"]):
            take = np.flatnonzero(np.asarray(components) == j)
            if not len(take):
                raise ValueError("empty robust component")
            if not np.all(estimable[take]):
                out["conditional_components"].append(
                    dict(
                        name=name,
                        status="nonidentifiable original coefficient request",
                        p=None,
                    )
                )
                continue
            part = cov[np.ix_(take, take)]
            pinverse, _, prank, _ = _span_inverse(part, require_resolved=False)
            if prank < len(take):
                out["conditional_components"].append(
                    dict(
                        name=name,
                        status="singular conditional coefficient covariance",
                        p=None,
                    )
                )
                continue
            stat = float(coef[take] @ pinverse @ coef[take])
            pv = float(chi2.sf(stat, prank))
            out["conditional_components"].append(
                dict(
                    name=name,
                    df=len(take),
                    wald=stat,
                    p=pv,
                    component_family_bonferroni_p=min(
                        1.0, len(summary.metadata["component_names"]) * pv
                    ),
                )
            )
    contrasts = (
        summary.metadata.get("coefficient_contrasts")
        if contrasts is None
        else contrasts
    )
    if contrasts is not None:
        if not isinstance(contrasts, Mapping) or not contrasts:
            raise ValueError("coefficient contrasts require a nonempty named mapping")
        out["coefficient_contrasts"] = []
        for name, values in contrasts.items():
            vector = np.asarray(values, dtype=float)
            if (
                not isinstance(name, str)
                or not name
                or vector.shape != (p,)
                or not np.isfinite(vector).all()
                or not np.any(vector)
            ):
                raise ValueError("invalid coefficient contrast")
            if np.linalg.norm(vector - vector @ inverse @ h) > 1e-8 * np.linalg.norm(
                vector
            ):
                out["coefficient_contrasts"].append(
                    dict(name=name, status="nonestimable contrast", p=None)
                )
                continue
            value, variance = float(vector @ coef), float(vector @ cov @ vector)
            if variance <= 0:
                out["coefficient_contrasts"].append(
                    dict(name=name, status="zero contrast covariance", p=None)
                )
                continue
            error = np.sqrt(variance)
            pv = float(2 * norm.sf(abs(value / error)))
            out["coefficient_contrasts"].append(
                dict(
                    name=name,
                    beta=value,
                    standard_error=float(error),
                    p=pv,
                    interval_95=[
                        value - 1.959963984540054 * error,
                        value + 1.959963984540054 * error,
                    ],
                    family_bonferroni_p=min(1.0, len(contrasts) * pv),
                )
            )
    if burden is not None:
        b = np.asarray(burden, dtype=float)
        if b.shape != (p,) or not np.all(np.isfinite(b)) or np.sum(b * b) == 0:
            raise ValueError("invalid robust burden direction")
        if b @ v @ b <= 1e-12 * np.linalg.norm(v, 2) * (b @ b):
            out["burden_p"] = None
            out["burden_status"] = "absorbed or zero-variance burden"
            return out
        out["burden_z"] = float(b @ s / np.sqrt(b @ v @ b))
        out["burden_p"] = float(2 * norm.sf(abs(out["burden_z"])))
        out["adaptive_bonferroni_p"] = min(
            1.0,
            3
            * min(
                1.0 if tail["p"] is None else tail["p"] + tail["absolute_error"],
                out["burden_p"],
                out["sparse_bonferroni_p"],
            ),
        )
    if "wild_bootstrap" in summary.metadata:
        out["wild_bootstrap"] = dict(summary.metadata["wild_bootstrap"][trait])
    return out


def robust_followup(summary, groups, *, trait=0, alpha=0.05):
    """Conditional tests corrected over the complete supplied feature universe.

    The extra group gate does not enlarge the rejection set. Validity requires
    the full universe, mean model, and all preprocessing fixed before using
    this phenotype; post-selection display alone does not change the family.
    """
    if not 0 < alpha < 1 or not isinstance(groups, dict) or not groups:
        raise ValueError("invalid robust follow-up family")
    if isinstance(trait, str):
        trait = summary.trait_names.index(trait)
    fit = robust_score_tests(summary, trait=trait)
    inverse, estimable, _, _ = _span_inverse(summary.information)
    covariance = inverse @ summary.score_covariance[trait] @ inverse
    beta = fit["beta"]
    p = len(beta)
    adjusted = np.minimum(1, p * fit["conditional_p"])
    result = []
    for name, selected in groups.items():
        take = np.asarray(selected)
        if (
            not isinstance(name, str)
            or not name
            or take.ndim != 1
            or take.dtype.kind not in "iu"
            or not len(take)
            or len(set(take)) != len(take)
            or np.any((take < 0) | (take >= p))
        ):
            raise ValueError("invalid robust follow-up group")
        if not np.all(estimable[take]):
            result.append(
                dict(
                    group=name,
                    reported=False,
                    status="nonidentifiable conditional coefficients",
                    pair_names=[summary.feature_names[i] for i in take],
                )
            )
            continue
        part = covariance[np.ix_(take, take)]
        pinverse, _, prank, _ = _span_inverse(part, require_resolved=False)
        if prank < len(take):
            result.append(
                dict(
                    group=name,
                    reported=False,
                    status="singular conditional coefficient covariance",
                    pair_names=[summary.feature_names[i] for i in take],
                )
            )
            continue
        statistic = float(beta[take] @ pinverse @ beta[take])
        gp = min(1.0, len(groups) * float(chi2.sf(statistic, len(take))))
        result.append(
            dict(
                group=name,
                group_adjusted_p=gp,
                reported=gp <= alpha,
                pair_names=[summary.feature_names[i] for i in take],
                pair_adjusted_p=adjusted[take],
                gated_pair_p=np.maximum(gp, adjusted[take]),
            )
        )
    return dict(
        groups=result,
        alpha=alpha,
        pair_universe=p,
        method=fit["method"],
        correction="conditional coefficients; Bonferroni over complete supplied universe plus group gate",
        inference=fit["inference"],
        selection="complete supplied universe; no post-selection refitting",
        **({"status": summary.metadata["status"]} if "status" in summary.metadata else {}),
    )


def write_robust_scores(summary, path):
    from .cli import _jsonable

    arrays = {
        k: getattr(summary, k) for k in ("scores", "information", "score_covariance")
    }
    manifest = dict(
        kind="summit.epistasis.robust_score",
        schema_version=1,
        feature_names=summary.feature_names,
        trait_names=summary.trait_names,
        metadata=dict(summary.metadata),
        digests={k: array_sha256(v) for k, v in arrays.items()},
    )
    return _publish_bundle(path, _jsonable(manifest), arrays)


def load_robust_scores(path):
    with np.load(path, allow_pickle=False) as a:
        m = json.loads(str(a["manifest"]))
        arrays = {k: a[k] for k in a.files if k != "manifest"}
    if m.get("kind") != "summit.epistasis.robust_score" or m.get("schema_version") != 1:
        raise ValueError("unsupported robust score artifact")
    if set(arrays) != {"scores", "information", "score_covariance"} or m["digests"] != {
        k: array_sha256(v) for k, v in arrays.items()
    }:
        raise ValueError("robust artifact fields or digests disagree")
    return RobustScoreSummary(
        **arrays,
        feature_names=m["feature_names"],
        trait_names=m["trait_names"],
        metadata=m["metadata"],
    )
