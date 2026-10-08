"""Saved Gaussian score experiments: exact unknown-scale and refitted-null tests.

These tests are distinct from the FAME coefficient/SE Wald approximation.
Exact linear-model inference conditions on the declared feature matrix and
fixed effects, with Gaussian iid residuals of unknown scale. Kernel-ratio
tails retain the dependence between numerator and estimated residual scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
import numpy as np
from scipy.stats import f as f_dist, t as t_dist, beta as beta_dist
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.spec import (
    array_sha256,
    canonical_sha256,
    owned_readonly_array,
    freeze_context_mapping,
)
from .summary import _publish_bundle
from .quadratic import quadratic_sf


@dataclass(frozen=True)
class LinearScoreSummary:
    scores: np.ndarray
    information: np.ndarray
    residual_ss: np.ndarray
    residual_rank: int
    feature_names: tuple
    trait_names: tuple
    metadata: dict

    def __post_init__(self):
        for key in ("scores", "information", "residual_ss"):
            value = owned_readonly_array(getattr(self, key), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError("nonfinite score summary")
            object.__setattr__(self, key, value)
        for key in ("feature_names", "trait_names"):
            names = tuple(getattr(self, key))
            if (
                not names
                or len(set(names)) != len(names)
                or any(not isinstance(v, str) or not v for v in names)
            ):
                raise ValueError("invalid score axes")
            object.__setattr__(self, key, names)
        p, t = len(self.feature_names), len(self.trait_names)
        if (
            self.scores.shape != (p, t)
            or self.information.shape != (p, p)
            or self.residual_ss.shape != (t,)
        ):
            raise ValueError("score axes disagree")
        if (
            type(self.residual_rank) is not int
            or self.residual_rank < 2
            or np.any(self.residual_ss <= 0)
        ):
            raise ValueError("invalid residual rank or sums of squares")
        if not np.allclose(self.information, self.information.T, atol=1e-10):
            raise ValueError("asymmetric score information")
        eig = np.linalg.eigvalsh(self.information)
        if eig[0] < -1e-10 * max(eig[-1], np.finfo(float).tiny):
            raise ValueError("indefinite score information")
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))

    @cached_property
    def decomposition(self):
        return np.linalg.eigh(self.information)


def prepare_linear_scores(
    features,
    phenotypes,
    fixed_effects,
    *,
    feature_names,
    trait_names,
    metadata,
    nn=None,
    tn=None,
):
    """No sample-square matrix; common projection after interaction multiplication."""
    f, y = np.asarray(features, dtype=float), np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if (
        f.ndim != 2
        or y.ndim != 2
        or len(f) != len(y)
        or not np.all(np.isfinite(f))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("score inputs must be finite and sample aligned")
    u = thin_rank_revealing_fixed_effect_basis(fixed_effects)
    if len(u) != len(y):
        raise ValueError("fixed effects must use the same sample mask")
    nn = nn or (lambda a, b: a @ b)
    tn = tn or (lambda a, b: a.T @ b)
    pf = f - nn(u, tn(u, f))
    py = y - nn(u, tn(u, y))
    return LinearScoreSummary(
        tn(pf, py),
        tn(pf, pf),
        np.einsum("nt,nt->t", py, py),
        len(y) - u.shape[1],
        tuple(feature_names),
        tuple(trait_names),
        dict(
            metadata,
            method="gaussian_linear_unknown_scale_score_v1",
            fixed_hash=array_sha256(np.asarray(fixed_effects)),
            feature_hash=array_sha256(f),
            n_samples=len(y),
            covariance_assumption="sigma2_I_after_declared_fixed_effects",
        ),
    )


def linear_score_tests(summary, *, trait=0, weights=None, burden=None, atol=1e-10):
    """Exact Gaussian unknown-scale tails, joint F, burden, sparse and adaptive.

    The kernel statistic is s'Ws / y'Py. Its tail at q is the probability
    sum_i (lambda_i-q) Z_i² - q chi²_(r-p) >= 0, retaining scale dependence.
    Generalized eigenvalues may be signed in this ratio calculation.
    """
    if isinstance(trait, str):
        trait = summary.trait_names.index(trait)
    if (
        isinstance(trait, bool)
        or not isinstance(trait, (int, np.integer))
        or not 0 <= trait < len(summary.trait_names)
    ):
        raise ValueError("invalid trait selection")
    s, h, ss, r = (
        summary.scores[:, trait],
        summary.information,
        float(summary.residual_ss[trait]),
        summary.residual_rank,
    )
    p = len(s)
    w = np.ones(p) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (p,) or np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("kernel weights must be finite, nonnegative and nonempty")
    eig, vec = summary.decomposition
    keep = eig > max(eig[-1] * 1e-10, np.finfo(float).tiny)
    rank = int(keep.sum())
    if not rank or r < rank:
        raise ValueError("interaction information exceeds residual sample rank")
    if np.linalg.norm(s - vec[:, keep] @ (vec[:, keep].T @ s)) > 1e-7 * max(
        np.linalg.norm(s), np.finfo(float).tiny
    ):
        raise ValueError("scores are outside the information span")
    # Representation in the nonzero feature-information space avoids artificial
    # zero eigenvalues and permits a single multiplicity for the complement.
    root = np.sqrt(eig[keep])[:, None] * vec[:, keep].T
    lam = eig[keep] if weights is None else np.linalg.eigvalsh((root * w) @ root.T)
    q = float(np.dot(w, s * s) / ss)
    information_fraction = float(
        (np.sum(lam * lam) - np.sum(lam) ** 2 / r)
        / max(np.sum(lam * lam), np.finfo(float).tiny)
    )
    if information_fraction < 1e-8:
        raise ValueError("interaction kernel is indistinguishable from residual scale")
    values = np.r_[lam - q, -q] if r > rank else lam - q
    multiplicities = np.r_[np.ones(rank), r - rank] if r > rank else np.ones(rank)
    tail = quadratic_sf(0.0, values, multiplicities=multiplicities, atol=atol)
    fitted = float(np.sum((vec[:, keep].T @ s) ** 2 / eig[keep]))
    if fitted > ss * (1 + 1e-9):
        raise ValueError("scores and residual sums of squares are incompatible")
    remaining = max(ss - fitted, np.finfo(float).tiny)
    joint = (fitted / rank) / (remaining / (r - rank)) if r > rank else np.nan
    diagonal = np.diag(h)
    identifiable = diagonal > np.finfo(float).eps * max(
        diagonal.max(), np.finfo(float).tiny
    )
    individual_ss = np.zeros(p)
    individual_ss[identifiable] = s[identifiable] ** 2 / diagonal[identifiable]
    pair_t = np.full(p, np.nan)
    pair_t[identifiable] = s[identifiable] / np.sqrt(
        diagonal[identifiable]
        * np.maximum(ss - individual_ss[identifiable], np.finfo(float).tiny)
        / (r - 1)
    )
    marginal_p = np.full(p, np.nan)
    marginal_p[identifiable] = 2 * t_dist.sf(abs(pair_t[identifiable]), r - 1)
    result = dict(
        method=summary.metadata["method"],
        trait=summary.trait_names[trait],
        kernel_ratio=q,
        kernel_p=tail["p"],
        kernel_p_error=tail["absolute_error"],
        kernel_tail_method=tail["method"],
        joint_f=joint,
        joint_df=(rank, r - rank),
        joint_p=float(f_dist.sf(joint, rank, r - rank)),
        marginal_t=pair_t,
        marginal_p=marginal_p,
        sparse_bonferroni_p=min(1.0, float(p * np.nanmin(marginal_p))),
        unidentifiable_feature_names=[
            summary.feature_names[i] for i in np.flatnonzero(~identifiable)
        ],
        information_rank=rank,
        feature_count=p,
        residual_variance_null=ss / r,
        scale_adjusted_information_fraction=information_fraction,
        null="no effect in the supplied interaction-feature span; declared additive mean is correctly specified",
        inference="finite_sample_gaussian_unknown_scale",
    )
    if rank == p and r > rank:
        beta = vec[:, keep] @ ((vec[:, keep].T @ s) / eig[keep])
        inverse_diagonal = np.sum(vec[:, keep] ** 2 / eig[keep], axis=1)
        se = np.sqrt(inverse_diagonal * remaining / (r - rank))
        critical = t_dist.ppf(0.975, r - rank)
        result.update(
            joint_beta=beta,
            joint_beta_se=se,
            joint_beta_interval_95=np.column_stack(
                [beta - critical * se, beta + critical * se]
            ),
            joint_beta_p=2 * t_dist.sf(abs(beta / se), r - rank),
            interval_estimand="fixed realized interaction effects conditional on all supplied features",
        )
    if burden is not None:
        b = np.asarray(burden, dtype=float)
        if b.shape != (p,) or not np.all(np.isfinite(b)) or np.sum(b * b) == 0:
            raise ValueError("invalid prespecified burden direction")
        burden_identifiable = b @ h @ b > np.finfo(float).eps * max(
            np.sum(b * b) * diagonal.max(), np.finfo(float).tiny
        )
        bss = float((b @ s) ** 2 / (b @ h @ b)) if burden_identifiable else 0.0
        stat = (
            bss / max((ss - bss) / (r - 1), np.finfo(float).tiny)
            if burden_identifiable
            else np.nan
        )
        bp = float(f_dist.sf(stat, 1, r - 1)) if burden_identifiable else np.nan
        result.update(
            burden_f=stat,
            burden_p=bp,
            burden_identifiable=bool(burden_identifiable),
            adaptive_bonferroni_p=min(
                1.0,
                3
                * min(
                    tail["p"] + tail["absolute_error"],
                    bp if burden_identifiable else 1.0,
                    result["sparse_bonferroni_p"],
                ),
            ),
        )
    return result


def conditional_feature_summary(summary, selected):
    """Exact Schur-complement adjustment for remaining supplied fixed pair effects.

    This is a conditional fixed-effect hypothesis, distinct from fitting
    conditional variance coefficients. No genotype access is needed.
    """
    selected = np.asarray(selected)
    p = len(summary.feature_names)
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or len(set(selected)) != len(selected)
        or not len(selected)
        or np.any((selected < 0) | (selected >= p))
    ):
        raise ValueError("invalid conditional feature selection")
    remaining = np.setdiff1d(np.arange(p), selected)
    h = summary.information
    s = summary.scores
    if len(remaining):
        values, vectors = np.linalg.eigh(h[np.ix_(remaining, remaining)])
        keep = values > max(values[-1] * 1e-10, np.finfo(float).tiny)
        inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
        cross = h[np.ix_(selected, remaining)] @ inverse
        scores = s[selected] - cross @ s[remaining]
        information = (
            h[np.ix_(selected, selected)] - cross @ h[np.ix_(remaining, selected)]
        )
        ss = summary.residual_ss - np.sum(
            s[remaining] * (inverse @ s[remaining]), axis=0
        )
        rank = summary.residual_rank - int(keep.sum())
    else:
        scores = s[selected]
        information = h[np.ix_(selected, selected)]
        ss = summary.residual_ss
        rank = summary.residual_rank
    return LinearScoreSummary(
        scores,
        (information + information.T) / 2,
        ss,
        rank,
        tuple(summary.feature_names[i] for i in selected),
        summary.trait_names,
        dict(
            summary.metadata,
            conditioned_feature_names=[summary.feature_names[i] for i in remaining],
            hypothesis="fixed selected interaction effects conditional on remaining supplied interactions",
        ),
    )


def prespecified_followup(summary, groups, *, trait=0, alpha=0.05):
    """Gate follow-up with family adjustment over the entire supplied pair union.

    The full pair universe must be prepared before phenotype selection. Joint
    conditional pair t tests are Bonferroni adjusted over that entire universe,
    even when only pairs in significant groups are subsequently displayed.
    Separate group tests are also Bonferroni adjusted over all supplied groups.
    """
    if not 0 < alpha < 1 or not groups:
        raise ValueError("invalid prespecified follow-up family")
    full = linear_score_tests(summary, trait=trait)
    if "joint_beta_p" not in full:
        raise ValueError(
            "conditional pair follow-up requires identifiable joint coefficients and residual degrees of freedom"
        )
    pair_p = np.minimum(1, len(summary.feature_names) * full["joint_beta_p"])
    results = []
    for name, selected in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("group names must be nonempty strings")
        selected = np.asarray(selected)
        sub = conditional_feature_summary(summary, selected)
        test = linear_score_tests(sub, trait=trait)
        group_p = min(1.0, len(groups) * (test["kernel_p"] + test["kernel_p_error"]))
        results.append(
            dict(
                group=name,
                group_adjusted_p=group_p,
                reported=group_p <= alpha,
                pair_names=sub.feature_names,
                pair_adjusted_p=pair_p[selected],
                gated_pair_p=np.maximum(group_p, pair_p[selected]),
            )
        )
    return dict(
        groups=results,
        alpha=alpha,
        pair_universe=len(summary.feature_names),
        correction="conditional joint pair t tests, Bonferroni over the full supplied universe; additional adjusted group gate",
        assumption="all hypotheses supplied before phenotype selection; iid Gaussian residuals and correctly specified full mean",
    )


def write_linear_scores(summary, path):
    arrays = {k: getattr(summary, k) for k in ("scores", "information", "residual_ss")}
    m = dict(
        kind="summit.epistasis.linear_score",
        schema_version=1,
        residual_rank=summary.residual_rank,
        feature_names=summary.feature_names,
        trait_names=summary.trait_names,
        metadata=summary.metadata,
        digests={k: array_sha256(v) for k, v in arrays.items()},
    )
    return _publish_bundle(path, m, arrays)


def load_linear_scores(path):
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != {"manifest", "scores", "information", "residual_ss"}:
            raise ValueError("unexpected linear-score fields")
        m = json.loads(str(a["manifest"]))
        if m["kind"] != "summit.epistasis.linear_score" or m["schema_version"] != 1:
            raise ValueError("unsupported linear-score artifact")
        arrays = {k: a[k] for k in ("scores", "information", "residual_ss")}
        if m["digests"] != {k: array_sha256(v) for k, v in arrays.items()}:
            raise ValueError("linear-score digest mismatch")
        return LinearScoreSummary(
            **arrays,
            **{
                k: m[k]
                for k in ("residual_rank", "feature_names", "trait_names", "metadata")
            },
        )


def refit_bootstrap(reference, rotated_kernels, phenotype, *, draws=199, seed=1):
    """Parametric Gaussian calibration, refitting REML in every bootstrap draw.

    All genotype/projection/sketch definitions are fixed. This accounts for
    nuisance refitting under the fitted Gaussian model, not misspecification.
    Plus-one P values have resolution 1/(B+1), not genome-wide precision.
    """
    if type(draws) is not int or not 19 <= draws <= 9999:
        raise ValueError("bootstrap draws must be an integer between 19 and 9999")
    if type(seed) is not int or seed < 0 or not len(rotated_kernels):
        raise ValueError(
            "bootstrap requires a nonnegative integer seed and nonempty kernel family"
        )
    z = reference.transform(phenotype)
    fit = reference.fit(z)
    observed = [reference.efficient_score(z, k, fit) for k in rotated_kernels]
    rng = np.random.default_rng(seed)
    simulated = reference.simulate(fit, draws, rng)
    boot = np.empty((draws, len(rotated_kernels)))
    boundary = 0
    for b in range(draws):
        null = reference.fit(simulated[:, b])
        boundary += null["genetic_boundary"]
        for j, k in enumerate(rotated_kernels):
            boot[b, j] = reference.efficient_score(simulated[:, b], k, null)[
                "statistic"
            ]
    statistics = np.array([r["statistic"] for r in observed])
    exceedances = (boot >= statistics).sum(axis=0)
    p = (exceedances + 1) / (draws + 1)
    lo = np.where(
        exceedances == 0, 0, beta_dist.ppf(0.025, exceedances, draws - exceedances + 1)
    )
    hi = np.where(
        exceedances == draws,
        1,
        beta_dist.ppf(0.975, exceedances + 1, draws - exceedances),
    )
    return dict(
        method="gaussian_profile_reml_refit_parametric_bootstrap_v1",
        statistics=statistics,
        bootstrap_statistics=boot,
        p=p,
        monte_carlo_interval=np.column_stack([lo, hi]),
        draws=draws,
        seed=seed,
        minimum_p=1 / (draws + 1),
        null_fit=fit,
        efficient_information_fraction=[r["information_fraction"] for r in observed],
        bootstrap_genetic_boundary_count=int(boundary),
        null_identity=reference.identity,
        calibration="parametric bootstrap with estimated nuisance; not finite-sample exact",
        stochastic_contract="genotypes, fixed effects, probes and pair sketches fixed; phenotypes regenerated; REML refit every draw",
    )
