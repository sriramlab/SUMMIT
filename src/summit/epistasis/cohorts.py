"""Harmonized common-effect score experiments, including supplied overlap covariance.

No overlap is inferred from counts. Cross-score covariance must be on the exact
score axes (after cohort nuisance projection), in the declared phenotype units.
Known-covariance Gaussian inference remains distinct from fitted-covariance use.
"""
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from .pairs import PairScores
from summit.context.spec import canonical_sha256, array_sha256


def harmonize_pair_scores(
    summary,
    canonical_variants,
    *,
    genome_build,
    trait_unit,
    phenotype_multiplier=1.0,
    pair_order=None,
    trait_order=None,
):
    """Transform to raw counted-dosage products and common phenotype units.

    Cohort x_j=a_j(d_j-mu_j). With both main effects and an intercept projected,
    a raw, allele-aligned product equals t_p*x_i*x_j modulo the fixed span,
    t_p=sign_i sign_j/(a_i a_j). For y*=u*y and V*=u²V,
    s*=diag(t/u)s, H*=diag(t/u)Hdiag(t/u). Means need not agree.
    Strand complements/liftover are deliberately not guessed.
    """
    m = summary.metadata
    if m.get("genome_build") != genome_build or not genome_build:
        raise ValueError("explicit matching genome builds are required")
    if (
        not m.get("trait_unit")
        or not trait_unit
        or isinstance(phenotype_multiplier, bool)
        or not np.isfinite(phenotype_multiplier)
        or phenotype_multiplier <= 0
    ):
        raise ValueError(
            "positive scientific phenotype unit conversion and unit names required"
        )
    variants = m["variants"]
    current = {v: i for i, v in enumerate(variants["ids"])}
    required = {v for pair in summary.pair_ids for v in pair}
    if not required.issubset(m.get("fixed_main_effect_variants", ())):
        raise ValueError("affine pair harmonization requires all pair main effects")
    scaling = m.get("inverse_scales", {})
    factors = {}
    for v in required:
        if (
            v not in canonical_variants
            or v not in scaling
            or not np.isfinite(scaling[v])
            or scaling[v] <= 0
        ):
            raise ValueError("missing canonical allele or affine scale contract")
        i = current[v]
        target = canonical_variants[v]
        if (str(target["chromosome"]), int(target["position"])) != (
            str(variants["chromosome"][i]),
            int(variants["position"][i]),
        ):
            raise ValueError("variant coordinates disagree")
        alleles = (variants["counted"][i], variants["other"][i])
        wanted = (target["counted"], target["other"])
        if alleles == wanted:
            sign = 1.0
        elif alleles == wanted[::-1]:
            sign = -1.0
        else:
            raise ValueError("alleles disagree; strand complements are not inferred")
        factors[v] = sign / scaling[v]
    order = (
        tuple(sorted(summary.pair_ids))
        if pair_order is None
        else tuple(tuple(sorted(p)) for p in pair_order)
    )
    if len(set(order)) != len(order) or set(order) != set(summary.pair_ids):
        raise ValueError("cohort pair hypotheses disagree")
    positions = [summary.pair_ids.index(p) for p in order]
    d = np.array([factors[a] * factors[b] / phenotype_multiplier for a, b in order])
    # Return the transform as well: cross-cohort covariance must transform too.
    transform = np.eye(len(order))[positions] * d[:, None]
    ids = sorted(required)
    axis = dict(
        ids=ids,
        chromosome=[str(canonical_variants[v]["chromosome"]) for v in ids],
        position=[int(canonical_variants[v]["position"]) for v in ids],
        counted=[canonical_variants[v]["counted"] for v in ids],
        other=[canonical_variants[v]["other"] for v in ids],
    )
    traits = (
        tuple(sorted(summary.trait_names))
        if trait_order is None
        else tuple(trait_order)
    )
    if len(set(traits)) != len(traits) or set(traits) != set(summary.trait_names):
        raise ValueError("trait hypotheses disagree")
    columns = [summary.trait_names.index(t) for t in traits]
    result = PairScores(
        transform @ summary.scores[:, columns],
        transform @ summary.information @ transform.T,
        order,
        traits,
        dict(
            m,
            variants=axis,
            genome_build=genome_build,
            trait_unit=trait_unit,
            inverse_scales={v: 1.0 for v in ids},
            effect_units="common_trait_unit_per_raw_dosage_product",
            genotype_scale_contract="raw_counted_dosage_product",
            harmonization="explicit_alleles_coordinates_affine_units_v1",
        ),
    )
    return result, transform


def combine_score_experiments(
    summaries, *, cross_score_covariance=None, independent=False
):
    """Common signed effects from s ~ N(A beta,C), with A=vertical stack H_c.

    S=A'C^-1s, I=A'C^-1A are sufficient for the common effect. For independent
    cohorts C=blockdiag(H_c), this reduces to sum s_c and sum H_c. With overlap,
    C is required in full, including its known diagonal blocks. This conditions
    on supplied covariance and is not a fitted-nuisance calibration claim.
    """
    summaries = tuple(summaries)
    if (
        type(independent) is not bool
        or len(summaries) < 2
        or (cross_score_covariance is None) == (not independent)
    ):
        raise ValueError(
            "declare independence or supply the full cross-score covariance"
        )
    first = summaries[0]
    for s in summaries:
        if s.pair_ids != first.pair_ids or s.trait_names != first.trait_names:
            raise ValueError("harmonize pair and trait order before combining")
        for key in (
            "variants",
            "genome_build",
            "trait_unit",
            "effect_units",
            "genotype_scale_contract",
        ):
            if not s.metadata.get(key) or s.metadata[key] != first.metadata.get(key):
                raise ValueError(
                    "cohort scientific units/alleles require harmonization"
                )
    ids = [
        identity
        for s in summaries
        for identity in s.metadata.get(
            "cohort_sample_identities", (s.metadata["sample_identity"],)
        )
    ]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cohort identities")
    h = np.vstack([s.information for s in summaries])
    scores = np.vstack([s.scores for s in summaries])
    c = (
        block_diag(*[s.information for s in summaries])
        if independent
        else np.asarray(cross_score_covariance, dtype=float)
    )
    p = len(first.pair_ids)
    if (
        c.shape != (len(h), len(h))
        or not np.all(np.isfinite(c))
        or not np.allclose(c, c.T, atol=1e-10)
    ):
        raise ValueError("invalid cross-score covariance")
    for i, s in enumerate(summaries):
        if not np.allclose(
            c[i * p : (i + 1) * p, i * p : (i + 1) * p],
            s.information,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError(
                "cross-score diagonal disagrees with the cohort covariance"
            )
    if np.linalg.cond(c) > 1e12:
        raise ValueError(
            "overlapping score covariance is singular or poorly conditioned"
        )
    cf = cho_factor(c, lower=True)
    information = h.T @ cho_solve(cf, h)
    score = h.T @ cho_solve(cf, scores)
    known = all(s.metadata["covariance_known"] for s in summaries)
    from scipy.stats import chi2

    total = np.sum(scores * cho_solve(cf, scores), axis=0)
    common = np.sum(score * np.linalg.solve(information, score), axis=0)
    heterogeneous = total - common
    if np.any(heterogeneous < -1e-8 * np.maximum(total, 1)):
        raise ArithmeticError("negative cohort heterogeneity contrast")
    cohort_tests = dict(
        heterogeneous_effect_omnibus=total.tolist(),
        omnibus_df=len(h),
        omnibus_p=chi2.sf(total, len(h)).tolist(),
        heterogeneity_statistic=np.maximum(heterogeneous, 0).tolist(),
        heterogeneity_df=len(h) - p,
        heterogeneity_p=chi2.sf(np.maximum(heterogeneous, 0), len(h) - p).tolist(),
        assumption="Gaussian score experiment with known covariance"
        if known
        else "plug-in fitted covariance, calibration unestablished",
    )
    return PairScores(
        score,
        (information + information.T) / 2,
        first.pair_ids,
        first.trait_names,
        dict(
            first.metadata,
            sample_identity=canonical_sha256(dict(cohorts=ids)),
            cohort_sample_identities=ids,
            covariance_identity=canonical_sha256(
                dict(cohorts=ids, covariance=array_sha256(c))
            ),
            covariance_known=known,
            inference="exact_gaussian_known_covariance"
            if known
            else "plugin_gaussian_score",
            estimand="common_signed_pair_effect",
            cohort_tests=cohort_tests,
            overlap_contract="independent"
            if independent
            else "full_score_covariance_supplied",
        ),
    )


def combine_shared_variance_summaries(summaries):
    """Block-diagonal shared variance hyperparameters, independent cohort effects.

    T,q,traces,C add for block-diagonal kernels. This is NOT a common realized
    random effect across cohorts (that model has off-block covariance). The
    explicit contract includes all component units and phenotype units, and
    assumes shared nuisance variance coefficients as well as shared epistasis.
    FAME uncertainty limitations remain unchanged.
    """
    from .summary import EpistasisSummary

    summaries = tuple(summaries)
    if len(summaries) < 2:
        raise ValueError("at least two cohorts required")
    first = summaries[0]
    contract = first.metadata.get("cohort_variance_contract")
    if (
        not contract
        or contract.get("effect_distribution")
        != "independent_cohort_effects_shared_variances"
        or not contract.get("phenotype_unit")
        or set(contract.get("component_units", {})) != set(first.component_names)
    ):
        raise ValueError("explicit shared variance/component unit contract required")
    ids = []
    for s in summaries:
        if s.probe_matrices is not None:
            raise ValueError(
                "shared-variance combination currently requires exact references; cross-cohort probe covariance is not supplied"
            )
        if (
            s.component_names != first.component_names
            or s.trait_names != first.trait_names
            or s.metadata.get("cohort_variance_contract") != contract
            or s.metadata.get("overlap_contract") != "mutually_disjoint_cohorts"
        ):
            raise ValueError("independent cohort variance definitions disagree")
        ids.extend(
            s.metadata.get("cohort_sample_identities", (s.metadata.get("sample_hash"),))
        )
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate cohort identity")
    return EpistasisSummary(
        *(
            sum(getattr(s, k) for s in summaries)
            for k in ("matrix", "rhs", "traces", "cubic")
        ),
        component_names=first.component_names,
        trait_names=first.trait_names,
        n_samples=sum(s.n_samples for s in summaries),
        residual_rank=sum(s.residual_rank for s in summaries),
        metadata=dict(
            first.metadata,
            cohort_sample_identities=ids,
            sample_hash=canonical_sha256(dict(cohorts=ids)),
            estimand="shared covariance coefficients, independent cohort-specific effect draws",
            inference="FAME plug-in analytic comparator; no new calibration claim",
        ),
    )


def combine_manifest(path, output):
    import json
    from pathlib import Path
    from scipy.linalg import block_diag
    from summit.prediction._validation import closed
    from .pairs import load_pair_scores, write_pair_scores

    path = Path(path)
    spec = json.loads(path.read_text())
    closed(
        spec,
        (
            "kind",
            "schema_version",
            "cohorts",
            "canonical_variants",
            "genome_build",
            "trait_unit",
        ),
        ("independent", "cross_score_covariance"),
        name="cohort combination",
    )
    if spec["kind"] != "summit.epistasis.combine" or spec["schema_version"] != 1:
        raise ValueError("unsupported cohort combination manifest")
    if not spec["cohorts"]:
        raise ValueError("cohorts cannot be empty")
    with np.load(
        path.parent / spec["cohorts"][0]["summary"], allow_pickle=False
    ) as archive:
        kind = json.loads(str(archive["manifest"]))["kind"]
    if kind == "summit.epistasis.robust_score":
        from .robust_cohorts import combine_robust_manifest

        return combine_robust_manifest(path, output, spec)
    summaries = []
    transforms = []
    for item in spec["cohorts"]:
        closed(item, ("summary", "phenotype_multiplier"), name="cohort")
        summary, transform = harmonize_pair_scores(
            load_pair_scores(path.parent / item["summary"]),
            spec["canonical_variants"],
            genome_build=spec["genome_build"],
            trait_unit=spec["trait_unit"],
            phenotype_multiplier=item["phenotype_multiplier"],
        )
        summaries.append(summary)
        transforms.append(transform)
    covariance = None
    if "cross_score_covariance" in spec:
        raw = np.load(path.parent / spec["cross_score_covariance"], allow_pickle=False)
        transform = block_diag(*transforms)
        if raw.shape != transform.shape:
            raise ValueError(
                "cross-score covariance must use input cohort order and pair axes"
            )
        covariance = transform @ raw @ transform.T
    if type(spec.get("independent", False)) is not bool:
        raise ValueError("independent must be an explicit Boolean")
    return write_pair_scores(
        combine_score_experiments(
            summaries,
            cross_score_covariance=covariance,
            independent=spec.get("independent", False),
        ),
        output,
    )
