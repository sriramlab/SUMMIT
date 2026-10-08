"""Common signed mean effects: harmonized estimating equations and robust meat."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from summit.context.spec import canonical_sha256
from summit.prediction._validation import closed
from .robust import RobustScoreSummary, load_robust_scores, write_robust_scores
from .pairs import PairScores
from .cohorts import harmonize_pair_scores


def harmonize_robust_pairs(
    summary, canonical_variants, *, genome_build, trait_unit, phenotype_multiplier=1.0
):
    """F*=F D, y*=u y gives s*=u D s, H*=D H D, S*=u² D S D.

    The existing pair helper supplies checked allele/affine/order mapping at
    u=1. Its known-V phenotype transformation is deliberately not reused.
    """
    if (
        isinstance(phenotype_multiplier, bool)
        or not np.isfinite(phenotype_multiplier)
        or phenotype_multiplier <= 0
    ):
        raise ValueError("positive phenotype unit conversion required")
    m = dict(summary.metadata)
    if "pair_ids" not in m:
        raise ValueError("robust harmonization currently requires supplied exact pairs")
    proxy = PairScores(
        summary.scores,
        summary.information,
        tuple(tuple(p) for p in m["pair_ids"]),
        summary.trait_names,
        m,
    )
    aligned, transform = harmonize_pair_scores(
        proxy,
        canonical_variants,
        genome_build=genome_build,
        trait_unit=trait_unit,
        phenotype_multiplier=1.0,
    )
    columns = [summary.trait_names.index(t) for t in aligned.trait_names]
    u = phenotype_multiplier
    covariance = np.stack(
        [u * u * transform @ summary.score_covariance[i] @ transform.T for i in columns]
    )
    metadata = dict(
        aligned.metadata,
        pair_ids=aligned.pair_ids,
        method="independent_mean_HC3_harmonized_v1",
        trait_unit=trait_unit,
        effect_units="common_trait_unit_per_raw_counted_dosage_product",
        component_names=["supplied_pairs"],
        component_index=[0] * len(aligned.pair_ids),
    )
    metadata.pop(
        "burden_weights", None
    )  # signed burdens must be supplied on the new axis
    return (
        RobustScoreSummary(
            u * aligned.scores,
            aligned.information,
            covariance,
            tuple(":".join(p) for p in aligned.pair_ids),
            aligned.trait_names,
            metadata,
        ),
        u * transform,
    )


def combine_robust_scores(summaries, *, independent=False, cross_score_covariance=None):
    """Sum estimating equations; full cross-score covariance required for overlap.

    Point estimate equals pooled OLS with cohort-specific nuisance coefficients
    after harmonization. Sum of cohort HC3 meat is asymptotically valid under a
    common mean effect, not numerically equal to finite-sample pooled HC3.
    """
    summaries = tuple(summaries)
    if (
        len(summaries) < 2
        or type(independent) is not bool
        or (cross_score_covariance is None) == (not independent)
    ):
        raise ValueError(
            "declare independent cohorts or provide full cross-score covariance"
        )
    first = summaries[0]
    p = len(first.feature_names)
    t = len(first.trait_names)
    for s in summaries:
        if s.metadata.get(
            "sampling_model", "fixed_design_correct_mean"
        ) != first.metadata.get("sampling_model", "fixed_design_correct_mean"):
            raise ValueError(
                "cohort sampling models differ; conditional and population targets cannot be silently combined"
            )
        if s.feature_names != first.feature_names or s.trait_names != first.trait_names:
            raise ValueError("harmonize robust cohort axes first")
        for key in (
            "pair_ids",
            "variants",
            "genome_build",
            "trait_unit",
            "effect_units",
            "genotype_scale_contract",
        ):
            if not s.metadata.get(key) or s.metadata[key] != first.metadata[key]:
                raise ValueError("robust cohort hypotheses/alleles/units disagree")
    samples = [s.metadata.get("sample_hash") for s in summaries]
    if any(not v for v in samples) or len(set(samples)) != len(samples):
        raise ValueError("distinct cohort sample identities required")
    if independent:
        covariance = sum(s.score_covariance for s in summaries)
    else:
        full = np.asarray(cross_score_covariance, dtype=float)
        if (
            full.shape != (t, p * len(summaries), p * len(summaries))
            or not np.all(np.isfinite(full))
            or not np.allclose(full, full.transpose(0, 2, 1))
        ):
            raise ValueError(
                "cross-score covariance must have trait by cohort-feature by cohort-feature axes"
            )
        for i, s in enumerate(summaries):
            if not np.allclose(
                full[:, i * p : (i + 1) * p, i * p : (i + 1) * p],
                s.score_covariance,
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError(
                    "cross-score diagonal disagrees with HC3 cohort covariance"
                )
        if any(np.linalg.eigvalsh(a)[0] < -1e-10 * np.linalg.norm(a, 2) for a in full):
            raise ValueError("indefinite cross-score covariance")
        add = np.tile(np.eye(p), (1, len(summaries)))
        covariance = np.stack([add @ a @ add.T for a in full])
    metadata = dict(
        first.metadata,
        method="common_mean_robust_estimating_equations_v1",
        inference="asymptotic common signed mean effect; estimated cohort covariance; cohort-specific nuisance means",
        sample_hash=canonical_sha256(samples),
        cohort_sample_identities=samples,
        cohort_diagnostics=[dict(s.metadata) for s in summaries],
        independent=independent,
        finite_sample_covariance="sum of local HC3 meat; differs from pooled HC3",
        n_samples=sum(s.metadata["n_samples"] for s in summaries),
        sample_count_interpretation="sum of cohort counts, including repeated individuals when overlap declared",
    )
    metadata.pop("wild_bootstrap", None)
    return RobustScoreSummary(
        sum(s.scores for s in summaries),
        sum(s.information for s in summaries),
        covariance,
        first.feature_names,
        first.trait_names,
        metadata,
    )


def combine_robust_manifest(path, output, spec):
    path = Path(path)
    summaries = []
    transforms = []
    original_trait_axes = []
    for item in spec["cohorts"]:
        closed(item, ("summary", "phenotype_multiplier"), name="robust cohort")
        original = load_robust_scores(path.parent / item["summary"])
        original_trait_axes.append(original.trait_names)
        s, d = harmonize_robust_pairs(
            original,
            spec["canonical_variants"],
            genome_build=spec["genome_build"],
            trait_unit=spec["trait_unit"],
            phenotype_multiplier=item["phenotype_multiplier"],
        )
        summaries.append(s)
        transforms.append(d)
    covariance = None
    if "cross_score_covariance" in spec:
        if any(axis != summaries[0].trait_names for axis in original_trait_axes):
            raise ValueError(
                "overlap covariance requires every input trait axis already in canonical sorted order"
            )
        original = np.load(
            path.parent / spec["cross_score_covariance"], allow_pickle=False
        )
        d = block_diag(*transforms)
        if original.ndim != 3 or original.shape[1:] != d.shape:
            raise ValueError("invalid input cross-score covariance axes")
        covariance = np.stack([d @ v @ d.T for v in original])
    return write_robust_scores(
        combine_robust_scores(
            summaries,
            independent=spec.get("independent", False),
            cross_score_covariance=covariance,
        ),
        output,
    )
