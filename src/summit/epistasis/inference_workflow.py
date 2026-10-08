"""Complete manifest preparation and summary-only inference for score methods."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import beta
from summit.context.spec import array_sha256, canonical_sha256
from summit.prediction._validation import closed
from .features import (
    prepare_feature_reference,
    feature_compatibility_id,
    load_feature_reference,
    write_feature_reference,
)
from .score import (
    prepare_linear_scores,
    write_linear_scores,
    linear_score_tests,
    LinearScoreSummary,
    refit_bootstrap,
)
from .null import GaussianNullReference, GeneralGaussianNullReference
from .summary import _publish_bundle


def prepare_score_job(
    study,
    job,
    annotations,
    y,
    *,
    trait_names,
    output,
    root,
    seed,
    shared_sources=None,
    trait_unit=None,
    resume=False,
    strata=None,
):
    settings = job["inference"]
    closed(
        settings,
        ("method",),
        (
            "main_effects",
            "dominance",
            "feature_sketch_dimensions",
            "bootstrap_draws",
            "reference",
            "save_reference",
            "wild_draws",
            "training_samples",
            "ridge_variance",
            "residual_variance",
            "storage",
            "solver",
            "sampling_model",
        ),
        name="score inference",
    )
    method = settings["method"]
    if "sampling_model" in settings and method != "robust_mean":
        raise ValueError("sampling_model requires robust_mean inference")
    if method not in (
        "linear_exact",
        "reml_bootstrap",
        "robust_mean",
        "orthogonal_mean",
    ):
        raise ValueError(
            "score method must be linear_exact, reml_bootstrap, robust_mean or orthogonal_mean"
        )
    if method == "orthogonal_mean" and "training_samples" not in settings:
        raise ValueError(
            "orthogonal_mean requires independent nuisance training_samples"
        )
    if method != "orthogonal_mean" and any(
        k in settings
        for k in (
            "training_samples",
            "ridge_variance",
            "residual_variance",
            "storage",
            "solver",
        )
    ):
        raise ValueError("nuisance training settings require orthogonal_mean")
    if "wild_draws" in settings and method != "robust_mean":
        raise ValueError("wild draws require robust_mean inference")
    if "nuisance" in job:
        raise ValueError(
            "estimated-null score jobs do not accept externally supplied nuisance coefficients"
        )
    options = dict(
        sketch_dimensions=settings.get("feature_sketch_dimensions"),
        seed=seed,
        main_effects=settings.get(
            "main_effects",
            "tested_variants" if method != "reml_bootstrap" else "declared",
        ),
        dominance=settings.get(
            "dominance",
            "tested_variants"
            if method == "robust_mean" and "pairs" in job
            else "declared",
        ),
        additive_null=method == "reml_bootstrap",
        strata=strata,
    )
    identity = feature_compatibility_id(study, job, annotations, **options)
    preparation_identity = canonical_sha256(
        dict(
            reference=identity,
            robust_algorithm="span_HC3_v2" if method == "robust_mean" else None,
            settings=settings,
            phenotype=array_sha256(np.asarray(y)),
            trait_names=list(trait_names),
            trait_unit=trait_unit,
        )
    )
    saved_reference = Path(output) / (job["id"] + ".cohort-reference.npz")
    if resume:
        if method != "robust_mean":
            raise ValueError("resume currently supports robust_mean jobs only")
        from .robust import load_robust_scores

        completed = Path(output) / (job["id"] + ".robust-score.npz")
        if completed.exists():
            previous = load_robust_scores(completed)
            if previous.metadata.get("preparation_identity") != preparation_identity:
                raise ValueError(
                    "completed summary does not match requested inputs; use a new output path"
                )
            if settings.get("save_reference"):
                load_feature_reference(saved_reference, compatibility_id=identity)
            return completed, dict(completed_summary_reused=True)
    if settings.get("reference"):
        reference = load_feature_reference(
            Path(root) / settings["reference"], compatibility_id=identity
        )
        passes = {"feature_reference_reused": True}
    elif resume and settings.get("save_reference") and saved_reference.exists():
        reference = load_feature_reference(saved_reference, compatibility_id=identity)
        passes = {"feature_reference_reused": True}
    else:
        reference = prepare_feature_reference(
            study, job, annotations, shared_sources=shared_sources, **options
        )
        passes = dict(reference.metadata["genotype_passes"])
    if settings.get("save_reference") and not (resume and saved_reference.exists()):
        write_feature_reference(
            reference, Path(output) / (job["id"] + ".cohort-reference.npz")
        )
    if trait_unit is not None and (not isinstance(trait_unit, str) or not trait_unit):
        raise ValueError("trait unit must be a nonempty scientific unit label")
    metadata = dict(
        reference.metadata,
        phenotype_units="raw",
        trait_unit=trait_unit,
        trait_names=list(trait_names),
        phenotypes_hash=array_sha256(np.asarray(y)),
        scientific_null="no effects in supplied interaction feature span",
        coefficient_contrasts=job.get("coefficient_contrasts"),
        preparation_identity=preparation_identity,
    )
    # Participant alignment tokens belong to the restricted cohort reference,
    # not to the portable trait-score artifact.
    metadata.pop("cohort_sample_tokens", None)
    if method in ("robust_mean", "orthogonal_mean"):
        from .robust import prepare_robust_scores, write_robust_scores

        if "pairs" in job:
            variants = study.source.variants
            lookup = {v: i for i, v in enumerate(variants.ids)}
            involved = sorted({v for pair in job["pairs"] for v in pair})
            metadata.update(
                pair_ids=[sorted(pair) for pair in job["pairs"]],
                fixed_main_effect_variants=involved,
                genome_build=variants.genome_build,
                inverse_scales={
                    v: float(study.scale.inverse_scale[lookup[v]]) for v in involved
                },
            )
        if (
            "bootstrap_draws" in settings
            or study.h > 1
            or job.get("residual_model", "iid") != "iid"
        ):
            raise ValueError(
                "robust mean inference estimates HC3 covariance; no Gaussian residual surfaces or refit-bootstrap options"
            )
        p, k, t = (
            reference.features.shape[1],
            reference.fixed_effects.shape[1],
            y.shape[1],
        )
        workspace = 8 * (
            study.n
            * (6 * p + 4 * k + 4 * t + (192 if settings.get("wild_draws") else 0))
            + 4 * t * p * p
        )
        cache_bytes = (
            0
            if shared_sources is None
            else sum(a.nbytes for a in shared_sources["sources"].values())
        )
        resident = sum(
            a.nbytes
            for a in (
                study.modifiers,
                study.fixed,
                study.u,
                study.weights,
                study.residual_basis,
            )
        )
        if (
            workspace
            + cache_bytes
            + resident
            + 32 * study.n * min(study.block_size, study.m)
            + 64 * 2**20
            > study.memory_bytes
        ):
            raise MemoryError("robust mean workspace exceeds memory budget")
        study.nn.begin_execution()
        study.tn.begin_execution()
        if method == "orthogonal_mean":
            from .orthogonal import prepare_orthogonal

            summary = prepare_orthogonal(
                study,
                reference,
                y,
                settings=settings,
                root=root,
                output=Path(output) / (job["id"] + ".nuisance"),
                trait_names=trait_names,
                metadata=dict(
                    metadata,
                    component_index=reference.component_index.tolist(),
                    burden_weights=job.get("burden_weights"),
                ),
            )
        else:
            summary = prepare_robust_scores(
                reference.features,
                y,
                reference.fixed_effects,
                feature_names=reference.metadata["feature_names"],
                trait_names=trait_names,
                metadata=dict(
                    metadata,
                    component_index=reference.component_index.tolist(),
                    burden_weights=job.get("burden_weights"),
                ),
                nn=study._nn,
                tn=study._tn,
                wild_draws=settings.get("wild_draws", 0),
                seed=seed,
                sampling_model=settings.get(
                    "sampling_model", "fixed_design_correct_mean"
                ),
            )
        study.nn.finish_execution()
        path = write_robust_scores(
            summary, Path(output) / (job["id"] + ".robust-score.npz")
        )
        passes.update(
            mean_fits_per_trait=1 + settings.get("wild_draws", 0),
            resampling_genotype_passes=0,
        )
        return path, passes
    if method == "linear_exact":
        if (
            "bootstrap_draws" in settings
            or study.h > 1
            or job.get("residual_model", "iid") != "iid"
        ):
            raise ValueError(
                "linear exact inference has no bootstrap or fitted residual surfaces"
            )
        study.nn.begin_execution()
        study.tn.begin_execution()
        summary = prepare_linear_scores(
            reference.features,
            y,
            reference.fixed_effects,
            feature_names=reference.metadata["feature_names"],
            trait_names=trait_names,
            metadata=dict(
                metadata,
                component_index=reference.component_index.tolist(),
                burden_weights=job.get("burden_weights"),
            ),
            nn=study._nn,
            tn=study._tn,
        )
        study.nn.finish_execution()
        path = write_linear_scores(
            summary, Path(output) / (job["id"] + ".linear-score.npz")
        )
        return path, passes
    g = reference.additive_kernel
    residual_basis = study.residual_basis
    residual_names = list(study.residual_names)
    if job.get("residual_model") == "feature_diagonal":
        diagonal = np.column_stack(
            [
                np.sum(
                    reference.features[:, reference.component_index == j] ** 2, axis=1
                )
                for j in range(len(reference.metadata["component_names"]))
            ]
        )
        residual_basis = np.column_stack([diagonal, np.ones(study.n)])
        residual_names = [
            "residual_feature_diagonal:" + n
            for n in reference.metadata["component_names"]
        ] + ["residual"]
    component_names = reference.metadata["component_names"]
    draws = settings.get("bootstrap_draws", 199)
    if type(draws) is not int or not 19 <= draws <= 9999:
        raise ValueError("bootstrap draws must be 19..9999")
    score_count = len(component_names) + (
        reference.features.shape[1] + int("burden_weights" in job)
        if "pairs" in job
        else 0
    )
    cache_bytes = (
        0
        if shared_sources is None
        else sum(a.nbytes for a in shared_sources["sources"].values())
    )
    nuisance_count = (g.shape[0] if g.ndim == 3 else 1) + residual_basis.shape[1]
    moment_count = nuisance_count + len(component_names)
    if (
        8
        * (
            # Include constructor temporaries, rotated/moment kernel copies
            # and retained trait-specific fitted covariances before allocating.
            (3 * score_count + 5 * nuisance_count + 16 + 2 * y.shape[1]) * study.n**2
            + 3 * study.n * draws
            + 4 * reference.features.size
            + y.shape[1] * (moment_count**3 + score_count * draws)
        )
        + cache_bytes
        + 64 * 2**20
        > study.memory_bytes
    ):
        raise MemoryError(
            "spectral score and bootstrap workspace exceeds memory budget"
        )
    if g.ndim == 3 or residual_basis.shape[1] > 1:
        kernels = list(g if g.ndim == 3 else g[None]) + [
            np.diag(e) for e in residual_basis.T
        ]
        names = list(reference.metadata["additive_kernel_names"]) + residual_names
        null = GeneralGaussianNullReference(
            kernels, reference.fixed_effects, names=names, identity=identity
        )
    else:
        null = GaussianNullReference(g, reference.fixed_effects, identity=identity)
    rotated = null.rotation.T @ reference.features
    kernels = [
        rotated[:, reference.component_index == j]
        @ rotated[:, reference.component_index == j].T
        for j in range(len(component_names))
    ]
    test_layout = None
    if "pairs" in job:
        if rotated.shape[1] > 64:
            raise ValueError(
                "refitted sparse pair bootstrap is bounded at 64 supplied pairs; linear exact panels allow 512"
            )
        test_layout = dict(
            kernel=0, pair_begin=len(kernels), pair_count=rotated.shape[1]
        )
        support = np.sum(rotated**2, axis=0) > 1e-12 * max(
            np.sum(rotated**2), np.finfo(float).tiny
        )
        test_layout["identifiable_pairs"] = np.flatnonzero(support).tolist()
        kernels.extend(
            np.outer(rotated[:, j], rotated[:, j]) for j in np.flatnonzero(support)
        )
        component_names = list(component_names) + [
            "pair:" + reference.metadata["feature_names"][j]
            for j in np.flatnonzero(support)
        ]
        if "burden_weights" in job:
            w = np.asarray(job["burden_weights"], dtype=float)
            if (
                w.shape != (rotated.shape[1],)
                or not np.all(np.isfinite(w))
                or np.sum((rotated @ w) ** 2)
                <= 1e-12
                * max(np.sum(rotated**2) * np.sum(w * w), np.finfo(float).tiny)
            ):
                raise ValueError("burden must be prespecified, finite and identifiable")
            test_layout["burden"] = len(kernels)
            burden = rotated @ w
            kernels.append(np.outer(burden, burden))
            component_names.append("burden")
    fits = [
        refit_bootstrap(null, kernels, y[:, i], draws=draws, seed=seed + i)
        for i in range(y.shape[1])
    ]
    arrays = dict(
        statistics=np.stack([f["statistics"] for f in fits]),
        bootstrap_statistics=np.stack([f["bootstrap_statistics"] for f in fits]),
    )
    # Preserve unconstrained estimation as a separate calculation on precisely
    # these kernels. Its FAME covariance is a named comparator, not the
    # uncertainty calculation used by the refitted score test.
    from .oracle import dense_summary

    additive_count = len(reference.metadata["additive_kernel_names"])
    nuisance = (
        null.kernels * null.scales[:, None, None]
        if isinstance(null, GeneralGaussianNullReference)
        else np.stack([np.diag(null.eigenvalues), np.eye(null.rank)])
    )
    epi_names = list(reference.metadata["component_names"])
    moment_kernels = np.concatenate(
        [
            nuisance[:additive_count],
            np.stack(kernels[: len(epi_names)]),
            nuisance[additive_count:],
        ]
    )
    moment_names = (
        tuple(reference.metadata["additive_kernel_names"])
        + tuple(epi_names)
        + tuple(residual_names)
    )
    moments = dense_summary(
        moment_kernels,
        null.transform(y),
        component_names=moment_names,
        trait_names=trait_names,
        residual_rank=null.rank,
        metadata=dict(
            method="unconstrained_moments_on_score_model_kernels",
            genetic_count=additive_count + len(epi_names),
            kernel_target=reference.metadata["kernel_target"],
        ),
    )
    arrays.update(
        {
            "moment_" + key: getattr(moments, key)
            for key in ("matrix", "rhs", "traces", "cubic")
        }
    )
    if "pairs" in job:
        scores = []
        information = []
        for i, result in enumerate(fits):
            reduced = null.transform(y[:, i])
            fitted = result["null_fit"]
            if "variance" in fitted:
                solved = rotated / fitted["variance"][:, None]
                solved_y = reduced / fitted["variance"]
            else:
                solved = np.linalg.solve(fitted["covariance"], rotated)
                solved_y = np.linalg.solve(fitted["covariance"], reduced)
            scores.append(rotated.T @ solved_y)
            information.append(rotated.T @ solved)
        arrays.update(
            signed_scores=np.stack(scores), score_information=np.stack(information)
        )
    fits_metadata = []
    for result in fits:
        record = {
            k: v
            for k, v in result.items()
            if k not in ("statistics", "bootstrap_statistics")
        }
        record["null_fit"] = {
            k: v
            for k, v in result["null_fit"].items()
            if k not in ("variance", "covariance")
        }
        fits_metadata.append(record)
    manifest = dict(
        kind="summit.epistasis.refit_bootstrap_score",
        schema_version=1,
        trait_names=list(trait_names),
        component_names=list(component_names),
        metadata=metadata,
        moment_projection=dict(
            component_names=moment_names,
            n_samples=study.n,
            residual_rank=null.rank,
            metadata=dict(moments.metadata),
        ),
        fits=fits_metadata,
        test_layout=test_layout,
        digests={k: array_sha256(v) for k, v in arrays.items()},
    )
    from .cli import _jsonable

    path = _publish_bundle(
        Path(output) / (job["id"] + ".bootstrap-score.npz"), _jsonable(manifest), arrays
    )
    passes.update(
        null_covariance_fits_per_trait=draws + 1,
        bootstrap_genotype_passes=0,
        bootstrap_explanation="fixed genotype reference reused; Gaussian reduced phenotypes regenerated; every null refit",
    )
    return path, passes


def fit_linear_artifact(summary):
    results = []
    component = np.asarray(summary.metadata["component_index"])
    names = summary.metadata["component_names"]
    for trait in range(len(summary.trait_names)):
        tests = []
        for j, name in enumerate(names):
            keep = np.flatnonzero(component == j)
            sub = LinearScoreSummary(
                summary.scores[keep],
                summary.information[np.ix_(keep, keep)],
                summary.residual_ss,
                summary.residual_rank,
                tuple(summary.feature_names[i] for i in keep),
                summary.trait_names,
                dict(summary.metadata),
            )
            burden = summary.metadata.get("burden_weights")
            if burden is not None and len(names) != 1:
                raise ValueError("a burden vector requires one declared feature panel")
            test = linear_score_tests(sub, trait=trait, burden=burden)
            test.update(
                component=name,
                component_family_bonferroni_p=min(
                    1.0, len(names) * (test["kernel_p"] + test["kernel_p_error"])
                ),
            )
            tests.append(test)
        results.append(
            dict(
                trait=summary.trait_names[trait],
                tests=tests,
                trait_unit=summary.metadata.get("trait_unit"),
                component_interpretation="marginal component tests under the shared null; not conditional variance-component attribution",
            )
        )
        if len(names) > 1:
            results[-1]["global_combined_kernel_test"] = linear_score_tests(
                summary, trait=trait
            )
            results[-1][
                "global_hypothesis"
            ] = "all supplied interaction components zero; sum of declared normalized kernels"
    return results


def fit_bootstrap_artifact(path):
    def mc_interval(hits, draws, alpha=0.05):
        # Interval for the fitted-null exceedance probability, separately from
        # the plus-one Monte Carlo test value. No zero-P extrapolation.
        return [
            0.0 if hits == 0 else float(beta.ppf(alpha / 2, hits, draws - hits + 1)),
            1.0
            if hits == draws
            else float(beta.ppf(1 - alpha / 2, hits + 1, draws - hits)),
        ]

    with np.load(path, allow_pickle=False) as a:
        m = json.loads(str(a["manifest"]))
        if (
            m["kind"] != "summit.epistasis.refit_bootstrap_score"
            or m["schema_version"] != 1
        ):
            raise ValueError("unsupported bootstrap artifact")
        arrays = {k: a[k] for k in a.files if k != "manifest"}
        required = {"statistics", "bootstrap_statistics"}
        if "moment_projection" in m:
            required |= {"moment_" + k for k in ("matrix", "rhs", "traces", "cubic")}
        if m.get("test_layout") is not None:
            required |= {"signed_scores", "score_information"}
        if set(arrays) != required or m["digests"] != {
            k: array_sha256(v) for k, v in arrays.items()
        }:
            raise ValueError("invalid bootstrap artifact fields or digests")
        observed, boot = arrays["statistics"], arrays["bootstrap_statistics"]
        if (
            boot.ndim != 3
            or observed.shape != (boot.shape[0], boot.shape[2])
            or not np.all(np.isfinite(boot))
            or not np.all(np.isfinite(observed))
            or observed.shape != (len(m["trait_names"]), len(m["component_names"]))
        ):
            raise ValueError("bootstrap axes or values disagree")
        results = []
        b = boot.shape[1]
        if not 19 <= b <= 9999 or len(m["fits"]) != len(observed):
            raise ValueError("bootstrap count or trait diagnostics disagree")
        layout = m.get("test_layout")
        if layout is not None:
            p = layout["pair_count"]
            s, h = arrays["signed_scores"], arrays["score_information"]
            if (
                s.shape != (len(observed), p)
                or h.shape != (len(observed), p, p)
                or not np.all(np.isfinite(s))
                or not np.all(np.isfinite(h))
                or not np.allclose(h, h.transpose(0, 2, 1), atol=1e-10)
            ):
                raise ValueError("invalid signed bootstrap score axes or information")
        for i, name in enumerate(m["trait_names"]):
            exceed = (boot[i] >= observed[i]).sum(axis=0)
            global_exceed = int(np.sum(boot[i].max(axis=1) >= observed[i].max()))
            results.append(
                dict(
                    trait=name,
                    trait_unit=m["metadata"].get("trait_unit"),
                    component_names=m["component_names"],
                    statistics=observed[i],
                    component_p=(exceed + 1) / (b + 1),
                    global_max_score_p=(global_exceed + 1) / (b + 1),
                    global_max_monte_carlo_interval=mc_interval(global_exceed, b),
                    minimum_p=1 / (b + 1),
                    null_fit=m["fits"][i]["null_fit"],
                    monte_carlo_interval=m["fits"][i]["monte_carlo_interval"],
                    efficient_information_fraction=m["fits"][i][
                        "efficient_information_fraction"
                    ],
                    inference=m["fits"][i]["calibration"],
                    hypothesis="all supplied interaction components zero; marginal score diagnostics are not conditional component tests",
                )
            )
            if "moment_projection" in m:
                from .summary import EpistasisSummary, fit_epistasis
                from summit.context.fit import ContextRankError

                moments = EpistasisSummary(
                    **{
                        k: arrays["moment_" + k]
                        for k in ("matrix", "rhs", "traces", "cubic")
                    },
                    trait_names=m["trait_names"],
                    **m["moment_projection"],
                )
                try:
                    moment_fit = fit_epistasis(moments, i)
                except ContextRankError as error:
                    moment_fit = dict(
                        status="unidentifiable_moment_components", error=str(error)
                    )
                results[-1]["unconstrained_moment_comparator"] = moment_fit
            layout = m.get("test_layout")
            if layout is not None:
                start = layout["pair_begin"]
                stop = start + len(layout["identifiable_pairs"])
                sparse_hits = int(
                    np.sum(
                        boot[i, :, start:stop].max(axis=1)
                        >= observed[i, start:stop].max()
                    )
                )
                sparse = (1 + sparse_hits) / (b + 1)
                tests = dict(
                    kernel_p=float((exceed[0] + 1) / (b + 1)),
                    sparse_max_score_p=float(sparse),
                    kernel_monte_carlo_interval=mc_interval(int(exceed[0]), b),
                    sparse_monte_carlo_interval=mc_interval(sparse_hits, b),
                    identifiable_pairs=layout["identifiable_pairs"],
                    signed_scores=arrays["signed_scores"][i],
                    fitted_score_information=arrays["score_information"][i],
                    signed_information_inference="estimated covariance plug-in; bootstrap calibrates the declared global tests",
                )
                if "burden" in layout:
                    intervals = np.array(
                        [
                            mc_interval(int(hits), b, alpha=0.05 / 3)
                            for hits in (
                                exceed[0],
                                sparse_hits,
                                exceed[layout["burden"]],
                            )
                        ]
                    )
                    bp = float((exceed[layout["burden"]] + 1) / (b + 1))
                    tests.update(
                        burden_p=bp,
                        burden_monte_carlo_interval=mc_interval(
                            int(exceed[layout["burden"]]), b
                        ),
                        adaptive_monte_carlo_interval=np.minimum(
                            1.0, 3 * intervals.min(axis=0)
                        ),
                        adaptive_bonferroni_p=min(
                            1.0, 3 * min(tests["kernel_p"], sparse, bp)
                        ),
                    )
                results[-1]["pair_tests"] = tests
        return results
