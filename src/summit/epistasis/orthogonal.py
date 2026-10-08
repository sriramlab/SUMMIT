"""Independent-split restricted additive projection (experimental).

Project BOTH outcomes and interaction features with SUMMIT's native prediction
solver. This is population restricted-space partialling out, not unrestricted
conditioning on all genotypes, nor a guarantee of negligible ridge bias.
"""
from pathlib import Path
import numpy as np
from dataclasses import replace
from summit.prediction.spec import TraitTraining, CandidatePrior, SolverSpec
from summit.prediction.batch import plan_prediction
from summit.prediction.api import fit_prediction
from summit.prediction.score import score_prediction, ScoreInput
from summit.prediction.cli import _rows
from summit.context.spec import canonical_sha256
from .robust import prepare_robust_scores


def prepare_orthogonal(
    study, reference, y, *, settings, root, output, trait_names, metadata
):
    training = _rows(study.source, Path(root) / settings["training_samples"])
    if not set(training) <= set(study.rows) or len(training) < 20:
        raise ValueError(
            "nuisance training samples must be a declared subset of analysis samples, with at least 20 rows"
        )
    train = np.isin(study.rows, training)
    test = ~train
    if test.sum() < 20:
        raise ValueError(
            "independent confirmation needs at least 20 samples outside nuisance training"
        )
    prior_variance = settings.get("ridge_variance", 1.0)
    residual = settings.get("residual_variance", 1.0)
    if (
        not np.isfinite(prior_variance)
        or not np.isfinite(residual)
        or min(prior_variance, residual) <= 0
    ):
        raise ValueError("prespecified ridge and residual variances must be positive")
    f, c = reference.features, reference.fixed_effects
    responses = np.column_stack([y, f])
    if responses.shape[1] > 32:
        raise ValueError(
            "independent-split nuisance preparation currently supports at most 32 outcome/feature RHSs"
        )
    context = dict(kind="restricted_additive_projection", names=["baseline"])
    fixed = dict(
        kind="declared_finite_mean", names=[f"c{i}" for i in range(c.shape[1])]
    )
    candidate = CandidatePrior(
        "prespecified",
        np.array([[prior_variance]]),
        np.full(train.sum(), residual),
        dict(
            method="restricted_additive_ridge",
            variance=prior_variance,
            residual=residual,
        ),
    )
    from summit.prediction.genotype import estimate_scale

    nuisance_scale = estimate_scale(
        study.source,
        study.rows[train],
        np.arange(study.m),
        threads=study.threads,
        block_size=study.block_size,
        memory_bytes=study.memory_bytes
        - sum(
            a.nbytes
            for a in (
                responses,
                f,
                c,
                study.fixed,
                study.u,
                study.modifiers,
                study.weights,
            )
        ),
    )
    # Admit caller-owned arrays before constructing per-RHS training designs.
    resident = sum(
        a.nbytes
        for a in (responses, f, c, study.fixed, study.u, study.modifiers, study.weights)
    )
    memory = study.memory_bytes - resident
    training_inputs = (
        8 * responses.shape[1] * (int(train.sum()) * (c.shape[1] + 5) + 3 * study.m)
    )
    if memory <= training_inputs + 256 * 2**20:
        raise MemoryError("caller-owned designs leave insufficient nuisance-fit memory")
    traits = [
        TraitTraining(
            f"rhs{j}",
            study.rows[train],
            np.arange(study.m),
            responses[train, j],
            np.ones((train.sum(), 1)),
            c[train],
            nuisance_scale,
            (candidate,),
            context,
            fixed,
            dict(units="common declared feature/outcome units", transform="raw"),
        )
        for j in range(responses.shape[1])
    ]
    plan = plan_prediction(
        traits,
        study.source,
        storage=settings.get("storage", "stream"),
        block_size=study.block_size,
        rhs_columns=min(32, len(traits)),
        threads=study.threads,
        memory_bytes=memory,
    )
    Path(output).mkdir(parents=True, exist_ok=False)
    models = fit_prediction(
        traits,
        study.source,
        output=Path(output) / "models",
        plan=plan,
        solver=SolverSpec(**settings.get("solver", {})),
        checkpoint=Path(output) / "solver.npz",
    )
    trait_ids = [t.id for t in traits]
    # The training designs are no longer needed while scoring the held-out set.
    del traits
    scoring_inputs = 8 * len(trait_ids) * int(test.sum()) * (c.shape[1] + 2)
    if memory <= scoring_inputs + 256 * 2**20:
        raise MemoryError(
            "held-out nuisance scoring inputs exceed remaining memory budget"
        )
    inputs = {
        name: ScoreInput(
            study.rows[test], np.ones((test.sum(), 1)), c[test], context, fixed
        )
        for name in trait_ids
    }
    result = score_prediction(
        models,
        study.source,
        inputs,
        block_size=study.block_size,
        rhs_columns=min(32, len(trait_ids)),
        threads=study.threads,
        memory_bytes=memory
        - sum(v.phi.nbytes + v.fixed.nbytes for v in inputs.values()),
    )
    predicted = np.column_stack([result.prediction[m.key] for m in models])
    residualized = responses[test] - predicted
    summary = prepare_robust_scores(
        residualized[:, y.shape[1] :],
        residualized[:, : y.shape[1]],
        c[test],
        feature_names=reference.metadata["feature_names"],
        trait_names=trait_names,
        metadata=metadata,
        nn=study._nn,
        tn=study._tn,
    )
    info = dict(
        summary.metadata,
        method="independent_split_restricted_additive_HC3_v1_experimental",
        inference="asymptotic restricted-population projection; independent nuisance training; requires vanishing approximation bias and product-rate nuisance error",
        estimand="conditional on frozen direction: coefficient after population projection onto declared additive genotype and finite covariate space",
        support="experimental: a ridge fit does not certify the required nuisance-error rates",
        nuisance_training_n=int(train.sum()),
        confirmation_n=int(test.sum()),
        nuisance_training_identity=canonical_sha256(
            [study.source.samples[int(i)] for i in study.rows[train]]
        ),
        common_scale_identity=study.scale.identity,
        nuisance_fit_identity=plan.fit_identity,
        nuisance_scoring_report=result.report,
    )
    return replace(summary, metadata=info)
