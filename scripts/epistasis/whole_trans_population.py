"""Whole-marker experiments conditional on a completed native model.

Reuse actual cohort features. Sample target-chromosome and remaining-genome
blocks independently from the empirical donor distribution. This imposed
independence supplies the biological-null implication; it is not inferred from
ordinary LD in the original cohort. The intact_rows experiment instead preserves
the full empirical joint law, and fixed_rows regenerates only outcomes. Their
possibly nonzero additive projection truths are computed separately from the
biological zero null. No genotype reads occur per replicate.
"""
import argparse, json, time, resource
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from summit.prediction.genotype import (
    source_from_spec,
    StandardizedBlock,
    native_module,
)
from summit.prediction.artifacts import load_prediction_models
from summit.prediction.score import align_variants
from summit.prediction.cli import _rows
from summit.prediction.runtime import configure_prediction_threads
from summit.epistasis.features import load_feature_reference
from summit.epistasis.robust import prepare_robust_scores
from summit.epistasis.cli import _jsonable


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=630179)
    p.add_argument(
        "--sampling",
        choices=("independent_blocks", "intact_rows", "fixed_rows"),
        default="independent_blocks",
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    root = a.inputs.resolve()
    spec = json.loads((root / "confirm.json").read_text())
    path = root / "confirmed/learned.cohort-reference.npz"
    with np.load(path, allow_pickle=False) as z:
        identity = json.loads(str(z["manifest"]))["metadata"]["compatibility_id"]
    ref = load_feature_reference(path, compatibility_id=identity)
    direction = json.loads((root / "trained/direction.json").read_text())
    identity = direction.get("additive_model_identity", direction["model_identity"])
    models = load_prediction_models(root / "trained/models")
    matches = [model for model in models if model.identity == identity]
    if len(matches) != 1:
        raise ValueError("saved additive model identity is unavailable")
    model = matches[0]
    local = list(ref.metadata["definitions"]["local_variants"])
    target = spec["jobs"][0]["trans_target"]
    # The feature builder appends its own intercept even when no additional
    # variant main effects were requested. It is redundant with the original
    # intercept, not a score column.
    if not np.all(ref.fixed_effects[:, -1] == 1):
        raise ValueError("unexpected additional feature-builder nuisance columns")
    c = ref.fixed_effects[:, :-1]
    n = len(c)
    if c.shape[1] != 3 + 2 * len(local):
        raise ValueError("unexpected benchmark nuisance definition")
    target_x = c[:, 2 + local.index(target)]
    score = c[:, -1]
    pgs = c[:, 1]
    if not np.allclose(ref.features[:, 0], target_x * score, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("cohort feature algebra differs")
    rng = np.random.default_rng(a.seed)
    native = native_module()
    configure_prediction_threads(native, a.num_threads)
    mean_a = np.zeros(n)
    mean_b = np.zeros(n)
    pgs_a = np.zeros(n)
    pgs_check = np.zeros(n)
    with source_from_spec(spec["genotypes"], root) as source:
        target_chromosome = source.variants.chromosome[source.variants.ids.index(target)]
        rows = np.sort(_rows(source, root / spec["samples"]))
        mr, sr, flips = align_variants(model.variants, source.variants)
        if np.any(flips) or len(mr) != len(model.variants.ids):
            raise ValueError(
                "benchmark expects the authenticated training source allele axis"
            )
        effects = rng.normal(size=len(mr)) * np.sqrt(0.7 / len(mr))
        source.prepare(rows, 128, a.num_threads)
        standard = StandardizedBlock(native, a.num_threads)
        for begin in range(0, len(sr), 128):
            take = sr[begin : begin + 128]
            mi = mr[begin : begin + 128]
            x = standard.prepare(
                source.read(take),
                np.arange(n),
                np.arange(len(take)),
                model.scale.mean[mi],
                model.scale.inverse_scale[mi],
            )
            on_target = np.array(
                [source.variants.chromosome[int(i)] == target_chromosome for i in take]
            )
            weights = np.column_stack(
                [
                    effects[begin : begin + len(take)] * on_target,
                    effects[begin : begin + len(take)] * ~on_target,
                    model.weights[mi, 0] * on_target,
                    model.weights[mi, 0],
                ]
            )
            product = np.empty((n, 4), order="F")
            native.prediction_product(
                np.asfortranarray(x),
                np.asfortranarray(weights),
                product,
                False,
                a.num_threads,
            )
            mean_a += product[:, 0]
            mean_b += product[:, 1]
            pgs_a += product[:, 2]
            pgs_check += product[:, 3]
        marker_count = len(sr)
    decomposition_error = float(np.max(abs(pgs_check - pgs)))
    if decomposition_error > 1e-9 * max(1, float(np.max(abs(pgs)))):
        raise ArithmeticError(
            "streamed baseline decomposition disagrees with native scoring"
        )
    pgs_b = pgs - pgs_a
    mean_a += 0.7 * target_x + 0.6 * c[:, 2 + len(local) + local.index(target)]
    preparation_seconds = time.perf_counter() - start
    target_var = np.var(target_x)
    score_var = np.var(score)
    strengths = (0.0, 0.0005, 0.002)
    signal_variance = (
        target_var * score_var
        if a.sampling == "independent_blocks"
        else np.var(target_x * score)
    )
    betas = np.sqrt(np.array(strengths) / signal_variance)
    null_truth = 0.0
    if a.sampling != "independent_blocks":
        f0 = target_x * score
        r0 = f0 - c @ np.linalg.lstsq(c, f0, rcond=1e-11)[0]
        null_truth = float(r0 @ (mean_a + mean_b) / (r0 @ r0))
    records = []
    max_equivalence = 0.0
    # Independent stream separates causal-effect generation from phenotype draws.
    rng = np.random.default_rng(a.seed + 1)
    for rep in range(a.draws):
        ia = np.arange(n) if a.sampling == "fixed_rows" else rng.integers(n, size=n)
        ib = rng.integers(n, size=n) if a.sampling == "independent_blocks" else ia
        cx = c[ia].copy()
        cx[:, 1] = pgs_a[ia] + pgs_b[ib]
        cx[:, -1] = score[ib]
        f = target_x[ia] * score[ib]
        mean = mean_a[ia] + mean_b[ib]
        sd = np.sqrt(0.4 + 0.6 * target_x[ia] ** 2)
        ys = np.column_stack(
            [
                mean + sd * rng.normal(size=n),
                mean + sd * rng.standard_t(5, size=n) / np.sqrt(5 / 3),
            ]
        )
        try:
            summary = prepare_robust_scores(
                f[:, None],
                ys,
                cx,
                feature_names=("direction",),
                trait_names=("gaussian", "t5"),
                metadata={},
                sampling_model="fixed_design_correct_mean"
                if a.sampling == "fixed_rows"
                else "iid_population_projection",
            )
            estimates = summary.scores[0] / summary.information[0, 0]
            ses = np.sqrt(summary.score_covariance[:, 0, 0]) / summary.information[0, 0]
            if rep == 0:
                checked = prepare_robust_scores(
                    f[:, None],
                    ys[:, 0, None] + f[:, None] * betas,
                    cx,
                    feature_names=("direction",),
                    trait_names=tuple(map(str, strengths)),
                    metadata={},
                )
                max_equivalence = float(
                    np.max(
                        abs(
                            checked.scores[0] / checked.information[0, 0]
                            - (estimates[0] + betas)
                        )
                    )
                )
                if max_equivalence > 1e-10:
                    raise ArithmeticError("linear shift reuse failed")
            for j, noise in enumerate(("gaussian", "t5")):
                for strength, beta in zip(strengths, betas):
                    records.append(
                        dict(
                            replicate=rep,
                            noise=noise,
                            signal_expected_variance=strength,
                            signal_realized_variance=float(beta * beta * np.var(f)),
                            truth=float(beta + null_truth),
                            biological_interaction_coefficient=float(beta),
                            beta=float(estimates[j] + beta),
                            se=float(ses[j]),
                            p=float(2 * norm.sf(abs((estimates[j] + beta) / ses[j]))),
                            coverage=float(
                                abs(estimates[j] - null_truth) <= 1.95996398454 * ses[j]
                            ),
                            failed=False,
                            outside_scope=";".join(
                                summary.metadata["outside_confirmation_design"]
                            ),
                        )
                    )
        except (ValueError, ArithmeticError) as error:
            for noise in ("gaussian", "t5"):
                for strength in strengths:
                    records.append(
                        dict(
                            replicate=rep,
                            noise=noise,
                            signal_expected_variance=strength,
                            failed=True,
                            p=np.nan,
                            error=str(error),
                        )
                    )
    pd.DataFrame(records).to_csv(a.out / "replicates.csv", index=False)
    (a.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    inputs=str(root),
                    n=n,
                    markers=marker_count,
                    seed=a.seed,
                    draws=a.draws,
                    sampling=a.sampling,
                    additive_projection_truth=null_truth,
                    genotype_passes=1,
                    genotype_passes_per_replicate=0,
                    reference_identity=identity,
                    model_identity=model.identity,
                    native_score_decomposition_error=decomposition_error,
                    max_shift_equivalence_error=max_equivalence,
                    fixed="frozen native training, causal effects across the full marker axis, real donor block distributions, feature and nuisance definitions",
                    regenerated=(
                        "both genotype blocks independently sampled per individual"
                        if a.sampling == "independent_blocks"
                        else "complete empirical donor rows"
                        if a.sampling == "intact_rows"
                        else "fixed complete rows; outcomes only"
                    )
                    + "; Gaussian/t5 residuals; every nuisance and HC3 fit",
                    caveat=(
                        "population block independence imposed; not original-cohort biological-null validation"
                        if a.sampling == "independent_blocks"
                        else "empirical joint genotype law only; nonzero additive projection is a biological-null failure even with correct projection coverage"
                        if a.sampling == "intact_rows"
                        else "fixed-panel omitted-mean diagnostic; HC3 need not estimate conditional error variance under misspecification"
                    ),
                    preparation_seconds=preparation_seconds,
                    seconds=time.perf_counter() - start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
