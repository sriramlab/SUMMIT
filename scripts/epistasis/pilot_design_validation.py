"""Bounded calibration checks for an actual frozen real-phenotype design.

Conditional residual draws test a correct finite mean. IID draws of complete
(C,F,Y) rows test inference for the empirical population's known projection.
Neither experiment certifies a biological zero null in the original cohort.
"""
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from summit.context.spec import canonical_sha256
from summit.prediction.cli import _table, _aligned_table
from summit.epistasis.features import load_feature_reference
from summit.epistasis.robust import prepare_robust_scores
from scripts.epistasis.trans_report import reduce


def influence_audit(f, c, y, frame, *, seed, draws):
    """Known empirical-population influence variance and matched RNG replay."""
    r = f[:, 0] - c @ np.linalg.lstsq(c, f[:, 0], rcond=None)[0]
    design = np.column_stack([c, f])
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    influence = r * residual / (r @ r)
    rng = np.random.default_rng(seed)
    linear = np.array(
        [influence[rng.integers(len(y), size=len(y))].sum() for _ in range(draws)]
    )
    population = frame[frame.experiment == "empirical_population"].sort_values(
        "replicate"
    )
    if population.failed.any():
        return dict(
            unavailable="failed empirical fits; retained in original accounting"
        )
    error = (population.beta - population.truth).to_numpy()
    return dict(
        asymptotic_empirical_population_sd=float(np.linalg.norm(influence)),
        matched_linearized_sd=float(linear.std(ddof=1)),
        actual_refit_error_sd=float(error.std(ddof=1)),
        mean_hc3_se=float(population.se.mean()),
        linearization_remainder_rms=float(np.sqrt(np.mean((linear - error) ** 2))),
        variance_scope="exact first-order IID empirical-population influence variance; not exact finite-N refitted variance; no correction applied",
    )


def run(
    inputs,
    output,
    *,
    draws=2000,
    seed=507219,
    reference="prepared/learned.cohort-reference.npz",
):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    spec = json.loads((inputs / "confirm.json").read_text())
    with np.load(inputs / reference, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["manifest"]))["metadata"]
    ref = load_feature_reference(
        inputs / reference, compatibility_id=metadata["compatibility_id"]
    )
    samples = list(_table(inputs / spec["samples"]).index)
    by_token = {canonical_sha256(list(s)): s for s in samples}
    samples = [by_token[t] for t in metadata["cohort_sample_tokens"]]
    phen = spec["phenotypes"]
    y = _aligned_table(inputs / phen["file"], samples)[phen["columns"][0]].to_numpy(
        float
    )
    f = ref.features
    c = ref.fixed_effects
    n = len(f)
    if f.shape[1] != 1:
        raise ValueError("scalar pilot reference required")
    kwargs = dict(feature_names=("direction",), trait_names=("y",), metadata={})
    fitted = prepare_robust_scores(
        f, y, c, **kwargs, sampling_model="iid_population_projection"
    )
    truth = float(fitted.scores[0, 0] / fitted.information[0, 0])
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(draws):
        idx = rng.integers(n, size=n)
        try:
            s = prepare_robust_scores(
                f[idx],
                y[idx],
                c[idx],
                **kwargs,
                sampling_model="iid_population_projection",
            )
            b = float(s.scores[0, 0] / s.information[0, 0])
            se = float(np.sqrt(s.score_covariance[0, 0, 0]) / s.information[0, 0])
            records.append(
                dict(
                    experiment="empirical_population",
                    noise="observed",
                    replicate=rep,
                    beta=b,
                    se=se,
                    truth=truth,
                    p=float(2 * norm.sf(abs((b - truth) / se))),
                    failed=False,
                    outside_scope=";".join(s.metadata["outside_confirmation_design"]),
                )
            )
        except (ValueError, ArithmeticError) as error:
            records.append(
                dict(
                    experiment="empirical_population",
                    noise="observed",
                    replicate=rep,
                    failed=True,
                    p=np.nan,
                    error=str(error),
                )
            )
    sd = np.sqrt(0.5 + 0.5 * (f[:, 0] / np.std(f[:, 0])) ** 2)
    # Batch many residual phenotypes; no design or genotype regeneration here.
    for noise in ("gaussian", "t5"):
        for begin in range(0, draws, 64):
            count = min(64, draws - begin)
            z = (
                rng.normal(size=(n, count))
                if noise == "gaussian"
                else rng.standard_t(5, size=(n, count)) / np.sqrt(5 / 3)
            )
            try:
                s = prepare_robust_scores(
                    f,
                    sd[:, None] * z,
                    c,
                    feature_names=("direction",),
                    trait_names=tuple(map(str, range(count))),
                    metadata={},
                )
                b = s.scores[0] / s.information[0, 0]
                se = np.sqrt(s.score_covariance[:, 0, 0]) / s.information[0, 0]
                for j in range(count):
                    records.append(
                        dict(
                            experiment="correct_finite_mean",
                            noise=noise,
                            replicate=begin + j,
                            beta=b[j],
                            se=se[j],
                            truth=0.0,
                            p=2 * norm.sf(abs(b[j] / se[j])),
                            failed=False,
                            outside_scope=";".join(
                                s.metadata["outside_confirmation_design"]
                            ),
                        )
                    )
            except (ValueError, ArithmeticError) as error:
                for j in range(count):
                    records.append(
                        dict(
                            experiment="correct_finite_mean",
                            noise=noise,
                            replicate=begin + j,
                            failed=True,
                            p=np.nan,
                            error=str(error),
                        )
                    )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "replicates.csv", index=False)
    reduce(frame, ["experiment", "noise"]).to_csv(output / "summary.csv", index=False)
    (output / "influence_audit.json").write_text(
        json.dumps(influence_audit(f, c, y, frame, seed=seed, draws=draws), indent=2)
    )
    (output / "design.json").write_text(
        json.dumps(
            dict(
                inputs=str(inputs),
                seed=seed,
                draws=draws,
                n=n,
                phenotype=phen["columns"][0],
                empirical_projection_truth=truth,
                reference_identity=metadata["compatibility_id"],
                original_scope=fitted.metadata["outside_confirmation_design"],
                fixed="independent native training, frozen empirical donor rows and features",
                empirical_random="IID complete C,F,Y rows; every finite nuisance/HC3 fit repeated, including rare-cell changes",
                residual_random="heteroskedastic Gaussian/t5 errors; C,F fixed; finite true coefficient zero",
                interpretation="design-specific checks, not certification of the original biological null or the real cohort sampling mechanism",
                genotype_passes=0,
                seconds=time.perf_counter() - start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    run(args.inputs, args.out, draws=args.draws)


if __name__ == "__main__":
    main()
