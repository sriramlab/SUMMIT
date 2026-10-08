"""Prespecified increasing-N fixed-design tails and random-design structure stress."""
import argparse, json, time, resource
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis as basis
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.trans import stratified_fixed_effects
from summit.epistasis.cli import _jsonable


def interval(k, n):
    return [
        0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1)),
        1 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k)),
    ]


def tail_panel(args):
    rng = np.random.default_rng(args.seed)
    records = []
    validation = []
    for n in (2048, 8192, 32768):
        for design, af in [("common", (0.3, 0.35)), ("sparse", (0.01, 0.03))]:
            raw = rng.binomial(2, af, size=(n, 2)).astype(float)
            x = (raw - 2 * np.array(af)) / np.sqrt(
                2 * np.array(af) * (1 - np.array(af))
            )
            c = np.column_stack([np.ones(n), x, raw == 1])
            f = (x[:, 0] * x[:, 1])[:, None]
            u = basis(c)
            r = f - u @ (u.T @ f)
            h = float((r.T @ r)[0, 0])
            a = r[:, 0] / h
            leverage = np.sum(u * u, axis=1) + r[:, 0] ** 2 / h
            if leverage.max() >= 1 - 1e-8:
                records.append(
                    dict(
                        n=n,
                        design=design,
                        failed=True,
                        reason="essential unit leverage",
                        scheduled=args.draws,
                    )
                )
                continue
            denom = 1 - leverage
            sd = np.sqrt(0.4 + 0.6 * x[:, 0] ** 2)
            for noise in ("hetero_gaussian", "hetero_t5"):
                hits = {}
                estimates = []
                ses = []
                first = []
                for start in range(0, args.draws, 128):
                    b = min(128, args.draws - start)
                    z = (
                        rng.normal(size=(n, b))
                        if noise == "hetero_gaussian"
                        else rng.standard_t(5, size=(n, b)) / np.sqrt(5 / 3)
                    )
                    y = sd[:, None] * z
                    py = y - u @ (u.T @ y)
                    coef = a @ y
                    residual = py - r @ coef[None, :]
                    se = np.sqrt(
                        np.sum((a[:, None] * residual / denom[:, None]) ** 2, axis=0)
                    )
                    if not np.isfinite(se).all() or np.any(se <= 0):
                        raise ArithmeticError("undefined validation fit")
                    estimates.extend(coef)
                    ses.extend(se)
                    if start == 0:
                        public = prepare_robust_scores(
                            f,
                            y[:, : args.wild_replicates],
                            c,
                            feature_names=("f",),
                            trait_names=tuple(map(str, range(args.wild_replicates))),
                            metadata={},
                        )
                        discrepancy = max(
                            np.max(
                                abs(public.scores[0] / h - coef[: args.wild_replicates])
                            ),
                            np.max(
                                abs(
                                    np.sqrt(public.score_covariance[:, 0, 0]) / h
                                    - se[: args.wild_replicates]
                                )
                            ),
                        )
                        if discrepancy > 1e-10:
                            raise ArithmeticError("independent batched ratio disagrees")
                        validation.append(
                            dict(
                                n=n,
                                design=design,
                                noise=noise,
                                max_error=discrepancy,
                                scope=list(
                                    public.metadata["outside_confirmation_design"]
                                ),
                            )
                        )
                        first = y[:, : args.wild_replicates].copy()
                estimates = np.asarray(estimates)
                ses = np.asarray(ses)
                for signal in (0.0, 0.02, 0.05):
                    for alpha in (0.05, 0.005, 0.0005):
                        k = int(
                            np.sum(
                                abs((estimates + signal) / ses) >= norm.isf(alpha / 2)
                            )
                        )
                        records.append(
                            dict(
                                n=n,
                                design=design,
                                noise=noise,
                                method="HC3",
                                signal=signal,
                                alpha=alpha,
                                hits=k,
                                draws=args.draws,
                                rate=k / args.draws,
                                interval=interval(k, args.draws),
                                failed=False,
                                empirical_sd=estimates.std(ddof=1),
                                mean_se=ses.mean(),
                                true_noise_sd=float(np.sqrt((a * a) @ (sd * sd))),
                                max_leverage=leverage.max(),
                            )
                        )
                if args.wild_replicates:
                    wild = prepare_robust_scores(
                        f,
                        first,
                        c,
                        feature_names=("f",),
                        trait_names=tuple(map(str, range(args.wild_replicates))),
                        metadata={},
                        wild_draws=args.wild_draws,
                        seed=args.seed + n,
                    )
                    ps = [v["p"] for v in wild.metadata["wild_bootstrap"]]
                    normal_ps = 2 * norm.sf(
                        abs(wild.scores[0]) / np.sqrt(wild.score_covariance[:, 0, 0])
                    )
                    validation.append(
                        dict(
                            n=n,
                            design=design,
                            noise=noise,
                            matched_wild_p=ps,
                            matched_normal_p=normal_ps,
                            wild_draws=args.wild_draws,
                            matched_replicates=args.wild_replicates,
                        )
                    )
                    for alpha in (0.05, 0.005):
                        k = int(np.sum(np.asarray(ps) <= alpha))
                        records.append(
                            dict(
                                n=n,
                                design=design,
                                noise=noise,
                                method="wild_HC3",
                                signal=0.0,
                                alpha=alpha,
                                hits=k,
                                draws=args.wild_replicates,
                                rate=k / args.wild_replicates,
                                interval=interval(k, args.wild_replicates),
                                resamples_per_fit=args.wild_draws,
                                minimum_p=1 / (args.wild_draws + 1),
                                failed=False,
                            )
                        )
            print("tails", n, design, "done", flush=True)
    return records, validation


def structure_panel(args):
    rng = np.random.default_rng(args.seed + 8137)
    records = []
    n = 4096
    for rep in range(args.structure_draws):
        s = rng.integers(2, size=n)
        af = 0.15 + 0.3 * s
        target = np.zeros(n)
        hidden = np.zeros(n)
        other = np.zeros(n)
        for hap in range(2):
            causal = rng.binomial(1, af)
            hidden += causal
            target += np.where(rng.random(n) < 0.7, causal, rng.binomial(1, af))
            other += rng.binomial(1, 0.2 + 0.25 * s)
        # Arbitrary local additive and dominance mean plus a distal main effect.
        # Blocks independent CONDITIONAL on S; hidden locus never supplied.
        y = (
            2 * hidden
            + 0.8 * (hidden == 1)
            + 1.5 * other
            + (0.5 + target) * rng.standard_t(5, size=n) / np.sqrt(5 / 3)
        )
        c = np.column_stack([np.ones(n), s, target, target == 1, other, other == 1])
        f = (target * other)[:, None]
        cs, _ = stratified_fixed_effects(c, s.astype(str), memory_bytes=2**30)
        for method, design in [
            ("stratum_intercept_only", c),
            ("stratum_specific_main_effects", cs),
        ]:
            try:
                summary = prepare_robust_scores(
                    f,
                    y,
                    design,
                    feature_names=("f",),
                    trait_names=("y",),
                    metadata={},
                    sampling_model="iid_population_projection",
                )
                fit = robust_score_tests(summary)
                records.append(
                    dict(
                        replicate=rep,
                        method=method,
                        p=fit["kernel_p"],
                        beta=fit["beta"][0],
                        se=fit["standard_errors"][0],
                        failed=False,
                        biological_null=True,
                        projection_null_justified=(
                            method == "stratum_specific_main_effects"
                        ),
                        outside_scope=";".join(
                            summary.metadata["outside_confirmation_design"]
                        ),
                    )
                )
            except (ValueError, ArithmeticError) as error:
                records.append(
                    dict(
                        replicate=rep,
                        method=method,
                        failed=True,
                        p=np.nan,
                        error=str(error),
                    )
                )
    return records


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=576219)
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--wild-replicates", type=int, default=50)
    p.add_argument("--wild-draws", type=int, default=1999)
    p.add_argument("--structure-draws", type=int, default=2000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if not 1 <= a.wild_replicates <= 100:
        raise ValueError("1..100 full wild-bootstrap fits")
    tails, comparison = tail_panel(a)
    (a.out / "tails.json").write_text(json.dumps(_jsonable(tails), indent=2))
    pd.DataFrame(structure_panel(a)).to_csv(a.out / "structure.csv", index=False)
    (a.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    seed=a.seed,
                    draws=a.draws,
                    wild_replicates=a.wild_replicates,
                    wild_draws=a.wild_draws,
                    independent_production_comparisons=comparison,
                    tails_fixed="genotypes, correct finite mean, variances; residuals regenerated; same fixed wild-sign bank across independent phenotypes within a design",
                    structure_random="IID strata, conditional-independent discrete genotype blocks, residuals; hidden causal locus; nuisance mean and HC3 refitted every draw",
                    thresholds="families 1,10,100 at family .05; no genome-wide-tail extrapolation",
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
