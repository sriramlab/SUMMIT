"""Conditional tail checks with every finite-dimensional mean/HC3 refitted.

Genotypes and any externally trained quantities are fixed. These cheap draws
do not establish tails after retraining a polygenic score or at genome-wide P.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from scripts.epistasis.robust_validation import load_panel
from scripts.epistasis.validate import interval


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-genotypes")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--variants", type=int, default=4096)
    parser.add_argument("--genotype-seed", type=int, default=583092)
    parser.add_argument("--phenotype-seed", type=int, default=911460)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    x, d, meta = load_panel(
        args.real_genotypes, args.genotype_seed, args.samples, args.variants
    )
    c = np.column_stack([np.ones(len(x)), x[:, :24], d])
    u = thin_rank_revealing_fixed_effect_basis(c)
    raw = x[:, 0, None] * x[:, 3:13]
    r = raw - u @ (u.T @ raw)
    inverse = np.linalg.inv(r.T @ r)
    influence = r @ inverse
    leverage = np.sum(u * u, axis=1) + np.sum(r * influence, axis=1)
    rng = np.random.default_rng(args.phenotype_seed)
    results = []
    max_error = 0.0
    for setting in ("gaussian", "heteroskedastic", "heavy_t5"):
        variances = (
            0.3 + 0.7 * x[:, 0] ** 2
            if setting == "heteroskedastic"
            else np.ones(len(x))
        )
        oracle_se = np.sqrt(np.sum(influence**2 * variances[:, None], axis=0))
        hits = {
            m: np.zeros(3, dtype=int)
            for m in ("HC3_single", "known_covariance_single", "HC3_family10")
        }
        for begin in range(0, args.draws, 100):
            count = min(100, args.draws - begin)
            errors = (
                rng.standard_t(5, (len(x), count)) * np.sqrt(3 / 5)
                if setting == "heavy_t5"
                else rng.normal(size=(len(x), count))
            )
            errors *= np.sqrt(variances)[:, None]
            coef = influence.T @ errors
            residual = errors - u @ (u.T @ errors) - r @ coef
            se = np.sqrt(
                (influence**2).T @ ((residual / (1 - leverage[:, None])) ** 2)
            )
            p = 2 * norm.sf(abs(coef / se))
            oracle = 2 * norm.sf(abs(coef / oracle_se[:, None]))
            values = dict(
                HC3_single=p[0],
                known_covariance_single=oracle[0],
                HC3_family10=np.minimum(1.0, 10 * p.min(axis=0)),
            )
            for method, v in values.items():
                hits[method] += np.array([(v <= a).sum() for a in (0.05, 0.005, 0.001)])
            if begin == 0:
                direct = prepare_robust_scores(
                    raw,
                    errors[:, :3],
                    c,
                    feature_names=tuple(map(str, range(10))),
                    trait_names=("a", "b", "c"),
                    metadata={},
                )
                for i in range(3):
                    fit = robust_score_tests(direct, trait=i)
                    max_error = max(
                        max_error,
                        float(np.max(abs(fit["standard_errors"] - se[:, i]))),
                        float(np.max(abs(fit["beta"] - coef[:, i]))),
                    )
        for method, counts in hits.items():
            for alpha, k in zip((0.05, 0.005, 0.001), counts):
                lo, hi = interval(int(k), args.draws)
                results.append(
                    dict(
                        setting=setting,
                        method=method,
                        alpha=alpha,
                        hits=int(k),
                        draws=args.draws,
                        rate=float(k / args.draws),
                        lower=lo,
                        upper=hi,
                    )
                )
    payload = dict(
        panel=meta,
        seed=args.phenotype_seed,
        draws=args.draws,
        results=results,
        independent_dense_formula_max_error=max_error,
        max_leverage=float(leverage.max()),
        scope_diagnostics={
            key: direct.metadata[key]
            for key in (
                "outside_confirmation_design",
                "minimum_feature_effective_support",
                "fixed_rank",
                "information_condition",
            )
            if key in direct.metadata
        },
        seconds=time.perf_counter() - start,
        qualification="conditional fixed-design mean/HC3 recomputed on every draw; true covariance comparator exact only for Gaussian errors; no retraining or reference randomness",
        testing_scope="predeclared moderate families; .005 and .001 are empirical diagnostics, no genome-wide claim",
    )
    (args.out / "tails.json").write_text(json.dumps(payload, indent=2) + "\n")
    import pandas as pd

    table = pd.DataFrame(results)
    table.to_csv(args.out / "tails.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
