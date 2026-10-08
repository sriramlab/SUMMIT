"""Independent spectral validation of estimated HC3 tails and many-covariate variance."""
import argparse, json, time, resource
from pathlib import Path
import numpy as np
from scipy.stats import norm
from summit.epistasis.robust_reference import (
    hc3_ratio_reference,
    many_covariate_reference,
)
from summit.epistasis.cli import _jsonable


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(712431)
    n = 512
    rows = []
    start = time.perf_counter()
    for case in ("common", "rare", "many_covariates", "concentrated"):
        g = rng.binomial(2, 0.3 if case != "rare" else 0.03, (n, 2)).astype(float)
        c = np.column_stack([np.ones(n), g, g == 1])
        if case == "many_covariates":
            c = np.column_stack([c, rng.normal(size=(n, 100))])
        f = np.prod(g - g.mean(0), axis=1)
        if case == "concentrated":
            f[0] += 25
        omega = 0.4 + 0.6 * (g[:, 0] - g[:, 0].mean()) ** 2
        d = np.column_stack([c, f])
        di = np.linalg.pinv(d)
        h = np.einsum("ij,ji->i", d, di)
        infl = di[-1]
        ts = []
        for _ in range(100):
            e = rng.normal(size=(n, 1000)) * np.sqrt(omega[:, None])
            r = e - d @ (di @ e)
            ts.extend((infl @ e) / np.sqrt((infl**2 / (1 - h) ** 2) @ (r * r)))
        ts = np.abs(ts)
        for alpha in (0.05, 0.005, 0.0005, 5e-6, 5e-8):
            record = dict(
                case=case,
                nominal_alpha=alpha,
                draws=len(ts),
                exceedances=int(np.sum(ts >= norm.isf(alpha / 2))),
                max_leverage=float(h.max()),
            )
            try:
                ref = hc3_ratio_reference(f, c, omega, norm.isf(alpha / 2), atol=1e-11)
                record.update(
                    {
                        k: v
                        for k, v in ref.items()
                        if k
                        not in (
                            "coefficient_influence",
                            "denominator_form",
                            "eigenvalues",
                        )
                    }
                )
            except (ValueError, ArithmeticError) as error:
                record["failure"] = str(error)
            rows.append(record)
        yy = c @ rng.normal(size=c.shape[1])
        ee = rng.normal(size=(n, 20000)) * np.sqrt(omega[:, None])
        many = many_covariate_reference(f, yy[:, None] + ee, c)
        truth = float(np.dot(infl**2, omega))
        for method in ("HC3", "leave_out", "hadamard"):
            v = many[method]
            valid = np.isfinite(v) & (v > 0)
            rows.append(
                dict(
                    case=case,
                    method=method,
                    true_coefficient_variance=truth,
                    mean_variance=float(v.mean()),
                    variance_mean_mcse=float(v.std() / np.sqrt(len(v))),
                    identified=bool(many["hadamard_identified"])
                    if method == "hadamard"
                    else True,
                    negative_or_zero=int((~valid).sum()),
                    draws=len(v),
                    null_rejection_valid=float(
                        np.mean(
                            abs(many["beta"][valid]) / np.sqrt(v[valid])
                            > norm.isf(0.025)
                        )
                    )
                    if valid.any()
                    else None,
                    covariance_hadamard_condition=many["hadamard_condition"],
                )
            )
    with a.out.open("x") as h:
        json.dump(
            _jsonable(
                dict(
                    seed=712431,
                    n=n,
                    results=rows,
                    seconds=time.perf_counter() - start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    fixed="design, realized correctly specified nuisance mean, heteroskedastic variances",
                    regenerated="independent Gaussian errors; complete OLS nuisance and variance refitted each draw",
                    interpretation="Known generating variances only enter validation spectrum; production uses estimated HC3. Monte Carlo cannot resolve zero-hit tails.",
                )
            ),
            h,
            indent=2,
        )


if __name__ == "__main__":
    main()
