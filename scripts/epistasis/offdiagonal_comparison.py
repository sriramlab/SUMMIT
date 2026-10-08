"""Bounded off-diagonal counterexample and known-covariance power comparison."""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.quadratic import quadratic_sf
from summit.epistasis.cli import _jsonable
from scripts.epistasis.validate import interval


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(420781)
    n = 128
    x = rng.binomial(2, 0.35, (n, 48)).astype(float)
    x = (x - 0.7) / np.sqrt(0.455)
    c = np.column_stack([np.ones(n), x[:, :12]])
    u = thin_rank_revealing_fixed_effect_basis(c)
    P = np.eye(n) - u @ u.T
    f = x[:, 0, None] * x[:, 16:48] / np.sqrt(32)
    k = f @ f.T
    naive = P @ (k - np.diag(np.diag(k))) @ P
    r = P @ f
    w = np.linalg.solve(P * P, np.sum(r * r, axis=1))
    corrected = P @ (k - np.diag(w)) @ P
    variance = 0.2 + 20 * (np.diag(naive) < 0)
    sd = np.sqrt(variance)
    kernels = dict(
        kernel=P @ k @ P, plain_off_diagonal=naive, corrected_off_diagonal=corrected
    )
    records = []
    checks = {}
    # A fixed mean direction is shared by every method. Signal strength is
    # set once, before all residual replicates, on the projected feature scale.
    b = rng.normal(size=f.shape[1])
    signal = r @ b
    signal *= np.sqrt(0.3 / np.mean(signal**2))
    for name, A in kernels.items():
        B = sd[:, None] * A * sd[None, :]
        lam, vec = np.linalg.eigh(B)
        expected = float(np.trace(B))
        var = 2 * float(np.sum(B * B))
        calibration = rng.normal(size=(n, 100000))
        cal = np.sum(lam[:, None] * calibration**2, axis=0)
        critical = float(np.quantile(cal, 0.95, method="higher"))
        checks[name] = dict(
            expectation=expected,
            known_variance=var,
            mean_mc=float(cal.mean()),
            mean_mc_se=float(cal.std(ddof=1) / np.sqrt(len(cal))),
            max_diagonal=float(abs(np.diag(A)).max()),
            fixed_leak=float(abs(A @ c).max()),
            critical_95=critical,
            calibration_draws=len(cal),
            critical_tail_mc_interval=interval(5000, 100000),
        )
        del calibration
        # Verify a signed-eigenvalue tail independently of its MC quantile.
        tail = quadratic_sf(critical, lam, atol=2e-6)
        checks[name]["exact_known_covariance_tail_at_critical"] = tail
        for strength in (0.0, 1.0):
            offset = vec.T @ (strength * signal / sd)
            z = rng.normal(size=(n, 50000)) + offset[:, None]
            q = np.sum(lam[:, None] * z * z, axis=0)
            hits = int((q >= critical).sum())
            records.append(
                dict(
                    method=name,
                    signal_variance=0.3 * strength,
                    draws=50000,
                    rejection=hits / 50000,
                    interval=interval(hits, 50000),
                    known_covariance=True,
                )
            )
    added = np.outer(x[:, 15], x[:, 15])
    from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
    from scripts.epistasis.robust_validation import reduction

    matched = []
    noise = sd[:, None] * rng.normal(size=(n, 100))
    for strength in (0.0, 1.0):
        Y = noise + strength * signal[:, None]
        robust = prepare_robust_scores(
            f,
            Y,
            c,
            feature_names=tuple(map(str, range(f.shape[1]))),
            trait_names=tuple(map(str, range(100))),
            metadata={},
        )
        for rep in range(100):
            for name, A in kernels.items():
                statistic = float(Y[:, rep] @ A @ Y[:, rep])
                # Indicator encoded as P=.05/1 solely for the shared count
                # reducer; the calibrated threshold and MC error are above.
                matched.append(
                    dict(
                        setting="signal" if strength else "null",
                        method="known_D_" + name,
                        replicate=rep,
                        p=0.05 if statistic >= checks[name]["critical_95"] else 1.0,
                        failed=False,
                    )
                )
            fit = robust_score_tests(robust, trait=rep)
            matched.append(
                dict(
                    setting="signal" if strength else "null",
                    method="estimated_HC3",
                    replicate=rep,
                    p=fit["kernel_p"],
                    failed=False,
                )
            )
    reduction(matched, a.out)
    payload = dict(
        seed=420781,
        n=n,
        features=f.shape[1],
        fixed_rank=u.shape[1],
        checks=checks,
        records=records,
        unremoved_additive_expectation=float(np.sum(corrected * added)),
        seconds=time.perf_counter() - start,
        matched_HC3_design=robust_score_tests(robust)["diagnostics"],
        matched_comparison="100 identical phenotypes per setting for every quadratic statistic and refitted HC3; known-D comparators receive more nuisance information; small high-leverage HC3 design is outside supported scope",
        estimand="fixed-feature quadratic mean test, not random interaction variance",
        uncertainty="All methods receive the same true diagonal variance. Thresholds from independent 100000 Gaussian draws, checked by signed quadratic tails; power/null on 50000 new draws. Not estimated-nuisance confirmation.",
        conclusion="Diagonal deletion requires a projection-aware correction. Even the corrected statistic responds to omitted additive covariance; no unknown-variance tail or high-dimensional mean guarantee is established, so this remains a research comparison.",
    )
    (a.out / "comparison.json").write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
