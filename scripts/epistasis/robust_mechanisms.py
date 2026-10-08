"""Prespecified scale, phantom-epistasis and off-diagonal investigations."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import norm
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.cli import _jsonable
from scripts.epistasis.robust_validation import reduction


def test(f, y, c, name, records, truth=0.0):
    summary = prepare_robust_scores(
        f,
        y,
        c,
        feature_names=("interaction",),
        trait_names=tuple(map(str, range(y.shape[1]))),
        metadata={},
    )
    for i in range(y.shape[1]):
        result = robust_score_tests(summary, trait=i)
        records.append(
            dict(
                setting=name,
                method="HC3",
                replicate=i,
                p=result["kernel_p"],
                failed=False,
                beta0=result["beta"][0],
                se0=result["standard_errors"][0],
                truth0=truth,
                coverage=float(
                    abs(result["beta"][0] - truth)
                    <= 1.95996398454 * result["standard_errors"][0]
                ),
            )
        )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=663041)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100:
        raise ValueError("at most 100 full fits per setting")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    records = []
    n = args.samples
    b = args.replicates
    started = time.perf_counter()
    # Four haplotypes; the untyped causal allele is a haplotype property.
    # Its dosage has a purely additive phenotype effect. Observed marker
    # products can tag it beyond marker additive+dominance main effects.
    panels = []
    for phase in ("discovery", "independent_confirmation"):
        hap = rng.binomial(1, 0.35, (n, 2, 2))
        g = hap.sum(axis=1).astype(float)
        causal = (hap[:, :, 0] * hap[:, :, 1]).sum(axis=1).astype(float)
        x = (g - 0.7) / np.sqrt(0.455)
        f = (x[:, 0] * x[:, 1])[:, None]
        c = np.column_stack([np.ones(n), x, g == 1])
        y = causal[:, None] + rng.normal(size=(n, b))
        test(f, y, c, phase + "_causal_hidden", records)
        test(
            f,
            y,
            np.column_stack([c, causal, causal == 1]),
            phase + "_dense_causal_restored",
            records,
        )
        panels.append(
            dict(
                phase=phase,
                causal_effect=1.0,
                max_marker_causal_ld=float(
                    abs(np.corrcoef(np.column_stack([g, causal]).T)[:2, 2]).max()
                ),
            )
        )
    # Fixed prespecified annotation score: a supplied predictor, not knowledge
    # of every unknown causal effect. Independent calibration outcomes fit the
    # smooth curve; test outcomes never enter that fit.
    x = rng.binomial(2, 0.35, (2 * n, 16)).astype(float)
    x = (x - 0.7) / np.sqrt(0.455)
    index = x @ np.ones(16) / 4
    f = (x[:, 0] * x[:, 1])[:, None]
    c = np.column_stack([np.ones(2 * n), x[:, :2]])
    basis = np.column_stack([np.ones(2 * n), index, index**2, index**3])
    for signal in (0.0, 0.06):
        calibration = (
            0.3 * index[:n]
            + 0.25 * index[:n] ** 2
            + signal * f[:n, 0]
            + rng.normal(size=n)
        )
        curve = basis[n:] @ np.linalg.lstsq(basis[:n], calibration, rcond=None)[0]
        mu = 0.3 * index[n:] + 0.25 * index[n:] ** 2 + signal * f[n:, 0]
        y = mu[:, None] + rng.normal(size=(n, b))
        for label, design in [
            ("linear", np.column_stack([c[n:], index[n:]])),
            ("frozen_curve", np.column_stack([c[n:], index[n:], curve])),
            ("declared_cubic_refitted", np.column_stack([c[n:], basis[n:, 1:]])),
        ]:
            test(
                f[n:],
                y,
                design,
                f"global_nonlinearity_signal{signal}_{label}",
                records,
                truth=signal,
            )
    # Misspecified single-index baseline: a second index not represented by
    # the first. Keeping this failure prevents a universal nonlinear claim.
    other = x[:, 0] + x[:, 1]
    y = (0.3 * index[n:] + 0.25 * other[n:] ** 2)[:, None] + rng.normal(size=(n, b))
    test(
        f[n:],
        y,
        np.column_stack([c[n:], basis[n:, 1:]]),
        "misspecified_single_index",
        records,
    )
    # Positive phenotype, multiplicative Gaussian noise on log scale. Both
    # scales are declared in advance; raw-scale interaction is not log-scale
    # causal interaction. Frozen external CDF keeps individuals independent.
    log_training = 0.25 * index[:n] + rng.normal(scale=0.5, size=n)
    log_y = 0.25 * index[n:, None] + rng.normal(scale=0.5, size=(n, b))
    raw = np.exp(log_y)
    cdf = np.sort(log_training)
    rank = norm.ppf((np.searchsorted(cdf, log_y, side="right") + 0.5) / (len(cdf) + 1))
    for name, y in [("raw", raw), ("log", log_y), ("frozen_rank_normal", rank)]:
        test(f[n:], y, np.column_stack([c[n:], index[n:]]), "scale_" + name, records)
    # Exact small reference to diagonal deletion before versus after fitting.
    n0 = 128
    z = rng.normal(size=(n0, 3))
    z[:, 0] += np.linspace(-2, 2, n0)
    fixed = np.column_stack([np.ones(n0), z[:, 0]])
    u = thin_rank_revealing_fixed_effect_basis(fixed)
    p = np.eye(n0) - u @ u.T
    ff = np.column_stack([z[:, 0] * z[:, 1], z[:, 0] * z[:, 2]])
    k = ff @ ff.T
    a = k - np.diag(np.diag(k))
    projected = p @ a @ p
    variance = 0.2 + z[:, 0] ** 2
    expected = float(np.diag(projected) @ variance)
    e = rng.normal(size=(n0, 20000)) * np.sqrt(variance[:, None])
    py = p @ e
    q = np.sum((ff.T @ py) ** 2, axis=0) - np.diag(k) @ (py * py)
    # Solve (P o P)w=diag(PKP) without building P o P in the candidate path.
    leverage = np.sum(u * u, axis=1)

    def multiply(w):
        return (1 - 2 * leverage) * w + np.sum(
            (u @ (u.T @ (w[:, None] * u))) * u, axis=1
        )

    r = p @ ff
    w, info = cg(
        LinearOperator((n0, n0), matvec=multiply),
        np.sum(r * r, axis=1),
        rtol=1e-12,
        atol=0.0,
    )
    if info:
        raise ArithmeticError("diagonal correction CG failed")
    corrected = p @ (k - np.diag(w)) @ p
    if not np.allclose(np.diag(corrected), 0, atol=1e-8):
        raise ArithmeticError("diagonal correction identity failed")
    corrected_q = np.sum((ff.T @ py) ** 2, axis=0) - w @ (py * py)
    unremoved_additive = np.outer(z[:, 1], z[:, 1])
    offdiag = dict(
        n=n0,
        replicates=20000,
        uncorrected_projected_expected=expected,
        uncorrected_empirical_mean=float(q.mean()),
        uncorrected_mean_mc_se=float(q.std(ddof=1) / np.sqrt(len(q))),
        corrected_max_diagonal=float(abs(np.diag(corrected)).max()),
        corrected_fixed_leakage=float(abs(corrected @ fixed).max()),
        corrected_expected=float(np.diag(corrected) @ variance),
        corrected_empirical_mean=float(corrected_q.mean()),
        corrected_mean_mc_se=float(corrected_q.std(ddof=1) / np.sqrt(len(q))),
        corrected_unremoved_additive_expectation=float(
            np.sum(corrected * unremoved_additive)
        ),
        conclusion="plain diagonal deletion fails after projection; matrix-free diagonal correction removes independent diagonal covariance but not an omitted additive mean/covariance; unknown-variance tail remains experimental",
    )
    table = reduction(records, args.out)
    (args.out / "mechanisms.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    seed=args.seed,
                    n=n,
                    replicates=b,
                    panels=panels,
                    off_diagonal=offdiag,
                    fixed="genotypes, causal effects and trained curve; all residual phenotypes regenerated independently",
                    nonlinear_basis="prespecified cubic of supplied annotation index; independently fitted curve and test-refitted finite-basis alternatives kept separate",
                    rank_interpretation="independent empirical phenotype CDF; not rank mutagenesis inference",
                    seconds=time.perf_counter() - started,
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(table[["setting", "rejection", "coverage"]].to_string(index=False))
    print(json.dumps(offdiag, indent=2))


if __name__ == "__main__":
    main()
