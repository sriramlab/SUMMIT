"""Prespecified structure, rare-cell and scalar wild-refit checks."""
import argparse, json, time
from pathlib import Path
import numpy as np
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.cli import _jsonable
from scripts.epistasis.robust_validation import load_panel, reduction


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--real-genotypes", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(923784)
    x, d, meta = load_panel(a.real_genotypes, 293760, 2048, 4096)
    n = len(x)
    records = []
    design = []
    c = np.column_stack([np.ones(n), x[:, :24], d])
    f = (x[:, 0] * x[:, 3])[:, None]
    y = (x[:, :24] @ rng.normal(size=24) / np.sqrt(24))[:, None] + rng.normal(
        size=(n, 100)
    ) * np.sqrt(0.25 + 0.75 * x[:, 0, None] ** 2)
    s = prepare_robust_scores(
        f,
        y,
        c,
        feature_names=("pair",),
        trait_names=tuple(map(str, range(100))),
        metadata={},
        wild_draws=999,
        seed=417892,
    )
    for rep in range(100):
        fit = robust_score_tests(s, trait=rep)
        for method, pv in [
            ("HC3", fit["kernel_p"]),
            ("wild_999", fit["wild_bootstrap"]["p"]),
        ]:
            records.append(
                dict(
                    setting="real_heterogeneous_null",
                    method=method,
                    replicate=rep,
                    p=pv,
                    failed=False,
                    beta0=fit["beta"][0],
                    truth0=0.0,
                    se0=fit["standard_errors"][0],
                    coverage=float(
                        abs(fit["beta"][0]) <= 1.95996398454 * fit["standard_errors"][0]
                    ),
                )
            )
    design.append(
        dict(
            setting="real_heterogeneous_null",
            diagnostics=robust_score_tests(s)["diagnostics"],
            wild="restricted mean and HC3 refitted 999 times, shared fixed signs across traits; minimum P=.001; not exact",
        )
    )
    # Modest allele-frequency structure and a nonlinear covariate mean.
    z = rng.normal(size=n)
    prob = np.clip(0.35 + 0.08 * z, 0.08, 0.8)
    g = np.column_stack([rng.binomial(2, prob) for _ in range(8)]).astype(float)
    f = ((g[:, 0] - g[:, 0].mean()) * (g[:, 1] - g[:, 1].mean()))[:, None]
    c = np.column_stack([np.ones(n), g, g == 1, z])
    mu = 0.8 * z + 0.8 * z * z + g @ np.ones(8) / 8
    y = mu[:, None] + rng.normal(size=(n, 100))
    cases = [
        ("structure_nonlinear_omitted", f, y, c),
        ("structure_nonlinear_adjusted", f, y, np.column_stack([c, z * z])),
    ]
    for maf in (0.2, 0.05, 0.01):
        raw = rng.binomial(2, [maf, 0.35], (n, 2)).astype(float)
        c = np.column_stack([np.ones(n), raw, raw == 1])
        f = np.prod(raw - raw.mean(0), axis=1)[:, None]
        y = (raw @ np.array([1.0, 0.5]) + 0.5 * (raw[:, 0] == 1))[
            :, None
        ] + rng.standard_t(5, (n, 100)) * np.sqrt(3 / 5)
        cases.append((f"rare_target_maf_{maf}", f, y, c))
    for name, f, y, c in cases:
        try:
            s = prepare_robust_scores(
                f,
                y,
                c,
                feature_names=("pair",),
                trait_names=tuple(map(str, range(100))),
                metadata={},
            )
            design.append(
                dict(setting=name, diagnostics=robust_score_tests(s)["diagnostics"])
            )
            for rep in range(100):
                fit = robust_score_tests(s, trait=rep)
                records.append(
                    dict(
                        setting=name,
                        method="HC3",
                        replicate=rep,
                        p=fit["kernel_p"],
                        failed=False,
                        beta0=fit["beta"][0],
                        truth0=0.0,
                        se0=fit["standard_errors"][0],
                        coverage=float(
                            abs(fit["beta"][0])
                            <= 1.95996398454 * fit["standard_errors"][0]
                        ),
                    )
                )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            for rep in range(100):
                records.append(
                    dict(
                        setting=name,
                        method="HC3",
                        replicate=rep,
                        p=np.nan,
                        failed=True,
                        error=str(error),
                    )
                )
    table = reduction(records, a.out)
    (a.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    seed=923784,
                    panel=meta,
                    settings=design,
                    seconds=time.perf_counter() - start,
                    fixed="genotypes, covariates and effects fixed; 100 independent residual draws per setting; conditional means and HC3 refitted",
                    status="independent prespecified stress checks; no method, threshold or scope-screen changes from these results",
                    coverage="biological zero is not a valid conditional coefficient under omitted nonlinear mean; label that row as failure control",
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        table[["setting", "method", "rejection", "coverage", "failures"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
