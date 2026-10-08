"""Repeated actual pinned FAME pipeline, matched dense moments and mean-score tests.

All genotypes/effects are fixed within a case. Outcomes and nuisance fits are
regenerated. Upstream's wall-clock probes are not controllable by this script.
"""
import argparse, json, re, subprocess, tempfile, time, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from bed_reader import to_bed
from scripts.epistasis.compare_fame import REVISION, printed_equations
from scripts.epistasis.robust_validation import load_panel, reduction
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.cli import _jsonable


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("source", "binary", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--real-genotypes", required=True)
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument(
        "--diagnose-only",
        action="store_true",
        help="reconstruct the exact generating means and compare sequential versus joint projection without executable runs",
    )
    a = p.parse_args()
    if not 1 <= a.replicates <= 100:
        raise ValueError("at most 100 full fits per setting")
    if (
        subprocess.check_output(
            ["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True
        ).strip()
        != REVISION
        or subprocess.check_output(
            ["git", "-C", str(a.source), "diff", "--name-only"], text=True
        ).strip()
    ):
        raise ValueError("clean pinned upstream required")
    a.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(529307)
    records = []
    checks = []
    diagnoses = []
    start = time.perf_counter()
    for label, n in [("synthetic_ld", 768), ("real", 2048)]:
        if label == "real":
            _, _, source_meta, raw = load_panel(
                a.real_genotypes, 620193, n, 2048, return_raw=True
            )
            keep = np.flatnonzero(np.isfinite(raw).all(axis=0))[:128]
            if len(keep) != 128:
                raise ValueError(
                    "pipeline comparison needs 128 complete hard-call loci"
                )
            raw = raw[:, keep]
        else:
            raw = rng.binomial(2, 0.35, (n, 128)).astype(float)
            for j in range(1, 128):
                if j % 4:
                    raw[:, j] = np.where(rng.random(n) < 0.5, raw[:, j - 1], raw[:, j])
            source_meta = dict(kind="synthetic_discrete_copy_LD")
        m = raw.shape[1]
        mean = raw.mean(0)
        x = (raw - mean) / np.sqrt(mean * (1 - mean / 2))
        cov = np.column_stack([np.ones(n), rng.normal(size=n)])
        fixed = np.column_stack([cov, raw[:, :8]])
        f = x[:, 2, None] * x[:, 8:] / np.sqrt(m - 8)
        kernels = np.stack(
            [
                x[:, :8] @ x[:, :8].T / 8,
                x[:, 8:] @ x[:, 8:].T / (m - 8),
                f @ f.T,
                np.eye(n),
            ]
        )
        exact_t = np.einsum("aij,bji->ab", kernels, kernels)
        inverse = np.linalg.inv(exact_t)
        local_mean = raw[:, :8] @ rng.normal(size=8) / np.sqrt(8) + cov[:, 1]
        effect = np.zeros(m - 8)
        effect[0] = np.sqrt(0.01 / np.var(f[:, 0]))
        with tempfile.TemporaryDirectory(
            prefix="fame-replicates-", dir="/data1/bronsonj/epistasis_20261002"
        ) as tmp:
            root = Path(tmp)
            ids = list(map(str, range(n)))
            to_bed(
                root / "input.bed",
                raw,
                properties=dict(
                    fid=ids,
                    iid=ids,
                    sid=[f"v{i}" for i in range(m)],
                    chromosome=["1"] * m,
                    bp_position=np.arange(1, m + 1),
                    allele_1=["A"] * m,
                    allele_2=["G"] * m,
                ),
            )
            for script in (
                "generate_ld_annotations.py",
                "linear_regression_annotation.py",
            ):
                (root / script).symlink_to(a.source.resolve() / "pipeline" / script)
            (root / "ld.txt").write_text(f"chr start stop\n1 1 8\n1 9 {m}\n")
            for setting in (
                "null_gaussian",
                "null_heterogeneous",
                "dominance_omitted",
                "dominance_adjusted",
                "sparse_signal",
            ):
                C = (
                    np.column_stack([fixed, raw[:, 2] == 1])
                    if setting == "dominance_adjusted"
                    else fixed
                )
                covariates = (
                    np.column_stack([cov, raw[:, 2] == 1])
                    if setting == "dominance_adjusted"
                    else cov
                )
                variance = (
                    0.25 + 0.75 * x[:, 2] ** 2
                    if setting == "null_heterogeneous"
                    else np.ones(n)
                )
                mu = local_mean.copy()
                if setting.startswith("dominance"):
                    mu += 2 * (raw[:, 2] == 1)
                if setting == "sparse_signal":
                    mu += f @ effect
                Y = mu[:, None] + np.sqrt(variance)[:, None] * rng.normal(
                    size=(n, a.replicates)
                )
                local = np.column_stack([np.ones(n), raw[:, :8]])
                sequential = mu - local @ np.linalg.lstsq(local, mu, rcond=None)[0]
                sequential -= (
                    covariates @ np.linalg.lstsq(covariates, sequential, rcond=None)[0]
                )
                joint = mu - C @ np.linalg.lstsq(C, mu, rcond=None)[0]
                diagnoses.append(
                    dict(
                        case=label,
                        setting=setting,
                        n=n,
                        m=m,
                        sequential_mean_square=float(np.mean(sequential**2)),
                        joint_mean_square=float(np.mean(joint**2)),
                    )
                )
                if a.diagnose_only:
                    continue
                robust = prepare_robust_scores(
                    f,
                    Y,
                    C,
                    feature_names=tuple(f"f{i}" for i in range(m - 8)),
                    trait_names=tuple(map(str, range(a.replicates))),
                    metadata={},
                )
                for rep in range(a.replicates):
                    trait = f"{setting}_{rep}"
                    y = Y[:, rep]
                    (root / (trait + ".pheno")).write_text(
                        "FID IID pheno\n"
                        + "".join(f"{i} {i} {v:.17g}\n" for i, v in enumerate(y))
                    )
                    (root / (trait + ".covar")).write_text(
                        "FID IID "
                        + " ".join(f"c{i}" for i in range(covariates.shape[1]))
                        + "\n"
                        + "".join(
                            f"{i} {i} " + " ".join(f"{v:.17g}" for v in row) + "\n"
                            for i, row in enumerate(covariates)
                        )
                    )
                    (root / "pairs.txt").write_text(f"{trait} v2\n")
                    cmd = [
                        "bash",
                        str(a.source.resolve() / "pipeline/run_fame_pipeline.sh"),
                        "0",
                        str(root / "input"),
                        str(root),
                        str(root / "pairs.txt"),
                        str(root / "ld.txt"),
                        str(a.binary.resolve()),
                    ]
                    run = subprocess.run(
                        cmd, cwd=root, capture_output=True, text=True, timeout=120
                    )
                    result = root / f"results/{trait}-v2.res.out.txt"
                    if run.returncode or not result.exists():
                        raise RuntimeError(run.stdout[-1000:] + run.stderr[-1000:])
                    residual = np.loadtxt(
                        root / f"residualized_pheno/{trait}-v2.pheno", skiprows=1
                    )[:, 2]
                    matched = y - LinearRegression().fit(
                        raw[:, :8].astype(np.float32), y
                    ).predict(raw[:, :8].astype(np.float32))
                    yy = (
                        residual
                        - covariates
                        @ np.linalg.lstsq(covariates, residual, rcond=None)[0]
                    )
                    yy = (yy - yy.mean()) / yy.std(ddof=1)
                    ky = kernels @ yy
                    q = ky @ yy
                    cubic = np.einsum("an,cnm,bm->acb", ky, kernels, ky, optimize=True)
                    printed_t, printed_q = printed_equations(
                        (root / f"results/{trait}-v2.res.out.full.txt").read_text(), 4
                    )
                    observed = np.array(
                        [
                            [float(c), float(s)]
                            for c, s in re.findall(
                                r"sigma\^2_\d+: ([^ ]+) se: ([^\n]+)",
                                result.read_text(),
                            )
                        ]
                    )
                    exact_beta = inverse @ q
                    exact_cov = (
                        inverse
                        @ (2 * np.einsum("c,acb->ab", exact_beta, cubic))
                        @ inverse.T
                    )
                    printed_inverse = np.linalg.inv(printed_t)
                    matched_beta = printed_inverse @ q
                    matched_cov = (
                        printed_inverse
                        @ (2 * np.einsum("c,acb->ab", matched_beta, cubic))
                        @ printed_inverse.T
                    )
                    fit = robust_score_tests(robust, trait=rep)
                    for method, b, se, pv in [
                        ("FAME_pipeline", observed[2, 0], observed[2, 1], np.nan),
                        (
                            "SUMMIT_dense_FAME_conventions",
                            exact_beta[2],
                            np.sqrt(exact_cov[2, 2]) if exact_cov[2, 2] > 0 else np.nan,
                            np.nan,
                        ),
                        ("HC3_same_features", np.nan, np.nan, fit["kernel_p"]),
                    ]:
                        if method != "HC3_same_features":
                            pv = (
                                float(norm.sf(b / se))
                                if np.isfinite(se) and se > 0
                                else np.nan
                            )
                        records.append(
                            dict(
                                setting=label + "_" + setting,
                                method=method,
                                replicate=rep,
                                p=pv,
                                failed=not np.isfinite(pv),
                                coefficient=b,
                                standard_error=se,
                            )
                        )
                    checks.append(
                        dict(
                            case=label,
                            setting=setting,
                            replicate=rep,
                            n=n,
                            m=m,
                            rank=int(np.linalg.matrix_rank(f)),
                            preprocessing_error=float(np.max(abs(residual - matched))),
                            rhs_error=float(np.max(abs(q - printed_q))),
                            beta_error=float(
                                np.max(abs(matched_beta - observed[:, 0]))
                            ),
                            se_error=float(
                                np.nanmax(
                                    abs(
                                        np.sqrt(
                                            np.where(
                                                np.diag(matched_cov) > 0,
                                                np.diag(matched_cov),
                                                np.nan,
                                            )
                                        )
                                        - observed[:, 1]
                                    )
                                )
                            ),
                            trace_relative_error=float(
                                np.linalg.norm(printed_t - exact_t)
                                / np.linalg.norm(exact_t)
                            ),
                        )
                    )
                print(label, setting, "completed", flush=True)
    if a.diagnose_only:
        (a.out / "projection.json").write_text(
            json.dumps(
                dict(
                    seed=529307,
                    replicates_used_to_reconstruct_rng=a.replicates,
                    records=diagnoses,
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(diagnoses, indent=2))
        return
    table = reduction(records, a.out)
    (a.out / "comparison.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    revision=REVISION,
                    binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                    seed=529307,
                    replicates=a.replicates,
                    checks=checks,
                    seconds=time.perf_counter() - start,
                    source=source_meta,
                    source_selection="128 complete hard-call loci from seeded 2048-marker panel; upstream local regression cannot accept missing local dosages; pipeline fixes 100 jackknife bins and crashes when M<100",
                    fixed="genotypes, effects, local block, conditional mean covariates; residuals and all trait nuisance fits regenerated",
                    random_reference="upstream wall-clock sample probes, 100 per invocation; may share draws within a clock second",
                    inference="FAME one-sided positive-variance Wald; HC3 global two-sided mean-feature alternative, same supplied feature span and declared mean covariates; point estimands differ",
                    normalization="upstream sequential local float32 regression, phenotype-only covariate projection and sample variance normalization; dense comparator reproduces those choices with exact trace products",
                    limitation="bounded empirical comparison, not biobank or extreme-tail calibration; sparse signal does not define a fixed realized FAME variance coefficient",
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(table[["setting", "method", "rejection", "failures"]].to_string(index=False))


if __name__ == "__main__":
    main()
