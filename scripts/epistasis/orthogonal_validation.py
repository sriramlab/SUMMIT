"""Full native independent nuisance fits on fixed real genotype backgrounds."""
import argparse, json, tempfile, time, resource
from pathlib import Path
import numpy as np
from bed_reader import to_bed
from summit.epistasis.cli import main as cli, _jsonable
from summit.epistasis.robust import load_robust_scores, robust_score_tests
from scripts.epistasis.robust_validation import load_panel, reduction


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--real-genotypes")
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--genotype-seed", type=int, default=619348)
    p.add_argument("--phenotype-seed", type=int, default=172438)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--variants", type=int, default=4096)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if not 1 <= a.replicates <= 100:
        raise ValueError("at most 100 full fits")
    start = time.perf_counter()
    x, dom, meta, raw = load_panel(
        a.real_genotypes, a.genotype_seed, a.samples, a.variants, return_raw=True
    )
    n, m = x.shape
    n0 = n // 2
    full_mu = np.nanmean(raw, axis=0)
    test_mu = np.nanmean(raw[n0:], axis=0)
    baseline_factor = float(
        np.sqrt(
            np.prod(test_mu[:2] * (1 - test_mu[:2] / 2))
            / np.prod(full_mu[:2] * (1 - full_mu[:2] / 2))
        )
    )
    rng = np.random.default_rng(a.phenotype_seed)
    records = []
    # Genotypes and causal identities fixed. Dense random effects are generated
    # once and retained across residual replicates: conditional-mean stress.
    causal = rng.normal(size=m) * np.sqrt(1.0 / m)
    f = x[:, 0] * x[:, 1]
    mean = x @ causal + 0.7 * dom[:, 0]
    sd = np.sqrt(0.4 + 0.6 * x[:, 0] ** 2)
    with tempfile.TemporaryDirectory(prefix="orthogonal-", dir=a.scratch) as tmp:
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
        for name, rows in [
            ("all", range(n)),
            ("train", range(n0)),
            ("test", range(n0, n)),
        ]:
            (root / (name + ".tsv")).write_text(
                "FID IID\n" + "".join(f"{i} {i}\n" for i in rows)
            )
        for setting, beta in [("null", 0.0), ("weak", 0.025), ("moderate", 0.06)]:
            for rep in range(a.replicates):
                y = mean + beta * f + sd * rng.normal(size=n)
                pheno = f"{setting}-{rep}.tsv"
                (root / pheno).write_text(
                    "FID IID y\n"
                    + "".join(f"{i} {i} {v:.17g}\n" for i, v in enumerate(y))
                )
                for method in ("robust_mean", "orthogonal_mean"):
                    options = dict(
                        method=method,
                        main_effects="tested_variants",
                        dominance="tested_variants",
                    )
                    if method == "orthogonal_mean":
                        options.update(
                            training_samples="train.tsv",
                            ridge_variance=1.0,
                            residual_variance=1.0,
                            storage="packed",
                            solver=dict(rtol=1e-6),
                        )
                    spec = dict(
                        kind="summit.epistasis.prepare",
                        schema_version=1,
                        genotypes=dict(geno="input.bed"),
                        samples="all.tsv"
                        if method == "orthogonal_mean"
                        else "test.tsv",
                        phenotypes=dict(file=pheno, columns=["y"], unit="simulated"),
                        annotations={},
                        jobs=[
                            dict(
                                id="pair",
                                additive_annotations=["all"],
                                pairs=[["v0", "v1"]],
                                local_variants=[f"v{i}" for i in range(24)],
                                dominance_variants=[f"v{i}" for i in range(24)],
                                inference=options,
                            )
                        ],
                    )
                    stem = f"{setting}-{rep}-{method}"
                    path = root / (stem + ".json")
                    path.write_text(json.dumps(spec))
                    try:
                        cli(["prepare", str(path), "--out", str(root / stem)])
                        summary = load_robust_scores(
                            root / stem / "pair.robust-score.npz"
                        )
                        fit = robust_score_tests(summary)
                        records.append(
                            dict(
                                setting=setting,
                                method=method,
                                replicate=rep,
                                p=fit["kernel_p"],
                                failed=False,
                                beta0=fit["beta"][0],
                                se0=fit["standard_errors"][0],
                                truth0=beta
                                * (baseline_factor if method == "robust_mean" else 1.0),
                                coverage=float(
                                    fit["beta_interval_95"][0, 0]
                                    <= beta
                                    * (
                                        baseline_factor
                                        if method == "robust_mean"
                                        else 1.0
                                    )
                                    <= fit["beta_interval_95"][0, 1]
                                ),
                                outside_scope=";".join(
                                    summary.metadata["outside_confirmation_design"]
                                ),
                            )
                        )
                    except (ValueError, ArithmeticError, RuntimeError) as e:
                        records.append(
                            dict(
                                setting=setting,
                                method=method,
                                replicate=rep,
                                failed=True,
                                p=np.nan,
                                error=str(e),
                            )
                        )
            print(setting, "done", time.perf_counter() - start, flush=True)
    table = reduction(records, a.out)
    (a.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    panel=meta,
                    seed=a.phenotype_seed,
                    training_n=n0,
                    test_n=n - n0,
                    replicates=a.replicates,
                    fixed="genotypes, realized dense causal effects and dominance, interaction coefficient, nuisance ridge prior, sample split",
                    regenerated="independent training/confirmation residuals; native additive outcome AND feature projection refitted per replicate",
                    seconds=time.perf_counter() - start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    caution="Population restricted-space target: arbitrary fixed-genotype omitted mean can remain. Biological coefficients converted between full/test HWE units. Missing genotypes can leave residual mean approximation error; zero biological interaction is distinct from a pseudo-true regression coefficient.",
                )
            ),
            indent=2,
        )
    )
    print(table[["setting", "method", "rejection", "failures"]].to_string(index=False))


if __name__ == "__main__":
    main()
