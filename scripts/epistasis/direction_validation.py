"""Independent training/confirmation on real genotypes with unknown dense additive effects.

Training is repeated in every replicate. Only aggregate results persist; the
temporary BED, phenotypes and fitted participant-derived models stay local.
"""
import argparse, json, tempfile, time, resource
from pathlib import Path
import numpy as np
from bed_reader import to_bed
from summit.epistasis.cli import main as epistasis_main, _jsonable
from summit.epistasis.directions import score_frozen
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.prediction.genotype import FileGenotypeSource, native_module
from scripts.epistasis.robust_validation import load_panel, reduction


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--real-genotypes")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--training-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--variants", type=int, default=4096)
    parser.add_argument("--genotype-seed", type=int, default=642081)
    parser.add_argument("--phenotype-seed", type=int, default=440763)
    parser.add_argument(
        "--diagnose-null-replicate",
        type=int,
        help="reconstruct one frozen null replicate and inspect learned nuisance leverage; no calibration claim",
    )
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100:
        raise ValueError("at most 100 full refits per setting")
    if (
        args.diagnose_null_replicate is not None
        and not 0 <= args.diagnose_null_replicate < args.replicates
    ):
        raise ValueError(
            "diagnostic replicate must belong to the declared original panel"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cpu = time.process_time()
    n0 = args.training_samples
    n1 = args.test_samples
    n = n0 + n1
    m = args.variants
    x, dom, meta, raw = load_panel(
        args.real_genotypes, args.genotype_seed, n, m, return_raw=True
    )
    rng = np.random.default_rng(args.phenotype_seed)
    records = []
    timings = []
    with tempfile.TemporaryDirectory(
        prefix="independent-direction-", dir=args.scratch
    ) as temporary:
        root = Path(temporary)
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
        regional_missing = np.isnan(raw[:, 24:88]).any(axis=1)
        del raw
        for name, rows in [("train", range(n0)), ("test", range(n0, n))]:
            (root / f"{name}.tsv").write_text(
                "FID IID\n" + "".join(f"{i} {i}\n" for i in rows)
            )
        (root / "variants.txt").write_text(
            "\n".join(f"v{i}" for i in range(1, m)) + "\n"
        )
        region = np.arange(24, 88)
        (root / "region.txt").write_text("\n".join(f"v{i}" for i in region) + "\n")
        pair_features = x[:, 0, None] * x[:, region] / np.sqrt(len(region))
        local = np.column_stack([np.ones(n), x[:, :24], dom[:, :24]])
        for setting, strength, architecture in [
            ("null", 0.0, "mixed"),
            ("mixed_weak", 0.002, "mixed"),
            ("mixed_moderate", 0.01, "mixed"),
            ("aligned_weak", 0.002, "aligned"),
        ]:
            if args.diagnose_null_replicate is not None and setting != "null":
                continue
            for rep in range(args.replicates):
                # Effects drawn from a fixed variance distribution, never
                # normalized using held-out phenotypes or realized variances.
                additive = rng.normal(size=m) * np.sqrt(0.3 / m)
                effects = rng.normal(size=len(region)) * np.sqrt(strength)
                if architecture == "aligned":
                    effects = np.full(len(region), np.sqrt(strength))
                mean = x @ additive + 0.5 * dom[:, 0] + pair_features @ effects
                sd = np.sqrt(0.4 + 0.6 * x[:, 0] ** 2)
                phenotype = mean + rng.normal(size=n) * sd
                if (
                    args.diagnose_null_replicate is not None
                    and rep != args.diagnose_null_replicate
                ):
                    continue
                stem = f"{setting}-{rep}"
                pheno = stem + ".tsv"
                (root / pheno).write_text(
                    "FID IID y\n"
                    + "".join(f"{i} {i} {v:.17g}\n" for i, v in zip(ids, phenotype))
                )
                spec = dict(
                    kind="summit.epistasis.train_direction",
                    schema_version=1,
                    genotypes=dict(geno="input.bed"),
                    samples="train.tsv",
                    phenotype=dict(file=pheno, column="y", unit="simulated"),
                    target="v0",
                    variants="variants.txt",
                    interaction_variants="region.txt",
                    local_variants=[f"v{i}" for i in range(24)],
                    dominance_variants=[f"v{i}" for i in range(24)],
                    prior=dict(additive=0.5, interaction=0.05, residual=1.0),
                    storage="packed",
                    solver=dict(rtol=1e-6, max_iterations=150),
                )
                manifest = root / (stem + ".json")
                manifest.write_text(json.dumps(spec))
                t0 = time.perf_counter()
                try:
                    epistasis_main(
                        ["train-direction", str(manifest), "--out", str(root / stem)]
                    )
                    training_seconds = time.perf_counter() - t0
                    definitions = [
                        dict(
                            name="pgs",
                            direction=stem + "/direction.json",
                            component=0,
                            adjust=True,
                        ),
                        dict(
                            name="interaction",
                            direction=stem + "/direction.json",
                            component=1,
                        ),
                    ]
                    with FileGenotypeSource(root / "input.bed") as source:
                        scores, adjust, report = score_frozen(
                            definitions,
                            root,
                            source,
                            np.arange(n0, n),
                            threads=1,
                            block_size=256,
                            memory_bytes=2**30,
                        )
                    # Identical nuisance information for every comparison,
                    # including all region main effects and both learned scores.
                    fixed = np.column_stack(
                        [
                            local[n0:],
                            x[n0:, region],
                            adjust,
                            scores["interaction"]["values"],
                        ]
                    )
                    if args.diagnose_null_replicate is not None:
                        from summit.context.fixed import (
                            thin_rank_revealing_fixed_effect_basis,
                        )
                        from summit.prediction.artifacts import load_prediction_models

                        base = fixed[:, :-1]
                        u = thin_rank_revealing_fixed_effect_basis(base)
                        full = thin_rank_revealing_fixed_effect_basis(fixed)
                        learned = fixed[:, -1]
                        remainder = learned - u @ (u.T @ learned)
                        missing = regional_missing[n0:]
                        model = load_prediction_models(root / stem / "models")[0]
                        model_region = np.array(
                            [v in {f"v{i}" for i in region} for v in model.variants.ids]
                        )
                        diagnosis = dict(
                            setting=setting,
                            replicate=rep,
                            base_rank=u.shape[1],
                            full_rank=full.shape[1],
                            base_max_leverage=float(np.sum(u * u, axis=1).max()),
                            full_max_leverage=float(np.sum(full * full, axis=1).max()),
                            score_residual_energy_fraction=float(
                                np.sum(remainder**2) / np.sum(learned**2)
                            ),
                            remainder_energy_on_rows_with_missing_region_genotypes=float(
                                np.sum(remainder[missing] ** 2) / np.sum(remainder**2)
                            ),
                            regional_missing_rows=int(missing.sum()),
                            outside_region_max_abs_interaction_weight=float(
                                np.max(abs(model.weights[~model_region, 1]))
                            ),
                            explanation="Test regional main columns use panel-mean genotype imputation; frozen scores preserve training-mean imputation. Their difference may create a nearly collinear missingness contrast. No column removed or leverage threshold changed.",
                        )
                        (args.out / "diagnosis.json").write_text(
                            json.dumps(diagnosis, indent=2) + "\n"
                        )
                    for name, features in [
                        (
                            "learned_direction",
                            x[n0:, 0, None] * scores["interaction"]["values"][:, None],
                        ),
                        ("additive_PGS_direction", x[n0:, 0, None] * adjust),
                        ("regional_kernel", pair_features[n0:]),
                        (
                            "supplied_burden",
                            pair_features[n0:].sum(axis=1, keepdims=True),
                        ),
                    ]:
                        summary = prepare_robust_scores(
                            features,
                            phenotype[n0:],
                            fixed,
                            feature_names=tuple(
                                f"f{i}" for i in range(features.shape[1])
                            ),
                            trait_names=("y",),
                            metadata={},
                        )
                        fit = robust_score_tests(
                            summary, burden=np.ones(features.shape[1])
                        )
                        records.append(
                            dict(
                                setting=setting,
                                method=name,
                                replicate=rep,
                                p=fit["kernel_p"],
                                failed=False,
                                beta0=fit["beta"][0],
                                se0=fit["standard_errors"][0],
                                truth0=np.nan,
                                coverage=np.nan,
                                expected_interaction_coefficient_variance=strength,
                                realized_interaction_variance=float(
                                    np.var((pair_features @ effects)[n0:])
                                ),
                                max_leverage=summary.metadata["max_leverage"],
                                outside_scope=";".join(
                                    summary.metadata["outside_confirmation_design"]
                                ),
                            )
                        )
                    timings.append(
                        dict(
                            setting=setting,
                            replicate=rep,
                            training_seconds=training_seconds,
                            total_seconds=time.perf_counter() - t0,
                            scoring_passes=report["ledger"],
                            training_report=json.loads(
                                (root / stem / "models/manifest.json").read_text()
                            ).get("run_report"),
                        )
                    )
                except (
                    ValueError,
                    ArithmeticError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ) as e:
                    for name in (
                        "learned_direction",
                        "additive_PGS_direction",
                        "regional_kernel",
                        "supplied_burden",
                    ):
                        if not any(
                            r["setting"] == setting
                            and r["replicate"] == rep
                            and r["method"] == name
                            for r in records
                        ):
                            records.append(
                                dict(
                                    setting=setting,
                                    method=name,
                                    replicate=rep,
                                    p=np.nan,
                                    failed=True,
                                    error=str(e),
                                )
                            )
            print(
                setting, "finished", round(time.perf_counter() - start, 2), flush=True
            )
    table = reduction(records, args.out)
    (args.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    panel=meta,
                    seed=args.phenotype_seed,
                    n_train=n0,
                    n_test=n1,
                    m=m,
                    replicates=args.replicates,
                    fixed="genotypes, supplied 64-SNP region, main-effect policy, ridge prior; no target discovery or phenotype tuning",
                    regenerated="dense additive and interaction effects, training/test residuals, learned direction and additive PGS, test nuisance mean and HC3",
                    estimand="conditional signed direction association; fitted PGS is not guaranteed to remove all additive mean",
                    comparison="same training/test individuals and prior information; regional kernel covers all 64 supplied alternatives; learned directions test one frozen alternative",
                    intervals="coefficient SE saved; no known scalar truth for a learned direction under misspecified polygenic baseline, so coverage omitted",
                    seconds=time.perf_counter() - start,
                    cpu_seconds=time.process_time() - cpu,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    timings=timings,
                    native_path=native_module().__file__,
                    native_build=native_module().build_info(),
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
