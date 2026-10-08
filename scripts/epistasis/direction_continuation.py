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
    parser.add_argument("--nested-draws", type=int, default=0)
    parser.add_argument("--training-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--variants", type=int, default=4096)
    parser.add_argument("--genotype-seed", type=int, default=642081)
    parser.add_argument("--phenotype-seed", type=int, default=440763)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100:
        raise ValueError("at most 100 full refits per setting")
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
        # Frozen training imputation for every simulated causal and tested
        # additive coordinate; observed values retain the common panel units.
        full_mean = np.nanmean(raw, axis=0)
        training_mean = np.nanmean(raw[:n0], axis=0)
        inverse = 1 / np.sqrt(full_mean * (1 - full_mean / 2))
        x = (np.where(np.isnan(raw), training_mean, raw) - full_mean) * inverse
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
        for setting, strength, architecture in [
            ("null", 0.0, "mixed"),
            ("mixed_weak", 0.002, "mixed"),
            ("mixed_moderate", 0.01, "mixed"),
            ("aligned_weak", 0.002, "aligned"),
        ]:
            if args.nested_draws and setting != "null":
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
                            main_variants=[f"v{i}" for i in range(88)],
                        )
                    public_spec = dict(
                        kind="summit.epistasis.prepare",
                        schema_version=1,
                        genotypes=dict(geno="input.bed"),
                        samples="test.tsv",
                        phenotypes=dict(file=pheno, columns=["y"], unit="simulated"),
                        annotations=dict(target=dict(v0=1)),
                        frozen_scores=definitions,
                        jobs=[
                            dict(
                                id="learned",
                                additive_annotations=["all"],
                                local_variants=[f"v{i}" for i in range(88)],
                                dominance_variants=[f"v{i}" for i in range(24)],
                                components=[
                                    dict(
                                        name="direction",
                                        frozen_score="interaction",
                                        background="target",
                                    )
                                ],
                                inference=dict(
                                    method="robust_mean",
                                    main_effects="declared",
                                    save_reference=True,
                                ),
                            )
                        ],
                    )
                    public_path = root / (stem + "-confirm.json")
                    public_path.write_text(json.dumps(public_spec))
                    prepared = root / (stem + "-prepared")
                    epistasis_main(
                        ["prepare", str(public_path), "--out", str(prepared)]
                    )
                    from summit.epistasis.features import load_feature_reference

                    reference_path = prepared / "learned.cohort-reference.npz"
                    with np.load(reference_path, allow_pickle=False) as archive:
                        reference_identity = json.loads(str(archive["manifest"]))[
                            "metadata"
                        ]["compatibility_id"]
                    reference = load_feature_reference(
                        reference_path, compatibility_id=reference_identity
                    )
                    # Identical nuisance information for every comparison,
                    # including all region main effects and both learned scores.
                    fixed = reference.fixed_effects
                    analysis_y = phenotype[n0:]
                    if args.nested_draws:
                        inner = np.random.default_rng(args.phenotype_seed + 19001 + rep)
                        analysis_y = mean[n0:, None] + sd[n0:, None] * inner.normal(
                            size=(n1, args.nested_draws)
                        )
                    # Saved real confirmation targets have no missing calls.
                    # The common-panel and public study target columns differ
                    # affinely; the score itself is in C, so their scalar tests
                    # agree after projection. Regional/burden alternatives keep
                    # their explicitly declared common-panel feature weights.
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
                            analysis_y,
                            fixed,
                            feature_names=tuple(
                                f"f{i}" for i in range(features.shape[1])
                            ),
                            trait_names=tuple(map(str, range(args.nested_draws or 1))),
                            metadata={},
                        )
                        for inner_rep in range(args.nested_draws or 1):
                            fit = robust_score_tests(
                                summary,
                                trait=inner_rep,
                                burden=np.ones(features.shape[1]),
                            )
                            records.append(
                                dict(
                                    setting=setting,
                                    method=name,
                                    replicate=inner_rep if args.nested_draws else rep,
                                    outer_replicate=rep,
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
                    nested_draws=args.nested_draws,
                    fixed="genotypes, supplied 64-SNP region, main-effect policy, ridge prior; no target discovery or phenotype tuning",
                    regenerated="full: dense additive and interaction effects, training/test residuals, learned direction and additive PGS, test nuisance mean and HC3; nested: true realized mean and trained direction fixed, confirmation residuals regenerated and full mean/HC3 refitted",
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
