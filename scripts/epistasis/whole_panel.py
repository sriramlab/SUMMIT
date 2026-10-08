"""Complete whole-marker real-BED workflow; new private cohort outputs only.

The HMGA2 target rs1042725 (GRCh37 12:66358347 C/T) is fixed from published
height main-effect literature. Its interaction is a supplied pilot hypothesis,
not an established association or a target selected on these phenotypes.
"""
import argparse, json, time, resource, os, sys, subprocess, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from summit.prediction.genotype import (
    FileGenotypeSource,
    StandardizedBlock,
    native_module,
)
from summit.epistasis.prepare import fit_scale
from summit.epistasis.cli import main as cli, _jsonable
from scripts.epistasis.benchmark_robust_workflow import io


def main():
    overall_start = time.perf_counter()
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--memory-gib", type=float, default=8)
    p.add_argument("--training-samples", type=int, default=6000)
    p.add_argument("--observed-phenotype", type=Path)
    p.add_argument("--covariates", type=Path)
    p.add_argument("--samples", type=int)
    p.add_argument("--trans", action="store_true")
    p.add_argument("--local-min-cell", type=int, default=0)
    p.add_argument("--interrupt-training", action="store_true")
    p.add_argument(
        "--sampling-model",
        default="fixed_design_correct_mean",
        choices=("fixed_design_correct_mean", "iid_population_projection"),
    )
    a = p.parse_args()
    if not np.isfinite(a.memory_gib) or a.memory_gib <= 0:
        raise ValueError("memory budget must be finite and positive")
    driver_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=False)
    a.out.chmod(0o700)
    root = a.out.resolve()
    launch_cpus = getattr(
        sys.modules.get("workflow"),
        "_PRE_NUMERICAL_CPU_AFFINITY",
        tuple(sorted(os.sched_getaffinity(0))),
    )
    launcher = Path(__file__).resolve().parents[2] / (
        "scripts/generalized_gxe/private_python.py"
        if native_module().build_info().get("private_blas_backend") == "upstream_blis"
        else "scripts/epistasis/checkout_python.py"
    )
    child_prefix = [
        "taskset",
        "-c",
        ",".join(map(str, launch_cpus)),
        sys.executable,
        str(launcher),
    ]
    records = []
    controls = [
        "--num-threads",
        str(a.num_threads),
        "--memory-gib",
        str(a.memory_gib),
        "--block-size",
        "128",
    ]

    def run(stage, call):
        t = time.perf_counter()
        cpu = time.process_time()
        before = io()
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        value = call()
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        after = io()
        records.append(
            dict(
                stage=stage,
                seconds=time.perf_counter() - t,
                cpu_seconds=time.process_time() - cpu,
                child_cpu_seconds=children_after.ru_utime
                + children_after.ru_stime
                - children.ru_utime
                - children.ru_stime,
                child_peak_rss_bytes=children_after.ru_maxrss * 1024,
                child_input_blocks=children_after.ru_inblock - children.ru_inblock,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                io={k: after[k] - before[k] for k in before},
            )
        )
        (root / "workload_progress.json").write_text(
            json.dumps(_jsonable(records), indent=2)
        )
        print(stage, round(records[-1]["seconds"], 2), flush=True)
        return value

    rng = np.random.default_rng(890641)
    with FileGenotypeSource(a.genotypes, genome_build="GRCh37") as source:
        native = native_module()
        from summit.prediction.runtime import configure_prediction_threads

        configure_prediction_threads(native, a.num_threads)
        source_n = len(source.samples)
        n = a.samples or source_n
        if not a.training_samples < n <= source_n or a.local_min_cell < 0:
            raise ValueError(
                "need 0 < training samples < selected samples <= source cohort and nonnegative cell threshold"
            )
        m = len(source.variants.ids)
        rows = (
            np.sort(np.random.default_rng(910583).choice(source_n, n, replace=False))
            if a.samples
            else np.arange(n)
        )
        target = "12:66358347"
        j = source.variants.ids.index(target)
        target_alleles = (source.variants.counted[j], source.variants.other[j])
        if set(target_alleles) != {"C", "T"}:
            raise ValueError("unexpected HMGA2 alleles")
        if a.sampling_model == "iid_population_projection":
            # Select the complete target mask before outcomes and retain entire
            # source rows. Background missingness is preserved by native scoring.
            source.prepare(np.arange(source_n), 1, a.num_threads)
            complete = np.flatnonzero(source.read(np.array([j]))[:, 0] != -127)
            if a.samples is None:
                n = len(complete)
            if n > len(complete):
                raise ValueError(
                    "requested population workload exceeds complete-target rows"
                )
            if a.training_samples >= n:
                raise ValueError("confirmation needs rows outside training")
            rows = np.sort(
                np.random.default_rng(910583).choice(complete, n, replace=False)
            )
        scale = run(
            "whole_panel_genotype_scale",
            lambda: fit_scale(
                source,
                rows,
                threads=a.num_threads,
                block_size=128,
                memory_bytes=int(a.memory_gib * 2**30),
            ),
        )
        # Local +/-100kb is specified before outcomes. Retain ALL variants,
        # with dominance, rather than phenotype-dependent pruning.
        local = np.flatnonzero(
            (np.asarray(source.variants.chromosome) == "12")
            & (abs(np.asarray(source.variants.position) - 66358347) <= 100000)
        )
        source.prepare(rows, 128, a.num_threads)
        standard = StandardizedBlock(native, a.num_threads)
        gx = []
        raw_local = []
        for lo in range(0, len(local), 128):
            take = local[lo : lo + 128]
            raw = source.read(take)
            raw_local.append(raw.copy())
            gx.append(
                standard.prepare(
                    raw,
                    np.arange(n),
                    np.arange(len(take)),
                    scale.mean[take],
                    scale.inverse_scale[take],
                ).copy()
            )
        xlocal = np.column_stack(gx)
        raw = np.column_stack(raw_local)
        del gx, raw_local
        local_candidate_count = len(local)
        if a.local_min_cell:
            support = (
                np.min(np.stack([(raw == g).sum(0) for g in (0, 1, 2)]), axis=0)
                >= a.local_min_cell
            )
            if not support[np.flatnonzero(local == j)[0]]:
                raise ValueError(
                    "supplied target fails the prespecified genotype-cell rule"
                )
            local, raw, xlocal = local[support], raw[:, support], xlocal[:, support]
        target_x = xlocal[:, np.flatnonzero(local == j)[0]]
        local_names = [source.variants.ids[i] for i in local]
        d = (raw == 1).astype(float)
        obs = raw != -127
        d = np.where(obs, d, (d.sum(0) / obs.sum(0))[None, :])
        dense = np.zeros(n)
        background = np.zeros(n)
        interaction_members = (
            np.asarray(source.variants.chromosome) != "12"
            if a.trans
            else np.arange(m) != j
        )
        interaction_mass = int(interaction_members.sum())
        effects = rng.normal(size=m) * np.sqrt(0.5 / m)
        start = time.perf_counter()
        source.prepare(rows, 128, a.num_threads)
        for lo in range(0, m, 128):
            take = np.arange(lo, min(m, lo + 128))
            xx = standard.prepare(
                source.read(take),
                np.arange(n),
                np.arange(len(take)),
                scale.mean[take],
                scale.inverse_scale[take],
            )
            weights = np.column_stack(
                [
                    effects[take],
                    np.where(
                        interaction_members[take], 1.0 / np.sqrt(interaction_mass), 0.0
                    ),
                ]
            )
            product = np.empty((n, 2), order="F")
            native.prediction_product(
                np.asfortranarray(xx),
                np.asfortranarray(weights),
                product,
                False,
                a.num_threads,
            )
            dense += product[:, 0]
            background += product[:, 1]
        records.append(
            dict(
                stage="simulated_mean_whole_marker_pass",
                seconds=time.perf_counter() - start,
            )
        )
        localmean = (
            xlocal @ rng.normal(size=len(local)) * 0.1
            + 0.4 * d[:, np.flatnonzero(local == j)[0]]
        )
        noise = rng.normal(size=(n, 3)) * np.sqrt(0.4 + 0.6 * target_x[:, None] ** 2)
        y = np.column_stack(
            [
                localmean + noise[:, 0],
                localmean + dense + noise[:, 1],
                localmean + dense + 0.03 * target_x * background + noise[:, 2],
            ]
        )
        samples = [source.samples[int(i)] for i in rows]
        axis = source.variants.ids
        source_identity = source.identity
    traits = ["finite_null", "polygenic_null", "direction_signal"]
    frame = pd.DataFrame(y, columns=traits)
    frame.insert(0, "IID", [s[1] for s in samples])
    frame.insert(0, "FID", [s[0] for s in samples])
    frame.to_csv(root / "simulated.tsv", sep=" ", index=False)
    sample = pd.DataFrame(samples, columns=["FID", "IID"])
    sample.to_csv(root / "all.tsv", sep=" ", index=False)
    chosen = np.sort(rng.choice(n, a.training_samples, replace=False))
    heldout = np.setdiff1d(np.arange(n), chosen)
    sample.iloc[chosen].to_csv(root / "training.tsv", sep=" ", index=False)
    sample.iloc[heldout].to_csv(root / "confirmation.tsv", sep=" ", index=False)
    (root / "background.txt").write_text(
        "\n".join(v for v in axis if v != target) + "\n"
    )
    region = (
        [v for v, keep in zip(axis, interaction_members) if keep]
        if a.trans
        else [v for v in local_names if v != target]
    )
    (root / "region.txt").write_text("\n".join(region) + "\n")
    job = dict(
        id="prespecified",
        additive_annotations=["all"],
        local_variants=local_names,
        dominance_variants=local_names,
        components=[
            dict(
                name="direction",
                score={
                    v: 1 / np.sqrt(interaction_mass)
                    for v, keep in zip(axis, interaction_members)
                    if keep
                },
                background="target",
            )
        ],
        inference=dict(
            method="robust_mean",
            main_effects="declared",
            sampling_model=a.sampling_model,
        ),
    )
    if a.trans:
        job["trans_target"] = target
        job["inference"]["save_reference"] = True
    spec = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno=str(Path(a.genotypes).resolve()), genome_build="GRCh37"),
        samples="all.tsv",
        phenotypes=dict(file="simulated.tsv", columns=traits, unit="simulated"),
        annotations=dict(target={target: 1}),
        jobs=[job],
    )
    path = root / "prespecified.json"
    path.write_text(json.dumps(spec))
    run(
        "public_prespecified_prepare_3_traits",
        lambda: cli(
            ["prepare", str(path), "--out", str(root / "prespecified"), *controls]
        ),
    )
    run(
        "validated_restart",
        lambda: cli(
            [
                "prepare",
                str(path),
                "--out",
                str(root / "prespecified"),
                "--resume",
                *controls,
            ]
        ),
    )
    run(
        "saved_fit",
        lambda: cli(
            [
                "fit",
                str(root / "prespecified/prespecified.robust-score.npz"),
                "--out",
                str(root / "prespecified.fit.json"),
            ]
        ),
    )
    # Fresh process receives only the artifact and cannot open cohort inputs.
    standalone = root / "summary_only"
    standalone.mkdir()
    import shutil

    shutil.copy2(
        root / "prespecified/prespecified.robust-score.npz", standalone / "scores.npz"
    )
    # Use the qualified checkout launcher, then run the stand-alone program.
    run(
        "fresh_process_summary_only",
        lambda: subprocess.run(
            child_prefix
            + [
                "scripts.epistasis.summary_only_fit",
                "fit",
                str(standalone / "scores.npz"),
                "--out",
                str(standalone / "fit.json"),
            ],
            check=True,
        ),
    )
    train = dict(
        kind="summit.epistasis.train_direction",
        schema_version=1,
        genotypes=spec["genotypes"],
        samples="training.tsv",
        phenotype=dict(
            file="simulated.tsv", column="direction_signal", unit="simulated"
        ),
        target=target,
        variants="background.txt",
        interaction_variants="region.txt",
        local_variants=local_names,
        dominance_variants=local_names,
        prior=dict(additive=0.5, interaction=0.05, residual=1.0),
        storage="packed",
        solver=dict(rtol=1e-6, max_iterations=150),
    )
    if a.trans:
        train["trans_only"] = True
    (root / "train.json").write_text(json.dumps(train))
    if a.interrupt_training:

        def interrupted():
            result = subprocess.run(
                child_prefix
                + [
                    "scripts.epistasis.checkpoint_stop",
                    "train-direction",
                    str(root / "train.json"),
                    "--out",
                    str(root / "trained"),
                    *controls,
                ]
            )
            if result.returncode != 75:
                raise RuntimeError(
                    f"checkpoint interruption returned unexpected exit {result.returncode}"
                )
            with np.load(root / "trained/solver.npz", allow_pickle=False) as saved:
                checkpoint = json.loads(saved["metadata"].tobytes())
            if checkpoint["iteration"] < 2 or (root / "trained/models").exists():
                raise RuntimeError("did not interrupt an incomplete training fit")
            (root / "interruption.json").write_text(
                json.dumps(
                    dict(
                        iteration=checkpoint["iteration"],
                        identity=checkpoint["identity"],
                        exit=75,
                    ),
                    indent=2,
                )
            )

        run("hard_process_interruption_after_atomic_checkpoint", interrupted)
    run(
        "complete_native_direction_training",
        lambda: cli(
            [
                "train-direction",
                str(root / "train.json"),
                "--out",
                str(root / "trained"),
                *controls,
                *(["--resume"] if a.interrupt_training else []),
            ]
        ),
    )
    frozen = [
        dict(name="PGS", direction="trained/direction.json", component=0, adjust=True),
        dict(name="learned", direction="trained/direction.json", component=1),
    ]
    learned = dict(
        job,
        id="learned",
        components=[dict(name="learned", frozen_score="learned", background="target")],
    )
    confirm = dict(
        spec, samples="confirmation.tsv", frozen_scores=frozen, jobs=[learned]
    )
    (root / "confirm.json").write_text(json.dumps(confirm))
    run(
        "independent_confirmation",
        lambda: cli(
            [
                "prepare",
                str(root / "confirm.json"),
                "--out",
                str(root / "confirmed"),
                *controls,
            ]
        ),
    )
    cli(
        [
            "fit",
            str(root / "confirmed/learned.robust-score.npz"),
            "--out",
            str(root / "confirmed.fit.json"),
        ]
    )
    payload = dict(
        n=n,
        source_n=source_n,
        m=m,
        training_n=len(chosen),
        confirmation_n=len(heldout),
        local_variants=len(local),
        local_candidates=local_candidate_count,
        minimum_local_genotype_cell=a.local_min_cell,
        trans=a.trans,
        sampling_model=a.sampling_model,
        memory_gib=a.memory_gib,
        interaction_markers=interaction_mass,
        target_counted_and_other_alleles=target_alleles,
        score_orientation="uniform signed weights in this source's recorded counted-allele units; not a cross-cohort harmonized burden",
        interruption_checked=a.interrupt_training,
        driver_sha256=driver_sha256,
        source_identity=source_identity,
        records=records,
        seed=890641,
        native_path=native.__file__,
        native_build=native.build_info(),
        hypothesis=target
        + (
            " times a score excluding chromosome 12; additive background retains chromosome 12"
            if a.trans
            else " times prespecified uniform standardized score, or independent learned local direction"
        ),
        observed="not requested",
        statistical_scope="workload and finite-mean pilot; polygenic null and learned direction are experimental",
        root=str(root),
        primary=json.loads((root / "prespecified.fit.json").read_text()),
        confirmation=json.loads((root / "confirmed.fit.json").read_text()),
    )
    if a.observed_phenotype:
        if not a.covariates:
            raise ValueError("observed height requires declared covariates")
        ph = pd.read_csv(
            a.observed_phenotype,
            sep=r"\s+",
            dtype={"FID": str, "IID": str},
            na_values=["-9", "NA"],
        )
        cv = pd.read_csv(
            a.covariates,
            sep=r"\s+",
            dtype={"FID": str, "IID": str},
            na_values=["-9", "NA"],
        )
        aligned = (
            sample.merge(ph, on=["FID", "IID"], validate="one_to_one")
            .merge(cv, on=["FID", "IID"], validate="one_to_one")
            .dropna()
        )
        columns = list(cv.columns[2:])
        aligned[["FID", "IID"]].to_csv(
            root / "height_samples.tsv", sep=" ", index=False
        )
        observed = dict(
            spec,
            samples="height_samples.tsv",
            phenotypes=dict(
                file=str(a.observed_phenotype.resolve()),
                columns=["height"],
                unit="legacy EUR height inverse-normal score",
            ),
            covariates=dict(file=str(a.covariates.resolve()), columns=columns),
        )
        (root / "height.json").write_text(json.dumps(observed))
        run(
            "observed_height_prepare",
            lambda: cli(
                [
                    "prepare",
                    str(root / "height.json"),
                    "--out",
                    str(root / "height"),
                    *controls,
                ]
            ),
        )
        cli(
            [
                "fit",
                str(root / "height/prespecified.robust-score.npz"),
                "--out",
                str(root / "height.fit.json"),
            ]
        )
        payload.update(
            observed=dict(
                n=len(aligned),
                covariates=columns,
                definition="existing EUR.height.txt; no new outcome transformation, no phenotype-based target selection; legacy phenotype provenance limits causal interpretation",
                results=json.loads((root / "height.fit.json").read_text()),
            )
        )
        if a.trans:
            # A real independently learned score uses the real training trait,
            # never the simulated direction or confirmation outcomes.
            keys = set(map(tuple, aligned[["FID", "IID"]].to_numpy(str)))
            for label, selected in (
                ("height_training", chosen),
                ("height_confirmation", heldout),
            ):
                part = sample.iloc[selected]
                part = part[[tuple(row) in keys for row in part.to_numpy(str)]]
                part.to_csv(root / (label + ".tsv"), sep=" ", index=False)
            height_train = dict(
                train,
                samples="height_training.tsv",
                phenotype=dict(
                    file=str(a.observed_phenotype.resolve()),
                    column="height",
                    unit="legacy EUR height inverse-normal score",
                ),
                covariates=observed["covariates"],
            )
            (root / "height_train.json").write_text(json.dumps(height_train))
            run(
                "observed_height_independent_training",
                lambda: cli(
                    [
                        "train-direction",
                        str(root / "height_train.json"),
                        "--out",
                        str(root / "height_trained"),
                        *controls,
                    ]
                ),
            )
            height_confirm = dict(
                observed,
                samples="height_confirmation.tsv",
                jobs=[learned],
                frozen_scores=[
                    dict(d, direction="height_trained/direction.json") for d in frozen
                ],
            )
            (root / "height_confirm.json").write_text(json.dumps(height_confirm))
            run(
                "observed_height_learned_confirmation",
                lambda: cli(
                    [
                        "prepare",
                        str(root / "height_confirm.json"),
                        "--out",
                        str(root / "height_confirmed"),
                        *controls,
                    ]
                ),
            )
            cli(
                [
                    "fit",
                    str(root / "height_confirmed/learned.robust-score.npz"),
                    "--out",
                    str(root / "height_confirmed.fit.json"),
                ]
            )
            payload["observed"]["independent_direction_confirmation"] = json.loads(
                (root / "height_confirmed.fit.json").read_text()
            )
    payload["total_wall_seconds_before_receipt"] = time.perf_counter() - overall_start
    payload[
        "wall_time_scope"
    ] = "includes setup, table writing, genotype passes, subprocesses, training and fitting; excludes final receipt serialization"
    with a.receipt.open("x") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


if __name__ == "__main__":
    main()
