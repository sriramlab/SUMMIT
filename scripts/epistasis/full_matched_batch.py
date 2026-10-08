"""Batched full-marker, matched learning experiments through public commands.

Outcome-dependent fits remain separate. Genotype decoding and affine products
are shared only when sample masks, scales and variant axes agree. Generating
means are simulation/diagnostic inputs, never fitting inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from summit.epistasis.cli import main as cli, _jsonable
from summit.epistasis.prepare import fit_scale
from summit.epistasis.robust import load_robust_scores
from summit.prediction.genotype import FileGenotypeSource, native_module
from summit.prediction.runtime import configure_prediction_threads
from scripts.epistasis.full_matched import write_json, diagnosis, load_reference, load_experiment_inputs, training_variant_input
from scripts.epistasis.benchmark_robust_workflow import io


def _genotypes_unavailable(*args, **kwargs):
    raise AssertionError("genotype access during portable fitting or phenotype reuse")


def run(a):
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    reference = a.reference.resolve()
    meta, data, samples = load_experiment_inputs(reference,
        local_stress="local_withheld" in a.settings, threads=a.num_threads)
    variant_input = training_variant_input(reference, root, meta)
    n0, n1 = a.training_samples, a.confirmation_samples
    if n0 <= 0 or n1 <= 0 or n0 + n1 > len(samples):
        raise ValueError("need distinct actual training and confirmation participants")
    if a.batch_size < 1 or a.replicates < 1 or a.replicates > 100:
        raise ValueError("positive batch size and 1..100 full replicates required")
    if not np.isfinite(a.signal_multiplier) or a.signal_multiplier <= 0:
        raise ValueError("signal multiplier must be fixed, finite and positive")
    settings = a.settings.split(",")
    if len(set(settings)) != len(settings) or set(settings) - set(meta["settings"]):
        raise ValueError("settings must be distinct reference scenarios")
    methods = getattr(a, "methods", "learned,burden,oracle,joint").split(",")
    if (len(set(methods)) != len(methods) or not set(methods) <= {"learned","burden","oracle","joint"}
            or a.primary not in methods):
        raise ValueError("methods must be distinct declared comparisons including the prespecified primary")
    order = np.random.default_rng(a.panel_seed).permutation(len(samples))
    train, test = np.sort(order[:n0]), np.sort(order[n0:n0+n1])
    samples.iloc[train].to_csv(root / "training.tsv", sep="\t", index=False)
    samples.iloc[test].to_csv(root / "confirmation.tsv", sep="\t", index=False)
    block_size = getattr(a, "block_size", 128)
    if block_size < 1:
        raise ValueError("positive genotype block size required")
    controls = ["--num-threads", str(a.num_threads), "--block-size", str(block_size),
                "--memory-gib", str(a.memory_gib)]
    measurements = []
    overall = time.perf_counter()

    def measure(stage, call):
        start, cpu, before = time.perf_counter(), time.process_time(), io()
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        status = "failed"
        try:
            result = call()
            status = "completed"
            return result
        finally:
            after, child_after = io(), resource.getrusage(resource.RUSAGE_CHILDREN)
            measurements.append(dict(stage=stage, status=status, seconds=time.perf_counter()-start,
                cpu_seconds=time.process_time()-cpu,
                child_cpu_seconds=child_after.ru_utime+child_after.ru_stime-child.ru_utime-child.ru_stime,
                child_peak_rss_bytes=child_after.ru_maxrss*1024,
                child_input_blocks=child_after.ru_inblock-child.ru_inblock,
                child_output_blocks=child_after.ru_oublock-child.ru_oublock,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                io={k: after[k]-before[k] for k in before}))
            print(stage, status, round(measurements[-1]["seconds"], 2), flush=True)

    native = native_module()
    configure_prediction_threads(native, a.num_threads)
    weights = None
    if set(methods) - {"learned"}:
        with FileGenotypeSource(meta["genotypes"], genome_build="GRCh37") as source:
            scale = measure("diagnostic_oracle_scale", lambda: fit_scale(source, data["rows"][test],
                threads=a.num_threads, block_size=block_size, memory_bytes=int(a.memory_gib*2**30)))
            weights = data["oracle_weights"] * data["reference_inverse_scale"][:, None] / scale.inverse_scale[data["background"], None]
    covariates = dict(file=str(reference / "covariates.tsv"), columns=meta["covariates"])
    if a.structure_covariates:
        covariates["varying_effects"] = a.structure_covariates.split(",")
    write_json(root / "design.json", dict(
        reference=str(reference), phase=a.phase,
        scenario_definitions={s: meta["definitions"][s] for s in settings},
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        sampling="fixed intact actual genotype rows; fresh independent errors in every full matched replicate",
        conditioning="truth evaluated separately for each learned direction and its confirmation nuisance span",
        primary=a.primary, thresholds=[0.05, 0.005], material_inflation_tolerances=[0.075, 0.01],
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    ))
    all_records = []
    for setting in settings:
        index = meta["settings"].index(setting)
        signal = a.signal_multiplier * data["signals"][:, index]
        mean = data["means"][:, index] - data["signals"][:, index] + signal
        oracle_column = {"aligned": 0, "mixed": 1, "sparse_interaction": 2}[meta["definitions"][setting]["direction"]]
        for begin in range(0, a.replicates, a.batch_size):
            reps = list(range(begin, min(a.replicates, begin+a.batch_size)))
            columns = [f"rep{rep:03d}" for rep in reps]
            work = root / f"{setting}_{begin:03d}"
            work.mkdir()
            outcomes = samples.copy()
            for rep, column in zip(reps, columns):
                rng = np.random.default_rng(np.random.SeedSequence([a.seed, index, rep]))
                error = rng.standard_t(5, len(samples))/np.sqrt(5/3) if setting == "heavy" else rng.normal(size=len(samples))
                outcomes[column] = mean + np.sqrt(data["variance"])*error
            outcomes.to_csv(work / "phenotypes.tsv", sep="\t", index=False)
            phenotype = dict(file="phenotypes.tsv", columns=columns, unit="fixed reference units")
            training = dict(kind="summit.epistasis.train_direction", schema_version=1,
                genotypes=dict(geno=meta["genotypes"], genome_build="GRCh37"),
                samples=str(root / "training.tsv"), phenotypes=phenotype,
                target=meta["target"], variants=str(variant_input),
                interaction_variants=str(reference / "interaction_variants.txt"), trans_only=True,
                covariates=covariates, local_variants=meta["local_variants"],
                dominance_variants=meta["local_variants"],
                prior=dict(additive=0.5, interaction=0.05, residual=1.0), storage="packed",
                solver=dict(rtol=1e-6, max_iterations=200))
            write_json(work / "train.json", training)
            jobs, scores = [], []
            for i, (rep, column) in enumerate(zip(reps, columns)):
                pgs, direction = f"PGS{rep}", f"direction{rep}"
                scores.extend([dict(name=pgs, direction=f"trained/direction.{i}.json", component=0),
                               dict(name=direction, direction=f"trained/direction.{i}.json", component=1)])
                learned = dict(name="learned", frozen_score=direction, background="target")
                burden = None if weights is None else dict(name="burden", score=dict(zip(meta["background_variants"], weights[:, 0])), background="target")
                oracle = None if weights is None else dict(name="oracle", score=dict(zip(meta["background_variants"], weights[:, oracle_column])), background="target")
                for method, components in [("learned", [learned]), ("burden", [burden]),
                                           ("oracle", [oracle]), ("joint", [learned, burden])]:
                    if method not in methods:
                        continue
                    jobs.append(dict(id=f"{column}_{method}", phenotypes=[column], adjust_scores=[pgs],
                        additive_annotations=["all"], trans_target=meta["target"], components=components,
                        local_variants=meta["local_variants"], dominance_variants=meta["local_variants"],
                        inference=dict(method="robust_mean", main_effects="declared", save_reference=True,
                                       sampling_model=a.sampling_model)))
            preparation = dict(kind="summit.epistasis.prepare", schema_version=1,
                genotypes=training["genotypes"], samples=str(root / "confirmation.tsv"),
                phenotypes=phenotype, covariates=covariates,
                annotations={"target": {meta["target"]: 1}}, frozen_scores=scores, jobs=jobs)
            write_json(work / "prepare.json", preparation)
            batch_records = []
            try:
                if getattr(a, "recover_models_from", None):
                    from scripts.epistasis.repack_prediction_artifact import repack
                    previous = a.recover_models_from / work.name / "trained/models"
                    converted = measure(f"{setting}/{begin}/recover_completed_models",
                        lambda: repack(previous, work / "trained/models", legacy_limit_bytes=512*2**20))
                    write_json(work / "model_recovery.json", converted)
                if a.interrupt_training and setting == settings[0] and begin == 0:
                    code = Path(__file__).resolve().parents[2]
                    launcher = code / ("scripts/generalized_gxe/private_python.py"
                        if native.build_info().get("private_blas_backend") == "upstream_blis"
                        else "scripts/epistasis/checkout_python.py")
                    cpus = getattr(sys.modules.get("workflow"), "_PRE_NUMERICAL_CPU_AFFINITY", tuple(sorted(os.sched_getaffinity(0))))
                    command = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable, str(launcher),
                               "scripts.epistasis.checkpoint_stop", "train-direction", str(work / "train.json"),
                               "--out", str(work / "trained"), *controls]
                    stopped = measure("hard_checkpoint_interruption", lambda: subprocess.run(command, check=False))
                    if stopped.returncode != 75 or not (work / "trained/solver.npz").exists():
                        raise RuntimeError("requested actual checkpoint interruption was not observed")
                resume = ["--resume"] if (work / "trained").exists() else []
                measure(f"{setting}/{begin}/train", lambda: cli(["train-direction", str(work / "train.json"),
                    "--out", str(work / "trained"), *controls, *resume]))
                measure(f"{setting}/{begin}/prepare", lambda: cli(["prepare", str(work / "prepare.json"),
                    "--out", str(work / "prepared"), *controls]))
                import summit.prediction.genotype as genotype
                for rep, column in zip(reps, columns):
                    for method in methods:
                        job = f"{column}_{method}"
                        artifact = work / f"prepared/{job}.robust-score.npz"
                        with patch.object(genotype, "source_from_spec", _genotypes_unavailable), patch.object(genotype, "FileGenotypeSource", _genotypes_unavailable):
                            cli(["fit", str(artifact), "--out", str(work / f"{job}.fit.json")])
                        fit = json.loads((work / f"{job}.fit.json").read_text())["fits"][0]
                        ref = load_reference(work / f"prepared/{job}.cohort-reference.npz")
                        batch_records.append(dict(setting=setting, replicate=rep, method=method,
                            failed=False, biological_null=meta["definitions"][setting]["biological_null"],
                            p=fit["joint_p"] if method == "joint" else fit["kernel_p"],
                            outside_scope=fit["diagnostics"]["outside_confirmation_design"],
                            max_leverage=fit["diagnostics"]["max_leverage"],
                            reference_signal_variance=meta["definitions"][setting]["reference_signal_variance"]*a.signal_multiplier**2,
                            realized_training_signal_variance=float(signal[train].var()),
                            realized_confirmation_signal_variance=float(signal[test].var()),
                            **diagnosis(ref, fit, mean[test], signal[test], data["variance"][test])))
                # Preparation with individual references remains cohort-side.
                # A second phenotype reuses the direction, not its fitted meat.
                reuse = dict(kind="summit.epistasis.prepare_traits", schema_version=1,
                    reference=f"prepared/{columns[0]}_{a.primary}.cohort-reference.npz",
                    samples=preparation["samples"], phenotypes=phenotype)
                write_json(work / "reuse.json", reuse)
                with patch.object(genotype, "source_from_spec", _genotypes_unavailable), patch.object(genotype, "FileGenotypeSource", _genotypes_unavailable):
                    measure(f"{setting}/{begin}/prepare_traits", lambda: cli(["prepare-traits", str(work / "reuse.json"),
                        "--out", str(work / "reuse.npz"), "--num-threads", str(a.num_threads), "--memory-gib", str(a.memory_gib)]))
                    cli(["fit", str(work / "reuse.npz"), "--out", str(work / "reuse.fit.json")])
                reused = load_robust_scores(work / "reuse.npz")
                original = load_robust_scores(work / f"prepared/{columns[0]}_{a.primary}.robust-score.npz")
                np.testing.assert_allclose(reused.information, original.information, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(reused.scores[:, :1], original.scores, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(reused.score_covariance[:1], original.score_covariance, rtol=1e-10, atol=1e-10)
                model = json.loads((work / "trained/models/manifest.json").read_text())
                write_json(work / "measurement.json", dict(native=model["run_report"],
                    convergence=[t["models"][0]["convergence"] for t in model["traits"]],
                    genotype_disabled_summary_fitting=True, genotype_disabled_phenotype_reuse=True,
                    first_trait_reproduction_verified=True))
            except (ValueError, ArithmeticError, RuntimeError, AssertionError, MemoryError) as error:
                batch_records = [dict(setting=setting, replicate=rep, method=method, failed=True, error=str(error))
                    for rep in reps for method in methods]
                print("FAILED", setting, begin, repr(error), flush=True)
            all_records.extend(batch_records)
            with (root / "replicates.jsonl").open("a") as handle:
                for record in batch_records:
                    handle.write(json.dumps(_jsonable(record))+"\n")
            write_json(work / "resources.json", dict(records=[r for r in measurements if r["stage"].startswith(f"{setting}/{begin}/")],
                cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    pd.DataFrame(all_records).to_csv(root / "replicates.csv", index=False)
    write_json(root / "resources.json", dict(records=measurements, seconds=time.perf_counter()-overall,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--training-samples", type=int, default=2048)
    p.add_argument("--confirmation-samples", type=int, default=4096)
    p.add_argument("--panel-seed", type=int, default=857491)
    p.add_argument("--seed", type=int, default=942873)
    p.add_argument("--replicates", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--settings", default="dense,structure,mixed,structure_mixed")
    p.add_argument("--structure-covariates", default="")
    p.add_argument("--phase", choices=["development", "confirmation"], default="development")
    p.add_argument("--primary", choices=["learned", "joint"], default="learned")
    p.add_argument("--methods", default="learned,burden,oracle,joint")
    p.add_argument("--sampling-model", choices=["fixed_design_correct_mean", "iid_population_projection"], default="fixed_design_correct_mean")
    p.add_argument("--signal-multiplier", type=float, default=1.0)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--memory-gib", type=float, default=16)
    p.add_argument("--interrupt-training", action="store_true")
    p.add_argument("--recover-models-from", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
