"""Trait-matched full-marker experiments through the public epistasis pipeline.

Reference means are generated once by streamed native products. They are used
only for simulation and diagnostics, never passed to the fitted procedure.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from summit.epistasis.cli import main as cli, _jsonable
from summit.epistasis.features import load_feature_reference
from summit.epistasis.prepare import fit_scale
from summit.prediction.genotype import (
    FileGenotypeSource,
    StandardizedBlock,
    native_module,
)
from summit.prediction.runtime import configure_prediction_threads


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(_jsonable(value), handle, indent=2)


def load_experiment_inputs(reference_dir, *, local_stress=False, threads=1):
    """Read frozen teachers and add an exact finite-mean positive control."""
    meta = json.loads((reference_dir / "reference.json").read_text())
    with np.load(reference_dir / "reference.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    samples = pd.read_csv(reference_dir / "samples.tsv", sep="\t", dtype=str)
    from summit.prediction.cli import _aligned_table
    pc = _aligned_table(reference_dir / "covariates.tsv", list(samples.itertuples(index=False, name=None)))["PC1"].to_numpy(float)
    finite = .5 * data["target"] + .2 * pc
    meta["settings"].append("finite")
    meta["definitions"]["finite"] = dict(biological_null=True, direction="aligned",
        reference_signal_variance=0., reference_mean_variance=float(finite.var()),
        mean=".5 reference-scaled target + .2 reference-scaled PC1")
    data["means"] = np.column_stack([data["means"], finite])
    data["signals"] = np.column_stack([data["signals"], np.zeros(len(finite))])
    if local_stress:
        with FileGenotypeSource(meta["genotypes"], genome_build="GRCh37") as source:
            if source.identity != meta["source_identity"]:
                raise ValueError("reference genotype source changed")
            target = source.variants.ids.index(meta["target"])
            positions = np.asarray(source.variants.position)
            chromosome = np.asarray(source.variants.chromosome)
            distance = positions - positions[target]
            candidates = np.flatnonzero((chromosome == chromosome[target]) & (distance >= 200000) & (distance <= 400000))
            meta["prediction_exclusion"] = dict(
                rule="prespecified target+[200,400] kb window unavailable to the additive predictor",
                variants=[source.variants.ids[j] for j in candidates])
            supported = []
            source.prepare(data["rows"], 128, threads)
            for begin in range(0, len(candidates), 128):
                selected = candidates[begin:begin+128]
                raw = source.read(selected)
                counts = np.stack([(raw == k).sum(0) for k in (0, 1, 2)])
                supported.extend(selected[counts.min(0) >= meta["minimum_reference_cell"]])
            if not supported:
                raise ValueError("no supported withheld marker 200..400 kb above the target")
            causal = min(supported, key=lambda j: (abs(distance[j]-250000), positions[j]))
            from summit.prediction.genotype import estimate_scale
            scale = estimate_scale(source, data["rows"], np.array([causal]), threads=threads)
            source.prepare(data["rows"], 1, threads)
            raw = source.read(np.array([causal]))
            gc = StandardizedBlock(native_module(), threads).prepare(raw, np.arange(len(raw)), np.array([0]),
                scale.mean, 1/np.sqrt(scale.mean*(1-scale.mean/2)))[:, 0].copy()
        dense = data["means"][:, meta["settings"].index("dense")]
        mixed = data["signals"][:, meta["settings"].index("mixed")]
        for name, signal in [("local_withheld", np.zeros(len(gc))), ("local_withheld_mixed", mixed)]:
            mean = dense + .8*gc + signal
            meta["settings"].append(name)
            meta["definitions"][name] = dict(biological_null=name == "local_withheld", direction="mixed",
                reference_signal_variance=float(signal.var()), reference_mean_variance=float(mean.var()),
                withheld_variant=source.variants.ids[causal],
                selection="closest supported marker to target+250 kb in [200,400] kb; genotype only",
                local_coefficient=.8, fitted_local_radius_bp=meta["local_radius_bp"])
            data["means"] = np.column_stack([data["means"], mean])
            data["signals"] = np.column_stack([data["signals"], signal])
    return meta, data, samples


def training_variant_input(reference, output, meta):
    """Genotype-only missing-region stress mask, shared by null and alternative."""
    original = reference / "variants.txt"
    excluded = set(meta.get("prediction_exclusion", {}).get("variants", []))
    if not excluded:
        return original
    variants = original.read_text().splitlines()
    selected = [v for v in variants if v not in excluded]
    if not selected or len(selected) == len(variants):
        raise ValueError("withheld window does not intersect the fitted variant axis")
    path = output / "training_variants.txt"
    with path.open("x") as handle:
        handle.write("\n".join(selected)+"\n")
    write_json(output / "prediction_exclusion.json", dict(meta["prediction_exclusion"],
        fitted_marker_count=len(selected), original_fitted_marker_count=len(variants)))
    return path


def reference(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    start = time.perf_counter()
    native = native_module()
    configure_prediction_threads(native, a.num_threads)
    with FileGenotypeSource(a.genotypes, genome_build="GRCh37") as source:
        cov = pd.read_csv(a.covariates, sep=r"\s+", dtype={"FID": str, "IID": str})
        if cov.duplicated(["FID", "IID"]).any():
            raise ValueError("duplicated covariate sample IDs")
        cov = cov.set_index(["FID", "IID"])
        names = list(cov.columns)
        aligned = cov.reindex(pd.MultiIndex.from_tuples(source.samples))
        valid = np.isfinite(aligned.to_numpy(float)).all(1)
        j = source.variants.ids.index(a.target)
        source.prepare(np.arange(len(source.samples)), 1, a.num_threads)
        valid &= source.read(np.array([j]))[:, 0] != -127
        rows = np.flatnonzero(valid)
        if a.reference_samples:
            if a.reference_samples > len(rows):
                raise ValueError("insufficient aligned reference rows")
            rows = np.sort(
                np.random.default_rng(674183).choice(
                    rows, a.reference_samples, replace=False
                )
            )
        c = aligned.iloc[rows].to_numpy(float)
        center = c.mean(0)
        spread = c.std(0)
        if np.any(spread == 0):
            raise ValueError(
                "constant supplied covariate; declare an appropriate column set"
            )
        c = (c - center) / spread
        covout = pd.DataFrame(c, columns=names)
        covout.insert(0, "IID", [source.samples[i][1] for i in rows])
        covout.insert(0, "FID", [source.samples[i][0] for i in rows])
        covout.to_csv(out / "covariates.tsv", sep="\t", index=False)
        scale = fit_scale(
            source,
            rows,
            threads=a.num_threads,
            block_size=128,
            memory_bytes=int(a.memory_gib * 2**30),
        )
        chrom = np.asarray(source.variants.chromosome)
        pos = np.asarray(source.variants.position)
        local = np.flatnonzero((chrom == chrom[j]) & (abs(pos - pos[j]) <= 100000))
        source.prepare(rows, 128, a.num_threads)
        raw = source.read(local)
        cells = np.stack([(raw == k).sum(0) for k in (0, 1, 2)])
        local = local[np.min(cells, axis=0) >= a.minimum_reference_cell]
        if j not in local:
            raise ValueError("target lacks prespecified reference support")
        source.prepare(rows, 128, a.num_threads)
        raw = source.read(local)
        standard = StandardizedBlock(native, a.num_threads)
        xlocal = standard.prepare(
            raw,
            np.arange(len(rows)),
            np.arange(len(local)),
            scale.mean[local],
            scale.inverse_scale[local],
        ).copy()
        target = xlocal[:, list(local).index(j)]
        causal = next((i for i in range(len(local) - 1, -1, -1) if local[i] != j), None)
        if causal is None:
            raise ValueError("need at least one supported nontarget local marker")
        background = np.flatnonzero(chrom == a.background_chromosome)
        if chrom[j] == a.background_chromosome or len(background) < 32:
            raise ValueError("invalid prespecified trans background")
        rng = np.random.default_rng(a.architecture_seed)
        m = len(chrom)
        w = np.zeros((m, 7), order="F")
        w[:, 0] = rng.normal(size=m)
        sparse = rng.choice(m, min(128, m), replace=False)
        w[sparse, 1] = rng.normal(size=len(sparse))
        dominant = rng.choice(m, min(256, m), replace=False)
        w[dominant, 2] = rng.normal(size=len(dominant))
        w[:, 3] = rng.normal(size=m)
        w[background, 4] = 1.0
        w[background, 5] = rng.normal(size=len(background))
        sparse_i = rng.choice(background, min(32, len(background)), replace=False)
        w[sparse_i, 6] = rng.normal(size=len(sparse_i))
        w[j, :] = 0.0
        genetic = np.zeros((len(rows), 7))
        source.prepare(rows, 128, a.num_threads)
        for lo in range(0, m, 128):
            take = np.arange(lo, min(m, lo + 128))
            raw_block = source.read(take)
            x = standard.prepare(
                raw_block,
                np.arange(len(rows)),
                np.arange(len(take)),
                scale.mean[take],
                scale.inverse_scale[take],
            )
            weights = w[take].copy(order="F")
            weights[:, 2] = 0.0
            product = np.empty_like(genetic, order="F")
            native.prediction_product(
                np.asfortranarray(x), weights, product, False, a.num_threads
            )
            genetic += product
            p = scale.mean[take] / 2
            het = np.where(
                raw_block == -127, 2 * p * (1 - p), (raw_block == 1).astype(float)
            )
            genetic[:, 2] += het @ w[take, 2]
        normalization = np.sqrt(np.var(genetic, axis=0))
        genetic /= normalization
        w /= normalization
        dense = np.sqrt(0.8) * genetic[:, 0]
        local_mean = 0.8 * xlocal[:, causal]
        dom = np.sqrt(0.5) * genetic[:, 2]
        structural = c[:, names.index("PC1")] * genetic[:, 3]
        structural *= np.sqrt(0.5 / structural.var())
        means = {
            "dense": dense,
            "sparse": np.sqrt(0.8) * genetic[:, 1] + local_mean,
            "dominance": dense + local_mean + dom,
            "structure": dense + local_mean + dom + structural,
            "heavy": dense + local_mean,
        }
        signals = {}
        for label, col in [("aligned", 4), ("mixed", 5), ("sparse_interaction", 6)]:
            f = target * genetic[:, col]
            signals[label] = f * np.sqrt(a.interaction_variance / f.var())
            means[label] = dense + local_mean + signals[label]
        means["structure_mixed"] = means["structure"] + signals["mixed"]
        definitions = {}
        for name, mean in means.items():
            alternative = name in signals or name == "structure_mixed"
            signal = (
                signals["mixed" if name == "structure_mixed" else name]
                if alternative
                else np.zeros(len(rows))
            )
            definitions[name] = dict(
                biological_null=not alternative,
                reference_signal_variance=float(signal.var()),
                reference_mean_variance=float(mean.var()),
                direction="mixed"
                if name == "structure_mixed"
                else name
                if alternative
                else "aligned",
            )
        np.savez(
            out / "reference.npz",
            rows=rows,
            target=target,
            means=np.column_stack(list(means.values())),
            signals=np.column_stack(
                [
                    signals.get(
                        "mixed" if k == "structure_mixed" else k, np.zeros(len(rows))
                    )
                    for k in means
                ]
            ),
            variance=0.4 + 0.6 * target**2,
            background=background,
            oracle_weights=w[background, 4:7],
            reference_inverse_scale=scale.inverse_scale[background],
        )
        pd.DataFrame([source.samples[i] for i in rows], columns=["FID", "IID"]).to_csv(
            out / "samples.tsv", sep="\t", index=False
        )
        (out / "variants.txt").write_text(
            "\n".join(v for k, v in enumerate(source.variants.ids) if k != j) + "\n"
        )
        (out / "interaction_variants.txt").write_text(
            "\n".join(source.variants.ids[k] for k in background) + "\n"
        )
        write_json(
            out / "reference.json",
            dict(
                genotypes=str(Path(a.genotypes).resolve()),
                source_identity=source.identity,
                n=len(rows),
                source_n=len(source.samples),
                m=m,
                target=a.target,
                background_chromosome=a.background_chromosome,
                local_variants=[source.variants.ids[k] for k in local],
                covariates=names,
                covariate_source=str(Path(a.covariates).resolve()),
                covariate_center=center,
                covariate_scale=spread,
                alignment=dict(
                    source_n=len(source.samples),
                    complete_covariates=int(
                        np.isfinite(aligned.to_numpy(float)).all(1).sum()
                    ),
                    selected_n=len(rows),
                ),
                background_variants=[source.variants.ids[k] for k in background],
                settings=list(means),
                definitions=definitions,
                architecture_seed=a.architecture_seed,
                interaction_variance=a.interaction_variance,
                minimum_reference_cell=a.minimum_reference_cell,
                local_radius_bp=100000,
                generating_additive_markers=m,
                generating_sparse_markers=len(sparse),
                generating_dominance_markers=len(dominant),
                seconds=time.perf_counter() - start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        )
    print("reference", len(rows), m, round(time.perf_counter() - start, 2), flush=True)


def load_reference(path):
    with np.load(path, allow_pickle=False) as z:
        identity = json.loads(str(z["manifest"]))["metadata"]["compatibility_id"]
    return load_feature_reference(path, compatibility_id=identity)


def diagnosis(ref, result, mean, signal, variance):
    c, f = ref.fixed_effects, ref.features
    # One independent least-squares factorization for all diagnostic responses.
    # This matters for large confirmation cohorts and never uses a known mean
    # in the production calculation.
    joined = np.column_stack([f, mean, signal])
    partial = joined - c @ np.linalg.lstsq(c, joined, rcond=1e-11)[0]
    r = partial[:, :f.shape[1]]
    h = r.T @ r
    inverse = np.linalg.pinv(h, rcond=1e-11)
    truth = inverse @ r.T @ mean
    leak = inverse @ r.T @ (mean - signal)
    influence = inverse @ r.T
    known_cov = (influence * variance) @ influence.T
    residual_signal = partial[:, -1]
    fitted_signal = r @ np.linalg.lstsq(r, residual_signal, rcond=1e-11)[0]
    row = dict(
        truth=truth.tolist(),
        additive_leakage=leak.tolist(),
        known_noise_se=np.sqrt(np.diag(known_cov)).tolist(),
        remaining_signal_variance=float(np.mean(residual_signal**2)),
        direction_alignment=float(
            fitted_signal @ fitted_signal / (residual_signal @ residual_signal)
        )
        if residual_signal @ residual_signal > 1e-18
        else None,
        nuisance_unexplained_mean_variance=float(np.mean((partial[:, -2] - partial[:, -1])**2)),
    )
    from scipy.stats import chi2
    coef = np.asarray(result["beta"], float)
    covariance = np.asarray(result["coefficient_covariance"], float)
    if np.all(np.isfinite(coef)) and np.all(np.isfinite(covariance)):
        delta = coef - truth
        row.update(error_vector=delta.tolist(), estimated_covariance=covariance.tolist(),
                   joint_coverage=bool(delta @ np.linalg.pinv(covariance) @ delta <= chi2.ppf(.95, len(delta))))
    if f.shape[1] == 1:
        b, se = result["beta"][0], result["standard_errors"][0]
        row.update(
            estimate=b,
            se=se,
            error=b - float(truth[0]),
            coverage=abs(b - truth[0]) <= 1.95996398454 * se,
            leakage_noise_ratio=float(leak[0] / np.sqrt(known_cov[0, 0])),
        )
    return row


def run(a):
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    reference_dir = a.reference.resolve()
    meta, data, samples = load_experiment_inputs(reference_dir,
        local_stress="local_withheld" in a.settings, threads=a.num_threads)
    variant_input = training_variant_input(reference_dir, root, meta)
    n0, n1 = a.training_samples, a.confirmation_samples
    if n0 + n1 > len(samples):
        raise ValueError(
            "fixed-panel training/confirmation need distinct reference individuals"
        )
    permutation = np.random.default_rng(a.panel_seed).permutation(len(samples))
    train = np.sort(permutation[:n0])
    test = np.sort(permutation[n0 : n0 + n1])
    samples.iloc[train].to_csv(root / "training.tsv", sep="\t", index=False)
    samples.iloc[test].to_csv(root / "confirmation.tsv", sep="\t", index=False)
    controls = [
        "--num-threads",
        str(a.num_threads),
        "--block-size",
        "128",
        "--memory-gib",
        str(a.memory_gib),
    ]
    settings = a.settings.split(",")
    unknown = set(settings) - set(meta["settings"])
    if unknown:
        raise ValueError(f"unknown settings: {unknown}")
    native = native_module()
    configure_prediction_threads(native, a.num_threads)
    # Oracle coefficients expressed on the exact confirmation HWE scale. Its
    # reference-imputation discrepancy remains in the measured projection truth.
    with FileGenotypeSource(meta["genotypes"], genome_build="GRCh37") as source:
        scale = fit_scale(
            source,
            data["rows"][test],
            threads=a.num_threads,
            block_size=128,
            memory_bytes=int(a.memory_gib * 2**30),
        )
        public_weights = (
            data["oracle_weights"]
            * data["reference_inverse_scale"][:, None]
            / scale.inverse_scale[data["background"], None]
        )
    write_json(
        root / "design.json",
        dict(
            reference=str(reference_dir),
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
            },
            phase="development",
            scenario_definitions={s: meta["definitions"][s] for s in settings},
            sampling="fixed intact genotype rows; regenerated independent training and confirmation errors",
            scientific_target="independent frozen score mean association; biological zero null evaluated separately",
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    records = []
    resources = []
    start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    for rep in range(a.replicates):
        for setting in settings:
            work = root / f"{setting}_{rep:03d}"
            work.mkdir()
            j = meta["settings"].index(setting)
            mean = data["means"][:, j]
            signal = data["signals"][:, j]
            errors = (
                rng.standard_t(5, len(samples)) / np.sqrt(5 / 3)
                if setting == "heavy"
                else rng.normal(size=len(samples))
            )
            y = mean + np.sqrt(data["variance"]) * errors
            outcomes = samples.copy()
            outcomes["y"] = y
            outcomes.to_csv(work / "phenotype.tsv", sep="\t", index=False)
            train_spec = dict(
                kind="summit.epistasis.train_direction",
                schema_version=1,
                genotypes=dict(geno=meta["genotypes"], genome_build="GRCh37"),
                samples=str(root / "training.tsv"),
                phenotype=dict(
                    file="phenotype.tsv", column="y", unit="fixed reference units"
                ),
                target=meta["target"],
                variants=str(variant_input),
                interaction_variants=str(reference_dir / "interaction_variants.txt"),
                trans_only=True,
                covariates=dict(
                    file=str(reference_dir / "covariates.tsv"),
                    columns=meta["covariates"],
                ),
                local_variants=meta["local_variants"],
                dominance_variants=meta["local_variants"],
                prior=dict(additive=0.5, interaction=0.05, residual=1.0),
                storage="packed",
                solver=dict(rtol=1e-6, max_iterations=200),
            )
            if getattr(a, "structure_covariates", ""):
                train_spec["covariates"]["varying_effects"] = a.structure_covariates.split(",")
            write_json(work / "train.json", train_spec)
            clock = time.perf_counter()
            cpu = time.process_time()
            case_records = []
            try:
                cli(
                    [
                        "train-direction",
                        str(work / "train.json"),
                        "--out",
                        str(work / "trained"),
                        *controls,
                    ]
                )
                resources.append(
                    dict(
                        setting=setting,
                        replicate=rep,
                        stage="train",
                        seconds=time.perf_counter() - clock,
                        cpu_seconds=time.process_time() - cpu,
                    )
                )
                oracle_col = {"aligned": 0, "mixed": 1, "sparse_interaction": 2}[
                    meta["definitions"][setting]["direction"]
                ]
                burden = dict(
                    name="burden",
                    score=dict(zip(meta["background_variants"], public_weights[:, 0])),
                    background="target",
                )
                oracle = dict(
                    name="oracle",
                    score=dict(
                        zip(meta["background_variants"], public_weights[:, oracle_col])
                    ),
                    background="target",
                )
                learned = dict(
                    name="learned", frozen_score="direction", background="target"
                )
                jobs = []
                for method, components in [
                    ("learned", [learned]),
                    ("burden", [burden]),
                    ("oracle", [oracle]),
                    ("joint", [learned, burden]),
                ]:
                    jobs.append(
                        dict(
                            id=method,
                            additive_annotations=["all"],
                            trans_target=meta["target"],
                            components=components,
                            local_variants=meta["local_variants"],
                            dominance_variants=meta["local_variants"],
                            inference=dict(
                                method="robust_mean",
                                main_effects="declared",
                                save_reference=True,
                            ),
                        )
                    )
                spec = dict(
                    kind="summit.epistasis.prepare",
                    schema_version=1,
                    genotypes=train_spec["genotypes"],
                    samples=str(root / "confirmation.tsv"),
                    phenotypes=dict(
                        file="phenotype.tsv",
                        columns=["y"],
                        unit="fixed reference units",
                    ),
                    covariates=train_spec["covariates"],
                    annotations={"target": {meta["target"]: 1}},
                    frozen_scores=[
                        dict(
                            name="PGS",
                            direction="trained/direction.json",
                            component=0,
                            adjust=True,
                        ),
                        dict(
                            name="direction",
                            direction="trained/direction.json",
                            component=1,
                        ),
                    ],
                    jobs=jobs,
                )
                write_json(work / "prepare.json", spec)
                clock = time.perf_counter()
                cpu = time.process_time()
                cli(
                    [
                        "prepare",
                        str(work / "prepare.json"),
                        "--out",
                        str(work / "prepared"),
                        *controls,
                    ]
                )
                resources.append(
                    dict(
                        setting=setting,
                        replicate=rep,
                        stage="prepare",
                        seconds=time.perf_counter() - clock,
                        cpu_seconds=time.process_time() - cpu,
                    )
                )
                for job in jobs:
                    method = job["id"]
                    artifact = work / "prepared" / f"{method}.robust-score.npz"
                    cli(
                        [
                            "fit",
                            str(artifact),
                            "--out",
                            str(work / f"{method}.fit.json"),
                        ]
                    )
                    result = json.loads((work / f"{method}.fit.json").read_text())[
                        "fits"
                    ][0]
                    ref = load_reference(
                        work / "prepared" / f"{method}.cohort-reference.npz"
                    )
                    row = dict(
                        setting=setting,
                        replicate=rep,
                        method=method,
                        failed=False,
                        biological_null=meta["definitions"][setting]["biological_null"],
                        p=result["joint_p"]
                        if method == "joint"
                        else result["kernel_p"],
                        outside_scope=result["diagnostics"][
                            "outside_confirmation_design"
                        ],
                        reference_signal_variance=meta["definitions"][setting][
                            "reference_signal_variance"
                        ],
                        realized_training_signal_variance=float(signal[train].var()),
                        realized_confirmation_signal_variance=float(signal[test].var()),
                        max_leverage=result["diagnostics"]["max_leverage"],
                        **diagnosis(
                            ref,
                            result,
                            mean[test],
                            signal[test],
                            data["variance"][test],
                        ),
                    )
                    case_records.append(row)
                model = json.loads((work / "trained/models/manifest.json").read_text())
                write_json(
                    work / "measurement.json",
                    dict(
                        resources=resources[-2:],
                        native=model["run_report"],
                        convergence=model["traits"][0]["models"][0]["convergence"],
                        peak_rss_bytes=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                    ),
                )
            except (ValueError, ArithmeticError, RuntimeError) as error:
                case_records = []
                for method in ["learned", "burden", "oracle", "joint"]:
                    case_records.append(
                        dict(
                            setting=setting,
                            replicate=rep,
                            method=method,
                            failed=True,
                            error=str(error),
                        )
                    )
                print("FAILED", setting, rep, str(error), flush=True)
            records.extend(case_records)
            print(
                setting,
                rep,
                "elapsed",
                round(time.perf_counter() - start, 2),
                flush=True,
            )
            # One append-sized compact record per completed full pipeline.
            with (root / "replicates.jsonl").open("a") as handle:
                for record in records[-4:]:
                    handle.write(json.dumps(_jsonable(record)) + "\n")
    write_json(
        root / "resources.json",
        dict(
            records=resources,
            seconds=time.perf_counter() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
    )
    pd.DataFrame(records).to_csv(root / "replicates.csv", index=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--genotypes", required=True)
    ref.add_argument("--covariates", required=True)
    ref.add_argument("--target", default="12:66358347")
    ref.add_argument("--background-chromosome", default="5")
    ref.add_argument("--architecture-seed", type=int, default=689173)
    ref.add_argument("--interaction-variance", type=float, default=0.1)
    ref.add_argument("--minimum-reference-cell", type=int, default=200)
    ref.add_argument("--reference-samples", type=int)
    fit = sub.add_parser("run")
    fit.add_argument("--reference", type=Path, required=True)
    fit.add_argument("--training-samples", type=int, default=2048)
    fit.add_argument("--confirmation-samples", type=int, default=4096)
    fit.add_argument("--panel-seed", type=int, default=857491)
    fit.add_argument("--seed", type=int, default=816293)
    fit.add_argument("--replicates", type=int, default=12)
    fit.add_argument("--structure-covariates", default="")
    fit.add_argument(
        "--settings",
        default="dense,sparse,dominance,structure,heavy,aligned,mixed,sparse_interaction,structure_mixed",
    )
    for command in (ref, fit):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--num-threads", type=int, default=2)
        command.add_argument("--memory-gib", type=float, default=8)
    a = p.parse_args()
    reference(a) if a.stage == "reference" else run(a)


if __name__ == "__main__":
    main()
