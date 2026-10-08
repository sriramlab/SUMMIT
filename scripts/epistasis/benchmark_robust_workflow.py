"""Complete disk-backed robust preparation, nuisance fitting and fresh-process inference."""
import argparse, json, os, re, resource, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
from bed_reader import to_bed
from summit.epistasis.cli import main as epistasis_main, _jsonable
from summit.prediction.genotype import native_module


def io():
    return {
        k: int(v)
        for k, v in (
            line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
        )
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--variants", type=int, default=8192)
    parser.add_argument("--traits", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--train-direction", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n, m, t = args.samples, args.variants, args.traits
    rng = np.random.default_rng(917360)
    records = []
    native = native_module()
    root_code = Path(__file__).resolve().parents[2]
    private = bool(os.environ.get("SUMMIT_PRIVATE_NATIVE_DIR"))
    cpus = (
        [int(v) for v in re.findall(r"\{(\d+)\}", os.environ.get("OMP_PLACES", ""))]
        if private
        else sorted(os.sched_getaffinity(0))
    )
    launcher = root_code / (
        "scripts/generalized_gxe/private_python.py"
        if private
        else "scripts/epistasis/checkout_python.py"
    )
    controls = [
        "--num-threads",
        str(args.num_threads),
        "--block-size",
        "128",
        "--memory-gib",
        str(args.memory_gib),
    ]

    def measured(name, call):
        start = time.perf_counter()
        cpu = time.process_time()
        before = io()
        value = call()
        after = io()
        records.append(
            dict(
                stage=name,
                seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu,
                peak_process_rss_bytes=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                peak_child_rss_bytes=resource.getrusage(
                    resource.RUSAGE_CHILDREN
                ).ru_maxrss
                * 1024,
                io_delta={k: after[k] - before[k] for k in before},
            )
        )
        print(name, round(records[-1]["seconds"], 3), flush=True)
        return value

    with tempfile.TemporaryDirectory(
        prefix="robust-workflow-", dir=args.scratch
    ) as directory:
        root = Path(directory)
        # Fixture generation is bounded too; it is reported separately from
        # analysis and can contribute to the whole-process RSS high watermark.
        raw = np.memmap(root / "fixture.raw", dtype=np.int8, mode="w+", shape=(n, m))
        for begin in range(0, m, 128):
            raw[:, begin : begin + 128] = rng.binomial(
                2, 0.35, (n, min(128, m - begin))
            )
        raw.flush()
        local = raw[:, :16].astype(float)
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
        del raw
        (root / "samples.tsv").write_text(
            "FID IID\n" + "".join(f"{i} {i}\n" for i in ids)
        )
        y = (
            local @ rng.normal(size=(16, t)) / 4
            + 0.4 * (local[:, 0, None] == 1)
            + rng.normal(size=(n, t))
            * np.sqrt(0.3 + 0.7 * (local[:, 0, None] - 0.7) ** 2 / 0.455)
        )
        traits = [f"y{i}" for i in range(t)]
        (root / "y.tsv").write_text(
            "FID IID "
            + " ".join(traits)
            + "\n"
            + "".join(
                f"{i} {i} " + " ".join(map(str, values)) + "\n"
                for i, values in enumerate(y)
            )
        )
        annotations = dict(
            target=dict(v0=1),
            A={f"v{i}": 1 for i in range(8)},
            B={f"v{i}": 1 for i in range(6, 14)},
            region1={f"v{i}": 1 for i in range(32, 64)},
            region2={f"v{i}": 1 + (i % 2) for i in range(48, 80)},
        )

        def job(name, **model):
            return dict(
                id=name,
                additive_annotations=["all"],
                local_variants=[f"v{i}" for i in range(16)],
                dominance_variants=[f"v{i}" for i in range(16)],
                inference=dict(method="robust_mean", main_effects="declared"),
                **model,
            )

        jobs = []
        for i in (0, 1):
            value = job(
                "target" + str(i),
                components=[dict(name="genome", target=f"v{i}", background="all")],
            )
            value["inference"]["feature_sketch_dimensions"] = 8
            jobs.append(value)
        jobs.append(
            job(
                "weighted_score",
                components=[
                    dict(
                        name="score_direction",
                        score={f"v{i}": 1 / np.sqrt(m - 1) for i in range(1, m)},
                        background="target",
                    )
                ],
            )
        )
        jobs.append(job("pairs", pairs=[["v0", f"v{i}"] for i in range(3, 7)]))
        jobs.append(
            job("within", groups=[dict(name="within", mode="within", left="A")])
        )
        for name, model in [
            (
                "overlap",
                dict(groups=[dict(name="overlap", mode="cross", left="A", right="B")]),
            ),
            (
                "remainder",
                dict(groups=[dict(name="remainder", mode="remainder", left="A")]),
            ),
            (
                "annotations",
                dict(
                    components=[
                        dict(name="one", target="v0", background="region1"),
                        dict(name="two", target="v0", background="region2"),
                    ]
                ),
            ),
        ]:
            value = job(name, **model)
            value["inference"]["feature_sketch_dimensions"] = 8
            jobs.append(value)
        spec = dict(
            kind="summit.epistasis.prepare",
            schema_version=1,
            genotypes=dict(geno="input.bed"),
            samples="samples.tsv",
            phenotypes=dict(file="y.tsv", columns=traits, unit="simulated"),
            annotations=annotations,
            jobs=jobs,
        )
        path = root / "prepare.json"
        path.write_text(json.dumps(spec))
        measured(
            "all_families_prepare_and_HC3_nuisance",
            lambda: epistasis_main(
                ["prepare", str(path), "--out", str(root / "prepared"), *controls]
            ),
        )
        preparation = json.loads((root / "prepared/preparation.json").read_text())
        fits = []
        for record in preparation["jobs"]:
            artifact = root / "prepared" / record["summary"]
            output = root / (record["id"] + ".fit.json")
            measured(
                record["id"] + "_saved_inference",
                lambda a=artifact, o=output: epistasis_main(
                    ["fit", str(a), "--out", str(o)]
                ),
            )
            result = json.loads(output.read_text())["fits"]
            fits.append(dict(job=record["id"], fits=result))
        # An actual separate process sees neither genotype nor phenotype path.
        (root / "input.bed").rename(root / "input.bed.unavailable")
        (root / "y.tsv").rename(root / "y.tsv.unavailable")
        cmd = [
            "taskset",
            "-c",
            ",".join(map(str, cpus)),
            sys.executable,
            str(launcher),
            "summit.epistasis.cli",
            "fit",
            str(root / "prepared/pairs.robust-score.npz"),
            "--out",
            str(root / "fresh-fit.json"),
        ]
        measured(
            "fresh_process_genotype_free_fit",
            lambda: subprocess.run(cmd, check=True, capture_output=True, text=True),
        )
        original = json.loads((root / "pairs.fit.json").read_text())
        fresh = json.loads((root / "fresh-fit.json").read_text())
        if original != fresh:
            raise ArithmeticError("fresh-process summary inference changed")
        (root / "input.bed.unavailable").rename(root / "input.bed")
        (root / "y.tsv.unavailable").rename(root / "y.tsv")
        training = None
        if args.train_direction:
            ntrain = min(2048, n // 3)
            for name, rows in [
                ("training", range(ntrain)),
                ("confirmation", range(ntrain, n)),
            ]:
                (root / f"{name}.tsv").write_text(
                    "FID IID\n" + "".join(f"{i} {i}\n" for i in rows)
                )
            (root / "background.txt").write_text(
                "\n".join(f"v{i}" for i in range(1, m)) + "\n"
            )
            (root / "interaction-region.txt").write_text(
                "\n".join(f"v{i}" for i in range(32, 96)) + "\n"
            )
            training = dict(
                kind="summit.epistasis.train_direction",
                schema_version=1,
                genotypes=dict(geno="input.bed"),
                samples="training.tsv",
                phenotype=dict(file="y.tsv", column="y0", unit="simulated"),
                target="v0",
                variants="background.txt",
                interaction_variants="interaction-region.txt",
                local_variants=[f"v{i}" for i in range(16)],
                dominance_variants=[f"v{i}" for i in range(16)],
                prior=dict(additive=0.5, interaction=0.05, residual=1.0),
                storage="packed",
                solver=dict(rtol=1e-6, max_iterations=150),
            )
            (root / "train.json").write_text(json.dumps(training))
            measured(
                "independent_direction_training",
                lambda: epistasis_main(
                    [
                        "train-direction",
                        str(root / "train.json"),
                        "--out",
                        str(root / "trained"),
                        *controls,
                    ]
                ),
            )
            confirm = dict(
                spec,
                samples="confirmation.tsv",
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
                jobs=[
                    job(
                        "confirmation",
                        components=[
                            dict(
                                name="frozen_direction",
                                frozen_score="direction",
                                background="target",
                            )
                        ],
                    )
                ],
            )
            (root / "confirm.json").write_text(json.dumps(confirm))
            measured(
                "heldout_scoring_prepare_and_HC3",
                lambda: epistasis_main(
                    [
                        "prepare",
                        str(root / "confirm.json"),
                        "--out",
                        str(root / "confirmed"),
                        *controls,
                    ]
                ),
            )
            measured(
                "heldout_saved_inference",
                lambda: epistasis_main(
                    [
                        "fit",
                        str(root / "confirmed/confirmation.robust-score.npz"),
                        "--out",
                        str(root / "confirmed-fit.json"),
                    ]
                ),
            )
            training = dict(
                n_train=ntrain,
                n_test=n - ntrain,
                model_report=json.loads(
                    (root / "trained/models/manifest.json").read_text()
                )["run_report"],
                preparation=json.loads(
                    (root / "confirmed/preparation.json").read_text()
                ),
                fits=json.loads((root / "confirmed-fit.json").read_text()),
            )
        payload = dict(
            n=n,
            m=m,
            traits=t,
            threads=args.num_threads,
            seed=917360,
            records=records,
            preparation=preparation,
            training=training,
            fit_results=fits,
            native_path=native.__file__,
            native_build=native.build_info(),
            input="temporary synthetic disk-backed PLINK; participant inputs not used",
            cache="warm after fixture generation; physical read_bytes distinguished from logical rchar",
            memory_scope="whole-process high watermark includes bounded fixture generation; child peak reported separately",
            statistical_scope="complete workflow benchmark under declared finite-dimensional mean; timing is not general polygenic calibration evidence",
        )
        # Do not duplicate the large inline scientific manifest in a benchmark.
        payload["preparation"].pop("manifest", None)
        if training:
            training["preparation"].pop("manifest", None)
        for case in payload["fit_results"]:
            for fit in case["fits"]:
                fit["diagnostics"] = {
                    k: fit["diagnostics"][k]
                    for k in (
                        "n_samples",
                        "fixed_rank",
                        "feature_rank",
                        "max_leverage",
                        "information_condition",
                        "minimum_feature_effective_support",
                        "outside_confirmation_design",
                    )
                }
        if training:
            for fit in training["fits"]["fits"]:
                fit["diagnostics"] = {
                    k: fit["diagnostics"][k]
                    for k in (
                        "n_samples",
                        "fixed_rank",
                        "feature_rank",
                        "max_leverage",
                        "information_condition",
                        "minimum_feature_effective_support",
                        "outside_confirmation_design",
                    )
                }
    with args.out.open("x") as handle:
        json.dump(_jsonable(payload), handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
