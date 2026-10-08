"""Prepare all intact held-out donor rows for already completed matched learners.

This is genotype-dependent population-validation preparation, not another
learning experiment or a correction of the parent experiment's P values.
The public prepare command evaluates the same frozen scalar procedure on a
larger empirical population. Population draws must still use the prespecified
confirmation sample size, and all donor rows remain jointly intact.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import time

import pandas as pd

from scripts.epistasis.full_matched import write_json
from scripts.epistasis.benchmark_robust_workflow import io
from summit.epistasis.cli import main as cli
from summit.prediction.artifacts import file_digest


def run(a):
    parent = a.input.resolve()
    original = json.loads((parent / "design.json").read_text())
    args = original["arguments"]
    if "batch_size" not in args:
        raise ValueError("donor expansion requires a batched matched learning run")
    settings = a.settings.split(",") if a.settings else args["settings"].split(",")
    if len(set(settings)) != len(settings) or set(settings) - set(
        args["settings"].split(",")
    ):
        raise ValueError("select distinct scheduled settings")
    records = [
        json.loads(line)
        for line in (parent / "replicates.jsonl").read_text().splitlines()
    ]
    selected = [
        r for r in records if r["setting"] in settings and r["method"] == "learned"
    ]
    expected = {(s, r) for s in settings for r in range(args["replicates"])}
    if (
        len(selected) != len(expected)
        or {(r["setting"], r["replicate"]) for r in selected} != expected
    ):
        raise ValueError(
            "every selected matched learning/preparation replicate must finish"
        )
    reference = Path(original["reference"])
    samples = pd.read_csv(reference / "samples.tsv", sep="\t", dtype=str)
    training = pd.read_csv(parent / "training.tsv", sep="\t", dtype=str)
    keys = pd.MultiIndex.from_frame(samples)
    train_keys = pd.MultiIndex.from_frame(training)
    if (
        not keys.is_unique
        or not train_keys.is_unique
        or not train_keys.isin(keys).all()
    ):
        raise ValueError("training/reference participant alignment failed")
    donors = samples.loc[~keys.isin(train_keys)].reset_index(drop=True)
    if len(donors) < args["confirmation_samples"]:
        raise ValueError(
            "expanded donor pool cannot remove original confirmation people"
        )
    original_test = pd.MultiIndex.from_frame(
        pd.read_csv(parent / "confirmation.tsv", sep="\t", dtype=str)
    )
    if not original_test.isin(pd.MultiIndex.from_frame(donors)).all():
        raise ValueError(
            "original confirmation participants are not contained in donor pool"
        )
    a.out.mkdir(parents=True, exist_ok=False)
    a.out.chmod(0o700)
    donors.to_csv(a.out / "confirmation.tsv", sep="\t", index=False)
    training.to_csv(a.out / "training.tsv", sep="\t", index=False)
    design = deepcopy(original)
    design.update(
        kind="matched_learning_donor_preparation",
        learning_source=str(parent),
        intended_confirmation_n=args["confirmation_samples"],
        sampling="uniform empirical population of all available intact nontraining rows; no chromosome rearrangement",
        independent_new_learning_models=0,
    )
    design["arguments"].update(
        settings=",".join(settings), methods="learned", confirmation_samples=len(donors)
    )
    write_json(a.out / "design.json", design)
    begin = time.perf_counter()
    measurements = []
    for setting in settings:
        for first in range(0, args["replicates"], args["batch_size"]):
            reps = list(
                range(first, min(first + args["batch_size"], args["replicates"]))
            )
            rows = [
                r
                for r in selected
                if r["setting"] == setting and r["replicate"] in reps
            ]
            work = a.out / f"{setting}_{first:03d}"
            source = parent / work.name
            work.mkdir()
            failures = [r for r in rows if r["failed"]]
            if failures:
                for r in rows:
                    if not r["failed"]:
                        raise ValueError(
                            "partial failed learning batches require explicit recovery before expansion"
                        )
                output_records = rows
            else:
                spec = json.loads((source / "prepare.json").read_text())
                spec["samples"] = str((a.out / "confirmation.tsv").resolve())
                for key in ("phenotypes", "covariates"):
                    if key in spec:
                        spec[key]["file"] = str((source / spec[key]["file"]).resolve())
                if "geno" in spec["genotypes"]:
                    spec["genotypes"]["geno"] = str(
                        (source / spec["genotypes"]["geno"]).resolve()
                    )
                spec["jobs"] = [j for j in spec["jobs"] if j["id"].endswith("_learned")]
                for score in spec["frozen_scores"]:
                    score["direction"] = str((source / score["direction"]).resolve())
                write_json(work / "prepare.json", spec)
                before, cpu, stage = io(), time.process_time(), time.perf_counter()
                cli(
                    [
                        "prepare",
                        str(work / "prepare.json"),
                        "--out",
                        str(work / "prepared"),
                        "--num-threads",
                        str(a.num_threads),
                        "--block-size",
                        "128",
                        "--memory-gib",
                        str(a.memory_gib),
                    ]
                )
                after = io()
                measurements.append(
                    dict(
                        setting=setting,
                        batch=first,
                        seconds=time.perf_counter() - stage,
                        cpu_seconds=time.process_time() - cpu,
                        io={k: after[k] - v for k, v in before.items()},
                        model_manifest_sha256=file_digest(
                            source / "trained/models/manifest.json"
                        ),
                    )
                )
                output_records = [
                    dict(
                        setting=setting,
                        replicate=r,
                        method="learned",
                        failed=False,
                        biological_null=rows[0]["biological_null"],
                        learning_source=str(source),
                        record_kind="donor feature preparation; not new learning",
                    )
                    for r in reps
                ]
                print(
                    setting,
                    first,
                    "donors",
                    len(donors),
                    "seconds",
                    round(measurements[-1]["seconds"], 2),
                    flush=True,
                )
            with (a.out / "replicates.jsonl").open("a") as handle:
                for record in output_records:
                    handle.write(json.dumps(record) + "\n")
    write_json(
        a.out / "resources.json",
        dict(
            seconds=time.perf_counter() - begin,
            records=measurements,
            donors=len(donors),
            independent_new_learning_models=0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--settings", default="")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--memory-gib", type=float, default=16)
    run(p.parse_args())


if __name__ == "__main__":
    main()
