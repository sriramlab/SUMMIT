"""Combine completed matched preparations and verify the public multi-target path.

Only compatible sample/covariate masks are combined. Each job keeps its own
trait-matched model, outcome, main effects and test. No learning is repeated or
counted as new statistical evidence.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.epistasis.benchmark_robust_workflow import io
from scripts.epistasis.full_matched import write_json, load_reference
from summit.epistasis.cli import main as cli
from summit.epistasis.robust import load_robust_scores
from summit.prediction.cli import _table, _aligned_table


def run(a):
    if len(a.preparations) != len(a.jobs) or len(a.jobs) < 2:
        raise ValueError("supply at least two preparation/job pairs")
    a.out.mkdir(parents=True, exist_ok=False)
    a.out.chmod(0o700)
    batch, comparisons, phenotype = None, [], None
    for i, (path, name) in enumerate(zip(a.preparations, a.jobs)):
        path = path.resolve()
        root = path.parent
        spec = json.loads(path.read_text())
        job = next(j for j in spec["jobs"] if j["id"] == name)
        if job.get("inference", {}).get("method") != "robust_mean":
            raise ValueError("this comparison requires robust mean preparations")
        sample_ids = list(_table(root / spec["samples"]).index)
        cv = spec["covariates"]
        cv_values = _aligned_table(root / cv["file"], sample_ids)[cv["columns"]]
        genotype = dict(spec["genotypes"], geno=str((root / spec["genotypes"]["geno"]).resolve()))
        columns = job.get("phenotypes", spec["phenotypes"]["columns"])
        if len(columns) != 1:
            raise ValueError("select one phenotype per target for this comparison")
        outcome = _aligned_table(root / spec["phenotypes"]["file"], sample_ids)[columns[0]].to_numpy()
        if batch is None:
            batch = deepcopy(spec)
            batch["genotypes"] = genotype
            batch["samples"] = "samples.tsv"
            batch["covariates"]["file"] = str((root / cv["file"]).resolve())
            batch["phenotypes"].update(file="phenotypes.tsv", columns=[])
            batch["annotations"], batch["jobs"], batch["frozen_scores"] = {}, [], []
            reference_ids, reference_cv = sample_ids, cv_values
            phenotype = pd.DataFrame(sample_ids, columns=["FID", "IID"])
            phenotype.to_csv(a.out / "samples.tsv", sep="\t", index=False)
        elif (sample_ids != reference_ids or genotype != batch["genotypes"]
              or cv["columns"] != batch["covariates"]["columns"]
              or cv.get("varying_effects", []) != batch["covariates"].get("varying_effects", [])
              or spec["phenotypes"]["unit"] != batch["phenotypes"]["unit"]
              or not np.allclose(cv_values.to_numpy(), reference_cv.to_numpy(),atol=1e-12,rtol=1e-12)):
            raise ValueError("multi-target comparison requires identical genotype, samples, covariates and outcome units")
        prefix = f"t{i}_"
        column = prefix + columns[0]
        phenotype[column] = outcome
        batch["phenotypes"]["columns"].append(column)
        for key, value in spec["annotations"].items():
            if not isinstance(value, dict):
                raise ValueError("this driver requires explicit annotation mappings")
            batch["annotations"][prefix + key] = value
        renamed = deepcopy(job)
        renamed["id"] = prefix + name
        renamed["phenotypes"] = [column]
        renamed["additive_annotations"] = [prefix + x if x in spec["annotations"] else x
                                          for x in job["additive_annotations"]]
        required = set(job.get("adjust_scores", []))
        renamed["adjust_scores"] = [prefix + x for x in job.get("adjust_scores", [])]
        for component in renamed["components"]:
            if "frozen_score" in component:
                required.add(component["frozen_score"])
                component["frozen_score"] = prefix + component["frozen_score"]
            component["background"] = prefix + component["background"]
        for score in spec["frozen_scores"]:
            if score["name"] in required:
                batch["frozen_scores"].append(dict(score, name=prefix + score["name"],
                    direction=str((root / score["direction"]).resolve())))
        batch["jobs"].append(renamed)
        comparisons.append(dict(parent=str(path), job=name, combined_job=renamed["id"],
            covariate_roundtrip_maximum_difference=float(np.max(abs(cv_values.to_numpy()-reference_cv.to_numpy())))))
    phenotype.to_csv(a.out / "phenotypes.tsv", sep="\t", index=False)
    write_json(a.out / "prepare.json", batch)
    before, start, cpu = io(), time.perf_counter(), time.process_time()
    cli(["prepare", str(a.out / "prepare.json"), "--out", str(a.out / "prepared"),
         "--num-threads", str(a.num_threads), "--block-size", "128", "--memory-gib", str(a.memory_gib)])
    after = io()
    preparation_seconds = time.perf_counter() - start
    for comparison in comparisons:
        parent = Path(comparison["parent"]).parent / "prepared" / comparison["job"]
        combined = a.out / "prepared" / comparison["combined_job"]
        original = load_robust_scores(str(parent) + ".robust-score.npz")
        together = load_robust_scores(str(combined) + ".robust-score.npz")
        differences = {}
        for field in ("scores", "information", "score_covariance"):
            left, right = getattr(original, field), getattr(together, field)
            np.testing.assert_allclose(left, right, atol=1e-9, rtol=1e-8)
            differences[field] = float(np.max(abs(left-right)))
        left = load_reference(Path(str(parent) + ".cohort-reference.npz"))
        right = load_reference(Path(str(combined) + ".cohort-reference.npz"))
        for field in ("features", "fixed_effects"):
            np.testing.assert_allclose(getattr(left, field), getattr(right, field), atol=1e-10, rtol=1e-8)
        from summit.epistasis.robust import robust_score_tests
        fit_original = robust_score_tests(original)
        fit_together = robust_score_tests(together)
        for field in ("beta", "coefficient_covariance", "joint_p"):
            np.testing.assert_allclose(fit_original[field], fit_together[field], atol=1e-10, rtol=1e-8)
        cli(["fit", str(combined) + ".robust-score.npz", "--out", str(combined) + ".fit.json"])
        comparison["maximum_absolute_differences"] = differences
    report = dict(comparisons=comparisons, preparation_seconds=preparation_seconds,
        total_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        preparation_io={k:after[k]-v for k,v in before.items()},
        independent_new_learning_models=0, n_samples=len(phenotype))
    write_json(a.out / "resources.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--preparations", type=Path, nargs="+", required=True)
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--memory-gib", type=float, default=16)
    run(p.parse_args())


if __name__ == "__main__":
    main()
