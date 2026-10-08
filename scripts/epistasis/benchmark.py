"""Bounded native workload timing; no extrapolation to biobank scale."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from summit.epistasis.models import annotation_weights, target_design
from summit.epistasis.prepare import SelectedStudy, fit_scale
from summit.epistasis.summary import fit_epistasis
from summit.epistasis.cli import _jsonable
from summit.prediction.genotype import ArrayGenotypeSource, native_module
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.spec import VariantAxis


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--variants", type=int, default=8192)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    native = native_module()
    configure_prediction_threads(native, args.num_threads)
    n, m = args.samples, args.variants
    rng = np.random.default_rng(274)
    raw = np.empty((n, m), dtype=np.int8, order="F")
    for start in range(0, m, 128):
        raw[:, start:start+128] = rng.binomial(2, .3, (n, min(128, m-start)))
    axis = VariantAxis(tuple(f"v{i}" for i in range(m)), ("1",)*m,
                       tuple(range(1, m+1)), ("A",)*m, ("G",)*m)
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis, hard_calls=True)
    del raw
    stages = {}
    start = time.perf_counter()
    scale = fit_scale(source, np.arange(n), block_size=128, threads=args.num_threads)
    stages["scale_seconds"] = time.perf_counter()-start
    design = target_design(source, np.arange(n), scale,
        components=[dict(name=f"epi{i}", target=f"v{i}", background="all") for i in range(4)],
        annotations=annotation_weights(axis.ids, {}), additive_annotations=["all"],
        block_size=128, threads=args.num_threads, native=native)
    study = SelectedStudy(source, np.arange(n), scale, **design, block_size=128,
                          threads=args.num_threads, memory_bytes=2**30)
    start = time.perf_counter()
    reference = study.reference(nvecs=64, seed=185)
    stages["reference_seconds"] = time.perf_counter()-start
    print(json.dumps(dict(phase="reference_complete", **stages)), flush=True)
    start = time.perf_counter()
    summary, _ = study.summarize(reference, rng.normal(size=(n, 4)), trait_names=("a", "b", "c", "d"))
    stages["trait_analytic_seconds"] = time.perf_counter()-start
    fits = [fit_epistasis(summary, i) for i in range(4)]
    record = dict(n=n, m=m, targets=4, traits=4, components=6, nvecs=64,
                  threads=args.num_threads, stages=stages,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  passes=study.stream.ledger.traversals, scale_passes=1,
                  reference_workspace_plan=reference.metadata["reference"]["planned_workspace_bytes"],
                  native_path=native.__file__, native_build=native.build_info(),
                  condition_number=fits[0]["condition_number"],
                  valid_covariance_count=sum(f["covariance_valid"] for f in fits),
                  purpose="throughput measurement; not statistical calibration; no biobank extrapolation")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(_jsonable(record), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(_jsonable(record)), flush=True)


if __name__ == "__main__":
    main()
