"""Measure public genotype-free trait preparation and batch/separate equivalence."""
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np
from summit.epistasis.cli import main as cli
from summit.epistasis.robust import load_robust_scores


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    root = args.inputs.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    original = json.loads((root / "confirm.json").read_text())
    phen = dict(original["phenotypes"], file=str(root / original["phenotypes"]["file"]))
    spec = dict(
        kind="summit.epistasis.prepare_traits",
        schema_version=1,
        reference=str(root / "confirmed/learned.cohort-reference.npz"),
        samples=str(root / original["samples"]),
        phenotypes=phen,
    )
    paths = []
    measures = []
    # Deliberately disable every genotype-source entry point for this process.
    import summit.prediction.genotype as genotype

    def unavailable(*a, **k):
        raise AssertionError("genotype access unavailable during phenotype reuse")

    genotype.source_from_spec = unavailable
    genotype.FileGenotypeSource = unavailable
    for label, columns in [("batch", phen["columns"])] + [
        (f"trait{j}", [name]) for j, name in enumerate(phen["columns"])
    ]:
        manifest = out / (label + ".json")
        manifest.write_text(
            json.dumps(dict(spec, phenotypes=dict(phen, columns=columns)))
        )
        start = time.perf_counter()
        cpu = time.process_time()
        path = out / (label + ".npz")
        cli(
            [
                "prepare-traits",
                str(manifest),
                "--out",
                str(path),
                "--num-threads",
                str(args.num_threads),
                "--memory-gib",
                "8",
            ]
        )
        cli(["fit", str(path), "--out", str(out / (label + ".fit.json"))])
        measures.append(
            dict(
                label=label,
                seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            )
        )
        paths.append(path)
    batch = load_robust_scores(paths[0])
    old = load_robust_scores(root / "confirmed/learned.robust-score.npz")
    errors = []
    for field in ("information", "scores", "score_covariance"):
        np.testing.assert_allclose(
            getattr(batch, field), getattr(old, field), rtol=1e-10, atol=1e-10
        )
        errors.append(float(np.max(abs(getattr(batch, field) - getattr(old, field)))))
    for j, path in enumerate(paths[1:]):
        single = load_robust_scores(path)
        np.testing.assert_allclose(
            single.scores[:, 0], batch.scores[:, j], rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            single.score_covariance[0],
            batch.score_covariance[j],
            rtol=1e-10,
            atol=1e-10,
        )
    receipt = dict(
        inputs=str(root),
        out=str(out),
        n=batch.metadata["n_samples"],
        traits=phen["columns"],
        maximum_absolute_reproduction_error=max(errors),
        genotype_sources_disabled=True,
        genotype_passes=0,
        batch_separate_verified=True,
        measurements=measures,
    )
    with args.receipt.open("x") as handle:
        json.dump(receipt, handle, indent=2)


if __name__ == "__main__":
    main()
