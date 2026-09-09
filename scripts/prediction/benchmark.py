"""Bounded synthetic BED benchmark; each mode belongs in a fresh process."""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--variants", type=int, default=4096)
    parser.add_argument("--models", type=int, default=10)
    parser.add_argument("--traits", type=int, default=2)
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mode", choices=["stream", "compact", "standardized"], default="compact")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--rhs-columns", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    if any(v <= 0 for k, v in vars(args).items() if k != "mode"):
        parser.error("all dimensions/counts must be positive")
    if args.samples*args.variants > 20_000_000 or args.samples < 32 or args.contexts < 2:
        parser.error("this local smoke benchmark requires N>=32, Q>=2 and N*M<=20,000,000")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import json
    import resource
    import tempfile
    import time
    from pathlib import Path
    from dataclasses import asdict
    import numpy as np
    from bed_reader import to_bed
    from summit.prediction.genotype import FileGenotypeSource, estimate_scale
    from summit.prediction.spec import TraitTraining, CandidatePrior
    from summit.prediction.batch import plan_prediction
    from summit.prediction.operator import GenotypeOperator
    rng = np.random.default_rng(20260909)
    n, m, q = args.samples, args.variants, args.contexts
    with tempfile.TemporaryDirectory(prefix="summit-prediction-benchmark-") as tmp:
        path = Path(tmp)/"calls.bed"
        calls = rng.binomial(2, .3, (n, m)).astype(np.float64)
        calls[rng.random((n, m)) < .01] = np.nan
        to_bed(path, calls, num_threads=1, properties=dict(chromosome=["1"]*m,
            bp_position=list(range(1, m+1)), allele_1=["A"]*m, allele_2=["G"]*m))
        del calls
        with FileGenotypeSource(path, genome_build="synthetic") as source:
            traits, vectors = [], {}
            for ti in range(args.traits):
                rows = np.flatnonzero(np.arange(n) % (7+ti) != ti)
                variants = np.arange(m)
                phi = np.column_stack([np.ones(len(rows)), rng.normal(size=(len(rows), q-1))])
                residual = np.exp(.2*phi[:, 1])
                covariance = np.diag(np.r_[.3, np.full(q-1, .03)])
                scale = estimate_scale(source, rows, variants, block_size=args.block_size, threads=args.threads)
                candidates = tuple(CandidatePrior(f"m{k}", covariance*(k+1)/args.models, residual, {"source": "synthetic"}) for k in range(args.models))
                traits.append(TraitTraining(f"t{ti}", rows, variants, rng.normal(size=len(rows)), phi,
                    phi[:, :1], scale, candidates, {"names": [f"q{j}" for j in range(q)]},
                    {"names": ["intercept"]}, {"units": "synthetic"}))
                for c in candidates:
                    vectors[(f"t{ti}", c.id)] = rng.normal(size=len(rows))
            plan = plan_prediction(traits, source, storage=args.mode, block_size=args.block_size,
                rhs_columns=args.rhs_columns, threads=args.threads, memory_bytes=2**30)
            operator = GenotypeOperator(source, traits, plan)
            start = time.perf_counter()
            operator.setup()
            setup_seconds = time.perf_counter()-start
            control = operator.apply(vectors, phase="warmup")
            timings = []
            error = 0.
            for _ in range(args.repetitions):
                start = time.perf_counter()
                result = operator.apply(vectors, phase="benchmark")
                timings.append(time.perf_counter()-start)
                error = max(error, max(float(np.linalg.norm(result[k]-control[k])/max(np.linalg.norm(control[k]), 1e-30)) for k in control))
            print(json.dumps(dict(dimensions=vars(args), selected_samples=[len(t.rows) for t in traits],
                setup_seconds=setup_seconds, operator_seconds=timings, median_operator_seconds=float(np.median(timings)),
                repeat_relative_error=error, process_peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                ledger=asdict(operator.ledger), plan=plan.to_dict(), native_build=operator.native.build_info()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
