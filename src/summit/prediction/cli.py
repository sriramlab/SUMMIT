"""Opt-in prediction commands; thread/NUMA setup precedes numerical imports."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _runtime(args):
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    from summit._early_numa import preconfigure_numa_from_argv
    preconfigure_numa_from_argv(["--numa-mode", args.numa_mode, "--numa-nodes", args.numa_nodes])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.num_threads)


def _table(path):
    import pandas as pd
    table = pd.read_csv(path, sep=r"\s+", dtype={"FID": str, "IID": str}, keep_default_na=False)
    if "FID" not in table or "IID" not in table or table.duplicated(["FID", "IID"]).any():
        raise ValueError("input table needs unique string FID/IID pairs")
    return table.set_index(["FID", "IID"])


def _aligned_table(path, samples):
    import pandas as pd
    table = _table(path)
    index = pd.MultiIndex.from_tuples(samples, names=["FID", "IID"])
    if not index.isin(table.index).all():
        raise ValueError("input table is missing selected samples")
    return table.loc[index]


def _rows(source, path):
    import numpy as np
    ids = list(_table(path).index)
    lookup = {key: i for i, key in enumerate(source.samples)}
    if not ids or any(key not in lookup for key in ids):
        raise ValueError("sample selection is empty or includes unknown IDs")
    return np.array([lookup[key] for key in ids], dtype=np.int64)


def _variants(source, path):
    import numpy as np
    if path is None:
        return np.arange(len(source.variants.ids), dtype=np.int64)
    ids = Path(path).read_text().split()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("variant selection must be nonempty and unique")
    lookup = {key: i for i, key in enumerate(source.variants.ids)}
    if any(key not in lookup for key in ids):
        raise ValueError("unknown variant in selection")
    return np.array(sorted(lookup[key] for key in ids), dtype=np.int64)


def _load_fit(path):
    import numpy as np
    from ._validation import closed
    from .artifacts import read_json, load_genotype_scale
    from .adapters import load_prior
    from .genotype import source_from_spec
    from .features import evaluate_contexts, evaluate_fixed
    from .priors import ResponseGeometry, common_scale, separate_scales
    from .spec import TraitTraining, CandidatePrior, SolverSpec
    path = Path(path).resolve()
    root = path.parent
    resolve = lambda value: root/Path(value)
    spec = read_json(path)
    closed(spec, ("kind", "schema_version", "genotypes", "traits", "solver"), name="fit specification")
    if spec["kind"] != "summit.prediction.fit_spec" or spec["schema_version"] != 1:
        raise ValueError("unsupported fit specification")
    source = source_from_spec(spec["genotypes"], root)
    traits = []
    try:
        for t in spec["traits"]:
            closed(t, ("id", "phenotype", "samples", "contexts", "context_spec", "covariates", "fixed_spec",
                       "genotype_scale", "architecture_prior", "residual_spec", "candidates"), ("variants",), name="trait specification")
            rows = _rows(source, resolve(t["samples"]))
            samples = [source.samples[int(i)] for i in rows]
            variants = _variants(source, resolve(t["variants"]) if "variants" in t else None)
            context_spec = read_json(resolve(t["context_spec"]))
            fixed_spec = read_json(resolve(t["fixed_spec"]))
            contexts = _aligned_table(resolve(t["contexts"]), samples)
            covariates = _aligned_table(resolve(t["covariates"]), samples)
            phi = evaluate_contexts(context_spec, contexts)
            fixed = evaluate_fixed(fixed_spec, covariates, phi, context_spec["names"])
            pheno = t["phenotype"]
            closed(pheno, ("file", "column", "units"), ("center", "scale"), name="phenotype")
            center, scale_y = float(pheno.get("center", 0)), float(pheno.get("scale", 1))
            if not np.isfinite(center) or not np.isfinite(scale_y) or scale_y <= 0 or not pheno["units"]:
                raise ValueError("invalid phenotype units/affine transform")
            y = (_aligned_table(resolve(pheno["file"]), samples)[pheno["column"]].to_numpy(float)-center)/scale_y
            scale = load_genotype_scale(resolve(t["genotype_scale"]))
            omega, parent = load_prior(resolve(t["architecture_prior"]), context_spec=context_spec, scale=scale)
            r = read_json(resolve(t["residual_spec"]))
            closed(r, ("kind", "schema_version", "file", "column", "units", "floor", "provenance"), name="residual specification")
            if r["kind"] != "summit.prediction.residual" or r["schema_version"] != 1 or r["units"] != "model_phenotype_variance":
                raise ValueError("unsupported residual variance specification/units")
            floor = float(r["floor"])
            if not np.isfinite(floor) or floor < 0 or not r["provenance"]:
                raise ValueError("invalid residual floor/provenance")
            residual_path = resolve(t["residual_spec"]).parent/r["file"]
            residual = _aligned_table(residual_path, samples)[r["column"]].to_numpy(float)
            if not np.all(np.isfinite(residual)):
                raise ValueError("nonfinite residual variance input")
            residual = np.maximum(residual, floor)
            geometry = None
            if "geometry" in parent:
                closed(parent["geometry"], ("metric", "reference", "anchor"), name="prior geometry")
                geometry = ResponseGeometry(omega, **parent["geometry"])
            candidates = []
            for c in t["candidates"]:
                operation = c.get("operation") if isinstance(c, dict) else None
                fields = {"common_scale": ("kappa",), "separate_scales": ("kappa_a", "kappa_h"),
                          "spectral_shrinkage": ("tau", "kappa"), "spectral_rank": ("rank", "kappa"),
                          "supplied": ("covariance", "provenance")}
                if operation not in fields:
                    raise ValueError("unknown candidate operation; profiled ranks require a supplied prior")
                closed(c, ("id", "operation", *fields[operation]), name="candidate")
                if operation == "common_scale":
                    covariance = common_scale(omega, c["kappa"])
                elif operation == "separate_scales":
                    covariance = separate_scales(omega, c["kappa_a"], c["kappa_h"])
                elif operation == "supplied":
                    covariance = c["covariance"]
                else:
                    if geometry is None:
                        raise ValueError("spectral prior needs a declared metric and anchor")
                    restricted = geometry.prior(tau=c["tau"]) if operation == "spectral_shrinkage" else geometry.prior(rank=c["rank"])
                    covariance = common_scale(restricted, c["kappa"])
                candidates.append(CandidatePrior(c["id"], covariance, residual,
                    {"candidate": c, "architecture": parent, "residual": r}))
            traits.append(TraitTraining(t["id"], rows, variants, y, phi, fixed, scale, tuple(candidates),
                context_spec, fixed_spec, {"units": pheno["units"], "center": center, "scale": scale_y,
                "transform": "linear", "prediction_units": "centered_scaled_phenotype"}, geometry))
        closed(spec["solver"], (), ("rtol", "atol", "max_iterations", "qr_rtol", "max_restarts"), name="solver")
        solver = SolverSpec(**spec["solver"])
        return source, traits, solver
    except BaseException:
        source.close()
        raise


def _score(args):
    import numpy as np
    from ._validation import closed
    from .artifacts import read_json, load_prediction_models, write_json, array_record, _sync_directory
    from .features import evaluate_contexts, evaluate_fixed
    from .genotype import source_from_spec
    from .score import ScoreInput, score_prediction
    models = load_prediction_models(args.models)
    path = Path(args.spec).resolve()
    spec = read_json(path)
    closed(spec, ("kind", "schema_version", "genotypes", "traits"), ("missing_variants",), name="score specification")
    if spec["kind"] != "summit.prediction.score_spec" or spec["schema_version"] != 1:
        raise ValueError("unsupported score specification")
    root = path.parent
    with source_from_spec(spec["genotypes"], root) as source:
        inputs = {}
        for t in spec["traits"]:
            closed(t, ("id", "samples", "contexts", "covariates"), name="scoring trait")
            if t["id"] in inputs:
                raise ValueError("duplicate scoring trait")
            model = next((m for m in models if m.trait_id == t["id"]), None)
            if model is None:
                raise ValueError("unknown scoring trait")
            rows = _rows(source, root/t["samples"])
            samples = [source.samples[int(i)] for i in rows]
            contexts = _aligned_table(root/t["contexts"], samples)
            phi = evaluate_contexts(model.context_spec, contexts)
            covariates = _aligned_table(root/t["covariates"], samples)
            fixed = evaluate_fixed(model.fixed_spec, covariates, phi, model.context_spec["names"])
            inputs[t["id"]] = ScoreInput(rows, phi, fixed, model.context_spec, model.fixed_spec)
        scores = score_prediction(models, source, inputs, block_size=args.block_size, rhs_columns=args.rhs_columns,
            threads=args.num_threads, memory_bytes=int(args.memory_gib*2**30), missing_variants=spec.get("missing_variants", "error"))
        output = Path(args.out)
        output.mkdir(exist_ok=False)
        entries = []
        for i, model in enumerate(models):
            arrays = {}
            for name, mapping in (("components", scores.components), ("genetic", scores.genetic), ("prediction", scores.prediction)):
                target = output/f"{name}-{i}.npy"
                with target.open("xb") as handle:
                    np.save(handle, mapping[model.key], allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                arrays[name] = array_record(target)
            entries.append(dict(trait=model.trait_id, model=model.model_id, model_identity=model.identity,
                phenotype_spec=model.phenotype_spec, arrays=arrays))
        write_json(output/"manifest.json", dict(kind="summit.prediction.scores", schema_version=1,
            samples=scores.samples, models=entries, report=scores.report))
        from .artifacts import file_digest
        write_json(output/"COMPLETE.json", dict(manifest_sha256=file_digest(output/"manifest.json")))
        _sync_directory(output)
    return dict(output=str(output), models=len(models))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="summit-pgs", allow_abbrev=False)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fit", "score", "scale", "inspect"):
        p = subs.add_parser(name, allow_abbrev=False)
        p.add_argument("--num-threads", type=int, default=1)
        p.add_argument("--numa-mode", choices=["none", "membind"], default="none")
        p.add_argument("--numa-nodes", default="all")
        if name in ("plan", "fit", "score"):
            p.add_argument("--spec", required=True)
            p.add_argument("--memory-gib", type=float, default=16)
            p.add_argument("--block-size", type=int, default=512)
            p.add_argument("--rhs-columns", type=int, default=64)
        if name in ("plan", "fit"):
            p.add_argument("--genotype-storage", choices=["stream", "compact", "standardized"], default="stream")
        if name in ("fit", "score", "scale"):
            p.add_argument("--out", required=True)
        if name == "score":
            p.add_argument("--models", required=True)
        if name == "inspect":
            p.add_argument("models")
        if name == "scale":
            p.add_argument("--geno", required=True)
            p.add_argument("--genome-build", required=True)
            p.add_argument("--samples", required=True)
            p.add_argument("--variants")
            p.add_argument("--block-size", type=int, default=512)
            p.add_argument("--memory-gib", type=float, default=16)
    args = parser.parse_args(argv)
    try:
        _runtime(args)
        if hasattr(args, "out") and Path(args.out).exists():
            raise FileExistsError(args.out)
        from ._validation import canonical
        if args.command in ("plan", "fit"):
            from .batch import plan_prediction
            from .api import fit_prediction
            source, traits, solver = _load_fit(args.spec)
            try:
                plan = plan_prediction(traits, source, storage=args.genotype_storage, block_size=args.block_size,
                    rhs_columns=args.rhs_columns, threads=args.num_threads, memory_bytes=int(args.memory_gib*2**30))
                if args.command == "plan":
                    result = plan.to_dict()
                else:
                    models = fit_prediction(traits, source, output=args.out, plan=plan, solver=solver)
                    result = dict(output=args.out, models=len(models))
            finally:
                source.close()
        elif args.command == "score":
            result = _score(args)
        elif args.command == "scale":
            from .genotype import FileGenotypeSource, estimate_scale
            from .artifacts import write_genotype_scale
            with FileGenotypeSource(args.geno, genome_build=args.genome_build) as source:
                scale = estimate_scale(source, _rows(source, args.samples), _variants(source, args.variants),
                    block_size=args.block_size, threads=args.num_threads, memory_bytes=int(args.memory_gib*2**30))
                write_genotype_scale(args.out, scale)
                result = dict(output=args.out, identity=scale.identity)
        else:
            from .artifacts import load_prediction_models
            models = load_prediction_models(args.models)
            result = dict(models=[dict(trait=m.trait_id, model=m.model_id, variants=len(m.variants.ids),
                context_terms=m.weights.shape[1], identity=m.identity, convergence=m.convergence) for m in models])
        print(canonical(result))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, MemoryError, ImportError, OverflowError, FloatingPointError) as exc:
        parser.exit(2, f"summit-pgs: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
