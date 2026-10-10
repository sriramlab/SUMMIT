"""Manifest-based preparation and genotype-free quantitative epistasis fitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, tuple, list)):
        return [_jsonable(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scale = sub.add_parser("scale-test", help="continuous Box-Cox union-null HC3 test on frozen finite features")
    scale.add_argument("arrays",type=Path)
    scale.add_argument("--out",type=Path,required=True)
    scale.add_argument("--groups",type=Path)
    scale.add_argument("--lambda-min",type=float,default=-2.)
    scale.add_argument("--lambda-max",type=float,default=2.)
    scale.add_argument("--alpha",type=float,default=.05)
    scale.add_argument("--max-evaluations",type=int,default=129)
    scale.add_argument("--num-threads",type=int,default=1)
    scale.add_argument("--memory-gib",type=float,default=4.)
    inputs = sub.add_parser(
        "make-inputs", help="construct supplied-target manifests using training genotype support"
    )
    inputs.add_argument("manifest", type=Path)
    inputs.add_argument("--out", type=Path, required=True)
    inputs.add_argument("--num-threads", type=int, default=1)
    prep = sub.add_parser(
        "prepare", help="study-derived reference and analytic trait moments"
    )
    prep.add_argument("manifest", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--num-threads", type=int, default=1)
    prep.add_argument("--block-size", type=int, default=256)
    prep.add_argument("--memory-gib", type=float, default=4)
    prep.add_argument("--nvecs", type=int, default=128)
    prep.add_argument("--seed", type=int, default=1)
    prep.add_argument(
        "--exact", action="store_true", help="tiny identity-probe numerical reference"
    )
    prep.add_argument(
        "--resume",
        action="store_true",
        help="resume cohort preparation and solver checkpoints, or validate completed jobs",
    )
    train = sub.add_parser(
        "train-direction",
        help="train a frozen target-specific direction on independent samples",
    )
    train.add_argument("manifest", type=Path)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--num-threads", type=int, default=1)
    train.add_argument("--block-size", type=int, default=256)
    train.add_argument("--memory-gib", type=float, default=4)
    train.add_argument(
        "--resume",
        action="store_true",
        help="resume the native prediction checkpoint or validate a completed training model",
    )
    fit = sub.add_parser(
        "fit", help="fit saved moments without individual-level inputs"
    )
    fit.add_argument("summary", type=Path)
    fit.add_argument("--out", type=Path, required=True)
    reuse = sub.add_parser(
        "prepare-traits",
        help="prepare new phenotypes using a cohort reference; conditional covariance also needs matched training and genotypes",
    )
    reuse.add_argument("manifest", type=Path)
    reuse.add_argument("--out", type=Path, required=True)
    reuse.add_argument("--num-threads", type=int, default=1)
    reuse.add_argument("--memory-gib", type=float, default=4)
    reuse.add_argument(
        "--block-size", type=int, default=512,
        help="genotype block size for conditional covariance preparation",
    )
    reuse.add_argument(
        "--resume", action="store_true",
        help="resume conditional covariance preparation and its solver checkpoints",
    )
    combine = sub.add_parser(
        "combine", help="harmonize common signed pair-effect summaries"
    )
    combine.add_argument("manifest", type=Path)
    combine.add_argument("--out", type=Path, required=True)
    followup = sub.add_parser(
        "followup",
        help="multiplicity-adjusted conditional follow-up for a supplied pair universe",
    )
    followup.add_argument("manifest", type=Path)
    followup.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "scale-test":
        from .scale_workflow import run_scale_arrays
        run_scale_arrays(args.arrays,args.out,
            groups=None if args.groups is None else json.loads(args.groups.read_text()),
            threads=args.num_threads,memory_bytes=int(args.memory_gib*2**30),
            bounds=(args.lambda_min,args.lambda_max),alpha=args.alpha,
            max_evaluations=args.max_evaluations)
        return 0
    if args.command == "make-inputs":
        from .inputs import prepare_inputs

        prepare_inputs(args.manifest, args.out, threads=args.num_threads)
        return 0
    if args.command == "prepare-traits":
        if json.loads(args.manifest.read_text()).get('inference',{}).get('method') != 'conditional_polygenic_mean':
            if args.resume:
                raise ValueError("prepare-traits --resume currently requires conditional_polygenic_mean")
            from .trait_reuse import prepare_traits

            prepare_traits(
                args.manifest,
                args.out,
                threads=args.num_threads,
                memory_bytes=int(args.memory_gib * 2**30),
            )
            return 0
    if args.command == "followup":
        from .score import load_linear_scores, prespecified_followup
        from summit.prediction._validation import closed

        spec = json.loads(args.manifest.read_text())
        closed(
            spec,
            ("kind", "schema_version", "summary", "groups"),
            ("alpha",),
            name="follow-up manifest",
        )
        if spec["kind"] != "summit.epistasis.followup" or spec["schema_version"] != 1:
            raise ValueError("unsupported follow-up manifest")
        import numpy as np

        summary_path = args.manifest.parent / spec["summary"]
        with np.load(summary_path, allow_pickle=False) as archive:
            kind = json.loads(str(archive["manifest"]))["kind"]
        if kind == "summit.epistasis.robust_score":
            from .robust import load_robust_scores, robust_followup

            summary = load_robust_scores(summary_path)
            procedure = robust_followup
        else:
            summary = load_linear_scores(summary_path)
            procedure = prespecified_followup
        results = [
            procedure(summary, spec["groups"], trait=i, alpha=spec.get("alpha", 0.05))
            for i in range(len(summary.trait_names))
        ]
        with args.out.open("x") as handle:
            json.dump(
                _jsonable(dict(traits=summary.trait_names, results=results)),
                handle,
                indent=2,
                allow_nan=False,
            )
        return 0
    if args.command == "combine":
        from .cohorts import combine_manifest

        combine_manifest(args.manifest, args.out)
        return 0
    if args.command == "fit":
        import numpy as np

        with np.load(args.summary, allow_pickle=False) as archive:
            kind = json.loads(str(archive["manifest"]))["kind"]
        if kind == "summit.epistasis.signed_pair_scores":
            from .pairs import pair_tests, load_pair_scores

            summary = load_pair_scores(args.summary)
            fits = [
                dict(
                    trait=summary.trait_names[i],
                    pair_ids=summary.pair_ids,
                    **pair_tests(
                        summary,
                        trait=i,
                        burden_weights=summary.metadata.get("burden_weights"),
                    ),
                )
                for i in range(len(summary.trait_names))
            ]
            if "cohort_tests" in summary.metadata:
                for i, result in enumerate(fits):
                    result["estimand"] = summary.metadata["estimand"]
                    result["cohort_tests"] = {
                        k: (v[i] if isinstance(v, (tuple, list)) else v)
                        for k, v in summary.metadata["cohort_tests"].items()
                    }
        elif kind == "summit.epistasis.linear_score":
            from .score import load_linear_scores
            from .inference_workflow import fit_linear_artifact

            fits = fit_linear_artifact(load_linear_scores(args.summary))
        elif kind == "summit.epistasis.robust_score":
            from .robust import load_robust_scores, robust_score_tests

            summary = load_robust_scores(args.summary)
            fits = [
                robust_score_tests(
                    summary, trait=i, burden=summary.metadata.get("burden_weights")
                )
                for i in range(len(summary.trait_names))
            ]
        elif kind == "summit.epistasis.refit_bootstrap_score":
            from .inference_workflow import fit_bootstrap_artifact

            fits = fit_bootstrap_artifact(args.summary)
        else:
            from .summary import fit_epistasis, load_summary

            summary = load_summary(args.summary)
            fits = [fit_epistasis(summary, i) for i in range(len(summary.trait_names))]
        with args.out.open("x") as handle:
            json.dump(
                _jsonable(
                    dict(kind="summit.epistasis.fit", schema_version=1, fits=fits)
                ),
                handle,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        return 0
    if args.num_threads < 1 or args.memory_gib <= 0:
        raise ValueError("threads and memory must be positive")
    # Same Linux process-local allocation guard as SUMMIT's research launchers.
    import sys

    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
            raise RuntimeError("process-local THP guard failed")
    # Establish threading before importing numerical libraries. A private-BLIS
    # launcher must already supply its placement contract; never modify it.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        if key in os.environ and os.environ[key] != str(args.num_threads):
            raise ValueError(f"{key} conflicts with --num-threads")
        os.environ.setdefault(key, str(args.num_threads))
    if args.command == "train-direction":
        from .directions import train_direction

        train_direction(
            args.manifest,
            args.out,
            threads=args.num_threads,
            block_size=args.block_size,
            memory_bytes=int(args.memory_gib * 2**30),
            resume=args.resume,
        )
        return 0
    import numpy as np
    from summit.prediction.cli import _aligned_table, _rows
    from summit.prediction.genotype import source_from_spec, native_module
    from summit.prediction.runtime import configure_prediction_threads
    from summit.prediction._validation import closed
    from summit.prediction.artifacts import read_json
    from .models import annotation_weights, target_design
    from .prepare import SelectedStudy, fit_scale
    from .summary import write_summary

    path = args.manifest.resolve()
    spec = read_json(path)
    if spec.get('inference',{}).get('method') == 'conditional_polygenic_mean':
        from .conditional_workflow import prepare_conditional

        prepare_conditional(path,args.out,threads=args.num_threads,
            block_size=getattr(args,'block_size',512),memory_bytes=int(args.memory_gib*2**30),
            resume=getattr(args,'resume',False),exact=getattr(args,'exact',False))
        return 0
    closed(
        spec,
        (
            "kind",
            "schema_version",
            "genotypes",
            "samples",
            "phenotypes",
            "annotations",
            "jobs",
        ),
        (
            "covariates",
            "genotype_scale",
            "phenotype_scale",
            "frozen_scores",
            "fixed_genotypes",
            "strata",
        ),
        name="epistasis manifest",
    )
    if spec["kind"] != "summit.epistasis.prepare" or spec["schema_version"] != 1:
        raise ValueError("unsupported epistasis manifest")
    if not spec["jobs"] or len({j["id"] for j in spec["jobs"]}) != len(spec["jobs"]):
        raise ValueError("job IDs must be nonempty and unique")
    for job in spec["jobs"]:
        closed(
            job,
            ("id", "additive_annotations"),
            (
                "components",
                "groups",
                "pairs",
                "nuisance",
                "burden_weights",
                "coefficient_contrasts",
                "local_variants",
                "dominance_variants",
                "residual_model",
                "inference",
                "trans_target",
                "phenotypes",
                "adjust_scores",
            ),
            name="job",
        )
        if "coefficient_contrasts" in job and job.get("inference", {}).get(
            "method"
        ) not in ("robust_mean", "orthogonal_mean"):
            raise ValueError(
                "coefficient contrasts require robust_mean or orthogonal_mean"
            )
        if ("phenotypes" in job or "adjust_scores" in job) and job.get(
            "inference", {}
        ).get("method") != "robust_mean":
            raise ValueError("job-specific traits and score adjustment require robust_mean")
        if sum(k in job for k in ("components", "groups", "pairs")) != 1:
            raise ValueError("job requires exactly one of components, groups or pairs")
        if (
            ("groups" in job or "pairs" in job)
            and not args.exact
            and "inference" not in job
        ):
            raise ValueError("group and pair manifest jobs currently require --exact")
        if (("pairs" in job) != ("nuisance" in job) and "inference" not in job) or (
            "burden_weights" in job and "pairs" not in job
        ):
            raise ValueError(
                "pair jobs require nuisance; burden weights apply only to pairs"
            )
        if ("pairs" in job or "inference" in job) and spec.get(
            "phenotype_scale", "raw"
        ) != "raw":
            raise ValueError(
                "pair covariance is in raw phenotype units; use phenotype_scale raw"
            )
        if (
            not job["id"]
            or Path(job["id"]).name != job["id"]
            or job["id"] in (".", "..")
        ):
            raise ValueError("job ID must be a file-name component")
    if args.resume and any(
        j.get("inference", {}).get("method") != "robust_mean" for j in spec["jobs"]
    ):
        raise ValueError("resume currently requires all jobs to use robust_mean")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (args.out / "preparation.json").exists():
        previous = read_json(args.out / "preparation.json")
        if previous["manifest"] != spec:
            raise ValueError(
                "completed preparation manifest changed; use a new output path"
            )
    native = native_module()
    configure_prediction_threads(native, args.num_threads)
    source = source_from_spec(spec["genotypes"], path.parent)
    try:
        rows = _rows(source, path.parent / spec["samples"])
        rows.sort()
        samples = [source.samples[i] for i in rows]
        strata = None
        if "strata" in spec:
            from .trans import stratum_definition

            closed(spec["strata"], ("file", "column"), name="fixed-effect strata")
            if any("trans_target" not in j for j in spec["jobs"]):
                raise ValueError(
                    "stratum-specific nuisance expansion currently requires supplied trans jobs"
                )
            strata = _aligned_table(path.parent / spec["strata"]["file"], samples)[
                spec["strata"]["column"]
            ].to_numpy(str)
            stratum_record = stratum_definition(strata, n_samples=len(samples))
        p = spec["phenotypes"]
        closed(p, ("file", "columns"), ("unit",), name="phenotypes")
        if "unit" in p and (not isinstance(p["unit"], str) or not p["unit"]):
            raise ValueError("phenotype unit must be a nonempty scientific unit label")
        y = _aligned_table(path.parent / p["file"], samples)[p["columns"]].to_numpy(
            float
        )
        cov = None
        structure_names = []
        if spec.get("covariates"):
            cv = spec["covariates"]
            closed(cv, ("file", "columns"), ("varying_effects",), name="covariates")
            cov = _aligned_table(path.parent / cv["file"], samples)[
                cv["columns"]
            ].to_numpy(float)
            from .nuisance import select_structure_covariates
            structure_names, structure_cov = select_structure_covariates(cv, cov)
            if structure_names and any(
                "trans_target" not in j or "components" not in j
                or j.get("inference", {}).get("method") != "robust_mean"
                or any("target" in component for component in j.get("components", []))
                for j in spec["jobs"]
            ):
                raise ValueError("varying_effects currently requires supplied trans target-by-score robust_mean jobs")
        frozen_scores = None
        frozen_report = None
        fixed_genotype_report = None
        if spec.get("fixed_genotypes"):
            from .fixed_genotypes import prepare_fixed_genotypes

            extra, fixed_genotype_report = prepare_fixed_genotypes(
                spec["fixed_genotypes"],
                path.parent,
                samples,
                threads=args.num_threads,
                block_size=args.block_size,
                memory_bytes=int(args.memory_gib * 2**30)
                - y.nbytes
                - (0 if cov is None else cov.nbytes),
            )
            cov = extra if cov is None else np.column_stack([cov, extra])
            del extra
        main_imputation = {}
        if spec.get("frozen_scores"):
            from .directions import score_frozen

            main_variants = set()
            for job in spec["jobs"]:
                policy = job.get("inference", {}).get("main_effects", "tested_variants")
                if (
                    policy in ("tested_variants", "all_genotypes")
                    and len(source.variants.ids) <= 4096
                ):
                    main_variants.update(source.variants.ids)
                main_variants.update(
                    c["target"] for c in job.get("components", ()) if "target" in c
                )
                main_variants.update(
                    v for c in job.get("components", ()) for v in c.get("score", {})
                )
                main_variants.update(job.get("local_variants", ()))
                main_variants.update(v for pair in job.get("pairs", ()) for v in pair)
                if (
                    job.get("inference", {}).get("main_effects", "tested_variants")
                    == "tested_variants"
                ):
                    for component in job.get("components", ()):
                        main_variants.update(
                            spec["annotations"].get(component["background"], {})
                        )
                    for group in job.get("groups", ()):
                        main_variants.update(spec["annotations"].get(group["left"], {}))
                        main_variants.update(
                            spec["annotations"].get(group.get("right"), {})
                        )
            frozen_scores, adjust, frozen_report = score_frozen(
                spec["frozen_scores"],
                path.parent,
                source,
                rows,
                threads=args.num_threads,
                block_size=args.block_size,
                memory_bytes=int(args.memory_gib * 2**30)
                - y.nbytes
                - (0 if cov is None else cov.nbytes),
                main_variants=main_variants,
            )
            for score in frozen_scores.values():
                for variant, mean in score["main_imputation"].items():
                    if variant in main_imputation and main_imputation[variant] != mean:
                        raise ValueError(
                            "frozen scores use incompatible main-effect imputation; prepare separately"
                        )
                    main_imputation[variant] = mean
            cov = adjust if cov is None else np.column_stack([cov, adjust])
            del adjust
        scale = fit_scale(
            source,
            rows,
            method=spec.get("genotype_scale", "hwe"),
            threads=args.num_threads,
            block_size=args.block_size,
            memory_bytes=int(args.memory_gib * 2**30)
            - y.nbytes
            - (0 if cov is None else cov.nbytes)
            - (
                sum(s["values"].nbytes for s in frozen_scores.values())
                if frozen_scores
                else 0
            ),
        )
        annotations = annotation_weights(source.variants.ids, spec["annotations"])
        base_bytes = (
            y.nbytes
            + (0 if cov is None else cov.nbytes)
            + sum(a.nbytes for a in annotations.values())
        )
        base_bytes += scale.mean.nbytes + scale.inverse_scale.nbytes
        if frozen_scores:
            base_bytes += sum(s["values"].nbytes for s in frozen_scores.values())
        available_bytes = int(args.memory_gib * 2**30) - base_bytes
        if available_bytes <= 64 * 2**20:
            raise MemoryError("aligned cohort inputs exceed memory budget")
        records = []
        shared_sources = {}
        prespecified_scores = {}
        shared_source_passes = 0
        for job in spec["jobs"]:
            trait_names = job.get("phenotypes", p["columns"])
            if (not isinstance(trait_names, list) or not trait_names
                    or len(set(trait_names)) != len(trait_names)
                    or any(name not in p["columns"] for name in trait_names)):
                raise ValueError("job phenotypes must select distinct prepared traits")
            job_y = y[:, [p["columns"].index(name) for name in trait_names]]
            job_cov = cov
            adjust_names = job.get("adjust_scores", [])
            if (not isinstance(adjust_names, list)
                    or len(set(adjust_names)) != len(adjust_names)
                    or any(name not in (frozen_scores or {}) for name in adjust_names)):
                raise ValueError("job adjust_scores must select distinct frozen scores")
            if adjust_names:
                extra = np.column_stack([frozen_scores[name]["values"] for name in adjust_names])
                job_cov = extra if cov is None else np.column_stack([cov, extra])
            trans_record = None
            if "trans_target" in job:
                from .trans import validate_trans_job

                trans_record = validate_trans_job(
                    job, source.variants, annotations, frozen_scores
                )
            local = list(job.get("local_variants", ()))
            dominance = list(job.get("dominance_variants", ()))
            if strata is not None and "pairs" in job:
                involved = sorted({v for pair in job["pairs"] for v in pair})
                local = list(dict.fromkeys(local + involved))
                if source.hard_calls:
                    dominance = list(dict.fromkeys(dominance + involved))
            if job.get("inference", {}).get("method") in (
                "robust_mean",
                "orthogonal_mean",
            ):
                required = set()
                for component in job.get("components", ()):
                    if "target" in component:
                        required.add(component["target"])
                    selected = np.flatnonzero(annotations[component["background"]] > 0)
                    # A SNP-by-score test must retain both main effects. The
                    # score itself is already included by target_design.
                    if len(selected) == 1:
                        required.add(source.variants.ids[selected[0]])
                local = list(dict.fromkeys(local + sorted(required)))
                if source.hard_calls:
                    dominance = list(dict.fromkeys(dominance + sorted(required)))
            design = target_design(
                source,
                rows,
                scale,
                components=job.get("components", ()),
                annotations=annotations,
                additive_annotations=job["additive_annotations"],
                covariates=job_cov,
                local_variants=local,
                dominance_variants=dominance,
                threads=args.num_threads,
                block_size=args.block_size,
                native=native,
                allow_additive_only="components" not in job,
                frozen_scores=frozen_scores,
                main_imputation=main_imputation,
                memory_bytes=available_bytes
                - sum(value.nbytes for value in prespecified_scores.values())
                - sum(
                    a.nbytes
                    for batch in shared_sources.values()
                    for a in batch["sources"].values()
                ),
                score_cache=prespecified_scores,
            )
            residual_model = job.get("residual_model", "iid")
            if structure_names:
                from .nuisance import varying_main_effects
                target_column = 1 + (0 if job_cov is None else job_cov.shape[1]) + local.index(job["trans_target"])
                mains = [design["fixed_effects"][:, target_column]]
                main_names = ["target:" + job["trans_target"]]
                begin = len(job["additive_annotations"])
                for i, component in enumerate(job["components"]):
                    mains.append(design["modifiers"][:, begin + i])
                    main_names.append("component:" + component["name"])
                for definition in spec.get("frozen_scores", []):
                    name = definition["name"]
                    powers = ([1] if definition.get("adjust") else definition.get("adjust_powers", []))
                    if name in adjust_names and 1 not in powers:
                        powers = [*powers, 1]
                    for power in powers:
                        mains.append(frozen_scores[name]["values"] ** power)
                        main_names.append(f"adjustment:{name}:{power}")
                design["fixed_effects"], design["definitions"]["varying_main_effects"] = varying_main_effects(
                    design["fixed_effects"], structure_cov, np.column_stack(mains),
                    covariate_names=structure_names, main_names=main_names,
                    memory_bytes=available_bytes,
                )
            if trans_record is not None:
                design["definitions"]["trans_membership"] = trans_record
            if strata is not None:
                design["definitions"]["fixed_effect_strata"] = stratum_record
            if fixed_genotype_report is not None:
                design["definitions"]["fixed_genotype_sources"] = fixed_genotype_report
            if (
                "components" not in job
                and residual_model != "iid"
                and "inference" not in job
            ):
                raise ValueError(
                    "group and pair manifests currently require iid residual"
                )
            if residual_model == "modifier_squares":
                if "components" not in job:
                    raise ValueError(
                        "modifier_squares requires declared modifier components"
                    )
                begin = len(job["additive_annotations"])
                design["residual_basis"] = np.column_stack(
                    [design["modifiers"][:, begin:] ** 2, np.ones(len(rows))]
                )
                design["residual_names"] = tuple(
                    "residual_square:" + n for n in design["component_names"][begin:]
                ) + ("residual",)
            elif residual_model == "feature_diagonal" and "inference" in job:
                pass
            elif residual_model != "iid":
                raise ValueError(
                    "residual_model must be iid, modifier_squares, or score-only feature_diagonal"
                )
            study = SelectedStudy(
                source,
                rows,
                scale,
                **design,
                threads=args.num_threads,
                block_size=args.block_size,
                memory_bytes=available_bytes,
                metadata_variants=sorted(
                    set(local) | set(dominance)
                    | {v for pair in job.get("pairs", ()) for v in pair}
                ) if job.get("inference", {}).get("method") == "robust_mean" else None,
            )
            if "inference" in job:
                from .inference_workflow import prepare_score_job

                dimensions = job["inference"].get("feature_sketch_dimensions")
                cache = None
                if (
                    dimensions is not None
                    and "components" in job
                    and not job["inference"].get("reference")
                    and not (
                        args.resume
                        and (args.out / (job["id"] + ".robust-score.npz")).exists()
                    )
                ):
                    if dimensions not in shared_sources:
                        # Retain only one dimension batch, so independently
                        # sized sketches cannot silently accumulate memory.
                        shared_sources.clear()
                        from .features import prepare_shared_target_sources

                        lookup = {v: i for i, v in enumerate(source.variants.ids)}
                        weights = []
                        for other in spec["jobs"]:
                            settings = other.get("inference", {})
                            if (
                                "components" not in other
                                or settings.get("feature_sketch_dimensions")
                                != dimensions
                                or settings.get("reference")
                            ):
                                continue
                            for component in other["components"]:
                                weight = annotations[component["background"]].copy()
                                excluded = set(component.get("exclude", ()))
                                if "target" in component:
                                    excluded.add(component["target"])
                                for variant in excluded:
                                    weight[lookup[variant]] = 0
                                weights.append(weight)
                        shared_sources[dimensions] = prepare_shared_target_sources(
                            study, weights, dimensions=dimensions, seed=args.seed
                        )
                        shared_source_passes += 1
                    cache = shared_sources[dimensions]
                output, passes = prepare_score_job(
                    study,
                    job,
                    annotations,
                    job_y,
                    trait_names=trait_names,
                    output=args.out,
                    root=path.parent,
                    seed=args.seed,
                    shared_sources=cache,
                    trait_unit=p.get("unit"),
                    resume=args.resume,
                    strata=strata,
                )
                records.append(
                    dict(
                        id=job["id"],
                        summary=output.name,
                        genotype_passes=passes,
                        target_and_fixed_variant_reads=design["definitions"][
                            "targeted_variant_reads"
                        ],
                    )
                )
                continue
            if "pairs" in job:
                from .pairs import prepare_bounded_pair_summary, write_pair_scores

                summary = prepare_bounded_pair_summary(
                    study,
                    job["pairs"],
                    y,
                    job["nuisance"],
                    trait_names=p["columns"],
                    burden_weights=job.get("burden_weights"),
                )
                if "unit" in p:
                    from dataclasses import replace

                    summary = replace(
                        summary, metadata=dict(summary.metadata, trait_unit=p["unit"])
                    )
                output = write_pair_scores(
                    summary, args.out / (job["id"] + ".pairs.npz")
                )
            elif "groups" in job:
                from .groups import prepare_group_summary

                summary = prepare_group_summary(
                    study,
                    job["groups"],
                    annotations,
                    y,
                    trait_names=p["columns"],
                    phenotype_scale=spec.get("phenotype_scale", "raw"),
                )
            else:
                reference = study.reference(
                    nvecs=args.nvecs, seed=args.seed, exact=args.exact
                )
                summary, _ = study.summarize(
                    reference,
                    y,
                    trait_names=p["columns"],
                    phenotype_scale=spec.get("phenotype_scale", "raw"),
                )
            if "pairs" not in job:
                output = write_summary(
                    summary, args.out / (job["id"] + ".epistasis.npz")
                )
            records.append(
                dict(
                    id=job["id"],
                    summary=output.name,
                    genotype_passes=study.stream.ledger.traversals,
                )
            )
        if not (args.out / "preparation.json").exists():
            with (args.out / "preparation.json").open("x") as handle:
                json.dump(
                    dict(
                        manifest=spec,
                        jobs=records,
                        scale_passes=1,
                        shared_target_source_passes=shared_source_passes,
                        frozen_score_report=frozen_report,
                        fixed_genotype_report=fixed_genotype_report,
                        native_module=native.__file__,
                        native_build=native.build_info(),
                    ),
                    handle,
                    indent=2,
                )
                handle.write("\n")
        from summit.prediction.artifacts import file_digest, write_json

        receipt = dict(
            kind="summit.epistasis.preparation_complete",
            schema_version=1,
            preparation_sha256=file_digest(args.out / "preparation.json"),
            summaries={
                r["summary"]: file_digest(args.out / r["summary"]) for r in records
            },
        )
        if (args.out / "COMPLETE.json").exists():
            if read_json(args.out / "COMPLETE.json") != receipt:
                raise ValueError("preparation completion digest mismatch")
        else:
            write_json(args.out / "COMPLETE.json", receipt)
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
