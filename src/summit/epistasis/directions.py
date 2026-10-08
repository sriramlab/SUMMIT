"""Independently train and score frozen interaction directions with SUMMIT prediction."""
from pathlib import Path
from dataclasses import asdict, replace
import json
import numpy as np
from summit.context.spec import canonical_sha256
from summit.prediction._validation import closed
from summit.prediction.artifacts import load_prediction_models


def sample_tokens(samples):
    # Local provenance, not a claim of anonymization. Keep with cohort artifacts.
    return [canonical_sha256([str(a), str(b)]) for a, b in samples]


def train_direction(
    manifest,
    output,
    *,
    threads=1,
    block_size=256,
    memory_bytes=4 * 2**30,
    resume=False,
):
    """Prespecified interaction learning and a separate additive-null predictor.

    The interaction direction retains its joint additive/interaction learner.
    The nuisance PGS has zero interaction prior variance, so fitted interaction
    effects cannot divert its additive prediction. Both use only training
    outcomes, without phenotype-dependent tuning.
    """
    from summit.prediction.cli import _rows, _variants, _aligned_table
    from summit.prediction.genotype import (
        source_from_spec,
        native_module,
        estimate_scale,
    )
    from summit.prediction.spec import TraitTraining, CandidatePrior, SolverSpec
    from summit.prediction.batch import plan_prediction
    from summit.prediction.api import fit_prediction
    from .models import target_design
    from .prepare import fit_scale

    path = Path(manifest).resolve()
    spec = json.loads(path.read_text())
    root = path.parent
    closed(
        spec,
        (
            "kind",
            "schema_version",
            "genotypes",
            "samples",
            "target",
            "variants",
            "prior",
        ),
        (
            "covariates",
            "local_variants",
            "dominance_variants",
            "storage",
            "solver",
            "id",
            "interaction_variants",
            "trans_only",
            "phenotype",
            "phenotypes",
        ),
        name="direction training",
    )
    if (
        spec["kind"] != "summit.epistasis.train_direction"
        or spec["schema_version"] != 1
    ):
        raise ValueError("unsupported direction training manifest")
    if ("phenotype" in spec) == ("phenotypes" in spec):
        raise ValueError("supply phenotype or a shared-mask phenotypes batch")
    batched = "phenotypes" in spec
    phenotype = spec["phenotypes" if batched else "phenotype"]
    closed(phenotype, ("file", "columns" if batched else "column", "unit"),
           name="training phenotypes")
    columns = phenotype["columns"] if batched else [phenotype["column"]]
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(c, str) or not c for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("training phenotype columns must be distinct nonempty names")
    closed(
        spec["prior"],
        ("additive", "interaction", "residual"),
        name="prespecified direction prior",
    )
    prior = spec["prior"]
    if any(not np.isfinite(v) or v <= 0 for v in prior.values()):
        raise ValueError("training ridge variances must be positive and prespecified")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    with source_from_spec(spec["genotypes"], root) as source:
        rows = np.sort(_rows(source, root / spec["samples"]))
        samples = [source.samples[i] for i in rows]
        variants = _variants(source, root / spec["variants"])
        if type(spec.get("trans_only", False)) is not bool:
            raise ValueError("trans_only must be boolean")
        trans_record = None
        if spec.get("trans_only"):
            from .trans import trans_membership

            selected = (
                _variants(source, root / spec["interaction_variants"])
                if spec.get("interaction_variants")
                else variants
            )
            trans_record = trans_membership(
                source.variants,
                spec["target"],
                [source.variants.ids[i] for i in selected],
            )
        if spec["target"] in set(source.variants.ids[i] for i in variants):
            raise ValueError("training direction must exclude the target locus")
        # Scale only the fitted background; target/local reads use a separate
        # common study scale. This wrapper currently requires the input axis
        # to have passed the same polymorphism checks as epistasis preparation.
        scale = fit_scale(
            source,
            rows,
            threads=threads,
            block_size=block_size,
            memory_bytes=memory_bytes,
        )
        cov = None
        structure_names = []
        if spec.get("covariates"):
            cv = spec["covariates"]
            closed(cv, ("file", "columns"), ("varying_effects",), name="training covariates")
            cov = _aligned_table(root / cv["file"], samples)[cv["columns"]].to_numpy(
                float
            )
            from .nuisance import select_structure_covariates
            structure_names, structure_cov = select_structure_covariates(cv, cov)
        design = target_design(
            source,
            rows,
            scale,
            components=[
                dict(name="interaction", target=spec["target"], background="all")
            ],
            annotations={"all": np.ones(len(source.variants.ids))},
            additive_annotations=["all"],
            covariates=cov,
            local_variants=spec.get("local_variants", ()),
            dominance_variants=spec.get("dominance_variants", ()),
            threads=threads,
            block_size=block_size,
            native=native_module(),
            memory_bytes=memory_bytes
            - scale.mean.nbytes
            - scale.inverse_scale.nbytes
            - (0 if cov is None else cov.nbytes),
        )
        if structure_names:
            from .nuisance import varying_main_effects
            design["fixed_effects"], design["definitions"]["varying_main_effects"] = varying_main_effects(
                design["fixed_effects"], structure_cov, design["modifiers"][:, 1:],
                covariate_names=structure_names, main_names=[spec["target"]],
                memory_bytes=memory_bytes - scale.mean.nbytes - scale.inverse_scale.nbytes,
            )
        from summit.prediction.spec import GenotypeScale

        fitted_scale = GenotypeScale(
            scale.mean[variants],
            scale.inverse_scale[variants],
            source.variants.subset(variants).identity,
            scale.sample_identity,
            dict(scale.provenance),
            ddof=scale.ddof,
        )
        y = _aligned_table(root / phenotype["file"], samples)[columns].to_numpy(float)
        phi = design["modifiers"]
        contexts = dict(
            kind="summit.epistasis.frozen_training_contexts",
            names=["baseline", "target"],
        )
        fixed = dict(
            kind="summit.epistasis.frozen_training_fixed",
            names=[f"c{i}" for i in range(design["fixed_effects"].shape[1])],
        )
        candidate = CandidatePrior(
            "prespecified",
            np.diag([prior["additive"], prior["interaction"]]),
            np.full(len(rows), prior["residual"]),
            dict(method="prespecified_ridge_direction", prior=prior),
        )
        if spec.get("interaction_variants"):
            from summit.prediction.annotations import AnnotationDesign, AnnotationPrior

            selected = _variants(source, root / spec["interaction_variants"])
            if not set(selected) <= set(variants):
                raise ValueError(
                    "interaction training variants must be within the background axis"
                )
            annotation = AnnotationDesign(
                np.column_stack([np.ones(len(variants)), np.isin(variants, selected)]),
                ("additive_background", "interaction_background"),
                fitted_scale.variant_identity,
            )
            annotated = AnnotationPrior(
                annotation,
                np.array(
                    [
                        [[prior["additive"], 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, prior["interaction"]]],
                    ]
                ),
            )
            candidate = annotated.candidate(
                "prespecified",
                np.full(len(rows), prior["residual"]),
                dict(method="prespecified_ridge_direction", prior=prior),
            )
        additive_null = CandidatePrior(
            "additive_null", np.diag([prior["additive"], 0.0]),
            np.full(len(rows), prior["residual"]),
            dict(method="prespecified_additive_null_nuisance",
                 additive=prior["additive"], residual=prior["residual"]),
        )
        traits = [TraitTraining(
            spec.get("id", "direction") + (f".{i}" if batched else ""),
            rows,
            variants,
            y[:, i],
            phi,
            design["fixed_effects"],
            fitted_scale,
            (candidate, additive_null),
            contexts,
            fixed,
            dict(units=phenotype["unit"], transform="raw"),
        ) for i in range(len(columns))]
        plan = plan_prediction(
            traits,
            source,
            storage=spec.get("storage", "stream"),
            block_size=block_size,
            rhs_columns=max(8, min(64, 4 * len(traits))),
            threads=threads,
            memory_bytes=memory_bytes
            - sum(v.nbytes for v in design.values() if isinstance(v, np.ndarray))
            - scale.mean.nbytes
            - scale.inverse_scale.nbytes
            - (0 if cov is None else cov.nbytes),
        )
        if resume and (output / "models").exists():
            models = load_prediction_models(output / "models")
            saved = json.loads((output / "models/manifest.json").read_text())
            if saved["run_report"]["plan"]["fit_identity"] != plan.fit_identity:
                raise ValueError(
                    "completed training model inputs changed; use a new output path"
                )
            if saved["provenance"]["solver"] != asdict(
                SolverSpec(**spec.get("solver", {}))
            ):
                raise ValueError(
                    "completed training solver settings changed; use a new output path"
                )
        else:
            checkpoint = output / "solver.npz"
            models = fit_prediction(
                traits,
                source,
                output=output / "models",
                plan=plan,
                solver=SolverSpec(**spec.get("solver", {})),
                checkpoint=checkpoint,
                resume=resume and checkpoint.exists(),
            )
        common = dict(
            kind="summit.epistasis.frozen_direction",
            schema_version=1,
            models="models",
            training_samples=sample_tokens(samples),
            training_source=source.identity,
            training_sample_identity=scale.sample_identity,
            training_target=spec["target"],
            training_manifest=spec,
            tuning="prespecified interaction learner and additive-null nuisance fit; no held-out outcomes used",
            roles=["additive_PGS", "learned_interaction_direction"],
        )
        if trans_record is not None:
            common["trans_membership"] = trans_record
        records = []
        by_model = {m.key: m for m in models}
        for i, trait in enumerate(traits):
            record = dict(common, model_identity=by_model[(trait.id, "prespecified")].identity,
                          additive_model_identity=by_model[(trait.id, "additive_null")].identity)
            if batched:
                record["phenotype_column"] = columns[i]
            filename = f"direction.{i}.json" if batched else "direction.json"
            path = output / filename
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError("completed frozen direction inputs changed")
            else:
                with path.open("x") as handle:
                    handle.write(json.dumps(record, indent=2) + "\n")
            path.chmod(0o600)
            records.append(dict(phenotype=columns[i], direction=filename))
        if batched:
            path = output / "directions.json"
            index = dict(kind="summit.epistasis.direction_batch", schema_version=1,
                         directions=records)
            if path.exists():
                if json.loads(path.read_text()) != index:
                    raise ValueError("completed direction batch changed")
            else:
                with path.open("x") as handle:
                    json.dump(index, handle, indent=2)
            path.chmod(0o600)
    return output / ("directions.json" if batched else "direction.json")


def score_frozen(
    definitions,
    root,
    source,
    rows,
    *,
    threads,
    block_size,
    memory_bytes,
    main_variants=(),
):
    """Align frozen alleles/scales, verify nonoverlap, share one prediction pass."""
    from summit.prediction.score import ScoreInput, score_prediction, align_variants

    if not isinstance(definitions, list) or len(
        {d["name"] for d in definitions}
    ) != len(definitions):
        raise ValueError("frozen scores require distinct names")
    heldout = set(sample_tokens([source.samples[i] for i in rows]))
    models = {}
    inputs = {}
    entries = []
    loaded_cache = {}
    model_aliases = {}
    original_model_names = {}
    for d in definitions:
        closed(
            d,
            ("name", "direction", "component"),
            ("adjust", "adjust_powers"),
            name="frozen score",
        )
        powers = d.get("adjust_powers", ())
        if (
            not isinstance(powers, (list, tuple))
            or len(set(powers)) != len(powers)
            or any(type(v) is not int or not 1 <= v <= 3 for v in powers)
        ):
            raise ValueError("score adjustment powers must be distinct integers 1..3")
        if d.get("adjust", False) and powers:
            raise ValueError("use adjust or adjust_powers, not both")
        path = (Path(root) / d["direction"]).resolve()
        record = json.loads(path.read_text())
        if (
            record.get("kind") != "summit.epistasis.frozen_direction"
            or record.get("schema_version") != 1
        ):
            raise ValueError("unsupported frozen direction")
        if not record.get("training_samples") or heldout.intersection(
            record["training_samples"]
        ):
            raise ValueError(
                "training and confirmation samples overlap or separation is unavailable"
            )
        cache_key = str(path.parent / record["models"])
        if cache_key not in loaded_cache:
            loaded_cache[cache_key] = load_prediction_models(
                path.parent / record["models"]
            )
        loaded = loaded_cache[cache_key]
        selected_identity = (record.get("additive_model_identity", record["model_identity"])
                             if d["component"] == 0 else record["model_identity"])
        matching = [m for m in loaded if m.identity == selected_identity]
        if len(matching) != 1:
            raise ValueError("frozen direction model identity mismatch")
        model = matching[0]
        # Model keys are local to each saved bundle. Two independently trained
        # targets can legitimately both be named "direction/prespecified".
        # Use an execution-local namespace without changing authenticated model
        # identities, numeric arrays, or published training records.
        if model.identity not in model_aliases:
            alias = replace(model, trait_id=f"frozen{len(model_aliases)}")
            model_aliases[model.identity] = alias
            original_model_names["/".join(alias.key)] = dict(
                trait_id=model.trait_id, model_id=model.model_id, identity=model.identity
            )
        model = model_aliases[model.identity]
        if (
            type(d["component"]) is not int
            or not 0 <= d["component"] < model.weights.shape[1]
        ):
            raise ValueError("invalid frozen score component")
        # Only genotype component scores are required. Do not allocate N times
        # the fitted nuisance rank in dummy zero fixed-effect values.
        models[model.key] = model
        owned_inputs = sum(v.phi.nbytes + v.fixed.nbytes for v in inputs.values())
        planned_input = (
            16 * len(rows) * model.weights.shape[1]
        )
        model_storage = sum(
            v.weights.nbytes + v.scale.mean.nbytes + v.scale.inverse_scale.nbytes
            for v in models.values()
        )
        if owned_inputs + planned_input + model_storage + 256 * 2**20 >= memory_bytes:
            raise MemoryError(
                "frozen scoring input designs exceed remaining cohort memory budget"
            )
        inputs[model.trait_id] = ScoreInput(
            rows,
            np.ones((len(rows), model.weights.shape[1])),
            np.empty((len(rows), 0)),
            model.context_spec,
            model.fixed_spec,
        )
        entries.append((d, record, model))
    result = score_prediction(
        models.values(),
        source,
        inputs,
        threads=threads,
        block_size=block_size,
        memory_bytes=memory_bytes
        - sum(v.phi.nbytes + v.fixed.nbytes for v in inputs.values()),
        components_only=True,
    )
    result.report["frozen_model_names"] = original_model_names
    scores = {}
    adjust = []
    requested_main = set(main_variants)
    for d, record, model in entries:
        values = result.components[model.key][:, d["component"]]
        # All fitted-axis variants are conservatively treated as constituents.
        mr, sr, flips = align_variants(model.variants, source.variants)
        imputation = {
            source.variants.ids[i]: float(
                2 - model.scale.mean[j] if flip else model.scale.mean[j]
            )
            for j, i, flip in zip(mr, sr, flips)
            if source.variants.ids[i] in requested_main
        }
        scores[d["name"]] = dict(
            values=values,
            main_imputation=imputation,
            identity=canonical_sha256([model.identity, d["component"]]),
            variants=model.variants.ids,
            nonzero_variants=[
                v
                for v, w in zip(model.variants.ids, model.weights[:, d["component"]])
                if w != 0
            ],
            role=record["roles"][d["component"]],
        )
        if d.get("adjust", False):
            adjust.append(values)
        adjust.extend(values**power for power in d.get("adjust_powers", ()))
    return (
        scores,
        (np.column_stack(adjust) if adjust else np.empty((len(rows), 0))),
        result.report,
    )
