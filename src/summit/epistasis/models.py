"""External target definitions and their explicit additive safeguards."""
from __future__ import annotations

import numpy as np

from summit.prediction.genotype import StandardizedBlock
from summit.context.spec import canonical_sha256, array_sha256


def annotation_weights(variant_ids, definitions):
    """Resolve named nonnegative SNP weights by ID, never by unchecked position."""
    lookup = {v: i for i, v in enumerate(variant_ids)}
    result = {"all": np.ones(len(variant_ids))}
    for name, entries in definitions.items():
        if name == "all" or not name or not isinstance(entries, dict) or not entries:
            raise ValueError(
                "annotations require distinct names and nonempty SNP-weight mappings"
            )
        weight = np.zeros(len(variant_ids))
        for variant, value in entries.items():
            if variant not in lookup:
                raise ValueError(f"unknown annotation variant {variant}")
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("annotation weights must be finite and nonnegative")
            weight[lookup[variant]] = value
        if weight.sum() <= 0:
            raise ValueError("empty annotation")
        result[name] = weight
    return result


def target_design(
    source,
    rows,
    scale,
    *,
    components,
    annotations,
    additive_annotations,
    covariates=None,
    local_variants=(),
    dominance_variants=(),
    threads=1,
    block_size=256,
    native=None,
    allow_additive_only=False,
    frozen_scores=None,
    main_imputation=None,
    dominance_imputation=None,
    memory_bytes=2**30,
    score_cache=None,
):
    """Prepare targets or fixed weighted scores, plus declared fixed effects.

    Target and score main effects are always fixed effects. Local adjustment
    and dominance are separate explicit safeguards, never exclusions inferred
    from physical proximity. ``exclude`` lists only interaction backgrounds.
    """
    lookup = {v: i for i, v in enumerate(source.variants.ids)}
    main_imputation = {} if main_imputation is None else dict(main_imputation)
    dominance_imputation = {} if dominance_imputation is None else dict(dominance_imputation)
    if set(dominance_imputation)-set(dominance_variants):
        raise ValueError('frozen heterozygosity must refer to declared dominance variants')
    if any(not np.isscalar(v) or not np.isfinite(v) or not 0<=v<=1
           for v in dominance_imputation.values()):
        raise ValueError('frozen heterozygosity must be a probability in [0,1]')
    if (
        (not components and not allow_additive_only)
        or not additive_annotations
        or additive_annotations[0] != "all"
    ):
        raise ValueError(
            "supply interactions and genome-wide additive adjustment first"
        )
    needed = set(local_variants) | set(dominance_variants)
    names, definitions = [], []
    for component in components:
        allowed = {"name", "target", "score", "frozen_score", "background", "exclude"}
        if set(component) - allowed or not {"name", "background"} <= set(component):
            raise ValueError("invalid interaction component fields")
        if sum(k in component for k in ("target", "score", "frozen_score")) != 1:
            raise ValueError(
                "declare exactly one target SNP, weighted score or frozen score"
            )
        if "target" in component:
            needed.add(component["target"])
        elif "score" in component:
            if not isinstance(component["score"], dict) or not component["score"]:
                raise ValueError("score requires prespecified SNP weights")
            if set(component["score"]) - lookup.keys():
                raise ValueError("unknown score variants")
        elif frozen_scores is None or component["frozen_score"] not in frozen_scores:
            raise ValueError("unknown frozen score")
        if component["background"] not in annotations:
            raise ValueError("unknown interaction background")
        if set(component.get("exclude", ())) - lookup.keys():
            raise ValueError("unknown excluded variant")
    if needed - lookup.keys():
        raise ValueError(
            f"unknown target/fixed/excluded variants: {sorted(needed-lookup.keys())}"
        )
    if dominance_variants and not source.hard_calls:
        raise ValueError(
            "heterozygote dominance adjustment requires a declared hard-call source"
        )
    score_components = [d for d in components if "score" in d]
    cached_scores = {}
    cache_keys = {}
    if score_cache is not None and score_components:
        source.check()
        common = dict(source=source.identity, rows=array_sha256(np.asarray(rows)),
                      scale=scale.identity, main_imputation=main_imputation)
        for j, component in enumerate(score_components):
            key = canonical_sha256(dict(common, weights=component["score"]))
            cache_keys[j] = key
            if key in score_cache:
                values = score_cache[key]
                if values.shape != (len(rows),) or values.flags.writeable:
                    raise ValueError("invalid internal prespecified-score cache")
                cached_scores[j] = values
    if covariates is not None and np.asarray(covariates).ndim != 2:
        raise ValueError("covariates must be a sample-by-covariate matrix")
    covariate_count = 0 if covariates is None else np.asarray(covariates).shape[1]
    fixed_count = (
        1
        + covariate_count
        + len(local_variants)
        + len(dominance_variants)
        + len(components)
    )
    planned = 8 * (
        len(rows)
        * (
            4 * fixed_count
            + 3 * len(needed)
            + 4 * len(score_components)
            + 3 * (len(components) + len(additive_annotations))
        )
        + len(lookup) * len(score_components)
    )
    planned += 32 * len(rows) * min(block_size, len(lookup)) + 64 * 2**20
    if planned > memory_bytes:
        raise MemoryError(
            "target, score and fixed-effect preparation exceeds memory budget"
        )
    score_weights = np.zeros((len(lookup), len(score_components)))
    for j, d in enumerate(score_components):
        for v, w in d["score"].items():
            if isinstance(w, bool) or not np.isfinite(w):
                raise ValueError("score weights must be finite")
            if j not in cached_scores:
                score_weights[lookup[v], j] = w
    scores = np.zeros((len(rows), len(score_components)))
    for j, values in cached_scores.items():
        scores[:, j] = values
    score_lookup = {d["name"]: j for j, d in enumerate(score_components)}
    variant_rows = np.array(
        sorted(
            {lookup[v] for v in needed}
            | set(np.flatnonzero(np.any(score_weights != 0, axis=1)))
        ),
        dtype=np.int64,
    )
    source.prepare(np.asarray(rows), block_size, threads)
    standardizer = StandardizedBlock(native, threads)
    x, raw_columns = {}, {}
    for start in range(0, len(variant_rows), block_size):
        selected = variant_rows[start : start + block_size]
        raw = source.read(selected)
        scaled = standardizer.prepare(
            raw,
            np.arange(len(rows)),
            np.arange(len(selected)),
            scale.mean[selected],
            scale.inverse_scale[selected],
        )
        for j, index in enumerate(selected):
            name = source.variants.ids[index]
            if name in main_imputation:
                mean = main_imputation[name]
                if not np.isfinite(mean) or not 0 <= mean <= 2:
                    raise ValueError("invalid frozen genotype imputation constant")
                scaled[raw[:, j] == -127, j] = (
                    mean - scale.mean[index]
                ) * scale.inverse_scale[index]
        if len(score_components) > len(cached_scores):
            weights_tile = np.asfortranarray(score_weights[selected])
            if native is None:
                scores += scaled @ weights_tile
            else:
                product = np.empty(scores.shape, order="F")
                native.prediction_product(
                    np.asfortranarray(scaled), weights_tile, product, False, threads
                )
                scores += product
        for j, index in enumerate(selected):
            name = source.variants.ids[index]
            if name in needed:
                x[name] = scaled[:, j].copy()
                raw_columns[name] = raw[:, j].copy()
    if score_cache is not None:
        for j, key in cache_keys.items():
            if key not in score_cache:
                values = scores[:, j].copy()
                values.setflags(write=False)
                score_cache[key] = values
    fixed = [np.ones((len(rows), 1))]
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != len(rows) or not np.all(np.isfinite(cov)):
            raise ValueError("covariates must be complete and sample aligned")
        fixed.append(cov)
    fixed.extend(x[v][:, None] for v in local_variants)
    for v in dominance_variants:
        raw = raw_columns[v]
        if np.any((raw != -127) & (raw != np.rint(raw))):
            raise ValueError("heterozygote dominance adjustment requires hard calls")
        observed = raw != -127
        h = (raw == 1).astype(float)
        h[~observed] = dominance_imputation[v] if v in dominance_imputation else h[observed].mean()
        fixed.append(h[:, None])
    modifiers, weights = [], []
    if len(set(additive_annotations)) != len(additive_annotations):
        raise ValueError("duplicate additive annotations")
    for annotation in additive_annotations:
        if annotation not in annotations:
            raise ValueError("unknown additive annotation")
        names.append("additive:" + annotation)
        modifiers.append(np.ones(len(rows)))
        weights.append(annotations[annotation])
    for component in components:
        target = component.get("target")
        if target is not None:
            modifier = x[target]
            excluded = set(component.get("exclude", ())) | {target}
            model = "target_independent_background_effects"
        elif "score" in component:
            score = component["score"]
            if any(isinstance(v, bool) or not np.isfinite(v) for v in score.values()):
                raise ValueError("score weights must be finite")
            modifier = scores[:, score_lookup[component["name"]]]
            excluded = set(component.get("exclude", ()))
            # A score interacting with its own constituent is a declared
            # within-locus square, not between-locus epistasis. Require exclusion.
            overlap = [
                v
                for v in score
                if annotations[component["background"]][lookup[v]] > 0
                and v not in excluded
            ]
            if overlap:
                raise ValueError(
                    "score background overlaps its variants; explicitly exclude them"
                )
            model = "prespecified_score_direction"
        else:
            frozen = frozen_scores[component["frozen_score"]]
            modifier = np.asarray(frozen["values"], dtype=float)
            if modifier.shape != (len(rows),) or not np.all(np.isfinite(modifier)):
                raise ValueError("invalid frozen score values")
            excluded = set(component.get("exclude", ()))
            overlap = [
                v
                for v in frozen.get("nonzero_variants", frozen["variants"])
                if v in lookup
                and annotations[component["background"]][lookup[v]] > 0
                and v not in excluded
            ]
            if overlap:
                raise ValueError(
                    "frozen score background overlaps its variants; exclude self interactions"
                )
            model = "independent_frozen_score_direction"
        weight = annotations[component["background"]].copy()
        for v in excluded:
            weight[lookup[v]] = 0
        if weight.sum() <= 0 or np.var(modifier) <= 0:
            raise ValueError("empty background or constant modifier")
        names.append(component["name"])
        modifiers.append(modifier)
        weights.append(weight)
        fixed.append(modifier[:, None])
        definition = dict(component, model=model, effective_exclusions=sorted(excluded))
        if "frozen_score" in component:
            definition.update(
                frozen_score_identity=frozen["identity"],
                # Zero-weight variants do not contribute to the frozen score.
                # The model identity still authenticates its complete axis,
                # affine scale and weights. Only actual constituents belong
                # in self-overlap checks and score-main-effect declarations.
                score_variants=list(frozen.get("nonzero_variants", frozen["variants"])),
            )
        definitions.append(definition)
    if len(set(names)) != len(names):
        raise ValueError("duplicate component names")
    result = dict(
        modifiers=np.column_stack(modifiers),
        weights=np.column_stack(weights),
        fixed_effects=np.column_stack(fixed),
        component_names=tuple(names),
        definitions=dict(
            interactions=definitions,
            additive_annotations=list(additive_annotations),
            local_variants=list(local_variants),
            dominance_variants=list(dominance_variants),
            fixed_main_effects="intercept_and_all_declared_modifiers",
            main_imputation=main_imputation,
            main_imputation_rule="frozen_raw_dosage_in_study_affine_units_v1",
            targeted_variant_reads=len(variant_rows),
        ),
    )
    if dominance_imputation:
        result['definitions']['dominance_imputation'] = dominance_imputation
        result['definitions']['dominance_imputation_rule'] = 'frozen_observed_training_heterozygosity_v1'
    return result
