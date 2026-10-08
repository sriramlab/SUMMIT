"""Cohort-side bounded exact features and implicit nonself pair sketches.

Pair sketches are distinct from the generalized engine's trace probes.
All sketches are completed before projection or score construction. Their
conditional score covariance is measured, never replaced by a kernel-product
unbiasedness argument. Large pair sets are never enumerated in sketch mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from summit.context.spec import (
    array_sha256,
    canonical_sha256,
    owned_readonly_array,
    freeze_context_mapping,
)
from summit.prediction.genotype import RawBlockStream, StandardizedBlock
from summit.ldscore.generalized_gxe_variant import (
    generate_global_variant_probes,
    native_global_variant_probes,
)
from .models import target_design
from .groups import remaining_genome_weights
from .summary import _publish_bundle


@dataclass(frozen=True)
class FeatureReference:
    features: np.ndarray
    fixed_effects: np.ndarray
    component_index: np.ndarray
    metadata: dict
    additive_kernel: np.ndarray | None = None

    def __post_init__(self):
        if np.asarray(self.component_index).dtype.kind not in "iu":
            raise ValueError("feature component labels must be integer indices")
        for key in ("features", "fixed_effects", "component_index", "additive_kernel"):
            value = getattr(self, key)
            if value is not None:
                value = owned_readonly_array(
                    value, dtype=np.int64 if key == "component_index" else float
                )
                if not np.all(np.isfinite(value)):
                    raise ValueError("nonfinite feature reference")
                object.__setattr__(self, key, value)
        if (
            self.features.ndim != 2
            or self.fixed_effects.ndim != 2
            or len(self.features) != len(self.fixed_effects)
        ):
            raise ValueError("feature reference sample axes disagree")
        if self.component_index.shape != (self.features.shape[1],) or np.any(
            self.component_index < 0
        ):
            raise ValueError("feature component axis disagrees")
        names = self.metadata.get("component_names", ())
        if (
            not names
            or self.features.shape[1] == 0
            or len(set(names)) != len(names)
            or set(self.component_index) != set(range(len(names)))
            or len(self.metadata.get("feature_names", ())) != self.features.shape[1]
        ):
            raise ValueError("feature reference metadata axes disagree")
        if self.additive_kernel is not None and (
            self.additive_kernel.ndim not in (2, 3)
            or self.additive_kernel.shape[-2:]
            != (len(self.features), len(self.features))
        ):
            raise ValueError("nuisance kernel sample axes disagree")
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))


def _read_variants(study, selected, phase):
    selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    planned = (
        8 * study.n * len(selected)
        + 32 * study.n * min(study.block_size, study.m)
        + 64 * 2**20
    )
    if planned > study.memory_bytes:
        raise MemoryError("selected genotype design exceeds memory budget")
    x = np.empty((study.n, len(selected)), order="F")
    stream = RawBlockStream(
        study.source,
        study.rows,
        selected,
        block_size=study.block_size,
        threads=study.threads,
    )
    for start, variants, raw in stream.blocks(phase):
        x[:, start : start + len(variants)] = study.standardize(raw, variants)
    return x, selected, stream.ledger


def _resume_full_stream(study):
    """Explicit ownership handoff after a completed targeted-design read.

    RawBlockStream authenticates source generations. Creating a fresh owner
    retains that check; mutating its expected generation would bypass it.
    Only completed traversals are handed off here.
    """
    ledger = study.stream.ledger
    study.stream = RawBlockStream(
        study.source,
        study.rows,
        np.arange(study.m),
        block_size=study.block_size,
        threads=study.threads,
    )
    study.stream.ledger = ledger


def _group_weights(groups, annotations):
    result = []
    for g in groups:
        if (
            g.get("mode") not in ("within", "cross", "remainder")
            or g.get("left") not in annotations
        ):
            raise ValueError("invalid group definition")
        if set(g) - {"name", "mode", "left", "right"} or not g.get("name"):
            raise ValueError("invalid group fields")
        a = annotations[g["left"]]
        if g["mode"] == "cross":
            if g.get("right") not in annotations:
                raise ValueError("cross group needs a right annotation")
            b = annotations[g["right"]]
        else:
            if "right" in g:
                raise ValueError("only cross groups accept right")
            b = a if g["mode"] == "within" else remaining_genome_weights(a)
        mass = float(a.sum() * b.sum() - a @ b)
        if mass <= 0:
            raise ValueError("group contains no nonself pairs")
        result.append((a, b, mass))
    return result


def _probe(study, indices, dimensions, seed, bank, side):
    if not hasattr(study, "_pair_probe_indices"):
        axis = study.source.variants
        # Stable variant labels, independent of input column order or tiling.
        labels = [
            int(
                canonical_sha256(
                    dict(
                        id=v,
                        chromosome=c,
                        position=int(p),
                        alleles=sorted((a, b)),
                        genome_build=axis.genome_build,
                    )
                )[:15],
                16,
            )
            for v, c, p, a, b in zip(
                axis.ids, axis.chromosome, axis.position, axis.counted, axis.other
            )
        ]
        if len(set(labels)) != len(labels):
            raise ValueError("pair-sketch variant counter collision")
        study._pair_probe_indices = np.asarray(labels, dtype=np.int64)
    indices = study._pair_probe_indices[np.asarray(indices, dtype=np.int64)]
    draw_seed = int(
        canonical_sha256(dict(seed=int(seed), bank=int(bank), side=side))[:16], 16
    )
    if study.native is None:
        return generate_global_variant_probes(
            indices, np.arange(dimensions), root_seed=draw_seed
        )
    return native_global_variant_probes(
        indices,
        np.arange(dimensions),
        root_seed=draw_seed,
        threads=study.threads,
        native_module=study.native,
    )


def feature_compatibility_id(
    study,
    job,
    annotations,
    *,
    sketch_dimensions=None,
    seed=1,
    bank=0,
    main_effects="declared",
    dominance="declared",
    additive_null=False,
    strata=None,
):
    scientific = {k: v for k, v in job.items() if k not in ("id", "inference")}
    stratum_fields = {}
    declared_strata = study.metadata["definitions"].get("fixed_effect_strata")
    if strata is not None:
        from .trans import stratum_definition

        record = stratum_definition(strata, n_samples=study.n)
        if declared_strata is not None and declared_strata != record:
            raise ValueError("feature strata differ from the declared nuisance design")
        stratum_fields["complete_nuisance_strata"] = record
    elif declared_strata is not None:
        raise ValueError(
            "stratified feature preparation requires sample-aligned labels"
        )
    # Group annotations need not be additive annotations. Resolve them before
    # looking up a cached reference, including the complement for remainder.
    resolved_groups = []
    for a, b, mass in _group_weights(job.get("groups", ()), annotations):
        for weight in (a, b):
            if (
                np.shape(weight) != (study.m,)
                or not np.all(np.isfinite(weight))
                or np.any(weight < 0)
            ):
                raise ValueError("invalid resolved group annotation weights")
        resolved_groups.append(
            dict(
                left=array_sha256(np.asarray(a, dtype=float)),
                right=array_sha256(np.asarray(b, dtype=float)),
                mass=mass,
            )
        )
    return canonical_sha256(
        dict(
            study=study.metadata,
            job=scientific,
            resolved_groups=resolved_groups,
            sketch_dimensions=sketch_dimensions,
            seed=seed,
            bank=bank,
            main_effects=main_effects,
            dominance=dominance,
            additive_null=additive_null,
            method="cohort_feature_reference_v3_common_imputation",
            draw_algorithm="stable_variant_independent_group_products_nonself_v1",
            **stratum_fields,
        )
    )


def _source_sketch_identity(study, dimensions, seed, bank):
    return canonical_sha256(
        dict(
            source=study.source.identity,
            samples=study.metadata["sample_hash"],
            scale=study.scale.identity,
            imputation=study.metadata["definitions"].get("main_imputation", {}),
            dimensions=dimensions,
            seed=seed,
            bank=bank,
            method="unprojected_target_source_sketch_v1",
        )
    )


def prepare_shared_target_sources(study, weights, *, dimensions, seed=1, bank=0):
    """One decoding pass for independent targets with separate later projections.

    Store bounded N-by-R sources, never target-by-SNP scores. The source key
    deliberately excludes target-specific projection: raw multiplication is
    shared, while multiplication by each modifier and projection happen later.
    """
    if type(dimensions) is not int or not 4 <= dimensions <= 4096:
        raise ValueError("invalid shared feature sketch dimension")
    unique = {
        array_sha256(np.asarray(w, dtype=float)): np.asarray(w, dtype=float)
        for w in weights
    }
    if not unique or any(
        w.shape != (study.m,)
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
        or w.sum() <= 0
        for w in unique.values()
    ):
        raise ValueError("invalid shared background weights")
    if (
        8 * study.n * dimensions * (len(unique) + 4) + 64 * 2**20
        > study.memory_bytes // 2
    ):
        raise MemoryError(
            "target source batch exceeds half the preparation memory budget; split the manifest"
        )
    values = {key: np.zeros((study.n, dimensions), order="F") for key in unique}
    axis = study.source.variants
    orientation = np.array(
        [1.0 if a <= b else -1.0 for a, b in zip(axis.counted, axis.other)]
    )
    _resume_full_stream(study)
    study.nn.begin_execution()
    study.tn.begin_execution()
    for lo, hi, x in study._blocks("shared_target_feature_sources"):
        z = (
            _probe(study, np.arange(lo, hi), dimensions, seed, bank, "left")
            * orientation[lo:hi, None]
        )
        for key, w in unique.items():
            values[key] += study._nn(x, np.sqrt(w[lo:hi, None]) * z)
    study.nn.finish_execution()
    for value in values.values():
        value.setflags(write=False)
    return dict(
        identity=_source_sketch_identity(study, dimensions, seed, bank),
        sources=values,
        source_passes=1,
        decoded_bytes=study.stream.ledger.source_decoded_bytes,
    )


def prepare_feature_reference(
    study,
    job,
    annotations,
    *,
    sketch_dimensions=None,
    seed=1,
    bank=0,
    main_effects="declared",
    dominance="declared",
    additive_null=False,
    shared_sources=None,
    strata=None,
):
    """Prepare a score experiment without an N-by-N interaction kernel.

    Exact pair panels are limited to 512 columns. A sketch uses R directions
    regardless of the number of pairs. Covariance is computed from these same
    directions during score preparation; its conditional null is therefore
    well defined even when the original group kernel approximation is noisy.
    """
    if sketch_dimensions is not None and (
        type(sketch_dimensions) is not int or not 4 <= sketch_dimensions <= 4096
    ):
        raise ValueError("pair/feature sketch dimensions must be 4..4096")
    if additive_null and study.n > 1024:
        raise ValueError("spectral REML null is bounded at N <= 1024")
    if type(seed) is not int or seed < 0 or type(bank) is not int or bank < 0:
        raise ValueError("sketch seed and bank must be nonnegative integers")
    if main_effects not in (
        "declared",
        "all_genotypes",
        "tested_variants",
    ) or dominance not in ("declared", "tested_variants"):
        raise ValueError("invalid main-effect or dominance policy")
    shared_bytes = (
        0
        if shared_sources is None
        else sum(a.nbytes for a in shared_sources["sources"].values())
    )
    available_bytes = study.memory_bytes - shared_bytes
    decoder_workspace = 32 * study.n * min(study.block_size, study.m)
    lookup = {v: i for i, v in enumerate(study.source.variants.ids)}
    groups = job.get("groups", ())
    group_weights = _group_weights(groups, annotations)
    if sketch_dimensions is not None:
        order = np.argsort(study.source.variants.ids)
        group_weights = [
            (a, b, mass)
            if array_sha256(a[order]) <= array_sha256(b[order])
            else (b, a, mass)
            for a, b, mass in group_weights
        ]
    if "pairs" in job:
        pairs = []
        for pair in job["pairs"]:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(v not in lookup for v in pair)
            ):
                raise ValueError("invalid supplied pair")
            pairs.append(tuple(sorted((lookup[pair[0]], lookup[pair[1]]))))
        if not pairs or len(set(pairs)) != len(pairs) or len(pairs) > 512:
            raise ValueError(
                "supplied pairs must be unique; maximum exact panel is 512"
            )
        if sketch_dimensions is not None:
            raise ValueError(
                "supplied pairs use their signed exact features, not a group sketch"
            )
        tested = set(i for pair in pairs for i in pair)
        component_names = ["supplied_pairs"]
    elif groups:
        tested = set(
            np.flatnonzero(
                np.any(
                    np.column_stack([a + b for a, b, _ in group_weights]) > 0, axis=1
                )
            )
        )
        component_names = [g["name"] for g in groups]
    else:
        interactions = study.metadata["definitions"]["interactions"]
        component_names = [d["name"] for d in interactions]
        # Main effects of background loci matter too in the fixed-mean model.
        start = study.c - len(interactions)
        tested = set(np.flatnonzero(np.any(study.weights[:, start:] > 0, axis=1)))
        for d in interactions:
            tested.update(
                lookup[v]
                for v in (
                    [d["target"]]
                    if "target" in d
                    else d.get("score", d.get("score_variants", ()))
                )
            )
    if not component_names or len(set(component_names)) != len(component_names):
        raise ValueError("component names must be nonempty and unique")
    extra_main = (
        set(range(study.m))
        if main_effects == "all_genotypes"
        else tested
        if main_effects == "tested_variants"
        else set()
    )
    if "pairs" in job:
        extra_main |= tested
    if additive_null and main_effects == "declared":
        required = {g["left"] for g in groups} | {
            g["right"] for g in groups if "right" in g
        }
        required |= {c["background"] for c in job.get("components", ())}
        if required - set(job["additive_annotations"]):
            raise ValueError(
                "fitted covariance requires additive nuisance annotations for every interaction background/group"
            )
    extra_dominance = tested if dominance == "tested_variants" else set()
    if extra_dominance and not study.source.hard_calls:
        raise ValueError("dominance policy requires discrete hard calls")
    if len(extra_main | extra_dominance) > 4096:
        raise ValueError(
            "large fixed main-effect design is unsupported; use a justified polygenic null"
        )
    if (
        8 * study.n * (4 * len(extra_main | extra_dominance) + 3 * study.fixed.shape[1])
        + decoder_workspace
        + 64 * 2**20
        > available_bytes
    ):
        raise MemoryError("fixed main-effect design exceeds memory budget")
    extra = target_design(
        study.source,
        study.rows,
        study.scale,
        components=[],
        annotations=annotations,
        additive_annotations=["all"],
        allow_additive_only=True,
        main_imputation=study.metadata["definitions"].get("main_imputation", {}),
        local_variants=[study.source.variants.ids[i] for i in sorted(extra_main)],
        dominance_variants=[
            study.source.variants.ids[i] for i in sorted(extra_dominance)
        ],
        block_size=study.block_size,
        threads=study.threads,
        native=study.native,
        memory_bytes=available_bytes,
    )
    fixed = np.column_stack([study.fixed, extra["fixed_effects"]])
    if strata is not None:
        from .trans import stratified_fixed_effects

        fixed, _ = stratified_fixed_effects(
            fixed,
            strata,
            memory_bytes=available_bytes
            - decoder_workspace
            - study.fixed.nbytes
            - study.u.nbytes
            - extra["fixed_effects"].nbytes,
        )
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis

    residual_rank = study.n - thin_rank_revealing_fixed_effect_basis(fixed).shape[1]
    if residual_rank < 3:
        raise ValueError("main-effect adjustment leaves insufficient residual rank")
    components = []
    feature_names = []
    indices = []
    passes = {
        "extra_main_and_dominance_variant_reads": len(extra_main | extra_dominance)
    }
    if sketch_dimensions is not None:
        r = sketch_dimensions
        planned = (
            8
            * (
                study.n * r * (7 * len(component_names) + 4)
                + study.n * fixed.shape[1] * 3
                + r * r * len(component_names) ** 2
            )
            + decoder_workspace
            + 64 * 2**20
        )
        if planned > available_bytes:
            raise MemoryError("implicit feature sketches exceed memory budget")
        study.nn.begin_execution()
        study.tn.begin_execution()
        _resume_full_stream(study)
        if groups:
            left = [np.zeros((study.n, r), order="F") for _ in groups]
            right = [np.zeros((study.n, r), order="F") for _ in groups]
            self_terms = [np.zeros((study.n, r), order="F") for _ in groups]
        else:
            if shared_sources is not None:
                if shared_sources["identity"] != _source_sketch_identity(
                    study, r, seed, bank
                ):
                    raise ValueError(
                        "shared target source mask, scale or draw mismatch"
                    )
                source = [
                    shared_sources["sources"][array_sha256(study.weights[:, start + j])]
                    for j in range(len(component_names))
                ]
            else:
                source = [np.zeros((study.n, r), order="F") for _ in component_names]
        # Canonical allele orientation makes sketch realizations invariant to
        # consistent counted-allele swaps, not merely equal in distribution.
        axis = study.source.variants
        orientation = np.array(
            [1.0 if a <= b else -1.0 for a, b in zip(axis.counted, axis.other)]
        )
        blocks = (
            study._blocks("interaction_feature_sketch")
            if groups or shared_sources is None
            else ()
        )
        for lo, hi, x in blocks:
            z = (
                _probe(study, np.arange(lo, hi), r, seed, bank, "left")
                * orientation[lo:hi, None]
            )
            if groups:
                zz = (
                    _probe(study, np.arange(lo, hi), r, seed, bank, "right")
                    * orientation[lo:hi, None]
                )
                for j, (a, b, mass) in enumerate(group_weights):
                    left[j] += study._nn(x, np.sqrt(a[lo:hi, None]) * z)
                    right[j] += study._nn(x, np.sqrt(b[lo:hi, None]) * zz)
                    self_terms[j] += study._nn(
                        x * x, np.sqrt((a * b)[lo:hi, None]) * (z * zz)
                    )
            else:
                for j in range(len(component_names)):
                    source[j] += study._nn(
                        x, np.sqrt(study.weights[lo:hi, start + j, None]) * z
                    )
        for j, name in enumerate(component_names):
            if groups:
                features = (left[j] * right[j] - self_terms[j]) / np.sqrt(
                    r * group_weights[j][2]
                )
            else:
                features = (
                    study.modifiers[:, start + j, None]
                    * source[j]
                    / np.sqrt(r * study.masses[start + j])
                )
            components.append(features)
            feature_names.extend(f"{name}:direction:{i}" for i in range(r))
            indices.extend([j] * r)
        study.nn.finish_execution()
        passes["interaction_feature_sketch"] = int(
            bool(groups) or shared_sources is None
        )
        if shared_sources is not None:
            passes["shared_target_sources_reused"] = True
    else:
        definitions = []
        if "pairs" in job:
            definitions = [[(a, b, 1.0) for a, b in pairs]]
        elif groups:
            for group, (a, b, mass) in zip(groups, group_weights):
                members = np.flatnonzero(a + b)
                pair_count = (len(members) * (len(members) - 1)) // 2
                if pair_count > 20000:
                    raise ValueError("large groups require an implicit pair sketch")
                pairs = [
                    (int(i), int(j), float((a[i] * b[j] + a[j] * b[i]) / mass))
                    for ii, i in enumerate(members)
                    for j in members[ii + 1 :]
                    if a[i] * b[j] + a[j] * b[i] > 0
                ]
                definitions.append(pairs)
        if definitions:
            count = sum(len(p) for p in definitions)
            if not count or count > 512:
                raise ValueError(
                    "exact group panels require 1..512 pairs; use pair sketches for larger groups"
                )
            if (
                8 * study.n * (3 * count + len(tested) + 3 * fixed.shape[1])
                + decoder_workspace
                + 64 * 2**20
                > available_bytes
            ):
                raise MemoryError("exact feature design exceeds memory budget")
            x, selected, ledger = _read_variants(
                study, tested, "interaction_exact_selected"
            )
            position = {int(v): i for i, v in enumerate(selected)}
            for c, pairs in enumerate(definitions):
                components.append(
                    np.column_stack(
                        [
                            x[:, position[i]] * x[:, position[j]] * np.sqrt(weight)
                            for i, j, weight in pairs
                        ]
                    )
                )
                feature_names.extend(
                    f"{component_names[c]}:{study.source.variants.ids[i]}:{study.source.variants.ids[j]}"
                    for i, j, _ in pairs
                )
                indices.extend([c] * len(pairs))
            passes.update(ledger.traversals)
        else:
            selected = sorted(
                set(np.flatnonzero(np.any(study.weights[:, start:] > 0, axis=1)))
            )
            count = int(np.count_nonzero(study.weights[:, start:]))
            if count > 512:
                raise ValueError(
                    "exact target features exceed 512 columns; specify feature sketch dimensions"
                )
            if (
                8 * study.n * (3 * count + len(selected) + 3 * fixed.shape[1])
                + 8 * count * count
                + decoder_workspace
                + 64 * 2**20
                > available_bytes
            ):
                raise MemoryError("exact target score features exceed memory budget")
            x, selected, ledger = _read_variants(
                study, selected, "interaction_exact_selected"
            )
            for j, name in enumerate(component_names):
                w = study.weights[selected, start + j]
                keep = w > 0
                components.append(
                    study.modifiers[:, start + j, None]
                    * x[:, keep]
                    * np.sqrt(w[keep] / study.masses[start + j])
                )
                feature_names.extend(
                    f"{name}:{study.source.variants.ids[i]}" for i in selected[keep]
                )
                indices.extend([j] * np.sum(keep))
            passes.update(ledger.traversals)
    g = None
    if additive_null:
        count = len(job["additive_annotations"])
        if (
            8
            * (
                (2 * count + 3) * study.n**2
                + study.n * (3 * len(indices) + 3 * fixed.shape[1])
            )
            + decoder_workspace
            + 64 * 2**20
            > available_bytes
        ):
            raise MemoryError("spectral nuisance reference exceeds memory budget")
        g = np.zeros((count, study.n, study.n))
        study.nn.begin_execution()
        study.tn.begin_execution()
        _resume_full_stream(study)
        for lo, hi, x in study._blocks("spectral_null_additive"):
            for c in range(count):
                g[c] += study._nn(x, (x * study.weights[lo:hi, c]).T) / study.masses[c]
        study.nn.finish_execution()
        if count == 1:
            g = g[0]
        passes["spectral_null_additive"] = 1
    metadata = dict(
        study.metadata,
        cohort_sample_tokens=[
            canonical_sha256(list(study.source.samples[int(i)])) for i in study.rows
        ],
        feature_names=feature_names,
        component_names=component_names,
        job=job,
        main_effects=main_effects,
        dominance=dominance,
        sketch_dimensions=sketch_dimensions,
        seed=seed,
        bank=bank,
        additive_null=additive_null,
        genotype_passes=passes,
        additive_kernel_names=["additive:" + a for a in job["additive_annotations"]],
        reference_kind="cohort_feature_reference_v3_common_imputation",
        residual_rank=residual_rank,
        projection_order="products_then_joint_projection",
        kernel_target="exact"
        if sketch_dimensions is None
        else "conditional_random_feature_kernel",
        sketch_algorithm=None
        if sketch_dimensions is None
        else "stable_variant_independent_group_products_nonself_v1",
    )
    if strata is not None:
        from .trans import stratum_definition

        metadata["definitions"] = dict(
            metadata["definitions"],
            fixed_effect_strata=stratum_definition(strata, n_samples=study.n),
        )
    metadata["compatibility_id"] = feature_compatibility_id(
        study,
        job,
        annotations,
        sketch_dimensions=sketch_dimensions,
        seed=seed,
        bank=bank,
        main_effects=main_effects,
        dominance=dominance,
        additive_null=additive_null,
        strata=strata,
    )
    if "pairs" in job:
        involved = sorted({v for pair in job["pairs"] for v in pair})
        lookup = {v: i for i, v in enumerate(study.source.variants.ids)}
        metadata.update(
            pair_ids=[sorted(pair) for pair in job["pairs"]],
            fixed_main_effect_variants=involved,
            genome_build=study.source.variants.genome_build,
            inverse_scales={
                v: float(study.scale.inverse_scale[lookup[v]]) for v in involved
            },
        )
    return FeatureReference(
        np.column_stack(components), fixed, np.array(indices), metadata, g
    )


def write_feature_reference(reference, path):
    arrays = {
        k: getattr(reference, k)
        for k in ("features", "fixed_effects", "component_index")
    }
    if reference.additive_kernel is not None:
        arrays["additive_kernel"] = reference.additive_kernel
    manifest = dict(
        kind="summit.epistasis.cohort_feature_reference",
        schema_version=2,
        metadata=reference.metadata,
        digests={k: array_sha256(v) for k, v in arrays.items()},
        data_class="cohort_side_genotype_derived_reference",
    )
    return _publish_bundle(path, manifest, arrays)


def load_feature_reference(path, *, compatibility_id):
    with np.load(path, allow_pickle=False) as a:
        m = json.loads(str(a["manifest"]))
        if (
            m["kind"] != "summit.epistasis.cohort_feature_reference"
            or m["schema_version"] != 2
        ):
            raise ValueError(
                "feature reference compatibility cannot be established for this schema; prepare a new version-2 reference at a new output path"
            )
        arrays = {k: a[k] for k in a.files if k != "manifest"}
        if (
            m["digests"] != {k: array_sha256(v) for k, v in arrays.items()}
            or m["metadata"]["compatibility_id"] != compatibility_id
        ):
            raise ValueError("feature reference digest or compatibility mismatch")
        return FeatureReference(**arrays, metadata=m["metadata"])
