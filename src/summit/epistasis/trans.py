"""Authenticate supplied trans hypotheses; no target selection or LD guarantee."""
import numpy as np
from summit.context.spec import canonical_sha256


def select_trans_background(source, target, *, chromosome=None, variants=None):
    """Resolve one prespecified chromosome or variant file on the source axis."""
    if (chromosome is None) == (variants is None):
        raise ValueError("supply exactly one background chromosome or interaction variant file")
    if variants is not None:
        from summit.prediction.cli import _variants

        selected = _variants(source, variants)
    else:
        names = np.array([str(c).removeprefix("chr") for c in source.variants.chromosome])
        selected = np.flatnonzero(names == str(chromosome).removeprefix("chr"))
    record = trans_membership(source.variants, target,
                              [source.variants.ids[i] for i in selected])
    return np.sort(selected), record


def stratum_definition(labels, *, n_samples):
    """Validate and identify the supplied sample-aligned nuisance strata."""
    labels = np.asarray(labels, str)
    if labels.shape != (n_samples,) or any(not s or s in ("nan", "NA") for s in labels):
        raise ValueError("stratum labels must be complete and sample aligned")
    levels = sorted(set(labels))
    return dict(
        levels=levels,
        counts=[int(np.sum(labels == s)) for s in levels],
        identity=canonical_sha256(labels.tolist()),
        preparation="complete_feature_nuisance_design_by_stratum_v1",
        model="separate declared finite nuisance mean by supplied stratum; common interaction coefficient",
    )


def stratified_fixed_effects(fixed, labels, *, memory_bytes):
    """Fit the complete finite nuisance mean separately in every stratum.

    Apply after automatic main/dominance columns have been assembled, so all
    representations of the same interaction use the same adjustment space.
    """
    c = np.asarray(fixed, float)
    labels = np.asarray(labels, str)
    record = stratum_definition(labels, n_samples=len(c))
    levels = record["levels"]
    # column_stack briefly owns both the per-stratum inputs and final output.
    planned = c.nbytes * (2 * len(levels) + 1)
    if planned + 64 * 2**20 > memory_bytes:
        raise MemoryError("stratum-specific nuisance columns exceed remaining memory")
    result = np.column_stack([c * (labels == s)[:, None] for s in levels])
    return result, record


def trans_membership(axis, target, members):
    lookup = {v: i for i, v in enumerate(axis.ids)}
    members = sorted(set(members))
    if target not in lookup or not members or set(members) - lookup.keys():
        raise ValueError(
            "trans hypothesis needs a known target and nonempty known members"
        )

    def chromosome(v):
        value = str(axis.chromosome[lookup[v]]).removeprefix("chr")
        if value not in {str(i) for i in range(1, 23)}:
            raise ValueError(
                "trans qualification requires declared autosomal chromosomes 1..22"
            )
        return value

    target_chr = chromosome(target)
    if any(chromosome(v) == target_chr for v in members):
        raise ValueError("trans interaction includes a marker on the target chromosome")
    return dict(
        target=target,
        target_chromosome=target_chr,
        member_count=len(members),
        member_identity=canonical_sha256(members),
        chromosomes=sorted({chromosome(v) for v in members}, key=int),
        variant_axis_identity=axis.identity,
        interpretation="chromosome membership only; does not establish independence or absence of phantom epistasis",
    )


def validate_trans_job(job, axis, annotations, frozen_scores):
    """Validate every actual component against the supplied common target."""
    target = job["trans_target"]
    records = []
    if job.get("inference", {}).get("method") not in ("robust_mean", "linear_exact"):
        raise ValueError("trans contract requires a declared finite-feature mean test")
    if "groups" in job:
        raise ValueError("trans_target is for target-by-score or supplied target pairs")
    if "pairs" in job:
        for pair in job["pairs"]:
            if len(pair) != 2 or pair.count(target) != 1:
                raise ValueError(
                    "all trans pairs must contain the declared target once"
                )
            records.append(
                trans_membership(axis, target, [v for v in pair if v != target])
            )
    else:
        for component in job["components"]:
            if "target" in component:
                if component["target"] != target:
                    raise ValueError("trans component target differs")
                members = [
                    v
                    for v, w in zip(axis.ids, annotations[component["background"]])
                    if w > 0 and v not in set(component.get("exclude", ())) | {target}
                ]
            else:
                background = [
                    v
                    for v, w in zip(axis.ids, annotations[component["background"]])
                    if w > 0
                ]
                if background != [target] or target in component.get("exclude", ()):
                    raise ValueError(
                        "trans score background must be the singleton target"
                    )
                if "score" in component:
                    members = [v for v, w in component["score"].items() if w != 0]
                else:
                    members = frozen_scores[component["frozen_score"]][
                        "nonzero_variants"
                    ]
            records.append(trans_membership(axis, target, members))
    return dict(
        components=records,
        scientific_target="supplied trans mean association",
        additive_adjustment="independent of interaction chromosome exclusions",
    )
