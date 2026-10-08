"""Aligned dense local main effects from an additional cohort genotype source."""
import numpy as np
from summit.prediction._validation import closed
from summit.prediction.genotype import (
    source_from_spec,
    estimate_scale,
    StandardizedBlock,
    native_module,
)
from summit.prediction.cli import _variants


def prepare_fixed_genotypes(
    definitions, root, samples, *, threads, block_size, memory_bytes
):
    """Two selected-marker passes; no interaction exclusion or target selection.

    All requested additive terms are retained, with optional hard-call
    heterozygote terms. Rank and leverage are checked by the chosen estimator.
    """
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("fixed_genotypes must be a nonempty list")
    names = set()
    columns = []
    reports = []
    resident = 0
    for entry in definitions:
        closed(
            entry,
            ("name", "genotypes", "variants"),
            ("dominance",),
            name="fixed genotype source",
        )
        if (
            not isinstance(entry["name"], str)
            or not entry["name"]
            or entry["name"] in names
        ):
            raise ValueError("fixed genotype sources require distinct names")
        names.add(entry["name"])
        dominance = entry.get("dominance", False)
        if type(dominance) is not bool:
            raise ValueError("dominance must be boolean")
        with source_from_spec(entry["genotypes"], root) as source:
            lookup = {v: i for i, v in enumerate(source.samples)}
            if any(v not in lookup for v in samples):
                raise ValueError(
                    "fixed genotype source does not contain every selected cohort sample"
                )
            requested = np.array([lookup[v] for v in samples])
            order = np.argsort(requested)
            undo = np.argsort(order)
            rows = requested[order]
            selected = _variants(source, root / entry["variants"])
            if len(selected) > 4096:
                raise ValueError(
                    "dense local mean adjustment is limited to 4096 declared variants"
                )
            if dominance and not source.hard_calls:
                raise ValueError(
                    "dominance needs hard calls, not dosage-only genotypes"
                )
            planned = (
                resident
                + 8 * len(rows) * len(selected) * (5 + 4 * dominance)
                + 32 * len(rows) * min(block_size, len(selected))
                + 64 * 2**20
            )
            if planned > memory_bytes:
                raise MemoryError("dense local main effects exceed memory budget")
            scale = estimate_scale(
                source,
                rows,
                selected,
                threads=threads,
                block_size=block_size,
                memory_bytes=memory_bytes - resident,
            )
            source.prepare(rows, block_size, threads)
            standardizer = StandardizedBlock(native_module(), threads)
            additive = np.empty((len(rows), len(selected)))
            heterozygote = np.empty_like(additive) if dominance else None
            for start in range(0, len(selected), block_size):
                stop = min(start + block_size, len(selected))
                raw = source.read(selected[start:stop])
                additive[:, start:stop] = standardizer.prepare(
                    raw,
                    np.arange(len(rows)),
                    np.arange(stop - start),
                    scale.mean[start:stop],
                    scale.inverse_scale[start:stop],
                )[undo]
                if dominance:
                    observed = raw != -127
                    h = (raw == 1).astype(float)
                    h = np.where(
                        observed, h, (h.sum(axis=0) / observed.sum(axis=0))[None, :]
                    )
                    heterozygote[:, start:stop] = h[undo]
            columns.append(additive)
            resident += additive.nbytes
            if dominance:
                columns.append(heterozygote)
                resident += heterozygote.nbytes
            reports.append(
                dict(
                    name=entry["name"],
                    source_identity=source.identity,
                    scale_identity=scale.identity,
                    variants=source.variants.subset(selected).to_dict(),
                    dominance=dominance,
                    selected_marker_passes=2,
                    sample_alignment="FID/IID; selected cohort order restored after sorted native reads",
                )
            )
    return np.column_stack(columns), reports
