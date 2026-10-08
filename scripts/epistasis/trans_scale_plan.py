"""Use the existing planner for larger trans shapes; no fit or time prediction.

Only target completeness is read. All other numerical arrays are explicitly
dimension placeholders, never estimated scales, phenotype data or fitted models.
The full local candidate count conservatively bounds fixed-effect storage.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.spec import GenotypeScale, TraitTraining
from summit.prediction.annotations import AnnotationDesign, AnnotationPrior
from summit.prediction.batch import plan_prediction
from summit.prediction._validation import digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--memory-gib", type=float, default=32)
    a = p.parse_args()
    plans = []
    with FileGenotypeSource(a.genotypes, genome_build="GRCh37") as source:
        j = source.variants.ids.index("12:66358347")
        source.prepare(np.arange(len(source.samples)), 1, 1)
        complete = np.flatnonzero(source.read(np.array([j]))[:, 0] != -127)
        variants = np.delete(np.arange(len(source.variants.ids)), j)
        axis = source.variants.subset(variants)
        local = sum(
            ch == "12" and abs(pos - 66358347) <= 100000
            for ch, pos in zip(source.variants.chromosome, source.variants.position)
        )
        fixed_columns = 2 + 2 * local
        annotation = AnnotationDesign(
            np.column_stack(
                [np.ones(len(variants)), np.array(axis.chromosome) != "12"]
            ),
            ("additive_background", "interaction_background"),
            axis.identity,
        )
        prior = AnnotationPrior(
            annotation, np.array([[[0.5, 0], [0, 0]], [[0, 0], [0, 0.05]]])
        )
        for total in (65536, 131072, len(complete)):
            if total > len(complete):
                continue
            n = total // 2
            rows = complete[:n]
            scale = GenotypeScale(
                np.ones(len(variants)),
                np.ones(len(variants)),
                axis.identity,
                digest([source.samples[int(i)] for i in rows]),
                dict(
                    source=source.identity,
                    purpose="dimension placeholders; never used for fitting",
                ),
            )
            trait = TraitTraining(
                "dimension_only",
                rows,
                variants,
                np.zeros(n),
                np.ones((n, 2)),
                np.zeros((n, fixed_columns)),
                scale,
                (
                    prior.candidate(
                        "prespecified",
                        np.ones(n),
                        dict(purpose="dimension-only planning"),
                    ),
                ),
                dict(names=["baseline", "target"]),
                dict(names=[f"c{i}" for i in range(fixed_columns)]),
                dict(units="not a phenotype; memory planning only"),
            )
            # Same caller-owned design and full-scale subtraction as train_direction.
            caller_owned = 8 * (n * (fixed_columns + 2) + 4 * len(source.variants.ids))
            for storage in ("packed", "stream"):
                plan = plan_prediction(
                    [trait],
                    source,
                    storage=storage,
                    block_size=128,
                    rhs_columns=8,
                    threads=2,
                    memory_bytes=int(a.memory_gib * 2**30) - caller_owned,
                ).to_dict()
                # This identity describes placeholders, so never offer it for resume.
                plan.pop("fit_identity")
                plans.append(
                    dict(
                        total_n=total,
                        training_n=n,
                        confirmation_n=total - n,
                        fixed_column_upper_bound=fixed_columns,
                        caller_owned_bytes=caller_owned,
                        peak_with_caller_bytes=caller_owned
                        + plan["estimated_peak_bytes"],
                        plan=plan,
                    )
                )
        report = dict(
            source_samples=len(source.samples),
            target_complete_samples=len(complete),
            source_markers=len(source.variants.ids),
            source_identity=source.identity,
            target_variant_reads=1,
            whole_marker_traversals=0,
            plans=plans,
            limitation="dimension-only admission with placeholder numerical arrays; not throughput, convergence, correctness of means, or calibration evidence; full source scaling and actual fit planning must run before production training",
        )
    with a.out.open("x") as h:
        json.dump(report, h, indent=2)
    for plan in plans:
        print(
            plan["total_n"],
            plan["plan"]["storage"],
            plan["peak_with_caller_bytes"] / 2**30,
        )


if __name__ == "__main__":
    main()
