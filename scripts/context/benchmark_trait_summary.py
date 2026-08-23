#!/usr/bin/env python3
"""Small synthetic benchmark for the experimental contextual trait summary."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    build_context_trait_summary,
    canonical_sha256,
    rank_revealing_projector,
    write_context_trait_summary,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    genotype = rng.normal(size=(args.n, args.m))
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    raw_basis = np.column_stack([np.ones(args.n), rng.normal(size=(args.n, 3))])
    phenotype = rng.normal(size=args.n)
    residual_basis = np.column_stack([np.ones(args.n), raw_basis[:, 1] ** 2])
    annotations = np.ones((args.m, 1))
    groups = [f"group:{index % 20}" for index in range(args.m)]
    records: list[dict[str, float | int]] = []
    with tempfile.TemporaryDirectory(prefix="summit-context-trait-benchmark-") as tmp:
        temporary = Path(tmp)
        for q in range(1, 5):
            basis = raw_basis[:, :q]
            fixed = np.column_stack(
                [np.ones(args.n), basis[:, 1:], rng.normal(size=args.n)]
            )
            projector = rank_revealing_projector(fixed)
            components = ContextComponentIndex(("all",), ContextPairIndex(q))
            summary = build_context_trait_summary(
                genotype=genotype,
                basis=basis,
                phenotype=phenotype,
                projector=projector,
                annotations=annotations,
                component_index=components,
                residual_basis=residual_basis,
                residual_names=("constant", "context_squared"),
                basis_hash=array_sha256(basis),
                fixed_effect_hash=array_sha256(fixed),
                variant_hash=canonical_sha256({"variants": list(range(args.m))}),
                loo_groups=groups,
                block_size=128,
            )
            manifest, arrays = write_context_trait_summary(
                summary, temporary / f"trait_q{q}"
            )
            records.append(
                {
                    "q": q,
                    "p_genetic": len(components),
                    "n": args.n,
                    "m": args.m,
                    "decode_passes": summary.decode_passes,
                    "decoded_blocks": summary.decoded_blocks,
                    "genotype_pass_seconds": summary.phase_times_seconds[
                        "genotype_pass"
                    ],
                    "total_seconds": summary.phase_times_seconds["total"],
                    "peak_rss_bytes": summary.peak_rss_bytes,
                    "manifest_bytes": manifest.stat().st_size,
                    "array_bytes": arrays.stat().st_size,
                }
            )
    payload = {
        "kind": "summit.context.trait_summary_benchmark",
        "seed": args.seed,
        "records": records,
    }
    (args.output_dir / "02_trait_summary_benchmark.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    q_values = np.asarray([record["q"] for record in records])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    color = "#1f6f8b"
    axes[0].plot(
        q_values, [record["total_seconds"] for record in records], "o-", color=color
    )
    axes[0].set_ylabel("Wall time (s)")
    axes[1].plot(
        q_values,
        [record["peak_rss_bytes"] / 2**20 for record in records],
        "o-",
        color=color,
    )
    axes[1].set_ylabel("Process peak RSS (MiB)")
    axes[2].plot(
        q_values,
        [record["array_bytes"] / 2**20 for record in records],
        "o-",
        color=color,
    )
    axes[2].set_ylabel("Compressed summary (MiB)")
    for axis in axes:
        axis.set_xlabel("Context basis dimension Q")
        axis.set_xticks(q_values)
        axis.grid(alpha=0.25)
    fig.suptitle(f"Contextual trait-summary prototype (N={args.n}, M={args.m})")
    fig.savefig(args.output_dir / "02_trait_summary_benchmark.png", dpi=300)
    fig.savefig(args.output_dir / "02_trait_summary_benchmark.pdf")


if __name__ == "__main__":
    main()
