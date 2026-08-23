"""Planning and inspection CLI for generalized per-variant GxE LD scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from summit.context.spec import canonical_json
from summit.ldscore.generalized_gxe_reference_v1 import (
    load_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_variant import (
    GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
    GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
    GeneralizedGxEPlanInputs,
    plan_generalized_gxe_variant_work,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summit-generalized-gxe-variant-ldscore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Plan or inspect the generalized per-variant SUMMIT-GxE LD-score "
            "reference. This is the variant-probe, exactly-two-pass estimator; "
            "it is not the sample-probe contextual action estimator."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan = subcommands.add_parser(
        "plan", help="dry-run the exact two-pass work and memory plan"
    )
    for option, destination in (
        ("--samples", "num_samples"),
        ("--variants", "num_variants"),
        ("--basis", "num_basis"),
        ("--annotations", "num_annotations"),
        ("--probes", "num_probes"),
        ("--jackknife-blocks", "num_jackknife_blocks"),
        ("--memory-bytes", "memory_limit_bytes"),
    ):
        plan.add_argument(option, dest=destination, type=int, required=True)
    plan.add_argument("--genotype-format", choices=("bed", "pgen"), required=True)
    plan.add_argument("--threads", type=int, default=1)
    plan.add_argument("--variant-block-width", type=int, default=4096)
    plan.add_argument("--rhs-tile-columns", type=int)
    plan.add_argument(
        "--rhs-policy", choices=("auto", "precompute", "tiled"), default="auto"
    )
    plan.add_argument("--omit-per-variant-panel", action="store_true")

    inspect = subcommands.add_parser(
        "inspect", help="validate and summarize a closed V1 reference artifact"
    )
    inspect.add_argument("artifact", type=Path)
    return parser


def _plan(args: argparse.Namespace) -> dict:
    work = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=args.num_samples,
            num_variants=args.num_variants,
            num_basis=args.num_basis,
            num_annotations=args.num_annotations,
            num_probes=args.num_probes,
            num_jackknife_blocks=args.num_jackknife_blocks,
            memory_limit_bytes=args.memory_limit_bytes,
            genotype_format=args.genotype_format,
            threads=args.threads,
            preferred_variant_block_width=args.variant_block_width,
            preferred_rhs_tile_columns=args.rhs_tile_columns,
            rhs_policy=args.rhs_policy,
            write_directional_panel=not args.omit_per_variant_panel,
        )
    )
    return {
        "command": "generalized-gxe-variant-ldscore-plan",
        "dry_run": True,
        "scientific_contract": GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
        "probe_axis": "variant",
        "jackknife_method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
        "planned_reference_genotype_passes": 2,
        "work_plan": work.to_dict(),
    }


def _inspect(path: Path) -> dict:
    artifact = load_generalized_gxe_variant_reference_v1(path)
    axes = artifact.manifest["axes"]
    return {
        "command": "generalized-gxe-variant-ldscore-inspect",
        "valid": True,
        "kind": artifact.manifest["kind"],
        "scientific_contract": artifact.manifest["scientific_contract"],
        "probe_axis": artifact.manifest["probe_axis"],
        "jackknife_method": artifact.manifest["jackknife_method"],
        "same_person_jackknife": artifact.manifest["same_person_jackknife"],
        "manifest_sha256": artifact.manifest_sha256,
        "dimensions": {
            "N": axes["samples"]["count"],
            "M": axes["variants"]["count"],
            "Q": len(axes["basis"]["names"]),
            "P": len(axes["pairs"]["table"]),
            "K": len(axes["annotations"]["names"]),
            "C": len(axes["components"]["table"]),
            "J": len(artifact.block_labels),
            "B": artifact.manifest["randomization"]["probe_count"],
        },
        "pass_ledger": dict(artifact.manifest["pass_ledger"]),
        "performance_ledger": dict(artifact.manifest["performance_ledger"]),
        "per_variant_panel": dict(artifact.manifest["per_variant_panel"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = _plan(args)
    elif args.command == "inspect":
        result = _inspect(args.artifact)
    else:
        parser.error("unsupported command")
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
