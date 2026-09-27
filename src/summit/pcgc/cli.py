"""Development CLI for typed binary preparation and inference."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from summit.context.spec import canonical_json, canonical_sha256
from summit.prediction.artifacts import load_genotype_scale, write_json, file_digest
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.spec import GenotypeScale
from summit.sumstats.binary import fit_binary_risk, prepare_binary_risk
from .artifacts import load_artifact, write_artifact
from .genotype import prepare_from_source
from .moments import fit_moments


def add_arguments(parser):
    group = parser.add_argument_group("Binary liability regression")
    group.add_argument("--binary-method", choices=("liability", "pcgc", "pcgc-inverse", "pcgc-basis", "pcgc-ld"),
                       help="Ascertainment-aware method; requires a typed binary artifact or raw score preparation.")
    group.add_argument("--make-binary-sumstats", metavar="TSV", help="Prepare binary moments from FID, IID, Y and optional risk covariates.")
    group.add_argument("--binary-scale", help="Population scale: sealed SUMMIT directory or SNP/A1/A2/MEAN/INV_SD TSV.")
    group.add_argument("--binary-reference-geno", help="Independent population BED/PGEN reference for the pcgc-ld approximation.")
    group.add_argument("--binary-prevalence", type=float, help="Externally supplied population prevalence.")
    group.add_argument("--binary-genome-build", help="Genome build matching the sealed genotype scale.")
    group.add_argument("--binary-covariates", help="Comma-separated exogenous covariate columns in the sample table; no intercept.")
    group.add_argument("--binary-risk-column", help="Column of supplied individual population risks instead of a probit fit.")
    group.add_argument("--binary-covariate-variance", type=float, help="Population variance of the covariate liability predictor.")
    group.add_argument("--binary-probes", type=int, default=256, help="Reference variant probes (default: 256).")
    group.add_argument("--binary-seed", type=int, default=0)
    group.add_argument("--binary-memory-gib", type=float, default=1.)
    group.add_argument("--binary-block-size", type=int, default=256)
    group.add_argument("--binary-basis-columns", help="Comma-separated sample-table basis columns for pcgc-basis.")
    group.add_argument("--binary-basis-coefficients", help="Comma-separated coefficients; basis must span the risk sensitivity exactly.")
    group.add_argument("--_binary-research", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--binary-research", action="store_true",
                       help="Enable unqualified inverse/external-LD methods or experimental jackknife uncertainty.")


def selected_columns(table, names):
    if names is None:
        return None
    names = names.split(",")
    if not names or len(set(names)) != len(names) or any(n not in table for n in names):
        raise ValueError("requested sample-table columns are missing or duplicated")
    return table[names].to_numpy(dtype=float)


def read_table(path, string_columns):
    # pandas otherwise silently renames duplicate columns (e.g. Y -> Y.1).
    header = pd.read_csv(path, sep=r"\s+", header=None, nrows=1, dtype=str, keep_default_na=False).iloc[0].tolist()
    if len(set(header)) != len(header):
        raise ValueError("binary input table has duplicate column names")
    return pd.read_csv(path, sep=r"\s+", dtype={name: str for name in string_columns}, keep_default_na=False)


def sample_table(path, source):
    table = read_table(path, ("FID", "IID"))
    if not {"FID", "IID", "Y"} <= set(table.columns):
        raise ValueError("binary sample table requires FID, IID and Y columns (Y coded 0/1)")
    if table.duplicated(["FID", "IID"]).any():
        raise ValueError("duplicate FID/IID in binary sample table")
    lookup = {pair: i for i, pair in enumerate(source.samples)}
    requested = list(zip(table.FID, table.IID))
    if any(pair not in lookup for pair in requested):
        raise ValueError("binary sample table includes unknown genotype sample IDs")
    rows = np.array([lookup[pair] for pair in requested], dtype=np.int64)
    order = np.argsort(rows)
    return rows[order], table.iloc[order].reset_index(drop=True)


def annotation_table(path, source):
    m = len(source.variants.ids)
    if path is None:
        return np.ones((m, 1)), ["all"]
    table = read_table(path, ("SNP",))
    if "SNP" not in table or table.SNP.duplicated().any() or set(table.SNP) != set(source.variants.ids):
        raise ValueError("binary annotation table must contain each genotype SNP exactly once")
    table = table.set_index("SNP").loc[list(source.variants.ids)]
    if len(table.columns) < 1:
        raise ValueError("binary annotations require at least one weight column")
    return table.to_numpy(dtype=float), list(table.columns)


def population_scale(path, source):
    if Path(path).is_dir():
        return load_genotype_scale(path)
    table = read_table(path, ("SNP", "A1", "A2"))
    if set(table.columns) != {"SNP", "A1", "A2", "MEAN", "INV_SD"}:
        raise ValueError("population scale TSV requires exactly SNP, A1, A2, MEAN, INV_SD columns")
    if table.SNP.duplicated().any() or set(table.SNP) != set(source.variants.ids):
        raise ValueError("population scale must contain each genotype SNP exactly once")
    table = table.set_index("SNP").loc[list(source.variants.ids)]
    if tuple(table.A1) != source.variants.counted or tuple(table.A2) != source.variants.other:
        raise ValueError("population scale A1/A2 must match the genotype reader's counted/other alleles (BED A1; PGEN REF)")
    identity = file_digest(path)
    return GenotypeScale(table.MEAN.to_numpy(dtype=float), table.INV_SD.to_numpy(dtype=float),
        source.variants.identity, canonical_sha256({"declared_population_scale": identity}),
        {"population_scale": True, "source_sha256": identity, "estimation": "externally_supplied"}, ddof=0)


def prepare(args):
    if any(value is None for value in (args.geno, args.binary_scale, args.binary_prevalence, args.binary_genome_build)):
        raise ValueError("--make-binary-sumstats requires --geno, --binary-scale, --binary-prevalence and --binary-genome-build")
    if args.binary_risk_column and args.binary_covariates:
        raise ValueError("choose supplied population risks or fitted risk covariates")
    if args.binary_covariate_variance is not None and not args.binary_risk_column:
        raise ValueError("--binary-covariate-variance accompanies supplied population risks")
    if not np.isfinite(args.binary_memory_gib) or args.binary_memory_gib <= 0:
        raise ValueError("binary reference memory must be finite and positive")
    with ExitStack() as stack:
        source = stack.enter_context(FileGenotypeSource(args.geno, genome_build=args.binary_genome_build))
        scale = population_scale(args.binary_scale, source)
        reference = None if args.binary_reference_geno is None else stack.enter_context(
            FileGenotypeSource(args.binary_reference_geno, genome_build=args.binary_genome_build))
        rows, table = sample_table(args.make_binary_sumstats, source)
        y = table.Y.to_numpy(dtype=float)
        if args.binary_risk_column:
            k = selected_columns(table, args.binary_risk_column)
            if k.shape[1] != 1:
                raise ValueError("supply one population risk column")
            risk = prepare_binary_risk(y, args.binary_prevalence, population_risk=k[:, 0],
                                       covariate_variance=args.binary_covariate_variance)
        else:
            risk = fit_binary_risk(y, args.binary_prevalence, selected_columns(table, args.binary_covariates))
        a, names = annotation_table(args.annot, source)
        basis = selected_columns(table, args.binary_basis_columns)
        c = None if args.binary_basis_coefficients is None else np.array([float(x) for x in args.binary_basis_coefficients.split(",")])
        return prepare_from_source(source, scale, rows, risk, a, annotation_names=names,
                                   method=args.binary_method, basis=basis, coefficients=c,
                                   probes=args.binary_probes, seed=args.binary_seed,
                                   memory_bytes=int(args.binary_memory_gib*2**30),
                                   threads=1 if args.num_threads is None else args.num_threads, block_size=args.binary_block_size,
                                   reference_source=reference)


def run(args, argv):
    """Only explicitly supported flags can enter the binary scientific path."""
    allowed = {"--binary-method", "--make-binary-sumstats", "--binary-scale", "--binary-prevalence",
               "--binary-genome-build", "--binary-covariates", "--binary-risk-column", "--binary-covariate-variance",
               "--binary-probes", "--binary-seed", "--binary-memory-gib", "--binary-block-size",
               "--binary-basis-columns", "--binary-basis-coefficients", "--binary-reference-geno", "--_binary-research", "--binary-research",
               "--geno", "--annot", "--out", "--h2", "--njack", "--num-threads"}
    explicit = {x.split("=", 1)[0] for x in argv if x.startswith("--")}
    if explicit-allowed:
        raise ValueError("unsupported options for the binary contract: "+", ".join(sorted(explicit-allowed)))
    if args.binary_method is None or args.out is None:
        raise ValueError("binary workflows require --binary-method and --out")
    research = args.binary_research or args._binary_research
    if not research and (args.binary_method in ("pcgc-inverse", "pcgc-ld") or "--njack" in explicit):
        raise ValueError("this method or uncertainty option has not passed general qualification; use --binary-research for validation only")
    qualification = "experimental" if research else "point_estimate_only"
    if bool(args.make_binary_sumstats) == bool(args.h2):
        raise ValueError("choose --make-binary-sumstats or --h2 with one typed binary artifact")
    prefix = Path(args.out)
    output = Path(str(prefix)+(".binary.npz" if args.make_binary_sumstats else ".binary.json"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.make_binary_sumstats:
        if "--njack" in explicit:
            raise ValueError("--njack belongs to inference after the reference is complete")
        artifact = prepare(args)
        write_artifact(artifact, output)
    else:
        preparation_options = explicit - {"--binary-method", "--h2", "--out", "--njack", "--num-threads", "--_binary-research", "--binary-research"}
        if preparation_options:
            raise ValueError("binary fit uses the sealed artifact; remove preparation options: "+", ".join(sorted(preparation_options)))
        artifact = load_artifact(args.h2)
        if args.binary_method != artifact.moments.method:
            raise ValueError("binary method disagrees with the prepared artifact; recompute raw scores/reference")
        blocks = None
        if "--njack" in explicit:
            from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
            try:
                count = int(args.njack)
            except ValueError as exc:
                raise ValueError("binary research jackknife requires an integer block count") from exc
            spec = JackknifeSpec.parse(count)
            if spec.mode != "block" or spec.nblocks < 2:
                raise ValueError("binary research jackknife currently requires an integer block count >=2")
            view = SimpleNamespace(nsnps=len(artifact.moments.rhs_rows))
            blocks = JackknifeDesign.from_trace_view(view, spec).unit_id
        result = fit_moments(artifact.moments, block_ids=blocks)
        result.update(kind="summit.pcgc.fit", schema_version=1, input_manifest_hash=artifact.manifest["manifest_hash"],
                      qualification=qualification, risk=artifact.manifest["risk"],
                      annotation_names=artifact.manifest["annotation_names"], diagnostics=artifact.manifest["diagnostics"])
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    print(canonical_json({"output": str(output), "method": args.binary_method, "qualification": qualification}))
    return 0
