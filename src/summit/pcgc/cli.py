"""CLI for typed binary preparation and SNP-block jackknife inference."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from summit.cli_options import explicit_options
from summit.context.spec import canonical_json, canonical_sha256
from summit.prediction.artifacts import load_genotype_scale, write_json, file_digest
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.spec import GenotypeScale
from summit.sumstats.binary import fit_binary_risk, prepare_binary_risk
from .artifacts import write_artifact
from .split_io import load_fit_artifact, output_paths, write_split_artifact
from .genotype import prepare_from_source
from .moments import fit_moments

DEFAULT_BINARY_NJACK = 200


def add_arguments(parser):
    group = parser.add_argument_group("Binary liability regression")
    group.add_argument("--binary-method", choices=("liability", "pcgc", "pcgc-inverse", "pcgc-basis", "pcgc-ld"),
                       help="Ascertainment-aware method; prepare binary scores or fit PCGC summary statistics with matching LD scores.")
    group.add_argument("--make-binary-sumstats", metavar="TSV", help="Prepare separate PCGC summary statistics and LD scores from FID, IID, Y and optional risk covariates.")
    group.add_argument("--binary-output-format", choices=("separate", "combined"), default="separate",
                       help="Preparation output: separate summary/reference files (default), or a combined .binary.npz file.")
    group.add_argument("--binary-scale", help="Population scale: SUMMIT scale directory or SNP/A1/A2/MEAN/INV_SD TSV.")
    group.add_argument("--binary-reference-geno", help="Independent population BED/PGEN reference for the pcgc-ld approximation.")
    group.add_argument("--binary-prevalence", type=float, help="Externally supplied population prevalence.")
    group.add_argument("--binary-covariates", help="Comma-separated exogenous covariate columns in the sample table; no intercept.")
    group.add_argument("--binary-risk-column", help="Column of supplied individual population risks instead of a probit fit.")
    group.add_argument("--binary-covariate-variance", type=float, help="Population variance of the covariate liability predictor.")
    group.add_argument("--binary-basis-columns", help="Comma-separated sample-table basis columns for pcgc-basis.")
    group.add_argument("--binary-basis-coefficients", help="Comma-separated coefficients; basis must span the risk sensitivity exactly.")
    from .gxe_cli import add_arguments as add_gxe_arguments
    add_gxe_arguments(group)


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
    if any(value is None for value in (args.geno, args.binary_scale, args.binary_prevalence)):
        raise ValueError("--make-binary-sumstats requires --geno, --binary-scale, --binary-prevalence")
    if args.binary_risk_column and args.binary_covariates:
        raise ValueError("choose supplied population risks or fitted risk covariates")
    if args.binary_covariate_variance is not None and not args.binary_risk_column:
        raise ValueError("--binary-covariate-variance accompanies supplied population risks")
    if args.memory_gib == "auto" or not np.isfinite(args.memory_gib) or args.memory_gib <= 0:
        raise ValueError("binary --memory-gib requires a finite positive number")
    if args.step_size == "auto":
        raise ValueError("binary --block-size requires a positive integer; auto is supported only by GxE reference generation")
    if getattr(args, "binary_context_columns", None):
        from .gxe_cli import prepare as prepare_contextual
        return prepare_contextual(args)
    if any(getattr(args, name, None) for name in ("binary_unit_liability", "binary_liability_sd_column", "binary_ld_factorization",
                                                "binary_genotype_covariates", "binary_reference_covariates", "binary_sampling_partners", "binary_architecture_probes")):
        raise ValueError("contextual binary options require --binary-context-columns")
    with ExitStack() as stack:
        source = stack.enter_context(FileGenotypeSource(args.geno, genome_build=args.genome_build))
        scale = population_scale(args.binary_scale, source)
        reference = None if args.binary_reference_geno is None else stack.enter_context(
            FileGenotypeSource(args.binary_reference_geno, genome_build=args.genome_build))
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
                                   probes=args.nvecs, seed=args.seed,
                                   memory_bytes=int(args.memory_gib*2**30),
                                   threads=1 if args.num_threads is None else args.num_threads, block_size=args.step_size,
                                   reference_source=reference)


def run(args, argv):
    """Only explicitly supported flags can enter the binary scientific path."""
    allowed = {"--binary-method", "--make-binary-sumstats", "--binary-scale", "--binary-prevalence",
               "--genome-build", "--binary-covariates", "--binary-risk-column", "--binary-covariate-variance",
               "--nvecs", "--seed", "--memory-gib", "--block-size",
               "--binary-basis-columns", "--binary-basis-coefficients", "--binary-reference-geno",
               "--geno", "--annot", "--out", "--h2", "--ldscores", "--binary-output-format", "--njack", "--num-threads"}
    from .gxe_cli import OPTIONS
    allowed |= OPTIONS
    explicit = explicit_options(argv)
    if explicit-allowed:
        raise ValueError("unsupported options for the binary analysis: "+", ".join(sorted(explicit-allowed)))
    if args.binary_method is None or args.out is None:
        raise ValueError("binary workflows require --binary-method and --out")
    if bool(args.make_binary_sumstats) == bool(args.h2):
        raise ValueError("choose --make-binary-sumstats or --h2 with saved PCGC inputs")
    prefix = Path(args.out)
    if args.make_binary_sumstats:
        if "--ldscores" in explicit:
            raise ValueError("--ldscores belongs to inference; preparation writes its matching PCGC reference")
        if "--njack" in explicit:
            raise ValueError("--njack belongs to inference after the reference is complete")
        separate = args.binary_output_format == "separate"
        outputs = output_paths(prefix) if separate else (Path(str(prefix) + ".binary.npz"),)
        for output in outputs:
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"refusing to overwrite {output}")
        artifact = prepare(args)
        if separate:
            output, reference_output = write_split_artifact(artifact, *outputs)
        elif getattr(args, "binary_context_columns", None):
            from .gxe_io import write_gxe_artifact
            output = write_gxe_artifact(artifact, outputs[0])
        else:
            output = write_artifact(artifact, outputs[0])
    else:
        preparation_options = explicit - {"--binary-method", "--h2", "--ldscores", "--out", "--njack", "--num-threads"}
        if preparation_options:
            raise ValueError("binary fit uses saved PCGC inputs; remove preparation options: "+", ".join(sorted(preparation_options)))
        output = Path(str(prefix) + ".binary.json")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        artifact = load_fit_artifact(args.h2, args.ldscores)
        from .gxe_io import GxEArtifact
        contextual = isinstance(artifact, GxEArtifact)
        if args.binary_method != artifact.moments.method:
            raise ValueError("binary method disagrees with the prepared artifact; recompute raw scores/reference")
        from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
        # The root parser's chromosome default belongs to HE/LDSC. Binary
        # inference uses equal-count SNP blocks and always computes uncertainty.
        try:
            count = int(args.njack) if "--njack" in explicit else DEFAULT_BINARY_NJACK
        except (ValueError, TypeError) as exc:
            raise ValueError("binary jackknife requires an integer block count >=2") from exc
        if count < 2:
            raise ValueError("binary jackknife requires an integer block count >=2")
        view = SimpleNamespace(nsnps=len(artifact.moments.rhs_rows))
        if count > view.nsnps:
            raise ValueError(f"binary --njack ({count}) exceeds the {view.nsnps} SNPs; choose a smaller block count >=2")
        blocks = JackknifeDesign.from_trace_view(view, JackknifeSpec.parse(count)).unit_id
        if contextual:
            from .gxe_cli import fit as fit_contextual
            result = fit_contextual(artifact, blocks, threads=args.num_threads)
        else:
            result = fit_moments(artifact.moments, block_ids=blocks)
            result.update(kind="summit.pcgc.fit", schema_version=2, input_manifest_hash=artifact.manifest["manifest_hash"],
                          risk=artifact.manifest["risk"],
                          annotation_names=artifact.manifest["annotation_names"], diagnostics=artifact.manifest["diagnostics"])
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    message = {"output": str(output), "method": args.binary_method}
    if args.make_binary_sumstats and separate:
        message["ldscores"] = str(reference_output)
    print(canonical_json(message))
    return 0
