"""Contextual binary CLI integration; additive PCGC remains unchanged."""
from contextlib import ExitStack
import numpy as np

from .gxe import EXTERNAL_CONTRACT, fit_gxe
from .gxe_io import prepare_gxe_from_source

OPTIONS = {"--binary-context-columns", "--binary-unit-liability", "--binary-liability-sd-column",
           "--binary-ld-factorization", "--binary-genotype-covariates", "--binary-reference-covariates",
           "--binary-sampling-partners", "--binary-architecture-probes"}


def add_arguments(group):
    group.add_argument("--binary-context-columns", help="Sample-table exposures/basis columns; an intercept is added, with no sample standardization.")
    scale = group.add_mutually_exclusive_group()
    scale.add_argument("--binary-unit-liability", action="store_true",
                       help="Declare total liability variance conditional on context to be one.")
    scale.add_argument("--binary-liability-sd-column",
                       help="Known positive conditional total liability SD on a common scale; requires supplied risks.")
    group.add_argument("--binary-ld-factorization", action="store_true",
                       help="Explicitly assert the stronger risk/context/genotype pair factorization needed by contextual pcgc-ld.")
    group.add_argument("--binary-genotype-covariates", help="Sample-table ancestry PC columns to regress out of genotypes before context/risk weighting; also included in fitted risks.")
    group.add_argument("--binary-reference-covariates", help="External-reference FID/IID/PC table, with the same PC column names; required with PC-adjusted pcgc-ld.")
    group.add_argument("--binary-sampling-partners",type=int,default=0,
        help="Experimental individual-sampling SEs: partner draws per person (>=2); 0 retains the SNP-block jackknife.")
    group.add_argument("--binary-architecture-probes",type=int,default=0,
        help="Experimental Gaussian SNP-effect uncertainty: variant probes per family (>=2), with individual-sampling SEs.")


def prepare(args):
    from .cli import selected_columns, sample_table, annotation_table, population_scale
    from summit.prediction.genotype import FileGenotypeSource
    from summit.sumstats.binary import fit_binary_risk, prepare_binary_risk
    if bool(args.binary_unit_liability) == bool(args.binary_liability_sd_column):
        raise ValueError("contextual PCGC requires --binary-unit-liability or --binary-liability-sd-column")
    if args.binary_liability_sd_column and not args.binary_risk_column:
        raise ValueError("nonunit liability SD requires supplied population risks; ordinary probit assumes unit conditional variance")
    if args.binary_covariate_variance is not None:
        raise ValueError("contextual population variance is derived on the declared scale; --binary-covariate-variance is additive-only")
    if args.binary_method == "liability":
        raise ValueError("contextual PCGC uses one of the four pcgc methods")
    if bool(args.binary_ld_factorization) != (args.binary_method == "pcgc-ld"):
        raise ValueError("contextual pcgc-ld requires --binary-ld-factorization; other methods do not use it")
    if args.binary_reference_covariates and (args.binary_method != "pcgc-ld" or not args.binary_genotype_covariates):
        raise ValueError("reference covariates require pcgc-ld and --binary-genotype-covariates")
    if args.binary_method == "pcgc-ld" and args.binary_genotype_covariates and not args.binary_reference_covariates:
        raise ValueError("PC-adjusted pcgc-ld requires --binary-reference-covariates")
    if args.binary_architecture_probes and not args.binary_sampling_partners:
        raise ValueError("architecture probes require --binary-sampling-partners")
    with ExitStack() as stack:
        source = stack.enter_context(FileGenotypeSource(args.geno, genome_build=args.genome_build))
        scale = population_scale(args.binary_scale, source)
        rows, table = sample_table(args.make_binary_sumstats, source)
        env = selected_columns(table, args.binary_context_columns)
        contexts = np.column_stack((np.ones(len(rows)), env))
        names = ["intercept", *args.binary_context_columns.split(",")]
        y = table.Y.to_numpy(dtype=float)
        risk_covariates = None
        if args.binary_risk_column:
            k = selected_columns(table, args.binary_risk_column)
            if k.shape[1] != 1:
                raise ValueError("supply one population risk column")
            risk = prepare_binary_risk(y, args.binary_prevalence, population_risk=k[:, 0])
        else:
            # Exposures can affect the risk mean as well as modify genetics.
            cov_names = list(dict.fromkeys([*args.binary_context_columns.split(","),
                *(args.binary_covariates.split(",") if args.binary_covariates else []),
                *(args.binary_genotype_covariates.split(",") if args.binary_genotype_covariates else [])]))
            risk_covariates = selected_columns(table, ",".join(cov_names))
            risk = fit_binary_risk(y, args.binary_prevalence,risk_covariates)
        sd = 1.
        if args.binary_liability_sd_column:
            sd = selected_columns(table, args.binary_liability_sd_column)
            if sd.shape[1] != 1:
                raise ValueError("supply one liability SD column")
            sd = sd[:, 0]
        a, annot_names = annotation_table(args.annot, source)
        options = dict(probes=args.nvecs, seed=args.seed, memory_bytes=int(args.memory_gib*2**30))
        if args.binary_sampling_partners:
            options.update(sampling_partners=args.binary_sampling_partners,sampling_seed=args.seed,
                           risk_covariates=risk_covariates,architecture_probes=args.binary_architecture_probes)
        if args.binary_genotype_covariates:
            options["genotype_covariates"] = selected_columns(table,args.binary_genotype_covariates)
        if args.binary_basis_columns or args.binary_basis_coefficients:
            options.update(basis=selected_columns(table, args.binary_basis_columns),
                           coefficients=None if args.binary_basis_coefficients is None else
                           np.array([float(x) for x in args.binary_basis_coefficients.split(",")]))
        reference = None
        if args.binary_method == "pcgc-ld":
            if any(key in options for key in ("basis", "coefficients")):
                raise ValueError("risk basis contraction is only available for pcgc-basis")
            if args.binary_reference_geno is None:
                raise ValueError("pcgc-ld requires --binary-reference-geno")
            reference = stack.enter_context(FileGenotypeSource(args.binary_reference_geno, genome_build=args.genome_build))
            options["factorization_contract"] = EXTERNAL_CONTRACT
            if args.binary_reference_covariates:
                from .cli import read_table
                import pandas as pd
                ref_table = read_table(args.binary_reference_covariates,("FID","IID"))
                if not {"FID","IID"} <= set(ref_table) or ref_table.duplicated(["FID","IID"]).any():
                    raise ValueError("reference ancestry table requires unique FID/IID")
                index = pd.MultiIndex.from_tuples(reference.samples,names=("FID","IID"))
                ref_table = ref_table.set_index(["FID","IID"])
                if set(ref_table.index) != set(index):
                    raise ValueError("reference ancestry table must match all reference samples")
                options["reference_genotype_covariates"] = selected_columns(ref_table.loc[index],args.binary_genotype_covariates)
        elif args.binary_reference_geno is not None:
            raise ValueError("study-specific contextual PCGC does not use an external reference")
        return prepare_gxe_from_source(source, scale, rows, risk, a, contexts,
            annotation_names=annot_names, context_names=names, liability_sd=sd,
            method=args.binary_method, reference_source=reference, block_size=args.step_size,
            threads=1 if args.num_threads is None else args.num_threads, **options)


def fit(artifact, blocks):
    result = fit_gxe(artifact.moments, block_ids=blocks)
    result.update(kind="summit.pcgc.context_fit", schema_version=1,
                  input_manifest_hash=artifact.manifest["manifest_hash"],
                  risk=artifact.manifest["risk"], annotation_names=artifact.manifest["annotation_names"],
                  context_names=artifact.manifest["context_names"], diagnostics=artifact.manifest["diagnostics"],
                  liability_scale_contract=artifact.manifest["liability_scale_contract"])
    return result
