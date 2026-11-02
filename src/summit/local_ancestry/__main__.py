"""Command line for local ancestry as a set of markers (`summit local-ancestry ...`).

  summit local-ancestry markers --calls PATTERN --format {lanc,msp} [--sites PATTERN] [--samples FILE] --fam FAM
                                --ancestry CODE [--spacing BP] [--fill CODE] --out PREFIX
  summit local-ancestry merge   --geno SNP_PREFIX --markers LA_PREFIX --annot ANNOT --out PREFIX
  summit local-ancestry test    --geno PREFIX --annot PREFIX.annot.tsv.gz --pheno FILE --covar FILE
                                [--pairs FILE] --out PREFIX

`markers` turns local-ancestry calls into ancestry dosages at evenly spaced positions and writes each individual's global
ancestry (PREFIX.theta.tsv), which belongs among the covariates. `merge` interleaves these markers with the SNPs and adds
an LA class to the annotation. The GWAS of the merged fileset (SNP associations and an admixture-mapping scan), its
in-sample LD scores and `summit --h2 --annot` then estimate SNP heritability and heritability tagged only by local
ancestry jointly. `test` tests the local-ancestry component and gives its interval, exactly from the genotypes.
"""
import argparse

from ..logger import Logger


def main(argv=None):
    ap = argparse.ArgumentParser(prog="summit local-ancestry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("markers", help="ancestry dosages at evenly spaced positions, and global ancestry")
    m.add_argument("--calls", required=True, help="local-ancestry calls per chromosome, a path containing {chr}")
    m.add_argument("--format", required=True, choices=["lanc", "msp"],
                   help="lanc: admix-kit .lanc; msp: RFMix 2 .msp.tsv")
    m.add_argument("--sites", default=None,
                   help="lanc only: the .pvar (or .bim) of the calling sites per chromosome, a path containing {chr}")
    m.add_argument("--samples", default=None,
                   help="lanc only: sample IDs in the order of the .lanc rows, one per line (default: the .psam "
                        "next to --sites)")
    m.add_argument("--fam", required=True, help="the .fam of the genotypes; markers follow its individuals")
    m.add_argument("--ancestry", type=int, required=True, help="ancestry code whose haplotypes are counted")
    m.add_argument("--spacing", type=int, default=250_000, help="distance between markers in bp (default 250000)")
    m.add_argument("--fill", type=int, default=None,
                   help="ancestry code given on both haplotypes to individuals of --fam without calls")
    m.add_argument("--out", required=True)
    g = sub.add_parser("merge", help="interleave local-ancestry markers with the SNPs and add an LA annotation class")
    g.add_argument("--geno", required=True, help="PLINK 1 prefix of the SNPs")
    g.add_argument("--markers", required=True, help="PLINK 1 prefix written by `markers` (same individuals and order)")
    g.add_argument("--annot", required=True, help="annotation of the SNPs (SNP + one 0/1 column per class)")
    g.add_argument("--out", required=True)
    t = sub.add_parser("test", help="test and interval for the local-ancestry component (exact, N x N per class)")
    t.add_argument("--geno", required=True, help="PLINK 1 prefix written by `merge`")
    t.add_argument("--annot", required=True, help="annotation written by `merge`")
    t.add_argument("--pheno", required=True)
    t.add_argument("--pheno-cols", default=None)
    t.add_argument("--covar", default=None, help="covariates, including the global ancestry from `markers`")
    t.add_argument("--pairs", default=None, help="related pairs (as for `summit relatives moments`)")
    t.add_argument("--kinship-col", default=None)
    t.add_argument("--kinship-min", type=float, default=None)
    t.add_argument("--no-closure", action="store_true")
    t.add_argument("--min-n", type=int, default=500, help="skip phenotypes with fewer individuals (default 500)")
    t.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    log = Logger()
    log.attach_file(f"{args.out}.{args.cmd}.log", mode="w")
    if args.cmd == "markers":
        from .markers import build_markers
        build_markers(args.calls, args.fam, args.out, fmt=args.format, code=args.ancestry, sites=args.sites,
                      samples=args.samples, spacing=args.spacing, fill=args.fill, log=log._log)
    elif args.cmd == "merge":
        from .markers import merge_markers
        merge_markers(args.geno, args.markers, args.annot, args.out, log=log._log)
    else:
        from .component import local_ancestry_test
        cols = None if args.pheno_cols is None else [c.strip() for c in args.pheno_cols.split(",") if c.strip()]
        local_ancestry_test(args.geno, args.annot, args.pheno, args.out, covar=args.covar, pheno_cols=cols,
                            pairs=args.pairs, kinship_col=args.kinship_col, kinship_min=args.kinship_min,
                            closure=not args.no_closure, min_n=args.min_n, log=log._log)


if __name__ == "__main__":
    main()
