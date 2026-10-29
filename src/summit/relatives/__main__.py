"""Command line for related-pair HE moments (`summit relatives ...`, or `python -m summit.relatives ...`).

  summit relatives moments --geno PREFIX --pheno FILE --pairs FILE [--covar FILE] [--annot FILE] [--inbreeding] --out PREFIX
  summit relatives append  --sumstats FILE --rel PREFIX.rel_m.tsv --pheno NAME [--annot FILE] --out FILE
  summit relatives cross-m12 --rel PREFIX.rel_m12.tsv --pheno1 A --pheno2 B [--annot FILE] --out FILE   (summit --rg --rel-m12)
  summit relatives traces  --geno PREFIX [--covar FILE] [--pairs FILE] [--annot FILE] [--inbreeding] --out PREFIX
                                                                                        (summit --h2 --jk-traces)
  summit relatives longrange --geno PREFIX [--covar FILE] [--pairs FILE] [--annot FILE] --out PREFIX
                                                                                        (summit --h2 --long-range-se)

By default `moments` writes one REL_M value per phenotype (and per class of --annot), the V-weighted mean of the
per-SNP related-pair moments, and `append` writes it as a constant REL_M column. `--per-snp` writes the exact per-SNP
values instead (PREFIX.tsv.gz), which `append` and `cross-m12` also accept.

The related pairs' share of in-sample LD scores (LD_REL_<column>) is written by the LD-score step:
  summit --geno PREFIX --covar FILE --relative-pairs FILE --out PREFIX
"""
import argparse
import sys

from ..logger import Logger
from .moments import (append_rel_m_to_sumstats, compute_relative_pair_moments, compute_relative_pair_totals,
                      extract_rel_m12)
from .traces import compute_jackknife_traces, compute_long_range_sums


def main(argv=None):
    ap = argparse.ArgumentParser(prog="summit relatives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("moments", help="related-pair HE moments: one REL_M value per phenotype (default) or per SNP")
    m.add_argument("--geno", required=True, help="PLINK 1 BED prefix or .bed path (the GWAS genotypes)")
    m.add_argument("--pheno", required=True, help="FID IID pheno1 pheno2 ...")
    m.add_argument("--pheno-cols", default=None, help="comma-separated phenotype columns (default: all)")
    m.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), without an intercept column")
    m.add_argument("--pairs", required=True, help="related pairs: IID1 IID2 [FID1 FID2] [kinship]")
    m.add_argument("--kinship-col", default=None)
    m.add_argument("--kinship-min", type=float, default=None)
    m.add_argument("--no-closure", action="store_true", help="exclude only listed pairs, not all within-family pairs")
    m.add_argument("--annot", default=None,
                   help="partitioning annotation (SNP + one 0/1 column per class, e.g. MAF/LD bins): also write one "
                        "REL_M value per class, for partitioned fits")
    m.add_argument("--per-snp", action="store_true",
                   help="write the exact per-SNP REL_M (PREFIX.tsv.gz; one genotype pass per complete-case sample) "
                        "as well as the per-phenotype values")
    m.add_argument("--step-size", type=int, default=None, help="SNPs per genotype block (default: about 512 MiB)")
    m.add_argument("--cross", action="store_true",
                   help="also REL_M12 for every phenotype pair (for rg; with --per-snp only pairs with the same "
                        "complete-case sample)")
    m.add_argument("--inbreeding", action="store_true",
                   help="also leave each individual paired with themselves (the diagonal of the GRM) out of the h2 "
                        "estimating equation, which removes bias from inbreeding (residual variance that grows with "
                        "inbreeding) and the covariate leverage of the related pairs. Writes REL_S, REL_LAMBDA and "
                        "REL_DIAG (and REL_S12 with --cross); `append` and `cross-m12` copy them, and `summit --h2` and `--rg` use "
                        "them. Costs one extra projection of the genotypes per complete-case sample, and a few percent "
                        "of precision in samples of a few thousand.")
    m.add_argument("--out", required=True)
    a_ = sub.add_parser("append", help="copy a sumstats file adding a REL_M column")
    a_.add_argument("--sumstats", required=True)
    a_.add_argument("--rel", required=True, help="PREFIX.rel_m.tsv (per phenotype) or PREFIX.tsv.gz (--per-snp)")
    a_.add_argument("--pheno", required=True)
    a_.add_argument("--annot", default=None,
                    help="with PREFIX.rel_m.tsv: the annotation given to `moments`, for per-class REL_M values")
    a_.add_argument("--out", required=True)
    x = sub.add_parser("cross-m12", help="write REL_M12 for one phenotype pair (summit --rg --rel-m12)")
    x.add_argument("--rel", required=True, help="PREFIX.rel_m12.tsv (moments --cross) or PREFIX.tsv.gz (--per-snp --cross)")
    x.add_argument("--pheno1", required=True)
    x.add_argument("--pheno2", required=True)
    x.add_argument("--annot", default=None,
                   help="with PREFIX.rel_m12.tsv: the annotation given to `moments`, for per-class values (partitioned rg)")
    x.add_argument("--out", required=True)
    t = sub.add_parser("traces", help="genotype-only block-pair traces for the rescaled jackknife (summit --h2 --jk-traces)")
    t.add_argument("--geno", required=True, help="PLINK 1 BED prefix or .bed path (the GWAS and LD-score genotypes)")
    t.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), without an intercept column")
    t.add_argument("--pairs", default=None, help="related pairs left out of the first equation (as for `moments`)")
    t.add_argument("--kinship-col", default=None)
    t.add_argument("--kinship-min", type=float, default=None)
    t.add_argument("--no-closure", action="store_true")
    t.add_argument("--annot", default=None, help="the partitioning annotation of the fits (every SNP in one class)")
    t.add_argument("--inbreeding", action="store_true",
                   help="also the traces without self-pairs, for fits with the `moments --inbreeding` values")
    t.add_argument("--keep", default=None, help="FID IID of the samples to use (default: all with complete covariates)")
    t.add_argument("--nblocks", type=int, default=200,
                   help="SUMMIT's contiguous jackknife blocks on this SNP axis (cells also split by chromosome, so "
                        "--njack chr fits can use the same traces)")
    t.add_argument("--probes", type=int, default=100, help="Rademacher probes (default 100)")
    t.add_argument("--seed", type=int, default=20261029)
    t.add_argument("--step-size", type=int, default=None)
    t.add_argument("--out", required=True)
    lr = sub.add_parser("longrange", help="LD-score mass between jackknife blocks, for the long-range LD correction of "
                                          "h2 and rg standard errors (summit --h2/--rg --long-range-se)")
    lr.add_argument("--geno", required=True, help="PLINK 1 BED prefix or .bed path (the GWAS and LD-score genotypes)")
    lr.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), without an intercept column")
    lr.add_argument("--pairs", default=None, help="related pairs left out of the first equation (as for `moments`)")
    lr.add_argument("--kinship-col", default=None)
    lr.add_argument("--kinship-min", type=float, default=None)
    lr.add_argument("--no-closure", action="store_true")
    lr.add_argument("--annot", default=None, help="the partitioning annotation of the fits (every SNP in one class)")
    lr.add_argument("--keep", default=None, help="FID IID of the samples to use (default: all with complete covariates)")
    lr.add_argument("--nblocks", type=int, default=200, help="SUMMIT's contiguous jackknife blocks (default 200)")
    lr.add_argument("--probes", type=int, default=100, help="Rademacher probes (default 100)")
    lr.add_argument("--seed", type=int, default=20261030)
    lr.add_argument("--step-size", type=int, default=None)
    lr.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "moments":
        log = Logger()
        log.attach_file(args.out + ".log", mode="w")
        cols = None if args.pheno_cols is None else [c.strip() for c in args.pheno_cols.split(",") if c.strip()]
        run = compute_relative_pair_moments if args.per_snp else compute_relative_pair_totals
        extra = {"inbreeding": True} if args.inbreeding else {}
        summary = run(
            args.geno, args.pheno, args.out, pairs=args.pairs, covar=args.covar, pheno_cols=cols,
            kinship_col=args.kinship_col, kinship_min=args.kinship_min, closure=not args.no_closure,
            step_size=args.step_size, cross=args.cross, annot=args.annot, log=log, **extra,
        )
        for name, info in summary["phenotypes"].items():
            log._log(f"[relatives] {name}: n={info['n']} pairs={info['pairs_used']} REL_M={info['rel_m']:.6g}")
        written = [args.out + ".rel_m.tsv"] + ([args.out + ".rel_m12.tsv"] if args.cross else [])
        log._log("[relatives] wrote " + ", ".join(written + ([args.out + ".tsv.gz"] if args.per_snp else [])))
    elif args.cmd == "traces":
        log = Logger()
        log.attach_file(args.out + ".jktraces.log", mode="w")
        compute_jackknife_traces(
            args.geno, args.out, covar=args.covar, pairs=args.pairs, kinship_col=args.kinship_col,
            kinship_min=args.kinship_min, closure=not args.no_closure, annot=args.annot, inbreeding=args.inbreeding,
            keep=args.keep, nblocks=args.nblocks, probes=args.probes, seed=args.seed, step_size=args.step_size, log=log,
        )
    elif args.cmd == "longrange":
        log = Logger()
        log.attach_file(args.out + ".longrange.log", mode="w")
        compute_long_range_sums(
            args.geno, args.out, covar=args.covar, pairs=args.pairs, kinship_col=args.kinship_col,
            kinship_min=args.kinship_min, closure=not args.no_closure, annot=args.annot, keep=args.keep,
            nblocks=args.nblocks, probes=args.probes, seed=args.seed, step_size=args.step_size, log=log,
        )
    elif args.cmd == "append":
        missing = append_rel_m_to_sumstats(args.sumstats, args.rel, args.pheno, args.out, annot=args.annot)
        print(f"wrote {args.out}; SNPs without REL_M: {missing}")
    else:
        c12 = extract_rel_m12(args.rel, args.pheno1, args.pheno2, args.out, annot=args.annot)
        print(f"wrote {args.out}; REL_M12={c12:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
