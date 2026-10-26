"""Command line for related-pair HE moments (`summit relatives ...`, or `python -m summit.relatives ...`).

  summit relatives moments --geno PREFIX --pheno FILE --pairs FILE [--covar FILE] [--annot FILE] --out PREFIX
  summit relatives append  --sumstats FILE --rel PREFIX.rel_m.tsv --pheno NAME [--annot FILE] --out FILE
  summit relatives cross-m12 --rel PREFIX.rel_m12.tsv --pheno1 A --pheno2 B --out FILE   (for summit --rg --rel-m12)

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
                   help="also REL_M12 for every phenotype pair with the same samples (for rg)")
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
    x.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "moments":
        log = Logger()
        log.attach_file(args.out + ".log", mode="w")
        cols = None if args.pheno_cols is None else [c.strip() for c in args.pheno_cols.split(",") if c.strip()]
        run = compute_relative_pair_moments if args.per_snp else compute_relative_pair_totals
        summary = run(
            args.geno, args.pheno, args.out, pairs=args.pairs, covar=args.covar, pheno_cols=cols,
            kinship_col=args.kinship_col, kinship_min=args.kinship_min, closure=not args.no_closure,
            step_size=args.step_size, cross=args.cross, annot=args.annot, log=log,
        )
        for name, info in summary["phenotypes"].items():
            log._log(f"[relatives] {name}: n={info['n']} pairs={info['pairs_used']} REL_M={info['rel_m']:.6g}")
        written = [args.out + ".rel_m.tsv"] + ([args.out + ".rel_m12.tsv"] if args.cross else [])
        log._log("[relatives] wrote " + ", ".join(written + ([args.out + ".tsv.gz"] if args.per_snp else [])))
    elif args.cmd == "append":
        missing = append_rel_m_to_sumstats(args.sumstats, args.rel, args.pheno, args.out, annot=args.annot)
        print(f"wrote {args.out}; SNPs without REL_M: {missing}")
    else:
        c12 = extract_rel_m12(args.rel, args.pheno1, args.pheno2, args.out)
        print(f"wrote {args.out}; REL_M12={c12:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
