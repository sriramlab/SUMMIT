"""Command line for related-pair HE moments.

  python -m summit.relatives moments --geno PREFIX --pheno FILE --pairs FILE [--covar FILE] --out PREFIX
  python -m summit.relatives append  --sumstats FILE --rel PREFIX.tsv.gz --pheno NAME --out FILE
  python -m summit.relatives adjust-ld --ldscores IN.gw.ldscore.gz --rel PREFIX.tsv.gz --out OUT.gw.ldscore.gz
"""
import argparse
import sys

from ..logger import Logger
from .moments import adjust_ldscores, append_rel_m_to_sumstats, compute_relative_pair_moments


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m summit.relatives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("moments", help="per-SNP related-pair HE moments (REL_M) and optional LD_REL")
    m.add_argument("--geno", required=True, help="PLINK 1 BED prefix (the GWAS genotypes)")
    m.add_argument("--pheno", required=True, help="FID IID pheno1 pheno2 ...")
    m.add_argument("--pheno-cols", default=None, help="comma-separated phenotype columns (default: all)")
    m.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), without an intercept column")
    m.add_argument("--pairs", required=True, help="related pairs: IID1 IID2 [FID1 FID2] [kinship]")
    m.add_argument("--kinship-col", default=None)
    m.add_argument("--kinship-min", type=float, default=None)
    m.add_argument("--no-closure", action="store_true", help="exclude only listed pairs, not all within-family pairs")
    m.add_argument("--step-size", type=int, default=5000)
    m.add_argument("--ld-rel", action="store_true", help="also write the related-pair share of in-sample LD scores")
    m.add_argument("--annot", default=None,
                   help="annotation of partitioned in-sample LD scores (LD_REL_<column> per annotation column)")
    m.add_argument("--out", required=True)
    a_ = sub.add_parser("append", help="copy a sumstats file adding a REL_M column")
    a_.add_argument("--sumstats", required=True)
    a_.add_argument("--rel", required=True)
    a_.add_argument("--pheno", required=True)
    a_.add_argument("--out", required=True)
    d = sub.add_parser("adjust-ld", help="subtract LD_REL from in-sample LD scores of the related sample")
    d.add_argument("--ldscores", required=True)
    d.add_argument("--rel", required=True, help="moments output (.tsv.gz) written with --ld-rel")
    d.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "moments":
        log = Logger()
        log.attach_file(args.out + ".log", mode="w")
        cols = None if args.pheno_cols is None else [c.strip() for c in args.pheno_cols.split(",") if c.strip()]
        summary = compute_relative_pair_moments(
            args.geno, args.pheno, args.out, pairs=args.pairs, covar=args.covar, pheno_cols=cols,
            kinship_col=args.kinship_col, kinship_min=args.kinship_min, closure=not args.no_closure,
            step_size=args.step_size, ld_rel=args.ld_rel, annot=args.annot, log=log,
        )
        for name, info in summary["phenotypes"].items():
            log._log(f"[relatives] {name}: n={info['n']} pairs={info['pairs_used']} c_rel={info['c_rel']:.6g}")
        if "ld_rel" in summary:
            sums = ", ".join(f"{c}={v:.6g}" for c, v in zip(summary["ld_rel"]["columns"], summary["ld_rel"]["sums"]))
            log._log(f"[relatives] LD_REL sums: {sums}")
    elif args.cmd == "append":
        missing = append_rel_m_to_sumstats(args.sumstats, args.rel, args.pheno, args.out)
        print(f"wrote {args.out}; SNPs without REL_M: {missing}")
    else:
        info = adjust_ldscores(args.ldscores, args.rel, args.out)
        for c, v in info["columns"].items():
            print(f"{c}: sum LD {v['sum_ld']:.6g} - sum {v['ld_rel_column']} {v['sum_ld_rel']:.6g}")
        print(f"wrote {args.out} ({info['nsnps']} SNPs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
