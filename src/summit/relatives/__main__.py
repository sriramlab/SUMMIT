"""Command line for related-pair HE moments (`summit relatives ...`, or `python -m summit.relatives ...`).

  summit relatives moments --geno PREFIX --pheno FILE --pairs FILE [--covar FILE] --out PREFIX
  summit relatives append  --sumstats FILE --rel PREFIX.tsv.gz --pheno NAME --out FILE
  summit relatives cross-m12 --rel PREFIX.tsv.gz --pheno1 A --pheno2 B --out FILE   (for summit --rg --rel-m12)

The related pairs' share of in-sample LD scores (LD_REL_<column>) is written by the LD-score step:
  summit --geno PREFIX --covar FILE --relative-pairs FILE --out PREFIX
"""
import argparse
import sys

from ..logger import Logger
from .moments import append_rel_m_to_sumstats, compute_relative_pair_moments, extract_rel_m12


def main(argv=None):
    ap = argparse.ArgumentParser(prog="summit relatives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("moments", help="per-SNP related-pair HE moments (REL_M, and REL_M12 with --cross)")
    m.add_argument("--geno", required=True, help="PLINK 1 BED prefix (the GWAS genotypes)")
    m.add_argument("--pheno", required=True, help="FID IID pheno1 pheno2 ...")
    m.add_argument("--pheno-cols", default=None, help="comma-separated phenotype columns (default: all)")
    m.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), without an intercept column")
    m.add_argument("--pairs", required=True, help="related pairs: IID1 IID2 [FID1 FID2] [kinship]")
    m.add_argument("--kinship-col", default=None)
    m.add_argument("--kinship-min", type=float, default=None)
    m.add_argument("--no-closure", action="store_true", help="exclude only listed pairs, not all within-family pairs")
    m.add_argument("--step-size", type=int, default=5000)
    m.add_argument("--cross", action="store_true",
                   help="also write REL_M12_<a>__<b> for every phenotype pair with the same samples (for rg)")
    m.add_argument("--out", required=True)
    a_ = sub.add_parser("append", help="copy a sumstats file adding a REL_M column")
    a_.add_argument("--sumstats", required=True)
    a_.add_argument("--rel", required=True)
    a_.add_argument("--pheno", required=True)
    a_.add_argument("--out", required=True)
    x = sub.add_parser("cross-m12", help="write SNP/REL_M12 for one phenotype pair (summit --rg --rel-m12)")
    x.add_argument("--rel", required=True, help="moments output (.tsv.gz) written with --cross")
    x.add_argument("--pheno1", required=True)
    x.add_argument("--pheno2", required=True)
    x.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "moments":
        log = Logger()
        log.attach_file(args.out + ".log", mode="w")
        cols = None if args.pheno_cols is None else [c.strip() for c in args.pheno_cols.split(",") if c.strip()]
        summary = compute_relative_pair_moments(
            args.geno, args.pheno, args.out, pairs=args.pairs, covar=args.covar, pheno_cols=cols,
            kinship_col=args.kinship_col, kinship_min=args.kinship_min, closure=not args.no_closure,
            step_size=args.step_size, cross=args.cross, log=log,
        )
        for name, info in summary["phenotypes"].items():
            log._log(f"[relatives] {name}: n={info['n']} pairs={info['pairs_used']} c_rel={info['c_rel']:.6g}")
        for name, info in summary.get("cross", {}).items():
            log._log(f"[relatives] {name}: c_rel12={info['c_rel12']:.6g}")
    elif args.cmd == "append":
        missing = append_rel_m_to_sumstats(args.sumstats, args.rel, args.pheno, args.out)
        print(f"wrote {args.out}; SNPs without REL_M: {missing}")
    else:
        c12 = extract_rel_m12(args.rel, args.pheno1, args.pheno2, args.out)
        print(f"wrote {args.out}; c_rel12={c12:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
