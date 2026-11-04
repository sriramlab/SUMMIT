"""Command line for LD scores of admixed cohorts without in-sample LD (`summit admixed-ld ...`).

  summit admixed-ld reference --panel AFR_PREFIX --panel EUR_PREFIX --names AFR,EUR [--window-kb 3000]
                              [--genetic-map MAP] --out REF_DIR
  summit admixed-ld curve     --geno COHORT --ref REF_DIR [--covar FILE] [--keep FILE] [--sources AFR,EUR]
                              [--relative-pairs FILE [--kinship-col COL --kinship-min X] [--closure]] --out PREFIX
  summit admixed-ld compose   --ref REF_DIR --sumstats FILE_OR_DIR [...] --annot ANNOT --out PREFIX
                              (--curve PREFIX | --admixture-generations T | --decay-per-cm LAMBDA)
                              [--sources AFR,EUR] [--ancestry-sd SD] [--relatedness X|curve] [--snps FILE]

`reference` is built once from unadmixed panels of the two source populations. `curve` is run by the cohort on its
genotypes and writes only aggregates (PREFIX.curve.tsv and PREFIX.curve.json). `compose` needs the GWAS summary
statistics (BETA, SE, N; CHR and BP for matching by position, or --snps), the annotation of the fit (e.g. MAF/LD bins),
and the decay of the local-ancestry covariance: the cohort's curve, a published admixture date in generations, or a
decay rate per cM. It writes PREFIX.gw.ldscore.gz and PREFIX.gw.M for `summit --h2/--rg`, and PREFIX.admixed_ld.json.
"""
import argparse
import sys

from ..logger import Logger


def _sources(value):
    return None if value is None else [x.strip() for x in value.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="summit admixed-ld")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reference", help="build the resource from source panels (once per reference)")
    r.add_argument("--panel", action="append", required=True,
                   help="PLINK 1 BED or PLINK 2 PGEN prefix of one source panel (repeat; the order sets the names)")
    r.add_argument("--names", required=True, help="comma-separated source names, one per --panel (e.g. AFR,EUR)")
    r.add_argument("--window-kb", type=int, default=3000,
                   help="pairs within this distance keep their source covariances (default 3000)")
    r.add_argument("--genetic-map", default=None,
                   help="genetic map: one file with a chromosome column, or a path containing {chr}. Without it the "
                        "CM column of the panel is used if filled, otherwise 1.2 cM per Mb")
    r.add_argument("--match", choices=["position", "id"], default="position",
                   help="match variants between panels by CHR:BP and alleles (default) or by ID and alleles")
    r.add_argument("--chromosomes", default=None, help="comma-separated chromosomes (default 1-22)")
    r.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="storage of covariances (float16 halves the size; default float32)")
    r.add_argument("--step", type=int, default=2048, help="variants per block (default 2048)")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--out", required=True)

    c = sub.add_parser("curve", help="the cohort's weighted-LD curve and global-ancestry moments (aggregates only)")
    c.add_argument("--geno", required=True, help="PLINK 1 BED or PLINK 2 PGEN prefix of the GWAS sample")
    c.add_argument("--ref", required=True, help="resource written by `summit admixed-ld reference`")
    c.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), as for the LD-score step")
    c.add_argument("--keep", default=None, help="samples to use (FID IID or IID per line)")
    c.add_argument("--sources", default=None,
                   help="two or more source names of the reference (default: its first two); the last is the baseline")
    c.add_argument("--match", choices=["position", "id"], default="position")
    c.add_argument("--cell-cm", type=float, default=0.25, help="cell width in cM (default 0.25)")
    c.add_argument("--max-cm", type=float, default=None,
                   help="largest distance of the curve in cM (default: the longest chromosome)")
    c.add_argument("--relative-pairs", default=None,
                   help="relative pairs (IID1 IID2 [FID1 FID2] [kinship]): also write the sum of their squared "
                        "relatedness after the covariates, computed from the genotypes, for `compose --relatedness "
                        "curve`. The listed pairs are used, selected with --kinship-col and --kinship-min")
    c.add_argument("--kinship-col", default=None, help="kinship column of --relative-pairs used with --kinship-min")
    c.add_argument("--kinship-min", type=float, default=None, help="keep only pairs whose kinship exceeds this value")
    c.add_argument("--closure", action="store_true",
                   help="also use every pair within a family (connected component of the listed pairs)")
    c.add_argument("--keep-panel-noise", "--no-noise-correction", dest="keep_panel_noise", action="store_true",
                   help="use raw products of the reference contrasts in the denominators and the global-ancestry "
                        "estimator (the behaviour before 4 Nov 2026), which shrinks the kernel by the frequencies' "
                        "sampling variance (small panels, derived sources)")
    c.add_argument("--step", type=int, default=4096)
    c.add_argument("--threads", type=int, default=1)
    c.add_argument("--out", required=True)

    m = sub.add_parser("compose", help="LD scores from the resource, the GWAS and the decay of local-ancestry covariance")
    m.add_argument("--ref", required=True)
    m.add_argument("--sumstats", nargs="+", required=True,
                   help="SUMMIT summary statistics (files or a directory); their SEs give mean ancestry and kappa(0)")
    m.add_argument("--annot", required=True, help="annotation of the fit: SNP and one 0/1 column per class")
    m.add_argument("--snps", default=None,
                   help="variant table of the GWAS (BIM, PVAR, or CHR BP SNP A1 A2) if the summary statistics lack "
                        "CHR and BP")
    m.add_argument("--sources", default=None,
                   help="two or more source names (default: the reference's first two); the last is the baseline of "
                        "the contrasts")
    m.add_argument("--match", choices=["position", "id"], default="position")
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--curve", default=None, help="prefix of the cohort's curve (`summit admixed-ld curve`)")
    g.add_argument("--admixture-generations", type=float, default=None,
                   help="generations since admixture (e.g. a published date); decay = generations / 100 per cM")
    g.add_argument("--decay-per-cm", type=float, default=None, help="decay rate of kappa per cM")
    m.add_argument("--mean-ancestry", default=None,
                   help="override the mean ancestry from the SEs: K-1 or K comma-separated values in source order")
    m.add_argument("--kappa0", default=None,
                   help="override kappa_res(0) from the SEs: one value for two sources, otherwise the upper triangle "
                        "of the contrast matrix, row by row")
    m.add_argument("--ancestry-sd", type=float, default=None,
                   help="SD of global ancestry for the same-person term (default: the curve's, else 0.12)")
    m.add_argument("--relatedness", default=None,
                   help="optional: sum over ordered pairs of relatives of their squared relatedness in the "
                        "covariate-adjusted genotypes, or 'curve' to take it from the curve")
    m.add_argument("--nu", type=float, default=None, help="sample size minus covariates (default from the GWAS)")
    m.add_argument("--grid-cm", type=float, default=0.02, help="grid of the long-range convolution (default 0.02)")
    m.add_argument("--cell-matrix", action="store_true",
                   help="use the cohort's cell matrix (written by curve) beyond --cell-switch-cm and between chromosomes")
    m.add_argument("--cell-switch-cm", type=float, default=10.0)
    m.add_argument("--out", required=True)

    v = sub.add_parser("derive", help="a source without a reference panel, estimated from an admixed cohort that "
                                      "carries it (writes a reference with the known sources and the derived one)")
    v.add_argument("--geno", required=True, help="PLINK 1 BED or PLINK 2 PGEN prefix of the admixed cohort")
    v.add_argument("--ref", required=True, help="reference with the other sources")
    v.add_argument("--name", required=True, help="name of the derived source (e.g. NAT)")
    v.add_argument("--known", default=None, help="sources of --ref to keep (default: all)")
    v.add_argument("--panel", default=None,
                   help="optional small panel of the derived source, whose members anchor its frequencies")
    v.add_argument("--covar", default=None, help="the cohort's covariates (FID IID ...), including its PCs")
    v.add_argument("--keep", default=None)
    v.add_argument("--match", choices=["position", "id"], default="position")
    v.add_argument("--als-snps", type=int, default=20000, help="SNPs used to estimate ancestry (default 20000)")
    v.add_argument("--als-iter", type=int, default=500,
                   help="most iterations of the ancestry fit (default 500; it stops when the fit no longer improves)")
    v.add_argument("--ld", choices=["inversion", "panel"], default="inversion",
                   help="window LD of the derived source: by inverting the cohort's composition, or that of the anchor "
                        "panel (--panel) with its own moments; the frequencies come from the cohort either way")
    v.add_argument("--no-calibration", action="store_true",
                   help="keep the fit's position on the family of equivalent sources instead of placing it with the "
                        "anchor panel")
    v.add_argument("--threads", type=int, default=1)
    v.add_argument("--out", required=True, help="output reference directory")

    a = ap.parse_args(argv)
    log = Logger()
    if a.cmd not in ("reference", "derive"):
        log.attach_file(a.out + f".{a.cmd}.log", mode="w")
    else:
        import os
        os.makedirs(a.out, exist_ok=True)
        log.attach_file(os.path.join(a.out, f"{a.cmd}.log"), mode="w")
    log._log("summit admixed-ld " + " ".join(argv if argv is not None else sys.argv[2:]))
    if a.cmd == "reference":
        from .reference import build_reference
        names = _sources(a.names)
        chroms = None if a.chromosomes is None else _sources(a.chromosomes)
        build_reference(a.panel, names, a.out, window_kb=a.window_kb, genetic_map=a.genetic_map, match=a.match,
                        chromosomes=chroms, step=a.step, threads=a.threads, dtype=a.dtype, log=log._log)
    elif a.cmd == "derive":
        from .derive import derive_source
        derive_source(a.geno, a.ref, a.out, name=a.name, known=_sources(a.known), panel=a.panel, covar=a.covar,
                      keep=a.keep, match=a.match, als_snps=a.als_snps, als_iter=a.als_iter, ld=a.ld,
                      calibrate=not a.no_calibration, threads=a.threads, log=log._log)
    elif a.cmd == "curve":
        from .curve import genotype_curve
        genotype_curve(a.geno, a.ref, a.out, sources=_sources(a.sources), covar=a.covar, keep=a.keep, match=a.match,
                       cell_cm=a.cell_cm, max_cm=a.max_cm, relative_pairs=a.relative_pairs, kinship_col=a.kinship_col,
                       kinship_min=a.kinship_min, closure=a.closure, step=a.step, threads=a.threads,
                       noise_correction=not a.keep_panel_noise, log=log._log)
    else:
        from .compose import compose_ldscores
        compose_ldscores(a.ref, a.sumstats, a.annot, a.out, sources=_sources(a.sources), snps=a.snps, match=a.match,
                         curve=a.curve, generations=a.admixture_generations, decay_per_cm=a.decay_per_cm,
                         mean_ancestry=a.mean_ancestry, kappa0=a.kappa0, theta_sd=a.ancestry_sd,
                         relatedness=a.relatedness, nu=a.nu, grid_cm=a.grid_cm, cell_matrix=a.cell_matrix,
                         cell_switch_cm=a.cell_switch_cm, log=log._log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
