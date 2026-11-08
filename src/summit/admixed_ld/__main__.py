"""Command line for LD scores of admixed cohorts without in-sample LD (`summit admixed-ld ...`).

  summit admixed-ld reference --panel AFR_PREFIX --panel EUR_PREFIX --names AFR,EUR [--window-kb 3000]
                              [--genetic-map MAP] [--remove-panel-admixture] --out REF_DIR
  summit admixed-ld clean-panels --ref REF_DIR --panel AFR_PREFIX --panel EUR_PREFIX --names AFR,EUR --out NEW_DIR
  summit admixed-ld curve     --geno COHORT --ref REF_DIR [--covar FILE] [--keep FILE] [--sources AFR,EUR]
                              [--relative-pairs FILE [--kinship-col COL --kinship-min X] [--closure]
                               [--relatedness-cell-cm 0.5] [--relatedness-controls N]] [--background] --out PREFIX
  summit admixed-ld compose   --ref REF_DIR --sumstats FILE_OR_DIR [...] --annot ANNOT --out PREFIX
                              (--curve PREFIX | --admixture-generations T | --decay-per-cm LAMBDA)
                              [--sources AFR,EUR] [--ancestry-sd SD] [--relatedness X|curve|uniform] [--snps FILE]
                              [--background-window panels|zero|flat] [--background-min-cm 3]
  summit admixed-ld matrix    --ref REF_DIR --params PREFIX.admixed_ld.json (--region CHR:START-END | --snps FILE)
                              (--curve PREFIX | --admixture-generations T | --decay-per-cm LAMBDA)
                              [--taper auto|none|TAU_CM] [--no-admixture] --out PREFIX

`reference` is built once from unadmixed panels of the two source populations. Panels drawn from a biobank carry some
admixture of their own; --remove-panel-admixture (or `clean-panels` on an existing reference) measures it with the
panel's weighted-LD curve and removes it from the panel's covariances and variances; both also record each panel's own
background LD (cryptic relatedness and structure among its members), which compose removes inside the band when a
cohort's background (`curve --background`) is applied at every distance. `curve` is run by the cohort on its
genotypes and writes only aggregates (PREFIX.curve.tsv and PREFIX.curve.json). `compose` needs the GWAS summary
statistics (BETA, SE, N; CHR and BP for matching by position, or --snps), the annotation of the fit (e.g. MAF/LD bins),
and the decay of the local-ancestry covariance: the cohort's curve, a published admixture date in generations, or a
decay rate per cM. It writes PREFIX.gw.ldscore.gz and PREFIX.gw.M for `summit --h2/--rg`, and PREFIX.admixed_ld.json.
`matrix` writes the regional correlation matrix of the covariate-adjusted dosages (PREFIX.ld, PREFIX.ld.npz,
PREFIX.snps.tsv with the counted allele) for fine-mapping or conditional analysis, from the reference, compose's
parameters (mean ancestry and kappa(0)) and the same decay inputs; --taper auto tapers the panel term against its
sampling noise.
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
    r.add_argument("--remove-panel-admixture", action="store_true",
                   help="remove each panel's own admixture LD (measured with the weighted-LD curve of the panel) from "
                        "its covariances and variances. Optional and off by default. The target's curve and standard "
                        "errors already hold all of the target's admixture LD (members from the panel's own population "
                        "included), so a panel's own admixture LD in the band is counted twice whoever the target is; "
                        "use it when the panel's kernel passes the gate (z >= 3, reported by clean-panels) and its "
                        "duplicated share of the target's kernel is not negligible (see the documentation)")
    r.add_argument("--out", required=True)

    cp = sub.add_parser("clean-panels", help="remove the panels' own admixture LD from an existing reference (copy) "
                                             "and record each panel's own background LD")
    cp.add_argument("--ref", required=True, help="reference written by `summit admixed-ld reference`")
    cp.add_argument("--panel", action="append",
                    help="PLINK prefix of a panel the reference was built from, or the anchor panel of a derived "
                         "source with a panel's LD (`derive --ld panel`) (repeat, in the order of --names)")
    cp.add_argument("--names", help="comma-separated source names of the panels given")
    cp.add_argument("--from-fits",
                    help="instead of --panel: a reference cleaned from the same panels (directory or manifest.json); "
                         "its fitted panel kernels are applied to --ref, e.g. a subset or converted copy of that "
                         "reference where the panels' genotypes are not available")
    cp.add_argument("--match", choices=["position", "id"], default="position")
    cp.add_argument("--fit-cm", default="3,40", help="distance range of the panel curve fit, in cM (default 3,40)")
    cp.add_argument("--threads", type=int, default=1)
    cp.add_argument("--no-panel-background", action="store_true",
                    help="do not measure the panels' own background LD (the cross-person LD of their members: cryptic "
                         "relatedness and structure, which their unbiased squared covariances hold inside the band). "
                         "By default it is measured with `curve --background` on each panel and recorded in the "
                         "manifest (panel_background, aggregates only), which `compose --relatedness curve` needs to "
                         "apply a cohort's background at every distance; --from-fits copies it")
    cp.add_argument("--panel-background-snps", type=int, default=20000,
                    help="SNPs used between chromosomes for the panels' background (default 20000)")
    cp.add_argument("--out", required=True, help="output reference directory (a cleaned copy)")

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
    c.add_argument("--relatedness-cell-cm", type=float, default=0.5,
                   help="cells of the relatives' profile in cM (default 0.5): with --relative-pairs the curve also "
                        "writes PREFIX.relatedness.tsv, the listed pairs' local relatedness products by distance on "
                        "one chromosome and between chromosomes, net of random control pairs (aggregates only), for "
                        "the within-chromosome relatives term of `compose --relatedness curve`")
    c.add_argument("--relatedness-controls", type=int, default=None,
                   help="random unlisted pairs that measure what unrelated pairs contribute to the same products "
                        "(default: as many as the listed pairs, 20,000 to 200,000)")
    c.add_argument("--relatedness-memory-gb", type=float, default=4.0,
                   help="memory for the pairs' cell sums; more pairs take extra genotype passes (default 4)")
    c.add_argument("--no-relatedness-profile", action="store_true",
                   help="only the sum of g^2 of the listed pairs (the behaviour before 5 Nov 2026)")
    c.add_argument("--background", action="store_true",
                   help="also measure the relatedness LD of all pairs of people, listed or not (distant and cryptic "
                        "relatives), from the cross-person LD among the least differentiated SNPs, by distance and "
                        "between chromosomes (PREFIX.background.tsv; aggregates only)")
    c.add_argument("--background-quantile", type=float, default=0.2,
                   help="share of SNPs with the smallest source contrasts used for --background (default 0.2)")
    c.add_argument("--background-snps", type=int, default=None,
                   help="of these, SNPs used between chromosomes (default: what --relatedness-memory-gb holds in "
                        "float32, at most 40,000)")
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
                   help="SD of global ancestry for the same-person term, e.g. a published value (default: the curve's "
                        "moments, else the SD implied by the SEs' m and kappa_res(0) under random mating)")
    m.add_argument("--relatedness", default=None,
                   help="optional relatives' term: 'curve' for the curve's profile (B between chromosomes, phi(d) on "
                        "one chromosome, which carries the segments that relatives share identical by descent; a "
                        "curve made before 5 Nov 2026 gives the uniform term), 'uniform' for the uniform term from "
                        "the curve's sum of g^2, or a number: the sum over ordered pairs of relatives of their squared "
                        "relatedness in the covariate-adjusted genotypes, added uniformly to every SNP pair")
    m.add_argument("--relatedness-smooth", type=float, default=0.1,
                   help="width of the bins that pool the profile's lags, as a fraction of the distance (default 0.1)")
    m.add_argument("--background-min-cm", type=float, default=3.0,
                   help="with a curve made with --background: the distance from which the smooth form of the "
                        "cohort's and the panels' background LD is fitted (default 3; its rates reach 1 / this value "
                        "per cM); with --background-window zero or flat, the distance from which the measurement is "
                        "used")
    m.add_argument("--background-window", choices=["panels", "zero", "flat"], default="panels",
                   help="with a curve made with --background: 'panels' (the default) applies the cohort's all-pairs "
                        "background LD at every distance, inside the reference band too, and removes there the "
                        "panels' own background LD that their unbiased squared covariances hold (recorded by "
                        "clean-panels); 'zero' and 'flat' are the rules of 5 Nov 2026 (the measurement from "
                        "--background-min-cm on, below it the listed pairs' profile plus nothing or plus the unlisted "
                        "part's value there)")
    m.add_argument("--nu", type=float, default=None, help="sample size minus covariates (default from the GWAS)")
    m.add_argument("--grid-cm", type=float, default=0.02, help="grid of the long-range convolution (default 0.02)")
    m.add_argument("--cell-matrix", action="store_true",
                   help="use the cohort's cell matrix (written by curve) beyond --cell-switch-cm and between chromosomes")
    m.add_argument("--cell-switch-cm", type=float, default=10.0)
    m.add_argument("--allow-flat-kernel-join", action="store_true",
                   help="proceed when the curve at 2 cM is not below kappa_res(0) from the standard errors (the "
                        "kernel is then held at the curve's value below 2 cM and the flag is recorded); by default "
                        "this inconsistency is an error")
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

    x = sub.add_parser("matrix", help="regional LD matrix of the GWAS sample (fine-mapping, conditional analysis)")
    x.add_argument("--ref", required=True)
    x.add_argument("--params", default=None, help="compose's PREFIX.admixed_ld.json (mean ancestry and kappa(0))")
    rg = x.add_mutually_exclusive_group(required=True)
    rg.add_argument("--region", default=None, help="CHR:START-END in base pairs of the reference (inclusive)")
    rg.add_argument("--snps", default=None, help="file with SNP IDs (first column) on one chromosome")
    x.add_argument("--sources", default=None, help="as in compose (the last is the baseline)")
    gx = x.add_mutually_exclusive_group(required=True)
    gx.add_argument("--curve", default=None)
    gx.add_argument("--admixture-generations", type=float, default=None)
    gx.add_argument("--decay-per-cm", type=float, default=None)
    x.add_argument("--mean-ancestry", default=None, help="override: K-1 or K comma-separated values")
    x.add_argument("--kappa0", default=None, help="override: upper triangle of kappa_res(0), row by row")
    x.add_argument("--taper", default="auto",
                   help="taper of the panel term: auto (scale from the panel's sampling noise), none, or a scale in cM")
    x.add_argument("--no-admixture", action="store_true", help="leave out the admixture term (for comparison only)")
    x.add_argument("--allow-flat-kernel-join", action="store_true",
                   help="as in compose: proceed when the curve at 2 cM is not below kappa_res(0), with the flag recorded")
    x.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    log = Logger()
    if a.cmd not in ("reference", "derive", "clean-panels"):
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
        if a.remove_panel_admixture:
            from .panel_admixture import remove_panel_admixture
            remove_panel_admixture(a.out, dict(zip(names, a.panel)), match=a.match, threads=a.threads,
                                   log=log._log)
    elif a.cmd == "clean-panels" and a.from_fits:
        if a.panel:
            raise SystemExit("Give either --panel (with --names) or --from-fits, not both.")
        from .panel_admixture import apply_panel_admixture
        apply_panel_admixture(a.ref, a.from_fits, out=a.out, log=log._log)
    elif a.cmd == "clean-panels":
        from .panel_admixture import remove_panel_admixture
        if not a.panel or not a.names:
            raise SystemExit("clean-panels needs --panel and --names, or --from-fits.")
        names = _sources(a.names)
        if len(names) != len(a.panel):
            raise SystemExit("Give one --panel per name in --names.")
        lo, hi = (float(x) for x in a.fit_cm.split(","))
        remove_panel_admixture(a.ref, dict(zip(names, a.panel)), out=a.out, match=a.match, lo=lo, hi=hi,
                               threads=a.threads, background=not a.no_panel_background,
                               background_snps=a.panel_background_snps, log=log._log)
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
                       noise_correction=not a.keep_panel_noise, relatedness_profile=not a.no_relatedness_profile,
                       relatedness_cell_cm=a.relatedness_cell_cm, relatedness_controls=a.relatedness_controls,
                       relatedness_memory_gb=a.relatedness_memory_gb, background=a.background,
                       background_quantile=a.background_quantile, background_between_snps=a.background_snps,
                       log=log._log)
    elif a.cmd == "matrix":
        from .matrix import compose_matrix
        compose_matrix(a.ref, a.out, region=a.region, snps=a.snps, sources=_sources(a.sources), params=a.params,
                       curve=a.curve, generations=a.admixture_generations, decay_per_cm=a.decay_per_cm,
                       mean_ancestry=a.mean_ancestry, kappa0=a.kappa0, taper=a.taper, admixture=not a.no_admixture,
                       allow_flat_kernel_join=a.allow_flat_kernel_join, log=log._log)
    else:
        from .compose import compose_ldscores
        compose_ldscores(a.ref, a.sumstats, a.annot, a.out, sources=_sources(a.sources), snps=a.snps, match=a.match,
                         curve=a.curve, generations=a.admixture_generations, decay_per_cm=a.decay_per_cm,
                         mean_ancestry=a.mean_ancestry, kappa0=a.kappa0, theta_sd=a.ancestry_sd,
                         relatedness=a.relatedness, relatedness_smooth=a.relatedness_smooth,
                         background_min_cm=a.background_min_cm, background_window=a.background_window, nu=a.nu,
                         grid_cm=a.grid_cm, cell_matrix=a.cell_matrix,
                         cell_switch_cm=a.cell_switch_cm, allow_flat_kernel_join=a.allow_flat_kernel_join,
                         log=log._log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
