"""SUMMIT-MIX: LD scores of admixed GWAS samples without in-sample LD (`summit admixed-ld ...`).

  summit admixed-ld reference    --panel AFR_PREFIX --panel EUR_PREFIX --names AFR,EUR [--genetic-map MAP] --out REF
  summit admixed-ld clean-panels --ref REF --panel AFR_PREFIX --panel EUR_PREFIX --names AFR,EUR --out REF_CLEAN
  summit admixed-ld curve        --geno COHORT --ref REF_CLEAN --keep GWAS_SAMPLE --covar GWAS_COVARIATES
                                 --relative-pairs PAIRS --out CURVE                     (curve mode; aggregates only)
  summit admixed-ld compose      --ref REF_CLEAN --sumstats FILE_OR_DIR [...] --annot ANNOT --out PREFIX
                                 (--curve CURVE | --admixture-generations T)
  summit admixed-ld sample-factors --geno COHORT --ref REF_CLEAN --keep GWAS_SAMPLE --covar GWAS_COVARIATES --out SF
  summit admixed-ld derive       --geno COHORT --ref REF --name NAT --panel NAT_ANCHORS --out REF_WITH_NAT
  summit admixed-ld matrix       --ref REF_CLEAN --params PREFIX.admixed_ld.json --region CHR:START-END
                                 (--curve CURVE | --admixture-generations T) --out REGION
  summit admixed-ld matrix ... --sumstats GWAS --error-variance      then SuSiE twice (example/finemap_two_stage.R)
  summit admixed-ld finemap-variance --matrix REGION --posterior FIT1.tsv --out REGION
  summit admixed-ld tractor-gwas --pgen PHASED[_{chr}] --lanc LANC[_{chr}].lanc --pheno PHENO --covar COVAR
                                 --sources AFR,EUR --out GWAS        (writes GWAS.<trait>.tractor.tsv with se_hcn)
  summit admixed-ld tractor-sumstats --tractor TRACTOR.tsv --sources AFR,EUR --covariates C --out TRACKS
  summit admixed-ld track-rg     --tracks COMPOSE_PREFIX --sumstats TRACKS --annot ANNOT [--external EUR.sumstats]
                                 --out OUT

`reference` is built once from unadmixed panels of the source populations, and `clean-panels` removes the panels' own
admixture LD and records their own background, which compose removes inside the reference band. In curve mode (the
default and recommended) the cohort runs `curve` on the GWAS sample with the GWAS's covariates; it writes only
aggregates (PREFIX.curve.tsv, .curve.json, .background.tsv, .cells.npz). Date mode needs only the summary statistics
and a published admixture date. `compose` takes the summary statistics (BETA, SE, N, COV_RANK; CHR and BP for
matching by position, or --snps), whose standard errors give the mean ancestry and kappa_res(0), the annotation of the
fit, and the curve or the date; it writes PREFIX.gw.ldscore.gz and PREFIX.gw.M for `summit --h2/--rg` (one file per
GWAS sample size when the traits' N differ, listed in PREFIX.n_groups.tsv) and PREFIX.admixed_ld.json with every
parameter and flag. `sample-factors` is an optional aggregate for samples of a few hundred people, `derive` estimates a
source without a panel, `matrix` writes regional LD matrices for fine-mapping (aligned to PREFIX.snps.tsv's counted
allele), and `tractor-sumstats`, `compose --tracks` and `track-rg` estimate the correlation of per-allele effects
across local-ancestry tracks. The wiki page "Admixed cohorts without in-sample LD" describes each step.
"""
import argparse
import sys

from ..logger import Logger


def _sources(value):
    return None if value is None else [x.strip() for x in value.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="summit admixed-ld",
                                 description="SUMMIT-MIX: LD scores of admixed GWAS samples without in-sample LD")
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
    r.add_argument("--threads", "--num-threads", dest="threads", type=int, default=1,
                    help="threads for genotype decoding and matrix products; also the BLAS limit (default 1)")
    r.add_argument("--remove-panel-admixture", action="store_true",
                   help=argparse.SUPPRESS)
    r.add_argument("--drop-ambiguous", action="store_true",
                    help="leave out strand-ambiguous (A/T, C/G) SNPs, which are otherwise matched as listed; use it when the "
                    "data sets may be on different strands")
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
    cp.add_argument("--threads", "--num-threads", dest="threads", type=int, default=1,
                     help="threads for genotype decoding and matrix products; also the BLAS limit (default 1)")
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
                   help="related pairs (IID1 IID2 [FID1 FID2] [kinship]), e.g. relatedness above 0.0442 in a GRM of "
                        "PC-adjusted genotypes: each family is kept within one half of the cell matrix. Select "
                        "pairs with --kinship-col and --kinship-min")
    c.add_argument("--kinship-col", default=None, help="kinship column of --relative-pairs used with --kinship-min")
    c.add_argument("--kinship-min", type=float, default=None, help="keep only pairs whose kinship exceeds this value")
    c.add_argument("--closure", action="store_true",
                   help="also use every pair within a family (connected component of the listed pairs)")
    c.add_argument("--relatedness-cell-cm", type=float, default=0.5,
                   help=argparse.SUPPRESS)
    c.add_argument("--relatedness-controls", type=int, default=None,
                   help=argparse.SUPPRESS)
    c.add_argument("--relatedness-memory-gb", type=float, default=4.0,
                   help="memory for the background term between chromosomes and for the listed pairs' cell sums "
                        "(default 4)")
    c.add_argument("--no-relatedness-profile", action="store_true",
                   help=argparse.SUPPRESS)
    c.add_argument("--no-sample-factors", dest="sample_factors", action="store_false", default=True,
                   help="do not write the sample statistic (PREFIX.sample_factors.json: the covariance of a few "
                        "person-level factors and the realized same-person mass, accumulated in the curve's own pass "
                        "at O(N M) extra operations), which compose uses with this curve for the same-person term. "
                        "Without this option the curve stops when PREFIX.sample_factors.json exists and was not "
                        "written by a curve")
    c.add_argument("--sample-factors-ld-mass", action="store_true",
                   help="also record the realized in-sample LD mass in the sample statistic (exact, a GRM's cost)")
    c.add_argument("--sample-factors-probes", type=int, default=0,
                   help="record the realized in-sample LD mass estimated with this many random probes (O(N M B); the "
                        "total to about 1%% with a few hundred, the between-chromosome part much less precisely)")
    c.add_argument("--no-background", dest="background", action="store_false",
                   help="leave out the background term (by default it is measured; it is part of curve mode)")
    c.add_argument("--background", dest="background", action="store_true", default=True,
                   help="also measure the background term, the LD that all pairs of people add by sharing segments "
                        "identical by descent (relatives included), from the least differentiated SNPs, by distance "
                        "and between chromosomes (PREFIX.background.tsv; aggregates only). Part of curve mode")
    c.add_argument("--background-memory-gb", type=float, default=32.0,
                   help="memory for each chromosome's background SNPs (people x SNPs float64); above it a random "
                        "subset of them is used, which keeps the term unbiased (default 32)")
    c.add_argument("--relatedness-profile", dest="relatedness_profile_on", action="store_true",
                   help=argparse.SUPPRESS)
    c.add_argument("--background-quantile", type=float, default=0.2,
                   help="share of SNPs with the smallest source contrasts used for --background (default 0.2)")
    c.add_argument("--background-snps", type=int, default=None,
                   help="of these, SNPs used between chromosomes (default: what --relatedness-memory-gb holds in "
                        "float32, at most 40,000)")
    c.add_argument("--keep-panel-noise", "--no-noise-correction", dest="keep_panel_noise", action="store_true",
                   help=argparse.SUPPRESS)
    c.add_argument("--step", type=int, default=4096)
    c.add_argument("--threads", "--num-threads", dest="threads", type=int, default=1,
                    help="threads for genotype decoding and matrix products; also the BLAS limit (default 1)")
    c.add_argument("--drop-ambiguous", action="store_true",
                    help="leave out strand-ambiguous (A/T, C/G) SNPs, which are otherwise matched as listed; use it when the "
                    "data sets may be on different strands")
    c.add_argument("--out", required=True)

    m = sub.add_parser("compose", help="LD scores from the resource, the GWAS and the decay of local-ancestry covariance")
    m.add_argument("--ref", required=True)
    m.add_argument("--sumstats", nargs="+", required=True,
                   help="SUMMIT summary statistics: files, or a directory whose every file (names not starting with a "
                        "dot) is read, as summit --h2 DIR reads it; their SEs give mean ancestry and kappa(0)")
    m.add_argument("--annot", required=True, help="annotation of the fit: SNP and one 0/1 column per class")
    m.add_argument("--snps", default=None,
                   help="variant table of the GWAS (BIM, PVAR, or CHR BP SNP A1 A2) if the summary statistics lack "
                        "CHR and BP")
    m.add_argument("--sources", default=None,
                   help="two or more source names (default: the reference's first two); the last is the baseline of "
                        "the contrasts")
    m.add_argument("--match", choices=["position", "id"], default="position")
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--curve", default=None, help="prefix of the cohort's curve (`summit admixed-ld curve`): curve mode")
    g.add_argument("--admixture-generations", type=float, default=None,
                   help="generations since admixture, a published date: date mode (decay generations / 100 per cM)")
    g.add_argument("--decay-per-cm", type=float, default=None, help=argparse.SUPPRESS)
    m.add_argument("--mean-ancestry", default=None,
                   help="override the mean ancestry from the SEs: K-1 or K comma-separated values in source order")
    m.add_argument("--kappa0", default=None,
                   help="override kappa_res(0) from the SEs: one value for two sources, otherwise the upper triangle "
                        "of the contrast matrix, row by row")
    m.add_argument("--ancestry-sd", type=float, default=None,
                   help="SD of global ancestry for the same-person term, a published value (default: the curve's "
                        "moments in curve mode, else the SD implied by the SEs' m and kappa_res(0) under random "
                        "mating)")
    m.add_argument("--relatedness", default="auto",
                   help="the background term: by default (with --curve) the cohort's background at every distance "
                        "from the curve; 'none' leaves it out. The curve's sums run over its own people and are "
                        "put on the GWAS's sample by N (N - 1) / (n (n - 1)), with a recorded flag when the two "
                        "differ by more than 1%%")
    m.add_argument("--relatedness-smooth", type=float, default=0.1,
                   help=argparse.SUPPRESS)
    m.add_argument("--background-min-cm", type=float, default=3.0,
                   help=argparse.SUPPRESS)
    m.add_argument("--background-window", choices=["panels", "zero", "flat"], default="panels",
                   help=argparse.SUPPRESS)
    m.add_argument("--nu", type=float, default=None, help="sample size minus covariates (default from the GWAS)")
    m.add_argument("--cov-rank", type=int, default=None,
                   help="the number of covariates of the GWAS, principal components included, for summary statistics "
                        "without a COV_RANK column (default: the curve's covariate rank, recorded with the flag "
                        "cov_rank_from_curve; in date mode 0, with the flag cov_rank_missing)")
    m.add_argument("--grid-cm", type=float, default=0.02, help=argparse.SUPPRESS)
    m.add_argument("--no-cell-matrix", dest="cell_matrix", action="store_const", const=False, default="auto",
                   help="leave out the cohort's cell matrix (used by default with --curve)")
    m.add_argument("--cell-matrix", dest="cell_matrix", action="store_const", const=True, default="auto",
                   help="use the cohort's cell matrix (written by curve) for pairs of 2-cM cells at least 10 cM apart "
                        "and between chromosomes (the default with --curve)")
    m.add_argument("--cell-switch-cm", type=float, default=10.0, help=argparse.SUPPRESS)
    m.add_argument("--allow-flat-kernel-join", action="store_true",
                   help="proceed when the curve at 2 cM is not below kappa_res(0) from the standard errors (the "
                        "kernel is then held at the curve's value below 2 cM and the flag is recorded); by default "
                        "this inconsistency is an error")
    m.add_argument("--cross-source", default=None,
                   help="also write PREFIX.cross_NAME.gw.ldscore.gz, the cross-population LD scores of this GWAS with "
                        "a GWAS of other people from an unadmixed population whose LD is that of source NAME's panel "
                        "in the reference (e.g. EUR): sum_k rho_1 rho_2 sqrt(V_1 V_2) within the window, for "
                        "cross-population genetic correlation")
    m.add_argument("--sample-factors", default=None,
                   help="a sample statistic made with `summit admixed-ld sample-factors` on one GWAS's people with the "
                        "GWAS covariates, for example a trait measured in part of the cohort: the same-person term "
                        "from it in place of the curve's own statistic")
    m.add_argument("--calibrate-between-chromosomes", action="store_true",
                   help="with sample factors that carry the cohort's realized LD mass: put the difference between it and "
                        "the composed mass on pairs of SNPs on different chromosomes, uniformly per pair (for structure "
                        "the sources do not represent); by default the difference is only checked and recorded. Skipped "
                        "with the flag between_chromosome_shift_skipped when the statistic has no realized LD mass "
                        "(the curve's default) or a stochastic one whose total has an SE above 10%% of the "
                        "between-chromosome mass")
    m.add_argument("--single-n", action="store_true",
                   help="write only the LD-score file for the traits' median N (by default, traits whose N differ by more "
                        "than 1%% get one file per sample size, listed in PREFIX.n_groups.tsv: the same-person and "
                        "background terms belong to each GWAS's sample)")
    m.add_argument("--n-tol", type=float, default=0.01, help=argparse.SUPPRESS)
    m.add_argument("--max-annotation-atoms", type=int, default=64, help=argparse.SUPPRESS)
    m.add_argument("--no-sample-factors", dest="release_sample_factors", action="store_false", default=True,
                   help="compose the same-person term under random mating even when the curve's pass wrote a sample "
                        "statistic next to it (PREFIX.sample_factors.json, listed in PREFIX.curve.json: used by "
                        "default, and the composition stops when the listed file is missing or not of the curve's "
                        "people and covariates)")
    m.add_argument("--dry-run", action="store_true",
                   help="write the plan (which decay, same-person term, relatives, cell matrix and sample sizes the "
                        "composition would use) to PREFIX.plan.json and stop before composing")
    m.add_argument("--require", default=None,
                   help="comma-separated components the composition must use (curve, statistic, cells, relatives, "
                        "background); stop with an error if one would be missing, e.g. in a pipeline")
    m.add_argument("--min-release-share", type=float, default=0.85,
                   help="flag traits whose GWAS N is below this share of the curve's people (default 0.85): compose "
                        "them alone with the sample statistic of their own people (with --sample-factors the share is "
                        "of that statistic's people)")
    m.add_argument("--sample-factors-no-calibration", action="store_true",
                   help=argparse.SUPPRESS)
    # development option: summit --h2/--rg refuse its LD scores (see compose_ldscores)
    m.add_argument("--partners", choices=["rows", "reference"], default="rows", help=argparse.SUPPRESS)
    m.add_argument("--tracks", action="store_true",
                   help="also write the LD scores of the local-ancestry tracks of a Tractor-style GWAS (own score of "
                        "each track, the cross score of each pair of tracks, and with --cross-source the tracks against "
                        "that population's GWAS), on the per-allele scale, for `summit admixed-ld track-rg`")
    m.add_argument("--track-alpha", type=float, default=None,
                   help=argparse.SUPPRESS)
    m.add_argument("--drop-ambiguous", action="store_true",
                    help="leave out strand-ambiguous (A/T, C/G) SNPs, which are otherwise matched as listed; use it when the "
                    "data sets may be on different strands")
    m.add_argument("--out", required=True)

    f = sub.add_parser("sample-factors", help="the cohort's person-factor statistic for the same-person term "
                                              "(aggregates only)")
    f.add_argument("--geno", required=True, help="PLINK 1 BED or PLINK 2 PGEN prefix of the GWAS sample")
    f.add_argument("--ref", required=True, help="resource written by `summit admixed-ld reference`")
    f.add_argument("--covar", default=None, help="the GWAS covariates (FID IID ...), as for the LD-score step")
    f.add_argument("--keep", default=None, help="samples to use (FID IID or IID per line)")
    f.add_argument("--sources", default=None,
                   help="two or more source names of the reference (default: its first two); the last is the baseline")
    f.add_argument("--match", choices=["position", "id"], default="position")
    f.add_argument("--threads", "--num-threads", dest="threads", type=int, default=1,
                    help="threads for genotype decoding and matrix products; also the BLAS limit (default 1)")
    f.add_argument("--no-ld-mass", action="store_true",
                   help="skip the realized in-sample LD mass (two n x n matrices, O(n^2 M))")
    f.add_argument("--memory-gb", type=float, default=32.0,
                   help="largest memory for the two n x n matrices of the LD mass; above it the mass is skipped")
    f.add_argument("--ld-mass-probes", type=int, default=0,
                   help="estimate the realized LD mass with this many random probes, at O(n M B) instead of a GRM's "
                        "O(n^2 M) (each Gram matrix's diagonal is exact; a few hundred probes give the total to about "
                        "1%%, the between-chromosome part, a small difference of large sums, much less precisely; both "
                        "standard errors are recorded). Default 0: exact")
    f.add_argument("--seed", type=int, default=0, help="seed of the probes (--ld-mass-probes)")
    f.add_argument("--drop-ambiguous", action="store_true",
                    help="leave out strand-ambiguous (A/T, C/G) SNPs, which are otherwise matched as listed; use it when the "
                    "data sets may be on different strands")
    f.add_argument("--out", required=True)

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
                   help=argparse.SUPPRESS)
    v.add_argument("--threads", "--num-threads", dest="threads", type=int, default=1,
                    help="threads for genotype decoding and matrix products; also the BLAS limit (default 1)")
    v.add_argument("--drop-ambiguous", action="store_true",
                    help="leave out strand-ambiguous (A/T, C/G) SNPs, which are otherwise matched as listed; use it when the "
                    "data sets may be on different strands")
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
    gx.add_argument("--decay-per-cm", type=float, default=None, help=argparse.SUPPRESS)
    x.add_argument("--mean-ancestry", default=None, help="override: K-1 or K comma-separated values")
    x.add_argument("--kappa0", default=None, help="override: upper triangle of kappa_res(0), row by row")
    x.add_argument("--taper", default="auto",
                   help="taper of the panel term: auto (scale from the panel's sampling noise), none, or a scale in cM")
    x.add_argument("--no-admixture", action="store_true", help=argparse.SUPPRESS)
    x.add_argument("--sumstats", default=None,
                   help="the GWAS summary statistics: the matrix keeps the region's SNPs in them, and PREFIX.z.tsv holds "
                        "their z-scores on the matrix's counted allele (strand-ambiguous SNPs left out unless "
                        "--keep-ambiguous)")
    x.add_argument("--keep-ambiguous", action="store_true",
                   help="with --sumstats: keep A/T and C/G SNPs, aligned as listed")
    x.add_argument("--error-variance", action="store_true",
                   help="also write PREFIX.ld_var.npz and PREFIX.ld_var: the predicted variance of the error of every "
                        "entry of the matrix, for the residual variance of the second SuSiE fit (finemap-variance)")
    x.add_argument("--n", type=float, default=None,
                   help="the GWAS sample size for --error-variance (default: the median N of --sumstats)")
    x.add_argument("--allow-flat-kernel-join", action="store_true",
                   help="as in compose: proceed when the curve at 2 cM is not below kappa_res(0), with the flag recorded")
    x.add_argument("--out", required=True)

    fv = sub.add_parser("finemap-variance", help="residual variance s^2 of the second SuSiE fit from a matrix made with "
                                                 "--error-variance and the first fit's posterior")
    fv.add_argument("--matrix", required=True, help="prefix of `matrix --error-variance`")
    fv.add_argument("--posterior", required=True,
                    help="the first fit's posterior second moments per SNP on the z scale (columns SNP and EB2; "
                         "susieR: colSums(alpha * mu2) / X_column_scale_factors^2)")
    fv.add_argument("--out", required=True, help="output prefix (PREFIX.s2.json)")
    tg = sub.add_parser("tractor-gwas", help="Tractor-style GWAS from phased genotypes and local ancestry, with the "
                                             "sandwich SEs that tractor-sumstats needs (se_hcn_anc<s>)")
    tg.add_argument("--pgen", required=True,
                    help="phased PLINK 2 prefix (.pgen/.pvar/.psam; biallelic, no missing calls), or a path containing "
                         "{chr} with one PGEN per chromosome")
    tg.add_argument("--lanc", required=True,
                    help="admix-kit .lanc local ancestry aligned to the PGEN's variants and samples (codes 0 and 1), "
                         "or a path containing {chr}")
    tg.add_argument("--pheno", required=True, help="FID IID and one column per trait (NA for missing values)")
    tg.add_argument("--covar", required=True,
                    help="FID IID and the covariates, PCs included; each trait uses the people with the trait and "
                         "every covariate")
    tg.add_argument("--keep", default=None, help="samples to use (FID IID or IID per line)")
    tg.add_argument("--sources", default=None,
                    help="names of the .lanc's ancestry codes 0 and 1, Tractor's anc0 and anc1 (e.g. AFR,EUR when code "
                         "0 is African), for the record and tractor-sumstats --sources (default anc0,anc1)")
    tg.add_argument("--chromosomes", default=None,
                    help="with {chr}: comma-separated chromosomes (default: those of 1-22 whose PGEN exists)")
    tg.add_argument("--hc1", action="store_true",
                    help="also write the HC1 sandwich from the full model's residuals (se_hc_anc<s>), which is biased "
                         "low at alleles rare on a track")
    tg.add_argument("--min-cases", type=int, default=50,
                    help="fewest complete cases a trait may have (default 50); a trait with fewer is an error")
    tg.add_argument("--step", type=int, default=None,
                    help="SNPs per block (default: 2 million people x SNPs per block, at most 2048; about 64 bytes "
                         "each per thread)")
    tg.add_argument("--threads", "--num-threads", dest="threads", type=int, default=1,
                    help="blocks fitted in parallel, each with one BLAS thread (default 1)")
    tg.add_argument("--out", required=True,
                    help="output prefix: OUT.<trait>.tractor.tsv per trait and OUT.tractor_gwas.json")

    ts = sub.add_parser("tractor-sumstats", help="split a Tractor GWAS into one summary-statistic file per "
                                                 "local-ancestry track (sandwich SEs required)")
    ts.add_argument("--tractor", required=True, help="Tractor output table (CHR POS ID REF ALT N, per ancestry s: "
                                                     "beta_anc{s}, se_anc{s} or tval_anc{s}, AF_anc{s}, "
                                                     "LAprop_anc{s}, and the sandwich SEs se_hcn_anc{s} from the "
                                                     "covariates-only residuals, or se_hc_anc{s} with a flag)")
    ts.add_argument("--sources", required=True,
                    help="comma-separated names of the ancestries anc0, anc1, ... as in the reference (e.g. AFR,EUR)")
    ts.add_argument("--se", choices=["hc", "ols"], default="hc",
                    help="standard errors used: 'hc' (sandwich, the default and required for the track moments; "
                         "se_hcn_anc<s> when present) or 'ols' (Tractor's own; needs --allow-ols)")
    ts.add_argument("--allow-ols", action="store_true",
                    help="accept OLS SEs, with the flag recorded (their own-track moments hold same-person terms that "
                         "the track LD scores do not model)")
    ts.add_argument("--covariates", type=int, default=None,
                    help="number of covariates of the Tractor model (for COV_RANK = covariates + 2 (K - 1), unless the "
                         "table has a COV_RANK column)")
    ts.add_argument("--out", required=True)

    tr = sub.add_parser("track-rg", help="correlation of per-allele causal effects across local-ancestry tracks (Hou "
                                         "et al.'s r_admix), and of each track with another population's GWAS")
    tr.add_argument("--tracks", required=True, help="prefix of `compose --tracks` (its .admixed_ld.json)")
    tr.add_argument("--sumstats", required=True, nargs="+",
                    help="the tractor-sumstats prefix, or the track files comma-separated in compose's source order; "
                         "several traits' prefixes pool r over the traits (ratio of their summed moments, one "
                         "jackknife)")
    tr.add_argument("--merge-ld-halves", action="store_true",
                    help="with several traits and classes named <maf class>_<half>: merge a MAF class's two LD halves "
                         "when any of its pooled own-track moments has a jackknife z below --merge-z")
    tr.add_argument("--merge-z", type=float, default=2.0, help="z threshold of --merge-ld-halves (default 2)")
    tr.add_argument("--annot", required=True, help="the annotation of the fit (as given to compose)")
    tr.add_argument("--external", default=None,
                    help="summary statistics of a GWAS of other people from the population of compose's "
                         "--cross-source (e.g. a European cohort): the triangle r(track, cohort) for every track")
    tr.add_argument("--cohort-track", default=None,
                    help="with --external: the track of the external cohort's ancestry, for the triangle check "
                         "r(other track, cohort) = r(tracks) r(this track, cohort) (default: the --cross-source name "
                         "when it is one of the sources)")
    tr.add_argument("--njack", default="200", help="jackknife blocks (default 200 contiguous blocks)")
    tr.add_argument("--group", action="append", default=None,
                    help="NAME=CLASS1,CLASS2,...: also report r over a group of the annotation's classes (repeat; "
                         "e.g. two MAF classes of the eight MAF/LD bins)")
    tr.add_argument("--max-chisq", default=None, help="chi-square threshold of the fits (default: none)")
    tr.add_argument("--keep-ambiguous", action="store_true",
                    help="keep strand-ambiguous SNPs in the fits against --external (the tracks of one Tractor run are "
                         "always aligned)")
    tr.add_argument("--allow-ols", action="store_true", help="accept track summary statistics with OLS SEs (flagged)")
    tr.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    log = Logger()
    if a.cmd not in ("reference", "derive", "clean-panels"):
        log.attach_file(a.out + f".{a.cmd}.log", mode="w")
    else:
        import os
        os.makedirs(a.out, exist_ok=True)
        log.attach_file(os.path.join(a.out, f"{a.cmd}.log"), mode="w")
    log._log("summit admixed-ld " + " ".join(argv if argv is not None else sys.argv[2:]))
    # BLAS threads follow --threads, as in SUMMIT's LD-score commands, so that plain numpy products do not take every
    # core of a shared machine; commands without --threads leave the BLAS setting as it is
    threads = getattr(a, "threads", None)
    try:
        from threadpoolctl import threadpool_limits
    except Exception:                                   # pragma: no cover - threadpoolctl is a SUMMIT dependency
        threadpool_limits = None
    if threads is not None and threadpool_limits is not None:
        with threadpool_limits(limits=max(1, int(threads)), user_api="blas"):
            return _run(a, log)
    return _run(a, log)


def _run(a, log):
    if a.cmd == "reference":
        from .reference import build_reference
        names = _sources(a.names)
        chroms = None if a.chromosomes is None else _sources(a.chromosomes)
        build_reference(a.panel, names, a.out, window_kb=a.window_kb, genetic_map=a.genetic_map, match=a.match,
                        chromosomes=chroms, step=a.step, threads=a.threads, dtype=a.dtype,
                        drop_ambiguous=a.drop_ambiguous, log=log._log)
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
    elif a.cmd == "sample-factors":
        from .sample_factors import sample_factors
        sample_factors(a.geno, a.ref, a.out, sources=_sources(a.sources), covar=a.covar, keep=a.keep, match=a.match,
                       threads=a.threads, ld_mass=not a.no_ld_mass, memory_gb=a.memory_gb,
                       ld_mass_probes=a.ld_mass_probes, seed=a.seed, drop_ambiguous=a.drop_ambiguous, log=log._log)
    elif a.cmd == "derive":
        from .derive import derive_source
        derive_source(a.geno, a.ref, a.out, name=a.name, known=_sources(a.known), panel=a.panel, covar=a.covar,
                      keep=a.keep, match=a.match, als_snps=a.als_snps, als_iter=a.als_iter, ld=a.ld,
                      calibrate=not a.no_calibration, threads=a.threads, drop_ambiguous=a.drop_ambiguous,
                      log=log._log)
    elif a.cmd == "curve":
        from .curve import genotype_curve
        genotype_curve(a.geno, a.ref, a.out, sources=_sources(a.sources), covar=a.covar, keep=a.keep, match=a.match,
                       cell_cm=a.cell_cm, max_cm=a.max_cm, relative_pairs=a.relative_pairs, kinship_col=a.kinship_col,
                       kinship_min=a.kinship_min, closure=a.closure, step=a.step, threads=a.threads,
                       noise_correction=not a.keep_panel_noise,
                       relatedness_profile=(False if a.no_relatedness_profile else True if a.relatedness_profile_on
                                            else None),
                       background_memory_gb=a.background_memory_gb,
                       relatedness_cell_cm=a.relatedness_cell_cm, relatedness_controls=a.relatedness_controls,
                       relatedness_memory_gb=a.relatedness_memory_gb, background=a.background,
                       background_quantile=a.background_quantile, background_between_snps=a.background_snps,
                       drop_ambiguous=a.drop_ambiguous, sample_factors=a.sample_factors,
                       sample_factors_ld_mass=a.sample_factors_ld_mass, sample_factors_probes=a.sample_factors_probes,
                       log=log._log)
    elif a.cmd == "tractor-gwas":
        from .tractor_gwas import tractor_gwas
        tractor_gwas(a.pgen, a.lanc, a.pheno, a.covar, a.out, keep=a.keep, sources=_sources(a.sources),
                     chromosomes=_sources(a.chromosomes), hc1=a.hc1, min_cases=a.min_cases, step=a.step,
                     threads=a.threads, log=log._log)
    elif a.cmd == "tractor-sumstats":
        from .tracks import tractor_sumstats
        tractor_sumstats(a.tractor, a.out, sources=_sources(a.sources), se=a.se, allow_ols=a.allow_ols,
                         n_covariates=a.covariates, log=log._log)
    elif a.cmd == "track-rg":
        from .tracks import track_rg
        njack = int(a.njack) if str(a.njack).isdigit() else a.njack
        track_rg(a.tracks, a.sumstats, a.annot, a.out, external=a.external, cohort_track=a.cohort_track,
                 njack=njack, groups=a.group,
                 allow_ols=a.allow_ols, max_chisq=a.max_chisq, keep_ambiguous=a.keep_ambiguous,
                 merge_ld_halves=a.merge_ld_halves, merge_z=a.merge_z, log=log._log)
    elif a.cmd == "matrix":
        from .matrix import compose_matrix
        compose_matrix(a.ref, a.out, region=a.region, snps=a.snps, sources=_sources(a.sources), params=a.params,
                       curve=a.curve, generations=a.admixture_generations, decay_per_cm=a.decay_per_cm,
                       mean_ancestry=a.mean_ancestry, kappa0=a.kappa0, taper=a.taper, admixture=not a.no_admixture,
                       allow_flat_kernel_join=a.allow_flat_kernel_join, sumstats=a.sumstats,
                       keep_ambiguous=a.keep_ambiguous, error_variance=a.error_variance, n=a.n, log=log._log)
    elif a.cmd == "finemap-variance":
        from .finemap import finemap_variance
        finemap_variance(a.matrix, a.posterior, out=a.out + ".s2.json", log=log._log)
    else:
        from .compose import compose_ldscores
        compose_ldscores(a.ref, a.sumstats, a.annot, a.out, sources=_sources(a.sources), snps=a.snps, match=a.match,
                         curve=a.curve, generations=a.admixture_generations, decay_per_cm=a.decay_per_cm,
                         mean_ancestry=a.mean_ancestry, kappa0=a.kappa0, theta_sd=a.ancestry_sd,
                         relatedness=a.relatedness, relatedness_smooth=a.relatedness_smooth,
                         background_min_cm=a.background_min_cm, background_window=a.background_window, nu=a.nu,
                         grid_cm=a.grid_cm, cell_matrix=a.cell_matrix,
                         cell_switch_cm=a.cell_switch_cm, allow_flat_kernel_join=a.allow_flat_kernel_join,
                         cross_source=a.cross_source, sample_factors=a.sample_factors,
                         sample_factors_calibrate=not a.sample_factors_no_calibration,
                         per_trait_n=not a.single_n, n_tol=a.n_tol,
                         calibrate_between_chromosomes=a.calibrate_between_chromosomes, tracks=a.tracks,
                         track_alpha=a.track_alpha, partners=a.partners, drop_ambiguous=a.drop_ambiguous,
                         release_sample_factors=a.release_sample_factors, min_release_share=a.min_release_share,
                         dry_run=a.dry_run, require=a.require, max_annotation_atoms=a.max_annotation_atoms,
                         cov_rank=a.cov_rank, log=log._log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
