#!/usr/bin/env bash
# SUMMIT-MIX on a real admixed cohort: a template. Fill in the inputs below, then run the steps in order. Only step 2
# (the cohort's release, and the optional 2b) and the Tractor-style GWAS of step 6 read individual-level data; they run
# where the cohort's data are and write aggregates or summary statistics only. Steps 1 and 3-5 need no individual-level
# data. See the wiki pages "Admixed cohorts without in-sample LD" and "SUMMIT-MIX details".
#
# Times and memory were measured on 3,800 and 6,100 people at 405,807 array SNPs with two sources (UK Biobank). The
# curve's main pass grows with people x SNPs. Its background term grows with people and with the square of the least
# differentiated SNPs per chromosome, and dominates at biobank scale. Plan for hours and tens of GB at 10^5 people at
# HapMap3 density.
set -euo pipefail

# ------------------------------------------------------------------------------------------------ inputs (fill in)
PANELS=(/path/to/panels/afr /path/to/panels/eur)  # PLINK prefixes of unadmixed source panels, e.g. 1000 Genomes YRI, ESN,
                                                # GWD, MSL and CEU, GBR, TSI, IBS, FIN, in the cohort's genome build
NAMES=AFR,EUR                                   # source names, in the order of PANELS (the last is the baseline)
MAP=/path/to/maps/genetic_map_chr{chr}_combined_b38.txt   # genetic map of the same build ({chr} per chromosome)
COHORT=/path/to/cohort/qc                       # QC'd PLINK 1 BED or PLINK 2 PGEN prefix of the cohort (autosomes)
PEOPLE=/path/to/gwas_people.txt                 # FID IID of everyone who enters any GWAS of the cohort (one release)
GWAS_COV=/path/to/gwas_covariates.txt           # FID IID and the covariates of every GWAS, PCs included
PAIRS=/path/to/related_pairs.tsv                # columns IID1 IID2 of related pairs (e.g. relatedness > 0.0442 in a GRM
                                                # of PC-adjusted genotypes); keeps families in one half of the cell matrix
SUMSTATS=/path/to/sumstats                      # SUMMIT summary statistics of the GWAS (BETA SE N COV_RANK, CHR BP),
                                                # least-squares or linear-probability, one file per trait
DATE=6                                          # published admixture date in generations, for date mode (6 for African
                                                # American and African Caribbean groups, 11 for US Latino groups)
THREADS=16
OUT=/path/to/output

missing() { echo "$1 not found; fill in the inputs at the top of $0" >&2; exit 2; }
for f in "${PANELS[@]}" "$COHORT"; do [ -e "$f.bed" ] || [ -e "$f.pgen" ] || missing "$f"; done
for f in "$PEOPLE" "$GWAS_COV" "$PAIRS" "$SUMSTATS"; do [ -e "$f" ] || missing "$f"; done
mkdir -p "$OUT"
PANEL_ARGS=(); for p in "${PANELS[@]}"; do PANEL_ARGS+=(--panel "$p"); done

# ------------------------------------------------------------------------------------------------ 1. reference (anyone)
# Once per set of panels. About 18 minutes on 16 cores for 17,000 panel members at array density (1000 Genomes panels
# are much faster), a few GB of memory; the reference is about 4 GB for 400,000 SNPs. Add --drop-ambiguous to
# reference, release and compose if the panels and the cohort may be on different strands (the release stops when it
# sees it).
summit admixed-ld reference "${PANEL_ARGS[@]}" --names "$NAMES" --genetic-map "$MAP" --threads "$THREADS" --out "$OUT/ref"
# The panels' own admixture LD and background: about 15 minutes on 4 cores for the same panels, 2.7 GB.
summit admixed-ld clean-panels --ref "$OUT/ref" "${PANEL_ARGS[@]}" --names "$NAMES" --threads 4 --out "$OUT/ref_clean"

# ------------------------------------------------------------------------------------------------ 2. release (cohort)
# One run on everyone who enters any GWAS, with the covariates of every GWAS. 10-20 minutes on 8 cores for 3,800-6,100
# people at 405,807 SNPs, about 3 GB. Writes one folder of aggregates: the curve (.curve.tsv, .curve.json), the
# background term (.background.tsv), the cell matrix (.cells.npz), the sample statistic (.sample_factors.json), the
# eight MAF/LD bins and the two MAF classes (.bins8.annot.gz, .maf2.annot.gz), release.json and README.txt, which lists
# what to share and what never to share. release.json records the number of people and of related pairs (check them
# against the cohort's rules on small counts).
summit admixed-ld release --geno "$COHORT" --ref "$OUT/ref_clean" --keep "$PEOPLE" --covar "$GWAS_COV" \
  --relative-pairs "$PAIRS" --threads 8 --out "$OUT/release"
ANNOT=$OUT/release/cohort.bins8.annot.gz          # the eight MAF/LD bins (cohort.maf2.annot.gz: two MAF classes)

# 2b (optional): a trait measured in less than 85% of the release's people (compose flags it as
# gwas_sample_smaller_than_release). The cohort also releases the statistic of the trait's people, made with the same
# covariates, when the trait's people and the release's people without the trait both number at least 50
# (TRAIT_PEOPLE: FID IID of the trait's people):
# summit admixed-ld sample-factors --geno "$COHORT" --ref "$OUT/ref_clean" --keep "$TRAIT_PEOPLE" --covar "$GWAS_COV" \
#   --threads 8 --out "$OUT/release_extra/trait"
# summit admixed-ld compose --ref "$OUT/ref_clean" --release "$OUT/release" --sumstats "$SUMSTATS/trait.tsv" \
#   --sample-factors "$OUT/release_extra/trait" --annot "$ANNOT" --out "$OUT/ld_trait"

# ------------------------------------------------------------------------------------------------ 3. compose (anyone)
# About 6 minutes on one core and 3.3 GB at array density, half of it writing one file per GWAS sample size; linear in
# the SNP pairs within 3 Mb. Read the warnings in OUT/ld_curve.compose.log and the flags in OUT/ld_curve.admixed_ld.json
# (the wiki's table of flags says which are expected with one release).
summit admixed-ld compose --ref "$OUT/ref_clean" --release "$OUT/release" --sumstats "$SUMSTATS" --annot "$ANNOT" \
  --out "$OUT/ld_curve"
# Date mode, from the summary statistics and a published date alone (add --ancestry-sd with a published value if one
# exists; without the cohort's bins, use two MAF classes, e.g. summit admixed-ld bins --sumstats FILE --out PREFIX):
summit admixed-ld compose --ref "$OUT/ref_clean" --sumstats "$SUMSTATS" --annot "$ANNOT" \
  --admixture-generations "$DATE" --out "$OUT/ld_date"

# ------------------------------------------------------------------------------------------------ 4. h2 and rg (anyone)
# One LD-score file per GWAS sample size: OUT/ld_curve.n_groups.tsv lists each trait's file by name, next to it (no
# table when every trait has the same N). summit --h2 with --ldscores OUT/ld_curve reads the table and fits each trait
# with its own file. For rg of two GWAS of the cohort, keep A/T and C/G SNPs and give each GWAS's phenotype and
# covariate files where they are at hand (--pheno-rg, --pheno-rg-cov).
summit --h2 "$SUMSTATS" --ldscores "$OUT/ld_curve" --annot "$ANNOT" --num-threads 4 --out "$OUT/h2"
# summit --rg "$SUMSTATS/height.tsv,$SUMSTATS/bmi.tsv" --ldscores "$OUT/ld_curve" --annot "$ANNOT" \
#   --pheno-rg PHENO_HEIGHT,PHENO_BMI --pheno-rg-cov "$GWAS_COV,$GWAS_COV" --keep-ambiguous --out "$OUT/rg"

# ------------------------------------------------------------------------------------------------ 5. fine-mapping (anyone)
# Per region (seconds to a minute): the matrix restricted to the GWAS's SNPs, the z-scores on its counted alleles and the
# predicted error variance; then the two SuSiE fits (R with susieR), the second with the residual variance s^2.
# mkdir -p "$OUT/fm"
# summit admixed-ld matrix --ref "$OUT/ref_clean" --params "$OUT/ld_curve.admixed_ld.json" --curve "$OUT/release/cohort" \
#   --region 4:1000000-4000000 --sumstats "$SUMSTATS/height.tsv" --error-variance --out "$OUT/fm/region1"
# Rscript "$(dirname "$0")/finemap_two_stage.R" "$OUT/fm/region1"

# ------------------------------------------------------------------------------------------------ 6. track rg (optional)
# Needs phased genotypes and local-ancestry calls of the GWAS sample, and a Tractor-style GWAS with sandwich SEs from
# the covariates-only residuals (se_hcn_anc<s>); tractor-gwas runs where the genotypes are, the rest anywhere.
# summit admixed-ld tractor-gwas --pgen /path/to/phased_chr{chr} --lanc /path/to/lanc_chr{chr}.lanc --pheno PHENO \
#   --covar "$GWAS_COV" --keep "$PEOPLE" --sources AFR,EUR --threads 16 --out "$OUT/tractor/gwas"
# for t in height bmi ldl; do summit admixed-ld tractor-sumstats --tractor "$OUT/tractor/gwas.$t.tractor.tsv" \
#   --sources AFR,EUR --out "$OUT/tracks/$t"; done
# summit admixed-ld compose --ref "$OUT/ref_clean" --sumstats "$OUT/tracks/height.AFR.sumstats" --annot MAF5_ANNOT \
#   --curve "$OUT/release/cohort" --mean-ancestry M_AFR,M_EUR --ancestry-sd SD --tracks --out "$OUT/ld_tracks"
# summit admixed-ld track-rg --tracks "$OUT/ld_tracks" --sumstats "$OUT"/tracks/height "$OUT"/tracks/bmi \
#   "$OUT"/tracks/ldl --annot MAF5_ANNOT --out "$OUT/track_rg"
echo "SUMMIT-MIX template finished; outputs in $OUT"
