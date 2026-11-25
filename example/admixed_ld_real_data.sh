#!/usr/bin/env bash
# SUMMIT-MIX on a real admixed cohort: a template. Fill in the inputs below, then run the steps in order. Steps 2-3
# (and the optional 2b) read individual-level genotypes and run where the cohort's data are; they write aggregates
# only. Steps 1 and 4-6 need no individual-level data. See the wiki page "Admixed cohorts without in-sample LD".
#
# Times and memory were measured on 3,787 and 6,111 people at 405,807 array SNPs with two sources (UK Biobank). The
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
GWAS_SAMPLE=/path/to/gwas_sample.txt            # FID IID of the GWAS's people (one file per GWAS sample)
GWAS_COV=/path/to/gwas_covariates.txt           # FID IID and exactly the GWAS's covariates, PCs included
PAIRS=/path/to/related_pairs.tsv                # IID1 IID2 of related pairs (e.g. relatedness > 0.0442 in a GRM of
                                                # PC-adjusted genotypes); keeps families in one half of the cell matrix
SUMSTATS=/path/to/sumstats                      # SUMMIT summary statistics of the GWAS (BETA SE N COV_RANK, CHR BP),
                                                # least-squares or linear-probability, one file per trait
ANNOT=/path/to/annot.tsv.gz                     # annotation of the fit (e.g. eight MAF/LD bins)
DATE=6                                          # published admixture date in generations, for date mode (6 for African
                                                # American and African Caribbean groups, 11 for US Latino groups)
THREADS=16
OUT=/path/to/output

missing() { echo "$1 not found; fill in the inputs at the top of $0" >&2; exit 2; }
for f in "${PANELS[@]}" "$COHORT"; do [ -e "$f.bed" ] || [ -e "$f.pgen" ] || missing "$f"; done
for f in "$GWAS_SAMPLE" "$GWAS_COV" "$PAIRS" "$SUMSTATS" "$ANNOT"; do [ -e "$f" ] || missing "$f"; done
mkdir -p "$OUT"
PANEL_ARGS=(); for p in "${PANELS[@]}"; do PANEL_ARGS+=(--panel "$p"); done

# ------------------------------------------------------------------------------------------------ 1. reference (anyone)
# Once per set of panels. About 18 minutes on 16 cores for 17,000 panel members at array density (1000 Genomes panels
# are much faster), a few GB of memory; the resource is about 4 GB for 400,000 SNPs. Add --drop-ambiguous to every step
# if the panels and the cohort may be on different strands (the curve stops when it sees it).
summit admixed-ld reference "${PANEL_ARGS[@]}" --names "$NAMES" --genetic-map "$MAP" --threads "$THREADS" --out "$OUT/ref"
# The panels' own admixture LD and background: about 15 minutes on 4 cores for the same panels, 3 GB.
summit admixed-ld clean-panels --ref "$OUT/ref" "${PANEL_ARGS[@]}" --names "$NAMES" --threads 4 --out "$OUT/ref_clean"

# ------------------------------------------------------------------------------------------------ 2. curve (cohort)
# Curve mode. Run where the genotypes are, on the GWAS's people with the GWAS's covariates. 9-16 minutes on 8 cores for
# 3,800-6,100 people at 405,807 SNPs, 3 GB. Writes aggregates only (may leave the controlled environment):
# curve.curve.tsv, .curve.json, .background.tsv, .cells.npz, .sample_factors.json (the sample statistic, which compose
# uses for the same-person term) and, with related pairs, .relatedness.tsv and .relatedness_chr.npz. The jsons record
# the number of people and of related pairs (check them against the cohort's rules on small counts).
summit admixed-ld curve --geno "$COHORT" --ref "$OUT/ref_clean" --keep "$GWAS_SAMPLE" --covar "$GWAS_COV" \
  --relative-pairs "$PAIRS" --threads 8 --out "$OUT/curve"

# 2b (optional): a trait measured in only part of the cohort. compose flags a GWAS N below 85% of the curve's people
# (gwas_sample_smaller_than_release); compose such a trait alone with the sample statistic of its own people and the
# GWAS covariates (aggregates only; by default with its exact realized LD mass, the cost of a GRM, which --no-ld-mass
# skips), under another --out than the curve's (TRAIT_SAMPLE: FID IID of the trait's people):
# summit admixed-ld sample-factors --geno "$COHORT" --ref "$OUT/ref_clean" --keep "$TRAIT_SAMPLE" --covar "$GWAS_COV" \
#   --threads 8 --out "$OUT/trait_sf"
# summit admixed-ld compose --ref "$OUT/ref_clean" --sumstats "$SUMSTATS/trait.tsv" --annot "$ANNOT" \
#   --curve "$OUT/curve" --sample-factors "$OUT/trait_sf" --out "$OUT/ld_trait"

# ------------------------------------------------------------------------------------------------ 3. compose (anyone)
# About 6 minutes on one core and 3 GB at array density, half of it writing one file per GWAS sample size; linear in
# the SNP pairs within 3 Mb. Read the warnings and the flags in each .admixed_ld.json.
summit admixed-ld compose --ref "$OUT/ref_clean" --sumstats "$SUMSTATS" --annot "$ANNOT" --curve "$OUT/curve" \
  --out "$OUT/ld_curve"
# Date mode, from the summary statistics and a published date alone (add --ancestry-sd with a published value if one
# exists):
summit admixed-ld compose --ref "$OUT/ref_clean" --sumstats "$SUMSTATS" --annot "$ANNOT" \
  --admixture-generations "$DATE" --out "$OUT/ld_date"

# ------------------------------------------------------------------------------------------------ 4. h2 and rg (anyone)
# One LD-score file per GWAS sample size: OUT/ld_curve.n_groups.tsv lists each trait's file by name, next to it
# (no table when every trait has the same N). summit --h2 with --ldscores OUT/ld_curve.gw.ldscore.gz reads the table and
# fits each trait with its own file (a trait the table does not list stops the run). For rg, give each GWAS's phenotype
# and covariate files (--pheno-rg, --pheno-rg-cov) where they are at hand, so that the overlapping covariance is
# computed from them.
for ss in "$SUMSTATS"/*; do
  t=$(basename "$ss")
  summit --h2 "$ss" --ldscores "$OUT/ld_curve.gw.ldscore.gz" --annot "$ANNOT" --num-threads 4 --out "$OUT/h2_${t%%.*}"
done

# ------------------------------------------------------------------------------------------------ 5. fine-mapping (anyone)
# Per region (seconds to a minute): the matrix restricted to the GWAS's SNPs, the z-scores on its counted alleles and the
# predicted error variance; then the two SuSiE fits (R with susieR), the second with the residual variance s^2.
# mkdir -p "$OUT/fm"
# summit admixed-ld matrix --ref "$OUT/ref_clean" --params "$OUT/ld_curve.admixed_ld.json" --curve "$OUT/curve" \
#   --region 4:1000000-4000000 --sumstats "$SUMSTATS/height.tsv" --error-variance --out "$OUT/fm/region1"
# Rscript "$(dirname "$0")/finemap_two_stage.R" "$OUT/fm/region1"

# ------------------------------------------------------------------------------------------------ 6. track rg (optional)
# Needs phased genotypes and local-ancestry calls of the GWAS sample, and a Tractor-style GWAS with sandwich SEs from
# the covariates-only residuals (se_hcn_anc<s>); the GWAS runs where the genotypes are, the rest anywhere. See the wiki.
# summit admixed-ld tractor-gwas --pgen /path/to/phased_chr{chr} --lanc /path/to/lanc_chr{chr}.lanc --pheno PHENO \
#   --covar "$GWAS_COV" --keep "$GWAS_SAMPLE" --sources AFR,EUR --threads 16 --out "$OUT/tractor/gwas"
# for t in height bmi ldl; do summit admixed-ld tractor-sumstats --tractor "$OUT/tractor/gwas.$t.tractor.tsv" \
#   --sources AFR,EUR --out "$OUT/tracks/$t"; done
# summit admixed-ld compose --ref "$OUT/ref_clean" --sumstats "$OUT/tracks/height.AFR.sumstats" --annot MAF5_ANNOT \
#   --curve "$OUT/curve" --mean-ancestry M_AFR,M_EUR --ancestry-sd SD --tracks --out "$OUT/ld_tracks"
# summit admixed-ld track-rg --tracks "$OUT/ld_tracks" --sumstats "$OUT"/tracks/height "$OUT"/tracks/bmi \
#   "$OUT"/tracks/ldl --annot MAF5_ANNOT --out "$OUT/track_rg"
echo "SUMMIT-MIX template finished; outputs in $OUT"
