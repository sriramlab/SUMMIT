#!/usr/bin/env bash
# SUMMIT-MIX on simulated data, end to end (about a minute): LD scores of an admixed cohort without its in-sample LD,
# composed from source panels, the GWAS summary statistics and the cohort's curve (curve mode) or an admixture date
# (date mode); h2 and genetic correlation with them and with in-sample LD scores; two-stage fine-mapping.
# For real data see admixed_ld_real_data.sh.
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_admixed_ld_inputs.py
D=./out/admixed_ld

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(env PYTHONPATH=../src python -m summit)
fi

# 1. The reference from the two source panels (once per set of panels; positions from the BIM CM column), then the
#    panels' own admixture LD and background measured and, where it passes its test, removed
"${SUMMIT[@]}" admixed-ld reference --panel "$D/panel_afr" --panel "$D/panel_eur" --names AFR,EUR --step 512 \
  --out "$D/ref"
"${SUMMIT[@]}" admixed-ld clean-panels --ref "$D/ref" --panel "$D/panel_afr" --panel "$D/panel_eur" --names AFR,EUR \
  --out "$D/ref_clean"

# 2. Curve mode: the cohort's curve, with the GWAS covariates and the background term (aggregates only; run where the
#    genotypes are). A real cohort also gives --keep (the GWAS sample) and --relative-pairs.
"${SUMMIT[@]}" admixed-ld curve --geno "$D/cohort" --ref "$D/ref_clean" --covar "$D/cohort.cov" --background \
  --cell-cm 0.5 --out "$D/cohort"

# 3. Composed LD scores: curve mode, and date mode from the summary statistics and the admixture date alone
"${SUMMIT[@]}" admixed-ld compose --ref "$D/ref_clean" --sumstats "$D/trait.sumstats" "$D/trait2.sumstats" \
  --annot "$D/cohort.annot" --curve "$D/cohort" --relatedness curve --cell-matrix --out "$D/composed_curve"
"${SUMMIT[@]}" admixed-ld compose --ref "$D/ref_clean" --sumstats "$D/trait.sumstats" "$D/trait2.sumstats" \
  --annot "$D/cohort.annot" --admixture-generations 8 --out "$D/composed_date"

# 4. In-sample LD scores with the GWAS covariates, for comparison
"${SUMMIT[@]}" --geno "$D/cohort.bed" --annot "$D/cohort.annot" --covar "$D/cohort.cov" --nvecs 200 \
  --block-size 256 --seed 1 --dtype float64 --num-threads 2 --out "$D/insample"

# 5. h2 and genetic correlation with each set of LD scores (the overlapping covariance from the phenotypes)
for ld in composed_curve composed_date insample; do
  "${SUMMIT[@]}" --h2 "$D/trait.sumstats" --ldscores "$D/$ld.gw.ldscore.gz" --annot "$D/cohort.annot" \
    --num-threads 2 --out "./out/admixed_ld_$ld"
  "${SUMMIT[@]}" --rg "$D/trait.sumstats,$D/trait2.sumstats" --ldscores "$D/$ld.gw.ldscore.gz" \
    --pheno-rg "$D/trait.pheno,$D/trait2.pheno" --pheno-rg-cov "$D/cohort.cov,$D/cohort.cov" \
    --annot "$D/cohort.annot" --num-threads 2 --out "./out/admixed_ld_rg_$ld"
done

# 6. Fine-mapping: a regional LD matrix (curve mode) with the trait's z-scores aligned to its counted alleles and the
#    predicted error variance of its entries; then SuSiE twice, the second time with the residual variance s^2 that
#    the first fit's posterior and the predicted error give (needs R with susieR; finemap-variance gives the same s^2)
"${SUMMIT[@]}" admixed-ld matrix --ref "$D/ref_clean" --params "$D/composed_curve.admixed_ld.json" \
  --curve "$D/cohort" --region 1:5000000-8000000 --sumstats "$D/trait.sumstats" --error-variance --out "$D/region1"
if command -v Rscript >/dev/null 2>&1 && Rscript -e 'quit(status = !requireNamespace("susieR", quietly = TRUE))'; then
  Rscript ./finemap_two_stage.R "$D/region1"
  "${SUMMIT[@]}" admixed-ld finemap-variance --matrix "$D/region1" --posterior "$D/region1.posterior1.tsv" \
    --out "$D/region1"
fi

echo "SUMMIT-MIX example finished; LD scores in $D, fits in ./out/admixed_ld_*"
