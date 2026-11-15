#!/usr/bin/env bash
# LD scores of an admixed cohort without its in-sample LD: composed from source panels, the GWAS summary statistics
# and the decay of local-ancestry covariance, then compared with in-sample LD scores in the same h2 fit.
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_admixed_ld_inputs.py
D=./out/admixed_ld

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(env PYTHONPATH=../src python -m summit)
fi

# 1. The reference resource from the two source panels (once per reference; positions from the BIM CM column)
"${SUMMIT[@]}" admixed-ld reference --panel "$D/panel_afr" --panel "$D/panel_eur" --names AFR,EUR --step 512 \
  --out "$D/ref"

# 2. The cohort's weighted-LD curve and global-ancestry moments (aggregates only; run where the genotypes are)
"${SUMMIT[@]}" admixed-ld curve --geno "$D/cohort" --ref "$D/ref" --covar "$D/cohort.cov" --cell-cm 0.5 \
  --out "$D/cohort"

# 3. Composed LD scores, with the decay from the curve and from a known admixture date
"${SUMMIT[@]}" admixed-ld compose --ref "$D/ref" --sumstats "$D/trait.sumstats" --annot "$D/cohort.annot" \
  --curve "$D/cohort" --out "$D/composed_curve"
"${SUMMIT[@]}" admixed-ld compose --ref "$D/ref" --sumstats "$D/trait.sumstats" --annot "$D/cohort.annot" \
  --admixture-generations 8 --ancestry-sd 0.12 --out "$D/composed_date"

# 4. In-sample LD scores with the GWAS covariates, for comparison
"${SUMMIT[@]}" --geno "$D/cohort.bed" --annot "$D/cohort.annot" --covar "$D/cohort.cov" --nvecs 200 \
  --block-size 256 --seed 1 --dtype float64 --num-threads 2 --out "$D/insample"

# 5. h2 with each set of LD scores
for ld in composed_curve composed_date insample; do
  "${SUMMIT[@]}" --h2 "$D/trait.sumstats" --ldscores "$D/$ld.gw.ldscore.gz" --annot "$D/cohort.annot" \
    --num-threads 2 --out "./out/admixed_ld_$ld"
done
