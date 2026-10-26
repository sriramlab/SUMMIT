#!/usr/bin/env bash
# h2 and rg in a cohort with sibling pairs, keeping everyone and excluding the related pairs.
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_related_inputs.py
D=./out/related

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(env PYTHONPATH=../src python -m summit)
fi

# 1. Related-pair moments: one REL_M value per trait and one REL_M12 value for the pair (one genotype pass)
"${SUMMIT[@]}" relatives moments --geno "$D/cohort.bed" --pheno "$D/traits.tsv" \
  --covar "$D/cohort.cov" --pairs "$D/pairs.tsv" --kinship-col RELATEDNESS --kinship-min 0.0442 \
  --cross --out "$D/relmoments"
for t in trait_a trait_b; do
  "${SUMMIT[@]}" relatives append --sumstats "$D/$t.sumstats" --rel "$D/relmoments.rel_m.tsv" \
    --pheno "$t" --out "$D/$t.rel.sumstats"
done
"${SUMMIT[@]}" relatives cross-m12 --rel "$D/relmoments.rel_m12.tsv" --pheno1 trait_a --pheno2 trait_b \
  --out "$D/trait_a_b.relm12.tsv"

# 2. In-sample LD scores with the related pairs' share (LD_REL) and SNP-variance weights (V)
"${SUMMIT[@]}" --geno "$D/cohort.bed" --covar "$D/cohort.cov" \
  --relative-pairs "$D/pairs.tsv" --kinship-col RELATEDNESS --kinship-min 0.0442 \
  --nvecs 200 --block-size 256 --seed 1 --dtype float64 --num-threads 2 --out "$D/cohort"

# 3. h2 with the related pairs excluded
"${SUMMIT[@]}" --h2 "$D/trait_a.rel.sumstats" --ldscores "$D/cohort.gw.ldscore.gz" \
  --njack chr --num-threads 2 --out ./out/related_h2

# 4. Genetic correlation with the related pairs excluded from every equation
"${SUMMIT[@]}" --rg "$D/trait_a.rel.sumstats,$D/trait_b.rel.sumstats" --rel-m12 "$D/trait_a_b.relm12.tsv" \
  --ldscores "$D/cohort.gw.ldscore.gz" \
  --pheno-rg "$D/trait_a.tsv,$D/trait_b.tsv" --pheno-rg-cov "$D/cohort.cov,$D/cohort.cov" \
  --njack chr --num-threads 2 --out ./out/related_rg

# For comparison: the same h2 with the related pairs kept (no REL_M), which is biased upward here
"${SUMMIT[@]}" --h2 "$D/trait_a.sumstats" --ldscores "$D/cohort.gw.ldscore.gz" \
  --njack chr --num-threads 2 --out ./out/related_h2_pairs_kept
