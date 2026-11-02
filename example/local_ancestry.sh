#!/usr/bin/env bash
# SNP and local-ancestry heritability in a synthetic admixed cohort.
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_local_ancestry_inputs.py
D=./out/local_ancestry

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(env PYTHONPATH=../src python -m summit)
fi
LD=(--nvecs 200 --block-size 256 --seed 1 --dtype float64 --num-threads 2)

# 1. Local-ancestry markers (ancestry 1 counted, one per 250 kb) and global ancestry from the calls
"${SUMMIT[@]}" local-ancestry markers --calls "$D/lanc/chr{chr}.lanc" --format lanc --sites "$D/lanc/chr{chr}.pvar" \
  --fam "$D/cohort.fam" --ancestry 1 --out "$D/markers"

# 2. SNPs and markers in one fileset, with an LA annotation class
"${SUMMIT[@]}" local-ancestry merge --geno "$D/cohort" --markers "$D/markers" --annot "$D/cohort.annot" --out "$D/joint"

# 3. GWAS of the SNPs and of the merged fileset (an admixture-mapping scan for the markers), with THETA as a covariate
python ./prepare_local_ancestry_inputs.py --sumstats

# 4. In-sample LD scores with the same covariates
"${SUMMIT[@]}" --geno "$D/cohort.bed" --annot "$D/cohort.annot" --covar "$D/cohort_theta.cov" "${LD[@]}" --out "$D/ld_snp"
"${SUMMIT[@]}" --geno "$D/joint.bed" --annot "$D/joint.annot.tsv.gz" --covar "$D/cohort_theta.cov" "${LD[@]}" \
  --out "$D/ld_joint"

# 5. h2 from the SNPs alone, and SNP and local-ancestry components jointly
"${SUMMIT[@]}" --h2 "$D/trait.snp.sumstats" --ldscores "$D/ld_snp.gw.ldscore.gz" --annot "$D/cohort.annot" \
  --num-threads 2 --out ./out/local_ancestry_snp_only
"${SUMMIT[@]}" --h2 "$D/trait.joint.sumstats" --ldscores "$D/ld_joint.gw.ldscore.gz" --annot "$D/joint.annot.tsv.gz" \
  --num-threads 2 --out ./out/local_ancestry_joint

# 6. Test of the local-ancestry component and its standard error (exact, from the genotypes)
"${SUMMIT[@]}" local-ancestry test --geno "$D/joint" --annot "$D/joint.annot.tsv.gz" --pheno "$D/trait.tsv" \
  --covar "$D/cohort_theta.cov" --out ./out/local_ancestry
