#!/bin/bash
# Final benchmark, per cohort: PLINK 2 GWAS of the 500 simulated phenotypes on everyone and on the unrelated set,
# `summit relatives moments` (REL_M) on everyone, and SUMMIT sumstats files (with COV_RANK) for three arms.
# usage: gwas_and_moments.sh POP NPC THREADS   (run in ~/summit_relatives on veena)
set -euo pipefail
cd ~/summit_relatives
pop=$1; npc=$2; th=${3:-4}
O=final/$pop
PY=~/anaconda3/envs/summit/bin/python
mkdir -p $O/gwas_all $O/gwas_unrel
glm="--glm hide-covar cols=chrom,pos,ref,alt,omitted,nobs,beta,se --covar-variance-standardize"
[ -e $O/gwas_all.done ] || { nice -n 5 ~/bin/plink2 --threads $th --memory 12000 --bfile panels/$pop/${pop}_full \
    --pheno $O/pheno.txt --covar $O/covar_all.txt $glm --out $O/gwas_all/g > $O/gwas_all.stdout 2>&1 && touch $O/gwas_all.done; }
[ -e $O/gwas_unrel.done ] || { nice -n 5 ~/bin/plink2 --threads $th --memory 12000 --bfile $O/geno_unrel \
    --pheno $O/pheno.txt --covar $O/covar_unrel.txt $glm --out $O/gwas_unrel/g > $O/gwas_unrel.stdout 2>&1 && touch $O/gwas_unrel.done; }
[ -e $O/rel.done ] || { nice -n 5 env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th \
    $PY -m summit relatives moments --geno panels/$pop/${pop}_full --pheno $O/pheno.txt --covar $O/covar_all.txt \
    --pairs $O/pairs.txt --kinship-col K_grm --kinship-min 0.0442 --out $O/rel > $O/rel.stdout 2>&1 && touch $O/rel.done; }
nice -n 5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/convert_sumstats.py $pop $npc $th
echo "$pop done"
