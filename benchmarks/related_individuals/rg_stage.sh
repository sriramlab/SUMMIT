#!/bin/bash
# rg arm of the final benchmark (veena, ~/summit_relatives): bivariate phenotypes, PLINK 2 GWAS on everyone and on
# the unrelated set, `summit relatives moments --cross` in batches of five pairs, REL_M12 per pair with
# `summit relatives cross-m12`, SUMMIT sumstats, and `summit --rg` for the three arms and both jackknifes.
# usage: rg_stage.sh POP NPC NPAIRS THREADS
set -euo pipefail
cd ~/summit_relatives
pop=$1; npc=$2; npairs=${3:-50}; th=${4:-4}
O=final/$pop/rg; F=final/$pop
PY=~/anaconda3/envs/summit/bin/python
SUMMIT="env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -m summit"
[ -s $O/pheno.txt ] || $PY scripts/sim_bivariate_final.py $pop $npairs
glm="--glm hide-covar cols=chrom,pos,ref,alt,omitted,nobs,beta,se --covar-variance-standardize"
mkdir -p $O/gwas_all $O/gwas_unrel $O/rel $O/ss/all $O/ss/unrel $O/ss/frel $O/m12 $O/fit
[ -e $O/gwas_all.done ] || { ~/bin/plink2 --threads $th --memory 12000 --bfile panels/$pop/${pop}_full --pheno $O/pheno.txt \
    --covar $F/covar_all.txt $glm --out $O/gwas_all/g > $O/gwas_all.stdout 2>&1 && touch $O/gwas_all.done; }
[ -e $O/gwas_unrel.done ] || { ~/bin/plink2 --threads $th --memory 12000 --bfile $F/geno_unrel --pheno $O/pheno.txt \
    --covar $F/covar_unrel.txt $glm --out $O/gwas_unrel/g > $O/gwas_unrel.stdout 2>&1 && touch $O/gwas_unrel.done; }
for ((b=0; b<npairs; b+=5)); do
  [ -e $O/rel/b$b.done ] && continue
  cols=$(seq $b $((b + 4 < npairs - 1 ? b + 4 : npairs - 1)) | awk '{printf "%sa%d,b%d", (NR>1?",":""), $1, $1}')
  env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th $PY -m summit relatives moments \
      --geno panels/$pop/${pop}_full --pheno $O/pheno.txt --pheno-cols $cols --covar $F/covar_all.txt \
      --pairs $F/pairs.txt --kinship-col K_grm --kinship-min 0.0442 --per-snp --cross --out $O/rel/b$b > $O/rel/b$b.stdout 2>&1
  touch $O/rel/b$b.done
done
for ((r=0; r<npairs; r++)); do
  b=$(( r / 5 * 5 ))
  [ -s $O/m12/p$r.tsv ] || $SUMMIT relatives cross-m12 --rel $O/rel/b$b.tsv.gz --pheno1 a$r --pheno2 b$r --out $O/m12/p$r.tsv > /dev/null
done
$PY scripts/convert_rg.py $pop $npc $npairs $th
jobs=$O/fit_jobs.txt; : > $jobs
for ((r=0; r<npairs; r++)); do
  for nj in 200 chr; do
    base="--keep-ambiguous --njack $nj --num-threads 1 --pheno-rg-cov $F/covar_all.txt,$F/covar_all.txt"
    o=$O/fit/naive_${nj}_$r; [ -s $o.log ] || echo "$SUMMIT --rg $O/ss/all/a$r.tsv.gz,$O/ss/all/b$r.tsv.gz --ldscores $F/ld/all.gw.ldscore.gz --pheno-rg $O/ph_all/a$r.txt,$O/ph_all/b$r.txt $base --out $o > /dev/null 2>&1" >> $jobs
    o=$O/fit/pairexcl_${nj}_$r; [ -s $o.log ] || echo "$SUMMIT --rg $O/ss/frel/a$r.tsv.gz,$O/ss/frel/b$r.tsv.gz --ldscores $F/ld/all.gw.ldscore.gz --pheno-rg $O/ph_all/a$r.txt,$O/ph_all/b$r.txt --rel-m12 $O/m12/p$r.tsv $base --out $o > /dev/null 2>&1" >> $jobs
    o=$O/fit/unrel_${nj}_$r; [ -s $o.log ] || echo "$SUMMIT --rg $O/ss/unrel/a$r.tsv.gz,$O/ss/unrel/b$r.tsv.gz --ldscores $F/ld/unrel.gw.ldscore.gz --pheno-rg $O/ph_unrel/a$r.txt,$O/ph_unrel/b$r.txt --keep-ambiguous --njack $nj --num-threads 1 --pheno-rg-cov $F/covar_unrel.txt,$F/covar_unrel.txt --out $o > /dev/null 2>&1" >> $jobs
  done
done
echo "$(wc -l < $jobs) rg jobs for $pop"
xargs -P ${5:-12} -I{} bash -c "{}" < $jobs
echo "$pop rg done"
