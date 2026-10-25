#!/bin/bash
# Final benchmark fits (veena, ~/summit_relatives): summit --h2 for every arm and jackknife, phenotypes in chunks.
#   unrel     GWAS on the unrelated set, in-sample LD of that set (V)
#   naive     GWAS on everyone, in-sample LD of everyone (V); no REL_M, so LD_REL is not used
#   pairexcl  GWAS on everyone with REL_M, in-sample LD of everyone (V); LD_REL subtracted
#   pairexcl_noV (AFR) as pairexcl with LD scores written with --snp-var-weights-off
# usage: fit_arms.sh POP NPROC
set -euo pipefail
cd ~/summit_relatives
pop=$1; nproc=${2:-12}
O=final/$pop
PY=~/anaconda3/envs/summit/bin/python
declare -A SS=( [unrel]=unrel [naive]=all [pairexcl]=frel [pairexcl_noV]=frel )
declare -A LD=( [unrel]=unrel [naive]=all [pairexcl]=all [pairexcl_noV]=all_off )
arms="unrel naive pairexcl"; [ "$pop" = AFR ] && arms="$arms pairexcl_noV"
jobs=$O/fit_jobs.txt; : > $jobs
mkdir -p $O/fit
for arm in $arms; do
  files=($(ls $O/ss/${SS[$arm]}/*.tsv.gz | sort))
  for k in $(seq 0 9); do
    d=$O/chunks/$arm/$k; mkdir -p $d
    for ((i=k; i<${#files[@]}; i+=10)); do ln -sf $(readlink -f ${files[$i]}) $d/; done
    for nj in 200 chr; do
      out=$O/fit/${arm}_${nj}_$k
      [ -s $out.results.tsv ] && [ $(($(wc -l < $out.results.tsv) - 1)) -eq $(ls $d | wc -l) ] && continue
      echo "env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -m summit --h2 $d --ldscores $O/ld/${LD[$arm]}.gw.ldscore.gz --keep-ambiguous --njack $nj --num-threads 1 --out $out > $out.stdout 2>&1" >> $jobs
    done
  done
done
echo "$(wc -l < $jobs) fit jobs for $pop"
xargs -P $nproc -I{} bash -c "{}" < $jobs
echo "$pop fits done"
