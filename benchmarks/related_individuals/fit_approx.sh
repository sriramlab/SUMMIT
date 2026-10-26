#!/bin/bash
# Fits of the approximate related-pair parts (veena, ~/summit_relatives), with the same settings as fit_arms.sh and
# rg_stage.sh: pairexcl_A1 / pairexcl_A2 use ss/frel_A1, ss/frel_A2 (and m12_A1, m12_A2 for rg) and the in-sample LD
# scores of everyone (V, LD_REL). 200-block jackknife only.
# usage: fit_approx.sh POP NPROC
set -euo pipefail
cd ~/summit_relatives
pop=$1; nproc=${2:-12}
O=final/$pop; R=$O/rg
PY=~/anaconda3/envs/summit/bin/python
SUMMIT="env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -m summit"
jobs=$O/fit_approx_jobs.txt; : > $jobs
mkdir -p $O/fit $R/fit
for v in A1 A2; do
  arm=pairexcl_$v
  files=($(ls $O/ss/frel_$v/*.tsv.gz | sort))
  for k in $(seq 0 9); do
    d=$O/chunks/$arm/$k; mkdir -p $d
    for ((i=k; i<${#files[@]}; i+=10)); do ln -sf $(readlink -f ${files[$i]}) $d/; done
    out=$O/fit/${arm}_200_$k
    [ -s $out.results.tsv ] && [ $(($(wc -l < $out.results.tsv) - 1)) -eq $(ls $d | wc -l) ] && continue
    echo "$SUMMIT --h2 $d --ldscores $O/ld/all.gw.ldscore.gz --keep-ambiguous --njack 200 --num-threads 1 --out $out > $out.stdout 2>&1" >> $jobs
  done
  npairs=$(ls $R/m12_$v/*.tsv | wc -l)
  for ((r=0; r<npairs; r++)); do
    o=$R/fit/${arm}_200_$r
    [ -s $o.log ] && continue
    echo "$SUMMIT --rg $R/ss/frel_$v/a$r.tsv.gz,$R/ss/frel_$v/b$r.tsv.gz --ldscores $O/ld/all.gw.ldscore.gz --pheno-rg $R/ph_all/a$r.txt,$R/ph_all/b$r.txt --rel-m12 $R/m12_$v/p$r.tsv --keep-ambiguous --njack 200 --num-threads 1 --pheno-rg-cov $O/covar_all.txt,$O/covar_all.txt --out $o > /dev/null 2>&1" >> $jobs
  done
done
echo "$(wc -l < $jobs) approximation fit jobs for $pop"
xargs -P $nproc -I{} bash -c "{}" < $jobs
echo "$pop approximation fits done"
