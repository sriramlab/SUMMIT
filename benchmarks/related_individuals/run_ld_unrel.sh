#!/bin/bash
# LD scores of the unrelated sets (veena, ~/summit_relatives): subset BEDs with PLINK 2, then in-sample V LD scores.
set -euo pipefail
cd ~/summit_relatives
th=${1:-6}
PY=~/anaconda3/envs/summit/bin/python
SUMMIT="env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th $PY -m summit"
common="--nvecs 20000 --block-size 16384 --dtype float64 --memory-gib 24 --num-threads $th"
for pop in AFR SAS EUR; do
  [ -s final/$pop/geno_unrel.bed ] || ~/bin/plink2 --threads 4 --bfile panels/$pop/${pop}_full --keep final/$pop/keep_unrel.txt \
      --make-bed --out final/$pop/geno_unrel > /dev/null
done
seed=41
for pop in AFR SAS EUR; do
  seed=$((seed + 2))
  [ -s final/$pop/ld/unrel.gw.ldscore.gz ] || $SUMMIT --geno final/$pop/geno_unrel $common --seed $seed \
      --covar final/$pop/covar_unrel.txt --out final/$pop/ld/unrel > final/$pop/ld/unrel.stdout 2>&1 &
done
wait
echo ld_unrel_done
