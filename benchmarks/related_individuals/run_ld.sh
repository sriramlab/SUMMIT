#!/bin/bash
# In-sample LD scores for the final benchmark (veena, ~/summit_relatives). Full cohort: V + LD_REL; unrelated set: V.
# AFR full-cohort files are those of unified_check_v2.sh (same genotypes, PCs and pairs).
set -euo pipefail
cd ~/summit_relatives
th=${1:-6}
PY=~/anaconda3/envs/summit/bin/python
SUMMIT="env PYTHONPATH=summit_dev_install_v2 OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th $PY -m summit"
common="--nvecs 20000 --block-size 16384 --dtype float64 --memory-gib 24 --num-threads $th"
mkdir -p final/AFR/ld
ln -sf ../../../unified_v2/AFR/ld_v.gw.ldscore.gz final/AFR/ld/all.gw.ldscore.gz
ln -sf ../../../unified_v2/AFR/ld_off.gw.ldscore.gz final/AFR/ld/all_off.gw.ldscore.gz
run() {  # pop name seed extra...
  local pop=$1 name=$2 seed=$3; shift 3
  mkdir -p final/$pop/ld
  [ -s final/$pop/ld/$name.gw.ldscore.gz ] || $SUMMIT --geno panels/$pop/${pop}_full $common --seed $seed "$@" \
      --out final/$pop/ld/$name > final/$pop/ld/$name.stdout 2>&1
}
run AFR unrel 41 --covar final/AFR/covar_unrel.txt &
run SAS all 42 --covar final/SAS/covar_all.txt --relative-pairs final/SAS/pairs.txt --kinship-col K_grm --kinship-min 0.0442 &
run SAS unrel 43 --covar final/SAS/covar_unrel.txt &
run EUR all 44 --covar final/EUR/covar_all.txt --relative-pairs final/EUR/pairs.txt --kinship-col K_grm --kinship-min 0.0442 &
run EUR unrel 45 --covar final/EUR/covar_unrel.txt &
wait
echo ld_done
