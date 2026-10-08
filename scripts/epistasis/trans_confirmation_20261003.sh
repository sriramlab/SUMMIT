#!/usr/bin/env bash
set -euo pipefail
cd /home/bronsonj/SUMMIT-integration
export PYTHONDONTWRITEBYTECODE=1
export SUMMIT_NATIVE_DIR=/data1/bronsonj/summit_release_20260926/build/portable
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
runner=(/home/bronsonj/anaconda3/bin/conda run --no-capture-output -n summit python scripts/epistasis/checkout_python.py scripts.epistasis.trans_validation)
common=(--genotypes /home/bronsonj/UKBB/geno/EUR/UKBB_EUR_unrel_3rd.no_mhc.bed --scratch /data1/bronsonj --replicates 100 --training-samples 1024 --test-samples 4096 --nested-models 3 --nested-draws 2000)
taskset -c 48 nice -n 10 "${runner[@]}" "${common[@]}" --out benchmarks/epistasis/trans_development_20261003 --background-chromosome 2 --genotype-seed 395741 --seed 168953
taskset -c 48 nice -n 10 "${runner[@]}" "${common[@]}" --out benchmarks/epistasis/trans_confirmation_20261003 --background-chromosome 5 --genotype-seed 572813 --seed 862459
