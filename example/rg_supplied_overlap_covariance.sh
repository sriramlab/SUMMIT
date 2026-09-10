#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_example_inputs.py
if [[ ! -f ./out/small.2bins.gw.ldscore.gz ]]; then
  bash ./estimate_partitioned_gwldscore.sh
fi

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(python ../src/summit.py)
fi

"${SUMMIT[@]}" \
  --rg ./out/synthetic/trait_a.sumstats,./out/synthetic/trait_b.sumstats \
  --overlap-covariance-rg 1 \
  --ldscores ./out/small.2bins.gw.ldscore.gz \
  --annot ./out/synthetic/small.annot \
  --align-alleles \
  --out ./out/rg_supplied_overlap_covariance \
  --njack chr \
  --num-threads 2
