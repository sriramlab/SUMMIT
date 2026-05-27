#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_example_inputs.py

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(python ../src/summit.py)
fi

"${SUMMIT[@]}" \
  --geno ./small.bed \
  --env ./out/small.env \
  --annot ./small.2bins_annot.txt \
  --covar ./small.cov \
  --out ./out/small.2bins.env \
  --nvecs 100 \
  --step_size 10000 \
  --seed 1 \
  --dtype float64 \
  --rand-samp 0.5 \
  --num-threads 2
