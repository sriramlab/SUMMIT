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
  --geno ./out/synthetic/small.bed \
  --env ./out/synthetic/small.env \
  --annot ./out/synthetic/small.annot \
  --covar ./out/synthetic/small.cov \
  --out ./out/small.2bins.env \
  --nvecs 100 \
  --step_size 256 \
  --seed 1 \
  --dtype float64 \
  --num-threads 2
