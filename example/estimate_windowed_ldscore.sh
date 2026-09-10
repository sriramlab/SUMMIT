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
  --annot ./out/synthetic/small.annot \
  --covar ./out/synthetic/small.cov \
  --ld-wind-kb 20000 \
  --out ./out/small.2bins.20mb \
  --seed 1 \
  --num-threads 2
