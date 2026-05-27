#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p out

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(python ../src/summit.py)
fi

"${SUMMIT[@]}" \
  --geno ./small.bed \
  --annot ./small.2bins_annot.txt \
  --out ./out/small.2bins \
  --nvecs 100 \
  --step_size 10000 \
  --seed 1 \
  --dtype float64 \
  --rand-samp 0.5 \
  --covar ./small.cov \
  --num-threads 2
