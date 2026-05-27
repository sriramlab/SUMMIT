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
  --covar ./small.cov \
  --ld-wind-kb 20000 \
  --out ./out/small.2bins.20mb \
  --rand-samp 0.5 \
  --seed 1 \
  --num-threads 2
