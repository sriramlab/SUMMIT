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
  --rg ./out/rg_manifest.fixed1.tsv \
  --rg-manifest-fast \
  --ldscores ./double_uniform_10k_stoc_k100.gw.ldscore.gz \
  --annot ./double_uniform_0.2.annot.txt \
  --out ./out/rg_manifest_fixed_intercept \
  --njack 100 \
  --num-threads 2
