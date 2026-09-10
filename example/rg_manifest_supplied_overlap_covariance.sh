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
  --rg ./out/synthetic/rg_manifest.tsv \
  --rg-manifest-fast \
  --ldscores ./out/small.2bins.gw.ldscore.gz \
  --annot ./out/synthetic/small.annot \
  --out ./out/rg_manifest_supplied_overlap_covariance \
  --njack chr \
  --num-threads 2
