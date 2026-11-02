#!/usr/bin/env bash
# Heritability due to direct genetic effects from synthetic sibships, beside the population heritability of the
# same people.
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_sibling_inputs.py
D=./out/siblings

if command -v summit >/dev/null 2>&1; then
  SUMMIT=(summit)
else
  SUMMIT=(env PYTHONPATH=../src python -m summit)
fi

# Age is regressed out before centering within sibships; PC1 enters the population estimate only.
"${SUMMIT[@]}" relatives direct --geno "$D/siblings" --sibships "$D/sibships.tsv" --annot "$D/siblings.annot" \
  --pheno "$D/traits.tsv" --covar "$D/siblings.cov" --within-covar age --nblocks 50 --grm "$D/grm" \
  --out ./out/direct
