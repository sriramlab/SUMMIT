#!/bin/bash
# Reuses the qualified eight-slot/native-BLIS launch contract. Output root is
# supplied explicitly; use jflint project storage when scratch is near quota.
set -euo pipefail
umask 077
phase=${1:?phase required}
output_root=${2:?exclusive output root required}
code_root=$PWD
python_exe=/u/home/b/bronsonj/.conda/envs/summit/bin/python
base=/u/scratch/b/bronsonj/general_gxe_chromosome_20260914
export TMPDIR="$output_root/tmp/${JOB_ID:?}.${SGE_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export LD_PRELOAD=/u/home/b/bronsonj/.conda/envs/summit/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/u/home/b/bronsonj/.conda/envs/summit/lib
export PYTHONPATH=/u/scratch/b/bronsonj/summit_pgs_qualification_20260913/test_deps
export SUMMIT_PRIVATE_NATIVE_DIR=$base/runtime_reference_v3/src/summit
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
[[ ${NSLOTS:?} == 8 ]]
launch=/u/scratch/b/bronsonj/general_gxe_rank_validation_20260907/runtime/launch_with_affinity.py
run=("$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py")
case $phase in
 qualify)
  exec "${run[@]}" pytest "$code_root/tests/test_cross_trait_uncertainty.py" \
    "$code_root/tests/test_cross_trait_requested_pairs.py" "$code_root/tests/test_cross_trait_study_publication.py" \
    "$code_root/tests/test_cross_trait_runtime_placement.py" -q -p no:cacheprovider
  ;;
 generate|score|fit)
  inputs=/u/project/jflint/bronsonj/cross_trait_20260923/simulation_inputs
  reference=/u/scratch/b/bronsonj/cross_trait_20260923/simulation_v3/simulation_reference/reference.generalized-gxe-variant-ldscore-v1.npz
  if [[ $phase == score || $phase == fit ]]; then [[ -s $output_root/simulation_generate/generated.npz ]]; fi
  if [[ $phase == fit ]]; then [[ -s $output_root/simulation_score/scores.npz ]]; fi
  exec "${run[@]}" cross_trait_simulation "$phase" --axes "$inputs/axes.npz" \
    --bed-prefix "$inputs/UKBB_EUR_50k_unrel_3rd.no_mhc_imp" --reference "$reference" \
    --generated "$output_root/simulation_generate/generated.npz" \
    --scores "$output_root/simulation_score/scores.npz" --output "$output_root/simulation_$phase" \
    --replicates 500 --seed 2026092402 --threads 8
  ;;
 within)
  exec "${run[@]}" cross_trait_refit --base "$base" --output "$output_root/within_directional" \
    --source-commit "${3:?immutable source commit required}"
  ;;
 study)
  chromosome=${3:-${SGE_TASK_ID:?chromosome required}}
  [[ $chromosome =~ ^([1-9]|1[0-9]|2[0-2])$ ]]
  # Names and pair orientation are authenticated in checkpoint identity.
  mapfile -t traits < <("$python_exe" -c 'import json,sys; print("\n".join(sorted({t for p in json.load(open(sys.argv[1])) for t in p})))' "$output_root/pairs.json")
  mkdir -p "$output_root/study"
  extra=(--checkpoint-every-blocks 128 --max-run-seconds 39600)
  if [[ $chromosome == 22 ]]; then extra=(--checkpoint-every-blocks 128 --max-run-seconds 18000); fi
  shopt -s nullglob
  checkpoints=("$output_root/study/chr$chromosome"/checkpoint_*.npz)
  shopt -u nullglob
  if (( ${#checkpoints[@]} )); then extra+=(--resume-from "${checkpoints[-1]}"); fi
  exec "${run[@]}" cross_trait_study study --base "$base" \
    --bed-prefix "/u/home/b/bronsonj/project-sriram/UKBB/imp/qc.v1/by_chr/imp.$chromosome" \
    --annotations "$base/imputed_maf3_design/annotations_chr$chromosome.npy" \
    --output "$output_root/study/chr$chromosome" --chromosome "$chromosome" --common-only \
    --trait-names "${traits[@]}" --pairs-file "$output_root/pairs.json" --threads 8 "${extra[@]}"
  ;;
 *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
