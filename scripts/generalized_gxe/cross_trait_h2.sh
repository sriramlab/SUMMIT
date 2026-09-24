#!/bin/bash
# qsub supplies -wd, -o, -pe shared 8, -binding set linear:1, and complete resources.
set -euo pipefail
umask 077
mode=${1:?qualify, zpass, simulation_reference, simulation or simulation_PHASE}
output_root=${2:?exclusive output root required}
export TMPDIR="$output_root/tmp/${JOB_ID:?}.${SGE_TASK_ID:-0}"
mkdir -p "$TMPDIR"
# UGE executes a spooled copy of this script, so $0 is not the source path.
# qsub -wd is the authenticated immutable checkout selected by the submitter.
code_root=$PWD
[[ -f $code_root/scripts/generalized_gxe/private_python.py ]]
base=/u/scratch/b/bronsonj/general_gxe_chromosome_20260914
python_exe=/u/home/b/bronsonj/.conda/envs/summit/bin/python
export LD_PRELOAD=/u/home/b/bronsonj/.conda/envs/summit/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/u/home/b/bronsonj/.conda/envs/summit/lib
export PYTHONPATH=/u/scratch/b/bronsonj/summit_pgs_qualification_20260913/test_deps
export SUMMIT_PRIVATE_NATIVE_DIR=$base/runtime_reference_v3/src/summit
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
[[ ${NSLOTS:?} == 8 ]]
launch=/u/scratch/b/bronsonj/general_gxe_rank_validation_20260907/runtime/launch_with_affinity.py
case $mode in
 qualify)
  exec "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" pytest \
    "$code_root/tests/test_cross_trait_gram.py" "$code_root/tests/test_cross_trait_zpass.py" \
    "$code_root/tests/test_cross_trait_batch.py" "$code_root/tests/test_cross_trait_fit.py" \
    "$code_root/tests/test_reference_zpass_cli.py" "$code_root/tests/test_cross_trait_runtime_placement.py" -q -p no:cacheprovider
  ;;
 zpass)
  chromosome=${3:-${SGE_TASK_ID:?chromosome required}}
  [[ $chromosome =~ ^([1-9]|1[0-9]|2[0-2])$ ]]
  exec "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    summit.context.reference_zpass_cli --manifest "$base/shared_reference_full_20260916/MANIFEST.json" \
    --reference-root "$base/shared_reference_full_20260916" --chromosome "$chromosome" \
    --master-input "$base/full_cohort_inputs_20260916/height_raw.npz" \
    --bed-prefix "/u/home/b/bronsonj/project-sriram/UKBB/imp/qc.v1/by_chr/imp.$chromosome" \
    --annotations "$base/imputed_maf3_design/annotations_chr$chromosome.npy" \
    --output "$output_root/zpass_chr$chromosome.npz" --num-threads 8 --width 128
  ;;
 simulation_reference|simulation|simulation_generate|simulation_score|simulation_fit)
  input_root=/u/project/jflint/bronsonj/cross_trait_20260923/simulation_inputs
  common_args=(--axes "$input_root/axes.npz" --bed-prefix "$input_root/UKBB_EUR_50k_unrel_3rd.no_mhc_imp" --threads 8)
  if [[ $mode == simulation_reference ]]; then
   "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    pytest "$code_root/tests/test_cross_trait_simulation_truth.py" -q -p no:cacheprovider
   exec "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    cross_trait_simulation reference "${common_args[@]}" --output "$output_root/simulation_reference" \
    --probes 1024 --blocks 200 --memory-gib 24
  fi
  ref="$output_root/simulation_reference/reference.generalized-gxe-variant-ldscore-v1.npz"
  [[ -f $output_root/simulation_reference/COMPLETE.json && -f $ref ]]
  phases=(generate score fit)
  if [[ $mode != simulation ]]; then phases=("${mode#simulation_}"); fi
  # Separate phase jobs allow resuming only the incomplete phase. Python
  # authenticates prerequisite arrays and publishes to exclusive new paths.
  for phase in "${phases[@]}"; do
   "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    cross_trait_simulation "$phase" "${common_args[@]}" --reference "$ref" \
    --output "$output_root/simulation_$phase" --replicates 100 \
    --generated "$output_root/simulation_generate/generated.npz" --scores "$output_root/simulation_score/scores.npz"
  done
  ;;
 real_gram)
  exec "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    cross_trait_real_gram --base "$base" --bed-prefix /u/home/b/bronsonj/project-sriram/UKBB/imp/qc.v1/by_chr/imp.22 \
    --annotations "$base/imputed_maf3_design/annotations_chr22.npy" --output "$output_root/real_gram_n40000" --n 40000 --threads 8
  ;;
 pilot|pilot_with_z|pilot_array)
  chromosome=${3:-${SGE_TASK_ID:?chromosome required}}
  [[ $chromosome =~ ^([1-9]|1[0-9]|2[0-2])$ ]]
  if [[ $mode == pilot_array ]]; then
   # A scheduler dependency is released after failure as well as success.
   # COMPLETE is written only after both authenticated artifacts are closed.
   # Require all three before doing any further genotype work.
   [[ -s $output_root/pilot_study/chr22/COMPLETE.json \
      && -s $output_root/pilot_study/chr22/cross_trait_summary.npz \
      && -s $output_root/zpass_chr22.npz ]] || {
    echo 'Fused chr22 qualification has no completed score/Z publication' >&2
    exit 1
   }
  fi
  z_args=()
  mkdir -p "$output_root/pilot_study"
  if [[ $mode != pilot ]]; then z_args=(--z-output "$output_root/zpass_chr$chromosome.npz"); fi
  if [[ $chromosome == 22 ]]; then
   "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    pytest "$code_root/tests/test_cross_trait_zpass.py" "$code_root/tests/test_cross_trait_batch.py" \
    "$code_root/tests/test_cross_trait_study_publication.py" -q -p no:cacheprovider
  fi
  exec "$python_exe" "$launch" --threads 8 -- "$python_exe" "$code_root/scripts/generalized_gxe/private_python.py" \
    cross_trait_study study --base "$base" \
    --bed-prefix "/u/home/b/bronsonj/project-sriram/UKBB/imp/qc.v1/by_chr/imp.$chromosome" \
    --annotations "$base/imputed_maf3_design/annotations_chr$chromosome.npy" \
    --output "$output_root/pilot_study/chr$chromosome" --chromosome "$chromosome" --common-only --traits 8 --threads 8 "${z_args[@]}"
  ;;
 *) exit 2 ;;
esac
