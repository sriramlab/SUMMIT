#!/bin/bash
# qsub supplies -wd, -o, -pe shared 8, -binding set linear:1, and complete resources.
set -euo pipefail
umask 077
mode=${1:?qualify or zpass}
output_root=${2:?exclusive output root required}
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
    "$code_root/tests/test_reference_zpass_cli.py" -q -p no:cacheprovider
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
 *) exit 2 ;;
esac
