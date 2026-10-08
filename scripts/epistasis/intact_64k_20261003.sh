#!/usr/bin/env bash
# Local measured workload. Select two available physical CPUs before launch.
set -euo pipefail
cd /home/bronsonj/SUMMIT-integration
: "${SUMMIT_CPUS:?Set two available physical CPUs, comma separated}"
IFS=',' read -r -a cpus <<< "$SUMMIT_CPUS"
[[ ${#cpus[@]} == 2 ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/bronsonj/SUMMIT-integration
export SUMMIT_PRIVATE_NATIVE_DIR=/data1/bronsonj/summit_release_20260926/build/private
export OPENBLAS_NUM_THREADS=2 BLIS_NUM_THREADS=2 OMP_NUM_THREADS=2
export OMP_THREAD_LIMIT=2 OMP_DYNAMIC=FALSE OMP_PROC_BIND=SPREAD
export OMP_PLACES="{${cpus[0]}},{${cpus[1]}}" OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
out=/data1/bronsonj/epistasis_trans_64k_intact_20261003
[[ ! -e $out && ! -e $out.json ]]
launcher=(taskset -c "$SUMMIT_CPUS" nice -n 10
  /home/bronsonj/anaconda3/bin/conda run --no-capture-output -n summit
  python scripts/generalized_gxe/private_python.py)
"${launcher[@]}" scripts.epistasis.whole_panel \
  --genotypes /home/bronsonj/UKBB/geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc.bed \
  --samples 65536 --training-samples 32768 --num-threads 2 \
  --trans --local-min-cell 100 --interrupt-training \
  --sampling-model iid_population_projection --out "$out" --receipt "$out.json"
"${launcher[@]}" scripts.epistasis.reuse_workflow --inputs "$out" \
  --out "${out}_reuse" --receipt "${out}_reuse.json" --num-threads 2
"${launcher[@]}" scripts.epistasis.whole_trans_population --inputs "$out" \
  --out "${out}_population" --num-threads 2 --sampling intact_rows --seed 992137 --draws 2000
"${launcher[@]}" scripts.epistasis.whole_trans_population --inputs "$out" \
  --out "${out}_fixed" --num-threads 2 --sampling fixed_rows --seed 992137 --draws 2000
