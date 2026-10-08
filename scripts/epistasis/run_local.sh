#!/usr/bin/env bash
# Run only on explicitly assigned local CPUs. For Hoffman use its verified
# scheduler launch requirements; this script does not request resources.
set -euo pipefail
local_host=$(hostname -s)
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ ${local_host,,} == tabla ]]; then
  case ${1:-} in
    make-inputs|prepare|train-direction|prepare-traits)
      echo 'Tabla cohort execution is disabled. An approved bounded pilot must use run_measured.py; see benchmarks/epistasis/execution_policy_20261005.md.' >&2
      exit 2
      ;;
  esac
fi
: "${SUMMIT_CPUS:?Set SUMMIT_CPUS to an explicit comma-separated list of assigned physical CPUs}"
IFS=',' read -r -a cpus <<< "$SUMMIT_CPUS"
threads=${#cpus[@]}
for cpu in "${cpus[@]}"; do [[ "$cpu" =~ ^[0-9]+$ ]] || { echo 'Explicit CPU IDs required' >&2; exit 2; }; done
if [[ ${local_host,,} == tabla ]]; then
  (( threads <= 8 )) || { echo 'Tabla permits at most eight assigned physical CPUs' >&2; exit 2; }
  for cpu in "${cpus[@]}"; do
    (( cpu >= 8 && cpu <= 63 )) || { echo 'Tabla compute CPUs must be in 8-63' >&2; exit 2; }
  done
fi
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads BLIS_NUM_THREADS=$threads
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
conda_root=${SUMMIT_CONDA_ROOT:-/home/bronsonj/anaconda3}
taskset -c "$SUMMIT_CPUS" "$conda_root/bin/conda" run --no-capture-output -n summit python - "$SUMMIT_CPUS" <<'PY'
import os,sys,socket
from pathlib import Path
cpus=list(map(int,sys.argv[1].split(',')))
if socket.gethostname().split('.')[0].lower()=='tabla' and (not set(cpus)<=set(range(8,64)) or len(cpus)>8):
    raise SystemExit('Tabla requires at most eight assigned physical CPUs in 8-63')
if len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):
    raise SystemExit('CPU set is duplicated or outside initial process affinity')
physical=[]
for cpu in cpus:
    root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    physical.append(((root/'physical_package_id').read_text(),(root/'core_id').read_text()))
if len(set(physical))!=len(cpus):raise SystemExit('Choose one hardware thread per physical core')
PY
if [[ ${SUMMIT_BACKEND:-portable} == private ]]; then
  : "${SUMMIT_PRIVATE_NATIVE_DIR:?Set the qualified private BLIS build directory}"
  places=''
  for cpu in "${cpus[@]}"; do places+="${places:+,}{$cpu}"; done
  export OMP_PLACES=$places OMP_PROC_BIND=SPREAD OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=$threads OMP_MAX_ACTIVE_LEVELS=1
  launcher="$repo/scripts/generalized_gxe/private_python.py"
  export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
elif [[ ${SUMMIT_BACKEND:-portable} == portable ]]; then
  : "${SUMMIT_NATIVE_DIR:?Set the qualified portable OpenBLAS build directory}"
  export OMP_PROC_BIND=FALSE OMP_DYNAMIC=FALSE
  launcher="$repo/scripts/epistasis/checkout_python.py"
else
  echo 'SUMMIT_BACKEND must be portable or private' >&2; exit 2
fi
controls=()
if [[ ${1:-} == make-inputs || ${1:-} == prepare || ${1:-} == train-direction || ${1:-} == prepare-traits ]]; then controls=(--num-threads "$threads"); fi
exec taskset -c "$SUMMIT_CPUS" "$conda_root/bin/conda" run --no-capture-output -n summit \
  python "$launcher" summit.epistasis.cli "$@" "${controls[@]}"
