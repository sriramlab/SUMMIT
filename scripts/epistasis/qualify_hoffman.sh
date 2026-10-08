#!/bin/bash
# Small native/public-workflow qualification, not a genomic throughput run.
set -euo pipefail
umask 077
: "${SUMMIT_RUNTIME_ROOT:?Set a new staged runtime under /u/scratch/b/bronsonj}"
case "$SUMMIT_RUNTIME_ROOT" in /u/scratch/b/bronsonj/*) ;; *) exit 2 ;; esac
[[ ${NSLOTS:?} == 8 ]]
root=$SUMMIT_RUNTIME_ROOT
export SUMMIT_QUALIFICATION_LAUNCHER=$(readlink -f "$0")
py=/u/home/b/bronsonj/.conda/envs/summit/bin/python
cmake=/u/home/b/bronsonj/.conda/envs/summit/bin/cmake
envroot=/u/home/b/bronsonj/.conda/envs/summit
blis=/u/scratch/b/bronsonj/summit_general_gxe/20260823_age_sex/build/blis_skx_cblas_install
export PYTHONDONTWRITEBYTECODE=1
export LD_PRELOAD=$envroot/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$envroot/lib
export PYTHONPATH=$root/code:/u/scratch/b/bronsonj/summit_pgs_qualification_20260913/test_deps
export TMPDIR=$root/tmp/${JOB_ID:?}
mkdir -p "$TMPDIR"
cd "$root/code"
"$py" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);meta=json.loads((r/'snapshot.json').read_text())
for name,expected in meta['files'].items():
    if hashlib.sha256((r/'code'/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('staged source changed: '+name)
PY
snapshot=$($py -c 'import json,os; print(json.load(open(os.environ["SUMMIT_RUNTIME_ROOT"]+"/snapshot.json"))["identity"])')
common=(-S "$root/code" -G Ninja -DCMAKE_MAKE_PROGRAM="$envroot/bin/ninja"
  -DCMAKE_CXX_COMPILER=/u/local/compilers/gcc/12.5.0/bin/g++
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$py" -DBLA_VENDOR=OpenBLAS
  -DOPENBLAS_INCLUDE_DIR=/usr/include/openblas -DBLAS_openblas_LIBRARY=/usr/lib64/libopenblas.so
  -DGWLDCORE_ENABLE_NATIVE_OPT=OFF -DGXELDCORE_USE_PRIVATE_OPENBLAS=OFF
  -DGXELDCORE_GEMM_INTEGRITY=ON -DGXELDCORE_GEMM_CHECKSUM=ON
  -DGWLDCORE_SOURCE_COMMIT=fc11388a017535ce73c19ea7a227a685acffb08e
  -DGWLDCORE_SOURCE_TREE_SHA256="$snapshot")
"$cmake" "${common[@]}" -B "$root/build/private" -DGXELDCORE_USE_PRIVATE_BLIS=ON \
  -DGXELDCORE_PRIVATE_BLIS_ARCHIVE="$blis/lib/libblis.a" \
  -DGXELDCORE_PRIVATE_BLIS_INCLUDE_DIR="$blis/include" \
  -DGXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT=e8566eb3e773fb54d11b33e371d13f22d2941e50 \
  -DGXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256=eefbd29a5cbb1d6982bdce76e8034037ea2a3f3ead33d90d9f3cef6da5728154 \
  -DGXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY=skx
"$cmake" --build "$root/build/private" --parallel 2
export SUMMIT_PRIVATE_NATIVE_DIR=$root/build/private
"$py" scripts/epistasis/hoffman_launch.py pytest -q -p no:cacheprovider \
  tests/test_epistasis_polygenic_operator.py \
  tests/test_epistasis_full_matched.py::test_full_matched_public_path \
  tests/test_epistasis_hoffman_launch.py
"$cmake" "${common[@]}" -B "$root/build/portable" -DGXELDCORE_USE_PRIVATE_BLIS=OFF
"$cmake" --build "$root/build/portable" --parallel 2
export SUMMIT_NATIVE_DIR=$root/build/portable
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1
unset SUMMIT_PRIVATE_NATIVE_DIR OMP_PLACES OMP_PROC_BIND
"$py" scripts/epistasis/checkout_python.py pytest -q -p no:cacheprovider \
  tests/test_epistasis*.py tests/test_prediction*.py
"$py" - "$root" <<'PY'
import hashlib,json,os,socket,sys
from pathlib import Path
r=Path(sys.argv[1]);meta=json.loads((r/'snapshot.json').read_text())
record=dict(source_snapshot=meta['identity'],job_id=os.environ['JOB_ID'],host=socket.gethostname(),
    launcher_sha256=hashlib.sha256(Path(os.environ['SUMMIT_QUALIFICATION_LAUNCHER']).read_bytes()).hexdigest(),
    initial_affinity=sorted(os.sched_getaffinity(0)),slots=int(os.environ['NSLOTS']),
    native={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()
        for kind in ('private','portable') for p in (r/'build'/kind).glob('*.so')},
    scope='bounded arithmetic and public workflow, portable regression, actual scheduler/native placement; no full-marker throughput claim')
with (r/'QUALIFIED.json').open('x') as handle:json.dump(record,handle,indent=2)
PY
