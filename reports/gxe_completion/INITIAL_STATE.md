# SUMMIT GxE completion: immutable initial state

Captured before implementation changes on 2026-08-15 UTC (2026-08-14 PDT). The
only filesystem changes made before this file was written were creation of the
requested empty report/test output directories. Factual statements below come
from the listed commands unless explicitly labeled as an inference.

## Repository

| Field | Observed value |
|---|---|
| Root (`git rev-parse --show-toplevel`) | `/home/bronsonj/SUMMIT` |
| Branch (`git branch --show-current`) | `main` |
| HEAD (`git rev-parse HEAD`) | `6bf009e850afd4b112bf46ee04d2cdea7acd8896` |
| Tracking state (`git status --short --branch`) | `## main...origin/main` |
| Dirty files | none (empty porcelain output) |
| Submodules (`git submodule status --recursive`) | none configured/reported |
| Remote tip | `origin/main` and `origin/HEAD` were both shown at HEAD by `git log --decorate` |

The observed recent commits were `6bf009e Project shared GxE source panels
algebraically`, `57bb402 Verify protected multi-environment provenance`, and
`7d883a7 Protect shared multi-environment GxE GEMMs`.

## Python and native build

The interpreter used for all audit commands is
`/home/bronsonj/anaconda3/envs/summit/bin/python` (Python 3.12.12), even though
the invoking shell inherited `CONDA_DEFAULT_ENV=base` and
`CONDA_PREFIX=/home/bronsonj/anaconda3`.

| Field | Observed value |
|---|---|
| Native module | `build/cp312-cp312-linux_x86_64/gxeldcore.cpython-312-x86_64-linux-gnu.so` |
| Module SHA-256 | `51e20f7e6d818d2b241cba19f2798baee922ea729d5d3382da9e0904d6ef1b6a` |
| Backend / version / API | `gxeldcore_direct` / `1.0` / `2` |
| Embedded commit | `6bf009e850afd4b112bf46ee04d2cdea7acd8896` |
| Embedded source-tree SHA-256 | `0bbe9a6b0398b3f647dc2109e8806470443f4ad8c7e7f6913acddf02bee29756` |
| Compiler | GNU C++ 12.2.0 (`/usr/bin/g++`) |
| CMake | 3.25.1 |
| C++ / build type / flags | C++17 / Release / `-O3 -DNDEBUG`, target also `-O3 -march=native` |
| OpenMP | enabled; compiled with `-fopenmp`; CMake found GNU `libgomp` from GCC 12 |
| Loaded OpenMP runtime | Conda `libgomp.so.1`, package 15.2.0 |

The handoff's native module hash did **not** match the observed binary hash.
The embedded commit and source-tree hash did match. This establishes source
provenance but not binary identity, so the two hashes are kept separate.

The repository-supported development build command, read from `CMakeLists.txt`,
is:

```bash
/home/bronsonj/anaconda3/envs/summit/bin/python -m pip install --no-build-isolation -ve .
```

The exact historical invocation that created the existing build directory is
not stored in `CMakeCache.txt`. Its material configuration was observed with:

```bash
grep -E '^(CMAKE_(BUILD_TYPE|CXX_COMPILER|CXX_FLAGS|CXX_FLAGS_RELEASE)|OpenMP|BLAS|BLA_|Python)' \
  build/cp312-cp312-linux_x86_64/CMakeCache.txt
ldd build/cp312-cp312-linux_x86_64/gxeldcore.cpython-312-x86_64-linux-gnu.so
```

`ldd` resolved the native module to the summit environment's
`libopenblas.so.0`, `libgomp.so.1`, `libstdc++.so.6`, `libgcc_s.so.1`,
`libgfortran.so.5`, and `libquadmath.so.0`, plus the system glibc, libm,
libpthread, libdl, and loader. No second BLAS or OpenMP runtime appeared in this
static dependency check. A loaded-process audit is still required by the GEMM
gate.

## BLAS runtime and ABI

The runtime library is
`/home/bronsonj/anaconda3/envs/summit/lib/libopenblas.so.0`, SHA-256
`78e63bed9f40a1877c93da99a0fc29be73ea76e4cfa83a8d14199109934a8305`.

| Query | Observed value |
|---|---|
| `openblas_get_config()` | `OpenBLAS 0.3.34 DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=128` |
| `openblas_get_corename()` | `Zen` |
| `openblas_get_parallel()` | `1` (pthreads) |
| `openblas_get_num_threads()` at audit | `128` |
| Conda package | `libopenblas 0.3.34 pthreads_h94d23a6_0` |
| Integer ABI | LP64 |

LP64 is directly supported by the installed header: `OPENBLAS_USE64BITINT` is
not defined and `blasint` therefore resolves to `int`; the library exports the
unsuffixed `cblas_dgemm` interface. The conda recipe builds with
`DYNAMIC_ARCH=1`, `NO_AFFINITY=1`, `USE_THREAD=1`, and `NUM_THREADS=128`.
`USE_LOCKING` is not passed by that recipe. This observation does not by itself
establish whether locking is enabled through an OpenBLAS default.

No `OMP_*`, `OPENBLAS_*`, `MKL_*`, `BLIS_*`, `GOTO_*`, `NUMEXPR_*`, `KMP_*`,
or `LD_*` variable was set in the invoking shell. Relevant inherited values
were:

```text
CONDA_DEFAULT_ENV=base
CONDA_PREFIX=/home/bronsonj/anaconda3
PATH=/home/bronsonj/bedtools2/bin:/home/bronsonj/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:/home/bronsonj/.local/bin:/home/bronsonj/bin
```

## Host, topology, affinity, and limits

| Field | Observed value |
|---|---|
| Host | `Tabla.CS.UCLA.EDU` |
| Kernel | Linux 4.19.0-21-amd64 x86_64 |
| CPU | AMD EPYC 7501 32-Core Processor, family 23 model 1 stepping 2 |
| Sockets / physical cores / SMT | 2 / 64 / SMT2 (128 logical CPUs) |
| Microcode | `0x800126f` |
| NUMA | 8 nodes; node CPU lists: `0-7,64-71`, `8-15,72-79`, `16-23,80-87`, `24-31,88-95`, `32-39,96-103`, `40-47,104-111`, `48-55,112-119`, `56-63,120-127` |
| Process CPU affinity | `0-127` |
| Process memory-node mask | `0-7` |
| RAM / swap | 1.0 TiB / 8.0 GiB |
| Address-space/data/CPU limits | unlimited |
| Stack | 8192 KiB |
| Locked memory | 132093224 KiB |
| Open files | 1048576 |
| `numactl` | not installed |

No cgroup memory or cpuset file among the standard v1/v2 paths queried was
readable/present, so no additional cgroup limit was observed.

## Interrupted local run and scheduler state

The following read-only process check returned no matching worker or watcher:

```bash
ps -eo pid,ppid,lstart,stat,cmd --sort=start_time | \
  grep -E 'SUMMIT_gxe_real_40x5|reference_multi|watch_reference|gxe.*(score|cache|merge|fit)|hoffman_population_multi_env'
```

The stop marker remains:

```text
exit_code=254
finished_utc=2026-08-15T02:14:38Z
```

Only `reference_multi/reference.gxe.log` was present under the interrupted
reference output root (7,963 bytes). No reference bundle was present or
resumed.

At 2026-08-15T02:24:38Z, a read-only `ssh h2` query showed:

| Job | Environment | State |
|---:|---|---|
| 14355909 | sex | `hqw`, tasks 1-40, four slots/task, user hold |
| 14355910 | BMI | `hqw`, tasks 1-40, four slots/task, user hold |
| 14355912 | alcohol frequency | `hqw`, tasks 1-40, four slots/task, user hold |
| 14355913 | smoking status | `hqw`, tasks 1-40, four slots/task, user hold |
| 14354718 | independent age | tasks 1-33 running and 34-40 queued, four slots/task |

All four held arrays reported `task_concurrency: 0` and the expected scratch
working directory. No scheduler mutation command was issued.

## Baseline test and reproduction commands

The baseline collection contained 352 tests before the new reproducibility
test was added.

| Gate | Exact command | Initial outcome |
|---|---|---|
| Existing GxE tests | `/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q tests/test_gxe*.py` | PASS: 245 passed in 15.74 s |
| Existing full suite | `/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q` | PASS: 352 passed in 33.24 s |
| Named algebra/oracle gates | command below | PASS: 7 passed in 1.27 s |
| Independent-reference simulation | `OPENBLAS_NUM_THREADS=1 /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q -vv tests/gxe_completion/test_reference_transfer_simulation.py` | PASS: 1 passed in 0.19 s |
| Production-shaped GEMM fault stress | command below | PASS: 20 2B plus 20 4B repeated targets, zero repairs/retries |

The named algebra/oracle command was:

```bash
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q -vv \
  tests/test_gxe_summary_mom.py::test_summary_equations_equal_explicit_genie_kernels \
  tests/test_gxe_file_pipeline.py::test_file_bundle_reconstructs_explicit_individual_level_fit \
  tests/test_gxe_summary_mom.py::test_population_trace_transfer_is_identity_at_the_reference_rank \
  tests/test_gxe_summary_mom.py::test_population_same_person_probe_u_statistic_matches_dense_target \
  tests/test_gxe_native_core.py::test_native_feature_source_target_match_dense_oracle \
  tests/test_gxe_native_core.py::test_native_pass_probe_tile_and_thread_determinism
```

The checked-in independent-reference reproduction generated
`benchmarks/reference_transfer_simulation.json`. Its median absolute relative
errors were 2.1716%, 3.1353%, and 2.2305% for the same-/different-person
transfer versus 9.2679%, 86.4641%, and 23.5139% for naive squared-rank scaling,
reproducing the report.

The production-shaped stress command was:

```bash
taskset -c 0-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=32 \
  /home/bronsonj/anaconda3/envs/summit/bin/python scripts/gxe/benchmark_native.py \
  --n 291273 --m 1207 --probe-counts 32 --seed 20260809 \
  --repeats 1 --warmups 0 --decode-threads 32 --blas-threads 1 \
  --step-size 1207 --target-panel-columns 128 --workspace-gib 16 \
  --max-abs-error 1e-5 --stress-repeats 20 --skip-legacy \
  --skip-materialized \
  --json reports/gxe_completion/benchmarks/production_shape_b32_baseline.json
```

The maximum source, 2B-target, and 4B-target absolute errors were respectively
`2.3341e-10`, `2.2820e-6`, and `2.2820e-6`. The protected path reported
zero repaired columns and zero fresh-decode input retries. Single-run native
times were 0.745 s (source), 0.863 s (2B target), and 1.557 s (4B target).

## Import-path clarification discovered during validation

The conda environment contains both an editable finder that redirects Python
modules to `build/cp312-cp312-linux_x86_64` and an installed package copy.
Consequently `PYTHONPATH=src` alone did not override every editable mapping.
At the immutable starting revision this did not change the Python source
revision exercised by the baseline suite, but it matters immediately after an
edit. All post-change tests therefore used Python `-S`, appended conda
site-packages after importing `/home/bronsonj/SUMMIT/src/summit`, and explicitly
prepended `/tmp/summit-gxe-completion-build` to `summit.__path__`.

The production-shaped *baseline* JSON records the installed native binary
actually loaded by that command, SHA-256
`84aef4da73d3580304ca2d077d4dc120f3331ac0aacb95500e094f0dc362022b`.
The `51e20f...` value above is the separately audited repository build copy.
The post-change exact-shape artifact explicitly records the rebuilt `/tmp`
binary and its distinct hash. These artifacts must not be compared as if they
were the same binary.
