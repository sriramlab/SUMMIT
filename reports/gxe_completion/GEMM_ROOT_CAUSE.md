# GEMM corruption investigation

## Native phenotype-score addendum (2026-08-16 01:17 PDT)

The legacy population phenotype scorer exposed one remaining process-shared
NumPy BLAS path. On the full height-by-age score, its output contained sparse
clusters of incorrect additive scores. The largest discrepancy from the new
native result was `7.68395597e-4`; the mean absolute difference over all
454,207 variants was only `3.88487e-9`, which explains why aggregate fit
diagnostics did not reveal it. An independent one-thread dense calculation of
the 12 discrepant variants matched every new native score within `6.7e-12` and
showed legacy errors from `2.49e-6` through `7.68e-4`. The interaction score
maximum difference was `3.032e-9`.

This localizes those output errors to the superseded phenotype-score execution
path rather than the sealed genotype/reference/design inputs. It does not
identify the exact internal OpenBLAS event for those particular calls. The
replacement API-6 path uses the descriptor-owned decoder, the private static
OpenBLAS, its immutable 32-thread configuration, and serialized vendor entry.
Matched scoring applies the already validated sealed feature scales; population
transfer computes study-specific factored feature moments and NxE diagonals.
Both perform one fused product against `[Y,EY]` without materializing projected
`N x block` X/W panels.

The full population-transfer rerun used one genotype pass, completed in
223.878 seconds, and peaked at 5.391 GiB sampled process-tree RSS. It recorded
`gemm_integrity_enabled=false`, duplicate feature-moment verification false,
and zero repaired feature columns, repaired GEMM columns, or retried inputs.
The corrected fit remained full rank with condition 5.85605; component
proportions changed from the legacy fit by at most `1.78e-9`. This is evidence
for the private application boundary and this real workload, not a general
certification of upstream OpenBLAS.

## Runtime/affinity addendum (20:05 PDT)

The private-runtime correction below resolves the sparse GEMM corruption, but
the subsequent production delay had a distinct cause: the private OpenMP team
was left in active barrier-wait mode while NumPy executed serial feature
reductions. A debugger showed the main thread inside NumPy reduction code and
31 OpenBLAS/OpenMP workers spinning at a GOMP barrier. `OMP_WAIT_POLICY=ACTIVE`
therefore reported nearly 32 busy cores while accomplishing almost no useful
parallel work. Private symbol isolation and serialized vendor entry provide
the safety boundary; active waiting does not. Private-OpenMP builds now
recommend `PASSIVE`, and the CLI sets `GOMP_SPINCOUNT=0` with that policy.

The earlier two local workers were also not physically disjoint: CPUs 64--95
are SMT siblings of CPUs 0--31 on tabla. In addition, the former
`force_affinity_all=true` default could expand either worker back to all online
CPUs after `taskset`. The default is now false. A supported two-process mode,
when selected, assigns one hardware thread from each physical core on socket 0
to one independent environment group and socket 1 cores to the other. The
processes share neither native BLAS state nor cores, and their NUMA policies are
applied independently. This process-isolated design is not the unsafe
concurrent-entry reproducer described below.

## Decision

**The two application-level triggers are identified and removed from the
supported private-runtime path.** The original large-GEMM corruption occurs
when multiple SUMMIT application threads enter the same pthread OpenBLAS
runtime concurrently. The real-cohort launch exposed a second failure after
repeated one-thread-to-32-thread mutations of that process-global runtime.
These explain the corruption. Two idle pools explain the separate throughput
loss, not sparse output damage. The failures are not specific to Zen dispatch,
adjacent output partitions, genotype decoding, or float32 storage. The exact
faulty statement inside OpenBLAS remains unlocalized.

Production Linux builds now embed a symbol-hidden private OpenBLAS in
`gxeldcore`, set its thread count exactly once, reject later changes, and
serialize every vendor entry. NumPy cannot resolve its OpenBLAS symbols. An
OpenMP private archive shares one worker runtime with decoder loops; a pthread
archive uses passive OpenMP waiting. Internal one-thread BLAS contexts and
1-to-N resets were removed. In private mode the checksum/fingerprint/
repair/rerun machinery is compiled out by default. The read-only `mprotect`
operand seal and structural ownership/dimension checks remain. If only a
process-shared OpenBLAS is available, the integrity implementation remains on
and CMake rejects an unchecked build.

## Optimization level

The native extension was already a Release build with `-O3 -march=native`.
`-O2` appeared only in the first standalone diagnostic build; sanitizer builds
used lower optimization to remain instrumentable. The decisive reproducer and
the final extension both use `-O3 -march=native`:

```bash
/usr/bin/g++ -O3 -march=native -DNDEBUG -std=c++17 -pthread \
  -I/home/bronsonj/anaconda3/envs/summit/include \
  tests/gxe_completion/gemm_concurrent_reproducer.cpp \
  -L/home/bronsonj/anaconda3/envs/summit/lib \
  -Wl,-rpath,/home/bronsonj/anaconda3/envs/summit/lib \
  -lopenblas -o /tmp/gemm_concurrent_reproducer_o3
```

## Minimal reproducer and caller audit

`tests/gxe_completion/gemm_concurrent_reproducer.cpp` is independent of Python,
NumPy, OpenMP, and genotype decoding. It creates deterministic binary64 inputs,
uses the production column-major transpose conventions, guards every allocation,
fingerprints A and B, and reports corruption frequency and location.

The audited dimensions and leading dimensions are:

| Product | `(m,n,k)` | Layout |
|---|---:|---|
| source NN | `(291273,64,1207)` | `lda=m, ldb=k, ldc=m` |
| target TN | `(1207,128,291273)` | `lda=k, ldb=k, ldc=m` |

All element counts and partition offsets use `size_t` or 64-bit intermediates.
A, B, reference C, and result C are disjoint. `beta=0`. LP64 headers and library
agree. Reduced ASan+UBSan and TSan caller tests were clean.

## Controlled results

Runs below were executed one process at a time. The threshold for a gross error
was `1e-10`; ordinary internal-reduction differences were about `3e-14`.

| Shape and mode | Faulty calls | Max abs. error | Inputs/guards | Time |
|---|---:|---:|---|---:|
| target, 32 concurrent Zen calls | 6/200 | 0.3557 | unchanged/intact | 107.0 s |
| target, one 32-thread Zen call | 0/200 | `2.84e-14` | unchanged/intact | 77.7 s |
| target, 32 concurrent Haswell calls | 17/200 | 33.14 | unchanged/intact | 151.4 s |
| source, 32 concurrent Zen calls | 4/100 | 3.256 | unchanged/intact | 50.9 s |
| source, one 32-thread Zen call | 0/100 | 0 | unchanged/intact | 28.3 s |

The first controlled target failure affected only eight cells across rows
544--547 and columns 60--64. The first source failure affected three cells in
one row across columns 0--8. Sparse output damage with bitwise-stable inputs and
intact red zones is inconsistent with a caller-wide dimension or leading-
dimension error.

Additional controls were decisive:

- column partitioning also corrupted;
- guarded private output for every row worker also corrupted, excluding overlap
  or a microkernel tail store into an adjacent worker's C;
- a global mutex was clean in the bounded exact-shape trial;
- Haswell dispatch failed at longer count, excluding a Zen-only conclusion; and
- a controlled OpenBLAS 0.3.34 `USE_THREAD=0 USE_LOCKING=1 TARGET=ZEN` build
  still corrupted, so absence of `USE_LOCKING` is not a sufficient explanation.

OpenBLAS 0.3.31 release notes separately document the reversion of a 0.3.30
GEMM partitioning optimization that could race and produce invalid results.
OpenBLAS usage documentation recommends one BLAS thread per call for externally
multithreaded programs, but that documented pattern is demonstrably unsafe with
this workload/runtime combination. See the upstream
[usage guide](https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md),
[release history](https://github.com/OpenMathLib/OpenBLAS/releases), and
[OpenMP/pthread handover FAQ](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/faq.md#openmp).

EDAC corrected/uncorrected counters were zero for all eight observed controllers.
That is supporting host evidence, not a hardware-stress exclusion.

Machine-readable results are in
`benchmarks/concurrent_gemm_stress_o3.json`.

## Production correction

`src/native/gxeldcore.cpp` now uses:

1. a symbol-hidden private static OpenBLAS archive;
2. one initialization-time `openblas_set_num_threads` call;
3. an immutable process-lifetime BLAS count, distinct from decoder threads;
4. one static mutex around every vendor GEMM entry; and
5. one whole internally threaded NN or TN DGEMM.

The feature-moment contraction was also routed through this executor after a
final caller audit found it bypassing the mutex. API 4 records
`gemm_execution_mode=serialized_fixed_private_openblas`, runtime isolation,
thread count/layer, archive SHA-256, and whether integrity code is compiled.
`[S,eS]` remains in a Linux read-only mapping. Common-covariate corrections use
an in-place disjoint-output native rank update.

Final source snapshot SHA-256 is
`d1047f3b082da5a1fdfca17d3f67ce98c0ceea2627b453d90174d7d77c6635ff`;
the build-tree/staged extension SHA-256 values are
`7a927a41f9a685cb34ddf6291501f32b2349372070b19b0ba0da219a92abb8ea`
and `2089f4146d0474e9fd7d925e73a445db7ab527f239afba4af04ad83d6400fd93`.

At `N=291273, M=1207, B=32`, the private OpenMP build completed 20 repeated
source, 2B-target, and 4B-target calls. Every repeat was bitwise identical to
its first result. Source error was `2.3341e-10`; both target errors were
`2.2823e-6` against the independent dense float64 oracle. The binary reported
`gemm_integrity_enabled=false`; repair/retry counters remained zero for schema
compatibility but those execution paths were absent. See
`../gxe_private_openblas/production_b32_private_openmp_active_stress20.json`.
The edited-source staged build passed 70 focused and 373 full tests.

## Real-cohort thread-pool handover reproducer

The first fresh five-environment launch passed source/input/binary preflight and
constructed all five 289,111-row estimators, then rejected the valid collinear
age projector before genotype streaming. An exact in-memory reproducer under a
32-thread outer pool failed twice, on age and then sex. The identical matrices
under one BLAS thread had projector-span residual norms `7.3e-15` to `1.1e-14`,
well below the `9.4e-9` gate; all input fingerprints remained bitwise stable.
A delayed threaded repetition was clean after the pool quiesced. This evidence
implicates the immediate OpenBLAS pool handover rather than rank tolerance or
input collinearity.

After the serial factorization boundary was added, the exact five-environment
reproducer passed under the same 32-thread outer state. It correctly returned
zero new directions for age and sex and unit directions for BMI, alcohol, and
smoking, and restored the outer pool to 32 threads afterward.

## Performance and remaining gate

The private OpenMP archive is a lean Zen, double-precision, CBLAS-only build and
shares libgomp with the decoder. In an initial active-wait exact-shape run,
source, 2B target, and 4B target took 0.654, 0.773, and 1.318 seconds. The prior
guarded build took 1.004, 1.007, and 2.275 seconds for the same hot paths, so
combined time fell about 36%. A pthread-private control remained deterministic,
but active OpenMP waiting caused severe starvation between vendor calls. CMake
now installs `PASSIVE` for the private OpenMP build; the CLI additionally sets
`GOMP_SPINCOUNT=0`. Pthread/shared fallbacks also remain passive and retain
`OPENBLAS_THREAD_TIMEOUT=1`.

The 10,000/100,000-call, second-host, and independent-BLAS certification matrix
was not completed. This does not reopen the application race: private symbols,
fixed lifetime configuration, and serialized entry remove both demonstrated
triggers. It does limit the claim to the tested host/toolchain rather than an
upstream OpenBLAS certification. B=1024 remains a separate scientific,
uncertainty, and end-to-end resource decision.
