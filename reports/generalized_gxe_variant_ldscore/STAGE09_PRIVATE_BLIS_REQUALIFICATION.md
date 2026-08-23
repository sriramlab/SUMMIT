# Stage 09 private-BLIS FP64 requalification

Date: 2026-08-22

Repository: `/home/bronsonj/SUMMIT`

Native candidate commit: `dc84b199e80050037c4193be2a0801a73d034573`

Scope: generalized per-variant GxE LD scores on descriptor-owned PLINK BED

## Verdict

**GO** within the qualified FP64 backend and topology envelope below.

The earlier `NO_GO` was caused by material GEMM corruption in the tested shared
OpenBLAS path, not by the generalized estimator's two-pass science. The existing
non-general GxE resolution applies here: use the pinned private pthread-BLIS
runtime and keep vendor thread ownership immutable from process start. On that
backend, the current generalized candidate completed the real full-variant,
production-row, `B=128`, and `B=1024` controls with exactly two passes, exactly
`2*M` retained-variant visits, and zero duplicate, retry, repair, fallback, or
integrity-failure counts.

The changed-`J` invariance experiment has been removed as a requirement. SNP
block deletion is an inference-time operation over fixed reference summaries;
reference construction accumulates block target-row numerators and masses but
does not run jackknife fits or materialize delete-block Gram cubes.

## Frozen native identity

| Field | Accepted value |
|---|---|
| Source commit | `dc84b199e80050037c4193be2a0801a73d034573` |
| Source-tree SHA-256 | `7e1641b2669467457bc656b6c057bf983354ee658b1861e8846121b6e9068449` |
| `gxeldcore` SHA-256 | `1a6e05733e81777092773c4e66171b1fc5a53566a0195b18aa6b0b4f3d3b2c7e` |
| Compiler/build | GNU 12.2, C++17, Release, `-O3 -march=native` |
| BLAS | private static upstream BLIS, `zen`, pthreads |
| BLIS commit | `e8566eb3e773fb54d11b33e371d13f22d2941e50` |
| BLIS source-tree SHA-256 | `eefbd29a5cbb1d6982bdce76e8034037ea2a3f3ead33d90d9f3cef6da5728154` |
| BLIS archive SHA-256 | `720068171eea951a0bc634d2d1a829561d5a2bae630bce41a24c4f0edbef9d9b` |
| Execution mode | `serialized_fixed_private_blis` |
| Integrity | protected inputs and independent audits; checksum fallback disabled |

The installed extension is 2,992,744 bytes. `ldd` shows no dynamic BLAS or
BLIS dependency, and `nm -D` shows no exported or unresolved `cblas_*`/`bli_*`
symbols. `build_info()` records private-static isolation, pthreads, immutable
environment ownership, and enforced owner-thread entry.

The exact qualified installation is
`/tmp/summit-generalized-private-blis-release.JniFXb/install`. On this host its
temporary CMake install has a base-Conda RUNPATH, so qualification commands
preload the active `summit` environment's `libstdc++.so.6`. Production builds
must be configured from the intended runtime environment so their RUNPATH and
C++ runtime agree.

## Current-build execution evidence

All measured controls used `Q=3`, `K=1`, `J=200`, dense FP64 protected NN/TN,
variant blocks of 4,096, probe tiles of four, 32 immutable BLIS threads, 32
OpenMP threads, and a socket-local 32-CPU taskset.

| Control | Wall time | Passes / visits | Reads | Independent audits | Peak RSS | Clean ledger |
|---|---:|---:|---:|---:|---:|---|
| `N=2,048,M=10,000,B=128` | 4.0898 s median (3.9575--4.1770) | 2 / 20,000 | 6 | 4 | 217,305,088 B | yes; four executions bitwise equal |
| `N=2,048,M=10,000,B=1024` | 31.8586 s | 2 / 20,000 | 6 | 24 | 250,146,816 B | yes |
| Real `N=9,401,M=454,207,B=128` | 343.8146 s | 2 / 908,414 | 222 | 112 | 677,158,912 B | yes |
| `N=300,000,M=4,096,B=128` | 96.2767 s | 2 / 8,192 | 2 | 2 | 11,474,591,744 B | yes |

Every row has zero checksum-recomputed columns, roundoff-only columns,
duplicates, retries, repairs, fallbacks, and integrity failures.

Additional current-build controls completed as follows. These were correctness
and work-ledger smokes, not an uncontended performance selection sweep:

| Control | Wall time | Reads | Clean ledger |
|---|---:|---:|---|
| One-thread `N=2,048,M=10,000,B=128` | 14.7211 s | 6 | yes |
| Eight-thread `N=2,048,M=10,000,B=128` | 4.5654 s | 6 | yes |
| Alternative width-512/probe-tile-8 plan | 18.9767 s | 40 | yes |
| `N=1,024,M=2,048,Q=4,K=4,B=32,J=16` | 4.0357 s | 8 | yes |

The alternative tiling and larger `Q,K` controls retained exactly two passes
and `2*M` visits. Their evidence digests are, respectively,
`1fde6f957c764c29c68d6613261e73dc2643b0e0b3756834f5e3b4eb99ab1bb7`
and `494a5ef4aa5878d8fa1b2562e263c7065d41c2ec22df733f24d1f7db8d406d22`.
The one- and eight-thread evidence digests are
`c2d987303817fb352a076b089ef116c7480591e3d8d0a6e3b0acc88b1a573222`
and `86162ad8b9782ca22c6e264d3cf7df6dd980f2c616638a7b99d80097e9a3998e`.

The current real full-variant run read 2,135,681,314 logical genotype bytes and
published 159,751,848 native output bytes. Its phases were 180.7594 seconds for
pass 1, 0.4066 seconds for the hard barrier, 162.4552 seconds for pass 2, and
0.0894 seconds for finalization. Recorded subphases include 44.5953 seconds of
decode, 128.7815 seconds of source NN, 131.0942 seconds of target TN, 12.3195
seconds of row-product reduction, and 22.0895 seconds of hashing/audits.

The current production-row run read 614,400,000 logical bytes and published
922,840,008 native output bytes. Its phases were 39.1111 seconds for pass 1,
14.9994 seconds for the barrier, 40.3503 seconds for pass 2, and 0.3445 seconds
for finalization. The measured peak remained below the admitted 64-GiB budget.

Evidence JSON and SHA-256 digests:

- `real-m454207-b128-dc84b19.json`:
  `7e326ee50592416e7fb0f5278f4c45585c61ae6710fb81934c6c2ccff0e13db9`;
- `primary-n300000-m4096-b128-dc84b19.json`:
  `c9a4125ff3ba7672fc45a976e93f5b4a16c2591d9c9748464b8d85b00e2b3a64`;
- `b128-dc84b19.json`:
  `f06f25fc500ec9033e744196318f0839ae23af7fd34db6e8f3361e59f3a86e85`;
- `b1024-dc84b19.json`:
  `2111c506f2a05ad874acf88bbaaf2a9dcc5667bb825040a9b94651c57babe4a2`.

The complete real source was
`/home/bronsonj/SUMMIT_gxe_pilot_20260808/validation/geno/age_dbp_cc`.
The production-row fixture was
`/tmp/summit-generalized-stage08-benchmark-data/primary_partial_n300000_m4096`.

## Target planner and dimension limitation

The exact combined `N approximately 300,000,M approximately 1,000,000` source
was not locally available. A planner-only `N=300,000,M=1,000,000,Q=3,K=1,
B=128,J=200` configuration with 64 GiB records:

```text
planned passes                         2
planned retained-variant visits        2,000,000
variant blocks per pass                245
leading FP64 work                      768,000,000,000,000 flops
modeled peak resident bytes            13,173,684,173
directional-panel output bytes         288,000,000
```

The complete real variant-axis and production-row controls separately validate
the two large dimensions with measured execution. This supports production
admission but does not claim a measured end-to-end runtime for their unavailable
Cartesian combination.

`B=1024` is feasible without changing the two-pass architecture. Its larger
probe axis changes resident sources and GEMM work, not genotype traversals.

## Correctness, reliability, and regression results

| Scope | Result |
|---|---|
| Release generalized oracle/contracts/pass 1/pass 2/native/artifact/fit/CLI/benchmark plus BLIS contract | 122 passed, 1 expected skip |
| Current mature non-general native-core regression under its immutable two-thread BLIS envelope | 38 passed |
| Current ASan+UBSan private-BLIS native/BLIS scope | 13 passed |
| Current UBSan-only private-BLIS native/BLIS scope | 13 passed |
| Authoritative package check | `PACKAGE_CHECK_OK` |
| Mathematical oracle | 10 passed |

The expected skip is the checksum-repair diagnostic: the accepted private-BLIS
backend disables numerical checksum/recompute fallback and therefore has no
repair path to exercise. Deterministic pass-1 and pass-2 fault injection still
fails closed before publication. Tests also cover descriptor mutation,
single-use native contexts, dense versus packed results, fixed global probes,
artifact truncation/corruption, atomic publication cleanup, inference-time
deletion, compatible trait scoring, full and leave-one-block-out fits, and
jackknife covariance construction.

The native executor now records per-variant affine means and inverse sample
standard deviations during the required pass-1 traversal, seals their hashes at
the barrier, and reproduces the hashes in pass 2. No extra decode or traversal
is introduced. The artifact adapter rejects any requested genotype-scale plan
that differs from this decoder-derived identity.

## Reproducible artifact example

The checked production example is
`example/estimate_generalized_gxe_variant_ldscore.py`. On the repository
`example/small` BED fixture it completed `N=8,430,M=14,821,Q=3,K=1,B=16,J=20`,
29,642 visits, and 15 reads per pass with a clean ledger. It atomically wrote
and reloaded a V1 artifact containing the full `[14821,6,6]` directional panel.
The retained-panel artifact was 4,135,603 bytes. A second clean execution with
the panel omitted wrote the same closed aggregate-summary family in 15,799
bytes, with `per_variant_panel.storage="omitted"`; inference-time deletion uses
neither form of the optional panel.

```bash
env BLIS_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 \
  python example/estimate_generalized_gxe_variant_ldscore.py \
  --output /tmp/summit-generalized-example \
  --probes 16 --blocks 20 --threads 4
```

The installed command
`summit-generalized-gxe-variant-ldscore` provides target-shape planning and
strict artifact inspection.

## Accepted scope and residual limitations

- Accepted execution is Linux x86-64, descriptor-owned PLINK BED, FP64, and the
  exact pinned private pthread-BLIS identity. PGEN is planner-only in this V1
  generalized native path and is not production-qualified.
- BLIS thread count and ownership must be fixed at process start. A request that
  disagrees with the sealed team fails rather than reconfiguring the vendor.
- The 32-CPU measurements used taskset and OpenMP placement. Strict early NUMA
  allocation binding was not requested, so complete NUMA locality is not
  claimed.
- The full combined million-variant/300,000-sample runtime remains unmeasured
  because that exact source was unavailable.
- Mixed precision is not required for release and was not opened. FP64 is the
  accepted scientific path.

Within these explicit boundaries, no unresolved scientific or reliability
stop gate remains.
