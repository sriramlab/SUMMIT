# Stage 08 production optimization and integrity

## Decision

**PASS for the integrity-qualified dense production path.** The generalized
variant-probe estimator retains one descriptor-owned global-source pass, one
hard barrier, and one descriptor-owned target-scoring pass. The accepted path
has no complete serial output witness, no tile-induced descriptor reread, no
duplicate retained-variant visit, measured phase and GEMM telemetry, bounded
native-output NUMA evidence, and fail-closed corruption detection before
publication.

The isolated `CODEX_LAUNCH_PROMPT.md` manifest mismatch in the authoritative
package remains the non-scientific, non-executable packaging discrepancy
documented and explicitly waived in Stage 00. The package mathematical oracle,
scientific documents, executable scripts, source archive, and all other
manifest entries passed. The mismatch was not modified and is not an estimator
or Stage 08 qualification failure.

## Settled candidate

- implementation and benchmark commit:
  `120b0ad3ea43d1bcc094e88c78c410ed45014e96`;
- embedded source-tree SHA-256:
  `cbf0d2b1279896cdb010e25ab127755522fd5ddd2ac0543ef97cd37efa9a0261`;
- Release root:
  `/tmp/summit-generalized-stage08-release.aYHX4I`;
- Release `gxeldcore` SHA-256:
  `c77facb6a1181a0b9867bbb9f2d89fd48e30e07f7b3a91f38c65dbba461d095b`;
- ASan+UBSan root and `gxeldcore` SHA-256:
  `/tmp/summit-generalized-stage08-asan.rkCgJc`,
  `826c3b81d24f6e5638f4fbf09b5688991d109f4f6aac2f07c6b9ecb345fbfa5d`;
- UBSan-only root and `gxeldcore` SHA-256:
  `/tmp/summit-generalized-stage08-ubsan.mSOwBP`,
  `1d2c334b0fa761d417adcd9acf5684f371e3388be30e2f08589ec05da7085123`.

The Release build reports GNU 12.2, C++17, `-O3 -march=native`, OpenBLAS
0.3.34 pthreads (`DYNAMIC_ARCH`, Zen kernel, maximum 128 threads), GNU OpenMP,
protected GEMM checksums and integrity enabled, and sanitizer mode `none`.
Both sanitizer builds use portable tuning, `-O1`, fatal sanitizer recovery,
and the same clean source identity.

## Measurement host and limits

| Property | Observed value |
|---|---|
| CPU | 2-socket AMD EPYC 7501, 32 physical cores per socket, 2 threads per core, 128 logical CPUs |
| NUMA | 8 nodes; physical CPUs 0--31 on nodes 0--3 and 32--63 on nodes 4--7; approximately 126 GiB per node |
| Memory | 1.0 TiB installed; 992 GiB available before the primary partial run; 8 GiB swap |
| Memory controllers | `/sys/devices/system/edac/mc` exposed `mc0`--`mc7`; exact DIMM/channel wiring was not exposed without privileged firmware data |
| Frequency policy | boost enabled; `ondemand` governor; 1.2--2.0 GHz reported range |
| Huge pages | transparent huge pages `always`; defrag `madvise`; zero reserved hugetlb pages |
| Filesystems | `/tmp` on ext4 `/dev/sda2`; repository and benchmark source on XFS `/dev/sda4` |
| Placement | every benchmark used `taskset`, singleton `OMP_PLACES`, immutable `OMP_PROC_BIND=close`, and native exact-team attestation |
| Counters | `perf` and `numactl` were absent; no bandwidth, cache, scheduling, false-sharing, or remote-memory cause is claimed |

Normal benchmark runs did not request the strict early-membind contract. Their
bounded native-output records therefore correctly say
`contract_required=false`, `allocation_mode=legacy_posix_memalign`, and
`complete=false`; they are allocation evidence, not page-locality proof. The
mature strict contract, bound-before-first-touch output allocation, complete
page query, and fail-closed placement checks passed the dedicated native NUMA
tests. Production must use that mature authenticated early-NUMA launcher when
strict placement is required.

## Accepted mechanisms

| Mechanism | Hypothesis and before evidence | After evidence | Correctness and memory effect | Decision |
|---|---|---|---|---|
| Decode/tile structure audit | A tile must not cause a reread or second decode. Stage 07 already decoded outside probe/annotation loops but had no closed measured ledger. | For `M=2,048`, four blocks per pass produced 8 block reads, 4 pass-1 decodes, 4 pass-2 decodes, 2 passes, and 4,096 visits. Native telemetry records zero tile rereads and zero duplicates. | All invariance tests pass. No additional resident genotype copy was introduced. | **Keep.** |
| Hoisted immutable work and phase-local arenas | Repeated annotation square roots and hot-loop output allocation were avoidable. | Annotation square roots are sealed once. Genotype, probe, RHS, reduction, and reusable mature protected-output allocations reserve phase capacity outside inner work. There are no Python callbacks in the two native passes. | Fixed-probe dense/oracle and backend comparisons remain within `3e-13`. Dense output capacity replaces, rather than duplicates, the prior vector output. | **Keep.** |
| Measured telemetry | Stage 07 native-to-artifact adaptation used placeholder zero phase timings. | Schema v2 records wall/CPU time for pass 1, barrier, pass 2, and finalization; nine subphases; actual GEMM shapes, calls, logical FLOPs, and throughput; logical genotype bytes; peak RSS; output/resident bytes; affinity; integrity; and bounded first-64 output NUMA records. Artifact performance ledgers now come only from these measured fields. | Native-to-artifact validation requires the closed telemetry policy. Publication remains atomic. | **Keep.** |
| Probe tile width | Wider calls might amortize dispatch, but pass-2 row reduction and RHS working-set growth could reverse the result. | On `N=1,024,M=2,048,Q=3,K=1,B=32`, the common screening medians for widths 4/8/16/32 were 0.14935/0.16635/0.18090/0.19057 s. A final width-4 confirmation was 0.11838 s, range 0.11835--0.13378 s. | Cross-tile outputs pass the declared FP64 tolerance; each repeated fixed configuration was bitwise stable. Width 4 reduces RHS capacity relative to wider tiles. | **Keep width 4 on this host.** Re-screen on a different BLAS/CPU. |
| Redundant generalized full audit removal | Large calls already receive the mature protected algebraic checksum. A second two-projection scan of the same clean operands was expected to dominate narrow production-shape GEMMs. | On `N=300,000,M=4,096,Q=3,K=1,B=4`, median time fell from 25.2431 s (25.0277--27.4353) to 14.1625 s (13.8789--14.3663), a 43.90% reduction. Hashing/audits fell from 12.7186 to 0.3902 s. The large source and target calls recorded two mature protected audits and zero independent clean-path audits. | No arithmetic product or reduction changed. Small sub-threshold calls still use two independent projections on the first and every 64th call. Qualification injection always forces the independent post-injection audit. No full serial-output witness exists. | **Keep.** |
| Integrity fault injection | A corrupted source or target product must fail before it contributes to a sealed source or public result. | Deterministic pass-1 NN and pass-2 TN injections each raise the expected checksum error, set terminal state `failed`, record one integrity failure, publish no result, and reject a second run as single-use. | Clean runs recorded zero retry, repair, fallback, and failure. ASan+UBSan and UBSan-only ran both fault cases cleanly. | **Keep qualification-only injection.** |
| Dense versus packed screen | The mature packed representation might win when genotype expansion dominates. | At `N=1,024,M=2,048,B=32`, dense/packed medians were 0.12217/0.11310 s. A one-run primary partial screen was 14.1625 s dense versus 5.4643 s packed and about 10.47 GiB versus 0.97 GiB peak RSS. | Packed and dense remain scientifically equivalent within the Stage 06 `3e-13` gate. The packed screen recorded no mature protected or independent product audit, so it does not meet this stage's production integrity gate. | **Dense is the production default.** Packed remains available but is not promoted by Stage 08. |
| Disjoint annotation compaction | Exact zero/disjoint annotations might justify a compact schedule. | With `Q=3,K=4,B=8`, overlap/disjoint medians were 0.12959/0.13227 s with overlapping ranges. | No scientific layout specialization was added, and no performance benefit is claimed. | **Do not add compaction.** |
| Thread/socket policy | More cores might help target TN calls, but cross-socket placement could amplify synchronization or locality costs. | The topology table below selects a 32-core socket-local process. | All one-, eight-, 32-, and 64-thread fixed configurations produced the same recorded output digest in their settled runs. No hardware-counter diagnosis is made. | **Keep socket-local 32-core policy on this host.** |

The row-product path already forms each target row's products and immediately
updates the per-SNP directional panel, full `DNUM`, and segmented `BDNUM` while
the block is resident. It does not materialize a second full cross-product
panel. Per-SNP artifact inclusion remains optional and bounded by the existing
atomic artifact layer; no asynchronous shard writer was added.

## Benchmark ladder

All measurements used one warm-up followed by three measured executions unless
explicitly marked as a screen. Timings exclude one-time BED generation because
the benchmark timer begins after the input exists.

| Fixture | Placement/configuration | Median seconds | Measured range | Peak RSS | Exact ledger |
|---|---|---:|---:|---:|---|
| Tiny oracle-format BED, `N=64,M=128,B=8` | 1 physical core, dense, width 4 | 0.001181 | 0.001170--0.001187 | 106.9 MB | 2 passes, 256 visits, 8 reads, 0 retry/repair/fallback/failure |
| Moderate BED, `N=1,024,M=2,048,B=32` | 8 cores, dense, width 4 | 0.118385 | 0.118347--0.133784 | 113 MB class | 2 passes, 4,096 visits, 8 reads, 0 retry/repair/fallback/failure |
| Real-format/realistic-missingness BED, `N=2,048,M=10,000,B=8` | 8 cores, dense, width 4, overlap annotations | 0.373194 | 0.365231--0.387462 | 138.4 MB | 2 passes, 20,000 visits, 20 reads, clean |
| Same, `B=32` | 8 cores, dense, width 4 | 0.839208 | 0.777613--1.113309 | 138.8 MB | 2 passes, 20,000 visits, 20 reads, clean |
| Primary-row partial, `N=300,000,M=4,096,B=4` | 32 socket-local cores, dense, width 4 | 14.162506 | 13.878893--14.366269 | 10.468 GB | 2 passes, 8,192 visits, 2 reads, 2 protected audits, clean |

The real-format fixtures are PLINK SNP-major BED/BIM/FAM with 0.5% seeded
missing calls and overlapping or strict-disjoint annotation layouts. They are
synthetic, not a claim of a biological cohort measurement.

The primary partial planner predicted 11.805 GB peak against 10.468 GB observed.
Its explicit resident terms were 9.830 GB decoded genotype, 86.4 MB maximum
RHS, 28.8 MB contextual sources, 9.6 MB base sources, 1.18 MB output, 268.4 MB
thread headroom, 67.1 MB telemetry/publication headroom, and 1.496 GB allocator
headroom. It read exactly 614.4 MB logically across the two 307.2 MB passes.
The pre-optimization run's 21.0 GB process peak is excluded from memory
comparison because that process had just generated and retained the 9.8 GB raw
BED-construction array; its execution timing remains valid.

### Topology screen

These runs used `N=2,048,M=10,000,Q=3,K=1,B=32`, width 4.

| Physical CPUs | Placement | Median seconds | Range |
|---|---|---:|---:|
| 0 | one core | 3.76917 | 3.73357--3.79027 |
| 0--7 | one NUMA node | 0.81190 | 0.79159--0.82330 |
| 0--31 | socket 0 | 0.57215 | 0.54956--0.65541 |
| 32--63 | socket 1 | 0.59240 | 0.56676--0.62822 |
| 0--63 | both sockets | 1.05201 | 0.87279--1.14682 |

The first 64-core screen observed more than one byte digest across its warm-up
and measured repetitions; an independent rerun was bitwise stable and matched
the one-, eight-, and 32-core digest. Byte identity is therefore reported but
is not the scientific gate. Cross-thread correctness uses the declared
`3e-13` FP64 tolerance, which passed.

An additional socket-0 run recorded 0.56151 s and actual generalized GEMMs:

| Operation | Calls and dimensions | Logical work | Achieved throughput |
|---|---|---:|---:|
| pass-1 source NN | 80 calls, `(2048 x 1000) (1000 x 4)` | 1.311 GFLOP | 13.14 GFLOP/s |
| pass-2 target TN | 80 calls, `(2048 x 1000)^T (2048 x 36)` | 11.796 GFLOP | 56.31 GFLOP/s |

The public mature protected kernels on the same operands, call counts, and
placement measured 16.02 GFLOP/s NN and 55.89 GFLOP/s TN. The generalized
target kernel is therefore equivalent to the mature target throughput at this
shape. No cache or bandwidth explanation is assigned to the smaller source
difference because hardware counters were unavailable.

At the primary partial shape, the actual source NN was one
`(300000 x 4096) (4096 x 4)` call at 4.12 GFLOP/s and the target TN was one
`(300000 x 4096)^T (300000 x 36)` call at 30.08 GFLOP/s. Decode took 7.625 s
across both passes, RHS formation 0.045 s, target row reduction 0.008 s,
projection 0.042 s, same-person 0.098 s, publication 0.004 s, and
hashing/audits 0.390 s in the recorded final execution.

### Primary and stress plans

Dry-run plans used `N=300,000,M=1,000,000,J=200`, a 4,096-variant block,
width 4, 64 requested threads, and a 900 GiB budget.

| `Q,K,B` | Leading FLOPs | Planned peak | Output | RHS columns | Passes/visits |
|---|---:|---:|---:|---:|---:|
| `3,1,128` | 768.0 TFLOP | 13.442 GB | 288 MB | 1,152 | 2 / 2,000,000 |
| `3,1,1024` | 6.144 PFLOP | 23.334 GB | 288 MB | 9,216 | 2 / 2,000,000 |
| `4,8,128` | 10.445 PFLOP | 26.486 GB | 6.4 GB | 16,384 | 2 / 2,000,000 |

Using the measured primary-dimension dense kernels, the `B=128` leading source
and target families project to 5.18 and 6.38 hours respectively. Linear decode
scaling adds approximately 0.52 hours; smaller measured phases make the total
projection about 12.2 hours. This is an inference, not a full-target runtime
claim. A full `M=1,000,000,B=128` execution was not started because its
projected CPU time was not a bounded qualification run. The exact planner and
one complete target-row/target-block partial execution cover the unavailable
full-run case without an open-ended job.

## Production configuration

| Setting | Settled value on this host |
|---|---|
| estimator | generalized per-variant directional G×E LD score, variant-axis probes |
| backend | descriptor-owned dense FP64 protected NN/TN |
| genotype traversals | exactly two complete passes with a hard barrier |
| genotype scale | one sealed mean-imputed affine FP64 scale shared by all `F_q` |
| primary dimensions | `Q=3,K=1,B=128`; dry-run larger `B,Q,K` before execution |
| variant block | 4,096, subject to the closed planner and actual memory budget |
| probe tile | 4 on EPYC 7501/OpenBLAS 0.3.34; re-screen after CPU/BLAS change |
| threads | 32 physical cores, one socket, no SMT, immutable OpenMP/BLAS count |
| affinity | explicit singleton places and native exact-team attestation |
| NUMA | mature early bind aligned to the selected socket when strict placement is requested; fail closed on incomplete evidence |
| integrity | mature checks on every threshold-eligible GEMM; two-projection periodic audits for smaller calls; qualification injection forces an independent audit |
| clean counters | retry 0, repair 0, fallback 0, integrity failure 0, duplicate visit 0 |
| jackknife | subtract fixed target rows from full `DNUM`; never recompute retained LD scores; reuse full same-person matrix |

The mature multi-process non-general estimator is not used as a hidden
generalized execution mode. Introducing it here would require a separately
qualified partition contract that still proves exactly two aggregate genotype
traversals. No such change was needed for this stage.

## Qualification

| Scope | Result |
|---|---:|
| focused generalized native/artifact/fit/CLI | **40 passed in 3.90 s** |
| benchmark planner plus generalized native | **14 passed in 6.11 s** |
| ASan+UBSan generalized native, including two fault injections | **11 passed in 5.08 s** |
| ASan+UBSan mature seven-file native scope | **83 passed, 1 skipped in 11.67 s** |
| UBSan-only generalized native | **11 passed in 3.33 s** |
| UBSan-only mature seven-file native scope | **83 passed, 1 skipped in 7.48 s** |
| full clean Release repository | **1704 passed, 6 skipped, 1 XPASS in 149.53 s** |

ASan+UBSan used
`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0`; both modes used
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. GCC 12 `libasan`, the Conda
`libstdc++.so.6`, and the linked Conda OpenBLAS were preloaded so isolated
`python -S` subprocess tests used the same runtime. No address- or
undefined-behavior diagnostic was emitted. LeakSanitizer and full-repository
sanitizer execution are not claimed.

The Release suite's six skips and one XPASS are repository-level outcomes. No
generalized Stage 08 test skipped. Required invariance coverage includes fixed
probe identity; block/probe/RHS/thread/backend tolerance; the exact two-pass
ledger; per-SNP rows, `DNUM`, `BDNUM`, Grams, same-person, fit, and standard
error comparisons; descriptor/lifecycle failures; corruption injection; and
mature old-path regressions.

## Stop gate

The stop gate passes. The clean dense FP64 path has zero full serial-output
witnesses, zero tile rereads, zero retry/fallback in qualification, measured
socket-local throughput, bounded memory below the closed plan, mature and
independent checksum coverage, successful pass-1 and pass-2 fault detection,
unchanged scientific results within the declared FP64 tolerance, and a clean
full repository regression. Stage 09 may begin from commit `120b0ad` plus this
report commit.
