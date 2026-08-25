# Stage 09 FP64 production qualification

> Historical record: this shared-OpenBLAS `NO_GO` is superseded by
> `STAGE09_PRIVATE_BLIS_REQUALIFICATION.md`. The pinned private pthread-BLIS
> candidate passed the same decisive clean-run gates and is the accepted
> production backend.

Date: 2026-08-22  
Repository: `/home/bronsonj/SUMMIT`  
Candidate implementation: `120b0ad3ea43d1bcc094e88c78c410ed45014e96`  
Stage 08 report commit: `019388c`  
Scope: measured qualification only; no estimator source was changed

## Verdict

**NO_GO**

The dense FP64 candidate is not production-ready. A complete two-pass run over
a real PLINK source with all 454,207 variants and `B=128` returned four material
protected-kernel repairs. A bounded `B=1024` run subsequently failed closed in
pass 1 with `Generalized pass-1 algebraic checksum detected corrupted output`.
Both observations violate the Stage 09 clean-run ledger, which requires
`repair_count=0` and `integrity_failure_count=0`.

The two-pass scientific architecture itself remained intact in every completed
run: one source pass, one barrier, one target pass, exactly `2*M` visits, no
duplicate visit, retry, or fallback, fixed target-row jackknife summaries, and
reuse of the full same-person matrix. Small and moderate scientific, artifact,
fit, corruption, and sanitizer tests remain clean. Those results do not waive a
material repair or a checksum abort at production probe counts.

No target reference was promoted, no release-candidate tag was made, and Stage
10 mixed precision is not eligible to begin under the prescribed stop gate.

## Authoritative-package disposition

The isolated package mismatch is **not an estimator issue** and remains waived
under the user's explicit direction. The Stage 09 rerun of
`scripts/check_package.py` again passed the mature-estimator audit, 16 snapshot
anchors, 43-file inventory, contract guardrails, source-archive SHA-256, Python
compilation, and repository patch check before reporting only:

```text
AssertionError: manifest hash mismatch: CODEX_LAUNCH_PROMPT.md
```

The manifest expects
`f0fcec724f5eb9f363327b169200e89c6de586713799d16348cc84b16f8a91fc`;
the Markdown file hashes to
`a3000ed8d9abda0eda92a81c208a29ad15697ce1366128ee09ac77391c4be160`.
It is a non-executable orchestration prompt, and every scientific/executable
package check reached before the manifest comparison passed. The independent
mathematical oracle passed all 10 tests in 1.93 seconds, with the same recorded
identity and approximation errors as Stage 00. Generated package-check caches
were removed using the package's own cleanup routine. The package mismatch is
therefore recorded but is not part of this `NO_GO` decision.

## Frozen build and host

The candidate is the exact Stage 08 Release installation at
`/tmp/summit-generalized-stage08-release.aYHX4I`:

- embedded source-tree SHA-256:
  `cbf0d2b1279896cdb010e25ab127755522fd5ddd2ac0543ef97cd37efa9a0261`;
- `gxeldcore` SHA-256:
  `c77facb6a1181a0b9867bbb9f2d89fd48e30e07f7b3a91f38c65dbba461d095b`;
- GNU 12.2, C++17, Release `-O3 -march=native`, OpenBLAS 0.3.34
  pthreads, GNU OpenMP, dense protected NN/TN; and
- one socket-local process on physical CPUs 0--31 of the two-socket AMD EPYC
  7501 host, with explicit OpenMP placement and exact-team attestation.

The same frozen-source Stage 08 ASan+UBSan and UBSan-only builds had already
passed the supported generalized and mature native scopes. Stage 09 made no
source change that would invalidate those precondition results.

## Available real sources and qualification limit

Direct inspection established:

| Source | Samples | Variants | BED bytes | Use |
|---|---:|---:|---:|---|
| `/home/bronsonj/SUMMIT_gxe_pilot_20260808/validation/geno/age_dbp_cc` | 9,401 | 454,207 | 1,067,840,660 | complete real-format Stage 09 run |
| `/home/bronsonj/UKBB/geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc_imp` | 291,273 | 454,207 | 33,074,899,536 | inspected, not executed |

The exact `N approximately 300,000, M approximately 1,000,000` real shape was
not locally available. The largest available source has 454,207 variants. A
complete `N=291,273` run was not started after the same-build Stage 08
primary-row measurement projected an approximately 12.2-hour `N=300,000,
M=1,000,000,B=128` leading-path run. Stage 09 instead used the full real
variant axis at `N=9,401`, the exact Stage 08 `N=300,000,M=4,096` two-pass
measurement, and target-shape planner projections. This split qualification
cannot establish a full-target runtime. More importantly, the completed real
subtarget already fails the mandatory clean ledger, so a longer run cannot be
accepted without first resolving integrity.

## Complete real-data run

The real BED/BIM/FAM files were opened read-only. Basis columns, fixed effects,
annotations, probes, and jackknife block assignments were deterministic
benchmark inputs; only the genotypes and missingness pattern came from the real
source.

```bash
env PYTHONPATH=/tmp/summit-generalized-stage08-release.aYHX4I/install \
  OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 \
  OMP_PROC_BIND=true OMP_PLACES=cores \
  taskset -c 0-31 \
  /home/bronsonj/anaconda3/envs/summit/bin/python \
  scripts/benchmark_generalized_gxe_variant.py run \
  --prefix /home/bronsonj/SUMMIT_gxe_pilot_20260808/validation/geno/age_dbp_cc \
  --samples 9401 --variants 454207 --basis 3 --annotations 1 \
  --probes 128 --njack 200 --threads 32 \
  --variant-block-width 4096 --probe-tile-width 4 \
  --sample-tile-width 4096 --memory-gib 64 \
  --warmups 1 --repeats 3 \
  --cpu-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
```

### Timing and resource evidence

| Measurement | Observed value |
|---|---:|
| Median end-to-end | 1,247.2118 s |
| Measured range | 1,038.6342--1,333.3071 s |
| Range relative to median | -16.72% / +6.90% |
| End-to-end target variants/s | 364.18 |
| Logical genotype payload read | 2,135,681,314 bytes |
| End-to-end logical decoded throughput | 0.001595 GiB/s |
| Decode-subphase logical throughput | 0.2278 GiB/s |
| Peak RSS | 728,596,480 bytes (694.84 MiB) |
| Native output bytes | 159,809,448 bytes (152.41 MiB) |
| Storage writes | zero; benchmark did not publish an artifact |

The last completed execution's subphases were:

| Subphase | Seconds |
|---|---:|
| decode | 8.7301 |
| RHS formation | 83.7405 |
| source NN | 80.0680 |
| barrier projection | 0.0471 |
| same-person | 0.1243 |
| target TN | 1,051.2153 |
| row-product/segmented reduction | 16.2281 |
| hashing and integrity audits | 8.3146 |
| publication/copy | 0.0605 |

Median phase times were 92.0363 s for pass 1, 0.3071 s for the hard barrier,
1,154.7080 s for pass 2, and 0.0836 s for finalization. Target TN is the
dominant measured subphase. The row reduction processed 2,092,985,856 logical
pair-product terms, or approximately 129.0 million terms/s.

All four warm-up/measured output byte digests differed. The run retained only
hashes, not all four result arrays, so no post hoc real-run FP64 tolerance
comparison is claimed. The 28.37% maximum/minimum runtime ratio and non-bitwise
repeat outputs are additional stability concerns, although neither replaces
the decisive repair failure.

### GEMM geometries

The 111 variant blocks per pass produced 3,552 source calls and 3,552 target
calls, split as follows:

| Operation | Calls | Geometry | Logical work | Vendor-GEMM GFLOP/s |
|---|---:|---|---:|---:|
| source NN, full blocks | 3,520 | `(9401 x 4096) (4096 x 4)` | 1.0843 TFLOP | 13.736 |
| source NN, tail | 32 | `(9401 x 3647) (3647 x 4)` | 8.777 GFLOP | 13.965 |
| target TN, full blocks | 3,520 | `(9401 x 4096)^T (9401 x 36)` | 9.7591 TFLOP | 11.788 |
| target TN, tail | 32 | `(9401 x 3647)^T (9401 x 36)` | 78.994 GFLOP | 6.020 |

The native subphase timers include surrounding protected-call overhead, so the
aggregate source and target rates computed from subphase wall time are lower
than the vendor-GEMM-only records (13.65 and 9.36 GFLOP/s, respectively).

Exact placement was attested for 32 threads. Strict early NUMA binding was not
requested, so the bounded output-allocation records are evidence of allocation
ownership, not complete page locality. No hardware counters were available.
An OS process observation was consistent with roughly 29--30 busy cores, but a
complete CPU-utilization trace was not retained and is not claimed as release
evidence.

### Failed clean ledger

```text
planned_reference_genotype_passes       2
observed_reference_genotype_passes      2
observed_retained_variant_visits        908414 = 2*M
duplicate_retained_variant_visits       0
observed_block_reads                    222
pass1_decoded_blocks                    111
pass2_decoded_blocks                    111
integrity_failure_count                 0
retry_count                             0
fallback_count                          0
checksum_recomputed_columns             10
roundoff_only_columns                   6
repair_count                            4    FAIL
```

The benchmark script's internal cleanliness assertion checked pass count,
visits, retry, fallback, and integrity failure, but did not check
`repair_count`. It therefore emitted this result even though the Stage 09
contract rejects it. This is a qualification-harness gap to fix before the
next attempt; it does not make the four repairs acceptable.

## Required scaling and controls

All bounded controls used the frozen dense FP64 build and read existing
synthetic real-format PLINK fixtures. No fixture generation time was included.

### Thread scaling at `N=2,048,M=10,000,Q=3,K=1,B=128,J=200`

| Threads/placement | Median seconds | Range | Bitwise repeat | Clean ledger |
|---|---:|---:|---|---|
| 1 core | 14.2253 | 13.9183--14.4894 | yes | yes |
| 8 cores | 3.2333 | 3.1505--3.6781 | yes | yes |
| 32 cores, socket 0 | 2.0272 | 1.9637--2.0536 | yes | yes |
| 64 cores, two sockets | 3.6353 | 3.4604--3.7214 | no | yes |

The 32-core socket-local configuration remained the fastest accepted placement.

### Probe scaling

| Probes | Configuration | Result |
|---:|---|---|
| 128 | `N=2,048,M=10,000`, 32 cores, width 4 | 2.0272 s median, clean |
| 256 | same | 4.2689 s median (4.1479--4.6280), clean and bitwise repeatable |
| 1,024 | same | **pass-1 checksum abort during warm-up** |

The `B=1024` command failed before it could emit a result:

```text
RuntimeError: Generalized pass-1 algebraic checksum detected corrupted output
```

This fail-closed behavior is correct once corruption is detected, but the
normal, uninjected production control is not acceptable.

### Jackknife, alternative tiling, and larger `Q,K`

- Identical one-run `J=100` and `J=200` controls at
  `N=2,048,M=10,000,B=128`, block width 4,096 and probe width 4 each recorded
  two passes, 20,000 visits, six reads, 96 source NN calls, 96 target TN calls,
  and zero repair/retry/fallback/failure. Thus changing `J` changed neither
  genotype work nor GEMM counts.
- An alternative 512-variant block/probe-width-8 plan at the same shape and
  `J=200` completed in 2.1977 s with two passes, 20,000 visits, 40 reads, 320
  source and 320 target calls, and a clean ledger. It was a control, not a
  replacement production configuration.
- A bounded `N=1,024,M=2,048,Q=4,K=4,B=32,J=16` execution completed in
  0.9345 s with two passes, 4,096 visits, eight reads, 128 source and 128 target
  calls, 129,380,352-byte peak RSS, and a clean ledger. The Stage 08 planner
  also accepted `N=300,000,M=1,000,000,Q=4,K=8,B=128` at 26.486 GB planned
  peak and 10.445 PFLOP leading work.

### Output retained versus omitted

Target-shape dry runs with the optional per-variant directional panel retained
and omitted produced identical work and descriptor plans: 768 TFLOP, exactly
two passes, and 2,000,000 visits. Retention planned a 288,000,000-byte artifact
panel and 13,442,119,629-byte peak; omission planned zero panel bytes and
13,440,763,034-byte peak. The 40-test focused Release run exercised both atomic
artifact forms and verified that compact deletion/fit results do not access or
depend on the optional panel. A target-scale artifact-write timing comparison
was not performed because no qualifying target result existed to publish.

## Scientific and reliability acceptance

The frozen Release build reran the focused native/reference/fit/CLI scope:

```bash
env PYTHONPATH=/tmp/summit-generalized-stage08-release.aYHX4I/install \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_generalized_gxe_native.py \
  tests/test_generalized_gxe_reference_v1.py \
  tests/test_generalized_gxe_fit_v1.py \
  tests/test_generalized_gxe_variant_cli.py
```

Result: **40 passed in 3.38 seconds**. This scope verifies:

- fixed-probe dense results against every reference layer at `3e-13`;
- pair/component/probe/common-scale identities and recorded pre-symmetry and
  block-reconstruction diagnostics;
- `sum_g BDNUM[g] = DNUM` within the declared FP64 reduction tolerance;
- frozen target-row deletion without recomputing retained LD scores and reuse
  of the full same-person matrix;
- compatible trait loading, full and every delete-block fit, covariance/SE
  construction, and explicit false flags for ridge, pseudoinverse, coefficient
  clipping, and PSD replacement;
- deterministic pass-1/pass-2 fault injection before publication;
- source descriptor mutation rejection;
- array mutation/missing-member rejection and cleanup after failed temporary
  validation; and
- compact and panel-retaining artifact load/fit equivalence.

A separate temporary smoke check truncated a correctly suffixed generalized
reference archive by 64 bytes. The loader rejected it with `classic
comment-free ZIP end record is missing`; the temporary directory was removed.

The exact candidate's Stage 08 qualification also remains relevant:

| Scope | Frozen-build result |
|---|---:|
| full Release repository | 1,704 passed, 6 skipped, 1 XPASS |
| ASan+UBSan generalized native | 11 passed |
| ASan+UBSan mature native scope | 83 passed, 1 skipped |
| UBSan-only generalized native | 11 passed |
| UBSan-only mature native scope | 83 passed, 1 skipped |

There was no sanitizer diagnostic in those supported scopes. The discrepancy
between clean bounded tests and failed high-call-count production controls is
the central unresolved qualification risk.

## `B=1024+` feasibility

The two-pass architecture does not need to change merely to hold `B=1024`.
The target planner reports 6.144 PFLOP leading work, 23.334 GB peak resident
memory, a 288 MB optional panel, two passes, and 2,000,000 visits. This fits the
stated 1 TiB host by a wide memory margin. Leading work is eight times the
`B=128` plan, so it is substantially more expensive.

The current implementation is nevertheless **not operationally qualified for
`B=1024+`**, because the bounded normal run aborted in pass 1. Capacity is
architecturally feasible; production use is not feasible until a clean run
meets the fixed ledger without relaxing, bypassing, or relabeling integrity.

## Unresolved risks and required next qualification

1. Determine whether the material repairs/checksum abort arise from a genuine
   GEMM/output fault, protected-checksum tolerance or recomputation logic, or a
   concurrency/backend defect. Stage 09 evidence does not distinguish these
   causes.
2. Make the benchmark fail whenever any clean-run contract counter is nonzero,
   including `repair_count` and `checksum_recomputed_columns` when material
   repair occurs.
3. Reproduce the failure with retained operands/output at the smallest failing
   shape, compare the protected call to an independent FP64 product, and keep
   fault-injection coverage fail-closed.
4. After a fix, rerun Release and sanitizer qualification, `B=128` and
   `B=1024`, repeated real-format controls, and the complete real-variant run.
5. Only after a clean subtarget should an exact `N approximately 300,000` real
   run, target artifact publication/load/fit, strict early-NUMA evidence, and
   output-write throughput be accepted.

No relaxation of the scientific contract, no hidden retry/fallback, no
promotion of recomputed columns as a clean result, and no mixed-precision work
is authorized by this report.

## Stop gate

Stage 09 stops with `NO_GO`. The required clean FP64 baseline does not exist,
so the release/documentation actions conditional on `GO` or scoped
`CONDITIONAL_GO` were not performed. Stage 10 was not opened or started.
