# Stage 04: pass 1 global sources and same-person matrix

Date: 2026-08-22

Base commit: `449b7c0e476f00f10de3f52e98ecf64a0b7f8f8a`

Scope: the first complete reference-genotype traversal only. No target score,
per-variant LD score, jackknife subtraction, or second descriptor pass is
implemented in this stage.

## Result

**PASS.** The generalized executor completes every base source and contextual
source from one monotonically ordered genotype traversal, seals the hard
pass-1 barrier, globally finalizes the signed same-person U-statistic, and
returns immutable source panels for pass 2. Observed pass-1 variant visits are
exactly `M` for every genotype/probe/annotation tiling tested.

All 15 focused Stage 04 tests and the full 1,645-test repository suite pass.

## Files changed

- `src/summit/ldscore/generalized_gxe_pass1.py`
  - stable-descriptor authentication and mature-reader composition adapter;
  - explicitly test-only in-memory genotype and NumPy NN operators;
  - protected native NN owner with repair, GEMM, allocation, integrity, NUMA,
    and overflow telemetry;
  - one-pass source executor, post-barrier projection, pairwise-tiled
    same-person finalization, immutable result panels, and allocation/phase
    ledgers.
- `src/summit/ldscore/generalized_gxe_variant.py`
  - a pass-1-only hard-barrier validator on the Stage 03 two-pass ledger.
- `tests/test_generalized_gxe_pass1.py`
  - 15 oracle, tiling, thread, failure, descriptor, protected-NN, and integrity
    tests.
- `scripts/benchmark_generalized_gxe_pass1.py`
  - reproducible source-only synthetic BED benchmark using the mature reader.
- `reports/generalized_gxe_variant_ldscore/STAGE04_REUSE_DESIGN.md`
  - required pre-code live-symbol and extraction decision record.
- this report.

No mature estimator file and no native genotype decoder was changed.

## Reuse design

The detailed symbol map is in `STAGE04_REUSE_DESIGN.md`. The implemented
boundary is composition:

```text
caller-owned stable BED/PGEN descriptors
  -> mature _read_genotype_block / PgenBlockReader operation
  -> already-imputed FP64 G block on one declared genotype_scale_id
  -> global variant-axis probe RHS
  -> existing gxeldcore.protected_matmul_nn
  -> complete V[K,N,B]
  -> hard end-of-descriptor barrier
  -> Y[K,Q,N,B] = P diag(phi_q) V[k]
  -> global same-person finalization
```

The generalized layer does not resolve or reopen genotype paths, decode BED or
PGEN bytes, impute values, or create a post-projection X/W scale. The adapter
for `GenomewideEnvLDScore` calls its live `_read_genotype_block` method and
declares the one common scale as
`mean_imputed_{sample|hwe}_ddof=..._eps=..._fp64_v1`.

`MailmanPackedBlock` remains native-private. Stage 04 therefore qualifies dense
protected NN and records `packed_backend_used=false`. Extracting an adaptive
descriptor-native dense/Mailman generalized consumer beside `DirectContext`
and `MultiEnvironmentDirectContext` is Stage 06 work. This avoids either a
Python-visible packed representation or a copied decoder.

## Scientific execution

For a decoded global block `[s,e)`, the executor generates probes only from
the global retained-variant indices, global probe indices, root seed, and named
namespace. It evaluates every annotation/probe source tile while the genotype
block is resident:

```text
V[k,:,probe_tile] +=
    G[:,s:e] @ (sqrt(A[s:e,k]) * Xi[s:e,probe_tile])
```

Only after the operator and ledger both attest end-of-descriptor does it form
all contextual panels as `P (phi_q * V[k])`. Normal execution then releases
base-source scratch. Diagnostic tests may retain it read-only for direct oracle
comparison.

For each annotation-major context-pair component `c=(k,q,r)`, the same-person
quantity uses signed values

```text
h[c,i,v] = kernel_factor(q,r) Y[k,q,i,v] Y[k,r,i,v] / M_k.
```

Only `a[c,i] = sum_v h[c,i,v]` and the global `C x C` same-probe products
persist. The finalized statistic is

```text
(a a' - sum_v h_v h_v') / (B(B-1)).
```

Component pairs are evaluated with at most two sample/probe tiles resident;
no `C x N x B` array is allocated and probe chunks are never finalized
separately.

## Fixed-probe differential errors

The fixture has `N=11`, `M=10`, `Q=3`, `K=2`, and `B=37`, with explicit
global counter probes at offset 13. Errors below are maximum absolute errors
against the independent Stage 02 dense oracle.

| Backend | Variant width | Probe width | Annotation width | Base source | Context source | Same-person | Visits |
|---|---:|---:|---:|---:|---:|---:|---:|
| NumPy differential | 1 | 1 | 1 | `1.7764e-15` | `3.5527e-15` | `7.1054e-15` | 10 |
| NumPy differential | 4 | 7 | 2 | `1.7764e-15` | `1.7764e-15` | `1.4211e-14` | 10 |
| NumPy differential | 10 | 37 | 1 | `0` | `0` | `3.5527e-15` | 10 |
| Protected native NN, 2 threads | 4 | 7 | 2 | `1.7764e-15` | `1.7764e-15` | `1.4211e-14` | 10 |

The maximum relative fixed-basis leakage was `2.1415e-16`. A fresh-process
comparison of protected native execution with one and two threads passed for
base sources, contextual sources, and the same-person matrix at
`rtol=atol=5e-14`. NumPy logical thread requests 1 and 4 were bit-identical.

Packed-versus-dense testing is not available at the Stage 04 Python boundary
because the mature packed genotype block is intentionally not exposed. It is
a required native differential in Stage 06.

## Pass ledger

For the protected fixed-probe fixture with three physical decode blocks:

```text
planned_reference_genotype_passes     2
observed_reference_genotype_passes    1
planned_retained_variant_visits      20
observed_retained_variant_visits     10
duplicate_retained_variant_visits     0
pass1_decoded_blocks                  3
pass2_decoded_blocks                  0
retry_count                           0
repair_count                          0
fallback_count                        0
integrity_failure_count               0
```

The independent genotype-operator counters agree: one pass, 10 visits, and
three blocks. Widths 1, 4, and 10 produced 10, 3, and 1 decoded blocks but
always exactly 10 variant visits. The barrier validator rejects an active,
partial, duplicated, retried, fallback, integrity-failed, or prematurely
started pass-2 state.

## Allocation ledger

The protected fixed-probe fixture recorded:

```text
base sources                       6,512 bytes
contextual sources                19,536 bytes
maximum decoded block                352 bytes
maximum probe/RHS tile               224 bytes each
maximum protected output             616 bytes
maximum projection scratch          7,696 bytes
same-person C x N accumulator       1,056 bytes
same-person C x C accumulator       1,152 bytes
maximum same-person pair tiles      1,232 bytes
```

All planned arrays fit under the admitted plan. Native protected outputs keep
their mature guarded allocation, integrity, and NUMA-evidence ownership; the
generalized layer adds no competing output allocator.

Normal execution seals `contextual_sources`, `same_person`, and annotation
masses read-only, returns no base-source array, and records
`base_source_scratch_released=true`.

## Failure and integrity tests

The focused suite verifies that publication fails before a completed barrier
for:

- negative, non-finite, zero-mass, or mass-inconsistent annotations;
- a decoded block carrying a different common genotype-scale identity; and
- in-place mutation of a caller-owned stable descriptor.

The integrity-build diagnostic was exercised at `512 x 2048` by
`2048 x 512`, above the one-billion-FLOP checking threshold. A `+1` corruption
was injected at output row 17, column 29. The mature interface classified the
vendor result outside its forward-error bound, repaired exactly one column,
and the executor's sealed base source agreed with the uncorrupted dense result
within `rtol=4e-14`, `atol=4e-12`. The ledger and telemetry both recorded one
repair; this test did not add a production fault-injection branch.

## Moderate source-only benchmark

Command:

```bash
PYTHONPATH=<fresh-release-install> \
  python scripts/benchmark_generalized_gxe_pass1.py \
  --samples 512 --variants 4096 --basis 3 --annotations 3 --probes 128 \
  --variant-block-width 512 --probe-tile-width 64 --threads 2 \
  --memory-gib 2
```

The benchmark generated a temporary synthetic BED trio, constructed the
mature estimator without running its LD-score method, and composed its
`_read_genotype_block`. Synthetic setup is excluded from pass-phase timings.

```text
decoded blocks / visits              8 / 4,096
observed genotype passes             1
duplicate visits                     0
retry / repair / fallback            0 / 0 / 0
decode time                          0.28323 s
protected NN calls                   48
protected NN dimensions              512 x 512 by 512 x 64
protected NN leading FLOPs           1,610,612,736
protected NN time                    0.56662 s
protected NN throughput              2.8425 GFLOP/s
projection time                      0.01235 s
same-person time                     0.04737 s
complete pass-1 time                 0.94091 s
process-lifetime peak RSS at exit    204,746,752 bytes
maximum relative projection leakage 6.4604e-17
same-person presymmetry error        0
```

The benchmark allocation ledger recorded 1,572,864 base-source bytes,
4,718,592 contextual-source bytes, a 2,097,152-byte decoded block, and a
524,288-byte maximum same-person pair tile. The admitted modeled peak was
94,977,048 bytes under a 2 GiB limit. This benchmark makes no target or
end-to-end scalability claim.

## Validation

Source-only qualification before the native rebuild:

```text
12 passed, 3 deselected in 0.98 s
```

Fresh Release focused qualification:

```text
15 passed in 7.80 s
```

Full repository regression:

```text
1645 passed, 5 skipped, 1 xpassed in 142.70 s
```

Static checks passed:

```text
Python py_compile
100-character Python line scan
git diff --check
```

The qualified Release extension used OpenBLAS with GEMM integrity and checksum
enabled. Its SHA-256 is
`29e6fd920cbe7b53c1d4f3fb498d7a11c2b05f4fa5be1f8853fd0153b27f27a5`.
It embeds source commit
`449b7c0e476f00f10de3f52e98ecf64a0b7f8f8a` and the exact staged build-input
tree SHA-256
`f5a609b07f0797e3654eba5a0d941e717539a7dc00de3bd9fcda2b0cc1e66b5d`.
The required final report was added after that build-input snapshot and does
not change executable code.

## Remaining bottlenecks and Stage 06 handoff

- Stage 04 uses Python block/tile orchestration and reaches only 2.84 GFLOP/s
  for moderate `512 x 512 x 64` protected products. No production performance
  conclusion follows from this prototype.
- Base-to-context projection and pairwise same-person finalization are correct
  but still Python-orchestrated.
- Dense-versus-Mailman selection, descriptor-native decode, NUMA placement of
  persistent generalized sources, and fused native phase telemetry remain for
  Stage 06.
- Pass 2, per-SNP directional generalized LD scores, fixed full-genome block
  sums, and jackknife row subtraction are exclusively Stage 05 work.

## Stop gate

Stage 04 passes. Every fixed-probe base source, contextual source, leakage, and
same-person comparison succeeds; observed physical pass-1 visits equal exactly
`M` independently of logical tile counts; the source panels are sealed; and no
target-scoring code exists in this stage.
