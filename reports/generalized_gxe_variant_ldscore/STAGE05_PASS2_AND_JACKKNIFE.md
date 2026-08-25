# Stage 05: pass 2 per-variant scores and fixed-row jackknife

> **Superseded 2026-08-24.** This stage coupled block reductions to pass 2 and
> must not be used as the current implementation contract. Pass 2 now emits
> only fixed per-SNP scores and full aggregates. `--njack` is applied by a
> separate post-hoc normal-equation reducer after reference and trait scans.

Date: 2026-08-22

Base commit: `9cbd1a63cea454164e236deb24c03f432ff22d2a`

Scope: historical second-pass implementation, retained for audit only.

## Result

**PASS.** The generalized scientific executor now completes the exact two-pass
contract. Pass 2 uses the same descriptor operator and common genotype-scale
identity as pass 1, authenticates every sealed source/basis/fixed-effect/
annotation identity before opening the descriptor, scores every retained SNP
once, and produces all normal and block numerators without recomputing any
retained SNP LD score.

All 32 focused pass-1/pass-2 tests and the full 1,662-test repository suite
pass. Every logical tiling and jackknife-count test observes exactly two passes
and `2M` retained-variant visits.

## Files changed

- `src/summit/ldscore/generalized_gxe_pass2.py`
  - immutable orientation-derived pair-product plan;
  - NumPy differential and mature protected-TN operators;
  - precomputed or reusable-tiled RHS construction;
  - one-pass target scoring and vectorized pair reductions;
  - contiguous SNP-block segmented numerator reductions;
  - full/block checksum, symmetrization, Gram normalization, and fixed-row
    delete-block postprocessing;
  - immutable result arrays and exact phase/allocation/native telemetry.
- `src/summit/ldscore/generalized_gxe_pass1.py`
  - sealed contextual-source, basis, fixed-effect, and annotation hashes plus
    the exact in-process genotype-operator identity required by pass 2.
- `src/summit/ldscore/generalized_gxe_variant.py`
  - tiled plans now require at least `Q^2` RHS columns, so one probe's complete
    cross-sketch family can always be reduced without a genotype reread.
- `tests/test_generalized_gxe_pass1.py`
  - Stage 04 barrier test extended for the identity hashes.
- `tests/test_generalized_gxe_pass2.py`
  - 17 pass-2 product-plan, oracle, tiling, jackknife, identity, sink, native,
    and fresh-process thread tests.
- `scripts/benchmark_generalized_gxe_pass2.py`
  - reproducible mature-BED-decode, wide-TN two-pass benchmark and equivalent
    protected-TN calibration.
- this report.

No mature estimator or genotype decoder was changed.

## Pass-2 execution

The executor requires an actual `GeneralizedGxEPass1Result` with:

- a clean sealed pass-1 ledger;
- read-only contextual sources and same-person matrix;
- a contextual-panel SHA-256 that recomputes exactly;
- matching basis, fixed-effect, and annotation SHA-256 values;
- the same in-process genotype operator and declared common scale; and
- one observed operator pass with exactly `M` visits.

The outer execution loop is always the genotype block. For each resident block
and source annotation, the executor either views a precomputed RHS or fills one
reusable Fortran-order arena with probe-major `Q^2` families:

```text
D_a Y[ell,b] = phi_a * Y[ell,b]
U[a,b,ell]   = G_block' (D_a Y[ell,b]) / residual_rank.
```

Each protected TN result is divided by residual rank once. Its column-major
buffer is viewed as `[Q,Q,V,B_tile]` without a layout copy. Python loops visit
only the compact target/source pair plan; every variant/probe product is an
FP64 vectorized `einsum`. Probe chunks accumulate into `LSUM` and divide by
the total `B` once, after the final chunk. No value is clamped, and signed
per-SNP values are retained.

For a complete row block `LROW[V,P,C]`, the full contribution is one vectorized
reduction:

```text
einsum("vk,vpc->kpc", A_block, LROW).reshape(C,C).
```

The same operation is applied to each contiguous jackknife segment intersecting
the decoded block. There is no per-variant atomic update and `J` never appears
outside compact mass/numerator postprocessing.

An optional sink receives one immutable, row-complete block. Aggregate results
do not reread or depend on sink output.

## Pair-product plan

The diagonal-first pair table supplies one orientation for each diagonal and
two orientations for each off-diagonal. For every target/source pair, the plan
is the Cartesian product of these orientation lists. The Q=2 multiplicity
matrix is therefore derived, not hard-coded:

```text
[[1, 1, 2],
 [1, 1, 2],
 [2, 2, 4]]
```

The Q=3 test verifies all 36 entries against the outer product of orientation
counts and verifies uniqueness of every listed four-index term. The immutable
plan has its own canonical digest.

## Fixed-probe differential errors

All fixtures use `N=11`, `M=10`, `B=23`, three contiguous SNP blocks, genotype
width 4, and RHS probe width 3. Values below are maximum absolute errors
against the independent Stage 02 oracle.

| Q | K | Per-SNP panel | Directed numerator | Genetic Gram | Block numerator | Block mass | Deleted Gram |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `1.7764e-15` | `3.5527e-15` | `7.1054e-15` | `0` | `0` | `2.8422e-14` |
| 2 | 1 | `7.1054e-15` | `1.4211e-14` | `1.4211e-14` | `1.4211e-14` | `0` | `2.8422e-14` |
| 3 | 1 | `7.1054e-15` | `1.4211e-14` | `2.8422e-14` | `1.4211e-14` | `0` | `5.6843e-14` |
| 3 | 2 | `1.7764e-15` | `3.5527e-15` | `1.4211e-14` | `3.5527e-15` | `0` | `2.8422e-14` |

The Q=2 fixture explicitly checks the mature X/W orientation multiplicities.
The Q=3,K=2 annotations are overlapping and continuous. Execution blocks cross
jackknife boundaries in the width-4 fixtures.

The protected native Q=3,K=2 fixture, using two threads, had maximum errors:

```text
per-SNP panel   1.4211e-14
genetic Gram    2.8422e-14
deleted Gram    1.1369e-13
```

It made 36 protected TN calls, recorded no repair, and completed exactly 20
total visits. Fresh-process one-thread and two-thread native executions agreed
for the per-SNP panel, directed numerator, and deleted Grams at
`rtol=atol=8e-14`.

## RHS and logical-tiling invariance

The same Q=3,K=2,B=37 fixture was run with:

- genotype width 2, one probe per tiled RHS;
- genotype width 4, four probes per tiled RHS; and
- genotype width 10, all RHS values precomputed.

All variants were decoded once in each pass. Narrow tiled versus full
precomputed output differed by at most `8.8818e-15` for per-SNP scores and
`1.1369e-13` for deleted Grams.

Changing `J` from 2 to 5 changed only block-array shapes and compact
postprocessing. The complete per-SNP panels were bit-identical, TN call counts
were identical, and both executions retained two passes and `2M` visits.

## Exact pass and block ledgers

For the protected native fixed-probe fixture:

```text
planned_reference_genotype_passes     2
observed_reference_genotype_passes    2
planned_retained_variant_visits      20
observed_retained_variant_visits     20
duplicate_retained_variant_visits     0
pass1_decoded_blocks                  3
pass2_decoded_blocks                  3
retry_count                           0
repair_count                          0
fallback_count                        0
integrity_failure_count               0
```

The genotype operator independently records exactly one new pass, `M` new
visits, and the planned decoded-block count during pass 2. The executor fails
if either ledger disagrees, if source hashes change during pass 2, or if native
telemetry or protected-output evidence overflows/fails.

For the four dense fixed-probe fixtures, maximum full-numerator reconstruction
error from `sum_g BDNUM[g]` was `1.4211e-14`; block-mass reconstruction was
exact. The benchmark's larger, differently ordered reduction had absolute
error `2.3283e-10`, below its recorded FP64 tolerance `3.6992e-4`.

## Jackknife semantics

For every block `g`, postprocessing uses only:

```text
retained DNUM = DNUM - BDNUM[g]
retained mass = MASS - block_mass[g]
```

It then symmetrizes and normalizes by retained masses. There is no genotype
access, source change, per-block same-person array, or per-SNP score mutation.
The result references the exact full same-person matrix object from pass 1 and
records `same_person_reused_for_all_deletions=true`.

The explicit Q=3,K=2,B=37 counterexample agrees with the Stage 02 fixed-row
deletion oracle but differs from literal two-sided genotype/source/kernel
deletion by maximum absolute `44.35973375800529`. This is expected and proves
that the production implementation did not silently substitute exact kernel
deletion.

## Moderate wide-TN benchmark

Command:

```bash
PYTHONPATH=<fresh-release-install> \
  python scripts/benchmark_generalized_gxe_pass2.py \
  --samples 512 --variants 4096 --basis 3 --annotations 1 --probes 128 \
  --variant-block-width 512 \
  --pass1-probe-width 64 --pass2-probe-width 128 \
  --threads 2 --memory-gib 2 --calibration-repeats 3
```

The benchmark uses a temporary synthetic BED trio and the mature
`GenomewideEnvLDScore._read_genotype_block` adapter in both passes.

```text
observed passes / visits             2 / 8,192
pass-1 / pass-2 decoded blocks       8 / 8
duplicate / retry / repair / fallback 0 / 0 / 0 / 0
pass-2 decode time                   0.27655 s
protected TN calls                   8
TN dimensions                        512 x 512 by 512 x 1,152
TN leading FLOPs                     4,831,838,208
protected TN phase                   0.67062 s
executor TN throughput               7.2050 GFLOP/s
RHS preparation                      0.04484 s
row-product reduction                0.04545 s
numerator/block reduction            0.00473 s
postprocessing                       0.00087 s
complete pass-2 time                 1.04951 s
process-lifetime peak RSS            205,680,640 bytes
```

Three separate same-dimension calls to the identical mature
`gxeldcore.protected_matmul_tn` binding had median throughput
`7.8351 GFLOP/s`. The executor achieved 91.96% of that calibration rate while
also dividing cross sketches and maintaining guarded output ownership. No
calibration call touched a genotype descriptor or changed the two-pass ledger.

The benchmark allocated a 4,718,592-byte reusable RHS arena, at most 9,437,184
bytes across raw/scaled cross output, a 1,179,648-byte complete directional
panel, 147,456-byte `LROW` and `LSUM` blocks, 5,760-byte block numerators, and
5,760-byte deleted Grams. Planned peak resident memory was 99,938,498 bytes
under the 2 GiB limit.

The pre-symmetry difference was `419.2673` absolute and `2.6389e-4` relative.
This is a recorded finite-probe directional-sketch diagnostic, not a failure:
the contract forms the full Gram from `0.5*(DNUM+DNUM')` and does not overwrite
the signed directional panel.

## Validation

Source-only pass-1/pass-2 qualification:

```text
27 passed, 5 deselected in 1.73 s
```

Fresh Release focused qualification after final provenance refresh:

```text
32 passed in 11.03 s
```

Full repository regression before the provenance-only rebuild:

```text
1662 passed, 5 skipped, 1 xpassed in 149.95 s
```

Static checks passed:

```text
Python py_compile
100-character Python line scan
git diff --check
```

The qualified Release extension used OpenBLAS with GEMM integrity and checksum
enabled. Its SHA-256 is
`7383fde933a8edd5a8293ea4c0f1946b63f77993ea76b03635f30efdc6a44f89`.
It embeds source commit
`9cbd1a63cea454164e236deb24c03f432ff22d2a` and exact staged build-input tree
SHA-256
`71a2cf5f458df309a665651a96dbde8c0b66ae7a05f9d2bcd2a9753647586b76`.
This required final report was added after that build-input snapshot and does
not change executable code.

## Remaining Stage 06 work

- Python still owns genotype-block, annotation, probe, and compact pair-plan
  orchestration. The complete scientific differential contract must move into
  a descriptor-owned native `GeneralizedGxELDScoreDirectContext` or equivalent.
- The production native owner must reuse mature packed Mailman/dense selection,
  direct decode/imputation/common scaling, protected NN/TN, phase arenas,
  threads/affinity/NUMA, integrity, and bounded telemetry.
- The Stage 05 correctness executor holds the complete directional panel in
  memory. The native/artifact path must publish row-complete shards or an
  admitted panel without adding a descriptor traversal.
- Packed-versus-dense and BED-versus-PGEN native differential qualification
  remain Stage 06 stop-gate requirements.

## Stop gate

Stage 05 passes. The complete signed per-SNP panel, full numerator, every block
numerator/mass, full Gram, and every frozen-row deleted Gram match the oracle;
all source identities remain fixed; same-person is reused; and every logical
tiling and `J` variation observes exactly two descriptor passes and `2M`
variant visits.
