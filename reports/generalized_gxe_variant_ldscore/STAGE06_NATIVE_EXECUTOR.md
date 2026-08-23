# Stage 06 native production executor

## Decision

**PASS for the descriptor-owned BED production path.** The generalized
variant-axis executor is fixed-probe equivalent to the Stage 05 scientific
path, reports exactly two complete genotype traversals and `2M` retained
variant visits, passes the full Release suite, and passes the supported native
ASan+UBSan and UBSan-only scopes. The contextual sample-probe estimator and
the mature non-general G×E paths remain separate and unchanged.

Native PGEN execution is not claimed. The live tree has no descriptor-owned
C++ PGEN context: PGEN decoding is owned by Python `pgenlib`, and the mature
direct G×E backend also rejects native PGEN execution. Stage 06 therefore did
not create a second PGEN decoder. PGEN planning and existing mature PGEN tests
remain covered by the Release and legacy-regression results below.

## Settled candidate

- implementation commit:
  `27e20f6a7e72a9e443b37b614c7e4c7d4b8f2f57`;
- source-tree SHA-256 embedded in all three native builds:
  `3edce0be52826c190842ac02900a1e3a7fa1c2e7c89a4c3a861baba142de8935`;
- Release root: `/tmp/summit-generalized-stage06-qual.0jCoyQ`;
- Release `gxeldcore` SHA-256:
  `2204610b94467aa7ba65e573c91ef529665c82c3fb9b107685e9d92090958877`;
- Release build: GNU 12.2, `Release`, OpenBLAS, integrity and exact checksums
  enabled, sanitizer mode `none`, `-march=native`;
- ASan+UBSan root: `/tmp/summit-generalized-stage06-asan.iuNmC4`, extension
  SHA-256
  `1da993ebccde8c0412081d816d09a23a7767e3cd8d853f48888835a9ff445d75`;
- UBSan-only root: `/tmp/summit-generalized-stage06-ubsan.uLGUye`, extension
  SHA-256
  `be5448c6de2d98faf1ff6dca0e8c292eda90717f12f2256c41862d35fb20239c`.

Both sanitizer builds report portable architecture tuning, `-O1`, native
optimization disabled, the expected sanitizer mode, and the same clean source
identity as the Release build.

## Native reuse map and scientific boundary

| Concern | Reused live owner | Generalized use |
|---|---|---|
| Descriptor identity and retained rows | `DirectContext` | One composed context duplicates the caller's BED/BIM/FAM descriptors, records file state, maps SNP-major BED, and owns retained row/variant order across both passes. |
| Decode, imputation, scale | `DirectContext::decode_block_into`; `read_block_mailman_mean_memory` | Dense and packed paths use the mature mean-imputation and one affine FP64 genotype transform under one `ddof`. No X/W post-projection scale is accepted. |
| Dense protected products | `dgemm_nn_partitioned_rows`, `dgemm_tn_partitioned_columns`, `dgemm_tn_partitioned_rows` | Pass 1 sources, projection, and pass 2 targets use the qualified protected NN/TN implementations and their integrity policy. |
| Packed products | `MailmanPackedBlock` and the mature Mailman primitives | The packed generalized adapter consumes the same decoded representation; it does not reuse the non-general X/W scientific layout. |
| Threads and integrity | fixed OpenBLAS thread freeze, OpenMP-capacity checks, GEMM checksums/canaries/retry policy | One positive admitted thread count is frozen and reused. `configured_blas_threads()` only exposes the already-frozen count so independently ordered tests obey the existing process-wide policy. |
| Scientific plan | Stage 03 pair/component tables and Stage 04 product plan | The generalized owner validates canonical table contents and Python/native SHA-256 digests. Pair factors and products are generic in `Q`; no `Q=2` formulas are hard-coded. |

The low-level descriptor-only `DirectContext` constructor contains no
annotation, pair, contextual, product-plan, or jackknife semantics. Those are
owned by `GeneralizedGxELDScoreDirectContext`. Existing
`DirectContext::source_block`, `MultiEnvironmentKernel`, and
`MultiEnvironmentDirectContext` group-action schedulers are not called.

## Lifecycle and traversal proof

The context is single-use and fail-closed:

```text
constructed -> pass1_running -> sources_sealed -> pass2_running -> finalized
                              \-> failed on any exception
```

In each pass the outermost loop is the ordered genotype-block loop. All
annotation/probe or target-product tiles are completed while that block is
resident. The pass-1 barrier checks file identity, one traversal, `M` visits,
canonical plan digests, finite projected sources, and the sealed contextual
source checksum before pass 2 begins. Pass 2 rechecks the descriptor and
source checksum and finalization requires two traversals and `2M` visits.

For the twelve-variant differential fixture, both dense and packed execution
reported:

| Ledger field | Observed |
|---|---:|
| planned/observed complete passes | 2 / 2 |
| planned/observed retained visits | 24 / 24 |
| block reads | 6 |
| duplicate visits | 0 |
| retries / repaired columns / fallbacks | 0 / 0 / 0 |
| integrity failures | 0 |
| terminal state | `finalized` |

The descriptor-mutation test changes the BED after construction and verifies
that execution raises without publication. A second `run()` also raises.

## Allocation and publication ownership

Immutable bases, annotations, tables, and all scientific outputs are copied
into private context-owned `std::vector` storage. Output sizes are checked and
pre-sized before either pass. Dense protected products use the mature native
protected GEMM allocation/integrity path; packed products use the mature
Mailman block owner. The admitted Python work plan must fit its memory limit
before native construction. Base sources are released at the pass-1 barrier
unless explicitly requested for differential qualification.

No Python array is published during either pass. After successful native
finalization, the context copies complete results into Python-owned arrays;
the wrapper then revalidates the source checksum, product-plan digest, pass
ledger, and block-to-full numerator reconstruction and makes returned arrays
read-only. An exception changes the context to `failed` and clears every
publishable scientific vector.

This stage does not claim per-output NUMA placement evidence for the
generalized context-owned vectors. It reuses the qualified protected GEMM and
packed-kernel policies but does not add a new generalized NUMA allocator.

## Changed symbols

- `src/native/gxeldcore.cpp`
  - added the science-neutral `DescriptorOnlyDirectContextTag` constructor;
  - added friendship for `GeneralizedGxELDScoreDirectContext`;
  - bound `GeneralizedGxELDScoreDirectContext` and the read-only
    `configured_blas_threads()` query.
- `src/native/generalized_gxe_variant.inc`
  - added the generalized owner, state machine, native pass 1/barrier/pass 2,
    dense protected and packed Mailman adapters, scientific finalization,
    bounded counters, and fail-closed publication.
- `src/summit/ldscore/generalized_gxe_native.py`
  - added `GeneralizedGxENativeBEDExecutor` and immutable
    `GeneralizedGxENativeResult` validation/publication surfaces.
- `tests/test_generalized_gxe_native.py`
  - added four generalized dense differential cases, packed/dense and
    one/multiple-thread comparisons, single-use/base-retention checks, and
    descriptor-mutation failure coverage.

All additions are new names. Existing CLIs, artifact schemas, probe streams,
and non-general/contextual public entry points were not modified.

## Fixed-probe numerical evidence

The fixture has `N=13`, `M=12`, 13 probes, mean missing-genotype imputation,
three unequal contiguous jackknife segments of sizes 3/4/5, and genotype block
widths 4 or 5 so block boundaries cross deletion segments. The maximum
absolute error over every tested entry was:

| Output | Dense native vs Stage 05 |
|---|---:|
| base sources | `2.220446049250313e-15` |
| contextual sources | `3.552713678800501e-15` |
| same-person matrix | `8.526512829121202e-14` |
| directional per-variant LD scores | `2.4868995751603507e-14` |
| directed numerator (`DNUM`) | `5.684341886080802e-14` |
| symmetric numerator | `5.684341886080802e-14` |
| full genetic Gram | `8.526512829121202e-14` |
| block directed numerator (`BDNUM`) | `2.842170943040401e-14` |
| block annotation mass | `0.0` |
| deleted genetic Grams | `2.2737367544323206e-13` |

Maximum absolute error over all output layers by scientific case was
`7.105427357601002e-15` for `Q=1,K=1`,
`7.105427357601002e-14` for `Q=2,K=1`,
`1.7053025658242404e-13` for `Q=3,K=1`, and
`2.2737367544323206e-13` for `Q=3,K=2`.

For the `Q=3,K=2` packed/dense comparison, maximum absolute errors were
`2.6645352591003757e-15` for base sources,
`6.217248937900877e-15` for contextual sources,
`5.684341886080802e-14` for same-person,
`1.0658141036401503e-14` for directional scores,
`4.263256414560601e-14` for `DNUM`,
`2.1316282072803006e-14` for `BDNUM`, and
`1.7053025658242404e-13` for deleted Grams. Fresh one- and two-thread packed
processes also matched at the test's `3e-13` relative/absolute tolerance.

## Test and sanitizer qualification

| Scope | Result |
|---|---:|
| Stage 02–06 oracle/contracts/pass-1/pass-2/native focused files | **82 passed in 15.11 s** |
| Required mature native/PGEN compatibility files plus feature conventions | **91 passed in 15.50 s** |
| Full Release repository | **1669 passed, 6 skipped, 1 XPASS in 139.21 s** |
| ASan+UBSan new generalized native file | **8 passed in 6.05 s** |
| ASan+UBSan required seven-file mature native scope | **88 passed in 21.58 s** |
| UBSan-only new generalized native file | **8 passed in 3.66 s** |
| UBSan-only required seven-file mature native scope | **88 passed in 15.48 s** |

ASan+UBSan used
`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, with GCC 12 `libasan` and
the Conda `libstdc++.so.6` preloaded. UBSan-only used the same fatal UBSan
settings. Test output contained no address- or undefined-behavior diagnostic.
Because `detect_leaks=0`, LeakSanitizer and leak freedom are explicitly not
claimed. Full-repository sanitizer execution is also not claimed; sanitizer
qualification is the new native file plus the required mature seven-file
native scope.

The complete Release suite's six skips and single XPASS are repository-level
existing outcomes; no Stage 06 generalized native test skipped in the recorded
Release, ASan+UBSan, or UBSan-only focused runs.

## Stop-gate conclusion

The descriptor-owned BED executor meets the Stage 06 stop gate: native and
Stage 05 fixed-probe outputs agree within FP64 tolerance; dense protected and
packed independent paths agree; normal tilings observe exactly two complete
traversals with no duplicate visits; descriptor mutation and invalid lifecycle
transitions fail without publication; mature paths, the full Release suite,
and supported sanitizer scopes pass. Stage 07 may begin from the settled
implementation commit and this report.
