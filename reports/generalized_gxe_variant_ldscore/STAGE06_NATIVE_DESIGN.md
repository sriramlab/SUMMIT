# Stage 06 native design review

This note records the implementation boundary before Stage 06 native edits.
Line numbers refer to the Stage 05 tree (`521f64a`).

## Existing ownership and reusable symbols

| Concern | Observed owner/symbol | Stage 06 use |
|---|---|---|
| BED descriptors and retained order | `DirectContext` in `src/native/gxeldcore.cpp:7075` duplicates the caller's BED/BIM/FAM descriptors with `CLOEXEC`, captures `FileState`, maps the SNP-major BED, copies and validates `row_sel`, and retains `rows_`, `n_`, and `m_total_`. `MultiEnvironmentDirectContext` already composes this object at `:13037`. | Compose one `DirectContext` for the entire generalized execution and add only the friendship needed to call its science-neutral private reader operations. Do not reopen or replace its decoder. |
| PGEN descriptors and retained order | The mature Python estimator owns `_genotype_descriptors`; `PgenGenotypeSource` in `src/summit/ldscore/genotype_source.py:345-440` owns `pgenlib.PgenReader`. No descriptor-owned C++ PGEN context exists in the live tree. | The Stage 06 native context is therefore a BED production context. PGEN remains available to the Stage 05 differential path and will be tested there when `pgenlib` is available. Adding a second PGEN decoder in Stage 06 is forbidden. |
| Decode, mean imputation, and sealed common scale | `DirectContext::decode_block_into` at `src/native/gxeldcore.cpp:8582` reads retained rows, imputes missing values to the observed mean, and applies the single per-SNP affine transform `(mean - dosage) * inverse_sd` with `ddof_`. `check_files_unchanged` authenticates the descriptors before publication. Packed decoding uses `MailmanPackedBlock` and `read_block_mailman_mean_memory` from `src/native/common/genotype.hpp:31,127`. | Use `decode_block_into` for dense execution and the existing packed reader for Mailman execution. The generalized context accepts no X/W scales and never rescales a projected feature. Its scale identity is the immutable descriptor/row-order/`ddof` transform. |
| Protected NN/TN | `dgemm_nn_partitioned_rows` (`gxeldcore.cpp:6204`), `dgemm_tn_partitioned_columns` (`:6145`), and `dgemm_tn_partitioned_rows` (`:6269`) implement the qualified protected kernels. Public qualification entry points are `protected_matmul_nn` (`:9850`) and `protected_matmul_tn` (`:15776`). | Call the same internal partitioned functions so generalized outputs inherit the mature output allocation, integrity snapshot, retry/repair, and deterministic fallback policy. No new generic matrix layer is introduced. |
| Packed and vendor kernels | `MailmanPackedBlock` plus `MultiEnvironmentKernel::source_block_packed` demonstrate packed source multiplication; dense shapes already reach the fixed vendor BLAS through the partitioned NN/TN functions. | Dense is the initial generalized implementation. A packed generalized adapter may call the same Mailman primitives only after matching the dense fixed-probe result; it must not adopt the X/W kernel layout. The native plan records which admitted backend ran. |
| Allocation and source sealing | `NativeGemmOutputAllocation` (`gxeldcore.cpp:2130`) owns private pre-sized protected output; `PackedSourcePanel` (`:6884`) owns pre-sized source storage, NUMA evidence, read-only sealing, and mutation detection. | Reuse these owners unchanged for protected products and the persistent contextual-source panel. The generalized owner publishes only after the source panel is sealed and all semantic checks pass. |
| Integrity, retry, and fallback | GEMM operand snapshots, checksums, canaries, `RetryableGemmInputMutation`, repair counters, and guarded long-double/tiled fallbacks are inside the existing protected functions. `DirectContext` records repairs and retries. | Preserve that policy and expose generalized observed repair/retry/fallback counts. A failed semantic digest, nonfinite result, file mutation, or telemetry overflow aborts the run; no partially filled output is returned. |
| Threads, affinity, and NUMA | `validate_configured_openmp_placement_for_entry`, `effective_openmp_capacity`, fixed-vendor thread configuration, `native_numa_contract_request`, `NativeGemmOutputAllocation`, and `PackedSourcePanel` implement the mature policy. | Validate one fixed positive thread request at construction, pass it to every existing protected kernel, bind reusable decode/source allocations with the existing policy, and return the existing NUMA evidence records. |
| Telemetry | `GemmTelemetryRecord`/buffer (`gxeldcore.cpp:2875-3255`), native output evidence, and the mature `append_call` semantic records are bounded native-owned evidence. | Reuse the global protected-operation records and add bounded generalized records for pass, block, and operation. Planned capacities are checked before execution. Observed traversal/visit counts, never inferred counts, are returned. |

## Composition boundary

The new generalized owner is split conceptually in two:

1. A science-neutral descriptor-backed block operator is the composed
   `DirectContext` plus its dense/packed block lease and protected NN/TN calls.
   It knows file identity, retained row/variant order, missingness and scale,
   allocation policy, threads, and counters. It does not know contextual
   pairs, orientation factors, annotations, components, or jackknife formulas.
2. `GeneralizedGxELDScoreDirectContext` owns immutable copies of the Stage 03
   basis/pair/component/product/jackknife tables and their Python-computed
   digests. It constructs source RHS panels during pass 1 and consumes target
   rows during pass 2. This is the scientific owner; it does not decode.

The minimal native change is composition plus a `friend` declaration on
`DirectContext`. There is no common-code extraction and therefore no
refactor-only commit is needed. The existing `DirectContext::source_block`,
`target_block`, `MultiEnvironmentKernel`, and `MultiEnvironmentDirectContext`
scientific schedulers are not called: they encode separate X/W scales or the
sample-probe multi-environment estimator.

## Lifecycle and loop order

The owner has a fail-closed single-use state machine:

```text
constructed -> pass1_running -> sources_sealed -> pass2_running -> finalized
```

For pass 1, the outermost loop is the ordered genotype-block loop. A block is
decoded once, then all admitted annotation and probe tiles are executed while
that lease is live. Projection and contextual row scaling occur only after the
last source block, with no descriptor access, followed by same-person
finalization and source sealing. The hard barrier authenticates file state,
table digests, source checksum, source read-only state, exactly `M` visits, and
one completed traversal.

For pass 2, one ordered genotype-block loop decodes each block once. Every
admitted target RHS/product tile is executed before releasing that block.
Row-complete directional numerators and the target block's contributions to
contiguous-segment `BDNUM` are reduced in native FP64 storage. There is no
probe- or SNP-level Python loop and no returned cross-sketch tensor. Finalize
requires two traversals and exactly `2M` visits.

Logical tiling may change the inner annotation, probe, pair-product, or RHS
loops; it cannot enclose or duplicate either genotype traversal.

## Compatibility and publication

The new context and Python wrapper are additive names. Existing CLI dispatch,
`DirectContext`, `MultiEnvironmentDirectContext`, schemas, probe streams,
scientific arrays, and artifact writers are untouched. Stage 06 tests first
exercise the new API directly. Later command integration will be an explicit
stage change. This makes legacy byte/science compatibility testable as an
unchanged old-path regression rather than an assumption.

The native result is accepted only if Python revalidates the immutable plan
digests, common genotype-scale identity, source checksum, pass ledger,
row/block reconstruction, Gram checksum, and fixed full-row deletion identity.
Publication remains downstream of that validation and is atomic under the
existing artifact machinery.
