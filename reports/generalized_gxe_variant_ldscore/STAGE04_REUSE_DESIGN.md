# Stage 04 pass-1 reuse design

Date: 2026-08-22

Scope: generalized per-variant SUMMIT-GxE LD-score pass 1 only. This note is the
required pre-code decision record. It does not authorize target scoring or a
third genotype traversal.

## Live reuse map

| Concern | Live mature symbol | Stage 04 decision |
|---|---|---|
| Input/trio resolution | `src/summit/ldscore/genotype_source.py:20` `GenotypeInput`; `:38` `resolve_genotype_input` | Reuse unchanged. The generalized layer accepts a validated sequential genotype operator; it does not resolve or reopen paths. |
| PGEN decode/imputation/scale | `src/summit/ldscore/genotype_source.py:330` `PgenBlockReader`; `:438` `read_standardized_block` | Reuse unchanged through an adapter. Its persistent buffer, missing sentinel, in-place mean imputation, and sample scale remain owned by the mature reader. |
| BED descriptor ownership | `src/summit/ldscore/gwe_ldscore.py:2161` construction of `gxeldcore.DirectContext`; `src/native/gxeldcore.cpp:7075` `DirectContext` | Stage 04 records and authenticates stable descriptor snapshots at the adapter boundary. The final descriptor-native generalized context is Stage 06 work; no path reopen or decoder is added here. |
| Sequential BED/PGEN block decode | `src/summit/ldscore/gwe_ldscore.py:2515` `GenomewideEnvLDScore._read_genotype_block` | Compose this mature operation through a callback adapter. Variant blocks are requested once, monotonically, for pass 1. No genotype decoding code is copied. |
| Imputation and one common sealed scale | `src/summit/ldscore/gwe_ldscore.py:2515-2815` `_read_genotype_block`; native `standardize_genotype_block` binding at `src/native/gxeldcore.cpp:16134` | The adapter contract requires already-imputed FP64 columns on one declared `genotype_scale_id`. Stage 04 never creates X/W-specific post-projection scales. All features use the same block values. |
| Protected dense NN | `src/native/gxeldcore.cpp:9850` `protected_matmul_nn`; binding at `:16101` | Reuse directly for `G_block @ (sqrt(annotation) * probe)` and expose native repaired-output counts. A NumPy implementation exists only as an independent differential test backend. |
| Integrity fault diagnostic | `src/native/gxeldcore.cpp:15674` `test_protected_matmul_nn_integrity_diagnostic`; binding at `:16107` in integrity builds | Exercise the mature diagnostic in qualification. Production routing stays on `protected_matmul_nn`. |
| Packed Mailman | `src/native/common/genotype.hpp:31` `MailmanPackedBlock`; packed descriptor consumers inside `src/native/gxeldcore.cpp:7075` `DirectContext` and `:13037` `MultiEnvironmentDirectContext` | Do not expose or copy packed internals into Python. Packed/dense adaptive source execution is extracted into the Stage 06 descriptor-native generalized context. Stage 04 establishes the scientific streaming result against dense protected NN. |
| Threading and vendor BLAS | `src/native/gxeldcore.cpp:15880` `configure_blas_threads`; protected NN's explicit `threads` argument | The execution object records its positive fixed thread request and passes it unchanged to every protected NN call. Build/runtime evidence remains native-owned. |
| NUMA/output allocation | `NativeGemmOutputAllocation`, native integrity snapshot machinery, and output evidence bindings in `src/native/gxeldcore.cpp`; `protected_matmul_nn` owns its output allocation | Reuse native ownership and telemetry. Stage 04 does not add a competing allocator. Python scientific arrays are admitted by the Stage 03 planner and recorded in an explicit allocation ledger. |
| Probe stream | `src/native/gxeldcore.cpp:10057` `global_variant_rademacher`; binding at `:16160`; Python contract in `src/summit/ldscore/generalized_gxe_variant.py` | Generate probes from `(global_variant_index, global_probe_index, seed, namespace)`. Variant blocks and probe tiles cannot change values. |
| Mature source formula (systems template only) | `src/summit/ldscore/gwe_ldscore.py:3320` `_accumulate_sketch_block`; native X/W helper at `:3475` `_native_source_block` | Reuse the streamed matrix-product shape, not the hard-coded X/W families or their separate scales. Generalized families are formed only as `F_q = P diag(phi_q) G` after the complete global source is assembled. |
| Telemetry | native `reset_gemm_telemetry`, `consume_gemm_telemetry`, protected-output NUMA evidence, and per-phase mature timing conventions | Preserve native GEMM records and add generalized pass/block/probe/allocation/timing counters. No counter is inferred from planned work when observed work is available. |

## Composition boundary

Stage 04 uses composition. A `SequentialGenotypeOperator` supplies validated,
already-imputed, already-common-scaled FP64 blocks in global variant order and
owns its descriptors/readers. A mature callback adapter wraps the existing
reader operation and stable descriptors. The pass-1 executor owns only
generalized annotations, global variant-axis probes, scientific accumulation,
projection, same-person sufficient statistics, and the Stage 03 two-pass
ledger.

The test-only array operator is not a production decoder. It makes the same
sequential contract observable for exact dense-oracle comparisons and reports
every physical visit.

## Pass-1 dataflow and lifetime

For each global variant block, exactly once:

1. authenticate descriptors and request the mature decoded/imputed/common-scale
   block;
2. for every annotation tile and probe tile, create globally addressed probe
   signs and accumulate `V[k] += G_block @ (sqrt(A_block[k]) * Z_block)` using
   protected NN;
3. record the exact visited interval in `TwoPassLedger`.

Only after all `M` variants have crossed the pass-1 barrier, form each
contextual source as `Y[q,k] = P (phi_q * V[k])`. The projection basis is the
one common sealed basis. Same-person statistics are reduced from sample and
probe tiles into `O(CN + C^2)` sufficient statistics; no `C x N x B` tensor is
allocated. Base-source scratch is released unless a diagnostic explicitly asks
to retain it. Contextual sources and finalized same-person matrices are made
read-only before return.

There is no target genotype input, target score, jackknife subtraction, or
second pass in this stage. The ledger must end Stage 04 with one observed pass,
exactly `M` distinct visits, no duplicate visit, and a sealed pass-1 barrier.

## Failure policy

The executor fails closed before returning scientific output for descriptor
mutation, non-monotone/duplicate/incomplete blocks, a genotype-scale mismatch,
non-finite genotype or annotation values, invalid annotation mass, shape or
plan mismatch, non-finite protected output, projection leakage, native telemetry
overflow, or an unsealed pass-1 barrier. The caller retains responsibility for
closing its mature genotype operator after failure.

## Stage 06 extraction decision

The final native implementation will extract the generalized pass-1 consumer
beside `DirectContext`/`MultiEnvironmentDirectContext`, so it can reuse the
descriptor mapper, BED decode, imputation, common genotype scale,
`MailmanPackedBlock`, dense/vendor BLAS selection, threads, NUMA placement,
integrity, and telemetry without Python-visible packed data. Stage 04's public
scientific result and ledger are the differential contract for that extraction.
