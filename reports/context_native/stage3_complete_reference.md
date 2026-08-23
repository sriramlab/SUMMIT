# Contextual native estimator — Stage 3 report

## Decision

`NARROW_GO` for Stage 4, restricted to private development of the contextual
trait-summary path against the same descriptor-owned executor and the stable
reference V1 boundary introduced here.

The Stage 3 private-use gate passes. The streamed executor matches the active
Python oracle for full Gram, grouped raw numerators, and the signed same-person
U-statistic over generated `Q=1..4` strict and generic cases. All individual
tile widths in the `Q=3` strict/generic tile fixture, `B_D=2` one-probe tiles,
explicit and counter variant probes,
factors 1/2/4, restricted and direct grouped algorithms, every single-group
and selected multi-group deletion through fits/Omegas/surfaces, all seven new
protected operations, compactness, stable schema isolation, and legacy
regression pass on the settled clean build. The stable publisher accepts only
the group-restricted result; direct grouped TN remains differential evidence.

This is not a full or production Stage 3 GO. Stage 4 may privately consume the
immutable reference artifact and extend the same executor, but may not inherit
claims of total-process memory admission, NUMA ownership, adversarial file
immutability, complete run provenance/telemetry, a qualified direct-grouped
planner, two-process variant-probe merging, or tabla-scale performance. The
specific blockers are listed below.

## Repository/build state

- Branch: `codex/contextual-stage3-complete-reference`, created from the
  Stage 2 report commit
  `6dbea4e48daf2fe41511fdcc2e8c56c3f7cfe2ec`.
- Settled implementation and tests commit:
  `07720fe0819cf3934abe1a28ba4922771480ad32`. The worktree was clean when the
  qualification build was configured. This report is added afterward in a
  documentation-only commit.
- Native implementation source SHA-256:
  `5c576664418123318e5279beee15b63f8d21e943119b035ee3a830119f05812a`.
- Embedded native provenance: source commit `07720fe...`; source-tree SHA-256
  `d66193b16e6fb2cbabb5b80e3aaa3cf7972c128e5f666486029f80c44977e140`.
- Clean build: `/tmp/summit-stage3-qual-build.Iw2s2T`; clean install:
  `/tmp/summit-stage3-qual-install.aUzjL8`. Existing build products were not
  overwritten.
- Installed `gxeldcore` SHA-256:
  `421fd4ddd06c483118167f5c7e91988ec17e5a07acd35b87e2020c81dd1be1fd`.
- Compiler/build: GNU C++ 12.2.0, C++17, Release,
  `-O3 -march=native -fopenmp`; native optimization, OpenMP, contextual GEMM
  integrity, and checksum compile options enabled.
- Native backend: unchanged `gxeldcore` module version 1.7, backend 1.9, API
  9; shared OpenBLAS 0.3.34 (`DYNAMIC_ARCH`, Zen, pthread), OpenMP 4.5. The
  contextual protected calls use the deterministic tiled fp64 primary,
  scalar witness, retry, and independently loop-ordered scalar fallback.
- Host: Linux `Tabla`, AMD EPYC 7501, 2 sockets, 64 physical/128 logical CPUs,
  8 NUMA nodes, 1,056,745,820 kB total memory.
- Compilation emitted three `-Wsubobject-linkage` warnings because the
  contextual executor owns the existing translation-unit-private `FileState`
  type. Tests emitted only the pre-existing pytest cleanup warnings for stale
  `/tmp/pytest-of-bronsonj/garbage-*` directories.

## Scope completed

### Native complete reference statistics

- Extended `ContextualReferenceExecutorV1` additively. Supplying no variant
  probes preserves the Stage 2 incomplete-result path and exact existing
  defaults; Stage 3 activates only with one valid explicit or NumPy-Philox
  variant-probe representation and `B_D >= 2`.
- Sealed a stable group-contiguous permutation with unique/complete coverage,
  group offsets/counts/masses, and a permutation digest. Group-restricted
  execution decodes every retained logical variant once per restricted pass.
- Implemented the production-selected group-restricted traversal:
  `H_gkt = G B_g A_k S_t`, unprojected contextual group actions, protected
  cross-Gram TN against resident full projected raw actions, one global
  `1/B_T` factor, symmetric per-group raw numerators, and reconstruction of
  the annotation-mass-unnormalized full Gram numerator.
- Implemented direct grouped TN with action-scaled and genotype-scaled row
  placement. Direct-only and differential modes are scientifically qualified
  by tests, but no production cost-model selector was added.
- Implemented the signed variant-probe estimator with `sqrt(A_k)` weighting,
  contextual projection, exact diagonal/off-diagonal `eta`, persistent global
  probe sums and within-probe cross-products across all probe tiles, and one
  final `B_D(B_D-1)` denominator after the global in-process merge. Negative
  and indefinite values are preserved.
- Added Stage 3 semantic coordinates, group ranges, call/pass/record ledgers,
  group reconstruction verification, group and same-person phase seals, and
  scratch release before final compact publication. Group-target variant
  coordinates are positions in the sealed group-execution permutation; the
  group range and permutation digest bind them to logical variants.
- Published only compact native statistics and identities. The native result
  truthfully reports `complete_reference_statistics=true` and
  `complete_reference_artifact=false`; it contains no sample- or variant-axis
  scientific array. Wide source/action panels remain native-internal, with a
  constructor-gated defensive differential snapshot for tests only.

### Stable Python reference V1 boundary

- Added the isolated `ContextualReferenceArtifactV1` family with exact logical
  schema/grouped encoding/native API/backend/build axes, canonical pair and
  component maps, exact scale/probe/map identities, raw grouped units,
  deletion policy, diagnostics, admission, and protected-event evidence.
- The adapter admits an exact 81-key native result, reconstructs protected
  call counts from events, validates lifecycle order, pass/block/visit and
  memory ledgers, policy strings, dtypes/shapes/digests, canonical maps,
  group reconstruction, NUMA non-claims, ownership, and terminal state before
  publication.
- Arrays are defensively copied into canonical little-endian, C-contiguous,
  bytes-backed immutable storage. Nested metadata is recursively frozen.
- Added one atomic `.contextual-reference-v1.npz` writer and strict loader with
  isolated suffix/magic, duplicate-key rejection, an externally stored
  canonical manifest digest, exact archive keys, and typed per-array digests.
  The V1 and development reference loaders fail closed across the tested
  cross-family cases; the V1 loader also rejects legacy/development suffixes.
- Added exact single/multi-group approximate deletion from raw numerators,
  rejecting unknown/duplicate groups and any deletion that empties an
  annotation. Full same-person `D_R` is intentionally reused unchanged.
- Added an explicit private bridge to the development grouped reference type
  for the pre-Stage-4 fitter. The bridge revalidates the stable artifact and
  requires caller-supplied development identities and an explicit basis
  metric for context surfaces; stable loaders/writers never emit or accept the
  development format.

### Repairs made during implementation review

The settled code includes repairs for: direct-TN diagonal components that were
initially skipped; source/group target scratch sizing; same-person projection
capacity; exact Stage 3 pass totals; strict group anchors; group ranges in
semantic coordinates; reconstruction fault magnitude; release of group
scratch before same-person work; canonical C-order native array digests;
native/adapter terminal-state and variant-probe policy vocabularies; exact
native-result key alignment; authoritative native core-plan identity;
event-derived publication validation; closed memory/NUMA/file/backend evidence;
non-vacuous group tiling; direct-only execution; and explicit variant-Philox
coverage.

No trait summary, native fit, hidden regularization, PSD projection, legacy
scientific refactor, public CLI path, NUMA placement, two-process merge API, or
production direct-grouped planner was added.

## Changed files

| Path | Symbols/areas | Reason |
|---|---|---|
| `src/native/contextual_streamed_reference_v1.inc` | Stage 3 constructor options, group plan/traversals, same-person phase, admission, telemetry, compact publication | Complete descriptor-owned contextual reference statistics without changing legacy bindings/defaults. |
| `src/summit/context/reference_v1.py` | stable artifact, publisher, strict loader/writer, deletion helper, development bridge | Add the isolated immutable V1 publication boundary. |
| `src/summit/context/reference.py` | `load_context_reference` suffix guard | Reject stable V1 containers before the development loader reads them. |
| `src/summit/context/__init__.py` | V1 imports and `__all__` | Export the explicit stable API. |
| `tests/test_context_stage3_complete_reference_native.py` | 119 native-path cases, including 90 tile configurations within the tile tests | Qualify real descriptor execution, algorithms, deletions, faults, admission, and compactness. |
| `tests/test_context_stage3_reference_v1.py` | 35 adapter/artifact cases | Qualify immutable schema, tamper rejection, round trip, loader isolation, and bridge behavior. |
| `reports/context_native/stage3_complete_reference.md` | this report | Record evidence, limits, and the Stage 4 gate. |

`gxeldcore.cpp` itself is unchanged from Stage 2; the existing isolated include
and additive binding call load the extended implementation. Module versions,
legacy classes, global BLAS state, and legacy telemetry defaults are unchanged.

## Scientific verification

Focused comparisons use `atol=1e-9, rtol=1e-11`. Maximum errors below were
collected from the settled qualification binary.

| Test/invariant | Command | Result | Maximum error |
|---|---|---|---:|
| Generated BED, `Q=1..4`, strict/generic annotations: full Gram versus Python oracle | focused Stage 3 command below | 8/8 passed | `4.97e-14` |
| Same cases: signed same-person `D_R` versus variant-probe Python oracle | same | passed, signed output preserved | `4.26e-14` |
| Same cases: group-restricted raw numerators versus Python grouped oracle | same | passed | `2.27e-12` |
| `sum_g GNUM_g` versus native probe-averaged raw full-Gram numerator | same | passed | `1.82e-12` |
| Restricted, action-scaled direct, and genotype-scaled direct versus Python, strict and generic | same | passed; direct-only uses one placement/pass | restricted `9.09e-13`; action `1.36e-12`; genotype `1.82e-12`; placement difference `4.55e-13` |
| `B_D=2`, `d=1`, cross-tile probe pairs | same | passed against global Python U-statistic | within declared tolerance |
| Named diagonal/diagonal, diagonal/off-diagonal, and off/off entries | same | explicit grouped and `D_R` factors 1/2/4 passed | within declared tolerance |
| Every individual width for variant, resident probe, probe, action, annotation, context, variant-probe, and group tiles | same | 90 strict/generic configurations passed; selected width and compact science exact within tolerance | within declared tolerance |
| Explicit Xi versus NumPy-Philox Xi, tile 1/full, decode threads 1/2 where available | same | policies/identities distinct and science invariant | within declared tolerance |
| Actual native artifact through full fit/Omega/surface | same | passed | coefficients/Omega `2.26e-13`; surface `3.89e-13` |
| Every single-group deletion through fit/Omega/surface | same | all 11 groups passed | coefficients/Omega `3.15e-12`; surface `9.76e-12` |
| Selected 2- and 3-group deletions through fit/surface | same | passed | coefficients `2.29e-13`; surface `4.28e-13` |
| Projection and symmetry terminal diagnostics | same | passed | leakage `3.72e-14`; Gram and same-person pre-symmetry `0` |
| Compact result and stable round trip | same | exact closed keys/shapes/dtypes/digests; no `N`/`M` scientific axis | structural equality |
| Stage 3 focused suite | `pytest -q tests/test_context_stage3_reference_v1.py tests/test_context_stage3_complete_reference_native.py` | 154 passed in 5.80 s | values above |

The Q fixtures use seeded/deterministic generated genotypes, bases,
annotations, groups, and probes. The tile qualification independently sweeps
every admissible width for each axis while holding the others full; it is not
a Cartesian product of all tile axes at all widths.

## Integrity and lifecycle verification

| Operation/failure | Injection/check | Detection/repair/fallback | Result |
|---|---|---|---|
| `group_target_nn`, `group_cross_gram_tn`, `direct_grouped_tn`, same-person target NN, projection TN/NN, and Gram TN | one-shot, `NaN`, `Inf`, canary damage | primary rejected; verified deterministic retry | 28/28 operation/mode cases passed |
| Same 7 operations | repeated primary/retry corruption, repair corruption, forced fallback | retry rejected when applicable; independently recomputed scalar fallback verified | 21/21 passed |
| Same 7 operations | fallback corruption or failure | terminal exception; no result | 14/14 failed closed |
| Same 7 operations | operand or authenticated runtime mutation | operand/runtime fingerprint mismatch | 14/14 failed closed |
| Semantic group reconstruction | corrupt one grouped numerator beyond scale-aware tolerance | global reconstruction check before publication | failed closed |
| Stable semantic targeting | one admitted point anchor for each new operation, including group execution-range coordinates | selector is consumed once; unmatched/invalid target rejects | passed in the selected generic schedules |
| Workspace and telemetry admission | exact reported capacities and each capacity minus one | exact accepted; minus one rejected before descriptor execution | passed |
| Group execution permutation | internal mutation at `post_seal` | plan fingerprint mismatch | failed closed |
| Phase and call lifecycle | exact source/action/group/same transitions, eight checkpoints, scratch-release and terminal publication events | adapter derives call counts from events and requires exact sequence/capacity/no drop | passed; tampered event streams rejected |
| Artifact ownership/tamper | caller aliasing, array writes, nested metadata mutation, dtype/shape/key/digest/manifest changes | bytes-backed immutable arrays and strict revalidation | passed or failed closed as appropriate |
| One-shot/lifetime | repeated native `run()`, executor destruction, native/stable result lifetime | state rejection and independently owned arrays | passed |

The event records are useful and fail closed in this envelope, but are not the
complete production records required by `COMMON/06`: they do not serialize
elapsed time, timestamps, process/slot identity, exact operand/runtime
fingerprint values, witness/checksum/canary values, or a canonical event digest.

## Exact execution ledgers

The frozen scientific units are explicit. For restricted group `g`,

- `H_gkt = G B_g A_k S_t`;
- `X_gc = D_q H_gkr + 1[q != r] D_r H_gkq`;
- `C_g = X_g,flat^T W_raw,flat / B_T`;
- `GNUM_g = (C_g + C_g^T) / 2`, with
  `sum_g GNUM_g = M_c M_d T_R`.

For same-person component `c=(k,q,r)`,
`g_cib = eta_c R_kq,ib R_kr,ib / M_k`. With
`S=sum_b g_b`, the signed statistic is
`D_R=(S^T S-sum_b g_b^T g_b)/(B_D(B_D-1))` after the global probe merge.
Deleting group set `S_g` uses
`T_R^(-S_g)=sum_(g not in S_g)GNUM_g/(M_c^(-S_g)M_d^(-S_g))` and reuses the
full `D_R` unchanged.

Let `L=ceil(M/V)`, `R=ceil(B_T/B_R)`, `P_T` be the sum of sample-probe tiles
over resident batches, `Q_t=ceil(Q/q_t)`, `K_t=ceil(K/k_t)`,
`A_t=ceil(C/a_t)`, `G_T=ceil(B_T/b_t)`,
`L_g=sum_g ceil(M_g/V)`, and `D_t=ceil(B_D/d_t)`.

The admitted protected-call ledger is:

- sample-probe projection TN/NN: `P_T` each;
- source TN: `P_T L Q_t`;
- generic full target NN: `P_T L K_t Q_t`; strict full target NN:
  `P_T Q_t` times the sum of active annotations over logical variant blocks;
- action projection TN/NN: `P_T A_t` each; full Gram TN: `R`;
- generic group target NN: `L_g G_T K_t Q_t`; strict group target NN:
  `G_T Q_t` times the sum of active annotations over group-execution blocks;
- group cross-Gram TN: `J A_t`;
- direct grouped TN: `placements L Q A_t G_T`, where `placements=2` only
  for the differential/both-placement plan and otherwise one when selected;
- same-person target NN: `D_t L`; projection TN/NN: `D_t` each;
  same-person Gram TN: `D_t+1` for within-tile terms plus the global sum term;
- trait operations: zero in Stage 3.

Descriptor passes are `R` source, `R` action, one per restricted grouped
traversal plus one per direct placement, and `D_t` same-person. Decoded blocks
are `R L`, `R L`, `L_g` plus `placements L`, and `D_t L`, respectively.
Retained logical visits equal each phase's passes times `M`. Logical BED bytes
touched equal visits times the SNP-major record width `ceil(N_total/4)`. These
are deterministic logical mmap-record touches; OS physical read bytes and page
faults are explicitly reported as unmeasured.

Telemetry capacity is admitted as `8 * total_protected_calls + 32` fixed POD
events. The checked payload threshold is:

`permanent + max(source_phase, action_phase) + group_phase + same_person_phase + integrity_reserve + compact_output + telemetry`.

This is the implemented accounting formula, not measured RSS or a complete
allocator-level peak guarantee.

## Legacy regression

All commands below used the settled qualification binary. Together they cover
all tests under `tests/`.

| Command | Result | Change from pre-Stage-3 baseline |
|---|---|---|
| `pytest -q tests/test_context_*.py` | 626 passed in 57.82 s | 154 new Stage 3 passes; no failures |
| Stage 0 fixtures/schema + Stage 1 dense + Stage 2 streamed focused files | 255 passed in 7.96 s | none observed |
| `pytest -q tests/test_gxe_[a-m]*.py` | 213 passed, 3 skipped, 1 XPASS | no failing change |
| `pytest -q tests/test_gxe_[n-z]*.py` | 152 passed, 1 skipped | none observed |
| Remaining root test files (`early_numa` through manifest tests) | 124 passed | none observed |
| `pytest -q tests/gxe_completion` | 283 passed, 1 skipped | none observed |
| **Settled total** | **1,398 passed, 5 skipped, 1 XPASS** | **154 new passes over the established 1,244-pass pre-Stage-3 partition; no failures** |

The only test warnings were stale pytest temporary-directory cleanup errors.
The no-variant-probe constructor branch remains the Stage 2 API/result, and
legacy DirectContext/MultiEnvironment classes, versions, algorithms, and
defaults are unchanged.

## Performance and admission

This is a small deterministic correctness benchmark, not a tabla performance
qualification. It imports the checked-in `_make_case`, `_variant_probes`, and
`_stage3_executor` helpers, times 15 sequential fresh constructor/run pairs
with `perf_counter_ns`, and collects the final preflight/result ledgers.

- Dimensions: `N=13`, `M=11`, `Q=4`, `K=2`, `C=20`, `J=3`, `B_T=5`,
  `B_D=5`, fp64, one decode thread and one protected-GEMM thread; 15 fresh
  executors per plan.
- Full plan (`V=11`, resident/probe tile 5, action 20, annotation 2, context 4,
  variant-probe tile 5, group tile 3): median constructor 0.783 ms; median run
  2.330 ms (2.283–5.610 ms); 4 descriptor passes, 6 decoded blocks, 44
  retained-variant visits, 18 protected calls.
- Tiled plan (`V=3`, resident 3, probe 2, action 3, annotation 1, context 2,
  variant-probe tile 2, group tile 1): median constructor 0.753 ms; median run
  5.971 ms (5.908–6.282 ms); 8 descriptor passes, 33 decoded blocks, 88
  visits, 225 protected calls.
- Full-plan tracked threshold: 294,742 bytes = 34,499 permanent + 63,835
  shared source/action + 76,024 group + 27,400 same-person + 42,112 integrity
  + 19,896 compact output + 30,976 telemetry; telemetry capacity 176.
- Tiled tracked threshold: 448,446 bytes = 34,499 permanent + 17,331 shared
  source/action + 27,104 group + 13,872 same-person + 13,312 integrity +
  19,896 compact output + 322,432 telemetry; telemetry capacity 1,832. On
  this small fixture, the conservative per-call event reserve outweighs the
  smaller numerical panels.
- Scratch payloads are allocated before the first descriptor decode. Compact
  NumPy outputs are allocated at publication and charged by formula. The
  published `tracked_high_water_bytes` is the planned tracked bound, not an
  observed high-water or RSS measurement.
- Only `unbound_first_touch_v1` with output node `-1` is accepted. The artifact
  truthfully records `numa_applicable=false` and `numa_verified=false`.
- No phase timings, process RSS, page placement, physical I/O/page faults,
  remote-access counters, B=128/B=1024 ladder, or realistic descriptor-scale
  throughput was measured.

## Remaining risks/blockers

1. **Admission is tracked payload accounting, not total-process peak
   admission.** Constructor sealing copies numeric plans, explicit probes, and
   M-scale IDs/allele strings and parses BIM before the workspace cap check.
   The ledger excludes string payload/object overhead, BIM parse temporaries,
   allocator/vector metadata, OpenMP/runtime allocations, Python event
   dictionaries, adapter immutable-copy coexistence, and explicit headroom.
   Publication allocates compact NumPy outputs after decode. Exact/cap-minus-
   one therefore proves formula enforcement, not measured or allocator-
   enforced peak memory. Stage 4 must not use it as a production tabla bound.
2. **File evidence is not adversarial immutability.** Checkpoints use duplicated
   `CLOEXEC` descriptors, bracketed fstat identities, and first/last-4-KiB
   boundary samples. There is no full BED content digest, immutable snapshot/
   lease, or elimination of the final check-to-return race. Interior changes
   that evade metadata are outside the claim.
3. **NUMA ownership remains unimplemented.** The executor accepts only the
   unbound policy and reports no placement verification on this eight-node
   host. No two-socket/process placement or remote-access test ran.
4. **Telemetry and provenance are incomplete.** Protected events lack full
   timing/fingerprint/checksum/process evidence. The stable artifact records
   native API/backend/build IDs and the embedded source-tree digest, but not a
   native binary digest, dirty flag, compiler flags, ISA, OpenMP/BLAS identity,
   full file identities, or a canonical event digest. Those facts were
   measured externally for this report, not frozen in each artifact. The new
   Stage 3 fault selectors are point-addressed, but the checked-in matrix does
   not retarget the same injected point under two different tile schedules.
5. **`sealed_plan_sha256` is a partial science/configuration digest, not a
   complete execution-plan digest.** It binds core scientific array/map,
   scaling, probe, group-permutation, and selected grouped-policy identities,
   but omits the expected/observed missingness digest, files, labels, tiles,
   threads/affinity, NUMA/output node, caps, backend/build, and fault policy.
   Rename or expand the contract before production use.
6. **Semantic label authority is incomplete.** The Python publication identity
   independently binds sample order, variant/allele order, fixed-effect and
   basis identities. Native weight and group-assignment hashes plus the final
   manifest bind returned annotation/group names, but there is no independent
   Python-owned annotation-name/order or deletion-group label/map identity.
   A trusted caller/native boundary is therefore required to prevent semantic
   relabeling.
7. **Direct grouped TN is not production-selected.** Both placements and the
   direct-only branch are scientifically correct in the test envelope, but no
   admitted cost model or representative benchmark qualifies selection. The
   stable publisher intentionally accepts only group-restricted attribution.
8. **Variant-probe merging is in-process only.** Probe tiles merge globally
   before finalization, but there is no public two-process/socket merge object
   for `probe_sums` and `same_probe`. A distributed plan must expose and
   validate that merge before computing `D_R`.
9. **Performance and randomized evidence remain fixture-scale.** No memory
   ladder, realistic BED size, physical I/O, page-fault, RSS/leak, NUMA, or
   protected-operation throughput benchmark exists. The checked-in Stage 3
   suite has no regenerated multi-seed randomized matrix, and individual tile
   widths were qualified rather than their full Cartesian product at
   production dimensions.
10. **The Stage 4 bridge is intentionally private.** Stable reference V1 is
    complete, but the current trait summary remains the development family.
    The explicit bridge needs caller-supplied development identities and
    basis metric. Stage 4 must add a matching stable trait boundary rather
    than silently relabel either artifact family.

## Reproduction commands

```bash
cd /home/bronsonj/SUMMIT
git checkout codex/contextual-stage3-complete-reference
git checkout 07720fe0819cf3934abe1a28ba4922771480ad32
git status --short --branch

stage3_build_dir=$(mktemp -d /tmp/summit-stage3-qual-build.XXXXXX)
stage3_install_dir=$(mktemp -d /tmp/summit-stage3-qual-install.XXXXXX)

cmake -S . -B "$stage3_build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=/home/bronsonj/anaconda3/envs/summit/bin/python \
  -DBLA_VENDOR=OpenBLAS \
  -DOPENBLAS_LIB=/home/bronsonj/anaconda3/envs/summit/lib/libopenblas.so \
  -DCBLAS_INCLUDE_DIR=/home/bronsonj/anaconda3/envs/summit/include \
  -DGWLDCORE_ENABLE_NATIVE_OPT=ON \
  -DGWLDCORE_USE_OPENMP=ON \
  -DGXELDCORE_GEMM_INTEGRITY=ON \
  -DGXELDCORE_GEMM_CHECKSUM=ON \
  -DGWLDCORE_SOURCE_COMMIT=auto \
  -DGWLDCORE_SOURCE_TREE_SHA256=auto
cmake --build "$stage3_build_dir" --parallel 2
cmake --install "$stage3_build_dir" --prefix "$stage3_install_dir"

export LD_LIBRARY_PATH=/home/bronsonj/anaconda3/envs/summit/lib
export PYTHONPATH="$stage3_install_dir"

/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage3_reference_v1.py \
  tests/test_context_stage3_complete_reference_native.py

/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_*.py
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_gxe_[a-m]*.py
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_gxe_[n-z]*.py
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_early_numa.py tests/test_exact_loco_fast.py tests/test_gwld_mc.py \
  tests/test_h2_batch_fast.py tests/test_harmonization_trace.py \
  tests/test_ldsc_h2.py tests/test_ldsc_rg.py tests/test_memory_budget.py \
  tests/test_pgen_gwld.py tests/test_rg_manifest_fast_parity.py \
  tests/test_rg_manifest_zero_overlap.py tests/test_rg_model_manifest.py
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/gxe_completion

/home/bronsonj/anaconda3/envs/summit/bin/python - <<'PY'
from summit import gxeldcore
print(dict(gxeldcore.build_info()))
print(tuple(gxeldcore.contextual_complete_reference_semantic_operations_v1()))
PY
```
