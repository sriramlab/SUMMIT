# Contextual native estimator — Stage 4 report

## Decision

`GO` for Stage 5 integrity and schema hardening.

The Stage 4 gate passes. The descriptor-owned native trait path projects and
normalizes an admitted phenotype batch once, visits every retained variant in
one logical traversal, matches the frozen Python trait oracle for all compact
moment families, and publishes through an isolated immutable trait V1
boundary. Stable reference and trait artifacts fit in a fresh process after
the row-level source has been removed. Full and deleted raw coefficients,
Omegas, and context surfaces match the established Python implementation for
strict-disjoint and generic-overlap annotations. Rank failure remains a hard
failure; no ridge, pseudoinverse, clipping, or hidden PSD projection was added.

The settled qualification contains 41 new Stage 4 tests. Stage 3 plus Stage 4
has 195 passing tests, the contextual suite has 667, and all repository test
partitions total 1,439 passed, 5 skipped, and 1 expected XPASS with no settled
failure. Stage 3's exact 81-key stable reference boundary and its zero-trait
call ledgers remain unchanged.

This decision does not claim release or tabla-scale readiness. Total-process
RSS admission, adversarial file immutability, complete execution provenance,
NUMA placement, production performance, and the other Stage 3 deferred
hardening items remain assigned to Stages 5 and 6. Stage 4 exposed no direct
scientific contradiction in those deferred items.

## Repository/build state

- Branch: `codex/contextual-stage4-trait-summary-fit`, created from the clean
  Stage 3 report commit
  `6c101fa26e29c826113cd831c0cc57337d575893`.
- Settled implementation/test commits:
  `8c46ba47ad8e6893a72d8d8f3ebb2847012fea90` and
  `d87909df21a49854886b7defcd1d6053f6c7dd01`. The worktree was clean when the
  qualification build was configured. This report is added afterward in a
  documentation-only commit.
- Native implementation source SHA-256:
  `8a0719ec482387c64bd5f0cf2d160da79d017208f0dea04be8e10de92f7dc292`.
- Embedded native provenance: source commit `d87909d...`; source-tree SHA-256
  `4d751ae6b99596085ebeaad3d4471f2cb379fcfb7f3f0b74b9b23dcf428f7840`.
- Clean build: `/tmp/summit-stage4-qual-build.pAQ9UJ`; clean install:
  `/tmp/summit-stage4-qual-install.sZK0N3`. Existing build products were not
  overwritten.
- Installed `gxeldcore` SHA-256:
  `3f6365844c62bd5c9112b4629e9eab02d16c29c5d87066a7205f7e29d3fa4e71`.
- Compiler/build: GNU C++ 12.2.0, C++17, Release,
  `-O3 -march=native -fopenmp`; native optimization, OpenMP, contextual GEMM
  integrity, and checksums enabled.
- Native module/backend/API remain 1.7/1.9/9. Shared OpenBLAS 0.3.34 uses the
  Zen pthread runtime. Stage 4 adds exactly three contextual semantic
  operations: `trait_score_tn`, `trait_feature_projection_tn`, and
  `trait_feature_projection_nn`.
- Host: Linux `Tabla`, AMD EPYC 7501, 2 sockets, 64 physical/128 logical CPUs,
  8 NUMA nodes, 1,082,107,719,680 bytes total memory.
- Compilation emitted only the existing anonymous-namespace linkage warnings
  for translation-unit-private descriptor state. Tests emitted only the
  pre-existing pytest cleanup warnings for stale
  `/tmp/pytest-of-bronsonj/garbage-*` directories.

The qualification configure set `CONDA_PREFIX` to the SUMMIT environment so
the installed extension retained the correct OpenBLAS runtime path. An earlier
temporary build made without that environment embedded the base-Anaconda
runtime path and was discarded from the evidence.

## Scope completed

### Descriptor-owned native trait statistics

- Added a trait-only mode behind the descriptor/integrity machinery already
  owned by `ContextualReferenceExecutorV1`, exposed as the separate callable
  `gxeldcore.ContextualTraitExecutorV1`. Reference construction, defaults, and
  compact publication remain on their exact Stage 3 branches.
- The immutable native plan owns duplicated BED/BIM/FAM descriptors, retained
  sample and variant maps, allele orientation, the sealed affine scale plan,
  fixed-effect basis, evaluated contextual basis, annotations and groups,
  phenotype batch, residual basis and names, tiles, caps, fault policy, and
  expected semantic call ledger.
- Each phenotype is projected against the fixed-effect basis and normalized
  exactly once. Residual RHS, traces, and Gram use compact low-rank formulas.
- Each admitted phenotype batch performs one descriptor traversal. Every
  retained genotype block is decoded once, all `G_v^T[D_q y_l]` scores are
  packed into one protected TN call per block, contextual features are formed
  and projected in the frozen order, and full/grouped moments are reduced
  before the block is released.
- Native compact arrays use canonical C-order layouts:
  `genetic_rhs[C,L]`, `residual_rhs[H,L]`,
  `genetic_residual[C,H]`, `group_rhs[J,C,L]`,
  `group_trace[J,C]`, and `group_genetic_residual[J,C,H]`. No scientific
  output has a sample or variant axis.
- Publication reports complete trait statistics but not a complete Python
  artifact. It requires exact descriptor/pass/visit/protected-call ledgers,
  projection and normalization diagnostics, grouped reconstruction, input and
  file checkpoints, scratch release, terminal lifecycle, and one-shot use.

### Stable trait V1 boundary

- Added the isolated `ContextualTraitArtifactV1` family and exact 76-key native
  adapter. The adapter validates the logical schema, separate grouped physical
  encoding, canonical pair/component maps, scale and allele identity, basis
  specification/calibration, annotations/groups, phenotype/residual labels,
  numeric and deletion policies, native API/backend/build, admission,
  diagnostics, lifecycle, and event-derived call counts.
- Core arrays are defensive canonical little-endian, C-contiguous,
  bytes-backed immutable copies; nested metadata is recursively frozen.
- Added atomic strict `.contextual-trait-v1.npz` I/O with isolated magic and
  suffix, exact keys, duplicate rejection, external canonical manifest digest,
  and typed per-array digests. The development trait loader rejects the stable
  suffix before interpreting the archive.
- Added full and arbitrary valid multi-group deletion of raw grouped
  numerators. Unknown or duplicate groups and deletion of all mass from any
  annotation fail closed.

### Stable summary-only fit V1 boundary

- Added an explicit compatibility check between stable reference and trait
  artifacts. It requires the shared scale, retained variant/allele order,
  basis specification/calibration, canonical component map,
  annotation/group maps and labels, grouped/numeric/deletion policies, native
  API, and build identity. Cohort-specific evaluated bases, sample maps,
  missingness, and residual rank are correctly not equated.
- The intentional native backend pair is closed and versioned:
  reference `plink_bed_descriptor_stream_stage2_v1:1`, trait
  `plink_bed_descriptor_stream_trait_v1:1`, under policy
  `exact_reference_stage2_trait_v1_backend_pair_v1`. The fit backend is
  separately identified as `python_numpy_scipy_summary_fit_v1`; any other pair
  fails closed.
- Normal-equation transfer uses study sample size `N`, not residual rank. A
  trait selector is mandatory for `L>1` and optional only when `L=1`.
- Full and every-group leave-one-out systems use retained annotation masses and
  the full signed same-person statistic. The solver is rank checked and never
  silently uses ridge or a pseudoinverse.
- The stable fit stores raw coefficients, packed raw Omegas, solve rank and
  residual diagnostics, every-group raw leave-one-out coefficients,
  balanced-jackknife covariance/standard errors, and raw full plus every-group
  deletion surfaces. Optional PSD output is separately named and never
  replaces the raw result.
- Strict-disjoint per-annotation surfaces are standalone. Generic-overlap
  per-annotation surfaces remain labeled conditional contributions, and the
  combined total is published by default.
- Added atomic strict `.contextual-fit-v1.npz` I/O with immutable arrays and
  recomputation of jackknife covariance, Omega packing/eigenvalues, basis
  metric checks, full/deleted surfaces, and combined overlap surfaces during
  verification/load. The development fit loader rejects the stable suffix.

### Fresh-process summary-only gate

The checked-in test creates row-level genotype and phenotype inputs in a
builder process, publishes one stable reference and one stable trait artifact,
waits for that process to terminate, removes the row-level source, and starts a
second process. The second process sees only those two stable NPZ inputs,
selects a trait, fits, writes and reloads a stable fit artifact, and verifies
all leave-one-out and surface panels. Serialized keys are inspected to reject
sample- or variant-axis leakage.

No public CLI, native fit, legacy scientific refactor, hidden regularization,
default PSD projection, production direct-grouped planner, NUMA placement, or
tabla-scale optimization was added.

## Changed files

| Path | Symbols/areas | Reason |
|---|---|---|
| `src/native/contextual_streamed_reference_v1.inc` | trait-only plan, preflight, residual moments, one-pass descriptor traversal, protected reductions, compact publication, semantic binding | Reuse the approved descriptor/integrity owner while preserving the Stage 3 reference boundary. |
| `src/summit/context/trait_v1.py` | stable identity/artifact/moments, exact native adapter, deletion helper, runner, strict I/O | Add the isolated immutable production trait boundary. |
| `src/summit/context/fit_v1.py` | compatibility, normal-equation assembly, selected-trait fit, LOO/jackknife, surfaces, strict I/O | Add summary-only stable fitting without changing the raw target. |
| `src/summit/context/summary.py` | stable-suffix guard | Prevent the development trait loader from accepting stable V1 containers. |
| `src/summit/context/fit.py` | stable-suffix guard | Prevent the development fit loader from accepting stable V1 containers. |
| `src/summit/context/__init__.py` | stable trait/fit exports | Expose the explicit V1 APIs without changing legacy APIs. |
| `tests/test_context_stage4_trait_native.py` | 19 native and end-to-end differential cases | Qualify Q/annotation/allele cases, one-pass ledgers, admission, faults, and native-to-stable fits. |
| `tests/test_context_stage4_trait_v1.py` | 7 stable trait cases | Qualify adapter, immutability, deletion, strict I/O, tamper rejection, and evidence validation. |
| `tests/test_context_stage4_fit_v1.py` | 15 stable fit cases | Qualify selectors, transfer, deletions, interpretations, rank/PSD separation, compatibility, I/O, and fresh-process operation. |
| `reports/context_native/stage4_trait_and_fit.md` | this report | Record the evidence, limits, and Stage 5 gate. |

`src/native/gxeldcore.cpp` is unchanged. The existing isolated include and
binding hook expose the additive callable; module versions, legacy classes,
global BLAS state, and legacy defaults are unchanged.

## Scientific verification

Focused numerical comparisons use `atol=1e-9, rtol=1e-11`. Maximum errors
below were collected from the settled qualification binary.

| Test/invariant | Command | Result | Maximum error |
|---|---|---|---:|
| Native trait versus Python oracle, `Q=1..4`, strict/generic, `K=2,H=2,L=3`, A1 counted | Stage 4 native focused suite | 8/8 cases passed | genetic RHS `2.13e-14`; traces/cross `1.07e-14` |
| A2 allele conversion, `Q=4`, generic overlap | same | passed with sealed orientation | included above |
| Residual-only RHS, traces, and Gram | same | all cases passed | RHS `3.55e-15`; traces `3.55e-15`; Gram `1.78e-15` |
| Grouped RHS/trace/genetic-residual raw numerators | same | reconstruction and oracle passed | `1.42e-13`; `1.42e-14`; `1.42e-14` |
| Annotation/group masses and counts | same | passed | masses `8.88e-16`; counts exact |
| Projection and phenotype normalization | same | passed | leakage `2.47e-15`; norm error `3.55e-15` |
| Variant/feature tiling | same | tiled/full compact science invariant | within declared tolerance |
| Strict native reference + native trait through stable fit: full/deleted raw coefficients/Omega/surfaces | same | strict and overlap passed | see rows below |
| Strict-disjoint end-to-end fit | same | full plus all groups passed | coefficients `5.66e-15`; LOO `1.55e-14`; Omega `5.66e-15`; surfaces `4.44e-14` |
| Generic-overlap end-to-end fit | same | conditional and combined full plus all groups passed | coefficients/Omega `8.37e-13`; LOO `1.52e-12`; surfaces `3.50e-12` |
| `N` versus residual-rank transfer | Stage 4 fit suite | exact study-`N` behavior and counterexample passed | structural/formula equality |
| Raw negative/indefinite result and optional PSD output | same | raw preserved; PSD separately named | invariant recomputation passed |
| Rank-deficient raw system | same | failed closed; no ridge/pseudoinverse | expected exception |
| Reference/trait compatibility axes | same | scale, build/backend, basis, annotation, variant, group mismatches rejected | expected exceptions |
| Fresh-process summary-only fit | same | row source removed; stable inputs only; fit round trip passed | exact compact invariants |
| Stage 4 focused suite | `pytest -q tests/test_context_stage4_trait_native.py tests/test_context_stage4_trait_v1.py tests/test_context_stage4_fit_v1.py` | 41 passed | values above |
| Stage 3 + Stage 4 focused suite | six focused files | 195 passed in 8.17 s | no Stage 3 change |

## Integrity and lifecycle verification

| Operation/failure | Injection/check | Detection/repair/fallback | Result |
|---|---|---|---|
| All three trait protected operations | one-shot semantic corruption | primary rejected; deterministic retry verified | 3/3 passed with unchanged science |
| All three trait protected operations | repeated corruption | primary/retry rejected; scalar fallback verified | 3/3 passed with unchanged science |
| Trait semantic coordinates | exact operation anchors and call-derived telemetry | selected target consumed once; calls tied to dimensions/transpose/anchor | passed |
| Workspace and telemetry admission | exact preflight capacity and cap minus one | exact accepted; insufficient capacity rejected before traversal | passed |
| Descriptor traversal | `Q=4,K=2,H=2,L=3`, `M=11`, `V=4` | exact planned/observed pass, block, and visit ledgers | 1 pass, 3 blocks, 11 visits |
| Protected-call ledger | same case, feature tile 5 | event-derived exact comparison | 3 score TN + 11 projection TN + 11 projection NN |
| Inputs and descriptors | sealed fingerprints plus lifecycle checkpoints | mutation or checkpoint inconsistency prevents publication | passed |
| Output ownership | native/stable write attempts and caller alias mutation | read-only native outputs; bytes-backed stable copies | passed/failed closed as appropriate |
| Stable archive schema | suffix/magic, exact keys, duplicate keys, manifest and typed-array digests | cross-family/tampered artifacts rejected before use | passed |
| Compact semantic invariants | grouped reconstruction, jackknife covariance, Omega, ranks, metrics, surfaces | recomputed at adaptation, verification, and load | passed; tampering rejected |
| One-shot/lifetime | repeated run, executor destruction, fresh-process load | state rejection and independently owned publication | passed |

The trait result has an exact 76-key native boundary. The stable reference
adapter continues to require its exact 81-key Stage 3 result and zero trait
calls, so trait fields cannot leak into or silently relabel that family.

## Exact trait execution ledger

For phenotype `l`, the admitted normalized phenotype is
`y_l = P_U y_l / ||P_U y_l||`. Every retained variant is decoded once into a
scaled genotype block. One packed score call per block computes every
`G_v^T[D_q y_l]`; contextual components follow the frozen
annotation-major, diagonal-then-off-diagonal map. The projected feature block
is reduced immediately into full and group-restricted RHS, trace, and
genetic-residual numerators.

Let `V_b` be each decoded block width, `V` the admitted variant block,
`F` the feature tile, `M` retained variants, and `L_y` the phenotype batch
size. The Stage 4 trait call ledger is:

- `trait_score_tn = ceil(M/V)`;
- `trait_feature_projection_tn = sum_b ceil(Q V_b/F)`;
- `trait_feature_projection_nn = sum_b ceil(Q V_b/F)`;
- every reference, group-Gram, same-person, and legacy semantic operation is
  zero in trait-only mode.

The authoritative traversal evidence is one descriptor pass,
`ceil(M/V)` decoded blocks, `M` retained-variant visits, and `L_y` phenotype
projections. This is a logical mmap-record ledger; OS physical read bytes and
page faults remain explicitly unmeasured.

## Legacy regression

All settled commands used the clean qualification install.

| Command | Result | Change from Stage 3 baseline |
|---|---|---|
| Stage 3 + Stage 4 focused files | 195 passed in 8.17 s | 41 new Stage 4 passes; no failure |
| `pytest -q tests/test_context_*.py` | 667 passed in 56.61 s | 41 new passes; no failure |
| `pytest -q tests/test_gxe_[a-m]*.py` | 213 passed, 3 skipped, 1 XPASS | unchanged |
| `pytest -q tests/test_gxe_[n-z]*.py` | 152 passed, 1 skipped | unchanged in settled rerun |
| Remaining root test files | 124 passed | unchanged |
| `pytest -q tests/gxe_completion` | 283 passed, 1 skipped | unchanged |
| **Settled total** | **1,439 passed, 5 skipped, 1 XPASS** | **41 new passes; no settled failure** |

One initial `tests/test_gxe_[n-z]*.py` run observed an integrity-repair counter
of one where `test_api9_vendor_telemetry_records_protected_nn_and_tn` expected
zero. The test passed immediately in isolation and the complete partition then
passed. No Stage 4 code was changed in response. This transient telemetry
observation is carried into Stage 5 integrity hardening; it is not a settled
scientific or regression failure.

## Performance and admission

Stage 4 is a correctness and publication qualification, not the Stage 6 tabla
performance gate. A representative admitted native case records:

- dimensions `N=13`, `M=11`, `Q=4`, `K=2`, `C=20`, `J=3`, `H=2`, `L=3`;
  variant block 4 and feature tile 5;
- expected and observed descriptor work: 1 pass, 3 decoded blocks, 11 retained
  variant visits, 3 phenotype projections, and 25 protected calls;
- checked payload threshold 61,087 bytes = 6,547 permanent + 528 residual
  phase + 6,564 trait phase + 4,024 compact output + 2,592 integrity reserve +
  40,832 telemetry; required event capacity 232; maximum protected output 65
  fp64 elements;
- exact capacity is accepted and capacity minus one is rejected before
  descriptor execution;
- only the inherited unbound NUMA nonclaim is admitted; no placement or
  physical-I/O evidence is asserted.

The reported `tracked_high_water_bytes` is the checked payload formula, not an
allocator high-water or RSS measurement. Constructor objects, string storage,
runtime allocations, Python adaptation, and temporary coexistence remain
outside this bound and are Stage 5/6 work. No production-size timing,
throughput, page-fault, RSS, leak, or NUMA benchmark was used for this gate.

## Remaining risks/blockers

These are not Stage 4 contradictions; they remain assigned to their packaged
later stages.

1. **Integrity evidence is not yet the Stage 5 release schema.** Protected
   events still lack the complete timing, fingerprint/checksum values,
   process/slot identity, canonical event digest, and binary/toolchain/runtime
   provenance required by the packaged integrity gate.
2. **Admission is not total-process peak enforcement.** The native ledger is
   exact for its declared payload categories but excludes allocator/runtime,
   descriptor-plan string/object, adapter-copy, and RSS coexistence costs.
3. **Descriptor files are not immutable snapshots.** Duplicated descriptors,
   fstat checkpoints, and boundary samples do not eliminate undetected
   interior mutation or the final check-to-publication race.
4. **NUMA ownership is unimplemented.** No first-touch placement, page-query
   attestation, affinity contract, or two-socket/process qualification applies
   to the contextual executor.
5. **Plan and provenance identities remain partial.** The inherited
   `sealed_plan_sha256` omits execution axes identified in Stage 3; artifact
   provenance lacks an embedded native binary digest and complete compiler,
   ISA, BLAS/OpenMP, file, and dirty-tree identities.
6. **Independent label authority remains incomplete.** Annotation-name/order
   and deletion-group label/map identities still depend on the trusted native
   input boundary rather than an independently supplied Python authority.
7. **Direct grouped and distributed probe paths remain outside Stage 4.** The
   stable reference accepts only group-restricted attribution, and no public
   two-process variant-probe sufficient-statistic merge was introduced.
8. **Performance evidence is fixture scale.** There is no realistic BED ladder,
   production tile cost model, measured RSS/I/O/page faults, NUMA verification,
   or tabla-scale throughput comparison. Those are explicit Stage 6 gates.

## Reproduction commands

```bash
cd /home/bronsonj/SUMMIT
git checkout codex/contextual-stage4-trait-summary-fit
git checkout d87909df21a49854886b7defcd1d6053f6c7dd01
git status --short --branch

stage4_build_dir=$(mktemp -d /tmp/summit-stage4-qual-build.XXXXXX)
stage4_install_dir=$(mktemp -d /tmp/summit-stage4-qual-install.XXXXXX)

CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit cmake -S . \
  -B "$stage4_build_dir" \
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
cmake --build "$stage4_build_dir" --parallel 2
cmake --install "$stage4_build_dir" --prefix "$stage4_install_dir"

export CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit
export LD_LIBRARY_PATH=/home/bronsonj/anaconda3/envs/summit/lib
export PYTHONPATH="$stage4_install_dir"

/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage4_trait_native.py \
  tests/test_context_stage4_trait_v1.py \
  tests/test_context_stage4_fit_v1.py

/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage2_streamed_reference.py \
  tests/test_context_stage3_complete_reference_native.py \
  tests/test_context_stage3_reference_v1.py \
  tests/test_context_stage4_trait_native.py \
  tests/test_context_stage4_trait_v1.py \
  tests/test_context_stage4_fit_v1.py

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
print(tuple(gxeldcore.contextual_trait_semantic_operations_v1()))
PY
```
