# Contextual native estimator — Stage 1 report

## Decision

`NARROW_GO` for Stage 2, restricted to using
`ContextualBlockExecutorV1` as an isolated in-memory scientific differential
oracle.

Fixed and randomized `Q=1..4` compact sufficient statistics, both grouped
algorithms and both direct-TN row-scaling placements, the global signed
same-person U-statistic, every deletion, transferred systems, raw
fits/Omegas/surfaces, operational tile schedules, and semantic NN/TN fault
injection match the active Python oracle. Legacy native defaults and scientific
helpers remain unchanged.

This is not a production integrity, admission, memory-scale, strict-routing,
descriptor, or artifact-schema GO. Stage 2 may add descriptor-owned decoding
and compare it against this oracle. It must not reuse the Stage 1 nominal dense
workspace estimate as a production memory bound or describe witness-copy
recovery as a qualified trusted fallback.

## Repository/build state

- Stage branch: `codex/contextual-stage1-dense-differential`, created at the
  Stage 0 commit `7681d120f79979cf1ef693841387ab5fe6487a65` without
  modifying the prior branch or worktree.
- Native implementation commit:
  `5b5d8eb501337fa38665d74b85425da2ebd1ebf3`. The tree was clean when
  CMake configured it. The report is added in a subsequent documentation-only
  commit; the final branch HEAD is reported in the handoff.
- Active native provenance: source commit `5b5d8eb...`, source-tree SHA-256
  `b7e710f885f48533e19dfbb258168005748ca3b1b1dd08287766af76cf35346e`.
- Clean-build directory: `/tmp/summit-stage1-clean-build.SOBpLF`; clean install:
  `/tmp/summit-stage1-clean-install.UViRQJ`. Existing build/install outputs
  were preserved.
- Compiler/build: GNU C++ 12.2.0, C++17, Release,
  `-O3 -march=native -fopenmp`.
- Native backend: existing `gxeldcore` module version 1.7, backend version 1.9,
  API 9, shared OpenBLAS 0.3.34 (`DYNAMIC_ARCH`, Zen, pthread, maximum 128
  threads), OpenMP enabled. Existing GEMM integrity/checksum remains enabled
  with the one-billion-FLOP legacy threshold and 16,384-record legacy
  telemetry capacity. No legacy version or default changed.
- Host: AMD EPYC 7501, 2 sockets, 64 physical/128 logical cores, AVX2, 8 NUMA
  nodes, 1.0 TiB RAM. Stage 1 differential tests used the contextual default of
  one execution thread and did not select or attest NUMA placement.

An intermediate extension configured before the implementation commit had no
canonical source provenance, so 14 legacy direct tests rejected it before
scientific execution. This was a build-state failure, not a behavioral
regression. Reconfiguring from the clean implementation commit produced the
final 103-passed/1-skipped legacy result below.

## Scope completed

### Native differential executor

- Added the contextual-only
  `summit::context_v1::ContextualBlockExecutorV1` in the same translation unit
  as the existing private protected-GEMM helpers. The legacy module change is
  an additive include and binding only.
- Accepted owned copies of pre-scaled dense `G`, orthonormal `U`, `Phi`, generic
  weights or strict-disjoint annotations, canonical groups/names, fixed sample
  probes, fixed variant probes, one phenotype, and a residual basis.
- Centralized diagonal-then-off-diagonal pair order, directional factors, and
  annotation-major component maps. Returned the complete compact maps and
  policy identities.
- Implemented block-local `P diag(phi_q) G` features, fixed-probe source and
  target actions, normalized and raw Gram units, group-restricted actions,
  direct grouped TN with action and genotype row scaling, the signed global
  variant-probe U-statistic, one logical trait traversal, and low-rank residual
  moments.
- Returned only compact scientific arrays. Population transfer, deletion
  renormalization, small rank-checked solves, raw Omega unpacking, covariance
  surfaces, interpretation, and immutable artifact construction remain in the
  active Python implementation.
- Made variant, sample-probe, variant-probe, action, annotation, context,
  group, and trait-feature tiles operational. Strict mode seals one annotation
  ID per variant and bypasses generic weight/square-root lookup while retaining
  the dense differential target schedule.

### Integrity and lifecycle

- Routed every scientific wide NN/TN through one contextual dispatcher using
  the 17 frozen semantic operation IDs. The primary protected result is checked
  against a deterministic tiled witness before accumulation.
- Added operation-plus-occurrence fault targeting, retry, witness-copy
  fallback, canaries, operand fingerprints, finiteness checks, fixed-capacity
  class-local telemetry, and fatal unknown/unconsumed targets.
- Added checked pre-copy dimensions/products and a dimension-only uint64
  overflow hook.
- Enforced one-shot lifecycle, owned-input fingerprints, exact semantic call
  counts, probe coverage, one logical trait pass, projection leakage,
  pre-symmetry, group counts/masses, grouped reference reconstruction, grouped
  trait reconstruction, and finite compact outputs before publication.
- Released source/action/retry/witness/fallback execution scratch before
  copying public results.

### Required repairs made during review

The review found and repaired four pre-commit defects: normal calls initially
bypassed witness comparison; annotation/context/trait-feature controls were
only recorded rather than physically tiled; phenotype projection consumed the
first trait-feature semantic fault occurrence; and telemetry admission counted
logical calls rather than the maximum three recovery events per call. The
committed implementation and final clean tests include all four repairs.

No descriptor reader, public CLI, on-disk artifact, native fit, hidden
regularization, PSD projection, NUMA policy, or legacy scientific refactor was
added.

## Changed files

| Path | Symbols/areas | Reason |
|---|---|---|
| `src/native/gxeldcore.cpp` | one include and `bind_contextual_dense_v1` call | Add the contextual binding without changing legacy binding order, versions, or defaults. |
| `src/native/contextual_dense_v1.inc` | executor, preflight, dispatcher, semantic ledger, compact publication | Implement the isolated Stage 1 dense differential boundary in the smallest same-TU seam. |
| `tests/test_context_stage1_dense_native.py` | 58 focused native tests | Verify frozen/random science through final fit, tiling, lifecycle, admission, and semantic injection. |
| `docs/context_native/stage1_dense_differential.md` | operation/layout/admission boundary | Document exact layouts, units, lifecycle, and explicit limitations without a production claim. |
| `reports/context_native/stage1_dense_differential.md` | this report | Record reproducible evidence and the Stage 2 gate. |

## Scientific verification

The focused numerical comparisons use `atol=1e-9, rtol=1e-11`. The maximum
stored-statistic discrepancy is scale-relative and passes that unchanged
tolerance.

| Test/invariant | Command | Result | Maximum error |
|---|---|---|---:|
| Frozen `Q=1..4` full/grouped reference, same-person, all trait/residual moments, maps, masses/counts, and raw units | `pytest -q tests/test_context_stage1_dense_native.py` on the clean install | passed in all four fixtures | `1.1641532182693481e-09` stored compact field; `2.3283064365386963e-09` derived raw Gram |
| Transfer and every single-group deleted reference/trait moment | same focused command | passed | transfer `1.4779288903810084e-12`; deleted Gram `1.8189894035458565e-12`; deleted RHS `4.2632564145606011e-14` |
| Raw coefficients, raw Omegas, every delete-group fit, and full/LOO covariance surfaces | same focused command | passed | coefficient/Omega `1.1551648526619829e-10`; deleted fit `2.5475621612258692e-11`; surface `1.3918111108068842e-10` |
| Seeded random strict/overlap `Q=1..4` versus active builders and fits | same focused command | 4/4 passed | within unchanged declared tolerance |
| Restricted grouped algorithm and both direct TN scaling placements, each through deletion fits | same focused command | passed independently | stored grouped maximum included above |
| Each tile knob at one versus full, plus combined all-one schedule | same focused command | all compact fields passed; expected/observed ledgers exact | Q4 combined all-one/full `5.8207660913467407e-10` |
| Strict annotation-ID route versus generic binary weights | same focused command | paths separately evidenced; outputs identical | `0` |
| `B_D=2` with variant-probe tile 1 and global cross-tile state | same focused command | passed against active U-statistic builder | within unchanged declared tolerance |
| `Q=3` nonorthogonal reparameterization | same focused command | transformed Omega and surface covariance passed | Omega `2.4868995751603507e-14`; surface `7.8603790143461083e-14` |
| No returned scientific ndarray has an `N` or `M` axis; no legacy/raw/vendor scientific call appears in the contextual source | same focused command plus source-boundary assertions | passed | exact structural check |
| Focused Stage 1 suite | `PYTHONPATH=/tmp/summit-stage1-clean-install.UViRQJ .../python -m pytest -q tests/test_context_stage1_dense_native.py` | 58 passed in 2.20 s | values above |
| Complete contextual suite | `PYTHONPATH=/tmp/summit-stage1-clean-install.UViRQJ .../python -m pytest -q tests/test_context_*.py` | 318 passed in 44.75 s | declared test tolerances |

The complete suite emitted only two pre-existing pytest cleanup warnings for a
concurrently populated `/tmp/pytest-of-bronsonj/garbage-*` directory.

## Integrity and lifecycle verification

| Operation/failure | Injection/check | Detection/repair/fallback | Result |
|---|---|---|---|
| All 17 contextual semantic NN/TN operations | one-shot corruption at occurrence 1, one run per operation | witness mismatch, one retry, verified publication | 17/17 passed; maximum Q1 discrepancy including derived raw units `8.731149137020111e-11` |
| Repeated corruption | corrupt primary and retry for `source_tn` | repair recorded, retry rejected, witness-copy fallback | passed; injection/repair/retry/fallback = `1/1/1/1` |
| Forced fallback | force `source_tn` directly to fallback | verified witness copied through admitted fallback buffer | passed; `1/1/0/1` |
| `NaN` and `Inf` output | corrupt primary output | finiteness/witness detection, successful retry | both passed; no fallback needed |
| Canary damage | damage guarded primary output | canary detection, retry, witness-copy fallback | passed; `1/1/1/1` |
| Repair corruption | corrupt retry output | terminal integrity exception before accumulation/publication | failed closed as required |
| Fallback corruption | corrupt fallback buffer | terminal integrity exception before publication | failed closed as required |
| Sealed operand mutation | mutate protected operand during injected call | fingerprint mismatch | failed closed as required |
| Unknown/unconsumed fault | invalid operation or unreachable occurrence | constructor validation / final ledger check | failed closed as required |
| Exact caller-object mutation | mutate every exact ndarray/list passed to C++ after construction | class-owned copies and sealed names/maps | all frozen outputs unchanged; passed |
| Second `run()` | call after publication | one-shot lifecycle rejection | passed |
| Workspace/telemetry cap minus one | construct at reported exact capacity and one below | pre-execution admission check | exact passed; minus one rejected |
| Worst-case telemetry arithmetic | fault-enabled reserve | checked `3 * total_calls + 10`; fixed vector cannot drop/overwrite | exact repeated-fault run passed; minus one rejected |
| uint64 layout overflow | dimension-only synthetic product | checked arithmetic before allocation | rejected as required |
| Terminal numeric invariants | leakage, pre-symmetry, masses/counts, probe coverage, trait visits, grouped reference/trait reconstruction, exact call map | validation before scratch release/publication | all fixed/random/tiled runs passed |

Contextual telemetry is class-local and fatal on its admitted event bound, and
every wide operation is verified against a deterministic witness. Injection is
currently addressed by semantic operation plus occurrence, and the fallback
buffer receives the already verified witness. Stable tile coordinates, full
attempt/layout records, and an independently invoked qualified fallback remain
future gates.

## Legacy regression

| Command | Result | Change from baseline |
|---|---|---|
| `pytest -q -rs tests/test_gxe_native_core.py tests/test_gxe_native_hardening.py tests/test_gxe_native_scratch_capacity.py tests/test_gxe_nn_integrity_diagnostic.py tests/test_gxe_multi_environment.py tests/test_gxe_reference_helpers.py tests/test_gxe_summary_mom.py tests/gxe_completion/test_feature_conventions.py` on the clean install | 103 passed, 1 skipped in 6.34 s | none; frozen baseline is 103 passed, 1 skipped |

The unchanged skip is `tests/test_gxe_multi_environment.py:2367`: explicit
placement requires pthread-BLIS or an OpenMP BLAS configuration accepted by
that test. The selected shared OpenBLAS build does not satisfy its placement
qualification. The run emitted the same two pytest temporary-directory cleanup
warnings noted above.

The final `gxeldcore.cpp` diff adds only three lines: the contextual include,
one blank separator, and the additive binding call. Module version 1.7, backend
version 1.9, API 9, binding order, legacy helpers, and all defaults are
unchanged.

## Performance and admission

- Illustrative fixed case: Q4 fixture, `N=64`, `M=96`, `Q=4`, `K=3`, `C=30`,
  `J=4`, `B_T=B_D=7`, residual width `H=2`, fp64, one execution thread.
- Full-tile plan: variant 96, sample probes 7, variant probes 7, actions 30,
  annotations 3, contexts 4, groups 4, trait features 384. The selected public
  grouped result is `group_restricted_action_v1`; both direct placements also
  execute and are returned under explicit diagnostic keys.
- Full-tile call ledger: 31 expected and observed protected calls. Counts are
  sample projection `1+1`, source `1`, full target `1`, action projection
  `1+1`, action Gram `1`, group target/cross-Gram `4+4`, direct grouped TN `8`,
  same-person target/projection/Gram `1+1+1+2`, and trait
  score/feature-projection `1+1+1`.
- Full-tile nominal reserve: 881,024 bytes; telemetry capacity 102; observed
  events 40. Five small-fixture runs had a median construction-plus-run time of
  0.022832 s (minimum 0.022760 s).
- All-one plan: 6,144 expected/observed protected calls, four outer group
  batches, nominal reserve 2,948,864 bytes, telemetry capacity 18,441, observed
  events 6,153. Five runs had a median of 0.578639 s (minimum 0.577220 s).
  These are differential-test timings, not production benchmarks.
- Meaningful pass evidence: this dense backend has
  `descriptor_decode_passes=0`; the trait phase reports one logical pass and
  exactly `M=96` variant visits. The compatibility field `decode_passes=1` is
  not a descriptor or whole-executor pass count and is not used as evidence.
- Full-tile maximum projection leakage was
  `1.971756091734278e-13`; grouped reconstruction maximum absolute discrepancy
  was `1.3969838619232178e-09` in raw numerator units.
- Admitted/observed peak bytes: no observed allocator/RSS high-water is
  published. `required_workspace_bytes` is a checked differential threshold,
  not an exact native peak. Phase times and protected call shapes/strides are
  also not emitted.
- NUMA/thread policy: one contextual thread, no binding/NUMA attestation. The
  active extension has OpenMP and shared pthread OpenBLAS, but Stage 1 makes no
  placement or multi-socket performance claim.

## Remaining risks/blockers

1. **Nominal rather than exact memory admission.**
   `contextual_dense_preflight_numbers_v1` does not represent one exact
   live-range ledger for class-owned inputs, all co-live local panels,
   helper-internal allocations, publication copies, allocator overhead,
   headroom, or measured high-water. Many scientific vectors allocate after
   `run()` begins. Consequence: exact/cap-minus-one proves only threshold
   enforcement. Stage 2 must use preallocated phase arenas, bind all categories
   into a checked maximum, measure/reconcile high-water, and repeat exact/minus
   one under every recovery path.
2. **Differential rather than production integrity evidence.**
   `protected_gemm` targets operation plus occurrence; telemetry records only
   event class, phase, operation, and resolution. It omits stable semantic tile
   coordinates, dimensions, strides, layouts, attempts, and digests. The
   fallback copies the deterministic witness, whose computation uses the
   selected thread count, instead of invoking an independent qualified
   one-thread backend. Stage 2 must add stable call keys, complete admitted
   records, and independently execute/qualify fallback.
3. **Dense panels remain resident.** Annotation/context/trait tiles change the
   protected call schedule, but several full target/action/feature assemblies
   remain live; selecting a full-width variant tile can reach a flattened
   `N x (Q*M)` feature extent. `group_tile` batches only the outer per-group
   schedule. Consequence: Stage 1 is not Tabla-scale or bounded-memory evidence.
   Descriptor stages must stream into genuinely bounded panels and verify
   physical record visits and peaks.
4. **Strict routing is only a dense ID fast path.** Strict annotations are
   validated and sealed as IDs, but no sparse/gather/disjoint genotype kernel
   was qualified. Stage 2 must compare a genuinely routed strict kernel against
   both this ID path and generic binary weights.
5. **Publication is not a stable native artifact.** Native NumPy results have
   capsule ownership but remain writeable. Basic names, maps, units, and
   policies are returned, but canonical map/scale-plan/build/telemetry/output
   digests and exact field admission are absent. The Python artifact boundary
   currently copies and freezes arrays. A later adapter/schema must enforce
   read-only ownership, exact keys/shapes/dtypes, independent version axes, and
   canonical per-array identities.
6. **No reference-only or multi-trait executor mode.** The combined constructor
   requires one phenotype and `H>=1` residual basis. Stage 2 must add explicit
   optional trait plans without weakening reference-only call/admission
   ledgers.
7. **Scale identity is not fully wired.** `pre_scaled_dense_v1` is a closed
   differential label, not a digest of means, inverse scales, allele
   orientation, variant identity, and provenance. Descriptor execution must
   bind the exact `GenotypeScalePlanV1` identity before comparison/publication.
8. **Compatibility pass label is ambiguous.** `decode_passes=1` means one
   resident dense-input compatibility load, not one overall visit to `G`.
   Authoritative Stage 1 fields are `descriptor_decode_passes=0`,
   `trait_logical_passes=1`, and `trait_variant_visits=M`. Remove or rename the
   alias before production telemetry is frozen.

## Reproduction commands

```bash
cd /home/bronsonj/SUMMIT
git status --short --branch
git rev-parse HEAD
git log -2 --oneline

stage1_build_dir=$(mktemp -d /tmp/summit-stage1-clean-build.XXXXXX)
stage1_install_dir=$(mktemp -d /tmp/summit-stage1-clean-install.XXXXXX)

env CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit \
  cmake -S . -B "$stage1_build_dir" \
  -DPython_EXECUTABLE=/home/bronsonj/anaconda3/envs/summit/bin/python \
  -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OpenBLAS \
  -DGXELDCORE_USE_PRIVATE_BLIS=OFF \
  -DGXELDCORE_USE_PRIVATE_OPENBLAS=OFF
cmake --build "$stage1_build_dir" --parallel 8
cmake --install "$stage1_build_dir" --prefix "$stage1_install_dir"

PYTHONPATH="$stage1_install_dir" \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage1_dense_native.py

PYTHONPATH="$stage1_install_dir" \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_*.py

PYTHONPATH="$stage1_install_dir" \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q -rs \
  tests/test_gxe_native_core.py \
  tests/test_gxe_native_hardening.py \
  tests/test_gxe_native_scratch_capacity.py \
  tests/test_gxe_nn_integrity_diagnostic.py \
  tests/test_gxe_multi_environment.py \
  tests/test_gxe_reference_helpers.py \
  tests/test_gxe_summary_mom.py \
  tests/gxe_completion/test_feature_conventions.py

PYTHONPATH="$stage1_install_dir" \
  /home/bronsonj/anaconda3/envs/summit/bin/python - <<'PY'
from summit import gxeldcore
print(dict(gxeldcore.build_info()))
print(tuple(gxeldcore.contextual_dense_semantic_operations_v1()))
PY
```

The exact layouts, units, operation list, and qualification boundary are in
`docs/context_native/stage1_dense_differential.md`. The frozen formula and
fixture provenance remain in the Stage 0 contract/report.
