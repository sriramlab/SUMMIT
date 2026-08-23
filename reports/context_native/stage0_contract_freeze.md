# Contextual native estimator — Stage 0 report

## Decision

`NARROW_GO` for Stage 1.

Stage 1 is permitted only as an isolated, in-memory contextual differential
class placed temporarily in `src/native/gxeldcore.cpp`, with the active Python
oracle and the committed Q1--Q4 corpus as its target. It may reuse private
generic projection and protected NN/TN machinery through a contextual semantic
dispatcher. It may not add the production streamed descriptor executor,
publish a production artifact, reuse legacy `X/W` scientific layouts, or
weaken the frozen scale/schema/integrity contracts.

This is not a production GO. Descriptor streaming, complete TN/Gram fault
injection and trusted fallback, allocation-free admitted execution, fatal
telemetry capacity, exact scale-plan wiring, and production V1 artifact
validation remain later-stage gates.

## Repository/build state

- Initial active worktree: clean
  `agent/gxe-reference-correctness-throughput` at `cfa385724`; its configured
  upstream is gone.
- Latest active implementation: `codex/contextual-covariance-core` at
  `251f197950775ca891f244dedc109476b2ad43b4`, a 70-commit descendant of
  `cfa385724` and the newest local contextual branch observed on 2026-08-20.
- Stage branch: `codex/contextual-stage0-contract-freeze`, created directly at
  `251f197950775ca891f244dedc109476b2ad43b4`. The existing branch/worktree was
  not modified. This report is part of the Stage 0 commit; the branch point is
  the reproducible source identity.
- Bundled comparison: the active contextual source, packaged contextual tests,
  `src/native/gxeldcore.cpp`, native common sources, `CMakeLists.txt`, and
  `pyproject.toml` were byte-identical to the package source snapshot at
  `251f197...`. The active repository contains additional unrelated tests; no
  material implementation mismatch was found. The active repository remained
  authoritative.
- Clean-build location: `/tmp/summit-stage0-build.NHd6yW`; fresh final install:
  `/tmp/summit-stage0-final-install.uS2Kvk`. No tracked or existing build output
  was overwritten.
- Compiler/build: GNU C++ 12.2.0, C++17, Release, `-O3 -march=native`.
- Native backend: `gxeldcore_direct` backend 1.9/API 9; shared OpenBLAS 0.3.34
  pthreads (`DYNAMIC_ARCH`, Zen, maximum 128 threads); OpenMP enabled; private
  BLIS/OpenBLAS disabled; GEMM integrity/checksum compiled on with the active
  one-billion-FLOP check threshold; telemetry capacity 16,384.
- Build provenance reported by `gxeldcore.build_info()`:
  source `251f197...`, tree
  `5f8ab2794b9fd74688e8bffdb95aeed56eb2634810c43b1aea42f8319f81621c`.
- Host: AMD EPYC 7501, 2 sockets, 64 physical/128 logical cores, AVX2, 8 NUMA
  nodes, 1.0 TiB RAM. `numactl` is not installed, so no `numactl --hardware`
  evidence was available; topology comes from `lscpu`.

Environment discoveries were not masked. Bare `python -m pytest` could not
import the source package; `PYTHONPATH=src` then exposed the missing native
extension. Default CMake configuration sought unavailable Intel/MKL, and an
OpenBLAS attempt under the base `CONDA_PREFIX` also failed. Configuration
succeeded only after selecting the `summit` environment explicitly and
disabling private BLAS builds. A package-relative temporary install was needed
for legacy tests that use `from summit import gxeldcore`.

## Scope completed

### Reconnaissance and contract freeze

- Located the active generalized implementation in
  `src/summit/context/{spec,oracle,reference,summary,fit,annotations}.py` and its
  context test modules.
- Audited descriptor/decode/projection/protected-GEMM/telemetry ownership in
  `src/native/gxeldcore.cpp`, `src/native/common/genotype.{hpp,cpp}`, and
  `src/native/common/mailman.hpp`.
- Froze the exact scientific formulas, active-symbol bindings, pair/component
  maps, scale and annotation modes, deletion semantics, raw/optional result
  separation, responsibility boundary, schema axes, lifecycle, semantic
  operation IDs, and checked admission formulas.
- Wrote a declaration-only C++17 API design. It is not referenced by CMake and
  implements no native execution.
- Imported the license-safe package fixtures without replacement, verified
  their package hashes, independently regenerated them from the active oracle,
  and committed a validation log.

### Observed defects

Before repair, `ContextBasisSpec` and `ContextComponentIndex` retained caller
lists; appending changed their effective contents/digests. A float64 input to
`build_disjoint_annotation_partition` was retained exactly
(`partition.weights is input` and writable), so caller mutation changed the
artifact while its recorded hash stayed stale. Core artifact arrays and nested
manifests were writable. `array_sha256` encoded native byte order and native
shape endianness. Core loaders accepted a changed declared
`component_index_hash` instead of recomputing it. Development manifest version
validation also accepted JSON `true` as integer version 1.

The audit additionally demonstrated development-format limits that were not
silently repaired into a new on-disk format: rehashing a modified NPZ container
can still admit semantic-array tampering; some development loaders do not
enforce exact key sets; scaling is still a free-form label; and generic overlap
outputs do not yet carry the production V1 interpretation metadata. These are
explicit production blockers, not hidden compatibility claims.

### Required minimal repair

- Canonicalized scalar array hashing across layout and byte order, rejected
  object/structured values, and made shape serialization little-endian.
- Added defensive C-contiguous copies with `writeable=False` plus recursively
  frozen JSON-compatible metadata.
- Applied ownership at construction/load boundaries for core specs,
  partitions, oracle results, references, trait summaries, normal equations,
  solves, fits, and PSD results.
- Recomputed component-map digests in reference, trait, and fit loaders.
- Rejected boolean schema versions.
- Added executable V1 schema/scale/annotation/deletion/admission types without
  changing legacy artifacts or promoting development NPZ formats.

No optional performance optimization and no production executor were added.

## Changed files

| Path | Symbols/areas | Reason |
|---|---|---|
| `src/summit/context/spec.py` | canonical array hash; frozen metadata/array helpers; basis/component ownership; bool version rejection | Close observed alias/hash defects with the smallest common helper. |
| `src/summit/context/schema.py` | V1 enums, identity/scale plan, annotation/deletion policy, responsibility boundary, checked uint64 call/event/memory ledger | Make future production identity and admission contracts executable without implementing execution. |
| `src/summit/context/{oracle,reference,summary,fit,annotations}.py` | dataclass `__post_init__` ownership; loader component-digest checks | Ensure core constructed and loaded artifacts cannot retain ordinary caller aliases or stale component claims. |
| `src/summit/context/__init__.py` | exports for the frozen Stage 0 schema | Make contract types available to tests/adapters. |
| `docs/context_native/scientific_contract_v1.md` | exact estimator formulas, active bindings, invariant-test map | Freeze the active scientific target. |
| `docs/context_native/schema_admission_contract_v1.md` | schema/lifecycle and checked memory/call/event formulas | Separate stable V1 axes from current development formats. |
| `docs/context_native/contextual_native_api_v1.hpp` | declaration-only plan/preflight/result interface | Freeze a concrete later-stage native boundary. |
| `docs/context_native/protected_operation_map.md` | protected-operation and reuse map | Record exact native seams, exclusions, and missing coverage. |
| `tests/test_context_stage0_schema.py` | 33 schema/ownership/cross-load/admission gates | Freeze maps, immutability, identities, interpretation, and overflow behavior. |
| `tests/test_context_stage0_fixtures.py` | 10 fixture/oracle gates | Independently replay Q1--Q4, all deletions/fits/surfaces, and five microcases. |
| `tests/fixtures/context_native_stage0/*` | 4 Q fixtures, 5 microcases, generator, manifest, README, validation log | Preserve the package oracle corpus and its provenance under the test tree. |
| `reports/context_native/stage0_contract_freeze.md` | this report | Record evidence, decision, and remaining gates. |

## Scientific verification

| Test/invariant | Command | Result | Maximum error |
|---|---|---|---:|
| Package integrity and independent reconstructions | `cd ~/SUMMIT_generalized_gxe_package && .../python TOOLS/validate_package.py` | PASS; 61 package files, 35 checks for each Q1--Q4, 5/5 microcases | `1.1641532182693481e-09` |
| Active-oracle regeneration versus imported files | active `src` with package `FIXTURES/generate_fixtures.py`, new `/tmp` output | All keys/shapes/nonfloating arrays match; reviewed reduction-order differences only | `4.656612873077393e-10` (Q3 grouped numerator) |
| Imported hashes | `sha256sum tests/fixtures/context_native_stage0/*.npz` | 9/9 match frozen manifest and test literals | exact |
| Full/grouped reference, U-statistic, group reconstruction, trait moments, transfer, all deletions, raw/LOO fits and surfaces, Q1--Q4 | `pytest -q tests/test_context_stage0_fixtures.py` | 10 passed | `atol=1e-9`, `rtol=1e-11`; active regeneration maximum above |
| `PDG` versus `PDPG`/`DPG`; factors 1/2/4; negative off-diagonal `Omega`; `N` versus rank; duplicate overlap rank; both grouped algorithms | same fixture command | all analytic microcases passed | overlap grouped algorithms `4.547473508864641e-13`; other regenerated microcases exact |
| Pair/component maps, deep ownership, independent schema axes, exact scale identity, strict/overlap interpretation, approximate deletion, raw/optional fields, cross-loading, checked admission | `pytest -q tests/test_context_stage0_schema.py` | 33 passed | exact structural checks |
| Complete generalized context suite | `PYTHONPATH=src:/tmp/summit-stage0-build.NHd6yW .../python -m pytest -q tests/test_context_*.py` | 260 passed after final bool-version gates | exact structural identities plus declared numerical tolerances |
| Header declaration syntax | `g++ -std=c++17 -x c++ -fsyntax-only docs/context_native/contextual_native_api_v1.hpp` | passed; expected `#pragma once in main file` warning | n/a |

The generalized baseline before Stage 0 additions was 217 passed in 42.95 s.
The final suite adds 43 Stage 0 tests. Pytest repeatedly emitted only two
pre-existing cleanup warnings for a concurrently populated
`/tmp/pytest-of-bronsonj/garbage-*` directory.

## Integrity and lifecycle verification

| Operation/failure | Injection/check | Detection/repair/fallback | Result |
|---|---|---|---|
| Caller base/view/list mutation | Construct core objects from mutable base, noncontiguous view, and nested lists; mutate originals | Defensive owned copies/tuples/frozen metadata | passed |
| Direct returned-array/metadata mutation | Write array cell or nested list/dict | read-only ndarray / `TypeError` | passed |
| Endian/layout checksum drift | C, Fortran, noncontiguous, little/big-endian equal arrays | one canonical digest; dtype/shape remain bound | passed |
| Component-map manifest tamper | Replace valid 64-hex digest before load | loader recomputation before scientific use | passed |
| Contextual/legacy cross-load | Give each reference loader the other family with missing artifacts | kind rejection before NPZ/table access | passed both directions |
| Admission overflow/incomplete call map | uint64 boundary, overflow products/sums, missing operation | checked failure before execution | passed |
| Existing protected NN one-shot fault | legacy `_test_protected_matmul_nn_integrity_diagnostic` suite | active detection/repair path | passed in legacy regression |
| Contextual TN/projection/Gram repeated fault | audit only; no Stage 0 native executor | no complete active targeted injection/retry/trusted-fallback path | production blocker |

## Legacy regression

| Command | Result | Change from baseline |
|---|---|---|
| `pytest -q tests/test_gxe_native_core.py tests/test_gxe_native_hardening.py tests/test_gxe_native_scratch_capacity.py tests/test_gxe_nn_integrity_diagnostic.py tests/test_gxe_multi_environment.py tests/test_gxe_reference_helpers.py tests/test_gxe_summary_mom.py tests/gxe_completion/test_feature_conventions.py` using the fresh temporary install | 103 passed, 1 skipped in 5.40 s | none; baseline was 103 passed, 1 skipped |

The single skip is
`tests/test_gxe_multi_environment.py:2367`: explicit placement requires
pthread-BLIS or an OpenMP BLAS. The selected shared OpenBLAS build does not
satisfy that test's placement requirement.

## Performance and admission

- Illustrative dimensions: `N=300000`, `M=1000000`, `Q=4`, `K=8`, `C=80`,
  `J=200`, fp64, group tile 1, action tile 80.
- At `B_T=B_D=128`, the package calculator reports core lower bounds of
  5.3248 GB source, 38.5024 GB action, 63.08864 GB group-restricted, and
  34.5984 GB same-person. Resident source scores are 4.096 GB and full raw
  actions 24.576 GB. Effective widths are 512 source, 4,096 generic
  group-restricted, 40,960 direct grouped TN, and about 512 useful
  strict-disjoint columns.
- At `B_T=B_D=1024`, corresponding lower bounds are 42.5984, 308.0192,
  504.63744, and 275.4432 GB; source scores are 32.768 GB and raw actions
  196.608 GB. These are not admitted peaks: decode, projection/packing,
  reductions, retry, separate trusted fallback, telemetry, allocator/NUMA
  duplication, output, and explicit headroom must be added.
- The exact frozen formula is in
  `docs/context_native/schema_admission_contract_v1.md`. It uses mutually
  exclusive phase maxima plus permanent, retry, trusted fallback, telemetry,
  compact output, and headroom ledgers; enumerates all 17 semantic operations;
  and makes overflow, capacity-minus-one, dropped records, or observed-plan
  disagreement terminal.
- Expected/observed descriptor passes: not applicable in Stage 0 because no
  contextual descriptor run exists. The future trait contract is one traversal
  per admitted phenotype batch; grouped plans must expose physical record
  visits rather than call indexed access a logical one-pass schedule.
- NUMA/thread policy: frozen as explicit plan input/evidence. No plan was
  selected or benchmarked in Stage 0. The active build has OpenMP available but
  reports no configured placement contract.

## Explicit Stage 0 questions

| Question | Resolved answer |
|---|---|
| Does active same-person code implement the variant-probe U-statistic? | Yes. `reference.same_person_ustatistic` and the Q1--Q4 replay use shared variant Rademacher probes, global probe sums, subtraction of same-probe products, and `B_D(B_D-1)` normalization. The fixture replay with probe tile 3 matches. |
| Do grouped contributions reconstruct the fixed-probe Gram in raw mass units? | Yes. Active grouped numerators sum to `M_k M_l T_R`; Q1--Q4 and the nontrivial overlap microcase enforce it. |
| Can a current loader cross-load contextual and legacy artifacts? | No. Both kinds are rejected before array/table I/O; executable tests cover both directions. |
| Is current reference/trait scale identity exact enough? | No. Development builders compare a free-form string, so `G` and `2G` can be mislabeled identically. `GenotypeScalePlanV1` freezes the required exact identity for Stage 1+, but development files were not silently migrated. |
| Which TN paths lack targeted fault injection, repair, or fallback? | Every proposed contextual TN semantic path: sample-probe projection, source, action projection/Gram, group cross-Gram/direct grouped TN, same-person projection/Gram, and trait score/feature projection. There is no dedicated protected Gram semantic wrapper; only NN has an explicit one-shot injection diagnostic. |
| Which reusable helpers are private, and what is the smallest seam? | Descriptor/FileState, `DirectContext` projection, protected GEMM routing, native output ownership, NUMA and telemetry helpers are in the `gxeldcore.cpp` anonymous namespace. Only common genotype/Mailman APIs are sibling-TU reusable. The smallest safe seam is a contextual-only class temporarily in the same translation unit. |
| Is coordinate-specific post-projection scaling already present? | Yes, in legacy `DirectContext::compute_feature_moments_from_genotype`/`scale_target_outputs` (`scale_x`, `scale_w`). It is explicitly outside the contextual reuse boundary. |
| Is the all-resident schedule compatible with active ownership? | Structurally yes under a new same-TU contextual owner and checked admission; no existing legacy class can be reused unchanged. Current layouts/pass schedules, per-call scratch allocation, group traversal, and telemetry behavior do not meet the contract. |

## Remaining risks/blockers

1. `src/summit/context/{reference,summary,fit}.py` still writes development NPZ
   formats. Container rehashing can conceal scientific-array changes, and some
   loaders accept extra keys. Stage 1 differential work must remain in memory;
   a later production schema must bind exact fields, keys, shapes, dtypes, and
   per-array canonical digests.
2. Development `genotype_scaling` is not a sealed plan. Stage 1 must receive
   one shared `GenotypeScalePlanV1` (or the exact native equivalent) and test
   allele orientation. Active legacy decode paths use opposite signs in
   different classes.
3. Generic overlap development outputs can be read as standalone annotation
   surfaces. Production adapters must apply `annotation_output_contract`, label
   them conditional contributions, and publish a combined total by default.
4. Native protected integrity is conditional by shape/build, checked calls
   allocate scratch internally, retry/fallback is incomplete, output canaries
   are absent, and telemetry drops oldest records. The Stage 1 dispatcher must
   begin closing these gaps and may not claim production readiness.
5. The private same-TU seam deliberately creates short-term code locality.
   Extracting common protected/descriptor infrastructure before differential
   correctness would broaden legacy risk and is outside the approved Stage 1
   increment.
6. The first-order Tabla estimates are not admissions or benchmarks. No
   descriptor passes, phase timings, high-water RSS, page-fault/record-visit
   evidence, or one/two-socket comparison exists for a contextual executor.

## Reproduction commands

```bash
cd /home/bronsonj/SUMMIT
git status --short --branch
git rev-parse HEAD
git log -1 --oneline

env CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit \
  cmake -S . -B /tmp/summit-stage0-build.NHd6yW \
  -DPython_EXECUTABLE=/home/bronsonj/anaconda3/envs/summit/bin/python \
  -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OpenBLAS \
  -DGXELDCORE_USE_PRIVATE_BLIS=OFF \
  -DGXELDCORE_USE_PRIVATE_OPENBLAS=OFF
cmake --build /tmp/summit-stage0-build.NHd6yW --parallel 8

PYTHONPATH=src:/tmp/summit-stage0-build.NHd6yW \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_*.py

stage0_install_dir=$(mktemp -d /tmp/summit-stage0-install.XXXXXX)
cmake --install /tmp/summit-stage0-build.NHd6yW \
  --prefix "$stage0_install_dir"
PYTHONPATH="$stage0_install_dir" \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q -rs \
  tests/test_gxe_native_core.py \
  tests/test_gxe_native_hardening.py \
  tests/test_gxe_native_scratch_capacity.py \
  tests/test_gxe_nn_integrity_diagnostic.py \
  tests/test_gxe_multi_environment.py \
  tests/test_gxe_reference_helpers.py \
  tests/test_gxe_summary_mom.py \
  tests/gxe_completion/test_feature_conventions.py

cd /home/bronsonj/SUMMIT_generalized_gxe_package
/home/bronsonj/anaconda3/envs/summit/bin/python TOOLS/validate_package.py
/home/bronsonj/anaconda3/envs/summit/bin/python FIXTURES/generate_fixtures.py \
  --source-root /home/bronsonj/SUMMIT/src \
  --output-dir "$(mktemp -d /tmp/summit-stage0-fixtures.XXXXXX)"
```

The exact imported-versus-regenerated comparison and all file hashes are in
`tests/fixtures/context_native_stage0/VALIDATION.md`; the protected symbol map
and line anchors are in `docs/context_native/protected_operation_map.md`.
