# Contextual native estimator — Stage 5 integrity hardening report

## Qualification status and decision

**Final Stage 5 decision: `NARROW_GO` for Stage 6.**

The Stage 5 implementation is settled at commit
`37317a85d5cbada9b06921c39642e164e24613d9`. Clean Release build, installed
binary, focused, contextual, legacy, and full-repository results are recorded
below. Both configured sanitizer builds passed the required scoped
qualification with fatal address/undefined-behavior diagnostics enabled; the
explicit LeakSanitizer and sanitizer-full-repository exclusions are recorded
below and are not presented as passes.

The supported decision is **`NARROW_GO` for Stage 6**. The narrowness is
deliberate: the
Stage 5 implementation closes the protected-operation, event, artifact-schema,
loader, and transient process-merge boundaries, while leaving the production
protected cross-process finalizer, total-process RSS enforcement, absolute
file snapshot/lease protection, verified NUMA placement, and the one-million-
variant performance demonstration to their packaged later stages. None of those
items changes the approved reference or trait science.

`GO` would require closing those residual production claims in Stage 5, which
this implementation does not claim. The clean Release suite, both configured
sanitizer scopes, true two-process summary-only test, exact 17-operation fault
matrix, and artifact compatibility/cross-loader gate passed; no `NO_GO`
condition in the decision rule below was observed.

## Repository and build state

- Branch: `codex/contextual-stage5-integrity-schema-hardening`, created from
  the clean Stage 4 report commit
  `c3a7be4c5a9d52c5b6290e94e3c2ff7cb41b0c93`.
- Final Stage 5 implementation/test commit:
  `37317a85d5cbada9b06921c39642e164e24613d9`.
- This report is committed separately after the qualified implementation;
  its commit is recorded in repository history and the Stage 6 handoff.
- Clean source commit embedded in the native binary:
  `37317a85d5cbada9b06921c39642e164e24613d9`.
- Clean source-tree SHA-256 embedded in the native binary:
  `9a6ddc2ef0d353d7bf1d0875dbbf1137131901e8e8eb2c884e4a6e691c512cae`.
- Native implementation source SHA-256
  (`src/native/contextual_streamed_reference_v1.inc`):
  `1146b0b7d8267231f751eeda53c7c8d1782a6303c0d9c6a7a220d96a9222025f`.
- Clean qualification source:
  `/tmp/summit-stage5-qual-src.Dd9JNl`.
- Clean Release build/install paths:
  `/tmp/summit-stage5-release-build.3FBu5E` and
  `/tmp/summit-stage5-release-install.PsC2Xv`.
- Installed Release `gxeldcore` SHA-256:
  `b16f2d1c9c5bab0e99f6a801dcfe52d5438a7d173582388b95e24e44f6676ad9`.
- Installed Release RUNPATH:
  `$ORIGIN:/home/bronsonj/anaconda3/envs/summit/lib`.
- Clean ASan+UBSan build/install paths and `gxeldcore` SHA-256:
  `/tmp/summit-stage5-final-asan-build.geKkop`,
  `/tmp/summit-stage5-final-asan-install.h5fuJ5`, and
  `e95a4c605a79bd3ee6000115e01c97915135614d455b021dbcaecba444be9640`.
- Clean UBSan-only build/install paths and `gxeldcore` SHA-256:
  `/tmp/summit-stage5-final-ubsan-build.6UNLPa`,
  `/tmp/summit-stage5-final-ubsan-install.V2D0xi`, and
  `3978799b4316cb213f21ef5c662187611b6b4952dfdbd2baaff608f778720f65`.
- Qualification inventory: Tabla, Linux `4.19.0-21-amd64`, `x86_64`; AMD EPYC
  7501, 128 logical/64 physical CPUs, 2 sockets, 8 NUMA nodes (`0`-`7`),
  automatic NUMA balancing `1`; CMake 3.25.1, GNU C++ 12.2.0, Python 3.12.12,
  NumPy 2.3.5, and SciPy 1.16.3. Release links the environment
  `libopenblas.so.0`, `libgomp.so.1`, and `libstdc++.so.6`; `build_info()`
  reports OpenBLAS 0.3.34, `DYNAMIC_ARCH`, `NO_AFFINITY`, `Zen`,
  `MAX_THREADS=128`, and pthreads.

The logical native API remains version 1. The native reference and trait
backend implementations move to physical backend version 2, while the stable
artifact families remain the finalized logical reference/trait/fit V1
families. The boundary is described precisely below; it is not a silent
migration.

## Scope completed in the implementation

Stage 5 adds or hardens the following release boundaries:

1. one protected dispatcher for all 17 contextual wide operations, plus a
   repository-wide static allowlist for vendor GEMM entry sites;
2. logical semantic targeting with phase, role, placement, canonical
   coordinate, and all existing axis ranges, including separate identities for
   same-person tile/global merge and direct-grouped action/genotype placement;
3. two-sided protected-output canaries, nonfinite checks, independent scalar
   witness and fallback, operand/runtime fingerprints, and process-wide
   serialized dispatcher entry;
4. exact, pre-admitted retry/fallback buffers and event capacity, including a
   reserved terminal slot and exact/capacity-minus-one tests;
5. metadata-only terminal failure reports and publication-time validation of
   the complete event sequence;
6. complete native scientific-array digest maps, post-phase SHA-256 evidence,
   an execution-plan digest, build provenance, and stronger descriptor/content
   identities;
7. independent Python-owned semantic authority for every stable reference and
   trait input that cannot be recovered safely from compact outputs;
8. strict reference/trait backend-v2 adapters, exact cross-artifact fit
   compatibility, canonical little-endian schemas, and bounded NPZ preflight;
9. a real builder-process/native-executor teardown followed by a separate
   stable-artifact-only fitting process; and
10. an immutable transient process/socket variant-probe partial and NumPy
    oracle merge for the signed global same-person U-statistic.

This stage does not alter the approved scientific estimand, component order,
group-deletion semantics, raw rank-checked fit, or raw-versus-PSD separation.

## Static protected-call coverage

`tests/test_context_stage5_build_integrity.py` implements two complementary
source checks.

- It scans every native C/C++/CUDA header, include, and translation-unit suffix
  after masking comments and literals. The matcher covers CBLAS and MKL
  GEMM/SYRK entries including batch and strided-batch variants, BLIS
  GEMM/SYRK variants, and Fortran `s`/`d` GEMM/SYRK spellings. Each entry must
  match the exact 15-call pre-existing wrapper allowlist in `gwldcore.cpp`,
  `gwldcore_cuda.cu`, `winldcore.cpp`, or the explicit legacy/private boundary
  in `gxeldcore.cpp`. A new entry, an extra call in an approved function, or a
  call outside an approved function fails the test.
- It walks the local-include transitive closure of both contextual roots across
  the same native suffix set. Vendor GEMM calls are forbidden. Calls to
  contextual `dgemm_*` or observed-vendor helpers must be lexically contained
  in `protected_gemm`; calls outside that function fail the test.

The production descriptor-owned contextual implementation declares exactly
the following 17 operations. Every call site reaches `protected_tn` or
`protected_nn`, and those wrappers reach `protected_gemm`. The contextual
primary implementation is the deterministic tiled FP64 backend and reports
`contextual_dispatch_vendor_calls=false`; there is no conditional production
vendor bypass.

Static check command and settled result:

```text
PYTHONPATH=/tmp/summit-stage5-release-install.PsC2Xv \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage5_build_integrity.py
3 passed in 1.21 s; exact vendor-entry allowlist count: 15
```

## Exact 17-operation integrity matrix

The mode vocabulary used in every matrix row is exact:

- `R7` = `one_shot`, `nan`, `inf`, `canary`, `repeated`,
  `repair_corruption`, and `force_fallback`;
- `T4` = `fallback_corruption`, `fallback_failure`, `operand_mutation`, and
  `runtime_mutation`.

For `one_shot`, `nan`, `inf`, and `canary`, the primary result is rejected and
a deterministic retry must agree with the independent scalar witness. For
`repeated` and `repair_corruption`, the primary and retry are rejected and the
independently loop-ordered one-thread scalar fallback must agree. For
`force_fallback`, the admitted fallback is exercised directly.
`fallback_corruption` and `fallback_failure` terminate without publication;
operand and runtime mutations terminate at their authenticated checks.

All attempt/verification events carry a contiguous sequence number, process
and thread ID, elapsed nanoseconds, full GEMM dimensions/strides/transpose,
operand/witness/accepted-output fingerprints, before/after runtime
fingerprints, prefix/suffix-canary results, finiteness, serialized-entry and
non-vendor-backend evidence, resolution/attempt, and the full logical semantic
coordinate. `E` in the table refers to that exact event contract.

| # | Protected operation | Form and phase | Logical target identity | Injection coverage | Detection and successful recovery | Telemetry and terminal behavior | Qualification owner/status |
|---:|---|---|---|---|---|---|---|
| 1 | `sample_probe_projection_tn` | TN, source | resident/probe ranges; ordinary/none; canonical coordinate | R7 + T4 | witness, finiteness, two-sided canaries, operand/runtime seal; retry or trusted fallback per mode | E; terminal event and metadata-only failure report for T4 | Stage 2 matrix + Stage 5 event regression; **PASS — clean Release** |
| 2 | `sample_probe_projection_nn` | NN, source | resident/probe ranges; ordinary/none; canonical coordinate | R7 + T4 | same protected policy | E; no compact publication after terminal fault | Stage 2 matrix + Stage 5 event regression; **PASS — clean Release** |
| 3 | `source_tn` | TN, source | resident/probe, retained-variant, and context ranges; ordinary/none; canonical coordinate | R7 + T4 | same protected policy | E; terminal result is metadata only | Stage 2 matrix + Stage 5 event regression; **PASS — clean Release** |
| 4 | `full_target_nn` | NN, source | retained-variant membership, annotation/context and probe ranges; ordinary/none; canonical coordinate prevents a strict-target enclosing range from claiming an absent member | R7 + T4 | same protected policy | E; unmatched target and terminal fault fail closed | Stage 2 matrix, tiled retargeting, strict membership; **PASS — clean Release** |
| 5 | `action_projection_tn` | TN, action | resident/probe and action/component ranges; ordinary/none; canonical coordinate | R7 + T4 | same protected policy | E; no output after terminal fault | Stage 2 matrix + Stage 5 event regression; **PASS — clean Release** |
| 6 | `action_projection_nn` | NN, action | resident/probe and action/component ranges; ordinary/none; canonical coordinate | R7 + T4 | same protected policy | E; no output after terminal fault | Stage 2 matrix + Stage 5 event regression; **PASS — clean Release** |
| 7 | `action_gram_tn` | TN, Gram | resident/probe and action/component ranges; ordinary/none; canonical coordinate | R7 + T4 | same protected policy; post-Gram output seal adds phase-boundary detection | E; terminal event precedes any publication | Stage 2 matrix + post-Gram mutation test; **PASS — clean Release** |
| 8 | `group_target_nn` | NN, group | group, group-execution variant, annotation/context and probe ranges; ordinary/none; canonical logical variant coordinate | R7 + T4 | same protected policy; group reconstruction checked before publication | E; terminal failure cannot yield a stable group result | Stage 3 matrix + Stage 5 event regression; **PASS — clean Release** |
| 9 | `group_cross_gram_tn` | TN, group | group, action/component, probe and canonical logical-variant ranges; ordinary/none | R7 + T4 | same protected policy; grouped numerator reconstruction remains required | E; no partial grouped artifact | Stage 3 matrix + Stage 5 event regression; **PASS — clean Release** |
| 10 | `direct_grouped_tn` | TN, group | group and canonical variant coordinates plus placement=`action_scaled` or `genotype_scaled` | R7 + T4 for the operation; placement-specific one-shot selectors | same protected policy; both placements are independently targetable and their differential agreement remains checked | E includes placement; a fault cannot be attributed to the other placement | Stage 3 matrix + Stage 5 two-placement targeting; **PASS — clean Release** |
| 11 | `same_person_target_nn` | NN, same-person | global probe, retained-variant, context/component and canonical coordinates; ordinary/none | R7 + T4 | same protected policy | E; terminal failure produces no `same_person` output | Stage 3 matrix + Stage 5 event regression; **PASS — clean Release** |
| 12 | `same_person_projection_tn` | TN, same-person | global probe, component/context and canonical ranges; ordinary/none | R7 + T4 | same protected policy | E; no partial compact publication | Stage 3 matrix + Stage 5 event regression; **PASS — clean Release** |
| 13 | `same_person_projection_nn` | NN, same-person | global probe, component/context and canonical ranges; ordinary/none | R7 + T4 | same protected policy | E; no partial compact publication | Stage 3 matrix + Stage 5 event regression; **PASS — clean Release** |
| 14 | `same_person_gram_tn` | TN, same-person | global probe/component coordinate plus role=`tile` or `global_merge` | R7 + T4 for the operation; role-specific one-shot selectors | same protected policy; local tile reduction and in-process global merge are no longer semantically ambiguous | E includes role; terminal failure publishes neither local raw state nor final statistic | Stage 3 matrix + Stage 5 tile/global-role targeting; **PASS — clean Release** |
| 15 | `trait_score_tn` | TN, trait | retained-variant block, context/trait packing, and canonical coordinate; ordinary/none | R7 + T4 | same protected policy | E; failed executor exposes only metadata and cannot rerun | Stage 5 full trait fault matrix; **PASS — clean Release** |
| 16 | `trait_feature_projection_tn` | TN, trait | retained-variant block and canonical flattened feature-tile coordinates; ordinary/none | R7 + T4 | same protected policy; canonical coordinate is independent of a local flattened buffer offset | E; no trait arrays on failure | Stage 5 full trait fault matrix; **PASS — clean Release** |
| 17 | `trait_feature_projection_nn` | NN, trait | retained-variant block and canonical flattened feature-tile coordinates; ordinary/none | R7 + T4 | same protected policy; canonical coordinate is independent of a local flattened buffer offset | E; no trait arrays on failure | Stage 5 full trait fault matrix; **PASS — clean Release** |

The Stage 2 and Stage 3 tests parameterize R7/T4 across their seven operations
each; the Stage 5 suite adds the same exhaustive treatment for all three trait
operations and validates the richer terminal report. Clean Release
qualification ran the four Stage 5 files (72 passed) and the six Stage 2-5
reference/trait/fit files (350 passed), so the matrix is not inferred from the
new Stage 5 files alone.

### Capacity, completeness, and ownership gates

- Preflight derives the maximum protected output, four admitted guarded
  buffers (primary, retry, witness, fallback), all phase scratch, compact
  outputs, and the maximum event ledger. No recovery allocation is admitted
  lazily after decoding begins.
- Exact workspace/event capacity is accepted; capacity minus one is rejected
  before descriptor traversal. The event estimate reserves publication and
  terminal slots.
- Publication validates that event sequences are contiguous, process/thread
  IDs are nonzero and constant for the one-shot executor, every attempt event
  has dimensions and integrity evidence, and protected-call and
  semantic-verification counts both equal the admitted operation ledger.
- Stable adapters independently enforce runtime-phase versus semantic-phase
  classification (including the explicit `full_target_nn` source/action
  distinction), strict membership commitments for strict full-target and both
  group-restricted operations, exact tile/global-merge and
  action/genotype-placement distributions, and a one-to-one protected-call to
  scalar-verification `(operation, semantic_anchor)` multiset.
- Recovered-result adaptation binds the top-level point selector to the
  injected protected range, requires that range to occur in the admitted
  protected-call ledger, and binds detection, retry, repair, and trusted
  fallback events to that same operation and range. Coherent phantom-range,
  operation, mode, selector, role, placement, phase, and membership grafts are
  rejected.
- A native terminal exception changes lifecycle to `failed`, appends a
  terminal event, and exposes only `failure_report()`. That report contains no
  ndarray. `run()` is one-shot after success or failure.
- Native compact arrays are owned, read-only copies. Stable arrays are
  little-endian, C-contiguous, bytes-backed immutable copies; nested metadata
  is recursively frozen.
- A process-wide dispatcher mutex, rather than an executor-local mutex,
  serializes protected entry across concurrent executors. Runtime identity
  binds the admitted BLAS thread count, OpenMP dynamic/max/capacity state, and
  Linux CPU affinity before and after protected work.
- The only admitted NUMA policy remains the exact unbound nonclaim. It cannot
  be relabeled as verified placement or a selected output node.

Capacity/terminal/thread qualification:

```text
PYTHONPATH=/tmp/summit-stage5-release-install.PsC2Xv \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage5_native_integrity.py
48 passed in 4.42 s; exact capacity accepted, capacity-minus-one rejected,
process-wide serialized entry and terminal metadata-only behavior passed
```

## Native result and backend-v2 boundary

| Layer | Final Stage 5 identity | Compatibility rule |
|---|---|---|
| Contextual native API | integer `1` | exact; any other integer rejects |
| Reference native implementation | backend name `plink_bed_descriptor_stream_stage2_v1`, physical version `2`; stable identity `plink_bed_descriptor_stream_stage2_v1:2` | exact; backend `:1`, missing version, or unknown version rejects |
| Trait native implementation | backend name `plink_bed_descriptor_stream_trait_v1`, physical version `2`; stable identity `plink_bed_descriptor_stream_trait_v1:2` | exact; backend `:1`, missing version, or unknown version rejects |
| Protected execution backend | `deterministic_tiled_fp64_with_scalar_witness_v1` | exact; vendor or unknown execution backend rejects |
| Stable logical schemas | `contextual_reference_v1`, `contextual_trait_v1`, `contextual_fit_v1` | remain V1; exact family/magic/suffix/key rules apply |
| Grouped physical encoding | `grouped_unnormalized_dense_v1` | exact and common across compatible reference/trait inputs |
| Python fit backend | `python_numpy_scipy_summary_fit_v1` | exact; source-module content digest is recorded in fit compatibility evidence |

The physical version increase is a release-hardening boundary, not a logical
science version increase. Stage 3/4 V1 files were draft artifacts created
before independent authority, complete execution evidence, and bounded loader
preflight were finalized. They are intentionally rejected and must be
regenerated from their original trusted inputs. No converter is provided: a
converter could copy bytes, but it cannot invent independent authority or
execution provenance that the draft never contained.

Reference and trait adapters require exact native result key sets. Missing or
extra keys, incomplete scientific digest maps, a mismatched execution-plan or
phase digest, an inconsistent build/source identity, nonterminal lifecycle,
incomplete telemetry, or an incorrect file identity fails adaptation before a
stable artifact can be written.

## Authority, digest, phase, file, and build evidence

### Independent semantic authority

| Boundary | Independent authority required by Stage 5 | Native/stable evidence compared | Failure behavior |
|---|---|---|---|
| Reference samples | sample order; retained-sample map; fixed-effect specification; exact fixed basis | Python authority digests versus native sample map/fixed-basis/source identities | mismatch rejects adaptation |
| Reference variants and scale | variant/allele order; retained-variant order; affine means/inverse scales; missingness | closed scale descriptor and native array/digest evidence | mismatch or unsupported descriptor rejects |
| Context basis | basis specification, calibration, and exact evaluated `Phi` | Python-owned specification/calibration plus native evaluated-basis digest | mismatch rejects |
| Annotation axis | numeric annotation map and exact ordered annotation names | native map digest, exact name sequence, component map | coherent numeric relabeling cannot pass |
| Deletion-group axis | numeric group assignment and exact ordered group labels | native group map, stable group sequence, group permutation and grouped arrays | relabel/reorder/mass/count mismatch rejects |
| Sample probes | exact policy plus explicit-probe or ordered Philox identity | native sample-probe policy/count/digest | policy/count/digest mismatch rejects |
| Variant probes | exact policy plus explicit-probe or ordered Philox identity | native variant-probe policy/count/digest | policy/count/digest mismatch rejects |
| Trait inputs | phenotype batch, residual basis, ordered trait IDs, ordered residual names | native phenotype/residual digests and exact names | relabeling or array mismatch rejects |
| Reference/trait fit pair | build, source tree, API, closed backend pair, scale, component/map/name order, masses/counts, numeric/group/deletion policy | exact digests and exact array equality; no tolerance-based mass compatibility | mismatch rejects before normal equations |
| Optional evaluation grid | explicit non-row role, provenance SHA-256, and trusted non-row assertion | grid/basis dimensions and stored role/provenance | absent assertion or row-like relabeling rejects |

The closed genotype scale descriptor requires the admitted affine policy,
counted-allele orientation, PLINK SNP-major diploid hard-call coding,
`provided_v1` centering source, exact centering/scaling formulas, sealed-mean
imputation, diploid ploidy, retained-variant order, and both affine-array
digests. Free-form alternative descriptors are not accepted by stable
reference or trait adapters.

### Native digests and phase seals

The reference native result publishes a complete digest map for:
`gram`, `raw_gram_numerator`, `annotation_masses`, `same_person`,
`group_gram_unnormalized_num`, `group_annotation_masses`,
`group_variant_counts`, `pair_q`, `pair_r`, `pair_eta`,
`component_annotation`, and `component_pair`. It also publishes a `post_gram`
phase-evidence SHA-256.

The trait native result publishes a complete digest map for its 12 compact
moment arrays plus `pair_q`, `pair_r`, `pair_eta`,
`component_annotation`, and `component_pair`. It publishes separate
`residual_derived_state` and `post_trait_outputs` phase-evidence SHA-256
values. Stable adapters recompute the native digest map from the returned
arrays and require the exact expected key set.

`execution_plan_sha256` binds the backend identities, mode/policies, probes,
trait input identities, all scientific maps/arrays, missingness, group order,
NUMA policy, dimensions, tiles, thread counts, differential flags, total call
count, maximum protected output, workspace/event admission, and the exact
17-operation call ledger. It is distinct from the scientific sealed-plan
identity.

### File identity

The native file evidence includes duplicated close-on-exec descriptors,
descriptor device/inode/size/link/time state, full BIM and FAM SHA-256,
the BED header SHA-256, a SHA-256 for every retained BED record, and the
logical retained-record stream SHA-256 accumulated during authoritative decode
passes. Repeated authoritative passes must reproduce that stream digest;
checkpoint fstat/boundary evidence and a private retained-record mutation test
add detection during execution.

This is stronger retained-input evidence, but it is intentionally not reported
as a full BED snapshot. The result explicitly records
`full_bed_file_sha256_claimed=false`,
`absolute_snapshot_or_lease_claimed=false`, and `toctou_closed=false`.

### Build and event provenance

Native build provenance records source commit/tree digest, compiler ID/version,
C++ standard, build type, sanitizer mode and enabled flags, effective
optimization, architecture tuning, configured compiler flags, BLAS identity,
integrity/checksum flags, private BLAS/OpenBLAS identities where configured,
OpenMP/native-tuning flags, the contextual dispatch backend, and the explicit
non-vendor dispatch claim. `gxeldcore.build_info()` derives optimization,
tuning, sanitizer, and flag values from the same CMake configuration rather
than a hard-coded `-O3` string.

The source-tree SHA-256 is framed directly over each tracked path, Git mode,
byte size, and file-content SHA-256 in canonical tracked-file order. It does
not wrap a Git SHA-1 tree identifier. Stable reference and trait validation
also requires the exact same internally consistent native build-provenance
object before fit compatibility records its canonical SHA-256.

The installed shared-object SHA-256 remains externally recorded in this
report. It is not claimed as a self-hash embedded inside the binary. Likewise,
the source-tree digest identifies the configured source snapshot, but the
native artifact does not independently attest a version-control dirty flag.

Event fingerprints are deterministic FNV-64 mutation detectors within one
native execution, while scientific arrays, plan/phase identities, manifests,
and file content use canonical SHA-256. FNV-64 event fields are not claimed to
be cryptographic authentication against a malicious in-process writer.

## Stable loader and durability hardening

All three stable families share a private bounded NPZ reader. Before loading a
scientific array it:

- opens one regular-file descriptor, bounds and validates the classic
  comment-free end-of-central-directory record before `ZipFile` construction,
  and rejects ZIP64 central directories, prefixes, gaps, data descriptors,
  comments, encryption, unsupported compression, invalid names, directories,
  duplicate members, noncanonical local/central metadata, and member counts
  outside the family bound;
- bounds member, manifest, NPY-header, and total uncompressed sizes;
- parses NPY magic/version/header before payload allocation, requires exactly
  `descr`, `fortran_order`, and `shape`, rejects duplicate header keys, object
  dtype, Fortran payloads, invalid/negative dimensions, comments or
  noncanonical padding, and inconsistent byte counts;
- requires exact little-endian `<f8` or `<i8` scientific dtypes and exact
  manifest-derived shapes;
- validates the family suffix, magic, strict manifest keys, canonical lowercase
  SHA-256, and manifest digest before loading scientific arrays; and
- streams each admitted member through the ZIP reader so CRC/truncation and
  trailing-payload failures are detected, then rechecks dtype and shape.

Writers fsync the temporary file, atomically replace the destination, and
fsync the parent directory. Loaders reject cross-family suffix/magic/key sets,
and development loaders do not interpret stable suffixes. Stable writers
accept only their own artifact class.

Loader/tamper qualification:

```text
PYTHONPATH=/tmp/summit-stage5-release-install.PsC2Xv \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage3_reference_v1.py \
  tests/test_context_stage4_trait_v1.py \
  tests/test_context_stage4_fit_v1.py
58 passed in 4.70 s; duplicate/CRC/truncation/header/size/key/canonical-layout
tamper cases and atomic durable publication checks passed
```

## Schema compatibility and migration matrix

| Input/producer | Consumer | Stage 5 result | Required action/rationale |
|---|---|---|---|
| Final reference V1, backend `stage2_v1:2`, exact manifest and arrays | reference V1 loader | accept after bounded preflight and full verification | none |
| Final trait V1, backend `trait_v1:2`, exact manifest and arrays | trait V1 loader | accept after bounded preflight and full verification | none |
| Final fit V1, exact Python backend and recorded backend-v2 pair | fit V1 loader | accept after bounded preflight and recomputation | none |
| Draft Stage 3 reference V1 with backend `stage2_v1:1` or incomplete authority/provenance | final reference loader or fit compatibility | reject closed | regenerate from trusted row-level inputs with Stage 5 code |
| Draft Stage 4 trait V1 with backend `trait_v1:1` or incomplete authority/provenance | final trait loader or fit compatibility | reject closed | regenerate; no provenance-inventing converter |
| Draft Stage 4 fit V1 recording the backend-v1 pair | final fit verification | reject closed | refit from regenerated final reference/trait artifacts |
| Reference V1 presented to trait or fit loader | wrong-family loader | reject suffix/magic/exact-key mismatch | use matching loader only |
| Trait V1 presented to reference or fit loader | wrong-family loader | reject suffix/magic/exact-key mismatch | use matching loader only |
| Fit V1 presented to reference or trait loader | wrong-family loader | reject suffix/magic/exact-key mismatch | use matching loader only |
| Development reference/trait/fit container | stable V1 loader | reject exact stable suffix/magic/schema | explicitly adapt from trusted source; do not relabel a file |
| Stable V1 container | development loader | reject stable suffix | use the stable family loader |
| Future logical schema, backend version, magic, grouped encoding, or unknown keys | any final V1 adapter/loader | reject closed | implement an explicit reviewed version boundary |
| Reference and trait with different build/source tree, scale, maps, labels, component order, masses/counts, deletion/numeric policy, or reference commitment | fit | reject before assembly | regenerate a compatible pair; no tolerance-based coercion |
| Transient variant-probe partial or oracle merge result | reference/trait/fit writer or loader | reject; no stable writer exists | consume only through the transient merge boundary |
| Archive with duplicate names, uppercase/noncanonical digest, wrong endian/dtype/shape, object array, corrupt CRC, truncated payload, oversized header/member, extra key, or missing key | matching stable loader | reject before publication/use | reproduce an exact canonical artifact |

## True fresh-process summary-only gate

`tests/test_context_stage5_schema_hardening.py` replaces the prior synthetic
helper-only interpretation with a real native teardown:

1. a spawned builder process creates BED/BIM/FAM and phenotype inputs, invokes
   the native Stage 3 reference and Stage 4 trait executors, adapts their exact
   backend-v2 results, writes stable reference and trait artifacts, and exits;
2. the parent removes BED, BIM, FAM, and phenotype input files and removes the
   source directory;
3. a separate fitter process receives only the two stable artifacts, loads
   them, performs the selected-trait fit, writes/reloads stable fit V1, and
   verifies the raw rank and leave-one-group-out panel; and
4. the process inspects exact archive keys/layout declarations and manifest
   text for sample-, variant-, or row-axis leakage and source-path/name
   leakage.

The stable reference, trait, and fit arrays are compact summary arrays; none
has a sample or retained-variant axis. The raw `[C,N]` state used for a
distributed same-person merge is deliberately transient and is forbidden from
all three stable writers.

Fresh-process settled result:

```text
PYTHONPATH=/tmp/summit-stage5-release-install.PsC2Xv \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage5_schema_hardening.py
12 passed in 5.67 s; real builder teardown, separate stable-only fitter,
schema migration/cross-loader, provenance, and coherent telemetry tamper gates passed
```

## Process/socket variant-probe merge

Stage 5 defines an immutable transient partial with:

- `probe_sums` in canonical logical `[C,N]` order;
- `within_probe_cross` in `[C,C]` order;
- exact global half-open probe range and ordered per-probe commitments;
- global probe-plan, science-identity, build/source/backend, scale/map/basis,
  numeric/feature, array, merge-plan, and range-commitment digests;
- process ID/slot, start method, hostname, CPU affinity, and socket evidence;
  and
- `contains_sample_axis=true`, `durable_artifact=false`, and a
  `partial_complete` lifecycle required by the merger.

After sorting by global range, the merger rejects duplicate, overlapping,
gapped, or incomplete coverage; mixed identities/dimensions; altered array,
range, or global-plan digests; nonfinite state; and already-finalized inputs.
Arrival order therefore does not change the result. For partials `b`, it forms

```text
S = sum_b S_b                    [C,N]
W = sum_b W_b                    [C,C]
D = (S S^T - W) / (B_D (B_D - 1))
```

and symmetrizes without clipping, PSD projection, or loss of sign. Spawned
tests cover `B_D=2` with one probe per process and `B_D=5` with reversed
arrival, including CPU/socket evidence where the host exposes it. Native
`same_probe_sums` and `same_probe_cross` are available only from the explicit
nonproduction differential snapshot and are adapted with layout and digest
checks.

The resulting compact statistic is explicitly labeled
`numpy_oracle_transient_v1_not_production_protected`,
`production_protected_finalizer=false`, and `durable_artifact=false`. It
qualifies the partition identity, transport, merge equation, and spawn/socket
integration. It does not satisfy the production protected finalizer gate.

Process-merge settled result:

```text
PYTHONPATH=/tmp/summit-stage5-release-install.PsC2Xv \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage5_probe_merge.py
9 passed in 3.75 s; spawned process/range-order oracle merge and available
CPU/socket evidence checks passed
```

## Release, sanitizer, and regression qualification

Release qualification below originated from the clean settled Stage 5
implementation/test commit in a separate source worktree and used new build
and install directories. Both sanitizer builds and their independently
inspected test partitions are also settled below.

### Release build

```bash
stage5_qual_source=/tmp/summit-stage5-qual-src.Dd9JNl
stage5_release_build=/tmp/summit-stage5-release-build.3FBu5E
stage5_release_install=/tmp/summit-stage5-release-install.PsC2Xv
cd "$stage5_qual_source"
git status --short --branch

CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit cmake \
  -S "$stage5_qual_source" \
  -B "$stage5_release_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$stage5_release_install"
cmake --build "$stage5_release_build" --parallel 2
cmake --install "$stage5_release_build"
```

Release configure, build, and install passed. `gxeldcore.build_info()` reported
the embedded commit and source-tree SHA-256 recorded above, `Release`,
sanitizer mode `none`, effective optimization `-O3`, and architecture tuning
`-march=native`. The installed extension SHA-256 is
`b16f2d1c9c5bab0e99f6a801dcfe52d5438a7d173582388b95e24e44f6676ad9`;
its RUNPATH is `$ORIGIN:/home/bronsonj/anaconda3/envs/summit/lib`.

### ASan+UBSan build

```bash
stage5_asan_build=/tmp/summit-stage5-final-asan-build.geKkop
stage5_asan_install=/tmp/summit-stage5-final-asan-install.h5fuJ5
CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit cmake \
  -S /tmp/summit-stage5-qual-src.Dd9JNl \
  -B "$stage5_asan_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGWLDCORE_ENABLE_ASAN_UBSAN=ON \
  -DCMAKE_INSTALL_PREFIX="$stage5_asan_install"
cmake --build "$stage5_asan_build" --parallel 4
cmake --install "$stage5_asan_build"
```

Configure, build, and install passed. `build_info()` reports the exact clean
commit/tree above, `asan_ubsan`, both sanitizer flags enabled, effective `-O1`,
portable architecture tuning, `-g`, frame pointers, no sanitizer recovery, and
`-fsanitize=address,undefined`. The installed extension SHA-256 is
`e95a4c605a79bd3ee6000115e01c97915135614d455b021dbcaecba444be9640`.

ASan qualification used:

```bash
export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/12/libasan.so:/home/bronsonj/anaconda3/envs/summit/lib/libstdc++.so.6
export ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
export CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit
export LD_LIBRARY_PATH=/home/bronsonj/anaconda3/envs/summit/lib
export PYTHONPATH=/tmp/summit-stage5-final-asan-install.h5fuJ5${PYTHONPATH:+:$PYTHONPATH}
```

Both preloads are required on this host; preloading `libasan` alone cannot
intercept the late-loaded `__cxa_throw`. Address and undefined-behavior checks
remained fatal. Leak checking is explicitly excluded: an isolated
`detect_leaks=1` clean-import control exited 134 on the host CPython baseline,
reporting 3,008 bytes in 3 allocations at `_PyType_AllocNoTrack`.

### UBSan-only build

```bash
stage5_ubsan_build=/tmp/summit-stage5-final-ubsan-build.6UNLPa
stage5_ubsan_install=/tmp/summit-stage5-final-ubsan-install.V2D0xi
CONDA_PREFIX=/home/bronsonj/anaconda3/envs/summit cmake \
  -S /tmp/summit-stage5-qual-src.Dd9JNl \
  -B "$stage5_ubsan_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGWLDCORE_ENABLE_UBSAN_ONLY=ON \
  -DCMAKE_INSTALL_PREFIX="$stage5_ubsan_install"
cmake --build "$stage5_ubsan_build" --parallel 4
cmake --install "$stage5_ubsan_build"
```

Configure, build, and install passed. `build_info()` reports the exact clean
commit/tree above, `ubsan_only`, only the undefined-behavior sanitizer enabled,
effective `-O1`, portable architecture tuning, `-g`, frame pointers, no
sanitizer recovery, and `-fsanitize=undefined`. The installed extension
SHA-256 is
`3978799b4316cb213f21ef5c662187611b6b4952dfdbd2baaff608f778720f65`.
Qualification used
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` and placed
`/tmp/summit-stage5-final-ubsan-install.V2D0xi` first on `PYTHONPATH`.

The sanitizer options are mutually exclusive. A normal Release build retains
the established `-O3` and configured native-tuning policy.

### Required test result table

| Qualification partition | Release | ASan+UBSan | UBSan-only |
|---|---:|---:|---:|
| Stage 5 focused files | **72 passed, 13.38 s** | **72 passed, 18.93 s** | **72 passed, 13.18 s** |
| Stage 2–5 reference/trait/fit focused regression | **350 passed, 14.67 s** | **350 passed, 31.82 s** | **350 passed, 15.95 s** |
| All `tests/test_context_*.py` | **740 passed, 70.32 s** | **740 passed, 112.18 s** | **740 passed, 70.81 s** |
| Required legacy native/context partitions | **103 passed, 1 skipped, 7.35 s** | **88 passed, 22.00 s** | **88 passed, 15.64 s** |
| Full repository settled total | **1512 passed, 5 skipped, 1 XPASS, 122.45 s** | not run; scoped Stage 5/context/legacy gate passed | not run; scoped Stage 5/context/legacy gate passed |

Sanitizer output was inspected for ASan/UBSan diagnostics rather than relying
only on pytest exit status. No sanitizer diagnostic occurred in an exercised
contextual or required legacy path.

The clean Release partitions used the installed tree first on `PYTHONPATH`:

```bash
cd /tmp/summit-stage5-qual-src.Dd9JNl
export PYTHONPATH=/tmp/summit-stage5-release-install.PsC2Xv${PYTHONPATH:+:$PYTHONPATH}

/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_context_stage5_build_integrity.py \
  tests/test_context_stage5_native_integrity.py \
  tests/test_context_stage5_probe_merge.py \
  tests/test_context_stage5_schema_hardening.py

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
  tests/test_gxe_native_core.py \
  tests/test_gxe_native_hardening.py \
  tests/test_gxe_native_scratch_capacity.py \
  tests/test_gxe_nn_integrity_diagnostic.py \
  tests/test_gxe_multi_environment.py \
  tests/test_gxe_reference_helpers.py \
  tests/test_gxe_summary_mom.py \
  tests/gxe_completion/test_feature_conventions.py

/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q
```

The sanitizer runs repeated the Stage 5 and six-file command lists above with
these exact basetemps:

| Configuration | Stage 5 basetemp | Stage 2-5 basetemp |
|---|---|---|
| ASan+UBSan | `/tmp/summit-stage5-final-pytest.ZA3kXh/asan-stage5` | `/tmp/summit-stage5-final-pytest.ZA3kXh/asan-stage2-5` |
| UBSan-only | `/tmp/summit-stage5-final-pytest.ZA3kXh/ubsan-stage5` | `/tmp/summit-stage5-final-pytest.ZA3kXh/ubsan-stage2-5` |

The sanitizer legacy partition was the following exact seven-file set, run
once per sanitizer environment with basetemps
`/tmp/summit-stage5-final-pytest.ZA3kXh/asan-legacy-native` and
`/tmp/summit-stage5-final-pytest.ZA3kXh/ubsan-legacy-native`:

```bash
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_gxe_native_core.py \
  tests/test_gxe_native_hardening.py \
  tests/test_gxe_native_openmp_placement.py \
  tests/test_gxe_native_output_numa.py \
  tests/test_gxe_native_scratch_capacity.py \
  tests/test_gwld_mc.py \
  tests/test_pgen_gwld.py
```

These focused and legacy runs used the clean qualification worktree as CWD and
did not override `TMPDIR`; their explicit basetemps are recorded above.
The all-context sanitizer runs used fresh `TMPDIR` values
`/tmp/summit-stage5-final-asan-context.qYyZWi` and
`/tmp/summit-stage5-final-ubsan-context.D37f3t`, respectively, without an
explicit basetemp. All five Release partitions and all four scoped partitions
under both sanitizer configurations passed. Sanitizer output inspection found
no address- or undefined-behavior diagnostic. The only pytest diagnostics were
known temporary-directory cleanup warnings. Full-repository sanitizer runs
were not performed; the required scope was the focused Stage 5, six-file,
all-context, and seven-file legacy partitions recorded above.

## Residual assumptions and explicit nonclaims

These items are not marked passed and must not be inferred from the Stage 5
fixture-scale integrity suite.

1. **No production protected cross-process finalizer.** Stage 5 proves the
   transient partial contract, spawned partition/merge behavior, and exact
   same-person equation with a NumPy oracle. A native protected finalizer with
   admitted scratch, event coverage, terminal behavior, and stable integration
   remains required before a distributed production claim.
2. **No total-process RSS cap.** Native preflight exactly accounts for its
   declared buffers, phase lifetimes, recovery reserve, telemetry, and compact
   output. It does not cap allocator metadata, Python/NumPy/SciPy state,
   loader copies, runtime libraries, page cache, or process-wide peak RSS.
3. **No absolute file lease/snapshot and no closed TOCTOU proof.** Full BIM/FAM
   and retained BED record identities plus repeated checkpoints detect the
   tested mutations, but unretained BED content is not fully hashed and no
   immutable snapshot, filesystem lease, or final check-to-publication lock
   excludes every concurrent writer.
4. **No verified NUMA page placement or NUMA performance claim.** CPU affinity
   is authenticated and spawned merge processes record observable socket IDs,
   but the native executor uses an unbound allocator and reports
   `numa_verified=false`. There is no `move_pages`/page-map attestation,
   first-touch ownership proof, remote-access measurement, or qualified output
   node.
5. **No one-million-variant dataset result.** Stage 5 is a release-integrity and
   schema gate. It does not provide the packaged Stage 6 one-million-variant
   memory, I/O, throughput, page-fault, NUMA, or end-to-end timing evidence.
6. **No embedded installed-binary self-hash or independent dirty-tree
   attestation.** The report records the installed binary hash externally and
   artifacts record source/build configuration. Those facts do not constitute
   a runtime self-measurement or a signed supply-chain attestation.
7. **No cryptographic in-process adversary model for event FNV-64.** The event
   fingerprints detect the injected and accidental mutations in scope. They
   are not a MAC, do not defend against a malicious process with arbitrary
   memory access, and do not replace the SHA-256 artifact/file identities.
8. **No new direct-grouped production selection claim.** Both placements are
   separately identifiable and protected. Stage 5 does not select the direct
   algorithm by a representative performance model; stable publication keeps
   the approved group-restricted attribution contract.
9. **No durable sample-bearing partial.** The transient `[C,N]` process-merge
   state is intentionally not serializable through stable writers. A future
   distributed transport must retain the same identity, privacy, lifecycle,
   and zeroization obligations rather than treating it as a compact artifact.
10. **No LeakSanitizer or sanitizer-full-repository claim.** Address and
    undefined-behavior checks were fatal over the required focused, contextual,
    and legacy partitions, but leak detection was disabled after the isolated
    clean-import control reproduced the 3,008-byte host CPython baseline. The
    entire repository passed under Release; sanitizer qualification used the
    explicit scope recorded above.

## Gate analysis for Stage 6

The final decision applies this rule:

- **`NARROW_GO`** if the exact 17-operation R7/T4 matrix, static call guards,
  Release suite, available sanitizer suites, schema/loader tamper matrix, true
  fresh-process teardown, and process/socket oracle merge all pass, with only
  the explicit later-stage nonclaims above remaining.
- **`NO_GO`** if any required test or sanitizer gate fails; if any contextual
  wide call can bypass `protected_gemm`; if telemetry can overflow/drop or a
  terminal failure can publish arrays; if backend-v1/unknown/cross-family
  artifacts are accepted; if compact stable artifacts leak row axes; or if
  reference/trait compatibility is tolerance-based or lacks independent
  authority.
- **`GO`** only if the residual production protected finalizer, process RSS,
  absolute file immutability, verified NUMA placement, and production-scale
  evidence are also closed in this stage. That is not the implementation
  described by this report.

Every `NARROW_GO` prerequisite above passed, and no `NO_GO` condition was
observed. Stage 6 may optimize only behind these exact identities, admission
formulas, event completeness checks, and protected dispatcher. No performance
path may bypass, weaken, conditionally disable, or under-size them in
production mode.
