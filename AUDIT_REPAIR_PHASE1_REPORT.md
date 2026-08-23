# Audit repair phase 1 report

Repairs implementing the first tranche of the independent audit
`SUMMIT_GxE_LDScore_Audit_2026-08-18`. Scope: make the phenotype-free GxE
LD-score reference constructor scientifically exact (Finding 1), truthfully
memory-admitted (Findings 2/3/6), fail-closed at the native boundary
(Finding 7), safe against destructive fused overwrite (Finding 4), and
release execution scratch before publication (Priority 0.3). The verified
estimator conventions — projected-feature algebra, XX/XW/WX/WW layout,
finite-probe normalization, Philox realization, same-person U-statistic,
multi-environment equality, the `B <= 10` Mailman rule, and the private
static pthread-BLIS boundary — are preserved exactly.

## 1. Commit and tree identities

| | commit | tree |
|---|---|---|
| audited checkpoint | `15dd349852b08f53cbedd403d199337b7eea70d0` | `40fe6ba115f27799801f78dc8803cb1fefb91f8e` |
| starting HEAD (branch `codex/gxe-throughput-optimization`) | `28ce38987b094e56a70dd8ac42c386e6b1f73a53` | `d2d2edf6f51a9d1fbd4995ac67a0f2f69194d4e7` |
| ending code commit (branch `fable/gxe-audit-repairs-phase1`) | `15dde5b0fb…` (full: see log) | `1add86bde1cb7f6e7421316ebe97370da857d1ee` |

(This report is committed on top of `15dde5b` as the final commit of the
phase.)

Commit series (oldest first), each landed only after its focused tests
passed:

1. `9418a75` — Commit A: canonical binary64 annotation semantics (Finding 1)
2. `7b72333` — Commit B: bounded maximum-capacity native scratch roles
   (Findings 2/3)
3. `cdfc877` — Commit C: q-panelled packed feature contraction with exact
   worker accounting (Finding 2)
4. `30eaa3c` — Commit D: `release_execution_scratch()` before publication
5. `851fdeb` — Commit E: fused overwrite fails safe (Finding 4)
6. `89d2498` — optional hardening: bounded semantic evidence + native mass
   validation (Findings 6/7)
7. `b38f585` — documentation reconciliation
8. `15dde5b` — semantic-record cap acts as a candidate-feasibility filter
   (production-scale correction to commit 6, found by the 10K gate)

## 2. Per-finding summary

### Finding 1 — annotations rounded to float32 (Commit A, `9418a75`)

Files/symbols: `gwe_ldscore.py` (`_read_annot`,
`_canonicalize_annotation_matrix` (new), `_annotation_digest`, the reference/
shard/moments manifest writers, sketch-path `annot_blk`), `gxe_multi.py`
(packed-path annotation casts), `gxe_score.py`
(`_validate_reference_manifest`, moments writers echo the reference schema),
`inference/gxe.py` (`_SUPPORTED_SCHEMA_VERSIONS`, pledge checks),
`gxe_merge.py` (shard v2/v3 handling, merged schema derivation), `cli.py`
(`--dtype` help), `scripts/gxe/hoffman/{deployment_config.json,hoffman_deploy.py}`.

Invariant: the canonical annotation matrix is always contiguous binary64 and
defines the estimand; the `dtype` option controls randomized probe/sketch
retained storage only. Conversion to canonical binary64 must be exact
(integer >2^53 and lossy extended-precision inputs are rejected), validation
runs after the final conversion, masses derive from the canonical matrix,
and annotation text parsing is correctly rounded
(`float_precision="round_trip"`). Artifact semantics are versioned:
reference manifests move to schema v4 with `annotation_value_dtype:
"float64"` and `annotation_digest`; shards move to v3 with the same pledge;
v3 shards seal v4 references while legacy v2 shards still seal v3
references; moments echo their reference's schema version and both
consumers reject cross-contract pairs and malformed pledges.

Tests added: `tests/test_gxe_annotation_precision.py` (5 tests: exact
conversion incl. `1` vs `1+2^-30`, `2^-150`, `2*float32_max`; matrix/mass/
digest identity across storage dtypes; full-reference output identity across
storage dtypes; binary-annotation storage-dtype invariance; v3/v4 pledge
acceptance/rejection matrix), plus updated shard/deployment contract tests.

### Findings 2/3 — exact-shape scratch caches (Commit B, `7b72333`)

Files/symbols: `gxeldcore.cpp` (`NativeGemmOutputAllocation` capacity model
with whole-capacity bind+pre-fault, `reuse_for_call` logical-fit semantics,
`prepare_reusable_native_gemm_output_mat2f`, kernel members
`source_output_scratch_`/`target_output_scratch_` (maps → single roles),
vector-scratch roles via `prepare_vector_scratch`, per-role
`ScratchRoleTelemetry`, `configure_scratch_capacities`, context-owned
`dense_decode_state_`, constructor capacity computation, capacity keys in
`info()`, NUMA evidence `capacity_byte_count` at `schema_version` 2),
`gxe_multi.py` (mirrored capacity formulas in the frozen `expected` info,
`_require_native_scratch_within_admitted_plan`, v1/v2 evidence validators,
`constructor_transient_overlap` plan component and `construction` phase,
Philox key table dropped immediately after construction; the annotation
constructor temp is eliminated by Commit A's zero-copy pass-through).

Invariant: one maximum-capacity physical allocation per execution-scratch
role, frozen from the admitted plan, reused with per-call logical shapes and
leading dimensions across full blocks, the terminal genotype block, terminal
environment tiles, terminal probe tiles, and every legal column width; a
request beyond the frozen capacity fails closed; native-reported capacities
must equal the planner's mirrored values (frozen-info equality) and never
exceed the admitted plan components (post-run check); the whole capacity is
bound and pre-faulted at allocation so exhaustive page verification remains
meaningful; source scratch keeps its deliberate release after each source
pass (the planner's phase model excludes it from the target phase), so its
allocation count equals the source-pass count with never two mappings live.

Tests added: `tests/test_gxe_native_scratch_capacity.py` (terminal geometry
blocks `[0,7)`/`[7,13)` = widths K and K-1, terminal environment tile,
terminal probe tile, multiple legal source/target widths; exactly one live
allocation per role; capacities ≤ plan; stale-data safety proven by exact
agreement with independent references when smaller shapes follow larger
ones; fused-plan capacity equalities), updated
`tests/test_gxe_native_output_numa.py` for evidence schema v2.

### Finding 2 (worker scratch) — q-panelled feature contraction (Commit C, `cdfc877`)

Files/symbols: `src/native/common/mailman.hpp` (`qpanel_width` gains a
`segment_buffers` count — shared policy, no second implementation),
`gxeldcore.cpp` (feature contraction q-panel loop; `MailmanWorkerArena`
context-owned per-worker scratch replacing all three `static thread_local`
sites; dirty-table tracking across pre/post kernels;
`configure_mailman_plan` freezing segment size, table size, per-path
q-panel widths, and exact per-worker capacities; per-block
`require_frozen_mailman_geometry`; Mailman keys in kernel/context info),
`gxe_multi.py` (`_frozen_mailman_environment` — canonical-only
`SUMMIT_MAILMAN_SEGMENT_SIZE`/`QPANEL`/`WORK_MB` with fail-closed rejection
of non-canonical values, `_mailman_segment_size`, `_mailman_qpanel_width`,
`_mailman_worker_scratch_bytes`; the flat 8 MiB-per-worker charge replaced
with the exact bytes; frozen values mirrored in the expected info).

Invariant: the packed feature contraction never allocates a lookup table
over the entire fused feature RHS width; per-worker scratch is context-owned
at `8*(table*max_panel + segment*max(feature,target)_panel +
segment*feature_panel)` bytes, frozen from the plan, eagerly allocated,
reported, bounded, and releasable; environment overrides are frozen and
exactly represented or rejected before native construction; column sums are
independent, so q-panelling leaves the moments bitwise identical (worst
observed coordinate difference attributable to panelling: 0).

Tests added: `tests/test_gxe_mailman_worker_scratch.py` (high-rank
5-environment, 175-covariate, segment-8 geometry reproducing the audited
>8 MiB full-width table regime, with exact planner/native agreement and
5e-12 agreement against independent references; frozen-or-rejected
behavior for all three overrides; unit mirror of the shared q-panel policy
including the audit's 11,154,592-byte counterexample dimensions).

### Priority 0.3 — release before publication (Commit D, `30eaa3c`)

Files/symbols: `gxeldcore.cpp` (kernel `release_execution_scratch`,
`released_scratch_bytes`, `live_scratch_capacity_bytes`; context
`release_execution_scratch` returning before/after capacity evidence;
`execution_scratch_released` state in info; nanobind binding),
`gxe_multi.py` (`_release_native_execution_scratch` invoked after the
post-run fail-closed validations and before any pandas construction or
serialization; release evidence incl. RSS published in the performance
telemetry; frozen-info expectation updated to the legitimate released state;
executor `close()` releases a completed context during exception cleanup).

Invariant: after the single-use run completed, its results were detached
into separately owned arrays (no returned array aliases scratch), and all
native validation passed, every execution-only mapping is freed; the release
is idempotent, observable, safe during ordinary exception cleanup, rejects
an incomplete context, and any later scratch request or execution attempt
fails clearly.

Tests added: `tests/test_gxe_scratch_release.py` (values remain valid after
release; live capacity drops to zero before publication with the released
bytes accounted; repeated release harmless; execution after terminal release
rejected; injected post-run failure exercises the exception-cleanup release
without masking the original error).

### Finding 4 — fused overwrite (Commit E, `851fdeb`)

Files/symbols: `gxe_multi.py` (`_require_common_contract`).

Invariant: `overwrite` is part of the common fused contract; mixed settings
are rejected; `overwrite=True` is rejected for fused batch publication
before any byte changes, because `os.replace` loses the original inode and a
later batch failure would roll back the replacement without restoring the
original. Single-environment generation keeps its existing overwrite
behavior; the existing no-overwrite rollback tests are retained unchanged.

Tests added: `tests/test_gxe_fused_overwrite_policy.py` (prepopulated valid
bundles; deterministic rejection with byte/inode/size snapshot equality and
no partial manifest, staging file, or orphan lock; mixed settings rejected
with nothing created).

### Findings 6/7 — bounded evidence and native masses (`89d2498` + `15dde5b`)

Finding 6: the context freezes `planned_semantic_call_maximum =
blocks*(1+2*tile_products) + 2*tile_products` and fails closed if the
appended records exceed it; the Python wrapper re-checks the count; the
planner charges 4096 bytes per record inside the 256 MiB telemetry
allowance and treats candidates whose record volume cannot fit it as
infeasible (`15dde5b` — at 10K scale the enumeration legitimately visits
small-tile candidates with huge record counts). Finding 7: the native
boundary verifies every supplied annotation mass against a long-double
column sum of its copied canonical annotation matrix under the documented
tolerance `64*eps*magnitude_sum + DBL_MIN`; the supplied binary64 pairwise
sums stay numerically authoritative so published values are unchanged.
Tests: `tests/test_gxe_native_hardening.py` (3 tests: bounded/admitted
record counts; all-infeasible enumeration surfaces the no-feasible-plan
error; halved masses with unchanged annotations rejected before
allocation-heavy execution).

## 3. Build and test commands

Authoritative environment: conda env `summit`
(`/home/bronsonj/anaconda3/envs/summit`, Python 3.12, NumPy 2.3.5), same as
CI (`.github/workflows/ldscore-tests.yml`).

```bash
# build + install (CI-identical)
conda activate summit && pip install -e . --no-deps
# full suite
python -X faulthandler -m pytest -q
# focused new tests
python -m pytest -q tests/test_gxe_annotation_precision.py \
  tests/test_gxe_native_scratch_capacity.py \
  tests/test_gxe_mailman_worker_scratch.py \
  tests/test_gxe_scratch_release.py \
  tests/test_gxe_fused_overwrite_policy.py \
  tests/test_gxe_native_hardening.py
# sanitizer build (ASan+UBSan, gcc 12.2, RelWithDebInfo)
cmake -C build/cp312-cp312-linux_x86_64/CMakeInit.txt \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g" \
  -G Ninja -S . -B <asan-build> && ninja -C <asan-build> gxeldcore
LD_PRELOAD="libasan.so.8 libstdc++.so.6" ASAN_OPTIONS=detect_leaks=0 \
  UBSAN_OPTIONS=print_stacktrace=1 python -S -m pytest -q <native test files>
# production private pthread-BLIS build
cmake -C build/cp312-cp312-linux_x86_64/CMakeInit.txt -DCMAKE_BUILD_TYPE=Release \
  -DGXELDCORE_USE_PRIVATE_BLIS=ON \
  -DGXELDCORE_PRIVATE_BLIS_ARCHIVE=/tmp/summit-blis-pthreads.GUubpW/install/lib/libblis.a \
  -DGXELDCORE_PRIVATE_BLIS_INCLUDE_DIR=/tmp/summit-blis-pthreads.GUubpW/install/include \
  -DGXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY=zen \
  -DGXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT=e8566eb3e773fb54d11b33e371d13f22d2941e50 \
  -DGXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256=eefbd29a5cbb1d6982bdce76e8034037ea2a3f3ead33d90d9f3cef6da5728154 \
  -G Ninja -S . -B <blis-build> && ninja -C <blis-build> gxeldcore
```

## 4. Test results

- Pre-change baseline (starting HEAD, correct env): targeted suite
  95 passed, 3 skipped; full behavior anchored by captured artifacts.
- Focused new tests: annotation 5, scratch capacity 2, Mailman worker 3,
  release 2, overwrite 2, hardening 3 — all passing.
- Full repository suite at the ending commit: see §"final suite" below.
- Sanitizers: ASan+UBSan build of `gxeldcore` (gcc 12.2,
  `-fsanitize=address,undefined -fno-sanitize-recover=all`); the six
  native-focused test files ran under
  `LD_PRELOAD="libasan.so.8 libstdc++.so.6"`: **53 passed, zero sanitizer
  reports** (no ASan errors, no UBSan runtime errors). Note: preloading
  libasan alone trips a known ASan interceptor CHECK
  (`real___cxa_throw != 0`) at the first intentional native exception
  because python itself is not linked against libstdc++; preloading
  libstdc++ after libasan resolves interception. TSan was not run: the
  system libgomp is not TSan-instrumented, which produces unusable false
  positives on every OpenMP barrier; the concurrency surface changed in
  this phase (per-worker Mailman arenas indexed by `omp_get_thread_num()`
  under kernel-serialized entry, with a team-size fail-closed guard) was
  exercised under ASan instead. This limitation is recorded as a deferred
  risk.
- Independent audit checker: rerun before and after the repairs under the
  authoritative env (NumPy 2.3.5): outputs byte-identical, `philox
  all_exact = true`, all algebra/U-statistic identities at the audited
  1e-15-scale errors. (The checker validates the estimator mathematics
  independently of the repository code; the pre/post identity plus the
  artifact comparison below is the regression evidence.)
- Production build identity (private pthread-BLIS module at the ending
  commit): API 9 / backend 1.9 / Release;
  `blas_runtime_isolation=private_static`, pthreads layer, immutable
  environment contract, serialized owner-thread entry enforced; pinned BLIS
  commit `e8566eb3…`, source tree `eefbd29a…`, archive `72006817…` (exact
  matches to the accepted identities); `gemm_integrity_enabled=true`,
  `gemm_checksum_enabled=false` (accepted private-BLIS boundary);
  `ldd`/`nm -D` show no dynamic or undefined BLAS/BLIS symbols; Mailman
  maximum probe count 10.

## 5. Numerical equivalence

Controlled pre/post comparison on captured artifacts (N=41, M=29, terminal
block width 1, three environments, dense B=64 and packed B=6, binary and
continuous annotations, storage dtypes float32/float64; all four
directional panels + diagonal tables + manifests):

| case | worst \|pre−post\| | normalized | verdict |
|---|---|---|---|
| binary dense float32 | 0.0 | 0.0 | bit-identical |
| binary dense float64 | 0.0 | 0.0 | bit-identical |
| binary packed float64 (Mailman) | 0.0 | 0.0 | bit-identical |
| non-partitioned dense float32 | 0.0 | 0.0 | bit-identical |
| continuous float64 | 1.776e-15 | 2.46e-16 (at age.gee row 9, CONT_C) | correctly-rounded text parsing only |
| continuous float32 | 1.134e-07 | 1.51e-08 (at age.gee row 27, CONT_C) | the intentional Finding-1 estimand repair |

Post-repair, `dtype=float32` and `dtype=float64` continuous-annotation
references are exactly identical (max |diff| = 0.0): both now name the same
binary64 annotation-defined estimand, and the production native path is
storage-dtype invariant. The continuous-float64 1.776e-15 difference is
fully attributable to the `float_precision="round_trip"` annotation parser
(one-ulp-correct parsing of extreme decimal values), not to any kernel
change; binary annotations are bit-identical under the unchanged
`rtol=atol=5e-12` gate. Multi-environment execution is compared against
independently constructed one-environment references (XX, XW, WX, WW
separately, never collapsed) inside the new capacity/Mailman/release tests
and the pre-existing equivalence suite, including terminal tile geometries.
Philox bits, probe identities, environment ordering, masses, and
missingness diagnostics are unchanged.

## 6. Planner / native / measured-memory reconciliation

Mechanism: the native context computes every role capacity from the same
frozen quantities as the planner; the planner mirrors the formulas into the
frozen `expected` info and `info != expected` fails closed (this is exact
equality, stronger than an inequality); after the run,
`_require_native_scratch_within_admitted_plan` additionally requires every
native-reported capacity ≤ its admitted `component_bytes` term.

Mid-size evidence run (N=4000, M=6001, step 750 → terminal block width 1,
three environments, dense B=64, 8 threads, shared-BLAS dev build;
`memory_evidence.json` in the session scratchpad):

- one allocation per role: decoded 1 (+17 reuses across 2 passes × 9
  blocks), source output 1 (fused single tile, +8 reuses), target output 1
  (+8 reuses), weights/annotation 1 each;
- native capacity ≤ admitted plan component for every role (decoded,
  source contribution, target output: all true, with equality of the
  context capacities against the mirrored expected values enforced by
  construction);
- release evidence: live scratch capacity 28,608,000 bytes → 0 before
  publication (decoded 24,000,000 + kernel 19,206,000 released bytes
  recorded), process RSS dropped 394.2 MB → 370.4 MB at the release
  boundary;
- RSS by phase: baseline 0.3145 GiB → post-construction 0.3238 GiB →
  post-publication 0.3477 GiB; process-lifetime `ru_maxrss` 0.4859 GiB;
  observed peak ≤ baseline + modeled complete-process peak (true).

What is proven versus allowed: the native scratch capacities are exact,
proven bounds (equality-checked against the plan); the total-process peak
statement remains a conservative measured allowance, because pandas,
compression, allocator, and runtime components are modeled through the
plan's explicit allowances (thread stacks, telemetry, allocator slack,
20% headroom), not derived exactly. The audit's H4 (publication memory is
a heuristic) therefore remains open by design in this phase; what changed
is that native execution scratch no longer overlaps publication at all.

## 7. Production-shaped 10K gate

Configuration (source-matched to
`reports/gxe_completion/CURRENT_STATUS_AND_REMAINING_WORK.md`): real UKBB
EUR genotypes (M=454,207), three environments (age, sex, bmi), 22
covariates, N=9,996 complete-case samples, B=256, K=2000 (228 blocks),
T=32, float32 retained storage, private pthread-BLIS build at the ending
commit, fused single-group route with explicit verified OpenMP placement
(`--gxe-parallel-environment-groups 1 --gxe-explicit-openmp-placement`),
`--target-xz-mem 4.0`, seed 20260808.

Cohort caveat: the accepted baseline's exact subsample is not reproducible
from its recorded metadata (its analysis fingerprint does not match any
plausible `--rand-samp` reconstruction; `--rand-samp 10063` reproduces the
same N=9,996 and identical design ranks but a different sample draw), so
value-level comparison against the accepted baseline artifacts is not
possible; the gate is therefore functional/memory evidence with structural
comparison (passes, calls, tiles, allocations) against the prior run, and
value-level regression evidence comes from §5. The machine carried light
background editor load (~3–4 cores of 128) during the run; per the
established policy this result is functional/memory evidence, not a
throughput claim.

Results (run completed 2026-08-19 02:08–02:14 PDT, exit 0; all fifteen
score/diagonal artifacts and the sealed batch manifest published):

- Structural comparison with the accepted baseline run: **exact match** —
  dense BLAS hybrid, Mailman ineligible at B=256, fused two-pass execution,
  one environment tile, exactly one width-256 probe tile, 228 blocks, 2
  genotype passes, 456 observed block reads, 685 planned protected output
  calls (all reconciled, 0 dropped records, 0 repaired/checksum/roundoff
  columns), matching the prior run's 456 reads / 685 outputs / 2 passes.
- New invariants at production shape: exactly one allocation per scratch
  role — decoded genotype 1 allocation + 455 reuses at 159,936,000 bytes
  capacity (9,996 x 2,000 x 8, the single maximum-capacity mapping the
  audit's Finding 3 demanded), source contribution 1 + 227 reuses at
  122,830,848 bytes, combined target output 1 + 227 reuses at 49,152,000
  bytes (the status document's "about 49 MiB" reused target output, now
  also capacity-frozen), weights/annotation 1 each; the prior
  implementation's three shape-keyed output mappings are gone.
- Semantic evidence bounded: 686 records planned maximum, 686 observed
  GEMM records, evidence buffers complete.
- Release before publication: live native scratch capacity 209,088,000
  bytes -> 0 at the release boundary; process RSS dropped 2,303.0 MB ->
  2,094.2 MB before pandas construction/serialization began;
  `execution_scratch_released = true` in the completed context info.
- Memory: peak process RSS at batch manifest 2.578 GiB <= modeled
  complete-process peak 3.793 GiB (plan peak phase: publication at 1.151
  GiB owned + allowances). The prior accepted run peaked at 2.537 GiB —
  the same envelope within 41 MiB.
- Timings (contended; functional/memory evidence, not a throughput
  claim): native descriptor end-to-end 240.2 s wall at 27.1 average
  active cores (prior accepted: 262.6 s), vendor GEMM 212.0 s wall /
  6,060.8 CPU-seconds at 28.6 average active cores across 686 calls
  (43.97 TFLOP), output compression staging 47.2 s, total wall ~6 min 05 s
  to the sealed batch manifest.
- Contention record: load average 4.4 before -> 13.6 after (the run's own
  32 threads plus a concurrently executing full pytest suite and editor
  tooling on the shared 128-core host).
- Value-level comparison against the accepted baseline artifacts was not
  possible for the cohort reason stated above (fingerprints differ under
  every plausible `--rand-samp` reconstruction of the archived run);
  value-level regression evidence is §5.

## 8. Deferred work and unresolved risks

Explicitly deferred to later phases (per the audit's fix order and the
phase instructions): Finding 5 (crash durability, fsync, stale-lock and
orphan-bundle recovery — publication remains namespace-atomic under
ordinary exceptions only, now stated as such in the invariants), Finding 8
(affinity-restoration poisoning), Finding 9 (`BLIS_THREAD_IMPL`
validation), Finding 10 (cryptographic source provenance), the
`accumulate_population` BLAS-blocking/parallel rewrite, streaming gzip
serialization, `sqrt(A)` block caching, block-width topology experiments,
true-FP32 kernels, altered Philox identity, Stage 3/sharding, B=1024
qualification, and the full N≈291K production run.

Unresolved risks carried forward:

- TSan was not run (non-instrumented libgomp); the OpenMP concurrency
  surface is mutex-serialized at kernel entry and was ASan-clean, but a
  data-race pass remains outstanding.
- The publication-phase peak remains a modeled allowance (audit H4), not
  an exact bound; streaming serialization (Priority 2.9) is the planned
  fix.
- The accepted 10K baseline's cohort draw is not reproducible from its
  recorded metadata; future baselines should persist the subsample
  identity (e.g., a keep-list digest) so value-level regression against
  archived production artifacts is possible.
- Legacy v3/v2 artifacts remain readable; only same-version
  reference/moments pairs are fittable, and legacy continuous-annotation
  artifacts name the old `fl32(A)` estimand by construction.

## 9. Recommended phase 2 scope

1. Parallelize/BLAS-block `accumulate_population` (audit Priority 1.5) —
   at B=256 it is the dominant non-GEMM native phase.
2. Stream gzip table serialization (Priority 2.9) and make the publication
   peak exact, closing H4.
3. Crash-durable publication: fsync + journaled batch commit + stale-lock
   recovery (Finding 5), after which a real overwrite transaction can lift
   the Commit-E restriction.
4. Affinity poisoning (Finding 8), `BLIS_THREAD_IMPL` contract fix
   (Finding 9), and source-provenance hardening (Finding 10).
5. The full-scale N≈291K, B=256 gate, then B=1024 qualification.

## Phase-2 increment 1 (implemented after this report was sealed)

Commits `0708b14` (Priority 1.5), `11ef0e8` (Priority 1.6), `8e9ad85` +
guard fix (automatic canonical block width):

- `accumulate_population` is parallelized over output coordinates with the
  serial accumulation order preserved per coordinate — bitwise identical to
  the serial pass (verified against all captured artifacts including the
  published same-individual products) and scaling with the family-pair
  count that grows under partitioned annotations and larger B.
- `sqrt(A)` is computed once per context and both source paths consume the
  cache via an `annotation_is_sqrt` precondition flag; bitwise unchanged.
- `--step_size auto` resolves `min(nsnps, 8192)` deterministically from the
  variant count only; the manifest records the resolved width plus a
  `step_size_selection` provenance tag, and mixed provenance across fused
  estimators is rejected.  The realization semantics are unchanged: auto is
  exactly the realization the same explicit width produces (tested
  bitwise).  The Hoffman2 shard/merge design was dropped by the project
  owner (excessive temporary data), so single-node width/topology is the
  scaling lever.
- Back-to-back 10K benchmark pair (same cohort/seed, private pthread-BLIS,
  T=32, light ~3-core background load): step 2000 → 243.0 s descriptor /
  213.7 s vendor GEMM (206 GFLOP/s, 228 blocks, 456 reads, peak RSS
  2.366 GiB); step auto=8192 → **166.0 s descriptor / 142.6 s vendor GEMM
  (308 GFLOP/s, 56 blocks, 112 reads, peak RSS 3.658 ≤ modeled 3.826
  GiB)** — a 1.46x descriptor speedup with all fail-closed gates and the
  pre-publication release intact.  ASan+UBSan on the changed paths: 15
  tests, zero findings.
- Updated N≈289K projection at the auto width: ~1,272 TFLOP at ~308
  GFLOP/s ≈ 69 min single-group T=32 under these conditions (~60–65 min
  clean); two socket-local groups ≈ 2/3 of that for three environments.

## 10. Final suite at the ending commit

`python -X faulthandler -m pytest -q` at `15dde5b`: **767 passed, 5
skipped, 1 xpassed** (64 s). The skips are the repository's established
environment-dependent skips (real-NUMA container facilities and
OpenMP-BLIS placement cases inapplicable to this host/build); nothing was
xfailed, skipped, or loosened to pass. The starting branch's CI-passing
state was 742 passed / 13 skipped at `15dd349`; the added tests account
for the growth.
