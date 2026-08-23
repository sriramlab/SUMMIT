# Contextual native estimator — Stage 7 operational runbook

> **Operational decision — NARROW_GO.** Stage 7 authorizes only bounded,
> private use of the stable reference, trait, and fit V1 artifact workflow at
> the exact release identity and T1 plan below. Public contextual CLI,
> configuration, or schema integration and large-real execution are
> **NO_GO/deferred**.

## Purpose and operating boundary

This runbook governs the compact contextual reference, trait, and fit V1
workflow. It is deliberately narrower than a public product interface.

This authorization supports bounded private stable-V1 execution, audit, and
reproduction, not:

- a public contextual CLI or stable public configuration schema;
- large real-data reference generation;
- a target-scale runtime or full-process memory claim;
- direct-action or direct-genotype grouped production;
- a two-process protected production finalizer;
- verified NUMA page placement or remote-byte accounting; or
- resumption of a native executor after terminal failure.

The selected Stage 7 execution policy is restricted grouped attribution,
FP64, full scalar-witness integrity, deterministic reductions, one process,
one decode thread, one BLAS thread, and a shared 128-variant block. The
eight-thread comparison plan was scientifically accepted on the same release
but was slower and was not selected. The earlier parallel-witness source
6bdb43f is rejected and must never be restored or promoted.

## Release identity header

An operator must verify this header before every private release run. A
missing or mismatched value stops the run.

| Identity | Required final value |
|---|---|
| Branch | codex/contextual-stage7-release-audit |
| Release decision | NARROW_GO — bounded private stable V1 only |
| Final source commit | 69203355d558792981993e2ac8fceb6851e41418 |
| Canonical tracked source SHA-256 | 5868faf8516eb46e0397091812fc2f6788ef387602d1a1d0be5204883b3b2c8c |
| Clean worktree status | Clean detached release source before and after qualification; see logs 01, 09, 12, 20, and 22 under the release evidence root |
| Release install prefix | /tmp/summit-stage7-release-6920335.yygYYi/install |
| Native module SHA-256 | 2591764e6ccedd67e0c92a8df2d4febcaa0f89967e6e2cb5615982c326f72eb4 |
| Installed package SHA-256 | 05e999e082b4cabc4126c1d5b899c8943cb53b8a015333c71c9e8099bbd6da19 |
| Python runtime SHA-256 | d525812398709daca5133a1cd12f4f7da127433a04f17a7d978151a8df6e9ac5 |
| Dependency runtime SHA-256 | 988388f2ca470dffaf208e8455771003149c9abbb495faaacf35036f9f252d10 |
| Qualification record path/SHA-256 | reports/context_native/stage7/config/stage7_fault_qualification.json / 7ac1361c88b8209df6bc5139d945cafc5499505050ee74d30af87d1d7e4a8be0 |
| Release build spec path/SHA-256 | reports/context_native/stage7/config/stage7_release_build.json / 99a19d06b0303531ee527223693366b85b00a54a1bd6812311a54417664bc1ad |
| Approved T1 plan path/SHA-256 | reports/context_native/stage7/config/stage7_core_node0_t1.json / 3d34606b5eef972ee65b45ce86fb5afb4efc1469eb3b701bb4b4c922c97b57f2 |
| Approved host/affinity identity | Tabla, physical CPUs 0-7 / 32abc5691e2cc75018c65d7903be7699fa35d37c654430a8e134a9b001340c8a |
| Frozen benchmark record/SHA-256 | reports/context_native/stage7/benchmark_core_node0_production_r3.json / 246211f8f564dace3f34ce3ef3efac7aac9d2eb71465c745f221dcf59137b670 |
| Frozen benchmark dry-run/SHA-256 | reports/context_native/stage7/benchmark_core_node0_dry_run.json / 79054829c6edc3570e18de5eff002555b6d03fd67e83342da06fc43953cb4867 |
| Release evidence capsule/SHA-256 | reports/context_native/stage7/stage7_release_evidence.json / 4876731149efbc5599d4b741368d594e216603f58177af518a00726cc751de61 |

The Stage 6 rollback build at commit 69fc6c1 is evidence for the Stage 7
candidate, not a substitute for this final header.

## Allowed private interface

The exact union of `reference_v1.__all__`, `trait_v1.__all__`, and
`fit_v1.__all__` is a frozen 28-name private stable-artifact module surface.
It is a subset of the 253-name `summit.context.__all__`, not a public API and
not a list of production entry points. Only the operations in the table below
are allowed in the audited driver; the other adapters, helpers, constants, and
classes in the 28-name module surface are not thereby authorized for direct
production use.

The installed `python -m summit --help` capture is distinct from the
deterministically normalized parser-help and parser-default captures. Neither
is a contextual V1 production interface. The final evidence keeps them
separate at `logs/18_installed_summit_help.txt`,
`logs/19_normalized_parser_help.txt`, and
`logs/19_normalized_parser_defaults.json` under the release evidence root.

| Operation | Allowed private symbol | Conditions |
|---|---|---|
| Native reference execution | summit.gxeldcore.ContextualReferenceExecutorV1, then preflight() and exactly one run through run_contextual_reference_v1 | Constructor is private and build-pinned; use only an audited driver with independently built publication identity. |
| Reference publication | write_contextual_reference_v1 | Only a verified ContextualReferenceArtifactV1; globally unique never-before-used generation/target; immediate official reload required. |
| Reference load | load_contextual_reference_v1 | Exact V1 suffix/magic and backend plink_bed_descriptor_stream_stage2_v1:2, exact keys, bounded members, manifest digest, and arrays only. |
| Native trait execution | summit.gxeldcore.ContextualTraitExecutorV1, then preflight() and exactly one run through run_contextual_trait_v1 | Same one-shot and pinned-driver rules; reference compatibility identity is mandatory. |
| Trait publication/load | write_contextual_trait_v1 / load_contextual_trait_v1 | Exact contextual trait V1 family and backend plink_bed_descriptor_stream_trait_v1:2 only; immediate official reload required. |
| Compatibility check | validate_contextual_fit_compatibility_v1 | Must pass exactly before assembly or fit; no tolerance-based coercion. |
| Summary-only fit | fit_contextual_model_v1 | Inputs must be freshly loaded stable V1 summaries; raw output remains authoritative. |
| Fit publication/load | write_contextual_fit_v1 / load_contextual_fit_v1 | Exact contextual fit V1 family and backend python_numpy_scipy_summary_fit_v1 only; immediate official reload required. |

The following are not allowed as private production entry points:

- adapt_native_contextual_reference_v1 and
  adapt_native_contextual_trait_v1 called on an unaudited mapping;
- a second run() call on any executor;
- direct publication of native result dictionaries;
- development reference/trait/fit writers or loaders for stable V1 output;
- the Stage 6 benchmark harness as a general production CLI;
- fault-injection, differential-snapshot, or trusted-fallback test controls;
- direct-action or direct-genotype grouped plans;
- transient ContextualVariantProbePartialV1 objects as durable artifacts; or
- any converter that relabels backend-v1 or development artifacts as final
  V1.

## Preflight

### 1. Establish an immutable software identity

1. Start from a clean detached source at the final audited commit.
2. Record git commit, source tree, status, compiler, flags, BLAS/OpenMP
   runtime, Python, NumPy, SciPy, native module, and installed package hashes.
3. Compare every value with the final release identity header.
4. Import summit and summit.gxeldcore from the audited install prefix, not the
   source tree or another environment.
5. Read gxeldcore.build_info() and require exact commit/tree, Release,
   sanitizer_mode=none, effective optimization, architecture tuning, native
   API/backend, compiler, and BLAS/OpenMP identity.
6. Reject a dirty tree, missing key, changed file hash, unqualified binary, or
   runtime dependency mismatch.

The final clean Release evidence root is
`/tmp/summit-stage7-release-6920335.yygYYi`. Exact configure, build, install,
RUNPATH/`ldd`, test, source-cleanliness, and identity commands and results are
preserved in `logs/07_configure_release.log` through
`logs/23_qualification_summary.log`; their ledger is
`qualification_log_sha256.txt`. The Release run recorded 1,588 passed, five
documented skips, and one documented XPASS across the repository, and 816
passed with no skip or XPASS across all contextual tests. The required
40-file contextual-plus-legacy sanitizer scope recorded 904/904 passed in
both the ASan+UBSan evidence root
`/tmp/summit-stage7-asan-6920335.0JVW73` and the UBSan-only evidence root
`/tmp/summit-stage7-definitive-ubsan-6920335.O7JawY`. The canonical
qualification record remains the hash-bound in-repository JSON named in the
release identity header.

### 2. Seal inputs and scientific authority

Before constructing an executor, record and verify:

- BED/BIM/FAM paths, sizes, device/inode, modification state, and content
  identities required by the final policy;
- retained sample and retained variant maps;
- ordered variant IDs, counted and other alleles, allele orientation, and
  count_A1 policy;
- affine means, inverse scales, missingness, ploidy, centering, and scaling
  plan;
- fixed-effect specification and fixed-basis digest;
- context basis specification, calibration, evaluated basis, and ordering;
- annotation mode, annotation weights, names, and map digest;
- deletion group membership, order, names, balance, and map digest;
- sample- and variant-probe policies, ordered keys, counts, and digests;
- trait order, phenotype batch digest, residual basis/order, and names; and
- the compatible reference manifest identity supplied to the trait
  publication identity.

Use independently constructed ContextualReferencePublicationIdentityV1 and
ContextualTraitPublicationIdentityV1 values. Do not accept labels or semantic
identities only because the native result repeats them.

The current file checks detect tested changes but do not prove an absolute
lease or immutable snapshot. For a private run, stage read-only inputs in an
operator-controlled snapshot and prevent concurrent writers. If that cannot be
guaranteed, stop rather than making a stronger TOCTOU claim.

### 3. Validate the execution plan

The approved bounded policy must retain:

- group_restricted_action_v1;
- strict_disjoint_binary_v1 where the selected plan requires it;
- contiguous_sealed_v1 group execution;
- protected_dense_gemm_v1;
- full_scalar_witness_v1 and
  deterministic_tiled_fp64_with_scalar_witness_v1;
- fp64_v1, deterministic reductions, and allocation reuse;
- one process, decode_threads=1, blas_threads=1;
- shared source/target/group/same-person/trait variant block 128;
- resident source scores and actions;
- unbound_first_touch_v1, output node -1, and explicit huge pages disabled;
  and
- Tabla physical CPUs 0-7 with affinity identity
  32abc5691e2cc75018c65d7903be7699fa35d37c654430a8e134a9b001340c8a,
  fixed headroom zero, and 2,000 basis points proportional headroom.

Do not generalize a plan across dimensions. The Stage 6 core plan used
resident probes 128 at N=128/M=512; the larger restricted plan used resident
probes 16 at N=512/M=2048. Only the former resident setting passed its
specific resident ladder. Neither result is a target-scale admission.

### 4. Require native preflight before traversal

Call preflight() once and persist its JSON before run(). Confirm:

- required_workspace_bytes is within the operator cap;
- telemetry capacity is at least the reported required capacity, including
  recovery/fallback and the terminal slot;
- all permanent, phase, compact-output, integrity-reserve, and telemetry
  ledgers are present and arithmetically consistent;
- descriptor passes, decoded blocks, logical visits, protected-call counts,
  and semantic operation ledger match the sealed plan;
- scratch is completely preallocated before decode;
- no output or intermediate has a forbidden M*C*C, Q*N*M, sample-axis, or
  retained-variant-axis stable shape;
- the maximum protected output and event storage fit their admitted types;
- the selected algorithm/backend/placement is supported; and
- the plan does not request an unsupported NUMA, process, packing, precision,
  or grouped mode.

Native admission is not total-process RSS. Add an operator RSS ceiling that
accounts for the interpreter, NumPy/SciPy, mappings, libraries, page cache,
publication copies, and load/fit coexistence. No workload outside the approved
bounded dimensions may proceed without a new audit that approves that ceiling.

### 5. Prepare output publication

1. Allocate a globally unique generation path with restrictive permissions
   and adequate free bytes/inodes. The audited wrapper must create it with one
   exclusive `os.mkdir(path, mode=0o700)` call, never `exist_ok=True`; an
   `EEXIST` result aborts the attempt. The generation directory and every final
   artifact target must not have existed before this attempt. The stable
   writer's later `mkdir(..., exist_ok=True)` parent check is not generation
   authorization and must be inert because the reviewed wrapper already
   created the generation exclusively.
2. Keep the generation on one filesystem so same-directory temporary-write,
   file fsync, atomic no-replace hard-link publication, and parent-directory
   fsync retain their intended semantics.
3. Do not overwrite or delete the last validated generation. Never reuse a
   generation path after success or failure, even if it appears empty.
4. Require the writer-side member, whole-container, classic-ZIP, and exact
   16-MiB manifest preflights before any output path is touched. The complete
   `manifest_json.npy` member, including its NPY header and UTF-32 payload,
   must be at most 16,777,216 bytes; an NPY header must be at most 65,536
   bytes. Every exact NPY member size and its conservative raw-DEFLATE bound
   must be at most `zipfile.ZIP64_LIMIT` = 2,147,483,647 bytes; the member
   count must be strictly less than `zipfile.ZIP_FILECOUNT_LIMIT` = 65,535;
   and the cumulative conservative local-header/stream offset and central
   directory size must each fit the same classic-ZIP field bound. The 8-GiB
   per-member and 16-GiB total-uncompressed parser caps do not widen these
   effective classic-ZIP writer limits and are not resource-admission claims.
5. The writer validates the fsynced temporary container with the stable
   reader, then atomically hard-links it to an absent final target. `EEXIST`
   is terminal and never replaces a file, directory, symlink, or concurrent
   winner.
6. Reject the 1,024-probe family and any workload that can approach the 16-MiB
   manifest bound. In the Stage 6 B1024 incident, canonical JSON length was at
   least 4,194,273 characters, the `<U{L}` payload was at least 16,777,092
   bytes, and the complete NPY member was at least 16,777,220 bytes. The
   exact deleted temporary length is unavailable. The final writer now fails
   before output creation, but B1024 remains unsupported and has no timing or
   science claim.
7. Reserve final names only for the exact stable suffixes:

| Family | Suffix | Magic | Accepted backend |
|---|---|---|---|
| Reference | .contextual-reference-v1.npz | SUMMIT_CONTEXTUAL_REFERENCE_V1 | plink_bed_descriptor_stream_stage2_v1:2 |
| Trait | .contextual-trait-v1.npz | SUMMIT_CONTEXTUAL_TRAIT_V1 | plink_bed_descriptor_stream_trait_v1:2 |
| Fit | .contextual-fit-v1.npz | SUMMIT_CONTEXTUAL_FIT_V1 | python_numpy_scipy_summary_fit_v1 |

## CPU and NUMA launch

### Qualified bounded launch shape

The measured Tabla policy is one process pinned to physical CPUs 0-7 on NUMA
node 0, with one OpenMP/decode thread and one OpenBLAS thread:

~~~bash
env PYTHONPATH=/tmp/summit-stage7-release-6920335.yygYYi/install PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=/home/bronsonj/anaconda3/envs/summit/lib LD_PRELOAD=/home/bronsonj/anaconda3/envs/summit/lib/libstdc++.so.6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 0-7 /home/bronsonj/anaconda3/envs/summit/bin/python3.12 -P /path/to/reviewed_private_driver.py --new-output /path/to/never-before-used-generation-id
~~~

No packaged production driver is part of this bounded private release. The
deployment-owned wrapper must be reviewed against the allowed call sequence,
exclusive-generation rule, sealed identities, preflight, monitoring, and
promotion checks in this runbook before use. The command above specifies its
launch envelope; it is not a public CLI and does not authorize any unreviewed
wrapper.

Before launch:

~~~bash
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
taskset -pc $$
numactl --hardware
~~~

Record the command outputs. Verify that the chosen CPU list contains distinct
physical cores on the intended node and excludes SMT siblings. Inside the
worker, require os.sched_getaffinity(0) to equal the plan CPU list exactly.

Do **not** add numactl --membind, --interleave, a second process, another
socket, or a different CPU list under the existing qualification. Stage 6
and Stage 7 used unbound first-touch allocation; the final record reports NUMA
remote bytes unavailable because `numa_maps` placement pages are not remote
traffic. A different memory policy changes the plan and requires a new clean
benchmark and acceptance record.

### Frozen benchmark reproduction only

The selected core benchmark is reproduced through the Stage 6 harness, not
through the private reference-generation driver. Run this from the root of a
clean checkout containing the report/evidence commit:

~~~bash
stage7_source=$(pwd)
env PYTHONPATH="$stage7_source" taskset -c 0-7 /home/bronsonj/anaconda3/envs/summit/bin/python3.12 "$stage7_source/scripts/context/benchmark_context_native_stage6.py" --build-spec reports/context_native/stage7/config/stage7_release_build.json --baseline-plan reports/context_native/stage7/config/stage7_core_node0_t1.json --candidate-plan reports/context_native/stage7/config/stage7_core_node0_t8.json --input-kind synthetic --retained-samples 128 --retained-variants 512 --q 4 --k 2 --groups 8 --sample-probes 128 --variant-probes 128 --traits 1 --residual-bases 2 --fixed-rank 2 --annotation-mode strict_disjoint_binary_v1 --group-layout contiguous --seed 6042026 --cpu-list 0-7 --warmups 1 --repeats 3 --worker-timeout-seconds 150 --sweep-timeout-seconds 540 --sample-interval-ms 100 --max-workspace-bytes 2147483648 --max-logical-variant-visits 50000000 --max-logical-bed-bytes 2147483648 --pilot-variant-ceiling 8192 --max-extrapolation-ratio 4 --estimate-safety-factor 2 --max-synthetic-cells 2000000 --max-input-hash-bytes 67108864 --preparation-variant-block 512 --production-qualified --work-root /tmp --output /path/to/never-before-used-stage7-benchmark-id.json
~~~

The selected T1 spec SHA-256 is
3d34606b5eef972ee65b45ce86fb5afb4efc1469eb3b701bb4b4c922c97b57f2;
the comparison-only T8 spec SHA-256 is
09c53d498b86880e5db2d7f9ba1549de5755126fc2a679cfb7ad8f415bf354bb.
The settled dry-run is
`reports/context_native/stage7/benchmark_core_node0_dry_run.json`, SHA-256
79054829c6edc3570e18de5eff002555b6d03fd67e83342da06fc43953cb4867.
The production record is
`reports/context_native/stage7/benchmark_core_node0_production_r3.json`,
SHA-256
246211f8f564dace3f34ce3ef3efac7aac9d2eb71465c745f221dcf59137b670.
Its T1 median was 4.752142690 seconds and its T8 median was 5.169534038
seconds. Both plans passed exact science and trace acceptance, but the
selector rejected T8 for lack of material end-to-end speedup and retained T1.
These existing record paths are immutable. A new authorized reproduction must
use separate globally unique dry-run and production output paths.

## Monitoring

Monitor the whole fresh worker process and any descendants without calling
executor methods concurrently.

Record at a bounded interval:

- PID/child process set and start time;
- exact CPU affinity and unexpected migrations outside the set;
- process-tree RSS, peak RSS, smaps_rollup, minor/major faults, and swap;
- user/system CPU, voluntary/involuntary context switches, and wall time;
- /proc/<pid>/io counters, while retaining the explicit mmap physical-read
  nonclaim;
- free memory, filesystem bytes/inodes, and kernel OOM messages;
- output-directory entries, ensuring no unexpected final artifact appears;
  and
- numa_maps placement-page observations only as placement observations, never
  as NUMA remote bytes.

Enforce the audited worker and sweep timeouts. A timeout is a failed run, not a
slow successful run.

After a successful run, require:

- lifecycle and terminal status published;
- exact semantic call ledger and descriptor-pass accounting;
- telemetry complete_without_drop, required capacity satisfied, and a
  terminal publication event;
- zero injection, repair, retry, fallback, and recovery counts for an
  ordinary production run;
- exact operand/runtime seals and file/input checkpoints;
- scratch released before publication;
- tracked native high-water within admission;
- expected protected trace, build/source identity, and phase digests; and
- performance_report() available and bound to the same run.

A nonzero normal-run recovery count is an incident even if science compares
within tolerance. Quarantine the output and investigate before rerun.

## Integrity or terminal failure response

Failure handling depends on the phase. Do not call a native
failure_report() merely because a later adapter, writer, parent-directory
fsync, or loader operation fails.

### Native run failure

For a native exception, telemetry overflow, fingerprint mismatch, nonfinite
result, canary/witness failure, trusted-fallback failure, file mutation,
timeout, OOM, or signal during run():

1. Stop the reference → trait → fit workflow immediately.
2. Do not invoke run() again on the terminal executor.
3. Do not call an artifact adapter or writer on partial native state.
4. Capture the executor's metadata-only failure_report(), process resources,
   stderr, build/plan/input hashes, and incident time.
5. Require the unique target path to remain absent and preserve the prior
   validated generation unchanged.

### Successful native run followed by adapter, writer, or loader failure

If native run() succeeded, the executor is not a failed executor. Preserve
its successful native telemetry/performance evidence and **do not call
failure_report()**.

- Adapter failure: publish nothing; require the unique final target to remain
  absent.
- Writer or manifest-preflight failure before the publication link: no new
  final target may exist; the writer must remove its temporary file and the
  prior generation remains unchanged.
- `EEXIST` at the publication link: the existing regular file, directory,
  symlink, or concurrent winner remains unchanged; the losing temporary file
  is removed and the generation is rejected.
- Failure after the publication link, including temporary-name unlink or
  parent-directory fsync: a complete new final file may already exist, but
  directory-entry durability is uncertain. Do not assert that publication was
  absent. Quarantine the entire unique generation, record the durability
  uncertainty, and use the official matching loader to inspect the file if it
  exists. A successful reload does not retroactively make the failed attempt
  promotable, and a retry into the same name must fail with `EEXIST`.
- Loader failure: quarantine the artifact and generation. Do not repair,
  overwrite, or resume from the rejected file.

### Common incident actions

1. Mark the never-before-used generation terminal failed and never reuse its
   path.
2. Treat controller wall time as incident metadata, not performance evidence.
3. Classify the failure before rerun: native integrity, input mutation,
   adapter, pre-replace writer, post-replace fsync durability, loader/schema,
   admission, resource exhaustion, or operator/configuration.
4. Correct the cause through review and qualification. Never weaken the
   witness, fallback, bounded loader, schema, compatibility, rank, or science
   tolerance gate to make a run pass.
5. Rerun only as a new process, new executor, and globally unique output
   generation from the beginning of the failed artifact phase.

The trusted-fallback failures observed for the rejected parallel witness and
direct-action grouped plan are fail-closed examples. They have no valid
benchmark record or resumable state.

## Resumption policy

There is no in-executor resume after a terminal failure. Native executor
states, scratch, telemetry, and transient sample-bearing probe partials are
not checkpoints.

The only allowed restart boundary is a stable artifact that:

- was fully published before the incident;
- was loaded and verified by its matching V1 loader in a fresh process;
- has the exact final release provenance and compatible upstream identities;
- has no row/sample/retained-variant leakage; and
- is listed in the operator's validated-generation inventory.

Examples:

- after a reference failure: restart reference from sealed individual-level
  inputs;
- after a trait failure: a previously validated reference may be reused, but
  trait restarts from sealed study inputs;
- after a fit failure: validated reference and trait summaries may be reloaded
  in a fresh process and fit restarted;
- after any loader failure: the rejected artifact is not a checkpoint and
  cannot be repaired in place.

## Artifact validation and promotion

For every newly written reference, trait, or fit:

1. Require a globally unique, never-before-used generation and absent final
   target before calling the writer; require the writer to return that exact
   stable-family path.
2. Open it only with the matching official V1 loader in a fresh process.
3. Require exact suffix, magic, family, logical schema, backend pair, native
   API version, grouped encoding, feature/numeric policy, exact member set,
   dtype/endian/shape, canonical manifest digest, and array digests.
4. Call the artifact verify() boundary and recompute fit invariants through
   the official loader.
5. Check source/build, file, scale, basis, annotation, group, probe,
   component/pair order, phase, telemetry, admission, and deletion identities.
6. Validate reference/trait compatibility before fit.
7. Require raw full rank and every-group leave-one-out rank. Never add a ridge
   or pseudoinverse to bypass rank failure.
8. Preserve raw coefficients, covariance, negative values, and indefinite
   surfaces. PSD output is optional, separately named, and unavailable for
   overlapping annotations.
9. Inspect manifest keys, array shapes, and serialized text for sample IDs,
   variant IDs, source paths, phenotype names tied to rows, or any N/M row
   axis. Stable arrays may have deletion-group, component, trait, residual, or
   surface axes only as declared by their schema.
10. Prove summary-only use: remove or make individual-level sources
    unavailable, then load reference/trait and produce/reload fit in a separate
    process.
11. Record artifact SHA-256, byte size, manifest SHA-256, loader result, and
    final release identity.
12. Promote the generation only after all checks pass.

### Compatibility and migration

The following is the required policy, not a claim that every migration path
has already been exercised:

- exact reference backend plink_bed_descriptor_stream_stage2_v1:2, trait
  backend plink_bed_descriptor_stream_trait_v1:2, and fit backend
  python_numpy_scipy_summary_fit_v1 may be accepted only after matching-family
  validation;
- development Stage 3/4 backend-v1 artifacts must be regenerated from trusted
  inputs and refit; no provenance-inventing converter is allowed;
- cross-family, future schema/backend, unknown key, wrong suffix/magic,
  duplicate member, object array, noncanonical digest/layout, oversized
  member/header, corrupt, or truncated archives must fail closed;
- transient probe-merge objects have no stable migration path; and
- a future schema requires a new explicit family/version and reviewed
  migration policy; it must not silently widen V1.

The final contextual suite exercises the focused Stage 7 schema-migration and
wrong-family/version rejection cases. That evidence enforces the listed V1
boundary on constructed cases; it does not demonstrate conversion of every
historical artifact, and no converter is authorized.

## Rollback

### Software rollback

1. Stop new launches and preserve the failed final candidate, logs, and hashes.
2. Select an already installed, immutable, previously audited generation by
   exact content hash. Do not rebuild a historical commit in place.
3. The bounded Stage 6 fallback lineage is f28adc0 containing the serial
   witness rollback 69fc6c1. It is not approved for the 1,024-probe family or
   large real-data generation.
4. Never roll back to parallel-witness commit 6bdb43f.
5. Re-run build_info, runtime dependency, qualification, input, preflight, and
   fresh loader checks before using the fallback.
6. Open a globally unique generation whose directory and artifact targets
   have never existed; do not append to, overwrite, or reuse any prior
   attempt.

### Artifact rollback

Keep the last validated generation immutable. Change an operator-controlled
release pointer only after the new generation passes fresh-process load and
summary-only fit. If promotion fails, point back to the prior validated
generation; do not modify, downgrade, or relabel the rejected artifact.

Rollback never converts incompatible schemas and never resumes a terminal
native state.

## Required incident and run record

Each run record must contain:

- operator, UTC start/end, host, kernel, NUMA topology, and command;
- final source/tree/build/package/runtime hashes and qualification record;
- input/file and every scientific authority identity;
- exact plan, caps, preflight, affinity, and environment;
- OS monitoring, native performance, telemetry, phase, and terminal evidence;
- artifact paths/hashes/sizes and fresh-loader verification;
- fit rank/raw/PSD/deletion result where applicable;
- recovery/failure classification and output publication status; and
- release decision or explicit reason the generation was not promoted.

## Operational authorization

This runbook is operationally authorized only under all of the following
constraints:

- exact commit, canonical source, Release install, native/package/runtime,
  qualification, and build-spec identities from the release header;
- the approved T1 plan at N=128, M=512, Q=4, K=2, eight groups, 128 sample
  probes, 128 variant probes, one trait, two residual bases, and fixed rank
  two, pinned to Tabla physical CPUs 0-7;
- bounded private use of the 28-name stable V1 module union through only the
  allowed operations and exact accepted backends listed above;
- a separately reviewed deployment wrapper that alone creates each globally
  unique generation with exclusive `os.mkdir`, then follows native preflight,
  one-shot execution, no-replace writer publication, official reload,
  monitoring, incident, and rollback controls;
- absence of any terminal native failure, recovery in an ordinary run,
  artifact validation failure, resource-cap breach, identity mismatch, or
  generation-path reuse; and
- continued exclusion of B1024, direct-action/direct-genotype grouped modes,
  unqualified NUMA policies, target-scale execution, and large-real
  generation.

The operational decision is **NARROW_GO** for this bounded private stable V1
workflow only. Public contextual CLI/configuration/schema integration is
**NO_GO/deferred**. Large-real execution is **NO_GO/deferred** because target
scale, full-process RSS admission, NUMA remote bytes, and repeated
representative real-data performance remain unqualified. Any broader use
requires a new reviewed release identity, plan, qualification, and benchmark
decision.
