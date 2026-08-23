# Contextual native estimator — Stage 7 release audit

> **Final decision:** **NARROW_GO for bounded private stable V1 use only.**
> Public contextual CLI/configuration/schema integration is **NO_GO and
> deferred**. Large real-data reference generation is **NO_GO**. This is not a
> package-wide production-readiness claim.

Evidence date: 2026-08-21.

## Decision surface

| Release surface | Decision | Evidence and boundary |
|---|---|---|
| Stable bounded private V1 use | **NARROW_GO** | Release, fatal-sanitizer, science, summary-only, leakage, migration, and unique-publication gates pass at the sealed candidate. Use is restricted to the exact 28-name stable V1 module union, pinned install, T1 plan, and runbook limits below. |
| Public contextual CLI/configuration/schema | **NO_GO; defer** | No contextual option or command is present in the 130 long options or normalized help. No public contextual configuration or migration schema is qualified. Existing legacy parser help, defaults, and source identities are frozen. |
| Large real-data reference generation | **NO_GO** | No measured N≈300,000/M≈1,000,000 execution, qualified target-scale RSS admission, verified NUMA ownership/remote traffic, repeated representative real-prefix timing, B=1,024 stable publication, or target-scale physical-I/O evidence exists. |
| Overall Stage 7 gate | **NARROW_GO, bounded private only** | The bounded private surface passes; public and large-real surfaces remain outside the release authorization. |

The common production definition of done is not satisfied. In particular,
this decision does not claim completion at N≈300,000, M≈1,000,000, B≥128,
Q≤4, public interface stability, or target-scale deployment readiness.

## Sealed candidate identity

| Field | Final value |
|---|---|
| Final implementation commit | `69203355d558792981993e2ac8fceb6851e41418` |
| Git tree | `114c707aa3d8769ce02459d280836a889f2f83df` |
| Canonical tracked-source SHA-256 | `5868faf8516eb46e0397091812fc2f6788ef387602d1a1d0be5204883b3b2c8c` |
| Tracked files / source status | 461; clean before configure and clean after all qualification |
| Release evidence root | `/tmp/summit-stage7-release-6920335.yygYYi` |
| Release install prefix | `/tmp/summit-stage7-release-6920335.yygYYi/install` |
| Release `gxeldcore` SHA-256 | `2591764e6ccedd67e0c92a8df2d4febcaa0f89967e6e2cb5615982c326f72eb4` |
| Release `gwldcore` / `winldcore` SHA-256 | `83cfad72c1fd17591afa5514e1df22173aabd416583d0aa1e0e5bdcd7210d40a` / `9a5563501b3cf803609c8d3ae406ac4ee03d58a0833889d5d3ff1ada638615e6` |
| Installed package SHA-256 | `05e999e082b4cabc4126c1d5b899c8943cb53b8a015333c71c9e8099bbd6da19` |
| Python executable / runtime SHA-256 | `59e1ed234f9cb1030d31d2db5e145e1ef409133540c78a387a2d34d269271fd8` / `d525812398709daca5133a1cd12f4f7da127433a04f17a7d978151a8df6e9ac5` |
| Dependency-runtime SHA-256 | `988388f2ca470dffaf208e8455771003149c9abbb495faaacf35036f9f252d10` |
| Canonical `build_info()` SHA-256 | `6f7cf813d1b4c8989c822a7294703c6e473e78dd421e3b81cb27bf5aae0053f3` |
| Qualification path / SHA-256 | `reports/context_native/stage7/config/stage7_fault_qualification.json` / `7ac1361c88b8209df6bc5139d945cafc5499505050ee74d30af87d1d7e4a8be0` |
| Build specification / SHA-256 | `reports/context_native/stage7/config/stage7_release_build.json` / `99a19d06b0303531ee527223693366b85b00a54a1bd6812311a54417664bc1ad` |
| Release evidence capsule / SHA-256 | `reports/context_native/stage7/stage7_release_evidence.json` / `4876731149efbc5599d4b741368d594e216603f58177af518a00726cc751de61` |
| Release evidence manifest SHA-256 | `e257791c9f7f8eb0093aa98b042475fd554c97330fb87c204d9eced10afc5557` |
| Report/evidence commit | The report/evidence commit is the Git commit containing this file; verify externally. |

The Release build used GNU 12.2.0, C++17, `Release`, `-O3`,
`-march=native`, native optimization, OpenMP, GEMM integrity, and GEMM
checksum. Sanitizers and private BLAS were disabled for this build. All three
native modules have exact RUNPATH
`$ORIGIN:/home/bronsonj/anaconda3/envs/summit/lib`; the three-module `ldd`
audit reported zero missing dependencies. Configure, build, and install took
1.173 s, 60.942 s, and 0.028 s, respectively.

The report distinguishes the immutable implementation identity above from
the commit that later records this evidence, avoiding a self-referential
report hash.

## Independent qualification matrix

| Configuration | Build and runtime identity | Test result | Diagnostic result | Evidence hashes |
|---|---|---|---|---|
| Release O3 native | Root `/tmp/summit-stage7-release-6920335.yygYYi`; CPUs 32–63; native/package hashes as above | Full repository: **1,588 passed, 5 skipped, 1 XPASS in 140.48 s**. Explicit `tests/test_context_*.py`: **816 passed, 0 skipped, 0 XPASS in 84.67 s** | Clean source pre/post; import, RUNPATH, `ldd`, package stability, and identity gates pass | Full log `c884ec0a023b5e2b67e272ece9f8f5aee522c45a76cc8dc55c21e3a4588f3a2d`; context log `35cb3945731c1a26f8846122b3d4a69dc596a14740f3d1e208c74f78fc874f65`; manifest `e257791c9f7f8eb0093aa98b042475fd554c97330fb87c204d9eced10afc5557` |
| ASan+UBSan | Root `/tmp/summit-stage7-asan-6920335.0JVW73`; portable `-O1`; `gxeldcore` `50ce3e882ee9ae5c608fef1abf4fb1a55dd2927db52e94f642f9a862f9d482ef`; package `fd562c654fe1d267993b785d332128faacb892f65b03863baf46a146e55dee39` | Required 33 contextual + 7 legacy files: **904 passed, 0 skipped in 141.83 s** | Fatal settings pass; diagnostic scan found zero sanitizer matches; installed package remained stable | Test log `5b02a9f77172d183edd985303f203118d3c3d67af47cdf668715ef99241a3efa`; manifest `25d3d2e13718d107f133f7da59030a2f4f47d8c73e0581253450475ba1a1e97d` |
| UBSan-only | Root `/tmp/summit-stage7-definitive-ubsan-6920335.O7JawY`; portable `-O1`; `gxeldcore` `13e28e8136de3d91900e80d411b101b366d091206955b2a559b05ce110f8cdd5`; package `4f2efb2673c46716ffb2b0fcdde7b64440cf3f03fdc05a526814c0a080ed7d5d` | Same required 33 contextual + 7 legacy files: **904 passed, 0 skipped in 101.05 s** | Fatal undefined-behavior scan clean; pre/post runtime identities equal; RUNPATH exact and zero missing dependencies | Test log `ca34e0b709748225ad21ad84ba11b0c7d3d6d074fc0b268fb61c3e054957e458`; manifest `749986c729ee31dedccdef8beb8a0a16d923275ee91db11c8d42309934bcb2ed` |

ASan+UBSan used exactly
`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, with GCC 12 `libasan`
before the environment `libstdc++.so.6` in `LD_PRELOAD`. UBSan-only used
exactly `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Because
`detect_leaks=0`, these results make no LeakSanitizer or leak-freedom claim.
The sanitizer matrix is the stated 904-test required scope, not a claim of
full-repository sanitizer execution.

The five Release skips were one unavailable preserved private OpenBLAS audit
archive, two non-BLIS-candidate cases, one unavailable two-socket integration
case, and one intentionally retired block-jackknife artifact. The sole XPASS
is the named API8 source-TT rejected-diagnostic test; it is not accepted as a
production primitive.

## Required Stage 7 audit

| # | Required audit | Result | Final evidence |
|---:|---|---|---|
| 1 | Frozen fixtures and native/Python differential | **PASS** | Stage 0–4 fixture, dense, streamed, complete-reference, trait, fit, deletion, randomized, and surface tests all pass inside the 816-test Release context suite and both 904-test sanitizer scopes. |
| 2 | Contextual, legacy, integrity, sanitizer, schema, and fresh-process suites | **PASS** | Release 1,588/816 aggregates and independent 904/904 sanitizer partitions above; no sanitizer diagnostic. |
| 3 | Selected Tabla benchmark against the frozen baseline | **PASS for bounded measurement; no optimization claim** | Dry-run SHA-256 `79054829c6edc3570e18de5eff002555b6d03fd67e83342da06fc43953cb4867`; production SHA-256 `246211f8f564dace3f34ce3ef3efac7aac9d2eb71465c745f221dcf59137b670`; detailed result below. |
| 4 | Artifact provenance and telemetry | **PASS for bounded V1** | Reference/trait/fit write-load-fit round trips, source/build/scale/basis/annotation/group/probe provenance, admission, phase, and integrity telemetry pass in Release and benchmark records. NUMA and physical-I/O nonclaims remain explicit. |
| 5 | Scientific documentation | **PASS** | Assertions in the scientific review below agree with the executable Stage 0/3/4 contracts and pass in the final 816-test context suite. |
| 6 | Row-leakage and family/version isolation | **PASS** | Nine parameterized Stage 7 leakage/public-surface cases reject undeclared raw grafts and inspect compatible sentinel-bearing reference, trait, fit, controller, publication, in-memory, and NPZ evidence. Stable layouts have no row/sample/variant axis; matching loaders reload; wrong-family/version paths remain closed. |
| 7 | Failure, rollback, and publication | **PASS for bounded V1** | Ten all-family Stage 7 writer tests cover unique generation, no-replace publication, file/symlink `EEXIST`, race loss, manifest/member/total/classic-ZIP bounds, temp validation, cleanup, and pre/postcommit fsync semantics. Stage 5 terminal-state/fault gates remain green in the final suites. |
| 8 | Migration and unsupported schema/backend rejection | **PASS** | Two Stage 7 migration tests cover an admitted canonical rewrite control and all-family rejection before scientific-array preflight; the broader Stage 5 compatibility/corruption matrix also passes. Unsupported migration is fail-closed. |
| 9 | Public names, help, defaults, and legacy behavior | **PASS for preservation; public context remains NO_GO** | Exact source/help/default/export identities below pass in Release. No contextual CLI exists. |
| 10 | Operational runbook and release operation | **PASS for bounded private operation** | `reports/context_native/stage7_operational_runbook.md`, the sealed configs, dry run, production reproduction, rollback exclusions, and unique-generation rules were reviewed against the final evidence. |

## Stable artifacts, leakage, and publication

### Admitted private surface

The stable private boundary is exactly the 28-name union of
`reference_v1.__all__`, `trait_v1.__all__`, and `fit_v1.__all__`. Its sorted
JSON SHA-256 is
`fc1bfac9c6b0a6b3a9ffac73cd0237ae9e989d89f87201edb4dbebb06a3cf6b6`:

`CONTEXTUAL_FIT_V1_MAGIC`, `CONTEXTUAL_FIT_V1_SUFFIX`,
`CONTEXTUAL_REFERENCE_V1_MAGIC`, `CONTEXTUAL_REFERENCE_V1_SUFFIX`,
`CONTEXTUAL_TRAIT_V1_MAGIC`, `CONTEXTUAL_TRAIT_V1_SUFFIX`,
`ContextualFitArtifactV1`, `ContextualReferenceArtifactV1`,
`ContextualReferencePublicationIdentityV1`, `ContextualTraitArtifactV1`,
`ContextualTraitMomentsV1`, `ContextualTraitPublicationIdentityV1`,
`adapt_native_contextual_reference_v1`, `adapt_native_contextual_trait_v1`,
`assemble_contextual_normal_equations_v1`,
`contextual_variant_order_allele_sha256_v1`, `fit_contextual_model_v1`,
`load_contextual_fit_v1`, `load_contextual_reference_v1`,
`load_contextual_trait_v1`, `reference_moments_after_deleting_groups_v1`,
`run_contextual_reference_v1`, `run_contextual_trait_v1`,
`trait_moments_after_deleting_groups_v1`,
`validate_contextual_fit_compatibility_v1`, `write_contextual_fit_v1`,
`write_contextual_reference_v1`, and `write_contextual_trait_v1`.

The 253 total names in `summit.context.__all__` are an inventory, not an
approved stable surface. The 28-name set is a subset, and the context export
list is unique. Native executor constructors remain private and one-shot;
native adapters are not general ingestion APIs; the benchmark harness is not
a production CLI.

| Family | Exact suffix | Magic | Accepted boundary |
|---|---|---|---|
| Reference V1 | `.contextual-reference-v1.npz` | `SUMMIT_CONTEXTUAL_REFERENCE_V1` | Matching reference V1 loader; native adaptation only through the admitted strict adapter |
| Trait V1 | `.contextual-trait-v1.npz` | `SUMMIT_CONTEXTUAL_TRAIT_V1` | Matching trait V1 loader; native adaptation only through the admitted strict adapter |
| Fit V1 | `.contextual-fit-v1.npz` | `SUMMIT_CONTEXTUAL_FIT_V1` | Matching fit V1 loader after exact reference/trait compatibility validation |

The leakage gate does more than compare serialized hashes. It injects sample,
variant, source-path, and phenotype sentinels; rejects raw top-level and file
grafts; constructs compatible sentinel-bound summary inputs; inspects native
terminal/controller evidence, publication identities, artifact manifests,
in-memory arrays, and every NPZ member; rejects stable row/sample/variant
axes; and reloads each family with its official loader. The existing true
native two-process test builds and writes reference/trait artifacts, tears
down the builder and removes the individual-level source, then fits and
reloads only stable summaries in a separately spawned process. That test
passes in the final 816-test Release suite and both sanitizer scopes.

### Unique publication and bounded archives

The shared writer boundary writes, fsyncs, and validates a same-directory
temporary NPZ, atomically links it to a previously unused final target, then
removes the temporary and fsyncs the parent. Publication cannot replace an
existing regular file or dangling symlink. A concurrent winner is preserved;
the losing writer receives `FileExistsError` with `errno.EEXIST`; temporary
files are cleaned. Operational generation creation uses exclusive
`os.mkdir`, and a second creation fails with `EEXIST` before the three
distinct family targets are published.

Precommit file-fsync or validation failure leaves a fresh target absent and
an existing target byte/inode-identical. If parent-directory fsync fails
after publication, the complete final remains officially loadable but its
directory durability is uncertain. A retry fails `EEXIST` and preserves its
bytes and inode. Operators must quarantine that generation rather than
promote or overwrite it.

All three writers preflight the exact 16 MiB manifest-NPY boundary, exact
member and total uncompressed NPY sizes, classic-ZIP member count, compressed
size, and central-directory offset limits before creating output. Exact
boundary artifacts reload; over-bound cases create no parent, temporary, or
final output and preserve any old target. These controls close publication of
known-unloadable archives without raising or bypassing the loader caps.
B=1,024 itself remains unavailable and unqualified; this writer closure is
not evidence of B=1,024 production support.

## Legacy CLI and public-surface audit

| Evidence | Observed value |
|---|---|
| Installed `python -m summit --help` SHA-256 | `3339d0a93cdd174e50a16794c4175a15749b27786beb8573addc2581b908aca1` |
| Normalized parser help SHA-256 | `01fd04cb2b640415372cce40ecde77838faea33add75e7f07eb0baa160a76f7c` |
| Normalized defaults SHA-256 | `29b7813d8dc5e8be8778405addd0a10cb0111eac7f79426846687bec06229210` |
| Parser actions / long options | 131 / 130 |
| Contextual option or help entry | None |
| `src/summit/cli.py` SHA-256 | `a469ba8f73594f5832b19ab6e249958850d6019f205b363f4ecb2611d6dfe3ca` |
| `src/summit/__init__.py` SHA-256 | `448f5770e516c134d875bdd95627aca81eb6520c9026681f959eea903e3e0153` |
| `src/summit/__main__.py` SHA-256 | `d7815f9da17bcd8e79d1d47170e5795e29cc5837f97d1049664813068a9e563f` |
| `pyproject.toml` SHA-256 | `16b39004a78c37c7d922bc281e7fcaec37e19522bfd6ab67d583b513de5434bc` |

These hashes establish preservation of the tested legacy parser and source
surface; they do not create or authorize a contextual public interface.
Legacy reference-backend V1 and trait-backend V1 artifacts are regenerated
from trusted inputs and refit, never relabeled or filled with invented
provenance. Wrong-family, unknown future version/backend, extra/missing key,
duplicate member, object/wrong-endian/wrong-shape array, corrupt CRC,
truncated, oversized, and transient sample-bearing inputs fail closed. Any
future public contextual schema requires a new explicit version boundary.

## Scientific review

| Topic | Reviewed assertion | Result |
|---|---|---|
| Common scale and transfer | One sealed affine SNP transform is shared across contexts. `F_q=P D_q G`, never `D_q P G` or `P D_q(PG)`, and there is no post-projection coordinate/SNP rescaling. Transfer uses study sample count, not residual rank. | **PASS** |
| Component order and factors | Pairs are diagonal then lexicographic off-diagonal; components are annotation-major/pair-minor with `C=KQ(Q+1)/2`; eta is 1 for diagonals and 2 for off-diagonals, with no extra packing factor. | **PASS** |
| Same-person statistic | The output is the shared-variant-probe global signed U-statistic with same-probe subtraction and denominator `B_D(B_D-1)`, never tile/process-local finalization, clipping, or PSD projection. | **PASS** |
| Deletions | Unnormalized group numerators and annotation masses sum to full Gram/RHS/trace/genetic-residual units; the selected group is subtracted and renormalized; full `D_R` is reused; deletion is explicitly approximate summary-only. | **PASS** |
| Rank | The full and every-group leave-one-out raw system must be full rank; failure is closed without ridge or pseudoinverse. | **PASS** |
| Raw versus PSD | Raw coefficients and raw Omega/surfaces are primary and may be negative or indefinite. `raw_jackknife_covariance` is separately retained and must pass its variance/PSD gate. Optional genetic PSD is separately named. | **PASS** |
| Overlap | Per-annotation outputs under generic overlap are conditional contributions; the combined total is primary, and genetic PSD interpretation is unavailable. | **PASS** |
| Trait traversal | Exactly one descriptor traversal occurs per admitted phenotype batch while each genotype block is resident, producing all `Q x L` scores/reductions and only compact RHS/traces/residual moments. | **PASS** |
| Measurement semantics | Native admission is not process RSS; logical mmap bytes are not physical reads; placement pages are not remote bytes. | **PASS** |

These assertions are supported by the Stage 0 contract, Stage 3 complete
reference/deletion tests, Stage 4 trait/fit/rank tests, and their successful
re-execution in the final Release and sanitizer suites.

## Final bounded Tabla reproduction

The final campaign used the exact restricted T1/T8 geometry, CPU list 0–7,
one warmup, three measured fresh workers, unique output paths, production
qualification, and the sealed build/qualification records. The dry-run
record SHA-256 is
`79054829c6edc3570e18de5eff002555b6d03fd67e83342da06fc43953cb4867`;
the production record SHA-256 is
`246211f8f564dace3f34ce3ef3efac7aac9d2eb71465c745f221dcf59137b670`.
Both production plan records terminate as accepted and production-qualified.

| Plan | Fresh-worker wall replicates (s) | Median (s) | Native / integrity overhead (s) | Whole RSS / admitted native bytes | Record SHA-256 | Selection |
|---|---|---:|---:|---:|---|---|
| T1, BLAS threads 1 | 4.750784788, 4.752142690, 4.787291578 | 4.752142690 | 1.014171892 / 0.761602139 | 162,144,256 / 14,783,488 | `bdc18ee15fe8628d2cbf02bc93179daa25b51ff9a737f0c76608e97e2713660d` | **Selected** |
| T8, BLAS threads 8 | 5.240278725, 4.998297064, 5.169534038 | 5.169534038 | 1.170807002 / 0.852754823 | 162,058,240 / 14,783,488 | `d4e0a28a5ba5224a5beb70c167dc76dc65e4b88e281fd3839feaedb50c5c6967` | Rejected only for insufficient material end-to-end speedup |

T8/T1 selection speedup is 0.9192593868360559. T1 is 2.40425% lower in
wall time than the frozen Stage 6 T1 median of 4.869210758 s
(`4.869210758 / 4.752142690 = 1.0246348`), but it does not meet the 1.05
optimization minimum and is not presented as an optimization win.

Every one of the six measured replicates has exact fixed-probe science,
deletion, scientific-array, rank, and protected-trace comparisons. Shared
identities are protected trace
`8a687648779ed9b976768bfa8f30b0eb3c27a11f8b22085abc9090e2f9e845dd`,
science
`7ef8ad8d19cbcf35a5e4ed2bed48da8bed2c9b9f5fa95173b7eda38ff53c918d`,
and rank
`693df1eb35dad74dca10ae46b3a8abb6e84fa79b2c61f7b8a8fdee3af7c93a6d`.
Retry, fallback, and recovery counts are zero. Logical descriptor passes are
action/group/same-person/source/trait = 1/1/8/1/1, and physical record visits
are 6,144 for both plans.

### Frozen comparison basis

| Stage 6 campaign | Campaign SHA-256 | Selected T1 SHA-256 | Median |
|---|---|---|---:|
| N128/M512 restricted core T1/T8 | `fa6041cb11f89b42279458b20d86502a9b92c58758e1b0d1695e9f0c94720680` | `e5c8125357f8b5157500a6b306cc99dea93c130cbd63df89023a3a78078176d3` | 4.869210758 s |
| N512/M2048 restricted T1/T8 | `16ce455eb58954117026840b39589ae9eeb81fdc592daa68b7cce67a022699db` | `76e1a9d0a1588fa6a6750c2c47253b91ab53488a87edc3635bb7bb35aa25bb2b` | 4.920812291 s |

The rejected parallel-witness commit
`6bdb43fca41ed792205f89acd0115d065c2425be` is not a comparison candidate.
The one-repeat example/small real-prefix pilot, SHA-256
`942e3bae34ae10313bc040c1abdca9a8ebe8303293c58a645faccf3a30ba04f9`,
remains screening evidence only. The modeled target record at
`reports/context_native/benchmarks/stage6/estimates/tabla_target_modeled.json`
is not a measurement or production admission.

## Rollback basis and rejected/unmeasured ledger

The immutable Stage 6 rollback basis remains:

- serial-witness rollback commit
  `69fc6c19d0b5b4cdc762c0ae92ad91a4cae61861`;
- source-tree SHA-256
  `aa63e704eb31954e70fabe09bbae7337e1bdb1653ba98de8d57be15f09bbb956`;
- Release native SHA-256
  `7a52639ba60e5e486f4cf60cecd7275cb77a1f49a04f660b00865324058ec8f7`;
- installed package SHA-256
  `a2c3614195f0ed96d82b4f7c0a79c01d98c3ffc99b38f6d1b518ebc1afc616b4`;
- qualification SHA-256
  `e170992b27d849f16b55e0e0744d8dc22c73d247ce725ddb91940b233cbaad5a`;
  and
- immutable native-build provenance SHA-256
  `69209d0528e361346e84f00333bf19b8422d0bdb65e5929d42012aea6434279d`.

Rollback selects only an immutable audited install by hash and never selects
the rejected parallel-witness commit. The last validated artifact generation
is preserved; terminal native failure requires a fresh executor and a unique
new generation.

Known alternatives remain rejected or unmeasured:

- T8 fails the material speedup minimum in the accepted restricted campaigns.
- Resident 16 and 32 fail rank/trace identity; resident 64 also fails
  fixed-probe science at the screened geometry.
- Direct-action T1 fails trusted-fallback verification; direct genotype is
  unmeasured.
- B=1,024 stable publication remains unavailable and unqualified.
- Generic/indexed packing, independent blocks, packed or Mailman comparison,
  a broader thread ladder, sockets/processes, bound NUMA, target-scale ladder,
  physical read bytes, remote bytes, and mixed precision are unmeasured.

No rejected or unmeasured option may be selected implicitly by a default,
cache, migration, or interface.

## Large-real NO_GO and explicit nonclaims

Large-real generation remains **NO_GO** because the following evidence does
not exist:

1. measured N≈300,000/M≈1,000,000 execution;
2. a qualified full-process RSS admission model and target-scale enforcement;
3. verified NUMA page ownership or remote-byte counters;
4. a qualified 32/64-core or two-process topology;
5. repeated representative real-prefix timing;
6. an admitted direct-action or measured direct-genotype route;
7. a measured B=1,024 transition and stable publication;
8. physical mmap read bytes and target I/O throughput; or
9. mixed-precision qualification under a separate version.

The final evidence therefore makes no claim about target-scale execution,
full-process memory admission, NUMA placement or remote traffic, physical
mmap reads, target-scale I/O, B=1,024 publication, public contextual CLI or
configuration schema, mixed precision, or LeakSanitizer freedom. Native
admitted bytes are not whole-process RSS, logical mmap bytes are not physical
reads, and observed placement pages are not remote bytes.

## Evidence inventory and sign-off

| Evidence | Final role |
|---|---|
| `reports/context_native/stage0_contract_freeze.md` through `stage4_trait_and_fit.md` | Frozen scientific contract and differential/summary-fit basis, re-executed by the final suites |
| `reports/context_native/stage5_integrity_hardening.md` | Integrity, stable schemas, two-process summary-only fit, compatibility, and failure-state basis, re-executed by the final suites |
| `reports/context_native/stage6_tabla_performance.md` (SHA-256 `d3456c3c33136ad2d0b1c62e06fd26479aa8c82c662d359540efc51d95680dae`) | Frozen T1 comparison, rollback, selection rules, and performance nonclaims |
| `reports/context_native/stage7_operational_runbook.md` | Bounded private operating, incident, unique-generation, and rollback control |
| `reports/context_native/stage7/stage7_release_evidence.json` | Machine-readable final identity, tests, sanitizers, benchmark, and decisions |
| Release root `/tmp/summit-stage7-release-6920335.yygYYi` | Clean Release build, install, full/context tests, CLI surface, dynamic linking, and log manifest |
| ASan+UBSan root `/tmp/summit-stage7-asan-6920335.0JVW73` | Fatal address/undefined-behavior required-scope qualification with leak nonclaim |
| UBSan root `/tmp/summit-stage7-definitive-ubsan-6920335.O7JawY` | Independent fatal undefined-behavior required-scope qualification |
| Dry and production benchmark JSON named above | Accepted bounded records, exact identities, measurement, and selector decision |

| Role | Reviewer/owner | Commit reviewed | Decision | Date/evidence |
|---|---|---|---|---|
| Scientific/documentation audit | Independent scientific audit agent | `69203355d558792981993e2ac8fceb6851e41418` | **PASS** for bounded science; retain interpretation nonclaims | 2026-08-21; final Release/context and sanitizer matrices |
| Native/integrity audit | Independent sanitizer/native audit agent | `69203355d558792981993e2ac8fceb6851e41418` | **PASS** for the stated Release and 904-test sanitizer scopes | 2026-08-21; three qualification roots and hashed logs |
| Artifact/schema/privacy audit | Independent artifact/privacy audit agents | `69203355d558792981993e2ac8fceb6851e41418` | **PASS** for bounded stable V1 publication, migration, and row-leakage gates | 2026-08-21; writer/leakage/migration tests in the 816 suite |
| Performance/operations audit | Independent performance/operations audit agents | `69203355d558792981993e2ac8fceb6851e41418` | **PASS** for restricted T1 measurement; large-real **NO_GO** | 2026-08-21; dry/production records and runbook |
| Release decision owner | Root Stage 7 release owner | `69203355d558792981993e2ac8fceb6851e41418` | **NARROW_GO bounded private only** | 2026-08-21; complete evidence inventory above |

## Final release decision

**NARROW_GO** is granted only for bounded private use of the exact stable V1
reference, trait, compatibility, fit, writer, and loader boundary through a
pinned audited install and the selected restricted T1 plan. Unique generation,
no-replace publication, strict loaders, and the operational runbook are
mandatory.

**NO_GO; defer** applies to any public contextual CLI, configuration, or
schema promise. **NO_GO** applies to large real-data reference generation.
Stage 6 rejected/unmeasured alternatives and all NUMA, physical-I/O,
target-scale, B=1,024, mixed-precision, full-RSS-admission, and leak-freedom
nonclaims remain in force.
