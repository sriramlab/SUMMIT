# Contextual native estimator — Stage 6 tabla performance report

> **Final Stage 6 gate record.** The parallel scalar-witness candidate at
> 6bdb43fca41ed792205f89acd0115d065c2425be is rejected. It reproducibly
> failed trusted-fallback verification in the restricted eight-thread plan at
> N=512, M=2048. Commit
> 69fc6c19d0b5b4cdc762c0ae92ad91a4cae61861 restores the serial witness,
> passed clean Release and sanitizer qualification, and passed both bounded
> production comparisons. No performance optimization was selected: the
> one-thread restricted baseline remains the retained policy.

## Decision

**Stage 6 decision: NARROW_GO to Stage 7. This is not a full performance GO
and does not authorize production or public-release performance claims.**

The rollback build is scientifically and operationally qualified for the
bounded evidence reported here. At N=128/M=512 and at N=512/M=2048, both the
one-thread and eight-thread restricted plans were accepted with exact arrays,
deletions, rank, and protected trace. The selector nevertheless retained T1:
T8 was slower in both three-replicate production comparisons and failed the
required 1.05x end-to-end speedup gate.

The handoff is narrow because required scale and implementation axes remain
unmeasured or rejected:

- the unsafe parallel-witness source is rejected and must not be restored,
  selected, cached, or promoted;
- resident counts below 128 were rejected in the one-repeat screen;
- direct-action grouped execution fails trusted-fallback verification, and
  direct-genotype execution was not run;
- the 1,024-probe artifact fails the existing 16 MiB bounded-manifest policy;
- the real-prefix pilot has only one replicate and is not production
  qualified;
- target-scale runtime, full-process admission, socket/process scaling,
  generic packing, alternate backends, and NUMA remote bytes are absent.

Stage 7 may audit the retained FP64 T1 policy, the rollback, the measurement
chain, and these explicit nonclaims. It must not turn modeled admission into a
tabla-scale measurement or interpret this NARROW_GO as a release gate.

## Repository, source, and build identity

- Branch: codex/contextual-stage6-performance-scale.
- Stage 6 foundation:
  e8f6331fe26780d5566b0983679edb1ebe4575ed.
- Rejected parallel-witness source:
  6bdb43fca41ed792205f89acd0115d065c2425be.
- Immutable-provenance harness repair:
  60a494cc6429543839640db200db4a582d071107.
- Qualified rollback and final measured source:
  69fc6c19d0b5b4cdc762c0ae92ad91a4cae61861.
- Rollback source-tree SHA-256:
  aa63e704eb31954e70fabe09bbae7337e1bdb1653ba98de8d57be15f09bbb956.
- Harness SHA-256:
  58c7b945ef263c16ebb6d227c8d1212e2d52413326bba0de12850090bfe4d57c.

The selected Release build is bound by
reports/context_native/benchmarks/stage6/config/rollback_candidate_69fc6c1_build.json
(SHA-256
766af756e6327b6b7ed0a19c423d7c3439e9b285b5262fd1c4fb5c81ce146187):

| Field | Exact value |
|---|---|
| Build label | rollback-69fc6c1 |
| Install prefix | /tmp/summit-stage6-serial-rollback-69fc6c1.Wndsw1/install |
| Release native module SHA-256 | 7a52639ba60e5e486f4cf60cecd7275cb77a1f49a04f660b00865324058ec8f7 |
| Installed package SHA-256 | a2c3614195f0ed96d82b4f7c0a79c01d98c3ffc99b38f6d1b518ebc1afc616b4 |
| Python runtime SHA-256 | d525812398709daca5133a1cd12f4f7da127433a04f17a7d978151a8df6e9ac5 |
| Dependency runtime SHA-256 | 988388f2ca470dffaf208e8455771003149c9abbb495faaacf35036f9f252d10 |
| Immutable native-build provenance SHA-256 | 69209d0528e361346e84f00333bf19b8422d0bdb65e5929d42012aea6434279d |

The measured Release build reports GNU C++ 12.2.0, C++17, Release -O3,
-march=native, OpenBLAS 0.3.34
(DYNAMIC_ARCH, NO_AFFINITY, Zen, MAX_THREADS=128), OpenMP enabled, FP64,
protected deterministic tiled GEMM, and the full scalar witness. Runtime BLAS
thread count is not part of immutable build identity after the 60a494c repair;
the allowlisted compile metadata, native-module hash, and installed-package
hash are.

Historical cross-build identities are retained for provenance only:

| Build | Source/tree | Native SHA-256 | Package SHA-256 | Immutable provenance SHA-256 |
|---|---|---|---|---|
| Foundation | e8f6331fe26780d5566b0983679edb1ebe4575ed / c659a2d27b181acb128b823e0ee1c69ab911ca538422f4fffd91d90854204b35 | 9b3ec6cd895ccc254449a3bb6ee5f35d950d934a6bdf7f424ba3374923496b0a | 2e974d329075a6d1ba648e17310b5456e7c94b852e63864d3fe0fcc0d167b20e | 8b2c8494417e2cfec4f03fba5ce3398cca405266c9b6d0b63a67229aacdcbddb |
| Rejected candidate | 6bdb43fca41ed792205f89acd0115d065c2425be / e1c44a2ab0077c3080ea36bf5b1ebca4e5caa028a3f3fb4694b8fcdbc12c114c | 5a9ad681955ae5fe5dad4168b009787b06a0153bd86f3063124cdf4d4c5beca8 | 755a3f4315a0869d16092521700cdff2100462bd18c1a9160c045bc1f1ef9f20 | cdb3f1fbd5dd15c3ce93e33e0def168440b23d4d9595673b98b60135a5dd04f4 |

The historical candidate's small accepted timing record does not make that
source selectable after its larger trusted-fallback failure.

## Host and measurement boundary

Measurements ran on Tabla: Linux 4.19.0-21-amd64, AMD EPYC 7501, 64
physical/128 logical CPUs, two sockets, eight NUMA nodes, and
1,082,107,719,680 bytes of memory. CPUs 0-7 are eight physical cores on NUMA
node 0; SMT siblings 64-71 were not admitted.

Stage 6 added or exercised:

1. a closed native performance report for every reference and trait phase,
   all 17 protected operations, exact shape histograms, logical FLOPs,
   descriptor accounting, admission, and explicit integrity overhead;
2. checked FP64 dimension/plan/admission schemas, 128-bit size arithmetic,
   modeled target estimates, a strict cache key, and a fail-closed selector;
3. a bounded fresh-process harness with full input hashes, immutable build and
   package hashes, exact science/trace capsules, CPU affinity, whole-worker
   monitoring, work/time caps, warmups, and replicates;
4. a candidate that parallelized independent scalar-witness output entries,
   later rejected at the broader geometry; and
5. the immutable-provenance repair followed by a serial-witness rollback.

The performance code remains measurement and selection infrastructure. It did
not promote a new execution default.

## Rollback qualification

The content-bound qualification record is
reports/context_native/benchmarks/stage6/config/rollback_candidate_69fc6c1_fault_qualification.json
(SHA-256
e170992b27d849f16b55e0e0744d8dc22c73d247ce725ddb91940b233cbaad5a).
It binds the exact rollback commit/tree and Release native module above, all
17 operations, the R7 recoverable modes, the T4 terminal modes, 57 focused
tests, 1,564 full-regression passes, and passing ASan+UBSan and UBSan gates.

The full Release command was:

~~~bash
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q
~~~

Its settled result was 1,564 passed, 6 skipped, with 1 XPASS recorded
separately. The focused scope was the same four files for both sanitizer
builds:

~~~text
tests/test_context_stage5_build_integrity.py
tests/test_context_stage5_native_integrity.py
tests/test_context_stage6_performance_instrumentation.py
tests/test_context_stage6_parallel_witness.py
~~~

### Fresh ASan+UBSan build

- source: /tmp/summit-stage6-rollback-asan-src.h1wZic;
- build: /tmp/summit-stage6-rollback-asan-build.qsmIV6;
- install: /tmp/summit-stage6-rollback-asan-install.SKg6fu;
- logs: /tmp/summit-stage6-rollback-asan-logs.qtPhF7;
- native SHA-256:
  3fc031ee8c46083d8e6d41dbc3714286afd4337b0b727fc69f6e89eef29dd859;
- installed package SHA-256:
  7f218300a425a64b06b6f05a91c86af8b5a9e8ae51b6e970955544f9f0ca01bf;
- build_info: GNU 12.2, Release, asan_ubsan, effective -O1, portable
  architecture, native optimization disabled, OpenMP enabled, OpenBLAS
  0.3.34;
- flags: -O3 -DNDEBUG -O1 -g -fno-omit-frame-pointer
  -fno-sanitize-recover=all -fsanitize=address,undefined;
- result: 57 passed in 12.13 s, exit 0, with no ASan or UBSan diagnostic;
- pytest log SHA-256:
  8fc284621dfabb4287cc0b6c811966c364538990f6e090f19470f6383583d5a5;
- build_info log SHA-256:
  9457ebcd0ae65e8aa6724d937d33f20dc7f1fca791e69eb6aae4b066d4a208e8.

The fatal runtime settings were
ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0 and
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1, with GCC 12 libasan and the
Conda libstdc++ preloaded. LeakSanitizer is explicitly **not claimed**:
detect_leaks=0 isolates the documented CPython baseline leak
(3,008 bytes in three allocations at a clean import).

### Fresh UBSan-only build

- detached source/build/install:
  /tmp/summit-stage6-rollback-ubsan-69fc6c1.im5vj6/{src,build-qualified,install-qualified};
- installed native SHA-256:
  1fefe58d6aad0dc2f9db57e4c8f6297e37b8c2fbefb6acf8cdb12f9f4392f926;
- build_info: GNU 12.2, Release, ubsan_only, ASan disabled, UBSan enabled,
  effective -O1, portable architecture, native optimization disabled;
- flags: -O3 -DNDEBUG -O1 -g -fno-omit-frame-pointer
  -fno-sanitize-recover=all -fsanitize=undefined;
- configuration: -DCMAKE_BUILD_TYPE=Release and
  -DGWLDCORE_ENABLE_UBSAN_ONLY=ON, Ninja, four build jobs;
- fatal setting:
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1;
- result: 57 passed in 10.02 s, exit 0, with no UBSan diagnostic;
- basetemp:
  /tmp/summit-stage6-rollback-ubsan-pytest.tWz9zw/focused.

These sanitizer artifacts have different native hashes because their compiler
flags differ from Release. They qualify their exact builds and test scope;
they are not timing artifacts.

## Bounded production plan

Both production campaigns used strict-disjoint binary annotations, contiguous
groups, seed 6042026, protected dense FP64 GEMM, the full scalar witness,
deterministic reductions, allocation reuse, one process, CPUs 0-7, one
warmup, and three fresh measured workers. T1 and T8 differ only in BLAS
threads (one versus eight); decode threads remain one.

The shared source, target, group, same-person, and trait variant blocks are
128. The campaign label “B16” refers to 16 sample probes, 16 variant probes,
and 16 resident sample probes; it does **not** mean 16-variant execution
blocks.

### N=512, M=2048 restricted production comparison

Machine record:
reports/context_native/benchmarks/stage6/screen/rollback_restricted_t1_t8_n512_m2048_b16_production_r3.json,
SHA-256
16ce455eb58954117026840b39589ae9eeb81fdc592daa68b7cce67a022699db.

| Metric | T1 retained baseline | T8 rejected candidate |
|---|---:|---:|
| Median fresh-worker E2E s | 4.920812291 | 5.079565637 |
| Replicate E2E s | 4.984766498, 4.648796166, 4.920812291 | 5.079565637, 5.141896185, 5.035676899 |
| Artifact write/load/fit s | 2.514860828 | 2.546352194 |
| Integrity overhead s | 1.661866011 | 1.854381536 |
| Whole-process peak RSS bytes | 145,080,320 | 145,768,448 |
| Native admitted payload bytes | 16,322,368 | 16,322,368 |
| Record SHA-256 | 76e1a9d0a1588fa6a6750c2c47253b91ab53488a87edc3635bb7bb35aa25bb2b | 46ec46b2d2e4fbb1c872077085755bdfe00ca41944d4341aaa90c29f233e4a05 |

Both records are terminal accepted and production qualified. They share
science identity
5a59c6dfafe4c33e473e88cfe48a82754ad4f6b43e14778937fbb6da561a1139,
rank identity
76dac5c439e4b1709f0804436aae291e0aee941fd4af78c64605a5918c2d93a3,
integrity identity
635622feba54571b96b11c17d2103f63b2d4adc0746b7ba5351e451cc526d935,
and exact combined trace
e28033d47f3bf7185277c778e86dd13a74c50206ff508fe76fd35d614bfddc46.
Every fixed-probe discrepancy is zero. The recorded speedup is
0.968746669037284; T8 is rejected for
material_end_to_end_speedup_not_met and T1 is selected.

### N=128, M=512 core production comparison

Machine record:
reports/context_native/benchmarks/stage6/screen/rollback_core_node0_t1_t8_n128_m512_b128_production_r3.json,
SHA-256
fa6041cb11f89b42279458b20d86502a9b92c58758e1b0d1695e9f0c94720680.

| Metric | T1 retained baseline | T8 rejected candidate |
|---|---:|---:|
| Median fresh-worker E2E s | 4.869210758 | 5.055546944 |
| Replicate E2E s | 4.869210758, 4.856138079, 4.896713629 | 5.055546944, 5.079560123, 5.034039049 |
| Artifact write/load/fit s | 3.503330724 | 3.469565057 |
| Integrity overhead s | 0.768468230 | 0.884370120 |
| Whole-process peak RSS bytes | 156,807,168 | 155,881,472 |
| Native admitted payload bytes | 14,783,488 | 14,783,488 |
| Record SHA-256 | e5c8125357f8b5157500a6b306cc99dea93c130cbd63df89023a3a78078176d3 | d8ff79e61e08bb7f86c63acfc1e567f925849a24843ddbcb41cf3668c2fb2723 |

Both records are terminal accepted and production qualified. They share
science identity
7ef8ad8d19cbcf35a5e4ed2bed48da8bed2c9b9f5fa95173b7eda38ff53c918d,
rank identity
693df1eb35dad74dca10ae46b3a8abb6e84fa79b2c61f7b8a8fdee3af7c93a6d,
integrity identity
635622feba54571b96b11c17d2103f63b2d4adc0746b7ba5351e451cc526d935,
and exact combined trace
8a687648779ed9b976768bfa8f30b0eb3c27a11f8b22085abc9090e2f9e845dd.
Every fixed-probe discrepancy is zero. The speedup is
0.9631422300961628; T8 is rejected for
material_end_to_end_speedup_not_met and T1 is selected under cache key
c159ae1955b3c95b11f3dbc6540b1854c4e71d2072421bf9bcc9ea0b50c432e8.

## Rejection of the unsafe parallel witness

On source 6bdb43f, the isolated restricted T8 plan at the N=512/M=2048
geometry failed before aggregate publication with:

~~~text
RuntimeError: ContextualReferenceExecutorV1 trusted fallback verification failed
~~~

The relevant preserved evidence is:

| Evidence | SHA-256 |
|---|---|
| screen_group_restricted_diagnostic_postfix_60a494c_FAILURE.txt | 251b9e49de14fe9d00073c4a822c81688e5532cc624b07983ed689bc36410e59 |
| screen_group_direct_action_diagnostic_postfix_60a494c_FAILURE.txt | b206caf51a79393b2862aac7796825905534a069defd003831f5c737f99f8a86 |
| screen_group_b16_postfix_60a494c_FAILURE.txt | 17dc873985d33daa6c896f405340936113b0114bbc53f0414c98a8b4ba45ea57 |
| Passing identical-geometry T1 control screen_group_restricted_t1_diagnostic_postfix_60a494c_r1.json | b2a40dfa7166afd2a1636cf13fe9684922372ab4764fd5b465781e9b499bc9d2 |

The failure published no final benchmark record or valid timing. The T1
control was accepted. After the serial-witness rollback, the clean
three-replicate restricted T1/T8 record at the same geometry is accepted for
both plans, with exact science and trace. This closes the rollback
requalification but does not rehabilitate the rejected 6bdb43f source.

The earlier clean small cross-build record
reports/context_native/benchmarks/stage6/screen/crossbuild_foundation_candidate_t8_n128_m512_b128_exclusive_r3.json
(SHA-256
e7fe2b29e76a09cd874473591e07195a704ed32bfff1546111e5e51478fd3ff4)
is preserved as historical measurement. Its candidate was already slower
end-to-end, and all of its rejected-source timings are non-selectable after
the larger integrity failure.

## Resident-probe screen

The one-repeat rollback screen is
reports/context_native/benchmarks/stage6/screen/rollback_screen_resident_t1_n128_m512_b128_r1.json
(SHA-256
caa21d85a225337e672207a616e70d0ea4b0b9c8f3a725155cc4566d0ba9342e).
Its dry run is SHA-256
4e08b8f57bbd6d8386cc6b4cc7b122634dd42e7122135eb3147f3494fe9fcbc7.
All plans are nonproduction because each has one measured replicate.

| Resident probes | E2E s | Native admitted bytes | Whole-process RSS bytes | Result |
|---:|---:|---:|---:|---|
| 128 | 4.805108911 | 14,783,488 | 155,758,592 | Accepted; exact science/rank/trace |
| 16 | 4.941055238 | 11,363,904 | 155,455,488 | Rejected: full_acceptance_evidence_failed; rank/trace identity differs |
| 32 | 4.884304732 | 11,843,392 | 156,631,040 | Rejected: full_acceptance_evidence_failed; rank/trace identity differs |
| 64 | 4.879713647 | 12,820,416 | 155,492,352 | Rejected: fixed_probe_science_comparison_failed and full_acceptance_evidence_failed |

For resident 64, fit.raw_coefficients has maximum absolute discrepancy
4.275889864402416e-09 and tolerance ratio 3.312716420117046. The smaller
resident counts are rejected alternatives; their lower native payloads do not
justify relaxing science, rank, or trace gates. No count below 128 was
selected.

## Grouped-algorithm screen

The restricted T1 baseline at N=512/M=2048 is accepted and production
qualified in the larger production record above. The rollback grouped sweep
then reached a separate direct-action integrity failure:

| Evidence | SHA-256 | Outcome |
|---|---|---|
| rollback_screen_group_t1_n512_m2048_b16_dryrun.json | 10f7fd342eb4f9322ac7b1f6d4b3fb9f67ac9d7906b507d522f2c5fa8ba7da84 | All three plans admitted |
| rollback_screen_group_t1_n512_m2048_b16_FAILURE.txt | cfffa55a0e503a695ce3bee95233acac82f74e7af6d32e4c99d0bb7bc95dd916 | Sweep stopped before final publication |
| rollback_screen_group_direct_action_t1_exact_diagnostic_dryrun.json | b8d076bca09d517bf44ace31d16af8ffaa9f3c32d715440f653ab7db4360a70b | Direct-action plan admitted |
| rollback_screen_group_direct_action_t1_exact_diagnostic_FAILURE.txt | 6e876f1ad48a6c0cacc56cbf71898eef5d37f01e4be36d6f9ae1412dd6f3fa81 | Direct-action T1 fails trusted-fallback verification |

The exact direct-action diagnostic exits 1 after 5.902389181 s of controller
wall time and creates no final record. That duration is not a benchmark
timing. Direct action is rejected. Direct genotype was not run by instruction,
so no direct-genotype science, runtime, or memory claim is made.

## 1,024-probe artifact screen

The “B1024” screen uses 1,024 sample probes and 1,024 variant probes. Its
source/target/group/same-person/trait variant blocks remain 128; it is not a
measurement of the modeled 1,024-variant-block plan.

The dry run
rollback_screen_b1024_t1_n64_m256_dryrun.json (SHA-256
38c27d0348ce0c4106bcd64ef3f5c1d14f4349ad492b12cee81d2d55354bd91e)
admits resident counts 128, 256, 512, and 1,024 at N=64/M=256. The official
writer completed, but the immediate official loader rejected the first
artifact:

~~~text
ValueError: Contextual reference V1 member 'manifest_json.npy' is not a bounded Unicode scalar.
~~~

Failure evidence
rollback_screen_b1024_t1_n64_m256_FAILURE.txt has SHA-256
4a650b4376105fbd2873c6163826cd5b737bc4225c0d1be06084ec32427b6e36.
The read-only postmortem has SHA-256
92ed61b48cbf05d3846bd74b9f899ce88e83ff5bcaec92e5389c5df566eb9dfd.
The paired writer guarantees canonical ASCII JSON, scalar shape (), and dtype
<U{L}; with a 128-byte NPY header, the loader failure proves L is at least
4,194,273, the UTF-32 payload at least 16,777,092 bytes, and the member at
least 16,777,220 bytes. Exact L is unavailable after temporary-workspace
cleanup. The executed reference ledger is reconstructed as 1,485 protected
calls and 2,987 no-fault events, with required event capacity 11,912 and
4,478,912 telemetry bytes.

The loader correctly failed closed under its 16,777,216-byte manifest limit.
The remaining defect is a writer/loader policy gap: the writer can publish an
artifact the official loader rejects. No loader/source change was made, no
final timing or science record exists, and resident 256/512/1024 workers were
not run. B1024 is unavailable. Stage 7 must preserve the cap and add shared
writer-side size preflight with a no-output regression.

## Real-prefix pilot

The final one-repeat real-prefix pilot uses the full hashes of
example/small.bed/.bim/.fam and a retained N=512/M=2048 case with Q=4, K=2,
eight groups, 16 sample/variant probes, resident 16, variant blocks 128, and
restricted T1/T8 plans.

Machine record:
reports/context_native/benchmarks/stage6/screen/rollback_screen_real_small_t1_t8_n512_m2048_b16_fullhash_r1.json,
SHA-256
942e3bae34ae10313bc040c1abdca9a8ebe8303293c58a645faccf3a30ba04f9.
Dry-run SHA-256:
7b55a3f1d3d8b269f6666735cc9ab901fd3f624b3693a61416e69b549b909329.

| Metric | T1 | T8 |
|---|---:|---:|
| E2E s, one replicate | 4.754753574 | 5.027633157 |
| Whole-process peak RSS bytes | 156,856,320 | 160,100,352 |
| Native admitted payload bytes | 16,322,368 | 16,322,368 |
| Record SHA-256 | c0ebb06bf470f6501dd822985b18e1446c49dfed3153caafd8ac68ab0cfa38f8 | b83d2612b0d0125fd4ce423bf967b5b784c86b0b3992e64e62fc0d68c72a6e86 |

Both records are terminal accepted with exact arrays, deletions, rank, and
trace and zero recovery across all four native reference/trait reports. They
share case/science identity
028d212beae5a4d1c11a96ad7c304f7f4480e0be61d5a829a55ae77e74b0c360,
rank identity
6eadbd59334e0f7c77d457d9ca2fa61bc83eaa620a50c7fb31b1878973390ade,
and trace
170653edf87afa5eb281e2a7504c354f72414869a07a47afb8e276617639b523.
The BED/BIM/FAM hashes are respectively
206eccc79cadad8def1dc17fdeb1c9166ac391625390d082fc69f56fabfc9f5f,
4859421b6b5928c8c2a3b63cb9b4e4937e78059ca71cf671bf940c070737e3b7,
and ada7c496e64ef9c9c0d04ad6db31f2a15676d07deec785a104815ab218d76b1e.

Both plans are nonproduction solely because fewer than three measured
replicates were run. The selector is unavailable with
record_not_production_qualified. The one-repeat ordering is screening
evidence, not a material speedup or real-data performance qualification.

## Admission, RSS, I/O, and NUMA nonclaims

admitted_peak_bytes is the maximum native reference/trait payload admitted
for sequential execution. It is **not total-process RSS**. For example, the
selected small T1 record admits 14,783,488 native bytes while the measured
worker reaches 156,807,168 RSS bytes. The larger selected T1 record admits
16,322,368 native bytes while the measured worker reaches 145,080,320 RSS
bytes.

The machine field admitted_measured_memory_agrees means only that RSS stayed
below native admitted bytes plus the explicit 268,435,456-byte harness
margin. It does not account the Python interpreter, NumPy/SciPy, allocator
metadata, runtime libraries, mappings, page cache, or all publication/load
copies as native admission categories. No full-RSS production admission claim
is made.

Physical record visits and logical BED bytes are recorded. Physical read
bytes are unavailable because the executor uses mmap; records identify this as
mmap_physical_read_bytes_unavailable_v1. NUMA placement pages sampled from
/proc/<pid>/numa_maps are not remote-memory traffic. NUMA_remote_bytes is
therefore unavailable. CPU affinity to node-0 cores is verified, but
allocation remains unbound_first_touch_v1, output node is -1, and there is no
first-touch ownership, local-output placement, or remote-byte claim.

## Modeled target estimate, not a measurement

reports/context_native/benchmarks/stage6/estimates/tabla_target_modeled.json
(SHA-256
68c5d714b6ddcd16ddae9ded6a30094a9dc9c5ed814cc39343634ff2dc52d018)
uses the checked planner for N=300,000, M=1,000,000, Q=4, K=8, C=80,
J=200, one process, and 64 modeled threads. It is labeled
modeled_not_measured, not_production_admission, and
no_runtime_extrapolation.

| Modeled scenario | Required native peak bytes | Source/action passes | Same-person passes | Under 1.082 TB planner cap |
|---|---:|---:|---:|---|
| Variant block 128, resident 128 | 134,803,508,468 | 1 / 1 | 4 | yes, modeled only |
| Variant block 1024, resident 128 | 481,511,502,682 | 8 / 8 | 32 | yes, modeled only |
| Variant block 1024, resident 1024 | 775,686,188,276 | 1 / 1 | 32 | yes, modeled only |

These values are first-order native lower bounds with headroom. They are not
measured RSS, runtime, bandwidth, NUMA, page-fault, I/O, or production
admission evidence. The failed 1,024-probe artifact screen does not measure
either modeled 1,024-variant-block scenario.

## Required-axis coverage and explicit nonclaims

| Required axis | Settled Stage 6 status |
|---|---|
| Resident probes 16/32/64/128 | One-repeat synthetic screen complete; only 128 accepted. |
| Resident versus tiled source/actions | Resident-count batching screened; true nonresident source/action plans are not executable. |
| Group-restricted/direct action/direct genotype | Restricted T1 accepted; direct action rejected by native integrity; direct genotype unrun. |
| Group/annotation/action/context/variant tile sweep | Baseline settings only; no independent tile sweep. |
| Independent source/target/group/same-person/trait blocks | Not executable; current executor requires one shared variant block. |
| Variant block 128/intermediate/1024 transitions | Variant block 128 measured only; 1024 is modeled, not measured. |
| Strict versus generic annotation packing | Strict contiguous only; generic and indexed routing unmeasured. |
| Protected dense GEMM versus packed/Mailman | No executable comparison in this harness. |
| T1 through Tn | T1 and T8 only on eight physical cores; no broader thread ladder. |
| One 32-core socket, qualified 64-core mode, two 32-core processes | Unmeasured; harness measurements use one process on CPUs 0-7. |
| NUMA first touch/output placement/remote traffic | Unbound only; output -1; remote bytes unavailable. |
| Allocation reuse/copy/huge pages | Reuse and deterministic reductions fixed on; alternatives and explicit huge pages unmeasured. |
| Synthetic smoke/medium/large/tabla ladder | N128/M512 and N512/M2048 only; no N2k/M10k through N300k/M1m measurement. |
| Real BED pilot | One retained N512/M2048 example/small replicate per T1/T8; nonproduction. |
| Full RSS admission | Not implemented; native payload plus a 256 MiB harness margin is not a full RSS model. |
| Physical read bytes and NUMA remote bytes | Explicitly unavailable. |
| Mixed precision | Not attempted; FP64 has only bounded qualification. |

The validator also fixes source_coordinate_tile=context_tile, full
scalar-witness integrity, FP64, deterministic reductions, one process,
unbound NUMA, disabled explicit huge pages, resident source/actions, and
event-count rather than byte-count telemetry admission. Those schema
constraints are not evidence for alternate implementation support.

## Machine-record validity and invalidations

Initial records affected by the immutable-provenance defect or overlapping
CPU use are preserved and invalidated by:

- core_node0_prior_results_INVALID.json, SHA-256
  7876411f8100f3ce0a5dcb1fcfb26ad095ca03661df2451512b6aa6dbf15b879;
- crossbuild_prior_results_INVALID.json, SHA-256
  149cfa16f07c13f001704fe231956905d4249148f5c904323895a395785b2546.

The invalidation manifests name each invalidated path/hash and its exclusive
replacement. The post-fix historical candidate records remain content-valid,
but they are non-selectable because their source is rejected. Final selection
uses only rollback-labeled records bound to commit 69fc6c1, tree aa63e704,
the Release native/package hashes, qualification hash, and immutable build
provenance above.

## Remaining release blockers and Stage 7 handoff

1. **No performance win was selected.** T1 is the retained bounded policy;
   T8 failed the 1.05x end-to-end gate in both production comparisons.
2. **The parallel-witness candidate is unsafe and rejected.** The serial
   rollback is qualified; reintroducing the candidate requires a new,
   independently gated design.
3. **Direct action is rejected and direct genotype is unknown.** The grouped
   algorithm comparison is incomplete.
4. **The 1,024-probe writer/loader policy is inconsistent.** The loader
   correctly fails closed; Stage 7 needs shared writer-side preflight. No
   B1024 science or performance result exists.
5. **Target scale and full RSS are unqualified.** Million-variant values are
   planner estimates only.
6. **NUMA/socket/process evidence is incomplete.** Affinity is not page
   ownership or remote-traffic measurement.
7. **Required packing, block, backend, scale-ladder, and repeated real-input
   axes remain open.**
8. **Build prefixes are ephemeral.** Content hashes bind them, but Stage 7
   should audit rebuildability and evidence references before any release
   decision.

These items prevent a full performance GO and public performance release.
They do not contradict the narrow Stage 7 handoff because the accepted,
qualified restricted T1 baseline is retained and every failed or unmeasured
alternative is excluded from selection.

## Reproduction commands

All campaign records bind plan JSONs, build content, dimensions, CPU affinity,
work caps, and output hashes. The exact focused sanitizer test command is:

~~~bash
/home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q tests/test_context_stage5_build_integrity.py tests/test_context_stage5_native_integrity.py tests/test_context_stage6_performance_instrumentation.py tests/test_context_stage6_parallel_witness.py
~~~

The exact direct-action and 1,024-probe controller commands, including all
caps and stderr, are preserved verbatim in:

- reports/context_native/benchmarks/stage6/screen/rollback_screen_group_direct_action_t1_exact_diagnostic_FAILURE.txt;
- reports/context_native/benchmarks/stage6/screen/rollback_screen_b1024_t1_n64_m256_FAILURE.txt.

The two final production outputs must be reproduced with the exact build spec
rollback_candidate_69fc6c1_build.json, their rollback T1/T8 plan JSONs, CPU
list 0-7, one warmup, three repeats, production qualification enabled, and
new output paths. Reusing rejected candidate build specs, invalidated results,
or the rejected source's T8 timing records is not requalification.
