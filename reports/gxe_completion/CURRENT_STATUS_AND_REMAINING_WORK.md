# GxE throughput: current status and remaining work

Date: 2026-08-18 PDT. Repository:
`/home/bronsonj/SUMMIT-gxe-throughput`, branch
`codex/gxe-throughput-optimization`.

The implementation checkpoint is commit
`15dd349852b08f53cbedd403d199337b7eea70d0` (Git tree
`40fe6ba115f27799801f78dc8803cb1fefb91f8e`; embedded tree SHA-256
`380b609e140b2c9f1b6bb18ea9d37b3a34f773a7126d62ade491680bb6c2242b`).

## Current decision

The phenotype-free one- and multi-environment reference estimators now use one
descriptor-owned C++ pipeline. Python validates arguments and small metadata,
constructs the execution descriptor, validates bounded native evidence, and
publishes the final M-length artifacts. C++ owns the BED descriptor, both BED
passes, genotype decode and standardization, Philox probes, feature moments,
source construction and projection, target products, reductions, and
normalization. No decoded `N x K` genotype block or large source/target panel
crosses the Python boundary.

Probe execution has one explicit, test-enforced policy:

- `B <= 10`: the shared native packed-genotype/Mailman kernels are eligible;
- `B > 10`: Mailman is forbidden and the planner uses dense BLIS;
- a feasible dense plan uses the widest memory-safe probe tile, so B=256 is
  one width-256 tile rather than eight width-32 calls;
- memory pressure may create dense tiles, but it never changes a wide job to
  Mailman.

The additive and GxE implementations share the Mailman primitives in
`src/native/common/mailman.hpp`; there is no second GxE-specific copy of the
pre/post kernel. The GxE native context chooses packed or dense execution from
the frozen probe count and records that choice in its versioned evidence.

## Dense GxE memory and call structure

The dense path retains reusable scratch rather than allocating block-sized
outputs repeatedly. Since the 2026-08 audit-repair phase 1, every native
execution-scratch role holds exactly one maximum-capacity allocation frozen
from the admitted plan (never one mapping per exact logical shape), the
planner/native/evidence capacities reconcile fail-closed, and
`release_execution_scratch()` frees all execution-only scratch after the
validated run and before publication:

- one decoded genotype mapping at `N x max_block_width` capacity is reused
  by every full and terminal block width;
- native feature, source, and target output mappings are capacity-based and
  reused;
- one page-bound persistent `[S, e*S]` mapping holds the source and its
  environment-weighted target half;
- source contributions accumulate directly into the first half, the weighted
  half is filled once, and the complete mapping is sealed and exhaustively
  NUMA-verified before target scoring;
- one combined target GEMM produces all four `XX`, `XW`, `WX`, and `WW`
  families for a block/tile;
- a K=2,000, L=3, B=256 target output is about 49 MiB and is reused, not
  emitted as a multi-gigabyte per-block artifact.

For the validated 10K run at commit `15dd349`, the context allocated three
native output mappings and reused them 453 times (the pre-repair behavior:
those were exact-shape-keyed allocations, one per distinct logical shape,
which the audit's Finding 3 identified). It planned 685 protected outputs,
one environment tile, one width-256 probe tile, two genotype passes, and 456
block reads. After the phase-1 repairs the same roles allocate once at their
frozen maximum capacity, terminal shapes reuse the same mapping, and the
mappings are released before publication.

`XW` and `WX` remain separate finite-probe directional estimators. Their
population expectation and trace relationship do not make their realized
random-probe vectors identical, so dropping either one would change the
estimator. Their shared operands and target product are fused; their reductions
remain distinct.

## BLAS corruption and checksum policy

The corruption was not caused by Python-to-C++ data transfer or by B=256
itself. Exact fixed-input native tests isolated real vendor failures in the
high-thread OpenMP-BLIS path; earlier testing also found failures in the
tested OpenBLAS configurations, including the known OpenBLAS 0.3.30 race.
Private static linkage prevents symbol/thread-pool ownership interference, but
it does not by itself fix a faulty vendor threading path.

The accepted production configuration is the pinned private pthread-BLIS
archive:

- BLIS commit `e8566eb3e773fb54d11b33e371d13f22d2941e50`;
- committed source-tree SHA-256
  `eefbd29a5cbb1d6982bdce76e8034037ea2a3f3ead33d90d9f3cef6da5728154`;
- archive SHA-256
  `720068171eea951a0bc634d2d1a829561d5a2bae630bce41a24c4f0edbef9d9b`.

BLIS owns a fixed pthread team. Application OpenMP owns decode and non-BLAS
loops. At vendor entry the caller temporarily exposes the authenticated CPU
set so BLIS workers inherit the intended affinity, then restores the caller's
singleton placement. The private runtime is immutable, serialized at entry,
and has no dynamic BLAS dependency or unresolved BLAS symbols.

For this accepted private pthread-BLIS boundary, numerical checksum/recompute
is disabled by default. The low-overhead safety boundary remains: exact
binary/source/archive identity, immutable thread ownership, dimensions and
finite-value checks, read-only protected inputs, full decoded/input-snapshot/
output NUMA evidence, exact output-call reconciliation, and zero repair/drop
requirements. Shared/system BLAS and private OpenBLAS compatibility builds keep
checksum protection enabled; they are not the accepted production GxE backend.

## Remote CI failure and branch state

The failed GitHub Actions run `32161333199` (job `95790693231`, pushed head
`1a8ac772`) built and linked successfully. Its 14 test failures had two
identified causes:

1. eleven tiny direct-planner tests inherited a generic 3-GiB vendor workspace
   even when their products were below the vendor threshold, inflating modeled
   peak memory beyond the runner's approximately 3.1-GiB limit;
2. three real-NUMA subprocess probes found libnuma in the container but the
   container rejected `membind` with `EINVAL`.

The planner now charges zero integrity workspace when every product is below
the native checksum threshold; the configured native-call workspace is a
ceiling, not an allocation. Large eligible products retain the configured
reserve.
The real-NUMA subprocess reports a narrowly defined exit-77
`NUMA_BIND_UNAVAILABLE` skip for an unavailable kernel/container facility;
mocked contract failures remain fail-closed. A local shared-OpenBLAS workflow
reproduction passed 750 tests (five skipped, one non-strict expected pass).
The obsolete remote `agent/gxe-reference-correctness-throughput` branch was a
strict ancestor of the active branch and has been deleted; no content was lost.

The first documentation-head rerun (`32198625457`) exposed five remaining tiny
dense-direct cases with the same ceiling-versus-allocation accounting error.
The generalized sub-threshold fix passed those exact tests locally under a
7-GiB process-limit override (approximately the runner's 3.1-GiB resolved auto
budget). Final GitHub Actions run `32199266483` passed at exact head `15dd349`:
742 passed, 13 skipped, one non-strict expected pass, and no failed job.

## Clean validation and 10K/B=256 result

The detached clean build root is
`/tmp/summit-gxe-clean-bad6388.YSVtdo`. Its installed `gxeldcore` SHA-256 is
`196066f381ca40a2bbeacd230b49fd4e54f95d3e46c1fc2d419496100c1e0b5e`.
Observed build information is API 9/backend 1.9, Release, direct-context v3,
private-static pthread-BLIS, integrity enabled, checksum disabled, and Mailman
maximum probe count 10. The module has no dynamic or unresolved BLAS symbol.
The clean source/install contain no bytecode caches.

Focused isolated installed-package gates passed 50 tests with one
environment-dependent integration skip: multi-environment, Stage-1
observability, BLIS contract, and native output-NUMA tests. The three
OpenMP-BLIS placement cases are inapplicable to the pthread-BLIS build and
skipped.

The single fresh B=256 development run is:

`/home/bronsonj/summit_gxe/worklogs/gxe_cpp_e2e_10k_B256_L3_dense_fused_blis_bad6388.gxe.multi.json`

- canonical SHA-256:
  `05eb04519a9853d21fbe44a1304dbc7dcddb93b8355d8e2946533ff8dcc7e6a5`;
- group SHA-256:
  `c3d6e37dbfaf2a070b5200862e7bacaef82d34b878d05812b46ca394950d070a`;
- N=9,996 complete-case samples, M=454,207, L=3, B=256, K=2,000,
  T=32, float32 retained storage and binary64 arithmetic;
- dense BLIS, Mailman ineligible, exactly one width-256 tile, two passes,
  456 reads, and 685 planned output calls;
- native descriptor 262.580 s, final reference log 315.900 s, vendor GEMMs
  240.313 s;
- peak RSS 2.537 GiB versus 3.275 GiB in the prior accepted implementation
  (22.5% lower);
- three scratch allocations and 453 reuses;
- zero checksum calls, roundoff resolutions, material repairs, failed output
  mappings, dropped evidence, or pass-count drift.

All 15 score/diagonal artifacts were compared with the prior accepted result:
29,977,662 numeric values over 6,813,105 rows. The worst absolute difference
was `4.263256414560601e-14`; its normalized error was
`0.0001734154332092469` under the unchanged `rtol=atol=5e-12` gate.

This run does not establish a throughput improvement. Its final log was slower
than the prior 274.447-s run. During the new run, unrelated editor analysis
processes consumed CPU, including a process scheduled on CPU 24 inside the
benchmark's selected CPU set. Therefore the one contaminated measurement is
retained as a successful functional/memory gate, not presented as a clean A/B
throughput result. The call structure and memory reduction are established;
an uncontended repetition would be required for a throughput claim.

## Remaining work

0. (2026-08-19) Priority-1 throughput work landed: exact-parallel
   same-person accumulation, a constructor sqrt(A) cache, and
   `--step_size auto` (deterministic `min(nsnps, 8192)` canonical width,
   recorded in the manifest with a selection tag). A back-to-back 10K
   B=256 pair on the private pthread-BLIS build measured 243.0 s -> 166.0 s
   native descriptor (vendor GEMM 213.7 s -> 142.6 s, about 308 GFLOP/s at
   T=32) with peak RSS 3.658 <= modeled 3.826 GiB. The Hoffman2
   shard/merge design is dropped (excessive temporary data); the N~289K
   and B=1024 workloads will run as single jobs, so block width and
   socket-group topology are the remaining scaling levers.
1. Run an uncontended, source-matched B=256 timing only if a stable throughput
   number is needed. Do not infer B=10,000 runtime by linearly scaling the
   dense B=256 run: the additive estimator and GxE estimator have different
   algebra and output families, and both use capacity-aware execution.
2. The new shared/adaptive architecture has not been rerun at N=289,111,
   B=256. The earlier accepted architecture completed that workload in
   64.518 minutes, but that result is not evidence for commit `15dd349`.
3. Do not start Stage 3, sharding, true FP32, B=1,024, topology experiments,
   or further vendor-specific micro-tuning at this checkpoint.

The next feature work should start from this clean boundary. In particular,
do not restore the width-32 BLIS rule, use Mailman above ten probes, or add
another checksum/repair layer to compensate for a backend choice.
