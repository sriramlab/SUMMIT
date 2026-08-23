# Fused multi-environment optimization

> **Archival, non-normative.** This document records the optimization history
> that led to the current implementation. Earlier sections describe superseded
> private-OpenBLAS/OpenMP and multi-pass designs; the normative contracts live
> in IMPLEMENTATION_INVARIANTS.md and CURRENT_STATUS_AND_REMAINING_WORK.md.

## Unified adaptive native pipeline (2026-08-18)

Single- and multi-environment phenotype-free reference construction now use
one descriptor-owned C++ direct context. Python supplies validated descriptors
and small design inputs, then validates evidence and publishes final arrays;
C++ owns BED reads, decode/standardization, probes, feature/source/projection/
target work, and reductions. The large numerical state never crosses the
language boundary.

The additive and GxE paths share the Mailman primitives in
`src/native/common/mailman.hpp`. Selection is intentionally narrow:

- B<=10 is eligible for packed Mailman execution;
- B>10 is dense BLIS only;
- dense execution uses the widest memory-safe probe tile and never applies a
  backend-specific width-32 rule.

The dense implementation allocates one bound `[S,e*S]` source/target pair,
accumulates directly into `S`, fills `e*S` once, seals the mapping, and issues
one combined target GEMM per genotype block/tile. Decode and output mappings
are retained by capacity. The 10K-sample B=256 gate used one probe tile, two
genotype passes, three output allocations plus 453 reuses, and 2.537 GiB peak
RSS. All 29,977,662 compared artifact values agreed with the prior accepted
implementation well inside `rtol=atol=5e-12`. Its one timing sample was
contaminated by unrelated CPU load, so it establishes function and memory—not
a throughput speedup.

## Native phenotype scoring (2026-08-16 01:17 PDT)

Reusable phenotype scoring now has a guard-free API-6 direct path. Matched
cohorts validate the exact genotype bytes, sample/design fingerprint, variant
axis, and hashed reference diagonal, then use the sealed scales directly. Each
block is one native decode plus one fused `G' [Y,EY]` product. Population
transfer cannot reuse reference scales: the same decode also supplies exact
study-specific scales and genetic-by-NxE diagonals through the factored
`[Q, E Q, E^2 Q, E^3 Q]'G` contraction. Neither mode constructs full projected
X/W panels, and both make one genotype pass.

On the real height-by-age population-transfer workload (`N=290641`,
`M=454207`, block width 2,000, 32 physical cores), the former NumPy scorer took
2:13:48.588 and averaged about 3.5--5 effective cores while retaining three
large genotype/feature panels. The native scorer took 223.878 seconds
(3:43.878), a 35.86-fold speedup, with 5.391 GiB sampled process-tree peak RSS
and about 25.5 effective cores during the measured middle interval. It wrote
one-pass provenance and zero repair/rerun counters. The complete staged API-6
suite passed 380 tests. The independent dense outlier audit is recorded in
`GEMM_ROOT_CAUSE.md`.

## Final algebraic and socket-isolated path (20:05 PDT)

The protected feature phase no longer materializes any environment-specific
`N x block` X/W matrix. One native column-parallel reduction returns
`sum(G^2)`, `sum(e G^2)`, `sum(e^2 G^2)`, and `sum(e^4 G^2)` for every
environment. One packed `Q'G` contraction supplies powers zero through three;
the exact DirectContext identities then recover normalization, NxE diagonals,
X--W correlation, and leakage diagnostics. The power-zero intercept/common
covariate basis is shared across environments, reducing the production
five-environment packed basis from approximately 460 to 372 columns. Five
independent dense oracles, including binary, correlated, skewed, and
covariate-collinear environments, pass at `3e-12`.

For large direct batches on two-socket hosts, `--gxe-parallel-environment-groups
auto` can split five independent references 3+2. Each child is pinned to
different physical cores, uses a private static OpenBLAS instance, and binds
memory to its socket's NUMA nodes. There is no shared numerical state and no
change to any score. Group manifests and every reference hash are validated
before the canonical batch manifest appears. The serial mode remains available
with value `1`.

Production-shaped measurements on tabla (`N=289111`, `M_block=2000`, `B=32`,
32 physical cores per group) gave 15.56 matrix minutes and 6.17 GiB peak RSS
for three environments, versus 14.91 minutes and 5.58 GiB for two. The slower
group determines parallel wall time. A real subprocess integration test and
the serial path produced all four score families within `3e-12`; the full
suite passed 377 tests in 39.78 seconds. These timings exclude BED decoding and
publication, so under 20 minutes is a measured engineering target rather than
a completed production wall-time claim.

Deletion-jackknife generation has since been retired. New reference
construction stores no block IDs or within-block sketches and writes no
jackknife artifact. The current one-tile path also accumulates the random
source immediately after each feature block determines its normalization,
reducing three genotype passes to two without changing the estimator. On real UKBB
data, native column-parallel mean imputation and standardization reduced a
289111-by-2000 block to about 0.53 seconds including BED decoding (three blocks
measured at 0.48--0.58 seconds).

## Algebra and implementation

Let `Q0` span centered covariates common to every environment and let `u_l` be
the normalized part of environment `e_l` orthogonal to the intercept and `Q0`.
SUMMIT verifies the reconstructed span against each environment's original
rank-revealing basis. If that residual has zero rank, `u_l=0` and the
environment contributes no extra projector direction. Otherwise

```text
P_l = P0 - u_l u_l',       P0 = I - 11'/N - Q0 Q0'.
```

For a decoded centered genotype block `G`, compute `G0=P0G` once and pack all
`u_l'G`. Then

```text
X_l = G0 - u_l(u_l'G),
||X_lj||^2 = ||G0j||^2 - 2 c_lj u_l'G0j + c_lj^2,
```

where `c_lj=u_l'G_j`. Interaction blocks start from `e_l .* G`; contractions
with all common covariates and `u_l` are packed across the environment tile.
Additive norms, `e_l^2` diagonals, and X--W cross-moments are accumulated
algebraically in float64. Only one streamed interaction block per environment
is materialized. Strict finite-value, rank, leakage (`1e-9`), and standardized
norm (`1e-9`) checks remain.

For each probe/environment tile, variant weights for all additive and
interaction panels are packed. Each genotype block contributes through one
wide `G @ weights` call. Completed sources are projected once using `Q0` plus
the low-rank environment directions. The final implementation applies the
common-covariate corrections to each environment in place, avoiding the former
`N x (L * block_width)` projected-feature temporary. The target phase forms a
bounded paired `[S_l, e_l .* S_l]` operand once, seals its backing mapping
read-only with `mprotect`, and uses one wide `G' @ [S_l, e_l .* S_l]` call per
block/tile. Views are unpacked into all four `XX,XW,WX,WW` families with the
original per-environment scales.

Environment-only contractions are precomputed, reductions use allocation-free
`einsum` forms, score arrays are normalized in place, and cohort/variant
fingerprint prefixes are reused during publication. The paired target call
eliminates repeated target operand snapshots and one vendor entry per target
block. Private-runtime builds compile output ABFT and repair out; the
process-shared fallback retains them.

With one environment tile and one probe tile, feature and source construction
are fused, leaving exactly two genotype passes: feature/source and target. The
general tiled fallback records `1 + 2 * (# environment tiles) * (# probe
tiles)`. No feature matrix or randomized sketch is persisted.

## Correctness gates

The new tests compare every score family and same-person population matrix with
independent one-environment construction for five continuous/binary/correlated/
skewed/collinear environments, duplicate covariates, one zero-rank environment
direction, two overlapping annotations, deterministic environment/probe tiling,
and differing-mask rejection. Existing dense-oracle, L=2, native fault-injection,
additive, and one-environment suites also pass. A fixed-seed test observes two
passes and a protected call count independent of environment count in one tile.
Manifest tests verify GEMM shapes/FLOPs, RSS scope, and absence of NPZ/cache/
sketch outputs.

The final source-matched API-4 private-OpenMP extension and edited Python source
passed all 373 tests in 39.74 seconds. Focused protected-pair, in-place
rank-update, fixed-runtime, and no-pool-resizing integration checks also passed.

## Historical measured performance

Synthetic profile: tabla CPUs 0–3, `N=8000`, `M=800`, `B=32`, four genotype
blocks, four SUMMIT threads, OpenBLAS one thread per protected call. Runs are
single measurements, so small differences are descriptive rather than stable
speedup estimates.

Pre/post shared runs were both under cProfile:

| L | Pre shared wall | Fused wall | Pre calls | Fused calls | Pre/Fused RSS |
|---:|---:|---:|---:|---:|---:|
| 2 | 3.578 s | 3.598 s | 86 | 34 | 224/259 MiB |
| 5 | 7.635 s | 7.272 s | 215 | 34 | 229/313 MiB |
| 10 | 14.715 s | 13.661 s | 430 | 34 | 272/380 MiB |

The L=10 call count fell 92.1% and measured GEMM time fell from 2.475 to
0.739 s, but end-to-end wall fell only 7.2%. At L=10, output finalization took
9.44 s cumulatively; metadata writing 5.98 s, CSV conversion 3.72 s, and
analysis fingerprints 2.53 s. Those per-environment file costs remain linear
in L and dominate this small-M benchmark.

Fair no-cProfile independent/fused measurements:

| L | Independent wall / passes | Fused wall / passes | Independent/Fused RSS | Decision |
|---:|---:|---:|---:|---|
| 2 | 0.828 s / 6 | 1.739 s / 3 | 225/231 MiB | fused slower |
| 5 | 1.981 s / 15 | 2.545 s / 3 | 230/287 MiB | fused 28.5% slower |
| 10 | 3.961 s / 30 | 3.917 s / 3 | 222/352 MiB | approximately equal |

These measurements are retained as historical architecture evidence; they are
not measurements of the corrected final executor.

## Final corrected-executor measurements

The corrected executor was re-profiled without cProfile at
`N=8000,M=800,B=32`, four threads, and four genotype blocks. The immediately
preceding corrected L=5 fused implementation took 3.327 s and 271,560 KiB RSS
in one baseline run. Three final in-place runs had wall times 2.568, 2.294, and
2.689 s and RSS values 245,724, 257,484, and 249,048 KiB. The medians therefore
improved wall time by 22.8% and RSS by 8.3% relative to that baseline. All three
runs used three genotype passes and reported zero repaired columns.

At L=10, three final fused runs had a 4.292 s median and 288,156 KiB median RSS.
Three corrected independent runs had a 4.234 s median and 226,652 KiB median
RSS. Thus the final fused path is approximately equal in wall time (1.4%
slower in these short runs) while reducing genotype passes from 30 to 3; its
single-process peak remains 27.1% above sequential independent construction.
This does not establish a general end-to-end speedup. It does establish that
the retained changes improve both throughput and memory over the immediately
preceding five-environment fused implementation, which matches the production
L=5 use case.

The corresponding artifacts are
`multi_corrected_preopt_shared_l5.json`,
`multi_corrected_inplace_shared_l5_rep{1,2,3}.json`,
`multi_corrected_inplace_shared_l10_rep{1,2,3}.json`, and
`multi_corrected_final_independent_l10_rep{1,2,3}.json`.

After private OpenBLAS isolation and guard removal, three otherwise identical
L=5 runs took 2.174, 2.161, and 2.115 seconds (median 2.161) with peak RSS
262,032, 258,204, and 263,968 KiB (median 262,032). The immediately preceding
corrected-candidate artifacts had medians 2.484 seconds and 269,508 KiB. The
private final build therefore improved median wall time by 13.0% and peak RSS
by 2.8%, while retaining exactly three genotype passes. Artifacts are
`multi_private_openmp_final_l5_rep{1,2,3}.json`.

## Tiling, memory, and remaining cost

The source panel contains both additive and interaction families. Protected
execution now allocates the final `[S,eS]` mapping up front, accumulates directly
into its first half, fills the weighted half once, and seals the same mapping
read-only. The panel budget therefore enforces the exact two-panel live peak
`4*N*A*L_tile*B_tile*sizeof(float64)` without a source copy or three-panel
overlap.

The planner enumerates all balanced environment/probe tile-count pairs and
minimizes exact genotype passes before preferring wider probe and environment
GEMMs. A separate `--gxe-total-memory-gib` contract bounds a phase-specific
complete-process model: the current RSS baseline, native context copies,
persistent feature/score/population arrays, decoded block, packed sources and
targets, integrity and vendor workspace, thread stacks, bounded telemetry,
allocator slack, output publication, and 20% headroom. Selected component and
phase arithmetic, budget margins, and planned/observed genotype reads are
validated before publication and retained through group combination.

Private pthread-BLIS executes the selected dense probe tile directly. The
former backend-specific width-32 subdivision is removed: a feasible B=256 plan
uses one width-256 tile and one wide source and paired-target shape. Tiling is
now solely a memory-planning decision. Packed Mailman is available only for
B<=10 and cannot be selected as a fallback for a wider job.

Unavoidable arithmetic remains at least linear in L: each `e_l .* G`, its
projected normalization, its source weighting, and its weighted target panel
are distinct. Packing reduces calls and genotype traffic; it cannot make that
environment-specific work sublinear. The clean 10K B=256 run is the current
validation step for this architecture. The earlier full-cohort B=256 result
belongs to the preceding implementation; the shared/adaptive implementation
has not been rerun at N=289,111. No B=1024 projection or run is authorized
from these measurements.
