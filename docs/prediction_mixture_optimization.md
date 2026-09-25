# Exact work reduction in the mixture PGS solver

Implemented on `working`, starting from `6111a12`, on 2026-09-24. No running
Hoffman job, frozen runtime, checkpoint or training result was changed. Builds
used for qualification are isolated under `build/pgs_optimization_{private,portable}`.

## Model and accuracy contract

For a candidate, let `B` be the orthonormal basis of the noise-weighted fixed
effects, `P = I - BB'`, and `W_j` the noise-weighted genotype-by-context design
at a SNP or block. The integrated design is **P W**, in that order. Write the
residual as `r = P y_w - sum_j P W_j beta_j`. The coordinate update uses

```
t_j = W_j' P r + H_jj beta_j,     H_jj = W_j' P' P W_j.
C_jl = L_jl (I + L_jl' H_jj L_jl)^-1 L_jl'
log w_jl = log p_l - 1/2 log det(I + L_jl' H_jj L_jl)
           + 1/2 t_j' C_jl t_j
beta_j = sum_l softmax(log w_jl)_l C_jl t_j.
```

Here `L_jl L_jl'` is a component prior covariance, including the existing
annotation weights and SNP normalization. Singular priors retain their existing
support cutoff. Fixed-effect integration, mixtures, annotations, genotype scales,
sample masks and residual variances are unchanged.

The original success criterion is retained: independently reconstruct the residual,
freeze all SNP weights during a simultaneous coordinate check, and compare the
sum of per-SNP squared changes in the diagonal-Gram metric against the same
absolute/relative tolerance. This is a VB fixed-point certificate, **not** a proof
of the global optimum of a nonconvex mixture objective. Fixed-effect projection
checks, the monotone objective guard, periodic residual-drift checks, FP64
arithmetic, and native GEMM integrity/checksum policies remain in force.

## 1. Stop updating certified candidates

Candidate objectives are independent; batching shares matrix operations rather
than parameters. Once a candidate passes its own simultaneous fixed-point and
fixed-effect checks, subsequent updates of another candidate cannot invalidate
its solution. The solver now removes certified candidates from native residual
owners and subsequent score, update, reconstruction and coordinate operations.
Full candidate-indexed weights and checkpoint arrays retain their original order.

Before export, all candidates pass a fresh joint reconstruction and check. If the
last iteration already checked the entire final batch, that check is reused;
previously frozen or checkpoint-restored certificates require a joint recheck.
This prevents a stale or corrupted checkpoint certificate from bypassing the
actual numerical acceptance criterion. The test suite includes a deliberately
forged inactive-candidate certificate and verifies that export fails.

This is **not validation-based pruning**: every requested model is still fitted,
returned and certified. An apparently poor ELBO or prediction score does not remove
a candidate. Work changes from approximately `K * max(T_k)` candidate-sweeps to
`sum(T_k)`, plus verification and shared decoding/design costs. Savings therefore
depend on how heterogeneous the candidates' convergence times are.

Checks normally share the existing periodic reconstruction rather than running
two additional genotype passes every iteration after weight changes become small.
An easy batch can request an earlier check when all its candidates first settle;
a failed check is retried periodically. Weight changes only schedule a check:
they never replace the fixed-point certificate.

## 2. Defer the fixed-effect correction

The previous native owner applied `B (B' W delta)` to every participant after
every block and recomputed `B' r` over every participant before each block score.
Both can instead be maintained by an exact small-matrix recurrence.

Store `r = a + B c` and `u = B' r`. With `J = B' W`, a block update is

```
a <- a - W delta
c <- c + J delta
u <- u + (B'B - I) J delta.
```

The score is then `W' a + J' (c - u)`. This is `W' P r`; it does not require
repeated participant-wide multiplication by the fixed-effect basis. Keeping
`B'B - I` is intentional: the implementation does not assume perfect numerical
orthogonality. Tests compare against explicit projections even when the supplied
basis has a small orthogonality error.

The full residual is materialized and the small projection recomputed at sweep
and reconstruction boundaries. The representation is canonicalized before a
checkpoint so uninterrupted and resumed execution have the same arithmetic state.
The native class defaults to its original explicit path for existing callers;
`MixtureSolverSpec.deferred_projection` enables the new path in the mixture solver.
Native residual API version 2 is required for that option.

For `N` participants, fixed-effect rank `r`, block width `d` and `K` active
candidates, two repeated `O(N r K)` products per block are replaced by small
`O(r^2 K)` work plus boundary materialization. The dominant `O(N d K)` products
remain. Thus this identity alone is not an order-of-magnitude speedup.

## 3. Reuse identical component posteriors

Mixed menus pad two-component priors to four components by repeating each
component and splitting its probability. Previously this repeated eigensolves,
linear solves and determinants on identical inputs. These systems are now solved
before padding. Exactly identical component covariance arrays, including the
Gaussian limit, also share the solve. Each original normalizer and probability
is retained, and the same output layout is restored afterward.

There is no approximate component merging. Bitwise comparisons against the
original calculation verify both posterior covariances and normalizers. A padded
Gaussian now needs one such eigensolve instead of four; a padded radial mixture
needs two instead of four.

## Qualification and runtime interpretation

The initial solver-only qualification passed all **102 prediction
tests** on both portable OpenBLAS and protected private BLIS (60.06 s and 57.99 s respectively), with explicit eight-worker placement.
Coverage includes heterogeneous
noise, different masks and variant sets, singular/zero priors, multiple residual
surfaces, whole inactive traits, basis transformations, native-versus-NumPy and
dense/exact-posterior comparisons, reconstruction failures, and checkpoint resume
after candidate compaction. Interrupted/resumed weights and fixed coefficients
agree bitwise with uninterrupted execution of the new implementation.

The benchmark script is `scripts/prediction/benchmark_mixture_optimization.py`.
It imports the original Python solver directly from the baseline commit and only
widens its native-version check to accept the backwards-compatible version-2
owner. The original solver uses the explicit projection path. The same native
coordinate kernel, starting values, candidates, inputs and tolerances are used;
the baseline already includes the September 22 contiguous-Gram optimization.
Run order alternates to reduce warm-up/order effects.

Final complete-fit measurements used 4,096 synthetic participants, 384 correlated
SNPs, five coordinates and 57 Gaussian/radial/separate-sparsity candidates. Median
time across two alternating-order repeats fell from 21.32 s to 6.69 s (**3.19x**).
Maximum final prediction difference was `7.87e-7`, with all
original convergence tolerances met. Candidate-block updates fell from 20,862
to 2,985 (about sevenfold less candidate-update work). Whole genotype-cache
traversals fell from 212 to 165, explaining why wall-time savings are smaller
than the candidate-work reduction. Frozen candidates may stop at slightly different weights within the
unchanged tolerance; equality to the old final weights is not claimed bitwise.

A separate 212,581-participant residual benchmark with width 640, rank 46 and
57 candidates measured **1.075x** for deferred projection alone. Residuals agreed
to `2.48e-15` relative error. This benchmark enters the protected large-GEMM path.

A complete-fit test with 212,581 synthetic participants and only 64 SNPs was an
easy case: every candidate converged together in two sweeps. Baseline and optimized
weights and predictions were bitwise identical. Median runtime was 6.18 s versus
6.28 s (about 1.7% slower), so the optimization does not help every workload.
This case prompted removal of a redundant final verification and duplicate
full-batch copies. Candidate compaction pays off when convergence times differ;
it cannot remove meaningful work from a batch that finishes together immediately.

Raw local evidence is under `build/pgs_optimization_evidence/`, including
`fit_final_code`, `fit_full_n_final_code` and `residual_full_n`. The final fit
`BENCHMARK.json` receipts contain source hashes, native identities and measured timings; model
manifests contain traversal counts, objective histories and convergence reports.

Additional qualification retained two failures: the unchanged baseline tripped
the residual-drift guard on CPUs 0–7 while a separate full-N benchmark ran on
CPUs 8–15 (`fit_complete/FAILURE.json`, drift 0.33984 versus phenotype norm 62.834).
The cause is not established; the optimized case had not yet run. Serial
qualification uses the previously qualified CPUs 8–15 with passive OpenMP waiting.
The original 8-GiB full-N benchmark budget also failed admission before fitting
(`fit_full_n/FAILURE.json`); its retry uses 16 GiB based on the unchanged planner.
These failures are not presented as successful qualifications, and no arithmetic
guard or acceptance tolerance was weakened to work around them.

These initial measurements are bounded synthetic qualifications, **not a measured
speedup or ETA for the running 454,207-SNP jobs**. Their realized savings depend on candidate
convergence, genotype/cache traffic, host contention and setup costs. A
representative full-array sweep remains necessary for a reliable genome-wide ETA.
The subsequent real-chromosome and migration qualifications are recorded below.

## Continuation and controls

The new solver writes active-candidate indices, first certification sweeps and
verification scheduling state into its atomic checkpoint, in addition to the
existing hashed weights, residuals and penalties. Checkpoint identity still binds
to the Python code, native extension, inputs and solver options. Old frozen-job
checkpoints cannot be resumed directly with a different implementation. Their
weights initialize a separately authenticated warm-start run through the migration
adapter described below. Imported residuals and old convergence flags are never
trusted by the new solver.

`freeze_converged=False` retains joint candidate updating, and
`deferred_projection=False` uses the explicit native residual path. These controls
are part of solver identity. Native builds remain isolated from running jobs;
the source-tree extension was not replaced. Using the optimized default path
requires rebuilding the native extension with residual API version 2, or explicitly
loading one of the qualified isolated builds.

Potential alternatives were not silently substituted: thresholding LD or SNP
weights would change the objective; discarding apparently poor candidates would
change the model menu; generic extrapolation can change the solution reached by
nonconvex VB. Exact low-rank prior-specific designs or a dedicated Gaussian solve
may provide further gains, but require additional layout, memory and continuation
qualification. The changes above preserve the existing coordinate updates and
numerical acceptance tests.


## 4. Lossless two-bit genotype cache

`storage="packed"` stores four decoded hard calls per byte: zero, one, two, or
missing. This changes no genotype arithmetic or scientific input. The shared BED
reader still performs decoding; the packing kernel operates on its output and
is not a second BED decoder. Each traversal unpacks only the current block into
a bounded int8 buffer, then uses the same imputation and affine scaling.
Non-hard-call sources are rejected. A partially built cache cannot be reused.
Both native and NumPy packing paths match compact-cache fits bitwise in focused
tests, including missing calls and incomplete final bytes.

Cache allocation changes from `N*M` to `ceil(N/4)*M` bytes. For the full 454,207-SNP
CRP training panel with 212,581 participants this is about 89.9 to 22.5 GiB.
Other solver allocations remain; this is a fourfold cache reduction, not a
fourfold reduction in total process memory. The memory planner includes the
unpacking buffer and conservative native-integrity reserves.

## Real array chromosome 22 qualification

The September 24 local comparison used the actual CRP discovery participants
(N=212,581), all 7,031 array chromosome-22 SNPs, five coordinates, fixed-effect
rank 46, all 57 candidates, and the complete available pilot/replication panels.
Both implementations used identical inputs, candidate priors, 32-SNP blocks,
FP64 arithmetic, and the original convergence tolerances. Annotation-bin priors
are normalized within the bounded chromosome panel; this is not claimed to be
an unchanged restriction of the genome-wide annotation prior.

| Measurement | Frozen September 19 implementation | Optimized, packed cache |
| --- | ---: | ---: |
| Complete fit | 2,799.11 s | 1,887.87 s |
| Fit, scoring and selection | 2,813.45 s | 1,902.19 s |
| Peak resident memory | 4,179,220 KiB | 3,120,956 KiB |
| Simultaneously certified candidate models | 57 | 57 |

This is **1.479x faster end to end** (46.9 to 31.7 minutes), with **25.3% lower
peak resident memory**. All selected candidates agree. Across all candidates,
the largest held-out score difference is 1.03e-6, maximum candidate score RMSE
is 1.31e-7, and relative score-vector error is at most 3.26e-6. The original
relative objective guard also passes. Independent `bed_reader` decoding and
scoring of 64 held-out participants across every SNP and candidate agrees to
less than 3.4e-16 relative error for both implementations.

These are single complete runs on the same eight physical tabla cores, with
passive OpenMP waiting and process-local THP disabled. Other qualifications ran
on disjoint core sets during parts of the measurements, so these are not repeated
whole-machine-isolation timing estimates. They establish representative real-data
behavior; they do not establish a genome-wide Hoffman completion time.

A wider 128-SNP block experiment with optional checked-BLAS products finished
about 50 seconds earlier than the chosen optimized run, but used 5,687,400 KiB
peak RSS and reached different fixed points for some candidates. Although its
objectives and selected models passed, its per-candidate prediction equivalence
gate failed. The experimental switch was removed from the production source.
Its frozen source and failed comparison receipt remain as evidence. Replacement
jobs retain the original 32-SNP update schedule.

Final production source passed **149 prediction tests on each backend**:
86.12 s on protected private BLIS and 89.62 s on portable OpenBLAS. These cover
packing, descriptor mutation, migration authentication and negative/tamper cases,
as well as the earlier solver and prediction checks.

## Authenticated old-checkpoint migration and portable new checkpoints

`prediction.migration.export_legacy_checkpoint` reconstructs the original
preparation and checkpoint identities from the frozen old Python/native runtime,
its exact six solver controls, protocol, and scientific inputs. It snapshots one
atomic generation, using a hard link on the same filesystem or a copy from one
open descriptor across filesystems. Replacement of the old checkpoint pathname
cannot change that open generation. Every checkpoint array is checked for its
shape, dtype, finite values and original content hash before a completion receipt
is written. Originals are never edited, renamed or locked.

When the export runs on another host, independently hashed original-host
metadata may attest the old descriptors. Shared paths, inode, size, mtime and
ctime must match; only the device number may differ. A metadata-only proxy
reconstructs the legacy identity. It is never used for genotype reads or fitting,
and the actual source's mutation checks remain intact.

Import authenticates the receipt hash, exact ordered scientific contract and
checkpoint bytes, then supplies **weights only** through `initial_weights`.
The new solver reconstructs its residuals and recomputes convergence certificates.
The job adapter additionally checks that the variational objective does not fall
below the imported checkpoint's last bound beyond the original 1e-8 relative guard.

For subsequent strict resume, `FileGenotypeSource.authenticate_content()` can be
called before preparation. It hashes all three open genotype files and the
ordered axes, giving a persistent identity independent of paths/device numbers.
Descriptor/path mutation checks still run during use. This is opt-in; ordinary
source identities and old checkpoints are unchanged.

Real chr22 migration imported the old sweep-3 checkpoint and completed in
1,621.11 s including scoring. All selected models agree with the old cold fit;
maximum held-out score difference is 5.60e-7. This warm-run time includes work
saved in the old checkpoint and should not be described as a pure implementation
speedup. Independent BED scoring passed at 3.05e-16 relative error.

A second qualification imported the final old checkpoint, deliberately exited
after its first new atomic checkpoint, and strictly resumed. All 57 weight
matrices, fixed coefficients, and complete held-out score arrays are **bitwise
identical** to uninterrupted execution. Independent BED scoring passed at
3.09e-16. This large local test uses the chosen block-32 arithmetic; the frozen
experimental-capable extension had its experimental switch disabled. Relocated
real-BED unit tests additionally verify content-identity resume. Actual Hoffman
cross-host qualification is a separate deployment gate.

Local receipts, model manifests, frozen runtimes and driver versions are under
`build/pgs_chr22_20260924`, notably `COMPARE_OPTIMIZED.json`, `COMPARE_WARM.json`,
`COMPARE_LATE_RESUMED.json`, `EXACT_LATE_RESUME.json`, and the independent score
audits in each output directory. Operational deployment records are under
`/data1/bronsonj/general_gxe_chromosome_20260914/pgs_optimized_20260925`.
The user approved `/u/project/jflint/bronsonj` for migration snapshots and
replacement outputs because scratch cannot safely hold both generations.


## Deployment continuation

A sealed hourly coordinator runs in tmux `pgs_optimized_rollout_20260925`.
Actual Hoffman build/test, original-checkpoint export, and cross-host interruption/
resume jobs are queued; full replacement fits remain gated on their success.
Original PGS jobs have not been canceled or modified. The workflow retains old
checkpoints, validates complete replacements independently, hands authenticated
local results to the existing manuscript collector, and only then retires matching
original processes. Soft time-budget exits occur after atomic checkpoints; other
failures require review. The operational record and current state are
`/data1/bronsonj/general_gxe_chromosome_20260914/pgs_optimized_20260925/ROLLOUT.md`
and `ROLLOUT_STATE.json`. Their changing deployment status supersedes this dated
source-document snapshot.
