# SUMMIT G+GxE+NxE completion audit

> **Current status:** the 2026-08-17 BLIS acceptance and rejected topology
> screen are recorded in `CURRENT_STATUS_AND_REMAINING_WORK.md`. That report
> supersedes the operational status and next-step wording below; the historical
> audit is retained unchanged.

Date: 2026-08-15 PDT. Repository: `/home/bronsonj/SUMMIT`, branch `main`,
starting HEAD `6bf009e850afd4b112bf46ee04d2cdea7acd8896`.

## 2026-08-16 implementation checkpoint

The B=32 five-environment real reference completed and passed its float64
height-by-age population-transfer sanity fit. A subsequent API-6 native
phenotype scorer replaced the remaining process-shared NumPy hot path. The
real score fell from 2:13:48.588 to 3:43.878, used one genotype pass, peaked at
5.391 GiB sampled process-tree RSS, and recorded no integrity guard, duplicate
moment check, repair, or rerun. A one-thread dense audit showed the new scores
corrected 12 sparse clustered legacy outliers while matching within `6.7e-12`;
the corrected component proportions differed by at most `1.78e-9`.

The precise scientific, input, numerical-runtime, throughput, and validation
constraints that future implementations must preserve are now normative in
`IMPLEMENTATION_INVARIANTS.md`. In particular, “guard-free” applies only to
the symbol-hidden private-static BLAS path. Structural ownership, input hashes,
read-only operands, finite checks, and atomic provenance remain mandatory; a
process-shared fallback remains guarded or must refuse execution.

## 20:30 PDT runtime addendum

Jackknife block count has no construction-time role. The one-tile protected
path now reuses each decoded block to build feature moments and its randomized
source contribution together, then decodes once more for the target product:
two genotype passes for every `--njack` value. Native column-parallel genotype
standardization also replaces the serial NumPy mask/reduction path. Three real
UKBB blocks (`N=289111`, width 2000, 32 physical cores) took 0.48--0.58 seconds
each including BED decoding, versus 2.6--10.5 seconds on the superseded paths.
This implementation and the private guard-free backend passed all **379 tests
in 41.59 seconds**, including five-environment comparisons to independent dense
oracles. These observations supersede the three-pass and 377-test statements
in the earlier addendum below.

## 20:05 PDT implementation addendum

This addendum supersedes the earlier run-status and performance statements
below. The two obsolete full-five-environment workers were terminated cleanly
at 19:39 PDT after profiling established that neither had reached a publishable
scientific output. Their status files record exit code 241; no reference bundle
or receipt existed. There are currently no SUMMIT reference workers running.

The apparent dependence on jackknife blocks was erroneous. Multi-environment
construction retains no within-block sketch, passes `within_jackknife=None` to
finalization, and makes the same three genotype passes regardless of `--njack`.
The block count assigns completed score rows to post-hoc deletions only. The
32-probe override affects Monte Carlo/jackknife calibration, not the number of
matrix products.

The hours-long run was traced to two operational defects plus an avoidable
feature algorithm: (i) an OpenMP private-BLAS team used active barrier waiting
while Python performed serial reductions, (ii) the CLI's former
`force_affinity_all=true` default could discard `taskset` placement, and (iii)
the old feature pass materialized and projected one `N x block` interaction
matrix per environment. The runtime policy is now `PASSIVE` with
`GOMP_SPINCOUNT=0`, inherited affinity is preserved by default, and exact
feature moments use one native scalar reduction plus one packed algebraic
contraction. The power-zero intercept/common-covariate basis is stored once,
not once per environment.

For four or more large direct references, the CLI can now run two independent
environment groups on disjoint physical sockets. This is exact because each
environment reference is an independent model; no cross-environment numerical
state is required. Each process has a symbol-hidden private BLAS runtime and a
socket-local NUMA policy. Group reference hashes are validated before a single
canonical manifest is transactionally published. A subprocess integration
test compared every XX/XW/WX/WW score against the serial protected path at
`rtol=atol=3e-12` and passed.

The final edited-source suite currently passes **377 tests in 39.78 seconds**.
At `N=289111`, block width 2,000, five environments split 3+2, and 32 physical
cores per socket, production-shaped matrix projections were 15.56 and 14.91
minutes. Measured process peaks were 6.17 and 5.58 GiB. These are hot-path
projections, not a completed real-genotype wall time, but they support a
credible under-20-minute target with a 16-GiB cap. The one-process option
remains available for the lowest aggregate memory footprint.

The four Hoffman arrays 14355909, 14355910, 14355912, and 14355913 were
re-queried at 20:04 PDT and remain explicit `hqw` arrays, tasks 1--40 with four
slots. Their remote root exists but contains no reference manifest, receipt, or
binding. Releasing them now would fail their mandatory input checks, so they
remain held until the new local reference is complete, hash-validated, staged,
and bound.

## Final decision

| Scope | Decision | Meaning |
|---|---|---|
| Production resumption / B=1024 | **FAIL (not approved)** | Required uncertainty, transport, backend, performance, and memory gates remain open. The stopped local reference and four held Hoffman arrays were not changed. |
| B=32 point-estimator implementation | **PARTIAL; real run active** | Dense/synthetic correctness, strict input semantics, solver, metadata, fused arithmetic, and final native integration pass. A fresh five-environment full-reference construction was launched and has not yet completed. |
| Broad manuscript claims | **FAIL** | No calibrated coverage/type-I grid, ancestry/shift grid, misspecification grid, comparator study, or backend acceptance matrix was completed. |
| Narrow algebra/software claims listed here | **PASS where individually marked** | Claims are limited to the observed tests and machine artifacts; they are not evidence for real-cohort operating characteristics. |

This is deliberately not called “complete.” The final acceptance definition in
the task requires all gates, not merely a normal-equation solution.

## Mathematical and scientific changes

### Residual-eliminated solve

For non-residual block `A`, traces `t`, RHS `q`, and residual trace/rank `r`,
the fitter now forms

```text
H = A - t t'/r,       b = q - t,
theta_nonres = H^+ b,
theta_0 = 1 - t' theta_nonres/r.
```

`H^+` is an explicit SVD pseudoinverse; no direct inverse is used. The output
records reduced singular values, full rank, condition, solve residual, normal
eigenvalues, component influence, identifiability, raw symmetry error, and
Cauchy--Schwarz violation. Material asymmetry, negative kernel norm, PSD/Cauchy
violation, inconsistent residual row, rank deficiency, and excessive condition
fail closed unless the existing explicit ill-conditioning override is used.

### Feature convention

Canonical version-1 modes are now `standardized_projected` and `raw_projected`.
Legacy `standardized`/`genie` spellings remain aliases. Reference, phenotype
moment, score, and fit inputs validate the same canonical convention. Dense
oracles cover both modes, realized traces, and cross-mode rejection.

### Population transport

The same-/different-individual formula remains numerically unchanged and still
uses exact individual counts. Material same-person asymmetry is rejected before
within-tolerance symmetrization. Diagnostics now record both sample counts and
ranks, both transport factors, `N_S>N_R` extrapolation, symmetry error, and the
finite-probe same-person minimum eigenvalue. No nearest-PSD projection is used.

### Missing genotype calls and variant axis

Mean imputation remains before projection/restandardization. Construction now
records call rate and missingness association with environment/phenotype, warns
at documented thresholds, and falls back from direct native decoding to the
auditable Python stream when calls are missing. The exact ordered variant-axis
policy remains fail-closed; heterogeneous per-variant `N`/`DF` remains rejected.

### Same-person and uncertainty diagnostics

A reproducible comparison quantifies the unbiased U-statistic, split-probe, and
PSD plug-in estimators. It supports retaining the unbiased point estimator and
reporting its finite-B indefiniteness. A dense deletion experiment demonstrates
that full-reference diagonal reuse is approximate (median 1.62%, maximum 9.37%
relative Frobenius error in a deliberately annotation-imbalanced design).
Under the revised calibration rather than equality criterion, a production-like
`K=1`, 100-block, 5,000-replicate experiment passed: GxE-null type-I error was
5.02%, moderate-effect coverage was conservatively 97.76%, and median SE
inflation relative to exact deletion was 17.55%. Exact deletion was not
implemented and is no longer required for this narrow operating regime.

### Fused multi-environment construction

The executor factors each environment projector into a common covariate basis
plus one rank-revealed residual direction, shares `P0G`, packs direction and
interaction contractions, accumulates additive moments algebraically, packs all
environment source weights in a tile, and performs one paired target product
against `[S,eS]`. Common-covariate feature corrections are applied in place,
environment-only contractions and publication fingerprints are reused, and
score arrays are normalized in place. At one environment/probe tile it
preserves three genotype passes. GEMM calls, shape histograms, FLOPs,
environment/probe tiles, resident/transient modeled workspace, repairs, and
measured process peak RSS are written to manifests.

### Native runtime isolation

The `-O3 -march=native` standalone reproducer demonstrates that concurrent
entry into pthread OpenBLAS is one corruption trigger. The real cohort exposed
a second trigger after repeated process-global one-to-many thread-pool
transitions. API 4 removes both boundaries: Linux `gxeldcore` embeds a
symbol-hidden private OpenBLAS, fixes its thread count once, rejects changes,
and serializes every vendor entry. The private OpenMP build shares one libgomp
worker runtime with decoding. Decoder and BLAS thread counts are represented
separately, so a decoder cap no longer resizes or underuses BLAS. Paired target
inputs remain in an `mprotect` read-only mapping. The checksum/fingerprint/
repair/rerun implementation is compiled out by default only for private
OpenBLAS; a process-shared fallback keeps it and cannot be configured
unchecked. Provenance records the mode, runtime layer/count, archive SHA-256,
and integrity state.

## Code paths changed

| Path | Purpose |
|---|---|
| `src/summit/inference/gxe.py` | residual-eliminated SVD solve, matrix diagnostics, population-transfer metadata, feature-convention checks, fit serialization |
| `src/summit/ldscore/gwe_ldscore.py` | canonical conventions, common basis exposure, serial analysis-defining trace contraction, shared fingerprint inputs, pre-imputation missingness evidence, manifests, native missing-call fallback, OpenBLAS version gate |
| `src/summit/ldscore/gxe_score.py` | convention validation and canonical score/moment metadata |
| `src/summit/ldscore/gxe_multi.py` | fused feature/source/paired-target executor, serial fixed-effect factorization, in-place projections/reductions, deterministic transient-aware tiling, protected call shapes/FLOPs, peak RSS |
| `src/native/gxeldcore.cpp` | signed checksum, gamma-based ABFT bound, serialized internally threaded OpenBLAS executor, read-only prepared pair, deterministic rank update |
| `src/summit/cli.py` | canonical mode names and retirement of hidden cache/shard construction controls |
| `scripts/gxe/hoffman/hoffman_deploy.py` | explicit runtime hard gate for retired persisted-artifact tasks |
| `scripts/gxe/hoffman/README.md` | archived-workflow warning |
| `tests/gxe_completion/` | dense/fusion/missingness/solver/transport/uncertainty tests, benchmarks, standalone reproducer |

Legacy cache readers/validators were intentionally retained so already sealed
artifacts can be checked. New cache/shard writers are unreachable through the
CLI and historical workers fail before plan execution.

## Validation results

| Gate | Result | Evidence |
|---|---|---|
| Immutable state/topology/BLAS/scheduler audit | **PASS** | `INITIAL_STATE.md`; four requested arrays observed held, no mutation |
| Report/code equation crosswalk | **PASS** | `MODEL_CODE_CROSSWALK.md` |
| Existing baseline suite | **PASS** | 352 passed before changes |
| Final edited-source/rebuilt-native suite | **PASS** | 373 passed in 42.02 s against the staged private-OpenMP API 4 extension; final rebuild pending documentation hash |
| Residual-eliminated solver/adversarial ranks | **PASS** | 7 focused new tests plus existing summary/file oracles |
| Canonical feature modes and cross-mode rejection | **PASS** | 3 focused tests plus existing raw/standardized dense oracles |
| Genotype missingness diagnostics/fallback | **PASS for implementation** | 2 focused tests; MNAR correction is not claimed |
| Exact variant-axis / heterogeneous-N rejection | **PASS** | existing schema-v3 file and score contract suite |
| Transfer identity and N smaller/equal/larger | **PASS for algebra** | solver tests and reference-transfer simulation |
| Shift/ancestry/overlap/PC transport grid | **FAIL** | not executed |
| Same-person estimator comparison | **PARTIAL** | 400-draw finite-probe result; unbiased estimator retained without PSD projection |
| Exact block-aware jackknife | **WAIVED for K=1 default** | exact equality not required; high-leverage annotation limitation retained |
| Block-local jackknife calibration | **PASS for limited K=1 gate** | 5,000 replicates; null type-I 5.02%, moderate-effect coverage 97.76% |
| Broader CI/reference/probe calibration | **FAIL** | reference-person, probe, overlap, annotation grid not executed |
| Nonzero G--GxE and residual--NxE covariance | **FAIL** | optional kernels and requested misspecification grid not implemented |
| Fused L=5 dense/independent correctness | **PASS** | all four panels and population diagonal; adversarial environment/covariate/annotation cases |
| Fixed-seed deterministic tiling / no sketch files | **PASS** | focused fused tests |
| Three passes at B=32-style one tile | **PASS** | manifest/read-count assertion |
| Native isolation/integrity fallback tests | **PASS for tests** | focused native and build-policy tests inside the full suite |
| Concurrent-entry root trigger | **PASS** | target 6/200 and source 4/100 faulty; private output/Haswell controls also fail |
| Private-runtime adversarial stress | **PASS for bounded gate** | private calls remained bitwise stable while NumPy's visible runtime changed between 1/8 threads |
| Exact-shape guard-free 20-cycle stress | **PASS for bounded gate** | source/2B/4B repeat deltas all zero; max target error `2.2823e-6` |
| 10k/100k two-host backend gate | **FAIL** | not executed; current backend retained |
| Reduced ASan/UBSan/TSan caller checks | **PASS for tested shapes** | clean guard/input results |
| Fused performance | **PARTIAL** | final L=5 median wall improved 22.8% and RSS 8.3% versus the preceding corrected fused baseline; L=10 remained approximately equal to independent wall time |
| Comprehensive enforced memory cap | **FAIL** | resident and sealing-transient sketch peaks are enforced, but the planner still does not include every simultaneous non-sketch allocation |
| No production feature/sketch cache | **PASS for supported construction** | CLI retirement, worker gate, no-file tests |

The staged private-OpenMP build was tested in an isolated interpreter so the
existing editable installation could not substitute its older extension:

```bash
/home/bronsonj/anaconda3/envs/summit/bin/python -S - /tmp/stage <<'PY'
import runpy, sys
sys.path[:0] = [sys.argv[1], "/home/bronsonj/anaconda3/envs/summit/lib/python3.12/site-packages"]
sys.argv = ["pytest", "-q"]
runpy.run_module("pytest", run_name="__main__")
PY
```

## Performance summary

At synthetic `N=8000,M=800,B=32,L=5`, three private-OpenMP runs had a 2.161 s
median and 262,032 KiB median RSS. The immediately preceding corrected-candidate
runs had medians 2.484 s and 269,508 KiB: the final private build was 13.0%
faster and used 2.8% less peak memory, while retaining three genotype passes.
At L=10, the earlier fused median was 4.292 s and 288,156 KiB versus 4.234 s
and 226,652 KiB for sequential independent construction. Details are in
`MULTI_ENVIRONMENT_OPTIMIZATION.md`.

In a sequential same-load control at production dimensions, the guarded build
took 0.951 s source, 3.041 s 2B target, and 4.783 s 4B target; the private build
took 0.638, 0.864, and 1.842 s. Combined hot-path time fell 61.9%. The complete
dense-oracle process peak fell from 13,515,824 to 13,504,000 KiB; that benchmark
is dominated by identical 13-GiB oracle arrays, so the L=5 estimator benchmark
is the informative memory comparison. A separate 20-cycle private stress had
zero bitwise repeat drift; the final five-cycle build repeated that result with
maximum target error `2.2823e-6`. The private binary contains no repair or retry
execution path.

The final runtime-source snapshot is
`d1047f3b082da5a1fdfca17d3f67ce98c0ceea2627b453d90174d7d77c6635ff`.
The build-tree extension SHA-256 is
`7a927a41f9a685cb34ddf6291501f32b2349372070b19b0ba0da219a92abb8ea`;
the staged installed extension SHA-256 is
`2089f4146d0474e9fd7d925e73a445db7ab527f239afba4af04ad83d6400fd93`.

## Remaining limitations and required next work

1. Extend block-local calibration to overlapping/small annotations and add a
   diagnostic threshold based on the annotation mass removed by each block;
   exact deletion is optional rather than the default acceptance requirement.
2. Add independent probe groups and reference-person folds/subsamples; calibrate
   component SE, coverage, type-I error, and overlap covariance.
3. Run the decisive fixed-trait Gram-source grid, ancestry/environment shifts,
   binary prevalence, PC/leverage, missingness, architectures, and fair external
   comparator analyses.
4. Implement or quantify nonzero G--GxE effect covariance and residual--NxE
   covariance, plus heavy-tail/endogeneity/measurement-error misspecification.
5. Complete AOCL/BLIS/Netlib comparisons, 10k/100k calls, persisted frozen
   inputs, second-host and hardware checks before claiming general upstream
   BLAS certification; private-runtime safety does not depend on that claim.
6. Make the tile planner enforce a comprehensive peak-memory model beyond its
   resident/transient sketch contract and complete the active B=32 real-cohort
   end-to-end gate.
7. Demonstrate a substantial fused wall-time reduction before projecting B=1024.

The interrupted historical reference was not resumed. The first fresh root,
`population_multi5_d43ac860_b32_20260815`, passed preflight and initialized all
five estimators, then failed before genotype streaming on the newly diagnosed
OpenBLAS handover bug; its status and resource diagnostics were preserved. The
corrected B=32 five-environment reference was launched in
`population_multi5_824deb6b_b32_20260815`. No held scheduler job was released,
cancelled, or modified, and no B=1024 calculation was launched.

At 15:40 PDT, a second B=32 five-environment reference was launched from the
final private-OpenMP build on disjoint CPUs 64--95 at
`population_multi5_d1047f3b_private_openmp_b32_20260815`. Its sealed software
copy, launcher, resource log, status, and eventual receipt are contained in
that run root. The original guarded run continues on CPUs 0--31; it was not
terminated or overwritten.

At the final read-only Hoffman check (2026-08-15 15:25 PDT), jobs 14355909,
14355910, 14355912, and 14355913 reported `hqw`, tasks 1--40, four slots. Their
full job records show the literal original submission command starts with
`qsub -h`; they are explicit user holds, not resource-queue starvation. The
required remote reference directory and binding were still absent, so release
would make all 160 tasks fail. A local inotify watcher will stage and validate
the completed manifests with the exact frozen phenotype-score loader, then run
`qrls` on the four original job IDs. At 16:26 PDT the inert watcher was rebound
from the stalled guarded run to the final private run; neither computation was
terminated. No job had been released at this check.

## Artifact index

- `IMPLEMENTATION_INVARIANTS.md` — normative constraints for future GxE changes
- `INITIAL_STATE.md` — immutable provenance and baseline
- `MODEL_CODE_CROSSWALK.md` — pre-change discrepancies and disposition
- `MISSINGNESS_AND_VARIANT_SET_CONTRACT.md` — accepted/rejected inputs
- `REFERENCE_TRANSPORT_THEORY.md` — identity versus transport assumptions
- `UNCERTAINTY_VALIDATION.md` — finite-probe and deletion evidence
- `GEMM_ROOT_CAUSE.md` — reproducer, demonstrated trigger, correction, remaining acceptance matrix
- `MULTI_ENVIRONMENT_OPTIMIZATION.md` — derivation, profiles, performance status
- `SCHEMA_MIGRATION.md` — metadata and retired-workflow migration
- `benchmarks/*.json` — machine-readable measurements
