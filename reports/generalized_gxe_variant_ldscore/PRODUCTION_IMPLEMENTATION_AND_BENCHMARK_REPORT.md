# Generalized GxE variant LD-score implementation and benchmark report

Date: 2026-08-23

## Executive summary

A concise illustrated interpretation is available in
[`BENCHMARK_VISUAL_SUMMARY.md`](BENCHMARK_VISUAL_SUMMARY.md).

SUMMIT now has a complete production generalized per-variant GxE LD-score
reference path. It is a
variant-probe estimator, not an extension of the contextual sample-probe
group-action estimator. It implements the full symmetric contextual
effect-covariance model on one sealed genotype scale, uses exactly two complete
reference-genotype traversals separated by a hard barrier, and publishes the
fixed full-genome summaries needed for inference-time SNP-block deletion.

Reference construction, real-data diagonal agreement, point recovery,
off-diagonal power, work accounting, and backend qualification pass. The
nuisance-matched 100-replicate B=1024 benchmark passes every sealed calibration
and signal-recovery gate. B=1024, rather than B=128, is the qualified probe
count for the full Q=3 configuration at N about 10,000. The inference design
must include all matching residual-context kernels
`P diag(eta_qr * phi_q * phi_r) P`; preliminary results produced without that
complete nuisance basis are not qualification evidence and are not published.

The implementation reuses the mature non-general GxE systems machinery:
descriptor-owned PLINK BED input, native decoding and imputation, protected
NN/TN GEMMs, private static pthread-BLIS, OpenMP, allocation and NUMA telemetry,
integrity audits, atomic artifact publication, and the established native
multi-phenotype trait executor. It does not copy the mature estimator's
hard-coded X/W layout or its separate post-projection X/W scales.

The benchmark harness added for this qualification provides:

- an N approximately 10,000 real-age comparison using DBP and SBP in one trait
  pass and reference probe counts B=128 and B=1024;
- a one-genotype-pass generalized GxE phenotype simulator;
- paired 100-replicate diagonal and off-diagonal two-environment simulations;
- full generalized fits, diagonal genetic restricted fits, inference-time
  SNP-block jackknife standard errors, and explicit acceptance gates; and
- one 100-replicate-per-scenario run using one shared 200-phenotype
  trait traversal, sealed B=128/B=1024 references, and a tightened 10% null
  rejection ceiling plus a 0.75--1.35 empirical/JK SE-ratio gate.

The empirical fit results are recorded in the final sections below.

## 1. Scientific model

Let G be the N by M retained genotype matrix after one per-variant affine
transform, U an orthonormal basis for the fixed-effect design, P=I-UU^T, and
Phi=[phi_0,...,phi_(Q-1)] the contextual basis. Every contextual feature uses

```text
F_q = P diag(phi_q) G.
```

The order is binding: the context is applied before covariate projection. A
single mean-imputed, sample-SD genotype column is shared by all q. There is no
context-specific SNP rescaling and no post-projection normalization.

The generalized random-effect parameter is a symmetric Q by Q matrix Omega.
The ordered unique pair axis is

```text
(0,0),...,(Q-1,Q-1),(0,1),(0,2),...,(Q-2,Q-1).
```

For q<r, one parameter Omega[q,r] multiplies the symmetrized SNP atom

```text
f_qj f_rj^T + f_rj f_qj^T.
```

The two orientations supply the factor of two; the stored coefficient is not
rescaled again. With K annotations there are C=K*Q*(Q+1)/2 genetic components,
ordered annotation-major and pair-minor.

For target SNP j and target pair p, the directional generalized LD score
against source component d=(annotation ell, pair s) is the full-genome sum of
products of contextual cross-correlations over every retained source SNP. Its
panel has shape

```text
[M, Q*(Q+1)/2, C].
```

Off-diagonal scores can be signed and are neither clipped nor PSD-projected.

## 2. Reference estimator

### 2.1 Admission and frozen identities

The Python planner validates N, M, Q, K, B, jackknife-block count, requested
memory, genotype format, variant-block width, and RHS tiling. It serializes and
hashes the ordered sample, variant/allele, basis, basis calibration, fixed
effect, annotation, component, pair, and block axes. Native execution receives
stable read-only descriptors rather than reopening genotype paths.

At the start of pass 1, the native BED decoder computes and seals each SNP's
mean, inverse sample standard deviation (ddof=1), missing-count declaration,
and retained-order identity. The exact same affine transform is replayed and
verified in pass 2. The affine vectors are also returned ephemerally so that
the existing trait executor can use the identical genotype scale; their hashes
must match the sealed plan.

### 2.2 Pass 1: global variant-probe sources

The probes are global retained-variant-axis Rademacher signs. A sign is a pure
function of root seed, global SNP index, global probe index, and namespace, so
it is independent of physical genotype blocking, probe tiling, thread count,
and backend.

For each annotation ell, pass 1 forms

```text
V_ell = G (sqrt(A_ell) .* Xi).
```

Each genotype block is decoded once and reused across all resident probe tiles
and annotations. Protected dense NN GEMMs accumulate the complete N by B source
sketch. Only after all M variants have contributed does the executor apply the
contexts and fixed-effect projection:

```text
Y_(ell,b) = P diag(phi_b) V_ell.
```

The resulting sources are complete and immutable. The same sources provide a
probe U-statistic estimate of the full same-person matrix. This matrix is
computed once and reused unchanged in every later jackknife replicate.

Source and target probe tiles are independently configurable execution
parameters. This is important for the real Q=2 workload: pass-1 NN products are
fastest with one wide source tile, whereas Q^2-packed pass-2 TN products are
fastest with target tiles of four probes. The default native API remains
backward compatible by making the source width equal the target width when it
is not specified. Tile width changes neither the global counter-based probe
stream nor the number of genotype traversals.

### 2.3 Hard barrier

The state machine forbids entry to pass 2 until all global sources are
complete, scale identity has been sealed, projection diagnostics have passed,
and the barrier transition has completed. This is scientifically necessary:
scoring a target block against a partial source would produce within-block,
not genome-wide, LD scores.

### 2.4 Pass 2: target scoring

Pass 2 makes one more complete traversal of G. For target coordinate a, source
coordinate b, and annotation ell it evaluates

```text
U_(a,b,ell) = G^T diag(phi_a) Y_(ell,b) / residual_rank.
```

Because Y is already in the range of P, this is algebraically F_a^T Y/r and
does not require materializing every contextual target matrix. Per decoded
target block, Q^2 coordinate families are packed into RHS panels and evaluated
by protected TN GEMMs. Products sharing the same global probe are reduced over
all B probes into directional per-SNP LD-score rows.

The executor accumulates both the full directional numerator and the target
row contribution assigned to each genomic block. It records the finite-probe
pre-symmetry discrepancy, symmetrizes the aggregate normal numerator, and
normalizes by residual rank and annotation masses. Optional per-SNP directional
panels are publication output only; inference does not require them.

### 2.5 Enforced work ledger

A successful normal reference run must report:

- two and only two complete reference genotype passes;
- exactly 2*M retained-variant visits;
- no duplicate visits, retry, repair, fallback, or integrity failure; and
- reconstruction of the full target numerator by summing block target rows.

Probe tiles create multiple GEMMs inside a resident decoded block. They do not
create additional genotype reads or traversals.

## 3. Inference and the jackknife boundary

Phenotype analysis is separate from reference construction. The existing
native trait executor makes one study-genotype traversal for an entire
complete-case phenotype batch. It projects and normalizes all traits, computes
the marginal contextual scores and residual-component moments, and accumulates
their SNP-additive block contributions. In the simulations, all 100 phenotypes
in each scenario, and both paired scenarios, are processed together in one
200-phenotype traversal.

The full model is solved from the reference normal matrix, the fixed
same-person correction, and trait moments. The diagonal comparison
fits a strict submodel by selecting only Omega[q,q] plus the diagonal
residual-context components.

The SNP-block jackknife is performed only at inference time. For block g it
subtracts the stored target-SNP rows from the fixed full-genome directional
LD-score sum and subtracts the corresponding SNP-additive trait rows. It does
not revisit genotype data, recompute scores for retained SNPs, delete source
SNPs from their already-computed full-genome LD scores, or recompute the
same-person matrix. The declared method is
`frozen_full_genome_variant_ldscore_delete_block_v1`.

Reference/trait compatibility and artifact integrity are verified before a
fit. That verification is amortized across the full system and every deleted
system in the same fit; the deletion loop does not repeatedly hash an optional
per-SNP panel. Public one-off assembly remains fail-closed and performs its own
verification.

## 4. Reused production machinery and backend contract

The generalized path is integrated into the existing `gxeldcore` extension and
reuses the mature descriptor validation, PLINK BED decoding, mean imputation,
genotype-scale sealing, protected input/output spans, NN/TN GEMM wrappers,
threading, NUMA/allocation telemetry, and artifact code. It adds generalized
pair/component indexing and Q^2 contextual source/target families rather than
adding a new decoder or routing through the sample-probe group-action path.

The accepted dense backend is the pinned private static pthread-BLIS build:

```text
native-build source commit: 923e9fdf1cf5f871dae5b30540c5b6ba4cf95d3d
native-build tracked-source SHA-256: 37a8b3295d6b091ede925ef242cb2c1507265706a9c67b0148e9725e3da7b9a7
sealed benchmark-workflow commit: 1ddf7f30aba16bd770fdad57bb77e6395e6de571
gxeldcore SHA-256: 8c8374fc1d01615319bb9fa7e4bed1acb7200a29b83e9e10ab5a317d4144ecb5
BLIS commit: e8566eb3e773fb54d11b33e371d13f22d2941e50
BLIS archive SHA-256: 720068171eea951a0bc634d2d1a829561d5a2bae630bce41a24c4f0edbef9d9b
execution mode: serialized_fixed_private_blis
```

BLIS and OpenMP thread capacity are fixed before module initialization. The
workflow captures the pre-numerical process affinity and requires the mature
canonical environment: one explicit singleton OpenMP place per worker CPU,
`OMP_PROC_BIND=SPREAD`, fixed thread limits, no nested OpenMP or conflicting
affinity variables, and automatic pthread-BLIS with no manual thread ways. It
then records the native exact-team placement attestation. At each protected
BLIS call, the native boundary temporarily expands the OpenMP owner's
singleton mask to the authenticated CPU set so new BLIS pthreads inherit the
whole set, and restores the singleton mask on return. The native layer seals
the owner thread, vendor, implementation, thread count, thread ways, and
relevant environment; a later disagreement fails instead of reconfiguring the
vendor. Protected GEMMs retain independent input/output audits; the old
checksum-repair behavior is not used as a normal execution mode.

## 5. Published artifacts

The reference V1 artifact contains frozen scientific identities, ordered axes,
probe identity, the genotype scale plan, full and per-block directional normal
numerators, the symmetrized/normalized full normal matrix, the full same-person
matrix, annotation masses and block masses, diagnostics, and the pass/
performance ledger. The directional per-variant panel may be included or
omitted without changing inference capability.

The trait V1 artifact records compatible sample, variant/allele, fixed-effect,
basis, genotype-scale, annotation, block, phenotype-batch, and residual-basis
identities together with full and per-block trait moments. Writers use staged
publication and readers revalidate schema, dimensions, hashes, and semantic
checksums before fitting.

The main implementation entry points are:

- `src/native/generalized_gxe_variant.inc`: descriptor-native two-pass source,
  barrier, same-person, target-scoring, reduction, and ledger state machine;
- `src/summit/ldscore/generalized_gxe_native.py`: validated native adapter and
  sealed-scale handoff;
- `src/summit/ldscore/generalized_gxe_reference_v1.py`: reference artifact
  construction, validation, and publication;
- `src/summit/ldscore/generalized_gxe_fit_v1.py`: normal-equation assembly,
  full/deleted fitting, and jackknife covariance;
- `scripts/generalized_gxe/workflow.py`: production reference, multi-trait,
  full-model, and diagonal-restriction orchestration;
- `scripts/generalized_gxe/run_real_age_sanity.py`: real age B=128/1024
  qualification;
- `scripts/generalized_gxe/simulate_generalized_gxe.py`: one-pass simulator;
  and
- `scripts/generalized_gxe/run_simulation_benchmarks.py`: shared-reference,
  multi-phenotype benchmark and stop gates.

## 6. Simulator design

The simulator uses a fixed imputed EUR BED and one sealed environment basis per
batch. It varies SNP effects and Gaussian noise across replicates. For each SNP
j and replicate it draws

```text
beta_j ~ Normal(0, Omega / M_causal)
y = sum_q phi_q .* (G beta_q) + C gamma + epsilon,
epsilon ~ Normal(0, sigma_e^2 I).
```

Here Q=3: phi_0 is the additive intercept basis and phi_1,phi_2 are two
centered, unit-SD, mutually sample-orthogonal environments. The fixed design is
[1,phi_1,phi_2], so the specified direct environmental effects are removed in
the same way they would be in an analysis. All contexts use the native
mean-imputed sample-SD SNP scale.

Efficiency properties are explicit:

- all replicate-by-context SNP effects are generated together;
- one decoded genotype block updates every replicate and context via one
  matrix multiplication;
- a simulation batch makes exactly one complete genotype traversal and M SNP
  visits;
- environment and fixed bases are shared across paired scenarios; and
- all 200 paired-scenario phenotypes are combined for one downstream trait
  pass.

The simulator records genotype file hashes, seeds and spawn keys, generating
Omega and its eigenvalues, effect-panel hash, affine vectors, traversal ledger,
realized effect covariance, projected component covariance, projected phenotype
variance, and phenotype-scale-normalized truth.

Phenotype generation remains homoskedastic: only the identity residual
coefficient is nonzero. The inference design nevertheless includes
the complete symmetric nuisance basis
`eta_qr * phi_q * phi_r`. The non-identity residual coefficients therefore
have truth zero. This matches the person-diagonal context structure required
by the fitted generalized model.

## 7. Benchmark designs

### 7.1 Diagonal genetic benchmark

The diagonal scenario uses

```text
Omega = diag(0.2, 0.2, 0.2), sigma_e^2 = 0.4.
```

It has 100 independent effect/noise replicates on the fixed N=10,060,
M=454,207 EUR genotype. Each of the additive and two GxE variance components
has nonzero signal, while all three off-diagonal components are true nulls.
This simultaneously tests recovery by the full generalized model, agreement
with its diagonal restriction, power for the intended components, and false
positive behavior for the additional generalized components.

Generation completed in 252.303 s with one genotype traversal, 454,207 SNP
visits, 111 decoded blocks, and zero missing calls. The maximum absolute error
between each replicate's realized SNP-effect covariance and generating Omega
was 0.001133. Mean realized genetic variance was 0.60070 and mean projected
phenotype variance was 0.99966.

### 7.2 Off-diagonal generalized benchmark

The paired generalized scenario uses the same environments, random innovations,
and noise streams, but

```text
Omega[1,2] = Omega[2,1] = 0.18
```

with all three diagonal elements 0.2. Its eigenvalues are 0.02, 0.2, and 0.38,
so the matrix is positive definite. The correlation between the two GxE SNP
effect vectors is 0.9, intentionally strong enough for an N approximately
10,000 validation benchmark.

Generation completed in 166.179 s with the same one-pass/454,207-visit ledger
and zero missing calls. The maximum absolute effect-covariance error was
0.001073. Mean realized genetic variance was 0.59831 and mean projected
phenotype variance was 0.99747.

The biological interpretation is pleiotropic environmental sensitivity: an
allele that increases the phenotype response to one exposure tends
systematically to increase the response to the second exposure (a negative
off-diagonal would mean opposing responses). The diagonal genetic restriction
assumes those per-SNP sensitivity effects are independent. It therefore has no
Omega[1,2] parameter or symmetrized cross-environment kernel and cannot test
this signal; it can only redistribute it among diagonal and residual terms.

The predeclared benchmark gates require all 100 fits in each scenario; at least
60% rejection of zero for each true diagonal component; no more than 10%
rejection for any true off-diagonal null in the diagonal scenario; at least
60% rejection, correct mean sign, and absolute mean bias no larger than 0.10
for Omega[1,2] in the generalized scenario; and confirmation that the
diagonal-restricted fit has no Omega[1,2] parameter. Null empirical-error-SD/
mean-JK-SE ratios must lie between 0.75 and 1.35. These thresholds are
qualification gates, not a replacement for a larger formal power/type-I-error
study.

## 8. Empirical qualification results

### 8.1 Real EUR age sanity

The real run used the established age/DBP complete-case BED at N=9,401 and
M=454,207, with fixed-effect rank 28 and residual rank 9,373. DBP and SBP were
the two phenotypes found to be complete on exactly these rows and were handled
together. Both references used 200 SNP blocks, one all-variant annotation,
source tiles equal to B, and target tiles of four.

| B | reference wall | pass 1 | pass 2 | passes / visits | clean ledger |
|---:|---:|---:|---:|---:|---|
| 128 | 186.16 s | 103.00 s | 79.01 s | 2 / 908,414 | yes |
| 1024 | 1,242.18 s | 617.13 s | 619.61 s | 2 / 908,414 | yes |

Both ledgers have 111 decoded blocks per pass and zero duplicate visits,
retry, repair, fallback, or integrity failure. The pre-symmetry relative
discrepancy decreased from 4.86e-5 at B=128 to 1.77e-5 at B=1024. The joint
DBP/SBP trait pass took 1,524.10 s and its accounting records exactly one BED
pass, 454,207 unique variant visits, 111 decoded blocks, zero duplicate
decodes, and 555 protected calls.

The diagonal-restricted estimates were highly stable as B increased:

| Trait | B | additive | age GxE | age-squared residual | identity |
|---|---:|---:|---:|---:|---:|
| DBP | 128 | 0.10228 | 0.11830 | -0.10778 | 0.88731 |
| DBP | 1024 | 0.10232 | 0.12011 | -0.10959 | 0.88726 |
| SBP | 128 | 0.11682 | 0.04565 | -0.11301 | 0.95061 |
| SBP | 1024 | 0.11686 | 0.04636 | -0.11372 | 0.95057 |

The largest B=128-to-B=1024 coefficient change was 0.00181 for DBP and
0.00071 for SBP. For DBP, the prior mature non-general SUMMIT coefficients
were 0.10666, 0.11614, -0.10560, and 0.88294; the B=1024 differences were
-0.00434, +0.00397, -0.00399, and +0.00432. Thus the generalized diagonal
restriction reproduces the mature result to much less than its sampling
uncertainty. Relative to the independent prior GENIE B=10/J=10 estimates, all
four B=1024 differences were within 0.51 combined jackknife standard errors.

This real-data sanity result qualifies only the diagonal mature-estimator
comparison. A future off-diagonal real-age analysis must use identity,
age-squared, and linear-age residual kernels. No off-diagonal coefficient from
an incomplete nuisance design is published here.

Per-SNP randomized panels remain noisy even at B=1024, as expected for
individual Hutchinson rows: B=128/B=1024 correlations were 0.44--0.58 and
relative Frobenius differences were approximately 0.117. Their genome-wide
means were extremely stable (absolute mean changes no larger than 0.055).
Against the independent mature B=10 probe stream, correlations are not a valid
identity test and were only 0.055--0.138, but directional means agreed within
0.141. The decisive comparisons are the exact fixed-probe Q=2 bridge and the
stable aggregate coefficient estimates above.

### 8.2 Nuisance-matched 100-replicate simulation benchmark

The production fit includes all six symmetric residual-context
nuisance kernels. Only the identity coefficient has nonzero generating truth;
the other five coefficients have truth zero.

At B=128, null Omega[0,1], Omega[0,2], and Omega[1,2] have empirical-error-SD/
mean-JK-SE ratios 1.10, 1.17, and 1.06. Thus the SE scale itself is reasonable,
but fixed finite-probe point shifts of -0.0057, +0.0266, and +0.0354 yield
rejection rates 8%, 13%, and 16%; B=128 fails the tightened gate. The matching
residual nuisance coefficients move almost exactly in the opposite direction.

At B=1024, the null means are -0.00332, -0.00558, and -0.00781. Their empirical
error SDs are 0.04149, 0.04384, and 0.04709, compared with mean SNP-JK SEs
0.03786, 0.03778, and 0.04513. The ratios are 1.096, 1.161, and 1.043, and the
rejection rates are 8%, 9%, and 4%. All pass the sealed 0.75--1.35 SE-ratio and
10% rejection gates. Omega[1,2] in the generalized scenario has mean truth
0.18051, mean estimate 0.17491, empirical error SD 0.05702, mean JK SE 0.04604,
and 94% detection. The full R=100 B=1024 stop gate passes.

The 200-phenotype trait artifact was generated in one 4,660.37 s genotype
traversal with 454,207 unique variant visits, 111 decoded blocks, zero duplicate
decodes, and exact accounting for all 777 protected calls. The B=128 and B=1024
fits took 336.24 s and 338.97 s. Both reused previously sealed references with
their original exact two-pass/908,414-visit clean ledgers; the B=1024 fit also
reused the trait artifact without genotype access.

## 9. Validation evidence and remaining scope

The dense mathematical oracle, pair/component conventions, two-pass state
machine, exact work ledger, block reconstruction, artifact integrity, trait
compatibility, inference-time deletion, deterministic probes, tiling/thread
controls, and private-BLIS contract are covered by focused tests. The exact
Q=2 matched-feature bridge reproduces the mature X/W XX, XW, WX, and WW
directions to approximately 3e-14 under a common scale. The new generalized
off-diagonal kernel intentionally has no mature-estimator counterpart.

The authoritative package checks also pass unchanged: `check_package.py`
reports `PACKAGE_CHECK_OK`, including all 43 required files, guardrail tokens,
Python compilation, clean patch application, and 49 manifest hashes;
`run_math_checks.py` passes all 10 oracle tests. Its dense exact identity error
is 7.82e-14, randomized relative Frobenius error is 0.004934, and same-person
relative Frobenius error is 0.01034. The complete generalized focused suite
passes 130 tests with one declared skip when tests are split across their
required process-start BLIS capacities. The skip is an optional checksum-based
integrity diagnostic; the qualified production build intentionally uses the
fixed private-BLIS path with `gemm_checksum_enabled=false`.

The complete repository suite, including all pre-existing estimators and CLI
surfaces, passes 1,718 tests with six declared skips and one expected xfail
when its exact one-, two-, and four-thread cases are run in fresh processes.
This split is required by the production contract that seals BLIS thread
ownership and capacity at process start.

The production tile choice was measured rather than assumed. On the real
N=9,401/M=454,207 Q=2 source with B=16, a shared width of four took 234.89 s
(pass 1 219.59 s; pass 2 13.49 s), and a shared width of 16 took 142.34 s
(pass 1 56.02 s; pass 2 84.52 s). Independent source/target widths 16/4 took
70.36 s (pass 1 56.44 s; pass 2 12.19 s) with the expected 111 source and 444
target GEMMs. On the N=2,048/M=10,000 control, full-width source plus width-four
target tiles took 2.32 s at B=128 and 13.66 s at B=1024; both runs had exactly
two traversals, 2M visits, and zero retry, repair, fallback, or integrity
failure. Cross-tile dense/packed tests reproduce the same scientific arrays to
the oracle tolerance.

A focused launch-affinity check reproduced the mature thread-ownership issue
and its solution. On the same N=2,048/M=10,000, Q=3, B=128 control, implicit
`OMP_PROC_BIND=TRUE`/`OMP_PLACES=cores` without native placement configuration
took 48.24 s because BLIS pthreads inherited the owner's CPU-0 singleton.
Disabling that binding took 5.23 s. The production solution is not to leave
binding disabled: canonical explicit singleton placement plus native
configuration also took 5.23 s, produced the identical SHA-256 scientific-
output digest in all three runs, and retained the exact two-pass/2M-visit
ledger. The generalized benchmark workflows now reject the implicit
environment and require the attested canonical contract before genotype
execution.

Machine-readable qualification outputs are:

- `/home/bronsonj/SUMMIT_generalized_GxE_benchmarks_20260823/simulation_r100_b128_full_residual_v3/simulation_benchmark.json`;
  and
- `/home/bronsonj/SUMMIT_generalized_GxE_benchmarks_20260823/simulation_r100_b1024_full_residual_final/simulation_benchmark.json`.

The production qualification scope is Linux x86-64, FP64, descriptor-owned
PLINK BED, and the exact pinned private pthread-BLIS identity. PGEN remains a
planner-only V1 format. Reference estimation is exactly two genotype
traversals; the additional one-pass trait calculation is a separate analysis
step and can be amortized across a multi-phenotype batch.

The final qualification decision is therefore split. The generalized
per-variant LD-score reference implementation is complete and passes its
scientific, two-pass, backend, real-data diagonal, and point-recovery checks.
The nuisance-matched 100-replicate B=1024 uncertainty run passes. The full
generalized inference
workflow is therefore qualified for the tested N about 10,000/Q=3/all-variant
configuration when it uses the complete matching residual-context basis and
B=1024. B=128 is not qualified for this configuration because of finite-probe
point-estimate bias. No exact delete-kernel or reference-time jackknife
recomputation is proposed.
