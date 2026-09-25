# Cross-trait genetic response covariance

This extension estimates whether traits share an environmental genetic
response program, whether response correlation exceeds baseline genetic
correlation, and whether (for example) one trait's age response covaries with
another trait's BMI response. A cross-trait response matrix is generally not
symmetric. Its transpose describes the reversed trait pair.

## Model and identifiable parameters

For trait X, write the context-specific SNP effect as
`beta_j^X(e) = phi(e)' b_j^X`, with
`Cov(b_j^X,b_j^Y) = Omega_XY / M`. The first context coordinate is one.
For annotation k, its kernel uses mass M_k and its own covariance matrix.
For observed rows X and Y and their respective fixed-effect projectors,

```
F_a^X = P_X D_a G_X
K_ab^XY = F_a^X (F_b^Y)' / M
E[y_X y_Y'] = sum_ab Omega_XY[a,b] K_ab^XY
              + P_X diag_overlap(phi' Psi_XY phi) P_Y
```

All Q² ordered genetic coordinates are retained. The residual term identifies
only symmetric exposure products on shared people; an antisymmetric part of
Psi is unobservable. The raw Q(Q+1)/2 residual products can also be redundant
(e.g. squared binary exposures). The implementation rank-reveals their
projected Gram, checks its null moments, and solves on the retained span.
No genetic symmetry constraint, PSD projection, or correlation clipping is
applied to XY.

The Frobenius normal equations have genetic entries

```
T[(ab),(cd)] = <K_ab^XY, K_cd^XY>
q[(ab)] = sum_j s_aj^X s_bj^Y / M
s_aj^X = (f_aj^X)' y_X.
```

`MaskedTraitBatch` preserves each trait's rows, projector and phenotype
normalization. `CrossTraitBatch` retains all scores in a decoded SNP tile and
reduces all requested pair products with one batched matrix multiplication
per annotation/target-block fragment. No SNP or trait-pair Python loop is
used for this score reduction. Exact overlap residual contractions are
computed separately and included in the complete-pass timing.

For large masked cohorts, every fixed-effect column must lie in the master
column span on that trait's rows. Its rank, orthonormal restriction and
rotation may differ by trait, including a zero-column fixed basis.
The dense oracle also supports unrelated fixed-effect spaces. Zero overlap
has no residual kernels and no same-person contribution.

## Reference reconstruction and cohort transfer

Ordered pairs use row-major `a*Q+b`. Saved symmetric pairs use
`ContextPairIndex`: diagonals first, followed by lexicographic off-diagonals.
Components are annotation-major. For Q=5 these are 25 ordered or 15 saved
coordinates per annotation.

`cross_trait_gram.py` reads the existing chromosome reference without
genotypes. The saved numerator is scaled as
`residual_rank**2 * block_directed / (M_target*M_source)` using global masses.
Reference diagonals already use global masses; off-diagonal saved components
include a factor of two. Ordered diagonal expansion halves those components.

The cohort exposure factor is computed directly from the master exposure
vectors restricted to each cohort:

```
F_XY[(ab),(cd)] = sum_X(phi_a phi_c) * sum_Y(phi_b phi_d)
                 - sum_overlap(phi_a phi_c phi_b phi_d).
```

Its intercept entry is `n_X*n_Y-n_overlap`; the reference entry is `N*(N-1)`.
Let `O_R,b = T_R,b - D_R,b`. The scalar for a target block and annotation pair
is `ell_b = O_R,b[(00),(00)] / (N*(N-1))`. Since target-block diagonals are
not saved, `D_R,b` apportions each chromosome's same-person matrix by the
target annotation's block mass. This is an approximation in scalar extraction,
not an observed block diagonal. `cross_trait_same_person_check.py` measures
its effect independently from baseline projected genotypes.

The default same-person term uses stored reference diagonals on the actual
overlap, `D_XY = d_overlap @ d_overlap.T`. It is exact for those supplied
diagonals. Substituting them for the two study projectors' diagonals remains
a reference approximation. It does not use the legacy `n_overlap/N` scaling.

| Mode | Different-person term |
|---|---|
| `factorized` (paper default) | `ell_b F_XY` |
| `factorized_plus_residual` | Default plus reconstructed `O_R,b - ell_b F_R`, scaled by the distinct-person ratio |
| `legacy_transport` | Reconstructed `O_R,b`, scaled by the distinct-person ratio |
| `legacy_transport_exact` | Z-repaired ordered `O_R,b`, with the same population scaling |

The ratio in this table is `(n_X*n_Y-n_overlap)/(N*(N-1))`.
The residual-preserving mode retains reference genotype–exposure dependence
and its sampling/probe noise. Factorization need not improve every selected
mask; the real-mask comparison below includes a counterexample.

For Q=5, reconstruction builds the 225×120 commuting-tensor design once and
caches its pseudoinverse composed with the ordered read-out (625×225).
It is exact on that tensor span, not for an arbitrary ordered Gram.

For within-trait fits, `within_trait_equations` defaults to factorized
different-person moments and own-row same-person moments. Explicit
`mode='legacy_transport', same_person_mode='scaled'` delegates to the
unchanged `transferred_chromosome_equations`, including its rounding and
deletion behavior. The regression test requires bit-identical matrices and
RHSs. Existing reference files and old study results are not changed.

## Z moments and the scope of exact repair

`Z_a = G' D_a U_R` identifies the projector's antisymmetric directional
sector. The artifact stores annotation-weighted target-block products
`V_b,k[a,d] = Z_a,b' diag(annotation_k,b) Z_d,b` and source products
`W_k = sum_b V_b,k`. Target weights are required for annotation-specific
repair; unweighted V alone is insufficient. Storage scales as
`8 * blocks * K * Q² * fixed_rank²` bytes, plus chromosome source products.

The four-trace formula in `antisymmetric_gram` obtains T^A, and repair is
`reconstruct(saved(T-T^A)) + T^A`. The repair reconstructs the symmetric target-block sector
used by the original reference normal equations; it does not supply every
directional per-block term for a changed projector. Dense tests establish this
identity for a **single annotation**, including a weighted annotation.
Arbitrary cross-annotation blocks also contain mixed symmetric/antisymmetric
terms; the quadratic products alone do not repair those terms. The exact
repair API therefore supports the common-bin arm, not an unverified exact
repair of all annotation pairs.

An orientation repair of the sealed B=128 reference still contains finite
probe noise. Equality to a dense Gram at 1e-12 is tested using exact saved
moments for identical rows/projectors. It is not a claim that a stochastic
reference equals a dense Gram.

`summit reference zpass` runs one guarded TN traversal with N×(Q C) right-hand
sides, preserves the sealed affine scale, and verifies completion and file
checksums before reading genotypes. Alternatively, the study driver's
`--z-output` obtains Z from `MaskedTraitBatch.fixed_weights` in the existing
shared product. That product uses the same protected TN operator, so no
additional genotype product or traversal is required. When collecting all
annotation Z moments, common-only studies still process all SNPs in that
shared product. Standalone and fused artifacts use the same schema.

## Fitting, uncertainty and derived quantities

`solve_cross_trait_normal_equations` delegates to the existing rank-revealing
context solver after reducing the residual span. Fits report rank, condition
number, minimum Gram eigenvalue, solve residual and deletion diagnostics.

The default uses 200 paired target-SNP blocks, `deletion_method="target_moments"`,
and `uncertainty_method="delta"`. Full estimates are unchanged. Write the
full residual-profiled equation as `A theta = h`, with full-mass moments

```
A = sym(sum_b O_b) + D - B C^+ B'
h = sum_b q_b - B C^+ r,             B = sum_b B_b.
```

`O_b` is the saved target-block off-person matrix; `B_b` and `q_b` are exact
study moments. `D` is computed after summing chromosome diagonals, on actual
overlap rows. If `F_b` is the diagonal matrix of each annotation's target
block mass fraction, the block equations are

```
A_b = O_b + F_b D - B_b C^+ B'
h_b = q_b - B_b C^+ r.
```

The full reference's antisymmetric remainder is removed by allocating
`sym(sum A_b) - sum A_b` with `F_b`; thus block matrices sum to the established
full profiled matrix. For exact symmetric full moments this correction is
zero up to roundoff. Deletion solves `(A-A_b) theta_-b = h-h_b` on **fixed
full-genome coefficient units**, without post-solve mass inflation. These
matrices are directional target/source moments, not symmetric Grams, and
use a rank-revealing SVD. Symmetrizing each block can change its population
RHS and is incorrect. Full and legacy systems retain the existing symmetric
spectral solver. Geometry is cached once, including for within-trait fits;
simulation right-hand sides share each factorization.

With exact block moments, the full and deleted coefficient estimators have
the same expectation. Their realized estimates need not agree. Nonlinear
correlations need not be unbiased. The sealed reference does not contain
per-target person diagonals: `F_b D` remains a mass-apportionment approximation.
Reference transport, probe noise and omitted cross-chromosome LD also remain
explicit approximations. This is target resampling with fixed source kernels,
not dense recomputation after removing both SNP axes.

The delta method propagates the **joint** paired deletion covariance of
`Omega_XX`, `Omega_YY`, and ordered `Omega_XY` through the full-fit Jacobian.
It includes centring, baseline projection and both denominators. It is not
an independent replacement for estimating coefficient covariance: the 200
paired block summaries are still required. A vectorized complex-step
Jacobian avoids finite-difference subtraction error; unit tests and the
report audit check it with independent central differences. Both delta and
nonlinear jackknife covariance matrices are stored. `--uncertainty-method
jackknife` selects the latter. Nonpositive variance denominators remain
undefined; correlations are never clipped to [-1,1].

For historical reproduction select `--deletion-method legacy
--uncertainty-method jackknife`. The old deletion equations and coefficient
multiplier `M_k/(M_k-M_bk)` remain available; `--unrestored-deletions` further
reproduces the earlier raw coefficients. Within-trait `legacy_transport`
with scaled diagonals and legacy deletion delegates to the unchanged old
assembler and is tested bit for bit. Historical round-2 tables below used
that former deletion convention; they are retained as archived results.

Intervals are conditional on the supplied reference and transport mode.
They do not include independent probe redraws or transport-model uncertainty;
the mode comparison reports that sensitivity. Delta linearization does not
cure a weak denominator, misspecified variance model or biased Gram.

The baseline covariance and correlation use the master-basis `[0,0]` entries.
For named-exposure orthogonal responses, first center each trait's intercept
at its own mean context (`Omega_XY_centered = C_X Omega_XY C_Y'`). With
`a_X = Omega_XX[0,1:]/Omega_XX[0,0]` in that centered basis,

```
H_XY = Omega_XY[1:,1:] - a_X Omega_XY[0,1:]
       - Omega_XY[1:,0] a_Y' + a_X Omega_XY[0,0] a_Y'.
```

The output includes the full response and H matrices, per-context response
correlations, `tr(S H_XY)`, and its correlation normalized by the corresponding
within-trait traces. One common, recorded exposure covariance S is used for
the two denominators. The pilot uses the master-cohort S. Correlations with
nonpositive estimated denominator variances are NaN. Values outside [-1,1]
are retained and flagged. Within/cross deletion IDs must agree before derived
jackknife covariance is computed. Paired response-minus-baseline contrasts
are included rather than treating those estimates as independent.

## Files and commands

New NPZ artifacts use schema version 1, JSON metadata, SHA-256 for every
array, no pickle, and exclusive publication. Kinds are
`summit.cross_trait.z_moments`, `.summary`, `.fit`, `.within_refit`, and the
simulation kinds described by the benchmark driver. Sealed formats remain
unchanged. Pair fits include `omega_xy`, `loo_omega_xy`, genetic covariance,
residual coefficients, block IDs, diagnostics, derived point/deletion arrays,
derived covariance matrices, Gram mode and input provenance. Pilot fits also
include basis names, annotation names, factorization residuals and
same-person shares per block/annotation pair. New pair files additionally
store full and deleted `omega_xx`/`omega_yy`, the joint primitive covariance,
`deletion_method`, `uncertainty_method`, and separately named
`*_delta_covariance` and `*_jackknife_covariance`. `*_covariance` is the selected
method. Tables distinguish `standard_error` from `jackknife_se`; interval
endpoints use the selected standard error.

Long studies may add `--checkpoint-every-blocks 128 --max-run-seconds 46800`.
This writes immutable `.study_checkpoint` artifacts every 128 decoded tiles
and exits with code 75 at a completed tile boundary when the time budget is
reached. `--resume-from PATH` restores the running sums and begins at the next
unread SNP. It checks input, Python-source and native-binary identities,
residual geometry and the guarded traversal ledger. Partial files are never
published as checkpoints. The final receipt records contiguous process
segments, so a graceful continuation retains one genotype traversal and the
original floating-point addition order. An unexpected kill can lose the
uncheckpointed tail; any replay of that tail must be reported as aborted work,
not silently included in the successful traversal claim. A final checkpoint
also precedes score/Z publication when checkpointing is enabled.

Run paths below are placeholders for authenticated inputs and **new** outputs:

```bash
summit reference zpass --manifest "$REF/MANIFEST.json" \
  --reference-root "$REF" --chromosome 22 --master-input "$MASTER" \
  --bed-prefix "$BED22" --annotations "$ANN22" --output "$NEW/zpass_chr22.npz" \
  --num-threads 8 --width 128

python scripts/generalized_gxe/cross_trait_study.py study \
  --base "$BASE" --bed-prefix "$BED22" --annotations "$ANN22" \
  --chromosome 22 --traits 8 --common-only --threads 8 \
  --output "$NEW/pilot_study/chr22" --z-output "$NEW/zpass_chr22.npz"

python scripts/generalized_gxe/cross_trait_refit.py --base "$BASE" \
  --output "$NEW/within_refits"

python scripts/generalized_gxe/cross_trait_pilot_fit.py --base "$BASE" \
  --study-root "$NEW/pilot_study" --z-root "$NEW" --output "$NEW/pilot_fits"

python scripts/generalized_gxe/cross_trait_pilot_report.py \
  --fits "$NEW/pilot_fits" --output "$NEW/pilot_report"

python scripts/generalized_gxe/cross_trait_bivariate.py --base "$BASE" \
  --study-root "$NEW/pilot_study" --pilot-fits "$NEW/pilot_fits" \
  --output "$NEW/baseline_regression"

python scripts/generalized_gxe/cross_trait_audit_pilot.py --repo "$PWD" \
  --fits "$NEW/pilot_fits" --report "$NEW/pilot_report" \
  --expected-blocks 200 --output "$NEW/pilot_audit"

python scripts/generalized_gxe/cross_trait_pilot_figures.py \
  --fits "$NEW/pilot_fits" --report "$NEW/pilot_report" \
  --audit "$NEW/pilot_audit" --output "$NEW/pilot_figures"
```

The pilot audit authenticates all 112 pair/mode artifacts and independently
recomputes the 5,936 estimates and paired intervals, covariance matrices,
1,484 mode shifts, and the named age/BMI report. It reads no genotypes and
does not refit. `cross_trait_audit_simulation.py` similarly recomputes every
simulation bias, RMSE, SE calibration and coverage value from the recorded
replicates and checks undefined correlations against their marginal variances.
The figure command requires the complete 22-chromosome, 200-block audit. It
exports PDF and PNG versions of the paired response-minus-baseline intervals
and the age–BMI covariance matrix, with file hashes and renderer versions.
Matrix rows denote the trait's age response and columns its BMI response;
the diagonal uses within-trait H. Intervals are nominal and exploratory.

Production guarded native runs use the qualified placement launcher
`scripts/generalized_gxe/private_python.py`; the Hoffman launcher is
`cross_trait_h2.sh`. Thread count variables alone do not qualify placement.
The study command defaults to eight traits and all three annotations unless
`--common-only` is supplied. `benchmark --traits 6` and `benchmark --traits 42`
time within-only and complete cross passes on each decoded tile, report peak
RSS and separate score and residual costs, and omit the first timing tile.
`--residual-workers 8` distributes exact pair residual contractions across
eight explicitly reserved CPUs while keeping each worker's NumPy BLAS
single-threaded. Recorded worker affinities must match those reservations.
Trait fixed bases are streamed at initialization and their row norms are
reused across pairs. Missing-row caches are exact work-sharing optimizations:
`--max-cached-missing-patterns` bounds the number of groups and
`--minimum-cached-pattern-rows` controls their minimum size. The optional
`--missing-cache-strategy pooled` also combines distinct patterns into
disjoint groups. A group serves a pair only if every member misses both
traits; uncached rows are still contracted exactly. Its
`--cache-addition-penalty-rows` controls a mask-only cost model, not a
statistical approximation. Defaults retain the individual-pattern cache.
`--parallel-cache-products` optionally uses the same reserved worker pool
for independent cached products, with one BLAS thread per worker. Every task
is joined before the process-wide BLAS limit is restored.
`--max-cached-pair-sums` optionally bounds reusable sums of identical cached
group combinations across trait pairs (default zero). These sums preserve
the same exact overlap contraction, with possible floating-point regrouping;
they do not approximate the masks or require another genotype traversal.

`cross_trait_bivariate.py` compares both context-model baseline centerings
with the existing ordinary SUMMIT HE fitter. It profiles baseline reference
moments through `joint_chromosome_equations` and preserves all SNP/block
counts. Expanding block means reproduces the original unweighted fitter's
sufficient statistics and deletions exactly; it does not reconstruct
per-SNP observations and cannot support SNP filtering or IRWLS. The ordinary
baseline-only model and a conditional GxE baseline need not agree under
nonzero context effects. Both comparisons and any one-SE failures are retained.

## Validation recorded on 23–24 September 2026

The detailed report, commands and checksum inventory are
`~/UKBB/manuscript/general_gxe_method/cross_trait_20260923/ROUND2_REPORT.md`,
`ROUND2_COMMANDS.md` and `ROUND2_VERIFICATION.json`; artifacts are in the
adjacent `round2/` directory.
These results do not by themselves establish all acceptance criteria.

| Check | Observed result |
|---|---|
| Portable suite including checkpoint/resume, pair-sum reuse and deletion mass restoration | 1,478 passed; 6 skipped; 1 xpassed |
| Checkpoint/resume through separate OS processes | Score and Z arrays bit-identical to uninterrupted traversal; changed tile width rejected; both BLAS configurations pass |
| Legacy within-trait full/deletion regression | Bit-identical |
| Legacy fits against all 22 available original saved study fits | Normal matrix, RHS, coefficients, Omega and raw deletion coefficients bit-identical; restored deletion matrices match the existing collector |
| Dense oracle, N=2,000, M=3,000, Q=3, about 60% overlap, fixed ranks 5 and 4 | Genetic Gram relative error 2.37e-15; coefficient error 3.23e-14 |
| Dense real chr22 reference, N=20,000, 41,275 common SNPs | Z repair relative error 2.26e-15 |
| Dense real chr22 reference, N=40,000, same 41,275 common SNPs | Z repair and production assembly errors 1.60e-15; 4,604.46 accounted seconds; RSS 28,357,036 KiB |
| Large simulation, N=50,112, M=454,207, 100 replicates per scenario | Default passes 33/38 summary gates and 17/18 Omega gates; full coverage acceptance fails (details below) |
| Q=1 baseline regression against bivariate SUMMIT | Agreement at 1e-12 on the same dense panel |
| Real chr22 downstream qualification, all eight traits | 112 pair/mode artifacts authenticate; rank 38 throughout; 28 contrasts and 112 age–BMI entries published |
| Real chr22 ordinary-bivariate comparison | All 28 pairs and four modes within one SE for both baseline centerings; maximum 0.076 SE; only three deletion blocks |
| Full-genome eight-trait common-bin pilot | 112 pair/mode artifacts, 200 paired blocks; independent audit verifies all 5,936 estimate/interval rows, 1,484 maximum-mode-shift rows and 112 age–BMI entries; rank 38 throughout, condition numbers 70.38–72.04 |
| Full-genome ordinary-bivariate regression | All 224 comparisons pass one context-model SE; maximum raw-baseline difference 0.03650 SE, centered-baseline difference 0.04105 SE |
| 42 traits × six within-trait arms, full-genome same-person diagonals and restored deletions, no genotype pass | 252 fits completed and authenticated; 142 comparison rows (71 distinct entries) above one SE across 11 traits; maximum 7.48 SE |
| Six-trait chr22 timing after streamed inputs, five measured tiles | 2.578 versus 2.229 s per 128-SNP block; 15.67% total increment; RSS 5,696,408 KiB |
| Eight-core 42-trait timing, 256 individual-pattern groups with minimum 16 rows, five measured tiles | 15.092 versus 10.516 s per 128-SNP block; 43.51% increment; RSS 5,943,260 KiB |
| Eight-core 42-trait timing, 256 pooled groups, minimum 16 rows, addition penalty eight | 14.574 versus 10.275 s per block; 41.84% increment; RSS 5,844,816 KiB |
| Same pooled cache, products built on eight reserved residual workers | 14.391 versus 10.459 s per block; 37.59% increment; RSS 6,182,660 KiB (fails 30% target) |
| Eight-core pooled cache, ceiling 512, minimum eight rows (331 eligible groups) | 15.459 versus 11.218 s per block; 37.80% increment; RSS 6,280,056 KiB |
| Eight-core pooled cache with 64 reused pair sums, 31 measured tiles on CPUs88–95 | 13.821 versus 10.635 s per block; 29.956% total increment; RSS 6,187,152 KiB; narrowly meets target |
| Longer 42-trait check, 31 measured tiles on CPUs96–103 | 27.477 versus 18.465 s per block; 48.81% increment; RSS 6,167,092 KiB; most memory allocated across sockets |
| 42 traits, 16 BLAS threads and 16 residual workers | 24.730 versus 22.100 s per block; 11.90% increment, but slower in absolute time than eight cores |
| 42 traits, eight BLAS threads and 16 residual workers on 16 reserved CPUs | 18.187 versus 13.363 s per block; 36.10% increment |
| 42-trait score accumulation alone, individual-pattern cache | 0.0192% of within-trait time; exact overlap residual work dominates |
| Block same-person apportionment, chr22 | Maximum relative LD-scalar error 0.000151906 (passes 0.001) |
| Hoffman placement qualification 14886620 | 20 tests passed; eight physical cores; NumPy and native worker affinities verified |
| Private-BLIS prediction and cross-trait checks at b271d90 | 95 passed |
| Private-BLIS batched residual checks at f00fa16 | 19 passed |
| Private-BLIS complete study publication, Z and batch checks at fe7bf6a | 20 passed |
| Private-BLIS optional pooled-cache and study publication checks at 5fc39bf | 20 passed |
| Private-BLIS parallel cache and study publication checks at 04515d0 | 23 passed |
| Reused pair-sum cache and study publication at 22f655c | 26 portable and 26 private-BLIS checks passed |
| Standalone chr22 Z, local versus Hoffman | Block products agree to 3.29e-16 relative; Hoffman traversal 651.87 s, one pass |
| Completed fused chr22 study/Z, eight traits | 99,273 SNPs in one traversal; 776 guarded TN calls, zero repairs; 3,792.20 accounted s; RSS 6,174,252 KiB |
| Completed fused study/Z across all 22 chromosomes | 7,774,235 SNPs; 60,747 guarded TN calls; zero repairs; one contiguous traversal per chromosome; 200 paired blocks; 354,263.502 aggregate accounted seconds |
| Fused versus standalone chr22 Z | Block products agree to 7.91e-15 relative; global products to 7.85e-15 |
| Reference probe tiling, N=50,112, M=1,024, B=1,024 | Tile 4 versus 128: 33.11 versus 13.70 s; Gram difference 1.65e-16 relative; same-person matrix bit-identical |

The 142 flagged within-trait comparison rows cover 71 distinct
trait/annotation/quantity entries: 102 rows are rare-bin, ten low-frequency
and 30 common-bin. Holding factorization fixed, the scaled versus actual-row
same-person choice shifts rare-bin testosterone Omega[0,3] by 5.07 SE.
Holding actual-row diagonals fixed, legacy versus factorized transport shifts
common-bin FVC Omega[0,1] by 3.44 SE. Factorized and plus-residual fits with
actual-row diagonals differ by at most 0.1492 SE across all reported entries.
These are observed mode sensitivities, not a guarantee of unbiasedness.

Real-mask Gram relative Frobenius errors at N=20,000:

| Mask pair | Factorized | Plus residual | Legacy | Z-repaired legacy |
|---|---:|---:|---:|---:|
| FEV1–FEV1 | 0.004745 | 0.004493 | 0.044780 | 0.044745 |
| LDL–LDL | 0.004316 | 0.001133 | 0.003688 | 0.003665 |
| FEV1–LDL | 0.004515 | 0.003241 | 0.029800 | 0.029773 |

Factorized transport improves the selected FEV1 mask, but does not outperform
legacy in Frobenius error for LDL alone. This result is retained, not filtered
out. The exact orientation repair cannot correct population-selection error.

At N=40,000, factorization improves all three selected comparisons:

| Mask pair | Factorized | Plus residual | Legacy | Z-repaired legacy |
|---|---:|---:|---:|---:|
| FEV1–FEV1 | 0.003414 | 0.003230 | 0.044525 | 0.044516 |
| LDL–LDL | 0.002704 | 0.001028 | 0.003831 | 0.003824 |
| FEV1–LDL | 0.003051 | 0.002332 | 0.029719 | 0.029712 |

The corresponding factorized coefficient errors are 0.003335, 0.002122 and
0.002844, compared with legacy errors 0.029885, 0.004954 and 0.037015.
The run report includes every signed Gram-entry error at both sample sizes
(7,500 entries each), audited against the saved dense matrices and summary
norms without another genotype pass. The N=20,000 LDL result remains a
counterexample to a universal improvement claim.

The reproducible simulation driver specifies nonsymmetric cross effects,
nonzero H, a positive-baseline/zero-H scenario after cohort centering, and
heteroskedastic correlated residuals. Generation and scoring each use one
traversal for all 100 replicates. The small complete BED pipeline is tested.
The original August simulation reference is obsolete under current
same-person validation, so the large benchmark constructs a new guarded
reference instead of bypassing that check. The reference and all simulation
phases have now completed and their artifacts authenticate. The genome-wide
pilot also completed; its exploratory findings are below. The complete eight-core 42-trait
pass with 64 reused pair sums adds
29.956% in 31 measured tiles, narrowly below the 30% target. This small
margin does not establish a robust bound across hosts or load conditions.
The 16-thread configuration has a
smaller percentage increment because its within-only pass is substantially
slower; it does not establish an absolute throughput improvement. All timing
configurations and CPU budgets are retained in the run report (21 completed
arms). The longer run's local NUMA nodes had little free memory; observed
remote allocation is a possible contributor, not an isolated causal result.
Streaming fixed bases and releasing unused overlaps reduced peak RSS from
approximately
11.8 million to 5.9 million KiB.

The prespecified factorized simulation arm fails the full coverage criterion.
Seventeen of 18 Omega entries meet 0.90–0.98 coverage; the remaining entry
has 0.99. Two cross-context H entries also have 0.99 coverage. The two
response-correlation coordinates in the shared-program scenario each have
99 valid replicates: a full-fit marginal response variance is negative in
replicate 41 (first context, Y) or 67 (second context, X). Undefined ratios
are retained; they are not clipped or removed from the acceptance denominator.
Overall, 33 of 38 summary-row gates pass.

Across the 18 Omega entries, bias ranges from -0.003419 to 0.000941, RMSE
from 0.010144 to 0.013210, and mean jackknife SE / empirical SD from 0.930
to 1.143. Orthogonal-response correlation coverage is 0.90 and 0.92 in the
shared and zero-program scenarios, respectively; SE calibration is 0.788
and 0.823. These uncertainty estimates need further qualification for weak
response variances. Diagnostic plus-residual and legacy refits of the same
saved summaries each pass 32/38 gates and retain the same undefined ratios.
They do not replace the prespecified default or change the generating truth.
The run report includes all entries and replicate values with SHA-256 hashes.

The simulation reference CLI exposes `--probe-tile-width` (default four).
This controls the existing native pass-2 product grouping, with no change to
the probe identities, count or two-pass estimator. The Hoffman simulation
launcher uses the qualified width 128. The bounded real-panel comparison
tested widths 4, 16, 64 and 128 at unchanged B=1,024; portable and private
native tests also compare their reference arrays. It is a tiling throughput
check, not evidence that the large simulation has completed or met coverage.

The first full-chromosome fused pilot traversed all 99,273 SNPs in 78.3
accounted minutes but failed artifact publication: an unmeasured timing was
represented as NaN in canonical JSON provenance. No reusable score or fused
Z artifact was published. Study timings now use JSON null, invalid provenance
is rejected before opening an artifact, and a complete-driver test exercises
publication and numerical read-back. The failed run is preserved. After its
full-chromosome traversal timing, a dependent array was submitted using that
measured workload. Each task authenticates the replacement chr22 completion
receipt, score/Z arrays and guarded traversal ledger before any genotype
work; scheduler dependency release alone is insufficient. The replacement
completed and authenticated both artifacts. Its full numerical pass averaged
4.458 seconds per 128-SNP tile, plus 0.312 seconds decoding; this run did not
measure a within-only comparator. After this qualification, chromosomes 1–21
were submitted with the same code and binding, eight cores, 16 GiB per task,
14 hours and no task-concurrency cap. All production tasks completed without
needing to resume a partial traversal.
All 22 traversals have now completed and authenticate. The independent
`cross_trait_audit_traversal.py` checks common cohort identities, all 200
block IDs, global annotation mass, guarded call counts and successful
eight-slot accounting. Its execution table records each chromosome's
runtime, peak RSS and score/Z hashes. These counts describe the successful
production run; the aborted attempts above remain part of the run history.

The chr22-only downstream qualification also runs all four fitting modes,
the paired report and the ordinary-bivariate comparison without genotypes.
It uses deletion blocks 197–199 and is explicitly labeled as chromosome 22
in its output paths and report scope. These artifacts verify the real input
schema and numerical chain; they are not the requested full-genome,
200-block pilot and do not establish genome-wide uncertainty calibration.

The corrected refit table is
`within_refits_full_diagonal_mass_restored/within_trait_mode_comparison.tsv`.
The older `within_refits` and `within_refits_full_diagonal` runs are preserved
as superseded outputs and must not supply paper uncertainty estimates.
The latter omitted the final post-solve coefficient restoration; its point
estimates and normal equations are unchanged. The corrected simulation
tables are under `simulation_v3/mass_restored_*`. Shared per-block diagnostics cover 221 chromosome fragments
of the 200 deletion blocks. Their maximum factorization residual is 20.1%
for a 285-SNP chr2 fragment (common-to-rare annotation pair); chromosome-average
errors are not bounds on individual block residuals.

The pilot hypothesis was recorded before fitting: a shared age–BMI response
direction among LDL, ApoB, total cholesterol, non-HDL, HbA1c and DBP, absent
for height and platelets. All 28 pairs are retained. Pilot findings are
exploratory; a nonsignificant control estimate does not establish absence.
The common-only model omits lower-frequency effects and is reported as such.

## Completed common-bin pilot

The successful production run visited 7,774,235 SNPs once per chromosome for
all eight traits and their 28 pairs, collecting Z moments in the same pass.
The common annotation contains 3,202,459 SNPs. All four modes were fitted
from these summaries with 200 paired, mass-restored deletions. Job 14888839
completed the fitting, report and bivariate comparison in 2,446.585 accounted
seconds with 2,734,568 KiB peak RSS and no genotype traversal. Independent
audits authenticate the traversal ledgers, array checksums, restored
coefficients, uncertainty tables and baseline-regression replicates.

Selected factorized/actual-overlap results are below. The last column uses
the paired jackknife for the difference; it is not a difference of separate
interval endpoints. Baseline rg uses the original master-context intercept.

| Trait pair | Baseline rg | Orthogonal-response rg | Paired difference, nominal 95% interval |
|---|---:|---:|---:|
| LDL–ApoB | 0.959 | 0.968 | 0.009 [-0.016, 0.034] |
| LDL–HbA1c | 0.046 | -0.297 | -0.342 [-0.631, -0.053] |
| LDL–DBP | -0.077 | 0.357 | 0.434 [0.225, 0.642] |
| Cholesterol–non-HDL | 0.907 | 0.960 | 0.053 [0.013, 0.093] |
| LDL–height | -0.107 | -0.107 | -0.000 [-0.279, 0.279] |
| LDL–platelets | 0.070 | 0.233 | 0.163 [-0.219, 0.545] |

Lipid–BP response alignment exceeds baseline alignment, while lipid–HbA1c
orthogonal-response estimates are negative. Among the 15 cardiometabolic
pairs, five difference intervals are above zero and three below zero; all
13 control-involving difference intervals include zero. These results do
not support a uniformly positive six-trait response program. Nor do they
establish absence in the controls: eight of 52 control-involving age/BMI
covariance intervals exclude zero nominally. No multiplicity correction is
applied, and the lipid traits are closely related.

The named, centered covariance entries retain direction. For example,
LDL-age with HbA1c-BMI is -0.001946 (95% interval -0.003535 to -0.000358),
and LDL-age with DBP-BMI is 0.001788 (0.000225 to 0.003351).
LDL-age with height-BMI is -0.001571 (-0.002720 to -0.000423), and with
platelet-BMI is 0.001724 (0.000403 to 0.003044). These are covariance
entries in the study's standardized named-exposure basis, not correlations.

Across finite default-SE comparisons, maximum shifts among the four modes
are 0.371 SE for Omega, 0.044 SE for H, 0.0242 SE for baseline rg,
0.0152 SE for orthogonal-response rg, and 0.0105 SE for their paired
difference. This stability does not resolve the failed simulation coverage
gate or establish that all modes are unbiased.

All 28 baseline and orthogonal-trace correlations have finite intervals and
point estimates within [-1,1]. Individual-context correlations are less
stable: of 112 coordinates, 18 smoking-response point estimates are
undefined, 34 jackknife intervals are undefined (27 smoking and seven BMI),
and 14 finite point estimates lie outside [-1,1]. The estimator imposes no
PSD constraint or clipping. Nonpositive estimated marginal variances make
ratios undefined; the tables retain these cases and identify undefined mode
shifts. Together with the simulation result, this limits paper inference
about weak individual-context responses.

The final artifacts are under `round2/pilot_fused_v3/` in the report tree:

- `pilot_fits/pilot_estimates.tsv`: every Omega and H entry and derived
  quantity, with paired uncertainty in all four modes; the corresponding
  112 NPZ files include deletion estimates, covariance and provenance.
- `pilot_fits/pilot_maximum_mode_shifts.tsv` and
  `pilot_gram_mode_differences.tsv`: sensitivity and Gram diagnostics.
- `pilot_report/baseline_vs_orthogonal_response.tsv` and
  `age_bmi_cross_exposure_covariance.tsv`: all 28 paired comparisons and
  112 named age/BMI entries under the default.
- `baseline_regression/baseline_rg_regression.tsv`: all 224 ordinary-SUMMIT
  comparisons, with authenticated replicate arrays beside the table.
- `pilot_figures/`: audited forest plot and oriented age/BMI covariance
  matrix, both PDF and PNG, with figure hashes and renderer versions.
- `traversal_audit/`, `pilot_audit/` and `FINAL_SUMMARY.json`: independently
  checked execution, numerical and report receipts.

The complete suite passed 1,478 tests (six skipped, one xpassed) in 153.61
seconds at commit 65105dd; subsequent changes are documentation only.
Every milestone push was retried but GitHub returned HTTP 403. Local SHAs
and exact commands are recorded in the report. No remote SHA or Actions
conclusion is claimed for these unpublished commits.

## Calibration investigation, 24 September 2026

The branch was subsequently pushed successfully using the conda `summit`
GitHub credential helper. Remote commit `8207302f66fc485eba785303fa1202bf2e672dd7`
passed [SUMMIT core tests](https://github.com/sriramlab/SUMMIT/actions/runs/36068248273).
The earlier HTTP403 statements above describe the original publication attempts.

The saved 100-replicate simulations identify two effects in the aggregate
orthogonal-response correlation. First, restored deletion coefficients have a
systematic center shift under the inherited frozen same-person convention.
For uniform target blocks with fraction f, let A be the full genetic Gram after
profiling the residual span and D the frozen same-person Gram. The current
single-chromosome, single-annotation equations imply

```
restored_theta_delete = solve(A - f*D, A @ theta_full)
```

This is an exact algebraic identity for uniform sufficient statistics, not a
claim that real blocks are uniform. A regression test verifies it independently.
Using the actual simulation geometry predicts the observed mean coefficient
shift with relative errors 5.70e-5 (XX), 5.08e-5 (YY) and 2.15e-4 (XY).
The shared-response scenario's mean orthogonal variances change from
0.07317/0.08499 at the full fit to 0.09050/0.10419 at the deletion center.
Differentiating the ratio at these inflated denominators compresses its SE.

Second, the nonlinear ratio has greater sampling dispersion than its
linearization at the generating truth. The nonlinear/linearized SD ratios are
1.1845 and 1.1035 in the shared-response and zero-H scenarios. By comparison,
RMS jackknife SE divided by the empirical SD of that *same linearized*
estimator is 1.0493 and 1.0428. Thus the observed ratio problem is not evidence
of a uniformly underestimated covariance matrix for the primitive coefficients.

`scripts/generalized_gxe/cross_trait_calibration_diagnostics.py` authenticates
saved fits and compares the original intervals, recentered deletion
coefficients, and a full-fit delta method retaining the joint XX/YY/XY
covariance. It also reports a truth-linearized diagnostic. No genotypes are
read, fits changed, or invalid replicates discarded from coverage denominators.
For the default Gram mode:

| Scenario | Original RMS SE / SD | Original coverage | Full-fit delta RMS SE / SD | Delta coverage |
|---|---:|---:|---:|---:|
| Shared response | 0.808 | 90/100 | 1.127 | 98/100 |
| Zero H, positive baseline covariance | 0.838 | 92/100 | 1.084 | 98/100 |

These are diagnostic comparisons on the existing replicates, not independent
validation of a replacement interval. The delta method still overshoots in
RMS and can fail near nonpositive response variances. A defensible next step
is to qualify a full-estimate influence/delta calculation and an estimating-
equation-consistent deletion convention on new seeds and multiple sample sizes,
with test inversion or validated parametric bootstrap for weak denominators.
Retain the legacy convention explicitly. Do not apply a universal SE multiplier,
clip correlations, or silently omit invalid draws. Reference/probe uncertainty
and transport bias require separate assessment; these simulations condition on
one genotype/exposure/reference realization.

A narrower pilot diagnostic applies the delta method to the saved paired
trace covariance C, variances Vx/Vy and baseline rg at their full-fit values.
It changes aggregate SEs by -0.6% to +2.0% across 28 pairs. It does not recompute
the H Jacobian from primitive within-trait coefficients, and therefore is not
a complete validation of full-cohort uncertainty. The LDL–DBP contrast changes
from 0.4338 [0.2252,0.6424] to 0.4338 [0.2253,0.6423].

The reproducible investigation and biological comparison are in
`cross_trait_20260923/calibration_20260924/REPORT.md`, with commands, source
hashes, all-entry simulation tables and the 28-pair pilot diagnostic. Original
fit files and intervals remain unchanged.


## Directional deletion and delta validation (24 September follow-up)

A population-moment oracle with heterogeneous directed blocks and unequal
annotation fractions exposed an additional error in symmetrizing individual
target/source equations. The corrected SVD path preserves the generating
coefficients at 1e-12; proportional-block, multi-chromosome, residual-profile,
cache-equivalence and paired-covariance tests also pass. Legacy assembly
remains bit-identical. The portable suite at the directional milestone passed
1,496 tests (six skipped, one xpassed); later publication receipts identify
the exact final test count and commit.

On the original 100-replicate panel, corrected directional deletions plus
full-fit delta propagation give aggregate orthogonal-response correlation
coverage of 0.97 and 0.98 in the shared- and zero-program scenarios. Mean
SE/empirical SD is 0.981 and 1.008; RMS SE/SD is 1.072 and 1.037. These
are diagnostics on the simulations used to identify the problem, not
independent validation. Some other entries still have 0.99 coverage, and
one replicate has an undefined individual-exposure correlation in the
shared-program scenario. Do not omit undefined intervals when describing
coverage. Batched fitting of all 100 replicates and both scenarios took
3.8 seconds locally, without reading genotypes.

The follow-up protocol specifies seed 2026092402 and 500 independent replicates
per scenario on the existing 50,112-person array panel, and 11 requested real-data
pairs over 15 traits. The new pairs expand the Namba single-exposure comparison
beyond LDL/total cholesterol and test whether the HbA1c pattern generalizes
to glucose. Job completion and prospective results must be read from the
follow-up report, not inferred from successful submission:
`cross_trait_20260923/round3_20260924/`.

`cross_trait_biology.py` exports baseline versus aggregate-response forests,
four single-exposure panels, an ordered age/BMI correlation matrix, and
matched Namba comparisons. Its numerical tables include paired asymmetry
contrasts, conditioning on both baselines as a sensitivity analysis, and
post-hoc lipid averages with covariance across all contributing pairs.
The four lipid measurements are correlated outcomes, not four independent
replications. The single aggregate response correlation measures alignment
of the two residual genetic response functions under the recorded exposure
covariance metric; it is not the arithmetic mean of four exposure-specific
correlations. Positive sharing does not identify the direction of the mean
phenotypic response to an exposure or establish a causal intervention effect.

External comparisons use ordinary same-exposure slope correlation, without
baseline orthogonalization. Namba et al.'s Table S22 contains significant
entries only. UKB comparisons overlap samples; BBJ is an independent cohort
with different ancestry and ascertainment. Exposure sets, phenotype transforms
and SNP panels differ. Current smoking is not equated with ever smoking, and
missing table entries are not null results. Nonsignificant height or platelet
contrasts do not establish absence of response sharing.
