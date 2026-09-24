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
`reconstruct(saved(T-T^A)) + T^A`. Frozen target blocks use their symmetric
part, which is what enters the normal equation. Dense tests establish this
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

The paper convention uses 200 paired target-SNP deletion blocks: both traits'
score products and matching reference target blocks are deleted together.
Genetic moments are mass-restored, reference source products stay frozen,
and the own-overlap same-person term stays frozen. For the full-genome
same-person term, chromosome diagonals are summed **before** taking their
Gram; this includes same-person products between chromosomes. Full-data
different-person contributions are summed across chromosomes. Deletions
retain the existing frozen-source residual-profile change relative to its
full-data value, using the study's residual geometry. Cross-chromosome LD is
not computed. These deletion and chromosome conventions are approximations,
not dense recomputation after removing both SNP axes.
These intervals are conditional on the supplied reference and transport
mode. They do not include independent probe redraws or transport-model
uncertainty; the mode comparison reports that additional sensitivity.

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
same-person shares per block/annotation pair.

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
```

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

The detailed run report and table checksums are under
`~/UKBB/manuscript/general_gxe_method/cross_trait_20260923/round2/`.
These results do not by themselves establish all acceptance criteria.

| Check | Observed result |
|---|---|
| Portable suite including checkpoint/resume and pair-sum reuse | 1,477 passed; 6 skipped; 1 xpassed |
| Checkpoint/resume through separate OS processes | Score and Z arrays bit-identical to uninterrupted traversal; changed tile width rejected; both BLAS configurations pass |
| Legacy within-trait full/deletion regression | Bit-identical |
| Dense oracle, N=2,000, M=3,000, Q=3, about 60% overlap, fixed ranks 5 and 4 | Genetic Gram relative error 2.37e-15; coefficient error 3.23e-14 |
| Dense real chr22 reference, N=20,000, 41,275 common SNPs | Z repair relative error 2.26e-15 |
| Dense real chr22 reference, N=40,000, same 41,275 common SNPs | Z repair and production assembly errors 1.60e-15; 4,604.46 accounted seconds; RSS 28,357,036 KiB |
| Q=1 baseline regression against bivariate SUMMIT | Agreement at 1e-12 on the same dense panel |
| Real chr22 downstream qualification, all eight traits | 112 pair/mode artifacts authenticate; rank 38 throughout; 28 contrasts and 112 age–BMI entries published |
| Real chr22 ordinary-bivariate comparison | All 28 pairs and four modes within one SE for both baseline centerings; maximum 0.076 SE; only three deletion blocks |
| 42 traits × six within-trait arms, full-genome same-person diagonals, no genotype pass | 252 fits completed and authenticated; 142 comparison rows (71 distinct entries) above one SE across 11 traits; maximum 7.52 SE |
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
| Fused versus standalone chr22 Z | Block products agree to 7.91e-15 relative; global products to 7.85e-15 |
| Reference probe tiling, N=50,112, M=1,024, B=1,024 | Tile 4 versus 128: 33.11 versus 13.70 s; Gram difference 1.65e-16 relative; same-person matrix bit-identical |

The 142 flagged within-trait comparison rows cover 71 distinct
trait/annotation/quantity entries: 102 rows are rare-bin, ten low-frequency
and 30 common-bin. Holding factorization fixed, the scaled versus actual-row
same-person choice shifts rare-bin testosterone Omega[0,3] by 5.09 SE.
Holding actual-row diagonals fixed, legacy versus factorized transport shifts
common-bin FVC Omega[0,1] by 3.45 SE. Factorized and plus-residual fits with
actual-row diagonals differ by at most 0.150 SE across all reported entries.
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
reference instead of bypassing that check. Large-simulation coverage and pilot results
remain pending in this version of the page. The complete eight-core 42-trait
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
14 hours and no task-concurrency cap. Checkpointed partial tasks must resume
before any full-genome fit is reported.

The chr22-only downstream qualification also runs all four fitting modes,
the paired report and the ordinary-bivariate comparison without genotypes.
It uses deletion blocks 197–199 and is explicitly labeled as chromosome 22
in its output paths and report scope. These artifacts verify the real input
schema and numerical chain; they are not the requested full-genome,
200-block pilot and do not establish genome-wide uncertainty calibration.

The corrected refit table is `within_refits_full_diagonal/within_trait_mode_comparison.tsv`.
The older `within_refits` run is preserved as provisional and must not supply
paper estimates. Shared per-block diagnostics cover 221 chromosome fragments
of the 200 deletion blocks. Their maximum factorization residual is 20.1%
for a 285-SNP chr2 fragment (common-to-rare annotation pair); chromosome-average
errors are not bounds on individual block residuals.

The pilot hypothesis was recorded before fitting: a shared age–BMI response
direction among LDL, ApoB, total cholesterol, non-HDL, HbA1c and DBP, absent
for height and platelets. All 28 pairs are retained. Pilot findings are
exploratory; a nonsignificant control estimate does not establish absence.
The common-only model omits lower-frequency effects and is reported as such.
