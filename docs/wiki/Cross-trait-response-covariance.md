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

For large masked cohorts, the fixed-effect column span must be shared on
master rows; its orthonormal restriction and rotation may differ by trait.
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
and the own-overlap same-person term stays frozen. Chromosomes use the
existing residual-profile combination rule; cross-chromosome LD is not
computed. These deletion and chromosome conventions are approximations,
not dense recomputation after removing both SNP axes.

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
```

Production guarded native runs use the qualified placement launcher
`scripts/generalized_gxe/private_python.py`; the Hoffman launcher is
`cross_trait_h2.sh`. Thread count variables alone do not qualify placement.
The study command defaults to eight traits and all three annotations unless
`--common-only` is supplied. `benchmark --traits 6` and `benchmark --traits 42`
time within-only and complete cross passes on each decoded tile, report peak
RSS and separate score and residual costs, and omit the first timing tile.

## Validation recorded on 23 September 2026

The detailed run report and table checksums are under
`~/UKBB/manuscript/general_gxe_method/cross_trait_20260923/round2/`.
These results do not by themselves establish all acceptance criteria.

| Check | Observed result |
|---|---|
| Portable suite at commit 62267ba | 1,446 passed; 5 skipped; 1 xpassed |
| Legacy within-trait full/deletion regression | Bit-identical |
| Dense oracle, N=2,000, M=3,000, Q=3, about 60% overlap | Genetic Gram relative error 5.58e-15; coefficient error 1.40e-13 |
| Dense real chr22 reference, N=20,000, 41,275 common SNPs | Z repair relative error 2.26e-15 |
| Q=1 baseline regression against bivariate SUMMIT | Agreement at 1e-12 on the same dense panel |
| 42 traits × six within-trait arms, no genotype pass | All 252 fits completed; 25,704 comparison rows |
| Within-trait shifts above one default jackknife SE | 85 rows across 11 traits; maximum 7.52 SE |
| Optimized six-trait chr22 timing | 3.317 versus 2.607 s per 128-SNP block; 27.2% total increment |

Real-mask Gram relative Frobenius errors at N=20,000:

| Mask pair | Factorized | Plus residual | Legacy | Z-repaired legacy |
|---|---:|---:|---:|---:|
| FEV1–FEV1 | 0.004745 | 0.004493 | 0.044780 | 0.044745 |
| LDL–LDL | 0.004316 | 0.001133 | 0.003688 | 0.003665 |
| FEV1–LDL | 0.004515 | 0.003241 | 0.029800 | 0.029773 |

Factorized transport improves the selected FEV1 mask, but does not outperform
legacy in Frobenius error for LDL alone. This result is retained, not filtered
out. The exact orientation repair cannot correct population-selection error.

The reproducible simulation driver specifies nonsymmetric cross effects,
nonzero H, a positive-baseline/zero-H scenario after cohort centering, and
heteroskedastic correlated residuals. Generation and scoring each use one
traversal for all 100 replicates. The small complete BED pipeline is tested.
The original August simulation reference is obsolete under current
same-person validation, so the large benchmark constructs a new guarded
reference instead of bypassing that check. Large-simulation coverage,
40,000-person validation, final 42-trait performance and pilot results remain
pending in this version of the page.

The pilot hypothesis was recorded before fitting: a shared age–BMI response
direction among LDL, ApoB, total cholesterol, non-HDL, HbA1c and DBP, absent
for height and platelets. All 28 pairs are retained. Pilot findings are
exploratory; a nonsignificant control estimate does not establish absence.
The common-only model omits lower-frequency effects and is reported as such.
