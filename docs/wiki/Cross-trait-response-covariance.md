# Cross-trait genetic response covariance

This extension estimates whether traits share an genetic responses to
environmental contexts, whether response correlation exceeds baseline genetic
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

## Reference reconstruction and cohort transfer

Ordered pairs use row-major `a*Q+b`. Saved symmetric pairs use
`ContextPairIndex`: diagonals first, followed by lexicographic off-diagonals.
Components are annotation-major. For Q=5 these are 25 ordered or 15 saved
coordinates per annotation.

Reference moments are reconstructed from saved chromosome summaries.
The numerator is scaled as
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
not an observed block diagonal.

The default same-person term uses stored reference diagonals on the actual
overlap, `D_XY = d_overlap @ d_overlap.T`. It is exact for those supplied
diagonals. Substituting them for the two study projectors' diagonals remains
a reference approximation.

| Mode | Different-person term |
|---|---|
| `factorized` (default) | `ell_b F_XY` |
| `factorized_plus_residual` | Default plus reconstructed `O_R,b - ell_b F_R`, scaled by the distinct-person ratio |
| `legacy_transport` | Reconstructed `O_R,b`, scaled by the distinct-person ratio |
| `legacy_transport_exact` | Z-repaired ordered `O_R,b`, with the same population scaling |

The ratio in this table is `(n_X*n_Y-n_overlap)/(N*(N-1))`.
The residual-preserving mode retains reference genotype–exposure dependence
and its sampling/probe noise. The appropriate approximation depends on
the genotype–environment distribution in the reference and study.

Reconstruction is exact for the commuting-tensor approximation, not for an
arbitrary ordered Gram matrix. Within-trait fits use factorized
different-person moments and same-person moments on their own sample rows.

## Z moments and the scope of exact repair

`Z_a = G' D_a U_R` identifies the projector's antisymmetric directional
sector. The artifact stores annotation-weighted target-block products
`V_b,k[a,d] = Z_a,b' diag(annotation_k,b) Z_d,b` and source products
`W_k = sum_b V_b,k`. Target weights are required for annotation-specific
repair; unweighted V alone is insufficient. Storage scales as
`8 * blocks * K * Q² * fixed_rank²` bytes, plus chromosome source products.

The four-trace formula in `antisymmetric_gram` obtains T^A. Corrected
moments are `reconstruct(saved(T-T^A)) + T^A`. This recovers the ordered
moments for a single annotation, including a weighted annotation. It does
not supply every directional block term for a different projector.
Cross-annotation products contain additional mixed terms, so exact
correction of arbitrary annotation pairs is not supported. Z moments also
leave reference sampling and random-vector error unchanged.

`summit reference zpass` computes Z moments from genotypes and the saved
reference scaling. The study driver can also collect them during scoring
with `--z-output`, avoiding a separate genotype pass.

## Fitting, uncertainty and derived quantities

`solve_cross_trait_normal_equations` delegates to the existing rank-revealing
context solver after reducing the residual span. Fits report rank, condition
number, minimum Gram eigenvalue, solve residual and deletion diagnostics.

The default uses 200 paired target-SNP blocks, `deletion_method="target_moments"`,
and `uncertainty_method="delta"`. Write the
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
RHS and is incorrect. The full system uses a symmetric spectral solver.

With exact block moments, the full and deleted coefficient estimators have
the same expectation. Their realized estimates need not agree. Nonlinear
correlations need not be unbiased. The saved reference does not contain
per-target person diagonals: `F_b D` remains a mass-apportionment approximation.
Reference transport, probe noise and omitted cross-chromosome LD also remain
approximations. This is target resampling with fixed source kernels,
not dense recomputation after removing both SNP axes.

The delta method propagates the **joint** paired deletion covariance of
`Omega_XX`, `Omega_YY`, and ordered `Omega_XY` through the full-fit Jacobian.
It includes centring, baseline projection and both denominators. It is not
an independent replacement for estimating coefficient covariance: the 200
paired block summaries are still required. `--uncertainty-method jackknife`
instead propagates the nonlinear deleted
estimates directly. Nonpositive variance denominators remain undefined;
correlations are never clipped to [-1,1].

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
the two denominators. Correlations with
nonpositive estimated denominator variances are NaN. Values outside [-1,1]
are retained and flagged. Within/cross deletion IDs must agree before derived
jackknife covariance is computed. Paired response-minus-baseline contrasts
are included rather than treating those estimates as independent.


## Output and interpretation

Pair fits contain the ordered covariance matrix, residual coefficients,
SNP-block estimates, joint coefficient covariance, and derived correlations.
The selected uncertainty method determines the reported standard errors.
Compare response and baseline correlations using their paired difference,
which includes their covariance.

Context-specific genetic correlation can also be evaluated at chosen
exposure values using the fitted covariance matrices. Keep the original
context means, scales, and covariance metric. These are cross-sectional
model estimates, not longitudinal trajectories or causal intervention effects.

The [usage guide](Cross-trait-analysis.md) describes the Python workflow.
