# Methods

This page collects the definitions needed to interpret the estimates. Usage
examples are in the individual analysis guides.

The binary-trait path has a separate [PCGC guide](Binary-traits-and-PCGC.md)
and [PCGC methods](PCGC-methods.md).
It uses ascertainment-aware risks and off-diagonal moments without the
quantitative variance row described below. Ordinary beta/SE inputs do not
implicitly become PCGC summaries.

Ordered cross-trait G×E equations and their paired uncertainty are described in
[Cross-trait genetic response covariance](Cross-trait-response-covariance.md).
Their directional target-block deletion convention is distinct from the
generalized within-trait convention below.
Study-specific binary/quantitative pairs with distinct bases and projections
use [Generalized cross-trait PCGC](Generalized-cross-trait-PCGC.md).

## h² and rg summary statistics

For a marginal linear-regression coefficient beta, standard error se, and
per-SNP sample size n, SUMMIT reconstructs a score statistic using

```text
z*_j = sqrt(N_max - c - 1) * beta_j
       / sqrt(beta_j² + (n_j - c - 2) * se_j²).
```

Here c is the non-intercept covariate rank. The current h² implementation sets
c=0; rg resolves c from the input metadata. Heritability denominators in rg
retain the h² convention. This distinction matters when comparing implementations.

Write `N* = N_max - c - 1` and `n*_j = n_j - c - 1`. Under the
linear-regression null, the variance of this reference-scaled score is
`d_j = N*/n*_j`. SUMMIT subtracts this SNP-specific null term:

```text
q_j = z*_j² - d_j
D_jk = N* L_jk / M_k.
```

The HE equations use annotation columns as estimating instruments. Equivalently,
the univariate variance equations receive `y_j = 1 + q_j`. Keeping the signal
on the reference scale allows traits to share the reference calculations even
when their SNP sample sizes differ. No average-N approximation is needed.

`--weight-mode ldsc` solves

```text
h = argmin_h ||sqrt(W) (q - D h)||².
```

Its iterative working variances use `d_j + N* h² L_j/M`, with floors applied
only to the weights. This is equivalent to regressing local-N score squares
minus one against `n*_j L_jk/M_k`, with the corresponding local-N weights.

For two traits, let `b_j = sqrt(d1_j d2_j)`. The bivariate response is
`q12_j = z1*_j z2*_j - c_ov b_j`, with design
`sqrt(N1* N2*) L_jk/M_k`. Each covariance estimate is divided by its matching
h² estimates to obtain rg. The overlap covariance is supplied or estimated in
a separate step; the latter uses `b_j` as its intercept column and refits it in
every jackknife replicate. All refits retain the same per-SNP sample scales.

This bivariate model assumes that the overlap covariance on the local score
scale is constant across SNPs. Shared genotype missingness with comparable
missing fractions in both traits and their overlap satisfies this assumption
to the usual large-sample approximation. Different contributing cohorts or
trait-specific missingness can violate it. The two N columns alone do not
determine SNP-specific overlap. Nonoverlapping studies use `c_ov = 0`.

The null subtraction follows the OLS partial-correlation identity; it is not
an exact finite-sample identity for logistic or arbitrary meta-analysis beta/SE
statistics. The usual LD-score signal model and its assumptions still apply.

Working-variance floors affect regression weights. They do not impose
nonnegative h² or bounded rg. For overlapping annotations, component estimates
are coefficient contributions, not estimates restricted to an annotation's SNP set.

### SNP-variance weights

Partial correlations rescale each SNP to unit variance after projection on the
covariates, so the ordinary kernel is `Σ_j x̃_j x̃_jᵀ / M` with every projected
SNP at full weight. The GRM of the genotypes before adjustment, projected on the
covariates as in GCTA or RHE-mc, is `P X Xᵀ P / M`, and `P x_j = sqrt(V_j) x̃_j`
on the scale of the standardized genotypes, with

```text
V_j = (1 - R²_j)(n - 1) / (n - p - 1),
```

one minus the adjusted R² of SNP j on the p covariates. For an infinitesimal
model on the unadjusted genotypes, the ordinary fit targets approximately
`h² (1 - Σ_j w_j R²_j)` with LD-dependent weights `w_j`. This target is below
h² when the covariates explain part of the genotype variance, as in admixed
cohorts.

With V, the LD scores weight partner SNPs by `a_j'k V_j'` and the fit uses
`a_jk V_j` as instruments, which is HE regression with the kernels
`P X_k X_kᵀ P`. The solve gives `σ_k = τ_k A^v_k`, with `A^v_k = Σ_j a_jk V_j`
and τ_k the per-SNP effect variance of the unadjusted genotypes, and the
residual σ_e, both in units of the adjusted phenotype's variance. SUMMIT reports

```text
G_k = τ_k M_k,   M_k = Σ_j a_jk,   V_p = Σ_l G_l + σ_e,   h²_k = G_k / V_p,
```

per-bin overlap h², enrichment and τ with the unweighted masses, and rg from
`γ_k = τ_12,k M_k / sqrt(V_p1 V_p2)`. The conversion is applied in every
jackknife replicate.

## Related individuals

Write `y` for the phenotype and `x_j` for SNP j, both projected on the GWAS
covariates and scaled to sum of squares `N_eff = n - p - 1`, and `K` for the
GRM of these genotypes. With a set P of related pairs, the model is

```text
E[y yᵀ | G] = σ²_g K + σ²_e I + Δ,     Δ_ii' unrestricted for (i,i') in P, zero elsewhere.
```

Δ holds any covariance between relatives beyond the kernel, for example from
untagged variants, household environment or indirect genetic effects. Removing
P from the HE quadratic form,

```text
yᵀ K_u y = yᵀ K y - Σ_{(i,i') in P, ordered} K_ii' y_i y_i',
E[yᵀ K_u y - yᵀ y] = σ²_g {tr(K_u²) - N_eff},
```

gives an estimator that is unbiased for every Δ. Its h²_g equals the
method-of-moments estimate of the two-kernel model σ²_g K + σ²_r K_P + σ²_e I.

The removed term splits over SNPs. With
`s_j = Σ_{(i,i') in P, ordered} x_ij x_i'j y_i y_i'`, the HE-scale moment loses

```text
REL_M_j = κ s_j,     κ = (n - 1)(n - p - 2) / {(n - 2) N_eff²},
```

the derivative of `z*_j²` (h² convention, c = 0) with respect to `(x_jᵀ y)²`
at zero association. The h² response becomes `1 + q_j - REL_M_j`. Without
covariates `(N*/M) Σ_j (z*_j² - REL_M_j) = yᵀ K_u y` holds exactly; with p
covariates the error is of relative order `r_j² p / n`. Because the correction
is per SNP, annotation sums and every jackknife deletion use it unchanged.
`c_rel = mean_j REL_M_j` is the relative-pair covariance in HE scale.

With LD scores from an unrelated sample of the same ancestry composition, the
trace needs no change. SUMMIT's in-sample LD scores satisfy
`N_eff + N_eff² Σ_j l_j / M² = tr(K²)`, which includes the relatives' pairs.
Their share of each LD score is

```text
LD_REL_{j,k} = N_eff⁻² Σ_{(i,i') in P, ordered} x_ij x_i'j W_k,ii',     W_k,ii' = Σ_j' a_j'k x_ij' x_i'j',
```

with `a_j'k V_j'` in place of `a_j'k` for V-weighted LD scores, and
`l_{j,k} - LD_REL_{j,k}` reconstructs the pair-excluded traces exactly.

For rg, the cross-trait moment `z*_1j z*_2j` has the same derivative κ with
respect to `(x_jᵀ y_1)(x_jᵀ y_2)` when both traits use one sample, so

```text
REL_M12_j = κ Σ_{(i,i') in P, ordered} x_ij x_i'j y_1i y_2i'
```

is subtracted from the bivariate response. rg resolves c from the input
metadata; at c = p the derivative differs from κ by a relative `p/n²`. The
overlap covariance concerns each person's own pair of phenotypes and keeps
its usual meaning.

## LD-score Monte Carlo variance

For target SNP i, annotation k, and random vector v, write its contribution as
Y_ikv. With B independent vectors, the conditional variance estimate is

```text
Var_MC(Lhat_ik) = [sum_v Y_ikv² - (sum_v Y_ikv)² / B] / [B (B-1)].
```

The finite-sample null subtraction is constant across vectors and does not
enter this variance. The annotation diagnostic sums these per-SNP variances.
It does not include cross-SNP covariance, reference sampling uncertainty, or
imputation uncertainty.

## Genetic features and covariance components

Let G be the centered and scaled genotype matrix, Phi the context matrix,
and U an orthonormal basis for the fixed-effect design. Projection is
`P X = X - U (U.T X)`.

Generalized contextual features use

```math
F_q = P\,\mathrm{diag}(\phi_q)G.
```

Every context uses the same genotype scale. Multiplication by context occurs
before projection; projecting G first would define a different model.

For annotation weights A_jk with mass `M_k = sum_j A_jk`, the kernels are

```math
K_{k,qq}=F_q A_k F_q^T/M_k,
\qquad
K_{k,qr}=(F_q A_k F_r^T+F_r A_k F_q^T)/M_k,\quad q<r.
```

Pairs are stored as diagonals first, then lexicographic off-diagonals.
Components are annotation-major. Each coefficient is one entry of Ω; the
symmetric off-diagonal contribution is included once in the kernel definition.
Residual components have the form `P diag(d_h) P`.

The default one-environment CLI additionally normalizes projected feature
columns to squared norm `rank(P)`. Its `raw_projected` option retains their
natural norms. Generalized models use the common scale without separate
post-projection normalization.

## Generalized per-SNP reference LD scores

Let `r = rank(P)`, and define `R_ab(j,m) = f_aj.T f_bm / r`. For a context pair
p, O(p) contains its two orientations, or one orientation for a diagonal pair.
The directional score from target pair p to source annotation/pair (ell,s) is

```math
L_{j,p\to(\ell,s)}=
\sum_m A_{m\ell}\sum_{(a,b)\in O(p)}\sum_{(c,d)\in O(s)}
R_{bc}(j,m)R_{ad}(j,m).
```

For components c=(k,p) and d=(ell,s), sum target rows as
`DNUM[c,d] = sum_j A_jk L_j,p→d`. The reference kernel Gram entry is

```math
T_{cd}=\frac{r^2}{M_k M_\ell}
       \frac{DNUM_{cd}+DNUM_{dc}}{2}.
```

The randomized implementation uses variant-axis Rademacher probes. Pass 1
completes all global source sketches. Pass 2 scores target variants against
those completed sketches. Tiling stays within a decoded block, giving exactly
two reference traversals.

Pass 2 also computes component-kernel diagonals d_ci. Their product `d @ d.T`
gives the exact same-person matrix. Reference construction accepts no
jackknife block assignment. A separate reducer subsequently groups the fixed
per-SNP scores into full and delete-block normal equations, updates retained
annotation masses, and reuses the full same-person matrix.

## Reference transfer and fitting

A reference Gram is moved to a study of N_S people by separating its
same-person part D from its different-person part O = T − D. Both depend on
the contexts of the people included, so both are formed for the study's own
people:

```math
T_S=D_S+s\,(T_R-D_R)+\Delta_O,\qquad s=\frac{N_S(N_S-1)}{N_R(N_R-1)}.
```

### Same-person part

A person's kernel diagonal is `d_(a,p)(i) = kappa_p phi_q(i) phi_r(i) h_a(i)`
for annotation a and context pair p = (q, r), with `kappa_p` equal to 1 when
q = r and 2 otherwise, and `h_a(i)` the person's projected squared-genotype mass.
Fits use the first available of three terms.

1. **Exact own rows.** With the reference's sample-aligned diagonals d
   (composable references) and the study's people S given as reference sample
   indices, `D_S = d[:, S] d[:, S]'`.
2. **Fourth-moment difference.** With the context fourth moments
   `G = sum_i u(i) u(i)'`, `u_p(i) = kappa_p phi_q(i) phi_r(i)`, of both cohorts,

   ```math
   D_S\approx fD_R+c\otimes(G_S-fG_R),\qquad f=N_S/N_R,
   ```

   where `c[a,b] = D_R[(a,0),(b,0)] / G_R[0,0]` uses the intercept pair. With
   the constant-one first basis column, `G_R[0,0] = N_R`. Components are
   annotation-major, which fixes the Kronecker order. The form is exact when
   the study is the reference, unbiased for an exchangeable subset, and
   otherwise corrects the difference in context composition. It neglects the
   weak dependence of `h` on the contexts.
3. **Legacy scaling** `D_S = (N_S/N_R) D_R`, used only when neither input is
   available.

The exact term uses diagonals computed with the reference projector; when the
covariates are few relative to N, they differ little from the study projector's.

### Different-person part

`s O_R` assumes the study is an exchangeable subset of the reference. The
composition correction is the difference form of the factorized model
(cross-trait mode `factorized_plus_residual`):

```math
\Delta_O=\sum_b \ell_b\otimes(F_S-sF_R),\qquad
F=\mathrm{saved}(M\otimes M)-G.
```

- `M = sum_i phi(i) phi(i)'` (Q×Q) and G are each cohort's context second and
  fourth moments. `F[p,s]` sums `M[a,c] M[b,d]` over the orientations (a,b) of
  pair p and (c,d) of pair s. It equals the sum over distinct people `i ≠ j`
  of the context products that multiply their kernel entries.
- `ell_b` (K×K) is block b's intercept-pair LD scalar. With the block's raw
  Gram numerator `N_b` (for example `residual_rank^2 * block_directed`) and the
  same-person matrix in raw units `D_raw[a,c] = D[(a,0),(c,0)] M_a M_c`,

  ```math
  \ell_b[a,c]=\frac{N_b[(a,0),(c,0)]-\mathrm{frac}_b[a]\,D_{\mathrm{raw}}[a,c]}
  {N_R(N_R-1)\,m_a m_c},\qquad \mathrm{frac}_b[a]=\frac{A_b[a]}{\sum_{b'} A_{b'}[a]}.
  ```

  `A_b` is the block's annotation mass. The same-person matrix is apportioned
  to blocks by each row annotation's block mass fraction. For chromosome
  references the sum over blocks and the same-person matrix are per
  chromosome. `m` are the annotation masses that normalize the reference Gram.

The correction is exactly zero when the study is the reference
(`F_S = F_R`, `s = 1`), and only corrects the difference in context
composition. A basis with only the constant column gives `F_S = s F_R`, so no
correction.

### Selection, labels and jackknife

Each part uses its corrected form whenever its inputs are present. Otherwise
the fit keeps the legacy term, which is biased when missingness depends on the
contexts. It warns (`SamePersonTransportWarning`,
`DifferentPersonTransportWarning`) when the legacy term is used with N_S ≠ N_R,
unless the caller requested it explicitly. Fits record the choices as
`same_person_transport` (`own_rows_exact`, `fourth_moment_difference` or
`scaled_legacy`) and `different_person_transport` (`composition_corrected` or
`scaled_legacy`).

Approximate jackknife deletions reuse the full study same-person term, as they
reuse the full reference D_R. For the different-person correction each
deletion drops the deleted blocks' `ell_b`. It divides the retained raw
numerators by that deletion's retained annotation masses, the masses that
normalize the deleted reference Gram. Equivalently, `ell_b` at full masses is
multiplied by `(M_a M_c)/(m_a m_c)`.

All inputs are person-summed aggregates (D, G, M and the block numerators)
and can be released with a summary reference.

These factors use sample counts, not residual rank. Genetic–residual and
residual-only moments are computed in the study cohort. Traits are projected
and normalized consistently with the chosen estimator. The resulting small
normal equations estimate genetic and residual coefficients jointly.

The sample-probe estimator in `summit.context.reference_v1` is a separate
method. It estimates aggregate kernel actions and stores grouped numerators;
its summary deletion is approximate. Its finite-probe outputs and deletion
procedure are not interchangeable with the per-variant estimator above.

### Between-block signal covariance

The delete-one-block jackknife treats the genetic moments of different SNP
blocks as independent. For a quantitative trait they are not. The phenotype
contains the genetic value of every block, so the moment of block `b`,
`y' H_u^b y`, contains products of block `b`'s realized genetic values with
those of every other block `b'`. The product for a pair of blocks appears in
the moments of both blocks; deleting block `b` removes only the copy carried by
block `b`'s own kernel. Summed over blocks, these products form the
participants' realized covariance between each context and their squared
genetic value, `sum_i c_ik g_i^2`, which has the form of a G–G×E covariance.
The jackknife therefore understates the variance of the G–G×E covariances
(amplification). G×E variances are not affected to leading order, because the
centered contexts cannot align a block's kernel with its own genetic values,
and the effect on the additive variance is small relative to its LD
fourth-moment variance.

For the residual-profiled genetic moments, the between-block covariance is,
to leading order,

```text
C[(a,u),(c,v)] = 4 sum_{b,d} G[a,b,c,d] sum_i w_u^(b)(i) w_v^(d)(i)
G[a,b,c,d]     = sum_{x != y} abar_x[a,b] abar_y[c,d]
w_qr^(a)(i)    = phi_iq z_r^(a)(i) + phi_ir z_q^(a)(i)   (q != r)
w_qq^(a)(i)    = phi_iq z_q^(a)(i)
z^(a)(i)       = phi_i' Omega^(a) Sigma_phi,   Sigma_phi = sum_i phi_i phi_i' / n
```

`abar_x[a,b] = (n - 1) ell_x[a,b] / (M_a M_b)` is block `x`'s different-person
additive (intercept-pair) trace between annotations `a` and `b`, divided by the
study size. `ell_x` are the per-block intercept LD scalars of the transport
(`intercept_ld_scalars`), which already exclude each person paired with
themselves. The sum over people `sum_i w w` is a contraction of the study's
context fourth moments (`context_fourth_moments`) with `Omega Sigma_phi`. The
corrected covariance of the coefficients is

```text
V = V_jk + B/(B-1) T^-1 [C 0; 0 0] T^-1
```

whose genetic block is `B/(B-1) S^-1 C S^-1`, with `S` the residual-profiled
genetic information. With one annotation, centered contexts and
`Omega = diag(omega_00, 0, ...)`, the added variance of the amplification
`omega_0k/omega_00` reduces to `(1 - sum_x s_x^2) [Sigma_c^-1]_kk / n`, with
`s_x` the block shares of the additive trace and `Sigma_c` the context
covariance: it depends on neither M, LD nor heritability. The jackknife's own
variance shrinks as heritability rises, so the shortfall grows with h2.

Derived quantities use the delta method: the jackknife variance of the derived
quantity over the deletion replicates plus `grad f' A grad f`, with `A` the
added covariance. When several nested models are fitted to one trait, C is
evaluated once at the estimate of the most general model, so the models share
it, and a paired contrast between models subtracts twice their cross term
`B/(B-1) T_a^-1 [C 0; 0 0] T_b^-1`.

All inputs are person-summed aggregates that a fit already has (the LD scalars,
annotation masses, normal matrix, coefficients and the study's context second
and fourth moments), so the term needs no genotype pass and no extra
parameter. It is the default (`standard_errors="crossblock"`) for
`fit_generalized_gxe_variant_model_v1` and
`fit_transferred_chromosome_models`; `standard_errors="jackknife"` keeps the
jackknife alone. Fits record `uncertainty.method`
(`snp_block_jackknife_plus_crossblock_signal_v1`) and `crossblock_status`.
Trait summaries without context moments fall back to the jackknife with a
`CrossBlockSignalUnavailableWarning`. In simulations on 448,562 array
variants and 317,694 participants with nine correlated contexts, the jackknife
z-scores of null amplification had a standard deviation of 1.065 (6.4%
rejected at 5%), and 1.020 (5.4%) with the correction; at h2 = 0.45 the
corrected ratio of the empirical standard deviation to the root-mean-square
standard error was 1.00. The classic one-environment model, binary G×E PCGC
(see [Generalized G×E PCGC](Generalized-GxE-PCGC.md)), h2, rg and cross-trait
fits keep their existing uncertainty estimates.

### Classic one-environment transfer

The one-environment G + G×E + N×E model (`--gxe-population-reference`) uses
the same decomposition with the basis `(1, e)` and only the G = (0,0) and G×E =
(1,1) kernels. Each cohort centers and per-SD scales its own environment,
projects out the intercept, covariates and environment, and rescales every
projected feature to squared norm `r` (the residual rank). For genotypes
independent of `e` the interaction feature is then about `g_ij e_i / sqrt(m2)`
with `m2 = sum e^2 / n`, so the scale of `e` cancels:

- kernel diagonals `K_G(i,i) ≈ (r/n) h(i)` and
  `K_GxE(i,i) ≈ (r/n) h(i) e_i^2 / m2`, giving `u(i) = (r/n)(1, e_i^2/m2)` and
  `G = (r/n)^2 [[n, S2/m2], [S2/m2, S4/m2^2]]`;
- different-person entries `K(i,j) ≈ kappa(i,j) psi(i,j)` with
  `psi = (1, e_i e_j / m2)`, giving
  `F = [[n(n-1), (S1^2-S2)/m2], [., (S2^2-S4)/m2^2]]`.

`S1, S2, S4` are each cohort's power sums of its analysis environment. A
cohort's composition enters only through `n`, `r` and the normalized fourth
moment `n S4 / S2^2`. The fit uses
`D_S = f D_R + kron(G_S - f G_R, c)`, with `c = D_R[G,G] / G_R[0,0]`, and adds
`kron(F_S - s F_R, L)`. Here `L` sums the reference's G–G LD scalars over the
retained jackknife blocks, divided by the retained annotation masses. The
components are ordered G for every annotation, then G×E for every annotation,
which fixes the Kronecker order.

Because each cohort re-standardizes its environment, the different-person
correction is of order `n` against entries of order `n^2`, and is negligible in
practice. The same-person G×E–G×E entry carries the change in the normalized
fourth moment. Under the old scaling it biases the G×E variance
multiplicatively when the environment distribution differs, while a null G×E
estimate stays unbiased. Per-person kernel diagonals are not stored by this
mode, so no exact own-row term is available. With unnormalized features
(`raw_projected`), `u = (1, e^2)` and `psi = (1, e_i e_j)`, the generalized
formulas restricted to the (0,0) and (1,1) pairs.

## PGS posterior mean

For each SNP j, let `beta_j ~ N(0, Lambda/M)`. Let R be a positive diagonal
residual covariance and Z the fixed-effect design. With elementwise matrix
multiplication denoted by ⊙,

```math
V=R+(GG^T/M)\odot(\Phi\Lambda\Phi^T).
```

The solver finds u in the orthogonal complement of Z:

```math
PVPu=Py,\qquad Pu=u.
```

It uses projected conjugate gradients with preconditioner
`P diag(V)^(-1) P`. Accepted solutions satisfy
`||Py - PVu|| <= max(atol, rtol*||Py||)`, checked with a fresh application of V.
Fixed-effect coefficients are recovered from `y-Vu` in the retained fixed-effect span.

Posterior SNP weights are

```math
B=G^T\mathrm{diag}(u)\Phi\Lambda/M.
```

The algorithm requires neither an inverse of Lambda nor an N-by-N matrix.
Singular, additive-only, amplification-only, and zero genetic priors are valid.

## Amplification and residual response

Partition a positive-baseline covariance as

```math
\Omega=\begin{pmatrix}a&b^T\\b&C\end{pmatrix},\qquad
\gamma=b/a,\qquad S=C-bb^T/a.
```

Separate prior scaling uses

```math
\Lambda=\kappa_A a
\begin{pmatrix}1\\\gamma\end{pmatrix}
\begin{pmatrix}1&\gamma^T\end{pmatrix}
+\kappa_H\begin{pmatrix}0&0\\0&S\end{pmatrix}.
```

The response spectrum is computed from `H^(1/2) S H^(1/2)`, with H the saved
reference response metric. Changing its eigenvalues or retained rank changes
the prior and requires refitting. Weighting score components after fitting
leaves the posterior SNP weights fixed.

If a supplied fitted prior changes its baseline or cross-covariance, its
amplification coefficient is derived from that fitted prior. At zero baseline
variance, the declared parent anchor supplies the decomposition and the fitted
baseline weights are zero. Repeated nonzero eigenvalues cannot be split by a rank cut.
