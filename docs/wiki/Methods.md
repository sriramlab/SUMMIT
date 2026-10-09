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

When reference and study sample the same relevant genotype–context distribution,
separate same-person and different-person contributions scale as

```math
T_S=\frac{N_S}{N_R}D_R+
\frac{N_S(N_S-1)}{N_R(N_R-1)}(T_R-D_R).
```

These factors use sample counts, not residual rank. Genetic–residual and
residual-only moments are computed in the study cohort. Traits are projected
and normalized consistently with the chosen estimator. The resulting small
normal equations estimate genetic and residual coefficients jointly.

The sample-probe estimator in `summit.context.reference_v1` is a separate
method. It estimates aggregate kernel actions and stores grouped numerators;
its summary deletion is approximate. Its finite-probe outputs and deletion
procedure are not interchangeable with the per-variant estimator above.

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
