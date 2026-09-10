# Methods

This page collects the definitions needed to interpret the estimates. Usage
examples are in the individual analysis guides.

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

The default HE equations use annotation columns as estimating instruments.
`--weight-mode ldsc` instead uses iteratively weighted LD-score columns. For
annotation mass M_k and `n* = N_max - 1`, its univariate regression is

```text
q_j = z*_j² - 1
D_jk = n* L_jk / M_k
h = argmin_h ||sqrt(W) (q - D h)||².
```

The bivariate version uses `q12 = z1* z2* - c_ov` and design
`sqrt(n1* n2*) L_jk / M_k`. Each covariance estimate is divided by its matching
h² estimates to obtain rg. The overlap covariance is supplied or estimated in
a separate step. When estimated from summaries, it is refitted in each
jackknife replicate.

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
F_q = P\,\operatorname{diag}(\phi_q)G.
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
B=G^T\operatorname{diag}(u)\Phi\Lambda/M.
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
