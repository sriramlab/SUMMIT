# Constrained score-scale LDSC and cov-LDSC

## Scope

`--weight-mode ldsc` fits constrained score-scale LDSC-style IRWLS for h2 and
constrained score-scale cov-LDSC IRWLS for bivariate genetic covariance. rg is
not directly regressed: every full or delete estimate is the corresponding
covariance estimate divided by its two matched h2 estimates. The mode is
supported by ordinary `--h2`, single-pair `--rg`, and regular rg manifests. The
default `--weight-mode he` is unchanged.

Here *constrained* means that the univariate null intercept is fixed at one, or
that the bivariate nuisance intercept `c` is resolved by SUMMIT before the main
covariance regression and is not jointly estimated with the covariance
coefficients. It does not impose non-negativity or boundedness constraints on
h2 or covariance coefficients. Clipping is used only in the working-variance
weights. These definitions retain SUMMIT's score-scale moments and intercept/
delete-refit semantics; they are not literal `ldsc.py` fits.

`--intercept-weight-mode` remains a separate option: it controls how SUMMIT
fits an unknown bivariate nuisance intercept, while `--weight-mode ldsc`
controls the two h2 equations and main cov-LDSC equation after that intercept
has been resolved. Fast cached h2 and fast rg manifests remain HE-only.

## Mean model and weighted estimating equation

For SNP `j`, SUMMIT reconstructs its exact score-scale statistic `z*_j` from
`BETA`, `SE`, and `N`, under the current h2 convention `cov_rank=0`, and uses

```text
q_j = (z*_j)^2 - 1.
```

Let `L_jk` be the LD score for annotation `k`, let `M_k` be that annotation's
fixed mass on the effect-reference SNP universe, and let
`n* = N_max - 1`. Define

```text
D_jk = n* L_jk / M_k.
```

The constrained score-scale LDSC estimate at one IRWLS update is

```text
h = (D' W D)^(-1) D' W q.
```

SUMMIT solves this as least squares on `sqrt(W) D`, using an SVD and requiring
full column rank. It does not solve the explicitly formed normal matrix, which
would square the condition number for correlated annotation columns.

This is a different estimating instrument from the default HE/SUMMIT system.
After eliminating the residual component, the default system uses annotation
columns as instruments, schematically

```text
(A' D) h = A' q,
```

whereas LDSC mode uses `W D` as the instrument. It therefore cannot be
implemented by changing a scalar weight inside the existing HE normal equations.

## Constrained cov-LDSC genetic covariance and rg

For trait `t`, let `n_t* = N_t,max - cov_rank_t - 1`, using the covariate-rank
metadata resolved by the rg path. (The h2 denominator fits retain SUMMIT's
current forced `cov_rank=0` h2 convention.) Let `z1*_j` and `z2*_j` be the
score-scale statistics and let `c` be the nuisance intercept resolved by the
existing SUMMIT intercept step. Define

```text
q12_j = z1*_j z2*_j - c,
D12_jk = sqrt(n1* n2*) L_jk / M_k.
```

At each update the component genetic-covariance vector is

```text
gamma = (D12' W12 D12)^(-1) D12' W12 q12.
```

This solve also uses least squares on `sqrt(W12) D12`, with a full-rank SVD.
Write `h1+`, `h2+`, and `gamma+` for the total h2 and covariance plug-ins from
the current matching replicate, and define

```text
a_j = 1 + clip(h1+, 0, 1) n1* max(L_j+, 1) / M_+,
b_j = 1 + clip(h2+, 0, 1) n2* max(L_j+, 1) / M_+,
c_j = c + clip(gamma+, -1, 1) sqrt(n1* n2*) max(L_j+, 1) / M_+,
w12_j = 1 / {max(Lw_j, 1) [a_j b_j + c_j^2]}.
```

The working variance `a_j b_j + c_j^2` is the Gaussian variance of the cross
product `z1*_j z2*_j`. As for h2, flooring affects only the weight model, not
the covariance design.

SUMMIT deliberately does not refit `c` jointly with `gamma`. If `c` came from
`--intercept-rg` or `--pheno-rg`, the established SUMMIT contract holds it
fixed in every SNP jackknife replicate. If SUMMIT estimated `c` from summary
statistics, covariance replicate `r` uses the corresponding intercept
delete-refit `c_r`. Each covariance delete refit also uses the matching LDSC
h2 delete estimates for both traits, then repeats covariance initialization
and every IRWLS update. Thus the intercept is resolved exactly as in the
default SUMMIT estimator; only the main h2 and covariance instruments change.

Component and total genetic correlations are computed from the matched
replicate estimates:

```text
rg_k     = gamma_k / sqrt(h1_k h2_k),
rg_total = sum_k gamma_k / sqrt(h1_total h2_total).
```

For overlapping annotations, these per-column quantities retain SUMMIT's
existing coefficient-component semantics: `gamma_k = M_k tau12_k`, and the
component `rg_k` denominator uses the corresponding raw h2 coefficient
components. They are not covariance or rg restricted to the SNP set carrying
annotation `k`. A set-restricted covariance would instead require applying the
full reference annotation-overlap matrix to `tau12`. The total covariance and
total rg are unaffected by this distinction, and the output schema is kept
consistent with SUMMIT's default HE estimator.

Only jackknife SEs are currently supported. HE-specific robust, delta, and
K-moment SE formulas are rejected rather than applied to a different
estimating equation.

## IRWLS weights

Write

```text
L_j+ = sum_k L_jk,
M_+  = sum_k M_k,
h_+  = sum_k h_k.
```

Given a scalar regression weight LD score `Lw_j`, SUMMIT uses

```text
mu_j = 1 + clip(h_+, 0, 1) n* max(L_j+, 1) / M_+,
w_j  = 1 / {2 max(mu_j, 1e-3)^2 max(Lw_j, 1)}.
```

Only the variance and overcounting weights are floored. The regression design
`D` is never floored, so negative unbiased or sub-unit LD scores remain in the
mean equation.

`--ldscores-w` should point to a one-column LD-score file computed over the
regression SNP set. Its SNP order may differ from the primary LD file; SUMMIT
aligns by SNP ID and restricts the regression axis to the intersection. Finite
values below one, including negative unbiased estimates, are retained and
floored only when forming `w_j`. If `--ldscores-w` is omitted, `Lw_j=L_j+`.
That fallback matches the experimental implementation but a separate scalar
weight LD score is preferable for overlapping annotations.

The default is three closed-form updates. For one component, initialization is
the usual aggregate constrained-LDSC estimate

```text
h0 = M_+ sum_j q_j / {n* sum_j L_j+}.
```

For multiple components, initialization is the full-rank unweighted
multivariate least-squares estimate. This matches SUMMIT's experimental
closed-form IRWLS implementation. It intentionally differs from historical
`ldsc.py`, which freezes its initial weights for partitioned regression instead
of continuing multivariate IRWLS.

The covariance fit uses the analogous aggregate one-component initializer,
with `n*` replaced by `sqrt(n1* n2*)` and `q` by `q12`. Its multicomponent
initializer is likewise unweighted least squares.

## Fixed reference moments

`M_k` is fixed across GWAS filtering and every jackknife deletion. It must not
be recomputed from the retained regression SNPs. When `--annot` is supplied,
SUMMIT reads its full pre-regression rows to calculate

```text
M_k    = sum_j A_jk,
O_kl   = sum_j A_jk A_jl,
M_ref  = number of reference annotation rows.
```

An explicit `--ldsc-m` overrides `M_k`, with chromosome-split `@` files summed,
but it must agree with the masses calculated from that same annotation file.
This validation prevents combining `.M_5_50` with an unfiltered annotation
overlap matrix. With no annotation and one LD-score column, an external `M`
directly supplies the all-ones reference count and overlap moment.

New genome-wide LD-score runs write `<out>.gw.M`. External LDSC `.l2.M` files
are also accepted when their annotation universe matches `--annot`.

For raw component contributions `h_k`, SUMMIT reports

```text
tau_k       = h_k / M_k,
h2_cat,k    = sum_l O_kl tau_l,
h2_total    = sum_k h_k.
```

Thus category h2 uses the same overlap adjustment as the existing H2 result
contract. `--enrich-mode` selects the enrichment definition; it does not change
the reported category h2 values.

## Jackknife semantics and cost

Every delete-block replicate repeats initialization and all IRWLS updates with
the same fixed reference moments. This is the implemented exact-refit
semantics. It differs from historical `ldsc.py`, which freezes
the full-data final weights when constructing delete values.

The implementation retains only the coefficient vector for each replicate and
the full-fit diagnostics, so memory does not grow as `R` full SNP-length weight
vectors. Runtime is nevertheless proportional to the number of exact refits:
roughly `O((R+1) I M K^2)` for `R` replicates, `I` updates, `M` regression SNPs,
and `K` annotation columns. Failed or rank-deficient delete refits are fatal;
SUMMIT does not silently compute an SE from a partial set.

## Difference from literal LDSC

Literal univariate LDSC ordinarily regresses a Wald `Z_j^2-1` response with
per-SNP sample size `N_j`. SUMMIT instead retains its exact score response and
the scalar `n*` used by the HE estimator. In the bivariate path, SUMMIT first
resolves `c` under its existing fixed-or-summary intercept contract, performs
matched h2 and covariance delete refits, and then forms rg as their ratio.
These conventions coincide with package LDSC only in special cases, such as
effectively constant sample size with equivalent score and Wald statistics.
The implemented modes should therefore be called constrained score-scale
LDSC-style IRWLS and constrained score-scale cov-LDSC IRWLS, not
package-identical LDSC.
