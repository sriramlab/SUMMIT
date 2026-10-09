# PCGC methods

PCGC estimates liability-scale genetic variance from case–control data.
This page describes additive PCGC. For inputs and commands, see
[Binary traits and PCGC](Binary-traits-and-PCGC.md); for context-dependent
SNP effects, see [Generalized G×E PCGC](Generalized-GxE-PCGC.md).

## Risk adjustment and estimating equations

Let population prevalence be K and the analyzed sample case fraction be P.
Sampling is assumed to depend on case status alone. For population risk k_i,

```
a   = K (1-P) / [P (1-K)]
p_i = k_i / [k_i + a (1-k_i)]
z_i = (y_i-p_i) / sqrt[p_i (1-p_i)]
d_i = normal_pdf(normal_quantile(k_i)) sqrt[p_i (1-p_i)] / [k_i (1-k_i)]
```

To first order in between-person liability correlation,
`E[z_i z_j] = d_i d_j sum_a theta_a K_a(i,j)`, for different people. The
population-scaled annotation kernel is `K_a = X diag(A_a) X.T / M_a`, with
`M_a = sum_j A_ja`. Residual liability variance, including genetic variance,
is one. The residual environmental variance is not fixed to one separately.

The standard method uses `B_a = diag(d) K_a diag(d)` and raw score response
`v = d*z`. Its equations are

```
H_ab = trace(B_a B_b) - dot(diag(B_a), diag(B_b))
b_a  = sum_j A_ja [(X_j.T v)^2 - sum_i X_ij^2 v_i^2] / M_a
H theta = b
```

The equations estimate the genetic components directly, without an additional
residual-variance parameter. Estimates are unconstrained. Singular normal
matrices, or a nonpositive symmetric part, cause the fit to fail. Too few
random vectors can produce such a matrix even when the exact matrix is positive.

Population risks can be supplied or fitted by the ascertainment-aware probit
likelihood with an intercept. Covariates must have full rank and must not
perfectly separate cases from controls. The response requires raw genotype
scores; ordinary logistic GWAS beta/SE or SAIGE statistics cannot be used in
these equations.

Marginal component estimates equal `theta/(1+V_cov)` under the stated
independent-covariate liability model. Supplied-risk input can include known
population `V_cov`; otherwise the variance of the risk predictor is estimated
with inverse-ascertainment weights. Outputs retain the scale and risk metadata.
The jackknife does not refit risks or capture uncertain external prevalence.

## Methods

| Method | Reference and score | Important limit |
|---|---|---|
| `liability` | Constant-risk PCGC, equivalent to a scalar off-diagonal HE conversion | Requires individual risks equal to population prevalence |
| `pcgc` | Trait-specific `d` features, raw `X.T (d*z)` scores | First-order liability approximation; unprojected, population-scaled genotypes |
| `pcgc-inverse` | Ordinary genotype features, raw `X.T (z/d)` scores | Different pair weighting and potentially heavy-tailed noise |
| `pcgc-basis` | Generalized features `diag(phi_q) X`, contracted using `d=Phi c` | Requires an exact sensitivity span |
| `pcgc-ld` | Independent population LD with finite-reference diagonal subtraction | Explicit factorization approximation requiring suitable risk/genotype dependence and reference matching |

Study-specific preparation computes reference moments and raw score rows in
two genotype passes. For `pcgc-basis`, forming `d = Phi c` before calculation
gives the same result as contracting the basis kernels afterward. Both
orientations of an off-diagonal basis pair are included in its kernel, so its
contraction coefficient is `c_q*c_r`, not twice that value.

## Population LD approximation

For an independent population reference of size R, `pcgc-ld` uses

```text
L_pop(j,b) = sum_k A_kb [(X_j.T X_k)^2 - sum_i X_ij^2 X_ik^2] / [R(R-1)].
```

Transfer to the study uses the risk factor
`sum_(i!=j) d_i^2 d_j^2 = (sum_i d_i^2)^2 - sum_i d_i^4`.
This approximates the risk-weighted study normal matrix. It assumes suitable
risk–genotype factorization and a matched population reference; ascertainment
can violate that factorization. Study and reference must have disjoint samples
and the same SNPs, alleles, and population genotype scale.

The additive methods above use unprojected genotype features. In general,
`P diag(d) X` differs from `diag(d) P X`, and projecting heteroskedastic binary
residuals creates off-diagonal noise. Risk covariates are assumed exogenous.
The generalized G×E path supports genotype-PC adjustment before context and
risk weighting; its [covariate-adjustment section](Generalized-GxE-PCGC.md#covariate-adjustment)
describes the required inputs and order of operations.

## SNP-block standard errors

Let `U_jb` be the directional kernel-product row after subtracting its exact
same-person term. For retained target SNPs T, the equations are

```text
H_ab(T) = sum_(j in T) A_ja U_jb / [M_a(T) M_b(full)]
b_a(T)  = sum_(j in T) A_ja rhs_j / M_a(T)
H(T) theta(T) = b(T).
```

Source kernels and source annotation masses stay fixed. Each replicate
therefore estimates the full-genome component parameters. The directional
matrix can be asymmetric and is solved without symmetrizing it.

The default divides the saved SNP order into 200 contiguous blocks. For J
blocks and replicate mean theta_bar, the coefficient covariance is

```text
Cov_JK(theta) = (J-1)/J * sum_b (theta_-b - theta_bar)(theta_-b - theta_bar).T.
```

Use chromosome- and position-ordered SNPs and blocks large enough to contain
local LD. `--njack` selects another integer count of at least two, no larger
than the SNP count. Both conditional and marginal SEs are reported.

These deletions hold people, case fraction, prevalence, risks, population
scales, and reference probes fixed. SEs exclude uncertainty from estimating
those quantities or sampling an independent reference. With overlapping
annotations, coefficients describe conditional contributions. Negative moment
estimates are retained.

## Limits of inverse weighting

At the null, `Var(z/d | C, sampled) = 1/d(C)^2`. Strong continuous risk
factors can make inverse weighting unstable or give it infinite variance.
For Gaussian population covariate C with risk predictor `eta = gamma*C-t`,
the lower-tail second-moment integrand is proportional to

```text
exp[((gamma^2-1)*C^2 - 2*gamma*t*C + t^2)/2] / |gamma*C-t|.
```

Considering both tails gives divergence for `gamma^2 >= 1`. This result is
specific to that Gaussian model. Inverse weights are not clipped, since
clipping changes the estimating equation.

## Cross-trait Python API

`summit.pcgc.cross.prepare_pair` accepts master genotypes, each cohort's
sample rows, and its risk model. A quantitative response must use a known
population residual scale. `fit_pair` estimates covariance and genetic
correlation; rg is undefined if either estimated genetic variance is
nonpositive.

The sampling assumption is `marginal_case_status_sampling_v1`: each cohort's
marginal inclusion probability depends only on its own case status. Selecting
shared controls by joint eligibility need not satisfy this assumption.
Shared samples must be identified individually because their same-person
correction depends on the product of the two responses, not just an overlap
count or ordinary phenotype covariance.

For unprojected features, `F_1.T F_2 = X.T diag(phi_1*phi_2) X` is symmetric.
The ordered rectangular reference moment can therefore be obtained as
`L_(01,01)/2 - L_(00,11)` from the generalized symmetric representation.
Within-trait and cross-trait moments share two genotype passes. Uncertainty
uses paired SNP-block deletions and joint delta propagation.

This is an experimental Python interface; binary `--rg` is not available.
Its selection assumptions require particular care for shared-control studies.

## References

The liability approximation follows
[Golan et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267399/).
SUMMIT's full-genome moments and target-SNP deletion equations differ from
the normalization, LD windows, and resampling used by
[PCGC-s](https://github.com/omerwe/PCGCs) and
[S-PCGC](https://github.com/omerwe/S-PCGC); their summary files are not
interchangeable.
