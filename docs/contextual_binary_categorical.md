# Binary and categorical contextual covariance

This experimental Python path provides categorical presets for the unified
common-scale contextual covariance model. It is opt-in and does not change the
existing SUMMIT additive or standardized GxE estimators.

## Preset and fixed effects

`build_categorical_context_preset` (also available as
`build_categorical_preset`) constructs all declared one-hot columns. The same
columns define the genetic context basis and the residual indicator kernels.
No extra genetic constant or identity residual kernel is added: either would
be redundant with a complete one-hot basis.

The preset also supplies `fixed_effect_columns`, containing categories 1
through C-1. `fixed_effect_design` combines those reference-coded columns with
an intercept and optional continuous covariates. The genetic model still keeps
all C categories; reference coding is only a nuisance-design convention.

Category order is either the supplied order or a deterministic order of typed
canonical scalar encodings. The basis-specification digest and a separate
category-order digest bind that choice. Heterogeneous scalar types are
preserved; typed values that NumPy's one-hot equality would collapse (for
example, `True` and `1`) are rejected rather than allowed to overlap. Every
declared category must occur, and every observed value must be declared.
Singleton categories are retained by default so the normal-equation rank
diagnostic can expose whether the requested model is identifiable. A larger
`minimum_category_count` can be declared as an earlier application-level gate.

`build_binary_context_preset` is the two-category convenience constructor.

## Binary coordinates

For one-hot coordinates h=(I[E=0], I[E=1]) and intercept coordinates
z=(1,E),

```text
z = A h,   A = [[1, 1],
                [0, 1]].
```

The binary one-hot coefficient order is `(v0, v1, gamma)`, corresponding to

```text
Omega_h = [[v0, gamma],
           [gamma, v1]].
```

The intercept/binary order is `(a, b, c)`. The exact maps are

```text
a = v0                         v0 = a
b = v0 + v1 - 2 gamma         v1 = a + b + 2 c
c = gamma - v0                gamma = a + c.
```

The coefficient and 3x3 covariance transform helpers apply these maps in the
canonical diagonal-then-off-diagonal order. For a full genetic/residual vector,
use the corresponding 3x3 map as the genetic block and an identity map for the
residual block. This preserves genetic–residual covariance terms.

## Fitting and reported quantities

Build the generic reference and trait summary with the preset's `basis`,
`residual_basis`, `component_index`, `residual_names`, and `basis_hash`, then
call `fit_context_model`. The trait and fit manifests bind the actual study
basis and residual-basis arrays, in addition to their declared order and basis
specification, so a different category count or swapped residual encoding is
rejected by the derived categorical API. After constructing the reference and
trait summaries, `preset.without_individual_data()` returns the aggregate-only
descriptor needed by fitting, derivation, and boundary testing; its fixed-effect
constructor then fails closed because the row-level context has been discarded.
`derive_categorical_context_fit` reports every category variance, pairwise
covariance and correlation, their nonlinear approximate-jackknife uncertainty,
residual coefficients, and trace-based category variance proportions. Pass the
deleted normal equations in fit-group order to obtain exact uncertainty for the
trace ratios under the declared SNP-contribution deletion approximation. The
recorded deleted-group identity is checked for every replicate.

The public feature aliases are `categorical_context_covariance` and
`binary_context_covariance`.

For a single-annotation binary fit, `derive_binary_context_fit` additionally
reports

```text
rho = gamma / sqrt(v0 v1)
log_sd_ratio = 0.5 log(v1 / v0)
tau2_1_given_0 = v1 - gamma^2 / v0.
```

`rho` and `log_sd_ratio` are undefined unless both raw context variances are
positive. `tau2_1_given_0` is undefined unless v0 is positive and can be
negative for an unconstrained raw moment estimate. Every scalar has an
explicit status. A non-significant difference test is not reported as evidence
that the context variances are equal.

Raw moment estimates remain primary. When `fit_context_model(...,
project_psd=True)` is requested for a disjoint-annotation model, the derived
result adds separately labelled PSD-interpretable quantities. It does not
replace the raw estimate. Projecting every approximate-LOO replicate is marked
as exploratory because the projection is nonsmooth near the PSD boundary. If
a singular jackknife metric prevents a unique numerical tie-break for any
projected replicate, the PSD point summary is retained and its projected-LOO
uncertainty is explicitly marked indeterminate.

The category-specific trace proportion is

```text
tr(D_c Sigma_g) / tr(D_c Sigma_total).
```

It is evaluated from the residual-category row of the complete normal system
and is labelled trace-based; it is not a covariance-surface coefficient.

## Boundary inference

`binary_context_boundary_test` (with `test_binary_boundary` retained as a
compatibility name) exposes an experimental route for `rho=1`, `tau2=0`, and
`equal_variances`. The matching binary preset is a required keyword argument;
this prevents an intercept-plus-binary or arbitrary two-column fit from being
silently interpreted as one-hot stratum coordinates. The test uses the full
joint approximate-LOO covariance, constructs equal-group pseudo-values, draws
Rademacher multipliers, and recomputes both the covariance and a
null-constrained minimum-distance statistic for every multiplier draw. The
nonlinear null projection uses a global one-dimensional rank-one
parameterization rather than a local two-parameter optimizer. The
equal-variance contrast includes `Cov(v0,v1)`; marginal standard errors are
never added as if the estimates were independent.

At least six equal-weight jackknife groups are required. Ideal Gaussian
group-influence simulations were calibrated at six or more groups, but noisy
nonlinear fits can still yield undefined raw domains or multiplier projections;
those cases return an explicit indeterminate status rather than a p-value.

Boundary p-values must remain labelled experimental until null simulations at
the intended sample size, category prevalence, and block design demonstrate
adequate type-I error. Failure to reject is indeterminate evidence, not proof of
equal variances, perfect correlation, or zero orthogonal heterogeneity.

## Interpretation and transport

No context dependence in one-hot coordinates is a rank-one matrix
`sigma2 * 11'`, not `sigma2 * I`. Proportional amplification is also rank one,
with unequal category loadings. A diagonal one-hot matrix instead describes
independent stratum-specific genetic effects.

Reference transfer assumes compatible ordered variants, alleles, annotations,
genotype scale, basis specification, and relevant genotype–context–covariate
moments. Different category prevalence or context–PC structure is a reference
mismatch diagnostic, not something the basis transform repairs.
