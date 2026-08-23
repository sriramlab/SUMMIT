# Uncertainty validation

## Decision

**The revised limited jackknife-calibration gate passes; overall uncertainty
validation remains PARTIAL.** Exact delete-block equality is not required. In
the production-like `K=1`, 100-block experiment below, full-reference diagonal
reuse gives approximately calibrated and mildly conservative inference. The
jackknife still does not propagate finite-reference-person or randomized-probe
variation, so this is not a general coverage claim.

## Same-person finite-probe estimators

`tests/gxe_completion/uncertainty_diagnostics_simulation.py` constructs four
overlapping additive/interaction components from 120 samples and 80 variants,
uses shared Rademacher or Gaussian variant probes, and compares 400 independent
probe draws. The exact target is the dense same-person matrix.

For source squares `z_ab(i)=S_ab(i)^2`, the estimators are:

```text
U_ab = sum_i [(sum_v z_av)(sum_w z_bw) - sum_v z_av z_bv] / [B(B-1)]
Split_ab = symmetrized product of mean squares in independent probe halves
Plugin_ab = sum_i mean_v(z_av) mean_v(z_bv).
```

The first two are unbiased across independent probe groups. The plug-in is PSD
for every realization but is biased when families share probes.

Selected Rademacher results (relative Frobenius scale):

| B | Estimator | Relative bias | Relative RMSE | Non-PSD fraction |
|---:|---|---:|---:|---:|
| 2 | U-statistic | 0.0385 | 0.6184 | 0.8275 |
| 2 | split probe | 0.0385 | 0.6184 | 0.8275 |
| 2 | PSD plug-in | 0.6491 | 1.0793 | 0.0000 |
| 16 | U-statistic | 0.0120 | 0.1904 | 0.5575 |
| 16 | split probe | 0.0123 | 0.1948 | 0.5875 |
| 16 | PSD plug-in | 0.0956 | 0.2269 | 0.0000 |
| 64 | U-statistic | 0.00675 | 0.0932 | 0.0525 |
| 64 | split probe | 0.00696 | 0.0936 | 0.1200 |
| 64 | PSD plug-in | 0.0164 | 0.0964 | 0.0000 |

Gaussian B=64 showed relative bias/RMSE `0.0118/0.1100` for the U-statistic,
`0.0118/0.1101` for split probes, and `0.0344/0.1173` for the plug-in. These
results support retaining the unbiased U-statistic for point estimation and
reporting, rather than hiding, its finite-B indefiniteness. They do not select
a confidence-interval procedure.

## Delete-block same-person approximation

The same simulation computed each deletion exactly as

```text
kappa_a^(-g)(i) = [R_a(i)-R_a,g(i)] / [M_a-M_a,g]
```

and compared its dense `D^(-g)` with reuse of full `D`. Across ten blocks, full
reuse had median relative Frobenius error `0.01622` and maximum `0.09366`; the
largest error occurred where a small annotation lost substantial mass. Thus
full-diagonal reuse is not exact and is not justified for small/high-leverage
annotations by this audit.

An exact production implementation would need global-by-block and block-by-block
rowwise feature-square cross-products, or a bounded recomputation strategy. It
is not necessary for the revised acceptance criterion, but the larger error for
small/high-leverage annotations remains a reason to diagnose annotation mass
lost per block.

## Conditional jackknife calibration

The version-2 simulation adds a `K=1` G/GxE design with 120 reference samples,
800 correlated variants, and 100 delete blocks. For each deletion it constructs
the exact renormalized same-person matrix and the production approximation that
reuses the full same-person matrix. It then propagates 5,000 independent
block-estimating-equation perturbations through both deleted normal equations.
This holds all other uncertainty sources fixed and directly tests the effect of
the approximation on the reported block-jackknife SE.

| Scenario | Exact 95% coverage | Approx. 95% coverage | Approx./exact median SE | Approx. SE / empirical SD |
|---|---:|---:|---:|---:|
| moderate GxE, G | 94.78% | 97.76% | 1.175 | 1.187 |
| moderate GxE, GxE | 94.78% | 97.76% | 1.175 | 1.187 |
| GxE null, G | 94.70% | 94.98% | 1.010 | 1.008 |
| GxE null, GxE | 94.70% | 94.98% | 1.010 | 1.008 |

The approximate GxE-null two-sided type-I error was 5.02%. Relative normal-
matrix error from full-diagonal reuse was 0.117% at the median and 0.479% at the
maximum. Under the moderate alternative, reuse inflated SE by about 17.5% and
coverage to 97.76%; this is conservative rather than anti-conservative in this
design. The predeclared limited gate required null type-I error in 3--7%,
alternative coverage in 93--99%, and median SE inflation no greater than 25%; it
passed.

This calibration applies to the ordinary one-annotation/100-block operating
regime. It does not override the adverse small-annotation example above, and it
does not establish calibration for many overlapping annotations or blocks that
remove a large fraction of annotation mass.

## Missing validation

The conditional block-local perturbation experiment establishes only the narrow
coverage/type-I result above. Reference-person folds, independent probe groups,
study/reference overlap covariance, power, and propagation through
`d theta = T^+ (d q - dT theta)` remain to be implemented. No uncertainty or
manuscript-wide coverage claim should rely on this conditional gate alone.

Machine-readable results are in
`benchmarks/uncertainty_diagnostics_simulation_v2.json` (SHA-256
`cc2de682f0fae955db28f0d72a196c6c8d355fb1f60b84f4e96150579b1ad195`).
