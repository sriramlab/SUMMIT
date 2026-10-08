# Cross-trait genetic response covariance

This model estimates how two traits' genetic responses covary across
contexts. For example, it can estimate the covariance between one trait's
age response and another trait's BMI response, or compare response correlation
with baseline genetic correlation. The cross-trait covariance matrix is
generally asymmetric; its transpose describes the reversed trait pair.
The [usage guide](Cross-trait-analysis.md) describes the Python workflow.

## Model

For trait X, write the context-specific SNP effect as
`beta_j^X(e) = phi(e)' b_j^X`, with
`Cov(b_j^X,b_j^Y) = Omega_XY / M`. The first context coordinate is one.
Each SNP annotation has its own covariance matrix and annotation mass.
For the two traits' context-weighted, fixed-effect-adjusted genotype features:

```text
F_a^X = P_X D_a G_X
K_ab^XY = F_a^X (F_b^Y)' / M
E[y_X y_Y'] = sum_ab Omega_XY[a,b] K_ab^XY
              + P_X diag_overlap(phi' Psi_XY phi) P_Y
```

The fit retains all Q² ordered genetic coefficients. Residual covariance
acts only on shared participants and identifies symmetric context products.
Redundant residual terms, such as a binary exposure and its square, are
reduced to an identifiable basis. Genetic estimates remain signed, with
no imposed symmetry or positive-semidefinite projection for Omega_XY.

## Reference assumptions

The reference and study must use compatible variants, genotype scaling,
context definitions, and fixed effects. Saved chromosome summaries supply
within-chromosome reference moments; cross-chromosome LD is omitted.
Cohort exposure moments use each trait's sample mask and the actual overlap.

| Mode | Treatment of different-person reference moments |
|---|---|
| `factorized` (default) | Combines reference LD with cohort exposure moments |
| `factorized_plus_residual` | Adds the reference's remaining genotype–context dependence, scaled to cohort size |
| `legacy_transport` | Transfers the reconstructed ordered reference moment, scaled to cohort size |
| `legacy_transport_exact` | Uses additional Z summaries to correct the ordered reference reconstruction before transfer |

Reconstructing ordered moments from the saved symmetric summaries is an
approximation. The additional Z summaries described below correct part of
that reconstruction.

These modes make different assumptions about the genotype–environment
distribution in the study and reference. Retaining more reference dependence
also retains its sampling and random-vector noise. Agreement among modes
provides a sensitivity check for the supplied data.

Same-person moments use saved reference diagonals restricted to the actual
overlap. These diagonals approximate the corresponding study diagonals when
the projectors differ. Where a target block's diagonal is unavailable, its
contribution is allocated by annotation mass. Within-trait fits use the
factorized different-person approximation and their own sample rows.

### Additional Z summaries

`summit reference zpass` computes Z summaries from genotypes and saved reference
scaling. The study driver can collect them during scoring with `--z-output`,
avoiding another genotype pass.

Z summaries correct the ordered reference reconstruction for a single
annotation, including a weighted annotation. Exact correction for arbitrary
pairs of different annotations is not supported. The correction also leaves
cohort-transfer assumptions, reference sampling error, and random-vector
error in place. The `legacy_transport_exact` name refers to this limited
reference correction.

## Fitting and uncertainty

`solve_cross_trait_normal_equations` fits the ordered genetic coefficients
and identifiable residual terms. Results include rank, condition number,
minimum Gram eigenvalue, solve residual, and deletion diagnostics.

The default uses 200 paired target-SNP blocks with
`deletion_method="target_moments"` and `uncertainty_method="delta"`.
The deletion removes target-block moments while holding the source reference
fixed. Coefficients retain their full-genome units. This approximation includes
annotation-mass allocation of unavailable block diagonals; it does not
recompute both SNP axes from genotypes.

The delta method propagates the joint deletion covariance of Omega_XX,
Omega_YY, and Omega_XY into derived correlations and their differences.
It includes uncertainty from centering, baseline projection, and both variance
denominators. `--uncertainty-method jackknife` propagates the nonlinear deleted
estimates directly. Both methods require matching deletion IDs across the
within-trait and cross-trait fits.

Intervals are conditional on the supplied reference and transfer mode.
They exclude independent reference sampling, probe redraws, and uncertainty
in the transfer assumptions. Weak variance denominators can make correlation
intervals unreliable; nonpositive denominators produce undefined values.
Raw correlations outside [-1,1] are retained and flagged.

## Derived quantities

Baseline covariance and correlation use the original context origin.
Centered baseline estimates move each trait's intercept to its own mean
context. Orthogonal responses remove the genetic association with that
centered baseline. If `a_X = Omega_XX[0,1:]/Omega_XX[0,0]` in the centered basis,
the cross-trait orthogonal-response covariance is

```text
H_XY = Omega_XY[1:,1:] - a_X Omega_XY[0,1:]
       - Omega_XY[1:,0] a_Y' + a_X Omega_XY[0,0] a_Y'.
```

Outputs include the full response and H matrices, per-context response
correlations, and the aggregate covariance `tr(S H_XY)`. Its correlation uses
the corresponding within-trait traces and one recorded exposure covariance S.
Response-minus-baseline differences use paired uncertainty.

Context-specific genetic correlation can also be evaluated at chosen exposure
values. Retain the original context means, scales, and covariance metric with
the estimates. The model describes cross-sectional genetic covariance;
causal or longitudinal interpretations require additional assumptions.
