# Cross-trait genetic response models

This research extension estimates covariance between two traits' genetic
responses to environmental contexts. It reports baseline genetic correlation,
response correlations, and paired differences between them. Ordinary additive
`--rg` remains available through the [h²/rg guide](Heritability-and-genetic-correlation.md).

For `[1, age, BMI]`, the cross-trait matrix contains nine ordered entries.
Trait X's age response can covary with trait Y's BMI response differently
from X's BMI response with Y's age response. Reversing the trait pair transposes
the matrix. Each SNP annotation has its own matrix.

## Workflow

1. Define a common genotype scale and saved context coding, including an
   intercept coordinate. Each trait retains its own observed sample rows,
   fixed effects, and phenotype normalization.
2. Prepare chromosome reference moments with the generalized G×E machinery.
   Keep the reference context definitions and sample identities with the files.
3. Use `CrossTraitBatch` around `MaskedTraitBatch` to collect the requested
   pairs in one study genotype traversal. The batch shares decoded genotypes
   and within-trait scores, and computes overlap residual moments.
4. Assemble compatible within-trait and ordered cross-trait equations from
   the saved summaries. Fit them on the same SNP blocks.
5. Derive correlations and their differences using the joint covariance of both
   within-trait estimates and the cross-trait estimate.

The API modules are `summit.ldscore.generalized_gxe_cross_trait_batch`,
`summit.context.cross_trait_gram`, and `summit.context.cross_trait_fit`.
`CrossTraitMomentPlan` and `fit_cross_trait` perform the summary-based fit.
`write_cross_trait_fit` adds derived quantities and paired uncertainty when
the matching within-trait fits are supplied. Across chromosomes, the full
same-person Gram must be formed from the **summed person-level diagonals**;
summing chromosome Grams would omit cross-chromosome diagonal products.

There is no general `summit` cross-trait G×E command yet. The research drivers
`scripts/generalized_gxe/cross_trait_study.py` and `cross_trait_pilot_fit.py`
show the complete workflow, including input checks and checkpoint
resume. These scripts use a study-specific directory layout; adapting them requires
preparing the same input files. They do not accept ordinary GWAS summaries.

## Reference modes

| Mode | Treatment of reference genotype–context dependence |
|---|---|
| `factorized` | Uses population LD and cohort exposure moments; default |
| `factorized_plus_residual` | Adds the nonfactorized reference remainder |
| `legacy_transport` | Transfers the reconstructed ordered reference moment |
| `legacy_transport_exact` | Transfers an ordered moment repaired with additional reference Z summaries |

All modes account for each trait's sample mask and actual overlap. Their
differences assess reference assumptions; close agreement on one dataset
does not establish calibration for every sampling design. The last mode needs
compatible Z summaries. Reference-mode choice does not change the scientific
definition of the ordered cross-trait covariance.

## Uncertainty and interpretation

`baseline_rg` uses the original context origin; `centered_baseline_rg` uses
each trait's mean context. Orthogonalization uses the centered baselines.
Retain the saved exposure units and covariance metric with the results.

The default is directional target-moment deletion with full-estimate delta
propagation. It retains covariance among XX, YY, and XY estimates. Comparisons
such as response correlation minus baseline correlation use paired uncertainty;
subtracting two independent confidence intervals is incorrect.

The research fitter also exposes `--uncertainty-method jackknife` for nonlinear
delete-block estimates. Both methods use the paired SNP blocks.

The aggregate orthogonal-response correlation measures shared response after
removing each trait's genetic association with its baseline, under the recorded
context covariance metric. It is not an average of exposure-specific correlations
and does not describe the direction of an environmental intervention.

Weak or nonpositive genetic-variance denominators can make correlations and
intervals undefined. Raw values outside [-1, 1] are retained and flagged.
Uncertainty is conditional on the supplied reference. It excludes reference
sampling, random-vector error, and error from transferring reference moments
to a different study cohort.

The [method description](Cross-trait-response-covariance.md) gives
the equations and reference approximations.
[Binary cross-trait PCGC](Binary-traits-and-PCGC.md) is a separate research path
with its own ascertainment and overlap assumptions.
