# Contextual Python API

`summit.context` provides continuous and categorical context coding, covariance
summaries, and research analyses. Dense builders are intended for small numerical
checks. The native reference/trait interface streams BED input; generalized
per-SNP estimation is described separately in [Multiple environments](Multiple-environments.md).

## Context coding

`calibrate_multienvironment_basis` fits continuous means/scales and categorical
frequencies on a reference cohort. `apply_multienvironment_calibration` applies
the same coding to study data. Unknown categories are rejected. Optional
`FixedEffectInteractionSpec` terms add specified context–covariate products to
the fixed-effect design.

For a category-only model, `build_categorical_context_preset` creates all
category indicator columns as the genetic basis. Fixed effects use an intercept
and all but one category indicator. `build_binary_context_preset` handles two
categories.

After constructing summaries, `preset.without_individual_data()` retains the
coding and aggregate information needed by derived analyses without retaining
participant context rows.

## Reference, trait summary, and fit

| Task | Dense Python | Native reference/trait interface |
|---|---|---|
| Reference | `build_context_reference` | `reference_v1.run_contextual_reference_v1` |
| Trait summary | `build_context_trait_summary` | `trait_v1.run_contextual_trait_v1` |
| Fit | `fit_context_model` | `fit_v1.fit_contextual_model_v1` |

Inputs specify genotype scaling, context and fixed-effect columns, variants,
annotations, residual components, and deletion groups. Use the same definitions
for reference and study. The native loaders and writers use their own versioned
formats; dense-development files cannot be substituted for them.

The fit returns raw covariance coefficients, joint jackknife covariance, and
rank/conditioning diagnostics. A rank-deficient model is reported explicitly.
Optional PSD projections are separate outputs.

For binary categories with variances v0 and v1 and covariance c, derived
quantities include `c/sqrt(v0*v1)`, `0.5*log(v1/v0)`, and `v1-c²/v0`. Each is
reported only on its valid domain. `binary_context_boundary_test` is experimental;
its p-values need simulation checks for the intended sample and group design.

## Annotations and context surfaces

`build_disjoint_annotation_partition` constructs non-overlapping bins;
`build_maf_ld_partition` applies specified MAF and LD edges. Grouped builders
retain group sums for summary deletion. A deletion that empties a bin is invalid.

`fit_annotation_context_model` estimates bin-specific matrices.
`derive_annotation_total` combines them, and `derive_annotation_contrast`
estimates differences with joint uncertainty. For overlapping annotations,
individual coefficients are conditional contributions; the combined surface is
usually the clearest summary.

`fit_multienvironment_model` estimates the full context covariance matrix.
`derive_context_contrast` evaluates a variance difference or covariance between
specified contexts. `derive_covariance_modes` uses the reference context metric
and reports repeated eigenvalues as subspaces.

## Phenotype transformations

`PhenotypeTransformSpec` supports specified identity, logarithmic, and Box–Cox
transformations. `build_context_transform_summary` computes their trait
summaries in one genotype traversal. `fit_context_transform_scan` then fits the
models against one compatible reference.

Each transformation is applied before fixed-effect projection and
normalization. Define transformations and valid domains in advance. Simultaneous
bands and `classify_scale_trajectory` are experimental; classification requires
specified equivalence margins. Failure to reject a difference does not establish
that a transformation removes context dependence.

## Learning a context direction

`optimize_context_direction` fits a reduced four-component model along a
normalized environmental direction. It excludes additive–interaction covariance.
The objectives are descriptive fitted coefficients or moment-fit improvement.

`crossfit_context_direction` learns on one group of variant blocks and evaluates
on another, then reverses the roles. Its two-fold variation is descriptive,
and it does not provide calibrated post-selection inference. Construction of
these contractions uses dense arrays and is intended for small experiments.

## Examples and limits

The synthetic validation scripts also serve as executable API examples:

```bash
python scripts/context/validate_context_fit.py --output-dir results/context-fit
python scripts/context/validate_multienvironment.py --output-dir results/multienv
python scripts/context/validate_context_annotations.py --output-dir results/annotations
```

These scripts may require plotting dependencies. Optional real-trait runs need
explicit local input paths. They write aggregate diagnostics; participant
inputs must remain in protected storage.

Dense routines can allocate feature and kernel matrices that are too large for
cohort-scale use. Native memory plans are estimates and should be checked
against measured RSS. Experimental boundary tests, transformation bands, and
learned directions need their own statistical validation before scientific use.
