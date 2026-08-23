# Contextual transformed-phenotype scans

The private `summit.context` development API can score several declared
phenotype transformations in one genotype pass. It remains a NumPy scientific
oracle, not the production-scale genotype backend.

## Transformation contract

Each scan uses one immutable retained-individual mask, fixed-effect projector,
context basis, residual basis, variant order, annotation matrix, genotype
scaling convention, and approximate-jackknife grouping. Supported
`PhenotypeTransformSpec` values are:

- `identity`;
- `log`, with an explicit shift;
- `box_cox`, with an explicit shift and lambda;
- `user_supplied`, naming an explicitly supplied transformed column.

For positive `x = y + shift`, Box-Cox is

\[
  \{\exp[\lambda\log(x)]-1\}/\lambda,
\]

with the log limit at lambda zero. A nonpositive or nonfinite retained value,
nonfinite transformed value, missing/malformed user column, or zero projected
variance makes that transformation invalid. It stays in the ordered manifest
with a diagnostic status and does not enter the genotype multiplication. The
implementation never repairs validity by changing the common mask.

Every valid transformed phenotype is projected by the same rank-revealing
projector and normalized so its residual sum of squares equals the common
residual rank. Manifests record the original trait, transform parameters,
mask count/hash, transform validity, projection/normalization diagnostics, and
the shared basis/fixed-effect/variant/annotation/group identities. Written
summaries contain aggregate moments and SNP-level approximate-LOO numerators,
not phenotype or context rows.

## Batched summary and fit

For a tile of transformed phenotypes `Y`, a genotype block is multiplied by

\[
  [D_0Y,\ldots,D_{Q-1}Y].
\]

The block is visited once. Within a tile, columns use the declared q-major,
transform-minor order. `transform_tile_size` bounds the RHS width by
`Q * transform_tile_size`; the context-dependent features, traces, and
genetic-residual moments are computed once and shared. The current development
artifact retains an `M x L x P_g` phenotype-dependent numerator. This is exact
for its declared grouped deletion rule but is not a production-scale storage
format.

`fit_context_transform_scan` assembles one full normal matrix and solves all
valid transformation RHSs with one eigendecomposition. Each selected
approximate-LOO group likewise uses one deletion-specific matrix factorization
for the full transformation trajectory. It does not recompute exact deleted
genotype kernels, and it retains the full-reference same-person term as in the
single-trait approximation.

The fit stores a compact `J x L x P` pseudo-value tensor rather than eagerly
materializing an `(LP) x (LP)` covariance matrix. The full covariance is
available through `joint_covariance()` when needed. `build_transform_trajectory`
applies a fixed coordinate function to the point estimate and every LOO
replicate, so nonlinear mechanism trajectories use the same joint deletion
states. Inferential output requires the complete, balanced set of declared
deletion groups; a proper subset is rejected because it is not a calibrated
delete-group jackknife. Stored pseudo-values are checked against
`J * theta - (J - 1) * theta_(-g)` before use.

## Simultaneous inference

`simultaneous_trajectory_bands` uses centered equal-group pseudo-values and a
Rademacher multiplier maximum over the entire declared
transformation-coordinate family. Every multiplier draw is recentered and
studentized using its own pseudo-value standard error. It requires at least six
balanced groups, finite values, standard errors reconstructed consistently from
the stored pseudo-values, and an exact partition of the declared scales into
analyzed and explicitly invalid transformations. A nonfinite critical value is
reported as indeterminate. This route is experimental and does not replace raw
estimates.
When amplification and heterogeneity are selected from a joint band,
`select_simultaneous_band_coordinate` preserves the common multiplier maxima
and critical value; separately calibrated bands cannot be classified as one
simultaneous family.

`classify_scale_trajectory` requires positive, user-declared equivalence
margins for amplification and heterogeneity:

- `removable_within_declared_family` requires the signed amplification band to
  lie inside its two-sided margin and the upper band for a declared
  nonnegative heterogeneity coordinate to lie below its margin at the same
  transformation;
- `robust_within_declared_family` requires at least one entire mechanism band
  to lie outside its margin at every declared, valid transformation, with no
  invalid predeclared scale;
- all other cases are `indeterminate`.

A negative raw heterogeneity estimate is indeterminate rather than evidence in
the opposite direction. A nonsignificant point estimate is therefore never
sufficient for removal.
The API does not select a scale by maximizing estimated genetic variance.

## Development example

```python
specs = [
    PhenotypeTransformSpec("identity", "identity"),
    PhenotypeTransformSpec("log", "log", shift=1.0),
    PhenotypeTransformSpec("bc_0_5", "box_cox", shift=1.0,
                           box_cox_lambda=0.5),
]

summary = build_context_transform_summary(
    genotype=genotype,
    basis=basis,
    original_phenotype=phenotype,
    transformations=specs,
    projector=projector,
    annotations=annotations,
    component_index=components,
    residual_basis=residual_basis,
    residual_names=residual_names,
    basis_hash=basis_hash,
    fixed_effect_hash=fixed_effect_hash,
    variant_hash=variant_hash,
    original_trait="trait",
    retained_mask=retained_mask,
    loo_groups=loo_groups,
    transform_tile_size=4,
)
fit = fit_context_transform_scan(reference, summary)
trajectory = build_transform_trajectory(fit)
bands = simultaneous_trajectory_bands(trajectory, seed=20260819)
```

The ordinary `build_context_trait_summary` and `fit_context_model` paths are
unchanged.
