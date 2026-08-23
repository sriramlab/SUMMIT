# Fixed multi-environment contextual covariance

The private `summit.context` development API supports a fixed context basis
with at most four columns. It reuses the generic full-\(\Omega\) contextual
engine and remains summary-only after the trait summary has been built. It
does not learn or optimize an environment direction.

## Reference calibration and masks

`calibrate_multienvironment_basis` fits one immutable calibration on a declared
reference mask. The v1 basis begins with an intercept. Each continuous source
contributes one reference-centered, reference-root-mean-square-scaled column.
Each categorical source contributes \(C-1\) typed, reference-centered indicator
columns relative to a declared reference category. Boolean, integer, floating,
and string category identities are not coerced into one another.

Every source and nuisance covariate must retain its original common row count.
`apply_multienvironment_calibration` applies one explicit Boolean mask to all
columns; it never constructs a different complete-case mask for an individual
column. A nonfinite retained continuous value, an unknown retained category,
or a differently prefiltered source fails closed. Values outside the declared
mask do not enter calibration or scoring. Artifacts retain the mask count and
digest rather than participant rows.

The genetic and nuisance designs are distinct:

- the genetic basis contains the retained calibrated basis columns;
- the nuisance design contains an intercept, each declared nonconstant basis
  main effect, named ordinary covariates, and only explicitly requested
  basis-by-covariate interactions.

The shared fixed-effect *specification* hash binds ordered names and
interactions across reference and study cohorts. The numeric fixed-effect
matrix hash is cohort-specific and may differ. The `preset.basis_hash` passed
to the generic reference and trait builders binds calibration, pruning, and
the shared nuisance specification, so a specification mismatch fails before a
cross-cohort fit.

The default residual basis is one identity component. The feature convention
remains `raw_projected`: genotypes use one declared pre-projection scale and
there is no basis-specific or post-projection genotype restandardization.

## Metric, conditioning, and explicit pruning

The default covariance-operator metric is the uncentered reference second
moment

\[
M_\phi=E_R[\phi(E)\phi(E)^\mathsf{T}].
\]

This is not the centered covariance of the nonconstant sources. Presets expose
both the immutable reference metric and the empirical study metric; modes use
the reference metric unless `metric="study"` or a matrix is explicitly
requested.

Conditioning diagnostics report the metric spectrum, rank and condition,
centered correlations among nonconstant context columns, nuisance-design rank
and leverage, and—when a diagnostic genotype array is supplied—the projected
genetic feature-family Gram spectrum. The last quantity is the relevant
projected-family diagnostic: `P @ phi` itself is zero when context main effects
are nuisance covariates. Normal-equation rank and condition remain properties
of the fitted model.

No direction is silently removed. `fit_multienvironment_pruning` constructs an
explicit reference-fitted, pivoted column subset with the intercept pinned. It
records requested and retained names, selected/dropped indices, the forward
selection matrix, a retained-to-requested reconstruction matrix, the rank
tolerance, and reference reconstruction error. The identical transform is
then applied to the study. A study violation of a reference dependency appears
as nonzero study reconstruction error rather than a local refit.

## Covariance surfaces, contrasts, and modes

For each annotation, the generic estimator returns the complete symmetric
coefficient matrix \(\Omega\), including every signed off-diagonal. A calibrated
grid \(\Phi_g\) produces the covariance surface

\[
C_g=\Phi_g\Omega\Phi_g^\mathsf{T}.
\]

`derive_context_contrast` provides two linear functionals with uncertainty
from the full joint approximate-jackknife covariance:

- `variance_difference`: \(x^\mathsf{T}\Omega x-z^\mathsf{T}\Omega z\);
- `covariance`: \(x^\mathsf{T}\Omega z\).

In diagonal-first packed coordinates, a variance contrast has diagonal weights
\(x_q^2-z_q^2\) and off-diagonal weights
\(2(x_qx_r-z_qz_r)\). A covariance has diagonal weights \(x_qz_q\) and
off-diagonal weights \(x_qz_r+x_rz_q\). Thus the kernel's factor of two is
applied exactly once.

`derive_covariance_modes` diagonalizes

\[
B=M_\phi^{1/2}\Omega M_\phi^{1/2}.
\]

If \(Bu_j=\lambda_j u_j\), the reported basis coefficients are
\(a_j=M_\phi^{-1/2}u_j\). They satisfy

\[
a_j^\mathsf{T}M_\phi a_k=\delta_{jk},\qquad
\Omega M_\phi a_j=\lambda_j a_j,
\]

and their evaluated context functions are \(\Phi_g a_j\). Under a nonsingular
basis change, the coefficient coordinates change but eigenvalues, evaluated
functions up to sign, and reconstructed covariance surfaces do not.

Deleted-group modes are matched to full-fit functions by maximum overlap.
Signs are aligned for isolated eigenvalues, while tied or near-tied clusters
are aligned as subspaces by an orthogonal Procrustes map. Individual function
standard errors are deliberately undefined for such clusters; eigenvalues and
eigenspaces remain reportable. Eigenvalue, evaluated-function, and rank-one
fraction uncertainty uses the ordinary equal-group approximate jackknife.

The rank-one fraction is \(\lambda_1/\sum_j\lambda_j\) only for a positive
semidefinite operator with positive total. A raw indefinite operator is
reported as a signed algebraic decomposition and is not called a covariance
mode. `use_psd=True` is a separate interpretation: it requires a requested PSD
fit and applies the same covariance-aware projection rule to every deleted-
group coefficient vector before computing mode uncertainty. Projection failure
is explicit and does not alter raw estimates.

## Development example

```python
specs = (
    MultiEnvironmentSourceSpec("age", "continuous"),
    MultiEnvironmentSourceSpec(
        "smoking",
        "categorical",
        categories=("never", "former", "current"),
        reference_category="never",
    ),
)
calibration = calibrate_multienvironment_basis(
    reference_sources,
    specs,
    mask=reference_mask,
)
interaction = FixedEffectInteractionSpec("age", "pc1", "age_by_pc1")
reference_preset = apply_multienvironment_calibration(
    calibration,
    reference_sources,
    mask=reference_mask,
    covariates={"pc1": reference_pc1},
    interactions=(interaction,),
)
study_preset = apply_multienvironment_calibration(
    calibration,
    study_sources,
    mask=study_mask,
    covariates={"pc1": study_pc1},
    interactions=(interaction,),
)

# Pass preset.basis_hash and preset.fixed_effect_hash to the existing generic
# reference/trait builders. Individual source rows are no longer needed after
# those summaries have been constructed.
fit = fit_multienvironment_model(
    reference,
    summary,
    reference_preset=reference_preset.without_individual_data(),
    study_preset=study_preset.without_individual_data(),
)
modes = derive_covariance_modes(
    fit,
    study_preset.without_individual_data(),
    annotation="all",
)
```

## Scope and performance

This is a NumPy correctness implementation for small validation fixtures. The
trait path performs one genotype pass, and the dominant feature/score products
grow linearly with \(Q\). The number of genetic covariance components grows as
\(Q(Q+1)/2\), however, so kernel assembly, Gram reductions, serialized
contributions, and exact dense work do not scale linearly. The current Python
reference materializes contextual features and development contribution
arrays; it is not safe for biobank-scale execution. Native block streaming,
bounded grouped deletion summaries, and protected GEMM integration remain
later gates.
