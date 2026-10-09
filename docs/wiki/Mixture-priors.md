# Mixture priors for polygenic scores

`fit_mixture_prediction` fits a joint mixture prior over each SNP's baseline
and environment-response coefficients. It uses SUMMIT's genotype readers,
matrix operations, annotation priors, model files and scoring API.
The Gaussian solver is the default.

For the existing per-SNP covariance `Lambda[j]`, the prior is

```text
b[j] ~ p N(0, (1-f)/p Lambda[j]) + (1-p) N(0, f/(1-p) Lambda[j]).
```

Thus `p` is the large-component probability and `f` is the fraction of total
variance assigned to the small component. The marginal covariance remains
`Lambda[j]`. For homogeneous priors this is `Omega/M`; annotated priors retain
their existing genome-wide mass normalization. Both components scale the
entire response covariance together.

`SeparateSparsitySpec(baseline=MixtureSpec(...), response=MixtureSpec(...))`
instead gives separate mixture indicators to the baseline-associated
component and the baseline-orthogonal response. For each SNP, it decomposes
`Lambda = A + U`, where `A = Lambda[:,0] Lambda[0,:] / Lambda[0,0]`
and `U` is the Schur complement embedded in the full coefficient space.
The four component covariances are `s_baseline*A + s_response*U`, with
product mixture probabilities by default. Their marginal covariance remains `Lambda`.
The first coefficient defines the baseline anchor; center environments at
the intended baseline before constructing this prior. Zero baseline variance
and pure amplification are supported. Optional `coupling` in `(-1,1)` moves
the joint indicator probabilities toward the positive or negative Fréchet
bound, preserving both marginal probabilities and the covariance. Zero means
independence; this parameter is a fraction of the feasible bound, not a Pearson
correlation. Tune it on validation data; it is not estimated during a fit.

`shrink_orthogonal_covariance(covariance, factors, environment_metric=S)`
supports direction-specific covariance shrinkage before prior construction.
It orders the eigen-directions of `S**(1/2) U S**(1/2)` by decreasing variance,
applies factors in `[0,1]`, and transforms back. It preserves baseline variance
and amplification. Obtain `S`, directions, and prior covariances from training
data. Equal eigenvalues require equal factors; otherwise directions would
depend on an arbitrary basis. The utility accepts a stack of annotation
covariances and can be combined with separate sparsity and indicator coupling.

## Python and CLI

```python
from summit.prediction import (
    MixtureSpec, MixtureSolverSpec, fit_mixture_prediction,
    plan_mixture_prediction,
)

options = dict(storage="stream", block_size=128, threads=8,
               memory_bytes=16 * 2**30)
plan = plan_mixture_prediction(traits, source, **options)
models = fit_mixture_prediction(
    traits, source,
    output="new_models",
    mixtures={(t.id, c.id): MixtureSpec(.01, .1)
              for t in traits for c in t.candidates},
    solver=MixtureSolverSpec(rtol=1e-7, max_sweeps=100),
    checkpoint="mixture_checkpoint.npz",
    **options,
)
```

Supply exactly one mixture specification per candidate. `p` must lie strictly
between zero and one; `f` can include zero or one, giving a spike component.
The setting `p=f=.5` reduces to a Gaussian prior. Inputs allow positive
participant-specific residual variances, rank-deficient fixed covariates,
singular genetic covariances and differing participant masks across traits.
The context dimension is limited to 32.

In the [CLI fit specification](Polygenic-scores.md#fit-and-inspect),
replace the `solver` field with the following setting. These snippets show the
fields to add to the full specification or candidate object.

```json
{
  "solver": {"kind": "mixture", "rtol": 1e-7, "max_sweeps": 100}
}
```

Add a `mixture` field to every candidate:

```json
{
  "mixture": {"probability": 0.01, "small_variance_fraction": 0.1}
}
```

For separate sparsity, use this candidate field:

```json
{
  "mixture": {
    "kind": "separate_sparsity",
    "baseline": {"probability": 0.01, "small_variance_fraction": 0.1},
    "response": {"probability": 0.1, "small_variance_fraction": 0.2}
  }
}
```

The usual `summit pgs plan`, `fit`, `--checkpoint`, `--resume`, and reloaded
`score` operations apply. Mixture candidates are rejected by the Gaussian
solver. Hyperparameter selection is a separate validation step; fitting a
candidate menu does not itself perform cross-validation.

## Computation and convergence

The variational distribution factors across SNPs and retains a joint response
distribution within each SNP. Fitting uses blockwise updates and cached local
Gram matrices. Candidates sharing a residual surface reuse genotype products.

No full genome-wide LD matrix is formed. Gram storage is proportional to
`M * block_size * Q**2` per distinct trait/residual group, with additional
posterior and fixed-projection caches included in memory planning. Compact
storage adds an `N*M` byte hard-call cache; streaming avoids that allocation.
Use the planner on the actual traits and candidate menu before choosing these
settings. An individual Gaussian or small-panel timing does not establish
full-menu throughput.

The fit checks simultaneous site updates and fixed-effect orthogonality
against independently reconstructed residuals. The saved model records
`method="mixture_vb_fixed_point"`, the convergence error, and its threshold.
The solution is a variational fixed point; an exact posterior or global
optimum is not guaranteed. Update order, initialization, and stopping criteria
can lead to different fitted weights across implementations.

Checkpoints save weights, residuals, site penalties, and objective history
after each complete sweep. Residuals are independently reconstructed after
the first sweep and every five sweeps by default (`residual_refresh`).
Appreciable numerical drift raises an error.

Resume requires matching inputs, solver settings, and implementation.
Rebuilding caches after restart requires another genotype pass. Existing model
directories are never overwritten. See [Installation](Installation.md)
for the native build requirements.

The Python API also accepts `initial_weights`, with exactly one finite array
per candidate, to initialize a new fit. Use either `initial_weights` or
checkpoint resume. Resuming uses the saved state and requires no original
initial-weight array.

## Matching an additive comparator

Match participants, variants, counted alleles, empirical versus HWE genotype
variance, phenotype units, fixed-covariate span, residual variance, total
genetic variance and mixture settings before comparing implementations.
For standardized genotypes with empirical raw variance `v[j]`, an LDAK power
parameter `alpha` corresponds to covariance weights proportional to
`v[j]**(1+alpha)`. These weights can be represented by `AnnotationDesign`.
The prior and residual variance must use the same residualized phenotype
variance, not an assumed unit variance.
