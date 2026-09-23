# Mixture shrinkage for SUMMIT-pgs

`fit_mixture_prediction` fits a joint mixture prior over each SNP's baseline
and environment-response coefficients. It uses SUMMIT's genotype readers,
protected matrix products, annotation priors, model artifacts and scoring API.
It does not invoke LDAK. The existing Gaussian solver remains the default.

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

For a CLI fit specification, set

```json
"solver": {"kind": "mixture", "rtol": 1e-7, "max_sweeps": 100}
```

and add to every candidate:

```json
"mixture": {"probability": 0.01, "small_variance_fraction": 0.1}
```

For separate sparsity, use:

```json
"mixture": {
  "kind": "separate_sparsity",
  "baseline": {"probability": 0.01, "small_variance_fraction": 0.1},
  "response": {"probability": 0.1, "small_variance_fraction": 0.2}
}
```

The usual `summit-pgs plan`, `fit`, `--checkpoint`, `--resume`, and reloaded
`score` operations apply. Mixture candidates are rejected by the Gaussian
solver. Hyperparameter selection is a separate validation step; fitting a
candidate menu does not itself perform cross-validation.

## Computation and convergence

The variational distribution factors across SNPs, retaining a full joint
response distribution within each SNP. Blockwise Gauss–Seidel updates use
cached projected Gram matrices. Candidates sharing a residual surface reuse
the same genotype products. Fixed-covariate projections and component
posterior precisions are cached once; later passes apply the projection to
candidate updates instead of projecting all SNP columns again.

No full genome-wide LD matrix is formed. Gram storage is proportional to
`M * block_size * Q**2` per distinct trait/residual group, with additional
posterior and fixed-projection caches included in admission planning. Compact
storage adds an `N*M` byte hard-call cache; streaming avoids that allocation.
Use the planner on the actual traits and candidate menu before choosing these
settings. An individual Gaussian or small-panel timing does not establish
full-menu throughput.

Convergence requires an independent simultaneous site-update check after
reconstructing the residual from all saved weights, as well as fixed-effect
orthogonality. The artifact records `method="mixture_vb_fixed_point"` and the
observed error and threshold. This certifies a variational fixed point; it is
neither an exact posterior certificate nor a guarantee of the global optimum.
The objective is monitored for decreases. LDAK can reach a different fixed
point because update order, initialization and stopping criteria differ.

Checkpoints atomically retain weights, residuals, site penalties and objective
history after each complete sweep. Residuals are independently reconstructed
after the first sweep and every five sweeps by default (`residual_refresh`); appreciable accumulated
drift raises an error. Mixture matrix products, including fixed-effect projection
and residual reconstruction, use the deterministic native tiled path. The fixed
basis also undergoes an independent orthonormality check. A C++ workspace owns
the residual and reconstruction state, while native interaction-design
construction avoids Python broadcast temporaries. Checkpoints receive explicit
snapshots and restore into the same native state. The private-BLIS
runtime protection contract remains required.
Resume authenticates source, inputs,
solver settings, Python implementation and native binary. Rebuilding caches
after restart requires another source traversal. Existing model directories
are never overwritten. The BLIS path requires both integrity and checksum
guards and the established native worker-placement contract.

The Python API also accepts `initial_weights`, with exactly one finite array
per candidate. This starts a new fit and reconstructs residuals and variational
state. It cannot be combined with checkpoint resume, and the initial arrays
are hashed in checkpoint initialization provenance. Resume authenticates the
unchanged fitting problem and saved iterative state without requiring the
original warm-start array again.

## Matching an additive comparator

Match participants, variants, counted alleles, empirical versus HWE genotype
variance, phenotype units, fixed-covariate span, residual variance, total
genetic variance and mixture settings before comparing implementations.
For standardized genotypes with empirical raw variance `v[j]`, an LDAK power
parameter `alpha` corresponds to covariance weights proportional to
`v[j]**(1+alpha)`. These weights can be represented by `AnnotationDesign`.
The prior and residual variance must use the same residualized phenotype
variance, not an assumed unit variance.

Qualification covers the Gaussian limit against dense GLS, joint posterior
quadrature, singular and annotated priors, basis rotations, native/reference
agreement, interruption/resume and artifact scoring. Full-array LDAK parity
and throughput remain empirical checks; small-panel agreement is insufficient.
