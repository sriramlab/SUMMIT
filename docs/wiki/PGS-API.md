# PGS Python API

The main entry points are `plan_prediction`, `fit_prediction`, and
`score_prediction` in `summit.prediction`.

```python
from summit.prediction import (
    FileGenotypeSource, SolverSpec, plan_prediction,
    fit_prediction, score_prediction,
)

with FileGenotypeSource("discovery.bed", genome_build="GRCh37") as source:
    plan = plan_prediction(traits, source, storage="compact",
                           memory_bytes=16 * 2**30, threads=8)
    models = fit_prediction(traits, source, output="models",
                            plan=plan, solver=SolverSpec(rtol=5e-4))
    scores = score_prediction(models, source, score_inputs, threads=8)
```

Here `traits` contains `TraitTraining` objects and `score_inputs` maps trait
names to `ScoreInput` objects. Set the numerical-library thread environment
before importing the backend; `summit-pgs` does this automatically.
For the eight-thread Python example above, launch your script with:

```bash
env BLIS_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python fit.py
```

`scripts/prediction/demo.py` is a complete synthetic construction of both inputs.

## Building a trait

`TraitTraining` combines source row and variant indices, y, the context matrix
Phi, fixed-effect matrix Z, genotype scale, candidate priors, and saved feature
recipes. Contexts start with a baseline column of ones. Variant indices follow
source order; each trait can have its own sample order and SNP subset.

`features.fit_contexts` estimates continuous centers/scales and categorical
coding on the discovery rows. It returns Phi, a reusable recipe, and a context
metric. `evaluate_contexts` applies that recipe to new data.

```python
from summit.prediction.features import fit_contexts

phi, context_spec, metric = fit_contexts(
    discovery_contexts,
    [{"name": "exposure", "kind": "continuous"},
     {"name": "group", "kind": "categorical",
      "categories": ["a", "b"], "reference_category": "a"}],
)
```

`evaluate_fixed` builds named fixed-effect terms from raw covariates and saved
context columns. An empty factor list creates an intercept. Products can
combine several factors, and each factor can have an integer power from 1 to 8.

```json
{
  "kind": "summit.prediction.fixed", "schema_version": 1,
  "terms": [
    {"name": "intercept", "factors": []},
    {"name": "exposure", "factors": [
      {"source": "context", "name": "exposure", "power": 1}]},
    {"name": "pc1", "factors": [
      {"source": "covariate", "name": "pc1", "power": 1}]}
  ]
}
```

`estimate_scale` and `write_genotype_scale` create a discovery scale when a new
architecture analysis is being prepared. When importing an existing generalized
reference, `adapters.scale_from_generalized_reference` uses its saved affine
parameters. It also requires the original sample and allele order: older
reference files do not establish those from dimensions alone.

## Priors and residual variance

`CandidatePrior(id, covariance, residual, specification)` accepts a symmetric
positive-semidefinite covariance and a positive residual vector. Singular and
zero genetic covariance are valid. Record how the prior and residual model were
estimated. The library does not fit that architecture from the phenotype for you.

For CLI use, `adapters.load_prior` reads a version-1 `summit.prediction.prior`
JSON containing covariance, context/scale identifiers, and provenance. Optional
geometry stores the response metric, reference description, and anchor.
`tests/test_prediction_cli.py` shows how to write the complete file set.

Residual JSON uses kind `summit.prediction.residual`, version 1, a table `file`
and `column`, `units: "model_phenotype_variance"`, a nonnegative `floor`, and
`provenance`. Its table path is relative to the residual JSON. Choose and record
the floor before comparing genetic candidates.

| Candidate operation | CLI parameters |
|---|---|
| Scale the complete covariance | `common_scale`: `kappa` |
| Scale amplification and residual response separately | `separate_scales`: `kappa_a`, `kappa_h` |
| Shrink response eigenvalues | `spectral_shrinkage`: `tau`, `kappa` |
| Retain a response rank | `spectral_rank`: `rank`, `kappa` |
| Use another estimated covariance | `supplied`: `covariance`, `provenance` |

Changing a prior requires fitting new SNP weights. `ResponseGeometry.weights`
instead combines components of an already fitted score. Those are separate
operations. A spectral rank cut cannot divide a nonzero tied eigenspace.

## Model files

`fit_prediction` writes a new directory containing a JSON manifest, numeric NPY
arrays, and a completion marker. Reload with `load_prediction_models` before
scoring elsewhere. Incomplete or modified bundles are rejected.

Saved models include weights, alleles and build, genotype scaling, feature
recipes, fixed coefficients, prior specification, and convergence results.
They omit training phenotypes, solver vectors, and sample-aligned residual
variances. Keep locally generated sample predictions in protected storage.

For standardized weights B, `model.raw_weights()` returns dosage weights
`W = inverse_scale * B` and offsets `c = -mean.T @ W`. The offset contribution
is `Phi @ c`, so it can depend on context. It cannot generally be replaced by
one global intercept.

The solver accepts a fit only after evaluating its true residual against the
requested tolerance. A failed candidate raises `ConvergenceError`; no complete
model bundle is published. See [Methods](Methods.md) for the linear system.

## Long fits and restart state

Pass `checkpoint="private/solver.npz"` to `fit_prediction` to save projected-PCG
state after each completed solver pass. If the process is interrupted, repeat
the call with the same arguments and `resume=True`. CLI equivalents are
`--checkpoint private/solver.npz` and `--resume`.

The checkpoint stores search directions, residuals, solutions, pending true
checks, and already verified candidates. It does not store the genotype cache;
resuming rebuilds that cache in one source pass. An interrupted pass is repeated.
The file is replaced atomically and an exclusive sidecar lock rejects duplicate
writers. The input, prior, solver, Python/native implementation, thread count,
and block/RHS sizes must match. Existing checkpoints require explicit resume;
existing output bundles are never overwritten. A solved checkpoint can export
to a new directory if interruption left an incomplete model directory.

Checkpoint vectors are individual-level training data. Keep the checkpoint and
its `.lock` alongside protected training inputs, outside portable model bundles.
Saving requires space for two generations during atomic replacement (about
`6 * 8 * N * candidates` bytes plus small metadata for a single trait).

The native executor fuses sample/SNP selection, allele orientation, missing-call
imputation and FP64 scaling into one reusable block. It also reuses GEMM scratch.
There is no genome-sized FP64 expansion in compact BED mode. Candidates awaiting
true-residual verification remain frozen while their checks share the next
genotype traversal with other candidates' CG steps. This does not relax any
convergence threshold. The pass ledger distinguishes `cg`, `verification`, and
`cg_and_verification`.

For bound OpenMP workers, set explicit singleton `OMP_PLACES` and the other
OpenMP controls before importing numerical libraries. The prediction API now
registers this placement with the native backend, which gives BLIS workers the
complete reserved CPU set. Ambiguous bound placement fails before cache setup.
For example, on an allocation containing physical CPUs 0, 2, 4 and 6:

```bash
env OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=FALSE \
  OMP_MAX_ACTIVE_LEVELS=1 OMP_PROC_BIND=SPREAD OMP_PLACES='{0},{2},{4},{6}' \
  BLIS_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python fit.py
```

Use the CPUs actually assigned by the scheduler. Native builds predating
`prediction_execution_version=2` must be rebuilt for this Python implementation.

## Calibration and evaluation

`select_and_calibrate` compares frozen score candidates on pilot data using
nested folds and ridge mean calibration. Centering and scaling are estimated
inside each training fold. Save the selected calibration and reload it with
`load_mean_calibration` for final evaluation.

`calibration.fit_calpred` calls a separately installed Gaussian CalPred R
implementation using explicit mean and variance features and a local temporary
directory. It retains the fitted interval parameters and backend diagnostics.
It does not install R packages.

`paired_r2_gain` gives a paired bootstrap interval for predictive R² differences
between fixed fitted models. It does not include uncertainty from retraining.
Mean calibration, phenotype prediction intervals, and genetic-effect posterior
uncertainty answer different questions; the current SNP fitter exports posterior means.
