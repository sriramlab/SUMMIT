# Contextual posterior prediction

`summit.prediction` fits joint Gaussian SNP posterior means from discovery
genotypes, outcomes, fixed effects, a sealed genotype scale, a genetic prior,
and positive residual variances. `summit-pgs` is a separate opt-in CLI. It does
not change LD reference estimation, architecture fitting, or jackknife deletion.

The implementation uses FP64 throughout. BED compact storage retains exact
0/1/2 calls with a -127 missing sentinel. PGEN retains fractional dosages and
uses FP64 compact storage. Native BED reading reuses SUMMIT's existing selected
row decode machinery and descriptor checks. Products use the existing guarded
FP64 native GEMMs. The NumPy backend is an explicit small reference backend.

## Install and run from a branch

Build/install this branch using the project's existing CMake/scikit-build
instructions. A normal installation supplies `summit-pgs`; the equivalent module
entry is `python -m summit.prediction`. A private BLIS build needs the existing
explicit archive/header and upstream provenance settings; prediction does not
silently select or download a BLAS implementation.

For a separate checkout sharing an environment with an older SUMMIT editable
installation, use `scripts/prediction/checkout.py`. It selects this checkout in
the current process without changing the shared environment. Build native
modules into `src/summit` with `-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=<checkout>/src/summit`.

```bash
python scripts/prediction/checkout.py test -q
python scripts/prediction/checkout.py demo --out /new/path/synthetic-pgs
python scripts/prediction/checkout.py plan --spec fit.json --memory-gib 128
```

The demo fits six models across two masks and scores held-out synthetic people.
`tests/test_prediction_cli.py` is a complete file-based BED/JSON/TSV example.

## Mathematical contract

For each trait, `G=(dosage-mean)*inverse_scale`, `Phi=[1,e]`, and
`beta_j ~ N(0,Lambda/M)`. Missing calls use the discovery mean and therefore
contribute zero in G. No context-specific post-projection normalization is used.
The exact covariance is

```text
V = diag(r) + (G G.T / M) * (Phi Lambda Phi.T)
P = I - U U.T,  U = thin rank-revealing basis of the fixed design Z
P V P u = P y,  P u = u
B = G.T @ (Phi * u[:,None]) @ Lambda / M
```

Projection is implicit and never allocates an N-by-N matrix. Projected PCG uses
`P diag(V)^(-1) P`; it does not replace GLS with an unprojected heteroskedastic
solve after residualization. Singular, amplification-only, additive and zero
priors are valid. No inverse or artificial ridge is applied to Lambda.

Each candidate has its own convergence and restart state. Every success is
verified by a new covariance application using the true residual. Success means
`||P y-P V u|| <= max(atol,rtol*||P y||)` with a fixed-span leakage check; there
is no hidden 1.05 tolerance allowance. Nonconvergence raises `ConvergenceError`
with per-model reports and prevents publishing a complete model bundle.
Fixed coefficients are recovered in exactly the retained QR span from `y-Vu`.
They use the minimum-norm representation on that identified span.

`common_scale`, `separate_scales`, and `ResponseGeometry.prior` implement the
prior changes in the design. `ResponseGeometry.decompose` and `.weights(tau)`
implement post-fit response-score weighting. Changing a prior requires a refit.
Spectral truncation differs from a covariance estimated under a profiled rank
constraint: supply the latter as its own covariance, with provenance.

The geometry uses the metric-weighted Schur spectrum. A loaded model with
positive fitted baseline variance derives gamma from **its fitted prior**;
profiled restrictions can change a and b. With zero fitted baseline variance,
the declared positive-baseline parent anchor is retained and fitted baseline
weights are zero. A rank cutoff cannot split an identified nonzero tied
eigenspace. A singular metric requires an explicit reference-fitted support
transform. Numerical rank and predictive regularization are not biological
rank estimates.

## Python API

```python
from summit.prediction import (
    FileGenotypeSource, TraitTraining, CandidatePrior, SolverSpec,
    plan_prediction, fit_prediction, score_prediction, ScoreInput,
)

with FileGenotypeSource("cohort.bed", genome_build="GRCh37") as source:
    # traits is a sequence of fully validated TraitTraining objects.
    plan = plan_prediction(traits, source, storage="compact",
                           memory_bytes=128 * 2**30, threads=16)
    models = fit_prediction(traits, source, output="new-models",
                            plan=plan, solver=SolverSpec(rtol=5e-4))
    scores = score_prediction(models, source, score_inputs, threads=16)
```

`TraitTraining` contains source row/variant indices; phenotype y; Phi and Z;
`GenotypeScale`; candidate covariance/residual inputs; context/fixed recipes;
phenotype units; and optional reference metric/anchor geometry. Rows may have
trait-specific order. Training variants follow increasing source order. Every
candidate's dimensions, symmetry, PSD, residual positivity and scientific axes
are checked before genotype work. Equal residual surfaces share storage.

The scale binds ordered samples, variants/alleles, affine vectors, arithmetic,
ddof and source identity. `estimate_scale` is an **explicit** discovery-scale
operation; use it before obtaining an exact architecture reference. An existing
reference must use its original scale. `adapters.scale_from_generalized_reference`
requires independent sample/variant authentication because older generalized
reference artifacts do not establish those identities by themselves.

Copied/repacked genotype sources have different descriptor identities. Establish
and record equivalence before constructing a scale adapter for the new source;
the fitter will not silently accept a changed source under an old scale.
`ShardedGenotypeSource` accepts ordered chromosome trios with exactly matching
sample axes and unique combined variants. Shared sample-ID metadata is retained
once. A source has one prepared stream owner at a time; stale stream reuse fails.

Context recipes reuse `context.multienvironment` calibration/evaluation.
`fit_contexts` fits continuous means/scales and categorical levels on discovery;
`evaluate_contexts` applies the frozen recipe, including an optional recorded
linear support/recoding transform. Unknown levels and nonfinite values fail.
`evaluate_fixed` supports explicit products and powers of frozen contexts and
raw numeric covariates. API callers may supply precomputed designs with complete
ordered names and transform metadata; scoring requires matching specifications.

## CLI specifications

```bash
summit-pgs plan --spec fit.json --genotype-storage compact --memory-gib 128 --num-threads 16
summit-pgs fit --spec fit.json --out new-models --genotype-storage compact --memory-gib 128 --num-threads 16
summit-pgs inspect new-models
summit-pgs score --models new-models --spec score.json --out new-scores --memory-gib 16 --num-threads 16
```

`plan` reads metadata, scale/prior artifacts and selected trait feature tables,
validates the scientific inputs and estimates allocations. It does not scan
genotypes or allocate iterative/cache workspaces. `--block-size` and
`--rhs-columns` control bounded SNP and complete-model tiles. Explicit NUMA
binding uses `--numa-mode membind --numa-nodes ...` before numerical imports.
Use fresh processes for different native thread ownership contracts.

Input tables and keep files have a header with string `FID IID` columns.
No positional join or integer conversion of IDs is performed. Paths in the fit
specification are relative to that specification; a residual table path is
relative to its residual JSON. Variant selection files contain whitespace-
separated SNP IDs without a header; the retained axis is source order.

The closed fit schema is:

```json
{
  "kind": "summit.prediction.fit_spec",
  "schema_version": 1,
  "genotypes": {"geno": "cohort.bed", "genome_build": "GRCh37"},
  "traits": [{
    "id": "trait1",
    "phenotype": {"file": "traits.tsv", "column": "trait1", "units": "trait units"},
    "samples": "discovery.keep",
    "contexts": "contexts.tsv",
    "context_spec": "contexts.json",
    "covariates": "fixed.tsv",
    "fixed_spec": "fixed.json",
    "genotype_scale": "discovery.scale",
    "architecture_prior": "prior.json",
    "residual_spec": "residual.json",
    "candidates": [{"id": "full_k0.5", "operation": "common_scale", "kappa": 0.5}]
  }],
  "solver": {"rtol": 0.0005, "max_iterations": 150}
}
```

Replace `geno` with `shards: ["chr1.bed", "chr2.pgen", ...]` for ordered trios.
Optional trait `variants` supplies a variant-selection file. Optional phenotype
`center` and positive `scale` apply a linear transformation to y; supplied R
must already be in that transformed variance unit. Scores remain in the model's
phenotype units; inverse linear conversion is `scale*prediction+center`.
Nonlinear inverse means require a separate statistical interpretation.

A prior JSON has kind `summit.prediction.prior`, schema version 1, `covariance`,
`context_identity=digest(context_spec)`, `scale_identity=scale.identity`, and
`provenance`. Optional fields are `raw_covariance`, `coefficient_covariance`, and
`geometry: {metric, reference, anchor}`. The latter metric is H for the response
terms, not the full basis metric. Raw covariance estimates remain provenance;
the covariance admitted for prediction must already be admissible.

A residual JSON has kind `summit.prediction.residual`, schema version 1, `file`,
`column`, `units: "model_phenotype_variance"`, nonnegative absolute `floor`, and
`provenance`. Record the floor fraction/reference median and architecture fit
in provenance when relevant. All genetic candidates of that trait share the
same admitted R. The library does not silently reprofile or change it.

Candidate operations and additional required fields:

| Operation | Fields |
|---|---|
| `common_scale` | `kappa` |
| `separate_scales` | `kappa_a`, `kappa_h` |
| `spectral_shrinkage` | `tau`, `kappa` |
| `spectral_rank` | `rank`, `kappa` |
| `supplied` | `covariance`, `provenance` |

The score specification has kind `summit.prediction.score_spec`, version 1,
`genotypes`, and `traits: [{id, samples, contexts, covariates}]`. It obtains the
context/fixed recipes from the saved model. Optional `missing_variants` is
`error` (default) or `mean_impute`. The latter retains the original M and reports
coverage; it never renormalizes remaining effects.

`summit-pgs scale --geno ... --genome-build ... --samples ... --out ...` explicitly
creates a new scale artifact. It is not an implicit replacement for a reference
scale. Sharded scale creation is available through the Python source API.

## Artifacts and scoring

Model bundles use a closed JSON manifest and uncompressed numeric NPY members.
Each model's FP64 weights are written sequentially in bounded variant blocks;
the whole M-by-Q candidate panel is never required in RAM. The manifest records
axes/alleles/build, scales and units, context/fixed recipes, fixed coefficients,
prior/residual provenance, geometry, true convergence, source and Python/native
identities, resource estimates and the traversal ledger. Training y, u, U and
sample-aligned R are absent from portable model bundles.

Publication requires a new directory, fsynced members, checked lengths/hashes,
and `COMPLETE.json` written last. Existing results are never replaced. Readers
reject incomplete bundles, unsafe/symlink members, unknown schema fields,
nonfinite arrays, wrong shapes/dtypes, corruption and mutable weight files.
An incomplete failed directory is left visibly incomplete for inspection; it
is not resumed or overwritten automatically.

Scoring aligns IDs, coordinates and explicit allele pairs. Biallelic swaps
transform calls into the model's counted allele; no strand inference occurs.
BED counts BIM A1, and the reused PGEN reader counts REF. Only autosomal diploid
biallelic SNPs are admitted in V1. Missing calls use the training mean.

For standardized B, raw dosage weights are `W=s*B` and offsets are
`c=-mean.T@W`. The full score offset is `Phi@c`, generally a context-dependent
quantity. `PredictionModel.raw_weights()` returns both arrays. Scoring uses the
persisted FP64 weights and reports component, genetic and total fixed-plus-
genetic predictions with sample/model identities.

## Shared execution, memory and traversal accounting

Each covariance application reads a raw union block once. Trait-local row and
variant views apply their own sealed scales. Identical genotype descriptors
share a standardized block. Context-weighted active RHS are packed once per
application; bounded full-model tiles then compute `G_block.T @ RHS`, mix
contexts by Lambda/M, immediately multiply back by G, and collapse into FP64
sample/model outputs. No genome-wide SNP intermediate or N-by-N matrix is used.
LD remains implicit in the global iterative solve. Trait sharing is
computational; it does not introduce a cross-trait statistical prior.

Storage modes are `stream`, `compact`, and `standardized`. Compact setup fuses
cache construction and exact row-diagonal setup. Standardized storage is FP64
and separate per incompatible scientific descriptor. The planner accounts for
cache, ragged state, packed RHS, native tiles/integrity reserve, axes, source
blocks and runtime reserve, and binds the plan to the concrete fit inputs.
These are conservative allocation estimates, not a guaranteed OS RSS ceiling.
The scorer also admits resident model weights and output arrays against its
budget; larger outputs require explicitly partitioned calls and extra passes.

Let I be CG rounds and C be verification calls. An uncached fit needs
`1 + I + C + 1` discovery traversals: setup, CG, verification, extraction.
Compact storage builds in one source pass; subsequent discovery traversals use
the cache. Standardized mode has the same source-pass count with different
resident storage. Scoring needs one union traversal for all admitted models and
traits. Candidate convergence can require more than one verification pass; all
are counted. Tilings never add genotype traversals. Logical blocks/variant
visits and decoded bytes are recorded separately from physical disk I/O.

The accepted native runtime checks remain active. Changing BLAS teams or
backend arithmetic requires a new process/run identity. FP32, supported-rank
RHS acceleration, persistent cache reuse, checkpoint/resume and automatic
memory waves are future optimizations; V1 rejects insufficient budgets rather
than silently changing arithmetic or rescanning per RHS tile. Full biobank
memory/throughput qualification remains necessary before claiming the earlier
96-GiB engineering target or a production speedup.

## Optional selection and uncertainty

`selection.select_and_calibrate` accepts a complete catalog of frozen score
feature candidates, penalty masks and ridge strengths. It uses fold-local
centering/scaling and augmented least squares, nested pilot selection, then
refits the selected mean calibration on pilot data. A supplied discovery sample
list is checked for overlap. Save the calibration JSON and the selection report;
`load_mean_calibration` applies the frozen representation. Candidates can include
an external additive score, its context products, full BLUP, and orthogonal
response additions. No discovery weights are modified.

`calibration.fit_calpred` invokes a caller-installed official Gaussian CalPred
R script with a pinned checksum, explicit full-rank mean/variance features and
an approved temporary root. It retains backend diagnostics and supports portable
interval parameters. It does not install packages, transform y, hide rank
reduction, or supply an interval model from R alone. The caller supplies
discovery/pilot/replication separation and matched comparison features.

Point calibration, phenotype prediction intervals, posterior genetic-effect
uncertainty and paired predictive-R2 uncertainty are distinct targets.
`paired_r2_gain` computes conditional paired bootstrap intervals with a
resampled phenotype-variance denominator. It does not include retraining
uncertainty. Including full BLUP in the catalog supplies no finite-sample
never-worse guarantee. The initial fitter exports posterior means, not posterior
genetic-effect uncertainty.

## Validation and observed small-run performance

The prediction tests independently check dense heteroskedastic GLS, singular
and zero priors, raw/standardized conversion, Schur/basis identities, masked
trait execution, convergence failure, partial blocks, dosage preservation,
allele and source alignment, sharded sources, corruption, artifact reload,
CLI operation, calibration isolation and traversal counts. Existing fixed,
multienvironment, generalized-native and PGEN regression tests are also checked.

The focused combined run passed 107 tests (six unrelated CLI/end-to-end cases
were deselected). The prediction-only run passed all 24 tests. Reproduce the
combined selection from the repository root with:

```bash
python scripts/prediction/checkout.py test \
  tests/test_prediction_core.py tests/test_prediction_io.py \
  tests/test_prediction_cli.py tests/test_prediction_selection.py \
  tests/test_prediction_edges.py tests/test_context_multienvironment.py \
  tests/test_context_fixed.py tests/test_context_fit.py \
  tests/test_generalized_gxe_native.py tests/test_pgen_gwld.py \
  -q -k 'not cli_dispatch and not end_to_end'
```

All three existing native targets (`gxeldcore`, `gwldcore`, `winldcore`) must be
built for this regression selection. The synthetic demo's six fits attained a
maximum relative true residual of `7.64e-11` at `rtol=1e-10`, then scored all
models on 24 held-out samples in one genotype traversal.

The local benchmark command is:

```bash
python scripts/prediction/checkout.py benchmark --mode compact --threads 1
```

Run modes in separate processes. The synthetic BED benchmark uses N=2,048,
M=4,096, two masks (1,755/1,792 rows), ten models per trait, Q=5, block=256,
RHS width=64, one thread, a warmup and three repetitions. On the reviewed tabla
environment with private pthread BLIS, initial observed measurements were:

| Storage | Setup seconds | Median operator seconds | Process peak RSS, Linux KiB |
|---|---:|---:|---:|
| Stream | 0.1738 | 0.3730 | 227,744 |
| Compact | 0.1780 | 0.3296 | 229,456 |
| Standardized FP64 | 0.1708 | 0.1788 | 267,384 |

Repeat operator differences were zero in these checks. Each mode made five
logical traversals: setup plus four operators. Stream visited 20,480 source
variants; cached modes visited 4,096 source and 16,384 cached variants. Peak RSS
includes synthetic fixture generation and is not an isolated fitting-phase
peak. These measurements establish a bounded smoke test and show the conversion
tradeoff; they do not predict full-cohort throughput. The optional official
CalPred adapter also passed a 160-row synthetic integration check with finite
intervals and no backend diagnostics.

Large comparisons belong in completion-gated Hoffman jobs with scratch outputs,
fixed total core/memory budgets, achieved true-residual controls and measured
time through export/reload/scoring. No large scientific jobs were launched for
this initial implementation.
