# Contextual covariance: development status and contracts

> **Estimator identity warning.** This status document concerns the sample-probe
> contextual covariance/action estimator. Do not use its grouped-action path as
> the generalized per-variant G×E LD-score implementation or its SNP-block
> jackknife. See `docs/generalized_gxe_variant_ldscore_contract.md`.

`summit.context` is a correctness-first programmatic framework with two
deliberately separate paths: dense small-data differential implementations and
private stable-V1 descriptor-backed reference/trait artifacts with a Python
summary fit. The stable path is implemented and under bounded release
qualification, but it is not exposed through the `summit` CLI and is not a
target-scale production claim. Existing additive SUMMIT and standardized
one-environment SUMMIT-GxE behavior is unchanged.

## Estimand and feature scale

For a fixed retained sample, rank-revealing residual projector `P`, declared
context basis `phi`, and one centered/pre-scaled genotype matrix `G`, the
features are

```text
F_q = P diag(phi_q) G.
```

The feature mode is recorded as `raw_projected`: all context columns share the
same pre-projection genotype scaling and there is no context- or SNP-specific
normalization after projection. The pre-projection scaling convention must be
recorded and compatible between a reference and study; the framework does not
repair cohort-specific scale mismatches.

Context pairs are ordered as all diagonals followed by lexicographic
off-diagonals. For `Q=3` this is

```text
(0,0), (1,1), (2,2), (0,1), (0,2), (1,2).
```

For annotation `k`, with mass `M_k`, the kernels are

```text
K_(k,q,q) = F_q A_k F_q' / M_k
K_(k,q,r) = (F_q A_k F_r' + F_r A_k F_q') / M_k, q < r.
```

The off-diagonal factor is present exactly once in the kernel and RHS. The
stored coefficient is the single matrix entry `Omega[q,r]`; it is not doubled
again. Components are annotation-major and pair-minor. Residual components are
declared separately as `P diag(d_h) P`.

Phenotypes are projected with the same `P` and normalized so `y'y=r`, where
`r=rank(P)`. Reference transport uses sample counts `N` and `N(N-1)`, never
`r` in place of `N`:

```text
T_S = (N_S/N_R) D_R
    + N_S(N_S-1)/(N_R(N_R-1)) (T_R-D_R).
```

## Input and summary contracts

The dense differential builders accept finite float64-compatible in-memory
arrays. The stable descriptor path reads sealed PLINK BED/BIM/FAM inputs,
binds file and retained-order identities, and applies one sealed variant
affine transform and missing-value policy during decode. In either path, the
sample order, variant order, basis calibration, fixed-effect specification,
residual order, genotype scale, and deletion grouping are fixed before
construction. Categorical levels use typed identities and unknown study
levels fail closed.

`build_context_reference` and `build_context_trait_summary` are the small-data
development builders. They bind the basis, fixed design, variants,
annotations, components, scaling, and grouping through canonical digests.
Reference objects contain genetic Gram and same-person moments; trait objects
contain exact study RHS, traces, genetic-residual terms, and residual moments.

`reference_v1.run_contextual_reference_v1` and
`trait_v1.run_contextual_trait_v1` each call one sealed native executor and
adapt only compact terminal results. `fit_v1.fit_contextual_model_v1` validates
the two stable inputs, performs transfer and approximate summary deletion, and
solves the small system in Python. Stable loaders accept only their exact
family, logical schema, backend, suffix, and published terminal state.

For a strict disjoint annotation partition,
`build_grouped_context_reference` and
`build_grouped_context_trait_summary` retain only lossless group aggregates.
Here `C = K Q(Q+1)/2` is the total genetic-component count:

```text
reference Gram numerators:       J x C x C
trait RHS and trace numerators:  J x C
trait genetic-residual terms:    J x C x H
annotation masses:               J x K
variant counts:                  J
```

After these summaries have been constructed, fitting reads no individual
phenotype, genotype, context, covariate, or sample-row data.

An optional stable fit evaluation grid is a trusted caller boundary. It must
be independently constructed and declared as `non_row_evaluation_grid_v1`;
subject context rows must not be republished as an evaluation grid.

## Approximate jackknife boundary

Summary-only inference uses SNP- or losslessly grouped contribution deletion.
For each frozen group, the implementation subtracts all genetic RHS, trace,
genetic-residual, and reference Gram numerators, divides by the retained
annotation masses, and reuses the full-reference same-person matrix `D_R`.
This is the accepted approximate LOO procedure. It is not an exact deleted
genotype-kernel or deleted-`D_R` jackknife.

All frozen balanced groups are required for reported equal-group covariance.
The fit stores every coefficient replicate and the complete joint jackknife
covariance. Rank failure in the full system or any replicate is explicit; no
ridge or hidden inverse is added.

## Output interpretation

Raw method-of-moments coefficients and covariance surfaces are primary. They
may be negative or indefinite and are never silently clipped. Optional
covariance-aware PSD projections are separate interpretive objects and can be
indeterminate when the optimization cannot be certified.

For strict disjoint annotations, each `Omega_k` is the total normalized-bin
contribution. `Omega_k/M_k` is a separately labelled per-unit-annotation-mass
coefficient. With overlapping annotations, coefficients and per-annotation
surfaces are conditional contributions rather than standalone covariance
matrices; the combined total surface is the default. Linear annotation
contrasts use the full joint LOO covariance and deleted-group traces/masses.

Derived nonlinear correlations, log scale ratios, conditional variances, and
individual covariance modes are reported only on their valid domains. Exact
or near eigenvalue ties identify a subspace rather than unique modes. Boundary
tests, simultaneous transformation bands, and learned directions remain
experimental and fail closed when their declared calibration or domain is not
available.

## Supported development modules

| module | status | scope |
|---|---|---|
| dense oracle, component index, transfer algebra | research-ready | exact small-data differential reference |
| trait/reference summaries and full-system fit | research-ready on bounded inputs | raw contextual moments and approximate LOO |
| binary/categorical presets | research-ready for deterministic estimation | nonlinear boundary tests remain experimental |
| transformed-phenotype scan | experimental inference | one-pass batching is validated; simultaneous bands are not production calibrated |
| fixed multi-environment basis and linear contrasts | research-ready on bounded inputs | covariance modes require the declared metric and tie diagnostics |
| learned context direction | experimental | exact two-fold selection/evaluation is descriptive, not calibrated inference |
| strict disjoint and nonnegative-overlap annotations | research-ready on bounded inputs | overlap outputs are conditional contributions; combined total is primary |
| private stable-V1 reference/trait/fit | bounded qualification candidate | descriptor-backed native summaries and Python compact fit; exact build and workload gates still apply |
| public contextual CLI and target-scale execution | not supported | no contextual CLI surface; modeled admission is not measured qualification |

## Numerical diagnostics

Every fit reports symmetric-system eigenvalues, numerical rank, condition,
effective threshold, null directions, and solve residual. Reference builders
record exact or fixed-probe method, probe identity, tiling, signed
same-person diagnostics, and phase timing. Resource admission reports
dimension-only conditioning as unknown unless an actual pilot matrix is
supplied; increasing probe count is not presented as a remedy for structural
rank deficiency.

## Performance boundary

The dense Python path remains a differential implementation and deliberately
materializes contextual features. The descriptor-backed native path implements
bounded source, action, grouped-numerator, same-person, and one-pass trait
phases with explicit allocation, call, event, and output admission. Its
existence does not establish target-scale readiness: each workload must fit an
exact plan and pass whole-process RSS, file-stability, NUMA, integrity, and
repeated-run gates on the selected build. Modeled target admission is labelled
modeled and is not a runtime extrapolation.

## Development examples and checks

Small end-to-end examples are the validation scripts under `scripts/context`.
They write aggregate JSON and PNG/PDF diagnostics to a fresh output directory.
For example:

```bash
python scripts/context/validate_context_fit.py --output-dir /tmp/context-fit
python scripts/context/validate_multienvironment.py --output-dir /tmp/context-multienv
python scripts/context/validate_context_annotations.py --output-dir /tmp/context-annotations
python scripts/context/dry_run_context_resources.py \
  --backend current --output-dir /tmp/context-resource-current
```

The deterministic, license-safe fixtures are generated in
`tests/test_context_*.py`. They include binary one-hot, continuous context,
nonorthogonal `Q=3` basis rotation, independent-reference transfer, explicit
rank failure, and strict disjoint annotations. Run them with:

```bash
python -m pytest -q tests/test_context_*.py
```

See the specialized contextual documentation and release audit for detailed
constructors, interpretation, exact validated commands, measured-versus-
modeled evidence, schema policy, and remaining release boundaries.
