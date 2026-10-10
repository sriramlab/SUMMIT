# Generalized G×E PCGC

PCGC estimates the covariance of additive and context-dependent SNP effects
for a binary trait, accounting for case–control sampling. A context basis
`[1, E1, E2]` gives six parameters per annotation: additive variance, two
interaction variances, and their three covariances. Continuous exposures,
categorical variables encoded numerically, and supplied nonlinear basis
columns use the same interface.

## Model and scale

For population-scaled genotypes X and context vector φ, the genetic model is

```math
g_i=\sum_j X_{ij}\phi_i^T\beta_j,\qquad
\mathrm{Cov}(\beta_j)=\sum_a A_{ja}\Omega_a/M_a,
\qquad M_a=\sum_j A_{ja}.
```

Each Ω is a symmetric effect-covariance matrix. Annotation weights must be
nonnegative; annotation sets may overlap. Overlapping annotation coefficients
describe their joint model and require identifiable kernels.

The binary outcome follows a liability threshold. Specify either unit total
liability variance conditional on covariates and contexts, or known positive
conditional liability SDs on a common scale. Disease risks alone cannot
identify an arbitrary liability-variance function. Under the unit-variance
model, changes in genetic variance are accompanied by changes in residual
variance that keep their sum equal to one.

PCGC uses the first-order relationship between distinct-person binary
covariance and liability covariance. It requires correctly specified disease
risks, case-status sampling, exogenous contexts, and compatible population
genotype scaling. Higher-order correlations, fitted nuisance parameters,
and a finite randomized reference can affect finite-sample estimates.

## Inputs

Use the genotype, population-scale, and sample-table formats described in
[Binary traits and PCGC](Binary-traits-and-PCGC.md). The sample table
must include `FID IID Y E1 E2`, with Y coded 0/1, plus any risk covariates.
Context columns are used as supplied; SUMMIT adds an intercept. Choose the
exposure centers, scales, and categorical reference levels before preparation.

Provide population prevalence with `--binary-prevalence`. A fitted probit
risk model includes the context columns and `--binary-covariates`.
Alternatively, `--binary-risk-column RISK` supplies individual population
risks. Ordinary logistic GWAS beta/SE files cannot replace these inputs.

Use `--binary-unit-liability` for the unit conditional variance model.
For known heterogeneous SDs, use `--binary-risk-column RISK` together with
`--binary-liability-sd-column SD`.

## Prepare reference and trait moments

`--make-binary-sumstats` computes both the PCGC directional LD scores and
the trait score moments:

```bash
summit --binary-method pcgc \
  --make-binary-sumstats people.tsv --geno study.bed \
  --binary-scale population_scale.tsv --binary-prevalence 0.05 \
  --binary-context-columns E1,E2 --binary-unit-liability \
  --binary-covariates AGE,PC1,PC2 \
  --binary-genotype-covariates PC1,PC2 \
  --binary-sampling-partners 128 --binary-architecture-probes 32 \
  --nvecs 256 --memory-gib 8 --num-threads 2 \
  --out results/disease_gxe
```

Preparation writes two NumPy archives:

| File | Contents |
|---|---|
| `results/disease_gxe.binary.ldscores.npz` | PCGC directional reference rows, annotations, and any reference uncertainty arrays |
| `results/disease_gxe.binary.sumstats.npz` | Per-SNP context-pair trait score products, population moments, and any sampling or SNP-effect covariance summaries |

Both the reference rows (`ldscores`) and trait score products (`rhs_rows`) have
same-person terms removed. The NPZ format retains their multiple context and
annotation dimensions. The files record variant order, alleles, context names,
and the sample, risk, genotype-scale, and liability-scale definitions.

Add `--annot annotations.tsv` for partitioned estimates. Changes to risks,
prevalence, genotype scaling, context coding, or sample selection require new
preparation. Choose a new output prefix for each preparation.

## Fit the saved moments

```bash
summit --binary-method pcgc \
  --h2 results/disease_gxe.binary.sumstats.npz \
  --ldscores results/disease_gxe.binary.ldscores.npz \
  --njack 100 --out results/disease_gxe_fit
```

Use the same PCGC method as in preparation. Fitting writes
`results/disease_gxe_fit.binary.json` and requires no genotype input. You can
reuse the two files with a different `--njack` value and a new output prefix.

### Matching reference and summary files

The summary records the expected reference identity. Fitting checks the stored
sample, risk, scaling, context, annotation, and variant definitions, including
alleles and any genome-build label. It also checks the reference array contents
and both files' checksums. A mismatch stops the fit.

Use the reference written with the summary. The checks require that particular
reference realization, including any uncertainty arrays, because the saved
sampling calculations can depend on it. Matching SNP names or dimensions alone
is insufficient. You can move or rename the files; matching uses their contents.
Standard PCGC's risk-weighted reference is generally specific to the disease
and analyzed sample.

Existing combined `.binary.npz` files remain accepted through `--h2` alone.
To prepare that format, add `--binary-output-format combined`. Separate files
are the default for all binary methods. The [input guide](Input-files.md#files-used-for-inference)
compares the formats across workflows.

## Covariate adjustment

Genotype-PC adjustment removes the supplied PC directions from X before
context and risk weighting. These PCs also enter fitted risks. The binary
response and weighted context features receive no additional least-squares
covariate projection. Add PC-by-context columns to the risk covariates when
the mean model requires them; SUMMIT does not create those columns.

## Four modes

Use `pcgc` for the primary analysis. Its study-specific directional LD scores
use features `F_q = diag(d/s) diag(phi_q) X`, where d is the PCGC risk
sensitivity and s is the supplied conditional liability SD. The sensitivity
depends on population risk and case–control sampling, as defined in
[PCGC methods](PCGC-methods.md#risk-adjustment-and-estimating-equations). Trait scores
use the same features, and preparation removes same-person terms from both
the trait and reference moments. Ordinary quantitative-trait directional LD
scores generally cannot be reused for this fit.

| Method | Reference and weighting | Additional inputs or assumptions |
|---|---|---|
| `pcgc` | Study-specific risk-weighted context reference | Population risks and genotype scale |
| `pcgc-basis` | Exact contraction of a supplied risk-sensitivity basis | `--binary-basis-columns` and `--binary-basis-coefficients` must reproduce sensitivity exactly |
| `pcgc-inverse` | Inverse risk weighting with an unweighted context reference | Inverse weights need finite variance; strong risk differences can cause instability |
| `pcgc-ld` | Independent population LD reference | `--binary-reference-geno` and `--binary-ld-factorization`; requires risk/context/genotype factorization |

The risk-sensitivity basis and the genetic context basis have separate roles.
An exact sensitivity basis reproduces standard PCGC. External LD is an
approximation whose validity depends on the stated factorization. With
genotype-PC adjustment, external LD also needs a matching PC table through
`--binary-reference-covariates`.

## Estimates and uncertainty

`omega` contains one matrix per annotation; `omega_total` is their sum.
For contexts x and z, `x.T @ omega_total @ z` describes genetic covariance
under the standardized kernel model. Estimates remain signed, and fitted
matrices can have negative eigenvalues. Inspect the conditioning and
eigenvalue diagnostics before interpreting individual components.

`population_heritability` is total SNP heritability, including G×E and its
covariances with additive effects. Its numerator averages the genetic kernel
diagonal over the population, using the actual adjusted genotype diagonals.
Its denominator is the marginal liability variance. When conditional genotype
variances are one, the numerator reduces to

```math
V_g=\sum_a\mathrm{tr}\{\Omega_a E_{\mathrm{pop}}[\phi\phi^T]\}.
```

Population moments use inverse ascertainment weights by default. The Python
API also accepts supplied population moments. Context-specific heritability
requires the corresponding context-specific liability variance.

`--binary-sampling-partners` enables participant-sampling covariance, including
a same-study fitted risk model, estimated population moments, and randomized
reference error. `--binary-architecture-probes` additionally models variation
in Gaussian SNP effects. Outputs include component SEs, their full covariance,
normal intervals, and score confidence sets. A score set can be unbounded.
The counts in the example are starting values; assess simulation precision
for the study size and model being analyzed.

These intervals remain experimental. Simulations have shown undercoverage
for some rare-disease inverse-weighted and overlapping-LD models. Supplied
prevalence, genotype scaling, liability SDs, genotype adjustment, and supplied
risks are treated as fixed. Uncertainty from estimating those external inputs
is not included.

With zero sampling partners, inference uses the SNP-block jackknife. It holds
risks, population moments, and reference probes fixed. When both calculations
are available, `snp_block_standard_errors` preserves the jackknife result and
`standard_errors` reports the sampling result.

## Computation and Python API

Study-specific preparation reads genotypes twice. External-LD preparation uses
one study pass and two reference passes, with a second study pass when sampling
covariance is requested. Computation uses streamed variant probes and avoids
participant-by-participant matrices. More contexts and annotations increase
memory and work; `--memory-gib` budgets workspace, so allow additional process
memory when sizing jobs.

All four modes use SUMMIT's native genotype readers and protected matrix
products. Sampled-pair relatedness, covariance preparation, and SNP-effect
trace calculations also run in the compiled extension. Python handles risk
fitting, preparation order, and the small estimating equations.
`--num-threads` controls native work during preparation and fitting.

Reference LD scores use randomized variant probes. Sampling covariance reuses
the genotype passes and accumulates sampled-pair moments. Products shared by
annotation and context combinations are evaluated once, then expanded into
the full coefficient covariance. This is an algebraic simplification and adds
no Monte Carlo approximation. Gaussian SNP-effect covariance uses separate
probe families and four participant groups.
When no projection follows context weighting, the reference calculation also
reuses the equal cross-products for context pairs `(u, v)` and `(v, u)`.

`summit.context.binary` exposes `prepare_gxe_moments`, `prepare_gxe_external`,
`fit_gxe`, `evaluate_contexts`, and `plan_gxe_reference`. File-backed
preparation is available through
`summit.pcgc.gxe_io.prepare_gxe_from_source`. Use
`summit.pcgc.split_io.write_split_artifact` and `load_split_artifact` to save
and reload the separate files. Quantitative G×E summaries and binary contextual
summaries have separate formats. Cross-trait contextual
PCGC is not currently exposed.
