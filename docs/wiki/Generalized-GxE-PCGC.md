# Generalized G×E PCGC

PCGC estimates the covariance of additive and context-dependent SNP effects
for a binary trait, accounting for case–control sampling. A context basis
`[1, E1, E2]` gives six parameters per annotation: additive variance, two
interaction variances, and their three covariances. Continuous exposures,
categorical variables encoded numerically, and supplied nonlinear basis
columns use the same interface.

For covariance between two traits, see
[Generalized cross-trait PCGC](Generalized-cross-trait-PCGC.md). That workflow
requires joint preparation and has separate selection and uncertainty limits.

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
  --binary-crossblock-blocks 200 \
  --nvecs 256 --memory-gib 8 --num-threads 2 \
  --out results/disease_gxe
```

`--binary-crossblock-blocks 200` records the participant rows for the
opt-in interval correction described under [Standard errors](#standard-errors);
fit with the default 200 jackknife blocks.

Preparation writes two NumPy archives:

| File | Contents |
|---|---|
| `results/disease_gxe.binary.ldscores.npz` | PCGC directional reference rows, annotations, and any reference uncertainty arrays |
| `results/disease_gxe.binary.sumstats.npz` | Per-SNP context-pair trait score products, population moments, and any sampling or SNP-effect covariance summaries |

Both the reference rows (`ldscores`) and trait score products (`rhs_rows`) have
same-person terms removed. The NPZ format retains their multiple context and
annotation dimensions. The files record variant order, alleles, context names,
and the sample, risk, genotype-scale, and liability-scale definitions.

Because the same-person terms are removed per SNP on the analyzed people, PCGC
never scales a reference same-person matrix by sample size. The own-row
correction used for transferred quantitative fits
([Methods](Methods.md#reference-transfer-and-fitting)) is therefore not needed
here. `pcgc-ld` likewise forms its context factor from the study's own contexts.

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

### Standard errors

Fits report SNP-block jackknife SEs. An opt-in correction adds covariance
that the jackknife cannot see. The jackknife treats SNP blocks as
independent. That holds for SNP effects and LD, but every block's estimating
equation is a sum over the same participants. Several participant-level
terms make block equations covary:

- threshold noise whose variance depends on the genetic value (positive for
  rare diseases, negative near a prevalence of 0.5);
- each participant's own genetic signal, which enters every block;
- selection on case status, which correlates blocks' genetic values within
  cases and within controls;
- risk and context weights shared by all blocks.

Their share grows with the signal in the data (roughly ĥ·N/M). With the
jackknife alone, a simulated rare disease (K = 0.02, N = 32,768, M = 1,024)
had 82% coverage for a null additive-by-context covariance. Near K = 0.5 the
jackknife was slightly conservative for that component.

With `--binary-crossblock-blocks`, preparation records each participant's
contribution to each SNP block's estimating equation, in the first genotype
pass. The fit adds the covariance between different blocks' contributions
from the same participants. The within-block part is already in the
jackknife and is not added again.
[The development report](../../reports/pcgc_development/pcgc_gxe_binary_intervals.md)
gives the derivation, assumptions and calibration results.

In simulations with 22 designs (prevalence 0.02–0.5, Q = 1–3, N/M = 8–128,
fitted and supplied risks, case–control ascertainment, LD and overlapping
annotations), the corrected variance was close to the true variance (ratio
0.89–1.14, mean 1.00) and null components had 4–7% type-I error, against
0.44–1.26 and up to 19% for the jackknife. Coverage of 95% intervals for
variance components (diagonal Ω entries) was still 90.8–94% in some designs:
the SE estimate is strongly correlated with the estimate, which affects any
normal interval, including the jackknife's where its variance is right. The
correction is therefore opt-in and labelled
`variance_calibrated_in_simulation_wald_coverage_can_be_below_nominal`. It is
recommended for rare diseases with supplied risks and large ĥ·N/M, where
the jackknife alone is most anti-conservative. Like the jackknife, it
requires blocks larger than local LD.

The correction needs the same SNP blocks at preparation and fit. Set
`--binary-crossblock-blocks` to the intended jackknife block count, usually
the binary default of 200. Without `--njack`, the fit uses the prepared
blocks. With a different `--njack`, the fit reports the jackknife alone and
says so. In the Python API, pass `crossblock_blocks` (and `risk_covariates`
for fitted risks) to `prepare_gxe_moments` and fit with the same `block_ids`.

With fitted risks (`--binary-covariates` or the context columns), the
correction includes uncertainty from the probit risk fit. Supplied risks,
prevalence, genotype scaling, liability SDs, genotype adjustment and
reference probes are treated as fixed, as is the population metric used for
`population_heritability`.

Output fields:

| Field | Meaning |
|---|---|
| `standard_errors`, `covariance` | Primary result: calibrated when `crossblock_status` is `applied` |
| `snp_block_standard_errors`, `snp_block_covariance` | SNP-block jackknife alone |
| `person_crossblock_covariance`, `risk_fit_covariance` | The two added terms |
| `uncertainty_method` | `snp_block_jackknife_plus_person_crossblock_v1` or the jackknife |
| `uncertainty_calibration_status` | Calibration status of the primary result |
| `crossblock_status` | `applied`, or why the correction was not applied |

A jackknife-only result is labelled
`not_calibrated_snp_block_jackknife_omits_between_block_person_covariance`.
In very small samples the between-block estimate can be noisy enough to give
a nonpositive variance; the fit then keeps the jackknife and reports
`not_applied_nonpositive_calibrated_variance`.

### Experimental sampling SEs

`--binary-sampling-partners` enables participant-sampling covariance, including
a same-study fitted risk model, estimated population moments, and randomized
reference error. `--binary-architecture-probes` additionally models variation
in Gaussian SNP effects. Outputs include component SEs, their full covariance,
normal intervals, and score confidence sets. A score set can be unbounded.
Counts such as 128 partners and 32 probes are starting values; assess
simulation precision for the study size and model being analyzed.

These SEs remain experimental. Without architecture probes they target a
fixed genome and omit SNP-effect variation, so they are much too small when
the estimand is the effect covariance Ω (variance ratios 0.10–0.72 for the
additive component in the 2026-10-13 simulations). With architecture probes
their variance ratios were 0.94–1.03 at N = 8,192, with coverage similar to
the cross-block correction, but preparation is far slower (about five
minutes single-threaded with 64 partners at N = 32,768 in our runs, against
seconds for the jackknife and correction). Supplied prevalence, genotype
scaling, liability SDs, genotype adjustment, and supplied risks are treated
as fixed.

When the cross-block correction is applied, the sampling result is reported
separately under `individual_sampling`. Otherwise, as before,
`snp_block_standard_errors` keeps the jackknife and `standard_errors` reports
the sampling result.

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

Cross-block participant rows add three small products per genotype block to
the first pass and O(N) memory; they need no participant pairs or extra pass.
Reference LD scores use randomized variant probes. Sampling covariance reuses
the genotype passes and accumulates sampled-pair moments. Products shared by
annotation and context combinations are evaluated once, then expanded into
the full coefficient covariance. This is an algebraic simplification and adds
no Monte Carlo approximation. Gaussian SNP-effect covariance uses separate
probe families and four participant groups.
When no projection follows context weighting, the reference calculation also
reuses the equal cross-products for context pairs `(u, v)` and `(v, u)`.
Its diagonal terms share genotype-square sums across context pairs.
For disjoint annotation bins, source sketches use only the variants in each
bin, preserving their global probe identities.
Annotation and context products are batched within the workspace budget.
Batching and block size do not change the probe identities or estimating equations.

`summit.context.binary` exposes `prepare_gxe_moments`, `prepare_gxe_external`,
`fit_gxe`, `evaluate_contexts`, and `plan_gxe_reference`. File-backed
preparation is available through
`summit.pcgc.gxe_io.prepare_gxe_from_source`. Use
`summit.pcgc.split_io.write_split_artifact` and `load_split_artifact` to save
and reload the separate files. Quantitative G×E summaries and binary contextual
summaries have separate formats. Cross-trait contextual
PCGC is not currently exposed.
