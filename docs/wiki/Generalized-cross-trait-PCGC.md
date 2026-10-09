# Generalized cross-trait PCGC

This workflow estimates context-dependent genetic covariance for continuous–
continuous, binary–continuous and binary–binary trait pairs. Preparation uses
the analyzed cohorts jointly and writes separate reference and trait files.
It supports different context bases, fixed effects, risks and cohort sizes.

Point estimates have independent dense-kernel checks, and uncertainty has
been tested with repeated simulations that vary sample size, SNP count and
block count separately. Inference remains experimental outside the validated
settings below, especially with rare outcomes, fitted nuisance models or
weak genetic-variance denominators.

## Estimand

For trait t, population-scaled, allele-aligned genotypes X and context basis
φ_t define genetic effects by

```math
g_{ti}=\sum_j X_{ij}\phi_{ti}^{T}\beta_{tj},\qquad
\operatorname{Cov}(\beta_{Aj},\beta_{Bj})
=\sum_a A_{ja}\Omega_{AB,a}/M_a,\quad M_a=\sum_j A_{ja}.
```

Each cross-trait Ω is an ordered Q_A × Q_B matrix. It need not be symmetric.
Reversing the traits transposes it. Within-trait Ω matrices are symmetric.
Annotation weights are nonnegative and may overlap; overlapping components
describe a joint model and require distinguishable kernels.

The effect covariance surface is φ_Aᵀ Ω_AB,a φ_B. Interpreting it as a
same-person genetic covariance assumes unit conditional genotype variance
on the declared scale. More generally the centered population covariance is

```math
C_{g,AB}=\sum_a\langle\Omega_{AB,a},H_{AB,a}\rangle,\qquad
H_{AB,a}=\sum_j A_{ja}\operatorname{Cov}_{\mathrm{pop}}
  (X_{ij}\phi_{Ai},X_{ij}\phi_{Bi})/M_a.
```

Each feature mean E_pop[X_j φ_t] must be removed. When these means are zero,
H reduces to the corresponding weighted genotype-squared/context second
moment. An uncentered second moment describes a different quantity when
these means are nonzero.

Supply H_AA and H_BB for the matching genetic variances. Genetic correlation
is C_g,AB / √(V_g,A V_g,B). Heritability instead divides a genetic variance by
the total population phenotype or liability variance. A covariance divided by
total phenotype SDs is a standardized covariance, not a genetic correlation.

## Supported sampling and scales

| Design | Support |
|---|---|
| Outcome-independent cohort membership, no/partial/full overlap | Supported, conditional on fixed exogenous contexts and covariates |
| Binary case–control sampling based solely on that trait’s own status, cohorts disjoint before selection | Supported by the first-order PCGC model |
| Ascertained overlapping trait cohorts | Rejected; needs a supported joint selection model |
| Controls restricted by several diagnoses | Rejected |
| Quantitative sample selected on a diagnosis | Rejected |
| Recognition conditional on measured illness, or treatment-restricted samples | Rejected without a separate target and selection model |

The binary approximation comes from differentiating the selected joint latent
distribution at zero distinct-person correlation. It assumes correctly
specified population risks and normally distributed liabilities conditional
on the declared covariates and contexts. Joint diagnosis selection generally
does not yield a product of marginal PCGC sensitivities.

Binary analyses require an explicit conditional liability scale: either unit
total variance or known positive SDs on a common liability unit. Disease risks
do not estimate a heterogeneous liability variance. Supplied heterogeneous
SDs require supplied population risks. Fitted probit risks currently assume
unit conditional liability variance.

Quantitative outcomes are divided by a supplied, fixed population SD. Use 1
to retain their original units. The program does not estimate or apply a
sample phenotype SD. Quantitative fixed effects include an intercept.

Distinct participants must have no residual covariance after the declared mean
model. Shared-household or other off-person residual dependence requires an
additional model and is not removed by the overlap correction.

Every analyzed person needs finite contexts, phenotype, covariates and risk
inputs. Absent rows define a cohort mask whose selection law must be justified.
They do not establish arbitrary missingness robustness. Choose context
centers, scales and categorical encodings before preparation.

## Prepare matched files

Use the genotype and population-scale formats in
[Binary traits and PCGC](Binary-traits-and-PCGC.md). Each trait table contains
unique string `FID IID`, a `Y` column, and its context and covariate columns.
Binary Y is 0/1. Tables may have different participants and row orders;
overlap is determined by matching both identifiers.

A preparation manifest can contain:

```json
{
  "genotype": "study.bed",
  "population_scale": "population_scale.tsv",
  "annotations": "annotations.tsv",
  "genome_build": "GRCh37",
  "traits": [
    {
      "name": "disease",
      "kind": "binary",
      "table": "disease.tsv",
      "contexts": ["E"],
      "selection": "outcome_independent",
      "risk_column": "RISK",
      "unit_liability": true
    },
    {
      "name": "biomarker",
      "kind": "quantitative",
      "table": "biomarker.tsv",
      "contexts": ["E", "E2"],
      "selection": "outcome_independent",
      "population_sd": 1.0,
      "fixed_effects": ["AGE", "PC1", "PC2"]
    }
  ]
}
```

Paths resolve relative to the manifest. The program adds a context intercept.
For a fitted binary risk model, omit `risk_column` and supply
`risk_covariates`; the context columns also enter that fit. Replace
`unit_liability` with `"liability_sd_column": "SD"` for known heterogeneous
SDs. For `selection: "own_case_status"`, supply population `prevalence`
or known `sampling_odds`, the case/control inclusion-probability ratio.
Outcome-independent sampling fixes that ratio at one.

Optional `genotype_adjustment` contains a `table` of unique FID/IID and
`columns` listing ancestry PCs. It projects population-scaled genotypes
before any trait-specific risk or context multiplication. These columns also
enter fitted binary risk models. Supplied risks must already account for the
appropriate mean effects. Quantitative projection occurs after context
multiplication; binary weighted features receive no extra OLS projection.

```bash
summit pcgc-cross prepare pair.json \
  --nvecs 256 --block-size 256 --probe-tile 32 \
  --memory-gib 8 --num-threads 2 --out results/pair
```

This creates `pair.cross.ldscores.npz` and `pair.cross.sumstats.npz`.
The first contains directional rows and correlated reference-draw summaries.
The second contains signed context scores and corrected pair products.
The input phenotype, context, risk, SD and PC tables contain participant-level
information. Both output archives contain sealed scientific identities and
checksums. Files default to
private permissions. They omit raw participant rows and identifiers, but
signed genetic scores remain sensitive and should be kept private.

Preparation includes every within-trait denominator and, by default, every
cross-trait pair. A manifest `pairs` list, such as
`[["disease", "biomarker"]]`, restricts requested cross-trait pairs. List
each pair in the order of the trait definitions.

## Fit without genotypes

```bash
summit pcgc-cross fit \
  --sumstats results/pair.cross.sumstats.npz \
  --ldscores results/pair.cross.ldscores.npz \
  --njack 200 --confidence 0.95 \
  --memory-gib 8 --num-threads 2 --out results/pair_fit
```

The output is `pair_fit.cross.json`. Pair keys refer to zero-based trait
indices, such as `0:1`. Within-trait and cross-trait coefficients have a
shared covariance matrix. Fits preserve signed estimates, negative variance
components and correlations outside [−1, 1]. Singular or nonpositive normal
equations stop the fit; weaker conditioning is reported diagnostically.

For population summaries, `--population-metrics population.json` accepts
`schema: "summit.pcgc.cross_population_metrics.v1"`, the `reference_identity`
printed by preparation, `target_population`,
`centering: "population_feature_means_removed"`,
`kernel_second_moments` keyed by these pair indices,
and `total_variances` in trait order. Every kernel metric has shape
annotation × left context × right context. Supply moments for all reported
pairs and their denominators, on a common target population. These quantities
are treated as fixed; their estimation uncertainty is not included. The file
must refer to the matched preparation and contain exactly its ordered pairs.
The fit records a checksum of these declared population moments.

The Python API provides `binary_trait`, `quantitative_trait`,
`prepare_cross_gxe`, `prepare_cross_from_source`, `fit_cross_gxe`,
`derived_covariance`, and `evaluate_cross_contexts` in
`summit.pcgc.cross_gxe_model`, `cross_gxe`, `cross_gxe_io`, and
`cross_gxe_fit`. The context evaluator reports an effect covariance surface
and matching genetic-variance denominators. A nonpositive denominator gives
an undefined correlation; a variance interval reaching zero makes the
correlation unreliable. Weak normal equations are reported separately.
The program does not clip a correlation or project the fitted joint effect
covariance to a positive semidefinite matrix.

Arbitrary existing within-trait PCGC files cannot be combined into these
inputs: their products can discard signs, and they lack the joint projected
overlap reference. The [within-trait split-file interface](Generalized-GxE-PCGC.md)
and its legacy combined files retain their existing behavior.

## Reference calculation and overlap

The reference uses shared global variant-axis probes and two complete genotype
passes. Source contributions cover the full retained genome. Genotype reads
and the first-pass genotype sketch are shared across contexts and requested
pairs. Variant and probe tiles limit workspace without making the reference
chromosome-local.

Same-person corrections are exact conditional on the supplied designs.
For quantitative traits, projection can spread residual covariance beyond
the original diagonal. The estimating instruments remove the entire projected
same-person nuisance space. This permits arbitrary same-person residual
cross-covariance under the supported sampling model, including unequal
projectors. High-leverage designs that make this correction unstable are
rejected.

Jackknife blocks are assigned only at fitting. Deleting a target block reuses
the saved rows, renormalizes retained target annotation mass, and keeps full
source mass and all global corrections fixed. The resulting normal equations
remain directed. No genotype refit or reference regeneration occurs.

## Uncertainty and validation limits

The fit reports paired SNP-block covariance and a joint delta approximation
for randomized-reference error. Shared participants and common probes are
retained when combining within-trait and cross-trait coefficients. Correlation
SEs use their joint covariance and both variance denominators.

The paired block procedure can capture response and SNP-effect variation when
blocks are approximately independent and no block dominates. It is not a
general decomposition into participant, effect and nuisance uncertainty.
Adding separate estimates of those pieces without their joint influence
calculation can double count variation. The output reports annotation mass
concentration across blocks and flags use of fewer than 30 blocks.

Intervals use a working Student approximation, with Welch–Satterthwaite
degrees of freedom from the SNP-block and reference-draw variance estimates.
This accounts for estimating variance from finitely many blocks or probes.
It is not an exact pivot under arbitrary LD, unequal blocks or fitted risks.
The reference contribution is a first-order delta approximation; finite-probe
inversion bias and higher-order interactions are not corrected.

External prevalence, genotype scale, supplied liability SDs and population
moments are treated as fixed. General fitted-risk and estimated-moment
influence calculations are not implemented. A correctly specified risk model
can change both point estimation and its sampling covariance; its effect need
not be an additional positive variance term.

Validation distinguishes Monte Carlo uncertainty from persistent deficits.
Independent exact person kernels check all trait-type combinations, partial
overlap, unequal projectors and distinct bases. Bounded variant-Gram oracles
allow larger known-truth simulations without changing the production
reference algorithm. They redraw Gaussian SNP effects and participant
residuals, with the declared liability scale, and compare every component
and the joint correlation denominator.

At N=M=512, eight-block normal intervals covered 89.6–93.3% in 1,000
continuous-pair replicates; Student intervals covered 94.3–96.1%. Increasing
the block count on the same responses removed most of this deficit. These
small-design results are not evidence of large-sample failure. Conversely,
with 32,768 participants, rare binary outcomes and 1,024 SNPs, independent
5,000-replicate confirmation retained 91.6–95.2% normal coverage across
components using 128 blocks. At 4,096 SNPs, coverage was 93.4–95.2%.
At 8,192 SNPs, coverage was 94.1–95.3%.
Sample size alone does not characterize this behavior.

With 8,192 participants, 4,096 SNPs in four-SNP LD groups, two disjoint
annotations and 128 blocks, 5,000-replicate confirmation covered approximately
94–96% across continuous, mixed and binary pairs. These experiments use
supplied risks and exact references; they do not establish calibration for
arbitrary prevalence, effect size, LD or nuisance estimation. With rare
outcomes, 32,768 participants, 4,096 SNPs and larger intercept genetic
variances (0.36 and 0.30), coverage fell to 87.9% for a within-trait component.
Cross-trait components covered 94.2–95.2% in that experiment. Increasing the
block count did not resolve the within-trait deficit. This is a reproducible
uncertainty limitation, and the favorable correlation coverage in that setting
does not validate all covariance-surface contrasts.

Separate retrospective case–control experiments use cohorts disjoint before
selection. At 2,048 people per cohort, 256 SNPs and 128 blocks, 1,000
replicates with correctly specified fitted two-stratum probit risks gave
93.1–96.2% coverage for common disease and 93.2–95.7% for rare disease.
Supplied-risk rare-disease variance components retained deficits. These
results qualify that particular nuisance model, not general fitted-risk
inference. Correlation coverage is reported with the frequencies of undefined
and weak denominators; conditional coverage among successful fits cannot
qualify weak correlations.

The validation drivers report bias and its Monte Carlo SE, empirical and
reported covariance, coverage Wilson intervals, null rejection and denominator
failures. They are
`scripts/pcgc/calibrate_cross_scaling.py`,
`scripts/pcgc/calibrate_cross_ascertainment.py` and
`scripts/pcgc/calibrate_cross_reference.py`.
The bounded timing driver is `scripts/pcgc/benchmark_cross_gxe.py`.
On shared research hosts, launch through
the existing `scripts/pcgc/gxe_python.py` guard with explicit affinity,
matching thread settings and the appropriate native backend. Memory admission
includes resident rows, source sketches, overlap corrections, native workspace
and serialization copies. Fitting separately checks archive loading, checksum
workspace, block reductions and joint covariance output against its memory budget.
Large context bases or many pairs can exceed the
budget even with small tiles; preparation then stops before decoding.

No cross-trait checkpoint/resume interface is currently provided. Before a
full-genome run, measure a bounded-SNP job with the full cohort, required probe
count and actual compute-node runtime. Small-cohort timing cannot establish
throughput or compliance with a queue time limit.

The derivation uses the first-order PCGC approach described by
[Weissbrod, Flint and Rosset (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6035374/).
The quantitative cohort-transport approximations in
[Cross-trait response covariance](Cross-trait-response-covariance.md) describe
a separate workflow.
