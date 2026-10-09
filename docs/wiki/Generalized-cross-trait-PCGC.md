# Generalized cross-trait PCGC

This workflow estimates context-dependent genetic covariance for continuous–
continuous, binary–continuous and binary–binary trait pairs. Preparation uses
the analyzed cohorts jointly and writes separate reference and trait files.
It supports different context bases, fixed effects, risks and cohort sizes.

The method is experimental. Point estimates have independent dense-kernel
checks. The available intervals are conditional approximations and have shown
undercoverage in simulations; they should not be described as calibrated.

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
  --njack 100 --memory-gib 8 --num-threads 2 --out results/pair_fit
```

The output is `pair_fit.cross.json`. Pair keys refer to zero-based trait
indices, such as `0:1`. Within-trait and cross-trait coefficients have a
shared covariance matrix. Fits preserve signed estimates, negative variance
components and correlations outside [−1, 1]. Singular or nonpositive normal
equations stop the fit; weaker conditioning is reported diagnostically.

For population summaries, `--population-metrics population.json` accepts
`target_population`, `centering: "population_feature_means_removed"`,
`kernel_second_moments` keyed by these pair indices,
and `total_variances` in trait order. Every kernel metric has shape
annotation × left context × right context. Supply moments for all reported
pairs and their denominators, on a common target population. These quantities
are treated as fixed; their estimation uncertainty is not included.

The Python API provides `binary_trait`, `quantitative_trait`,
`prepare_cross_gxe`, `prepare_cross_from_source`, `fit_cross_gxe`,
`derived_covariance`, and `evaluate_cross_contexts` in
`summit.pcgc.cross_gxe_model`, `cross_gxe`, `cross_gxe_io`, and
`cross_gxe_fit`. The context evaluator reports an effect covariance surface
and matching genetic-variance denominators. A nonpositive denominator gives
an undefined correlation; a weak denominator is marked unreliable.

Arbitrary existing within-trait PCGC files cannot be combined into these
inputs: their products can discard signs, and they lack the joint projected
overlap reference. The [within-trait split-file interface](Generalized-GxE-PCGC.md)
and its legacy combined files retain their existing behavior.

## Reference calculation and overlap

The reference uses shared global variant-axis probes and two complete genotype
passes. Source contributions cover the full retained genome. Genotype reads
and the first-pass genotype sketch are shared across contexts and requested
pairs. Context and probe tiles limit workspace without making the reference
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

These calculations omit participant-sampling covariance, selected binary
fourth moments, Gaussian SNP-effect uncertainty beyond what the target-block
procedure captures, fitted-risk influence and estimated population-moment
influence. External prevalence, genotype scale and supplied SDs are fixed.
Finite-probe inversion bias is not removed.

In a fixed-design validation with 500 replicates per setting, N=128 per trait,
64 SNPs and eight target blocks, nominal 95% component intervals covered
86.2–93.2% across the principal continuous/binary and overlap settings.
A rare-disease setting with heterogeneous supplied SDs reached 76.2% for
one component. These are small-design conditional experiments with exact
references, and do not establish accuracy for ascertained or full-cohort
inference. The scripts report bias MCSEs, empirical SDs, SE ratios, coverage
Wilson intervals and null rejection rates. Passing numerical checks does
not resolve this undercoverage. A follow-up at N=512 reduced undefined
correlations but retained component undercoverage. Rare-disease fitted-risk
simulations also showed finite-sample bias and separated risk fits. Fitting
risks does not supply their missing influence contribution to uncertainty.

The bounded benchmark and calibration drivers are
`scripts/pcgc/benchmark_cross_gxe.py` and
`scripts/pcgc/calibrate_cross_gxe.py`. The separate
`scripts/pcgc/calibrate_cross_ascertainment.py` checks retrospective
case–control point estimation. On shared research hosts, launch through
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
