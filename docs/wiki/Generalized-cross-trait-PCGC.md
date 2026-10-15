# Generalized cross-trait PCGC

This workflow estimates context-dependent genetic covariance for continuous–
continuous, binary–continuous and binary–binary trait pairs. Preparation uses
the analyzed cohorts jointly and writes separate reference and trait files.
It supports different context bases, fixed effects, risks and cohort sizes.

Point estimates have independent dense-kernel checks, and uncertainty has
been tested with repeated simulations that vary sample size, SNP count and
block count separately. Inference remains experimental outside the validated
settings below, especially with rare outcomes, fitted nuisance models or
weak genetic-variance denominators. Within-trait binary intervals are
anti-conservative for rare outcomes or strong per-SNP signal; the fit marks
them uncalibrated (see [Uncertainty](#uncertainty-and-validation-limits)).
Validation used at most 8,192 SNPs and 32,768 participants per trait; the
genome-wide array regime has not been simulated.

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
State whether a heritability denominator includes variation in fixed mean
effects. Unit conditional liability variance does not imply unit total
population liability variance when those means vary.

## Supported sampling and scales

| Design | Support |
|---|---|
| Outcome-independent cohort membership, no/partial/full overlap | Supported, conditional on fixed exogenous contexts and covariates |
| Binary case–control sampling based solely on that trait’s own status, cohorts disjoint before selection | Supported by the first-order PCGC model, with fitted risks whose design includes the contexts |
| Ascertained overlapping trait cohorts | Unsupported; rejected only when declared (an `own_case_status` trait overlapping another cohort) |
| Controls restricted by several diagnoses | Unsupported, not detected |
| Quantitative sample selected on a diagnosis | Unsupported; rejected only when declared (`own_case_status` on a quantitative trait) |
| Recognition conditional on measured illness, or treatment-restricted samples | Unsupported without a separate target and selection model; not detected |

The program checks the declared `selection` labels and cohort overlap only.
It cannot recognize an undeclared ascertainment, a restricted control group
or treatment-based exclusions in the supplied tables, so a design listed as
unsupported can run without error when it is not declared. The fit records
this as `selection_law_declared_not_verified`; the analyst attests to the
sampling design.

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

A fitted probit risk design always contains every context column; columns
not already spanned by the intercept and the risk covariates are added. At
the fit, the score equations make the risk-weighted contexts orthogonal to
the binary responses. Case–control (`own_case_status`) sampling requires
these fitted risks. With supplied risks, even the true population risks,
null within-trait context covariances were rejected too often in
retrospective case–control calibration: 8.1–9.2% of rare-disease
replicates in the original experiment and 8.7% when repeated, against
4.3–4.8% with fitted risks (see the results below). The excess behaves like
sample-level variation shared by all SNP blocks, which the block jackknife
cannot see and the fitted risk model absorbs. Recentring the supplied-risk
responses on the risk-weighted contexts removed only part of it, so it is
not a substitute for the fit. Supplied risks in that design are therefore
refused unless `allow_uncalibrated_supplied_risks` is set, and the fit then
marks that trait's within-trait intervals uncalibrated. This also applies
to heterogeneous liability SDs, which need supplied risks. Under
outcome-independent sampling, supplied risks remain supported and should
come from a model that includes the contexts.

Quantitative outcomes are divided by a supplied, fixed population SD. Use 1
to retain their original units. The program does not estimate or apply a
sample phenotype SD. Quantitative fixed effects include an intercept, and by
default the context columns also enter the projected design as main
effects. An omitted context main effect remains in the residual and is
correlated with the context-weighted kernels. Set `context_main_effects` to
false only when the contexts are known to have no mean effect.

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
`risk_covariates`, or neither; the context columns always enter that fit.
Replace `unit_liability` with `"liability_sd_column": "SD"` for known
heterogeneous SDs. For `selection: "own_case_status"`, supply either the
population `prevalence` or the known `sampling_odds`, the case/control
inclusion-probability ratio, and use fitted risks; a `risk_column` there also
needs `"allow_uncalibrated_supplied_risks": true`. Outcome-independent
sampling fixes the ratio at one and rejects a `prevalence`, which it would
not use. A quantitative trait accepts `"context_main_effects": false` to
leave its context columns out of the fixed design.

Optional `genotype_adjustment` contains a `table` of unique FID/IID and
`columns` listing ancestry PCs. It projects population-scaled genotypes
before any trait-specific risk or context multiplication. These columns also
enter fitted binary risk models. Supplied risks must already account for the
appropriate mean effects. Quantitative projection occurs after context
multiplication; binary weighted features receive no extra OLS projection.

```bash
summit pcgc-cross prepare pair.json \
  --nvecs 256 --block-size 256 --probe-tile 32 --context-tile 4 \
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
must use the same trait scales as preparation. The population quantities
are treated as fixed; their estimation uncertainty is not included. The file
must refer to the matched preparation and contain exactly its ordered pairs.
The fit records a checksum of these declared population moments.

The Python API provides `binary_trait`, `quantitative_trait`,
`prepare_cross_gxe`, `prepare_cross_from_source`, `fit_cross_gxe`,
`derived_covariance`, and `evaluate_cross_contexts` in
`summit.pcgc.cross_gxe_model`, `cross_gxe`, `cross_gxe_io`, and
`cross_gxe_fit`. `quantitative_trait(fixed_effects=None)` projects an
intercept; pass an n × 0 array for no fixed design. Its
`context_main_effects` and `binary_trait`'s
`allow_uncalibrated_supplied_risks` match the manifest fields.
The context evaluator reports an effect covariance surface
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
pairs. Variant, probe and context tiles limit workspace without making the
reference chromosome-local. Projection coefficients are retained on the
fixed-effect axis, so each context does not need a separate participant-sized
source sketch. The ordered projection corrections remain distinct when
context products share a calculation.

Same-person corrections are exact conditional on the supplied designs.
For quantitative traits, projection can spread residual covariance beyond
the original diagonal. The estimating instruments remove the entire projected
same-person nuisance space. This permits arbitrary same-person residual
cross-covariance under the supported sampling model, including unequal
projectors. High-leverage designs that make this correction unstable are
rejected before genotype reading. The same-person diagonal 1 − h_A − h_B,
for combined leverage h_A + h_B, must exceed 10⁻⁴ times its largest value,
and the condition number of the overlap system must stay below 10⁸.

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

The delete-block replicates keep the realized reference rows, so their spread
already contains part of the randomized-reference error, which the reference
term then adds in full. This double counting is conservative; in the
independent review's checks it was 0.1–0.2% of the reported variance.

### Calibration flags

Within-trait binary intervals are anti-conservative when the outcome is rare
or the per-SNP signal is strong. The cause is threshold heteroscedasticity.
For a binary trait, the variance of the response given the genetic liability
changes with that liability, by approximately δ = (η² − 1)/2 at
η = Φ⁻¹(K). This correlates the signal-by-noise parts of the estimating
function across SNP blocks, Cov(u_b, u_b′) ∝ 2δ h_b h_b′. The delete-block
jackknife assumes independent blocks and omits this covariance, which is
positive, making intervals too narrow, when K < Φ(−1) ≈ 0.16, and grows with
h·N/M. Drift in the realized liability variance does not explain the
deficit: once that drift was removed, the predicted variance ratio (0.47)
matched the observed one (0.49). In a quantitative control, genotype-dependent
residual variance alone lowered the variance ratio of a null within-trait
context covariance from 0.81 to 0.60.

An independent reproduction used two binary traits with K = 0.02,
N = 32,768, M = 1,024 and intercept genetic variance 0.36. The reported
variance was 0.32 times the empirical variance for the intercept component
(72.8% coverage). A null within-trait context covariance had variance ratio
0.44 and 80.3% coverage, so it was rejected in about 20% of replicates at a
nominal 5% level. Cross-trait components were affected only mildly, with
variance ratios 0.87–0.95. The single-trait
[Generalized G×E PCGC](Generalized-GxE-PCGC.md) fit uses the same
within-trait estimator and SNP-block intervals.

The fit output therefore includes `calibration` and `component_calibration`.
Within-trait binary components are marked `uncalibrated` when the trait's
reference prevalence is below Φ(−1) or ĥ·N/M ≥ 1. Here ĥ is the sample
liability genetic variance implied by the fitted Ω and the mean context
second moment, N is that trait's sample size and M the number of SNPs. With
LD, M overstates the number of independent SNPs, so this screen is
optimistic. Supplied risks kept under case–control sampling are also marked.
Every interval reports the flag of its contrast. Genetic variances,
heritabilities and correlations that use a flagged component inherit the
flag; cross-trait covariances do not. `not_flagged` means only that no known
deficit applies, not that calibration is established. No variance correction
is applied. `fit_cross_gxe(participant_covariance=...)` accepts a
participant-level covariance for within-trait binary pairs as an additional
known term, reserved for a calibrated model of the omitted between-block
covariance; none is computed here. Single-trait binary
[Generalized G×E PCGC](Generalized-GxE-PCGC.md) fits apply a correction for this
between-block participant covariance by default (`--binary-crossblock-blocks`); it is not yet
wired into cross-trait fits.

Intervals use a working Student approximation, with Welch–Satterthwaite
degrees of freedom from the SNP-block and reference-draw variance estimates.
This accounts for estimating variance from finitely many blocks or probes.
It is not an exact pivot under arbitrary LD, unequal blocks or fitted risks.
The reference contribution is a first-order delta approximation; finite-probe
inversion bias and higher-order interactions are not corrected. The fit also
reports reference perturbation after multiplication by the inverse normal
matrix and a second-order inversion-bias estimate on each pair's coefficient
axis. These are diagnostic approximations. Large inverse perturbation calls
for more probes and a stability check; a small raw reference error alone does
not establish accurate inversion.

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
block count did not resolve the within-trait deficit, as expected from the
between-block covariance described under [Calibration flags](#calibration-flags),
and the favorable correlation coverage in that setting does not validate all
covariance-surface contrasts.

Follow-up held the larger genetic variance and 8,192 SNPs fixed. With 8,192
participants, coverage was 93.4–95.6%; with 32,768 participants it was
91.1–95.1%. In the latter experiment, cross-trait components covered
94.2–95.1%, while some within-trait variances remained underestimated.
Increasing the sample size raises h·N/M and so the omitted covariance.
These results concern the tested LD and genetic architecture; they do not
establish a universal sample-size threshold.

Further 5,000-replicate tests used shared contexts, unequal disease risks,
strong genetic covariance and same-person residual correlation 0.8.
Coefficient coverage remained approximately 94–96%, but rare-outcome
correlations often had weak or undefined denominators. Favorable coverage
among the defined correlations does not qualify those weak estimates.
An eight-annotation, zero-cross-covariance experiment gave 4.54–6.28% null
rejection across the ordered cross-trait coefficients. For a biomarker and
its exact binary threshold, component coverage was 93.78–95.44%; correlation
coverage at the boundary value one was conservative (96.9%). Unconstrained
correlations above one were retained. These examples use an unrestricted
cohort; they do not validate analysis conditional on having measured illness.

With eight disjoint annotations, 32,768 participants and 8,192 SNPs, the
strong-correlation binary experiment covered 93.98–95.38% across components
and 95.1% for correlation, with no weak or undefined denominators.
Allowing different genetic covariance matrices across eight annotations at
N=32,768, M=4,096 gave 93.62–95.32% component coverage and 94.48% correlation
coverage in 5,000 further replicates. Reported component variances were
0.94–1.04 times empirical variances. A heterogeneous, overlapping-annotation
mixed pair covered 94.36–95.20% across components. Neither experiment had
weak or undefined correlations.

Independent randomized-reference confirmation redrew both responses and
probes in 5,000 binary-pair replicates at N=2,048, M=512. Reported reference
variances were 0.94–1.03 times the paired empirical errors at 32 probes and
0.97–1.03 at 128 probes. The largest discrepancy across the joint reference
covariance matrix was 2.85 Monte Carlo SEs. This checks correlated reference
draws in that model; it does not remove finite-probe inversion bias.

All of these experiments used at most 8,192 SNPs and 32,768 participants per
trait, with independent SNPs or short LD groups. The genome-wide array
regime, about 448,000 SNPs with long-range LD, has not been simulated. There
the effective number of independent SNPs is much smaller than M, which raises
the effective h·N/M; neither the interval approximations nor the calibration
screen have been tested in that regime.

Separate retrospective case–control experiments use cohorts disjoint before
selection. At 2,048 people per cohort, 256 SNPs and 128 blocks, 1,000
replicates with correctly specified fitted two-stratum probit risks gave
93.1–96.2% coverage for common disease and 93.2–95.7% for rare disease.
Supplied-risk rare-disease variance components retained deficits. These
results qualify that particular nuisance model, not general fitted-risk
inference. Correlation coverage is reported with the frequencies of undefined
and weak denominators; conditional coverage among successful fits cannot
qualify weak correlations.

The same design was repeated after fitted risks with contexts became the
required default, with 3,000 rare-disease (population prevalences 0.020 and
0.026) and 2,000 common-disease (0.16 and 0.18) replicates. The two null within-trait context
covariances were rejected in 4.70% and 4.97% of rare-disease replicates
(pooled 4.83%, Monte Carlo SE 0.28%) and in 4.25% and 4.75% of
common-disease replicates (pooled 4.50%, SE 0.33%). Within-trait coverage
was 94.3–95.3% for rare and 93.8–95.8% for common disease; cross-trait
coverage was 94.1–95.4%. Supplied true risks, kept with the override, gave
9.00% and 8.47% for rare disease (pooled 8.73%, SE 0.36%), with within-trait
coverage down to 90.7% and variance ratios down to 0.80, and 5.27% (SE 0.35%)
for common disease. Responses recentred on the risk-weighted contexts gave
7.37% (SE 0.34%) for rare disease. With fitted risks, the largest component
bias was 1% of the rare-disease intercept variance (3.1 Monte Carlo SEs).
The calibration screen flags these rare-disease within-trait components
(prevalence below Φ(−1), ĥ·N/M about 1.0–1.2). The screen follows the
unascertained derivation; in these balanced case–control samples the
fitted-risk intervals were nonetheless calibrated, so it is conservative
there. The driver is `scripts/pcgc/calibrate_cross_ascertainment.py --variant
--n 2048 --m 256 --blocks 128`, with `--risk-modes` and `--settings`
selecting the arms.

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

On a Hoffman compute node with eight physical cores, 337,845 participants,
eight annotations, one binary and one quantitative trait with two contexts
each and 256 probes, private
BLIS preparation plus input, serialization and fitting took 165.2 seconds
for 1,024 SNPs and 320.5 seconds for 2,048 SNPs. Both used genotype chunks of
1,024 SNPs, ten genotype PCs, two quantitative fixed-effect columns and two
genotype passes. The maximum observed RSS was 42.9 GiB;
the full 448,562-SNP memory plan reserved 73.6 GiB within an 88 GiB program
budget and a 96 GiB allocation. Portable OpenBLAS took 163.7 seconds for the
1,024-SNP case. These timings use synthetic outcomes.

The measured private-BLIS slope projects approximately 18.9 hours for that
448,562-SNP configuration, below the 24-hour limit. This is a full-cohort,
bounded-SNP forecast, not an observed genome-wide completion. Larger bases,
additional pairs, different projections or a different runtime need their
own measurement. Genotype decoding was a small fraction of elapsed time;
reference products, feature construction and exact overlap corrections
dominated. Timings for protected matrix products include operand preparation
and integrity checks, and overlap the broader phase timers.
The native vendor timers separately measure CBLAS calls. A matched 1,024-SNP
run spent 17.2 seconds in OpenBLAS calls or 16.5 seconds in private BLIS calls,
within approximately 150 seconds total. Those calls used nearly all eight
cores; the whole preparation averaged about three active cores. Data movement,
projection and exact correction work therefore matter more than vendor GEMM
speed alone in this configuration.

No cross-trait checkpoint/resume interface is currently provided. Before a
full-genome run, measure a bounded-SNP job with the full cohort, required probe
count and actual compute-node runtime. Small-cohort timing cannot establish
throughput or compliance with a queue time limit.

The derivation uses the first-order PCGC approach described by
[Weissbrod, Flint and Rosset (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6035374/).
The quantitative cohort-transport approximations in
[Cross-trait response covariance](Cross-trait-response-covariance.md) describe
a separate workflow.
