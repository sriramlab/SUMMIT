# PCGC implementation and qualification

Date: 2026-09-26. Branch: `feat/pcgc`, based on `working` at
`af6430efdfb3995b2905a878e35114148bc0863d`.

**Historical qualification at `4a96bb1`.** The PCGC jackknife below used an
incorrect deletion scaling. It is superseded by the
[jackknife audit and local-data checks](pcgc_jackknife_audit.md). The original
results are retained for paired comparison; they do not qualify v2 uncertainty.
The [release validation](pcgc_release_validation.md) evaluates the corrected
equations across methods, annotation components, and binary cross-trait fits.

Standard PCGC point estimates passed the predeclared bias-equivalence screens
in the five independently confirmed univariate scenarios. Its frozen-target
jackknife did not pass the calibration screens with strong covariates.
Accordingly, the ordinary binary interface provides point estimates;
uncertainty, inverse weighting and external-LD transfer require an explicit
research option. This is a bounded scientific qualification, not a claim of
validity for arbitrary ascertainment, ancestry structure or reference panels.

## What is implemented

The [scientific and input contract](pcgc.md) defines all five methods. Risk
preparation fits an ascertainment-aware population-probit likelihood or accepts
supplied population risks. Standard, inverse and scalar methods share the
same moment representation, solver and block reduction. Exact basis
contraction reuses the generalized contextual feature representation.

Production preparation reuses SUMMIT's BED/PGEN readers, population affine
scales, descriptor checks, fixed variant probes, protected matrix products,
memory planner and two-pass reference engine. Scores and exact same-person
terms are collected during those passes. There is no additional genotype
decoder or change to the native numerical kernels. Dense exact references
exist only in the small-data research oracle.

Typed joint artifacts bind scores to their variant/allele axes, annotation
names, sample and scale identities, prevalence, risks and reference contract.
Ordinary beta/SE files cannot enter this path implicitly. Quantitative HE and
generalized GxE retain their existing equations and defaults.

| Approach | Point-estimate interface | Qualification boundary |
|---|---|---|
| `liability` | Available | Constant-risk model; agrees with standard/inverse PCGC algebraically |
| `pcgc` | Available | Exogenous risk covariates, population genotype scale, case-status sampling; confirmation below |
| `pcgc-basis` | Available for exact spans | Exact fixed-probe identity; an approximate span is a separate research approximation |
| `pcgc-inverse` | `--binary-research` | Strong Gaussian risk covariates can cause infinite variance; S4 did not establish bias equivalence |
| `pcgc-ld` | `--binary-research` | Reference matching and risk/genotype factorization remain assumptions; not an S-PCGC format adapter |
| Any binary standard errors | `--binary-research` | Frozen target-row jackknife passed some settings and failed others |
| Binary–binary / binary–quantitative | Research Python API | Explicit marginal sampling/overlap contract; no public binary `--rg` or pair artifact |

There is no automatic selection of the apparently best method from an input
dataset. Unsupported assumptions cannot generally be diagnosed from the
typed summaries alone.

## Validation design and counts

The [approved plan](pcgc_implementation_plan.md) specified the screens before
confirmation. Numerical checks preceded statistical runs. Methods were paired
on identical simulated datasets; supplied and estimated risks were evaluated
separately. Fits were not clipped or silently regularized.

Gaussian scenarios use 4,000 study people, 4,000 markers, 100 independent
40-marker AR blocks alternating correlation 0.2/0.7, an independent population
reference of 2,000 people, and 50 frozen target-row blocks. Covariates are
independent of population genotypes except for the explicitly invalid S6
sentinel. Conditional liability variance is one, with genetic variance 0.25
unless specified otherwise.

| Scenario | Population K / sample P | Risk covariate | Pilot datasets analyzed | Independent confirmation |
|---|---|---|---:|---:|
| S0 | 0.10 / 0.10 | None | 19 | 80 |
| S1 | 0.10 / 0.50 | None | 19 | — |
| S2 | 0.10 / 0.50 | Binary ±1, effect 0.5 | 19 | 80 |
| S3 | 0.10 / 0.50 | Gaussian, effect 0.5 | 19 | 80 |
| S4 | 0.01 / 0.50 | Gaussian, effect 1 | 19 | 80 |
| S5 | 0.10 / 0.50 | Binary ±1, effect 0.5; components 0.05/0.20 | 19 | — |
| S6 | 0.10 / 0.50 | Deliberately mismatched genotype–covariate kernel | 19 | Withheld |
| S7 | 0.10 / 0.50 | Gaussian, effect 1; genetic variance 0 | 19 | 80 |

The Gaussian S2 timing benchmark adds one generated dataset, so its total is
100; the other expanded Gaussian scenarios total 99. That benchmark is not
included in the main 19-dataset paired pilot comparison. Two independent
discrete-haplotype scenarios, S2 and S5, used 20 datasets each. Each of the
three bivariate designs used 20 pilots and 80 independent confirmation
datasets. Six existing pilot seeds per selected univariate scenario were
reused for risk/reference sensitivity diagnostics. Re-evaluating an estimator
on the same data does not add a replicate. No statistical setting exceeds
100 distinct datasets.

The efficient Gaussian conditional sampler was checked against independent
population rejection. The discrete generator has diploid Markov haplotypes
with allele frequency 0.3, independently checked LD, and a prevalence
threshold obtained by inversion of its liability characteristic function.
The latter calculation was checked against Gaussian and exact single-SNP
mixtures, then independent population draws.

Total-h² bias equivalence requires its 95% Monte Carlo interval to lie inside
`±max(0.02, 0.10*truth)`. Component/covariance margins are
`±max(0.01, 0.10*abs(truth))`; the rg margin is ±0.05. The coarse uncertainty
screen requires RMS(SE)/empirical SD in [0.8, 1.25], a Wilson coverage interval
containing 0.95 with lower bound above 0.85, and, at a null, a rejection
interval containing 0.05 with upper bound below 0.15. Failed or undefined
intervals count against coverage. These modest-run screens cannot establish
fine calibration.

## Univariate confirmation

The table reports standard PCGC with estimated risks on the marginal scale;
each row contains 80 independent confirmation datasets. All five passed the
total-h² bias-equivalence criterion. Known-risk versions also passed that
point-estimate criterion.

| Scenario | Truth | Mean estimate | 95% interval for bias | RMS(SE)/SD | Coverage | Combined screen |
|---|---:|---:|---|---:|---:|---|
| S0 | 0.250 | 0.2498 | [−0.0128, 0.0123] | 1.012 | 79/80 | Pass |
| S2 | 0.200 | 0.2003 | [−0.0046, 0.0052] | 1.224 | 79/80 | Pass |
| S3 | 0.200 | 0.1998 | [−0.0059, 0.0054] | 1.043 | 76/80 | Pass |
| S4 | 0.125 | 0.1249 | [−0.0045, 0.0043] | 0.898 | 74/80 | Uncertainty unqualified |
| S7 | 0 | 0.0006 | [−0.0030, 0.0041] | 0.832 | 73/80 | Uncertainty unqualified |

S4's coverage interval is [0.846, 0.965]; S7's is [0.830, 0.957]. Both include
nominal coverage but fail the predeclared lower-bound requirement. They are
not promoted by rounding coverage or relaxing the threshold. With supplied
risks, S4 coverage was 72/80, so the limitation cannot be attributed solely
to fitting nuisance risks. No ad hoc SE multiplier was fitted to these draws.

Inverse weighting agreed in the no-covariate setting and performed adequately
with binary risk strata. With fitted continuous risks, S3 coverage was 74/80.
In S4 its mean was 0.1087, bias interval [−0.0345, 0.0020], empirical SD 0.0819
and coverage 68/80. The bias interval does not establish equivalence; it is
not evidence for a precise nonzero bias either. Standard PCGC's SD there was
0.0200. The separate infinite-second-moment counterexample in the contract
explains why general inverse-weighting qualification would be inappropriate.

Four-bin approximate basis fits tracked standard PCGC in the tested
scenarios, but S4 coverage was 72/80 and S7 73/80 with fitted risks. The exact
basis interface is justified by an algebraic identity and fixed-probe tests;
these four-bin results do not authorize arbitrary approximations.

External-LD point estimates passed the five confirmation bias screens after
the distinct-person risk-factor correction. Its fitted-risk S4 coverage was
75/80, while the supplied-risk version and the strong-covariate null remained
unqualified. Matching these simulations does not validate transfer to an
arbitrary reference.

The unchanged SUMMIT HE fit followed by global liability conversion was also
competitive in these exogenous-covariate simulations. Its marginal means were
0.2506, 0.2051, 0.2002, 0.1263 and −0.0012 for S0/S2/S3/S4/S7. These experiments
do not establish universal superiority of PCGC over a correctly interpreted
global conversion. The extension supplies explicit covariate-risk moments and
their statistical contract.

The two-component Gaussian pilot passed the component point-bias screens;
its uncertainty was not confirmed. Discrete S2 and S5 total point estimates
were compatible with the margins. The smaller discrete S5 component was
borderline (bias interval approximately [−0.0079, 0.0101] against ±0.01).
Twenty replicates do not qualify component/enrichment uncertainty.

## Bivariate confirmation

The three designs each used 4,000 people per cohort, 4,000 markers, genetic
variance 0.25 for each trait and binary risk covariates with effects 0.5.
Binary cohorts had K=0.1 and P=0.5. BB0 used disjoint cohorts and true rg=0;
BB_shared used rg=0.5 and 1,000 shared controls with compensating private
sampling; BQ used a disjoint population quantitative cohort and rg=0.5.
The quantitative residual had known population variance one.

Each row below uses standard PCGC with fitted binary risks and 80 independent
confirmation datasets. All covariance and rg screens passed for both
standard and inverse PCGC, with supplied or fitted risks. There were no
undefined point ratios or nonfinite ratio deletions in these confirmations.

| Design / quantity | Truth | Mean | 95% interval for bias | RMS(SE)/SD | Coverage |
|---|---:|---:|---|---:|---:|
| BB0 covariance | 0 | −0.0014 | [−0.0063, 0.0035] | 1.091 | 76/80 |
| BB0 rg | 0 | −0.0070 | [−0.0270, 0.0130] | 1.083 | 76/80 |
| BB_shared covariance | 0.125 | 0.1271 | [−0.0029, 0.0070] | 1.132 | 78/80 |
| BB_shared rg | 0.500 | 0.5027 | [−0.0134, 0.0188] | 1.111 | 77/80 |
| BQ covariance | 0.125 | 0.1209 | [−0.0085, 0.0003] | 1.134 | 77/80 |
| BQ rg | 0.500 | 0.4878 | [−0.0281, 0.0038] | 1.178 | 78/80 |

BB0 rejected the null in 4/80 datasets; its Wilson interval was
[0.0196, 0.1216]. The low BQ covariance and conservative BB0 uncertainty seen
in the 20-dataset pilots did not prevent the independent confirmations from
meeting the frozen criteria. Pilot and confirmation results were not pooled
to make those decisions.

These results support the explicitly declared research sampling design.
Shared controls drawn only by joint eligibility, with uncompensated private
controls, need a different derivation. A general binary `--rg` interface and
its cross-study artifact/alignment contract remain unexposed; the tested
extension is a Python API on a common master genotype axis.

## Risk and reference diagnostics

These are six paired pilot datasets per scenario, not additional confirmation.
For S2/S3/S4, replacing the exact study reference with an independent weighted
reference from the same sampling design changed marginal estimates by mean
−0.0006 / +0.0027 / +0.0007. The corrected population-factorization changes
were approximately 0 / 0 / +0.0005. Estimating population means/scales from the
independent 2,000-person reference changed them by about −0.0011 / −0.0010 /
−0.0008. These checks support the implementation under matching; they do not
cover arbitrary small panels or fixed-reference inference.

A reference with AR correlation 0.85 instead of alternating 0.2/0.7 reduced
those estimates by approximately 0.132 / 0.134 / 0.092. In S2 and S3, doubling
or halving the assumed prevalence shifted estimates by roughly 0.03. The
artifact therefore seals prevalence and the reference/scale identity.

Sample-logistic fitting and population-probit fitting agree for saturated
binary strata. In S4, logistic back-transformation exceeded the paired probit
estimate by about 0.017 on average in this small diagnostic. The logistic
helper remains a research comparison, not the production risk fit.

The planned small/richer basis comparison reused six S3 and six S4 pilot
seeds with supplied risks. Increasing four to eight bins reduced mean
relative sensitivity error from 5.75% to 3.45% in S3 and 12.66% to 6.40% in
S4. Relative normal-matrix errors fell from 0.66% to 0.24% and 3.21% to 0.82%,
respectively. RMS marginal-estimate differences from exact PCGC fell from
0.00113 to 0.00093 and 0.00175 to 0.00143. Individual relative sensitivity
errors remained large in the tails, especially S4. Those results support
reporting both global and individual errors and retaining the exact-span
requirement; they do not qualify arbitrary binning.

S6 adds a covariate-dependent genotype mean while generating liability from
the within-stratum genotype. Its exact study-kernel estimate is badly biased
relative to the intended residual genetic target. A design-matched reference
does not repair that incompatible kernel. The population-reference result
can look much better by cancellation; it is not a qualification for ancestry
adjustment. The tiny projection counterexample separately demonstrates that
projection and risk weighting do not commute and can create off-diagonal
residual noise.

## Numerical and integration evidence

The combined portable-build run passed 141 tests, including all binary tests,
generalized reference pass 1/2, HE batching, LDSC h² and the existing GxE CLI
contract. The private BLIS build passed 20 binary I/O/CLI/cross tests. A bound
two-worker BLIS smoke check verified native/NumPy agreement, exactly two
genotype passes, and the existing authenticated singleton OpenMP placement
contract. Shared native runtime safeguards are retained.
After adding probe/build provenance, the affected suites passed another
24 portable and 11 private-BLIS checks.

Tests include liability-integral derivatives; likelihood finite differences;
separation and rank failures; explicit off-diagonal pair equations; weighted
overlapping annotations; full and frozen-delete equations; finite-reference
and finite-risk-pair corrections; basis contraction; case and allele recoding;
trait exchange; zero/partial/full overlap; fractional PGEN dosage and missing
values; typed archive integrity; strict input alignment; and actual CLI
preparation followed by inference. The independent cross-sampler check was
rerun after adding risk-fit failure accounting and passed.

For one 4,000×4,000 continuous-risk dataset and eight probe seeds per count,
standard PCGC's RMS relative normal-matrix error was 3.89%, 1.65% and 0.40%
with 16, 64 and 256 probes. Corresponding RMS conditional-h² errors were
0.00833, 0.00364 and 0.00090. Inverse results were similar numerically. This
supports 256 as the initial default for these dimensions; it is not a bound
for every annotation system or ill-conditioned fit.

A separate private-BLIS check used one 4,000-marker binary–binary dataset
with 4,000 people per cohort and 1,000 shared controls. Across eight probe
seeds, 64/256 probes gave RMS relative cross-normal-matrix errors of
0.947%/0.359%, covariance errors of 0.00147/0.00055, and rg errors of
0.00057/0.00044. Common probes correlate numerator/denominator errors, so the
small rg error does not imply each primitive moment is exact.

Reference probes are on the variant axis; all source sketches finish before
target scoring. Reference estimation has no jackknife block IDs. The existing
frozen deletion convention is preserved, and a test explicitly distinguishes
it from recomputing both sides of a retained-variant kernel.

## Reproduction and output provenance

All generated results are outside the checkout under
`/data1/bronsonj/summit_pcgc_20260926/`. No individual simulation genotypes,
participant inputs or result bundles are added to Git. Relevant output
directories are:

- `pilot5`, `pilot_next`, `pilot_report`: Gaussian pilots and paired summaries.
- `confirmation_s0`, `confirmation_s2`, `confirmation_s3`, `confirmation_s4`,
  `confirmation_s7`, `confirmation_report`: independent univariate confirmation.
- `discrete_benchmark`, `discrete_s2`, `discrete_s5`: 20 discrete datasets per scenario.
- `external_finite_pair_confirmation`, `external_finite_pair_report`: corrected
  external estimates on the same 80 confirmation draws per scenario.
- `risk_reference_finite_pair`: current reference/risk sensitivity checks.
- `numerics_representative`: univariate probe diagnostics.
- `cross_numerics_private_blis`: rectangular probe diagnostics with overlap.
- `cross_bb0_pilot`, `cross_benchmark`, `cross_bb_shared_pilot`,
  `cross_bq_pilot`, `cross_pilot_report`: bivariate pilots.
- `cross_bb0_confirmation`, `cross_bb_shared_confirmation`,
  `cross_bq_confirmation`, `cross_confirmation_report`: independent bivariate confirmation.
- `basis_resolution`: four/eight-bin comparison on reused seeds.

Simulation manifests record arguments, seeds, source hashes and the starting
revision. Runs began before the branch's implementation commit, so the source
hashes, rather than the base revision alone, identify the executed estimator.
The finite-pair correction was identified in equation review after the first
confirmation run: external estimates and SEs were analytically rescaled using
regenerated risks on the same datasets. The maximum multiplier was 1.000271.
This is a documented paired correction, not a new untouched confirmation.
The earlier `risk_reference` directory is superseded by
`risk_reference_finite_pair`.

For example, from a built source checkout with its matching native extensions:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/pcgc/validate_pcgc.py --scenarios S3 \
  --replicates 80 --seed-offset 20 --out /new/output/s3_confirmation

PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/pcgc/validate_pcgc.py --generator discrete --scenarios S5 \
  --replicates 20 --out /new/output/discrete_s5

PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/pcgc/validate_cross.py --scenario BB_shared \
  --replicates 80 --seed-offset 20 --out /new/output/cross_confirmation

PYTHONPATH=src python scripts/pcgc/report_pcgc.py \
  --runs /new/output/s3_confirmation --out /new/output/s3_report
```

The local native tests used Python 3.12 in
`/home/bronsonj/anaconda3/envs/summit`, the existing ABI-matching
`build/pgs_optimization_portable` or `build/pgs_optimization_private`
extensions in `/data1/bronsonj/SUMMIT-cross-trait-response-covariance`, and
`build/private_blis` for the auxiliary extensions. The local environment
requires its `lib/libstdc++.so.6` in `LD_PRELOAD`. The checked-in
`scripts/pcgc/run_native_tests.py` temporarily directs both pytest and its
subprocesses to this source checkout without changing the installed package.

## Remaining scientific boundaries

General binary uncertainty needs a justified covariance calculation that
accounts for the relevant sampling, risk and reference variation. A nuisance
variance cannot simply be added without its cross-covariances. The strong-risk
results should guide that derivation before any scalability expansion.

Ancestry projection, covariate-dependent genotype kernels, arbitrary joint
selection, uncertain prevalence, tiny reference panels, and general
annotation enrichment remain outside the advertised qualification. Existing
PCGC/S-PCGC summary files need an explicit validated adapter. Further
performance work should preserve these contracts and begin with measured
representative workloads rather than enlarging the simulation grid.
