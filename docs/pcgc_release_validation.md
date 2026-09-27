# PCGC release validation

Date: 2026-09-27.

The binary cross-trait settings and the PCGC annotation components pass the
prespecified screens. Univariate calibration remains setting-dependent,
including conservative SEs with binary risk covariates and incomplete
qualification under strong continuous risks. The corrected SNP-block
jackknife is implemented for every binary method and annotation component.
All five CLI methods now report SEs by default, using 200 SNP blocks unless
`--njack` is supplied. The [interface review](pcgc_interface_audit.md) explains
why the earlier release gate was removed. The historical simulation results
below remain unchanged; their 50-block design is distinct from the CLI default.

This audit checks the corrected SNP-block jackknife across the implemented
binary methods, including annotation components and binary cross-trait fits.
It uses `pcgc_frozen_offdiagonal_estimating_equations_v2`; risks, population
scales, full source kernels, and source annotation masses remain fixed.
The earlier [jackknife audit](pcgc_jackknife_audit.md) and
[upstream comparison and scaling audit](pcgc_second_audit.md) describe the
estimator correction and the reference-engine optimization.

## Design

Each setting has 80 datasets, using replicate indices 20–99. Methods share
the same datasets. S0, S2, S3, S4, S7, and the three cross-trait settings replay
the original confirmation seeds with the corrected equations; they are not
independent confirmation of a correction chosen on those datasets. S1 and S5
extend the earlier pilots with 80 separate confirmation draws. No setting
exceeds 100 distinct datasets across pilot and confirmation.
S6 deliberately violates the kernel assumptions and remains a negative
control in the original qualification record; it is not a calibration target.

The univariate simulations use 4,000 people, 4,000 markers, and 50 contiguous
SNP blocks. LD blocks contain 40 markers, with alternating AR(1) correlations
of 0.2 and 0.7. External references contain 2,000 independent population
samples. Reference moments are exact, so reference-probe noise is excluded.
The generator uses Gaussian markers under a conditional liability model;
these runs do not establish calibration for arbitrary genotype distributions
or population structure.

| Setting | Population prevalence K | Sample case fraction P | Covariate predictor | Conditional genetic variance |
|---|---:|---:|---|---:|
| S0 | 0.10 | 0.10 | None | 0.25 |
| S1 | 0.10 | 0.50 | None | 0.25 |
| S2 | 0.10 | 0.50 | Binary, variance 0.25 | 0.25 |
| S3 | 0.10 | 0.50 | Gaussian, variance 0.25 | 0.25 |
| S4 | 0.01 | 0.50 | Gaussian, variance 1 | 0.25 |
| S5 | 0.10 | 0.50 | Binary, variance 0.25 | Components 0.05 and 0.20 |
| S7 | 0.10 | 0.50 | Gaussian, variance 1 | 0 |

Both supplied and fitted risks are evaluated. `pcgc-basis-4` denotes a
four-bin approximation used only in this experiment. The public exact-basis
mode is instead checked against standard PCGC by dense and fixed-probe
equivalence tests. The HE comparator calls SUMMIT's existing HE fit with
actual marginal linear-regression beta/SE values, then applies a global
liability conversion. Its helper exports total uncertainty only; missing
component intervals are not a test of HE component calibration.

The screens are those specified before the confirmation runs: a bias interval
inside the equivalence margin, RMS SE / empirical SD between 0.8 and 1.25,
and a Wilson coverage interval containing 0.95 with lower bound above 0.85.
The margin is the larger of 0.02 and 10% of the truth for total variance,
0.01 and 10% for components or covariance, and 0.05 for genetic correlation.
Null settings also require a rejection-rate interval containing 0.05 with
upper bound below 0.15. Invalid estimates and intervals remain in the attempted
denominator. These are coarse screens with Monte Carlo uncertainty, not a
guarantee of calibration outside the tested settings.
No SE multiplier, block count, or estimator was selected after inspecting
these results.

The binary cross-trait runs use 4,000 people per cohort and 4,000 markers,
with conditional heritability 0.25 for each trait and a binary risk predictor
of variance 0.25. Binary cohorts have K = 0.1 and P = 0.5. `BB0` uses disjoint
cohorts and true genetic correlation zero. `BB_shared` uses correlation 0.5
and 1,000 shared controls, with private controls sampled to preserve each
trait's marginal ascertainment law. `BQ` pairs a binary cohort with a disjoint
quantitative cohort at correlation 0.5. Standard and inverse PCGC are evaluated
with supplied and fitted risks, for both covariance and genetic correlation.

## Univariate results

All 5,600 method/risk fits completed without a recorded fit failure. The
table shows **marginal total variance with fitted risks**; scalar liability
and HE use global conversion. Each cell is RMS SE / empirical SD, followed
by 95% interval coverage. An asterisk means the combined bias and calibration
screen did not pass.

| Setting | PCGC | Inverse | External LD | Four-bin basis | Scalar liability | HE + global conversion |
|---|---|---|---|---|---|---|
| S0 | 1.043; 79/80 | 1.043; 79/80 | 1.043; 79/80 | 1.043; 79/80 | 1.043; 79/80 | 1.039; 79/80 |
| S1 | 0.994; 75/80 | 0.994; 75/80 | 0.989; 75/80 | 0.994; 75/80 | 0.994; 75/80 | 0.990; 75/80 |
| S2 | 1.262; 79/80 * | 1.221; 79/80 | 1.267; 79/80 * | 1.262; 79/80 * | 1.240; 79/80 | 1.245; 79/80 |
| S3 | 1.076; 77/80 | 0.998; 75/80 | 1.077; 77/80 | 1.078; 77/80 | 1.062; 78/80 | 1.065; 78/80 |
| S4 | 0.935; 75/80 | 0.888; 70/80 * | 0.936; 75/80 | 0.920; 73/80 * | 0.975; 74/80 * | 0.976; 74/80 * |
| S5 | 1.141; 79/80 | 1.151; 79/80 | 1.146; 79/80 | 1.141; 79/80 | 1.086; 79/80 | 1.090; 79/80 |
| S7 | 0.861; 74/80 * | 0.916; 73/80 * | 0.860; 74/80 * | 0.859; 75/80 | 0.924; 74/80 * | 0.923; 74/80 * |

S0 and S1 support the constant-risk equivalence. S3 passes the total-variance
screens for every method with both risk sources. S2 has conservative fitted-risk
PCGC SEs: the marginal SE/SD ratio is 1.262, with a bootstrap interval of
1.117–1.479. Its point estimates pass the bias-equivalence screen. With supplied
risks the ratio is 1.198 and the coarse screen passes; that does not mean the
ratio is known to equal one.

The rare-disease, strong-risk S4 setting needs more care. With fitted risks,
standard PCGC passes the marginal screen, but conditional coverage is 74/80
and misses the required Wilson lower bound. With supplied risks, standard
PCGC has SE/SD 0.861 and coverage 73/80. The bootstrap ratio interval is
0.765–1.002 and the Wilson coverage interval is 0.830–0.957: the screen fails,
but these 80 draws do not establish a remaining implementation defect. The
previous matched-equation upstream comparison addresses that question separately.

S4 inverse weighting has a fitted-risk marginal mean of 0.1087 against truth
0.125, with bias interval −0.0345 to 0.0020. It fails bias equivalence; the
interval includes zero, so the observed downward shift is not conclusive
evidence of population bias. Its coverage is 70/80, or 68/80 with supplied
risks. The heavy-tail concern described in the
[scaling and uncertainty audit](pcgc_second_audit.md) still applies. Four-bin
approximation also misses the S4 coverage screen. In the constant and binary-risk
settings its contraction is exact, so those successes do not test approximation
error for continuous risks.

The strong-risk null S7 also prevents a general calibration claim. Standard
PCGC covers 74/80 with either risk source; fitted-risk SE/SD is 0.861 with
bootstrap interval 0.742–1.029. The scalar and HE conversions do not pass that
setting either. No method was substituted after observing these results.

### Annotation components

S5's two disjoint annotations have marginal truths 0.04 and 0.16. Standard,
inverse, external-LD, and four-bin PCGC pass for **both components and the total,
on both scales and with both risk sources**. For fitted-risk standard PCGC:

| Quantity | Mean | RMS SE / SD | Coverage |
|---|---:|---:|---:|
| Component 1 | 0.03963 | 1.046 | 78/80 |
| Component 2 | 0.15737 | 1.146 | 77/80 |
| Total | 0.19700 | 1.141 | 79/80 |

Scalar liability passes the total screen but covers only 74/80 for the smaller
component. HE component uncertainty was not exported by the comparator and
was not evaluated. This experiment tests a partition; overlapping annotations
are covered by numerical oracle tests, not a separate calibration panel here.

The [complete univariate table](../reports/pcgc_release_20260927/univariate.tsv)
includes both scales, risk sources, Monte Carlo intervals, and finite-interval
counts. Single-component duplicates are omitted. HE component rows have zero
finite intervals because the helper does not return those SEs.

## Binary cross-trait results

All 960 method/risk fits completed. Every covariance and genetic-correlation
screen passed, with 80 finite point estimates, 80 finite intervals, and finite
ratio estimates for all 50 deletions in each dataset. Fitted-risk results are:

| Setting | Method | Covariance mean | Covariance SE/SD; coverage | Genetic correlation mean | Correlation SE/SD; coverage |
|---|---|---:|---|---:|---|
| BB0 | PCGC | −0.00141 | 1.113; 76/80 | −0.00703 | 1.105; 76/80 |
| BB0 | Inverse | −0.00131 | 1.115; 76/80 | −0.00668 | 1.111; 76/80 |
| BB_shared | PCGC | 0.12707 | 1.159; 78/80 | 0.50270 | 1.137; 78/80 |
| BB_shared | Inverse | 0.12681 | 1.153; 78/80 | 0.50227 | 1.123; 78/80 |
| BQ | PCGC | 0.12088 | 1.158; 77/80 | 0.48785 | 1.203; 78/80 |
| BQ | Inverse | 0.12072 | 1.161; 77/80 | 0.48776 | 1.208; 77/80 |

Covariance is on the conditional scale: truth is zero for BB0 and 0.125 for
the other settings. Genetic-correlation truths are zero and 0.5. Supplied-risk
SE/SD ratios range from 1.106 to 1.208, with coverage 76/80 to 78/80; all pass
the same screens. The [complete bivariate table](../reports/pcgc_release_20260927/bivariate.tsv)
includes the Monte Carlo intervals and counts.

These tests support the implementation under the stated marginal sampling
design. They do not qualify arbitrary overlap recruitment, strong continuous
risks, cross-trait annotation components, or weak heritability denominators.
The binary cross-trait API remains research-only.

## Numerical and integration checks

The final review retained the production thread and affinity safeguards.
Several PCGC tests had requested one native worker after another test had
initialized a different process-wide worker count. They now use the existing
test helper that respects that contract. The numerical estimator was unchanged
in this release pass.

The treatment-aware cross-trait launcher now honors its time-budget override
on chromosome 22 and handles optional paths without relying on an empty array
under older Bash `set -u`. Tests exercise paths containing spaces, additional
phenotype roots, manifest authentication, checkpoint resume, and the fixed
master height definition.

Quantitative cross-trait response models are a separate path. The integrated
tests exercise all four reference modes, including authenticated fitting with
additional phenotype inputs. Their statistical evidence remains the
[existing validation record](wiki/Cross-trait-response-covariance.md): corrected
aggregate response-correlation coverage was 97/100 and 98/100 on the saved
panel, with RMS SE/SD 1.072 and 1.037. Those were reanalyses of the datasets
used to diagnose the earlier problem, and some other coordinates still failed
the screens. This release does not claim a new independent calibration panel
for all four quantitative reference modes.

| Check | Result |
|---|---|
| Full portable OpenBLAS suite | 1,639 passed, 9 skipped, 1 expected-failure test passed |
| Private BLIS prediction, PCGC, cross-trait, and chromosome suites, one worker | 302 passed, 9 skipped |
| Private BLIS production launcher, four workers | 69 passed |
| Additional four-worker cross-trait residual and placement checks | 24 passed |

The single-worker skips require parallel workers and are covered by the
four-worker checks. The portable expected-failure test concerns a previously
rejected native layout; one passing run does not qualify that layout for use.
Both native builds were compiled from the integrated source. Tests used the
`summit` conda environment.

Actual private BLIS worker placement was measured during protected matrix
products. All 48 newly observed worker threads inherited the four selected
physical cores; at most three were active alongside the main thread. The
main thread returned to its prescribed OpenMP core. The product oracle passed
without repairs. This verifies placement for that launch, not full-genome
throughput. The existing [scaling measurements](pcgc_second_audit.md) remain
the evidence for large-sample memory and bounded throughput.

## Reproduction

The drivers are `scripts/pcgc/validate_pcgc.py` and
`scripts/pcgc/validate_cross.py`. Each output directory records arguments,
source hashes, seed rules, and per-dataset results. Use a fresh output path
for each invocation; the drivers refuse to overwrite an existing directory.

```bash
conda run --no-capture-output -n summit \
  python scripts/pcgc/validate_pcgc.py \
  --out results/pcgc_S5 --scenarios S5 --replicates 80 --seed-offset 20

conda run --no-capture-output -n summit \
  python scripts/pcgc/validate_cross.py \
  --out results/pcgc_BB_shared --scenario BB_shared \
  --replicates 80 --seed-offset 20
```

Run from an installed checkout of this source. The release audit used one
NumPy/OpenMP/BLIS worker per simulation process, fixed CPU affinity, and the
local transparent-huge-page guard before allocating genotype arrays.
`report_pcgc.py` and `report_cross.py` summarize completed runs without pooling
the pilot and confirmation phases. Local audit outputs are under
`/data1/bronsonj/summit_release_20260926/`.
