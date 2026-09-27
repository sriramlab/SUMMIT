# PCGC jackknife audit

This audit supersedes the uncertainty results in
[the original qualification](pcgc_qualification.md). Work is on `feat/pcgc`,
based on `working`; the original quantitative and generalized-GxE estimators
are unchanged. Results are retained under
`/data1/bronsonj/summit_pcgc_audit_20260926/`.

## Identified errors and corrected equations

The first PCGC implementation deleted target SNP rows, divided both annotation
axes by retained masses, and reused the full same-person matrix. That combination
does not describe the estimating equation of the retained rows. The earlier
test reproduced that formula, so it could not identify the statistical error.

For population-standardized genotypes X, risk sensitivities d, and full source
annotation masses M, define

```
t_j  = (sum_i X_ij d_i z_i)^2 - sum_i X_ij^2 d_i^2 z_i^2
U_jb = sum_(i != l) d_i^2 d_l^2 X_ij X_lj sum_k A_kb X_ik X_lk
E[t_j] ≈ sum_b U_jb theta_b / M_b(full)
```

Thus, with retained target set T,

```
H_ab(T) = sum_(j in T) A_ja U_jb / [M_a(T) M_b(full)]
b_a(T)  = sum_(j in T) A_ja t_j / M_a(T)
```

The target normalization cancels in the solve; the full source normalization
does not. Every deletion estimates the same full-genome component parameters.
Deleting only target rows can make H asymmetric even with exact references.
Symmetrizing it would change its expectation. The solver retains the directional
equation, checks its condition number, and checks the symmetric part of the
full Gram for positive definiteness. No eigenvalue clipping or regularization
is introduced.

The adapter now subtracts each SNP's exact same-person reference term before
storing LD rows. It obtains those terms in the existing two genotype passes:
accumulate `X^2 A` in pass 1, then score its risk-weighted products in pass 2.
This also supplies exact overlap corrections in cross-trait deletions. No
jackknife labels enter reference construction, and no source sketches are
recomputed per deletion. The ordinary equal-group jackknife covariance is
`(B-1)/B sum_b (theta_-b - mean(theta_-b)) (theta_-b - mean(theta_-b)).T`.
Comparable genomic block sizes remain a requirement.

Other repairs include arbitrary integer block-label handling, explicit failure
with fewer than two groups, direct marginal SE output, and admission of score
buffers before allocation. Nonfinite squared inverse responses fail before any
reference traversal. Schema-2 artifacts identify corrected LD rows in
their checksummed manifests. Schema-1 artifacts retain point-estimate support
but require regeneration for corrected uncertainty.

Deleting SNPs does not change the participants, case fraction, or risk-model
fit. Keeping those quantities fixed is appropriate for this resampling unit.
It does not propagate uncertainty from fitted risks, population prevalence,
the covariate-variance conversion, or independent reference/probe estimation.
Those limitations are explicit in fit output. The original
[PCGC-s individual-level routine](https://github.com/omerwe/PCGCs/blob/fdc5089f485fe25c04a8972665fee4216570764f/deprecated/pcgcs_direct.py)
uses person deletions, whereas its
[S-PCGC summary implementation](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_main.py)
uses SNP blocks. Their resampling units and sufficient statistics differ;
neither provides a reason to rescale both axes of SUMMIT's frozen target rows.

## Verification strategy

Independent tests now enumerate retained-target/full-source **person pairs**
with overlapping annotations. A noise-free score-expectation test requires
every deletion to return the original component parameters. These tests fail
under the former scaling or deletion symmetrization. An exhaustive test of all
64 binary outcomes for six people with unequal risks verifies the exact null
identity `Cov(theta | X, risks) = 2 H^-1` for independent Bernoulli responses.
It checks diagonal subtraction without Monte Carlo error or a fitted nuisance
model. This identity is not a claim about fixed-case-count sampling or general
nonnull uncertainty.

Streamed/native tests cover fixed-probe agreement, two physical passes,
fractional PGEN dosage, noncontiguous variant subsets, population-scale and
allele alignment, typed archives, CLI behavior, and overlapping cross-trait
cohorts. The quantitative HE, LDSC and generalized-GxE regression tests remain
part of the focused suite.

Final verification: **152 tests passed** with the portable OpenBLAS extensions;
**44 focused tests passed** with private BLIS, including BED/PGEN preparation,
artifact/CLI behavior, cross-trait moments and the exact OLS comparator. There
were no skips in those native runs. A separate pure-Python run passed with
native-only tests skipped. These are local tests; no remote CI run is claimed.

The statistical audit replays the **same 80 confirmation seeds** for S3, S4,
and S7, using 4,000 people, 4,000 SNPs, exact references, and 50/100 SNP blocks.
No estimator is selected or tuned using the results. Both supplied and fitted
risks are retained. A 20-seed S5 component pilot exercises the multi-annotation
solver. The earlier largest null outlier, seed 827065, reproduces exactly in
a single-threaded run and remains in the results.

## Calibration results

All 240 replayed datasets completed for both risk treatments and block counts.
The original 50-block SEs were reproduced before comparing the correction.
Full point estimates are unchanged to floating-point precision except for one
historical fitted-risk result differing by `1.85e-8`, far below its sampling
uncertainty. This small discrepancy is reported rather than treated as evidence
of a statistical effect.

The 50-block correction increased SEs by approximately **3.2–4.1%**. Results
below use RMS estimated SE divided by empirical SD; coverage counts refer to
nominal 95% normal intervals out of 80 datasets.

| Scenario | Risk | Old 50-block SE/SD | Corrected 50-block SE/SD | Coverage, old → corrected | Corrected 100-block coverage |
|---|---|---:|---:|---:|---:|
| S3: moderate continuous covariate | supplied | 1.027 | 1.060 | 76 → 76 | 77 |
| S3 | fitted | 1.043 | 1.076 | 76 → 77 | 77 |
| S4: rare trait, strong covariate | supplied | 0.828 | 0.862 | 72 → 73 | 74 |
| S4 | fitted | 0.898 | 0.935 | 74 → 75 | 75 |
| S7: null, strong covariate | supplied | 0.832 | 0.861 | 73 → 74 | 75 |
| S7 | fitted | 0.832 | 0.861 | 73 → 74 | 75 |

The correction resolves the fitted-risk S4 failure under the existing screen.
S7 passes that screen with 100 blocks, but remains just below its coverage
threshold with 50. Supplied-risk S4 does not pass at either block count. Its
corrected 50-block coverage Wilson interval is `[0.830, 0.957]`; S7's is
`[0.846, 0.965]`. All six 50-block SE/SD bootstrap intervals include 1, so these
80-dataset results do not establish a precisely measured universal SE deficit.
Increasing blocks was a prespecified comparison, not a correction factor fitted
to the observed coverage. **Binary SEs remain research-only.**

The S5 component pilot is also insufficient for general component calibration:
corrected fitted-risk SE/SD ratios are 1.017 and 1.443 for its two components
over 20 datasets. The correction must not be judged solely by total h2.

No risk fits are refitted per SNP block. Known-risk failures show that omitted
risk-fitting variability cannot be the sole explanation for the earlier screen
failures. Normal-interval approximation, finite block information, model
approximation and Monte Carlo uncertainty remain separate issues. The audit
does not add an empirical SE multiplier, drop an outlier, or claim that every
remaining failure is an implementation error.

The final reducer reuses SUMMIT's contiguous-block and sparse-annotation fast
paths. Replaying two S5 datasets after that optimization changed point estimates
by zero, covariance entries by at most `2.17e-19`, and total SEs by at most
`6.94e-18`. The single-component path uses three linear grouped reductions.

An additional two-component numerical check used one 2,000-person/2,000-SNP S5
dataset and eight probe seeds. At 256 probes, standard PCGC's component RMS
errors were 0.022 and 0.033 times their exact-reference jackknife SEs; inverse
PCGC gave 0.023 and 0.033. The directional matrices were deliberately not
symmetrized. This checks numerical approximation, not population calibration.

## Efficiency

For one requested basis contraction, `Phi c` is formed before reference
calculation. This computes exactly the requested rank-one feature instead of
constructing all basis-pair kernels and discarding them after contraction.
The shared cross-trait engine still uses multiple features where needed.

In a controlled NumPy comparison with 2,000 people, 1,600 SNPs, four basis
columns and 64 probes, median preparation time fell from 1.117 to 0.518 seconds
across three repeats: **2.16× faster**. Relative normal-matrix error was
`1.82e-16`. Planned reference memory fell from 142.6 MB to 96.9 MB (the latter
includes the new score/diagonal buffers). This is a small reference benchmark,
not a production throughput forecast. See `basis_benchmark.json`.

Score and reference diagonal products reuse each block's squared genotypes,
batch right-hand sides, and release the pass-1 diagonal accumulator after
constructing pass-2 right-hand sides. Corrected LD rows replace the old rows;
there is no second persistent M-by-annotation diagonal array in artifacts.
BED/PGEN decoding, protected native products, probe generation and memory
planning continue to use SUMMIT's existing machinery.

For the small exact population-reference oracle, the smaller of the sample and
variant axes now determines the square matrix. With R=4,000 and M=20,000,
the square workspace is 128 MB instead of 3.2 GB. The sample-axis identity
`X_j.T (X A X.T - diag(X A X.T)) X_j` gives exactly the same per-SNP U statistic.
Target products are tiled, and the local run uses protected native products.
This is a research oracle; the production reference remains the two-pass
variant-probe estimator.

## Local real-data design

The existing `code_ht_fix` hypertension phenotype uses corrected prescription
classes, documented by the local producer
`/home/bronsonj/ml_4h/02_genetics_of_recognition/scripts/77_rx_fix_phenotypes.py`.
It has 290,868 eligible people in the unrelated EUR genotype panel. The target
population here is that eligible UKBB cohort; its case fraction,
**0.4289333993**, supplies the prevalence for artificial case-status sampling.
This is not an assumption about general UK hypertension prevalence.

Before fitting, the driver selects 20,000 common SNPs without reference to
association statistics, covering all 22 autosomes. It holds out a random
4,000-person population reference, then samples two 16,000-person studies with
50% and 90% cases. Their risk model includes age, sex, their interaction, age
squared and its interaction with sex. Allele frequencies from the full EUR
cohort define the common population genotype scale. Frequency alleles are
checked against the native BED reader's counted allele.

Each study shares two native genotype passes between PCGC reference estimation
and PCGC/OLS score preparation. HE and constrained LDSC use the same SNPs and
independent population LD reference. Their reported marginal estimates apply
global liability conversion and the observed residual-variance fraction after
covariate adjustment. Intercept-only comparisons are retained separately.
The 512- and 1,024-probe runs use identical sample, variant, scale and reference
identities; only the probe count changes.

Doubling probes changed the PCGC estimates by only 0.000024 and 0.000047.
The covariate-adjusted LDSC comparison at 90% cases moved by 0.00870, showing
greater sensitivity to noise in individual LD rows. The final comparison
therefore uses an **exact** population LD reference. The final OLS baselines
also use their residual degrees of freedom and SUMMIT's actual beta/SE score
reconstruction; the initial driver used normalized cross-products directly.
Those initial baseline values are retained as diagnostics, not final results.

To avoid repeating PCGC reference work for this baseline correction, the final
driver authenticates and reuses the PCGC fits, then makes one scoring pass per
study. It saves compact SNP-level OLS beta/SE arrays and population LD rows so
subsequent baseline checks require no genotype traversal. It saves no
participant-level phenotype, risk or genotype rows.

Final marginal panel estimates (estimate ± SNP-jackknife SE):

| Case fraction | PCGC | External-LD PCGC | HE, age/sex adjusted | LDSC, age/sex adjusted |
|---|---:|---:|---:|---:|
| 0.50 | 0.06951 ± 0.01641 | 0.06942 ± 0.01638 | 0.06826 ± 0.01628 | 0.06784 ± 0.01618 |
| 0.90 | 0.02752 ± 0.04593 | 0.02751 ± 0.04591 | 0.03121 ± 0.05256 | 0.03110 ± 0.05215 |

PCGC and HE/LDSC are close in this dataset; the differences do not establish a
material PCGC advantage. The 90%-case design has much larger uncertainty, as
expected with few controls. Final results and compact score products are in
`real_hypertension_final/`; the earlier probe and approximate-score comparisons
remain in their original directories. The final baseline correction took
35.8 seconds, including 28.8 seconds for exact reference LD and one study
scoring pass per cohort (3.65 and 3.14 seconds). Authenticated reuse avoided
another two PCGC reference calculations.

The combined machine-readable report, including Monte Carlo intervals and
paired source checks, is `report_final/report.json`. `confirmation/` contains
the 80-seed replay; `components/` the component pilot;
`component_numerics/` the fixed-probe comparison; and
`final_reducer_parity/` the check after restoring shared reduction fast paths.

Runs use four physical cores (58–61), process-local THP disablement, and private
BLIS. Output records native singleton-place OpenMP attestation and the BLIS
four-worker placement contract. A live check also observed active workers
restricted to those cores. The 512-probe run took approximately 190 seconds
for the population reference and both study fits. Study reference/score work
planned 336.4 MiB; sampled process RSS was approximately 0.714 GiB, including
metadata and other runtime allocations.

These estimates concern a restricted SNP panel. Strong ascertainment and
varying age/sex risks motivate expecting different PCGC and global-conversion
weights; large numerical differences are not guaranteed. Residual population
structure, risk-model misspecification and uncertain general-population
prevalence remain unresolved. Real-data agreement cannot establish unbiasedness
or SE calibration. Only aggregate results and input/axis hashes are saved.

## Reproduction

Drivers refuse to overwrite existing output paths. Source hashes are captured
before numerical work, and prior simulation/real-data files are preserved.

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  taskset -c 56-57 python scripts/pcgc/validate_jackknife.py \
  --out NEW_AUDIT_DIRECTORY --scenarios S3 S4 S7 --replicates 80 --blocks 50 100

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/pcgc/benchmark_basis.py --out NEW_BENCHMARK_JSON
```

The real-data driver requires this checkout plus ABI-compatible extensions.
The local runs used `/home/bronsonj/anaconda3/envs/summit/bin/python`,
`LD_PRELOAD=/home/bronsonj/anaconda3/envs/summit/lib/libstdc++.so.6`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=4`, `BLIS_NUM_THREADS=4`,
`OMP_PROC_BIND=true`, `OMP_PLACES='{58},{59},{60},{61}'`,
`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, and `taskset -c 58-61`:

```
python -B scripts/pcgc/validate_real_data.py \
  --extension-dir /data1/bronsonj/SUMMIT-cross-trait-response-covariance/build/pgs_optimization_private \
  --auxiliary-extension-dir /data1/bronsonj/SUMMIT-cross-trait-response-covariance/build/private_blis \
  --out NEW_REAL_DATA_DIRECTORY --samples 16000 --variants 20000 \
  --reference-samples 4000 --probes 1024 --threads 4 --exact-reference
```

CPU availability must be checked before reusing the recorded local placement.
`--reuse-pcgc PARENT_DIRECTORY` additionally checks input hashes, sample/variant/
scale identities, risk diagnostics and the producing PCGC implementation before
reusing a point/SE fit. A code-identity mismatch requires a fresh PCGC reference;
it is not silently ignored.
