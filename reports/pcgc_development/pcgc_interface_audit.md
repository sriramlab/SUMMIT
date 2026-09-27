# PCGC standard errors and CLI defaults

Date: 2026-09-27.

All five binary methods now report component and total SNP-block jackknife
SEs by default. Both research flags have been removed. The default is 200
contiguous SNP blocks, with an integer `--njack` override. This change removes
an interface restriction; it does not alter the corrected estimating equations
or apply an empirical SE multiplier.

## Comparison with upstream PCGC

SUMMIT is not an end-to-end replica of the upstream packages. The differences
are explicit:

| Calculation | Upstream implementation | SUMMIT |
|---|---|---|
| Direct covariate-aware PCGC | Conditional response, individual sensitivity weights, off-diagonal pair regression | Same pair equations for supplied risks and the same genotype scale; tested against explicit pairs |
| Direct PCGC uncertainty | Delete one person and all pairs involving that person | Delete target SNP blocks with full source kernels fixed |
| S-PCGC SNP jackknife | Recalculate retained score sums and masses; hold the full normal matrix and supplied diagonal intercept fixed | Recalculate target-row contributions and retained target masses, including exact per-SNP diagonal removal |
| S-PCGC score preparation | Globally standardized phenotype times sensitivity; sample-logistic risk fit | Conditionally standardized phenotype times sensitivity; ascertainment-aware population-probit risk fit |
| S-PCGC annotation convention | Squared annotation weights | Kernel weights enter linearly; the comparison supplies square roots to upstream |

The inspected sources are pinned to
[S-PCGC inference at 5211d17](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_main.py),
[its summary creator](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_sumstats_creator.py),
and [direct PCGC at fdc5089](https://github.com/omerwe/PCGCs/blob/fdc5089f485fe25c04a8972665fee4216570764f/deprecated/pcgcs_direct.py).
The [second audit](pcgc_second_audit.md) records the executable comparison,
source hashes, and the reference/normalization differences.

The comparison adapts upstream inputs to match SUMMIT's full normal equations,
then executes the actual upstream deletion routine. It establishes arithmetic
agreement with that routine under the adapter. It does not establish equality
of independently prepared S-PCGC and SUMMIT files. Inverse weighting and
population-LD transfer are separate estimators; exact basis contraction is
algebraically the same as standard PCGC.

Deleting target rows from both sides of the estimating equation preserves the
same genetic parameter: the retained target mass changes, while the source
mass remains the full-genome mass. Keeping the entire normal matrix fixed can
instead introduce deletion variation from LD heterogeneity even when score
rows equal their model expectations. The existing independent pair and null
covariance checks support retaining SUMMIT's corrected deletion equations.
Neither package's jackknife can guarantee calibration outside its assumptions.

## Why the release gate was removed

The earlier screen mixed bias equivalence, a fixed SE/SD tolerance, and a
coverage rule. Failing it is not a statistical demonstration that an SE
implementation is wrong. For 80 independent replicates, the coverage rule
accepts only 75 through 79 covered intervals. In particular:

- 74/80 coverage is 92.5%, with Wilson interval [84.59%, 96.52%]. It fails
  because the lower endpoint is just below 85%. Its one-sided exact binomial
  undercoverage p-value against 95% is 0.211.
- Even 80/80 fails because its Wilson interval excludes 95%.
- Under exactly 95% independent coverage, this coverage rule alone rejects
  with probability 22.729%, calculated from a Binomial(80, 0.95) distribution.
  Requiring many method/component screens to pass compounds this problem;
  the screens share datasets and are not independent.

The paired 80-dataset audit found nearly identical 50-block SE/SD ratios for
SUMMIT and the upstream rule in the difficult S4 and S7 settings. All eight
SUMMIT SE/SD bootstrap intervals in that audit contained one. These findings
do not prove calibration, but they do not support withholding all binary SEs.
The [release tables](pcgc_release_validation.md) remain unchanged, including
the conservative binary-covariate results and weak inverse-weighting results.
The reporting scripts retain their original screens so old analyses remain
reproducible. No threshold was retuned to make their tables pass.

Finite-variance assumptions still matter. Inverse weighting can have infinite
variance with sufficiently strong Gaussian risk predictors. Population-LD
transfer needs reference matching and a valid factorization of risk and
genotype moments. Fitted risks, prevalence, genotype scaling and probe draws
remain fixed during SNP deletion. Those limitations belong in the method
description and output conditioning metadata.

## Block counts and shared machinery

The previous binary CLI ignored the root parser's jackknife default and only
computed SEs when `--njack` was explicitly supplied with a research flag.
The value 50 came from the validation drivers and examples. The ordinary
HE/LDSC CLI currently defaults to `chr`; that behavior is unchanged.

The validation simulations had 4,000 SNPs in 100 independent 40-SNP LD blocks.
Fifty deletion groups kept two complete LD blocks per group. Using 200 groups
on those same data cuts each LD block in half, so increasing the count is not
automatically an improvement. On 454,000 SNPs, 200 equal-count groups instead
contain 2,270 SNPs each. Users should choose a count appropriate to the genomic
span and LD structure. A count exceeding the SNP count fails explicitly.

A bounded recheck reused four original seeds each from S4 and S7, with
4,000 people and 4,000 SNPs, supplied/fitted risks, and 50/100/200 groups.
All adapted upstream full fits and deletions passed their independent
arithmetic comparisons. These eight reused datasets are a numerical check,
not an additional calibration experiment. For the supplied-risk S7 design
at seed 827020, the exact conditional expected jackknife variance divided by
the true variance was:

| Blocks | Expected variance ratio |
|---:|---:|
| 50 | 1.000790 |
| 100 | 1.002908 |
| 200 | 0.977286 |

This calculation integrates over independent Bernoulli outcomes conditional
on the fixed genotypes and risks. It excludes fixed-case-count recruitment,
fitted-risk uncertainty and reference error. The small decrease at 200 is
consistent with splitting LD blocks; it is not grounds for an SE multiplier.

Binary CLI block membership uses `JackknifeSpec` and `JackknifeDesign`.
Multi-annotation sums reuse the generalized reference reducer; covariance now
calls the existing helper used by contextual and cross-trait inference.
Single-component fits retain the linear-time accumulation path. Reference
preparation still takes exactly two study-genotype traversals, independent
of the jackknife count. No decoder or native thread contract was changed.

Fit JSON schema 2 records SEs, covariance, replicates, block count and sizes,
and fixed-nuisance conditioning. Legacy preparation schema 1 lacks the exact
per-SNP diagonal information and must be regenerated for CLI inference.
The low-level Python API still permits point-only numerical checks.

## Reproduction

The new local outputs are under
`/data1/bronsonj/summit_pcgc_interface_20260927/blocks_200/`; the manifest records
arguments, source hashes and the pinned upstream SHA256. From a source
environment without a competing editable installation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts/pcgc \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/pcgc/compare_official_jackknife.py \
  --out NEW_OUTPUT_DIRECTORY --scenarios S4 S7 \
  --replicates 4 --seed-offset 20 --blocks 50 100 200 --null-analytic
```

CLI tests exercise every method with overlapping annotations, 200 default
groups, explicit overrides and nondivisible SNP counts. They recompute each
deletion directly and compare component and total covariance. The native CLI
test prepares BED moments and checks that inference reports SEs without an
extra flag. Invalid counts and removed flags are tested separately.

Local release verification passed 1,646 portable OpenBLAS tests (9 skipped,
1 expected-failure test passed) and 318 private-BLIS tests with four workers
(no skips). The latter includes prediction, PCGC and cross-trait paths under
the production placement launcher. Native sources and runtime safeguards are
unchanged. Tracked-file and published-history content checks had no findings.
